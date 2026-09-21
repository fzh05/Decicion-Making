"""
=========================================================================
PEMODELAN POTENSI PERINGATAN DINI BANJIR BERBASIS ML & CURAH HUJAN HARIAN
DAS Way Sekampung, Provinsi Lampung
=========================================================================

Pipeline:
1. Load & cleaning data NASA POWER (harian)
2. Feature engineering (curah hujan kumulatif, API, dll)
3. Labeling kategori potensi banjir (berbasis intensitas curah hujan BMKG)
4. Split data & training 3 model: Random Forest, XGBoost, SVM
5. Evaluasi model (accuracy, precision, recall, F1, confusion matrix)
6. Feature importance
7. Simpan model & hasil

CATATAN PENTING:
- Data ini baru 1 titik lokasi x 1 tahun (365 hari). Untuk hasil yang lebih
  kuat & publishable, sangat disarankan menambah data historis (3-5 tahun)
  dan/atau beberapa titik lain di DAS Way Sekampung (lihat script
  02_download_data_nasa_power.py untuk download otomatis banyak titik/tahun).
- Label "potensi banjir" di sini dibuat dari AMBANG curah hujan (proxy),
  BUKAN dari data kejadian banjir aktual. Kalau kalian sudah punya data
  kejadian banjir riil (dari BNPB/BPBD/berita), sebaiknya dipakai untuk
  memvalidasi/mengganti label ini -- lihat bagian LABELING di bawah.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

INPUT_CSV = "POWER_Point_Daily_20250101_20251231_005d16S_105d11E_UTC.csv"
OUTPUT_DIR = "output"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================================
# 1. LOAD & CLEANING DATA
# =========================================================================
def load_power_data(filepath):
    """
    File NASA POWER punya header ~17 baris sebelum data mulai.
    Baris data dimulai setelah '-END HEADER-'.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_end_idx = next(
        i for i, line in enumerate(lines) if "-END HEADER-" in line
    )
    data_start_idx = header_end_idx + 1  # baris berikutnya = nama kolom

    df = pd.read_csv(filepath, skiprows=data_start_idx)
    return df


print(">> Loading data...")
df = load_power_data(INPUT_CSV)
print(f"Jumlah baris awal: {len(df)}")
print(df.head())

# Ubah YEAR + DOY (day of year) jadi kolom tanggal
df["date"] = pd.to_datetime(df["YEAR"].astype(str), format="%Y") + \
             pd.to_timedelta(df["DOY"] - 1, unit="D")
df = df.sort_values("date").reset_index(drop=True)

# NASA POWER pakai -999 untuk data hilang -> ganti jadi NaN
df = df.replace(-999, np.nan)

# Cek missing value
print("\n>> Missing value per kolom:")
print(df.isna().sum())

# Pakai PRECTOTCORR (curah hujan terkoreksi) sebagai variabel utama.
# Kalau mau pakai IMERG_PRECTOT sebagai pembanding, tinggal ganti nama kolom.
df["rain"] = df["PRECTOTCORR"]

# Isi missing value curah hujan dengan interpolasi linear (aman untuk time-series pendek)
df["rain"] = df["rain"].interpolate(method="linear").fillna(0)
df["T2M_MAX"] = df["T2M_MAX"].interpolate(method="linear")
df["T2M_MIN"] = df["T2M_MIN"].interpolate(method="linear")
df["GWETTOP"] = df["GWETTOP"].interpolate(method="linear")
df["WS2M"] = df["WS2M"].interpolate(method="linear")

print(f"\nJumlah baris setelah cleaning: {len(df)}")


# =========================================================================
# 2. FEATURE ENGINEERING
# =========================================================================
print("\n>> Feature engineering...")

# Curah hujan kumulatif (rolling sum) -- ini fitur kunci untuk potensi banjir,
# karena banjir biasanya dipicu akumulasi hujan beberapa hari, bukan cuma hari itu saja
df["rain_cum_3d"] = df["rain"].rolling(window=3, min_periods=1).sum()
df["rain_cum_5d"] = df["rain"].rolling(window=5, min_periods=1).sum()
df["rain_cum_7d"] = df["rain"].rolling(window=7, min_periods=1).sum()

# Antecedent Precipitation Index (API) -- indeks kelembaban tanah dari hujan sebelumnya
# semakin tinggi k, semakin besar pengaruh hujan hari-hari sebelumnya
def compute_api(rain_series, k=0.85):
    api = np.zeros(len(rain_series))
    api[0] = rain_series.iloc[0]
    for i in range(1, len(rain_series)):
        api[i] = rain_series.iloc[i] + k * api[i - 1]
    return api

df["API"] = compute_api(df["rain"], k=0.85)

# Rentang suhu harian (bisa jadi proxy kondisi atmosfer sebelum hujan ekstrem)
df["T2M_range_calc"] = df["T2M_MAX"] - df["T2M_MIN"]

# Fitur waktu (musiman)
df["month"] = df["date"].dt.month
df["is_rainy_season"] = df["month"].isin([10, 11, 12, 1, 2, 3]).astype(int)  # musim hujan umum Lampung

# Lag features -- curah hujan & kelembaban tanah hari sebelumnya
df["rain_lag1"] = df["rain"].shift(1).fillna(0)
df["GWETTOP_lag1"] = df["GWETTOP"].shift(1).fillna(df["GWETTOP"].mean())

print(df[["date", "rain", "rain_cum_3d", "rain_cum_7d", "API"]].head(10))


# =========================================================================
# 3. LABELING -- KATEGORI POTENSI BANJIR (berbasis ambang curah hujan)
# =========================================================================
# Klasifikasi intensitas curah hujan harian mengikuti kategori umum BMKG:
#   Ringan       : < 20 mm/hari
#   Sedang       : 20 - 50 mm/hari
#   Lebat        : 50 - 100 mm/hari
#   Sangat Lebat : 100 - 150 mm/hari
#   Ekstrem      : > 150 mm/hari
#
# PENTING: ini PROXY, bukan label kejadian banjir aktual. Kalian bisa:
#   (a) pakai kategori ini apa adanya sebagai "tingkat potensi banjir", ATAU
#   (b) ganti logikanya kalau sudah punya data kejadian banjir riil
#       (misal: tanggal-tanggal banjir dari BPBD/BNPB -> jadi label 1,
#        sisanya label 0)
#
# Di sini kita pakai pendekatan gabungan: curah hujan HARIAN ekstrem ATAU
# curah hujan KUMULATIF 3 hari tinggi -> berpotensi banjir. Ini konsisten
# dengan temuan riset banjir Way Semangka, Lampung yang mencatat hujan
# ekstrem >200 mm/hari selama 2 hari berturut-turut sebagai pemicu banjir.

def label_potensi_banjir_4kelas(row):
    if row["rain"] > 150 or row["rain_cum_3d"] > 200:
        return "Awas"
    elif row["rain"] > 100 or row["rain_cum_3d"] > 150:
        return "Siaga"
    elif row["rain"] > 50 or row["rain_cum_3d"] > 100:
        return "Waspada"
    else:
        return "Aman"

df["potensi_banjir_4kelas"] = df.apply(label_potensi_banjir_4kelas, axis=1)

# --- PENTING (baca ini) ---
# Dengan data 1 titik x 1 tahun, kejadian hujan ekstrem itu LANGKA (rare event).
# Coba cek sendiri: distribusi 4 kelas di atas biasanya sangat timpang
# (mis. 359 hari "Aman" vs cuma 2-4 hari di kelas lain). Model ML yang
# dilatih dengan data setimpang ini gampang bias / gagal belajar kelas minoritas,
# dan classification_report bisa error karena kelas tertentu bahkan
# tidak muncul di data test.
#
# Solusi paling jujur & robust untuk skala data ini: sederhanakan jadi
# klasifikasi BINER -- "Aman" vs "Berpotensi Banjir" (gabungan Waspada+Siaga+Awas).
# Kalau nanti data kalian sudah beberapa tahun / beberapa titik, silakan
# kembali pakai label 4 kelas di atas (df["potensi_banjir_4kelas"]).
df["potensi_banjir"] = df["potensi_banjir_4kelas"].apply(
    lambda x: "Aman" if x == "Aman" else "Berpotensi Banjir"
)

print("\n>> Distribusi label potensi banjir:")
print(df["potensi_banjir"].value_counts())
print("\n(Kalau ada kelas yang jumlahnya sangat sedikit/0, sesuaikan ambang")
print(" di atas dengan kondisi curah hujan riil DAS Way Sekampung, atau")
print(" gabungkan jadi lebih sedikit kelas, misal cuma 'Aman' vs 'Berpotensi Banjir')")

# Visualisasi distribusi label (versi 4 kelas, untuk dilihat sebarannya walau tidak dipakai model)
plt.figure()
order4 = ["Aman", "Waspada", "Siaga", "Awas"]
sns.countplot(data=df, x="potensi_banjir_4kelas",
              order=[o for o in order4 if o in df["potensi_banjir_4kelas"].unique()],
              palette="YlOrRd")
plt.title("Distribusi Kategori Potensi Banjir Harian (2025)")
plt.xlabel("Kategori")
plt.ylabel("Jumlah Hari")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distribusi_label.png", dpi=150)
plt.close()

# Visualisasi time-series curah hujan + kategori
plt.figure(figsize=(14, 5))
colors = {"Aman": "green", "Waspada": "gold", "Siaga": "orange", "Awas": "red"}
plt.plot(df["date"], df["rain"], color="steelblue", linewidth=0.8, label="Curah hujan harian (mm)")
for cat, color in colors.items():
    sub = df[df["potensi_banjir_4kelas"] == cat]
    plt.scatter(sub["date"], sub["rain"], color=color, s=15, label=cat)
plt.title("Curah Hujan Harian & Kategori Potensi Banjir - Titik DAS Way Sekampung (2025)")
plt.xlabel("Tanggal")
plt.ylabel("Curah hujan (mm/hari)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/timeseries_curah_hujan.png", dpi=150)
plt.close()


# =========================================================================
# 4. SPLIT DATA & TRAINING MODEL
# =========================================================================
print("\n>> Training model...")

feature_cols = [
    "rain", "rain_lag1", "rain_cum_3d", "rain_cum_5d", "rain_cum_7d", "API",
    "T2M_MAX", "T2M_MIN", "T2M_range_calc", "WS2M", "GWETTOP", "GWETTOP_lag1",
    "PS", "is_rainy_season"
]

X = df[feature_cols].copy()
y = df["potensi_banjir"].copy()

# Encode label kategori jadi angka
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Kelas label: {dict(zip(le.classes_, range(len(le.classes_))))}")

# NOTE: karena ini data 1 tahun (time-series harian), idealnya split TIDAK
# diacak (shuffle=False) supaya tidak ada "data leakage" dari masa depan ke
# masa lalu. Tapi karena jumlah data terbatas & beberapa kelas jarang,
# di sini pakai stratified split biasa. Untuk data multi-tahun nanti,
# sebaiknya pakai TimeSeriesSplit dari sklearn.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Standardisasi khusus untuk SVM (RF & XGBoost tidak wajib, tapi tidak masalah juga)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results["Random Forest"] = y_pred_rf

# --- XGBoost ---
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    random_state=42, eval_metric="mlogloss"
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
results["XGBoost"] = y_pred_xgb

# --- SVM ---
svm_model = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
results["SVM"] = y_pred_svm


# =========================================================================
# 5. EVALUASI MODEL
# =========================================================================
print("\n" + "=" * 60)
print("HASIL EVALUASI MODEL")
print("=" * 60)

summary_rows = []
for model_name, y_pred in results.items():
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    summary_rows.append({
        "Model": model_name, "Accuracy": acc,
        "Precision": prec, "Recall": rec, "F1-Score": f1
    })
    print(f"\n--- {model_name} ---")
    print(f"Accuracy : {acc:.3f}")
    print(classification_report(
        y_test, y_pred, target_names=le.classes_, zero_division=0
    ))

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUTPUT_DIR}/ringkasan_evaluasi_model.csv", index=False)
print("\n>> Ringkasan perbandingan model:")
print(summary_df.round(3))

# Confusion matrix untuk tiap model
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (model_name, y_pred) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150)
plt.close()

# Bar chart perbandingan metrik antar model
plt.figure(figsize=(9, 5))
summary_melt = summary_df.melt(id_vars="Model", var_name="Metrik", value_name="Skor")
sns.barplot(data=summary_melt, x="Metrik", y="Skor", hue="Model")
plt.title("Perbandingan Kinerja Model (Random Forest vs XGBoost vs SVM)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/perbandingan_model.png", dpi=150)
plt.close()


# =========================================================================
# 6. FEATURE IMPORTANCE (Random Forest & XGBoost)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

rf_importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
rf_importance.plot(kind="barh", ax=axes[0], color="seagreen")
axes[0].set_title("Feature Importance - Random Forest")

xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
xgb_importance.plot(kind="barh", ax=axes[1], color="steelblue")
axes[1].set_title("Feature Importance - XGBoost")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
plt.close()

print("\n>> Fitur paling berpengaruh (Random Forest):")
print(rf_importance.sort_values(ascending=False).head(5))


# =========================================================================
# 7. SIMPAN MODEL & DATA HASIL FEATURE ENGINEERING
# =========================================================================
joblib.dump(rf, f"{OUTPUT_DIR}/model_random_forest.pkl")
joblib.dump(xgb_model, f"{OUTPUT_DIR}/model_xgboost.pkl")
joblib.dump(svm_model, f"{OUTPUT_DIR}/model_svm.pkl")
joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
joblib.dump(le, f"{OUTPUT_DIR}/label_encoder.pkl")

df.to_csv(f"{OUTPUT_DIR}/data_hasil_feature_engineering.csv", index=False)

print("\n" + "=" * 60)
print("SELESAI. Semua output tersimpan di folder:", OUTPUT_DIR)
print("=" * 60)
print("""
File yang dihasilkan:
- data_hasil_feature_engineering.csv  -> data bersih + fitur, siap dianalisis lanjut
- ringkasan_evaluasi_model.csv        -> tabel perbandingan Accuracy/Precision/Recall/F1
- distribusi_label.png                -> distribusi kategori potensi banjir
- timeseries_curah_hujan.png          -> plot hujan harian + kategori
- confusion_matrices.png              -> confusion matrix 3 model
- perbandingan_model.png              -> bar chart perbandingan performa model
- feature_importance.png              -> fitur paling berpengaruh
- model_*.pkl, scaler.pkl, label_encoder.pkl -> model tersimpan (siap dipakai ulang)
""")
