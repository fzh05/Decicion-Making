"""
=========================================================================
DOWNLOAD DATA NASA POWER -- MULTI TAHUN & MULTI TITIK, DAS Way Sekampung
=========================================================================
NASA POWER API gratis, tanpa API key, dan datanya sama sumbernya dengan
file yang kalian upload (satelit MERRA-2 / IMERG).

Kenapa ini penting:
- File yang kalian punya sekarang cuma 1 titik x 1 tahun -> hasil model
  di script 01 akurasinya kelihatan tinggi (98-100%) TAPI itu bukan bukti
  model bagus, itu karena kejadian "berpotensi banjir" cuma 6 dari 365 hari
  (imbalance parah) dan data test cuma 1 sampel kelas minoritas.
  Angka setinggi itu TIDAK BOLEH langsung diklaim sebagai hasil valid
  di jurnal tanpa data yang lebih banyak.
- Solusi: tarik data 5-10 tahun & beberapa titik yang mewakili DAS Way
  Sekampung (hulu, tengah, hilir) -> jumlah "hari ekstrem" jadi lebih
  banyak & representatif, hasil evaluasi model jadi lebih bisa dipercaya.

Cara pakai:
1. Ganti/tambah titik koordinat di POINTS di bawah (titik-titik ini contoh
   sebaran di DAS Way Sekampung -- SESUAIKAN dengan sub-DAS yang mau kalian
   teliti, misal titik dekat pos hujan BMKG atau titik tengah tiap sub-DAS)
2. Atur START_YEAR dan END_YEAR
3. Jalankan: python3 02_download_data_nasa_power.py
4. Hasil tersimpan di folder data_raw/, satu file per titik
"""

import requests
import pandas as pd
import time
import os

OUTPUT_DIR = "data_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Titik-titik contoh yang mencakup sebaran hulu-tengah-hilir DAS Way Sekampung ---
# GANTI koordinat ini sesuai kebutuhan riset kalian (misal titik pos duga
# air / pos hujan BMKG yang relevan, atau titik tengah tiap sub-DAS:
# Batutegi, Way Sekampung, Argoguruh, Margatiga, Jabung, Sekampung Hilir)
POINTS = {
    "hulu_batutegi":     (-5.05, 104.85),
    "tengah_argoguruh":  (-5.16, 105.11),   # titik yang sama dengan file yang kalian upload
    "hilir_sekampung":   (-5.45, 105.60),
}

START_YEAR = 2015
END_YEAR = 2025

PARAMETERS = "PRECTOTCORR,T2M_MAX,T2M_MIN,WS2M,GWETTOP,PS"
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def download_point(name, lat, lon, start_year, end_year):
    start_date = f"{start_year}0101"
    end_date = f"{end_year}1231"
    params = {
        "parameters": PARAMETERS,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }
    print(f">> Downloading {name} ({lat}, {lon}) {start_year}-{end_year} ...")
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    props = data["properties"]["parameter"]
    df = pd.DataFrame(props)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.reset_index().rename(columns={"index": "date"})
    df["lokasi"] = name
    df["lat"] = lat
    df["lon"] = lon

    out_path = f"{OUTPUT_DIR}/power_{name}_{start_year}_{end_year}.csv"
    df.to_csv(out_path, index=False)
    print(f"   Selesai -> {out_path} ({len(df)} baris)")
    return df


if __name__ == "__main__":
    all_dfs = []
    for name, (lat, lon) in POINTS.items():
        try:
            df = download_point(name, lat, lon, START_YEAR, END_YEAR)
            all_dfs.append(df)
            time.sleep(1)  # sopan santun ke API, jangan spam request
        except Exception as e:
            print(f"   GAGAL download {name}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(f"{OUTPUT_DIR}/power_gabungan_semua_titik.csv", index=False)
        print(f"\n>> Semua titik digabung -> {OUTPUT_DIR}/power_gabungan_semua_titik.csv")
        print(f"   Total baris: {len(combined)}")
        print("\nLangkah selanjutnya: pakai file gabungan ini sebagai input")
        print("script 01 (sesuaikan bagian load_power_data untuk format JSON->CSV ini,")
        print("kolom rain di sini adalah PRECTOTCORR, sudah tanpa header 17 baris seperti")
        print("file CSV manual dari website).")
