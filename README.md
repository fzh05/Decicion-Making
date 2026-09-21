# Pemodelan Potensi Peringatan Dini Banjir — DAS Way Sekampung

## Isi folder
- `01_preprocessing_dan_model.py` — pipeline utama: cleaning data, feature
  engineering, labeling, training 3 model (Random Forest, XGBoost, SVM), evaluasi.
- `02_download_data_nasa_power.py` — download otomatis data NASA POWER
  multi-tahun & multi-titik (jalankan ini di komputer kalian sendiri, bukan
  di sini, karena akses internet di sandbox ini dibatasi ke domain tertentu
  dan `power.larc.nasa.gov` tidak termasuk).
- `requirements.txt` — daftar library yang dibutuhkan.
- `output/` — hasil run script 01 (grafik, tabel evaluasi, model tersimpan).

## Cara pakai
```bash
pip install -r requirements.txt
python3 01_preprocessing_dan_model.py
```

## CATATAN PENTING — WAJIB DIBACA SEBELUM MASUK KE NASKAH JURNAL

### 1. Data saat ini masih sangat terbatas
File yang kalian upload cuma **1 titik lokasi x 1 tahun (365 hari)**. Ini
cukup buat ngetes pipeline-nya jalan atau nggak, tapi **belum cukup buat
klaim ilmiah yang kuat**. Masalah konkret yang kelihatan pas kode dijalankan:

- Dari 365 hari, cuma **6 hari** yang masuk kategori "berpotensi banjir"
  (curah hujan >50 mm/hari atau kumulatif 3 hari >100 mm). Kejadian ekstrem
  itu jarang (rare event) — normal secara klimatologi, tapi bikin data
  latihnya sangat timpang (imbalanced).
- Karena timpang, model kelihatan akurasinya 98-100% — **ini BUKAN bukti
  model bagus**, itu accuracy paradox: model bisa tebak "Aman" terus buat
  semua hari dan tetap dapat akurasi tinggi, karena kelas "Aman" mendominasi.
  Metrik yang lebih jujur buat kondisi ini adalah **Recall & F1-score di
  kelas minoritas** ("Berpotensi Banjir"), bukan accuracy keseluruhan —
  itu kenapa script ini print `classification_report` lengkap, bukan cuma
  accuracy.
- Test set kelas minoritas isinya cuma **1 sampel**. Kalau ditulis di jurnal
  "model mencapai akurasi 98%", reviewer yang paham ML pasti nanya "diuji
  di berapa sampel kelas positif?" — dan jawabannya bakal kelihatan lemah.

### 2. Yang perlu dilakukan sebelum submit ke jurnal
- **Tambah data tahun**: idealnya minimal 5-10 tahun (pakai
  `02_download_data_nasa_power.py`, NASA POWER punya data dari 1981/2000-an
  sampai sekarang).
- **Tambah titik lokasi**: minimal wakili hulu-tengah-hilir DAS Way Sekampung
  (contoh titik sudah disiapkan di script 02, sesuaikan koordinatnya).
- **Kalau memungkinkan, validasi label** pakai data kejadian banjir riil
  (BNPB DIBI / BPBD Lampung / arsip berita), bukan cuma ambang curah hujan.
  Ambang di script ini (kategori BMKG: Ringan/Sedang/Lebat/Sangat Lebat/
  Ekstrem) itu **proxy**, bukan pengganti data kejadian aktual.
- Setelah data lebih banyak, boleh balik pakai label 4 kelas
  (`potensi_banjir_4kelas` di script 01) kalau tiap kelas sudah cukup
  sampelnya (idealnya puluhan sampel per kelas, minimal).

### 3. Kalau tetap mau lanjut dengan data yang ada sekarang
Boleh, tapi framing di naskah harus jujur, misalnya:
> "Penelitian ini merupakan studi awal (preliminary study/proof of concept)
> untuk menguji kelayakan pipeline machine learning dalam mendeteksi
> potensi hari-hari berisiko tinggi curah hujan di satu titik representatif
> DAS Way Sekampung, dengan keterbatasan periode data 1 tahun. Hasil ini
> perlu divalidasi lebih lanjut dengan data historis yang lebih panjang."

Ini pendekatan yang jauh lebih aman secara akademik dibanding klaim
"model mencapai akurasi 98% dalam memprediksi potensi banjir" tanpa
konteks jumlah data.
