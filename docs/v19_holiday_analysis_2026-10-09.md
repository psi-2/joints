# v19: analisis periode libur dan hasilnya (9 Oktober 2026)

Tidak ada model yang dilatih secara lokal. Semua angka berasal dari data resmi, OOF v18 yang tersimpan, dan
eksekusi cell preprocessing v19 tanpa model.

## Sumber 0.38111 (v10a teman)

`notebooks/v10a.ipynb` adalah resep v8 dengan satu perubahan: `LEB_RHO` 0,5 diganti 0,92. Nilai 0,92 diukur dari train,
yaitu porsi tiket slate Lebaran 2025 pada 1–7 April. Public turun dari 0,39991 ke 0,38111 (−0,0188), sementara 93,9% baris
lain tidak berubah.

Pengecekan ulang membuktikan nilai 0,92 sah. Isi slate-nya adalah lima film rilis 31 Maret 2025:

| Film | Porsi |
| :--- | ---: |
| PABRIK GULA (2 versi) | 33% |
| QODRAT 2 | 19% |
| JUMBO | 18% |
| KOMANG | 15% |
| NORMA | 7% |

Film lama yang masih tayang hanya mengambil sekitar 8%.

## Temuan EDA teman yang relevan

**Pergeseran utama ada di retensi, bukan di libur.** Di test, pasangan lemah lebih jarang dicopot dibanding train:
- retensi D3 untuk pasangan dengan tiket/show D2 ≤ 8: train 46% vs test 77%;
- setelah distandardisasi per klaster, selisihnya +15 poin;
- pola ini muncul di semua bulan.

Februari–Maret lebih lemah karena level pasarnya rendah. Bentuk kurva film saat Ramadan sendiri normal.

**Kalender:**
- Cuti bersama tidak dikodekan di `holidays.csv`. Ini sudah ditangani di pipeline kita sejak v5.
- Anomali 1–5 Juni (EDA 106 teman) belum kita tangani. Kini ditangani di v19.

**Perlu ditanyakan ke teman.** `eda/geografi_mudik/mudik.py` berisi `truth` = "rasio_aktual" untuk enam film Lebaran 2026,
padahal label test disembunyikan. Sumbernya harus ditanyakan. Angka ini **tidak dipakai** di v19.

## Analisis baru (`eda/120`–`eda/127`)

| Script | Temuan | Keputusan |
| :--- | :--- | :--- |
| 120 holiday level | Pasangan film lama: libur hari kerja 2,33× tetangga vs kalender 1,59×. Film nasional D2/D3: 0,97–0,99× kalender. OOF target libur MASE 0,51 vs 0,29 | Diuji di 125 |
| 121 Lebaran saturation | Okupansi slate 2025 median 73% vs 37% normal, show tetap. Kursi slate 2026 1,0 juta/hari vs 0,92 juta (2025). Total kapasitas 570 ribu vs analog ρ0,92 543 ribu | Level nasional dikonfirmasi dua jalur |
| 122 surge allocation | Rasio klaster benar: MASE Lebaran 2025 0,628 vs nasional 1,238. Okupansi di dalam klaster −0,010 (CI lewat 0). Antar film slope 0 | Pembagian antar film tidak diubah |
| 123 weekday regime | Minggu sekolah Sabtu/Sen–Kam 1,77, minggu libur 1,40, profil 1,62. D1–D3 test lonjakan Jum→Sab +0,13 log | Tidak dipakai (efek orde dua) |
| 124 cluster surge stability | 7 lonjakan libur berlabel, split-half 0,65. Skill: pola masa lalu 0,20, okupansi 0,39, rata-rata 0,44 | **Rasio klaster Lebaran = gabungan analog + kapasitas** |
| 125 holiday target calibration | Leave-one-holiday-out: optimum 0,95; TW baris libur 0,5601 → 0,5675; 3/10 tanggal membaik | **Gagal**: varians, bukan bias |
| 126 June anomaly | 2.058 baris target (5 judul) median y/s 0,05–0,29 vs 0,44–0,98 | **Blok 1–16 Juni dibuang** |
| 127 Lebaran v2 prepro | Total harian identik; faktor 0,77–1,24; Padang (Basko XXI) 1,99 | Diterapkan |

Pemeriksaan lain yang ditutup:
- Kepdis DKI 89/2025 (dokumen primer, ditetapkan 19 Juni 2025) menetapkan libur semester 20 Des–3 Jan. Ini tidak mengubah
  c(t), hanya flag `t_school`.
- Rekonstruksi label dari varian format: 0 baris.
- Hipotesis bahwa Ramadan sepi rilis film salah: median ada 6 rilis sejak D1.

## Dekomposisi gap (bukan probing, hanya aritmetika dari skor yang sudah ada)

Jika error baris non-Lebaran ≈ TW validasi (0,33–0,36), baris Lebaran (6,08%) masih menanggung error sekitar 0,6–1,0 per
baris setelah ρ = 0,92.

Backtest Lebaran 2025 memberi gambaran batasnya: dengan rasio klaster sempurna, MASE masih 0,63. Sisa error ini berasal dari
heterogenitas film, dan heterogenitas itu tidak bisa diprediksi dari D1–D3 (EDA 122: slope antar film 0; rating dan genre
lemah). Jadi gap ke top 1 (0,345) tidak terlihat bisa ditutup oleh satu tuas lain yang tervalidasi.

## v19 (`notebooks/build_v19.py` → `v19.ipynb`)

Resep v18 dipertahankan, ditambah perubahan berikut:
- `OUTAGE` = 1–16 Juni (sebelumnya 5 tanggal);
- `LEB_RHO` diukur dari train (0,92);
- `LEB_CAP_W` = 0,5 (gabungan rasio kapasitas; 0 = analog v18);
- `SCHOOL_BREAKS` Desember diganti tanggal primer.

Ekspektasi public sekitar 0,378–0,381: v8 → v18 menyumbang ~0,001–0,002, ditambah atau dikurangi ±0,003 dari pembagian
klaster Lebaran. Ini bukan breakthrough.

TW v19 tidak boleh dibandingkan langsung dengan v18, karena 2.058 baris keluar dari populasi. Bandingkan pada baris yang ada
di kedua versi.
