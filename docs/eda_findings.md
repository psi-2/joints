# Temuan EDA dan Keputusan Preprocessing

Semua angka di bawah berasal dari script di `eda/` (jalankan dengan `.venv/bin/python eda/NN_*.py`,
gambar tersimpan di `outputs/eda/`). Modul bersama: `src/common.py` (load, rekonstruksi D1, simulasi,
MASE), `src/features.py` (fitur), `src/model.py` (LightGBM).

## 01_data_audit.py: integritas dan cara panitia membangun test

| Temuan | Angka | Keputusan |
| :--- | :--- | :--- |
| Data bersih | 0 NA, 0 duplikat (tanggal, klaster, film), 1 klaster = 1 kota | tidak perlu imputasi |
| **Aturan seleksi pasangan** | 10.373/10.373 pasangan test laku di D3; 1.450 pasangan yang terakhir laku di D1/D2 semuanya dibuang | simulasi train WAJIB memakai aturan "laku di D3" |
| D1 = tanggal rilis resmi | 10 film test sudah ada di train sebagai pratinjau (1 sampai 10 klaster) | D1 direkonstruksi dari lonjakan cakupan, bukan transaksi pertama |
| D1 hampir selalu Rabu/Kamis | Rabu 67, Kamis 65, Jumat 24, Sabtu 5, Selasa 2 | fitur `d1_dow` + normalisasi kalender |
| Klaster baru | 4 klaster test tidak ada di train (602 baris) | fitur klaster boleh NaN |

## 02_target_scale.py: apa yang sebenarnya dinilai MASE

- MASE = MAE pada rasio `r = y / s`. Median `r` = 0,42, rata-rata 0,61 (menceng kanan) sehingga
  **objective harus L1**; konstanta terbaik `k*s` dengan k = 0,42 (MASE 0,504), naive `s` = 0,731.
- 31% target nol; porsi nol naik dari 10% (D4) ke 54% (D10) karena film dicopot.
- Pasangan kecil (skala < 20) menyumbang galat per baris terbesar tetapi hanya 15% baris.

## 03_release_lifecycle.py: rekonstruksi D1 dan verifikasi terhadap test

- Aturan D1 "cakupan pertama >= 50% puncak" tanpa filter: cakupan D3 median 29 klaster vs test 70,
  Rabu/Kamis 71% vs 81%. **Secara teori benar, secara distribusi salah.**
- Penyebab: test hanya berisi **rilis luas**. 56 judul di `movies.csv` (Bollywood terbatas, konser
  K-pop, nobar) tidak ada di train maupun test; film 2D terkecil di test dibuka di 28 klaster.
- Setelah filter film dasar >= 25 klaster pada D1 dan D1 dihitung di level judul dasar (2D/3D/IMAX
  berbagi D1): cakupan 48 sampai 51, Rabu/Kamis 78%. Sisa selisih berasal dari musim (lihat 07).
- Kurva nasional median (/rata-rata D1 sampai D3): D4 0,84, D7 0,38, D10 0,17. Momentum D3/D1
  hanya berkorelasi 0,18 dengan retensi D4 sampai D10: prediksi tidak bisa sekadar ekstrapolasi tren.

## 04_calendar.py: efek kalender dan periode yang tidak ada di train

- Pengali hari: Sen-Jum 0,78 sampai 0,85, Sabtu 1,29, Minggu 1,28.
- Libur di hari kerja = level akhir pekan (Hari Buruh 1,85x, Waisak 1,88x, Kenaikan 1,98x hari yang
  sama); libur di akhir pekan hampir tidak menambah.
- Libur sekolah: hari kerja +15%.
- **30% baris test ada di periode tanpa padanan di train**: Ramadan 17%, Natal/Tahun Baru 7%, Lebaran 6%.
  Pohon tidak bisa ekstrapolasi, sehingga kalender dimasukkan sebagai **pengali target struktural**
  `c_mult = c(t) / mean c(D1..D3)`.
- Volume D1 sampai D3 film rilis Ramadan setengah dari film lain, TETAPI film yang D1 sampai D3-nya
  melintasi awal Ramadan (CRIME 101, BLADES OF THE GUARDIANS) tidak menunjukkan penurunan rasio.
  Kesimpulan: penurunan volume adalah efek seleksi (film besar menghindari Ramadan), bukan efek
  kalender, sehingga `RAMADAN_F = 1.0`.
- Perbaikan hasil verifikasi: pada versi pertama, cuti bersama 18 sampai 20 Maret 2026 (hari mudik di
  akhir Ramadan) dianggap "hari libur tinggi" sehingga prediksi minggu Lebaran justru turun. Uplift
  libur kini dimatikan di dalam Ramadan.

## 05_cinema_city.py: klaster

- Ukuran klaster timpang (p10 539, p90 5.345 tiket/hari); 5 klaster terbesar = 23% tiket.
- Residual klaster (r - median film) punya korelasi split-half 0,29, tetapi target encoding OOF
  memperburuk CV (lihat 09), jadi tidak dipakai.
- Harga tiket kota berkorelasi lemah dengan ukuran klaster (0,15); dipakai sebagai fitur ringan.

## 06_movie_meta.py: metadata film

- Retensi film (mean D4-10 / mean D1-3) median 0,43; Family 0,12, Romance 0,17, Adventure 0,78,
  IMAX 0,58. Sinyal ada tetapi kecil; dipakai sebagai flag genre dan rating.

## 07_preprocessing_check.py: verifikasi preprocessing

- Skala dari `build` identik dengan fungsi `hitung_skala` resmi; urutan `id` test terjaga.
- Adversarial validation **harus di-group per film**: split acak memberi AUC palsu 1,0 karena fitur
  level film konstan dalam satu film.
- AUC grouped: fitur level absolut 0,82, fitur relatif 0,66. Periode test adalah musim sepi
  (okupansi median 10% vs 22%, tiket/pertunjukan 18,6 vs 32,9) sehingga fitur level absolut dibuang
  kecuali `log_s` dan `f_logT`.

## Pembersihan data (ditemukan di 08)

- Outage pelaporan di train: 7, 9, 10, 16 Juni 2025 (1 sampai 67 klaster) dan 13 Juni hilang total.
  Zero-fill menciptakan nol palsu. Baris target di tanggal tersebut dibuang dan film yang D1 sampai
  D3-nya kena outage tidak dijadikan sampel.

## 08_baselines_cv.py / 09_model_improve.py: validasi dan eksperimen

| Varian | GroupKFold | Temporal |
| :--- | :--- | :--- |
| median r/c per (horizon, hari D1) | 0,422 | 0,388 |
| LightGBM r, semua fitur | 0,363 | 0,342 |
| LightGBM r/c, fitur relatif + log_s + f_logT (+ kalender resmi) | **0,358** | **0,328** |
| + augmentasi D1 +-1 hari | +0,007 (lebih buruk) | |
| + target encoding klaster OOF | +0,0025 (lebih buruk) | |
| CatBoost MAE / XGBoost MAE | 0,372 / 0,379 | |
| blend LGB+CB+XGB | 0,366 | |

Noise floor CV (3 penugasan fold berbeda): std 0,003.

## 10_tabpfn.py: model pretrained

TabPFN v2 (Prior-Labs/TabPFN-v2-reg, Nature 9 Jan 2025, lolos batas 30 Sep 2025). Foundation model
deret waktu (Chronos-Bolt, TimesFM, Moirai) tidak dipakai karena tiap seri hanya punya 3 titik.
TabPFN-2.5 (Nov 2025) tidak boleh dipakai karena terbit setelah batas.

| Model (split temporal, sub-sampel 5.000 baris) | MASE |
| :--- | :--- |
| LightGBM 3 seed | 0,3373 |
| TabPFN v2, median prediktif, konteks 5.000 | 0,3301 |
| 0,3 LightGBM + 0,7 TabPFN | **0,3291** |

Korelasi prediksi 0,92. Di notebook, konteks 10.000 di GPU T4 dan bobot blend dipilih otomatis dari
argmin kurva temporal.


---

# v3: Mengapa CV 0,30 tetapi public LB 0,45 (script 12 sampai 19)

## 12_gap_analysis.py: jarak CV -> LB sebagian besar adalah komposisi

| Estimasi | LightGBM | v2 (TabPFN-3.5) |
| :--- | :--- | :--- |
| OOF biasa | 0,328 | 0,300 |
| OOF dibobot ke komposisi skala uji | 0,414 | 0,377 |
| Density-ratio adversarial | 0,398 | - |
| Public LB | ~0,456 (v1) | 0,4506 |

Uji memuat 13% baris ber-skala <= 20 (train-sim 3%); galat bucket itu 0,6 sampai 2,6. Metrik keputusan v3 =
**TW-MASE** (`src/evaluate.py`).

## 13_proxy_test_period.py: validasi berlabel di dalam periode uji

Tugas proxy D1,D2 -> D3 bisa dihitung di `test_history`. Galat per bucket skala train CV vs periode uji
hampir sama; setelah dibobot ke skala uji, CV (0,498) bahkan sedikit di atas periode uji (0,474).
Level shift kecil: k* = 1,06 di uji (Februari 1,16), gain hanya 0,002.

## 14_small_pairs.py: pasangan kecil

- Bobot komposisi uji di training: memperburuk (0,4094 -> 0,4136).
- Film rilis terbatas sebagai data latih tambahan: membantu (-0,0035), konsisten di 2 seed fold.
- Thinning binomial: **cek prepro gagal** - zero-rate pasangan kecil hasil thinning 0,52 vs asli 0,76
  (thinning meniru penonton sedikit, bukan bioskop mencopot film). TW-MASE memburuk (0,4171).
- Pasangan s <= 20: 76% target nol, model ~ prediksi nol; galat bucket ini hampir tak tereduksi.

## 15/16: fitur tambahan

Kompetisi per klaster pada tanggal target (film baru di jendela D1-D3 di klaster yang sama, data resmi):
zero-rate naik monoton 0,13 -> 0,73 menurut kuintilnya, distribusi train/uji sama. Tetapi pada ablation
2 seed fold, semua grup (jadwal program, first-day, pasar relatif, kompetisi) berada di dalam noise
(~0,0035). LightGBM jenuh di sisi fitur.

## 17/18: arti "kecil" bergeser

Pada skala absolut yang sama, zero-rate D3 di periode uji jauh lebih kecil (20-50: 0,39 train vs 0,11 uji).
Di train, kecil = film gagal; di uji, kecil = pasar sepi. Skala relatif pasar menyejajarkan distribusi
(`log_s` bergeser 0,67, `pair_vs_mkt` 0,14). Bukti validasi bertentangan: proxy periode uji memilih fitur
"both" (0,467 vs 0,4735), tetapi GroupKFold (+0,0045) dan transfer ramai->sepi dalam train (+0,012)
memilih "abs". Bulan sepi train tidak cukup sepi (zero-rate hanya turun 0,78 -> 0,72). Keputusan: default
`abs`, varian `both` diuji A/B di LB.

## 19_segment_stakes.py: segmen kalender tanpa padanan di train (30% baris)

Sisa jarak ~0,07 membutuhkan level asli 2 sampai 3x dari asumsi model di Ramadan dan/atau Lebaran
(k=3: Ramadan +0,066, Lebaran +0,024). v2 memprediksi ~0 untuk 54% baris Ramadan. Tidak ada probing
leaderboard: asumsi segmen harus diputuskan dari validasi dan logika domain di dalam notebook.

## Model pretrained: kepatuhan batas 30 Sep 2025

| Model | Revisi / versi | Tanggal | Status |
| :--- | :--- | :--- | :--- |
| TabPFN v2 reg | HF `213f8e38`, `tabpfn==2.1.4` | 11 Jun 2025 / 11 Sep 2025 | dipakai v3 |
| LimiX-16M | HF `4fd2dbad` | 1 Sep 2025 | patuh, belum diuji |
| TabDPT 1.1 | HF `514eadca`, `tabdpt==1.1.5` | Agu/Sep 2025 | gagal jalan (faiss) |
| TabPFN-2.5 | - | 6 Nov 2025 | melanggar |
| TabPFN-3.5 (dipakai v2) | `tabpfn-v3.5-20260909` | Sep 2026 | **melanggar** |

---

# v4: Sumber offset LB tetap ~0,07 (script 20 sampai 25)

Titik LB: v1 0,45625; v2 0,45061 (TW 0,377); v3 0,45731 (TW 0,388). TW-MASE memberi peringkat yang sama
dengan LB tetapi offset ~0,07 muncul di kedua model -> kesalahan sistematis yang diwarisi dari train.

## 20_pull_policy_shift.py: bioskop di periode uji lebih jarang mencopot film

| tiket/show D1-D2 | berhenti di D3 (train) | berhenti di D3 (uji) |
| :--- | :--- | :--- |
| <= 7,8 | 0,50 | 0,22 |
| 7,8-14,8 | 0,26 | 0,06 |
| 14,8-25 | 0,085 | 0,03 |

Bukan musim: November/Desember 2025 punya level pasar setara bulan train (153/259 tiket per klaster-hari)
tetapi zero-rate D3-nya tetap rendah. Bukan data bolong: pola kehadiran D1-D3 train vs uji mirip
(111: 78% vs 80%), lubang di luar outage Juni median 0,25%.

## 21_pull_odds_correction.py: besar pergeseran

Rekalibrasi logit P(D3 = 0) di periode uji: a = -1,27 (odds x 0,28), sd antar split-half 0,075,
konsisten per bulan. Di D3 sendiri koreksi tidak menurunkan MASE (nol hanya 8%).

## 22_d3_propensity_transfer.py: jembatan D3 -> D4-D10 lemah

Di train, kecenderungan mencopot di D3 per klaster x bulan hampir tidak memprediksi pencopotan D4-D10
(korelasi 0,13, slope 0,02). Pergeseran D3 tidak bisa diasumsikan berlaku penuh di horizon target.
Juga dicek: hipotesis "pencopotan karena kapasitas layar" tidak didukung (penurunan terjadi dengan atau
tanpa film baru di klaster).

## 23_holiday_calibration.py: kalender libur tidak meremehkan

Grid HOL_LEVEL x SCHOOL_WD: TW-MASE datar 0,4062-0,4108 (noise); bias di baris target hari libur ~0.
Parameter kalender tetap.

## 24_pull_shift_decision.py: model hurdle + analisis keputusan

Hurdle (klasifier p0 + 19 regresi kuantil, median campuran) vs L1, TW-MASE train OOF:
L1 0,4087 -> hurdle 0,3874 (-0,021, jauh di atas noise 0,0035). Dunia bergeser (pencopotan dibatalkan
sesuai a): L1 memburuk ke 0,4479, hurdle lambda=1 0,4211.

## 25_hurdle_lambda_proxy.py: validasi di periode uji membalik

Proxy D3 periode uji (held-out per film, 10 split): hurdle murni kalah dari L1 di 10/10 split
(0,481-0,485 vs 0,473). Penyebab: hurdle lebih baik pada baris yang laku (0,479 vs 0,492) tetapi jauh
lebih buruk pada baris yang ternyata nol (0,491 vs 0,275). Blend menang di kedua sisi:

| proxy D3 | L1 | blend 50% hurdle lambda 0 | lambda 0,5 | lambda 1 |
| :--- | :--- | :--- | :--- | :--- |
| periode uji held-out | 0,4734 | 0,4728 | 0,4701 | 0,4680 |
| train-sim OOF | 0,4219 | 0,4172 | - | - |

Tugas utama (TW-MASE), baris lambda x kolom bobot hurdle -> minimax regret atas dunia nyata, dunia
bergeser, dan proxy periode uji memilih **bobot hurdle 0,75, lambda 0,5** (regret maks ~0,008).

## Model pretrained lain yang dicek (patuh batas 30 Sep 2025)

- LimiX-16M: output point prediction, butuh flash-attn 2.8 (tidak mendukung T4), retrieval > RTX 4090 -> tidak layak.
- TabDPT 1.1.5: gagal dengan faiss terbaru.
- GPU lokal Intel Arc: PyTorch XPU butuh `intel-compute-runtime` + `level-zero-loader` (sudo), belum terpasang.

---

# v5: setelah v4 = 0,44809 (script 26 sampai 28)

## Kalibrasi validasi terhadap LB

| kappa (porsi pergeseran D3 yang berlaku di D4-D10) | TW v3 | TW v4 | v3 - v4 |
| :--- | :--- | :--- | :--- |
| 0 (validasi biasa) | 0,3879 | 0,3911 | -0,003 |
| 0,5 | 0,4189 | 0,4082 | 0,011 |
| 1 | 0,4526 | 0,4312 | 0,021 |

Selisih LB v3 - v4 = 0,0092 -> kappa ~ 0,45-0,5; lambda = 0,5 dipertahankan. Offset tersisa ~0,039.

## 26_test_regime_calibration.py: bagian positif sudah terkalibrasi di periode uji

PIT rata-rata 0,513 (train 0,480), rasio median aktual/prediksi 1,005. Tidak ada pergeseran level umum.
Rasio antar-hari dalam film yang sama mirip train (kecuali Sabtu/Jumat film rilis Kamis 1,19 vs 1,01,
yang sudah terlihat model lewat bentuk D1-D3).

## 27_zero_classifier_features.py: fitur untuk klasifier pencopotan

| grup | AUC | TW hurdle (nyata, lambda 0) |
| :--- | :--- | :--- |
| base | 0,9267 | 0,3874 |
| + kompetisi per klaster | 0,9304 | 0,3836 |
| + pergantian program | 0,9250 | 0,3890 |
| + first day / shows | 0,9270 | 0,3870 |
| + pasar relatif | 0,9256 | 0,3878 |

Hanya kompetisi per klaster yang dipakai, khusus di klasifier.

## 28_lebaran_analog.py: minggu Lebaran (6,1% baris uji)

- Slate Lebaran 2026 (7 judul, D1 = 18 Mar): D1-D3 rata-rata 167 ribu tiket/hari, ~1,0 juta kursi/hari,
  okupansi 15-23%.
- Lebaran 2025 di train (1-7 Apr): 608 ribu tiket/hari (2,8x normal), slate okupansi median 73%;
  8-11 Apr masih 2,3-3x normal.
- v4 memprediksi slate 129 ribu tiket/hari (rasio 1,37 -> 0,25), setara okupansi ~13%.
- Rasio tersirat dari analog 2025 (g x phi): pesimis 0,42 -> 1,40; tengah 0,72 -> 2,4; per klaster median 3,9 (g=phi=1).
- Tidak bisa divalidasi di train (ekor Lebaran 2025 tidak bias: itu fase turun, bukan lonjakan).
- v5: analog top-down per klaster dengan rho = 0,5 (rasio rata-rata ~1,7).

---

# v6: setelah v5 = 0,40865 (script 29 sampai 32)

## Validasi sekarang terkalibrasi ke LB

| versi | TW real | TW kappa=0,5 | LB | offset |
| :--- | :--- | :--- | :--- | :--- |
| v3 | 0,3879 | 0,4189 | 0,45731 | +0,038 |
| v4 | 0,3911 | 0,4082 | 0,44809 | +0,040 |
| v5 | 0,3894 | 0,4066 | 0,40865 | +0,002 |

Analog Lebaran menghapus sisa offset. TW-MASE di dunia kappa=0,5 = metrik keputusan v6 (memprediksi LB).

## 29_cinema_day_demand.py: indeks permintaan klaster-hari dari film lain (ditolak)

Korelasi dengan residual v5 -0,018; TW kappa 0,4047 -> 0,4051. Klaster "absen" pada tanggal target justru
zero-rate lebih rendah (0,195 vs 0,302): bukan sinyal outage.

## 30_hurdle_tuning.py

| komponen | setelan | TW real | TW kappa |
| :--- | :--- | :--- | :--- |
| klasifier | 63 daun / 300 / 100 (v5) | 0,3898 | 0,4058 |
| klasifier | **31 daun / 300 / 100** | 0,3872 | **0,4038** |
| kuantil | 63 / 300 / 100, 19 level (v5) | 0,3870 | 0,4040 |
| kuantil | 63 / 300 / 100, 39 level | 0,3867 | 0,4038 |

## Natal-Tahun Baru: cek kapasitas (tidak dioverride)

AVATAR: FIRE AND ASH rasio 0,87, okupansi tersirat 53% (D1-D3 61%); rilis 24 Des rasio 0,43-0,58
(D1-D3 jatuh di puncak libur 24-26 Des). Masih di kisaran okupansi libur sekolah Juli 2025 (23-40%).

## Model pretrained: TabPFN-3.5

Diputuskan tim (broadcast panitia: batas 30 Sep 2025 untuk data eksternal) memakai TabPFN-3.5
(`Prior-Labs/tabpfn_3_5`, revisi `06bf2ba3`, `tabpfn==9.0.0`). Catatan teknis: unduh dengan `local_dir`,
karena loader memilih format dari akhiran `.safetensors` dan blob cache HF tidak berakhiran.

## 31/32: TabPFN-3.5 vs TabPFN v2, dan cara blending (OOF lokal, 4 anggota ensemble)

| model | TW real | TW kappa=0,5 |
| :--- | :--- | :--- |
| v5 final (Kaggle) | 0,3894 | 0,4054 |
| LightGBM mix (setelan v6) | 0,3898 | 0,4063 |
| TabPFN v2 lambda 0,5 | 0,3951 | 0,4136 |
| TabPFN-3.5 lambda 0 | 0,3826 | 0,4137 |
| TabPFN-3.5 lambda 0,5 | 0,3851 | 0,4052 |
| blend median w=0,5 | 0,3847 | **0,4025** |
| blend distribusi w=0,5 | 0,3865 | 0,4039 |

Blend distribusi (rata-rata p0 dan kuantil) secara teori lebih konsisten dengan MASE tetapi kalah tipis.
Perkiraan v6 ~ v5 - 0,003 pada metrik terkalibrasi LB.

## 33_shift_heterogeneity.py: apakah pergeseran pull seragam?

Proxy periode uji (D1,D2 -> D3), global a = -1,322. Per bucket p0 model-train, a stabil (sd split-half kecil)
dan sangat heterogen: p0 <= 0,02 a = +0,67; p0 0,4-1 a = -1,91. Per skala/hari/ukuran film, perbedaan
sebagian besar ikut dari p0. Polanya = kemiringan (slope), bukan intersep: model proxy terlalu yakin.

## 34_classifier_calibration.py: kalibrasi Platt sebelum median campuran

- Tugas utama: OOF p0 cukup terkalibrasi (Platt a = -0,03, b = 0,79).
  Platt cross-fitted tidak membantu (TW kappa 0,4063 -> 0,4064).
- Proxy: model D1,D2 -> D3 punya slope OOF 0,53. Setelah dikalibrasi, sisa pergeseran periode uji murni
  intersep (a = -1,05, b = 1,04).
- Jadi heterogenitas di 33 berasal dari miskalibrasi model proxy, bukan dari kebijakan pull periode uji.
  Pergeseran seragam lambda*a dipertahankan.

## 35/36: TabICL v2 sebagai model fondasi kedua (ditolak)

| komponen | TW real | TW kappa |
| :--- | :--- | :--- |
| LightGBM mix | 0,3898 | 0,4063 |
| TabPFN-3.5 lambda 0,5 | 0,3851 | 0,4052 |
| TabICL v2 lambda 0,5 | 0,4339 | 0,4564 |
| best simplex (0,5 / 0,5 / 0) | 0,3847 | **0,4025** |

Korelasi error LGB-TabPFN 0,988 dan LGB-TabICL 0,913. TabICL lebih beragam, tetapi jauh lebih buruk di
massa nol, jadi bobot optimalnya 0. TabDPT (faiss gagal) dan LimiX (butuh flash-attn, tidak ada di T4)
juga gugur.

## 37_error_oracles.py: di mana sisa error?

| oracle (satu bagian diketahui benar) | TW real |
| :--- | :--- |
| blend v6 | 0,3847 |
| rasio film x horizon | 0,3299 |
| rasio klaster x tanggal | 0,3252 |
| nol/non-nol | 0,3259 |

- Tiap tuas (level film, permintaan klaster-hari, pull) bernilai sekitar 0,055-0,06, tetapi hanya
  sebagai oracle penuh.
- Error terbesar per baris ada di h = 4-5 (akhir pekan pertama).
- Bucket skala <= 20 = 13% bobot uji, tetapi menyumbang 0,11 dari MASE.

## v7 (Kaggle): TabPFN-3.5 fine-tuned (ditolak)

| komponen | TW real |
| :--- | :--- |
| TabPFN-3.5 in-context, lambda 0,5 | 0,3840 |
| TabPFN-3.5 fine-tuned (CRPS, lr 1e-5, 600 s/fold), lambda 0,5 | 0,3968 |

- Korelasi error fine-tuned vs in-context 0,994, jadi tidak menambah keragaman. Bobot blend = 0.
- `submission.csv` v7 identik dengan v6 (selisih maksimum 0,0).

---

# v8: setelah v7 (= v6, 0,40119), top 1 = 0,34456 (script 38 sampai 48)

## 38_film_level_residual.py: galat level film tidak bisa ditaksir

- Rasio OOF per film (log jumlah aktual / jumlah prediksi): sd 0,72.
- Spearman terkuat: reissue -0,26, comp_maxT 0,20. Asal film (heuristik judul/pemain) tidak berbeda nyata.
- Model level film dengan grouped CV: R2 = 0,00. Koreksi x0,25 sampai x1 memperburuk TW (0,3846 -> 0,3867 sampai 0,4010).
- Breakout (SORE: T3/T1 2,29, cakupan 58 -> 87 klaster) memang terlihat sebagian, tetapi terlalu jarang untuk dipelajari.

## 39_lb_segment_inference.py: level Lebaran dari LB nyata v4 -> v5 (bukan probing)

- Perubahan LB bagian Lebaran -0,0378 dari maksimum -0,0702.
- Multiplier kebenaran m terhadap v5 konsisten 0,95-2,05, tergantung noise.
- Skala ulang override 0,8 sampai 1,25 mengubah skor harapan < 0,003. Rho = 0,5 dipertahankan.

## 40_cluster_momentum.py: momentum klaster dari film lain (ditolak)

- Tren relatif D1 -> D3 film lain di klaster yang sama, 21 hari terakhir: korelasi dengan residual -0,016.
- Uji vs train: porsi pencopotan klaster lebih rendah (0,128 vs 0,194). Ini pergeseran yang sudah ditangani lambda*a.

## 41_integer_median.py: pembulatan ke bilangan bulat

Untuk s <= 20, TW kappa -0,0005; untuk s <= 2, -0,03 di dalam bucket-nya. Efek total dapat diabaikan.

## 42_late_starters.py: pasangan yang baru laku di D3 + koreksi bobot validasi

- Late starter (first_day = 3): bimodal. 60-68% nol di D4-D10, sisanya laku penuh dengan skala kecil
  (r rata-rata 4,7-6,2 vs prediksi 0,7-0,9).
- Porsinya di bobot per bucket 4,2%, di data uji 1,8%.
- **Bobot baru = bucket skala x hari penjualan pertama.** v6 TW kappa 0,4023 -> 0,3389.
- Spesialis late starter dan target r' = y / max(s, y3) tidak membantu (0,3389 -> 0,3496 / 0,3612).

## 43_proxy_segments.py: periode uji tidak lebih sulit di D3

- Proxy berlabel periode uji: nyata 0,4735 vs harapan dari komposisi train 0,4789.
- Ramadan -0,023, Natal -0,019, per bulan +-0,03 (Des +0,068).

## 44_adversarial_weights.py: pergeseran kovariat

- AUC adversarial 0,736. Fitur utama f_occ 0,30, f_nc_trend 0,18, f_logT 0,11: film uji lebih kecil, okupansi
  lebih rendah, lebih sering menambah klaster.
- Offset LB - validasi (kappa 0,5) per lensa bobot:

| lensa bobot | offset |
| :--- | :--- |
| bucket (lama) | ~0 |
| bucket x fd | 0,062-0,067 |
| adversarial | 0,039-0,045 |

- Offset sama untuk semua versi, artinya tidak bergantung model. Penjelasan paling masuk akal: beberapa film
  breakout yang tidak bisa diprediksi di set uji (lihat 38).

## 45_period_dow.py: amplitudo mingguan (tidak meyakinkan)

- Model FE film/pasangan + decay + DOW: Sabtu/Kamis uji 1,30 vs train 1,18. Namun decay dan DOW
  terkonfundasi di jendela 3 hari.
- Bias proxy per hari D3 menunjukkan arah sebaliknya. eda/26 (kuantil positif) menemukan level terkalibrasi.
  Tidak diterapkan.

## 46_retune_corrected.py: knob di metrik baru

- Grid lambda x bobot hurdle x bobot TabPFN x pembulatan, 3 lensa x kappa 0,25/0,5/0,75 (minimax regret).
- Titik terbaik (lambda 0,5, hurdle 1,0, TabPFN 0,6) hanya -0,0014 vs v6. Lambda 0,5 tetap optimal.
  Hurdle tetap 0,75 karena proxy D3 periode uji (eda/25) memihak blend dengan L1.

## 47_tabpfn_context.py: konteks TabPFN-3.5

Fold 0, h = 4 dan 8, 2 anggota, TW-fd kappa:

| konteks | baris | TW-fd kappa | waktu |
| :--- | :--- | :--- | :--- |
| per horizon | 6,4 ribu | **0,3866** | |
| tetangga h +- 1 | | 0,3937 | |
| gabungan semua h | 45 ribu | 0,3904 | 17x lebih lambat |

Blend 30% gabungan memberi -0,0015. Tidak sepadan dengan biaya dan risiko memori T4.

## 48_hurdle_seed_bagging.py

3 seed 0,3435 vs seed tunggal 0,3432-0,3440: noise, tidak dipakai.

## Kesimpulan v8

- Tidak ada tuas sah > 0,003 yang tersisa di data ini.
- Sisa galat adalah level film (word-of-mouth) yang tidak teramati di D1-D3.
- v8 = v6 + metrik validasi yang benar komposisinya + TabPFN 16 anggota; fine-tuning dihapus.

---

# v9: setelah v8 = 0,39991, kepatuhan batas bobot model 200 MB

## Ukuran checkpoint tabular foundation model (Hugging Face, dicek 4 Okt 2026)

| model | berkas regresi | ukuran | status |
| :--- | :--- | :--- | :--- |
| TabPFN-3.5 (v6-v8) | tabpfn-v3.5-20260909.safetensors | 876 MB (fast 334 MB) | > 200 MB |
| TabPFN-3 | tabpfn-v3-regressor-v3_default.ckpt | 233 MB | > 200 MB |
| TabPFN-2.6 | tabpfn-v2.6-regressor-v2.6_default.ckpt | 51,6 MB | dipakai v9 |
| TabPFN-2.5 | tabpfn-v2.5-regressor-v2.5_* | 40,8 MB | muat |
| EXAONE-Tabular (LG AI) | exaone-tabular-regressor-v1_default.safetensors | 84,5 MB | dipakai v9 |
| TabICL v2 | tabicl-regressor-v2-20260212.ckpt | 114 MB | ditolak (eda/35) |
| LimiX-16M | LimiX-16M.ckpt | 66 MB | keluaran titik, flash-attn |
| TabDPT 1.1-1.3 | tabdpt1_*.safetensors | 252-308 MB | > 200 MB |
| Mitra | model.safetensors | 303 MB | > 200 MB |
| Google TabFM 1.0 | regression/model.safetensors | 6,6 GB | > 200 MB |

"Causilo foundation model" tidak ditemukan di Hugging Face.

## EXAONE-Tabular: catatan teknis

- Paket `exaonetabular` (commit `8638e07d`) hanya mengembalikan titik (trimmed mean) dari 999 kuantil per
  anggota. `src/exaone_q.py` membaca kuantil langsung. Uji mandiri: coverage interval 80% = 0,89.
- Default attention memakai FlashAttention, padahal T4 (sm75) tidak punya kernelnya. Notebook mengalihkan
  ke kernel memory-efficient atau math.
- CPU float32: 2303 detik per fold x horizon (2 pass SVD x 4 anggota), sehingga evaluasi lokal dihentikan.
  EXAONE dievaluasi out-of-fold di notebook Kaggle, dengan bobot blend otomatis.

## Paket bobot

- Model LightGBM dikompresi zlib.
- Checkpoint foundation model yang terpakai disimpan dalam `model_weights.pkl`, dengan cek <= 200 MB.
- Notebook mencari checkpoint di input Kaggle, lalu di pkl, baru kemudian di HF (revisi terkunci).

---

# v10: audit 5 Okt 2026 -> alat ukur dulu, satu perubahan per kandidat

## 52_talent_signal.py: identitas producer/director/writer/casts (sinyal lemah)

Track record (shrinkage) dari film lain yang berbagi nama, terhadap miss level film v8:

| field | fold-local | hanya film lebih awal | koreksi cross-fitted |
| :--- | :--- | :--- | :--- |
| casts | rho 0,27 (p 0,01; 80 film) | 0,15 (p 0,25) | TW +0,0010 |
| producer | 0,35 (p 0,05; 31 film) | 0,27 (p 0,24) | +0,0010 |
| director/writer | ~0 | ~0 | memburuk |

Catatan audit lanjutan: "past" belum menjamin label D4-D10 lengkap saat cutoff, residual OOF film sebelumnya
berasal dari model yang melihat film masa depan, dan koreksinya belum nested. Kesimpulan yang tepat:
implementasi sederhana ini belum menunjukkan manfaat. v10 hanya memakai metadata resmi bebas label
(`has_number`, `has_subtitle`, `is_reissue`, `cast_overlap`) sebagai satu ablation.

## 53_d1_rule_audit.py: aturan D1 (dikoreksi setelah review)

- Versi awal keliru dalam dua hal:
  - "Aturan kausal" R2 masih memilih segmen lewat puncak (`idxmax`).
  - Median rasio skala 1 dibaca sebagai "skala tidak berubah". Hitungan pasti: 125 pasangan bersama berubah skala.
- Aturan benar-benar kausal R3 (jendela D1-D3 sendiri >= 25 klaster per hari, tanpa puncak):
  - 6 film pindah (MINECRAFT ke pratayang 4 Apr, BELIEVE -20 hari).
  - 191 pasangan berubah skala.
  - 15 film terbuang: buka >= 25 klaster lalu runtuh, rasio D3/D1 median 0,34.
  - KS rasio cakupan ke uji 0,196 -> 0,106, tetapi hanya karena ekor bawah dibuang.
- Uji justru memuat pembukaan runtuh: 8 film punya hari < 25 klaster (SEND HELP 52, 49, 1), dan 30 film
  memiliki rasio < 0,8.
- Kesimpulan: aturan cakupan D1 (R0) konsisten dengan seleksi panitia. Dipertahankan atas dasar itu.

## Kalender sekolah wilayah

- 82% baris uji di luar Jabodetabek. Banten, Jateng, Jatim, DIY, Bali, Lampung, Sulsel libur 20/22 Des
  sampai 1-4 Jan: jendela Senin-Kamis sama dengan DKI.
- **Jawa Barat (22% baris uji) berbeda**: SE Kadisdik Jabar 14995/TU.03/PSMA (20 Juni 2025) menetapkan libur
  29 Des 2025 sampai 10 Jan 2026.
- Train Jabar (SE 21808, 10 Juni 2024: 30 Juni sampai 12 Juli 2025) = DKI pada hari kerja.
- Efek di v10 (cek prepro lokal): 1.134 baris uji (1,56%) berganti flag sekolah; di train hanya 202 baris
  akhir pekan (28-29 Juni), `cal_mult` train tidak berubah.

## Statistik klaster fold-local (cek prepro lokal)

- `cin_size` pada baris validasi bergeser median 0,052 log10 (p95 0,124). Sebagian besar ini pergeseran level
  karena ~20% film dikeluarkan. Notebook juga mencetak korelasi peringkat global vs fold-local.
- Dampaknya pada skor (B0 vs B1) diukur di notebook, tidak diasumsikan.

## Desain v10

- Satu notebook, tangga ablation di dalamnya:
  - B0: statistik global = v8.
  - B1: fold-local = baseline bersih.
  - A1: + fitur perubahan show dan tiket per show (nested cross-fitting).
  - A2: + metadata resmi.
  - C: gabungan, bila keduanya lolos.
- Aturan adopsi ditulis sebelum hasil:
  - TW grouped CV turun >= 0,0015.
  - Rata-rata temporal turun, dan tidak ada bulan yang naik > 0,003.
  - Membaik di >= 3/5 fold.
  - Dunia kappa (generator tetap dari B1) tidak naik > 0,002.
- Lambda 0,5 dan bobot hurdle 0,75 dibekukan. Sensitivitas lambda hanya dilaporkan.
- Foundation model (`FM_MODE` exaone | tabpfn_v2 rev 213f8e38 11 Jun 2025 | none) hanya pada konfigurasi
  terpilih. Bobot dipilih dengan label asli; temporal dan kappa hanya sebagai syarat tidak memburuk.
- `STRICT = True` (gagal keras); berkas bobot <= 200 MB.

---

# v11: setelah v10 = 0,40823 (dasar: docs/deep_analysis_2026-10-06.md, eda/54-61)

- v10 di Kaggle: B0 vs B1 bocor TW +0,0007. A1 (show/attendance) dan A2 (metadata) gagal aturan adopsi
  (A2 gagal di September +0,0035), sehingga terpilih B1.
- Bobot EXAONE 0,7 dipilih dari label asli. Dunia kappa terbaik di 0,4, dan public turun ke 0,40823
  (v8: 0,39991).
- v11 hanya memuat dua perubahan yang bisa dibaca terpisah:
  - Kandidat tunggal **A3**: fitur input kalender `cal_p1-3` dan `cal_log31` (tiket D1-D3 / nilai kalender
    harinya). Label dan skala tidak berubah. Median train dan uji mirip (`cal_log31` -0,42 vs -0,39).
  - Aturan bobot blend **minimax regret** atas TW label asli, temporal, dan kappa. Diterapkan ulang pada tabel
    v10, aturan ini memilih 0,5, bukan 0,7.
- Panel bioskop, show/TPS/okupansi, dan metadata dihapus dari notebook.
- Fold-local vs global `cin_size`: korelasi peringkat 0,9989. Pergeseran median 0,052 hanyalah pergeseran level.
- Revisi v11 (permintaan tim):
  - Model diganti menjadi keluarga GBDT, yaitu LightGBM, CatBoost (`MultiQuantile`), dan XGBoost
    (`reg:quantileerror`). Ketiganya berstruktur L1 + hurdle yang sama, pada frame fold dan temporal yang sama.
  - Ditambah Causilo (`causilo==1.0.3`, commit rilis `94f2bd91`, regressor 148,4 MB, kuantil native).
    EXAONE tidak bisa ditanam bersama Causilo di bawah 200 MB.
  - Blend: grid simpleks 4 komponen, filter syarat terhadap LightGBM saja, lalu minimax regret.
  - Causilo butuh torch >= 2.13; notebook berhenti di Settings bila GPU tidak terlihat setelah instalasi.
  - Batas bobot dicek sebelum inferensi Causilo final.

---

# v12: setelah v11 (submisi = LightGBM saja)

## Hasil v11 di Kaggle

| komponen | TW | catatan |
| :--- | :--- | :--- |
| A3 kalender | 0,3319 -> 0,3310 | temporal 0,3058 -> 0,3027; gagal ambang 0,0015, syarat lain lolos |
| CatBoost | 0,3367 | korelasi galat 0,993 dengan LightGBM |
| XGBoost | 0,3342 | korelasi galat 0,997 dengan LightGBM |
| Causilo | 0,3954 | sangat buruk, 52 detik per fold |

- Tidak ada blend yang lolos. Bobot 25 MB.

## Desain v12 (revisi Astra)

- Fitur B1 dibekukan selama perbandingan model.
- Urutan: kontrol LightGBM -> TabPFN-2.5 Quantiles -> TabM -> A/B kalender pada kombinasi terpilih.
- TabPFN-2.5 Quantiles: `tabpfn-v2.5-regressor-v2.5_quantiles.ckpt`, rev `6c45f3a6`, 40,8 MB, SHA-256 `6dd4dbcd…`,
  per horizon.
- TabM: `tabm==0.0.3`, k = 32, PLE 48 bin versi B, QuantileTransformer fit di bagian latih, 49 kuantil, pinball
  dibobot cal_mult, early stopping pada 10% film bagian latih.
- Koreksi pencopotan dipilih per model: lambda 0 vs 0,5, minimax regret 3 lensa.
- Blend: filter + minimax seperti v11.
- CatBoost, XGBoost, Causilo, dan EXAONE dihapus.
- Perbaikan v12 sebelum run (review Astra):
  - TabM: `compute_bins` menolak kolom konstan. `t_ramadan` dan `cin_new` konstan di train-sim. Fitur sekarang
    disaring di inner-training (terverifikasi di data nyata: 45 fitur dipakai, 2 dibuang).
  - Median imputasi, QuantileTransformer, dan bin kini di-fit hanya pada inner-training, setelah split early
    stopping.
  - Inferensi final hanya membangun komponen berbobot positif, sehingga blend tanpa LightGBM tidak lagi gagal
    assertion.
  - Dunia kappa berasal dari LightGBM, jadi pilihan lambda per model dan blend juga dicetak tanpa lensa kappa.


## 65-73 (7 Oktober 2026): level relatif dan pasar sepi

Detail lengkap di `docs/v13_deep_analysis_2026-10-07.md`. Inti: offset validasi-LB yang sama untuk semua model berasal
terutama dari fitur level absolut. Proxy periode uji: MAE absolut 0,402 vs relatif 0,370, bias +0,116 vs 0; pada skala
5-50 bias absolut +0,32 sampai +0,38. Pergeseran pencopotan D3 (logit -1,35) menyusut ke -0,21 (eda/68) atau -0,44
(definisi notebook) dengan tolok ukur relatif klaster. Transfer D3 ke nol D4-D10: kemiringan 0,09. Ramadan dan Natal sesuai
kalender; hanya Tahun Baru menyimpang (n = 3, tidak diubah).


## 79-90 (8 Oktober 2026): binomial thinning

Detail di `docs/v14_thinning_breakthrough_2026-10-08.md`. Offset val-public dijelaskan oleh komposisi, bukan bias label;
oracle klaster x tanggal adalah artefak; informasi yang hilang = lintasan film setelah D3. Thinning (pi 0,35-0,5)
mereproduksi label proxy periode uji, komposisi uji, dan 75% pergeseran pencopotan; lookup D4-D10 di dunia thinned
0,380 (latih mentah) -> 0,355 (latih thinned).

## 120-127: periode libur (9 Oktober 2026)

Ringkasan dan keputusan ada di `docs/v19_holiday_analysis_2026-10-09.md`. Diadopsi di v19:
- blok anomali 1-16 Jun dibuang (126);
- rho Lebaran terukur 0,92 dengan level nasional dikonfirmasi lewat kapasitas kursi (121);
- rasio klaster Lebaran = gabungan pola 2025 dan kapasitas 2026 (124, 127).

Ditolak:
- koreksi target hari libur (125: optimum 0,95, out-of-sample memburuk);
- profil hari masa sekolah (123: efek orde dua);
- alokasi okupansi antar film (122: slope antar film 0).
