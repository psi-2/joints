# Status saat ini (7 Oktober 2026)

## Sedang di mana

- **v15 public 0,39918 (terbaik); v14 LightGBM saja 0,41283. Panitia: pretrained rilis kapan pun boleh, kompresi boleh (berkas ≤ 200 MB). v18 siap di-run** (`docs/v18_plan_2026-10-08.md`): partial-window + TabPFN-3.5 int7 + Causilo int7 (198,4 MB).

- **v15 sudah di-run: TabPFN-3.5 int7 0,9 + LGB 0,1, gain vs kontrol -0,0117 (CI tidak melewati 0), prediksi uji paling mirip v8; disarankan disubmit.** Rencana awal: **v15 siap di-run** (`notebooks/build_v15.py`): CV kohort minggu rilis, TabPFN-3.5 penuh int7 (169 MB), Causilo log1p (per horizon dan konteks penuh), Fast, TabM. Lihat `docs/v15_plan_2026-10-08.md`.

- **v14 sudah di-run: thinning gagal di model; final LightGBM saja, validasi lebih buruk dari v13 (TW +0,005), tidak disarankan disubmit.** Sebelumnya: **8 Okt: v13 public 0,40244. v14 siap di-run** (`notebooks/build_v14.py` -> `notebooks/v14.ipynb`): binomial thinning
  (periode uji = pasar yang sama dengan pembeli lebih sedikit, terverifikasi di label proxy periode uji), lensa utama dunia
  validasi thinned. Ringkasan: `docs/v14_thinning_breakthrough_2026-10-08.md`, script `eda/79`-`eda/90`.

- v12 sudah di-run dan disubmit: **public 0,41060**, lebih buruk dari v8 (0,39991).
- User menduga public dan private akan "jomplang" dan private lebih baik. Itu belum bisa diverifikasi; skor
  private baru keluar setelah lomba.
- **v13 sudah di-run** (review: `eda/74`, `docs/v13_deep_analysis_2026-10-07.md`): R1 level relatif ditolak, final B1 +
  Fast FP16 0,5 / LGB 0,2 / TabM 0,3; TW 0,3267 (setara v12, CI melewati 0), temporal 0,2994. Rencana awal:
- v13 (`notebooks/build_v13.py` -> `notebooks/v13.ipynb`, belum di-run). Dasarnya analisis
  `eda/65`-`eda/73`, ringkasan di `docs/v13_deep_analysis_2026-10-07.md`: fitur level absolut (`log_s`, `f_logT`)
  diganti level relatif terhadap film lain +-14 hari, diputuskan dengan proxy periode uji dan lensa pasar sepi;
  kandidat TabPFN-3.5 Fast FP16 (167 MB). Setelah run, taruh hasil di `results/v13/` lalu putuskan submit.

## Kandidat submisi final

Di Kaggle biasanya bisa memilih 2 submisi final untuk private.

| Kandidat | Public | Validasi TW (bfd) | Kepatuhan | Catatan |
| :--- | :--- | :--- | :--- | :--- |
| v8 | 0,39991 | 0,3251 | Bermasalah: TabPFN-3.5 876 MB > 200 MB; cutoff tanggal checkpoint juga tidak jelas | Tidak bisa dipaketkan dalam ZIP 200 MB kecuali panitia menyatakan checkpoint unduhan tidak dihitung |
| v12 | 0,41060 | 0,3257 (temporal 0,3019) | Bobot 72 MB, checkpoint tertanam; TabPFN-2.5 rilis Nov 2025 (cutoff tidak jelas) | Validasi paling seimbang di antara versi patuh |
| v9 | tidak diketahui | 0,3261 | 109,6 MB; EXAONE rilis Agu 2026 | Cek apakah pernah disubmit |
| v11 | tidak diketahui | 0,3319 | 25 MB, **hanya LightGBM: paling aman** dari sisi aturan pretrained | Validasi terburuk dari kandidat |

Untuk ZIP final yang aman dari semua tafsiran aturan: LightGBM saja (v11) atau TabPFN v2 revisi Juni 2025
(`FM_MODE = 'tabpfn_v2'` di builder v12). Keputusan tergantung jawaban panitia (lihat
`07_langkah_berikutnya/sisa_pekerjaan.md`).

## Yang sudah selesai

- Pipeline lengkap dan teruji: rekonstruksi, fitur, validasi multi-lensa, hurdle, foundation model, analog
  Lebaran, dan paket bobot ≤ 200 MB dengan SHA-256.
- 60 script EDA (`eda/01`-`eda/61`, tanpa 11) dengan angka dan gambar di `outputs/eda/`. Temuan tercatat di
  `docs/eda_findings.md`.
- Notebook v1-v12 beserta builder-nya.

## Yang belum selesai atau tertunda

- Konfirmasi panitia: cutoff checkpoint, 200 MB untuk unduhan runtime, dan cakupan "AI Agent".
- Causilo belum dieksplorasi tuntas: hanya 1 konfigurasi, hasil janggal (TW 0,3954, 52 detik/fold,
  korelasi galat 0,75). Saya menawarkan diagnosis keluaran, perbandingan λ 0 vs 0,5, dan konteks gabungan.
  User belum menjawab.
- Pemilihan 2 submisi final di Kaggle, dan ZIP final (nama tim, notebook, bobot ≤ 200 MB).
- Isi nama tim dan anggota di cell banner notebook: masih placeholder `[Nama Tim]`.

## Pembaruan 9 Oktober 2026

- Terbaik public: v10a teman **0,38111** (= v8 + `LEB_RHO` 0,92 yang diukur dari train). v15 0,39918, v18 tidak disubmit.
- v19 siap run (`notebooks/build_v19.py`): resep v18 + blok anomali Juni 1-16 dibuang + rho terukur + rasio klaster Lebaran
  gabungan kapasitas. Detail dan semua angka: `docs/v19_holiday_analysis_2026-10-09.md`, script `eda/120`-`eda/127`.
- Pertanyaan terbuka ke teman: sumber angka `truth` di `eda/geografi_mudik/mudik.py` (label Lebaran 2026 tidak boleh dipakai).
