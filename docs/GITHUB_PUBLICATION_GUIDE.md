# GitHub Publication Guide

Dokumen ini menjelaskan cara menyiapkan repositori publik tanpa membocorkan
credential, dataset privat, output eksperimen lokal, atau checkpoint model.

## Rekomendasi Struktur Publik

Gunakan root proyek ini sebagai root GitHub repo, lalu kontrol isi publik
dengan `.gitignore`.

Alasannya:

- Struktur kode, notebook, SQL, dan dokumentasi tetap sesuai path runtime.
- Tidak ada risiko folder publik tertinggal dari perubahan terbaru.
- File privat bisa dicegah masuk Git sejak awal sebelum `git add`.

Buat folder publik terpisah hanya jika tujuannya adalah snapshot arsip statis,
bukan repo yang akan terus dikembangkan. Untuk proyek yang masih aktif, folder
duplikat biasanya lebih berisiko karena mudah tidak sinkron dengan source of
truth.

## File dan Folder yang Tidak Boleh Dipush

Jangan commit:

- `.env`, `.env.*`, DSN database, password PostgreSQL, token Supabase, dan API key.
- `.rclone/`, `rclone.conf`, token Google Drive, dan file `tskey-auth-*.txt`.
- Dataset citra: `dataset 10 kelas/`, `dataset 3 kelas/`,
  `dataset_cleaned_preprocessed/`, dan `archive/`.
- Output eksperimen: `outputs/`, `curriculum_outputs/`,
  `architecture_worker_artifacts/`, CSV, plot, dan executed notebook.
- Checkpoint dan model artifact: `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`,
  `*.joblib`, `*.npy`, dan `*.npz`.
- Environment lokal: `.conda/`, `.venv/`, `venv/`, cache Python, dan folder IDE.
- Bundle besar: `*.zip`, deployment bundle, dan source upload mentah.

## File yang Aman Dipush

Umumnya aman dipush:

- Modul Python di `patched_program_files/`, `03_model_architecture/`,
  `04_runtime_final/`, dan script test/smoke di `tests/`.
- Notebook wrapper di `notebooks/`, selama tidak berisi output sensitif.
- SQL schema/function/seed di `sql/`.
- Dokumentasi `.md` di root dan `docs/`.
- `.env.example`, karena hanya berisi placeholder.
- `requirements.txt`, `requirements-lightning-gpu.txt`, dan `environment.yml`.

## Checklist Sebelum `git push`

Jalankan dari root proyek:

```powershell
git init
git status --ignored -s
git add .
git status -s
```

Periksa `git status -s`. Jika terlihat file privat ter-stage, batalkan dari
index tanpa menghapus file lokal:

```powershell
git rm --cached -- <path>
```

Cari pola credential yang umum:

```powershell
rg -n --hidden --glob '!**/.conda/**' --glob '!**/.rclone/**' --glob '!outputs/**' --glob '!dataset*/**' --glob '!archive/**' --glob '!*.ipynb' "(password|secret|token|api[_-]?key|postgresql://|postgres://|client_secret|access_token|refresh_token|tskey-auth)" .
```

Hasil yang masih dapat diterima adalah placeholder di `.env.example` atau
dokumentasi setup. Nilai credential nyata tidak boleh muncul.

## Saran Deskripsi GitHub

Short description:

```text
Controlled Classical CNN vs Hybrid QCQ-CNN pipeline for Indonesian spice image classification with EDA, curriculum learning, Optuna HPO, worker orchestration, and reproducible checkpointing.
```

Topics:

```text
hybrid-quantum-machine-learning, cnn-qnn, pennylane, pytorch, optuna, postgresql, image-classification, reproducible-ml, checkpointing, xai
```

## Batasan Klaim Publik

Gunakan formulasi:

- "Hybrid QCQ-CNN dievaluasi sebagai varian quantum-classical terkontrol."
- "Perbandingan dilakukan secara apple-to-apple terhadap baseline CNN klasik."
- "Checkpointing dan worker runtime dirancang untuk reproducible training."

Hindari formulasi:

- "Quantum model lebih unggul secara umum."
- "Quantum advantage terbukti."
- "Model hybrid pasti lebih efisien daripada CNN klasik."

Klaim performa hanya boleh ditulis setelah hasil eksperimen final, seed,
fold, metric, dan uji reliabilitas sudah tersedia.
