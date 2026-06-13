# Indonesian Spice Image Classification with Classical CNN and Hybrid QCQ-CNN

Repositori ini berisi pipeline skripsi untuk klasifikasi citra rempah Indonesia
dengan dua model pembanding:

1. Classical Fully Spatial CNN sebagai baseline klasik.
2. Hybrid QCQ-CNN sebagai varian quantum-classical terkontrol.

Tujuan riset ini bukan membuktikan "quantum advantage". Model hybrid
ditempatkan sebagai varian CNN-QNN yang diuji secara apple-to-apple terhadap
baseline klasik dengan backbone, input, dan jumlah kelas yang sama.

## Ruang Lingkup Riset

- Dataset: citra rempah Indonesia 10 kelas.
- Input model: citra grayscale berukuran `[B, 1, 128, 128]`.
- Evaluasi: accuracy, macro-F1, balanced accuracy, confusion matrix,
  per-class precision, recall, dan F1.
- Prinsip utama: reproducible training, checkpointing, dan pemisahan jelas
  antara orchestration metadata, Optuna study, checkpoint fisik, dan output EDA.

## Fitur yang Sudah Dibangun

- Pipeline EDA bertahap untuk profiling dataset, ekstraksi fitur citra,
  audit outlier, dan justifikasi fitur.
- Desain curriculum learning Stage 1 sampai Stage 5, termasuk holdout awal,
  5-fold CV, HPO Stage 3-4, dan evaluasi final Stage 5.
- Arsitektur Classical Fully Spatial CNN dengan shared CNN backbone,
  BlurPool, GroupNorm, CBAM attention, bottleneck spasial, dan head spasial.
- Arsitektur Hybrid QCQ-CNN dengan shared CNN backbone, spatial collapse,
  amplitude encoding, StronglyEntanglingLayers, Pauli-Z expectation readout,
  dan linear classifier 10 kelas.
- Integrasi Optuna PostgreSQL untuk study, trial, sampled hyperparameters,
  objective value, dan best trial.
- Orchestration PostgreSQL untuk stage, task, worker, heartbeat, checkpoint
  metadata, resume, dan stale-task hijacking.
- Worker runtime dengan polling stage signal, task claim, heartbeat,
  checkpoint-before-tell order, dan `study.tell()` setelah hasil training siap.
- Checkpoint helper untuk `latest.pt` lokal per worker, interval/best/final
  checkpoint, SHA-256, ukuran file, dan metadata resume.
- Script preflight, smoke test, dan dokumen setup untuk validasi lingkungan.
- Struktur `outputs/` untuk CSV, JSON, plot, executed notebook, dan artifact
  eksperimen lokal tanpa memasukkannya ke database.

## Pemisahan Penyimpanan

| Komponen | Peran | Status publikasi |
| --- | --- | --- |
| Orchestration PostgreSQL | Stage, task, worker, heartbeat, checkpoint metadata | hanya DSN di `.env`, tidak dipublikasi |
| Optuna PostgreSQL | Study, trial, hyperparameter, objective, best trial | hanya DSN di `.env`, tidak dipublikasi |
| Checkpoint `.pt` | Bobot model dan state training | file fisik tidak masuk Git |
| `outputs/` | CSV, JSON, plot, notebook tereksekusi | lokal, tidak masuk Git kecuali README |
| Dataset | Citra mentah/olah | tidak masuk Git |

Implementasi checkpoint yang ada saat ini menyediakan backend rclone/Google
Drive. Jika deployment akhir memakai Supabase Storage, simpan credential di
`.env` atau secret manager dan tetap pertahankan prinsip yang sama: PostgreSQL
hanya menyimpan metadata, sedangkan file `.pt` berada di object storage.

## Struktur Direktori Utama

```text
01_eda_pipeline/              Notebook dan output EDA lokal
02_curriculum_pipeline/       Curriculum split dan audit stage
03_model_architecture/        Definisi arsitektur classical dan hybrid
04_runtime_final/             Optuna orchestrator, worker, dan DB runtime
patched_program_files/        Modul Python yang dipakai notebook/runtime
notebooks/                    Entry point orchestrator dan worker
sql/                          Schema, function, seed, dan readiness checks
docs/                         Dokumentasi setup dan perhitungan eksperimen
tests/                        Smoke test, integrity test, dan worker probes
outputs/                      Output lokal, tidak dipublikasi
```

## Menjalankan Pipeline

1. Buat environment Python dari `requirements.txt`.
2. Salin `.env.example` menjadi `.env`, lalu isi nilai private secara lokal.
3. Jalankan SQL sesuai `sql/README.md`.
4. Ikuti urutan lengkap pada `RUN_ORDER_END_TO_END.md`.
5. Untuk detail alur dari EDA sampai worker, baca
   `WORKFLOW_DETAIL_FROM_EDA_TO_WORKER.md`.
6. Untuk publikasi GitHub dan privasi, baca
   `docs/GITHUB_PUBLICATION_GUIDE.md`.

## Catatan Klaim Ilmiah

- Novelty utama: desain komparasi terkontrol antara baseline CNN spasial dan
  varian Hybrid QCQ-CNN untuk klasifikasi citra rempah.
- Novelty aplikatif: pipeline klasifikasi citra rempah Indonesia dengan EDA,
  curriculum learning, dan evaluasi multi-metrik.
- Engineering contribution: orchestration berbasis PostgreSQL, Optuna RDB
  storage, worker execution, heartbeat, checkpoint metadata, dan recovery.
- Future work: validasi statistik lintas seed, perbandingan backend quantum,
  integrasi storage final, dan perluasan XAI pada checkpoint terbaik.

Semua klaim performa harus diturunkan dari hasil eksperimen yang sudah
direproduksi. Repositori ini tidak mengklaim quantum advantage tanpa bukti
empiris dan analisis statistik yang memadai.
