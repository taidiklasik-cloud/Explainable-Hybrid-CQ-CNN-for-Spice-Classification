# Output folder: Optuna PostgreSQL rolling orchestrator

Notebook source: `01_optuna_orchestrator_postgres.ipynb`

Simpan semua output lokal dari notebook ini di folder ini. Output notebook **tidak masuk PostgreSQL**.

Subfolder standar:
- `csv/` untuk tabel ringkas dan manifest CSV.
- `json/` untuk konfigurasi, audit log, dan metric summary.
- `figures/` untuk plot PNG/SVG/PDF bila ada.
- `logs/` untuk log eksekusi lokal.
- `reports/` untuk markdown/html/txt summary.
- `manifests/` khusus notebook curriculum bila ada.
- `executed_notebook/` untuk hasil notebook yang sudah dijalankan.
- `checkpoints_temp/` hanya untuk cache/recovery lokal; checkpoint interval, best, dan final di-upload ke Google Drive via rclone.

