# Local notebook outputs

Semua CSV, JSON, gambar, executed notebook, dan laporan hasil notebook disimpan lokal di folder `outputs/` ini.

Prinsip final:
- PostgreSQL lokal `cqcnn_orchestration` **tidak menyimpan artifact notebook**.
- PostgreSQL lokal hanya untuk stage/task/worker/heartbeat/checkpoint metadata.
- Google Drive via rclone menyimpan checkpoint `.pt` penting: interval, best, dan final.
- Local checkpoint cache menyimpan `latest.pt` untuk self-recovery worker.
- Output EDA, curriculum, plot, dan laporan tetap lokal agar database tidak membengkak.

Gunakan satu folder per notebook agar audit dan lampiran skripsi lebih rapi.

