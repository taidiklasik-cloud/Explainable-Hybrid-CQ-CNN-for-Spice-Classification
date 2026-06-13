# Panduan Deployment Worker Node (Kaggle / Colab / PC Tambahan)

Dokumen ini adalah SOP (*Standard Operating Procedure*) untuk mendistribusikan komputasi riset Anda ke PC lain, Kaggle, Google Colab, Deepnote, atau Databricks.

---

## 📦 1. Isi dari `worker_deployment_bundle.zip`

Script `bundle_worker.py` secara otomatis merangkum kerangka minimum yang dibutuhkan oleh setiap *Worker* tanpa memberatkan proses upload.

| Path di dalam Zip | Fungsi di Worker |
| :--- | :--- |
| **Konfigurasi Lingkungan** | |
| `.env` | Menyimpan kredensial rahasia (URL Database & Rclone). Harus diedit manual di Worker. |
| `requirements.txt` / `environment.yml`| Daftar *library* (PyTorch, Optuna, psycopg2) agar lingkungan worker sama persis dengan PC Induk. |
| **Kode Sumber (Logika Utama)** | |
| `03_model_architecture/` | Definisi arsitektur *Classical Fully Spatial CNN* dan *Hybrid QCQ-CNN*. |
| `patched_program_files/` | Penghubung antara worker dengan fungsi *Stored Procedure* di PostgreSQL dan Rclone. |
| `04_runtime_final/worker/` | Jantung pekerja: Berisi `worker_loop.py` dan `worker_task_template.py`. |
| **Dataset & Kurikulum** | |
| `dataset_cleaned_preprocessed/` | Berisi seluruh 2.001 gambar mentah (Tanpa perlu dipisah foldernya secara fisik). |
| `curriculum_outputs/` | "Buku Panduan" CSV (Stage 1-5 dan Holdout) yang memandu DataLoader gambar mana yang boleh dibaca. |
| **Entry Point (Eksekutor)** | |
| `notebooks/02_worker_pc_template.ipynb` | *Thin wrapper* (Tombol Start) yang Anda klik "Run All" di Jupyter/Colab si Worker. |

---

## 🛠️ 2. Setup Manual di Komputer Worker

Setelah file `.zip` di-ekstrak di PC Worker / Kaggle / Colab, Anda harus melakukan langkah manual ini **sekali saja**:

### A. Konfigurasi Rahasia (`.env`)
Buka file `.env` di *Worker* dan sesuaikan koneksinya agar menunjuk ke PC Utama (Orchestrator).

> [!WARNING]
> Jangan biarkan IP bernilai `localhost` atau `127.0.0.1` jika Worker berada di jaringan yang berbeda atau di Cloud!

```env
# Koneksi ke DB PC Utama (Gunakan IP Public/Ngrok jika Worker di Cloud, IP LAN jika di rumah)
DB_HOST=<PRIMARY_PC_IP>
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=<DB_PASSWORD>
DB_NAME=cqcnn_orchestration

OPTUNA_DB_URL=postgresql://<OPTUNA_DB_USER>:<DB_PASSWORD>@<PRIMARY_PC_IP>:5432/optuna_skripsi

# Pastikan rclone sudah tersetting dan mengarah ke Google Drive
RCLONE_CONFIG_PATH=/root/.config/rclone/rclone.conf
RCLONE_DRIVE_NAME=gdrive_skripsi
```

### B. Integrasi DataLoader Sebenarnya
Di dalam file `04_runtime_final/worker/worker_task_template.py`, pastikan DataLoader **TIDAK** meload seluruh folder `dataset_cleaned_preprocessed`. DataLoader harus di-filter oleh CSV Kurikulum.

> [!TIP]
> **Logika DataLoader yang Benar:**
> 1. Worker membaca parameter `stage_no` dari PostgreSQL.
> 2. Worker membaca file `curriculum_outputs/stage_0{stage_no}/combined_train_plan_original_plus_augmented.csv`
> 3. PyTorch Dataset `__getitem__` akan mengambil baris spesifik dari CSV tersebut, membaca nama filenya, lalu menarik gambarnya dari `dataset_cleaned_preprocessed/`.

---

## ☁️ 3. Strategi Penyimpanan & Checkpoint di Cloud Notebooks

Jika Anda menjalankan *Worker* di platform *Cloud* gratisan, mesin dapat mati kapan saja (Ephemeral Storage). Worker harus dipastikan hanya menyimpan model yang sudah **selesai total evaluasinya dalam 1 trial**.

> [!IMPORTANT]
> **Aturan Emas Checkpoint Trial:**
> Jangan meng-upload/menyimpan checkpoint yang baru setengah jalan (kecuali fitur resume diaktifkan secara ketat). Worker harus mengeksekusi 1 Trial hingga **Selesai Total**, menyimpan metrik akhirnya, meng-upload `best_model.pt` final trial tersebut ke penyimpanan eksternal, barulah memanggil `study.tell()`.

Berikut adalah strategi spesifik per platform:

### 🔹 Google Colab (Free)
*   **Kondisi:** 100% *Ephemeral* (Hilang jika *disconnect* / max 12 jam).
*   **Keuntungan Khusus:** Integrasi langsung dengan Google Drive.
*   **Aksi Spesifik:** 
    *   Gunakan perintah `from google.colab import drive; drive.mount('/content/drive')` di baris pertama notebook.
    *   Ubah direktori target *checkpoint* di `worker_loop.py` agar mengarah langsung ke `/content/drive/MyDrive/skripsi_checkpoints`. (Rclone menjadi opsional di sini).

### 🔹 Kaggle Notebooks
*   **Kondisi:** 100% *Ephemeral* (Hilang jika reset / max 12 jam).
*   **Lokasi Kerja:** Direktori wajib adalah `/kaggle/working/`.
*   **Aksi Spesifik:** 
    *   Pastikan file bundle di-*unzip* ke dalam `/kaggle/working/`.
    *   **Wajib Rclone:** Karena Kaggle tidak bisa di-*mount* persisten, Rclone adalah kunci utama. Worker harus membuang `best_trial_X.pt` ke Google Drive melalui Rclone segera setelah suatu trial dinyatakan selesai total.

### 🔹 Deepnote (Free Edition)
*   **Kondisi:** **Persisten**, tetapi dibatasi **Maksimal 5GB**.
*   **Bahaya Tersembunyi:** Batas memori 5GB akan sangat cepat penuh jika Worker menyimpan puluhan model `interval.pt` atau `best.pt`. Jika penuh, Worker akan *Crash*.
*   **Aksi Spesifik:** 
    *   Pastikan Worker selalu melakukan "Sapu Bersih". Tambahkan logika `os.remove("nama_checkpoint_lama.pt")` segera setelah Rclone selesai mengunggah checkpoint tersebut ke Google Drive.

### 🔹 Databricks Community Edition
*   **Kondisi:** Mesin (Cluster) mati otomatis setelah 1-2 jam *idle*. 
*   **Lokasi Kerja:** `/local_disk0` (Ephemeral/Sementara) vs `/dbfs/` (Persisten/Databricks File System).
*   **Aksi Spesifik:**
    *   Anda bisa menyimpan output trial sementara ke dalam direktori persisten `/dbfs/FileStore/skripsi/checkpoints/`. File tidak akan hilang meski *cluster* mati.
    *   Meski begitu, integrasi Rclone tetap disarankan agar file model Anda terpusat di Google Drive yang sama dengan Worker lainnya.

---

## 🛡️ 4. Toleransi Kegagalan (Fault Tolerance) & Checkpoint Recovery

Keuntungan utama arsitektur ini adalah kemampuannya untuk melanjutkan pekerjaan (*resume*) yang terputus di tengah jalan.

**Skenario Putus Koneksi (Disconnect):**
Misalkan *Worker* Anda sedang memproses 1 Trial yang berisi 5-Fold. Mesin tiba-tiba mati saat sedang melatih **Fold ke-5, Epoch ke-99** (dari total 100 epoch).
*   **Yang Terjadi:** *Worker* lain (atau mesin yang sama saat Anda *restart*) akan mengambil alih (*hijack*) tugas yang terbengkalai (tidak ada *heartbeat* dari database).
*   **Aksi Resume:** Berkat sistem *checkpoint* lokal dan Rclone, PyTorch **TIDAK AKAN** mengulang *training* dari Fold 1 Epoch 1. Model akan membaca file `latest.pt` (atau *checkpoint* terakhir di Google Drive), memuat ulang *model state* beserta *optimizer*-nya, dan **langsung melanjutkan training dari titik putus (misal: Fold 5, Epoch 99)**.

Itulah mengapa sangat penting memastikan logika *checkpoint* per-epoch Anda berjalan lancar agar Anda tidak kehilangan progres komputasi berjam-jam saat Cloud mematikan mesin Anda.

---

## 🚦 5. SOP Awal Setiap Berpindah Stage

Transisi dari Stage 1 ke Stage 2 (dan seterusnya) membutuhkan kerja sama antara **Orchestrator (PC Utama)** dan **Worker**.

### A. Tugas Orchestrator (PC Utama Anda)
Orchestrator harus "memukul gong" tanda stage baru dimulai.
1. Jalankan `optuna_stage_manager.py` dengan parameter stage yang ingin dijalankan (Misal: `stage_no=3`).
2. Script ini akan **menghasilkan (generate)** baris-baris tugas (tasks) baru ke dalam tabel `task` di PostgreSQL dan menyuruh Optuna menyiapkan algoritma Bayesian Optimization.
3. Status sinyal di database akan berubah menjadi `HAS_WAITING_TASK` atau `WAIT_FOR_DISPATCHER`.

### B. Tugas Worker
1. Pekerja murni pasif. Buka `notebooks/02_worker_pc_template.ipynb`.
2. Klik **Run All**.
3. Worker akan memanggil fungsi `worker_loop.py`. 
   - Jika DB merespon `HAS_WAITING_TASK`, ia otomatis mengunduh hyperparameter, men-setup PyTorch, dan memulai iterasi epoch.
   - Jika DB merespon `WAIT_FOR_DISPATCHER`, Worker akan *sleep* secara otomatis selama beberapa detik tanpa membebani sistem hingga tugas baru muncul.

---

## 🚀 6. Persiapan Khusus Migrasi ke Azure VM & Google Cloud (GCP)

Berbeda dengan Colab/Kaggle yang merupakan *managed environment*, Azure VM dan GCP Compute Engine biasanya adalah mesin kosong (Linux/Ubuntu *headless*). Sebelum Anda mengunggah *file bundle* ke sana, ada 4 persiapan wajib yang harus Anda lakukan:

### A. Jembatan Koneksi Database (Networking)
*Worker* di Azure/GCP membutuhkan akses internet untuk mencapai *Orchestrator* (PC Lokal Anda).
*   **Aksi:** Anda wajib membuka *Port Forwarding* 5432 di *router* rumah Anda, **ATAU** menggunakan *tunneling* gratis seperti **Ngrok** atau **Tailscale** agar PC Lokal Anda bisa memberikan IP Publik/Virtual kepada *Worker* Cloud untuk mengakses database PostgreSQL.

### B. Siapkan `rclone.conf` Terlebih Dahulu
Sangat sulit melakukan *login* (otentikasi *browser*) Google Drive di dalam mesin Azure/GCP yang tidak memiliki layar UI (*headless*).
*   **Aksi:** Lakukan otentikasi Rclone di PC lokal Anda terlebih dahulu. Setelah berhasil, *copy* file `rclone.conf` milik PC Anda dan masukkan file tersebut ke dalam `worker_deployment_bundle.zip`. Di Cloud nanti, Anda tinggal meletakkan file tersebut di `/root/.config/rclone/rclone.conf`.

### C. Script Instalasi CUDA & PyTorch
Mesin Azure/GCP datang tanpa *library*. File `environment.yml` Anda harus solid.
*   **Aksi:** Pastikan Anda menyertakan perintah instalasi **Miniconda** dan instalasi spesifik PyTorch yang cocok dengan GPU yang Anda sewa (misal: Nvidia T4 atau L4 di GCP membutuhkan versi CUDA tertentu, biasanya diatur via perintah `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`).

### D. Beralih dari Notebook ke Terminal (Tmux / Screen)
Anda tidak akan menggunakan Jupyter Notebook (`02_worker_pc_template.ipynb`) di Azure/GCP. Jika koneksi SSH Anda ke server terputus, proses di notebook akan mati seketika.
*   **Aksi:** Gunakan aplikasi `tmux` atau `screen` di terminal Ubuntu. Di dalam sesi *tmux* tersebut, eksekusi kode Anda murni melalui Python *script*: `python -m 04_runtime_final.worker.worker_loop`. Dengan begitu, Anda bisa menutup laptop Anda di rumah, dan *Worker* di Azure akan tetap berlatih semalaman suntuk di latar belakang (*background*).

### E. Kebutuhan Spesifikasi Hardware & Storage di Cloud
Jika Anda merakit *Worker* di Azure/GCP, ini adalah "belanjaan" yang harus Anda siapkan:
1.  **Virtual Machine (VM):** Wajib menyewa instans server (Pilih sistem operasi Linux **Ubuntu 22.04 LTS** karena dukungan pustaka AI-nya paling stabil).
2.  **GPU (Kartu Grafis):** **Sangat Wajib**. Jangan menyewa mesin CPU saja karena komputasi CNN & Quantum (QCQ) akan mandek. Ajukan kuota GPU seperti **Nvidia T4** (Paling hemat biaya) atau **L4 / A100** (Jika ada *budget* lebih).
3.  **Storage Disk (Kapasitas):** Anda tidak butuh ratusan Giga. Karena dataset Anda hanya 2.001 gambar (sangat kecil), Anda hanya butuh ruang untuk instalasi OS Ubuntu, CUDA, dan PyTorch. **SSD berkapasitas 50 GB - 100 GB** sudah jauh dari cukup. (Catatan: *Worker* akan bolak-balik menulis `latest.pt` dan membuang file ke Google Drive, sehingga hardisk lokal tidak akan menumpuk).
4.  **Struktur Direktori:** Di dalam Linux, cukup buat satu folder kerja (misal: `mkdir ~/skripsi_worker`). Anda cukup melakukan perintah `unzip worker_deployment_bundle.zip` di dalam folder tersebut. Semua skema direktori akan otomatis mengikuti struktur proyek tanpa perlu Anda susun ulang!
