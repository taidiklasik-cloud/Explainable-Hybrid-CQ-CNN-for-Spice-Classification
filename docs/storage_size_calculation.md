# Storage & Component Size Breakdown

Berdasarkan analisis file lokal proyek dan `environment.yml`, berikut adalah estimasi ukuran file/komponen utama yang akan dikelola dan didistribusikan dalam arsitektur **Hybrid QCQ-CNN Orchestration**.

## 1. Estimasi Ukuran Komponen Utama

| Komponen | Ukuran | Keterangan |
|----------|-------:|------------|
| **Environment Dependencies** | **~4.81 GB** | Sangat besar. Sebagian besar ukuran berasal dari *PyTorch* dengan CUDA 11.8 (~2 GB), `pennylane-lightning`, OpenCV, dan SciPy. Sangat disarankan worker menyiapkan sisa *disk space* minimal 10 GB untuk instalasi environment. |
| **Dataset (10 Kelas Original)** | **142.27 MB** | Koleksi gambar rempah asli beresolusi tinggi sebelum tahap preprocessing. |
| **Dataset (Cleaned/Preprocessed)** | **17.54 MB** | Dataset yang telah di-resize (128x128), di-filter dari duplikat, dan dinormalisasi. Sangat ringan untuk ditransfer ke worker. |
| **Curriculum Plan Outputs** | **235.72 MB** | Agak besar karena mencakup seluruh JSON konfigurasi stage, mapping pembagian fold CV, CSV tracking, dan list index gambar untuk tiap repetisi dari Stage 1 hingga Stage 5. |
| **Worker Bundle (`04_runtime_final`)** | **0.16 MB** | Script Python inti untuk worker (`worker_loop.py`, `worker_task_template.py`, dll). Sangat ringan dan instan untuk di-copy ke node eksekusi. |
| **Checkpoint Model (`.pt`)** | **< 1.0 MB / file** | Model Shared Backbone berukuran sangat kecil (~99.000 parameter) ditambah lapisan quantum (48 parameter rotasi). File `.pt` sangat ringan untuk di-upload otomatis ke Google Drive via `rclone`. |

## 2. Implikasi Terhadap Arsitektur Jaringan (Bandwidth)

> [!TIP]
> **Distribusi Payload Cepat & Ringan**
> Ukuran bundle eksekusi (`04_runtime_final`) ditambah dengan data latih bersih (`dataset_cleaned_preprocessed`) ukurannya secara total **kurang dari 20 MB**. 

Hal ini memberikan beberapa keuntungan arsitektural:
1. **Zero-Latency Orchestration**: Proses komunikasi *task claim* dari PostgreSQL dan transfer script antar *Node Worker* tidak akan pernah menjadi *bottleneck* jaringan.
2. **Stateless Worker Ready**: *Worker node* (misalnya dari Google Colab, Kaggle, atau PC cloud) dapat men-download dataset preprocessed + script bundle dalam hitungan **< 10 detik**.
3. **Efisiensi Checkpoint Drive**: Mengingat model berukuran < 1 MB per epoch, frekuensi *upload checkpoint* interval ke Google Drive tidak akan memakan kuota bandwidth yang signifikan dan aman dijalankan bahkan setiap selesai *epoch*.

---

*Catatan: Environment size dihitung berdasarkan library di `environment.yml` yang berisi module deep learning standard. Untuk worker CPU-Only (tanpa GPU NVIDIA), ukuran environment dapat berkurang ~2 GB jika menggunakan versi `pytorch-cpu`.*
