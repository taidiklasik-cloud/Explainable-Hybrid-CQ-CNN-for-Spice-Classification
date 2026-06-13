# Feasibility Storage dan Pemetaan Worker  
## Hybrid QCQ-CNN Orchestration: Stage 2–5

Dokumen ini merangkum kelayakan storage lokal pada setiap worker, kebutuhan file yang perlu tersedia, strategi checkpoint lokal, serta pemetaan apakah worker lebih cocok digunakan untuk **Classical CNN** atau **Hybrid QCQ-CNN**.

---

## 1. Prinsip Umum

Arsitektur orchestration dinilai **feasible** untuk menjalankan Stage 2–5, karena ukuran dataset dan script runtime relatif kecil. Bottleneck utama bukan berada pada transfer data atau ukuran file, tetapi pada:

1. Waktu instalasi environment.
2. Stabilitas worker gratis seperti Google Colab dan Kaggle.
3. Ketersediaan GPU untuk Hybrid QCQ-CNN.
4. Manajemen checkpoint agar file lokal tidak menumpuk.
5. Pembagian task agar worker CPU yang lambat tetap produktif tanpa menahan seluruh stage.

Secara umum:

- **GPU worker** lebih baik diprioritaskan untuk **Hybrid QCQ-CNN Stage 3–5**.
- **CPU worker** lebih baik diprioritaskan untuk **Classical CNN Stage 2–5**.
- **CPU worker tetap boleh ikut Hybrid**, tetapi hanya sebagai background/slow lane dan tidak boleh menjadi bottleneck utama.
- Semua worker sebaiknya mengambil task dalam unit kecil, bukan stage penuh.

---

## 2. Dasar Workload

Berdasarkan perhitungan workload eksperimen:

| Model | Total Batch | Catatan |
|---|---:|---|
| **Classical CNN** | ±507.566 batch | Lebih ringan, cocok untuk CPU worker |
| **Hybrid QCQ-CNN** | ±2.025.243 batch | Jauh lebih berat, sebaiknya prioritas GPU |

Dari sisi ukuran model:

| Model | Parameter | Memory dasar model | Implikasi |
|---|---:|---:|---|
| **Classical CNN** | ±100.630 parameter | ±0,39 MB | Checkpoint kecil |
| **Hybrid QCQ-CNN** | ±1.148.562 parameter | ±4,39 MB | Checkpoint lebih besar karena projection layer |

Catatan penting: checkpoint Hybrid tidak sebaiknya diasumsikan selalu `<1 MB`, karena jika menyimpan optimizer state, scheduler, dan metadata training, ukurannya bisa menjadi beberapa MB hingga belasan MB.

---

## 3. Kebutuhan Storage Dasar per Worker

| Komponen | GPU Worker | CPU Worker |
|---|---:|---:|
| Environment | ±4,81 GB | ±2,5–3 GB |
| Dataset cleaned/preprocessed | ±17,54 MB | ±17,54 MB |
| Curriculum plan | ±235,72 MB | ±235,72 MB |
| Worker bundle | ±0,16 MB | ±0,16 MB |
| Cache pip/conda/torch | ±3–10 GB | ±2–5 GB |
| Checkpoint + log aktif | ±1–5 GB | ±0,5–2 GB |
| **Minimum free disk** | **20–30 GB** | **10–20 GB** |
| **Rekomendasi nyaman** | **50–100 GB** | **30–50 GB** |

Kesimpulan storage:

- **Dataset bukan bottleneck**.
- **Worker bundle bukan bottleneck**.
- **Environment dan cache instalasi adalah komponen terbesar**.
- **Checkpoint aman**, asalkan tidak menyimpan semua epoch sebagai file permanen.

---

## 4. File yang Perlu Ada di Setiap Worker

Setiap worker cukup membawa file berikut:

```text
hqcq_worker/
├── runtime/
│   ├── worker_loop.py
│   ├── worker_task_template.py
│   ├── db_client.py
│   ├── checkpoint_rclone.py
│   └── train_entrypoint.py
│
├── data/
│   └── dataset_cleaned_preprocessed/
│
├── curriculum/
│   ├── stage2_manifest.json
│   ├── stage3_manifest.json
│   ├── stage4_manifest.json
│   └── stage5_manifest.json
│
├── configs/
│   ├── environment-gpu.yml
│   ├── environment-cpu.yml
│   ├── model_config.json
│   ├── split_seed_config.json
│   └── rclone_remote_name.txt
│
├── runs/
│   └── <worker_id>/
│       └── <task_id>/
│           ├── latest.pt
│           ├── prev.pt
│           ├── best.pt
│           ├── final.pt
│           ├── metrics.jsonl
│           ├── task_state.json
│           └── train.log
│
└── uploads_pending/
```

Dataset original tidak perlu disalin ke semua worker. Worker cukup menggunakan dataset cleaned/preprocessed.

---

## 5. Estimasi Ukuran Checkpoint

| Jenis Checkpoint | Estimasi Ukuran |
|---|---:|
| Classical weights only | `<1 MB` |
| Classical full checkpoint + optimizer | ±1–5 MB |
| Hybrid weights only | ±4–5 MB |
| Hybrid full checkpoint + optimizer | ±10–20 MB |
| Metrics/log per task | Biasanya kecil, `<10 MB` |

Checkpoint tetap feasible untuk semua worker. Namun, jangan menyimpan checkpoint semua epoch secara permanen.

---

## 6. Skema Checkpoint Lokal

Gunakan pola checkpoint lokal berikut:

```text
runs/
└── <worker_id>/
    └── <task_id>/
        ├── latest.pt
        ├── prev.pt
        ├── best.pt
        ├── final.pt
        ├── metrics.jsonl
        ├── task_state.json
        └── train.log
```

### Fungsi Setiap File

| File | Fungsi |
|---|---|
| `latest.pt` | Checkpoint terbaru untuk resume |
| `prev.pt` | Backup checkpoint sebelumnya |
| `best.pt` | Model terbaik berdasarkan metric |
| `final.pt` | Model akhir task |
| `metrics.jsonl` | Log metrik setiap epoch |
| `task_state.json` | Status task, epoch terakhir, dan metadata resume |
| `train.log` | Log proses training |

### Pola Simpan yang Disarankan

Setiap epoch:

```text
1. Simpan latest.pt lokal.
2. Sebelum overwrite latest.pt, pindahkan latest lama menjadi prev.pt.
3. Simpan metrics.jsonl.
4. Update task_state.json.
```

Jika metric membaik:

```text
1. Simpan best.pt lokal.
2. Upload best.pt ke Google Drive/rclone.
```

Setiap 5 atau 10 epoch:

```text
1. Upload interval checkpoint ke Google Drive.
2. Hapus interval checkpoint lokal setelah upload sukses.
```

Akhir task:

```text
1. Simpan final.pt.
2. Upload final.pt, best.pt, metrics.jsonl, task_state.json, dan train.log.
3. Tandai task COMPLETED di PostgreSQL.
4. Bersihkan file lokal jika upload sudah terverifikasi.
```

### Hindari Pola Ini

```text
epoch_001.pt
epoch_002.pt
epoch_003.pt
...
epoch_100.pt
```

Pola tersebut aman untuk eksperimen kecil, tetapi tidak efisien untuk orchestration multi-worker karena file checkpoint akan menumpuk.

---

## 7. Feasibility Storage dan Penempatan Worker

### 7.1 Google Cloud L4

| Aspek | Penilaian |
|---|---|
| Spek | 4 vCPU, 16 GB RAM, 1× NVIDIA L4, 50 GB storage |
| Storage feasibility | Sangat feasible |
| Risiko storage | Rendah, tetapi cache perlu dikontrol |
| Lebih cocok | **Hybrid QCQ-CNN** |
| Classical | Bisa, tetapi tidak prioritas |
| Stage cocok | Hybrid Stage 3, 4, 5 |

**Kesimpulan:**  
Google Cloud L4 adalah worker utama untuk Hybrid QCQ-CNN. Storage 50 GB cukup, tetapi 100 GB lebih nyaman untuk install ulang, cache, checkpoint, dan retry.

| Stage | Rekomendasi |
|---|---|
| Stage 2 | Hybrid test |
| Stage 3 | Hybrid utama |
| Stage 4 | Hybrid utama |
| Stage 5 | Hybrid utama / rescue failed task |

---

### 7.2 Google Colab GPU

| Aspek | Penilaian |
|---|---|
| Spek | 2 CPU, ±12,6 GB RAM, 1 GPU |
| Storage feasibility | Feasible tetapi ephemeral |
| Risiko storage | Sedang-tinggi karena runtime bisa putus |
| Lebih cocok | **Hybrid QCQ-CNN** |
| Classical | Jangan diprioritaskan |
| Stage cocok | Hybrid Stage 3, 4, 5 |

**Kesimpulan:**  
Colab GPU cocok sebagai akselerator Hybrid, tetapi semua checkpoint penting harus rutin di-upload ke Drive/rclone.

| Stage | Rekomendasi |
|---|---|
| Stage 2 | Tidak perlu, simpan kuota |
| Stage 3 | Hybrid |
| Stage 4 | Hybrid prioritas tinggi |
| Stage 5 | Hybrid final evaluation |

---

### 7.3 Kaggle GPU

| Aspek | Penilaian |
|---|---|
| Spek | 4 CPU, ±31 GB RAM, 2 GPU terdeteksi |
| Storage feasibility | Feasible tetapi runtime tetap terbatas |
| Risiko storage | Sedang |
| Lebih cocok | **Hybrid QCQ-CNN** |
| Classical | Bisa, tetapi GPU sebaiknya disimpan untuk Hybrid |
| Stage cocok | Hybrid Stage 3, 4, 5 |

**Kesimpulan:**  
Kaggle GPU cocok sebagai worker Hybrid tambahan. Jika 2 GPU benar-benar dapat dipakai, jalankan 2 worker process terpisah.

Contoh:

```bash
CUDA_VISIBLE_DEVICES=0 python worker.py --worker-id kaggle-gpu-0
CUDA_VISIBLE_DEVICES=1 python worker.py --worker-id kaggle-gpu-1
```

| Stage | Rekomendasi |
|---|---|
| Stage 2 | Tidak perlu |
| Stage 3 | Hybrid |
| Stage 4 | Hybrid |
| Stage 5 | Hybrid |

---

### 7.4 PC Lokal RTX2050 RAM 32 GB

| Aspek | Penilaian |
|---|---|
| Spek | RTX2050, RAM 32 GB |
| Storage feasibility | Feasible jika free disk minimal 50 GB |
| Risiko storage | Rendah jika disk lokal cukup |
| Lebih cocok | **Classical + Hybrid ringan** |
| Classical | Sangat cocok |
| Hybrid | Bisa, tetapi terbatas |
| Stage cocok | Classical Stage 2–5, Hybrid Stage 2–3, sedikit Stage 4/5 |

**Kesimpulan:**  
PC lokal cocok sebagai worker persistent karena tidak memiliki limit platform. Namun RTX2050 sebaiknya tidak diberi beban Hybrid Stage 4 terlalu banyak.

| Stage | Rekomendasi |
|---|---|
| Stage 2 | Classical + Hybrid |
| Stage 3 | Classical + sebagian Hybrid |
| Stage 4 | Classical utama + Hybrid kecil |
| Stage 5 | Classical + Hybrid kecil |

---

### 7.5 DeepOcean CPU 16 Core RAM 62 GB

| Aspek | Penilaian |
|---|---|
| Spek | CPU 16 core, ±62 GB RAM |
| Storage feasibility | Feasible jika free disk minimal 30–50 GB |
| Risiko storage | Rendah-sedang |
| Lebih cocok | **Classical CNN** |
| Hybrid | Bisa sebagai background lambat |
| Stage cocok | Classical Stage 2–5 |

**Kesimpulan:**  
DeepOcean CPU adalah CPU worker terbaik. Cocok untuk Classical Stage 3–5 yang memiliki banyak trial.

| Stage | Rekomendasi |
|---|---|
| Stage 2 | Classical + Hybrid ringan |
| Stage 3 | Classical utama + Hybrid background kecil |
| Stage 4 | Classical utama |
| Stage 5 | Classical utama |

---

### 7.6 Azure CPU

| Aspek | Penilaian |
|---|---|
| Spek | 4 vCPU, ±32 GB RAM |
| Storage feasibility | Sangat feasible |
| Risiko storage | Rendah |
| Lebih cocok | **Classical CNN + orchestrator/monitoring** |
| Hybrid | Hanya Stage 2 atau background kecil |
| Stage cocok | Classical Stage 2–5 |

**Kesimpulan:**  
Azure CPU bagus untuk worker persistent Classical, dispatcher, heartbeat monitoring, atau PostgreSQL client. Tidak cocok untuk Hybrid berat.

| Stage | Rekomendasi |
|---|---|
| Stage 2 | Classical + Hybrid ringan |
| Stage 3 | Classical |
| Stage 4 | Classical |
| Stage 5 | Classical |

---

### 7.7 Databricks CPU

| Aspek | Penilaian |
|---|---|
| Spek | 4 CPU, ±30,7 GB RAM |
| Storage feasibility | Feasible terbatas |
| Risiko storage | Sedang karena runtime tidak sefleksibel VM biasa |
| Lebih cocok | **Classical kecil + analisis hasil** |
| Hybrid | Tidak disarankan |
| Stage cocok | Classical Stage 2, sebagian Stage 3–5 |

**Kesimpulan:**  
Databricks CPU bisa dipakai, tetapi jangan dijadikan worker training utama. Lebih cocok untuk analisis metrics, rekap hasil, visualisasi, dan Classical kecil.

| Stage | Rekomendasi |
|---|---|
| Stage 2 | Classical |
| Stage 3 | Classical sedikit |
| Stage 4 | Classical sedikit |
| Stage 5 | Classical sedikit |
| Hybrid | Hindari |

---

### 7.8 Kaggle CPU

| Aspek | Penilaian |
|---|---|
| Spek | 4 CPU, ±31 GB RAM |
| Storage feasibility | Feasible terbatas |
| Risiko storage | Sedang |
| Lebih cocok | **Classical CNN** |
| Hybrid | Tidak disarankan |
| Stage cocok | Classical Stage 2–5 kecil/sedang |

**Kesimpulan:**  
Kaggle CPU tetap bisa menjadi slow Classical worker.

| Stage | Rekomendasi |
|---|---|
| Stage 2 | Classical |
| Stage 3 | Classical |
| Stage 4 | Classical kecil |
| Stage 5 | Classical kecil |

---

### 7.9 Colab CPU

| Aspek | Penilaian |
|---|---|
| Spek | 2 CPU, ±12,6 GB RAM |
| Storage feasibility | Feasible kecil, tetapi tidak nyaman |
| Risiko storage | Tinggi karena runtime mudah putus |
| Lebih cocok | **Smoke test / Classical kecil** |
| Hybrid | Hindari |
| Stage cocok | Stage 2 dan Classical kecil |

**Kesimpulan:**  
Colab CPU jangan diberi task berat. Pakai hanya untuk smoke test atau Classical kecil.

| Stage | Rekomendasi |
|---|---|
| Stage 2 | Classical |
| Stage 3 | Classical kecil |
| Stage 4 | Hindari atau sangat kecil |
| Stage 5 | Hindari atau 1 fold kecil |

---

## 8. Tabel Final: Worker Lebih Cocok Classical atau Hybrid

| Worker | Storage Feasibility | Lebih Cocok | Boleh Hybrid? | Prioritas Stage |
|---|---|---|---|---|
| Google Cloud L4 | Sangat feasible | **Hybrid** | Ya, utama | Hybrid 3–5 |
| Colab GPU | Feasible tetapi ephemeral | **Hybrid** | Ya, utama | Hybrid 3–5 |
| Kaggle GPU | Feasible tetapi terbatas | **Hybrid** | Ya, utama | Hybrid 3–5 |
| PC RTX2050 | Feasible jika disk cukup | **Classical + Hybrid ringan** | Ya, terbatas | Classical 2–5, Hybrid 2–3 |
| DeepOcean CPU | Feasible | **Classical** | Ya, background kecil | Classical 2–5 |
| Azure CPU | Sangat feasible | **Classical + monitoring** | Hanya ringan | Classical 2–5 |
| Databricks CPU | Feasible terbatas | **Classical kecil + analisis** | Tidak disarankan | Classical 2–5 kecil |
| Kaggle CPU | Feasible terbatas | **Classical** | Tidak disarankan | Classical kecil |
| Colab CPU | Terbatas | **Smoke test / Classical kecil** | Tidak | Stage 2 / Classical kecil |

---

## 9. Pemetaan Stage Berdasarkan Kemampuan Storage dan Compute

### 9.1 Classical CNN

| Stage | Worker Utama | Worker Pendukung |
|---|---|---|
| Stage 2 | PC, Azure, DeepOcean | Databricks, Kaggle CPU, Colab CPU |
| Stage 3 | DeepOcean, Azure, PC | Databricks, Kaggle CPU |
| Stage 4 | DeepOcean, Azure, PC | Databricks/Kaggle CPU sedikit |
| Stage 5 | DeepOcean, Azure, PC | Databricks/Kaggle CPU sedikit |

### 9.2 Hybrid QCQ-CNN

| Stage | Worker Utama | Worker Pendukung |
|---|---|---|
| Stage 2 | PC RTX2050, Google Cloud L4 | DeepOcean/Azure untuk smoke test |
| Stage 3 | Google Cloud L4, Colab GPU, Kaggle GPU | PC RTX2050, DeepOcean kecil |
| Stage 4 | Google Cloud L4, Colab GPU, Kaggle GPU | PC RTX2050 sedikit |
| Stage 5 | Google Cloud L4, Colab GPU, Kaggle GPU | PC RTX2050 sedikit |

---

## 10. Unit Task yang Disarankan

Agar worker lambat tetap produktif, jangan berikan stage penuh. Gunakan unit kecil:

```text
1 task = 1 model + 1 stage + 1 trial + 1 fold
```

Untuk Stage 5:

```text
1 task = 1 model + 1 repeated-fold
```

Dengan unit kecil, CPU lambat tetap bisa jalan persistent tanpa menahan seluruh stage.

---

## 11. Prioritas Queue Worker

### 11.1 GPU Worker

```text
Prioritas 1: Hybrid Stage 4
Prioritas 2: Hybrid Stage 5
Prioritas 3: Hybrid Stage 3
Prioritas 4: Hybrid Stage 2
Prioritas 5: Classical hanya jika semua Hybrid selesai
```

### 11.2 CPU Worker

```text
Prioritas 1: Classical Stage 4
Prioritas 2: Classical Stage 5
Prioritas 3: Classical Stage 3
Prioritas 4: Classical Stage 2
Prioritas 5: Hybrid Stage 2 / Hybrid Stage 3 background kecil
Prioritas 6: Hybrid Stage 4/5 hanya jika tidak ada GPU sama sekali
```

---

## 12. Disk Guardrail

Agar worker tidak gagal karena disk penuh:

### GPU Worker

```text
Jika free disk < 10 GB:
1. Stop claim task baru.
2. Upload semua pending checkpoint.
3. Hapus interval checkpoint lokal.
4. Bersihkan cache pip/conda jika perlu.
5. Lanjutkan hanya setelah free disk aman.
```

### CPU Worker

```text
Jika free disk < 5 GB:
1. Stop claim task baru.
2. Upload pending metrics/checkpoint.
3. Hapus log lama dan checkpoint interval.
4. Lanjutkan setelah free disk aman.
```

---

## 13. Retention Policy

### Simpan Lokal

```text
latest.pt
prev.pt
best.pt
final.pt
metrics.jsonl
task_state.json
train.log
```

### Upload ke Google Drive/rclone

```text
best.pt
final.pt
metrics.jsonl
task_state.json
train.log
interval checkpoint setiap 5/10 epoch
```

### Hapus Lokal Setelah Verified

```text
interval checkpoint
task folder lama yang sudah COMPLETED dan upload verified
cache temporary
```

---

## 14. Rekomendasi Praktis

1. Pisahkan environment:
   - `environment-gpu.yml`
   - `environment-cpu.yml`

2. Jangan kirim dataset original ke worker.
   - Worker hanya perlu dataset cleaned/preprocessed.

3. Pecah curriculum manifest per stage:
   - `stage2_manifest.json`
   - `stage3_manifest.json`
   - `stage4_manifest.json`
   - `stage5_manifest.json`

4. Gunakan checkpoint overwrite:
   - `latest.pt`
   - `prev.pt`
   - `best.pt`
   - `final.pt`

5. Gunakan dynamic queue berbasis PostgreSQL:
   - Worker cepat mengambil banyak task.
   - Worker lambat tetap bekerja tanpa mengunci stage.

6. Google Cloud L4 50 GB cukup, tetapi 100 GB lebih aman.

7. CPU worker tetap boleh jalan terus, tetapi diarahkan ke Classical CNN dan Hybrid ringan.

---

## 15. Kesimpulan

Feasibility storage untuk semua worker adalah **aman**, dengan catatan pengelolaan checkpoint harus disiplin.

Ringkasan akhir:

| Jenis Worker | Penempatan Terbaik |
|---|---|
| GPU worker | Hybrid QCQ-CNN Stage 3–5 |
| GPU lokal kecil | Classical + Hybrid ringan |
| CPU besar | Classical CNN Stage 2–5 |
| CPU cloud sedang | Classical CNN + monitoring |
| CPU notebook/free runtime | Smoke test + Classical kecil |

Arsitektur ini layak dijalankan dengan strategi **persistent multi-worker**, selama:

1. setiap task dibuat kecil,
2. checkpoint lokal tidak menumpuk,
3. checkpoint penting selalu di-upload,
4. environment dipisah GPU/CPU,
5. queue mengatur prioritas GPU dan CPU secara berbeda.
