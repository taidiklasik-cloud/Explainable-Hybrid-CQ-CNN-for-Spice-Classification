# Audit & Patch Minimal: CQ-CNN Diagnostics & Metrics

Pipeline saat ini memiliki orkestrasi yang kokoh, namun secara teknis **kekurangan lapisan persisten untuk menyimpan metrik eksperimen lanjutan (multiclass & convergence)**. Semuanya masih bergantung pada Optuna SQLite/PostgreSQL yang hanya menyimpan metrik objektif tunggal per trial.

Berikut adalah hasil audit dan usulan rencana implementasi (*Implementation Plan*) yang mematuhi batas arsitektur ketat Anda (Tanpa Refactor Besar, Tanpa BLOB).

---

## A-D. Audit Checklist (PASS/FAIL)
- **A. FAIL**: Pencatatan Stage 2 multiclass metrics belum ada di `worker_task_template.py` dan belum memiliki struktur tabel di PostgreSQL.
- **B. FAIL**: Pencatatan Stage 3 convergence + multiclass metrics (termasuk deteksi loss plateau, gradient plateau, dan F1) belum didukung oleh `epoch_metric_log` maupun `convergence_diagnostic_summary`.
- **C. FAIL**: Data statistical significance Stage 3–5 belum dikumpulkan secara agregat; tidak ada tabel `fold_run_result` atau `statistical_test_result`.
- **D. FAIL**: Effect size pipeline dan tabel penyimpannya (`effect_size_result`) sama sekali belum ada.

## E-F. Analisis Skema Database
**E. Tabel yang sudah ada (Orkestrasi Inti):**
`stage_information`, `worker_node`, `worker_heartbeat`, `worker_monitoring_event`, `task`, `checkpoint_slot`, `checkpoint_file`.

**F. Tabel yang perlu ditambah (Sesuai Permintaan):**
`fold_run_result`, `epoch_metric_log`, `convergence_diagnostic_summary`, `statistical_test_result`, `effect_size_result`, `experiment_artifact_pointer`.

---

## Proposed Changes

### 1. Lapisan Database (SQL Migration)
#### [NEW] sql/08_curriculum_metrics_and_diagnostics.sql
Sebuah file migrasi SQL baru yang bersifat *idempotent* (`CREATE TABLE IF NOT EXISTS`) untuk mencetak 6 tabel di atas.
* **Tidak menggunakan bytea/BLOB.** Semua grafik PNG dan metrik detail CSV hanya disimpan sebagai path lokal (`local_path`) atau URL gdrive (`remote_uri`) di tabel `experiment_artifact_pointer`.
* Istilah untuk quantum hybrid diakomodir: kolom `barren_plateau_indicator` bertipe boolean *nullable* disediakan di tabel diagnostik konvergensi.

### 2. Lapisan Penghubung Python
#### [MODIFY] patched_program_files/postgres_orchestration_db.py
Menambahkan fungsi utilitas ringan tanpa merusak arsitektur kelas, khusus untuk logging metrik ke 6 tabel baru:
- `log_epoch_metrics(**kwargs)`
- `log_fold_run_result(**kwargs)`
- `log_convergence_diagnostic(**kwargs)`
- `log_experiment_artifact(**kwargs)`
- `log_statistical_test(**kwargs)`

### 3. Lapisan Eksekusi Worker
#### [MODIFY] patched_program_files/worker_task_template.py
Meng-update pseudo-trainer di `train_one_task` agar setiap *epoch* mengkompilasi *dictionary* yang berisi `train_acc`, `val_macro_f1`, dan `grad_norm_global` (dummy, sebagai simulasi).
*   Metrik ini kemudian dilempar melalui pemanggilan ke fungsi DB (contoh: `db.log_epoch_metrics(...)`).
*   Pada akhir task, logika diagnostik *convergence* dipanggil dan dimasukkan ke `convergence_diagnostic_summary`.

---

## User Review Required

> [!IMPORTANT]
> **Keputusan Storage**: Sesuai dengan instruksi, `epoch_metric_log` akan mencatat data numerik per epoch ke PostgreSQL. Secara arsitektur, ini aman jika epoch dibatasi (misal < 100). Harap konfirmasi bahwa Anda setuju fungsi `log_epoch_metrics()` akan mengeksekusi `INSERT` ke PostgreSQL pada akhir setiap iterasi epoch *worker*.

> [!TIP]
> **Artifact Pointers**: Tabel `experiment_artifact_pointer` akan menunjuk ke lokasi `.png` dan `.csv`. Kami akan menggunakan format JSON di kolom `metadata` tabel ini jika ada atribut tambahan (misal: *bbox* plot). 

## Verification Plan

### J. Smoke Test Minimal
Kami akan membuat file **`tests/test_diagnostics_pipeline.py`** yang:
1. Mengeksekusi SQL migration untuk 6 tabel baru.
2. Mensimulasikan jalannya 2 *fold* dummy di Stage 3.
3. Mencatat log epoch (loss & macro F1), gradient norm, dan membuat indikator dummy `loss_plateau_detected`.
4. Menyimpan pointer artifact lokal fiktif (`stage3_loss_curve.png`).
5. Memvalidasi bahwa baris-baris tersebut telah tersimpan di database Orchestration tanpa error.
