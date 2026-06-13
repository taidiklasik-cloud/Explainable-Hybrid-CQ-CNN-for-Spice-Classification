# Workflow Detail dari EDA sampai Worker

## 1. EDA Dataset Awal
Notebook: `eda_datasetawal_generated.ipynb`.
Output lokal: `outputs/01_eda_datasetawal_generated/`.
Tujuan: profiling dataset, distribusi kelas, ukuran citra, kualitas awal, dan manifest awal.

## 2. EDA Fitur Gambar
Notebook: `eda_fiturgambar_generated_updated.ipynb`.
Output lokal: `outputs/02_eda_fiturgambar_generated_updated/`.
Tujuan: analisis fitur warna/tekstur/bentuk, PCA, ANOVA, overlap fitur, dan justifikasi EDA sebagai a priori decision layer.

## 3. Postprocessing EDA + Outlier Detection
Notebook: `posprocessing_eda_outlierdetection_generated.ipynb`.
Output lokal: `outputs/03_posprocessing_eda_outlierdetection_generated/`.
Tujuan: audit outlier, noise, dan validasi dataset final. Outlier tidak otomatis dihapus; dipakai untuk audit data.

## 4. Curriculum Design
Notebook final: `curriculum_stage_data_mechanism_80_20_audit_patched.ipynb`.
Output lokal: `outputs/04b_curriculum_stage_data_mechanism_80_20_audit_patched/`.
Strategi final:
- Stage 1: sanity test, holdout 80/20, 1 trial formal.
- Stage 2: warm start, holdout 80/20, 1 trial formal.
- Stage 3: convergence tuning, 5-fold CV, HPO aktif, minimize `val_loss`.
- Stage 4: maximum accuracy tuning, 5-fold CV, HPO aktif, maximize `val_macro_f1`.
- Stage 5: final repeated 5-fold evaluation, tanpa HPO, pakai best config Stage 4.

## 5. PostgreSQL Orchestration Setup
PostgreSQL lokal database `cqcnn_orchestration` menyimpan orchestration:
- `stage_information`
- `worker_node`
- `task`
- `checkpoint_slot`
- `checkpoint_file`

File fisik checkpoint `.pt` tidak disimpan di PostgreSQL. Checkpoint interval, best, dan final di-upload ke Google Drive via rclone. PostgreSQL hanya menyimpan metadata, pointer, hash, ukuran file, metric, epoch, dan status upload.

## 6. Optuna PostgreSQL Lokal
PostgreSQL lokal database `optuna_skripsi` adalah authoritative source untuk:
- study
- trial
- sampled hyperparameter
- objective value
- best trial

`task.trial_params_json` di `cqcnn_orchestration` hanya snapshot audit/fallback.

## 7. Optuna Orchestrator
Notebook: `notebooks/01_optuna_orchestrator_postgres.ipynb`.
Peran: `study.ask()`, `trial.suggest_*`, buat task rolling di `cqcnn_orchestration`, menjaga active task <= jumlah worker.

## 8. Worker PC Template
Notebook: `notebooks/02_worker_pc_template.ipynb`.
Peran: claim task dari PostgreSQL lokal, baca params dari Optuna PostgreSQL, training, heartbeat, local latest checkpoint cache, upload interval/best/final checkpoint ke Google Drive via rclone, register metadata checkpoint, `study.tell()`, mark task TOLD/DONE.

## 9. Idle dan Stage Completion
Worker membaca `get_worker_stage_signal()`:
- `HAS_WAITING_TASK`: claim task.
- `WAIT_FOR_DISPATCHER`: tunggu trial Bayesian berikutnya.
- `WAIT_FOR_RUNNING_TASKS`: tunggu task worker lain selesai.
- `WAIT_FOR_OPTUNA_TELL`: tunggu result masuk Optuna bila mode dispatcher-tell dipakai.
- `HAS_STALE_TASK`: hijack jika diizinkan.
- `STAGE_COMPLETE`: worker boleh berhenti.
