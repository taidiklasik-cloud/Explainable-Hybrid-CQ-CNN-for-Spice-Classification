# Run Order End-to-End

## A. Persiapan lokal
1. Install dependencies dari `requirements.txt`.
2. Setup `.env` dari `.env.example`.
3. Setup PostgreSQL lokal untuk `cqcnn_orchestration`, lalu isi `ORCHESTRATION_DB_DSN`.
4. Setup PostgreSQL lokal untuk Optuna `optuna_skripsi`, lalu isi `OPTUNA_STORAGE_URL`.
5. Setup rclone remote Google Drive, lalu isi `RCLONE_REMOTE_NAME`, `GDRIVE_CHECKPOINT_ROOT`, dan `WORKER_LOCAL_CHECKPOINT_ROOT`.
6. Pastikan SQL orchestration lokal sudah dijalankan berurutan.

## B. Jalankan SQL PostgreSQL Orchestration Lokal
Jalankan di database PostgreSQL lokal `cqcnn_orchestration`:
1. `sql/01_schema_tables_and_views.sql`
2. `sql/02_functions_run_once.sql`
3. `sql/03_seed_stage_information_stage3_4_5_revised.sql`
4. `sql/06_readiness_checks.sql`

Opsional:
- `sql/05_optional_cron_stale_monitor.sql` jika ingin stale checker otomatis via `pg_cron`.

Jangan jalankan `sql/04_worker_sql_repeatable_calls.sql` secara mentah; file itu contoh panggilan RPC berulang dari Python/notebook.

## C. Jalankan pipeline data lokal
1. `01_eda_pipeline/final_recommended/eda_datasetawal_generated.ipynb`
2. `01_eda_pipeline/final_recommended/eda_fiturgambar_generated_updated.ipynb`
3. `01_eda_pipeline/final_recommended/posprocessing_eda_outlierdetection_generated.ipynb`
4. `02_curriculum_pipeline/final_recommended/curriculum_stage_data_mechanism_80_20_audit_patched.ipynb`

Simpan output masing-masing notebook ke folder `outputs/<nama_notebook>/`.

## D. Jalankan arsitektur model sanity
- `03_model_architecture/final_recommended/model_architecture_resource_checked_worker_runtime.ipynb`

## E. Jalankan Optuna + Worker
Pada laptop pribadi gunakan 2 kernel Jupyter berbeda:
1. Kernel A: `notebooks/01_optuna_orchestrator_postgres.ipynb`
2. Kernel B: `notebooks/02_worker_pc_template.ipynb`

Worker PC lain hanya menjalankan:
- `notebooks/02_worker_pc_template.ipynb`

## F. Stage logic
- Stage 3: convergence tuning, objective `val_loss`, direction `minimize`.
- Stage 4: maximum accuracy tuning, objective `val_macro_f1`, direction `maximize`.
- Stage 5: final repeated 5-fold evaluation, tanpa HPO; gunakan best config dari Stage 4.
