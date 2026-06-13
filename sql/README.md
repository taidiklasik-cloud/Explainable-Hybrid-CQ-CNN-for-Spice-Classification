# SQL execution order for local PostgreSQL orchestration

Run these files in the local PostgreSQL database `cqcnn_orchestration` in order:

1. `01_schema_tables_and_views.sql`  
   Creates orchestration tables, checkpoint metadata pointers, indexes, triggers, and monitoring views.

2. `02_functions_run_once.sql`  
   Creates worker/orchestrator SQL functions. This version supports worker-direct Optuna `study.tell()` through `mark_task_told_by_worker()`.

3. `03_seed_stage_information_stage3_4_5_revised.sql`  
   Optional seed for `stage_information` with Stage 3 convergence, Stage 4 maximum accuracy, and Stage 5 final repeated K-Fold without HPO.

4. `06_readiness_checks.sql`  
   Readiness checks after setup.

`04_worker_sql_repeatable_calls.sql` is reference only. Do not execute it as one raw script.

`05_optional_cron_stale_monitor.sql` is optional. Use it only if you want local `pg_cron` to mark stale tasks automatically. Heartbeat itself must still be sent by worker code.

## Optuna PostgreSQL design

The orchestration database does not store the authoritative hyperparameter configuration. The source of truth is Optuna PostgreSQL storage in `optuna_skripsi`. `task.trial_params_json` is only an audit/fallback snapshot.

## Checkpoint file design

Physical `.pt` checkpoint files are not stored in PostgreSQL. Interval, best, and final checkpoints are uploaded to Google Drive using rclone. PostgreSQL stores only metadata: `gdrive_relative_path`, `checkpoint_uri`, `sha256`, file size, epoch, metric, upload status, and checkpoint content flags.
