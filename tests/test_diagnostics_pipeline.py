import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from _path_setup import PROJECT_ROOT, configure_paths

configure_paths()

from patched_program_files.postgres_orchestration_db import PostgresOrchestrationDb, PostgresOrchestrationDbConfig
from worker_task_template import dry_run_train_one_task

def main():
    load_dotenv(PROJECT_ROOT / ".env")
    dsn = os.environ.get("ORCHESTRATION_DB_DSN")
    if not dsn:
        print("ERROR: ORCHESTRATION_DB_DSN tidak ditemukan di .env")
        sys.exit(1)
        
    db = PostgresOrchestrationDb(PostgresOrchestrationDbConfig(dsn=dsn))
    print("[1] Memeriksa koneksi DB:", db.test_connection())
    
    migration_path_8 = PROJECT_ROOT / "sql" / "08_curriculum_metrics_and_diagnostics.sql"
    with open(migration_path_8, "r", encoding="utf-8") as f:
        sql_content_8 = f.read()

    migration_path_9 = PROJECT_ROOT / "sql" / "09_curriculum_functions.sql"
    with open(migration_path_9, "r", encoding="utf-8") as f:
        sql_content_9 = f.read()
        
    try:
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_content_8)
                cur.execute(sql_content_9)
    except Exception as e:
        print("Error SQL Migration:", e)
        sys.exit(1)
    
    print("[2] Migration SQL 08_curriculum_metrics_and_diagnostics.sql tereksekusi (idempotent).")

    # 2. Simulasi 2 Fold Stage 3
    print("\n[3] Memulai Simulasi 2-Fold Dummy di Stage 3...")
    for fold in range(2):
        print(f"\n--- Mensimulasikan Fold {fold} ---")
        dummy_task = {
            "task_id": 9990 + fold,
            "stage_no": 3,
            "model_type": "hybrid_qcqcnn",
            "trial_nr": 1,
            "repeat_id": 0,
            "fold_id": fold,
            "objective_metric_name": "val_loss",
            "optuna_tell_status": "PENDING"
        }
        
        def dummy_heartbeat():
            pass
            
        result = dry_run_train_one_task(
            task=dummy_task,
            db=db,
            worker_uid="test_diagnostics_worker",
            trial_params={"lr": 0.001},
            heartbeat_callback=dummy_heartbeat
        )
        print(f"Hasil Akhir Fold {fold}:", result)

    print("\n[4] Memverifikasi data tersimpan di PostgreSQL...")
    logs = db._fetchall("SELECT * FROM public.epoch_metric_log WHERE task_id IN (9990, 9991)")
    print(f"  -> Ditemukan {len(logs)} baris epoch log metric!")
    
    diags = db._fetchall("SELECT * FROM public.convergence_diagnostic_summary WHERE task_id IN (9990, 9991)")
    print(f"  -> Ditemukan {len(diags)} baris diagnostic summary!")
    
    if len(logs) > 0 and len(diags) > 0:
        print("\n>>> SMOKE TEST PASS: Tabel metrik & diagnostik berfungsi penuh tanpa BLOB!")
    else:
        print("\n>>> SMOKE TEST FAIL: Data tidak ditemukan.")
        sys.exit(1)
        
    print("\n[test] Cleaning up test task data 9990 and 9991 from DB...")
    try:
        db.delete_task_data(9990)
        db.delete_task_data(9991)
        print("[test] Cleanup complete.")
    except Exception as e:
        print(f"[test] Cleanup failed: {e}")

if __name__ == "__main__":
    main()
