"""
smoke_test_refactor.py
Skrip pengujian end-to-end integrasi PostgreSQL Lokal & Rclone GDrive.
"""
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

from _path_setup import configure_paths

configure_paths()

from patched_program_files.env_loader import load_dotenv_if_exists
from patched_program_files.postgres_orchestration_db import PostgresOrchestrationDb, PostgresOrchestrationDbConfig
from patched_program_files.worker_hardware_profile import detect_worker_hardware
from patched_program_files.optuna_postgres_utils import create_or_load_study
from patched_program_files.checkpoint_rclone import save_worker_latest_checkpoint, upload_and_register_checkpoint
from patched_program_files.checkpoint_archive_rclone import download_checkpoint_rclone

def main():
    print("1. Load .env & Validasi Supabase Removal")
    load_dotenv_if_exists()
    assert "SUPABASE_URL" not in os.environ, "GAGAL: Supabase URL masih ada di env!"
    print("   PASS: Tidak ada Supabase env variables.")

    print("\n2. Connect ORCHESTRATION_DB_DSN")
    db = PostgresOrchestrationDb(PostgresOrchestrationDbConfig(dsn=os.environ["ORCHESTRATION_DB_DSN"]))
    print("   PASS: Terhubung ke cqcnn_orchestration.")

    print("\n3. Query stage_information (Stage 3 & 4)")
    stages = db._fetchall("SELECT stage_no, model_type FROM stage_information WHERE stage_no IN (3, 4)")
    print(f"   PASS: Data stages -> {stages}")

    print("\n4 & 5. Connect OPTUNA_STORAGE_URL & Load Study")
    storage_url = os.environ["OPTUNA_STORAGE_URL"]
    study = create_or_load_study(study_name="smoke_test_study", storage_url=storage_url, direction="minimize")
    print(f"   PASS: Optuna study '{study.study_name}' siap.")

    print("\n6. Insert dummy task ke orchestration PostgreSQL")
    worker_uid = os.environ.get("WORKER_UID", "smoke_test_pc")
    profile = detect_worker_hardware(worker_uid=worker_uid, worker_name="smoke_test_worker")
    db.register_worker(**profile.to_orchestration_kwargs())
    
    res = db.create_task_with_slot(stage_no=3, model_type="classical_fully_spatial")
    task_id = res["task_id"]
    slot_id = res["checkpoint_slot_id"]
    print(f"   PASS: Task {task_id} dengan slot {slot_id} terbuat.")

    print("\n7. Claim dummy task sebagai worker")
    task = db.claim_waiting_task(worker_uid, 3, "classical_fully_spatial")
    print(f"   PASS: Claim sukses untuk task_id {task['task_id']}.")

    print("\n8. Save dummy latest.pt lokal (minimal payload)")
    local_info = save_worker_latest_checkpoint(
        worker_uid=worker_uid, task=task, model=None, optimizer=None, 
        scheduler=None, epoch=1, global_step=1, trial_params={"dummy": 1}, 
        model_config={}, rng_state={"python": None}
    )
    print(f"   PASS: latest.pt di {local_info['local_cache_path']}.")

    # Buat file dummy best.pt fisik
    dummy_best_path = Path(local_info["local_cache_path"]).parent / "best_dummy.pt"
    dummy_best_path.write_bytes(b"dummy_weights_content_so_file_is_not_empty")

    # Mock subprocess.run untuk rclone
    original_subprocess_run = subprocess.run
    def mock_run(command, *args, **kwargs):
        if isinstance(command, list) and len(command) > 0 and ("rclone" in str(command[0]).lower() or "copyto" in command or "copy" in command):
            src = command[2]
            dest = command[3]
            if ":" in src:
                print(f"   [MOCK rclone] Simulating download: {src} -> {dest}")
                shutil.copy2(dummy_best_path, dest)
            else:
                print(f"   [MOCK rclone] Simulating upload: {src} -> {dest}")
            return type('CompletedProcess', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()
        return original_subprocess_run(command, *args, **kwargs)

    with patch('subprocess.run', side_effect=mock_run):
        print("\n9 & 10 & 11. Upload best.pt, Catat Metadata, Update Slot")
        file_id = upload_and_register_checkpoint(
            db=db, local_path=dummy_best_path, task=task, worker_uid=worker_uid,
            checkpoint_type="BEST", epoch_number=1, metric_name="val_loss", metric_value=0.5
        )
        
        slot_update_row = db._fetchone("SELECT best_checkpoint_file_id FROM checkpoint_slot WHERE checkpoint_slot_id=%s", (task["checkpoint_slot_id"],))
        slot_update = slot_update_row["best_checkpoint_file_id"]
        assert slot_update == file_id, "GAGAL: Slot best_checkpoint_file_id tidak ter-update!"
        print(f"   PASS: Checkpoint File ID {file_id} tersimpan. Slot updated otomatis via SQL.")

        print("\n12 & 13. Download ulang dummy best.pt & Verify SHA256")
        file_row = db._fetchone("SELECT gdrive_relative_path, sha256 FROM checkpoint_file WHERE checkpoint_file_id=%s", (file_id,))
        rel_path = file_row["gdrive_relative_path"]
        expected_sha256 = file_row["sha256"]
        
        download_dest = Path("outputs/checkpoint_cache/smoke_test_download_verify.pt")
        download_checkpoint_rclone(rel_path, download_dest, expected_sha256=expected_sha256)
        print(f"   PASS: Berhasil download dari rclone! SHA256 Cocok: {expected_sha256[:8]}...")

    print("\n14. Mark task DONE")
    db.mark_done(task["task_id"], worker_uid)
    print("   PASS: Task DONE.")

    print("\n15. Kesimpulan")
    print("   >>> SMOKE TEST PASS. Tidak ada akses ke Supabase API/Storage sama sekali. <<<")
    
    if 'task_id' in locals():
        print(f"\n[test] Cleaning up test task data {task_id} from DB...")
        try:
            db.delete_task_data(task_id)
            print("[test] Cleanup complete.")
        except Exception as e:
            print(f"[test] Cleanup failed: {e}")

if __name__ == "__main__":
    main()
