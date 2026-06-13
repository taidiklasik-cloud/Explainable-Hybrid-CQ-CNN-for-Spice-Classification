import os
import time
from pathlib import Path
from patched_program_files.env_loader import load_dotenv_if_exists
from patched_program_files.postgres_orchestration_db import PostgresOrchestrationDb, PostgresOrchestrationDbConfig
from patched_program_files.worker_hardware_profile import detect_worker_hardware
from patched_program_files.checkpoint_rclone import save_worker_latest_checkpoint, upload_and_register_checkpoint
import subprocess
import sys

from _path_setup import configure_paths

configure_paths()

# Mock torch for save_worker_latest_checkpoint
class DummyTorch:
    def save(self, obj, f):
        with open(f, 'w') as file:
            file.write(str(obj))
sys.modules['torch'] = DummyTorch()

def main():
    load_dotenv_if_exists()
    db = PostgresOrchestrationDb(PostgresOrchestrationDbConfig(dsn=os.environ["ORCHESTRATION_DB_DSN"]))
    worker_uid = os.environ.get("WORKER_UID", "test_pc")
    profile = detect_worker_hardware(worker_uid=worker_uid, worker_name="Test Worker")
    db.register_worker(**profile.to_orchestration_kwargs())

    # Create dummy task
    res = db.create_task_with_slot(stage_no=3, model_type="classical_fully_spatial")
    task_id = res["task_id"]
    task = db.claim_waiting_task(worker_uid, 3, "classical_fully_spatial")

    # Simulate 12 epochs
    for epoch in range(1, 13):
        print(f"\n--- Epoch {epoch} ---")
        
        # 1. Save local latest.pt
        local_info = save_worker_latest_checkpoint(
            worker_uid=worker_uid, task=task, model=None, optimizer=None, 
            scheduler=None, epoch=epoch, global_step=epoch*100, trial_params={"dummy": 1}, 
            model_config={}, rng_state={"python": None}
        )
        
        # 2. Write physical dummy file
        dummy_path = Path(local_info["local_cache_path"])
        with open(dummy_path, 'w') as f:
            f.write(f"dummy epoch {epoch}")

        # 3. Save metadata locally (RECOVERY) per epoch
        db.register_checkpoint_file(
            task_id=task_id, worker_uid=worker_uid, checkpoint_type="RECOVERY",
            gdrive_relative_path="NOT_UPLOADED_LOCAL", file_name=local_info["file_name"],
            sha256=local_info["sha256"], file_size_bytes=local_info["file_size_bytes"],
            epoch_number=epoch, global_step=epoch*100, metric_name="val_loss", metric_value=0.5,
            upload_status="LOCAL_ONLY", local_cache_path=str(dummy_path),
            rclone_remote="", storage_backend="gdrive_rclone"
        )
        print(f"Saved local RECOVERY metadata for epoch {epoch}")

        # Upload best at epoch 2
        if epoch == 2:
            print("Uploading BEST checkpoint...")
            file_id = upload_and_register_checkpoint(
                db=db, local_path=dummy_path, task=task, worker_uid=worker_uid,
                checkpoint_type="BEST", epoch_number=epoch, metric_name="val_loss", metric_value=0.4
            )
            print(f"Uploaded BEST checkpoint. File ID: {file_id}")

        # Upload interval at epoch 10
        if epoch == 10:
            print("Uploading INTERVAL checkpoint...")
            file_id = upload_and_register_checkpoint(
                db=db, local_path=dummy_path, task=task, worker_uid=worker_uid,
                checkpoint_type="INTERVAL", epoch_number=epoch, metric_name="val_loss", metric_value=0.6
            )
            print(f"Uploaded INTERVAL checkpoint. File ID: {file_id}")

    # Upload final at the end
    print("\n--- Training Finished ---")
    print("Uploading FINAL checkpoint...")
    file_id = upload_and_register_checkpoint(
        db=db, local_path=dummy_path, task=task, worker_uid=worker_uid,
        checkpoint_type="FINAL", epoch_number=12, metric_name="val_loss", metric_value=0.55
    )
    print(f"Uploaded FINAL checkpoint. File ID: {file_id}")
    
    db.mark_done(task_id, worker_uid, "val_loss", 0.55)
    print(f"Task {task_id} marked DONE.")

    # Check remote with rclone ls
    print("\n--- Rclone LS Verification ---")
    cmd = [
        os.environ["RCLONE_EXE_PATH"], 
        "ls", 
        f"{os.environ.get('RCLONE_REMOTE_NAME', 'gdrive')}:{os.environ.get('GDRIVE_CHECKPOINT_ROOT', 'cqcnn_checkpoints')}",
        "--config", os.environ["RCLONE_CONFIG_PATH"]
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(res.stdout)
    
    if 'task_id' in locals():
        print(f"\n[test] Cleaning up test task data {task_id} from DB...")
        try:
            db.delete_task_data(task_id)
            print("[test] Cleanup complete.")
        except Exception as e:
            print(f"[test] Cleanup failed: {e}")

if __name__ == "__main__":
    main()
