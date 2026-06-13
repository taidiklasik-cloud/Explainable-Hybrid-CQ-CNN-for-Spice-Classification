"""
test_heartbeat_smoke.py
Smoke test heartbeat monitoring: 12 epoch dummy, checkpoint events,
cek views v_worker_health / v_checkpoint_health / v_stale_workers.
Tidak ada training berat. File .pt adalah dummy teks kecil.
"""
import os
import sys
import socket
from pathlib import Path

from _path_setup import configure_paths

configure_paths()

from patched_program_files.env_loader import load_dotenv_if_exists
from patched_program_files.postgres_orchestration_db import PostgresOrchestrationDb, PostgresOrchestrationDbConfig
from patched_program_files.worker_hardware_profile import detect_worker_hardware
from patched_program_files.checkpoint_rclone import (
    save_worker_latest_checkpoint, upload_and_register_checkpoint, RcloneCheckpointConfig
)
from patched_program_files.worker_heartbeat import (
    update_worker_heartbeat, mark_worker_idle, mark_worker_failed, record_checkpoint_event
)

# Mock torch to avoid heavy dependency
class _DummyTorch:
    def save(self, obj, f):
        with open(f, "w") as fh:
            fh.write(str(obj))
sys.modules.setdefault("torch", _DummyTorch())

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def check(label: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    print(f"  {tag} {label}" + (f" | {detail}" if detail else ""))
    results.append((label, condition))
    return condition


def main():
    load_dotenv_if_exists()
    dsn = os.environ["ORCHESTRATION_DB_DSN"]
    db = PostgresOrchestrationDb(PostgresOrchestrationDbConfig(dsn=dsn))
    worker_uid = os.environ.get("WORKER_UID", "pc_01")
    worker_name = os.environ.get("WORKER_NAME", "Laptop Pribadi")

    # ---------- Register worker ----------
    print("\n=== Setup ===")
    profile = detect_worker_hardware(worker_uid=worker_uid, worker_name=worker_name)
    db.register_worker(**profile.to_orchestration_kwargs())
    check("register_worker", True)

    # ---------- Create + claim task ----------
    # Create + claim task — use actual claimed task_id (may differ from newly created)
    res = db.create_task_with_slot(stage_no=3, model_type="classical_fully_spatial")
    task = db.claim_waiting_task(worker_uid, 3, "classical_fully_spatial")
    task_id = int(task["task_id"])   # authoritative task_id from claim
    check("create_and_claim_task", task is not None, f"claimed task_id={task_id}")

    # ---------- Heartbeat on start ----------
    print("\n=== Heartbeat: start ===")
    update_worker_heartbeat(
        db, worker_uid, worker_name=worker_name,
        status="RUNNING", current_task_id=task_id,
        stage_no=3, model_type="classical_fully_spatial", current_epoch=0,
    )
    row = db._fetchone(
        "select status, current_task_id from public.worker_heartbeat where worker_uid = %s",
        (worker_uid,)
    )
    check("heartbeat on start", row is not None and row["status"] == "RUNNING",
          f"status={row and row['status']}")

    # ---------- Simulate 12 epochs ----------
    print("\n=== Epoch Loop 1-12 ===")
    dummy_path: Path | None = None
    best_file_id: int | None = None
    interval_file_id: int | None = None
    final_file_id: int | None = None

    for epoch in range(1, 13):
        # Local latest.pt
        local_info = save_worker_latest_checkpoint(
            worker_uid=worker_uid, task=task, model=None, optimizer=None,
            scheduler=None, epoch=epoch, global_step=epoch * 100,
            trial_params={}, model_config={}, rng_state={"python": None},
        )
        dummy_path = Path(local_info["local_cache_path"])
        with open(dummy_path, "w") as f:
            f.write(f"dummy epoch {epoch}")

        # Heartbeat per epoch
        update_worker_heartbeat(
            db, worker_uid, status="RUNNING", current_task_id=task_id,
            stage_no=3, model_type="classical_fully_spatial", current_epoch=epoch,
            last_checkpoint_local=str(dummy_path),
        )

        # BEST at epoch 2
        if epoch == 2:
            record_checkpoint_event(
                db, worker_uid, task=task, epoch=epoch,
                local_path=str(dummy_path),
                remote_path=f"gdrive:cqcnn_checkpoints/stage_03/{worker_uid}/non_hpo/best.pt",
                status="UPLOADING",
            )
            best_file_id = upload_and_register_checkpoint(
                db=db, local_path=dummy_path, task=task, worker_uid=worker_uid,
                checkpoint_type="BEST", epoch_number=epoch, metric_name="val_loss", metric_value=0.40,
            )
            record_checkpoint_event(
                db, worker_uid, task=task, epoch=epoch,
                local_path=str(dummy_path),
                remote_path=f"gdrive:cqcnn_checkpoints/stage_03/{worker_uid}/non_hpo/best.pt",
                status="RUNNING",
            )

        # INTERVAL at epoch 10
        if epoch == 10:
            record_checkpoint_event(
                db, worker_uid, task=task, epoch=epoch,
                local_path=str(dummy_path),
                remote_path=f"gdrive:cqcnn_checkpoints/stage_03/{worker_uid}/non_hpo/epoch_0010.pt",
                status="UPLOADING",
            )
            interval_file_id = upload_and_register_checkpoint(
                db=db, local_path=dummy_path, task=task, worker_uid=worker_uid,
                checkpoint_type="INTERVAL", epoch_number=epoch, metric_name="val_loss", metric_value=0.55,
            )
            record_checkpoint_event(
                db, worker_uid, task=task, epoch=epoch,
                local_path=str(dummy_path),
                remote_path=f"gdrive:cqcnn_checkpoints/stage_03/{worker_uid}/non_hpo/epoch_0010.pt",
                status="RUNNING",
            )

    # ---------- FINAL checkpoint ----------
    print("\n=== Final Checkpoint ===")
    record_checkpoint_event(
        db, worker_uid, task=task, epoch=12,
        local_path=str(dummy_path),
        remote_path=f"gdrive:cqcnn_checkpoints/stage_03/{worker_uid}/non_hpo/final.pt",
        status="UPLOADING",
    )
    final_file_id = upload_and_register_checkpoint(
        db=db, local_path=dummy_path, task=task, worker_uid=worker_uid,
        checkpoint_type="FINAL", epoch_number=12, metric_name="val_loss", metric_value=0.50,
    )

    # ---------- Mark done + idle heartbeat ----------
    db.mark_done(task_id, worker_uid, "val_loss", 0.50)
    mark_worker_idle(db, worker_uid)

    # ============================================================
    # CHECK: v_worker_health
    # ============================================================
    print("\n=== CHECK: v_worker_health ===")
    wh = db._fetchone(
        "select * from public.v_worker_health where worker_uid = %s", (worker_uid,)
    )
    check("v_worker_health row exists", wh is not None)
    check("worker status IDLE after done", wh and wh["status"] == "IDLE",
          f"got={wh and wh['status']}")
    check("hostname populated", wh and bool(wh.get("hostname")))
    check("last_checkpoint_epoch populated", wh and wh.get("last_checkpoint_epoch") == 12,
          f"epoch={wh and wh.get('last_checkpoint_epoch')}")
    check("last_checkpoint_local populated", wh and bool(wh.get("last_checkpoint_local_path")))

    # ============================================================
    # CHECK: v_checkpoint_health
    # ============================================================
    print("\n=== CHECK: v_checkpoint_health ===")
    checkpoints = db._fetchall(
        "select * from public.v_checkpoint_health where task_id = %s order by created_at", (task_id,)
    )
    types_uploaded = {r["checkpoint_type"] for r in checkpoints if r.get("upload_status") == "UPLOADED"}
    check("BEST checkpoint registered", "BEST" in types_uploaded, f"types={types_uploaded}")
    check("INTERVAL checkpoint registered", "INTERVAL" in types_uploaded)
    check("FINAL checkpoint registered", "FINAL" in types_uploaded)
    check("best_file_id > 0", best_file_id is not None and best_file_id > 0, f"id={best_file_id}")
    check("interval_file_id > 0", interval_file_id is not None and interval_file_id > 0)
    check("final_file_id > 0", final_file_id is not None and final_file_id > 0)

    # Epoch 10 interval name check
    interval_row = next((r for r in checkpoints if r["checkpoint_type"] == "INTERVAL"), None)
    check("INTERVAL epoch_number=10", interval_row and interval_row["epoch_number"] == 10,
          f"epoch={interval_row and interval_row['epoch_number']}")
    check("INTERVAL gdrive_path contains epoch_0010", interval_row and "epoch_0010" in (interval_row.get("gdrive_relative_path") or ""),
          f"path={interval_row and interval_row.get('gdrive_relative_path')}")

    # ============================================================
    # CHECK: v_stale_workers (should be empty since we just heartbeated)
    # ============================================================
    print("\n=== CHECK: v_stale_workers ===")
    stale = db._fetchall(
        "select * from public.v_stale_workers where worker_uid = %s", (worker_uid,)
    )
    check("v_stale_workers empty for active worker", len(stale) == 0,
          f"stale_count={len(stale)}")

    # ============================================================
    # CHECK: monitor_stale_workers function callable
    # ============================================================
    print("\n=== CHECK: monitor_stale_workers function ===")
    monitor_result = db._fetchone("select public.monitor_stale_workers() as n")
    check("monitor_stale_workers() callable", monitor_result is not None,
          f"returned={monitor_result}")

    # ============================================================
    # CHECK: no Supabase
    # ============================================================
    print("\n=== CHECK: No Supabase ===")
    check("SUPABASE_URL not in env", "SUPABASE_URL" not in os.environ)
    check("SUPABASE_KEY not in env", "SUPABASE_KEY" not in os.environ)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"RESULT: {passed}/{total} checks passed")
    if passed == total:
        print(">>> ALL PASS <<<")
    else:
        print(">>> SOME FAILED <<<")
        for label, ok in results:
            if not ok:
                print(f"  FAIL: {label}")
    print("=" * 50)

    print(f"\n[test] Cleaning up test task data {task_id} from DB...")
    try:
        db.delete_task_data(task_id)
        print("[test] Cleanup complete.")
    except Exception as e:
        print(f"[test] Cleanup failed: {e}")

if __name__ == "__main__":
    main()
