"""worker_heartbeat.py
Helper functions for worker heartbeat, status updates, and checkpoint event tracking.
Dipanggil dari training loop setiap epoch, saat claim task, upload checkpoint, dan exception.
"""
from __future__ import annotations

import os
import socket
import platform
from typing import Any, Optional

try:
    from postgres_orchestration_db import PostgresOrchestrationDb
except ImportError:
    from patched_program_files.postgres_orchestration_db import PostgresOrchestrationDb  # type: ignore


def _pid() -> int:
    return os.getpid()


def _hostname() -> str:
    return socket.gethostname()


def update_worker_heartbeat(
    db: PostgresOrchestrationDb,
    worker_uid: str,
    *,
    worker_name: str | None = None,
    status: str = "RUNNING",
    current_task_id: int | None = None,
    stage_no: int | None = None,
    model_type: str | None = None,
    current_epoch: int | None = None,
    last_checkpoint_epoch: int | None = None,
    last_checkpoint_local: str | None = None,
    last_checkpoint_remote: str | None = None,
    error_message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Upsert worker heartbeat in PostgreSQL. Called every epoch and on status changes."""
    try:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(db.config.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """select public.upsert_worker_heartbeat(
                        p_worker_uid := %(worker_uid)s,
                        p_worker_name := %(worker_name)s,
                        p_hostname := %(hostname)s,
                        p_pid := %(pid)s,
                        p_status := %(status)s,
                        p_current_task_id := %(current_task_id)s,
                        p_stage_no := %(stage_no)s,
                        p_model_type := %(model_type)s,
                        p_current_epoch := %(current_epoch)s,
                        p_last_checkpoint_epoch := %(last_checkpoint_epoch)s,
                        p_last_checkpoint_local := %(last_checkpoint_local)s,
                        p_last_checkpoint_remote := %(last_checkpoint_remote)s,
                        p_error_message := %(error_message)s,
                        p_metadata := %(metadata)s
                    )""",
                    {
                        "worker_uid": worker_uid,
                        "worker_name": worker_name or os.environ.get("WORKER_NAME"),
                        "hostname": _hostname(),
                        "pid": _pid(),
                        "status": status,
                        "current_task_id": current_task_id,
                        "stage_no": stage_no,
                        "model_type": model_type,
                        "current_epoch": current_epoch,
                        "last_checkpoint_epoch": last_checkpoint_epoch,
                        "last_checkpoint_local": last_checkpoint_local,
                        "last_checkpoint_remote": last_checkpoint_remote,
                        "error_message": error_message,
                        "metadata": psycopg2.extras.Json(metadata) if metadata else None,
                    },
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        # Heartbeat must never crash training
        print(f"[heartbeat] Warning: failed to update heartbeat for {worker_uid}: {e}")


def mark_worker_idle(db: PostgresOrchestrationDb, worker_uid: str) -> None:
    """Mark worker as IDLE (no active task)."""
    update_worker_heartbeat(
        db, worker_uid,
        status="IDLE",
        current_task_id=None,
        stage_no=None,
        model_type=None,
        current_epoch=None,
    )


def mark_worker_failed(
    db: PostgresOrchestrationDb,
    worker_uid: str,
    *,
    current_task_id: int | None = None,
    error_message: str,
) -> None:
    """Mark worker as FAILED with error message."""
    update_worker_heartbeat(
        db, worker_uid,
        status="FAILED",
        current_task_id=current_task_id,
        error_message=error_message[:2000] if error_message else None,
    )


def record_checkpoint_event(
    db: PostgresOrchestrationDb,
    worker_uid: str,
    *,
    task: dict[str, Any],
    epoch: int,
    local_path: str,
    remote_path: str,
    status: str = "UPLOADING",
) -> None:
    """Update heartbeat immediately before/after checkpoint upload."""
    update_worker_heartbeat(
        db, worker_uid,
        status=status,
        current_task_id=int(task.get("task_id", 0)),
        stage_no=int(task.get("stage_no", 0)),
        model_type=task.get("model_type"),
        current_epoch=epoch,
        last_checkpoint_epoch=epoch,
        last_checkpoint_local=local_path,
        last_checkpoint_remote=remote_path,
    )
