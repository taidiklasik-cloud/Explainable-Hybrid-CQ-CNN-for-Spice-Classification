"""worker_loop.py
Polling worker loop for manual Jupyter workers.

This worker is a template/orchestration wrapper. The real training function is passed
as a callback so the loop stays small and reusable.

Expected train_one_task signature:
    train_one_task(task, db, worker_uid, trial_params, heartbeat_callback) -> dict

Expected result for HPO stages:
    {"objective_metric_name": "val_macro_f1", "objective_value": 0.87, "requires_optuna_tell": True}

Expected result for final/non-HPO stages:
    {"objective_metric_name": "val_macro_f1", "objective_value": 0.86, "requires_optuna_tell": False}
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable

from optuna_postgres_utils import create_or_load_study, get_trial_params, tell_trial_result
from postgres_orchestration_db import PostgresOrchestrationDb

TrainOneTask = Callable[[dict[str, Any], PostgresOrchestrationDb, str, dict[str, Any], Callable[[], bool]], dict[str, Any]]


def _load_optuna_params_for_task(task: dict[str, Any], optuna_storage_url: str) -> dict[str, Any]:
    study_name = task.get("optuna_study_name")
    trial_nr = task.get("trial_nr")
    direction = task.get("objective_direction") or "maximize"

    if not study_name or trial_nr is None:
        return dict(task.get("trial_params_json") or {})

    study = create_or_load_study(
        study_name=str(study_name),
        storage_url=optuna_storage_url,
        direction=str(direction),
    )
    params = get_trial_params(study, int(trial_nr))
    if not params:
        # Audit/fallback snapshot from the orchestration DB. This should not be needed if the
        # orchestrator has already called trial.suggest_* before creating the task.
        params = dict(task.get("trial_params_json") or {})
    return params


def _tell_optuna_for_task(task: dict[str, Any], optuna_storage_url: str, objective_value: float) -> None:
    study_name = task.get("optuna_study_name")
    trial_nr = task.get("trial_nr")
    direction = task.get("objective_direction") or "maximize"
    if not study_name or trial_nr is None:
        return
    study = create_or_load_study(
        study_name=str(study_name),
        storage_url=optuna_storage_url,
        direction=str(direction),
    )
    tell_trial_result(study, int(trial_nr), float(objective_value))


def run_worker_loop(
    *,
    db: PostgresOrchestrationDb,
    worker_uid: str,
    stage_no: int,
    model_type: str,
    train_one_task: TrainOneTask,
    optuna_storage_url: str | None = None,
    poll_seconds: int = 30,
    stale_after: str = "15 minutes",
    allow_hijack: bool = True,
    stop_when_stage_complete: bool = True,
) -> None:
    """Run a worker that polls local PostgreSQL, trains tasks, and optionally tells Optuna.

    Idle logic:
    - HAS_WAITING_TASK: claim and train.
    - Before reading the idle signal, mark heartbeat-expired RUNNING tasks as STALE.
    - WAIT_FOR_DISPATCHER: no new Bayesian trial yet; keep polling.
    - WAIT_FOR_RUNNING_TASKS / WAIT_FOR_OPTUNA_TELL: keep polling.
    - HAS_STALE_TASK: hijack if enabled.
    - STAGE_COMPLETE: stop if requested.
    """
    print(f"[worker:{worker_uid}] starting stage={stage_no}, model={model_type}")

    while True:
        task = db.claim_waiting_task(worker_uid, stage_no, model_type)

        if task is None:
            stale_count = db.mark_stale_tasks(stale_after)
            if stale_count:
                print(f"[worker:{worker_uid}] marked stale tasks: {stale_count}")

            signal = db.get_worker_stage_signal(stage_no, model_type)
            code = signal.get("signal", "IDLE")
            reason = signal.get("reason", "")
            print(f"[worker:{worker_uid}] {code}: {reason}")

            if code == "HAS_STALE_TASK" and allow_hijack:
                task = db.hijack_stale_task(worker_uid, stage_no, model_type, stale_after=stale_after)
                if task:
                    print(f"[worker:{worker_uid}] hijacked task={task['task_id']}")
                else:
                    time.sleep(poll_seconds)
                    continue
            elif code == "STAGE_COMPLETE" and stop_when_stage_complete:
                print(f"[worker:{worker_uid}] stage complete. stopping.")
                return
            else:
                time.sleep(poll_seconds)
                continue

        task_id = int(task["task_id"])
        print(f"[worker:{worker_uid}] running task={task_id}, trial={task.get('trial_nr')}")

        try:
            requires_tell = bool(task.get("optuna_tell_status") == "PENDING")
            trial_params: dict[str, Any] = {}
            if requires_tell:
                if not optuna_storage_url:
                    raise RuntimeError("optuna_storage_url is required for HPO worker tasks.")
                trial_params = _load_optuna_params_for_task(task, optuna_storage_url)
            else:
                trial_params = dict(task.get("trial_params_json") or {})

            def heartbeat_callback() -> bool:
                return db.heartbeat(task_id, worker_uid)

            result = train_one_task(task, db, worker_uid, trial_params, heartbeat_callback)
            metric_name = result.get("objective_metric_name") or task.get("objective_metric_name")
            metric_value = float(result["objective_value"])
            result_requires_tell = bool(result.get("requires_optuna_tell", requires_tell))

            if result_requires_tell:
                if not optuna_storage_url:
                    raise RuntimeError("optuna_storage_url is required to tell Optuna.")
                _tell_optuna_for_task(task, optuna_storage_url, metric_value)
                ok = db.mark_told_by_worker(task_id, worker_uid, metric_name, metric_value)
            else:
                ok = db.mark_done(task_id, worker_uid, metric_name, metric_value)

            if not ok:
                raise RuntimeError(f"Failed to mark task {task_id} completed")

        except Exception as exc:  # noqa: BLE001
            msg = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            db.mark_failed(task_id, worker_uid, msg)
            print(f"[worker:{worker_uid}] task={task_id} failed: {msg}")
            raise
