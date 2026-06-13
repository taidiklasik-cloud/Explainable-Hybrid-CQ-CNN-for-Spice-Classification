"""optuna_stage_manager.py
Rolling Optuna orchestrator for PostgreSQL Optuna storage + local PostgreSQL task queue.

Recommended final architecture:
- Optuna orchestrator on local laptop: ask/suggest and create orchestration tasks.
- Optuna PostgreSQL local/shared: authoritative study/trial/parameter/objective state.
- Local PostgreSQL cqcnn_orchestration: task orchestration, heartbeat, checkpoint metadata, resume/hijack.
- Workers: claim local PostgreSQL tasks, load params from Optuna PostgreSQL, train, tell result.

The orchestrator intentionally does NOT train models. It only keeps the active task
window close to the number of available workers, so Bayesian/TPE updates can occur
between rolling batches instead of pre-generating all trials.

Stage 5 is non-HPO. It is dispatched as one final repeated 5-fold task that points
back to the Stage 4 Optuna best trial. The worker then runs all repeats/folds with
that fixed configuration and marks the task DONE, not TOLD.
"""
from __future__ import annotations

import time
from typing import Any

from optuna_postgres_utils import create_or_load_study, suggest_from_search_space
from postgres_orchestration_db import PostgresOrchestrationDb


def create_final_repeated_kfold_task_from_stage4_best(
    *,
    db: PostgresOrchestrationDb,
    final_stage_info: dict[str, Any],
    optuna_storage_url: str,
    source_stage_no: int = 4,
    seed: int = 42,
    dispatcher_batch_no: int = 1,
    final_objective_metric_name: str = "mean_macro_f1_repeated_5fold",
) -> dict[str, Any]:
    """Create the single Stage 5 non-HPO task from the Stage 4 Optuna best trial.

    The task stores the best trial params as a flat audit/fallback snapshot in
    task.trial_params_json. Optuna PostgreSQL remains authoritative because the
    task also stores the source study name and best trial number.
    """
    final_stage_no = int(final_stage_info["stage_no"])
    model_type = str(final_stage_info["model_type"])
    optuna_trials = int(final_stage_info.get("optuna_trials") or 0)
    split_strategy = str(final_stage_info.get("split_strategy") or "")

    if optuna_trials != 0:
        raise ValueError("Final repeated K-Fold task generation is only for non-HPO stages.")
    if split_strategy != "REPEATED_5_FOLD":
        raise ValueError(f"Expected split_strategy='REPEATED_5_FOLD', got {split_strategy!r}.")

    signal = db.get_worker_stage_signal(final_stage_no, model_type)
    if int(signal.get("total_tasks") or 0) > 0:
        print(
            f"[orchestrator] final stage already has task(s): "
            f"stage={final_stage_no}, model={model_type}, signal={signal.get('signal')}"
        )
        return {"created": False, "signal": signal}

    source_stage_info = db.get_stage_info(source_stage_no, model_type)
    source_study_name = str(source_stage_info.get("optuna_study_name") or "")
    source_direction = str(source_stage_info.get("optuna_direction") or "")
    if not source_study_name or source_direction not in {"minimize", "maximize"}:
        raise ValueError(
            f"Stage {source_stage_no} for {model_type} must have optuna_study_name and valid optuna_direction."
        )

    study = create_or_load_study(
        study_name=source_study_name,
        storage_url=optuna_storage_url,
        direction=source_direction,
        seed=seed,
        constant_liar=False,
    )
    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    if not best_params:
        raise ValueError(f"Best trial in study {source_study_name!r} has no sampled parameters.")

    task = db.create_task_with_slot(
        stage_no=final_stage_no,
        model_type=model_type,
        optuna_study_name=source_study_name,
        trial_nr=int(best_trial.number),
        trial_params_json=best_params,
        objective_metric_name=final_objective_metric_name,
        objective_direction="maximize",
        optuna_tell_status="NOT_REQUIRED",
        dispatcher_batch_no=dispatcher_batch_no,
    )
    print(
        f"[orchestrator] created final repeated 5-fold task={task['task_id']} "
        f"from stage={source_stage_no} study={source_study_name} "
        f"best_trial={best_trial.number} best_value={best_trial.value}"
    )
    return {
        "created": True,
        **task,
        "source_stage_no": source_stage_no,
        "source_optuna_study_name": source_study_name,
        "source_best_trial_nr": int(best_trial.number),
        "source_best_value": float(best_trial.value),
        "trial_params": best_params,
    }


def run_dispatcher_from_postgres(
    *,
    db: PostgresOrchestrationDb,
    stage_no: int,
    model_type: str,
    optuna_storage_url: str,
    max_parallel_tasks: int = 1,
    poll_seconds: int = 30,
    seed: int = 42,
    stop_when_complete: bool = True,
    dispatcher_batch_start: int = 1,
    source_best_stage_no: int = 4,
) -> dict[str, Any] | None:
    """Dispatch tasks for one stage/model using local PostgreSQL stage metadata."""
    stage_info = db.get_stage_info(stage_no, model_type)
    optuna_trials = int(stage_info.get("optuna_trials") or 0)
    split_strategy = str(stage_info.get("split_strategy") or "")

    print(
        f"[orchestrator] loaded stage_information: "
        f"stage={stage_no}, model={model_type}, optuna_trials={optuna_trials}, "
        f"split_strategy={split_strategy}"
    )

    if optuna_trials > 0:
        run_rolling_orchestrator(
            db=db,
            stage_info=stage_info,
            optuna_storage_url=optuna_storage_url,
            max_parallel_tasks=max_parallel_tasks,
            poll_seconds=poll_seconds,
            seed=seed,
            stop_when_complete=stop_when_complete,
            dispatcher_batch_start=dispatcher_batch_start,
        )
        return None

    if int(stage_info.get("stage_no")) == 5 and split_strategy == "REPEATED_5_FOLD":
        return create_final_repeated_kfold_task_from_stage4_best(
            db=db,
            final_stage_info=stage_info,
            optuna_storage_url=optuna_storage_url,
            source_stage_no=source_best_stage_no,
            seed=seed,
            dispatcher_batch_no=dispatcher_batch_start,
        )

    raise ValueError(f"Stage non-HPO belum punya dispatcher otomatis: {stage_info}")


def run_rolling_orchestrator(
    *,
    db: PostgresOrchestrationDb,
    stage_info: dict[str, Any],
    optuna_storage_url: str,
    max_parallel_tasks: int = 1,
    poll_seconds: int = 30,
    seed: int = 42,
    stop_when_complete: bool = True,
    dispatcher_batch_start: int = 1,
) -> None:
    """Keep HPO tasks rolling for one stage/model.

    stage_info comes from cqcnn_orchestration.stage_information and must contain:
    stage_no, model_type, optuna_study_name, optuna_direction, optuna_trials,
    search_space_json, and early_stop_monitor.
    """
    stage_no = int(stage_info["stage_no"])
    model_type = str(stage_info["model_type"])
    study_name = str(stage_info["optuna_study_name"])
    direction = str(stage_info["optuna_direction"])
    optuna_trials = int(stage_info["optuna_trials"])
    search_space = stage_info.get("search_space_json") or {}
    objective_metric_name = stage_info.get("early_stop_monitor") or (
        "val_loss" if direction == "minimize" else "val_macro_f1"
    )

    if optuna_trials <= 0:
        raise ValueError("Rolling orchestrator is only for HPO stages with optuna_trials > 0.")
    if max_parallel_tasks < 1:
        raise ValueError("max_parallel_tasks must be >= 1.")

    study = create_or_load_study(
        study_name=study_name,
        storage_url=optuna_storage_url,
        direction=direction,
        seed=seed,
        constant_liar=True,
    )

    dispatcher_batch_no = dispatcher_batch_start
    print(
        f"[orchestrator] started stage={stage_no}, model={model_type}, "
        f"study={study_name}, storage=PostgreSQL, max_parallel_tasks={max_parallel_tasks}"
    )

    while True:
        status = db.get_dispatcher_stage_status(stage_no, model_type, max_parallel_tasks)
        signal = status.get("dispatcher_signal")
        can_generate = int(status.get("can_generate_count") or 0)

        print(
            "[orchestrator] "
            f"signal={signal}, active={status.get('active_tasks')}, "
            f"told={status.get('told_tasks')}, waiting={status.get('waiting_tasks')}, "
            f"running={status.get('running_tasks')}, can_generate={can_generate}"
        )

        if signal == "STAGE_COMPLETE":
            print(f"[orchestrator] stage complete: stage={stage_no}, model={model_type}")
            if stop_when_complete:
                return

        for _ in range(can_generate):
            # ask + suggest is centralized here so trial generation remains controlled.
            trial = study.ask()
            params = suggest_from_search_space(trial, search_space)

            task = db.create_task_with_slot(
                stage_no=stage_no,
                model_type=model_type,
                optuna_study_name=study_name,
                trial_nr=trial.number,
                trial_params_json=params,  # audit snapshot only; Optuna PostgreSQL remains authoritative.
                objective_metric_name=objective_metric_name,
                objective_direction=direction,
                optuna_tell_status="PENDING",
                dispatcher_batch_no=dispatcher_batch_no,
            )
            print(
                f"[orchestrator] generated task={task['task_id']} "
                f"trial={trial.number} batch={dispatcher_batch_no}"
            )
            dispatcher_batch_no += 1

        time.sleep(poll_seconds)
