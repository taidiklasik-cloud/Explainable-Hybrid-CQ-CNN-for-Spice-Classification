"""PostgreSQL helper for CQ-CNN / Hybrid QCQ-CNN orchestration.

This module connects to the local PostgreSQL orchestration database
(`cqcnn_orchestration`) through ORCHESTRATION_DB_DSN. It owns stage/task/worker,
heartbeat, checkpoint metadata, resume, and hijacking queries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb

    _DB_DRIVER = "psycopg"
except ImportError:
    import psycopg2
    import psycopg2.extras

    Jsonb = psycopg2.extras.Json
    _DB_DRIVER = "psycopg2"


@dataclass(frozen=True)
class PostgresOrchestrationDbConfig:
    dsn: str


class PostgresOrchestrationDb:
    def __init__(self, config: PostgresOrchestrationDbConfig):
        self.config = config

    def _fetchone(self, sql: str, params: Any = None) -> Optional[dict[str, Any]]:
        if _DB_DRIVER == "psycopg":
            with psycopg.connect(self.config.dsn, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params or {})
                    row = cur.fetchone()
                    conn.commit()
                    return dict(row) if row else None

        conn = psycopg2.connect(self.config.dsn)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params or {})
                row = cur.fetchone()
                conn.commit()
                return dict(row) if row else None
        finally:
            conn.close()

    def _fetchall(self, sql: str, params: Any = None) -> list[dict[str, Any]]:
        if _DB_DRIVER == "psycopg":
            with psycopg.connect(self.config.dsn, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params or {})
                    rows = cur.fetchall()
                    conn.commit()
                    return [dict(r) for r in rows]

        conn = psycopg2.connect(self.config.dsn)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params or {})
                rows = cur.fetchall()
                conn.commit()
                return [dict(r) for r in rows]
        finally:
            conn.close()

    def _execute(self, sql: str, params: Any = None) -> None:
        if _DB_DRIVER == "psycopg":
            with psycopg.connect(self.config.dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params or {})
                    conn.commit()
            return

        conn = psycopg2.connect(self.config.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or {})
                conn.commit()
        finally:
            conn.close()

    # ---------- Smoke tests / direct reads ----------
    def test_connection(self) -> dict[str, Any]:
        row = self._fetchone(
            "select current_database() as database_name, current_user as user_name, now() as checked_at"
        )
        if not row:
            raise RuntimeError("PostgreSQL orchestration connection test returned no row.")
        return row

    def query_stage_information(
        self,
        *,
        stage_no: int | None = None,
        model_type: str | None = None,
        only_active: bool | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if stage_no is not None:
            clauses.append("stage_no = %(stage_no)s")
            params["stage_no"] = int(stage_no)
        if model_type is not None:
            clauses.append("model_type = %(model_type)s")
            params["model_type"] = model_type
        if only_active is not None:
            clauses.append("is_active = %(only_active)s")
            params["only_active"] = bool(only_active)

        where_sql = f"where {' and '.join(clauses)}" if clauses else ""
        return self._fetchall(
            f"""
            select *
            from public.stage_information
            {where_sql}
            order by stage_no, model_type
            """,
            params,
        )

    # ---------- Worker registration ----------
    def register_worker(self, **kwargs: Any) -> int:
        row = self._fetchone(
            """select public.register_worker(
                p_worker_uid := %(worker_uid)s::text,
                p_worker_name := %(worker_name)s::text,
                p_hostname := %(hostname)s::text,
                p_worker_type := %(worker_type)s::text,
                p_cpu_name := %(cpu_name)s::text,
                p_cpu_count := %(cpu_count)s::integer,
                p_ram_gb := %(ram_gb)s::numeric,
                p_has_gpu := %(has_gpu)s::boolean,
                p_gpu_name := %(gpu_name)s::text,
                p_gpu_count := %(gpu_count)s::integer,
                p_gpu_vram_gb := %(gpu_vram_gb)s::numeric,
                p_python_version := %(python_version)s::text,
                p_platform_name := %(platform_name)s::text
            ) as worker_id""",
            kwargs,
        )
        return int(row["worker_id"])

    # ---------- Dispatcher helpers ----------
    def get_stage_info(self, stage_no: int, model_type: str) -> dict[str, Any]:
        row = self._fetchone(
            "select * from public.stage_information where stage_no = %s and model_type = %s",
            (stage_no, model_type),
        )
        if not row:
            raise ValueError(f"stage_information not found: stage_no={stage_no}, model_type={model_type}")
        return row

    def get_dispatcher_stage_status(self, stage_no: int, model_type: str, max_parallel_tasks: int) -> dict[str, Any]:
        row = self._fetchone(
            "select * from public.get_dispatcher_stage_status(%s, %s, %s)",
            (stage_no, model_type, max_parallel_tasks),
        )
        return row or {}

    def create_task_with_slot(
        self,
        *,
        stage_no: int,
        model_type: str,
        optuna_study_name: str | None = None,
        trial_nr: int | None = None,
        trial_params_json: dict[str, Any] | None = None,
        objective_metric_name: str | None = None,
        objective_direction: str | None = None,
        optuna_tell_status: str | None = None,
        dispatcher_batch_no: int | None = None,
    ) -> dict[str, int]:
        row = self._fetchone(
            """select * from public.create_task_with_slot(
                p_stage_no := %(stage_no)s::integer,
                p_model_type := %(model_type)s::text,
                p_optuna_study_name := %(optuna_study_name)s::text,
                p_trial_nr := %(trial_nr)s::integer,
                p_trial_params_json := %(trial_params_json)s::jsonb,
                p_objective_metric_name := %(objective_metric_name)s::text,
                p_objective_direction := %(objective_direction)s::text,
                p_optuna_tell_status := %(optuna_tell_status)s::text,
                p_dispatcher_batch_no := %(dispatcher_batch_no)s::integer
            )""",
            {
                "stage_no": stage_no,
                "model_type": model_type,
                "optuna_study_name": optuna_study_name,
                "trial_nr": trial_nr,
                "trial_params_json": Jsonb(trial_params_json or {}),
                "objective_metric_name": objective_metric_name,
                "objective_direction": objective_direction,
                "optuna_tell_status": optuna_tell_status,
                "dispatcher_batch_no": dispatcher_batch_no,
            },
        )
        return {"task_id": int(row["task_id"]), "checkpoint_slot_id": int(row["checkpoint_slot_id"])}

    def get_tasks_ready_for_tell(self, stage_no: int, model_type: str, limit: int = 50) -> list[dict[str, Any]]:
        return self._fetchall(
            "select * from public.get_tasks_ready_for_tell(%s::integer, %s::text, %s::integer)",
            (stage_no, model_type, limit),
        )

    def mark_task_told(self, task_id: int) -> bool:
        row = self._fetchone("select public.mark_task_told(%s::integer) as ok", (task_id,))
        return bool(row and row["ok"])

    # ---------- Worker polling helpers ----------
    def get_worker_stage_signal(self, stage_no: int, model_type: str) -> dict[str, Any]:
        row = self._fetchone(
            "select * from public.get_worker_stage_signal(%s::integer, %s::text)",
            (stage_no, model_type),
        )
        return row or {"signal": "IDLE", "reason": "No signal row returned."}

    def claim_waiting_task(self, worker_uid: str, stage_no: int, model_type: str) -> Optional[dict[str, Any]]:
        rows = self._fetchall(
            "select * from public.claim_waiting_task(%s::text, %s::integer, %s::text)",
            (worker_uid, stage_no, model_type),
        )
        return rows[0] if rows else None

    def heartbeat(self, task_id: int, worker_uid: str) -> bool:
        row = self._fetchone(
            "select public.update_task_heartbeat(%s::integer, %s::text) as ok",
            (task_id, worker_uid),
        )
        return bool(row and row["ok"])

    def mark_stale_tasks(self, stale_after: str = "15 minutes") -> int:
        row = self._fetchone("select public.mark_stale_tasks(%s::interval) as n", (stale_after,))
        return int(row["n"] if row else 0)

    def hijack_stale_task(
        self,
        worker_uid: str,
        stage_no: int,
        model_type: str,
        stale_after: str = "15 minutes",
    ) -> Optional[dict[str, Any]]:
        rows = self._fetchall(
            "select * from public.hijack_stale_task(%s::text, %s::integer, %s::text, %s::interval)",
            (worker_uid, stage_no, model_type, stale_after),
        )
        return rows[0] if rows else None

    # ---------- Checkpoint metadata ----------
    def register_checkpoint_file(self, **kwargs: Any) -> int:
        params = {
            "task_id": kwargs["task_id"],
            "worker_uid": kwargs["worker_uid"],
            "checkpoint_type": kwargs["checkpoint_type"],
            "gdrive_relative_path": kwargs.get("gdrive_relative_path"),
            "file_name": kwargs.get("file_name"),
            "sha256": kwargs.get("sha256"),
            "file_size_bytes": kwargs.get("file_size_bytes"),
            "epoch_number": kwargs.get("epoch_number"),
            "global_step": kwargs.get("global_step"),
            "repeat_id": kwargs.get("repeat_id", 0),
            "fold_id": kwargs.get("fold_id", 0),
            "metric_name": kwargs.get("metric_name")
            if kwargs.get("metric_name") is not None
            else kwargs.get("best_metric_name"),
            "metric_value": kwargs.get("metric_value")
            if kwargs.get("metric_value") is not None
            else kwargs.get("best_metric_value"),
            "upload_status": kwargs.get("upload_status", "UPLOADED"),
            "local_cache_path": kwargs.get("local_cache_path"),
            "rclone_remote": kwargs.get("rclone_remote", "gdrive"),
            "storage_backend": kwargs.get("storage_backend", "gdrive_rclone"),
            "has_model_state": kwargs.get("has_model_state", True),
            "has_optimizer_state": kwargs.get("has_optimizer_state", True),
            "has_scheduler_state": kwargs.get("has_scheduler_state", True),
            "has_model_config": kwargs.get("has_model_config", True),
            "has_runtime_plan": kwargs.get("has_runtime_plan", True),
            "has_seed": kwargs.get("has_seed", True),
            "optimizer_name": kwargs.get("optimizer_name", "AdamW"),
        }
        if not params["gdrive_relative_path"]:
            raise ValueError("gdrive_relative_path is required for checkpoint metadata.")

        row = self._fetchone(
            """select public.register_checkpoint_file(
                p_task_id := %(task_id)s::bigint,
                p_worker_uid := %(worker_uid)s::text,
                p_checkpoint_type := %(checkpoint_type)s::text,
                p_gdrive_relative_path := %(gdrive_relative_path)s::text,
                p_file_name := %(file_name)s::text,
                p_sha256 := %(sha256)s::text,
                p_file_size_bytes := %(file_size_bytes)s::bigint,
                p_epoch_number := %(epoch_number)s::integer,
                p_global_step := %(global_step)s::bigint,
                p_repeat_id := %(repeat_id)s::integer,
                p_fold_id := %(fold_id)s::integer,
                p_metric_name := %(metric_name)s::text,
                p_metric_value := %(metric_value)s::numeric,
                p_upload_status := %(upload_status)s::text,
                p_local_cache_path := %(local_cache_path)s::text,
                p_rclone_remote := %(rclone_remote)s::text,
                p_storage_backend := %(storage_backend)s::text,
                p_has_model_state := %(has_model_state)s::boolean,
                p_has_optimizer_state := %(has_optimizer_state)s::boolean,
                p_has_scheduler_state := %(has_scheduler_state)s::boolean,
                p_has_model_config := %(has_model_config)s::boolean,
                p_has_runtime_plan := %(has_runtime_plan)s::boolean,
                p_has_seed := %(has_seed)s::boolean,
                p_optimizer_name := %(optimizer_name)s::text
            ) as checkpoint_file_id""",
            params,
        )
        return int(row["checkpoint_file_id"])

    def get_resume_checkpoint(self, checkpoint_slot_id: int, prefer: str = "LATEST") -> Optional[dict[str, Any]]:
        rows = self._fetchall("select * from public.get_resume_checkpoint(%s, %s)", (checkpoint_slot_id, prefer))
        return rows[0] if rows else None

    # ---------- Finish / fail ----------
    def mark_told_by_worker(
        self,
        task_id: int,
        worker_uid: str,
        objective_metric_name: str,
        objective_value: float,
    ) -> bool:
        """Mark HPO task as TOLD after the worker already called study.tell()."""
        row = self._fetchone(
            "select public.mark_task_told_by_worker(%s, %s, %s, %s) as ok",
            (task_id, worker_uid, objective_metric_name, objective_value),
        )
        return bool(row and row["ok"])

    def mark_done_waiting_tell(
        self,
        task_id: int,
        worker_uid: str,
        objective_metric_name: str,
        objective_value: float,
    ) -> bool:
        row = self._fetchone(
            "select public.mark_task_done_waiting_tell(%s, %s, %s, %s) as ok",
            (task_id, worker_uid, objective_metric_name, objective_value),
        )
        return bool(row and row["ok"])

    def mark_done(
        self,
        task_id: int,
        worker_uid: str,
        objective_metric_name: str | None = None,
        objective_value: float | None = None,
    ) -> bool:
        row = self._fetchone(
            "select public.mark_task_done(%s::integer, %s::text, %s::text, %s::numeric) as ok",
            (task_id, worker_uid, objective_metric_name, objective_value),
        )
        return bool(row and row["ok"])

    def mark_failed(self, task_id: int, worker_uid: str, error_message: str) -> bool:
        row = self._fetchone("select public.mark_task_failed(%s, %s, %s) as ok", (task_id, worker_uid, error_message))
        return bool(row and row["ok"])

    # ---------- Metrics & Diagnostics Logging ----------
    def log_epoch_metrics(self, **kwargs: Any) -> int:
        params_str = ", ".join([f"p_{k} := %({k})s" for k in kwargs.keys()])
        sql = f"SELECT public.log_epoch_metrics({params_str}) as log_id"
        row = self._fetchone(sql, kwargs)
        return int(row["log_id"])

    def log_fold_run_result(self, **kwargs: Any) -> int:
        params_str = ", ".join([f"p_{k} := %({k})s" for k in kwargs.keys()])
        sql = f"SELECT public.log_fold_run_result({params_str}) as fold_run_id"
        row = self._fetchone(sql, kwargs)
        return int(row["fold_run_id"])

    def log_convergence_diagnostic(self, **kwargs: Any) -> int:
        params_str = ", ".join([f"p_{k} := %({k})s" for k in kwargs.keys()])
        sql = f"SELECT public.log_convergence_diagnostic({params_str}) as diagnostic_id"
        row = self._fetchone(sql, kwargs)
        return int(row["diagnostic_id"])

    def log_experiment_artifact(self, **kwargs: Any) -> int:
        params_str = ", ".join([f"p_{k} := %({k})s" for k in kwargs.keys()])
        sql = f"SELECT public.log_experiment_artifact({params_str}) as artifact_id"
        row = self._fetchone(sql, kwargs)
        return int(row["artifact_id"])

    def log_statistical_test(self, **kwargs: Any) -> int:
        params_str = ", ".join([f"p_{k} := %({k})s" for k in kwargs.keys()])
        sql = f"SELECT public.log_statistical_test({params_str}) as analysis_id"
        row = self._fetchone(sql, kwargs)
        return int(row["analysis_id"])

    def log_effect_size(self, **kwargs: Any) -> int:
        params_str = ", ".join([f"p_{k} := %({k})s" for k in kwargs.keys()])
        sql = f"SELECT public.log_effect_size({params_str}) as analysis_id"
        row = self._fetchone(sql, kwargs)
        return int(row["analysis_id"])

    # ---------- Cleanup / Restart Mechanisms ----------
    def delete_task_data(self, task_id: int) -> None:
        """Purge all metadata and task traces for a specific task_id."""
        self._execute("DELETE FROM public.epoch_metric_log WHERE task_id = %s", (task_id,))
        self._execute("DELETE FROM public.fold_run_result WHERE task_id = %s", (task_id,))
        self._execute("DELETE FROM public.convergence_diagnostic_summary WHERE task_id = %s", (task_id,))
        self._execute("DELETE FROM public.checkpoint_file WHERE task_id = %s", (task_id,))
        
        row = self._fetchone("SELECT checkpoint_slot_id FROM public.task WHERE task_id = %s", (task_id,))
        self._execute("DELETE FROM public.task WHERE task_id = %s", (task_id,))
        
        if row and row.get("checkpoint_slot_id"):
            self._execute("DELETE FROM public.checkpoint_slot WHERE checkpoint_slot_id = %s", (row["checkpoint_slot_id"],))

    def reset_stage_tasks(self, stage_no: int, model_type: str) -> None:
        """Purge all tasks and metrics for a specific stage from scratch."""
        # Find all tasks
        tasks = self._fetchall(
            "SELECT task_id FROM public.task WHERE stage_no = %s AND model_type = %s",
            (stage_no, model_type)
        )
        for t in tasks:
            self.delete_task_data(t["task_id"])
        
        # Reset the stage signal
        self._execute(
            "UPDATE public.stage_information SET is_active = true WHERE stage_no = %s AND model_type = %s",
            (stage_no, model_type)
        )


# Backward-compatible aliases for code that has not yet moved to the canonical name.
LocalOrchestrationDbConfig = PostgresOrchestrationDbConfig
LocalOrchestrationDb = PostgresOrchestrationDb
