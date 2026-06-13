-- =============================================================
-- 06_readiness_checks.sql
-- Readiness checks setelah 01, 02, dan 03 dijalankan.
-- =============================================================

-- 1. Stage 3/4/5 policy check.
select
    stage_no,
    model_type,
    stage_objective,
    split_strategy,
    k_folds,
    n_repeats,
    optuna_trials,
    optuna_direction,
    tuning_focus,
    early_stop_monitor,
    early_stop_mode
from public.stage_information
where stage_no in (3,4,5)
order by stage_no, model_type;

-- 2. Checkpoint metadata columns for Google Drive/rclone pointers exist.
select
    column_name,
    data_type,
    is_nullable
from information_schema.columns
where table_schema = 'public'
  and table_name = 'checkpoint_file'
  and column_name in (
      'storage_backend',
      'rclone_remote',
      'gdrive_relative_path',
      'checkpoint_uri',
      'local_cache_path',
      'sha256',
      'file_size_bytes',
      'metric_name',
      'metric_value',
      'upload_status',
      'has_model_state',
      'has_optimizer_state',
      'has_scheduler_state',
      'has_model_config',
      'has_runtime_plan',
      'has_seed'
  )
order by column_name;

-- 3. Task columns for Optuna PostgreSQL dispatcher exist.
select
    column_name,
    data_type,
    is_nullable
from information_schema.columns
where table_schema = 'public'
  and table_name = 'task'
  and column_name in (
      'trial_params_json',
      'objective_metric_name',
      'objective_value',
      'objective_direction',
      'optuna_tell_status',
      'optuna_told_at',
      'dispatcher_batch_no',
      'dispatcher_generated_at',
      'completed_by_worker_at'
  )
order by column_name;

-- 4. Required functions exist.
select routine_name
from information_schema.routines
where specific_schema = 'public'
  and routine_name in (
      'register_worker',
      'create_task_with_slot',
      'get_dispatcher_stage_status',
      'get_tasks_ready_for_tell',
      'get_worker_stage_signal',
      'claim_waiting_task',
      'update_task_heartbeat',
      'mark_stale_tasks',
      'hijack_stale_task',
      'register_checkpoint_file',
      'get_resume_checkpoint',
      'mark_task_done_waiting_tell',
      'mark_task_done',
      'mark_task_told',
      'mark_task_failed'
  )
order by routine_name;

-- 5. Worker signal dry-run for Stage 4 hybrid.
select *
from public.get_worker_stage_signal(4, 'hybrid_qcqcnn');

-- 6. Dispatcher status dry-run for Stage 4 hybrid.
select *
from public.get_dispatcher_stage_status(4, 'hybrid_qcqcnn', 1);

-- 7. Monitoring views smoke test.
select * from public.v_stage_progress order by stage_no, model_type;
select * from public.v_task_monitor order by task_id limit 20;

-- Extra readiness check for Optuna PostgreSQL worker-direct tell support.
select
    'function_mark_task_told_by_worker' as check_name,
    case when exists (
        select 1
        from pg_proc p
        join pg_namespace n on n.oid = p.pronamespace
        where n.nspname = 'public'
          and p.proname = 'mark_task_told_by_worker'
    ) then 'OK' else 'MISSING' end as status;
