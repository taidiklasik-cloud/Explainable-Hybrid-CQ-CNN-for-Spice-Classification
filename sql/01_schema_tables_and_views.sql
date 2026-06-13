-- =============================================================
-- 01_schema_tables_and_views.sql
-- Local PostgreSQL schema for CQ-CNN / QCQ-CNN worker orchestration.
-- Target database: cqcnn_orchestration.
-- Versi: local-postgres-rclone refactor.
-- Jalankan di PostgreSQL lokal. Aman untuk rerun karena memakai IF NOT EXISTS / DROP TRIGGER.
-- =============================================================

begin;


-- -------------------------------------------------------------
-- 1. Utility trigger updated_at
-- -------------------------------------------------------------
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

-- -------------------------------------------------------------
-- 2. STAGE_INFORMATION
-- -------------------------------------------------------------
create table if not exists public.stage_information (
    stage_no integer not null,
    model_type text not null,

    stage_name text not null,
    stage_objective text not null,
    is_active boolean not null default false,

    train_ratio numeric(4,2) not null default 0.80,
    validation_ratio numeric(4,2) not null default 0.20,

    split_strategy text not null,
    k_folds integer,
    n_repeats integer not null default 1,

    max_epoch integer not null,

    optuna_study_name text,
    optuna_trials integer not null default 0,
    optuna_direction text,
    tuning_focus text,

    search_space_json jsonb,

    early_stop_monitor text,
    early_stop_mode text,
    early_stop_patience integer,
    early_stop_min_delta numeric,
    min_epoch_before_stop integer,

    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    primary key (stage_no, model_type)
);

alter table public.stage_information drop constraint if exists chk_stage_model_type;
alter table public.stage_information add constraint chk_stage_model_type check (
    model_type in ('classical_fully_spatial', 'hybrid_qcqcnn')
);
alter table public.stage_information drop constraint if exists chk_stage_objective;
alter table public.stage_information add constraint chk_stage_objective check (
    stage_objective in ('SANITY', 'WARM_START', 'CONVERGENCE', 'MAX_ACCURACY', 'FINALIZATION')
);
alter table public.stage_information drop constraint if exists chk_stage_split_strategy;
alter table public.stage_information add constraint chk_stage_split_strategy check (
    split_strategy in ('SIMPLE_80_20', 'CV_5_FOLD', 'REPEATED_5_FOLD')
);
alter table public.stage_information drop constraint if exists chk_stage_optuna_direction;
alter table public.stage_information add constraint chk_stage_optuna_direction check (
    optuna_direction is null or optuna_direction in ('minimize', 'maximize')
);
alter table public.stage_information drop constraint if exists chk_stage_tuning_focus;
alter table public.stage_information add constraint chk_stage_tuning_focus check (
    tuning_focus is null or tuning_focus in ('NONE', 'CONVERGENCE', 'ACCURACY', 'FINAL_EVALUATION')
);
alter table public.stage_information drop constraint if exists chk_stage_early_stop_mode;
alter table public.stage_information add constraint chk_stage_early_stop_mode check (
    early_stop_mode is null or early_stop_mode in ('min', 'max')
);
alter table public.stage_information drop constraint if exists chk_stage_early_stop_monitor;
alter table public.stage_information add constraint chk_stage_early_stop_monitor check (
    early_stop_monitor is null or early_stop_monitor in ('val_loss', 'val_accuracy', 'val_macro_f1', 'val_balanced_accuracy')
);
alter table public.stage_information drop constraint if exists chk_stage_ratio;
alter table public.stage_information add constraint chk_stage_ratio check (
    train_ratio > 0 and validation_ratio >= 0 and train_ratio <= 1 and validation_ratio <= 1
);
alter table public.stage_information drop constraint if exists chk_stage_epoch;
alter table public.stage_information add constraint chk_stage_epoch check (max_epoch > 0);
alter table public.stage_information drop constraint if exists chk_stage_trials;
alter table public.stage_information add constraint chk_stage_trials check (optuna_trials >= 0);

drop trigger if exists trg_stage_information_updated_at on public.stage_information;
create trigger trg_stage_information_updated_at
before update on public.stage_information
for each row execute function public.set_updated_at();

-- -------------------------------------------------------------
-- 3. WORKER_NODE
-- -------------------------------------------------------------
create table if not exists public.worker_node (
    worker_id bigint generated always as identity primary key,

    worker_uid text not null unique,
    worker_name text,
    hostname text,
    worker_type text not null default 'LOCAL_PC',

    cpu_name text,
    cpu_count integer,
    ram_gb numeric(8,2),

    has_gpu boolean not null default false,
    gpu_name text,
    gpu_count integer,
    gpu_vram_gb numeric(8,2),

    python_version text,
    platform_name text,

    last_seen_at timestamptz,

    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table public.worker_node drop constraint if exists chk_worker_type;
alter table public.worker_node add constraint chk_worker_type check (
    worker_type in ('LOCAL_PC', 'LAPTOP', 'LAB_PC', 'CLOUD_CPU', 'CLOUD_GPU')
);

drop trigger if exists trg_worker_node_updated_at on public.worker_node;
create trigger trg_worker_node_updated_at
before update on public.worker_node
for each row execute function public.set_updated_at();

-- -------------------------------------------------------------
-- 4. CHECKPOINT_SLOT
-- -------------------------------------------------------------
create table if not exists public.checkpoint_slot (
    checkpoint_slot_id bigint generated always as identity primary key,

    latest_checkpoint_file_id bigint,
    best_checkpoint_file_id bigint,
    final_checkpoint_file_id bigint,

    current_repeat_id integer not null default 0,
    current_fold_id integer not null default 0,
    current_epoch integer not null default 0,
    global_step bigint not null default 0,

    best_metric_name text,
    best_metric_value numeric,

    has_model_state boolean not null default true,
    has_optimizer_state boolean not null default true,
    has_scheduler_state boolean not null default true,
    optimizer_name text not null default 'AdamW',

    resume_status text not null default 'EMPTY',

    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table public.checkpoint_slot drop constraint if exists chk_checkpoint_slot_resume_status;
alter table public.checkpoint_slot add constraint chk_checkpoint_slot_resume_status check (
    resume_status in ('EMPTY', 'READY', 'FINALIZED', 'BROKEN')
);

drop trigger if exists trg_checkpoint_slot_updated_at on public.checkpoint_slot;
create trigger trg_checkpoint_slot_updated_at
before update on public.checkpoint_slot
for each row execute function public.set_updated_at();

-- -------------------------------------------------------------
-- 5. TASK
-- -------------------------------------------------------------
create table if not exists public.task (
    task_id bigint generated always as identity primary key,

    stage_no integer not null,
    model_type text not null,

    checkpoint_slot_id bigint not null unique,
    fk_worker_id bigint,

    optuna_study_name text,
    trial_nr integer,

    status_task text not null default 'WAITING',

    last_heartbeat timestamptz,
    started_at timestamptz,
    finished_at timestamptz,
    stale_marked_at timestamptz,
    hijack_count integer not null default 0,

    error_message text,

    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    constraint fk_task_stage
        foreign key (stage_no, model_type)
        references public.stage_information(stage_no, model_type)
        on update cascade
        on delete restrict,

    constraint fk_task_checkpoint_slot
        foreign key (checkpoint_slot_id)
        references public.checkpoint_slot(checkpoint_slot_id)
        on update cascade
        on delete restrict,

    constraint fk_task_worker
        foreign key (fk_worker_id)
        references public.worker_node(worker_id)
        on update cascade
        on delete set null
);

-- Idempotent columns for Optuna PostgreSQL dispatcher + worker queue.
alter table public.task add column if not exists trial_params_json jsonb;
alter table public.task add column if not exists objective_metric_name text;
alter table public.task add column if not exists objective_value numeric;
alter table public.task add column if not exists objective_direction text;
alter table public.task add column if not exists optuna_tell_status text not null default 'NOT_REQUIRED';
alter table public.task add column if not exists optuna_told_at timestamptz;
alter table public.task add column if not exists dispatcher_batch_no integer;
alter table public.task add column if not exists dispatcher_generated_at timestamptz;
alter table public.task add column if not exists completed_by_worker_at timestamptz;

alter table public.task drop constraint if exists chk_task_model_type;
alter table public.task add constraint chk_task_model_type check (
    model_type in ('classical_fully_spatial', 'hybrid_qcqcnn')
);
alter table public.task drop constraint if exists chk_task_status;
alter table public.task add constraint chk_task_status check (
    status_task in (
        'WAITING',
        'RUNNING',
        'DONE_WAITING_TELL',
        'TOLD',
        'DONE',
        'FAILED',
        'STALE',
        'HIJACKED',
        'CANCELLED'
    )
);
alter table public.task drop constraint if exists chk_task_hijack_count;
alter table public.task add constraint chk_task_hijack_count check (hijack_count >= 0);
alter table public.task drop constraint if exists chk_task_objective_direction;
alter table public.task add constraint chk_task_objective_direction check (
    objective_direction is null or objective_direction in ('minimize', 'maximize')
);
alter table public.task drop constraint if exists chk_task_optuna_tell_status;
alter table public.task add constraint chk_task_optuna_tell_status check (
    optuna_tell_status in ('NOT_REQUIRED', 'PENDING', 'TOLD', 'FAILED')
);

drop trigger if exists trg_task_updated_at on public.task;
create trigger trg_task_updated_at
before update on public.task
for each row execute function public.set_updated_at();

-- -------------------------------------------------------------
-- 6. CHECKPOINT_FILE
-- Metadata dan pointer checkpoint. File fisik .pt tidak disimpan di PostgreSQL;
-- interval/best/final checkpoint disimpan di Google Drive via rclone.
-- -------------------------------------------------------------
create table if not exists public.checkpoint_file (
    checkpoint_file_id bigint generated always as identity primary key,

    checkpoint_slot_id bigint not null,
    task_id bigint not null,
    worker_id bigint,

    stage_no integer not null,
    model_type text not null,

    trial_nr integer,
    repeat_id integer not null default 0,
    fold_id integer not null default 0,
    epoch_number integer,
    global_step bigint,

    checkpoint_type text not null,

    storage_backend text not null default 'gdrive_rclone',
    rclone_remote text not null default 'gdrive',
    gdrive_relative_path text not null,
    checkpoint_uri text generated always as (
        'gdrive://' || rclone_remote || '/' || gdrive_relative_path
    ) stored,
    local_cache_path text,

    file_name text,
    sha256 text,
    file_size_bytes bigint,

    has_model_state boolean not null default true,
    has_optimizer_state boolean not null default true,
    has_scheduler_state boolean not null default true,
    has_model_config boolean not null default true,
    has_runtime_plan boolean not null default true,
    has_seed boolean not null default true,
    optimizer_name text not null default 'AdamW',

    upload_status text not null default 'UPLOADED',

    metric_name text,
    metric_value numeric,

    created_at timestamptz not null default now(),

    constraint fk_checkpoint_file_slot
        foreign key (checkpoint_slot_id)
        references public.checkpoint_slot(checkpoint_slot_id)
        on update cascade
        on delete restrict,

    constraint fk_checkpoint_file_task
        foreign key (task_id)
        references public.task(task_id)
        on update cascade
        on delete restrict,

    constraint fk_checkpoint_file_worker
        foreign key (worker_id)
        references public.worker_node(worker_id)
        on update cascade
        on delete set null,

    constraint fk_checkpoint_file_stage
        foreign key (stage_no, model_type)
        references public.stage_information(stage_no, model_type)
        on update cascade
        on delete restrict
);

alter table public.checkpoint_file add column if not exists storage_backend text not null default 'gdrive_rclone';
alter table public.checkpoint_file add column if not exists rclone_remote text not null default 'gdrive';
alter table public.checkpoint_file add column if not exists gdrive_relative_path text not null default '';
alter table public.checkpoint_file add column if not exists checkpoint_uri text generated always as (
    'gdrive://' || rclone_remote || '/' || gdrive_relative_path
) stored;
alter table public.checkpoint_file add column if not exists local_cache_path text;
alter table public.checkpoint_file add column if not exists has_model_config boolean not null default true;
alter table public.checkpoint_file add column if not exists has_runtime_plan boolean not null default true;
alter table public.checkpoint_file add column if not exists has_seed boolean not null default true;
alter table public.checkpoint_file add column if not exists metric_name text;
alter table public.checkpoint_file add column if not exists metric_value numeric;

alter table public.checkpoint_file drop constraint if exists chk_checkpoint_file_model_type;
alter table public.checkpoint_file add constraint chk_checkpoint_file_model_type check (
    model_type in ('classical_fully_spatial', 'hybrid_qcqcnn')
);
alter table public.checkpoint_file drop constraint if exists chk_checkpoint_file_type;
alter table public.checkpoint_file add constraint chk_checkpoint_file_type check (
    checkpoint_type in ('INTERVAL', 'EPOCH', 'BEST', 'FINAL', 'RECOVERY')
);
alter table public.checkpoint_file drop constraint if exists chk_checkpoint_upload_status;
alter table public.checkpoint_file add constraint chk_checkpoint_upload_status check (
    upload_status in ('LOCAL_ONLY', 'PENDING_UPLOAD', 'UPLOADED', 'VERIFIED', 'FAILED')
);
alter table public.checkpoint_file drop constraint if exists chk_checkpoint_storage_backend;
alter table public.checkpoint_file add constraint chk_checkpoint_storage_backend check (
    storage_backend in ('gdrive_rclone')
);
alter table public.checkpoint_file drop constraint if exists chk_checkpoint_file_size;
alter table public.checkpoint_file add constraint chk_checkpoint_file_size check (
    file_size_bytes is null or file_size_bytes >= 0
);

-- Add FK from slot to checkpoint_file after checkpoint_file exists.
alter table public.checkpoint_slot drop constraint if exists fk_slot_latest_checkpoint;
alter table public.checkpoint_slot
add constraint fk_slot_latest_checkpoint
foreign key (latest_checkpoint_file_id)
references public.checkpoint_file(checkpoint_file_id)
on update cascade
on delete set null;

alter table public.checkpoint_slot drop constraint if exists fk_slot_best_checkpoint;
alter table public.checkpoint_slot
add constraint fk_slot_best_checkpoint
foreign key (best_checkpoint_file_id)
references public.checkpoint_file(checkpoint_file_id)
on update cascade
on delete set null;

alter table public.checkpoint_slot drop constraint if exists fk_slot_final_checkpoint;
alter table public.checkpoint_slot
add constraint fk_slot_final_checkpoint
foreign key (final_checkpoint_file_id)
references public.checkpoint_file(checkpoint_file_id)
on update cascade
on delete set null;

-- -------------------------------------------------------------
-- 7. Indexes
-- -------------------------------------------------------------
create index if not exists idx_stage_information_active on public.stage_information(is_active);
create index if not exists idx_task_status on public.task(status_task);
create index if not exists idx_task_stage_model_status on public.task(stage_no, model_type, status_task);
create index if not exists idx_task_heartbeat on public.task(last_heartbeat);
create index if not exists idx_task_worker_status on public.task(fk_worker_id, status_task);
create index if not exists idx_task_optuna_tell on public.task(stage_no, model_type, optuna_tell_status, status_task);
create index if not exists idx_checkpoint_file_task on public.checkpoint_file(task_id);
create index if not exists idx_checkpoint_file_slot on public.checkpoint_file(checkpoint_slot_id);
create index if not exists idx_checkpoint_file_type on public.checkpoint_file(checkpoint_type);
create index if not exists idx_checkpoint_file_stage_model on public.checkpoint_file(stage_no, model_type, trial_nr);
create index if not exists idx_checkpoint_file_upload_status on public.checkpoint_file(upload_status);
create index if not exists idx_checkpoint_file_gdrive_path on public.checkpoint_file(gdrive_relative_path);
create index if not exists idx_worker_node_uid on public.worker_node(worker_uid);

-- -------------------------------------------------------------
-- 8. Monitoring views
-- -------------------------------------------------------------
create or replace view public.v_stage_progress as
select
    s.stage_no,
    s.model_type,
    s.stage_name,
    s.stage_objective,
    s.optuna_study_name,
    s.optuna_trials,
    count(t.task_id) as total_tasks,
    count(*) filter (where t.status_task = 'WAITING') as waiting_tasks,
    count(*) filter (where t.status_task = 'RUNNING') as running_tasks,
    count(*) filter (where t.status_task = 'DONE_WAITING_TELL') as done_waiting_tell_tasks,
    count(*) filter (where t.status_task = 'TOLD') as told_tasks,
    count(*) filter (where t.status_task = 'DONE') as done_tasks,
    count(*) filter (where t.status_task = 'FAILED') as failed_tasks,
    count(*) filter (where t.status_task = 'STALE') as stale_tasks,
    count(*) filter (where t.status_task = 'CANCELLED') as cancelled_tasks,
    count(*) filter (where t.optuna_tell_status = 'PENDING') as optuna_pending_tell_tasks,
    count(*) filter (where t.optuna_tell_status = 'TOLD') as optuna_told_tasks,
    greatest(s.optuna_trials - count(*) filter (where t.status_task in ('WAITING','RUNNING','DONE_WAITING_TELL','TOLD','FAILED','CANCELLED')), 0) as remaining_hpo_trials_to_generate,
    max(t.updated_at) as last_task_update
from public.stage_information s
left join public.task t
    on t.stage_no = s.stage_no
   and t.model_type = s.model_type
group by
    s.stage_no,
    s.model_type,
    s.stage_name,
    s.stage_objective,
    s.optuna_study_name,
    s.optuna_trials;

create or replace view public.v_task_monitor as
select
    t.task_id,
    t.stage_no,
    t.model_type,
    s.stage_name,
    t.status_task,
    t.optuna_study_name,
    t.trial_nr,
    t.trial_params_json,
    t.objective_metric_name,
    t.objective_value,
    t.objective_direction,
    t.optuna_tell_status,
    t.optuna_told_at,
    t.dispatcher_batch_no,
    t.checkpoint_slot_id,
    t.fk_worker_id,
    w.worker_uid,
    w.worker_name,
    t.last_heartbeat,
    now() - t.last_heartbeat as heartbeat_age,
    t.started_at,
    t.completed_by_worker_at,
    t.finished_at,
    t.stale_marked_at,
    t.hijack_count,
    cs.current_repeat_id,
    cs.current_fold_id,
    cs.current_epoch,
    cs.global_step,
    cs.resume_status,
    cs.best_metric_name,
    cs.best_metric_value,
    t.error_message,
    t.created_at,
    t.updated_at
from public.task t
join public.stage_information s
    on s.stage_no = t.stage_no
   and s.model_type = t.model_type
left join public.worker_node w
    on w.worker_id = t.fk_worker_id
left join public.checkpoint_slot cs
    on cs.checkpoint_slot_id = t.checkpoint_slot_id;

create or replace view public.v_stale_candidates as
select *
from public.v_task_monitor
where status_task = 'RUNNING'
  and last_heartbeat is not null
  and last_heartbeat < now() - interval '15 minutes';

create or replace view public.v_worker_monitor as
select
    w.worker_id,
    w.worker_uid,
    w.worker_name,
    w.hostname,
    w.worker_type,
    w.has_gpu,
    w.gpu_name,
    w.gpu_vram_gb,
    w.last_seen_at,
    now() - w.last_seen_at as last_seen_age,
    count(t.task_id) filter (where t.status_task = 'RUNNING') as running_tasks,
    count(t.task_id) filter (where t.status_task in ('DONE','TOLD')) as completed_tasks,
    count(t.task_id) filter (where t.status_task = 'FAILED') as failed_tasks
from public.worker_node w
left join public.task t
    on t.fk_worker_id = w.worker_id
group by w.worker_id;

create or replace view public.v_checkpoint_monitor as
select
    cf.checkpoint_file_id,
    cf.task_id,
    cf.stage_no,
    cf.model_type,
    cf.trial_nr,
    cf.repeat_id,
    cf.fold_id,
    cf.epoch_number,
    cf.global_step,
    cf.checkpoint_type,
    cf.storage_backend,
    cf.rclone_remote,
    cf.gdrive_relative_path,
    cf.checkpoint_uri,
    cf.local_cache_path,
    cf.sha256,
    cf.file_size_bytes,
    cf.metric_name,
    cf.metric_value,
    cf.upload_status,
    cf.created_at
from public.checkpoint_file cf;

commit;
