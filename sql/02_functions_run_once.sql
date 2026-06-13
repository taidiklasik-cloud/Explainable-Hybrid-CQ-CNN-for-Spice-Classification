-- =============================================================
-- 02_functions_run_once.sql
-- Local PostgreSQL functions/RPC for CQ-CNN / QCQ-CNN worker orchestration.
-- Target database: cqcnn_orchestration.
-- Versi: local-postgres-rclone refactor.
-- Jalankan setelah 01_schema_tables_and_views.sql. Aman untuk rerun.
-- =============================================================

-- Cleanup changed signatures safely.
do $$
declare
    r record;
begin
    for r in
        select p.oid::regprocedure as func_signature
        from pg_proc p
        join pg_namespace n on n.oid = p.pronamespace
        where n.nspname = 'public'
          and p.proname in (
              'register_artifact_file',
              'get_artifacts_for_task',
              'create_task_with_slot',
              'claim_waiting_task',
              'hijack_stale_task',
              'register_checkpoint_file',
              'get_resume_checkpoint',
              'mark_task_done',
              'mark_task_done_waiting_tell',
              'mark_task_told',
              'mark_task_failed',
              'update_task_heartbeat',
              'mark_stale_tasks',
              'get_worker_stage_signal',
              'get_dispatcher_stage_status',
              'get_tasks_ready_for_tell'
          )
    loop
        execute 'drop function if exists ' || r.func_signature || ' cascade';
    end loop;
end $$;

begin;

-- -------------------------------------------------------------
-- 1. Register / upsert worker
-- -------------------------------------------------------------
create or replace function public.register_worker(
    p_worker_uid text,
    p_worker_name text default null,
    p_hostname text default null,
    p_worker_type text default 'LOCAL_PC',
    p_cpu_name text default null,
    p_cpu_count integer default null,
    p_ram_gb numeric default null,
    p_has_gpu boolean default false,
    p_gpu_name text default null,
    p_gpu_count integer default null,
    p_gpu_vram_gb numeric default null,
    p_python_version text default null,
    p_platform_name text default null
)
returns bigint
language plpgsql
security definer
set search_path = public
as $$
declare
    v_worker_id bigint;
begin
    insert into public.worker_node (
        worker_uid,
        worker_name,
        hostname,
        worker_type,
        cpu_name,
        cpu_count,
        ram_gb,
        has_gpu,
        gpu_name,
        gpu_count,
        gpu_vram_gb,
        python_version,
        platform_name,
        last_seen_at
    )
    values (
        p_worker_uid,
        p_worker_name,
        p_hostname,
        p_worker_type,
        p_cpu_name,
        p_cpu_count,
        p_ram_gb,
        p_has_gpu,
        p_gpu_name,
        p_gpu_count,
        p_gpu_vram_gb,
        p_python_version,
        p_platform_name,
        now()
    )
    on conflict (worker_uid) do update
    set
        worker_name = excluded.worker_name,
        hostname = excluded.hostname,
        worker_type = excluded.worker_type,
        cpu_name = excluded.cpu_name,
        cpu_count = excluded.cpu_count,
        ram_gb = excluded.ram_gb,
        has_gpu = excluded.has_gpu,
        gpu_name = excluded.gpu_name,
        gpu_count = excluded.gpu_count,
        gpu_vram_gb = excluded.gpu_vram_gb,
        python_version = excluded.python_version,
        platform_name = excluded.platform_name,
        last_seen_at = now(),
        updated_at = now()
    returning worker_id into v_worker_id;

    return v_worker_id;
end;
$$;

-- -------------------------------------------------------------
-- 2. Create task + checkpoint slot
-- Dipakai dispatcher/master. Untuk Optuna PostgreSQL rolling dispatcher,
-- trial_params_json adalah snapshot audit/fallback; sumber utama tetap Optuna PostgreSQL.
-- -------------------------------------------------------------
create or replace function public.create_task_with_slot(
    p_stage_no integer,
    p_model_type text,
    p_optuna_study_name text default null,
    p_trial_nr integer default null,
    p_trial_params_json jsonb default null,
    p_objective_metric_name text default null,
    p_objective_direction text default null,
    p_optuna_tell_status text default null,
    p_dispatcher_batch_no integer default null
)
returns table (
    task_id bigint,
    checkpoint_slot_id bigint
)
language plpgsql
security definer
set search_path = public
as $$
declare
    v_slot_id bigint;
    v_task_id bigint;
    v_stage public.stage_information%rowtype;
    v_tell_status text;
    v_objective_direction text;
begin
    select * into v_stage
    from public.stage_information
    where stage_no = p_stage_no
      and model_type = p_model_type;

    if not found then
        raise exception 'stage_information not found for stage_no %, model_type %', p_stage_no, p_model_type;
    end if;

    v_objective_direction := coalesce(p_objective_direction, v_stage.optuna_direction);

    if p_optuna_tell_status is not null then
        v_tell_status := p_optuna_tell_status;
    elsif v_stage.optuna_trials > 0 then
        v_tell_status := 'PENDING';
    else
        v_tell_status := 'NOT_REQUIRED';
    end if;

    insert into public.checkpoint_slot (resume_status)
    values ('EMPTY')
    returning public.checkpoint_slot.checkpoint_slot_id into v_slot_id;

    insert into public.task (
        stage_no,
        model_type,
        checkpoint_slot_id,
        optuna_study_name,
        trial_nr,
        trial_params_json,
        objective_metric_name,
        objective_direction,
        optuna_tell_status,
        dispatcher_batch_no,
        dispatcher_generated_at,
        status_task
    )
    values (
        p_stage_no,
        p_model_type,
        v_slot_id,
        p_optuna_study_name,
        p_trial_nr,
        p_trial_params_json,
        p_objective_metric_name,
        v_objective_direction,
        v_tell_status,
        p_dispatcher_batch_no,
        now(),
        'WAITING'
    )
    returning public.task.task_id into v_task_id;

    return query select v_task_id, v_slot_id;
end;
$$;

-- -------------------------------------------------------------
-- 3. Dispatcher status: berapa task aktif, berapa yang bisa digenerate.
-- -------------------------------------------------------------
create or replace function public.get_dispatcher_stage_status(
    p_stage_no integer,
    p_model_type text,
    p_max_parallel_tasks integer default 1
)
returns table (
    stage_no integer,
    model_type text,
    optuna_trials integer,
    total_tasks bigint,
    waiting_tasks bigint,
    running_tasks bigint,
    done_waiting_tell_tasks bigint,
    told_tasks bigint,
    done_tasks bigint,
    failed_tasks bigint,
    stale_tasks bigint,
    active_tasks bigint,
    remaining_trials_to_generate bigint,
    can_generate_count integer,
    dispatcher_signal text
)
language plpgsql
security definer
set search_path = public
as $$
begin
    return query
    with s as (
        select *
        from public.stage_information si
        where si.stage_no = p_stage_no
          and si.model_type = p_model_type
    ), c as (
        select
            s.stage_no,
            s.model_type,
            s.optuna_trials,
            count(t.task_id) as total_tasks,
            count(*) filter (where t.status_task = 'WAITING') as waiting_tasks,
            count(*) filter (where t.status_task = 'RUNNING') as running_tasks,
            count(*) filter (where t.status_task = 'DONE_WAITING_TELL') as done_waiting_tell_tasks,
            count(*) filter (where t.status_task = 'TOLD') as told_tasks,
            count(*) filter (where t.status_task = 'DONE') as done_tasks,
            count(*) filter (where t.status_task = 'FAILED') as failed_tasks,
            count(*) filter (where t.status_task = 'STALE') as stale_tasks
        from s
        left join public.task t
          on t.stage_no = s.stage_no
         and t.model_type = s.model_type
        group by s.stage_no, s.model_type, s.optuna_trials
    ), d as (
        select
            c.*,
            (c.waiting_tasks + c.running_tasks + c.done_waiting_tell_tasks) as active_tasks,
            greatest(c.optuna_trials - (c.told_tasks + c.waiting_tasks + c.running_tasks + c.done_waiting_tell_tasks + c.failed_tasks), 0)::bigint as remaining_trials_to_generate
        from c
    )
    select
        d.stage_no,
        d.model_type,
        d.optuna_trials,
        d.total_tasks,
        d.waiting_tasks,
        d.running_tasks,
        d.done_waiting_tell_tasks,
        d.told_tasks,
        d.done_tasks,
        d.failed_tasks,
        d.stale_tasks,
        d.active_tasks,
        d.remaining_trials_to_generate,
        greatest(least(p_max_parallel_tasks - d.active_tasks, d.remaining_trials_to_generate), 0)::integer as can_generate_count,
        case
            when d.optuna_trials = 0 then 'NO_HPO_STAGE'
            when d.told_tasks >= d.optuna_trials and d.active_tasks = 0 then 'STAGE_COMPLETE'
            when d.done_waiting_tell_tasks > 0 then 'HAS_RESULTS_TO_TELL'
            when greatest(least(p_max_parallel_tasks - d.active_tasks, d.remaining_trials_to_generate), 0) > 0 then 'GENERATE_TRIAL'
            else 'WAIT_FOR_WORKERS'
        end as dispatcher_signal
    from d;
end;
$$;

-- -------------------------------------------------------------
-- 4. Get task results ready for Optuna tell().
-- -------------------------------------------------------------
create or replace function public.get_tasks_ready_for_tell(
    p_stage_no integer,
    p_model_type text,
    p_limit integer default 50
)
returns table (
    task_id bigint,
    optuna_study_name text,
    trial_nr integer,
    objective_metric_name text,
    objective_value numeric,
    objective_direction text
)
language sql
security definer
set search_path = public
as $$
    select
        t.task_id,
        t.optuna_study_name,
        t.trial_nr,
        t.objective_metric_name,
        t.objective_value,
        t.objective_direction
    from public.task t
    where t.stage_no = p_stage_no
      and t.model_type = p_model_type
      and t.status_task = 'DONE_WAITING_TELL'
      and t.optuna_tell_status = 'PENDING'
      and t.objective_value is not null
    order by t.completed_by_worker_at nulls last, t.task_id
    limit p_limit;
$$;

-- -------------------------------------------------------------
-- 5. Worker signal: worker tahu harus claim, idle, tunggu dispatcher, atau stop.
-- -------------------------------------------------------------
create or replace function public.get_worker_stage_signal(
    p_stage_no integer,
    p_model_type text
)
returns table (
    signal text,
    reason text,
    optuna_trials integer,
    total_tasks bigint,
    waiting_tasks bigint,
    running_tasks bigint,
    done_waiting_tell_tasks bigint,
    told_tasks bigint,
    done_tasks bigint,
    failed_tasks bigint,
    stale_tasks bigint
)
language plpgsql
security definer
set search_path = public
as $$
begin
    return query
    with s as (
        select * from public.stage_information
        where stage_no = p_stage_no
          and model_type = p_model_type
    ), c as (
        select
            s.optuna_trials,
            count(t.task_id) as total_tasks,
            count(*) filter (where t.status_task = 'WAITING') as waiting_tasks,
            count(*) filter (where t.status_task = 'RUNNING') as running_tasks,
            count(*) filter (where t.status_task = 'DONE_WAITING_TELL') as done_waiting_tell_tasks,
            count(*) filter (where t.status_task = 'TOLD') as told_tasks,
            count(*) filter (where t.status_task = 'DONE') as done_tasks,
            count(*) filter (where t.status_task = 'FAILED') as failed_tasks,
            count(*) filter (where t.status_task = 'STALE') as stale_tasks
        from s
        left join public.task t
          on t.stage_no = p_stage_no
         and t.model_type = p_model_type
        group by s.optuna_trials
    )
    select
        case
            when c.waiting_tasks > 0 then 'HAS_WAITING_TASK'
            when c.stale_tasks > 0 then 'HAS_STALE_TASK'
            when c.running_tasks > 0 then 'WAIT_FOR_RUNNING_TASKS'
            when c.done_waiting_tell_tasks > 0 then 'WAIT_FOR_OPTUNA_TELL'
            when c.optuna_trials > 0 and c.told_tasks >= c.optuna_trials then 'STAGE_COMPLETE'
            when c.optuna_trials > 0 and c.told_tasks < c.optuna_trials then 'WAIT_FOR_DISPATCHER'
            when c.optuna_trials = 0 and c.total_tasks = 0 then 'WAIT_FOR_TASK_GENERATION'
            when c.optuna_trials = 0 and c.waiting_tasks = 0 and c.running_tasks = 0 and c.stale_tasks = 0 and c.failed_tasks = 0 then 'STAGE_COMPLETE'
            when c.failed_tasks > 0 then 'HAS_FAILED_TASK'
            else 'IDLE'
        end as signal,
        case
            when c.waiting_tasks > 0 then 'Ada task WAITING; worker sebaiknya claim task.'
            when c.stale_tasks > 0 then 'Ada task STALE; worker boleh hijack jika tidak ada WAITING task.'
            when c.running_tasks > 0 then 'Task lain masih RUNNING; worker idle/polling.'
            when c.done_waiting_tell_tasks > 0 then 'Ada hasil training menunggu Optuna dispatcher melakukan study.tell().'
            when c.optuna_trials > 0 and c.told_tasks >= c.optuna_trials then 'Seluruh trial HPO sudah TOLD; stage selesai.'
            when c.optuna_trials > 0 and c.told_tasks < c.optuna_trials then 'Belum ada task WAITING; tunggu dispatcher generate trial Bayesian berikutnya.'
            when c.optuna_trials = 0 and c.total_tasks = 0 then 'Stage non-HPO belum dibuatkan task.'
            when c.optuna_trials = 0 and c.waiting_tasks = 0 and c.running_tasks = 0 and c.stale_tasks = 0 and c.failed_tasks = 0 then 'Semua task non-HPO selesai.'
            when c.failed_tasks > 0 then 'Ada task gagal; perlu inspeksi manual.'
            else 'Tidak ada aksi.'
        end as reason,
        c.optuna_trials,
        c.total_tasks,
        c.waiting_tasks,
        c.running_tasks,
        c.done_waiting_tell_tasks,
        c.told_tasks,
        c.done_tasks,
        c.failed_tasks,
        c.stale_tasks
    from c;
end;
$$;

-- -------------------------------------------------------------
-- 6. Claim WAITING task.
-- -------------------------------------------------------------
create or replace function public.claim_waiting_task(
    p_worker_uid text,
    p_stage_no integer,
    p_model_type text
)
returns table (
    task_id bigint,
    stage_no integer,
    model_type text,
    checkpoint_slot_id bigint,
    worker_id bigint,
    optuna_study_name text,
    trial_nr integer,
    trial_params_json jsonb,
    objective_metric_name text,
    objective_direction text,
    optuna_tell_status text,
    status_task text,
    stage_name text,
    stage_objective text,
    split_strategy text,
    k_folds integer,
    n_repeats integer,
    max_epoch integer,
    optuna_trials integer,
    optuna_direction text,
    tuning_focus text,
    search_space_json jsonb,
    early_stop_monitor text,
    early_stop_mode text,
    early_stop_patience integer,
    early_stop_min_delta numeric,
    min_epoch_before_stop integer
)
language plpgsql
security definer
set search_path = public
as $$
declare
    v_worker_id bigint;
begin
    select wn.worker_id into v_worker_id
    from public.worker_node wn
    where wn.worker_uid = p_worker_uid;

    if v_worker_id is null then
        raise exception 'Worker UID % is not registered. Call register_worker() first.', p_worker_uid;
    end if;

    update public.worker_node
    set last_seen_at = now(), updated_at = now()
    where public.worker_node.worker_id = v_worker_id;

    return query
    with next_task as (
        select t.task_id
        from public.task t
        where t.status_task = 'WAITING'
          and t.stage_no = p_stage_no
          and t.model_type = p_model_type
        order by t.task_id
        limit 1
        for update skip locked
    ), claimed as (
        update public.task t
        set
            status_task = 'RUNNING',
            fk_worker_id = v_worker_id,
            started_at = coalesce(t.started_at, now()),
            last_heartbeat = now(),
            stale_marked_at = null,
            error_message = null,
            updated_at = now()
        from next_task nt
        where t.task_id = nt.task_id
        returning t.*
    )
    select
        c.task_id,
        c.stage_no,
        c.model_type,
        c.checkpoint_slot_id,
        c.fk_worker_id as worker_id,
        c.optuna_study_name,
        c.trial_nr,
        c.trial_params_json,
        c.objective_metric_name,
        c.objective_direction,
        c.optuna_tell_status,
        c.status_task,
        s.stage_name,
        s.stage_objective,
        s.split_strategy,
        s.k_folds,
        s.n_repeats,
        s.max_epoch,
        s.optuna_trials,
        s.optuna_direction,
        s.tuning_focus,
        s.search_space_json,
        s.early_stop_monitor,
        s.early_stop_mode,
        s.early_stop_patience,
        s.early_stop_min_delta,
        s.min_epoch_before_stop
    from claimed c
    join public.stage_information s
      on s.stage_no = c.stage_no
     and s.model_type = c.model_type;
end;
$$;

-- -------------------------------------------------------------
-- 7. Heartbeat
-- -------------------------------------------------------------
create or replace function public.update_task_heartbeat(
    p_task_id bigint,
    p_worker_uid text
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
    v_worker_id bigint;
    v_updated integer;
begin
    select worker_id into v_worker_id
    from public.worker_node
    where worker_uid = p_worker_uid;

    if v_worker_id is null then
        return false;
    end if;

    update public.task
    set last_heartbeat = now(), updated_at = now()
    where task_id = p_task_id
      and fk_worker_id = v_worker_id
      and status_task = 'RUNNING';

    get diagnostics v_updated = row_count;

    update public.worker_node
    set last_seen_at = now(), updated_at = now()
    where worker_id = v_worker_id;

    return v_updated = 1;
end;
$$;

-- -------------------------------------------------------------
-- 8. Mark RUNNING tasks as STALE.
-- -------------------------------------------------------------
create or replace function public.mark_stale_tasks(
    p_stale_after interval default interval '15 minutes'
)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
    v_count integer;
begin
    update public.task
    set
        status_task = 'STALE',
        stale_marked_at = now(),
        updated_at = now()
    where status_task = 'RUNNING'
      and last_heartbeat is not null
      and last_heartbeat < now() - p_stale_after;

    get diagnostics v_count = row_count;
    return v_count;
end;
$$;

-- -------------------------------------------------------------
-- 9. Hijack stale task if no WAITING task exists.
-- -------------------------------------------------------------
create or replace function public.hijack_stale_task(
    p_worker_uid text,
    p_stage_no integer,
    p_model_type text,
    p_stale_after interval default interval '15 minutes'
)
returns table (
    task_id bigint,
    stage_no integer,
    model_type text,
    checkpoint_slot_id bigint,
    worker_id bigint,
    optuna_study_name text,
    trial_nr integer,
    trial_params_json jsonb,
    objective_metric_name text,
    objective_direction text,
    status_task text,
    resume_checkpoint_file_id bigint,
    resume_storage_backend text,
    resume_rclone_remote text,
    resume_gdrive_relative_path text,
    resume_checkpoint_uri text,
    resume_local_cache_path text,
    current_repeat_id integer,
    current_fold_id integer,
    current_epoch integer,
    global_step bigint
)
language plpgsql
security definer
set search_path = public
as $$
declare
    v_worker_id bigint;
    v_waiting_count integer;
begin
    perform public.mark_stale_tasks(p_stale_after);

    select worker_id into v_worker_id
    from public.worker_node
    where worker_uid = p_worker_uid;

    if v_worker_id is null then
        raise exception 'Worker UID % is not registered. Call register_worker() first.', p_worker_uid;
    end if;

    select count(*) into v_waiting_count
    from public.task t
    where t.status_task = 'WAITING'
      and t.stage_no = p_stage_no
      and t.model_type = p_model_type;

    if v_waiting_count > 0 then
        return;
    end if;

    return query
    with stale_task as (
        select t.task_id
        from public.task t
        where t.status_task = 'STALE'
          and t.stage_no = p_stage_no
          and t.model_type = p_model_type
        order by t.stale_marked_at nulls last, t.task_id
        limit 1
        for update skip locked
    ), claimed as (
        update public.task t
        set
            status_task = 'RUNNING',
            fk_worker_id = v_worker_id,
            last_heartbeat = now(),
            hijack_count = t.hijack_count + 1,
            updated_at = now()
        from stale_task st
        where t.task_id = st.task_id
        returning t.*
    )
    select
        c.task_id,
        c.stage_no,
        c.model_type,
        c.checkpoint_slot_id,
        c.fk_worker_id as worker_id,
        c.optuna_study_name,
        c.trial_nr,
        c.trial_params_json,
        c.objective_metric_name,
        c.objective_direction,
        c.status_task,
        cs.latest_checkpoint_file_id as resume_checkpoint_file_id,
        cf.storage_backend as resume_storage_backend,
        cf.rclone_remote as resume_rclone_remote,
        cf.gdrive_relative_path as resume_gdrive_relative_path,
        cf.checkpoint_uri as resume_checkpoint_uri,
        cf.local_cache_path as resume_local_cache_path,
        cs.current_repeat_id,
        cs.current_fold_id,
        cs.current_epoch,
        cs.global_step
    from claimed c
    join public.checkpoint_slot cs
      on cs.checkpoint_slot_id = c.checkpoint_slot_id
    left join public.checkpoint_file cf
      on cf.checkpoint_file_id = cs.latest_checkpoint_file_id;
end;
$$;

-- -------------------------------------------------------------
-- 10. Register checkpoint metadata after rclone upload to Google Drive.
-- -------------------------------------------------------------
create or replace function public.register_checkpoint_file(
    p_task_id bigint,
    p_worker_uid text,
    p_checkpoint_type text,
    p_gdrive_relative_path text,
    p_file_name text default null,
    p_sha256 text default null,
    p_file_size_bytes bigint default null,
    p_epoch_number integer default null,
    p_global_step bigint default null,
    p_repeat_id integer default 0,
    p_fold_id integer default 0,
    p_metric_name text default null,
    p_metric_value numeric default null,
    p_upload_status text default 'UPLOADED',
    p_local_cache_path text default null,
    p_rclone_remote text default 'gdrive',
    p_storage_backend text default 'gdrive_rclone',
    p_has_model_state boolean default true,
    p_has_optimizer_state boolean default true,
    p_has_scheduler_state boolean default true,
    p_has_model_config boolean default true,
    p_has_runtime_plan boolean default true,
    p_has_seed boolean default true,
    p_optimizer_name text default 'AdamW'
)
returns bigint
language plpgsql
security definer
set search_path = public
as $$
declare
    v_worker_id bigint;
    v_task public.task%rowtype;
    v_checkpoint_file_id bigint;
begin
    select worker_id into v_worker_id
    from public.worker_node
    where worker_uid = p_worker_uid;

    if v_worker_id is null then
        raise exception 'Worker UID % is not registered.', p_worker_uid;
    end if;

    select * into v_task
    from public.task
    where task_id = p_task_id;

    if not found then
        raise exception 'Task % not found.', p_task_id;
    end if;

    if nullif(trim(p_gdrive_relative_path), '') is null then
        raise exception 'p_gdrive_relative_path must not be empty.';
    end if;

    insert into public.checkpoint_file (
        checkpoint_slot_id,
        task_id,
        worker_id,
        stage_no,
        model_type,
        trial_nr,
        repeat_id,
        fold_id,
        epoch_number,
        global_step,
        checkpoint_type,
        storage_backend,
        rclone_remote,
        gdrive_relative_path,
        local_cache_path,
        file_name,
        sha256,
        file_size_bytes,
        has_model_state,
        has_optimizer_state,
        has_scheduler_state,
        has_model_config,
        has_runtime_plan,
        has_seed,
        optimizer_name,
        upload_status,
        metric_name,
        metric_value
    )
    values (
        v_task.checkpoint_slot_id,
        v_task.task_id,
        v_worker_id,
        v_task.stage_no,
        v_task.model_type,
        v_task.trial_nr,
        p_repeat_id,
        p_fold_id,
        p_epoch_number,
        p_global_step,
        p_checkpoint_type,
        p_storage_backend,
        p_rclone_remote,
        p_gdrive_relative_path,
        p_local_cache_path,
        p_file_name,
        p_sha256,
        p_file_size_bytes,
        p_has_model_state,
        p_has_optimizer_state,
        p_has_scheduler_state,
        p_has_model_config,
        p_has_runtime_plan,
        p_has_seed,
        p_optimizer_name,
        p_upload_status,
        p_metric_name,
        p_metric_value
    )
    returning checkpoint_file_id into v_checkpoint_file_id;

    if p_checkpoint_type in ('INTERVAL', 'EPOCH', 'RECOVERY') then
        update public.checkpoint_slot
        set
            latest_checkpoint_file_id = v_checkpoint_file_id,
            current_repeat_id = p_repeat_id,
            current_fold_id = p_fold_id,
            current_epoch = coalesce(p_epoch_number, current_epoch),
            global_step = coalesce(p_global_step, global_step),
            resume_status = 'READY',
            updated_at = now()
        where checkpoint_slot_id = v_task.checkpoint_slot_id;
    elsif p_checkpoint_type = 'BEST' then
        update public.checkpoint_slot
        set
            best_checkpoint_file_id = v_checkpoint_file_id,
            best_metric_name = p_metric_name,
            best_metric_value = p_metric_value,
            updated_at = now()
        where checkpoint_slot_id = v_task.checkpoint_slot_id;
    elsif p_checkpoint_type = 'FINAL' then
        update public.checkpoint_slot
        set
            final_checkpoint_file_id = v_checkpoint_file_id,
            latest_checkpoint_file_id = coalesce(latest_checkpoint_file_id, v_checkpoint_file_id),
            current_repeat_id = p_repeat_id,
            current_fold_id = p_fold_id,
            current_epoch = coalesce(p_epoch_number, current_epoch),
            global_step = coalesce(p_global_step, global_step),
            resume_status = 'FINALIZED',
            updated_at = now()
        where checkpoint_slot_id = v_task.checkpoint_slot_id;
    end if;

    return v_checkpoint_file_id;
end;
$$;

-- -------------------------------------------------------------
-- 11. Get resume checkpoint metadata.
-- -------------------------------------------------------------
create or replace function public.get_resume_checkpoint(
    p_checkpoint_slot_id bigint,
    p_prefer text default 'LATEST'
)
returns table (
    checkpoint_file_id bigint,
    checkpoint_type text,
    storage_backend text,
    rclone_remote text,
    gdrive_relative_path text,
    checkpoint_uri text,
    local_cache_path text,
    sha256 text,
    file_size_bytes bigint,
    epoch_number integer,
    global_step bigint,
    repeat_id integer,
    fold_id integer
)
language plpgsql
security definer
set search_path = public
as $$
declare
    v_file_id bigint;
begin
    if upper(p_prefer) = 'BEST' then
        select best_checkpoint_file_id into v_file_id
        from public.checkpoint_slot
        where checkpoint_slot_id = p_checkpoint_slot_id;
    elsif upper(p_prefer) = 'FINAL' then
        select final_checkpoint_file_id into v_file_id
        from public.checkpoint_slot
        where checkpoint_slot_id = p_checkpoint_slot_id;
    else
        select latest_checkpoint_file_id into v_file_id
        from public.checkpoint_slot
        where checkpoint_slot_id = p_checkpoint_slot_id;
    end if;

    if v_file_id is null then
        return;
    end if;

    return query
    select
        cf.checkpoint_file_id,
        cf.checkpoint_type,
        cf.storage_backend,
        cf.rclone_remote,
        cf.gdrive_relative_path,
        cf.checkpoint_uri,
        cf.local_cache_path,
        cf.sha256,
        cf.file_size_bytes,
        cf.epoch_number,
        cf.global_step,
        cf.repeat_id,
        cf.fold_id
    from public.checkpoint_file cf
    where cf.checkpoint_file_id = v_file_id;
end;
$$;

-- -------------------------------------------------------------
-- 12a. Mark HPO task done but waiting for Optuna tell().
-- -------------------------------------------------------------
create or replace function public.mark_task_done_waiting_tell(
    p_task_id bigint,
    p_worker_uid text,
    p_objective_metric_name text,
    p_objective_value numeric
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
    v_worker_id bigint;
    v_updated integer;
begin
    select worker_id into v_worker_id
    from public.worker_node
    where worker_uid = p_worker_uid;

    if v_worker_id is null then
        return false;
    end if;

    update public.task
    set
        status_task = 'DONE_WAITING_TELL',
        objective_metric_name = p_objective_metric_name,
        objective_value = p_objective_value,
        optuna_tell_status = 'PENDING',
        completed_by_worker_at = now(),
        finished_at = now(),
        updated_at = now()
    where task_id = p_task_id
      and fk_worker_id = v_worker_id
      and status_task = 'RUNNING';

    get diagnostics v_updated = row_count;
    return v_updated = 1;
end;
$$;

-- -------------------------------------------------------------
-- 12b. Mark non-HPO task done.
-- -------------------------------------------------------------
create or replace function public.mark_task_done(
    p_task_id bigint,
    p_worker_uid text,
    p_objective_metric_name text default null,
    p_objective_value numeric default null
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
    v_worker_id bigint;
    v_updated integer;
begin
    select worker_id into v_worker_id
    from public.worker_node
    where worker_uid = p_worker_uid;

    if v_worker_id is null then
        return false;
    end if;

    update public.task
    set
        status_task = 'DONE',
        objective_metric_name = coalesce(p_objective_metric_name, objective_metric_name),
        objective_value = coalesce(p_objective_value, objective_value),
        optuna_tell_status = 'NOT_REQUIRED',
        completed_by_worker_at = now(),
        finished_at = now(),
        updated_at = now()
    where task_id = p_task_id
      and fk_worker_id = v_worker_id
      and status_task = 'RUNNING';

    get diagnostics v_updated = row_count;
    return v_updated = 1;
end;
$$;

-- -------------------------------------------------------------
-- 13. Mark task as TOLD after dispatcher calls study.tell().
-- -------------------------------------------------------------
create or replace function public.mark_task_told(
    p_task_id bigint
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
    v_updated integer;
begin
    update public.task
    set
        status_task = 'TOLD',
        optuna_tell_status = 'TOLD',
        optuna_told_at = now(),
        updated_at = now()
    where task_id = p_task_id
      and status_task = 'DONE_WAITING_TELL'
      and optuna_tell_status = 'PENDING';

    get diagnostics v_updated = row_count;
    return v_updated = 1;
end;
$$;

-- -------------------------------------------------------------
-- 14. Mark task failed.
-- -------------------------------------------------------------
create or replace function public.mark_task_failed(
    p_task_id bigint,
    p_worker_uid text,
    p_error_message text
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
    v_worker_id bigint;
    v_updated integer;
begin
    select worker_id into v_worker_id
    from public.worker_node
    where worker_uid = p_worker_uid;

    if v_worker_id is null then
        return false;
    end if;

    update public.task
    set
        status_task = 'FAILED',
        error_message = left(coalesce(p_error_message, 'Unknown error'), 5000),
        optuna_tell_status = case when optuna_tell_status = 'PENDING' then 'FAILED' else optuna_tell_status end,
        finished_at = now(),
        updated_at = now()
    where task_id = p_task_id
      and fk_worker_id = v_worker_id
      and status_task in ('RUNNING', 'STALE', 'HIJACKED');

    get diagnostics v_updated = row_count;
    return v_updated = 1;
end;
$$;

commit;

-- -------------------------------------------------------------
-- 13b. Mark HPO task as TOLD directly by worker after worker calls study.tell().
-- Recommended when Optuna uses local/shared PostgreSQL storage accessible by workers.
-- -------------------------------------------------------------
create or replace function public.mark_task_told_by_worker(
    p_task_id bigint,
    p_worker_uid text,
    p_objective_metric_name text,
    p_objective_value numeric
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
    v_worker_id bigint;
    v_updated integer;
begin
    select worker_id into v_worker_id
    from public.worker_node
    where worker_uid = p_worker_uid;

    if v_worker_id is null then
        return false;
    end if;

    update public.task
    set
        status_task = 'TOLD',
        objective_metric_name = p_objective_metric_name,
        objective_value = p_objective_value,
        optuna_tell_status = 'TOLD',
        completed_by_worker_at = now(),
        optuna_told_at = now(),
        finished_at = now(),
        updated_at = now()
    where task_id = p_task_id
      and fk_worker_id = v_worker_id
      and status_task = 'RUNNING'
      and optuna_tell_status = 'PENDING';

    get diagnostics v_updated = row_count;
    return v_updated = 1;
end;
$$;
