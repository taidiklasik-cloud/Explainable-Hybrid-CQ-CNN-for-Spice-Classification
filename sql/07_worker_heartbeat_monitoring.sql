-- =============================================================
-- 07_worker_heartbeat_monitoring.sql
-- Worker heartbeat detail, monitoring views, dan pgAgent job script.
-- Target database: cqcnn_orchestration.
-- Jalankan setelah 01 dan 02. Aman untuk rerun.
-- Tidak pakai pg_cron. Scheduler via pgAgent.
-- =============================================================

begin;

-- -------------------------------------------------------------
-- 1. WORKER_HEARTBEAT: status runtime detail worker
-- Tabel ini diupdate oleh worker setiap epoch, checkpoint, dan perubahan status.
-- worker_node tetap untuk metadata hardware (tidak duplikat).
-- -------------------------------------------------------------
create table if not exists public.worker_heartbeat (
    heartbeat_id        bigint generated always as identity primary key,
    worker_uid          text not null unique,    -- FK ke worker_node.worker_uid
    worker_name         text,
    hostname            text,
    pid                 integer,

    status              text not null default 'IDLE',
    current_task_id     bigint,
    stage_no            integer,
    model_type          text,
    current_epoch       integer,

    last_checkpoint_epoch       integer,
    last_checkpoint_local_path  text,
    last_checkpoint_remote_path text,

    last_seen_at        timestamptz not null default now(),
    error_message       text,
    metadata            jsonb,

    created_at          timestamptz not null default now(),
    updated_at          timestamptz not null default now()
);

alter table public.worker_heartbeat drop constraint if exists chk_heartbeat_status;
alter table public.worker_heartbeat add constraint chk_heartbeat_status check (
    status in ('IDLE', 'RUNNING', 'UPLOADING', 'FAILED', 'OFFLINE')
);

drop trigger if exists trg_worker_heartbeat_updated_at on public.worker_heartbeat;
create trigger trg_worker_heartbeat_updated_at
before update on public.worker_heartbeat
for each row execute function public.set_updated_at();

-- -------------------------------------------------------------
-- 2. WORKER_MONITORING_EVENT: log stale/offline events
-- Diisi oleh pgAgent job, bukan oleh worker.
-- -------------------------------------------------------------
create table if not exists public.worker_monitoring_event (
    event_id            bigint generated always as identity primary key,
    worker_uid          text not null,
    event_type          text not null,   -- 'STALE' | 'OFFLINE' | 'RECOVERED'
    last_seen_at        timestamptz,
    detected_at         timestamptz not null default now(),
    detail              text
);

alter table public.worker_monitoring_event drop constraint if exists chk_event_type;
alter table public.worker_monitoring_event add constraint chk_event_type check (
    event_type in ('STALE', 'OFFLINE', 'RECOVERED')
);

create index if not exists idx_worker_mon_event_uid on public.worker_monitoring_event(worker_uid);
create index if not exists idx_worker_mon_event_type on public.worker_monitoring_event(event_type, detected_at);

-- -------------------------------------------------------------
-- 3. SQL FUNCTION: upsert_worker_heartbeat
-- Dipanggil dari Python: update_worker_heartbeat(...)
-- -------------------------------------------------------------
create or replace function public.upsert_worker_heartbeat(
    p_worker_uid                text,
    p_worker_name               text        default null,
    p_hostname                  text        default null,
    p_pid                       integer     default null,
    p_status                    text        default 'IDLE',
    p_current_task_id           bigint      default null,
    p_stage_no                  integer     default null,
    p_model_type                text        default null,
    p_current_epoch             integer     default null,
    p_last_checkpoint_epoch     integer     default null,
    p_last_checkpoint_local     text        default null,
    p_last_checkpoint_remote    text        default null,
    p_error_message             text        default null,
    p_metadata                  jsonb       default null
)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.worker_heartbeat (
        worker_uid, worker_name, hostname, pid, status,
        current_task_id, stage_no, model_type, current_epoch,
        last_checkpoint_epoch, last_checkpoint_local_path, last_checkpoint_remote_path,
        last_seen_at, error_message, metadata
    )
    values (
        p_worker_uid, p_worker_name, p_hostname, p_pid, p_status,
        p_current_task_id, p_stage_no, p_model_type, p_current_epoch,
        p_last_checkpoint_epoch, p_last_checkpoint_local, p_last_checkpoint_remote,
        now(), p_error_message, p_metadata
    )
    on conflict (worker_uid) do update set
        worker_name                 = coalesce(p_worker_name, worker_heartbeat.worker_name),
        hostname                    = coalesce(p_hostname, worker_heartbeat.hostname),
        pid                         = coalesce(p_pid, worker_heartbeat.pid),
        status                      = p_status,
        current_task_id             = p_current_task_id,
        stage_no                    = p_stage_no,
        model_type                  = p_model_type,
        current_epoch               = p_current_epoch,
        last_checkpoint_epoch       = coalesce(p_last_checkpoint_epoch, worker_heartbeat.last_checkpoint_epoch),
        last_checkpoint_local_path  = coalesce(p_last_checkpoint_local, worker_heartbeat.last_checkpoint_local_path),
        last_checkpoint_remote_path = coalesce(p_last_checkpoint_remote, worker_heartbeat.last_checkpoint_remote_path),
        last_seen_at                = now(),
        error_message               = p_error_message,
        metadata                    = coalesce(p_metadata, worker_heartbeat.metadata),
        updated_at                  = now();

    -- Also keep worker_node.last_seen_at fresh
    update public.worker_node
    set last_seen_at = now(), updated_at = now()
    where worker_uid = p_worker_uid;
end;
$$;

-- -------------------------------------------------------------
-- 4. SQL FUNCTION: monitor_stale_workers
-- Dipanggil oleh pgAgent job setiap 1 menit.
-- -------------------------------------------------------------
create or replace function public.monitor_stale_workers()
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
    v_count integer := 0;
    r record;
begin
    -- Mark STALE: last_seen_at between 2 and 5 minutes ago
    for r in
        select worker_uid, last_seen_at, status
        from public.worker_heartbeat
        where last_seen_at < now() - interval '2 minutes'
          and last_seen_at >= now() - interval '5 minutes'
          and status not in ('OFFLINE', 'IDLE')
    loop
        insert into public.worker_monitoring_event (worker_uid, event_type, last_seen_at, detail)
        values (r.worker_uid, 'STALE', r.last_seen_at,
            format('Worker %s was %s, stale for %s', r.worker_uid, r.status,
                   now() - r.last_seen_at));

        update public.worker_heartbeat
        set status = 'OFFLINE', updated_at = now()
        where worker_uid = r.worker_uid;

        v_count := v_count + 1;
    end loop;

    -- Mark OFFLINE: last_seen_at > 5 minutes ago and not already OFFLINE
    for r in
        select worker_uid, last_seen_at, status
        from public.worker_heartbeat
        where last_seen_at < now() - interval '5 minutes'
          and status != 'OFFLINE'
    loop
        insert into public.worker_monitoring_event (worker_uid, event_type, last_seen_at, detail)
        values (r.worker_uid, 'OFFLINE', r.last_seen_at,
            format('Worker %s offline since %s', r.worker_uid, r.last_seen_at));

        update public.worker_heartbeat
        set status = 'OFFLINE', updated_at = now()
        where worker_uid = r.worker_uid;

        v_count := v_count + 1;
    end loop;

    return v_count;
end;
$$;

-- -------------------------------------------------------------
-- 5. MONITORING VIEWS
-- -------------------------------------------------------------

-- v_worker_health: status realtime tiap worker
create or replace view public.v_worker_health as
select
    wh.worker_uid,
    wh.worker_name,
    wh.hostname,
    wh.pid,
    wh.status,
    wh.current_task_id,
    wh.stage_no,
    wh.model_type,
    wh.current_epoch,
    wh.last_checkpoint_epoch,
    wh.last_checkpoint_local_path,
    wh.last_checkpoint_remote_path,
    wh.last_seen_at,
    now() - wh.last_seen_at as last_seen_age,
    wh.error_message,
    wn.has_gpu,
    wn.gpu_name,
    wn.gpu_vram_gb
from public.worker_heartbeat wh
left join public.worker_node wn using (worker_uid);

-- v_checkpoint_health: checkpoint terbaru per task
create or replace view public.v_checkpoint_health as
select
    cf.task_id,
    cf.stage_no,
    cf.model_type,
    cf.trial_nr,
    cf.checkpoint_type,
    cf.epoch_number,
    cf.global_step,
    cf.gdrive_relative_path,
    cf.checkpoint_uri,
    cf.local_cache_path,
    cf.upload_status,
    cf.sha256,
    cf.file_size_bytes,
    cf.metric_name,
    cf.metric_value,
    cf.created_at,
    wn.worker_uid,
    wn.worker_name
from public.checkpoint_file cf
left join public.worker_node wn on wn.worker_id = cf.worker_id
order by cf.created_at desc;

-- v_stale_workers: worker yang tidak heartbeat > 2 menit
create or replace view public.v_stale_workers as
select
    wh.worker_uid,
    wh.worker_name,
    wh.status,
    wh.current_task_id,
    wh.last_seen_at,
    now() - wh.last_seen_at as stale_duration,
    case
        when now() - wh.last_seen_at > interval '5 minutes' then 'OFFLINE'
        when now() - wh.last_seen_at > interval '2 minutes' then 'STALE'
        else 'OK'
    end as health_status
from public.worker_heartbeat wh
where now() - wh.last_seen_at > interval '2 minutes';

commit;

-- =============================================================
-- pgAgent Job Script (jalankan manual di pgAdmin Query Tool):
-- Buat satu pgAgent job bernama "cqcnn_worker_monitor_every_1min"
-- dengan schedule setiap 1 menit, step SQL berikut:
-- =============================================================
-- Step SQL untuk pgAgent job:
--   select public.monitor_stale_workers();
--
-- Cara buat job di pgAdmin:
--   1. pgAgent Jobs -> New Job
--   2. Name: cqcnn_worker_monitor_every_1min
--   3. Steps tab -> Add step:
--      - Name: run_monitor_stale_workers
--      - Kind: SQL
--      - Code: select public.monitor_stale_workers();
--      - Database: cqcnn_orchestration
--   4. Schedules tab -> Add schedule:
--      - Name: every_1_minute
--      - Start: now
--      - Minutes: setiap menit (checklist semua menit 0-59)
--   5. Save
-- =============================================================
