-- =============================================================
-- 05_optional_cron_stale_monitor.sql
-- Opsional: pg_cron lokal untuk menandai task RUNNING menjadi STALE.
-- Jalankan hanya kalau PostgreSQL lokal Anda sudah mengaktifkan pg_cron.
-- Jika workflow Anda manual via notebook, file ini TIDAK WAJIB.
-- =============================================================

-- Aktifkan extension jika tersedia/diizinkan.
create extension if not exists pg_cron;

-- Jadwalkan stale checker setiap 5 menit.
-- Nama job: cqcnn_mark_stale_tasks_every_5min
select cron.schedule(
    'cqcnn_mark_stale_tasks_every_5min',
    '*/5 * * * *',
    $$select public.mark_stale_tasks(interval '15 minutes');$$
);

-- Cek daftar cron job:
select * from cron.job;

-- Jika ingin membatalkan:
-- select cron.unschedule('cqcnn_mark_stale_tasks_every_5min');
