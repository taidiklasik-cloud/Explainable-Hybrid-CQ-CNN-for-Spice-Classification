-- =============================================================
-- 08_monitoring_queries.sql
-- Kumpulan query untuk memonitor worker heartbeat dan checkpoint.
-- File ini tidak membuat struktur baru, hanya untuk dieksekusi 
-- (run) secara berulang kapan pun Anda ingin memonitor sistem.
-- =============================================================

-- 1. Status realtime worker (termasuk status IDLE/RUNNING/OFFLINE, spesifikasi GPU, task aktif)
select * from public.v_worker_health;

-- 2. Checkpoint yang sudah berhasil ter-upload ke Google Drive via rclone
select * from public.v_checkpoint_health 
where upload_status = 'UPLOADED' 
order by created_at desc;

-- 3. Daftar worker yang terdeteksi tidak merespon/hilang (heartbeat > 2 menit)
select * from public.v_stale_workers;

-- 4. Catatan log historis kapan saja worker menjadi STALE atau OFFLINE yang direkam oleh pgAgent
select * from public.worker_monitoring_event 
order by detected_at desc;
