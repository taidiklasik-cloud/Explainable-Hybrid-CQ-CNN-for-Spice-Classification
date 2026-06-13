-- =============================================================
-- 04_worker_sql_repeatable_calls.sql
-- Contoh SQL yang dieksekusi berkali-kali dari notebook orchestrator/worker.
-- File ini bukan untuk dijalankan sekaligus tanpa edit.
-- =============================================================

-- A. ORCHESTRATOR: cek status stage dan apakah perlu generate task baru.
select *
from public.get_dispatcher_stage_status(
    p_stage_no := 4,
    p_model_type := 'hybrid_qcqcnn',
    p_max_parallel_tasks := 2
);

-- B. ORCHESTRATOR: create one task + checkpoint slot dengan snapshot parameter Optuna.
-- trial_params_json berasal dari Optuna PostgreSQL setelah study.ask() + trial.suggest_*.
-- Snapshot ini untuk audit/fallback. Sumber utama parameter tetap Optuna PostgreSQL.
select *
from public.create_task_with_slot(
    p_stage_no := 4,
    p_model_type := 'hybrid_qcqcnn',
    p_optuna_study_name := 'cqcnn_stage4_hybrid_accuracy',
    p_trial_nr := 7,
    p_trial_params_json := '{
        "lr_backbone": 0.0005,
        "lr_head": 0.001,
        "lr_quantum": 0.0005,
        "weight_decay": 0.0001,
        "dropout": 0.20,
        "label_smoothing": 0.10,
        "grad_clip_norm": 1.0,
        "quantum_measurement": "pauli_z_linear"
    }'::jsonb,
    p_objective_metric_name := 'val_macro_f1',
    p_objective_direction := 'maximize',
    p_optuna_tell_status := 'PENDING',
    p_dispatcher_batch_no := 1
);

-- C. WORKER START: register worker.
select public.register_worker(
    p_worker_uid := 'pc_local_01',
    p_worker_name := 'PC Local 01',
    p_hostname := 'localhost',
    p_worker_type := 'LOCAL_PC',
    p_cpu_name := 'auto-detected-cpu',
    p_cpu_count := 8,
    p_ram_gb := 32.0,
    p_has_gpu := true,
    p_gpu_name := 'auto-detected-gpu',
    p_gpu_count := 1,
    p_gpu_vram_gb := 12.0,
    p_python_version := '3.11',
    p_platform_name := 'Windows/Linux'
);

-- D. WORKER LOOP: cek apakah harus claim, idle, tunggu dispatcher, atau stage selesai.
select *
from public.get_worker_stage_signal(
    p_stage_no := 4,
    p_model_type := 'hybrid_qcqcnn'
);

-- E. WORKER LOOP: claim satu WAITING task.
-- Hasil kosong berarti tidak ada task WAITING; worker harus membaca get_worker_stage_signal().
select *
from public.claim_waiting_task(
    p_worker_uid := 'pc_local_01',
    p_stage_no := 4,
    p_model_type := 'hybrid_qcqcnn'
);

-- F. WORKER LOOP: heartbeat berkala.
select public.update_task_heartbeat(
    p_task_id := 12,
    p_worker_uid := 'pc_local_01'
);

-- G. WORKER CHECKPOINT: setelah upload .pt ke Google Drive via rclone.
select public.register_checkpoint_file(
    p_task_id := 12,
    p_worker_uid := 'pc_local_01',
    p_checkpoint_type := 'INTERVAL',
    p_gdrive_relative_path := 'cqcnn_checkpoints/stage_04/hybrid_qcqcnn/task_000012/trial_000007/interval_epoch_005.pt',
    p_file_name := 'interval_epoch_005.pt',
    p_sha256 := 'replace_with_sha256',
    p_file_size_bytes := 12345678,
    p_epoch_number := 5,
    p_global_step := 250,
    p_repeat_id := 0,
    p_fold_id := 0,
    p_metric_name := 'val_macro_f1',
    p_metric_value := 0.873,
    p_upload_status := 'UPLOADED',
    p_local_cache_path := 'outputs/checkpoint_cache/task_000012/latest.pt',
    p_rclone_remote := 'gdrive'
);

-- H. WORKER FINISH HPO TASK: setelah Python worker menjalankan study.tell() ke Optuna PostgreSQL.
select public.mark_task_told_by_worker(
    p_task_id := 12,
    p_worker_uid := 'pc_local_01',
    p_objective_metric_name := 'val_macro_f1',
    p_objective_value := 0.873
);

-- I. FALLBACK: kalau tell dilakukan oleh orchestrator, worker tandai dulu DONE_WAITING_TELL.
select public.mark_task_done_waiting_tell(
    p_task_id := 12,
    p_worker_uid := 'pc_local_01',
    p_objective_metric_name := 'val_macro_f1',
    p_objective_value := 0.873
);

-- J. ORCHESTRATOR FALLBACK: setelah Python orchestrator menjalankan study.tell(), tandai task sebagai TOLD.
select public.mark_task_told(
    p_task_id := 12
);

-- K. WORKER FINISH NON-HPO TASK: contoh Stage 5 final evaluation tanpa Optuna.
select public.mark_task_done(
    p_task_id := 99,
    p_worker_uid := 'pc_local_01',
    p_objective_metric_name := 'mean_macro_f1_repeated_5fold',
    p_objective_value := 0.861
);

-- L. WORKER ERROR.
select public.mark_task_failed(
    p_task_id := 12,
    p_worker_uid := 'pc_local_01',
    p_error_message := 'replace with traceback summary'
);

-- M. STALE CHECKER: manual dari notebook/watcher.
select public.mark_stale_tasks(interval '15 minutes');

-- N. TASK HIJACKING: ambil stale task jika tidak ada waiting task.
select *
from public.hijack_stale_task(
    p_worker_uid := 'pc_local_02',
    p_stage_no := 4,
    p_model_type := 'hybrid_qcqcnn',
    p_stale_after := interval '15 minutes'
);

-- O. RESUME CHECKPOINT.
select *
from public.get_resume_checkpoint(
    p_checkpoint_slot_id := 1,
    p_prefer := 'LATEST'
);

-- P. MONITORING.
select * from public.v_stage_progress order by stage_no, model_type;
select * from public.v_task_monitor order by task_id;
select * from public.v_stale_candidates order by task_id;
select * from public.v_worker_monitor order by worker_id;
select * from public.v_checkpoint_monitor order by checkpoint_file_id desc;
