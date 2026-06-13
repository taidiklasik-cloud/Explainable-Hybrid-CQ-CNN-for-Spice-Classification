-- =============================================================
-- 03_seed_stage_information_stage3_4_5_revised.sql
-- Seed awal untuk 5 stage x 2 model.
-- Revisi fokus:
--   Stage 3 = convergence-oriented hyperparameter tuning
--   Stage 4 = maximum-accuracy-oriented hyperparameter tuning
--   Stage 5 = repeated k-fold final evaluation only, without HPO
-- Jalankan setelah 01_schema_tables_and_views.sql dan 02_functions_run_once.sql.
-- Jika file 03 lama belum pernah dijalankan, jalankan file ini.
-- Jika file 03 lama sudah pernah dijalankan, file ini tetap aman karena memakai ON CONFLICT DO UPDATE.
-- =============================================================

begin;

insert into public.stage_information (
    stage_no,
    model_type,
    stage_name,
    stage_objective,
    is_active,
    train_ratio,
    validation_ratio,
    split_strategy,
    k_folds,
    n_repeats,
    max_epoch,
    optuna_study_name,
    optuna_trials,
    optuna_direction,
    tuning_focus,
    search_space_json,
    early_stop_monitor,
    early_stop_mode,
    early_stop_patience,
    early_stop_min_delta,
    min_epoch_before_stop
)
values
-- =============================================================
-- CLASSICAL FULLY SPATIAL CNN
-- =============================================================
(1, 'classical_fully_spatial', 'Stage 1 - Sanity Test', 'SANITY', true,
 0.80, 0.20, 'SIMPLE_80_20', null, 1, 2,
 'cqcnn_stage1_classical_sanity', 1, 'minimize', 'NONE',
 '{
    "lr_backbone": {"type": "float", "low": 0.0005, "high": 0.0005},
    "lr_head": {"type": "float", "low": 0.001, "high": 0.001},
    "dropout": {"type": "float", "low": 0.20, "high": 0.20}
  }'::jsonb,
 null, null, null, null, null),

 (2, 'classical_fully_spatial', 'Stage 2 - Warm Start', 'WARM_START', true,
 0.80, 0.20, 'SIMPLE_80_20', null, 1, 5,
 'cqcnn_stage2_classical_warmstart', 1, 'minimize', 'NONE',
 '{
    "lr_backbone": {"type": "float", "low": 0.0005, "high": 0.0005},
    "lr_head": {"type": "float", "low": 0.001, "high": 0.001},
    "dropout": {"type": "float", "low": 0.20, "high": 0.20}
  }'::jsonb,
 'val_loss', 'min', 5, 0.0001, 5),

-- Stage 3: mencari konfigurasi yang paling stabil/konvergen.
-- Objective Optuna = minimize val_loss.
-- Fold = 5-fold biasa, bukan repeated, agar biaya tuning masih masuk akal.
(3, 'classical_fully_spatial', 'Stage 3 - Convergence Tuning', 'CONVERGENCE', true,
 0.80, 0.20, 'CV_5_FOLD', 5, 1, 25,
 'cqcnn_stage3_classical_convergence', 20, 'minimize', 'CONVERGENCE',
 '{
    "lr_backbone": {"type": "float", "low": 0.0001, "high": 0.001, "log": true},
    "lr_head": {"type": "float", "low": 0.0003, "high": 0.003, "log": true},
    "weight_decay": {"type": "float", "low": 0.000001, "high": 0.001, "log": true},
    "dropout": {"type": "float", "low": 0.10, "high": 0.35},
    "activation_fn": {"type": "categorical", "choices": ["leaky_relu"]},
    "leaky_relu_negative_slope": {"type": "float", "low": 0.001, "high": 0.10, "log": true},
    "label_smoothing": {"type": "float", "low": 0.00, "high": 0.15},
    "grad_clip_norm": {"type": "float", "low": 0.50, "high": 2.00}
  }'::jsonb,
 'val_loss', 'min', 7, 0.0001, 8),

-- Stage 4: mencari performa maksimum setelah ruang konfigurasi lebih stabil.
-- Objective Optuna = maximize val_macro_f1.
-- Tetap 5-fold non-repeated supaya HPO tidak terlalu mahal.
(4, 'classical_fully_spatial', 'Stage 4 - Maximum Accuracy Tuning', 'MAX_ACCURACY', true,
 0.80, 0.20, 'CV_5_FOLD', 5, 1, 50,
 'cqcnn_stage4_classical_accuracy', 40, 'maximize', 'ACCURACY',
 '{
    "lr_backbone": {"type": "float", "low": 0.0001, "high": 0.001, "log": true},
    "lr_head": {"type": "float", "low": 0.0003, "high": 0.003, "log": true},
    "weight_decay": {"type": "float", "low": 0.000001, "high": 0.001, "log": true},
    "dropout": {"type": "float", "low": 0.10, "high": 0.35},
    "activation_fn": {"type": "categorical", "choices": ["leaky_relu"]},
    "leaky_relu_negative_slope": {"type": "float", "low": 0.001, "high": 0.10, "log": true},
    "label_smoothing": {"type": "float", "low": 0.00, "high": 0.15},
    "grad_clip_norm": {"type": "float", "low": 0.50, "high": 2.00}
  }'::jsonb,
 'val_macro_f1', 'max', 12, 0.001, 15),

-- Stage 5: final evaluation only.
-- Tidak ada hyperparameter optimization.
-- Worker memakai best configuration dari Stage 4, lalu menjalankan repeated 5-fold.
(5, 'classical_fully_spatial', 'Stage 5 - Repeated K-Fold Final Evaluation', 'FINALIZATION', true,
 0.80, 0.20, 'REPEATED_5_FOLD', 5, 5, 100,
 null, 0, null, 'FINAL_EVALUATION',
 null,
 'val_macro_f1', 'max', 15, 0.001, 20),

-- =============================================================
-- HYBRID QCQ-CNN
-- =============================================================
(1, 'hybrid_qcqcnn', 'Stage 1 - Sanity Test', 'SANITY', true,
 0.80, 0.20, 'SIMPLE_80_20', null, 1, 2,
 'cqcnn_stage1_hybrid_sanity', 1, 'minimize', 'NONE',
 '{
    "lr_backbone": {"type": "float", "low": 0.0005, "high": 0.0005},
    "lr_head": {"type": "float", "low": 0.001, "high": 0.001},
    "lr_quantum": {"type": "float", "low": 0.0005, "high": 0.0005},
    "q_depth": {"type": "categorical", "choices": [2]},
    "quantum_measurement": {"type": "categorical", "choices": ["pauli_z_linear"]}
  }'::jsonb,
 null, null, null, null, null),

(2, 'hybrid_qcqcnn', 'Stage 2 - Warm Start', 'WARM_START', true,
 0.80, 0.20, 'SIMPLE_80_20', null, 1, 5,
 'cqcnn_stage2_hybrid_warmstart', 1, 'minimize', 'NONE',
 '{
    "lr_backbone": {"type": "float", "low": 0.0005, "high": 0.0005},
    "lr_head": {"type": "float", "low": 0.001, "high": 0.001},
    "lr_quantum": {"type": "float", "low": 0.0005, "high": 0.0005},
    "q_depth": {"type": "categorical", "choices": [2]},
    "quantum_measurement": {"type": "categorical", "choices": ["pauli_z_linear"]}
  }'::jsonb,
 'val_loss', 'min', 5, 0.0001, 5),

-- Stage 3: convergence-oriented HPO untuk hybrid.
-- Objective = minimize val_loss.
-- Search space memprioritaskan learning rate dan regularization agar trainability stabil.
(3, 'hybrid_qcqcnn', 'Stage 3 - Convergence Tuning', 'CONVERGENCE', true,
 0.80, 0.20, 'CV_5_FOLD', 5, 1, 25,
 'cqcnn_stage3_hybrid_convergence', 20, 'minimize', 'CONVERGENCE',
 '{
    "lr_backbone": {"type": "float", "low": 0.0001, "high": 0.001, "log": true},
    "lr_head": {"type": "float", "low": 0.0003, "high": 0.003, "log": true},
    "lr_quantum": {"type": "float", "low": 0.00005, "high": 0.001, "log": true},
    "weight_decay": {"type": "float", "low": 0.000001, "high": 0.001, "log": true},
    "dropout": {"type": "float", "low": 0.10, "high": 0.35},
    "activation_fn": {"type": "categorical", "choices": ["leaky_relu"]},
    "leaky_relu_negative_slope": {"type": "float", "low": 0.001, "high": 0.10, "log": true},
    "label_smoothing": {"type": "float", "low": 0.00, "high": 0.15},
    "grad_clip_norm": {"type": "float", "low": 0.50, "high": 2.00},
    "q_depth": {"type": "categorical", "choices": [2]},
    "quantum_measurement": {"type": "categorical", "choices": ["pauli_z_linear"]}
  }'::jsonb,
 'val_loss', 'min', 7, 0.0001, 8),

-- Stage 4: maximum-accuracy-oriented HPO untuk hybrid.
-- Objective = maximize val_macro_f1.
-- Search space sedikit lebih luas, tetapi depth tetap dikunci ke 2 agar apple-to-apple dan trainability aman.
(4, 'hybrid_qcqcnn', 'Stage 4 - Maximum Accuracy Tuning', 'MAX_ACCURACY', true,
 0.80, 0.20, 'CV_5_FOLD', 5, 1, 50,
 'cqcnn_stage4_hybrid_accuracy', 40, 'maximize', 'ACCURACY',
 '{
    "lr_backbone": {"type": "float", "low": 0.0001, "high": 0.001, "log": true},
    "lr_head": {"type": "float", "low": 0.0003, "high": 0.003, "log": true},
    "lr_quantum": {"type": "float", "low": 0.00005, "high": 0.001, "log": true},
    "weight_decay": {"type": "float", "low": 0.000001, "high": 0.001, "log": true},
    "dropout": {"type": "float", "low": 0.10, "high": 0.35},
    "activation_fn": {"type": "categorical", "choices": ["leaky_relu"]},
    "leaky_relu_negative_slope": {"type": "float", "low": 0.001, "high": 0.10, "log": true},
    "label_smoothing": {"type": "float", "low": 0.00, "high": 0.15},
    "grad_clip_norm": {"type": "float", "low": 0.50, "high": 2.00},
    "q_depth": {"type": "categorical", "choices": [2]},
    "quantum_measurement": {"type": "categorical", "choices": ["pauli_z_linear"]}
  }'::jsonb,
 'val_macro_f1', 'max', 12, 0.001, 15),

-- Stage 5: final repeated k-fold evaluation only.
-- Tidak ada HPO. Worker memakai best configuration dari Stage 4.
(5, 'hybrid_qcqcnn', 'Stage 5 - Repeated K-Fold Final Evaluation', 'FINALIZATION', true,
 0.80, 0.20, 'REPEATED_5_FOLD', 5, 5, 100,
 null, 0, null, 'FINAL_EVALUATION',
 null,
 'val_macro_f1', 'max', 15, 0.001, 20)

on conflict (stage_no, model_type) do update
set
    stage_name = excluded.stage_name,
    stage_objective = excluded.stage_objective,
    is_active = excluded.is_active,
    train_ratio = excluded.train_ratio,
    validation_ratio = excluded.validation_ratio,
    split_strategy = excluded.split_strategy,
    k_folds = excluded.k_folds,
    n_repeats = excluded.n_repeats,
    max_epoch = excluded.max_epoch,
    optuna_study_name = excluded.optuna_study_name,
    optuna_trials = excluded.optuna_trials,
    optuna_direction = excluded.optuna_direction,
    tuning_focus = excluded.tuning_focus,
    search_space_json = excluded.search_space_json,
    early_stop_monitor = excluded.early_stop_monitor,
    early_stop_mode = excluded.early_stop_mode,
    early_stop_patience = excluded.early_stop_patience,
    early_stop_min_delta = excluded.early_stop_min_delta,
    min_epoch_before_stop = excluded.min_epoch_before_stop,
    updated_at = now();

commit;

-- Verifikasi hasil seed.
select
    stage_no,
    model_type,
    stage_name,
    stage_objective,
    split_strategy,
    k_folds,
    n_repeats,
    max_epoch,
    optuna_trials,
    optuna_direction,
    tuning_focus,
    early_stop_monitor,
    early_stop_mode,
    early_stop_patience,
    min_epoch_before_stop
from public.stage_information
order by model_type, stage_no;
