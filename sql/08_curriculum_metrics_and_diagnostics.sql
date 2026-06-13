-- 08_curriculum_metrics_and_diagnostics.sql
-- Migration file untuk tabel metrik, diagnostik, dan pointer artifact
-- Sesuai dengan spesifikasi Curriculum CQ-CNN.

-- -------------------------------------------------------------
-- 1. FOLD RUN RESULT
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.fold_run_result (
    fold_run_id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id              TEXT,
    task_id             BIGINT,
    stage_no            INTEGER NOT NULL,
    model_type          TEXT NOT NULL,
    optuna_study_name   TEXT,
    trial_number        INTEGER,
    repeat_id           INTEGER,
    fold_id             INTEGER,
    seed                INTEGER,
    
    train_size          INTEGER,
    val_size            INTEGER,
    test_size           INTEGER,
    
    accuracy            DOUBLE PRECISION,
    balanced_accuracy   DOUBLE PRECISION,
    macro_precision     DOUBLE PRECISION,
    macro_recall        DOUBLE PRECISION,
    macro_f1            DOUBLE PRECISION,
    weighted_precision  DOUBLE PRECISION,
    weighted_recall     DOUBLE PRECISION,
    weighted_f1         DOUBLE PRECISION,
    
    val_loss            DOUBLE PRECISION,
    train_loss          DOUBLE PRECISION,
    
    best_epoch          INTEGER,
    runtime_seconds     DOUBLE PRECISION,
    checkpoint_file_id  BIGINT, -- Nullable, referensi ke tabel checkpoint_file
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------------------------------------------
-- 2. EPOCH METRIC LOG
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.epoch_metric_log (
    log_id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_id             BIGINT,
    stage_no            INTEGER NOT NULL,
    model_type          TEXT NOT NULL,
    trial_number        INTEGER,
    repeat_id           INTEGER,
    fold_id             INTEGER,
    epoch               INTEGER NOT NULL,
    
    train_loss          DOUBLE PRECISION,
    val_loss            DOUBLE PRECISION,
    train_acc           DOUBLE PRECISION,
    val_acc             DOUBLE PRECISION,
    val_balanced_accuracy  DOUBLE PRECISION,
    val_macro_precision DOUBLE PRECISION,
    val_macro_recall    DOUBLE PRECISION,
    val_macro_f1        DOUBLE PRECISION,
    val_weighted_precision DOUBLE PRECISION,
    val_weighted_recall DOUBLE PRECISION,
    val_weighted_f1     DOUBLE PRECISION,
    
    lr_backbone         DOUBLE PRECISION,
    lr_head             DOUBLE PRECISION,
    lr_quantum          DOUBLE PRECISION,
    
    grad_norm_global    DOUBLE PRECISION,
    grad_norm_backbone  DOUBLE PRECISION,
    grad_norm_head      DOUBLE PRECISION,
    grad_norm_quantum   DOUBLE PRECISION,
    
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------------------------------------------
-- 3. CONVERGENCE DIAGNOSTIC SUMMARY
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.convergence_diagnostic_summary (
    diagnostic_id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_id             BIGINT,
    stage_no            INTEGER NOT NULL,
    model_type          TEXT NOT NULL,
    trial_number        INTEGER,
    repeat_id           INTEGER,
    fold_id             INTEGER,
    
    loss_plateau_detected       BOOLEAN,
    plateau_start_epoch         INTEGER,
    vanishing_gradient_detected BOOLEAN,
    exploding_gradient_detected BOOLEAN,
    nan_loss_detected           BOOLEAN,
    overfitting_detected        BOOLEAN,
    underfitting_detected       BOOLEAN,
    
    train_val_loss_gap          DOUBLE PRECISION,
    train_val_acc_gap           DOUBLE PRECISION,
    train_val_macro_f1_gap      DOUBLE PRECISION,
    
    best_val_loss               DOUBLE PRECISION,
    best_val_accuracy           DOUBLE PRECISION,
    best_val_macro_f1           DOUBLE PRECISION,
    best_epoch                  INTEGER,
    
    diagnostic_note             TEXT,
    barren_plateau_indicator    BOOLEAN, -- Untuk hybrid/quantum gradient stagnancy
    quantum_gradient_plateau_indicator BOOLEAN,
    
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------------------------------------------
-- 4. STATISTICAL TEST RESULT
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.statistical_test_result (
    analysis_id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    stage_no            INTEGER NOT NULL,
    metric_name         TEXT NOT NULL,
    model_a             TEXT NOT NULL,
    model_b             TEXT NOT NULL,
    test_name           TEXT NOT NULL,
    statistic_value     DOUBLE PRECISION,
    p_value             DOUBLE PRECISION,
    alpha               DOUBLE PRECISION,
    significant         BOOLEAN,
    n_a                 INTEGER,
    n_b                 INTEGER,
    data_source         TEXT NOT NULL, -- fold, repeated_fold, holdout
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------------------------------------------
-- 5. EFFECT SIZE RESULT
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.effect_size_result (
    analysis_id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    stage_no            INTEGER NOT NULL,
    metric_name         TEXT NOT NULL,
    model_a             TEXT NOT NULL,
    model_b             TEXT NOT NULL,
    effect_size_name    TEXT NOT NULL,
    effect_size_value   DOUBLE PRECISION,
    interpretation      TEXT,
    confidence_interval_low  DOUBLE PRECISION,
    confidence_interval_high DOUBLE PRECISION,
    data_source         TEXT NOT NULL, -- fold, repeated_fold, holdout
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------------------------------------------
-- 6. EXPERIMENT ARTIFACT POINTER
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.experiment_artifact_pointer (
    artifact_id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_id             BIGINT,
    stage_no            INTEGER NOT NULL,
    model_type          TEXT NOT NULL,
    trial_number        INTEGER,
    repeat_id           INTEGER,
    fold_id             INTEGER,
    
    artifact_type       TEXT NOT NULL,
    -- e.g., epoch_metric_csv, gradient_csv, prediction_csv, 
    -- loss_plot, accuracy_plot, macro_f1_plot, confusion_matrix_png
    
    storage_backend     TEXT NOT NULL, -- 'local' or 'google_drive_rclone'
    local_path          TEXT,
    remote_uri          TEXT,
    sha256              TEXT,
    
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
