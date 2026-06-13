-- 09_curriculum_functions.sql
-- Kumpulan fungsi SQL untuk mencatat metrics, diagnostics, dll.
-- Ini mencegah query INSERT dinamis dari level aplikasi (Python).

-- -------------------------------------------------------------
-- 1. FOLD RUN RESULT
-- -------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.log_fold_run_result(
    p_run_id TEXT DEFAULT NULL,
    p_task_id BIGINT DEFAULT NULL,
    p_stage_no INTEGER DEFAULT NULL,
    p_model_type TEXT DEFAULT NULL,
    p_optuna_study_name TEXT DEFAULT NULL,
    p_trial_number INTEGER DEFAULT NULL,
    p_repeat_id INTEGER DEFAULT NULL,
    p_fold_id INTEGER DEFAULT NULL,
    p_seed INTEGER DEFAULT NULL,
    
    p_train_size INTEGER DEFAULT NULL,
    p_val_size INTEGER DEFAULT NULL,
    p_test_size INTEGER DEFAULT NULL,
    
    p_accuracy DOUBLE PRECISION DEFAULT NULL,
    p_balanced_accuracy DOUBLE PRECISION DEFAULT NULL,
    p_macro_precision DOUBLE PRECISION DEFAULT NULL,
    p_macro_recall DOUBLE PRECISION DEFAULT NULL,
    p_macro_f1 DOUBLE PRECISION DEFAULT NULL,
    p_weighted_precision DOUBLE PRECISION DEFAULT NULL,
    p_weighted_recall DOUBLE PRECISION DEFAULT NULL,
    p_weighted_f1 DOUBLE PRECISION DEFAULT NULL,
    
    p_val_loss DOUBLE PRECISION DEFAULT NULL,
    p_train_loss DOUBLE PRECISION DEFAULT NULL,
    
    p_best_epoch INTEGER DEFAULT NULL,
    p_runtime_seconds DOUBLE PRECISION DEFAULT NULL,
    p_checkpoint_file_id BIGINT DEFAULT NULL
) RETURNS BIGINT AS $$
DECLARE
    v_id BIGINT;
BEGIN
    INSERT INTO public.fold_run_result (
        run_id, task_id, stage_no, model_type, optuna_study_name, trial_number,
        repeat_id, fold_id, seed, train_size, val_size, test_size,
        accuracy, balanced_accuracy, macro_precision, macro_recall, macro_f1,
        weighted_precision, weighted_recall, weighted_f1,
        val_loss, train_loss, best_epoch, runtime_seconds, checkpoint_file_id
    ) VALUES (
        p_run_id, p_task_id, p_stage_no, p_model_type, p_optuna_study_name, p_trial_number,
        p_repeat_id, p_fold_id, p_seed, p_train_size, p_val_size, p_test_size,
        p_accuracy, p_balanced_accuracy, p_macro_precision, p_macro_recall, p_macro_f1,
        p_weighted_precision, p_weighted_recall, p_weighted_f1,
        p_val_loss, p_train_loss, p_best_epoch, p_runtime_seconds, p_checkpoint_file_id
    ) RETURNING fold_run_id INTO v_id;
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;


-- -------------------------------------------------------------
-- 2. EPOCH METRIC LOG
-- -------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.log_epoch_metrics(
    p_task_id BIGINT DEFAULT NULL,
    p_stage_no INTEGER DEFAULT NULL,
    p_model_type TEXT DEFAULT NULL,
    p_trial_number INTEGER DEFAULT NULL,
    p_repeat_id INTEGER DEFAULT NULL,
    p_fold_id INTEGER DEFAULT NULL,
    p_epoch INTEGER DEFAULT NULL,
    
    p_train_loss DOUBLE PRECISION DEFAULT NULL,
    p_val_loss DOUBLE PRECISION DEFAULT NULL,
    p_train_acc DOUBLE PRECISION DEFAULT NULL,
    p_val_acc DOUBLE PRECISION DEFAULT NULL,
    p_val_balanced_accuracy DOUBLE PRECISION DEFAULT NULL,
    p_val_macro_precision DOUBLE PRECISION DEFAULT NULL,
    p_val_macro_recall DOUBLE PRECISION DEFAULT NULL,
    p_val_macro_f1 DOUBLE PRECISION DEFAULT NULL,
    p_val_weighted_precision DOUBLE PRECISION DEFAULT NULL,
    p_val_weighted_recall DOUBLE PRECISION DEFAULT NULL,
    p_val_weighted_f1 DOUBLE PRECISION DEFAULT NULL,
    
    p_lr_backbone DOUBLE PRECISION DEFAULT NULL,
    p_lr_head DOUBLE PRECISION DEFAULT NULL,
    p_lr_quantum DOUBLE PRECISION DEFAULT NULL,
    
    p_grad_norm_global DOUBLE PRECISION DEFAULT NULL,
    p_grad_norm_backbone DOUBLE PRECISION DEFAULT NULL,
    p_grad_norm_head DOUBLE PRECISION DEFAULT NULL,
    p_grad_norm_quantum DOUBLE PRECISION DEFAULT NULL
) RETURNS BIGINT AS $$
DECLARE
    v_id BIGINT;
BEGIN
    INSERT INTO public.epoch_metric_log (
        task_id, stage_no, model_type, trial_number, repeat_id, fold_id, epoch,
        train_loss, val_loss, train_acc, val_acc, val_balanced_accuracy,
        val_macro_precision, val_macro_recall, val_macro_f1,
        val_weighted_precision, val_weighted_recall, val_weighted_f1,
        lr_backbone, lr_head, lr_quantum,
        grad_norm_global, grad_norm_backbone, grad_norm_head, grad_norm_quantum
    ) VALUES (
        p_task_id, p_stage_no, p_model_type, p_trial_number, p_repeat_id, p_fold_id, p_epoch,
        p_train_loss, p_val_loss, p_train_acc, p_val_acc, p_val_balanced_accuracy,
        p_val_macro_precision, p_val_macro_recall, p_val_macro_f1,
        p_val_weighted_precision, p_val_weighted_recall, p_val_weighted_f1,
        p_lr_backbone, p_lr_head, p_lr_quantum,
        p_grad_norm_global, p_grad_norm_backbone, p_grad_norm_head, p_grad_norm_quantum
    ) RETURNING log_id INTO v_id;
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;


-- -------------------------------------------------------------
-- 3. CONVERGENCE DIAGNOSTIC SUMMARY
-- -------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.log_convergence_diagnostic(
    p_task_id BIGINT DEFAULT NULL,
    p_stage_no INTEGER DEFAULT NULL,
    p_model_type TEXT DEFAULT NULL,
    p_trial_number INTEGER DEFAULT NULL,
    p_repeat_id INTEGER DEFAULT NULL,
    p_fold_id INTEGER DEFAULT NULL,
    
    p_loss_plateau_detected BOOLEAN DEFAULT NULL,
    p_plateau_start_epoch INTEGER DEFAULT NULL,
    p_vanishing_gradient_detected BOOLEAN DEFAULT NULL,
    p_exploding_gradient_detected BOOLEAN DEFAULT NULL,
    p_nan_loss_detected BOOLEAN DEFAULT NULL,
    p_overfitting_detected BOOLEAN DEFAULT NULL,
    p_underfitting_detected BOOLEAN DEFAULT NULL,
    
    p_train_val_loss_gap DOUBLE PRECISION DEFAULT NULL,
    p_train_val_acc_gap DOUBLE PRECISION DEFAULT NULL,
    p_train_val_macro_f1_gap DOUBLE PRECISION DEFAULT NULL,
    
    p_best_val_loss DOUBLE PRECISION DEFAULT NULL,
    p_best_val_accuracy DOUBLE PRECISION DEFAULT NULL,
    p_best_val_macro_f1 DOUBLE PRECISION DEFAULT NULL,
    p_best_epoch INTEGER DEFAULT NULL,
    
    p_diagnostic_note TEXT DEFAULT NULL,
    p_barren_plateau_indicator BOOLEAN DEFAULT NULL,
    p_quantum_gradient_plateau_indicator BOOLEAN DEFAULT NULL
) RETURNS BIGINT AS $$
DECLARE
    v_id BIGINT;
BEGIN
    INSERT INTO public.convergence_diagnostic_summary (
        task_id, stage_no, model_type, trial_number, repeat_id, fold_id,
        loss_plateau_detected, plateau_start_epoch, vanishing_gradient_detected,
        exploding_gradient_detected, nan_loss_detected, overfitting_detected,
        underfitting_detected, train_val_loss_gap, train_val_acc_gap, train_val_macro_f1_gap,
        best_val_loss, best_val_accuracy, best_val_macro_f1, best_epoch,
        diagnostic_note, barren_plateau_indicator, quantum_gradient_plateau_indicator
    ) VALUES (
        p_task_id, p_stage_no, p_model_type, p_trial_number, p_repeat_id, p_fold_id,
        p_loss_plateau_detected, p_plateau_start_epoch, p_vanishing_gradient_detected,
        p_exploding_gradient_detected, p_nan_loss_detected, p_overfitting_detected,
        p_underfitting_detected, p_train_val_loss_gap, p_train_val_acc_gap, p_train_val_macro_f1_gap,
        p_best_val_loss, p_best_val_accuracy, p_best_val_macro_f1, p_best_epoch,
        p_diagnostic_note, p_barren_plateau_indicator, p_quantum_gradient_plateau_indicator
    ) RETURNING diagnostic_id INTO v_id;
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;


-- -------------------------------------------------------------
-- 4. STATISTICAL TEST RESULT
-- -------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.log_statistical_test(
    p_stage_no INTEGER DEFAULT NULL,
    p_metric_name TEXT DEFAULT NULL,
    p_model_a TEXT DEFAULT NULL,
    p_model_b TEXT DEFAULT NULL,
    p_test_name TEXT DEFAULT NULL,
    p_statistic_value DOUBLE PRECISION DEFAULT NULL,
    p_p_value DOUBLE PRECISION DEFAULT NULL,
    p_alpha DOUBLE PRECISION DEFAULT NULL,
    p_significant BOOLEAN DEFAULT NULL,
    p_n_a INTEGER DEFAULT NULL,
    p_n_b INTEGER DEFAULT NULL,
    p_data_source TEXT DEFAULT NULL
) RETURNS BIGINT AS $$
DECLARE
    v_id BIGINT;
BEGIN
    INSERT INTO public.statistical_test_result (
        stage_no, metric_name, model_a, model_b, test_name,
        statistic_value, p_value, alpha, significant, n_a, n_b, data_source
    ) VALUES (
        p_stage_no, p_metric_name, p_model_a, p_model_b, p_test_name,
        p_statistic_value, p_p_value, p_alpha, p_significant, p_n_a, p_n_b, p_data_source
    ) RETURNING analysis_id INTO v_id;
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;


-- -------------------------------------------------------------
-- 5. EFFECT SIZE RESULT
-- -------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.log_effect_size(
    p_stage_no INTEGER DEFAULT NULL,
    p_metric_name TEXT DEFAULT NULL,
    p_model_a TEXT DEFAULT NULL,
    p_model_b TEXT DEFAULT NULL,
    p_effect_size_name TEXT DEFAULT NULL,
    p_effect_size_value DOUBLE PRECISION DEFAULT NULL,
    p_interpretation TEXT DEFAULT NULL,
    p_confidence_interval_low DOUBLE PRECISION DEFAULT NULL,
    p_confidence_interval_high DOUBLE PRECISION DEFAULT NULL,
    p_data_source TEXT DEFAULT NULL
) RETURNS BIGINT AS $$
DECLARE
    v_id BIGINT;
BEGIN
    INSERT INTO public.effect_size_result (
        stage_no, metric_name, model_a, model_b, effect_size_name,
        effect_size_value, interpretation, confidence_interval_low,
        confidence_interval_high, data_source
    ) VALUES (
        p_stage_no, p_metric_name, p_model_a, p_model_b, p_effect_size_name,
        p_effect_size_value, p_interpretation, p_confidence_interval_low,
        p_confidence_interval_high, p_data_source
    ) RETURNING analysis_id INTO v_id;
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;


-- -------------------------------------------------------------
-- 6. EXPERIMENT ARTIFACT POINTER
-- -------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.log_experiment_artifact(
    p_task_id BIGINT DEFAULT NULL,
    p_stage_no INTEGER DEFAULT NULL,
    p_model_type TEXT DEFAULT NULL,
    p_trial_number INTEGER DEFAULT NULL,
    p_repeat_id INTEGER DEFAULT NULL,
    p_fold_id INTEGER DEFAULT NULL,
    p_artifact_type TEXT DEFAULT NULL,
    p_storage_backend TEXT DEFAULT NULL,
    p_local_path TEXT DEFAULT NULL,
    p_remote_uri TEXT DEFAULT NULL,
    p_sha256 TEXT DEFAULT NULL
) RETURNS BIGINT AS $$
DECLARE
    v_id BIGINT;
BEGIN
    INSERT INTO public.experiment_artifact_pointer (
        task_id, stage_no, model_type, trial_number, repeat_id, fold_id,
        artifact_type, storage_backend, local_path, remote_uri, sha256
    ) VALUES (
        p_task_id, p_stage_no, p_model_type, p_trial_number, p_repeat_id, p_fold_id,
        p_artifact_type, p_storage_backend, p_local_path, p_remote_uri, p_sha256
    ) RETURNING artifact_id INTO v_id;
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

