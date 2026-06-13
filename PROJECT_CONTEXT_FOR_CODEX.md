# Project Context for Codex

## 1. Research Goal

This repository supports an undergraduate thesis pipeline for Indonesian spice image classification using two apple-to-apple models:

1. Classical Fully Spatial CNN
2. Hybrid QCQ-CNN

The goal is not to overclaim quantum advantage. The hybrid model is evaluated as a quantum-classical classifier variant under a controlled comparison against a classical CNN baseline.

## 2. Research Style

Use Indonesian academic language when generating thesis-related explanations.

Important principles:
- Avoid overclaiming quantum advantage.
- Separate core novelty, applied novelty, engineering contribution, and future work.
- Use IEEE references when citations are requested.
- Prefer defensible claims based on actual code, data, and literature.

## 3. Model Architecture

Both models use the same shared CNN backbone.

Input:
- grayscale image
- shape: [B, 1, 128, 128]
- number of classes: 10

Shared backbone:
- ConvBlock 1: 1 → 32, output [B, 32, 64, 64]
- ConvBlock 2: 32 → 64, output [B, 64, 32, 32]
- ConvBlock 3: 64 → 128, output [B, 128, 16, 16]
- CBAM attention
- Bottleneck Conv2D 1×1: 128 → 16
- final feature map: [B, 16, 16, 16]

Classical Fully Spatial CNN:
- uses shared backbone
- no flatten before classifier
- Spatial Conv2D head: 16 → 10
- GroupNorm
- ReLU
- Global Average Pooling
- output: [B, 10]

Hybrid QCQ-CNN:
- uses shared backbone
- flatten [B, 16, 16, 16] into [B, 4096]
- Linear projection 4096 → 256
- LayerNorm
- L2 normalization
- AmplitudeEmbedding into 8 qubits because 2^8 = 256
- StronglyEntanglingLayers depth 2
- Pauli-Z expectation readout: [B, 8]
- Linear readout: 8 to 10
- output logits: [B, 10]

## 4. Curriculum Learning Design

Dataset workflow:
1. EDA dataset awal
2. EDA fitur gambar
3. post-processing EDA and outlier audit
4. final dataset manifest
5. curriculum split

Global split:
- development set
- locked holdout test set

Stages:
- Stage 1: sanity test, 5% dev, holdout 80/20, 1 formal trial
- Stage 2: warm start, 25% dev, holdout 80/20, 1 formal trial
- Stage 3: convergence tuning, 50% dev, 5-fold CV, Optuna active, objective = minimize val_loss
- Stage 4: maximum accuracy tuning, 100% dev, 5-fold CV, Optuna active, objective = maximize val_macro_f1
- Stage 5: final repeated 5-fold evaluation, no HPO, use best config from Stage 4

## 5. Database and Storage Decisions

Local PostgreSQL database `cqcnn_orchestration` is the passive orchestration database. It stores:
- stage_information
- worker_node
- task
- checkpoint_slot
- checkpoint_file

Local PostgreSQL database `optuna_skripsi` is the Optuna RDB storage. It stores:
- study
- trial
- sampled hyperparameters
- trial state
- objective value
- best trial

Google Drive via rclone stores physical checkpoint .pt files:
- interval_epoch_XXX.pt
- best.pt
- final.pt

Do not store notebook artifacts in PostgreSQL:
- no artifact_file table
- no experiment-artifacts bucket
- EDA plots, CSV, JSON, figures, and executed notebooks stay local under outputs/

Orchestration task is orchestration/log layer:
- not the authoritative source of hyperparameter configuration
- may store trial_params_json only as an audit snapshot/fallback
- authoritative trial parameters come from Optuna PostgreSQL

Checkpoint metadata is also a pointer/index for resume and hijacking control. It stores Google Drive relative path, checkpoint URI, SHA-256, file size, upload status, epoch, metric, and checkpoint content flags.

## 6. Runtime Architecture

Laptop personal machine:
- Kernel A: optuna_orchestrator_postgres.ipynb
- Kernel B: worker_pc_template.ipynb

Other worker PCs:
- worker_pc_template.ipynb only

Optuna orchestrator:
- opens local PostgreSQL Optuna storage
- reads stage_information from cqcnn_orchestration
- creates/loads Optuna study
- calls study.ask()
- calls trial.suggest_*
- inserts task into cqcnn_orchestration
- keeps active tasks approximately equal to active worker count
- does not generate all trials at once

Worker:
- registers hardware profile
- claims task from cqcnn_orchestration
- reads optuna_study_name and trial_nr
- loads trial params from Optuna PostgreSQL
- trains model
- updates heartbeat to cqcnn_orchestration
- writes latest.pt to local checkpoint cache every epoch for fast self-recovery
- uploads interval/best/final checkpoint .pt files to Google Drive via rclone
- registers checkpoint metadata in cqcnn_orchestration
- calls study.tell(objective_value)
- marks task TOLD/DONE

## 7. Idle and Completion Logic

Worker must not assume that no WAITING task means stage is complete.

Use worker stage signal logic:
- HAS_WAITING_TASK: claim task
- WAIT_FOR_DISPATCHER: idle and poll; dispatcher has not generated next Bayesian trial yet
- WAIT_FOR_OPTUNA_TELL: training result exists but tell/update has not completed
- WAIT_FOR_RUNNING_TASKS: other workers are still running
- HAS_STALE_TASK: stale task can be hijacked if no WAITING task exists
- STAGE_COMPLETE: all trials are completed/told; worker can stop

## 8. Checkpointing

Checkpoint .pt should contain:
- model_state_dict
- optimizer_state_dict
- scheduler_state_dict
- epoch
- global_step
- best_metric
- model_config
- runtime_plan
- seed

Google Drive via rclone:
- stores interval/best/final .pt files

Local worker cache:
- stores latest.pt for fast self-recovery on the same machine

Local PostgreSQL cqcnn_orchestration:
- checkpoint_file stores metadata and Google Drive pointers
- checkpoint_slot points to latest, best, and final checkpoint metadata rows

## 9. Refactor Policy for Codex

When editing code:
- keep structure efficient
- do not add unnecessary SQL files
- do not add artifact database tables
- do not move EDA outputs into PostgreSQL
- do not make worker depend on SQLite Optuna
- preserve cqcnn_orchestration as orchestration layer
- preserve optuna_skripsi as Optuna storage
- preserve Google Drive/rclone as physical checkpoint storage
- prefer small utility functions
- avoid bloated abstractions
- keep notebook wrappers thin
- put reusable logic in .py files

## 10. Files Codex Should Check First

Start from:
- README.md
- RUN_ORDER_END_TO_END.md
- WORKFLOW_DETAIL_FROM_EDA_TO_WORKER.md
- docs/LOCAL_POSTGRES_OPTUNA_SETUP.md
- sql/README.md
- patched_program_files/optuna_postgres_utils.py
- patched_program_files/optuna_stage_manager.py
- patched_program_files/worker_loop.py
- patched_program_files/postgres_orchestration_db.py
- patched_program_files/checkpoint_rclone.py
- patched_program_files/worker_hardware_profile.py
- patched_program_files/curriculum_stage_utils.py
- patched_program_files/model_architecture_modules.py
