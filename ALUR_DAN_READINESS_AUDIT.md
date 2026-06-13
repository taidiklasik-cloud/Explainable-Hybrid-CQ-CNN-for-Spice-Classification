# Alur dan Readiness Audit — Final Optuna PostgreSQL Version

## 1. EDA and local outputs

Notebook EDA outputs stay local:

- CSV
- JSON
- plots
- executed notebooks
- XAI images

These are not stored in PostgreSQL.

## 2. Curriculum design

Use patched curriculum files:

- Stage 1: sanity, 80/20, 1 formal task/trial
- Stage 2: warm start, 80/20, 1 formal task/trial
- Stage 3: convergence tuning, CV 5-fold, HPO active, minimize `val_loss`
- Stage 4: maximum accuracy tuning, CV 5-fold, HPO active, maximize `val_macro_f1`
- Stage 5: final repeated 5-fold evaluation, no HPO, use best Stage 4 config

## 3. Local PostgreSQL orchestration role

Local PostgreSQL database `cqcnn_orchestration` stores operational state:

- `stage_information`
- `worker_node`
- `task`
- `checkpoint_slot`
- `checkpoint_file`

The orchestration database does not become the source of hyperparameter decisions. It records trial execution.

## 4. Optuna PostgreSQL role

Local/shared PostgreSQL database `optuna_skripsi` is the authoritative Optuna storage:

- study
- trial
- params
- objective values
- best trial

Python still runs `ask`, `suggest`, and `tell`. PostgreSQL only stores state.

## 5. Orchestrator role

Laptop local runs `01_optuna_orchestrator_postgres.ipynb`:

1. read `stage_information`
2. open Optuna PostgreSQL study
3. `study.ask()`
4. `trial.suggest_*()` from `search_space_json`
5. insert orchestration task with `trial_nr`
6. keep active task count near `MAX_PARALLEL_TASKS`
7. stop when stage is complete

It does not train.

## 6. Worker role

Laptop and worker PCs run `02_worker_pc_template.ipynb`:

1. register hardware profile
2. claim WAITING task
3. read `optuna_study_name` and `trial_nr`
4. load params from Optuna PostgreSQL
5. train model
6. send heartbeat
7. upload interval/best/final checkpoint `.pt` to Google Drive via rclone
8. register checkpoint metadata
9. call `study.tell()`
10. mark orchestration task as `TOLD`

## 7. Idle/wait mechanism

If no task is available, worker calls `get_worker_stage_signal()`:

- `WAIT_FOR_DISPATCHER`: no Bayesian task generated yet, keep polling
- `WAIT_FOR_RUNNING_TASKS`: other workers are still running, keep polling
- `WAIT_FOR_OPTUNA_TELL`: fallback mode waiting for tell, keep polling
- `HAS_STALE_TASK`: can hijack if enabled
- `STAGE_COMPLETE`: stop

## 8. Checkpoint design

Physical interval/best/final `.pt` files are stored in Google Drive via rclone. Worker-local `latest.pt` is kept in local checkpoint cache for fast self-recovery.

PostgreSQL stores metadata and resume/hijacking pointers only.

## 9. Readiness checklist

- [ ] Local orchestration SQL 01 and 02 rerun with latest bundle
- [ ] Local orchestration SQL 03 seed applied
- [ ] Local orchestration SQL 06 readiness check OK
- [ ] local PostgreSQL database `cqcnn_orchestration` created
- [ ] local PostgreSQL database `optuna_skripsi` created
- [ ] rclone Google Drive remote configured
- [ ] `.env` configured
- [ ] `00_optuna_postgres_connectivity_check.ipynb` passes
- [ ] dispatcher notebook uses its own kernel
- [ ] worker notebook uses separate kernel
- [ ] dry run succeeds before real training
