# Local PostgreSQL for Orchestration and Optuna Storage

## Purpose

Local PostgreSQL now owns both database roles, but they should stay in separate databases:

- `cqcnn_orchestration`: task queue, heartbeat, worker status, checkpoint metadata, resume, and hijacking.
- `optuna_skripsi`: authoritative Optuna RDB storage for study, trial, sampled hyperparameters, objective values, and best trial.

Physical checkpoint `.pt` files are not stored in PostgreSQL. Important interval, best, and final checkpoints are uploaded to Google Drive through rclone; each worker keeps a local checkpoint cache for fast self-recovery.

## Optuna database

- study metadata
- trial numbers
- sampled hyperparameters
- trial states
- objective values
- best trial

## Orchestration database

- task queue
- heartbeat
- checkpoint metadata
- resume/hijacking

## Minimal PostgreSQL setup

Create databases and users in pgAdmin or `psql`:

```sql
create database cqcnn_orchestration;
create database optuna_skripsi;
create user optuna_user with encrypted password '<DB_PASSWORD>';
create user cqcnn_orchestration_user with encrypted password '<DB_PASSWORD>';
grant all privileges on database optuna_skripsi to optuna_user;
grant all privileges on database cqcnn_orchestration to cqcnn_orchestration_user;
```

If PostgreSQL 15/16 restricts public schema privileges, connect to each database and run:

```sql
-- run inside optuna_skripsi
grant all on schema public to optuna_user;
alter schema public owner to optuna_user;

-- run inside cqcnn_orchestration
grant all on schema public to cqcnn_orchestration_user;
alter schema public owner to cqcnn_orchestration_user;
```

## Connection string

Laptop-only:

```text
ORCHESTRATION_DB_DSN=postgresql://<ORCHESTRATION_DB_USER>:<DB_PASSWORD>@localhost:5432/cqcnn_orchestration
OPTUNA_STORAGE_URL=postgresql+psycopg2://<OPTUNA_DB_USER>:<DB_PASSWORD>@localhost:5432/optuna_skripsi
```

Worker over LAN:

```text
ORCHESTRATION_DB_DSN=postgresql://<ORCHESTRATION_DB_USER>:<DB_PASSWORD>@<LAPTOP_IP>:5432/cqcnn_orchestration
OPTUNA_STORAGE_URL=postgresql+psycopg2://<OPTUNA_DB_USER>:<DB_PASSWORD>@<LAPTOP_IP>:5432/optuna_skripsi
```

## LAN access checklist for worker PCs

Only needed if worker PCs access the laptop PostgreSQL:

1. PostgreSQL `listen_addresses` allows LAN access.
2. `pg_hba.conf` allows worker IPs.
3. Windows Firewall allows inbound TCP 5432.
4. Laptop does not sleep during experiments.
5. Laptop IP is stable.

For initial testing, run only on laptop with `localhost`.

## rclone checklist for Google Drive checkpoints

1. Configure a Google Drive remote, for example `rclone config` then remote name `gdrive`.
2. Set `RCLONE_REMOTE_NAME=gdrive`.
3. Set `GDRIVE_CHECKPOINT_ROOT=cqcnn_checkpoints`.
4. Set `WORKER_LOCAL_CHECKPOINT_ROOT=outputs/checkpoint_cache`.
5. Verify with `rclone lsd gdrive:`.

## Notebook kernels

Use separate kernels/processes:

- Kernel A: `notebooks/01_optuna_orchestrator_postgres.ipynb`
- Kernel B: `notebooks/02_worker_pc_template.ipynb`

Worker PCs run only `02_worker_pc_template.ipynb`.
