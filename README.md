# Explainable Hybrid CQ-CNN for Indonesian Spice Classification

This repository contains an undergraduate thesis pipeline for Indonesian spice
image classification using two controlled comparison models:

1. Classical Fully Spatial CNN as the classical baseline.
2. Hybrid CQ-CNN as a quantum-classical classifier variant.

The project does not claim quantum advantage. The hybrid model is treated as a
controlled CNN-QNN variant and is compared with the classical baseline under the
same input shape, dataset protocol, and evaluation scheme.

## Research Scope

- Dataset domain: Indonesian spice images with 10 classes.
- Input shape: grayscale images with shape `[B, 1, 128, 128]`.
- Evaluation metrics: accuracy, macro-F1, balanced accuracy, confusion matrix,
  and per-class precision, recall, and F1.
- Engineering focus: reproducible training, checkpointing, worker-based
  execution, and strict separation between orchestration metadata, Optuna trial
  storage, physical checkpoints, and local EDA outputs.

## Implemented Features

- Multi-stage EDA pipeline for dataset profiling, image-feature extraction,
  outlier auditing, and feature-level analysis.
- Curriculum learning design for Stage 1 to Stage 5, including early holdout
  splits, 5-fold cross-validation, Stage 3-4 hyperparameter optimization, and
  final Stage 5 evaluation.
- Classical Fully Spatial CNN architecture with a shared CNN backbone,
  BlurPool, GroupNorm, CBAM attention, spatial bottleneck, and spatial
  classification head.
- Hybrid CQ-CNN architecture with the same shared CNN backbone, spatial
  collapse, amplitude encoding, StronglyEntanglingLayers, Pauli-Z expectation
  readout, and a 10-class linear classifier.
- Optuna PostgreSQL integration for studies, trials, sampled hyperparameters,
  objective values, and best-trial metadata.
- PostgreSQL orchestration layer for stages, tasks, workers, heartbeat records,
  checkpoint metadata, resume handling, and stale-task hijacking.
- Worker runtime with stage-signal polling, task claiming, heartbeat updates,
  checkpoint-before-tell ordering, and `study.tell()` after training results
  are ready.
- Checkpoint helpers for worker-local `latest.pt`, interval/best/final
  checkpoints, SHA-256 hashes, file sizes, and resume metadata.
- Preflight, smoke-test, and worker-probe scripts for environment validation.
- Local `outputs/` structure for CSV, JSON, plots, executed notebooks, and
  experiment artifacts without storing them in database tables.

## Storage Separation

| Component | Role | Publication policy |
| --- | --- | --- |
| Orchestration PostgreSQL | Stage, task, worker, heartbeat, checkpoint metadata | DSN stays in `.env`; not committed |
| Optuna PostgreSQL | Study, trial, hyperparameter, objective value, best trial | DSN stays in `.env`; not committed |
| Checkpoint `.pt` files | Model weights and training state | physical files are not committed |
| `outputs/` | CSV, JSON, plots, executed notebooks | local only, except README placeholders |
| Dataset | Raw and processed spice images | not committed |

The current checkpoint helper implementation supports a Google Drive-compatible
rclone backend. If the final deployment uses Supabase Storage, credentials must
remain in `.env` or a secret manager, and the same rule still applies:
PostgreSQL stores metadata only, while physical `.pt` files live in object
storage.

## Main Directory Structure

```text
01_eda_pipeline/              EDA notebooks and local EDA workflow
02_curriculum_pipeline/       Curriculum split and stage-audit workflow
03_model_architecture/        Classical CNN and Hybrid CQ-CNN definitions
04_runtime_final/             Optuna orchestrator, worker, and DB runtime
patched_program_files/        Python modules used by notebooks/runtime
notebooks/                    Orchestrator and worker entry points
sql/                          Schema, functions, seed data, readiness checks
docs/                         Setup and experiment-planning documentation
tests/                        Smoke tests, integrity tests, worker probes
outputs/                      Local generated outputs, not published
```

## Running the Pipeline

1. Create a Python environment from `requirements.txt`.
2. Copy `.env.example` to `.env`, then fill in local/private values.
3. Run the SQL files according to `sql/README.md`.
4. Follow the full execution order in `RUN_ORDER_END_TO_END.md`.
5. Read `WORKFLOW_DETAIL_FROM_EDA_TO_WORKER.md` for the EDA-to-worker flow.
6. Read `docs/GITHUB_PUBLICATION_GUIDE.md` before publishing or pushing
   repository changes.

## Public Test Commands

Run the smoke-test orchestrator:

```powershell
python tests/run_all_smoke_tests.py
```

Run the worker environment probe:

```powershell
python tests/worker_smoke_estimate.py --model hybrid --stage-from 3 --stage-to 5
```

Some tests require local PostgreSQL, Optuna PostgreSQL, dataset manifests, and
rclone-compatible checkpoint configuration. Credential values must stay outside
Git.

## Research Contribution Framing

- Core novelty: a controlled comparison design between a fully spatial
  classical CNN baseline and a Hybrid CQ-CNN variant for Indonesian spice image
  classification.
- Applied novelty: an Indonesian spice image-classification pipeline combining
  EDA, curriculum learning, explainability preparation, and multi-metric
  evaluation.
- Engineering contribution: PostgreSQL-based orchestration, Optuna RDB storage,
  worker execution, heartbeat monitoring, checkpoint metadata, and recovery
  support.
- Future work: statistical validation across seeds, additional quantum-backend
  comparison, final object-storage integration, and XAI analysis on selected
  best checkpoints.

All performance claims must be derived from reproduced experiments. This
repository does not claim quantum advantage without sufficient empirical and
statistical evidence.
