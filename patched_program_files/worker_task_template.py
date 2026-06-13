"""worker_task_template.py
Real training implementation for CQ-CNN / Hybrid QCQ-CNN worker orchestration.

Contains:
- dry_run_train_one_task(): Safe dry-run for flow validation (preserved as fallback).
- real_train_one_task(): Full PyTorch training with heartbeat, checkpoint, and metrics.
- train_one_task: Default alias pointing to real_train_one_task.
"""
from __future__ import annotations

import math
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from postgres_orchestration_db import PostgresOrchestrationDb

# Model architecture (no training logic lives here)
from model_architecture_modules import (
    ClassicalFullySpatialCNN,
    HybridQCQCNN,
    ModelConfig,
    build_loss,
    build_optimizer,
    build_scheduler,
    clip_gradients,
    coerce_model_config,
    normalize_model_type,
    set_global_seed,
)

# Checkpoint helpers
from checkpoint_rclone import (
    RcloneCheckpointConfig,
    build_worker_latest_checkpoint_path,
    capture_rng_state,
    load_worker_latest_checkpoint,
    restore_rng_state,
    save_worker_latest_checkpoint,
    upload_and_register_checkpoint,
)

# Dataset
from curriculum_dataset import get_train_val_loaders

# Worker runtime
from worker_runtime_config import (
    WorkerRuntimeProfile,
    apply_runtime_to_model_config,
    build_runtime_plan,
    configure_torch_backend,
    detect_worker_resources,
    resolve_torch_device,
)

# Heartbeat helpers (best-effort, never crash training)
try:
    from worker_heartbeat import (
        mark_worker_failed,
        record_checkpoint_event,
        update_worker_heartbeat,
    )
    _HEARTBEAT_HELPERS = True
except ImportError:
    _HEARTBEAT_HELPERS = False


# ---------------------------------------------------------------------------
# Dry-run trainer (preserved as fallback/debug)
# ---------------------------------------------------------------------------

def dry_run_train_one_task(
    task: dict[str, Any],
    db: PostgresOrchestrationDb,
    worker_uid: str,
    trial_params: dict[str, Any],
    heartbeat_callback: Callable[[], bool],
) -> dict[str, Any]:
    """A safe dry-run trainer for checking dispatcher/worker/local PostgreSQL flow.

    It does not train a model. It simulates epochs, sends heartbeat, and returns a
    synthetic objective. Use this to validate queue/idle/tell mechanics before GPU work.
    """
    stage_no = int(task["stage_no"])
    objective_metric = task.get("objective_metric_name") or ("val_loss" if stage_no == 3 else "val_macro_f1")
    requires_tell = task.get("optuna_tell_status") == "PENDING"

    print(f"[dry-run] task={task['task_id']} trial={task.get('trial_nr')} params={trial_params}")
    for epoch in range(1, 4):
        time.sleep(0.5)
        heartbeat_callback()
        print(f"[dry-run] epoch={epoch} heartbeat sent")

        try:
            db.log_epoch_metrics(
                task_id=task["task_id"],
                stage_no=stage_no,
                model_type=task.get("model_type", "classical_cnn"),
                trial_number=task.get("trial_nr"),
                repeat_id=task.get("repeat_id", 0),
                fold_id=task.get("fold_id", 0),
                epoch=epoch,
                train_loss=0.5 / epoch,
                val_loss=0.6 / epoch,
                train_acc=0.8 + (0.05 * epoch),
                val_acc=0.75 + (0.05 * epoch),
                val_macro_f1=0.7 + (0.05 * epoch),
                grad_norm_global=2.5 - (0.1 * epoch)
            )
        except (AttributeError, Exception):
            pass

    trial_nr = int(task.get("trial_nr") or 0)
    random.seed(42 + trial_nr)
    if objective_metric == "val_loss":
        value = 0.6 + random.random() * 0.3
    else:
        value = 0.55 + random.random() * 0.25

    return {
        "objective_metric_name": objective_metric,
        "objective_value": value,
        "requires_optuna_tell": requires_tell,
    }


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_multiclass_metrics(preds: list[int], labels: list[int]) -> dict:
    from sklearn.metrics import (
        f1_score, balanced_accuracy_score, precision_score, recall_score,
    )
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }


def compute_epoch_seed(base_seed: int, repeat_id: int, fold_id: int, epoch: int) -> int:
    """Deterministic per-epoch seed with positional spacing.

    Formula: base_seed + repeat_id * 1000 + fold_id * 100 + epoch

    Slot allocation (no overlap for epoch<100, fold<10):
      epoch  : occupies [0..99]
      fold_id: occupies [0, 100, 200, ..., 900]
      repeat : occupies [0, 1000, 2000, ...]

    Stage 1-2: repeat=0, fold=0  -> base + epoch
    Stage 3-4: repeat=0, fold=k  -> base + k*100 + epoch
    Stage 5  : repeat=n, fold=k  -> base + n*1000 + k*100 + epoch
    """
    return base_seed + repeat_id * 1000 + fold_id * 100 + epoch


# ---------------------------------------------------------------------------
# Real trainer
# ---------------------------------------------------------------------------

def real_train_one_task(
    task: dict[str, Any],
    db: PostgresOrchestrationDb,
    worker_uid: str,
    trial_params: dict[str, Any],
    heartbeat_callback: Callable[[], bool],
) -> dict[str, Any]:
    """Full PyTorch training loop for a single task/trial.

    Supports all 5 curriculum stages for both ClassicalFullySpatialCNN and
    HybridQCQCNN. Includes checkpoint resume for fault tolerance.
    """
    # ── 1. Validate task ──────────────────────────────────────────────
    task_id = int(task["task_id"])
    stage_no = int(task["stage_no"])
    raw_model_type = str(task.get("model_type") or "classical_fully_spatial")
    model_type = normalize_model_type(raw_model_type)
    trial_nr = task.get("trial_nr")
    fold_id = int(task.get("fold_id") or 0)
    repeat_id = int(task.get("repeat_id") or 0)
    seed = int(task.get("seed") or 42)
    max_epoch = int(task.get("max_epoch") or 2)
    objective_metric = task.get("objective_metric_name") or ("val_loss" if stage_no <= 3 else "val_macro_f1")
    objective_direction = str(task.get("objective_direction") or "minimize").lower()
    requires_tell = task.get("optuna_tell_status") == "PENDING"

    print(f"[train] task={task_id} stage={stage_no} model={model_type} "
          f"trial={trial_nr} fold={fold_id} repeat={repeat_id} max_epoch={max_epoch}")

    # ── 2. Set seed ───────────────────────────────────────────────────
    set_global_seed(seed)

    # ── 3. Build ModelConfig from trial_params ────────────────────────
    cfg_overrides: dict[str, Any] = {
        "lr_backbone": trial_params.get("lr_backbone"),
        "lr_head": trial_params.get("lr_head"),
        "weight_decay": trial_params.get("weight_decay"),
        "dropout": trial_params.get("dropout"),
        "activation_fn": trial_params.get("activation_fn"),
        "leaky_relu_negative_slope": trial_params.get("leaky_relu_negative_slope"),
        "label_smoothing": trial_params.get("label_smoothing"),
        "grad_clip_norm": trial_params.get("grad_clip_norm"),
    }
    if model_type == "hybrid":
        cfg_overrides["lr_quantum"] = trial_params.get("lr_quantum")
        cfg_overrides["q_depth"] = trial_params.get("q_depth")
        cfg_overrides["quantum_measurement"] = trial_params.get("quantum_measurement")

    cfg = coerce_model_config(None, **{k: v for k, v in cfg_overrides.items() if v is not None})

    # ── 4. Resolve device ─────────────────────────────────────────────
    profile = WorkerRuntimeProfile(worker_id=worker_uid, base_seed=seed)
    configure_torch_backend(profile)
    device_str = resolve_torch_device(profile)
    device = torch.device(device_str)
    resources = detect_worker_resources(profile)
    plan = build_runtime_plan(model_type, profile, resources)
    batch_size = plan.micro_batch_size
    accum_steps = plan.gradient_accumulation_steps

    if model_type == "hybrid":
        cfg = apply_runtime_to_model_config(cfg, profile)

    print(f"[train] device={device_str} micro_batch={batch_size} accum_steps={accum_steps} "
          f"effective_batch={plan.effective_batch_size} "
          f"lr_backbone={cfg.lr_backbone} lr_head={cfg.lr_head} dropout={cfg.dropout}")

    # ── 5. Build model ────────────────────────────────────────────────
    if model_type == "classical":
        model = ClassicalFullySpatialCNN(cfg).to(device)
    else:
        print(f"[train] Building HybridQCQCNN: q_depth={cfg.q_depth} "
              f"measurement={cfg.quantum_measurement} lr_quantum={cfg.lr_quantum}")
        model = HybridQCQCNN(cfg).to(device)

    # ── 6. Build optimizer, scheduler, loss ────────────────────────────
    optimizer = build_optimizer(model, cfg, model_type)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = build_loss(cfg)

    # ── 7. Load data ──────────────────────────────────────────────────
    train_loader, val_loader, class_names = get_train_val_loaders(
        stage_no=stage_no,
        repeat_id=repeat_id,
        fold_id=fold_id,
        batch_size=batch_size,
        num_workers=plan.dataloader_workers,
    )
    print(f"[train] train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)}")

    # ── 8. Heartbeat awal ─────────────────────────────────────────────
    heartbeat_callback()
    if _HEARTBEAT_HELPERS:
        try:
            update_worker_heartbeat(
                db, worker_uid,
                status="RUNNING",
                current_task_id=task_id,
                stage_no=stage_no,
                model_type=raw_model_type,
            )
        except Exception:
            pass

    # ── 9. Resume from checkpoint (fault tolerance) ────────────────────
    start_epoch = 1
    best_metric_value = -math.inf if objective_direction == "maximize" else math.inf
    best_epoch = 0
    best_metrics_dict: dict[str, float] = {}
    global_step = 0
    rclone_cfg = RcloneCheckpointConfig.from_env()

    latest_ckpt_path = build_worker_latest_checkpoint_path(
        worker_uid=worker_uid, task=task, config=rclone_cfg,
    )
    if latest_ckpt_path.exists():
        try:
            ckpt = load_worker_latest_checkpoint(latest_ckpt_path, map_location=device_str)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict") and scheduler is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if ckpt.get("rng_state"):
                restore_rng_state(ckpt["rng_state"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            global_step = int(ckpt.get("global_step", 0))
            extra = ckpt.get("extra") or {}
            if objective_metric == "val_loss" and "val_loss" in extra:
                best_metric_value = float(extra["val_loss"])
            elif "val_macro_f1" in extra:
                best_metric_value = float(extra["val_macro_f1"])
            best_epoch = int(ckpt.get("epoch", 0))
            print(f"[train] RESUMED from checkpoint: epoch={start_epoch - 1}, "
                  f"global_step={global_step}, best_metric={best_metric_value:.4f}")
        except Exception as e:
            print(f"[train] Warning: failed to resume from {latest_ckpt_path}: {e}. Starting fresh.")
            start_epoch = 1
            global_step = 0
            best_metric_value = -math.inf if objective_direction == "maximize" else math.inf
    else:
        print(f"[train] No existing checkpoint found. Starting from epoch 1.")

    if start_epoch > max_epoch:
        print(f"[train] Already completed all {max_epoch} epochs. Returning cached result.")
        return {
            "objective_metric_name": objective_metric,
            "objective_value": float(best_metric_value),
            "requires_optuna_tell": requires_tell,
            "stage_no": stage_no,
            "model_type": model_type,
            "best_epoch": best_epoch,
            "final_epoch": max_epoch,
            "status": "OK",
        }

    for epoch in range(start_epoch, max_epoch + 1):
        # ── Reseed per-epoch for deterministic data ordering ───────
        epoch_seed = compute_epoch_seed(seed, repeat_id, fold_id, epoch)
        set_global_seed(epoch_seed)

        # ── Train phase ───────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        for micro_idx, (batch_imgs, batch_labels) in enumerate(train_loader):
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_imgs)
            loss = criterion(logits, batch_labels)
            loss = loss / accum_steps

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss detected at epoch {epoch}, step {global_step}. Training aborted.")

            loss.backward()

            train_loss_sum += loss.item() * accum_steps * batch_imgs.size(0)
            train_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            train_total += batch_imgs.size(0)

            if (micro_idx + 1) % accum_steps == 0 or (micro_idx + 1) == len(train_loader):
                grad_norm = clip_gradients(model, cfg)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        scheduler.step()
        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ── Validation phase ──────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)

                logits = model(batch_imgs)
                loss = criterion(logits, batch_labels)

                val_loss_sum += loss.item() * batch_imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_imgs.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch_labels.cpu().tolist())

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        mc_metrics = _compute_multiclass_metrics(all_preds, all_labels)

        print(
            f"[train] epoch={epoch}/{max_epoch} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"macro_f1={mc_metrics['macro_f1']:.4f}"
        )

        # ── Heartbeat ─────────────────────────────────────────────
        heartbeat_callback()
        if _HEARTBEAT_HELPERS:
            try:
                update_worker_heartbeat(
                    db, worker_uid,
                    status="RUNNING",
                    current_task_id=task_id,
                    stage_no=stage_no,
                    model_type=raw_model_type,
                    current_epoch=epoch,
                )
            except Exception:
                pass

        # ── Log epoch metrics to PostgreSQL ───────────────────────
        try:
            db.log_epoch_metrics(
                task_id=task_id,
                stage_no=stage_no,
                model_type=raw_model_type,
                trial_number=trial_nr,
                repeat_id=repeat_id,
                fold_id=fold_id,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                val_macro_f1=mc_metrics["macro_f1"],
                val_balanced_accuracy=mc_metrics["balanced_accuracy"],
                val_macro_precision=mc_metrics["macro_precision"],
                val_macro_recall=mc_metrics["macro_recall"],
                val_weighted_f1=mc_metrics["weighted_f1"],
                grad_norm_global=grad_norm,
            )
        except Exception as e:
            print(f"[train] Warning: log_epoch_metrics failed: {e}")

        # ── Save local latest.pt ──────────────────────────────────
        rng_state = capture_rng_state()
        local_ckpt_info = save_worker_latest_checkpoint(
            worker_uid=worker_uid,
            task=task,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            trial_params=trial_params,
            model_config=cfg,
            rng_state=rng_state,
            extra={"val_loss": val_loss, "val_macro_f1": mc_metrics["macro_f1"]},
        )

        # ── Track best metric ─────────────────────────────────────
        if objective_metric == "val_loss":
            current_metric = val_loss
        else:
            current_metric = mc_metrics.get("macro_f1", val_acc)

        is_best = (
            (objective_direction == "maximize" and current_metric > best_metric_value)
            or (objective_direction == "minimize" and current_metric < best_metric_value)
        )

        if is_best:
            best_metric_value = current_metric
            best_epoch = epoch
            best_metrics_dict = {
                "val_loss": val_loss,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "train_acc": train_acc,
                "balanced_accuracy": mc_metrics["balanced_accuracy"],
                "macro_precision": mc_metrics["macro_precision"],
                "macro_recall": mc_metrics["macro_recall"],
                "macro_f1": mc_metrics["macro_f1"],
                "weighted_f1": mc_metrics["weighted_f1"]
            }

            # Save best.pt locally then upload
            best_path = Path(local_ckpt_info["local_cache_path"]).parent / "best.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": rng_state,
                "epoch": epoch,
                "global_step": global_step,
                "best_metric_name": objective_metric,
                "best_metric_value": best_metric_value,
                "trial_params": trial_params,
                "model_config": dict(cfg.__dict__),
                "task_id": task_id,
                "stage_no": stage_no,
                "model_type": raw_model_type,
            }, best_path)

            # Heartbeat before checkpoint upload
            heartbeat_callback()
            if _HEARTBEAT_HELPERS:
                try:
                    record_checkpoint_event(
                        db, worker_uid, task=task, epoch=epoch,
                        local_path=str(best_path), remote_path="uploading...",
                        status="UPLOADING",
                    )
                except Exception:
                    pass

            try:
                upload_and_register_checkpoint(
                    db=db,
                    local_path=best_path,
                    task=task,
                    worker_uid=worker_uid,
                    checkpoint_type="BEST",
                    epoch_number=epoch,
                    global_step=global_step,
                    metric_name=objective_metric,
                    metric_value=best_metric_value,
                    config=rclone_cfg,
                )
                print(f"[train] best.pt uploaded (epoch={epoch}, {objective_metric}={best_metric_value:.4f})")
            except Exception as e:
                if rclone_cfg.require_upload:
                    raise
                print(f"[train] Warning: best.pt upload failed (soft-fail): {e}")

            # Heartbeat after checkpoint upload
            heartbeat_callback()

    # ── 10. Final checkpoint ──────────────────────────────────────────
    final_path = Path(local_ckpt_info["local_cache_path"]).parent / "final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "epoch": max_epoch,
        "global_step": global_step,
        "best_metric_name": objective_metric,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "trial_params": trial_params,
        "model_config": dict(cfg.__dict__),
        "task_id": task_id,
        "stage_no": stage_no,
        "model_type": raw_model_type,
    }, final_path)

    heartbeat_callback()
    try:
        upload_and_register_checkpoint(
            db=db,
            local_path=final_path,
            task=task,
            worker_uid=worker_uid,
            checkpoint_type="FINAL",
            epoch_number=max_epoch,
            global_step=global_step,
            metric_name=objective_metric,
            metric_value=best_metric_value,
            config=rclone_cfg,
        )
        print(f"[train] final.pt uploaded (epoch={max_epoch})")
    except Exception as e:
        if rclone_cfg.require_upload:
            raise
        print(f"[train] Warning: final.pt upload failed (soft-fail): {e}")

    heartbeat_callback()

    # ── 10.5. Log Fold Run Result and Convergence ─────────────────────
    try:
        db.log_fold_run_result(
            task_id=task_id,
            stage_no=stage_no,
            model_type=raw_model_type,
            trial_number=trial_nr,
            repeat_id=repeat_id,
            fold_id=fold_id,
            seed=seed,
            train_size=len(train_loader.dataset),
            val_size=len(val_loader.dataset),
            test_size=0,
            accuracy=best_metrics_dict.get("val_acc"),
            balanced_accuracy=best_metrics_dict.get("balanced_accuracy"),
            macro_precision=best_metrics_dict.get("macro_precision"),
            macro_recall=best_metrics_dict.get("macro_recall"),
            macro_f1=best_metrics_dict.get("macro_f1"),
            weighted_f1=best_metrics_dict.get("weighted_f1"),
            val_loss=best_metrics_dict.get("val_loss"),
            train_loss=best_metrics_dict.get("train_loss"),
            best_epoch=best_epoch,
        )
    except Exception as e:
        print(f"[train] Warning: log_fold_run_result failed: {e}")

    try:
        db.log_convergence_diagnostic(
            task_id=task_id,
            stage_no=stage_no,
            model_type=raw_model_type,
            trial_number=trial_nr,
            repeat_id=repeat_id,
            fold_id=fold_id,
            nan_loss_detected=False, # We raise error on NaN before reaching here
            best_val_loss=best_metrics_dict.get("val_loss"),
            best_val_accuracy=best_metrics_dict.get("val_acc"),
            best_val_macro_f1=best_metrics_dict.get("macro_f1"),
            best_epoch=best_epoch,
        )
    except Exception as e:
        print(f"[train] Warning: log_convergence_diagnostic failed: {e}")

    # ── 11. Return ────────────────────────────────────────────────────
    print(f"[train] task={task_id} COMPLETE. best_epoch={best_epoch} "
          f"{objective_metric}={best_metric_value:.4f}")

    return {
        "objective_metric_name": objective_metric,
        "objective_value": float(best_metric_value),
        "requires_optuna_tell": requires_tell,
        "stage_no": stage_no,
        "model_type": model_type,
        "best_epoch": best_epoch,
        "final_epoch": max_epoch,
        "status": "OK",
    }


# Default: real trainer. Set DRY_RUN=1 env var to use dry-run.
if os.environ.get("DRY_RUN", "").strip() in {"1", "true", "yes"}:
    train_one_task = dry_run_train_one_task
else:
    train_one_task = real_train_one_task
