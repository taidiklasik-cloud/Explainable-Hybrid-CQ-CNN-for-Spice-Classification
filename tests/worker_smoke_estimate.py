"""Worker resource, smoke-test, and runtime estimate probe.

This script is intended for a freshly deployed worker node. It does not write
artifacts; it prints the detected resource profile, runtime plan, one-batch
forward/backward smoke result, and rough Stage 2-5 estimates.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import math
import sys
import time
from pathlib import Path

from _path_setup import PROJECT_ROOT, configure_paths

ROOT = PROJECT_ROOT
configure_paths()

import torch

from curriculum_dataset import get_train_val_loaders
from model_architecture_modules import (
    ClassicalFullySpatialCNN,
    HybridQCQCNN,
    ModelConfig,
    build_loss,
    build_optimizer,
    normalize_model_type,
)
from worker_hardware_profile import detect_worker_hardware
from worker_runtime_config import (
    WorkerRuntimeProfile,
    build_runtime_plan,
    configure_torch_backend,
    detect_worker_resources,
)


STAGE_CONFIGS = {
    2: {"trials": 1, "folds": 1, "repeats": 1, "max_epoch": 5},
    3: {"trials": 20, "folds": 5, "repeats": 1, "max_epoch": 25},
    4: {"trials": 40, "folds": 5, "repeats": 1, "max_epoch": 50},
    5: {"trials": 1, "folds": 5, "repeats": 5, "max_epoch": 100},
}


def _stage_counts(stage_no: int) -> tuple[int, int, int]:
    metadata_path = ROOT / "curriculum_outputs" / f"stage_{stage_no:02d}" / "stage_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Stage metadata not found: {metadata_path}")

    import csv

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Stage metadata is empty: {metadata_path}")

    augmented_train = int(float(rows[0]["n_train_total_after_augmentation_plan"]))

    val_path = (
        ROOT
        / "curriculum_outputs"
        / f"stage_{stage_no:02d}"
        / "train_validation_subsets"
        / f"validation_natural_stage_{stage_no:02d}_repeat_00_fold_00.csv"
    )
    with val_path.open("r", encoding="utf-8-sig", newline="") as f:
        val_count = max(sum(1 for _ in f) - 1, 0)

    natural_train_path = (
        ROOT
        / "curriculum_outputs"
        / f"stage_{stage_no:02d}"
        / "train_validation_subsets"
        / f"train_natural_stage_{stage_no:02d}_repeat_00_fold_00.csv"
    )
    with natural_train_path.open("r", encoding="utf-8-sig", newline="") as f:
        natural_train = max(sum(1 for _ in f) - 1, 0)

    return natural_train, augmented_train, val_count


def _build_model(model_type: str, cfg: ModelConfig):
    if model_type == "classical":
        return ClassicalFullySpatialCNN(cfg)
    if model_type == "hybrid":
        return HybridQCQCNN(cfg)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _probe_quantum_backends() -> list[tuple[str, bool, str]]:
    try:
        import pennylane as qml
    except Exception as exc:
        return [("pennylane", False, f"import failed: {type(exc).__name__}: {exc}")]

    results: list[tuple[str, bool, str]] = []
    for device_name in ("lightning.gpu", "lightning.qubit", "default.qubit"):
        try:
            dev = qml.device(device_name, wires=2)
            results.append((device_name, True, getattr(dev, "name", type(dev).__name__)))
        except Exception as exc:
            results.append((device_name, False, f"{type(exc).__name__}: {exc}"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test and estimate this worker node.")
    parser.add_argument("--model", choices=["classical", "hybrid", "classical_fully_spatial", "hybrid_qcqcnn"], default="classical")
    parser.add_argument("--torch-mode", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--quantum-mode", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--stage-from", type=int, default=2)
    parser.add_argument("--stage-to", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=8)
    parser.add_argument("--worker-id", default="smoke_estimate_worker")
    args = parser.parse_args()

    model_type = normalize_model_type(args.model)
    profile = WorkerRuntimeProfile(
        worker_id=args.worker_id,
        torch_mode=args.torch_mode,
        quantum_mode=args.quantum_mode,
    )
    configure_torch_backend(profile)
    hardware = detect_worker_hardware(worker_uid=args.worker_id)
    resources = detect_worker_resources(profile)
    plan = build_runtime_plan(model_type, profile, resources)

    cfg = ModelConfig(in_channels=1, num_classes=10, quantum_measurement="pauli_z_linear")
    if model_type == "hybrid":
        cfg.q_device = plan.quantum_device
        cfg.q_device_fallbacks = plan.quantum_device_fallbacks
        cfg.q_diff_method = plan.quantum_diff_method

    device = torch.device(plan.torch_device)
    model = _build_model(model_type, cfg).to(device)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(model, cfg, model_type)

    loader, _, _ = get_train_val_loaders(stage_no=max(args.stage_from, 2), repeat_id=0, fold_id=0, batch_size=1, num_workers=0)
    img, label = next(iter(loader))
    img = img.to(device)
    label = label.to(device)
    batch = max(1, int(plan.micro_batch_size))
    xb = img.repeat(batch, 1, 1, 1)
    yb = label.repeat(batch)

    def train_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        if logits.shape != (batch, 10):
            raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise AssertionError("Non-finite logits detected.")
        loss = criterion(logits, yb)
        if not torch.isfinite(loss):
            raise AssertionError("Non-finite loss detected.")
        loss.backward()
        has_finite_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if not has_finite_grad:
            raise AssertionError("No finite gradient detected.")
        optimizer.step()
        return float(loss.detach().cpu())

    for _ in range(2):
        train_step()
    _sync(plan.torch_device)

    iters = max(1, int(args.benchmark_iters))
    start = time.perf_counter()
    loss = 0.0
    for _ in range(iters):
        loss = train_step()
    _sync(plan.torch_device)
    train_step_sec = (time.perf_counter() - start) / iters
    train_per_image_sec = train_step_sec / batch

    with torch.no_grad():
        for _ in range(2):
            logits = model(xb)
            if logits.shape != (batch, 10):
                raise AssertionError(f"Unexpected forward logits shape: {tuple(logits.shape)}")
    _sync(plan.torch_device)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(xb)
    _sync(plan.torch_device)
    forward_step_sec = (time.perf_counter() - start) / iters
    forward_per_image_sec = forward_step_sec / batch

    print("=" * 70)
    print("WORKER SMOKE + ESTIMATE REPORT")
    print("=" * 70)
    print(f"worker_id          : {args.worker_id}")
    print(f"model_type         : {model_type}")
    print(f"hostname           : {hardware.hostname}")
    print(f"worker_type        : {hardware.worker_type}")
    print(f"cpu_count          : {hardware.cpu_count}")
    print(f"ram_gb             : {hardware.ram_gb}")
    print(f"has_gpu            : {hardware.has_gpu}")
    print(f"gpu_name           : {hardware.gpu_name}")
    print(f"gpu_count          : {hardware.gpu_count}")
    print(f"gpu_vram_gb        : {hardware.gpu_vram_gb}")
    print(f"torch_device       : {plan.torch_device}")
    print(f"micro_batch        : {plan.micro_batch_size}")
    global_batch = getattr(plan, "global_batch_size", getattr(plan, "effective_batch_size", plan.micro_batch_size))
    print(f"global_batch       : {global_batch}")
    print(f"accum_steps        : {plan.gradient_accumulation_steps}")
    print(f"dataloader_workers : {plan.dataloader_workers}")
    print(f"quantum_device     : {plan.quantum_device}")
    print(f"quantum_diff       : {plan.quantum_diff_method}")
    print(f"pennylane          : {_package_version('pennylane')}")
    print(f"pennylane-lightning: {_package_version('pennylane-lightning')}")
    print(f"pl-lightning-gpu   : {_package_version('pennylane-lightning-gpu')}")
    backend_results = _probe_quantum_backends()
    for backend_name, ok, detail in backend_results:
        status = "OK" if ok else "FAIL"
        print(f"qml_backend {backend_name:15}: {status} - {detail}")
    if not any(name == "lightning.gpu" and ok for name, ok, _ in backend_results):
        print("lightning_gpu_hint : Linux/WSL worker usually needs cuQuantum/cuStateVec plus pennylane-lightning-gpu.")
    print(f"smoke_loss         : {loss:.6f}")
    print(f"train_step_sec     : {train_step_sec:.6f}")
    print(f"train_per_image_s  : {train_per_image_sec:.8f}")
    print(f"forward_per_image_s: {forward_per_image_sec:.8f}")
    print("-" * 70)

    total_sec = 0.0
    for stage_no in range(args.stage_from, args.stage_to + 1):
        if stage_no not in STAGE_CONFIGS:
            continue
        natural_train, augmented_train, val_count = _stage_counts(stage_no)
        stage_cfg = STAGE_CONFIGS[stage_no]
        units = stage_cfg["trials"] * stage_cfg["folds"] * stage_cfg["repeats"] * stage_cfg["max_epoch"]
        epoch_sec = augmented_train * train_per_image_sec + val_count * forward_per_image_sec
        stage_sec = epoch_sec * units
        total_sec += stage_sec
        print(
            f"stage {stage_no}: train_aug={augmented_train}, train_natural={natural_train}, "
            f"val={val_count}, units={units}, estimate_hours={stage_sec / 3600:.2f}"
        )

    print("-" * 70)
    print(f"total_estimate_hours: {total_sec / 3600:.2f}")
    print(f"total_estimate_days : {total_sec / 86400:.2f}")
    print("note: estimate excludes early stopping, queue delay, network, and checkpoint overhead.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
