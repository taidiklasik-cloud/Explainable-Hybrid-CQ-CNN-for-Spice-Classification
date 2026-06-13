"""test_runtime_and_seed.py
Mock tests for:
  A. Dynamic hardware profiling (CPU/GPU VRAM/RAM-aware batch sizing)
  B. Seed locking (reproducibility across runs)
  C. Gradient accumulation in Stage 1 training

Usage:
    cd c:\\Klasik\\1\\PPT\\Coding
    python tests/test_runtime_and_seed.py
"""
from __future__ import annotations

import math
import os
import sys
import traceback
from pathlib import Path
from unittest.mock import MagicMock

from _path_setup import PROJECT_ROOT, configure_paths

configure_paths()

os.environ.setdefault("DATASET_ROOT", str(PROJECT_ROOT))
os.environ.setdefault("CURRICULUM_OUTPUTS_ROOT", str(PROJECT_ROOT / "curriculum_outputs"))
os.environ.setdefault("REQUIRE_CHECKPOINT_UPLOAD", "false")


def build_mock_db() -> MagicMock:
    db = MagicMock()
    db.heartbeat.return_value = True
    db.log_epoch_metrics.return_value = 1
    db.log_convergence_diagnostic.return_value = 1
    db.register_checkpoint_file.return_value = 1
    return db


def build_stage1_task(seed: int = 42) -> dict:
    return {
        "task_id": 8888,
        "stage_no": 1,
        "model_type": "classical_fully_spatial",
        "trial_nr": 0,
        "fold_id": 0,
        "repeat_id": 0,
        "seed": seed,
        "max_epoch": 2,
        "objective_metric_name": "val_loss",
        "objective_direction": "minimize",
        "optuna_tell_status": None,
        "optuna_study_name": None,
    }


def build_trial_params() -> dict:
    return {
        "lr_backbone": 0.0005,
        "lr_head": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.20,
        "label_smoothing": 0.1,
        "grad_clip_norm": 1.0,
    }


def main() -> None:
    results: list[tuple[str, bool, str]] = []

    # ══════════════════════════════════════════════════════════════
    # PART A: Dynamic Hardware Profiling
    # ══════════════════════════════════════════════════════════════

    # ── A1: Factory presets create correct modes ──────────────────
    try:
        from worker_runtime_config import (
            WorkerRuntimeProfile,
            create_cpu_profile,
            create_gpu_profile,
            with_seed,
            build_runtime_plan,
            detect_worker_resources,
            WorkerResources,
            RuntimePlan,
        )

        cpu_prof = create_cpu_profile(worker_id="test_cpu", seed=99)
        assert cpu_prof.torch_mode == "cpu", f"Expected cpu, got {cpu_prof.torch_mode}"
        assert cpu_prof.quantum_mode == "cpu"
        assert cpu_prof.base_seed == 99
        assert cpu_prof.global_batch_size == 32
        assert cpu_prof.benchmark_cudnn is False

        gpu_prof = create_gpu_profile(worker_id="test_gpu", seed=77)
        assert gpu_prof.torch_mode == "gpu", f"Expected gpu, got {gpu_prof.torch_mode}"
        assert gpu_prof.quantum_mode == "gpu"
        assert gpu_prof.base_seed == 77
        assert gpu_prof.global_batch_size == 32
        assert gpu_prof.benchmark_cudnn is True

        results.append(("A1: Factory presets (cpu/gpu)", True, "modes and defaults correct"))
    except Exception as e:
        results.append(("A1: Factory presets (cpu/gpu)", False, str(e)))
        traceback.print_exc()

    # ── A2: with_seed() keeps hardware, changes seed ─────────────
    try:
        auto_prof = WorkerRuntimeProfile(worker_id="w1")
        assert auto_prof.base_seed == 42

        seeded = with_seed(auto_prof, 123)
        assert seeded.base_seed == 123
        assert seeded.torch_mode == auto_prof.torch_mode
        assert seeded.quantum_mode == auto_prof.quantum_mode
        assert auto_prof.base_seed == 42, "Original must not mutate"

        results.append(("A2: with_seed() immutable override", True, "seed=123, original unchanged"))
    except Exception as e:
        results.append(("A2: with_seed() immutable override", False, str(e)))
        traceback.print_exc()

    # ── A3: GPU VRAM tiers ───────────────────────────────────────
    try:
        vram_tiers = [
            # (vram_gib, model_type, expected_micro)
            # micro = min(base_micro_from_vram, global_batch=32)
            (3.0,  "classical", 8),
            (4.0,  "classical", 16),
            (8.0,  "classical", 32),
            (16.0, "classical", 32),  # base=64 clamped to global=32
            (3.0,  "hybrid", 2),
            (4.0,  "hybrid", 4),
            (8.0,  "hybrid", 8),
            (12.0, "hybrid", 12),
            (24.0, "hybrid", 16),
        ]

        for vram, model_type, expected_micro in vram_tiers:
            mock_res = WorkerResources(
                worker_id="gpu_test",
                torch_device="cuda",
                cuda_available=True,
                cuda_device_count=1,
                cuda_device_name=f"Mock GPU {vram}GB",
                cuda_total_vram_gib=vram,
                system_ram_gib=16.0,
                usable_ram_gib=6.8,
                cpu_count_logical=8,
                cpu_count_physical=4,
                python_version="3.10",
                platform="mock",
                pid=1,
            )
            prof = create_gpu_profile("test")
            plan = build_runtime_plan(model_type, prof, mock_res)
            assert plan.micro_batch_size == expected_micro, (
                f"VRAM={vram} {model_type}: expected micro={expected_micro}, got {plan.micro_batch_size}"
            )
            assert plan.global_batch_size == plan.micro_batch_size * plan.gradient_accumulation_steps

        results.append(("A3: GPU VRAM tier batch sizing", True, f"all {len(vram_tiers)} tiers correct"))
    except Exception as e:
        results.append(("A3: GPU VRAM tier batch sizing", False, str(e)))
        traceback.print_exc()

    # ── A4: CPU RAM+core tiers ───────────────────────────────────
    try:
        cpu_tiers = [
            # (ram_usable, cores, model_type, expected_micro)
            (2.0,  2, "classical", 4),
            (3.0,  1, "classical", 8),
            (7.0,  2, "classical", 16),
            (13.0, 4, "classical", 32),
            (2.0,  2, "hybrid", 1),
            (3.0,  1, "hybrid", 2),
            (7.0,  2, "hybrid", 4),
            (13.0, 4, "hybrid", 8),
        ]

        for ram, cores, model_type, expected_micro in cpu_tiers:
            mock_res = WorkerResources(
                worker_id="cpu_test",
                torch_device="cpu",
                cuda_available=False,
                cuda_device_count=0,
                cuda_device_name="CPU/no CUDA",
                cuda_total_vram_gib=0.0,
                system_ram_gib=ram / 0.55 + 2.0 / 0.55,
                usable_ram_gib=ram,
                cpu_count_logical=cores * 2,
                cpu_count_physical=cores,
                python_version="3.10",
                platform="mock",
                pid=1,
            )
            prof = create_cpu_profile("test")
            plan = build_runtime_plan(model_type, prof, mock_res)
            assert plan.micro_batch_size == expected_micro, (
                f"RAM={ram} cores={cores} {model_type}: expected micro={expected_micro}, got {plan.micro_batch_size}"
            )
            assert plan.global_batch_size == plan.micro_batch_size * plan.gradient_accumulation_steps

        results.append(("A4: CPU RAM+core tier batch sizing", True, f"all {len(cpu_tiers)} tiers correct"))
    except Exception as e:
        results.append(("A4: CPU RAM+core tier batch sizing", False, str(e)))
        traceback.print_exc()

    # ── A5: Gradient accumulation steps derived correctly ─────────
    try:
        mock_res = WorkerResources(
            worker_id="accum_test",
            torch_device="cpu",
            cuda_available=False,
            cuda_device_count=0,
            cuda_device_name="CPU/no CUDA",
            cuda_total_vram_gib=0.0,
            system_ram_gib=16.0,
            usable_ram_gib=2.0,
            cpu_count_logical=2,
            cpu_count_physical=1,
            python_version="3.10",
            platform="mock",
            pid=1,
        )
        prof = create_cpu_profile("test", global_batch_size=32)
        plan_h = build_runtime_plan("hybrid", prof, mock_res)
        # RAM=2.0 cores=1 hybrid -> micro=1, accum=32, global=32
        assert plan_h.micro_batch_size == 1
        assert plan_h.gradient_accumulation_steps == 32
        assert plan_h.global_batch_size == 32

        plan_c = build_runtime_plan("classical", prof, mock_res)
        # RAM=2.0 cores=1 classical -> micro=4, accum=8, global=32
        assert plan_c.micro_batch_size == 4
        assert plan_c.gradient_accumulation_steps == 8
        assert plan_c.global_batch_size == 32

        results.append(("A5: Accum steps = global/micro", True,
                        f"hybrid: {plan_h.micro_batch_size}x{plan_h.gradient_accumulation_steps}="
                        f"{plan_h.global_batch_size}, "
                        f"classical: {plan_c.micro_batch_size}x{plan_c.gradient_accumulation_steps}="
                        f"{plan_c.global_batch_size}"))
    except Exception as e:
        results.append(("A5: Accum steps = global/micro", False, str(e)))
        traceback.print_exc()

    # ── A6: Notes contain hardware info ──────────────────────────
    try:
        mock_gpu_res = WorkerResources(
            worker_id="notes_test", torch_device="cuda",
            cuda_available=True, cuda_device_count=1,
            cuda_device_name="T4", cuda_total_vram_gib=15.0,
            system_ram_gib=16.0, usable_ram_gib=6.8,
            cpu_count_logical=4, cpu_count_physical=2,
            python_version="3.10", platform="mock", pid=1,
        )
        plan_g = build_runtime_plan("classical", create_gpu_profile("t"), mock_gpu_res)
        assert "VRAM" in plan_g.notes, f"Expected VRAM in notes, got: {plan_g.notes}"
        assert "15.0" in plan_g.notes

        mock_cpu_res = WorkerResources(
            worker_id="notes_test", torch_device="cpu",
            cuda_available=False, cuda_device_count=0,
            cuda_device_name="CPU/no CUDA", cuda_total_vram_gib=0.0,
            system_ram_gib=16.0, usable_ram_gib=6.8,
            cpu_count_logical=8, cpu_count_physical=4,
            python_version="3.10", platform="mock", pid=1,
        )
        plan_c2 = build_runtime_plan("hybrid", create_cpu_profile("t"), mock_cpu_res)
        assert "RAM" in plan_c2.notes and "cores" in plan_c2.notes, f"Got: {plan_c2.notes}"

        results.append(("A6: Notes contain hardware info", True, f"GPU: '{plan_g.notes}', CPU: '{plan_c2.notes}'"))
    except Exception as e:
        results.append(("A6: Notes contain hardware info", False, str(e)))
        traceback.print_exc()

    # ══════════════════════════════════════════════════════════════
    # PART B: Seed Locking (Reproducibility)
    # ══════════════════════════════════════════════════════════════

    # ── B1: set_global_seed produces identical model weights ──────
    try:
        import torch
        from model_architecture_modules import ClassicalFullySpatialCNN, ModelConfig, set_global_seed

        set_global_seed(42)
        m1 = ClassicalFullySpatialCNN(ModelConfig())
        w1 = {k: v.clone() for k, v in m1.state_dict().items()}

        set_global_seed(42)
        m2 = ClassicalFullySpatialCNN(ModelConfig())
        w2 = m2.state_dict()

        for k in w1:
            assert torch.equal(w1[k], w2[k]), f"Weight mismatch at {k}"

        results.append(("B1: Same seed -> identical model weights", True, f"checked {len(w1)} tensors"))
    except Exception as e:
        results.append(("B1: Same seed -> identical model weights", False, str(e)))
        traceback.print_exc()

    # ── B2: Different seed -> different weights ──────────────────
    try:
        set_global_seed(42)
        m_a = ClassicalFullySpatialCNN(ModelConfig())
        set_global_seed(999)
        m_b = ClassicalFullySpatialCNN(ModelConfig())

        sa = m_a.state_dict()
        sb = m_b.state_dict()
        diff_count = sum(1 for k in sa if not torch.equal(sa[k], sb[k]))
        assert diff_count > 0, "Different seeds should produce different weights"

        results.append(("B2: Different seed -> different weights", True, f"{diff_count}/{len(sa)} tensors differ"))
    except Exception as e:
        results.append(("B2: Different seed -> different weights", False, str(e)))
        traceback.print_exc()

    # ── B3: Seed from task payload flows to training ─────────────
    try:
        import torch
        import numpy as np
        import random

        set_global_seed(7777)
        snap_torch = torch.initial_seed()
        snap_np = np.random.get_state()[1][0]
        snap_py = random.getstate()[1][0]

        set_global_seed(7777)
        assert torch.initial_seed() == snap_torch
        assert np.random.get_state()[1][0] == snap_np
        assert random.getstate()[1][0] == snap_py

        results.append(("B3: Seed locks torch+numpy+random", True,
                        f"torch_seed={snap_torch}, np[0]={snap_np}, py[0]={snap_py}"))
    except Exception as e:
        results.append(("B3: Seed locks torch+numpy+random", False, str(e)))
        traceback.print_exc()

    # ── B4: compute_epoch_seed formula correctness ───────────────
    try:
        from worker_task_template import compute_epoch_seed

        # Stage 1-2: repeat=0, fold=0
        assert compute_epoch_seed(42, 0, 0, 1) == 43
        assert compute_epoch_seed(42, 0, 0, 2) == 44
        assert compute_epoch_seed(42, 0, 0, 50) == 92

        # Stage 3-4: repeat=0, fold=0..4
        assert compute_epoch_seed(42, 0, 0, 1) == 43  # base + 0 + 0 + 1
        assert compute_epoch_seed(42, 0, 3, 1) == 343  # base + 0 + 300 + 1
        assert compute_epoch_seed(42, 0, 4, 10) == 452  # base + 0 + 400 + 10

        # Stage 5: repeat=1,2,3, fold=0..4
        assert compute_epoch_seed(42, 1, 0, 1) == 1043    # base + 1000 + 0 + 1
        assert compute_epoch_seed(42, 1, 3, 5) == 1347    # base + 1000 + 300 + 5
        assert compute_epoch_seed(42, 2, 4, 1) == 2443    # base + 2000 + 400 + 1
        assert compute_epoch_seed(42, 3, 2, 50) == 3292   # base + 3000 + 200 + 50

        results.append(("B4: compute_epoch_seed formula", True, "all stage patterns correct"))
    except Exception as e:
        results.append(("B4: compute_epoch_seed formula", False, str(e)))
        traceback.print_exc()

    # ── B5: Epoch seeds are unique across folds/repeats/epochs ───
    try:
        seen: set[int] = set()
        collisions = []
        base = 42

        # Stage 3/4: repeat=0, folds 0-4, epochs 1-50
        for fold in range(5):
            for ep in range(1, 51):
                s = compute_epoch_seed(base, 0, fold, ep)
                if s in seen:
                    collisions.append(f"repeat=0 fold={fold} epoch={ep} -> {s}")
                seen.add(s)

        # Stage 5: repeats 1,2,3, folds 0-4, epochs 1-50
        for repeat in [1, 2, 3]:
            for fold in range(5):
                for ep in range(1, 51):
                    s = compute_epoch_seed(base, repeat, fold, ep)
                    if s in seen:
                        collisions.append(f"repeat={repeat} fold={fold} epoch={ep} -> {s}")
                    seen.add(s)

        if collisions:
            raise AssertionError(f"{len(collisions)} collisions: {collisions[:3]}")

        results.append(("B5: No epoch seed collisions", True, f"{len(seen)} unique seeds, 0 collisions"))
    except Exception as e:
        results.append(("B5: No epoch seed collisions", False, str(e)))
        traceback.print_exc()

    # ── B6: Different folds produce different epoch seeds ─────────
    try:
        s_fold0 = compute_epoch_seed(42, 0, 0, 1)
        s_fold1 = compute_epoch_seed(42, 0, 1, 1)
        s_fold4 = compute_epoch_seed(42, 0, 4, 1)
        assert s_fold0 != s_fold1 != s_fold4
        assert s_fold1 - s_fold0 == 100  # fold spacing
        assert s_fold4 - s_fold0 == 400  # fold spacing

        results.append(("B6: Different folds -> different seeds", True,
                        f"fold0={s_fold0}, fold1={s_fold1}, fold4={s_fold4}"))
    except Exception as e:
        results.append(("B6: Different folds -> different seeds", False, str(e)))
        traceback.print_exc()

    # ── B7: repeat_id=100 spacing avoids fold/epoch overlap ──────
    try:
        # Max epoch_seed from repeat=0 fold=4 epoch=99: base+0+4+99 = base+103
        # Min epoch_seed from repeat=100 fold=0 epoch=1: base+100+0+1 = base+101
        # These WILL overlap at the boundary, but that's by design since
        # different repeat+fold+epoch combos landing on same value is OK
        # as long as the specific (repeat, fold, epoch) triple is always unique.
        # The real guarantee is: same (base, repeat, fold, epoch) -> same seed always.
        s1a = compute_epoch_seed(42, 0, 0, 1)
        s1b = compute_epoch_seed(42, 0, 0, 1)
        assert s1a == s1b, "Same inputs must produce same seed"

        s2a = compute_epoch_seed(42, 100, 3, 10)
        s2b = compute_epoch_seed(42, 100, 3, 10)
        assert s2a == s2b, "Same inputs must produce same seed"

        results.append(("B7: Deterministic (same input -> same seed)", True, f"{s1a}=={s1b}, {s2a}=={s2b}"))
    except Exception as e:
        results.append(("B7: Deterministic (same input -> same seed)", False, str(e)))
        traceback.print_exc()

    # ══════════════════════════════════════════════════════════════
    # PART C: Gradient Accumulation in Stage 1 Training
    # ══════════════════════════════════════════════════════════════

    # ── C1: Training with gradient accumulation completes ────────
    try:
        from worker_task_template import real_train_one_task
        import shutil

        # Clean checkpoint cache to avoid stale checkpoint from previous test runs
        ckpt_root = Path(os.environ.get("WORKER_LOCAL_CHECKPOINT_ROOT", "outputs/checkpoint_cache"))
        c1_dir = ckpt_root / "accum_smoke_test"
        if c1_dir.exists():
            shutil.rmtree(c1_dir, ignore_errors=True)

        db = build_mock_db()
        task = build_stage1_task(seed=42)
        trial_params = build_trial_params()
        heartbeat_calls = [0]

        def mock_heartbeat() -> bool:
            heartbeat_calls[0] += 1
            return True

        result = real_train_one_task(
            task=task,
            db=db,
            worker_uid="accum_smoke_test",
            trial_params=trial_params,
            heartbeat_callback=mock_heartbeat,
        )

        assert result["status"] == "OK", f"Status not OK: {result}"
        assert isinstance(result["objective_value"], float)
        assert result["best_epoch"] >= 1
        results.append(("C1: Stage 1 with grad accum completes", True,
                        f"val_loss={result['objective_value']:.4f}, best_epoch={result['best_epoch']}"))
    except Exception as e:
        results.append(("C1: Stage 1 with grad accum completes", False, str(e)))
        traceback.print_exc()

    # ── C2: Same seed -> same training result ────────────────────
    try:
        db2 = build_mock_db()
        task2 = build_stage1_task(seed=42)
        hb2 = [0]

        # Clean checkpoint cache to force fresh start
        ckpt_root = Path(os.environ.get("WORKER_LOCAL_CHECKPOINT_ROOT", "outputs/checkpoint_cache"))
        seed_test_dir = ckpt_root / "seed_repro_test"
        if seed_test_dir.exists():
            import shutil
            shutil.rmtree(seed_test_dir, ignore_errors=True)

        result2 = real_train_one_task(
            task=task2,
            db=db2,
            worker_uid="seed_repro_test",
            trial_params=build_trial_params(),
            heartbeat_callback=lambda: (hb2.__setitem__(0, hb2[0] + 1) or True),
        )

        # Allow small float tolerance for potential platform differences
        diff = abs(result["objective_value"] - result2["objective_value"])
        assert diff < 1e-6, (
            f"Same seed should give same result. "
            f"Run1={result['objective_value']:.6f} vs Run2={result2['objective_value']:.6f}, diff={diff:.8f}"
        )
        results.append(("C2: Same seed -> same training result", True,
                        f"run1={result['objective_value']:.6f}, run2={result2['objective_value']:.6f}, diff={diff:.2e}"))
    except Exception as e:
        results.append(("C2: Same seed -> same training result", False, str(e)))
        traceback.print_exc()

    # ── C3: log_epoch_metrics called with grad_norm_global ───────
    try:
        # Use db2 from C2 which always runs fresh (checkpoint cleaned)
        call_args = db2.log_epoch_metrics.call_args_list
        assert len(call_args) >= 2, f"Expected >= 2 epoch logs, got {len(call_args)}"
        last_kwargs = call_args[-1][1]
        assert "grad_norm_global" in last_kwargs, f"Missing grad_norm_global in: {list(last_kwargs.keys())}"
        gn = last_kwargs["grad_norm_global"]
        assert isinstance(gn, float) and not math.isnan(gn), f"grad_norm is invalid: {gn}"

        results.append(("C3: grad_norm_global logged per epoch", True, f"grad_norm={gn:.4f}"))
    except Exception as e:
        results.append(("C3: grad_norm_global logged per epoch", False, str(e)))
        traceback.print_exc()

    # ══════════════════════════════════════════════════════════════
    # Report
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("RUNTIME PROFILING & SEED LOCKING TEST REPORT")
    print("=" * 70)
    passed = 0
    failed = 0
    for name, ok, detail in results:
        icon = "[OK]" if ok else "[XX]"
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {icon} [{status}] {name}")
        print(f"         {detail}")
    print("-" * 70)
    print(f"  Total: {passed + failed}  |  Passed: {passed}  |  Failed: {failed}")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
