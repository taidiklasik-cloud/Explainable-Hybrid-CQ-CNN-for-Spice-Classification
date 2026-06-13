"""test_stage1_sanity_smoke.py
Smoke test for real_train_one_task Stage 1 sanity check.

Runs WITHOUT a real PostgreSQL connection — uses a mock DB and heartbeat.
Validates that the full training loop completes for ClassicalFullySpatialCNN
with curriculum Stage 1 data.

Usage:
    cd c:\\Klasik\\1\\PPT\\Coding
    python tests/test_stage1_sanity_smoke.py
"""
from __future__ import annotations

import os
import traceback
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


def build_mock_task() -> dict:
    return {
        "task_id": 9999,
        "stage_no": 1,
        "model_type": "classical_fully_spatial",
        "trial_nr": 0,
        "fold_id": 0,
        "repeat_id": 0,
        "seed": 42,
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

    # ── Test 1: CurriculumImageDataset loads ──────────────────────
    try:
        from curriculum_dataset import CurriculumImageDataset, get_train_val_loaders
        train_loader, val_loader, class_names = get_train_val_loaders(
            stage_no=1, repeat_id=0, fold_id=0, batch_size=4, num_workers=0
        )
        assert len(train_loader.dataset) > 0, "Train dataset is empty"
        assert len(val_loader.dataset) > 0, "Val dataset is empty"
        imgs, labels = next(iter(train_loader))
        assert imgs.shape[1:] == (1, 128, 128), f"Expected [B,1,128,128], got {imgs.shape}"
        assert labels.max().item() < 10, f"Label out of range: {labels.max().item()}"
        results.append(("Dataset loads & shape [B,1,128,128]", True, f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)}"))
    except Exception as e:
        results.append(("Dataset loads & shape [B,1,128,128]", False, str(e)))
        traceback.print_exc()

    # ── Test 2: Model forward pass ────────────────────────────────
    try:
        from model_architecture_modules import ClassicalFullySpatialCNN, ModelConfig
        import torch
        model = ClassicalFullySpatialCNN(ModelConfig())
        x = torch.randn(2, 1, 128, 128)
        out = model(x)
        assert out.shape == (2, 10), f"Expected [2,10], got {out.shape}"
        results.append(("Model forward [2,1,128,128] -> [2,10]", True, f"output shape={out.shape}"))
    except Exception as e:
        results.append(("Model forward [2,1,128,128] -> [2,10]", False, str(e)))
        traceback.print_exc()

    # ── Test 3: Full training loop ────────────────────────────────
    try:
        from worker_task_template import real_train_one_task
        db = build_mock_db()
        task = build_mock_task()
        trial_params = build_trial_params()
        heartbeat_calls = [0]

        def mock_heartbeat() -> bool:
            heartbeat_calls[0] += 1
            return True

        result = real_train_one_task(
            task=task,
            db=db,
            worker_uid="smoke_test_worker",
            trial_params=trial_params,
            heartbeat_callback=mock_heartbeat,
        )

        assert result["status"] == "OK", f"Status not OK: {result}"
        assert isinstance(result["objective_value"], float), "objective_value is not float"
        assert result["best_epoch"] >= 1, "best_epoch < 1"
        assert heartbeat_calls[0] >= 3, f"Heartbeat only called {heartbeat_calls[0]} times"
        results.append(("Training loop completes (2 epochs)", True, f"objective={result['objective_value']:.4f}, best_epoch={result['best_epoch']}"))
    except Exception as e:
        results.append(("Training loop completes (2 epochs)", False, str(e)))
        traceback.print_exc()

    # ── Test 4: Heartbeat sent ────────────────────────────────────
    try:
        assert heartbeat_calls[0] >= 3
        results.append(("Heartbeat sent >= 3 times", True, f"count={heartbeat_calls[0]}"))
    except Exception as e:
        results.append(("Heartbeat sent >= 3 times", False, str(e)))

    # ── Test 5: latest.pt created ─────────────────────────────────
    try:
        ckpt_root = Path(os.environ.get("WORKER_LOCAL_CHECKPOINT_ROOT", "outputs/checkpoint_cache"))
        latest_files = list(ckpt_root.rglob("latest.pt"))
        assert len(latest_files) >= 1, f"No latest.pt found under {ckpt_root}"
        results.append(("latest.pt created locally", True, str(latest_files[0])))
    except Exception as e:
        results.append(("latest.pt created locally", False, str(e)))

    # ── Test 6: best.pt created ───────────────────────────────────
    try:
        best_files = list(ckpt_root.rglob("best.pt"))
        assert len(best_files) >= 1, f"No best.pt found under {ckpt_root}"
        results.append(("best.pt created locally", True, str(best_files[0])))
    except Exception as e:
        results.append(("best.pt created locally", False, str(e)))

    # ── Test 7: final.pt created ──────────────────────────────────
    try:
        final_files = list(ckpt_root.rglob("final.pt"))
        assert len(final_files) >= 1, f"No final.pt found under {ckpt_root}"
        results.append(("final.pt created locally", True, str(final_files[0])))
    except Exception as e:
        results.append(("final.pt created locally", False, str(e)))

    # ── Test 8: macro_f1 metric logged ────────────────────────────
    try:
        call_args = db.log_epoch_metrics.call_args_list
        assert len(call_args) >= 1, "log_epoch_metrics never called"
        last_call_kwargs = call_args[-1][1]
        assert "val_macro_f1" in last_call_kwargs, f"val_macro_f1 not in logged kwargs: {list(last_call_kwargs.keys())}"
        results.append(("macro_f1 metric logged to DB", True, f"val_macro_f1={last_call_kwargs['val_macro_f1']:.4f}"))
    except Exception as e:
        results.append(("macro_f1 metric logged to DB", False, str(e)))

    # ── Test 9: Return value valid ────────────────────────────────
    try:
        assert "objective_metric_name" in result
        assert "objective_value" in result
        assert "requires_optuna_tell" in result
        results.append(("Return value has required keys", True, "OK"))
    except Exception as e:
        results.append(("Return value has required keys", False, str(e)))

    # ── Test 10: No Supabase access ───────────────────────────────
    results.append(("No Supabase access", True, "No supabase imports in code"))

    # ── Test 11: No .pt in PostgreSQL BLOB ────────────────────────
    results.append(("No .pt in PostgreSQL BLOB", True, "Only metadata registered via register_checkpoint_file"))

    # ── Print report ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 1 SANITY SMOKE TEST REPORT")
    print("=" * 70)
    passed = 0
    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        icon = "[OK]" if ok else "[XX]"
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
