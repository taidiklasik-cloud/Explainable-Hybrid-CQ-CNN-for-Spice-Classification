"""Smoke test for Hybrid QCQ-CNN Pauli-Z linear readout.

Usage:
    cd c:\\Klasik\\1\\PPT\\Coding
    python tests/test_hybrid_pauli_z_linear_smoke.py
"""
from __future__ import annotations

import torch
import torch.nn as nn

from _path_setup import configure_paths

configure_paths()

from model_architecture_modules import HybridQCQCNN, ModelConfig, build_optimizer, count_parameters


def _first_trainable_parameter(module: nn.Module) -> torch.nn.Parameter:
    for parameter in module.parameters():
        if parameter.requires_grad:
            return parameter
    raise AssertionError(f"No trainable parameter found in {module.__class__.__name__}")


def _assert_grad_ok(name: str, parameter: torch.nn.Parameter) -> None:
    assert parameter.grad is not None, f"{name} grad is None"
    assert torch.isfinite(parameter.grad).all(), f"{name} grad contains non-finite values"


def main() -> None:
    torch.manual_seed(42)
    cfg = ModelConfig(
        n_qubits=8,
        latent_dim=256,
        q_depth=2,
        quantum_measurement="pauli_z_linear",
        q_device="default.qubit",
        q_device_fallbacks=("default.qubit",),
        q_diff_method="backprop",
    )
    model = HybridQCQCNN(cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg, "hybrid")

    x = torch.randn(2, 1, 128, 128)
    y = torch.tensor([0, 1])

    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    assert logits.shape == (2, 10), f"Expected logits shape (2,10), got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "Logits contain non-finite values"

    loss = criterion(logits, y)
    assert torch.isfinite(loss), "Loss is non-finite"
    loss.backward()

    backbone_parameter = _first_trainable_parameter(model.backbone)
    quantum_weights = model.quantum_head.qlayer.weights
    readout_weights = model.quantum_head.readout.weight
    _assert_grad_ok("CNN backbone parameter", backbone_parameter)
    _assert_grad_ok("quantum weights", quantum_weights)
    _assert_grad_ok("linear readout weights", readout_weights)

    optimizer.step()

    total_params = count_parameters(model)["total_params"]
    quantum_params = sum(p.numel() for p in model.quantum_head.quantum_parameters())
    readout_params = sum(p.numel() for p in model.quantum_head.readout_parameters())
    hybrid_head_params = quantum_params + readout_params

    assert quantum_params == 48, f"Expected 48 quantum params, got {quantum_params}"
    assert readout_params == 90, f"Expected 90 readout params, got {readout_params}"
    assert hybrid_head_params == 138, f"Expected 138 hybrid head params, got {hybrid_head_params}"

    print("Hybrid QCQ-CNN Pauli-Z linear smoke test")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss value: {loss.item():.6f}")
    print(f"total parameters: {total_params}")
    print(f"quantum params: {quantum_params}")
    print(f"linear readout params: {readout_params}")
    print(f"hybrid head params: {hybrid_head_params}")
    print("grad OK: backbone=True quantum=True linear_readout=True")


if __name__ == "__main__":
    main()
