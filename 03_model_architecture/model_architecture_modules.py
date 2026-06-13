
"""
model_architecture_modules.py
Arsitektur apple-to-apple untuk:
1) CNN Klasik Fully Spatial
2) Hybrid QCQ-CNN dengan amplitude encoding 8 qubit

File ini dirancang untuk diimpor oleh notebook training.
Catatan: Hybrid quantum head membutuhkan PennyLane. Jika PennyLane belum terpasang,
model hybrid akan memberi error informatif saat dibuat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import os
import random
import sys

def _ensure_windows_conda_cuda_dll_dirs() -> None:
    if os.name != "nt":
        return
    for rel in ("bin", "Library/bin", "Library/mingw-w64/bin", "Library/usr/bin"):
        dll_dir = Path(sys.prefix) / rel
        if not dll_dir.exists():
            continue
        dll_path = str(dll_dir)
        if dll_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = dll_path + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(dll_path)
            except OSError:
                pass


_ensure_windows_conda_cuda_dll_dirs()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pennylane as qml
    _PENNYLANE_AVAILABLE = True
except Exception:
    qml = None
    _PENNYLANE_AVAILABLE = False


# ============================================================
# 1. Konfigurasi umum
# ============================================================

@dataclass
class ModelConfig:
    in_channels: int = 1
    num_classes: int = 10
    image_size: int = 128

    # Shared backbone
    channels: Tuple[int, int, int] = (32, 64, 128)
    bottleneck_channels: int = 16
    group_norm_groups: int = 8
    dropout: float = 0.20
    cbam_reduction: int = 8
    use_blurpool: bool = True
    activation_fn: str = "leaky_relu"
    leaky_relu_negative_slope: float = 0.01

    # Hybrid quantum head
    latent_dim: int = 256
    n_qubits: int = 8
    q_depth: int = 2
    q_init_scale: float = 1e-2
    quantum_measurement: str = "pauli_z_linear"
    q_device: str = "default.qubit"
    q_device_fallbacks: Tuple[str, ...] = ("lightning.qubit", "default.qubit")
    q_diff_method: str = "backprop"
    q_shots: Optional[int] = None
    q_gpu_batch_obs: bool = False
    q_gpu_mpi: bool = False

    # Optimization default
    lr_backbone: float = 5e-4
    lr_head: float = 1e-3
    lr_quantum: float = 5e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    grad_clip_norm: float = 1.0
    t0_warm_restart: int = 10
    t_mult: int = 2


def _nested_get(obj: Any, *names: str, default: Any = None) -> Any:
    current = obj
    for name in names:
        if current is None or not hasattr(current, name):
            return default
        current = getattr(current, name)
    return current


def coerce_model_config(cfg: Any = None, **overrides: Any) -> ModelConfig:
    """Accept the flat ModelConfig or the nested ArchitectureConfig notebook schema."""
    if cfg is None:
        model_cfg = ModelConfig()
    elif isinstance(cfg, ModelConfig):
        model_cfg = cfg
    elif hasattr(cfg, "image") and hasattr(cfg, "backbone") and hasattr(cfg, "quantum_head"):
        optim = getattr(cfg, "optim_groups", None)
        model_cfg = ModelConfig(
            in_channels=_nested_get(cfg, "image", "in_channels", default=1),
            num_classes=_nested_get(cfg, "image", "num_classes", default=10),
            image_size=_nested_get(cfg, "image", "image_size", default=128),
            channels=_nested_get(cfg, "backbone", "channels", default=(32, 64, 128)),
            bottleneck_channels=_nested_get(cfg, "backbone", "bottleneck_channels", default=16),
            group_norm_groups=_nested_get(cfg, "backbone", "group_norm_groups", default=8),
            dropout=_nested_get(cfg, "backbone", "dropout", default=0.20),
            use_blurpool=_nested_get(cfg, "backbone", "use_blurpool", default=True),
            activation_fn=_nested_get(cfg, "backbone", "activation_fn", default="leaky_relu"),
            leaky_relu_negative_slope=_nested_get(cfg, "backbone", "leaky_relu_negative_slope", default=0.01),
            latent_dim=_nested_get(cfg, "quantum_head", "latent_dim", default=256),
            n_qubits=_nested_get(cfg, "quantum_head", "n_qubits", default=8),
            q_depth=_nested_get(cfg, "quantum_head", "depth", default=2),
            q_init_scale=_nested_get(cfg, "quantum_head", "init_scale", default=1e-2),
            quantum_measurement=_nested_get(cfg, "quantum_head", "measurement", default="pauli_z_linear"),
            lr_backbone=getattr(optim, "lr_backbone", 5e-4),
            lr_head=getattr(optim, "lr_classical_head", getattr(optim, "lr_projection_head", 1e-3)),
            lr_quantum=getattr(optim, "lr_quantum", 5e-4),
            weight_decay=getattr(optim, "weight_decay_head", 1e-4),
        )
    else:
        raise TypeError("cfg harus ModelConfig atau ArchitectureConfig dari model_architecture_settings.py")

    values = dict(model_cfg.__dict__)
    for key, value in overrides.items():
        if value is not None and key in values:
            values[key] = value
    return ModelConfig(**values)


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_model_type(model_type: str) -> str:
    """Normalize model aliases from SQL/task payload to internal keys."""
    key = str(model_type).lower().strip()
    classical_aliases = {"classical", "classical_fully_spatial", "classical_cnn", "fully_spatial"}
    hybrid_aliases = {"hybrid", "hybrid_qcqcnn", "qcqcnn", "hybrid_qcqc_nn"}
    if key in classical_aliases:
        return "classical"
    if key in hybrid_aliases:
        return "hybrid"
    raise ValueError(f"model_type tidak dikenal: {model_type!r}")



def safe_group_count(num_channels: int, requested_groups: int) -> int:
    """Pilih jumlah group valid untuk GroupNorm."""
    g = min(requested_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return max(g, 1)


def make_activation(name: str, negative_slope: float = 0.01, inplace: bool = True) -> nn.Module:
    name = str(name).lower().strip()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    raise ValueError(f"activation_fn tidak didukung: {name!r}")


# ============================================================
# 2. Layer pendukung: BlurPool, CBAM, ConvBlock
# ============================================================

class BlurPool2d(nn.Module):
    """Anti-aliased downsampling sederhana.

    Digunakan untuk mengurangi sensitivitas terhadap pergeseran kecil.
    Implementasi ini menggunakan kernel binomial 3x3 dan depthwise convolution.
    """

    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        kernel = torch.tensor([1.0, 2.0, 1.0])
        filt = kernel[:, None] * kernel[None, :]
        filt = filt / filt.sum()
        self.register_buffer("filt", filt[None, None, :, :].repeat(channels, 1, 1, 1))
        self.channels = channels
        self.stride = stride
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(self.pad(x), self.filt, stride=self.stride, groups=self.channels)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, activation_fn: str = "leaky_relu", negative_slope: float = 0.01):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            make_activation(activation_fn, negative_slope),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention lalu spatial attention."""

    def __init__(self, channels: int, reduction: int = 8, activation_fn: str = "leaky_relu", negative_slope: float = 0.01):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction, activation_fn, negative_slope)
        self.spatial = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


class ConvBlock(nn.Module):
    """Conv2D + GroupNorm + activation + BlurPool/MaxPool + Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
        dropout: float = 0.2,
        use_blurpool: bool = True,
        activation_fn: str = "leaky_relu",
        negative_slope: float = 0.01,
    ):
        super().__init__()
        gn = safe_group_count(out_channels, groups)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(gn, out_channels)
        self.act = make_activation(activation_fn, negative_slope)
        self.down = BlurPool2d(out_channels, stride=2) if use_blurpool else nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.down(x)
        x = self.drop(x)
        return x


# ============================================================
# 3. Shared backbone
# ============================================================

class SharedSpiceBackbone(nn.Module):
    """Backbone bersama untuk model klasik dan hybrid.

    Output terakhir disimpan untuk kebutuhan Grad-CAM++.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c1, c2, c3 = cfg.channels
        act_kwargs = {"activation_fn": cfg.activation_fn, "negative_slope": cfg.leaky_relu_negative_slope}
        self.block1 = ConvBlock(cfg.in_channels, c1, cfg.group_norm_groups, cfg.dropout, cfg.use_blurpool, **act_kwargs)
        self.block2 = ConvBlock(c1, c2, cfg.group_norm_groups, cfg.dropout, cfg.use_blurpool, **act_kwargs)
        self.block3 = ConvBlock(c2, c3, cfg.group_norm_groups, cfg.dropout, cfg.use_blurpool, **act_kwargs)
        self.cbam = CBAM(c3, cfg.cbam_reduction, **act_kwargs)
        gn = safe_group_count(cfg.bottleneck_channels, cfg.group_norm_groups)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c3, cfg.bottleneck_channels, kernel_size=1, bias=False),
            nn.GroupNorm(gn, cfg.bottleneck_channels),
            make_activation(cfg.activation_fn, cfg.leaky_relu_negative_slope),
        )
        self.last_feature_map: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.cbam(x)
        x = self.bottleneck(x)
        self.last_feature_map = x
        return x


# ============================================================
# 4. Model klasik fully spatial
# ============================================================

class ClassicalFullySpatialCNN(nn.Module):
    """Baseline CNN klasik dengan head spasial.

    Head ini mempertahankan struktur spasial sehingga cocok untuk Grad-CAM++.
    """

    def __init__(self, cfg: Any = None):
        super().__init__()
        cfg = coerce_model_config(cfg)
        self.cfg = cfg
        self.backbone = SharedSpiceBackbone(cfg)
        hidden_channels = cfg.bottleneck_channels
        gn = safe_group_count(hidden_channels, cfg.group_norm_groups)
        self.classical_head = nn.Sequential(
            nn.Conv2d(cfg.bottleneck_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn, hidden_channels),
            make_activation(cfg.activation_fn, cfg.leaky_relu_negative_slope),
            nn.Conv2d(hidden_channels, cfg.num_classes, kernel_size=1, bias=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)
        logits_map = self.classical_head(fmap)
        logits = self.pool(logits_map).flatten(1)
        return logits

    def get_last_feature_map(self) -> Optional[torch.Tensor]:
        return self.backbone.last_feature_map


# ============================================================
# 5. Quantum head dan Hybrid QCQ-CNN
# ============================================================

class PennyLaneQuantumClassifier(nn.Module):
    """Quantum classifier berbasis PennyLane TorchLayer.

    Menggunakan Pauli-Z expectation readout dan linear classifier 10 kelas.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if not _PENNYLANE_AVAILABLE:
            raise ImportError(
                "PennyLane belum terpasang. Install dengan `pip install pennylane` "
                "untuk memakai Hybrid QCQ-CNN."
            )
        if cfg.latent_dim != 2 ** cfg.n_qubits:
            raise ValueError(f"latent_dim harus 2^n_qubits. Diterima latent_dim={cfg.latent_dim}, n_qubits={cfg.n_qubits}")

        measurement = cfg.quantum_measurement.lower().strip()
        if measurement != "pauli_z_linear":
            raise ValueError("quantum_measurement hanya mendukung 'pauli_z_linear'.")
        self.cfg = cfg
        self.measurement = measurement
        self.dev = self._make_device(cfg)
        weight_shapes = {"weights": (cfg.q_depth, cfg.n_qubits, 3)}
        init_method = {"weights": lambda w: torch.nn.init.normal_(w, mean=0.0, std=cfg.q_init_scale)}

        @qml.qnode(self.dev, interface="torch", diff_method=cfg.q_diff_method)
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(cfg.n_qubits), normalize=True, pad_with=0.0)
            qml.StronglyEntanglingLayers(weights, wires=range(cfg.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(cfg.n_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)
        self.readout = nn.Linear(cfg.n_qubits, cfg.num_classes)

    @staticmethod
    def _device_kwargs(device_name: str, cfg: ModelConfig) -> Dict[str, object]:
        kwargs: Dict[str, object] = {"wires": cfg.n_qubits}
        if cfg.q_shots is not None:
            kwargs["shots"] = cfg.q_shots
        if device_name == "lightning.gpu":
            if cfg.q_gpu_batch_obs:
                kwargs["batch_obs"] = True
            if cfg.q_gpu_mpi:
                kwargs["mpi"] = True
        return kwargs

    @classmethod
    def _make_device(cls, cfg: ModelConfig):
        errors: List[str] = []
        candidates = (cfg.q_device, *cfg.q_device_fallbacks)
        for device_name in dict.fromkeys(candidates):
            try:
                return qml.device(device_name, **cls._device_kwargs(device_name, cfg))
            except Exception as exc:
                errors.append(f"{device_name}: {exc}")
        raise RuntimeError(
            "Tidak ada device PennyLane yang bisa dibuat. Percobaan: "
            + " | ".join(errors)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_expval = self.qlayer(x)
        z_expval = z_expval.to(dtype=self.readout.weight.dtype, device=self.readout.weight.device)
        return self.readout(z_expval)

    def quantum_parameters(self) -> Iterable[nn.Parameter]:
        return self.qlayer.parameters()

    def readout_parameters(self) -> Iterable[nn.Parameter]:
        return self.readout.parameters()


class HybridQCQCNN(nn.Module):
    """Hybrid QCQ-CNN: CNN feature extractor + quantum classifier."""

    def __init__(
        self,
        cfg: Any = None,
        q_device: Optional[str] = None,
        q_diff_method: Optional[str] = None,
        q_device_fallbacks: Optional[Tuple[str, ...]] = None,
        q_shots: Optional[int] = None,
        q_gpu_batch_obs: Optional[bool] = None,
        q_gpu_mpi: Optional[bool] = None,
    ):
        super().__init__()
        cfg = coerce_model_config(
            cfg,
            q_device=q_device,
            q_diff_method=q_diff_method,
            q_device_fallbacks=q_device_fallbacks,
            q_shots=q_shots,
            q_gpu_batch_obs=q_gpu_batch_obs,
            q_gpu_mpi=q_gpu_mpi,
        )
        self.cfg = cfg
        self.backbone = SharedSpiceBackbone(cfg)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.LayerNorm(cfg.latent_dim, elementwise_affine=False),
        )
        self.quantum_head = PennyLaneQuantumClassifier(cfg)
        self.quantum_runtime = f"{self.quantum_head.dev.name}/{cfg.q_diff_method}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)
        z = self.projection(fmap)
        z = F.normalize(z, p=2, dim=1)
        logits = self.quantum_head(z)
        return logits

    def get_last_feature_map(self) -> Optional[torch.Tensor]:
        return self.backbone.last_feature_map


# ============================================================
# 6. Optimizer, scheduler, loss, regularization utilities
# ============================================================

def build_loss(cfg: ModelConfig) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)


def build_optimizer(model: nn.Module, cfg: ModelConfig, model_type: str) -> torch.optim.Optimizer:
    """Parameter-group LR untuk fairness.

    Classical:
        backbone -> lr_backbone
        classical_head -> lr_head
    Hybrid:
        backbone -> lr_backbone
        projection -> lr_head
        quantum weights -> lr_quantum
        linear readout -> lr_head
    """
    model_type = normalize_model_type(model_type)
    if model_type == "classical":
        groups = [
            {"params": model.backbone.parameters(), "lr": cfg.lr_backbone, "name": "backbone"},
            {"params": model.classical_head.parameters(), "lr": cfg.lr_head, "name": "classical_head"},
        ]
    elif model_type == "hybrid":
        groups = [
            {"params": model.backbone.parameters(), "lr": cfg.lr_backbone, "name": "backbone"},
            {"params": model.projection.parameters(), "lr": cfg.lr_head, "name": "projection_head"},
            {"params": model.quantum_head.quantum_parameters(), "lr": cfg.lr_quantum, "name": "quantum_weights"},
            {"params": model.quantum_head.readout_parameters(), "lr": cfg.lr_head, "name": "linear_readout"},
        ]
    else:
        raise ValueError("model_type harus 'classical' atau 'hybrid'.")

    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: ModelConfig):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.t0_warm_restart, T_mult=cfg.t_mult
    )


def clip_gradients(model: nn.Module, cfg: ModelConfig) -> float:
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm))


def add_small_gradient_noise(model: nn.Module, std: float = 0.0) -> None:
    """Perturbed gradient descent opsional untuk saddle point.

    Gunakan std kecil, misalnya 1e-5 atau 1e-6. Default 0.0 tidak mengubah gradien.
    """
    if std <= 0:
        return
    for p in model.parameters():
        if p.grad is not None:
            p.grad.add_(torch.randn_like(p.grad) * std)


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def freeze_backbone(model: nn.Module) -> None:
    set_module_trainable(model.backbone, False)


def unfreeze_backbone(model: nn.Module) -> None:
    set_module_trainable(model.backbone, True)


def freeze_quantum(model: nn.Module) -> None:
    if hasattr(model, "quantum_head"):
        set_module_trainable(model.quantum_head, False)


def unfreeze_quantum(model: nn.Module) -> None:
    if hasattr(model, "quantum_head"):
        set_module_trainable(model.quantum_head, True)


def get_gradient_norms_by_group(model: nn.Module) -> Dict[str, float]:
    groups: Dict[str, List[float]] = {"backbone": [], "head": [], "quantum": []}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        norm = float(p.grad.detach().norm().cpu())
        if "quantum_head.qlayer" in name or ".qlayer" in name:
            groups["quantum"].append(norm)
        elif "backbone" in name:
            groups["backbone"].append(norm)
        else:
            groups["head"].append(norm)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in groups.items()}


class EarlyStopping:
    """Early stopping yang mendukung objective minimize dan maximize.

    mode="min" cocok untuk val_loss.
    mode="max" cocok untuk val_macro_f1 / val_accuracy.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        mode = str(mode).lower().strip()
        if mode not in {"min", "max"}:
            raise ValueError("mode EarlyStopping harus 'min' atau 'max'.")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.count = 0
        self.should_stop = False

    def is_improvement(self, value: float) -> bool:
        value = float(value)
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float) -> bool:
        if self.is_improvement(value):
            self.best = float(value)
            self.count = 0
            self.should_stop = False
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True
        return self.should_stop


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "non_trainable_params": int(total - trainable),
    }


def parameter_table(model: nn.Module) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, p in model.named_parameters():
        rows.append(
            {
                "parameter_name": name,
                "shape": tuple(p.shape),
                "num_params": int(p.numel()),
                "trainable": bool(p.requires_grad),
                "dtype": str(p.dtype),
                "device": str(p.device),
            }
        )
    return rows


def _optim_values(cfg: Any) -> Dict[str, float]:
    if hasattr(cfg, "optim_groups"):
        optim = cfg.optim_groups
        return {
            "lr_backbone": getattr(optim, "lr_backbone", 5e-4),
            "lr_classical_head": getattr(optim, "lr_classical_head", 1e-3),
            "lr_projection_head": getattr(optim, "lr_projection_head", 1e-3),
            "lr_quantum": getattr(optim, "lr_quantum", 5e-4),
            "weight_decay_backbone": getattr(optim, "weight_decay_backbone", 1e-4),
            "weight_decay_head": getattr(optim, "weight_decay_head", 1e-4),
            "weight_decay_quantum": getattr(optim, "weight_decay_quantum", 0.0),
        }
    model_cfg = coerce_model_config(cfg)
    return {
        "lr_backbone": model_cfg.lr_backbone,
        "lr_classical_head": model_cfg.lr_head,
        "lr_projection_head": model_cfg.lr_head,
        "lr_quantum": model_cfg.lr_quantum,
        "weight_decay_backbone": model_cfg.weight_decay,
        "weight_decay_head": model_cfg.weight_decay,
        "weight_decay_quantum": model_cfg.weight_decay,
    }


def make_classical_parameter_groups(model: nn.Module, cfg: Any) -> List[Dict[str, object]]:
    optim = _optim_values(cfg)
    return [
        {
            "name": "backbone",
            "params": list(model.backbone.parameters()),
            "lr": optim["lr_backbone"],
            "weight_decay": optim["weight_decay_backbone"],
        },
        {
            "name": "classical_head",
            "params": list(model.classical_head.parameters()),
            "lr": optim["lr_classical_head"],
            "weight_decay": optim["weight_decay_head"],
        },
    ]


def make_hybrid_parameter_groups(model: nn.Module, cfg: Any) -> List[Dict[str, object]]:
    optim = _optim_values(cfg)
    return [
        {
            "name": "backbone",
            "params": list(model.backbone.parameters()),
            "lr": optim["lr_backbone"],
            "weight_decay": optim["weight_decay_backbone"],
        },
        {
            "name": "projection_head",
            "params": list(model.projection.parameters()),
            "lr": optim["lr_projection_head"],
            "weight_decay": optim["weight_decay_head"],
        },
        {
            "name": "quantum_weights",
            "params": list(model.quantum_head.quantum_parameters()),
            "lr": optim["lr_quantum"],
            "weight_decay": optim["weight_decay_quantum"],
        },
        {
            "name": "linear_readout",
            "params": list(model.quantum_head.readout_parameters()),
            "lr": optim["lr_projection_head"],
            "weight_decay": optim["weight_decay_head"],
        },
    ]


def parameter_group_summary(groups: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, group in enumerate(groups):
        params = list(group.get("params", []))
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        rows.append(
            {
                "group_index": idx,
                "group_name": group.get("name", f"group_{idx}"),
                "learning_rate": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
                "num_params": int(total),
                "trainable_params": int(trainable),
            }
        )
    return rows


def forward_layer_summary(
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (1, 128, 128),
    batch_size: int = 2,
    device: Optional[str] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    hooks = []

    def register_hook(name: str, module: nn.Module):
        def hook(module: nn.Module, inputs: Tuple[object, ...], output: object) -> None:
            if len(list(module.children())) > 0:
                return
            inp = inputs[0] if inputs else None
            in_shape = tuple(inp.shape) if hasattr(inp, "shape") else str(type(inp))
            if isinstance(output, (list, tuple)):
                out_shape = [tuple(o.shape) if hasattr(o, "shape") else str(type(o)) for o in output]
            else:
                out_shape = tuple(output.shape) if hasattr(output, "shape") else str(type(output))
            counts = count_parameters(module)
            rows.append(
                {
                    "layer_name": name,
                    "layer_type": module.__class__.__name__,
                    "input_shape": in_shape,
                    "output_shape": out_shape,
                    **counts,
                }
            )

        return hook

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(register_hook(name, module)))

    was_training = model.training
    model.eval()
    if device is None:
        try:
            torch_device = next(model.parameters()).device
        except StopIteration:
            torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(device)
        model = model.to(torch_device)
    x = torch.randn(batch_size, *input_shape, device=torch_device)
    try:
        with torch.no_grad():
            _ = model(x)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return rows


def quantum_circuit_detail_table(cfg: Any, selected_q_device: str = "default.qubit") -> List[Dict[str, object]]:
    model_cfg = coerce_model_config(cfg)
    trainable_quantum_params = model_cfg.q_depth * model_cfg.n_qubits * 3
    return [
        {"item": "encoding", "value": "AmplitudeEmbedding", "rationale": "maps normalized latent vector to amplitudes"},
        {"item": "n_qubits", "value": model_cfg.n_qubits, "rationale": "2^n amplitudes match latent_dim"},
        {"item": "latent_dim", "value": model_cfg.latent_dim, "rationale": "spatial collapse output before L2 normalization"},
        {"item": "ansatz", "value": "StronglyEntanglingLayers", "rationale": "ring-style variational entanglement"},
        {"item": "depth", "value": model_cfg.q_depth, "rationale": "number of variational layers"},
        {"item": "trainable_quantum_params", "value": trainable_quantum_params, "rationale": "PennyLane weight shape depth x wires x 3"},
        {"item": "measurement", "value": model_cfg.quantum_measurement, "rationale": "class readout strategy"},
        {"item": "selected_device", "value": selected_q_device, "rationale": "worker runtime plan"},
        {"item": "diff_method", "value": model_cfg.q_diff_method, "rationale": "quantum gradient method"},
    ]


def regularization_trainability_table() -> List[Dict[str, str]]:
    return [
        {
            "component": "shared_backbone",
            "risk": "overfitting on small image dataset",
            "mitigation": "dropout, GroupNorm, BlurPool, CBAM attention, AdamW weight decay",
        },
        {
            "component": "classical_head",
            "risk": "unfair parameter comparison",
            "mitigation": "separate optimizer group and spatial-only classifier head",
        },
        {
            "component": "projection_spatial_collapse",
            "risk": "unstable amplitude encoding scale",
            "mitigation": "parameter-free 4x4 spatial collapse, LayerNorm without affine parameters, and L2 normalization before quantum circuit",
        },
        {
            "component": "quantum_head",
            "risk": "barren plateau or noisy gradients",
            "mitigation": "shallow depth, small initialization, Pauli-Z expectation readout, runtime fallbacks",
        },
    ]


def _get_pyplot():
    import matplotlib
    import matplotlib.backends

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def make_architecture_block_diagram(out_png: str | Path) -> Path:
    plt = _get_pyplot()
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.axis("off")
    boxes = [
        ("Input\n1x128x128", (0.04, 0.70, 0.12, 0.13)),
        ("ConvBlock 1\n1->32", (0.21, 0.70, 0.13, 0.13)),
        ("ConvBlock 2\n32->64", (0.39, 0.70, 0.13, 0.13)),
        ("ConvBlock 3\n64->128", (0.57, 0.70, 0.13, 0.13)),
        ("CBAM + Bottleneck\n128->16", (0.75, 0.70, 0.18, 0.13)),
        ("Classical Head\nspatial conv + GAP", (0.20, 0.32, 0.25, 0.15)),
        ("Hybrid Spatial Collapse\n4x4 pool + norm", (0.55, 0.32, 0.27, 0.15)),
        ("Quantum Head\n8 qubits, depth 2", (0.55, 0.08, 0.27, 0.15)),
        ("10 classes", (0.86, 0.20, 0.10, 0.14)),
    ]
    for text, (x, y, w, h) in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.6))

    for x1, x2 in [(0.16, 0.21), (0.34, 0.39), (0.52, 0.57), (0.70, 0.75)]:
        arrow(x1, 0.765, x2, 0.765)
    arrow(0.84, 0.70, 0.33, 0.47)
    arrow(0.84, 0.70, 0.69, 0.47)
    arrow(0.45, 0.39, 0.86, 0.27)
    arrow(0.69, 0.32, 0.69, 0.23)
    arrow(0.82, 0.15, 0.86, 0.27)
    ax.set_title("Classical Fully Spatial CNN vs Hybrid QCQ-CNN", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _build_qiskit_qcq_circuit(
    n_qubits: int = 8,
    depth: int = 2,
    measurement: str = "pauli_z_linear",
):
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate, ParameterVector

    theta = ParameterVector("theta", length=depth * n_qubits * 3)
    qc = QuantumCircuit(n_qubits, name="Hybrid QCQ-CNN")
    qc.append(Gate(f"AmplitudeEncoding({2 ** n_qubits})", n_qubits, []), range(n_qubits))
    cursor = 0
    for layer_idx in range(depth):
        qc.barrier()
        for q in range(n_qubits):
            qc.ry(theta[cursor], q)
            qc.rz(theta[cursor + 1], q)
            qc.rx(theta[cursor + 2], q)
            cursor += 3
        for q in range(n_qubits):
            qc.cx(q, (q + 1) % n_qubits)
        qc.barrier(label=f"SEL layer {layer_idx + 1}")
    qc.append(Gate(f"{measurement} readout", n_qubits, []), range(n_qubits))
    return qc


def make_qiskit_quantum_circuit_image(
    out_png: str | Path,
    n_qubits: int = 8,
    depth: int = 2,
    measurement: str = "pauli_z_linear",
) -> Path:
    """Render the quantum circuit using Qiskit's native Matplotlib drawer."""
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    circuit = _build_qiskit_qcq_circuit(n_qubits=n_qubits, depth=depth, measurement=measurement)
    fig = circuit.draw(output="mpl", filename=str(out_path), fold=-1, idle_wires=False)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return out_path


def make_quantum_ring_circuit_fallback(
    out_png: str | Path,
    n_qubits: int = 8,
    depth: int = 2,
    prefer_qiskit: bool = True,
) -> Path:
    if prefer_qiskit:
        try:
            return make_qiskit_quantum_circuit_image(out_png, n_qubits=n_qubits, depth=depth)
        except Exception:
            pass

    plt = _get_pyplot()
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, max(4, n_qubits * 0.55)))
    ax.axis("off")
    y_positions = list(range(n_qubits))[::-1]
    for q, y in enumerate(y_positions):
        ax.hlines(y, 0.05, 0.95, color="black", linewidth=1)
        ax.text(0.01, y, f"q{q}", va="center", ha="left", family="monospace")
        ax.add_patch(plt.Rectangle((0.10, y - 0.16), 0.10, 0.32, fill=False, linewidth=1.4))
        ax.text(0.15, y, "Amp", ha="center", va="center", fontsize=8)
    for d in range(depth):
        x0 = 0.30 + d * 0.25
        for y in y_positions:
            ax.add_patch(plt.Rectangle((x0, y - 0.16), 0.11, 0.32, fill=False, linewidth=1.4))
            ax.text(x0 + 0.055, y, f"Rot {d+1}", ha="center", va="center", fontsize=8)
        for q in range(n_qubits):
            y1 = y_positions[q]
            y2 = y_positions[(q + 1) % n_qubits]
            ax.plot([x0 + 0.15, x0 + 0.15], [y1, y2], color="black", linewidth=1)
            ax.scatter([x0 + 0.15, x0 + 0.15], [y1, y2], s=18, color="black")
    ax.text(0.86, y_positions[0] + 0.5, "Pauli-Z expval\n+ Linear readout", ha="center", fontsize=9)
    ax.set_ylim(-1, n_qubits)
    ax.set_xlim(0, 1)
    ax.set_title("Hybrid QCQ-CNN Quantum Circuit Fallback Schematic", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_module_tree_text(model: nn.Module) -> str:
    lines: List[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        depth = name.count(".")
        params = sum(p.numel() for p in module.parameters())
        lines.append(f"{'  ' * depth}{name} ({module.__class__.__name__}) params={params:,}")
    return "\n".join(lines)


def make_text_image(text: str, out_png: str | Path, title: str = "") -> Path:
    plt = _get_pyplot()
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = text.splitlines() or [""]
    fig_h = max(4, 0.28 * len(lines) + 1.2)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.text(0.01, 0.98, text, ha="left", va="top", family="monospace", fontsize=8, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def model_summary_dict(model: nn.Module) -> Dict[str, int]:
    counts = count_parameters(model)
    return {"total_params": counts["total_params"], "trainable_params": counts["trainable_params"]}


def smoke_test_model(model: nn.Module, cfg: ModelConfig, batch_size: int = 2, device: str = "cpu") -> Tuple[torch.Size, Dict[str, int]]:
    model = model.to(device)
    x = torch.randn(batch_size, cfg.in_channels, cfg.image_size, cfg.image_size, device=device)
    with torch.no_grad():
        y = model(x)
    return y.shape, model_summary_dict(model)
