"""
Model architecture settings for the spice classification experiment.

Scope:
- Shared CNN backbone configuration
- Classical fully spatial head configuration
- Hybrid QCQ-CNN quantum head configuration
- Training/runtime-independent architecture constants

This file intentionally does NOT contain training loops or XAI logic.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple


@dataclass(frozen=True)
class ImageInputConfig:
    image_size: int = 128
    in_channels: int = 1
    num_classes: int = 10

    @property
    def input_shape_chw(self) -> Tuple[int, int, int]:
        return (self.in_channels, self.image_size, self.image_size)


@dataclass(frozen=True)
class BackboneConfig:
    channels: Tuple[int, int, int] = (32, 64, 128)
    bottleneck_channels: int = 16
    group_norm_groups: int = 8
    dropout: float = 0.25
    use_cbam: bool = True
    use_blurpool: bool = True
    activation_fn: str = "leaky_relu"
    leaky_relu_negative_slope: float = 0.01


@dataclass(frozen=True)
class ClassicalHeadConfig:
    kernel_size: int = 1
    dropout: float = 0.10


@dataclass(frozen=True)
class QuantumHeadConfig:
    n_qubits: int = 8
    latent_dim: int = 256
    depth: int = 2
    ansatz: str = "StronglyEntanglingLayers"
    entanglement_pattern: str = "ring/circular"
    encoding: str = "AmplitudeEmbedding"
    measurement: str = "pauli_z_linear"
    init_scale: float = 0.01
    allow_surrogate_if_no_quantum: bool = True

    @property
    def amplitude_dim(self) -> int:
        return 2 ** self.n_qubits

    @property
    def n_quantum_parameters(self) -> int:
        # PennyLane StronglyEntanglingLayers weights shape: (n_layers, n_wires, 3)
        return self.depth * self.n_qubits * 3


@dataclass(frozen=True)
class OptimizerGroupConfig:
    lr_backbone: float = 5e-4
    lr_classical_head: float = 1e-3
    lr_projection_head: float = 1e-3
    lr_quantum: float = 5e-4
    weight_decay_backbone: float = 1e-4
    weight_decay_head: float = 1e-4
    weight_decay_quantum: float = 0.0


@dataclass(frozen=True)
class ArchitectureConfig:
    image: ImageInputConfig = ImageInputConfig()
    backbone: BackboneConfig = BackboneConfig()
    classical_head: ClassicalHeadConfig = ClassicalHeadConfig()
    quantum_head: QuantumHeadConfig = QuantumHeadConfig()
    optim_groups: OptimizerGroupConfig = OptimizerGroupConfig()

    def to_dict(self) -> Dict[str, object]:
        return {
            "image": asdict(self.image),
            "backbone": asdict(self.backbone),
            "classical_head": asdict(self.classical_head),
            "quantum_head": asdict(self.quantum_head),
            "optim_groups": asdict(self.optim_groups),
        }


def default_architecture_config() -> ArchitectureConfig:
    """Return the locked architecture configuration for the thesis experiment."""
    return ArchitectureConfig()
