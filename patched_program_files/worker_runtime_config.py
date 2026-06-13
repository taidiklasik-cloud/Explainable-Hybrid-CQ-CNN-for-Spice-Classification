"""
worker_runtime_config.py
Konfigurasi runtime terpisah untuk eksperimen model dan worker node.

Tujuan:
1. Memisahkan setelan GPU/CPU dari definisi arsitektur model.
2. Menentukan torch device secara aman: auto/gpu/cpu.
3. Menentukan backend PennyLane quantum secara aman: lightning.gpu, lightning.qubit, atau default.qubit.
4. Menyediakan konfigurasi yang dapat diekspor sebagai CSV untuk audit eksperimen.

File ini sengaja dibuat ringan agar bisa dipakai ulang oleh notebook arsitektur,
notebook training, dan worker node terdistribusi.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math
import os
import platform

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except Exception:
    pd = None
    _PANDAS_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    _PSUTIL_AVAILABLE = False

try:
    import pennylane as qml
    _PENNYLANE_AVAILABLE = True
except Exception:
    qml = None
    _PENNYLANE_AVAILABLE = False


@dataclass
class WorkerRuntimeProfile:
    """Profil runtime untuk satu worker eksperimen.

    torch_mode:
        - "auto": pakai CUDA jika tersedia, jika tidak CPU.
        - "gpu": paksa CUDA. Jika CUDA tidak tersedia, fallback ke CPU kecuali strict=True.
        - "cpu": paksa CPU.

    quantum_mode:
        - "auto": pakai lightning.gpu jika tersedia; jika tidak, default.qubit/backprop pada CUDA.
        - "gpu": target utama lightning.gpu; fallback default.qubit/backprop pada CUDA.
        - "cpu": target utama lightning.qubit.

    strict:
        Jika True, mode "gpu" akan error jika CUDA tidak tersedia.
        Untuk notebook eksplorasi sebaiknya False; untuk worker produksi bisa True.
    """

    worker_id: str = "local_worker"
    torch_mode: str = "auto"            # auto | gpu | cpu
    quantum_mode: str = "auto"          # auto | gpu | cpu
    strict: bool = False

    # PennyLane runtime preference
    q_gpu_device: str = "lightning.gpu"
    q_cpu_device: str = "lightning.qubit"
    q_safe_device: str = "default.qubit"
    q_diff_method: str = "adjoint"
    q_shots: Optional[int] = None
    q_gpu_batch_obs: bool = False
    q_gpu_mpi: bool = False
    q_device_preference_gpu: Tuple[str, ...] = ("lightning.gpu", "lightning.qubit", "default.qubit")
    q_device_preference_cpu: Tuple[str, ...] = ("lightning.qubit", "default.qubit")

    # Worker sizing preference
    global_batch_size: int = 32
    min_micro_batch_size: int = 1
    max_micro_batch_size: int = 64
    safety_vram_fraction: float = 0.70
    safety_ram_fraction: float = 0.55
    dataloader_worker_cap: int = 8
    reserve_ram_gib: float = 2.0
    dry_run_batch_probe: bool = False

    # Reproducibility and logging
    base_seed: int = 42
    deterministic_torch: bool = False
    benchmark_cudnn: bool = True


def with_seed(profile: WorkerRuntimeProfile, seed: int) -> WorkerRuntimeProfile:
    """Return a copy of profile with base_seed overridden.

    Use this when the seed comes from the task/trial payload rather than the
    default profile value.  Keeps seed configuration decoupled from hardware.
    """
    return replace(profile, base_seed=seed)


def create_cpu_profile(
    worker_id: str = "local_worker",
    seed: int = 42,
    global_batch_size: int = 32,
) -> WorkerRuntimeProfile:
    """Factory for CPU-only workers (laptop tanpa GPU, Colab CPU mode, dll)."""
    return WorkerRuntimeProfile(
        worker_id=worker_id,
        torch_mode="cpu",
        quantum_mode="cpu",
        strict=False,
        global_batch_size=global_batch_size,
        base_seed=seed,
        deterministic_torch=False,
        benchmark_cudnn=False,
    )


def create_gpu_profile(
    worker_id: str = "local_worker",
    seed: int = 42,
    global_batch_size: int = 32,
    strict: bool = False,
) -> WorkerRuntimeProfile:
    """Factory for GPU workers (T4, L4, A100, local NVIDIA card, dll)."""
    return WorkerRuntimeProfile(
        worker_id=worker_id,
        torch_mode="gpu",
        quantum_mode="gpu",
        strict=strict,
        global_batch_size=global_batch_size,
        base_seed=seed,
        deterministic_torch=False,
        benchmark_cudnn=True,
    )


def cuda_available() -> bool:
    return bool(_TORCH_AVAILABLE and torch.cuda.is_available())


def cuda_device_name() -> str:
    if not cuda_available():
        return "CPU/no CUDA"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "CUDA available, device name unavailable"


def _qml_device_available(device_name: str) -> bool:
    if not _PENNYLANE_AVAILABLE:
        return False
    try:
        qml.device(device_name, wires=1)
        return True
    except Exception:
        return False


def resolve_torch_device(profile: WorkerRuntimeProfile):
    mode = profile.torch_mode.lower().strip()
    if mode not in {"auto", "gpu", "cpu"}:
        raise ValueError("torch_mode harus 'auto', 'gpu', atau 'cpu'.")

    has_cuda = cuda_available()
    if mode == "cpu":
        return "cpu"
    if mode == "gpu":
        if has_cuda:
            return "cuda"
        if profile.strict:
            raise RuntimeError("torch_mode='gpu' tetapi CUDA tidak tersedia.")
        return "cpu"
    # auto
    return "cuda" if has_cuda else "cpu"


def resolve_quantum_settings(profile: WorkerRuntimeProfile) -> Dict[str, object]:
    mode = profile.quantum_mode.lower().strip()
    if mode not in {"auto", "gpu", "cpu"}:
        raise ValueError("quantum_mode harus 'auto', 'gpu', atau 'cpu'.")

    has_cuda = cuda_available()
    q_diff_method = profile.q_diff_method
    if mode == "gpu":
        if not has_cuda and profile.strict:
            raise RuntimeError("quantum_mode='gpu' tetapi CUDA tidak tersedia.")
        if has_cuda and not _qml_device_available(profile.q_gpu_device):
            preferences = (profile.q_safe_device, profile.q_cpu_device)
            q_diff_method = "backprop"
        else:
            preferences = profile.q_device_preference_gpu
    elif mode == "cpu":
        preferences = profile.q_device_preference_cpu
    else:
        if has_cuda and not _qml_device_available(profile.q_gpu_device):
            preferences = (profile.q_safe_device, profile.q_cpu_device)
            q_diff_method = "backprop"
        else:
            preferences = profile.q_device_preference_gpu if has_cuda else profile.q_device_preference_cpu

    q_device = preferences[0]
    fallbacks = tuple(preferences[1:])

    return {
        "q_device": q_device,
        "q_device_fallbacks": fallbacks,
        "q_diff_method": q_diff_method,
        "q_shots": profile.q_shots,
        "q_gpu_batch_obs": profile.q_gpu_batch_obs,
        "q_gpu_mpi": profile.q_gpu_mpi,
    }


def apply_runtime_to_model_config(model_cfg, profile: WorkerRuntimeProfile):
    """Mengembalikan ModelConfig baru yang sudah diberi setelan runtime quantum.

    Fungsi ini memakai dataclasses.replace agar konfigurasi awal tidak termutasi.
    """
    q = resolve_quantum_settings(profile)
    return replace(
        model_cfg,
        q_device=q["q_device"],
        q_device_fallbacks=q["q_device_fallbacks"],
        q_diff_method=q["q_diff_method"],
        q_shots=q["q_shots"],
        q_gpu_batch_obs=q["q_gpu_batch_obs"],
        q_gpu_mpi=q["q_gpu_mpi"],
    )


def configure_torch_backend(profile: WorkerRuntimeProfile) -> None:
    """Set preferensi deterministik PyTorch/CuDNN.

    Untuk reproducibility kuat, deterministic_torch=True. Namun ini bisa menurunkan speed.
    Untuk eksperimen QML berat, benchmark_cudnn=True sering lebih cepat bila input size konstan.
    """
    if not _TORCH_AVAILABLE:
        return
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = bool(profile.deterministic_torch)
        torch.backends.cudnn.benchmark = bool(profile.benchmark_cudnn and not profile.deterministic_torch)


def runtime_summary(profile: WorkerRuntimeProfile) -> Dict[str, object]:
    torch_device = resolve_torch_device(profile)
    q = resolve_quantum_settings(profile)
    return {
        "worker_id": profile.worker_id,
        "torch_mode_requested": profile.torch_mode,
        "torch_device_resolved": torch_device,
        "cuda_available": cuda_available(),
        "cuda_device_name": cuda_device_name(),
        "quantum_mode_requested": profile.quantum_mode,
        "q_device_resolved": q["q_device"],
        "q_device_fallbacks": ", ".join(q["q_device_fallbacks"]),
        "q_diff_method": q["q_diff_method"],
        "q_shots": q["q_shots"],
        "q_gpu_batch_obs": q["q_gpu_batch_obs"],
        "q_gpu_mpi": q["q_gpu_mpi"],
        "strict": profile.strict,
        "base_seed": profile.base_seed,
        "deterministic_torch": profile.deterministic_torch,
        "benchmark_cudnn": profile.benchmark_cudnn,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
    }


def runtime_summary_dataframe(profile: WorkerRuntimeProfile):
    if not _PANDAS_AVAILABLE:
        return runtime_summary(profile)
    items = runtime_summary(profile).items()
    return pd.DataFrame([{"item": k, "value": v} for k, v in items])


def save_runtime_summary(profile: WorkerRuntimeProfile, out_csv: str | Path) -> None:
    if not _PANDAS_AVAILABLE:
        Path(out_csv).write_text(str(runtime_summary(profile)), encoding="utf-8")
        return
    df = runtime_summary_dataframe(profile)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def get_torch_device_object(profile: WorkerRuntimeProfile):
    if not _TORCH_AVAILABLE:
        return "cpu"
    return torch.device(resolve_torch_device(profile))


@dataclass
class WorkerResources:
    worker_id: str
    torch_device: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str
    cuda_total_vram_gib: float
    system_ram_gib: float
    usable_ram_gib: float
    cpu_count_logical: int
    cpu_count_physical: int
    python_version: str
    platform: str
    pid: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class RuntimePlan:
    model_type: str
    torch_device: str
    micro_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int
    dataloader_workers: int
    quantum_device: str
    quantum_device_fallbacks: Tuple[str, ...]
    quantum_diff_method: str
    notes: str

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["quantum_device_fallbacks"] = ", ".join(self.quantum_device_fallbacks)
        return data


def _system_ram_gib() -> float:
    if _PSUTIL_AVAILABLE:
        return float(psutil.virtual_memory().total / (1024 ** 3))
    return 0.0


def _cuda_total_vram_gib() -> float:
    if not cuda_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(0)
        return float(props.total_memory / (1024 ** 3))
    except Exception:
        return 0.0


def detect_worker_resources(profile: WorkerRuntimeProfile) -> WorkerResources:
    """Collect a compact resource snapshot for architecture/runtime audit tables."""
    cpu_logical = os.cpu_count() or 1
    if _PSUTIL_AVAILABLE:
        cpu_physical = psutil.cpu_count(logical=False) or cpu_logical
    else:
        cpu_physical = cpu_logical
    total_ram = _system_ram_gib()
    usable_ram = max(total_ram * profile.safety_ram_fraction - profile.reserve_ram_gib, 0.0)
    return WorkerResources(
        worker_id=profile.worker_id,
        torch_device=resolve_torch_device(profile),
        cuda_available=cuda_available(),
        cuda_device_count=torch.cuda.device_count() if cuda_available() else 0,
        cuda_device_name=cuda_device_name(),
        cuda_total_vram_gib=round(_cuda_total_vram_gib(), 3),
        system_ram_gib=round(total_ram, 3),
        usable_ram_gib=round(usable_ram, 3),
        cpu_count_logical=int(cpu_logical),
        cpu_count_physical=int(cpu_physical),
        python_version=platform.python_version(),
        platform=platform.platform(),
        pid=os.getpid(),
    )


def _clamp_batch(value: int, profile: WorkerRuntimeProfile) -> int:
    return max(profile.min_micro_batch_size, min(profile.max_micro_batch_size, int(value)))



def normalize_model_type(model_type: str) -> str:
    """Normalize model aliases from SQL/task payload to runtime planner keys."""
    key = str(model_type).lower().strip()
    classical_aliases = {"classical", "classical_fully_spatial", "classical_cnn", "fully_spatial"}
    hybrid_aliases = {"hybrid", "hybrid_qcqcnn", "qcqcnn", "hybrid_qcqc_nn"}
    if key in classical_aliases:
        return "classical"
    if key in hybrid_aliases:
        return "hybrid"
    raise ValueError(f"model_type tidak dikenal: {model_type!r}")

def build_runtime_plan(model_type: str, profile: WorkerRuntimeProfile, resources: WorkerResources) -> RuntimePlan:
    """Build a hardware-aware runtime plan.

    GPU path: micro_batch sized by VRAM tier.
    CPU path: micro_batch sized by usable RAM and physical core count.
    Gradient accumulation steps = ceil(global_batch / micro_batch).
    """
    model_key = normalize_model_type(model_type)
    torch_device = resources.torch_device
    target = max(1, int(profile.global_batch_size))

    if torch_device.startswith("cuda"):
        vram = resources.cuda_total_vram_gib
        if model_key == "hybrid":
            if vram >= 24:
                base_micro = 16
            elif vram >= 12:
                base_micro = 12
            elif vram >= 8:
                base_micro = 8
            elif vram >= 4:
                base_micro = 4
            else:
                base_micro = 2
        else:
            if vram >= 16:
                base_micro = 64
            elif vram >= 8:
                base_micro = 32
            elif vram >= 4:
                base_micro = 16
            else:
                base_micro = 8
    else:
        ram = resources.usable_ram_gib
        cores = resources.cpu_count_physical
        if model_key == "hybrid":
            if ram >= 12 and cores >= 4:
                base_micro = 8
            elif ram >= 6 and cores >= 2:
                base_micro = 4
            elif ram >= 3:
                base_micro = 2
            else:
                base_micro = 1
        else:
            if ram >= 12 and cores >= 4:
                base_micro = 32
            elif ram >= 6 and cores >= 2:
                base_micro = 16
            elif ram >= 3:
                base_micro = 8
            else:
                base_micro = 4

    micro = _clamp_batch(min(base_micro, target), profile)
    accum = max(1, math.ceil(target / micro))
    global_bs = micro * accum
    workers = max(0, min(profile.dataloader_worker_cap, max(resources.cpu_count_logical // 2, 1)))

    q = resolve_quantum_settings(profile)
    if model_key != "hybrid":
        quantum_device = "not_used"
        fallbacks: Tuple[str, ...] = ()
        if torch_device.startswith("cuda"):
            notes = f"classical on GPU (VRAM {resources.cuda_total_vram_gib:.1f} GiB)"
        else:
            notes = f"classical on CPU (RAM {resources.usable_ram_gib:.1f} GiB, {resources.cpu_count_physical} cores)"
    else:
        quantum_device = str(q["q_device"])
        fallbacks = tuple(q["q_device_fallbacks"])
        if torch_device.startswith("cuda"):
            notes = f"hybrid on GPU (VRAM {resources.cuda_total_vram_gib:.1f} GiB) + {quantum_device}"
        else:
            notes = f"hybrid on CPU (RAM {resources.usable_ram_gib:.1f} GiB, {resources.cpu_count_physical} cores) + {quantum_device}"

    return RuntimePlan(
        model_type=model_key,
        torch_device=torch_device,
        micro_batch_size=micro,
        gradient_accumulation_steps=accum,
        global_batch_size=global_bs,
        dataloader_workers=workers,
        quantum_device=quantum_device,
        quantum_device_fallbacks=fallbacks,
        quantum_diff_method=str(q["q_diff_method"]),
        notes=notes,
    )


def export_runtime_audit(resources: WorkerResources, plans: List[RuntimePlan], out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    resources_data = resources.to_dict()
    plans_data = [p.to_dict() for p in plans]
    (out_path / "worker_resources.json").write_text(json.dumps(resources_data, indent=2), encoding="utf-8")
    (out_path / "runtime_plans.json").write_text(json.dumps(plans_data, indent=2), encoding="utf-8")
    if _PANDAS_AVAILABLE:
        pd.DataFrame([resources_data]).to_csv(out_path / "worker_resources.csv", index=False)
        pd.DataFrame(plans_data).to_csv(out_path / "runtime_plans.csv", index=False)


def print_runtime_summary(resources: WorkerResources, plans: List[RuntimePlan]) -> None:
    print(f"Worker: {resources.worker_id}")
    print(f"Torch device: {resources.torch_device}")
    print(f"CUDA: {resources.cuda_available} ({resources.cuda_device_name})")
    print(f"RAM GiB: total={resources.system_ram_gib}, usable_plan={resources.usable_ram_gib}")
    if resources.cuda_available:
        print(f"VRAM GiB: {resources.cuda_total_vram_gib}")
    print(f"CPU cores: physical={resources.cpu_count_physical}, logical={resources.cpu_count_logical}")
    for plan in plans:
        print(
            f"{plan.model_type}: micro_batch={plan.micro_batch_size}, "
            f"accum={plan.gradient_accumulation_steps}, global_batch={plan.global_batch_size}, "
            f"quantum_device={plan.quantum_device}"
        )
        print(f"  -> {plan.notes}")
