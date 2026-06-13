"""worker_hardware_profile.py
Lightweight hardware profiling for CQ-CNN worker registration.

This utility intentionally avoids mandatory heavy dependencies. It uses psutil/torch
when available and degrades gracefully when they are missing.
"""
from __future__ import annotations

import os
import platform
import socket
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class WorkerHardwareProfile:
    worker_uid: str
    worker_name: str | None
    hostname: str
    worker_type: str
    cpu_name: str
    cpu_count: int
    ram_gb: float | None
    has_gpu: bool
    gpu_name: str | None
    gpu_count: int
    gpu_vram_gb: float | None
    python_version: str
    platform_name: str

    def to_orchestration_kwargs(self) -> dict[str, Any]:
        return asdict(self)


def _get_ram_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        return None


def _get_gpu_info() -> tuple[bool, str | None, int, float | None]:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0) if count else None
            vram = None
            try:
                props = torch.cuda.get_device_properties(0)
                vram = round(props.total_memory / (1024 ** 3), 2)
            except Exception:
                pass
            return True, name, count, vram
    except Exception:
        pass
    return False, None, 0, None


def detect_worker_hardware(
    *,
    worker_uid: str | None = None,
    worker_name: str | None = None,
    worker_type: str = "LOCAL_PC",
) -> WorkerHardwareProfile:
    hostname = socket.gethostname()
    uid = worker_uid or os.environ.get("CQ_WORKER_UID") or f"{hostname}_{os.getpid()}"
    cpu_name = platform.processor() or platform.machine() or "unknown_cpu"
    has_gpu, gpu_name, gpu_count, gpu_vram_gb = _get_gpu_info()

    return WorkerHardwareProfile(
        worker_uid=uid,
        worker_name=worker_name,
        hostname=hostname,
        worker_type=worker_type,
        cpu_name=cpu_name,
        cpu_count=os.cpu_count() or 1,
        ram_gb=_get_ram_gb(),
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_vram_gb=gpu_vram_gb,
        python_version=sys.version.split()[0],
        platform_name=platform.platform(),
    )
