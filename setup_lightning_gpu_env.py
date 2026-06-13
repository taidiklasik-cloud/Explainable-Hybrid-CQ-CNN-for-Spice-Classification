"""Probe and optionally install PennyLane Lightning-GPU for this project.

Run this inside the target worker environment. The script prints a compact
readiness report and can call the project smoke benchmark without writing
artifacts.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import platform
import site
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GPU_REQUIREMENTS = ROOT / "requirements-lightning-gpu.txt"


def _run(cmd: list[str], env: dict[str, str] | None = None) -> int:
    print("+ " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, env=env, text=True)
    return int(completed.returncode)


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _is_wsl() -> bool:
    if platform.system().lower() != "linux":
        return False
    release = platform.release().lower()
    if "microsoft" in release or "wsl" in release:
        return True
    version_path = Path("/proc/version")
    if version_path.exists():
        return "microsoft" in version_path.read_text(errors="ignore").lower()
    return False


def _cuquantum_sdk_path() -> str | None:
    for site_dir in site.getsitepackages():
        candidate = Path(site_dir) / "cuquantum"
        if candidate.exists():
            return str(candidate)
    user_candidate = Path(site.getusersitepackages()) / "cuquantum"
    if user_candidate.exists():
        return str(user_candidate)
    return None


def _print_header(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70, flush=True)


def probe_environment() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    results.append(("python", True, sys.executable))
    results.append(("platform", True, platform.platform()))
    results.append(("is_wsl", True, str(_is_wsl())))

    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gib = props.total_memory / (1024**3)
            detail = f"torch={torch.__version__}, gpu={gpu_name}, vram={vram_gib:.2f} GiB"
            results.append(("torch_cuda", True, detail))
        else:
            results.append(("torch_cuda", False, f"torch={torch.__version__}, CUDA not available"))
    except Exception as exc:
        results.append(("torch_cuda", False, f"{type(exc).__name__}: {exc}"))

    for package_name in (
        "pennylane",
        "pennylane-lightning",
        "custatevec-cu12",
        "pennylane-lightning-gpu",
    ):
        version = _package_version(package_name)
        results.append((package_name, version != "NOT_INSTALLED", version))

    cuquantum_sdk = os.environ.get("CUQUANTUM_SDK") or _cuquantum_sdk_path()
    results.append(("CUQUANTUM_SDK", bool(cuquantum_sdk), cuquantum_sdk or "NOT_SET"))

    try:
        import pennylane as qml

        for device_name in ("lightning.gpu", "lightning.qubit", "default.qubit"):
            try:
                dev = qml.device(device_name, wires=2)
                results.append((f"qml_device:{device_name}", True, getattr(dev, "name", type(dev).__name__)))
            except Exception as exc:
                results.append((f"qml_device:{device_name}", False, f"{type(exc).__name__}: {exc}"))
    except Exception as exc:
        results.append(("pennylane_import", False, f"{type(exc).__name__}: {exc}"))

    return results


def install_lightning_gpu() -> int:
    if platform.system().lower() == "windows" and not _is_wsl():
        print("INSTALL_SKIPPED: Windows native terdeteksi.")
        print("lightning.gpu umumnya perlu Linux/WSL + cuQuantum/cuStateVec.")
        print("Gunakan skrip ini di WSL2 Ubuntu, Lovelace Linux, atau Google Cloud L4.")
        return 2

    if not GPU_REQUIREMENTS.exists():
        print(f"INSTALL_FAILED: {GPU_REQUIREMENTS} tidak ditemukan.")
        return 1

    code = _run([sys.executable, "-m", "pip", "install", "-U", "pip"])
    if code != 0:
        return code
    code = _run([sys.executable, "-m", "pip", "install", "-r", str(GPU_REQUIREMENTS)])
    if code != 0:
        return code

    cuquantum_sdk = _cuquantum_sdk_path()
    if cuquantum_sdk:
        print(f"CUQUANTUM_SDK_HINT={cuquantum_sdk}")
        print("Jika shell worker belum punya variabel ini, export/set CUQUANTUM_SDK ke path di atas.")
    else:
        print("CUQUANTUM_SDK_HINT=NOT_FOUND")
    return 0


def run_project_benchmark(stage: int, iters: int, worker_id: str) -> int:
    benchmark_script = ROOT / "tests" / "worker_smoke_estimate.py"
    if not benchmark_script.exists():
        print(f"BENCHMARK_SKIPPED: {benchmark_script} tidak ditemukan.")
        return 1

    env = os.environ.copy()
    cuquantum_sdk = env.get("CUQUANTUM_SDK") or _cuquantum_sdk_path()
    if cuquantum_sdk:
        env["CUQUANTUM_SDK"] = cuquantum_sdk

    return _run(
        [
            sys.executable,
            str(benchmark_script),
            "--model",
            "hybrid",
            "--torch-mode",
            "gpu",
            "--quantum-mode",
            "gpu",
            "--stage-from",
            str(stage),
            "--stage-to",
            str(stage),
            "--benchmark-iters",
            str(iters),
            "--worker-id",
            worker_id,
        ],
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup/probe PennyLane lightning.gpu for this project.")
    parser.add_argument("--install", action="store_true", help="Install optional Lightning-GPU packages on Linux/WSL.")
    parser.add_argument("--benchmark", action="store_true", help="Run project hybrid smoke benchmark after probing.")
    parser.add_argument("--stage", type=int, default=3, help="Stage number for worker_smoke_estimate.")
    parser.add_argument("--iters", type=int, default=1, help="Benchmark iterations for worker_smoke_estimate.")
    parser.add_argument("--worker-id", default="lightning_gpu_env_probe")
    args = parser.parse_args()

    _print_header("LIGHTNING GPU ENV PROBE")
    if args.install:
        install_code = install_lightning_gpu()
        print(f"install_status_code: {install_code}")

    results = probe_environment()
    for name, ok, detail in results:
        status = "OK" if ok else "FAIL"
        print(f"{name:32}: {status} | {detail}")

    has_lightning_gpu = any(name == "qml_device:lightning.gpu" and ok for name, ok, _ in results)
    if not has_lightning_gpu:
        print("lightning_gpu_status           : NOT_READY")
        print("recommended_next_step          : Jalankan di Linux/WSL atau worker cloud, lalu ulangi dengan --install --benchmark.")
    else:
        print("lightning_gpu_status           : READY")

    if args.benchmark:
        _print_header("PROJECT HYBRID BENCHMARK")
        return run_project_benchmark(args.stage, args.iters, args.worker_id)

    return 0 if has_lightning_gpu else 2


if __name__ == "__main__":
    raise SystemExit(main())
