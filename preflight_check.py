"""
preflight_check.py — Pengecekan kelengkapan environment sebelum worker mulai bekerja.

Skrip ini dirancang untuk dijalankan di awal notebook worker (Colab/Kaggle/lokal).
Ia akan:
  1. Mengecek setiap library yang dibutuhkan sudah bisa di-import
  2. Otomatis menginstal library yang belum ada (pip install)
  3. Memverifikasi dependensi silang (cross-dependency):
     - pennylane + pennylane-lightning -> lightning.qubit device
     - torch + CUDA -> GPU terdeteksi
     - optuna + psycopg2 -> PostgreSQL storage
     - sqlalchemy + psycopg -> orchestration DB
  4. Memuat dan memvalidasi file .env
  5. Menampilkan laporan lengkap

Usage:
    python preflight_check.py
    # atau di notebook:
    %run preflight_check.py
"""
from __future__ import annotations

import subprocess
import sys
import os
import platform
from pathlib import Path


# ==============================================================
# 1. DAFTAR LIBRARY -- module_name -> pip_name
#    Diurutkan berdasarkan prioritas dependensi.
# ==============================================================

# Library yang biasanya SUDAH ADA di platform cloud (Colab/Kaggle)
PLATFORM_LIBRARIES: dict[str, str] = {
    "torch":        "torch",
    "torchvision":  "torchvision",
    "numpy":        "numpy",
    "pandas":       "pandas",
    "scipy":        "scipy",
    "sklearn":      "scikit-learn",
    "skimage":      "scikit-image",
    "matplotlib":   "matplotlib",
    "seaborn":      "seaborn",
    "PIL":          "Pillow",
    "statsmodels":  "statsmodels",
    "joblib":       "joblib",
    "tqdm":         "tqdm",
    "numba":        "numba",
    "psutil":       "psutil",
    "cv2":          "opencv-python",
}

# Library yang HARUS diinstal tambahan di cloud
PROJECT_LIBRARIES: dict[str, str] = {
    "pennylane":            "pennylane",
    "pennylane_lightning":  "pennylane-lightning",
    "optuna":               "optuna",
    "psycopg":              "psycopg[binary]",
    "psycopg2":             "psycopg2-binary",
    "sqlalchemy":           "sqlalchemy",
    "dotenv":               "python-dotenv",
    "imagehash":            "imagehash",
}


def _pip_install(packages: list[str]) -> bool:
    if not packages:
        return True
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + packages,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def check_and_install_libraries() -> tuple[list[str], list[str], list[str]]:
    """Cek semua library, instal yang kurang. Return (ok, installed, failed)."""
    ok: list[str] = []
    to_install: list[str] = []
    to_install_names: list[str] = []

    all_libs = {**PLATFORM_LIBRARIES, **PROJECT_LIBRARIES}

    for module_name, pip_name in all_libs.items():
        try:
            __import__(module_name)
            ok.append(module_name)
        except ImportError:
            to_install.append(pip_name)
            to_install_names.append(module_name)

    if not to_install:
        return ok, [], []

    print(f"\n[INSTALL] Menginstal {len(to_install)} library yang belum ada: {to_install}")
    success = _pip_install(to_install)

    installed: list[str] = []
    failed: list[str] = []
    for module_name, pip_name in zip(to_install_names, to_install):
        try:
            __import__(module_name)
            installed.append(module_name)
        except ImportError:
            failed.append(f"{module_name} ({pip_name})")

    return ok, installed, failed


# ==============================================================
# 2. CROSS-DEPENDENCY VERIFICATION
# ==============================================================

def verify_cross_dependencies() -> list[tuple[str, bool, str]]:
    """Verifikasi dependensi silang antar library. Return list (name, ok, detail)."""
    results: list[tuple[str, bool, str]] = []

    # ── 2a: PyTorch + CUDA ────────────────────────────────────
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            results.append((
                "torch -> CUDA GPU",
                True,
                f"[OK] GPU terdeteksi: {gpu_name} ({vram:.1f} GiB VRAM)"
            ))
        else:
            results.append((
                "torch -> CUDA GPU",
                True,
                "[WARN] Tidak ada GPU NVIDIA. Worker akan berjalan di mode CPU."
            ))
    except Exception as e:
        results.append(("torch -> CUDA GPU", False, f"[FAIL] {e}"))

    try:
        import importlib.metadata as metadata
        import pennylane as qml

        versions = {
            "pennylane": metadata.version("pennylane"),
            "pennylane-lightning": metadata.version("pennylane-lightning"),
        }
        try:
            versions["pennylane-lightning-gpu"] = metadata.version("pennylane-lightning-gpu")
        except metadata.PackageNotFoundError:
            versions["pennylane-lightning-gpu"] = "NOT_INSTALLED"

        dev = qml.device("lightning.gpu", wires=2)
        results.append((
            "pennylane -> lightning.gpu",
            True,
            f"[OK] Device '{dev.name}' tersedia. Versions: {versions}"
        ))
    except Exception as e:
        results.append((
            "pennylane -> lightning.gpu",
            False,
            "[WARN] lightning.gpu belum tersedia. "
            f"Detail: {e}. Untuk Linux GPU worker, instal cuQuantum/cuStateVec dan pennylane-lightning-gpu yang kompatibel."
        ))

    # ── 2b: PennyLane → lightning.qubit device ────────────────
    try:
        import pennylane as qml
        dev = qml.device("lightning.qubit", wires=2)
        results.append((
            "pennylane -> lightning.qubit",
            True,
            f"[OK] Device '{dev.name}' berhasil dibuat (2 wires test)"
        ))
    except Exception as e:
        # Fallback: cek default.qubit
        try:
            import pennylane as qml
            dev = qml.device("default.qubit", wires=2)
            results.append((
                "pennylane -> lightning.qubit",
                False,
                f"[WARN] lightning.qubit gagal ({e}). Fallback ke default.qubit OK, tapi lebih lambat."
            ))
        except Exception as e2:
            results.append((
                "pennylane -> lightning.qubit",
                False,
                f"[FAIL] Baik lightning.qubit maupun default.qubit gagal: {e2}"
            ))

    # ── 2c: PennyLane → adjoint diff method ───────────────────
    try:
        import pennylane as qml
        import torch

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def test_circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = torch.tensor(0.5, requires_grad=True)
        out = test_circuit(x)
        out.backward()
        grad = float(x.grad)
        results.append((
            "pennylane -> adjoint differentiation",
            True,
            f"[OK] Adjoint diff bekerja. Test grad = {grad:.4f}"
        ))
    except Exception as e:
        results.append((
            "pennylane -> adjoint differentiation",
            False,
            f"[FAIL] Adjoint diff gagal: {e}. Hybrid model mungkin harus pakai backprop."
        ))

    # ── 2d: Optuna → PostgreSQL storage (psycopg2) ───────────
    try:
        import optuna
        from optuna.storages import RDBStorage
        import sqlalchemy

        # Hanya cek apakah class bisa diinstansiasi dengan URL format yang valid
        # Kita TIDAK benar-benar connect ke DB di sini
        results.append((
            "optuna -> sqlalchemy + psycopg2",
            True,
            f"[OK] Optuna {optuna.__version__}, SQLAlchemy {sqlalchemy.__version__} -- siap untuk RDB storage"
        ))
    except Exception as e:
        results.append(("optuna -> sqlalchemy + psycopg2", False, f"[FAIL] {e}"))

    # ── 2e: psycopg (v3) → bisa import ───────────────────────
    try:
        import psycopg
        results.append((
            "psycopg (v3) -> PostgreSQL driver",
            True,
            f"[OK] psycopg {psycopg.__version__} -- orchestration DB driver ready"
        ))
    except Exception as e:
        results.append(("psycopg (v3) -> PostgreSQL driver", False, f"[FAIL] {e}"))

    # ── 2f: python-dotenv → load .env ─────────────────────────
    try:
        from dotenv import load_dotenv, find_dotenv
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
            worker_uid = os.environ.get("WORKER_UID", "<not set>")
            db_dsn = os.environ.get("ORCHESTRATION_DB_DSN", "<not set>")
            db_host = db_dsn.split("@")[1].split(":")[0] if "@" in db_dsn else "?"
            results.append((
                "python-dotenv -> .env file",
                True,
                f"[OK] Loaded '{Path(env_path).name}'. WORKER_UID={worker_uid}, DB_HOST={db_host}"
            ))
        else:
            results.append((
                "python-dotenv -> .env file",
                False,
                "[WARN] File .env tidak ditemukan. Upload .env atau mount dari Google Drive."
            ))
    except Exception as e:
        results.append(("python-dotenv -> .env file", False, f"[FAIL] {e}"))

    # ── 2g: scikit-learn metrics (dipakai di training loop) ───
    try:
        from sklearn.metrics import f1_score, balanced_accuracy_score
        f1 = f1_score([0, 1, 2], [0, 1, 1], average="macro", zero_division=0)
        results.append((
            "sklearn -> metrics (f1, balanced_acc)",
            True,
            f"[OK] Metrics OK. Test macro_f1 = {f1:.4f}"
        ))
    except Exception as e:
        results.append(("sklearn -> metrics (f1, balanced_acc)", False, f"[FAIL] {e}"))

    # ── 2h: torchvision transforms ────────────────────────────
    try:
        import torchvision.transforms as T
        t = T.Compose([T.Resize((128, 128)), T.ToTensor()])
        results.append((
            "torchvision -> transforms pipeline",
            True,
            "[OK] Resize(128x128) + ToTensor() pipeline ready"
        ))
    except Exception as e:
        results.append(("torchvision -> transforms pipeline", False, f"[FAIL] {e}"))

    return results


# ==============================================================
# 3. REPORT
# ==============================================================

def print_report(
    ok: list[str],
    installed: list[str],
    failed_imports: list[str],
    cross_results: list[tuple[str, bool, str]],
) -> bool:
    print("\n" + "=" * 70)
    print("  PREFLIGHT CHECK -- HYBRID QCQ-CNN WORKER ENVIRONMENT")
    print("=" * 70)

    # System info
    print(f"\n  [PC]     Platform  : {platform.platform()}")
    print(f"  [Py]     Python    : {sys.version.split()[0]} ({sys.executable})")
    try:
        import torch
        cuda_ver = torch.version.cuda or "CPU-only"
        print(f"  [Torch]  PyTorch   : {torch.__version__} (CUDA: {cuda_ver})")
    except ImportError:
        print("  [Torch]  PyTorch   : NOT INSTALLED")

    # Section A: Library availability
    total_libs = len(ok) + len(installed) + len(failed_imports)
    print(f"\n{'-' * 70}")
    print(f"  PART A: Library Availability ({len(ok) + len(installed)}/{total_libs})")
    print(f"{'-' * 70}")

    if ok:
        print(f"  [OK] Sudah tersedia ({len(ok)}): {', '.join(ok[:8])}")
        if len(ok) > 8:
            print(f"       ... dan {len(ok) - 8} lainnya")
    if installed:
        print(f"  [++] Baru diinstal ({len(installed)}): {', '.join(installed)}")
    if failed_imports:
        print(f"  [!!] GAGAL diinstal ({len(failed_imports)}):")
        for f in failed_imports:
            print(f"       - {f}")

    # Section B: Cross-dependency checks
    print(f"\n{'-' * 70}")
    print(f"  PART B: Cross-Dependency Verification")
    print(f"{'-' * 70}")

    all_cross_ok = True
    for name, ok_flag, detail in cross_results:
        icon = "[OK]" if ok_flag else "[!!]"
        print(f"  {icon} {name}")
        print(f"       {detail}")
        if not ok_flag:
            all_cross_ok = False

    # Summary
    total_pass = len(ok) + len(installed) + sum(1 for _, o, _ in cross_results if o)
    total_fail = len(failed_imports) + sum(1 for _, o, _ in cross_results if not o)

    print(f"\n{'=' * 70}")
    if total_fail == 0:
        print("  [PASS] PREFLIGHT CHECK PASSED -- Worker siap beroperasi!")
    else:
        print(f"  [WARN] PREFLIGHT CHECK: {total_fail} masalah ditemukan.")
        print("         Periksa item yang gagal sebelum memulai training.")
    print(f"  Total: {total_pass} OK, {total_fail} masalah")
    print("=" * 70)

    return total_fail == 0


# ==============================================================
# 4. MAIN
# ==============================================================

def run_preflight() -> bool:
    """Entry point. Returns True if all checks pass."""
    ok, installed, failed = check_and_install_libraries()
    cross_results = verify_cross_dependencies()
    return print_report(ok, installed, failed, cross_results)


if __name__ == "__main__":
    success = run_preflight()
    if not success:
        sys.exit(1)
