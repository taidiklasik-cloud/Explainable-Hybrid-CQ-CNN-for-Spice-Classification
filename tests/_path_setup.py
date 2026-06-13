"""Shared path setup for repository smoke tests."""
from __future__ import annotations

import sys
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
PATCHED_PROGRAM_FILES = PROJECT_ROOT / "patched_program_files"
WORKER_RUNTIME = PROJECT_ROOT / "04_runtime_final" / "worker"
MODEL_ARCHITECTURE = PROJECT_ROOT / "03_model_architecture"


def configure_paths(*, include_worker_runtime: bool = True) -> None:
    paths = [PATCHED_PROGRAM_FILES]
    if include_worker_runtime:
        paths.append(WORKER_RUNTIME)
    paths.append(MODEL_ARCHITECTURE)
    paths.append(PROJECT_ROOT)

    for path in reversed(paths):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
