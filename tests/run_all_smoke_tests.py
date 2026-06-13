"""Run repository smoke tests in a fixed order.

Usage:
    cd c:\\Klasik\\1\\PPT\\Coding
    python tests/run_all_smoke_tests.py
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from _path_setup import PROJECT_ROOT


def run_script(script_path: Path) -> bool:
    display_name = str(script_path.relative_to(PROJECT_ROOT))
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {display_name}")
    print(f"{'=' * 60}")

    try:
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)
        print(f"\n[PASS] {display_name} completed.\n")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"\n[FAIL] {display_name} stopped with exit code {exc.returncode}.")
        print("Dummy database rows should still be cleaned by each test's cleanup block.")
        return False
    except FileNotFoundError:
        print(f"\n[FAIL] File not found: {display_name}")
        return False


def main() -> None:
    scripts_to_run = [
        PROJECT_ROOT / "preflight_check.py",
        PROJECT_ROOT / "tests" / "worker_connection_test.py",
        PROJECT_ROOT / "tests" / "test_heartbeat_smoke.py",
        PROJECT_ROOT / "tests" / "smoke_test_refactor.py",
        PROJECT_ROOT / "tests" / "test_stage1_real_db.py",
    ]

    print("=" * 64)
    print("MASTER SMOKE TEST RUNNER")
    print("=" * 64)
    print("Runs system checks and Stage 1 smoke validation in sequence.")
    print("Each test owns its database cleanup routine.")
    print("=" * 64)

    passed = 0
    failed_scripts: list[str] = []

    for script in scripts_to_run:
        if run_script(script):
            passed += 1
            continue
        failed_scripts.append(str(script.relative_to(PROJECT_ROOT)))
        print("\n[!] Stopping master runner after first failure.")
        break

    print(f"\n{'=' * 60}")
    print("MASTER TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total run : {passed + len(failed_scripts)} of {len(scripts_to_run)}")
    print(f"Passed    : {passed}")

    if failed_scripts:
        print(f"Failed    : {len(failed_scripts)} ({', '.join(failed_scripts)})")
        raise SystemExit(1)

    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
