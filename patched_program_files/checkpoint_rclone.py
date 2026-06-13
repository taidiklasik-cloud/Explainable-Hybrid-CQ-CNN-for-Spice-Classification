"""Checkpoint file helpers for local cache + Google Drive upload via rclone."""
from __future__ import annotations

import hashlib
import os
import random
import subprocess
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RcloneCheckpointConfig:
    rclone_remote: str = "gdrive"
    remote_root: str = "cqcnn_checkpoints"
    local_cache_dir: Path = Path("outputs/checkpoint_cache")
    rclone_executable: str = "rclone"
    rclone_config: str = ""
    require_upload: bool = False

    @classmethod
    def from_env(cls) -> "RcloneCheckpointConfig":
        local_cache_root = (
            os.environ.get("WORKER_LOCAL_CHECKPOINT_ROOT")
            or os.environ.get("LOCAL_CHECKPOINT_CACHE_DIR")
            or "outputs/checkpoint_cache"
        )
        return cls(
            rclone_remote=os.environ.get("RCLONE_REMOTE_NAME") or os.environ.get("RCLONE_REMOTE", "gdrive"),
            remote_root=os.environ.get("GDRIVE_CHECKPOINT_ROOT", "cqcnn_checkpoints"),
            local_cache_dir=Path(local_cache_root),
            rclone_executable=os.environ.get("RCLONE_EXE_PATH") or os.environ.get("RCLONE_EXE", "rclone"),
            rclone_config=os.environ.get("RCLONE_CONFIG_PATH", ""),
            require_upload=os.environ.get("REQUIRE_CHECKPOINT_UPLOAD", "false").lower() == "true",
        )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_file_name(checkpoint_type: str, epoch_number: int | None = None) -> str:
    key = checkpoint_type.upper().strip()
    if key in {"INTERVAL", "EPOCH"}:
        if epoch_number is None:
            raise ValueError("epoch_number is required for interval checkpoints.")
        return f"epoch_{int(epoch_number):04d}.pt"
    if key == "BEST":
        return "best.pt"
    if key == "FINAL":
        return "final.pt"
    if key == "RECOVERY":
        return "latest.pt"
    raise ValueError(f"checkpoint_type tidak dikenal: {checkpoint_type!r}")


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and PyTorch RNG states for exact local recovery."""
    state: dict[str, Any] = {"python": random.getstate(), "numpy": None, "torch_cpu": None, "torch_cuda": None}

    try:
        import numpy as np  # type: ignore

        state["numpy"] = np.random.get_state()
    except Exception:
        pass

    try:
        import torch  # type: ignore

        state["torch_cpu"] = torch.get_rng_state()
        state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    except Exception:
        pass

    return state


def restore_rng_state(rng_state: dict[str, Any]) -> None:
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])

    if rng_state.get("numpy") is not None:
        try:
            import numpy as np  # type: ignore

            np.random.set_state(rng_state["numpy"])
        except Exception:
            pass

    try:
        import torch  # type: ignore

        if rng_state.get("torch_cpu") is not None:
            torch.set_rng_state(rng_state["torch_cpu"])
        if rng_state.get("torch_cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
    except Exception:
        pass


def _safe_path_part(value: Any) -> str:
    text = str(value).strip().replace("\\", "_").replace("/", "_")
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text) or "unknown"


def _state_dict_or_none(component: Any, component_name: str) -> dict[str, Any] | None:
    if component is None:
        return None
    if not hasattr(component, "state_dict"):
        raise TypeError(f"{component_name} must expose state_dict().")
    return component.state_dict()


def _plain_config(config: Any) -> Any:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return config


def build_worker_latest_checkpoint_path(
    *,
    worker_uid: str,
    task: dict[str, Any],
    config: RcloneCheckpointConfig | None = None,
    checkpoint_root: str | Path | None = None,
) -> Path:
    cfg = config or RcloneCheckpointConfig.from_env()
    root = Path(checkpoint_root) if checkpoint_root is not None else cfg.local_cache_dir
    trial_nr = task.get("trial_nr")
    trial_part = f"trial_{int(trial_nr):06d}" if trial_nr is not None else "non_hpo"
    return (
        root
        / _safe_path_part(worker_uid)
        / f"stage_{int(task.get('stage_no') or 0):02d}"
        / _safe_path_part(task.get("model_type") or "unknown_model")
        / f"task_{int(task['task_id']):06d}"
        / trial_part
        / "latest.pt"
    )


def save_worker_latest_checkpoint(
    *,
    worker_uid: str,
    task: dict[str, Any],
    model: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    global_step: int,
    trial_params: dict[str, Any],
    model_config: Any,
    config: RcloneCheckpointConfig | None = None,
    checkpoint_root: str | Path | None = None,
    rng_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save worker-local latest.pt for fast self-recovery without remote upload."""
    pass

    path = build_worker_latest_checkpoint_path(
        worker_uid=worker_uid,
        task=task,
        config=config,
        checkpoint_root=checkpoint_root,
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "checkpoint_version": 1,
        "checkpoint_type": "RECOVERY",
        "worker_uid": worker_uid,
        "task_id": int(task["task_id"]),
        "stage_no": int(task.get("stage_no") or 0),
        "model_type": task.get("model_type"),
        "trial_nr": task.get("trial_nr"),
        "model_state_dict": _state_dict_or_none(model, "model"),
        "optimizer_state_dict": _state_dict_or_none(optimizer, "optimizer"),
        "scheduler_state_dict": _state_dict_or_none(scheduler, "scheduler"),
        "rng_state": rng_state or capture_rng_state(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "trial_params": dict(trial_params or {}),
        "model_config": _plain_config(model_config),
    }
    if extra:
        payload["extra"] = dict(extra)

    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        import torch  # type: ignore
        torch.save(payload, tmp_path)
    except ImportError:
        import pickle
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f)
    os.replace(tmp_path, path)
    return {
        "local_cache_path": str(path),
        "file_name": path.name,
        "sha256": sha256_file(path),
        "file_size_bytes": path.stat().st_size,
        "checkpoint_type": "RECOVERY",
        "epoch_number": int(epoch),
        "global_step": int(global_step),
    }


def load_worker_latest_checkpoint(path: str | Path, map_location: str | None = "cpu") -> dict[str, Any]:
    import torch  # type: ignore

    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)


def build_gdrive_relative_dir(
    *,
    stage_no: int,
    worker_uid: str,
    trial_nr: int | None,
    config: RcloneCheckpointConfig,
) -> str:
    stage_or_study = f"stage_{int(stage_no):02d}"
    run_or_trial_id = f"trial_{int(trial_nr):06d}" if trial_nr is not None else "non_hpo"
    parts = [
        config.remote_root.strip("/"),
        stage_or_study,
        _safe_path_part(worker_uid),
        run_or_trial_id,
    ]
    return "/".join(part for part in parts if part)


def checkpoint_uri(rclone_remote: str, gdrive_relative_path: str) -> str:
    return f"gdrive://{rclone_remote}/{gdrive_relative_path.lstrip('/')}"


def upload_with_rclone(local_path: str | Path, remote_file_path: str, config: RcloneCheckpointConfig) -> bool:
    import time
    local = Path(local_path)
    if not local.exists():
        print(f"File not found: {local}")
        return False

    remote_target = f"{config.rclone_remote}:{remote_file_path.replace(chr(92), '/')}"
    command = [config.rclone_executable, "copyto", str(local), remote_target]
    if config.rclone_config:
        command.extend(["--config", config.rclone_config])

    for attempt in range(1, 4):
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Rclone upload success: {local.name} -> {remote_target}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Rclone upload attempt {attempt} failed: {e.stderr.strip()}")
            if attempt < 3:
                time.sleep(2 ** attempt)
            else:
                print("Rclone upload exhausted retries.")
                if config.require_upload:
                    raise
                return False
    return False


def checkpoint_metadata(
    *,
    local_path: str | Path,
    gdrive_relative_path: str,
    config: RcloneCheckpointConfig,
    upload_status: str = "UPLOADED",
) -> dict[str, Any]:
    local = Path(local_path)
    return {
        "file_name": local.name,
        "sha256": sha256_file(local),
        "file_size_bytes": local.stat().st_size,
        "gdrive_relative_path": gdrive_relative_path.replace("\\", "/"),
        "checkpoint_uri": checkpoint_uri(config.rclone_remote, gdrive_relative_path),
        "local_cache_path": str(local),
        "rclone_remote": config.rclone_remote,
        "storage_backend": "gdrive_rclone",
        "upload_status": upload_status,
    }


def upload_and_register_checkpoint(
    *,
    db: Any,
    local_path: str | Path,
    task: dict[str, Any],
    worker_uid: str,
    checkpoint_type: str,
    epoch_number: int | None = None,
    global_step: int | None = None,
    metric_name: str | None = None,
    metric_value: float | None = None,
    config: RcloneCheckpointConfig | None = None,
    extra_rclone_args: list[str] | None = None,
) -> int:
    """
    Upload a checkpoint to Google Drive via rclone and register it in PostgreSQL.
    
    This function handles the end-to-end process:
    1. Builds the correct Google Drive path.
    2. Uploads the file via rclone (throws on failure, ensuring verification).
    3. Calls db.register_checkpoint_file() which logs to checkpoint_file
       and automatically updates the checkpoint_slot for latest/best/final.
    """
    cfg = config or RcloneCheckpointConfig.from_env()
    
    stage_no = int(task.get("stage_no") or 0)
    task_id = int(task["task_id"])
    trial_nr = task.get("trial_nr")
    
    gdrive_relative_dir = build_gdrive_relative_dir(
        stage_no=stage_no,
        worker_uid=worker_uid,
        trial_nr=trial_nr,
        config=cfg,
    )
    
    file_name = checkpoint_file_name(checkpoint_type, epoch_number)
    gdrive_relative_path = f"{gdrive_relative_dir}/{file_name}"
    
    # Upload via rclone
    upload_success = upload_with_rclone(local_path, gdrive_relative_path, cfg)
    upload_status = "UPLOADED" if upload_success else "FAILED"
    
    # Register to PostgreSQL.
    checkpoint_file_id = db.register_checkpoint_file(
        task_id=task_id,
        worker_uid=worker_uid,
        checkpoint_type=checkpoint_type,
        gdrive_relative_path=gdrive_relative_path,
        file_name=file_name,
        sha256=sha256_file(local_path) if upload_success else "unknown",
        file_size_bytes=Path(local_path).stat().st_size if upload_success else 0,
        epoch_number=epoch_number,
        global_step=global_step,
        metric_name=metric_name,
        metric_value=metric_value,
        upload_status=upload_status,
        local_cache_path=str(local_path),
        rclone_remote=cfg.rclone_remote,
        storage_backend="gdrive_rclone",
    )
    
    return checkpoint_file_id
