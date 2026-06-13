"""
checkpoint_archive_rclone.py

Utility for uploading and downloading checkpoint archives to/from Google Drive via rclone.
Includes functions for sha256 hashing and file size verification to ensure integrity.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Optional


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def rclone_check_available(rclone_exe: str = "rclone") -> bool:
    """Check if rclone executable is available."""
    try:
        subprocess.run([rclone_exe, "version"], check=True, capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def build_remote_dir(stage_or_study: str, worker_uid: str, run_or_trial_id: str) -> str:
    """Build remote directory path following deterministic convention."""
    return f"{stage_or_study}/{worker_uid}/{run_or_trial_id}"



def verify_file_size(path: str | Path, expected_size_bytes: int) -> bool:
    """Verify that a local file's size matches the expected size in bytes."""
    local_path = Path(path)
    if not local_path.is_file():
        return False
    return local_path.stat().st_size == expected_size_bytes


def upload_checkpoint_rclone(
    local_path: str | Path,
    remote_path: str,
    rclone_remote: str = "gdrive",
    rclone_exe: str = "rclone",
    extra_args: list[str] | None = None
) -> bool:
    """
    Upload a local checkpoint file to Google Drive via rclone.
    
    Args:
        local_path: Path to the local file.
        remote_path: Destination path on the remote (including remote directory structure).
        rclone_remote: Name of the rclone remote (default: "gdrive").
        rclone_exe: Path or command for rclone executable.
        extra_args: Additional arguments for rclone.
        
    Returns:
        True if successful.
    """
    local = Path(local_path)
    if not local.is_file():
        raise FileNotFoundError(f"Local file not found for upload: {local}")

    # Ensure remote_path uses forward slashes for cross-platform consistency
    remote_target = f"{rclone_remote}:{remote_path.replace(chr(92), '/')}"
    
    command = [rclone_exe, "copyto", str(local), remote_target]
    if extra_args:
        command.extend(extra_args)

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Rclone upload failed: {e.stderr}")
        raise


def download_checkpoint_rclone(
    remote_path: str,
    local_path: str | Path,
    rclone_remote: str = "gdrive",
    rclone_exe: str = "rclone",
    expected_size_bytes: Optional[int] = None,
    expected_sha256: Optional[str] = None,
    extra_args: list[str] | None = None
) -> bool:
    """
    Download a checkpoint file from Google Drive via rclone.
    Optionally verifies the file size and sha256 hash after download.
    
    Args:
        remote_path: Path on the remote.
        local_path: Destination path on the local filesystem.
        rclone_remote: Name of the rclone remote (default: "gdrive").
        rclone_exe: Path or command for rclone executable.
        expected_size_bytes: Expected file size in bytes to verify after download.
        expected_sha256: Expected SHA-256 hash to verify after download.
        extra_args: Additional arguments for rclone.
        
    Returns:
        True if download and all verifications succeed.
    """
    local = Path(local_path)
    
    # Ensure parent directories exist
    local.parent.mkdir(parents=True, exist_ok=True)
    
    remote_source = f"{rclone_remote}:{remote_path.replace(chr(92), '/')}"
    
    command = [rclone_exe, "copyto", remote_source, str(local)]
    if extra_args:
        command.extend(extra_args)

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Rclone download failed: {e.stderr}")
        raise

    if not local.is_file():
        raise FileNotFoundError(f"Downloaded file not found at expected location: {local}")

    # Verification steps
    if expected_size_bytes is not None:
        if not verify_file_size(local, expected_size_bytes):
            actual_size = local.stat().st_size
            raise ValueError(
                f"Size verification failed for {local}. "
                f"Expected: {expected_size_bytes}, got: {actual_size}"
            )
            
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(local)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"SHA-256 verification failed for {local}. "
                f"Expected: {expected_sha256}, got: {actual_sha256}"
            )

    return True

# Aliases requested by checklist
upload_with_rclone = upload_checkpoint_rclone
download_with_rclone = download_checkpoint_rclone
