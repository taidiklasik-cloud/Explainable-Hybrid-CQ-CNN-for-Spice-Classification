"""curriculum_dataset.py
PyTorch Dataset backed by curriculum CSV manifests.

Reads the per-stage train/validation split CSVs produced by the curriculum
pipeline and loads images as grayscale [1, 128, 128] tensors.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image
except ImportError:
    import importlib
    Image = importlib.import_module("PIL.Image")

SPICE_CLASSES: list[str] = [
    "adas",
    "bawang merah",
    "daun ketumbar",
    "jahe",
    "kayu manis",
    "kencur",
    "kunyit",
    "lengkuas",
    "pala",
    "serai",
]

CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(SPICE_CLASSES)}

IMAGE_SIZE = 128


def _default_dataset_root() -> Path:
    return Path(os.environ.get("DATASET_ROOT", "."))


def _default_curriculum_root() -> Path:
    return Path(os.environ.get("CURRICULUM_OUTPUTS_ROOT", "curriculum_outputs"))


def _base_transform(img: Image.Image) -> torch.Tensor:
    img = img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = torch.from_numpy(
        __import__("numpy").array(img, dtype="float32") / 255.0
    )
    return arr.unsqueeze(0)


class CurriculumImageDataset(Dataset):
    """Load images listed in a curriculum manifest CSV.

    Each row must have at least ``path`` (relative to *dataset_root*) and
    ``class`` (one of the 10 spice class names).
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        dataset_root: str | Path | None = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root) if dataset_root else _default_dataset_root()
        self.transform = transform or _base_transform

        df = pd.read_csv(manifest_csv)
        if "path" not in df.columns or "class" not in df.columns:
            raise ValueError(f"Manifest CSV harus memiliki kolom 'path' dan 'class'. Ditemukan: {list(df.columns)}")

        self.paths: list[str] = df["path"].tolist()
        self.labels: list[int] = [CLASS_TO_IDX[c] for c in df["class"]]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rel_path = self.paths[idx].replace("\\", os.sep).replace("/", os.sep)
        full_path = self.dataset_root / rel_path
        img = Image.open(full_path)
        tensor = self.transform(img)
        return tensor, self.labels[idx]


def get_train_val_loaders(
    stage_no: int,
    repeat_id: int = 0,
    fold_id: int = 0,
    batch_size: int = 16,
    num_workers: int = 0,
    dataset_root: Path | str | None = None,
    curriculum_root: Path | str | None = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build train and validation DataLoaders from curriculum manifests.

    For Stage 1 sanity we use the natural (original-only) train/val CSVs
    located at ``curriculum_outputs/stage_0X/train_validation_subsets/``.
    """
    ds_root = Path(dataset_root) if dataset_root else _default_dataset_root()
    cur_root = Path(curriculum_root) if curriculum_root else _default_curriculum_root()

    stage_dir = cur_root / f"stage_{int(stage_no):02d}" / "train_validation_subsets"
    train_csv = stage_dir / f"train_natural_stage_{int(stage_no):02d}_repeat_{int(repeat_id):02d}_fold_{int(fold_id):02d}.csv"
    val_csv = stage_dir / f"validation_natural_stage_{int(stage_no):02d}_repeat_{int(repeat_id):02d}_fold_{int(fold_id):02d}.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Train manifest tidak ditemukan: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation manifest tidak ditemukan: {val_csv}")

    train_ds = CurriculumImageDataset(train_csv, dataset_root=ds_root, transform=train_transform)
    val_ds = CurriculumImageDataset(val_csv, dataset_root=ds_root, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, list(SPICE_CLASSES)
