
"""
curriculum_stage_utils.py

Utility untuk membuat manifest data curriculum learning:
1) global stratified split 90/10;
2) stage subset 5%, 25%, 50%, 100%;
3) train/validation atau Stratified K-Fold;
4) adaptive unique balancing hanya pada training subset;
5) augmentation/upscale plan hanya pada training subset;
6) leakage guard berbasis source_image_id.

File ini sengaja tidak menjalankan training model.
Notebook training lain cukup import fungsi dari file ini.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import itertools
import json
import math
import hashlib

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, RepeatedStratifiedKFold


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class StageConfig:
    stage_id: int
    stage_name: str
    dev_fraction: float
    epoch: int
    cv_mode: str
    n_splits: int = 5
    n_repeats: int = 1
    val_size: float = 0.20
    final_augmented_per_class: int = 1000
    augmentation_fraction_of_final: float = 1.0
    max_unique_per_class: int = 210
    random_state: int = 42

    @property
    def augmented_target_per_class(self) -> int:
        return max(1, int(round(self.final_augmented_per_class * self.augmentation_fraction_of_final)))


DEFAULT_STAGE_CONFIGS: List[StageConfig] = [
    StageConfig(
        stage_id=1,
        stage_name="Sanity Test",
        dev_fraction=0.05,
        epoch=2,
        cv_mode="holdout",
        val_size=0.20,
        augmentation_fraction_of_final=0.05,
    ),
    StageConfig(
        stage_id=2,
        stage_name="Warm Start",
        dev_fraction=0.25,
        epoch=10,
        cv_mode="holdout",
        val_size=0.20,
        augmentation_fraction_of_final=0.25,
    ),
    StageConfig(
        stage_id=3,
        stage_name="Tuning for Convergence",
        dev_fraction=0.50,
        epoch=25,
        cv_mode="kfold",
        n_splits=5,
        n_repeats=1,
        augmentation_fraction_of_final=0.50,
    ),
    StageConfig(
        stage_id=4,
        stage_name="Tuning for Max Accuracy",
        dev_fraction=1.00,
        epoch=50,
        cv_mode="kfold",
        n_splits=5,
        n_repeats=1,
        augmentation_fraction_of_final=1.00,
    ),
    StageConfig(
        stage_id=5,
        stage_name="Final Repeated K-Fold Evaluation",
        dev_fraction=1.00,
        epoch=100,
        cv_mode="repeated_kfold",
        n_splits=5,
        n_repeats=5,
        augmentation_fraction_of_final=1.00,
    ),
]


def stable_id(value: str, length: int = 16) -> str:
    """Membuat id stabil dari string."""
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:length]


def normalize_manifest_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi kolom minimal: path, filename, class, source_image_id."""
    df = df.copy()

    # Cari kolom path
    if "path" not in df.columns:
        candidates = [c for c in df.columns if c.lower() in {"filepath", "file_path", "image_path", "img_path"}]
        if candidates:
            df = df.rename(columns={candidates[0]: "path"})
        else:
            raise ValueError("Manifest harus memiliki kolom 'path' atau kolom sepadan seperti image_path/file_path.")

    # Cari kolom class
    if "class" not in df.columns:
        candidates = [c for c in df.columns if c.lower() in {"label", "kelas", "category", "target"}]
        if candidates:
            df = df.rename(columns={candidates[0]: "class"})
        else:
            raise ValueError("Manifest harus memiliki kolom 'class' atau kolom sepadan seperti label/kelas.")

    df["path"] = df["path"].astype(str)
    df["class"] = df["class"].astype(str)

    if "filename" not in df.columns:
        df["filename"] = df["path"].map(lambda p: Path(p).name)

    # source_image_id penting untuk leakage guard.
    # Karena dataset sudah hasil deduplikasi final, default id bisa dari path/filename.
    if "source_image_id" not in df.columns:
        df["source_image_id"] = df["path"].map(lambda p: stable_id(Path(p).as_posix()))

    # Simpan original_path jika belum ada
    if "original_path" not in df.columns:
        df["original_path"] = df["path"]

    return df.reset_index(drop=True)


def load_manifest(manifest_path: Union[str, Path]) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")

    if manifest_path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(manifest_path)
    elif manifest_path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(manifest_path)
    else:
        raise ValueError("Format manifest belum didukung. Gunakan CSV atau Parquet.")

    return normalize_manifest_columns(df)


def save_manifest(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def class_distribution(df: pd.DataFrame, class_col: str = "class") -> pd.DataFrame:
    out = df[class_col].value_counts().sort_index().rename_axis(class_col).reset_index(name="count")
    out["percent"] = out["count"] / out["count"].sum()
    return out


def stratified_global_split(
    df: pd.DataFrame,
    dev_size: float = 0.90,
    random_state: int = 42,
    class_col: str = "class",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Membagi dataset final menjadi development dan holdout test secara stratified."""
    if not 0 < dev_size < 1:
        raise ValueError("dev_size harus antara 0 dan 1.")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=dev_size,
        random_state=random_state,
    )
    idx_dev, idx_test = next(splitter.split(df, df[class_col]))
    dev = df.iloc[idx_dev].copy().reset_index(drop=True)
    test = df.iloc[idx_test].copy().reset_index(drop=True)
    dev["split"] = "development"
    test["split"] = "holdout_test"
    return dev, test


def make_stage_subset(
    dev_df: pd.DataFrame,
    dev_fraction: float,
    stage_id: int,
    random_state: int = 42,
    class_col: str = "class",
) -> pd.DataFrame:
    """
    Mengambil subset development untuk stage tertentu secara stratified.
    Jika dev_fraction = 1, mengembalikan seluruh development set.
    """
    if not 0 < dev_fraction <= 1:
        raise ValueError("dev_fraction harus >0 dan <=1.")

    if dev_fraction == 1:
        stage_df = dev_df.copy().reset_index(drop=True)
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=dev_fraction,
            random_state=random_state + stage_id,
        )
        idx_stage, _ = next(splitter.split(dev_df, dev_df[class_col]))
        stage_df = dev_df.iloc[idx_stage].copy().reset_index(drop=True)

    stage_df["stage_id"] = stage_id
    stage_df["stage_fraction"] = dev_fraction
    return stage_df


def split_stage_holdout(
    stage_df: pd.DataFrame,
    val_size: float = 0.20,
    stage_id: int = 1,
    random_state: int = 42,
    class_col: str = "class",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split train/validation untuk stage kecil atau final holdout internal."""
    if len(stage_df[class_col].unique()) < 2:
        raise ValueError("Minimal harus ada dua kelas untuk stratified split.")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=random_state + stage_id,
    )
    idx_train, idx_val = next(splitter.split(stage_df, stage_df[class_col]))
    train_df = stage_df.iloc[idx_train].copy().reset_index(drop=True)
    val_df = stage_df.iloc[idx_val].copy().reset_index(drop=True)
    train_df["subset"] = "train"
    val_df["subset"] = "validation"
    train_df["fold_id"] = 0
    val_df["fold_id"] = 0
    return train_df, val_df


def make_stage_folds(
    stage_df: pd.DataFrame,
    stage_id: int,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    class_col: str = "class",
) -> pd.DataFrame:
    """
    Membuat assignment fold untuk K-Fold atau Repeated K-Fold.
    Output panjang: satu baris per sampel per repeat/fold dengan subset train/validation.
    """
    y = stage_df[class_col].to_numpy()
    records = []

    if n_repeats <= 1:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + stage_id)
        iterator = ((0, fold_idx, tr, va) for fold_idx, (tr, va) in enumerate(splitter.split(stage_df, y)))
    else:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state + stage_id,
        )
        iterator = []
        k = 0
        for tr, va in splitter.split(stage_df, y):
            repeat_id = k // n_splits
            fold_idx = k % n_splits
            iterator.append((repeat_id, fold_idx, tr, va))
            k += 1

    for repeat_id, fold_idx, tr_idx, va_idx in iterator:
        tr = stage_df.iloc[tr_idx].copy()
        va = stage_df.iloc[va_idx].copy()
        tr["subset"] = "train"
        va["subset"] = "validation"
        tr["repeat_id"] = repeat_id
        va["repeat_id"] = repeat_id
        tr["fold_id"] = fold_idx
        va["fold_id"] = fold_idx
        records.append(tr)
        records.append(va)

    out = pd.concat(records, ignore_index=True)
    return out


def validate_no_source_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    source_col: str = "source_image_id",
) -> bool:
    """Memastikan tidak ada source image yang sama di train dan validation."""
    tr = set(train_df[source_col].astype(str))
    va = set(val_df[source_col].astype(str))
    overlap = tr.intersection(va)
    if overlap:
        raise ValueError(f"Data leakage terdeteksi: {len(overlap)} source_image_id muncul di train dan validation.")
    return True


def _filter_repeat_fold(
    df: pd.DataFrame,
    fold_id: int = 0,
    repeat_id: int = 0,
) -> pd.DataFrame:
    """Filter dataframe split/manifest ke repeat dan fold tertentu jika kolomnya tersedia."""
    out = df.copy()
    if "repeat_id" in out.columns:
        out = out[out["repeat_id"].astype(int) == int(repeat_id)]
    if "fold_id" in out.columns:
        out = out[out["fold_id"].astype(int) == int(fold_id)]
    return out.reset_index(drop=True)


def export_train_validation_subsets(
    split_df: pd.DataFrame,
    stage_dir: Union[str, Path],
    stage_id: int,
    stage_name: str = "",
    cv_mode: str = "",
    source_col: str = "source_image_id",
    class_col: str = "class",
) -> Dict[str, Path]:
    """
    Menyimpan subset train/validation natural per stage-repeat-fold.

    File ini dipakai untuk lampiran/audit split sebelum balancing dan augmentasi.
    """
    stage_dir = Path(stage_dir)
    subset_dir = stage_dir / "train_validation_subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {"subset", "fold_id"}
    missing = required_cols.difference(split_df.columns)
    if missing:
        raise ValueError(f"split_df harus memiliki kolom: {sorted(missing)}")

    work_df = split_df.copy()
    if "repeat_id" not in work_df.columns:
        work_df["repeat_id"] = 0

    index_records = []
    distribution_records = []

    for (repeat_id, fold_id), g in work_df.groupby(["repeat_id", "fold_id"], sort=True):
        repeat_id = int(repeat_id)
        fold_id = int(fold_id)

        train_df = g[g["subset"] == "train"].copy().reset_index(drop=True)
        val_df = g[g["subset"] == "validation"].copy().reset_index(drop=True)
        validate_no_source_overlap(train_df, val_df, source_col=source_col)

        train_path = subset_dir / f"train_natural_stage_{stage_id:02d}_repeat_{repeat_id:02d}_fold_{fold_id:02d}.csv"
        val_path = subset_dir / f"validation_natural_stage_{stage_id:02d}_repeat_{repeat_id:02d}_fold_{fold_id:02d}.csv"
        save_manifest(train_df, train_path)
        save_manifest(val_df, val_path)

        train_counts = train_df[class_col].value_counts().sort_index()
        val_counts = val_df[class_col].value_counts().sort_index()
        for subset_name, counts in (("train", train_counts), ("validation", val_counts)):
            total = int(counts.sum())
            for cls, count in counts.items():
                distribution_records.append({
                    "stage_id": int(stage_id),
                    "stage_name": stage_name,
                    "cv_mode": cv_mode,
                    "repeat_id": repeat_id,
                    "fold_id": fold_id,
                    "subset": subset_name,
                    class_col: cls,
                    "count": int(count),
                    "percent": float(count / total) if total else 0.0,
                })

        index_records.append({
            "stage_id": int(stage_id),
            "stage_name": stage_name,
            "cv_mode": cv_mode,
            "repeat_id": repeat_id,
            "fold_id": fold_id,
            "n_train_natural": int(len(train_df)),
            "n_validation_natural": int(len(val_df)),
            "train_path": str(train_path),
            "validation_path": str(val_path),
            "train_class_count_json": json.dumps(train_counts.astype(int).to_dict(), ensure_ascii=False),
            "validation_class_count_json": json.dumps(val_counts.astype(int).to_dict(), ensure_ascii=False),
        })

    index_df = pd.DataFrame(index_records)
    distribution_df = pd.DataFrame(distribution_records)

    index_path = save_manifest(index_df, stage_dir / "train_validation_subset_index.csv")
    distribution_path = save_manifest(distribution_df, stage_dir / "train_validation_distribution_summary.csv")

    return {
        "train_validation_subset_index": index_path,
        "train_validation_distribution_summary": distribution_path,
        "train_validation_subsets_dir": subset_dir,
    }


def adaptive_unique_balance(
    train_df: pd.DataFrame,
    max_unique_per_class: int = 210,
    mode: str = "equal_unique_cap",
    random_state: int = 42,
    class_col: str = "class",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Balancing unik sebelum augmentasi, hanya untuk training subset.

    mode='equal_unique_cap':
        target_unique = min(max_unique_per_class, min jumlah kelas pada train subset)
        semua kelas di-downsample ke target_unique.
        Ini paling fair, tetapi bisa membuang data kelas besar.

    mode='cap_only':
        kelas > max_unique_per_class di-downsample ke max_unique_per_class,
        kelas <= max_unique_per_class dipakai seluruhnya.
        Ini memakai lebih banyak data, tetapi pre-augmentation tidak sepenuhnya seimbang.

    Return:
        balanced_train_df, metadata
    """
    train_df = train_df.copy()
    counts = train_df[class_col].value_counts().to_dict()
    if not counts:
        raise ValueError("train_df kosong.")

    rng = np.random.default_rng(random_state)

    if mode == "equal_unique_cap":
        target_unique = min(max_unique_per_class, min(counts.values()))
    elif mode == "cap_only":
        target_unique = max_unique_per_class
    else:
        raise ValueError("mode harus 'equal_unique_cap' atau 'cap_only'.")

    chunks = []
    actual_per_class = {}
    for cls, group in train_df.groupby(class_col):
        n = len(group)
        if mode == "equal_unique_cap":
            take_n = min(n, target_unique)
        else:
            take_n = min(n, max_unique_per_class)

        sampled = group.sample(n=take_n, random_state=random_state + stable_int(cls))
        chunks.append(sampled)
        actual_per_class[str(cls)] = take_n

    out = pd.concat(chunks, ignore_index=True)
    out["is_original"] = True
    out["is_augmented"] = False
    out["augmentation_id"] = ""
    out["augmentation_ops"] = ""
    out["base_source_image_id"] = out["source_image_id"].astype(str)

    metadata = {
        "mode": mode,
        "max_unique_per_class": int(max_unique_per_class),
        "target_unique_equal_cap": int(target_unique),
        "actual_per_class": actual_per_class,
    }
    return out.reset_index(drop=True), metadata


def stable_int(value: str, modulo: int = 10_000_000) -> int:
    return int(hashlib.sha1(str(value).encode("utf-8")).hexdigest(), 16) % modulo


DEFAULT_AUGMENTATION_GROUPS: Dict[str, List[str]] = {
    "geometric": [
        "rotate_small",
        "shift_small",
        "zoom_small",
        "horizontal_flip_safe",
    ],
    "photometric": [
        "brightness_small",
        "contrast_small",
        "gamma_small",
    ],
    "texture_noise": [
        "gaussian_noise_low",
        "blur_light",
        "sharpen_light",
    ],
    "spatial": [
        "random_crop_pad_small",
        "affine_small",
    ],
}


def generate_aug_op_pairs(
    augmentation_groups: Optional[Dict[str, List[str]]] = None,
    combo_size: int = 2,
) -> List[Tuple[str, ...]]:
    """
    Membuat kombinasi operasi augmentasi.
    Default: kombinasi dua operasi dari kelompok berbeda agar tidak terlalu agresif.
    """
    if augmentation_groups is None:
        augmentation_groups = DEFAULT_AUGMENTATION_GROUPS

    pairs = []
    group_names = list(augmentation_groups.keys())
    for g1, g2 in itertools.combinations(group_names, combo_size):
        for op1 in augmentation_groups[g1]:
            for op2 in augmentation_groups[g2]:
                pairs.append((op1, op2))
    return pairs


def make_augmentation_plan_for_train(
    balanced_train_df: pd.DataFrame,
    target_augmented_per_class: int,
    stage_id: int,
    fold_id: int = 0,
    repeat_id: int = 0,
    random_state: int = 42,
    class_col: str = "class",
    source_col: str = "source_image_id",
    augmentation_groups: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Membuat rencana augmentasi untuk training subset.

    Output:
    - train_original_plan: data original balanced.
    - aug_plan: baris tambahan untuk mencapai target_augmented_per_class per kelas.

    Catatan:
    Fungsi ini membuat manifest rencana augmentasi, belum menulis file gambar augmentasi.
    Eksekutor augmentasi bisa membaca kolom augmentation_ops.
    """
    base = balanced_train_df.copy().reset_index(drop=True)
    if "is_original" not in base.columns:
        base["is_original"] = True
    base["is_augmented"] = False
    base["augmentation_id"] = ""
    base["augmentation_ops"] = ""
    base["base_source_image_id"] = base[source_col].astype(str)
    base["stage_id"] = stage_id
    base["fold_id"] = fold_id
    base["repeat_id"] = repeat_id

    op_pairs = generate_aug_op_pairs(augmentation_groups=augmentation_groups, combo_size=2)
    rng = np.random.default_rng(random_state + stage_id * 1000 + fold_id * 100 + repeat_id)

    aug_records = []
    for cls, group in base.groupby(class_col):
        n_base = len(group)
        n_needed = max(0, int(target_augmented_per_class) - n_base)
        if n_needed == 0:
            continue

        # sampling base images with replacement untuk membuat rencana augmentasi.
        base_indices = rng.choice(group.index.to_numpy(), size=n_needed, replace=True)
        op_indices = rng.choice(len(op_pairs), size=n_needed, replace=True)

        for j, (base_idx, op_idx) in enumerate(zip(base_indices, op_indices), start=1):
            row = base.loc[base_idx].copy()
            ops = op_pairs[int(op_idx)]
            aug_id = f"S{stage_id}_R{repeat_id}_F{fold_id}_{cls}_{j:05d}_{stable_id(row[source_col] + '_' + '_'.join(ops), 8)}"
            row["is_original"] = False
            row["is_augmented"] = True
            row["augmentation_id"] = aug_id
            row["augmentation_ops"] = "+".join(ops)
            row["base_source_image_id"] = row[source_col]
            row["planned_output_path"] = str(Path("augmented") / f"stage_{stage_id}" / f"repeat_{repeat_id}" / f"fold_{fold_id}" / str(cls) / f"{aug_id}.png")
            aug_records.append(row)

    aug_plan = pd.DataFrame(aug_records)
    if not aug_plan.empty:
        aug_plan = aug_plan.reset_index(drop=True)

    return base, aug_plan


def build_stage_assets(
    dev_df: pd.DataFrame,
    config: StageConfig,
    out_dir: Union[str, Path],
    balance_mode: str = "equal_unique_cap",
    random_state: Optional[int] = None,
    class_col: str = "class",
) -> Dict[str, Path]:
    """
    Membuat semua manifest untuk satu stage:
    - stage subset
    - split/fold assignment
    - balanced train original
    - augmentation plan
    - combined train plan
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rs = config.random_state if random_state is None else random_state

    stage_dir = out_dir / f"stage_{config.stage_id:02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    stage_df = make_stage_subset(
        dev_df=dev_df,
        dev_fraction=config.dev_fraction,
        stage_id=config.stage_id,
        random_state=rs,
        class_col=class_col,
    )
    stage_path = save_manifest(stage_df, stage_dir / "stage_subset_manifest.csv")

    saved = {"stage_subset": stage_path}

    # Split/fold
    if config.cv_mode == "holdout":
        train_df, val_df = split_stage_holdout(
            stage_df,
            val_size=config.val_size,
            stage_id=config.stage_id,
            random_state=rs,
            class_col=class_col,
        )
        validate_no_source_overlap(train_df, val_df)
        split_df = pd.concat([train_df, val_df], ignore_index=True)
    elif config.cv_mode == "kfold":
        split_df = make_stage_folds(
            stage_df,
            stage_id=config.stage_id,
            n_splits=config.n_splits,
            n_repeats=1,
            random_state=rs,
            class_col=class_col,
        )
    elif config.cv_mode == "repeated_kfold":
        split_df = make_stage_folds(
            stage_df,
            stage_id=config.stage_id,
            n_splits=config.n_splits,
            n_repeats=config.n_repeats,
            random_state=rs,
            class_col=class_col,
        )
    else:
        raise ValueError("cv_mode harus holdout, kfold, atau repeated_kfold.")

    split_path = save_manifest(split_df, stage_dir / "stage_split_or_fold_assignment.csv")
    saved["split_or_fold_assignment"] = split_path
    saved.update(export_train_validation_subsets(
        split_df=split_df,
        stage_dir=stage_dir,
        stage_id=config.stage_id,
        stage_name=config.stage_name,
        cv_mode=config.cv_mode,
        class_col=class_col,
    ))

    # Build balance + aug per repeat/fold
    all_balanced = []
    all_aug = []
    all_combined = []
    meta_records = []

    group_cols = ["repeat_id", "fold_id"] if "repeat_id" in split_df.columns else ["fold_id"]
    for keys, g in split_df.groupby(group_cols):
        if isinstance(keys, tuple):
            repeat_id, fold_id = keys if len(keys) == 2 else (0, keys[0])
        else:
            repeat_id, fold_id = 0, keys

        tr = g[g["subset"] == "train"].copy().reset_index(drop=True)
        va = g[g["subset"] == "validation"].copy().reset_index(drop=True)

        if len(va):
            validate_no_source_overlap(tr, va)

        balanced, meta = adaptive_unique_balance(
            tr,
            max_unique_per_class=config.max_unique_per_class,
            mode=balance_mode,
            random_state=rs + config.stage_id * 1000 + int(fold_id),
            class_col=class_col,
        )
        balanced["stage_id"] = config.stage_id
        balanced["repeat_id"] = int(repeat_id)
        balanced["fold_id"] = int(fold_id)
        balanced["subset"] = "train"

        base, aug_plan = make_augmentation_plan_for_train(
            balanced,
            target_augmented_per_class=config.augmented_target_per_class,
            stage_id=config.stage_id,
            fold_id=int(fold_id),
            repeat_id=int(repeat_id),
            random_state=rs,
            class_col=class_col,
        )

        if not aug_plan.empty:
            aug_plan["subset"] = "train"
            combined = pd.concat([base, aug_plan], ignore_index=True)
        else:
            combined = base.copy()

        all_balanced.append(balanced)
        all_aug.append(aug_plan)
        all_combined.append(combined)

        meta_records.append({
            "stage_id": config.stage_id,
            "stage_name": config.stage_name,
            "repeat_id": int(repeat_id),
            "fold_id": int(fold_id),
            "cv_mode": config.cv_mode,
            "stage_dev_fraction": config.dev_fraction,
            "augmented_target_per_class": config.augmented_target_per_class,
            "augmentation_fraction_of_final": config.augmentation_fraction_of_final,
            "balance_mode": meta["mode"],
            "max_unique_per_class": meta["max_unique_per_class"],
            "target_unique_equal_cap": meta["target_unique_equal_cap"],
            "actual_per_class_json": json.dumps(meta["actual_per_class"], ensure_ascii=False),
            "n_train_unique_after_balance": len(balanced),
            "n_augmented_rows": len(aug_plan),
            "n_train_total_after_augmentation_plan": len(combined),
        })

    balanced_df = pd.concat(all_balanced, ignore_index=True) if all_balanced else pd.DataFrame()
    aug_df = pd.concat([x for x in all_aug if not x.empty], ignore_index=True) if any(not x.empty for x in all_aug) else pd.DataFrame()
    combined_df = pd.concat(all_combined, ignore_index=True) if all_combined else pd.DataFrame()
    meta_df = pd.DataFrame(meta_records)

    saved["balanced_train_original"] = save_manifest(balanced_df, stage_dir / "balanced_train_original_manifest.csv")
    saved["augmentation_plan"] = save_manifest(aug_df, stage_dir / "augmentation_plan.csv")
    saved["combined_train_plan"] = save_manifest(combined_df, stage_dir / "combined_train_plan_original_plus_augmented.csv")
    saved["stage_metadata"] = save_manifest(meta_df, stage_dir / "stage_metadata.csv")

    return saved


def build_all_stage_assets(
    dev_df: pd.DataFrame,
    out_dir: Union[str, Path],
    stage_configs: Optional[Sequence[StageConfig]] = None,
    balance_mode: str = "equal_unique_cap",
    random_state: int = 42,
    class_col: str = "class",
) -> Dict[int, Dict[str, Path]]:
    if stage_configs is None:
        stage_configs = DEFAULT_STAGE_CONFIGS

    all_saved = {}
    for cfg in stage_configs:
        saved = build_stage_assets(
            dev_df=dev_df,
            config=cfg,
            out_dir=out_dir,
            balance_mode=balance_mode,
            random_state=random_state,
            class_col=class_col,
        )
        all_saved[cfg.stage_id] = saved
    return all_saved


def summarize_stage_outputs(curriculum_dir: Union[str, Path]) -> pd.DataFrame:
    curriculum_dir = Path(curriculum_dir)
    rows = []
    for meta_path in sorted(curriculum_dir.glob("stage_*/stage_metadata.csv")):
        df = pd.read_csv(meta_path)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_train_validation_subset_indices(curriculum_dir: Union[str, Path]) -> pd.DataFrame:
    """Menggabungkan semua train_validation_subset_index.csv dari setiap stage."""
    curriculum_dir = Path(curriculum_dir)
    rows = []
    for index_path in sorted(curriculum_dir.glob("stage_*/train_validation_subset_index.csv")):
        df = pd.read_csv(index_path)
        rows.append(df)
    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    sort_cols = [c for c in ["stage_id", "repeat_id", "fold_id"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def get_stage_train_plan(
    curriculum_dir: Union[str, Path],
    stage_id: int,
    fold_id: int = 0,
    repeat_id: int = 0,
) -> pd.DataFrame:
    path = Path(curriculum_dir) / f"stage_{stage_id:02d}" / "combined_train_plan_original_plus_augmented.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return _filter_repeat_fold(df, fold_id=fold_id, repeat_id=repeat_id)


def get_stage_validation_manifest(
    curriculum_dir: Union[str, Path],
    stage_id: int,
    fold_id: int = 0,
    repeat_id: int = 0,
) -> pd.DataFrame:
    path = Path(curriculum_dir) / f"stage_{stage_id:02d}" / "stage_split_or_fold_assignment.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = _filter_repeat_fold(df, fold_id=fold_id, repeat_id=repeat_id)
    df = df[df["subset"] == "validation"]
    return df.reset_index(drop=True)


def get_stage_natural_train_manifest(
    curriculum_dir: Union[str, Path],
    stage_id: int,
    fold_id: int = 0,
    repeat_id: int = 0,
) -> pd.DataFrame:
    """Mengambil natural train subset sebelum balancing dan augmentasi."""
    curriculum_dir = Path(curriculum_dir)
    path = (
        curriculum_dir
        / f"stage_{stage_id:02d}"
        / "train_validation_subsets"
        / f"train_natural_stage_{stage_id:02d}_repeat_{repeat_id:02d}_fold_{fold_id:02d}.csv"
    )
    if path.exists():
        return pd.read_csv(path).reset_index(drop=True)

    split_path = curriculum_dir / f"stage_{stage_id:02d}" / "stage_split_or_fold_assignment.csv"
    if not split_path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(split_path)
    df = _filter_repeat_fold(df, fold_id=fold_id, repeat_id=repeat_id)
    return df[df["subset"] == "train"].reset_index(drop=True)


def get_stage_natural_validation_manifest(
    curriculum_dir: Union[str, Path],
    stage_id: int,
    fold_id: int = 0,
    repeat_id: int = 0,
) -> pd.DataFrame:
    """Mengambil natural validation subset sebelum balancing dan augmentasi."""
    curriculum_dir = Path(curriculum_dir)
    path = (
        curriculum_dir
        / f"stage_{stage_id:02d}"
        / "train_validation_subsets"
        / f"validation_natural_stage_{stage_id:02d}_repeat_{repeat_id:02d}_fold_{fold_id:02d}.csv"
    )
    if path.exists():
        return pd.read_csv(path).reset_index(drop=True)

    split_path = curriculum_dir / f"stage_{stage_id:02d}" / "stage_split_or_fold_assignment.csv"
    if not split_path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(split_path)
    df = _filter_repeat_fold(df, fold_id=fold_id, repeat_id=repeat_id)
    return df[df["subset"] == "validation"].reset_index(drop=True)


def make_stage_table(stage_configs: Optional[Sequence[StageConfig]] = None) -> pd.DataFrame:
    if stage_configs is None:
        stage_configs = DEFAULT_STAGE_CONFIGS
    rows = []
    for c in stage_configs:
        rows.append({
            "stage_id": c.stage_id,
            "stage_name": c.stage_name,
            "dev_fraction": c.dev_fraction,
            "epoch": c.epoch,
            "cv_mode": c.cv_mode,
            "n_splits": c.n_splits,
            "n_repeats": c.n_repeats,
            "val_size": c.val_size,
            "max_unique_per_class": c.max_unique_per_class,
            "augmentation_fraction_of_final": c.augmentation_fraction_of_final,
            "augmented_target_per_class": c.augmented_target_per_class,
            "augmented_total_10_classes": c.augmented_target_per_class * 10,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optional: materialisasi augmentasi offline menggunakan PIL.
# Jika training memakai online augmentation di PyTorch Dataset, bagian ini tidak wajib.
# ---------------------------------------------------------------------------

def _pil_apply_single_op(img, op: str, rng: np.random.Generator):
    from PIL import ImageOps, ImageEnhance, ImageFilter

    if op == "rotate_small":
        angle = float(rng.uniform(-12, 12))
        return img.rotate(angle, resample=2, expand=False, fillcolor=0)

    if op == "shift_small":
        dx = int(rng.integers(-8, 9))
        dy = int(rng.integers(-8, 9))
        canvas = img.transform(img.size, 2, (1, 0, dx, 0, 1, dy), resample=2)
        return canvas

    if op == "zoom_small":
        w, h = img.size
        scale = float(rng.uniform(1.02, 1.12))
        nw, nh = int(w * scale), int(h * scale)
        resized = img.resize((nw, nh), resample=2)
        left = max(0, (nw - w) // 2)
        top = max(0, (nh - h) // 2)
        return resized.crop((left, top, left + w, top + h))

    if op == "horizontal_flip_safe":
        # Untuk rempah, flip horizontal umumnya tidak mengubah kelas.
        if rng.random() < 0.5:
            return ImageOps.mirror(img)
        return img

    if op == "brightness_small":
        factor = float(rng.uniform(0.85, 1.15))
        return ImageEnhance.Brightness(img).enhance(factor)

    if op == "contrast_small":
        factor = float(rng.uniform(0.85, 1.20))
        return ImageEnhance.Contrast(img).enhance(factor)

    if op == "gamma_small":
        gamma = float(rng.uniform(0.85, 1.15))
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.power(np.clip(arr, 0, 1), gamma)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        from PIL import Image
        return Image.fromarray(arr, mode=img.mode)

    if op == "gaussian_noise_low":
        arr = np.asarray(img).astype(np.float32)
        noise = rng.normal(0, 4.0, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        from PIL import Image
        return Image.fromarray(arr, mode=img.mode)

    if op == "blur_light":
        radius = float(rng.uniform(0.25, 0.75))
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    if op == "sharpen_light":
        return img.filter(ImageFilter.SHARPEN)

    if op == "random_crop_pad_small":
        w, h = img.size
        crop_pct = float(rng.uniform(0.92, 0.98))
        nw, nh = int(w * crop_pct), int(h * crop_pct)
        left = int(rng.integers(0, max(1, w - nw + 1)))
        top = int(rng.integers(0, max(1, h - nh + 1)))
        cropped = img.crop((left, top, left + nw, top + nh))
        return cropped.resize((w, h), resample=2)

    if op == "affine_small":
        # Affine ringan: shear sangat kecil.
        shear = float(rng.uniform(-0.08, 0.08))
        return img.transform(img.size, 2, (1, shear, 0, shear, 1, 0), resample=2)

    # Unknown op: biarkan gambar tidak berubah agar pipeline tetap aman.
    return img


def apply_augmentation_ops(image_path: Union[str, Path], ops: Union[str, Sequence[str]], seed: int = 42, mode: str = "L"):
    """Menerapkan operasi augmentasi ringan ke satu gambar dan mengembalikan PIL Image."""
    from PIL import Image

    image_path = Path(image_path)
    img = Image.open(image_path)
    if mode:
        img = img.convert(mode)

    if isinstance(ops, str):
        ops_list = [x for x in ops.split("+") if x]
    else:
        ops_list = list(ops)

    rng = np.random.default_rng(seed)
    for op in ops_list:
        img = _pil_apply_single_op(img, op, rng)
    return img


def materialize_augmentation_plan(
    augmentation_plan: Union[str, Path, pd.DataFrame],
    output_root: Union[str, Path] = ".",
    overwrite: bool = False,
    mode: str = "L",
) -> pd.DataFrame:
    """
    Membuat file gambar hasil augmentasi offline berdasarkan augmentation_plan.csv.

    Input augmentation_plan wajib punya kolom:
    - path
    - augmentation_ops
    - planned_output_path

    Return augmentation manifest dengan kolom materialized_path dan status.
    """
    if isinstance(augmentation_plan, (str, Path)):
        plan_df = pd.read_csv(augmentation_plan)
    else:
        plan_df = augmentation_plan.copy()

    output_root = Path(output_root)
    rows = []

    for _, row in plan_df.iterrows():
        src = Path(str(row["path"]))
        rel_out = Path(str(row.get("planned_output_path", "")))
        if not str(rel_out):
            rel_out = Path("augmented") / f"{row.get('augmentation_id', stable_id(str(src)))}.png"
        dst = output_root / rel_out
        dst.parent.mkdir(parents=True, exist_ok=True)

        status = "created"
        if dst.exists() and not overwrite:
            status = "exists"
        else:
            try:
                seed = stable_int(str(row.get("augmentation_id", "")) + str(src))
                img = apply_augmentation_ops(src, row.get("augmentation_ops", ""), seed=seed, mode=mode)
                img.save(dst)
            except Exception as exc:
                status = f"error: {exc}"

        new_row = row.to_dict()
        new_row["materialized_path"] = str(dst)
        new_row["materialize_status"] = status
        rows.append(new_row)

    return pd.DataFrame(rows)
