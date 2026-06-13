import pandas as pd
from pathlib import Path

from _path_setup import PROJECT_ROOT

def test_curriculum_integrity():
    base_dir = PROJECT_ROOT / "outputs/04b_curriculum_stage_data_mechanism_80_20_audit_patched/csv"
    
    print("=== UJI COBA INTEGRITAS CURRICULUM ===")
    
    # 1. Uji Global Split (Development vs Holdout Test)
    print("\n1. Menguji Global Split (Development vs Holdout Test)...")
    dev_df = pd.read_csv(base_dir / "development_manifest.csv")
    test_df = pd.read_csv(base_dir / "holdout_test_manifest.csv")
    
    dev_ids = set(dev_df["source_image_id"].astype(str))
    test_ids = set(test_df["source_image_id"].astype(str))
    overlap_global = dev_ids.intersection(test_ids)
    
    print(f"   - Jumlah data Development: {len(dev_df)}")
    print(f"   - Jumlah data Holdout Test: {len(test_df)}")
    if len(overlap_global) == 0:
        print("   -> [PASS] TIDAK ADA KEBOCORAN. Development dan Holdout Test terpisah 100%.")
    else:
        print(f"   -> [FAIL] Ditemukan {len(overlap_global)} gambar bocor!")

    # 2. Uji Stage 1 Train vs Validation
    print("\n2. Menguji Train vs Validation (Stage 1 Fold 0)...")
    val_path = base_dir / "stage_01" / "train_validation_subsets" / "validation_natural_stage_01_repeat_00_fold_00.csv"
    train_natural_path = base_dir / "stage_01" / "train_validation_subsets" / "train_natural_stage_01_repeat_00_fold_00.csv"
    
    val_df = pd.read_csv(val_path)
    train_nat_df = pd.read_csv(train_natural_path)
    
    val_ids = set(val_df["source_image_id"].astype(str))
    train_ids = set(train_nat_df["source_image_id"].astype(str))
    overlap_stage = val_ids.intersection(train_ids)
    
    print(f"   - Jumlah data Train Natural: {len(train_nat_df)}")
    print(f"   - Jumlah data Validation Natural: {len(val_df)}")
    if len(overlap_stage) == 0:
        print("   -> [PASS] TIDAK ADA KEBOCORAN antara Train dan Validation.")
    else:
        print(f"   -> [FAIL] Ditemukan {len(overlap_stage)} gambar bocor ke validation!")

    # 3. Uji apakah Validation mengandung gambar augmentasi
    print("\n3. Menguji apakah file Validation murni natural (tanpa augmentasi)...")
    # Cek apakah ada path yang mengarah ke folder 'augmented' di file validation
    has_augmented_val = val_df["path"].str.contains(r"^augmented\\|/augmented/").any()
    if not has_augmented_val:
        print("   -> [PASS] Data Validation murni natural. Tidak ada gambar augmentasi di Validation.")
    else:
        print("   -> [FAIL] Data Validation tercemar gambar augmentasi!")

    # 4. Uji file Combined Train (Apakah menggabungkan data asli dan augmentasi?)
    print("\n4. Menguji file Combined Train Plan untuk Worker...")
    combined_path = base_dir / "stage_01" / "combined_train_plan_original_plus_augmented.csv"
    combined_df = pd.read_csv(combined_path)
    
    # Hitung jumlah gambar asli vs augmentasi
    num_original = len(combined_df[combined_df["is_original"] == True])
    num_augmented = len(combined_df[combined_df["is_augmented"] == True])
    
    # Cek path nya
    paths = combined_df["path"]
    if "planned_output_path" in combined_df.columns:
        paths_aug = combined_df[combined_df["is_augmented"] == True]["planned_output_path"]
        aug_paths_correct = paths_aug.str.contains("augmented").all()
    else:
        aug_paths_correct = combined_df[combined_df["is_augmented"] == True]["path"].str.contains("augmented").all()
        
    orig_paths_correct = combined_df[combined_df["is_original"] == True]["path"].str.contains("dataset_cleaned_preprocessed").all()
    
    print(f"   - Total baris di Combined Train: {len(combined_df)}")
    print(f"   - Jumlah baris Original: {num_original}")
    print(f"   - Jumlah baris Augmentasi: {num_augmented}")
    
    if orig_paths_correct and aug_paths_correct and (num_original > 0) and (num_augmented > 0):
        print("   -> [PASS] CSV Worker berhasil menggabungkan path Original dan Augmentasi dengan benar dalam 1 file.")
    else:
        print("   -> [FAIL] Ada kesalahan format penggabungan di CSV Combined Train.")
        
    print("\n=== UJI COBA SELESAI ===")

if __name__ == "__main__":
    test_curriculum_integrity()
