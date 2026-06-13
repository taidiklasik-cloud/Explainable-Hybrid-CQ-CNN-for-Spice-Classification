import pandas as pd
from pathlib import Path

from _path_setup import PROJECT_ROOT

def test_stratified_split():
    base_dir = PROJECT_ROOT / "outputs/04b_curriculum_stage_data_mechanism_80_20_audit_patched/csv"
    
    # Load data
    dev_df = pd.read_csv(base_dir / "development_manifest.csv")
    test_df = pd.read_csv(base_dir / "holdout_test_manifest.csv")
    
    # Combine back to get the original "full" dataset distribution
    full_df = pd.concat([dev_df, test_df], ignore_index=True)
    
    def get_distribution(df, label):
        counts = df["class"].value_counts().sort_index()
        total = len(df)
        percentages = (counts / total * 100).round(2)
        
        # Format as string
        dist_str = "\n".join([f"      {cls}: {pct}% ({cnt} images)" for cls, pct, cnt in zip(percentages.index, percentages.values, counts.values)])
        return total, dist_str, percentages

    print("=== UJI COBA INTEGRITAS STRATIFIED SPLIT ===")
    
    total_full, dist_full, pct_full = get_distribution(full_df, "FULL DATASET")
    print(f"\n1. Distribusi Dataset Penuh (Total: {total_full} gambar) [Kondisi Imbalance Awal]:")
    print(dist_full)
    
    total_dev, dist_dev, pct_dev = get_distribution(dev_df, "DEVELOPMENT (90%)")
    print(f"\n2. Distribusi Development Set (Total: {total_dev} gambar):")
    print(dist_dev)
    
    total_test, dist_test, pct_test = get_distribution(test_df, "HOLDOUT TEST (10%)")
    print(f"\n3. Distribusi Holdout Test Set (Total: {total_test} gambar):")
    print(dist_test)
    
    # Check max deviation between full and dev/test
    max_dev_diff = (pct_full - pct_dev).abs().max()
    max_test_diff = (pct_full - pct_test).abs().max()
    
    print("\n=== KESIMPULAN GLOBAL SPLIT ===")
    if max_dev_diff <= 1.5 and max_test_diff <= 1.5:
        print(f"[PASS] Split terbukti Stratified! Maksimal deviasi persentase antar kelas sangat kecil (Deviasi maks: {max_test_diff:.2f}%).")
        print("Proporsi kelas yang imbalance tetap dipertahankan secara akurat di Development dan Holdout.")
    else:
        print(f"[FAIL] Terdeteksi Inkonsistensi Stratified Split! Deviasi maks: {max_test_diff:.2f}%")
        
    # Let's also check Stage 1 subset
    stage_1_val = pd.read_csv(base_dir / "stage_01/train_validation_subsets/validation_natural_stage_01_repeat_00_fold_00.csv")
    stage_1_train = pd.read_csv(base_dir / "stage_01/train_validation_subsets/train_natural_stage_01_repeat_00_fold_00.csv")
    
    # Stage 1 natural (before balancing)
    stage_1_natural = pd.concat([stage_1_train, stage_1_val], ignore_index=True)
    total_s1, dist_s1, pct_s1 = get_distribution(stage_1_natural, "STAGE 1 NATURAL (5%)")
    
    print(f"\n4. Distribusi Stage 1 Subset Natural (Total: {total_s1} gambar):")
    print(dist_s1)
    
    max_s1_diff = (pct_full - pct_s1).abs().max()
    if max_s1_diff <= 3.0: # Margin slightly higher for very small datasets due to rounding/integer division
        print(f"   -> [PASS] Subset Stage 1 juga terbukti Stratified (Deviasi maks dari Full Dataset: {max_s1_diff:.2f}%).")
    else:
        print(f"   -> [FAIL/WARNING] Subset Stage 1 kurang Stratified (Deviasi maks: {max_s1_diff:.2f}%).")
        
    print("\n=== UJI COBA SELESAI ===")

if __name__ == "__main__":
    test_stratified_split()
