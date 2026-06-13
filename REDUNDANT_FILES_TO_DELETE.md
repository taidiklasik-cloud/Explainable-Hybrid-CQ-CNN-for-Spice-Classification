# Redundant Files to Delete After Pipeline Execution

## 🔴 SAFE TO DELETE (From Execution)

### 1. Empty Temporary Folders (8 KB)
**SAFE TO DELETE** - already moved contents

```bash
rm -rf curriculum_outputs/
rm -rf architecture_worker_artifacts/
```

✅ All outputs already moved to:
- `outputs/04b_curriculum_stage_data_mechanism_80_20_audit_patched/`
- `outputs/05_model_architecture_resource_checked_worker_runtime/`

---

### 2. Python Cache Directories (Can delete anytime)
**SAFE TO DELETE** - auto-regenerated when code runs

```bash
rm -rf 02_curriculum_pipeline/__pycache__/
rm -rf 03_model_architecture/__pycache__/
find . -type d -name "__pycache__" -exec rm -rf {} \; 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} \; 2>/dev/null
```

✅ These are just compiled Python bytecode

---

### 3. Old Curriculum Version (8 KB)
**SAFE TO DELETE** - use 04b (patched) version instead

```bash
rm -rf outputs/04_curriculum_stage_data_mechanism_80_20_audit/
```

✅ Keep only: `outputs/04b_curriculum_stage_data_mechanism_80_20_audit_patched/`

---

### 4. Source Backup Folder (27 MB)
**SAFE TO DELETE** - original extracted files, duplicates current code

```bash
rm -rf 00_source_uploads/
```

**Contains duplicates of:**
- Notebooks (eda_datasetawal_generated.ipynb, etc.)
- Python modules (curriculum_stage_utils.py, model_architecture_modules.py)

**Keep if:** You want to preserve original/unpatched versions

---

## 🟡 OPTIONAL TO DELETE (Large, but keep as reference)

### 5. Preprocessed Dataset Folder
**DO NOT DELETE during training** - contains actual preprocessed images!

```bash
# DO NOT RUN:
# rm -rf dataset_cleaned_preprocessed/
```

⚠️ **NEEDED**: This folder contains the 2,001 preprocessed PNG images
- Used by training pipeline
- Referenced in `dataset_final_manifest.csv`
- Safe to delete ONLY after training is complete and checkpoints are saved

---

### 6. Dataset ZIP Archive (130 MB)
**OPTIONAL - delete only if you have backups**

```bash
# Only if you have other backup copies
rm "dataset 10 kelas.zip"
```

✅ You already have extracted: `dataset 10 kelas/` folder

---

### 7. Original Bundle ZIP (33 MB)
**OPTIONAL - delete if not needed as backup**

```bash
# Only if you don't need original package as reference
rm cqcnn_end_to_end_complete_bundle_v2.zip
```

✅ All code is extracted and organized in working folders

---

### 8. Archive Folder (432 MB - Largest!)
**OPTIONAL - check contents first, then delete if not needed**

```bash
# First check what's inside
ls -la archive/

# Then delete if confirmed not needed
rm -rf archive/
```

⚠️ Unknown contents - verify before deletion

---

## 🟢 DO NOT DELETE (Needed for Training)

```
✗ patched_program_files/              (current production code)
✗ 04_runtime_final/                   (worker runtime code)
✗ notebooks/                          (orchestrator & worker entry points)
✗ 01_eda_pipeline/                    (reference: how data was processed)
✗ 02_curriculum_pipeline/             (reference & utilities)
✗ 03_model_architecture/              (reference & model definitions)
✗ 03_model_architecture/__pycache__/  (SAFE to delete, but not critical)
✗ dataset_cleaned_preprocessed/       (ACTUAL IMAGES - DO NOT DELETE)
✗ dataset 10 kelas/                   (ACTUAL IMAGES - DO NOT DELETE)
✗ dataset_final_manifest.csv          (CRITICAL - canonical manifest)
✗ outputs/                            (ALL OUTPUT FILES - NEEDED FOR AUDIT)
✗ sql/                                (NEEDED FOR LOCAL POSTGRESQL ORCHESTRATION SETUP)
✗ .env (when created)                 (NEEDED FOR TRAINING)
```

---

## 📊 Recommended Cleanup Plan

### **TIER 1: Safe Cleanup (Almost no risk)**
**Space freed: ~45 KB** | Delete with confidence

```bash
rm -rf curriculum_outputs/
rm -rf architecture_worker_artifacts/
rm -rf 02_curriculum_pipeline/__pycache__
rm -rf 03_model_architecture/__pycache__
rm -rf outputs/04_curriculum_stage_data_mechanism_80_20_audit/
find . -type d -name "__pycache__" -exec rm -rf {} \; 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} \; 2>/dev/null
```

✅ Safe to do now

---

### **TIER 2: Reference Cleanup (Verify first)**
**Space freed: ~27 MB** | Optional after reviewing

```bash
# Only delete if you confirm no need for original unpatched versions
rm -rf 00_source_uploads/
```

---

### **TIER 3: Large Archive Cleanup (Check first)**
**Space freed: ~432 MB** | Only if sure

```bash
# First verify contents
ls -la archive/

# Then delete if confirmed not needed
# rm -rf archive/
```

---

### **TIER 4: Dataset Backup Cleanup (AFTER training)**
**Space freed: ~163 MB** | Only after training completes

```bash
# WARNING: DO NOT DELETE UNTIL AFTER TRAINING COMPLETE
# rm "dataset 10 kelas.zip"
# rm cqcnn_end_to_end_complete_bundle_v2.zip
```

⚠️ Keep for now - might need backup during development

---

## ✅ Recommended Action (Do This Now)

```bash
# SAFE: Clean up execution artifacts
echo "Cleaning up temporary/cache files..."
rm -rf curriculum_outputs/
rm -rf architecture_worker_artifacts/
rm -rf 02_curriculum_pipeline/__pycache__
rm -rf 03_model_architecture/__pycache__
rm -rf outputs/04_curriculum_stage_data_mechanism_80_20_audit/
find . -type d -name "__pycache__" -exec rm -rf {} \; 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} \; 2>/dev/null

echo "✅ Cleanup complete - freed ~45 KB"
echo "✅ Safe to proceed with training setup"
```

**Result**:
- ✅ Clean project structure
- ✅ No duplicate code or outputs
- ✅ All necessary files preserved
- ✅ Ready for environment configuration & training

---

## Summary

| Category | Space | Safe? | Action |
|----------|-------|-------|--------|
| **Temp folders** | 8 KB | ✅ | DELETE NOW |
| **Python cache** | ~1 MB | ✅ | DELETE NOW |
| **Old outputs** | 8 KB | ✅ | DELETE NOW |
| **Source backup** | 27 MB | ⚠️ | Delete if no need |
| **Archive folder** | 432 MB | ⚠️ | Check first, then delete |
| **Dataset ZIPs** | 163 MB | ⚠️ | Keep until after training |

**Conservative approach**: Delete only TIER 1 (~45 KB) now. Delete TIER 2+ after training succeeds.
