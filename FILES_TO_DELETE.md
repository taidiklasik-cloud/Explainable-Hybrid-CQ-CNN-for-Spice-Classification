# Files to Delete After Pipeline Execution

## 🟢 SAFE TO DELETE (No Risk)

### 1. Temporary Working Folders (12 KB)
**These folders are now empty after moving outputs:**

```bash
rm -rf curriculum_outputs/
rm -rf architecture_worker_artifacts/
```

**Reason**: All content already moved to `outputs/04b/` and `outputs/05/`

---

### 2. Python Cache Directories (Can be deleted anytime)
**Auto-generated, will regenerate if needed:**

```bash
rm -rf 02_curriculum_pipeline/__pycache__/
rm -rf 03_model_architecture/__pycache__/
```

**Reason**: Just compiled bytecode, not needed

---

### 3. Old Curriculum Version (8 KB)
**Use 04b (patched) version instead:**

```bash
rm -rf outputs/04_curriculum_stage_data_mechanism_80_20_audit/
```

**Reason**: Keep only the patched (04b) version which is current

---

## 🟡 OPTIONAL - DELETE IF SPACE IS TIGHT

### 4. Large ZIP Archives (163 MB total)
**Only if you have backups elsewhere:**

```bash
# Delete if you don't need as backup
rm "dataset 10 kelas.zip"                          # 130 MB
rm "cqcnn_end_to_end_complete_bundle_v2.zip"      # 33 MB
```

**Reason**: You already have extracted `dataset 10 kelas/` folder and all code is in working directories

**Keep if**: You want backup copies of original data

---

### 5. Source Upload Folder (27 MB)
**Only if you don't need original backup:**

```bash
# Delete if you have git history or other backup
rm -rf 00_source_uploads/
```

**Reason**: Contains duplicate old notebooks and utilities already in current folders

**Keep if**: You want reference to original unpatched versions

---

### 6. Archive Folder (432 MB - Largest!)
**Check contents first:**

```bash
ls -la archive/
rm -rf archive/  # if you don't need it
```

**Reason**: Unknown contents; delete only if confirmed not needed

---

## 🔴 DO NOT DELETE

```
✗ 01_eda_pipeline/                    (source notebooks for reference)
✗ 02_curriculum_pipeline/             (source modules + notebooks)
✗ 03_model_architecture/              (source modules + notebooks)
✗ 04_runtime_final/                   (worker runtime code)
✗ patched_program_files/              (production code modules)
✗ outputs/                            (all outputs - needed as audit trail)
✗ notebooks/                          (orchestrator & worker entry points)
✗ sql/                                (local PostgreSQL orchestration schema - needed for setup)
✗ dataset 10 kelas/                   (actual dataset images - DO NOT DELETE)
✗ dataset_cleaned_preprocessed/       (preprocessed dataset - DO NOT DELETE)
✗ dataset_final_manifest.csv          (canonical manifest - DO NOT DELETE)
```

---

## 📊 Cleanup Summary

### Recommended: Delete Everything Safe (20 KB freed)
```bash
rm -rf curriculum_outputs/
rm -rf architecture_worker_artifacts/
rm -rf 02_curriculum_pipeline/__pycache__/
rm -rf 03_model_architecture/__pycache__/
rm -rf outputs/04_curriculum_stage_data_mechanism_80_20_audit/
```

**Space freed**: ~20 KB (negligible)

---

### Aggressive: Delete Safe + Large Backups (620 MB freed)
```bash
# All of above, plus:
rm "dataset 10 kelas.zip"                          # 130 MB
rm "cqcnn_end_to_end_complete_bundle_v2.zip"      # 33 MB
rm -rf 00_source_uploads/                          # 27 MB
rm -rf archive/                                    # 432 MB
```

**Space freed**: ~620 MB

**WARNING**: Only do this if you have confirmed these files are backed up elsewhere or not needed

---

## 🎯 Recommended Action

**Delete safe files (20 KB)**:
```bash
# Cleanup temporary working folders and cache
rm -rf curriculum_outputs/ architecture_worker_artifacts/
rm -rf 02_curriculum_pipeline/__pycache__/ 03_model_architecture/__pycache__/
rm -rf outputs/04_curriculum_stage_data_mechanism_80_20_audit/
```

**Space impact**: Negligible but cleans up clutter

**Keep everything else** until you've confirmed training runs successfully, then decide on backups.

---

## Why Keep the Rest?

- **Source notebooks** (`01_eda_pipeline/` etc): Reference documentation for how data was processed
- **outputs/** folder: Complete audit trail of pipeline execution (required for thesis appendix)
- **dataset_cleaned_preprocessed/**: Actual image files used for training (essential)
- **patched_program_files/**: Production code modules used by worker/orchestrator

These are all referenced by the training pipeline and will be needed.
