import os
import zipfile
from pathlib import Path

# Mendefinisikan target
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ZIP = PROJECT_ROOT / "worker_deployment_bundle.zip"

# Daftar folder dan file wajib untuk Worker (Sesuai arsitektur)
REQUIRED_PATHS = [
    ".env",
    "requirements.txt",
    "requirements-lightning-gpu.txt",
    "environment.yml",
    "04_runtime_final/worker",
    "patched_program_files",
    "03_model_architecture",
    "curriculum_outputs",
    "dataset_cleaned_preprocessed",
    "notebooks/02_worker_pc_template.ipynb",
    "preflight_check.py",
    "setup_lightning_gpu_env.py",
    "tests/_path_setup.py",
    "tests/worker_smoke_estimate.py",
    "tests/worker_connection_test.py"
]

def create_worker_bundle():
    print(f"[MULAI] Memulai pembuatan paket deployment worker: {OUTPUT_ZIP.name}")
    
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for rel_path in REQUIRED_PATHS:
            target_path = PROJECT_ROOT / rel_path
            
            if not target_path.exists():
                print(f"[PERINGATAN] {rel_path} tidak ditemukan, dilewati.")
                continue
                
            if target_path.is_file():
                print(f"Menambahkan file: {rel_path}")
                zipf.write(target_path, arcname=rel_path)
            elif target_path.is_dir():
                print(f"Menambahkan direktori: {rel_path}/ ...")
                for root, dirs, files in os.walk(target_path):
                    # Hindari file cache
                    dirs[:] = [d for d in dirs if d not in ('__pycache__', '.ipynb_checkpoints')]
                    
                    for file in files:
                        if file.endswith('.pyc'):
                            continue
                        
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(PROJECT_ROOT)
                        zipf.write(file_path, arcname=arcname)
                        
    print(f"\n[SELESAI] Paket worker berhasil dibuat di: {OUTPUT_ZIP}")
    print("[INFO] Anda tinggal membawa/meng-upload file .zip ini ke Kaggle/Colab/PC Lain, lalu di-unzip di sana.")

if __name__ == "__main__":
    create_worker_bundle()
