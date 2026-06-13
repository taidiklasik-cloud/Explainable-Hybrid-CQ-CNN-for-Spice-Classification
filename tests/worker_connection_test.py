"""
worker_connection_test.py
Skrip ini dirancang untuk dijalankan di node worker untuk memverifikasi
koneksi eksternal yang diperlukan:
1. PostgreSQL Orchestration (cqcnn_orchestration)
2. PostgreSQL Optuna (optuna_skripsi)
3. Rclone -> Google Drive (Read/Write permissions)

Usage:
    python tests/worker_connection_test.py
"""
import os
import sys
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv

from _path_setup import PROJECT_ROOT, configure_paths


def check_env():
    load_dotenv()
    required_vars = [
        "ORCHESTRATION_DB_DSN",
        "OPTUNA_STORAGE_URL",
        "RCLONE_EXE_PATH",
        "RCLONE_CONFIG_PATH",
        "RCLONE_REMOTE_NAME",
        "GDRIVE_CHECKPOINT_ROOT"
    ]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        print(f"[FAIL] Variabel environment berikut belum diset: {missing}")
        print("Harap pastikan file .env sudah diisi dengan benar.")
        sys.exit(1)
    
    # Validasi Rclone binary exists
    rclone_exe = Path(os.environ["RCLONE_EXE_PATH"])
    if not rclone_exe.exists():
        print(f"[FAIL] Rclone binary tidak ditemukan di path: {rclone_exe}")
        sys.exit(1)

def test_orchestration_db():
    print("\n--- [1] Menguji Koneksi Orchestration Database ---")
    try:
        from patched_program_files.postgres_orchestration_db import PostgresOrchestrationDb, PostgresOrchestrationDbConfig
        dsn = os.environ["ORCHESTRATION_DB_DSN"]
        db = PostgresOrchestrationDb(PostgresOrchestrationDbConfig(dsn=dsn))
        conn_res = db.test_connection()
        print(f"[PASS] Berhasil terhubung ke Orchestration DB: {conn_res['database_name']} as {conn_res['user_name']}")
    except Exception as e:
        print(f"[FAIL] Gagal terhubung ke Orchestration DB.\nError: {e}")
        sys.exit(1)

def test_optuna_db():
    print("\n--- [2] Menguji Koneksi Optuna Database ---")
    try:
        import optuna
        optuna_url = os.environ["OPTUNA_STORAGE_URL"]
        studies = optuna.get_all_study_summaries(storage=optuna_url)
        print(f"[PASS] Berhasil terhubung ke Optuna DB. Ditemukan {len(studies)} studies.")
    except Exception as e:
        print(f"[FAIL] Gagal terhubung ke Optuna DB.\nError: {e}")
        sys.exit(1)

def test_rclone_gdrive():
    print("\n--- [3] Menguji Konektivitas dan Akses Rclone Google Drive ---")
    rclone_exe = os.environ["RCLONE_EXE_PATH"]
    rclone_config = os.environ["RCLONE_CONFIG_PATH"]
    remote_name = os.environ["RCLONE_REMOTE_NAME"]
    root_dir = os.environ["GDRIVE_CHECKPOINT_ROOT"]
    
    test_folder_name = f".test_connection_tmp_{int(time.time())}"
    target_remote_path = f"{remote_name}:{root_dir}/{test_folder_name}"
    
    # 1. Test WRITE (mkdir)
    print(f"-> Mencoba membuat folder sementara di {target_remote_path} ...")
    cmd_mkdir = [
        rclone_exe, "mkdir", target_remote_path,
        "--config", rclone_config
    ]
    try:
        subprocess.run(cmd_mkdir, check=True, capture_output=True, text=True)
        print("[PASS] Operasi Rclone 'mkdir' berhasil (Akses Tulis diizinkan).")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Operasi Rclone 'mkdir' gagal!\nError: {e.stderr}")
        sys.exit(1)
        
    # 2. Test READ (lsd)
    print("-> Memverifikasi folder dapat dibaca ...")
    cmd_lsd = [
        rclone_exe, "lsd", f"{remote_name}:{root_dir}",
        "--config", rclone_config
    ]
    try:
        res = subprocess.run(cmd_lsd, check=True, capture_output=True, text=True)
        if test_folder_name in res.stdout:
            print("[PASS] Operasi Rclone 'lsd' berhasil melihat folder (Akses Baca diizinkan).")
        else:
            print("[FAIL] Folder tes dibuat tapi tidak terlihat oleh lsd.")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Operasi Rclone 'lsd' gagal!\nError: {e.stderr}")
        # Lanjut ke rmdir agar tidak nyangkut
        
    # 3. Test DELETE (rmdir)
    print(f"-> Menghapus folder sementara di {target_remote_path} ...")
    cmd_rmdir = [
        rclone_exe, "rmdir", target_remote_path,
        "--config", rclone_config
    ]
    try:
        subprocess.run(cmd_rmdir, check=True, capture_output=True, text=True)
        print("[PASS] Operasi Rclone 'rmdir' berhasil (Akses Hapus diizinkan).")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Operasi Rclone 'rmdir' gagal, folder mungkin tertinggal.\nError: {e.stderr}")

def main():
    print("===========================================================")
    print("              WORKER CONNECTION TEST RUNNER                ")
    print("===========================================================")
    configure_paths()
    
    check_env()
    test_orchestration_db()
    test_optuna_db()
    test_rclone_gdrive()
    
    print("\n===========================================================")
    print(">>> KESIMPULAN: SEMUA KONEKSI WORKER BERFUNGSI NORMAL <<<")
    print("Worker ini sudah siap untuk menjalankan skrip training (02_worker_pc_template.ipynb).")

if __name__ == "__main__":
    main()
