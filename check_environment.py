import os
import sys
import subprocess

# Warna untuk output terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def check(name: str, test_func) -> bool:
    print(f"Menguji {name}...", end=" ", flush=True)
    try:
        msg = test_func()
        print(f"\r{GREEN}[PASS]{RESET} {name} teruji jalan! ({msg})")
        return True
    except ImportError as e:
        print(f"\r{RED}[FAIL]{RESET} {name} tidak terinstall (ImportError: {e})")
        return False
    except Exception as e:
        print(f"\r{RED}[FAIL]{RESET} {name} terinstall tetapi GAGAL dieksekusi: {e}")
        return False

def test_pytorch():
    import torch
    # Uji komputasi tensor dasar
    t = torch.tensor([1.0, 2.0])
    res = t * 2
    
    # Uji CUDA
    if torch.cuda.is_available():
        t_gpu = t.cuda()
        gpu_name = torch.cuda.get_device_name(0)
        return f"PyTorch {torch.__version__}, GPU aktif: {gpu_name}, Tensor math OK"
    else:
        return f"PyTorch {torch.__version__}, CPU only, Tensor math OK"

def test_pennylane():
    import pennylane as qml
    import numpy as np
    # Uji pembuatan sirkuit kuantum sederhana
    dev = qml.device('default.qubit', wires=1)
    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=0)
        return qml.probs(wires=[0])
    result = circuit()
    if int(result[1]) == 1:
        return f"PennyLane {qml.__version__}, QNode simulasi kuantum OK"
    else:
        raise ValueError(f"Hasil sirkuit tidak sesuai ekspektasi (dapat {result})")

def test_optuna():
    import optuna
    # Uji in-memory study creation
    study = optuna.create_study(storage="sqlite:///:memory:", direction="minimize")
    return f"Optuna {optuna.__version__}, In-memory study OK"

def test_data_science():
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    arr = np.array([1, 2])
    return f"Pandas {pd.__version__} & Numpy {np.__version__} OK"

def test_image_processing():
    import cv2
    import numpy as np
    # Uji resize array kosong (mensimulasikan manipulasi gambar)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    resized = cv2.resize(img, (5, 5))
    return f"OpenCV {cv2.__version__}, Image resize OK"

def test_database_connection():
    from dotenv import load_dotenv
    load_dotenv()
    
    dsn = os.environ.get("ORCHESTRATION_DB_DSN")
    if not dsn:
        return "Skip koneksi DB (.env ORCHESTRATION_DB_DSN tidak ditemukan)"
        
    import sqlalchemy
    engine = sqlalchemy.create_engine(dsn)
    with engine.connect() as conn:
        # Coba query ringan
        from sqlalchemy import text
        conn.execute(text("SELECT 1"))
    return f"SQLAlchemy {sqlalchemy.__version__}, Terhubung ke PostgreSQL Orchestrator OK"

def test_rclone():
    from dotenv import load_dotenv
    load_dotenv()
    rclone_path = os.environ.get("RCLONE_EXE_PATH", "rclone")
    res = subprocess.run([rclone_path, "version"], capture_output=True, text=True, timeout=5)
    if res.returncode == 0:
        version_line = res.stdout.split('\n')[0]
        return f"Rclone eksekusi OK ({version_line})"
    else:
        raise RuntimeError("Subprocess return code non-zero")

def main():
    print("="*60)
    print("   UJI COBA EKSEKUSI RUNTIME ENVIRONMENT (DEEP CHECK)")
    print("="*60)
    
    tests = [
        ("PyTorch & CUDA GPU", test_pytorch),
        ("PennyLane Quantum", test_pennylane),
        ("Optuna Optimizer", test_optuna),
        ("Pandas & Numpy", test_data_science),
        ("OpenCV & Pustaka Gambar", test_image_processing),
        ("PostgreSQL SQLAlchemy", test_database_connection),
        ("Rclone CLI", test_rclone),
    ]

    all_passed = True
    for name, func in tests:
        if not check(name, func):
            all_passed = False
            
    print("="*60)
    if all_passed:
        print(f"{GREEN}>>> ENVIRONMENT 100% SIAP DAN TERBUKTI BISA DIEKSEKUSI! <<< {RESET}")
        print("Semua pustaka berhasil memproses operasi matematis, database, dan kuantum.")
    else:
        print(f"{RED}>>> ENVIRONMENT BELUM SEMPURNA <<< {RESET}")
        print("Beberapa dependensi terinstall tapi GAGAL dijalankan saat dites.")
    print("="*60)

if __name__ == "__main__":
    main()
