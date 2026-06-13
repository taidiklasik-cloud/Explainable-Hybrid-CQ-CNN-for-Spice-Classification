import json
from pathlib import Path

notebook_path = Path("notebooks/02_worker_pc_template.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Buat cell baru
new_markdown = {
  "cell_type": "markdown",
  "metadata": {},
  "source": [
    "### 0. Universal Setup & Environment Diagnostic\n",
    "\n",
    "Jalankan cell di bawah ini **pertama kali** (terutama jika Anda berada di platform *cloud* seperti **Kaggle, Google Colab, atau PC baru**). \n",
    "Sistem akan otomatis:\n",
    "1. Membaca `requirements.txt` dan mengunduh semua library yang kurang.\n",
    "2. Menjalankan skrip `check_environment.py` untuk memastikan (Deep Check) bahwa PyTorch, GPU, Database, dan Rclone Anda berfungsi secara riil.\n",
    "\n",
    "> **TIPS PENTING**: Jika *pip install* baru saja menginstal banyak hal untuk pertama kalinya, Anda **Wajib melakukan Restart Kernel** (Menu bar: Kernel -> Restart) agar Jupyter bisa mendeteksi *library* yang baru diunduh. Setelah *restart*, jalankan ulang sel-sel di bawahnya."
  ]
}

new_code = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "outputs": [],
  "source": [
    "import os\n",
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()\n",
    "req_path = PROJECT_ROOT / \"requirements.txt\"\n",
    "check_path = PROJECT_ROOT / \"check_environment.py\"\n",
    "\n",
    "print(\"1. Memastikan semua dependensi terinstall...\")\n",
    "!{sys.executable} -m pip install -q -r {req_path}\n",
    "\n",
    "print(\"\\n2. Menjalankan Uji Coba Kesiapan Environment...\")\n",
    "!{sys.executable} {check_path}"
  ]
}

# Insert after the first markdown cell (index 1 and 2)
if nb["cells"][1]["cell_type"] != "markdown" or "0. Universal Setup" not in "".join(nb["cells"][1]["source"]):
    nb["cells"].insert(1, new_markdown)
    nb["cells"].insert(2, new_code)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook updated successfully.")
