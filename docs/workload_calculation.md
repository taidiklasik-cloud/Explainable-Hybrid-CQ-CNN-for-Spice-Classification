# Kalkulasi Workload Lengkap: Arsitektur, Volume, dan Estimasi Waktu

Dokumen ini merangkum seluruh kalkulasi workload untuk pipeline training
CQ-CNN (Classical) dan Hybrid QCQ-CNN di 5 stage curriculum learning.

---

## 1. Arsitektur Model

### 1.1 Classical Fully Spatial CNN

```
Input [B, 1, 128, 128]
  |
  +-- ConvBlock1: Conv2d(1->32) + GroupNorm + ReLU + BlurPool(stride=2) + Dropout
  |     Output: [B, 32, 64, 64]          288 params
  |
  +-- ConvBlock2: Conv2d(32->64) + GroupNorm + ReLU + BlurPool(stride=2) + Dropout
  |     Output: [B, 64, 32, 32]        18,432 params
  |
  +-- ConvBlock3: Conv2d(64->128) + GroupNorm + ReLU + BlurPool(stride=2) + Dropout
  |     Output: [B, 128, 16, 16]       73,728 params
  |
  +-- CBAM: ChannelAttention + SpatialAttention
  |     Output: [B, 128, 16, 16]        4,242 params
  |
  +-- Bottleneck: Conv2d(128->16) + GroupNorm + ReLU
  |     Output: [B, 16, 16, 16]         2,080 params
  |
  +-- Classical Head: Conv2d(16->10) + GroupNorm + ReLU
  |     Output: [B, 10, 16, 16]         1,460 params
  |
  +-- AdaptiveAvgPool2d -> Flatten
        Output: [B, 10]                 (logits)
```

### 1.2 Hybrid QCQ-CNN

```
Input [B, 1, 128, 128]
  |
  +-- SharedSpiceBackbone (SAMA dengan Classical)
  |     Output: [B, 16, 16, 16]        99,170 params
  |
  +-- Projection: AdaptiveAvgPool2d(4x4) + Flatten + LayerNorm
  |     Output: [B, 256]            0 params
  |
  +-- L2 Normalize
  |     Output: [B, 256] (unit norm)
  |
  +-- PennyLane Quantum Circuit:
  |     AmplitudeEmbedding(256 amplitudes, 8 qubits)
  |     StronglyEntanglingLayers(depth=2)
  |     Measurement: Pauli-Z expval(wires=0..7)
  |     Output: [B, 8]                     48 params (quantum rotations)
  |
  +-- Linear readout: Linear(8->10)
  |     Output: [B, 10] logits             90 params
```

### 1.3 Perbandingan Parameter

| Komponen | Classical | Hybrid | Catatan |
|----------|---:|---:|------|
| SharedSpiceBackbone | 99,170 | 99,170 | Identik (apple-to-apple) |
| Classical Head | 1,460 | — | Conv2d spatial head |
| Projection (Linear + LayerNorm) | — | 1,049,344 | Flatten 4096 → 256 |
| Quantum Head | — | 48 | 2 × 8 × 3 rotation angles |
| **Total Parameter** | **100,630** | **1,148,562** | Hybrid ~11.4× lebih besar |
| **Memory (params+buffers)** | **0.39 MB** | **4.39 MB** | Karena projection layer |

> **Catatan:** Hybrid memiliki lebih banyak parameter bukan karena quantum circuit-nya (hanya 48 params), melainkan karena projection layer Linear(4096 → 256) yang diperlukan untuk memetakan feature map ke amplitudo quantum.

### 1.4 Ukuran Checkpoint

| Jenis | Classical | Hybrid |
|-------|---:|---:|
| Model state_dict saja | 410 KB | 4,504 KB (4.4 MB) |
| Full checkpoint (model + optimizer + scheduler) | 410 KB | 4,504 KB (4.4 MB) |

> AdamW menyimpan momentum (m) dan variance (v) per parameter, tetapi karena model ini kecil, overhead optimizer negligible.

---

## 2. Quantum Circuit Detail

| Parameter | Nilai | Keterangan |
|-----------|------:|------------|
| n_qubits | 8 | 2^8 = 256 amplitudes |
| q_depth | 2 | StronglyEntanglingLayers |
| Trainable quantum params | 48 | 2 × 8 × 3 rotation angles |
| Linear readout params | 90 | 8 × 10 weights + 10 bias |
| Hybrid head params | 138 | 48 quantum params + 90 linear readout params |
| Statevector dimension | 256 | Ukuran state yang disimulasi |
| Encoding | AmplitudeEmbedding | Normalize + pad_with=0 |
| Ansatz | StronglyEntanglingLayers | Ring-style entanglement |
| Measurement | pauli_z_linear | Pauli-Z expectation [B,8] + Linear(8,10) logits |
| Diff method | **adjoint** | C++ optimized (lightning.qubit/gpu) |
| Device priority | lightning.gpu → lightning.qubit → default.qubit | Auto-fallback |

### Mengapa Adjoint Diferensiasi Cepat

| Metode | Forward Passes | Backward | Memori |
|--------|---:|---:|---:|
| Parameter-shift | 2P+1 = 97 | O(P) evaluasi | Rendah |
| Backprop (default.qubit) | 1 | 1 (Python autograd) | Tinggi |
| **Adjoint (lightning.qubit)** | **1** | **1 (C++ optimized)** | **Rendah** |
| **Adjoint (lightning.gpu)** | **1** | **1 (CUDA kernel)** | **Rendah** |

> Adjoint tidak menyimpan intermediate states. Ia memutar balik circuit secara analitik. Memori konstan O(2 × 2^n) dan waktu proporsional jumlah gates.

---

## 3. Volume Dataset per Stage

| Stage | Nama | Fraksi Data | Folds | Train/fold | Val/fold | Total/fold |
|:---:|------|:---:|:---:|---:|---:|---:|
| 1 | Sanity Test | 5% | 1 | 72 | 18 | 90 |
| 2 | Warm Start | 25% | 1 | 360 | 90 | 450 |
| 3 | Convergence | 50% | 5 | 720 | 180 | 900 |
| 4 | Max Accuracy | 100% | 5 | 1,440 | 360 | 1,800 |
| 5 | Final Eval | 100% | 5×5=25 | 1,440 | 360 | 1,800 |

Total gambar unik dalam dataset: **1,800** (setelah preprocessing).
Holdout test set terpisah: **tidak dipakai selama training**.

---

## 4. Konfigurasi Training per Stage

| Stage | max_epoch | Trials (HPO) | Folds | Objective | Split | Early Stop |
|:---:|---:|---:|---:|------|------|:---:|
| 1 | 2 | 1 | 1 | minimize val_loss | 80/20 | Tidak |
| 2 | 5 | 1 | 1 | minimize val_loss | 80/20 | Tidak |
| 3 | 25 | 20 | 5 | minimize val_loss | 5-fold CV | Ya |
| 4 | 50 | 30 | 5 | maximize val_macro_f1 | 5-fold CV | Ya |
| 5 | 100 | 1 | 25 | — (evaluasi final) | repeated 5-fold | Tidak |

---

## 5. Workload Detail per Model

### 5.1 Classical CNN (batch_size = 32)

| Stage | Trials | Folds | Epoch/fold | Total Epochs | Batches/epoch | Total Batches | Total Gambar |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 2 | 2 | 4 | 8 | 180 |
| 2 | 1 | 1 | 5 | 5 | 15 | 75 | 2,250 |
| 3 | 20 | 5 | 25 | 2,500 | 29 | 72,500 | 2,250,000 |
| 4 | 30 | 5 | 50 | 7,500 | 57 | 427,500 | 13,500,000 |
| 5 | 1 | 25 | 100 | 2,500 | 57 | 142,500 | 4,500,000 |
| **TOTAL** | | | | **12,507** | | **642,583** | **20,252,430** |

### 5.2 Hybrid QCQ-CNN (batch_size = 8)

| Stage | Trials | Folds | Epoch/fold | Total Epochs | Batches/epoch | Total Batches | Total Gambar |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 2 | 2 | 13 | 26 | 180 |
| 2 | 1 | 1 | 5 | 5 | 57 | 285 | 2,250 |
| 3 | 20 | 5 | 25 | 2,500 | 113 | 282,500 | 2,250,000 |
| 4 | 30 | 5 | 50 | 7,500 | 225 | 1,687,500 | 13,500,000 |
| 5 | 1 | 25 | 100 | 2,500 | 225 | 562,500 | 4,500,000 |
| **TOTAL** | | | | **12,507** | | **2,532,811** | **20,252,430** |

### 5.3 Ringkasan Gabungan

| Metrik | Classical | Hybrid | Total |
|--------|---:|---:|---:|
| Total parameter model | 100,630 | 1,148,562 | — |
| Total epochs | 12,507 | 12,507 | **25,014** |
| Total batches | 642,583 | 2,532,811 | **3,175,394** |
| Total gambar diproses | 20,252,430 | 20,252,430 | **40,504,860** |

---

## 6. Estimasi Waktu per Hardware

### 6.1 Estimasi ms/batch

| Hardware | Classical (bs=32) | Hybrid adjoint (bs=8) |
|----------|---:|---:|
| NVIDIA A100 (40 GB) | ~5 ms | ~30 ms |
| NVIDIA T4 (16 GB) | ~15 ms | ~50 ms |
| NVIDIA L4 (24 GB) | ~10 ms | ~40 ms |
| CPU + lightning.qubit | ~200 ms | ~90 ms |
| CPU + default.qubit (backprop) | ~200 ms | ~410 ms |

> **Catatan:** Hybrid di CPU + lightning.qubit (adjoint) bisa **lebih cepat per batch** dari classical CPU karena circuit 8 qubit sangat kecil (256 element statevector) dan C++ backend sangat efisien. Bottleneck hybrid sebenarnya di **jumlah batch** (4× lebih banyak karena batch_size=8 vs 32).

### 6.2 Estimasi Waktu Total

#### Classical CNN (642,583 batches)

| Hardware | ms/batch | Total | Jam | Hari |
|----------|---:|---:|---:|---:|
| A100 | 5 | 3,213 s | **0.9 jam** | 0.04 |
| T4 | 15 | 9,639 s | **2.7 jam** | 0.11 |
| L4 | 10 | 6,426 s | **1.8 jam** | 0.07 |
| CPU only | 200 | 128,517 s | **35.7 jam** | 1.49 |

#### Hybrid QCQ-CNN — Adjoint (2,532,811 batches)

| Hardware | ms/batch | Total | Jam | Hari |
|----------|---:|---:|---:|---:|
| A100 + lightning.gpu | 30 | 75,984 s | **21.1 jam** | 0.88 |
| T4 + lightning.qubit | 50 | 126,641 s | **35.2 jam** | 1.47 |
| L4 + lightning.qubit | 40 | 101,312 s | **28.1 jam** | 1.17 |
| CPU + lightning.qubit | 90 | 227,953 s | **63.3 jam** | 2.64 |

#### Grand Total (Classical + Hybrid)

| Hardware | Classical | Hybrid | **Grand Total** |
|----------|---:|---:|---:|
| **A100 GPU** | 0.9 jam | 21.1 jam | **22 jam (< 1 hari)** |
| **T4 GPU** | 2.7 jam | 35.2 jam | **38 jam (1.6 hari)** |
| **L4 GPU** | 1.8 jam | 28.1 jam | **30 jam (1.2 hari)** |
| **CPU only** | 35.7 jam | 63.3 jam | **99 jam (4.1 hari)** |

---

## 7. Estimasi Storage

### 7.1 Checkpoint di Google Drive

Per trial disimpan: `best.pt` + `final.pt` = 2 checkpoint.

| Model | Stage 2 | Stage 3 | Stage 4 | Stage 5 | **Total** |
|-------|---:|---:|---:|---:|---:|
| Classical (410 KB/ckpt) | 0.8 MB | 16.0 MB | 24.0 MB | 0.8 MB | **41.6 MB** |
| Hybrid (4.4 MB/ckpt) | 8.8 MB | 176.0 MB | 264.0 MB | 8.8 MB | **457.6 MB** |
| **Combined** | | | | | **~500 MB** |

### 7.2 Checkpoint Lokal (latest.pt per worker)

Setiap worker menyimpan 1 `latest.pt` aktif untuk self-recovery:
- Classical: ~410 KB (negligible)
- Hybrid: ~4.4 MB (negligible)

### 7.3 Dataset di Worker

| Item | Ukuran |
|------|---:|
| Dataset 1,800 gambar grayscale 128×128 PNG | ~15 MB |
| Curriculum manifests (CSV) | ~2 MB |
| Kode Python + dependencies | ~50 MB |
| **Total per worker** | **~70 MB** |

---

## 8. Bottleneck Analysis

```
Distribusi workload Hybrid (total 2,532,811 batches):

Stage 4: ████████████████████████████████████████  1,687,500 (66.6%)
Stage 5: █████████████████                          562,500 (22.2%)
Stage 3: ████████                                   282,500 (11.2%)
Stage 2: ▏                                              285  (0.01%)
Stage 1: ▏                                               26  (0.00%)
```

**Stage 4 Hybrid** = bottleneck terbesar:
- 30 trials × 5 folds × 50 epoch × 225 batches = **1,687,500 batches**
- Dengan T4: ~23.4 jam sendiri
- Bisa diparalelkan: 3 worker → ~7.8 jam per worker

---

## 9. Rekomendasi Eksekusi

### Urutan Optimal

1. **Stage 1 (kedua model)** — Sanity check, < 1 menit
2. **Stage 2 (kedua model)** — Warm start, < 5 menit
3. **Stage 3 Classical → Stage 3 Hybrid** — Atau paralel di 2 worker
4. **Stage 4 Classical → Stage 4 Hybrid** — Classical selesai dalam jam, Hybrid butuh 1+ hari
5. **Stage 5 (kedua model)** — Pakai config terbaik dari Stage 4

### Strategi Paralel

| Worker | Tugas | Estimasi (T4) |
|--------|-------|---:|
| Worker 1 | Classical Stage 1→5 (semua) | 2.7 jam |
| Worker 2 | Hybrid Stage 1→3 | 5.5 jam |
| Worker 3 | Hybrid Stage 4 (trial 1-15) | 11.7 jam |
| Worker 4 | Hybrid Stage 4 (trial 16-30) | 11.7 jam |
| Worker 2 | Hybrid Stage 5 (setelah Stage 3) | 7.8 jam |
| **Total wallclock** | | **~20 jam** |

### Tanpa GPU (CPU + lightning.qubit adjoint)

| Worker | Tugas | Estimasi |
|--------|-------|---:|
| 1 PC | Semua (serial) | ~4.1 hari |
| 2 PC | Classical + Hybrid paralel | ~2.6 hari |
| 3 PC | Split Stage 4 Hybrid | ~1.8 hari |
