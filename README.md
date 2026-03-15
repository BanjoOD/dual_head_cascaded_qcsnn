# A Cascaded Quantized Spiking Neural Network for Real-Time ECG Arrhythmia Detection on Edge Hardware


This repository implements an end-to-end **Quantized Convolution Spiking Neural Network (QCSNN)** for ECG arrhythmia classification on the **MIT-BIH Arrhythmia Database**, with hardware deployment on a **PYNQ-Z2 FPGA**.

The QCSNN uses a **dual-head architecture** (binary + 4-class heads) jointly trained via surrogate gradient descent and quantization-aware training (QAT), then converted to a **two-stage cascaded inference pipeline** with hardware-resident early termination for efficient edge deployment.

---

## Key Results

| Metric | GPU (RTX A2000) | FPGA (PYNQ-Z2) |
|--------|----------------|-----------------|
| System Accuracy | 99.02% | 98.61% |
| Stage-1 Macro F1 | 93.22% | 92.51% |
| Stage-2 Macro F1 | 91.10% | 89.63% |
| Power | 63.96 W | 2.02 W (0.33 W accelerator) |
| Latency | 1.46 ms/beat | 11.54 ms/beat |
| Energy | 93.55 mJ/inf | 23.31 mJ/inf |

**31.66× power reduction** and **4.01× energy reduction** versus GPU, within 1.5% macro F1.

### Loss–RR Interaction Finding

A factorial ablation across 8 configurations ({CE, FL} × {No RR, RR→S1, RR→S2, RR→Both}) reveals that:
- Without RR features, Focal Loss outperforms Cross-Entropy by ~2.5% macro F1
- With RR features, this advantage **closes or reverses**
- RR features supply at the input level the discriminative information that Focal Loss recovers at the optimization level

---

## Repository Structure

```
dual_head_cascaded_qcsnn/
├── notebooks                   # PyTorch training and evaluation
│                               # Dual-head QCSNN joint training + weight export
│                               # Cascaded inference evaluation + ablation
├── csnn_cpp/                   # C++/HLS implementation
│   ├── include/                # Layer implementations (Conv1D, LIF, Linear, BatchNorm)
|   |   ├── hls4csnn1d
|   |       ├── model24
│   |       ├── weights_sd/     # Exported quantized weights (from PyTorch)
│   └── src/                    # HLS simulation testbench
└── README.md
```

---

## End-to-End Workflow

### Phase 1: Model Development (PyTorch)

**Requirements:** PyTorch, [snnTorch](https://github.com/jeshraghian/snntorch), [Brevitas](https://github.com/Xilinx/brevitas), NumPy, scikit-learn

1. **Train the dual-head QCSNN for baseline**
   - Open the training notebook in `notebooks/`
   - The model jointly trains binary (Normal vs Abnormal) and 4-class (N/SVEB/VEB/F) heads on a shared convolutional backbone
   - Training uses surrogate gradient descent with QAT (INT8 via Brevitas + snnTorch)
   - Export quantized weights

2. **Run ablation study (optional)**
   - Evaluate all 8 configurations: {CE, FL} × {RR→S1, RR→S2, RR→Both}
   - Generate per-class metrics, confusion matrices, and interaction analysis

3. **Copy exported weights to C++ project**
   - Move exported weight files to `csnn_cpp/weights_sd/`

### Phase 2: Model Acceleration (C++/HLS → Vivado → PYNQ-Z2)

**Requirements:** AMD Vitis HLS 2025.1, AMD Vivado 2025.1, PYNQ-Z2 board

#### A. HLS Synthesis (Vitis)

1. Create an HLS Component in Vitis Unified IDE
2. Add C++ sources from `csnn_cpp/src/` with `topFunction` as the top-level entry
3. Run the HLS flow:
   - **C Simulation** — functional verification against Python reference
   - **C Synthesis** — generate RTL from C++
   - **C/RTL Co-simulation** — verify RTL matches C++ (recommended)
4. Package/export the synthesized component as an IP for Vivado

#### B. Vivado Block Design

Create a block design with the following IP blocks:

| IP Block | Role |
|----------|------|
| Zynq-7000 Processing System | ARM PS running PYNQ Linux |
| Custom QCSNN IP (from HLS) | Combined 2-stage spiking inference engine |
| AXI DMA (×2) | Stream ECG data in, predictions out (Stage-1 + Stage-2) |
| AXI Interconnect / SmartConnect | Route AXI-Lite control between PS and IP |
| AXI Memory Interconnect | Connect DMA to DDR via HP ports |
| Processor System Reset | Reset sequencing for all AXI logic |
| xlconstant | Tie-off for unused/constant signals |

**Dataflow:**
1. Zynq PS (Python/PYNQ) configures AXI DMA and loads ECG segments into DDR
2. DMA (MM2S) streams ECG data from DDR → QCSNN IP via AXI4-Stream
3. QCSNN IP performs Stage-1 binary classification
4. If Normal → early termination on PL (no PS involvement)
5. If Abnormal → Stage-2 four-class classification within the same IP core
6. DMA (S2MM) streams predictions back to DDR
7. PS reads predictions and computes metrics

**Generate outputs:**
- Validate block design (no connection/width errors)
- Run Synthesis → Implementation → Generate Bitstream (`.bit`)
- Export hardware handoff (`.hwh`)

#### C. PYNQ-Z2 Deployment

1. Copy `.bit` and `.hwh` to the PYNQ-Z2 board
2. From a PYNQ Jupyter notebook:
   ```python
   from pynq import Overlay, allocate
   overlay = Overlay('path/to/qcsnn.bit')
   dma = overlay.axi_dma_0
   # Allocate buffers, stream ECG data, collect predictions
   ```
3. Measure accuracy, latency, throughput, and power

---

## Model Architecture

```
ECG Signal (180 samples) + RR Features (4-dim)
         │
         ▼
┌─────────────────────────────┐
│   Shared Backbone           │
│   3× QuantConv1D + BN +     │
│   LIF + MaxPool             │
│   → 480-dim features        │
└──────────┬──────────────────┘
           │ + RR features (4-dim) → 484-dim
           ├──────────────────┐
           ▼                  ▼
┌──────────────────┐ ┌──────────────────┐
│  Binary Head     │ │  4-Class Head    │
│  Linear(484→2)   │ │  Linear(484→128) │
│  + LIF           │ │  + LIF           │
│  Normal vs Abn   │ │  Linear(128→4)   │
│                  │ │  + LIF           │
│                  │ │  N/SVEB/VEB/F    │
└──────────────────┘ └──────────────────┘
```

- **Parameters:** 65,539 (INT8 quantized, 0.064 MB)
- **LIF neurons:** β=0.5, threshold=0.5 (both learnable)
- **Surrogate gradient:** Fast sigmoid, slope=25
- **Quantization:** 8-bit weights and activations via Brevitas QAT

---

## Cascaded Inference

During deployment, the dual-head model is reorganized into a cascaded pipeline:

```
ECG Beat → [Stage-1: Binary] → Normal? ──Yes──→ Exit (early termination)
                                  │
                                  No
                                  │
                                  ▼
                         [Stage-2: 4-Class]
                          N / SVEB / VEB / F
```

- Cascade routing is implemented **entirely on the FPGA PL** (no PS-in-the-loop)
- Both stages execute within a **single HLS IP core** (`topFunction`)
- Single bitstream, no partial reconfiguration
- Majority of beats (Normal) exit early → lower average latency and energy

---

## Hardware Resource Utilization (PYNQ-Z2, Zynq-7020)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUT | 19,494 | 53,200 | 36.80% |
| FF | 29,630 | 106,400 | 22.22% |
| DSP | 220 | 220 | **100%** |
| BRAM | 94 | 140 | 67.14% |

Clock frequency: 100 MHz

---

## Reproducibility Checklist

1. ☐ Install Python environment (PyTorch, snnTorch, Brevitas)
2. ☐ Download MIT-BIH Arrhythmia Database
3. ☐ Run dual-head QCSNN joint training notebook with QAT
4. ☐ Export quantized weights to `csnn_cpp/weights_sd/`
5. ☐ Build and verify C++ implementation (CPU simulation)
6. ☐ Run Vitis HLS: C Simulation → C Synthesis → Export IP
7. ☐ Build Vivado block design and generate bitstream
8. ☐ Deploy to PYNQ-Z2 and validate against GPU baseline


%\iffalse
%---
%
%## Citation
%
%If you use this code in your research, please cite:
%
%```bibtex
%@article{banjo2026qcsnn,
%  author  = {Olamilekan Banjo and Behnaz Ghoraani},
%  title   = {A Cascaded Quantized Spiking Neural Network for Real-Time ECG Arrhythmia Detection on Edge Hardware},
%  journal = {Biomedical Signal Processing and Control},
%  year    = {2026}
%}
%```
%
%---
%
%## License
%
%This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
%
%## Acknowledgments
%
%Department of Electrical Engineering and Computer Science, Florida Atlantic University.
%\fi
