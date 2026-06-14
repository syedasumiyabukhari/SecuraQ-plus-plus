# QEGVD — Quantum-Enhanced Graph-Based Vulnerability Detection

End-to-end hybrid quantum-classical pipeline for binary vulnerability detection
in C/C++ source code. Detects **Buffer Overflow (CWE-121/122)**, **Format String
(CWE-134)**, and **Use-After-Free (CWE-416)** using the Juliet Test Suite.

---

## Project Structure

```
QEGVD/
├── configs/
│   └── config.yaml                   ← all hyperparameters
├── data/
│   ├── raw/                          ← place the 3 sanitized CSV files here
│   ├── processed/{bo,fs,uaf}/        ← Stage 1 output
│   ├── graphs/{bo,fs,uaf}/           ← Stage 2 output
│   ├── embeddings/{bo,fs,uaf}/       ← Stage 3 output  (N, 128)
│   ├── compressed/{bo,fs,uaf}/       ← Stage 4 output  (N, 32)
│   ├── qafa/{bo,fs,uaf}/             ← Stage 5 output  (N, 8) x2
│   ├── quantum/{bo,fs,uaf}/          ← Stage 6 output  (N, 4)
│   └── hybrid/{bo,fs,uaf}/           ← Stage 7+8 output (N, 36)
├── models/
│   ├── checkpoints/                  ← per-stage best checkpoints
│   └── final/                        ← final hybrid models
├── results/
│   ├── metrics/                      ← JSON metrics per stage
│   └── explanations/{bo,fs,uaf}/     ← SHAP reports
├── src/
│   ├── stage1_preprocessing.py       ← data cleaning + splitting
│   ├── stage2_graph_construction.py  ← 7-graph static analysis
│   ├── stage3_gat.py                 ← multi-view GAT (128-dim)
│   ├── stage4_classical_encoder.py   ← 128 → 64 → 32
│   ├── stage5_qafa.py                ← QAFA feature selection
│   ├── stage6_vqc.py                 ← 4-qubit VQC (PennyLane)
│   ├── stage7_fusion.py              ← fusion + MLP classifier
│   ├── stage9_explainability.py      ← SHAP + GNNExplainer
│   ├── inference.py                  ← end-to-end inference
│   └── utils/
│       ├── audit.py                  ← 15-point leakage checker
│       └── metrics.py                ← F1, MCC, AUC, FPR/FNR
├── logs/                             ← per-stage log files
├── requirements.txt
├── run_pipeline.sh                   ← one-command full run
└── README.md
```

---

## Pipeline Overview

```
Raw CSV
  │
  ▼ Stage 1 ─── Leakage audit · Dedup · Stratified 70/15/15 split
  │
  ▼ Stage 2 ─── Static analysis → 7 graphs per sample
  │               AST · CFG · DFG · PDG · TPG · MAG · CG
  │               Each node: 64-dim feature vector
  │
  ▼ Stage 3 ─── Multi-view GAT
  │               4-layer GATConv (heads=8) per graph
  │               Mean+Max readout → attention fusion
  │               Output: (N, 128)
  │
  ▼ Stage 4 ─── Classical Encoder
  │               128 → FC(64) → ReLU+BN+Dropout(0.3)
  │                   → FC(32) → ReLU+BN+Dropout(0.3)
  │               Output: (N, 32)
  │
  ▼ Stage 5 ─── QAFA  (Quantum-Aware Feature Alignment)
  │               Composite score: S_i = 0.40·MI + 0.35·SHAP + 0.25·Centrality
  │               Top-16 selected → split into Stage1(8) + Stage2(8)
  │               tanh(x)·π → angles ∈ [-π, π]
  │               Output: (N,8) + (N,8)
  │
  ▼ Stage 6 ─── VQC  (4 qubits, PennyLane)
  │               H⁴ → [RY/RZ s1] → CNOT-ring
  │                  → Var_A(×3) → [RY/RZ s2] → CNOT-ring → Var_B(×3)
  │                  → ⟨Z₀⟩,⟨Z₁⟩,⟨Z₂⟩,⟨Z₃⟩
  │               Output: (N, 4)
  │
  ▼ Stage 7 ─── Residual Hybrid Fusion
  │               h_hybrid = concat(classical[32], quantum[4]) → 36-dim
  │
  ▼ Stage 8 ─── MLP Classifier
  │               36 → FC(16) → ReLU → Dropout(0.2) → FC(1) → Sigmoid
  │               Loss: Focal(γ=2.0, α=0.25)
  │               Output: P(vulnerable) ∈ [0,1]
  │
  ▼ Stage 9 ─── Explainability
                  SHAP KernelExplainer on 36-dim hybrid vector
                  GNNExplainer via degree+betweenness+PageRank centrality
                  Output: per-prediction structured vulnerability report
```

---

## Datasets

Place the three files in `data/raw/`:

| File | Vulnerability | Rows | Balance |
|------|--------------|------|---------|
| `bo_dataset_sanitized.csv` | Buffer Overflow (CWE-121/122) | 11,417 | 50.3/49.7% |
| `fs_dataset_sanitized.csv` | Format String (CWE-134) | 5,390 | 50.1/49.9% |
| `uaf_dataset_sanitized.csv` | Use-After-Free (CWE-416) | 891 | 48.1/51.9% |

---

## Installation

```powershell
# 1. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate           # Windows PowerShell
# source venv/bin/activate      # Linux/Mac

# 2. Install PyTorch (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install PyTorch Geometric
pip install torch-geometric

# 4. Install remaining dependencies
pip install pennylane shap scikit-learn scipy pyyaml numpy pandas tqdm

# 5. Optional (better SHAP performance)
pip install lightgbm
```

---

## Running the Full Pipeline

### Option A — Run all stages manually (recommended)

```powershell
# Navigate to project root
cd E:\fyp\givingUpVersion

# Stage 1: Data cleaning and splitting
python src/stage1_preprocessing.py --dataset all

# Stage 2: Graph construction (7 graphs per sample)
python src/stage2_graph_construction.py --dataset all

# Stage 3: GAT training (produces 128-dim embeddings)
python src/stage3_gat.py --dataset all

# Stage 4: Classical encoder (128 -> 32)
python src/stage4_classical_encoder.py --dataset all

# Stage 5: QAFA feature selection (32 -> top-16 -> 2x8 angles)
python src/stage5_qafa.py --dataset all

# Stage 6: VQC training (4-dim quantum vectors) -- SLOW on CPU
python src/stage6_vqc.py --dataset all

# Stage 7+8: Hybrid fusion + MLP classifier (36-dim -> binary)
python src/stage7_fusion.py --dataset all

# Stage 9: Explainability (SHAP + GNNExplainer reports)
python src/stage9_explainability.py --dataset all
```

### Option B — Run a single dataset

```powershell
# Replace 'all' with 'bo', 'fs', or 'uaf'
python src/stage1_preprocessing.py --dataset bo
python src/stage2_graph_construction.py --dataset bo
python src/stage3_gat.py --dataset bo
python src/stage4_classical_encoder.py --dataset bo
python src/stage5_qafa.py --dataset bo
python src/stage6_vqc.py --dataset bo
python src/stage7_fusion.py --dataset bo
python src/stage9_explainability.py --dataset bo
```

### Option C — One-command bash runner (Linux/Mac/Git Bash)

```bash
bash run_pipeline.sh all
# or for one dataset:
bash run_pipeline.sh bo
```

---

## Stage-by-Stage Details

### Stage 1 — Preprocessing
```powershell
python src/stage1_preprocessing.py --dataset all
```
- Runs 15-point leakage audit
- Removes whitespace, filters by length (10–8000 chars)
- MD5 exact dedup + near-dedup
- Stratified 70/15/15 split
- Output: `data/processed/{bo,fs,uaf}/{train,val,test}.csv`

### Stage 2 — Graph Construction
```powershell
python src/stage2_graph_construction.py --dataset all
```
- Pure Python static analyser (no tree-sitter required)
- Builds 7 NetworkX DiGraphs per sample: AST, CFG, DFG, PDG, TPG, MAG, CG
- Node features: 64-dim float32 vector
- Output: `data/graphs/{bo,fs,uaf}/{train,val,test}.pkl`

### Stage 3 — GAT Embedding
```powershell
python src/stage3_gat.py --dataset all
# Skip training, load checkpoint only:
python src/stage3_gat.py --dataset all --eval-only
```
- 4-layer GATConv with 8 heads per graph view
- Mean+Max global readout → attention-weighted 7-view fusion
- Output: `data/embeddings/{bo,fs,uaf}/{train,val,test}.npy`  shape `(N, 128)`

### Stage 4 — Classical Encoder
```powershell
python src/stage4_classical_encoder.py --dataset all
python src/stage4_classical_encoder.py --dataset all --eval-only
```
- Architecture: `128 → FC(64) → ReLU+BN+Dropout(0.3) → FC(32) → ReLU+BN+Dropout(0.3)`
- Output: `data/compressed/{bo,fs,uaf}/{train,val,test}.npy`  shape `(N, 32)`

### Stage 5 — QAFA
```powershell
python src/stage5_qafa.py --dataset all
```
- Composite importance: `S_i = 0.40·MI + 0.35·SHAP + 0.25·Centrality`
- Selects top-16 from 32 features
- Splits into Stage1 (top-8) + Stage2 (next-8)
- Rescales to `[-π, π]` via `tanh(x)·π`
- Output: `data/qafa/{bo,fs,uaf}/{split}_stage{1,2}.npy`  shape `(N, 8)`

### Stage 6 — VQC  ⚠️ SLOW
```powershell
python src/stage6_vqc.py --dataset all
# After first training, skip to extraction only:
python src/stage6_vqc.py --dataset all --eval-only
```
> **Warning:** VQC on CPU takes ~0.5–2 seconds per sample. For 11K BO samples
> this can take several hours. Use `--eval-only` after first successful training.
>
> The circuit is: `H⁴ → [RY/RZ s1] → CNOT-ring → Var(×3) → [RY/RZ s2] → CNOT-ring → Var(×3) → ⟨Z⟩⁴`
>
> Output: `data/quantum/{bo,fs,uaf}/{split}_qvec.npy`  shape `(N, 4)`

### Stage 7+8 — Hybrid Fusion + Classifier
```powershell
python src/stage7_fusion.py --dataset all
python src/stage7_fusion.py --dataset all --eval-only
```
- Fusion: `concat(classical[32], quantum[4]) → 36-dim`  (no learnable params)
- Classifier: `36 → FC(16) → ReLU → Dropout(0.2) → FC(1) → Sigmoid`
- Loss: Focal Loss `γ=2.0, α=0.25`
- Optimizer: AdamW `lr=1e-3, wd=1e-4` with cosine annealing
- Early stopping: patience=15
- Output: `data/hybrid/{bo,fs,uaf}/{split}_hybrid.npy`  shape `(N, 36)`
- Models: `models/final/{bo,fs,uaf}_hybrid_best.pt`

### Stage 9 — Explainability
```powershell
# Explain 50 test samples (TP/TN/FP/FN mix)
python src/stage9_explainability.py --dataset all

# Explain a specific sample by index
python src/stage9_explainability.py --dataset bo --sample-id 42

# Control how many samples to explain
python src/stage9_explainability.py --dataset bo --n-explain 100
```
- SHAP KernelExplainer on 36-dim hybrid vector
- GNNExplainer via graph centrality (degree + betweenness + PageRank)
- Output: `results/explanations/{bo,fs,uaf}/test_reports.json`

### Inference — New Code
```powershell
# Analyse a code snippet
python src/inference.py --code "void foo(char *s){char buf[10]; strcpy(buf,s);}"

# Analyse a file
python src/inference.py --file path/to/function.c

# With SHAP attribution
python src/inference.py --file path/to/function.c --explain

# Save JSON output
python src/inference.py --file func.c --explain --output result.json
```

---

## Eval-Only Mode (resume after crash)

Every stage that trains a model supports `--eval-only` to skip training and
load the saved checkpoint directly:

```powershell
python src/stage3_gat.py --dataset bo --eval-only
python src/stage4_classical_encoder.py --dataset bo --eval-only
python src/stage6_vqc.py --dataset bo --eval-only
python src/stage7_fusion.py --dataset bo --eval-only
```

---

## Output Files Reference

| Path | Shape | Produced by |
|------|-------|-------------|
| `data/processed/<ds>/{split}.csv` | N rows | Stage 1 |
| `data/graphs/<ds>/{split}.pkl` | N GraphBundles | Stage 2 |
| `data/embeddings/<ds>/{split}.npy` | (N, 128) | Stage 3 |
| `data/compressed/<ds>/{split}.npy` | (N, 32) | Stage 4 |
| `data/qafa/<ds>/{split}_stage1.npy` | (N, 8) | Stage 5 |
| `data/qafa/<ds>/{split}_stage2.npy` | (N, 8) | Stage 5 |
| `data/quantum/<ds>/{split}_qvec.npy` | (N, 4) | Stage 6 |
| `data/hybrid/<ds>/{split}_hybrid.npy` | (N, 36) | Stage 7+8 |
| `models/checkpoints/<ds>_gat_best.pt` | — | Stage 3 |
| `models/checkpoints/<ds>_encoder_best.pt` | — | Stage 4 |
| `models/checkpoints/<ds>_vqc_best.pt` | — | Stage 6 |
| `models/final/<ds>_hybrid_best.pt` | — | Stage 7+8 |
| `results/metrics/<ds>_stage8_test.json` | — | Stage 7+8 |
| `results/explanations/<ds>/test_reports.json` | — | Stage 9 |

---

## Key Hyperparameters (configs/config.yaml)

| Parameter | Value | Section |
|-----------|-------|---------|
| GAT hidden dim | 128 | Stage 3 |
| GAT attention heads | 8 | Stage 3 |
| GAT layers | 4 | Stage 3 |
| Encoder: 128→64→32 | dropout=0.3, BN | Stage 4 |
| QAFA weights α/β/γ | 0.40/0.35/0.25 | Stage 5 |
| VQC qubits | 4 | Stage 6 |
| VQC var layers | 6 | Stage 6 |
| Hybrid dim | 36 (32+4) | Stage 7 |
| Classifier hidden | 16 | Stage 8 |
| Focal Loss γ/α | 2.0/0.25 | Stage 8 |
| Optimizer | AdamW lr=1e-3 | Stages 3,4,6,8 |
| Early stopping | patience=15 | All |
| Grad clip | 1.0 | All |

---

## Troubleshooting

**`AttributeError: Can't get attribute 'GraphBundle'`**
→ Already patched in `stage3_gat.py`. If it appears elsewhere, add before `pickle.load()`:
```python
import stage2_graph_construction as _s2
import sys; sys.modules["__main__"].GraphBundle = _s2.GraphBundle
```

**`FileNotFoundError: Missing data/quantum/bo/train_qvec.npy`**
→ Run stages in order. Each stage depends on the previous one's output.

**Stage 6 VQC is very slow**
→ Normal on CPU. VQC is ~0.5–2s per sample. Run Stage 6 overnight or use
`--eval-only` if you already have a checkpoint from a previous run.

**SHAP not installed (Stage 5 / Stage 9)**
→ Falls back automatically to permutation importance (Stage 5) and gradient×input
attribution (Stage 9). Install with `pip install shap` for full SHAP support.

**`UnicodeEncodeError` on Windows**
→ All logging handlers are forced to UTF-8. If you still see issues, set:
```powershell
$env:PYTHONIOENCODING="utf-8"
```
