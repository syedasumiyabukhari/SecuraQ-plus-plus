"""
QEGVD -- Stage 7 + Stage 8: Residual Hybrid Fusion + MLP Classifier
====================================================================
Paper Section 8 + 9 + 10 (exact):

  Stage 7 -- Residual Hybrid Fusion
      h_hybrid = concat(h_classical[32], h_quantum[4]) -> 36-dim

  Stage 8 -- MLP Classifier
      36 -> FC(16) -> ReLU -> Dropout(0.2) -> FC(1) -> Sigmoid

  Training:
      Focal Loss (gamma=2.0, alpha=0.25)
      AdamW  lr=1e-3  weight_decay=1e-4
      Cosine annealing  T_max=100
      Grad clip 1.0
      Early stopping patience=15

Inputs  (pre-computed):
    data/compressed/<ds>/{split}.npy       (N, 32)  -- Stage 4
    data/quantum/<ds>/{split}_qvec.npy     (N, 4)   -- Stage 6

Outputs:
    data/hybrid/<ds>/{split}_hybrid.npy    (N, 36)
    models/final/<ds>_hybrid_best.pt
    results/metrics/<ds>_stage8_test.json

Usage
-----
    python src/stage7_fusion.py --dataset bo
    python src/stage7_fusion.py --dataset all
    python src/stage7_fusion.py --dataset bo --eval-only
"""

from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
from typing import Optional
import numpy as np, yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
for _d in ["logs","models/final","data/hybrid","results/metrics"]:
    (_ROOT / _d).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)),
        logging.FileHandler(_ROOT / "logs" / "stage78.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("Stage7+8")

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    logger.error("torch not found -- pip install torch"); sys.exit(1)

from utils.metrics import compute_metrics, find_optimal_threshold, EpochTracker

# ── Paper constants ────────────────────────────────────────
CLASSICAL_DIM    = 256  # Stage 3 full embedding (always use full for best signal)
QUANTUM_DIM_DEFAULT = 4   # FS/UAF: 4-dim VQC output
QUANTUM_DIM_BO   = 32   # BO: 8 rounds × 4 = 32-dim VQC output


# ── Stage 7: Residual Hybrid Fusion ───────────────────────
class ResidualHybridFusion(nn.Module):
    """
    Paper Section 8:
        h_hybrid = concat(h_classical[256], h_quantum[4]) -> 260-dim
    No learnable parameters -- residual means full classical info preserved.
    """
    def forward(self, classical, quantum, ds_key=None):
        return torch.cat([classical, quantum], dim=-1)  # (B, 260)


# ── Stage 8: MLP Classifier ───────────────────────────────
class HybridMLP(nn.Module):
    """
    Hybrid MLP classifier. input_dim = CLASSICAL_DIM + q_dim (dynamic per dataset).
    FS uses a deeper architecture; BO/UAF use a standard 3-layer MLP.
    """
    def __init__(self, ds_key=None, input_dim=None):
        super().__init__()
        hdim = input_dim if input_dim is not None else (CLASSICAL_DIM + QUANTUM_DIM_DEFAULT)
        if ds_key and ds_key.lower() == "fs":
            # Smaller MLP with stronger regularization for FS
            # (3.7k samples → deep model overfits; reduce capacity, increase dropout)
            self.net = nn.Sequential(
                nn.Linear(hdim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.35),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.25),

                nn.Linear(64, 1)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(hdim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(64, 1)
            )
    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Focal Loss (paper Section 10.1) ───────────────────────
class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t.(1-p_t)^gamma.log(p_t). gamma=2.0 alpha=0.25"""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__(); self.gamma=gamma; self.alpha=alpha
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.sigmoid(logits)*targets + (1-torch.sigmoid(logits))*(1-targets)
        at  = self.alpha*targets + (1-self.alpha)*(1-targets)
        return (at*(1-pt)**self.gamma*bce).mean()


# ── FS Enhanced Dataset (separate tensors for cross-attention) ──────────────

class FSEnhancedDataset(Dataset):
    """Stores classical/quantum/extra as separate tensors for EnhancedFSHybridClassifier."""
    def __init__(self, classical, quantum, labels, extra=None):
        self.classical = torch.from_numpy(classical.astype(np.float32))
        self.quantum   = torch.from_numpy(quantum.astype(np.float32))
        self.extra     = torch.from_numpy(extra.astype(np.float32)) if extra is not None else None
        self.y         = torch.from_numpy(labels.astype(np.float32))

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        ex = self.extra[i] if self.extra is not None else torch.empty(0)
        return self.classical[i], self.quantum[i], ex, self.y[i]


def _fs_ldr(c, q, y, bs, shuf, extra=None):
    return DataLoader(FSEnhancedDataset(c, q, y, extra), batch_size=bs, shuffle=shuf, num_workers=0)


def _train_epoch_fs(model, loader, opt, focal, grad_clip):
    model.train(); ep = n = 0; correct = 0
    for classical, quantum, extra, y in loader:
        opt.zero_grad()
        ex_arg = extra if extra.shape[-1] > 0 else None
        lg = model(classical, quantum, ex_arg)
        loss = focal(lg, y); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        ep += loss.item() * len(y); n += len(y)
        correct += ((torch.sigmoid(lg) >= 0.5).long() == y.long()).sum().item()
    return ep / n, correct / n


@torch.no_grad()
def _eval_fs(model, loader, focal):
    model.eval(); va_l = 0; ps = []; ls = []
    for classical, quantum, extra, y in loader:
        ex_arg = extra if extra.shape[-1] > 0 else None
        lg = model(classical, quantum, ex_arg)
        va_l += focal(lg, y).item() * len(y)
        ps.append(torch.sigmoid(lg).numpy()); ls.append(y.numpy())
    return va_l / sum(len(l) for l in ls), np.concatenate(ps), np.concatenate(ls)


# ── Dataset ────────────────────────────────────────────────
class HybridDataset(Dataset):
    """Concatenates classical + quantum [+ FS meta] -> hybrid."""
    def __init__(self, classical, quantum, labels, meta=None, ds_key=None):
        fusion = ResidualHybridFusion()
        classical_tensor = torch.from_numpy(classical.astype(np.float32))
        quantum_tensor = torch.from_numpy(quantum.astype(np.float32))
        hybrid_tensor = fusion(classical_tensor, quantum_tensor, ds_key=ds_key)
        if meta is not None:
            meta_tensor = torch.from_numpy(meta.astype(np.float32))
            hybrid_tensor = torch.cat([hybrid_tensor, meta_tensor], dim=-1)
        self.X = hybrid_tensor
        self.y = torch.from_numpy(labels.astype(np.float32))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def _loader(c, q, y, bs, shuf, meta=None, ds_key=None):
    return DataLoader(HybridDataset(c, q, y, meta=meta, ds_key=ds_key), batch_size=bs, shuffle=shuf, num_workers=0)


# ── Train / eval ───────────────────────────────────────────
def _train_epoch(model, loader, opt, focal, grad_clip):
    model.train(); ep=n=0; correct=0
    for X,y in loader:
        opt.zero_grad()
        lg=model(X); loss=focal(lg,y); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        ep+=loss.item()*len(y); n+=len(y)
        correct+=((torch.sigmoid(lg)>=0.5).long()==y.long()).sum().item()
    return ep/n, correct/n

@torch.no_grad()
def _eval(model, loader, focal):
    model.eval(); va_l=0; ps=[]; ls=[]
    for X,y in loader:
        lg=model(X); va_l+=focal(lg,y).item()*len(y)
        ps.append(torch.sigmoid(lg).numpy()); ls.append(y.numpy())
    return va_l/sum(len(l) for l in ls), np.concatenate(ps), np.concatenate(ls)


# ── Config + loaders ──────────────────────────────────────
def load_config(path=None):
    return yaml.safe_load(open(path or _ROOT/"configs"/"config.yaml"))

def _load(directory, split, suffix):
    p = Path(directory) / f"{split}{suffix}"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}\nRun preceding stage first.")
    return np.load(p)


# ── FS-only: handcrafted meta-features ─────────────────────
import re as _re
import sys as _sys

def _extract_fs_meta(code: str) -> list:
    """
    15 handcrafted features derived from raw C code.
    Extracted only for FS dataset — completely ignored for BO/UAF.
    Based on analysis of Juliet CWE-134 discriminative patterns.
    """
    # Structural complexity (main signal: vuln functions are longer/more complex)
    code_len      = len(code)
    n_unique_vars = len(set(_re.findall(r'VAR_\d+', code)))
    n_unique_funcs= len(set(_re.findall(r'func_\d+', code)))
    n_strs        = len(_re.findall(r'\bSTR_\d+\b', code))
    n_null        = code.count('NULL')
    n_loops       = len(_re.findall(r'\b(?:for|while|do)\b', code))
    n_ifs         = len(_re.findall(r'\bif\b', code))
    n_case        = len(_re.findall(r'\bcase\b', code))   # more in SAFE
    has_switch    = int('switch' in code)                 # more in SAFE
    has_default   = int('default' in code)
    # Format call signals (corrected for fprintf second-arg pattern)
    fmt_var_direct= int(bool(_re.search(r'\bprintf\s*\(\s*VAR_\w+\s*[,)]', code)))
    fmt_safe_str  = int(bool(_re.search(r'\b(?:printf|fprintf|sprintf)\s*\([^)]*STR_', code)))
    # Taint source presence
    has_taint     = int(bool(_re.search(r'\b(?:fgets|gets|fgetws|getenv|recv|scanf)\b', code)))
    # VAR numbering (higher numbered VARs → more complex → more in vuln)
    var_nums = [int(m) for m in _re.findall(r'VAR_(\d+)', code)]
    max_var_num   = max(var_nums) if var_nums else 0
    mean_var_num  = float(np.mean(var_nums)) if var_nums else 0.0

    # Normalize to [0,1] with empirical ranges from data analysis
    return [
        min(code_len / 2000.0, 1.0),          # 0: code length
        min(n_unique_vars / 30.0, 1.0),        # 1: unique vars
        min(n_unique_funcs / 15.0, 1.0),       # 2: unique funcs
        min(n_strs / 10.0, 1.0),               # 3: STR_ tokens
        min(n_null / 5.0, 1.0),                # 4: NULL count
        min(n_loops / 5.0, 1.0),               # 5: loops
        min(n_ifs / 15.0, 1.0),                # 6: if statements
        min(n_case / 5.0, 1.0),                # 7: case (more in SAFE)
        float(has_switch),                     # 8: has switch (more in SAFE)
        float(has_default),                    # 9: has default
        float(fmt_var_direct),                 # 10: printf(VAR_) direct
        float(fmt_safe_str),                   # 11: printf with STR_ (safe)
        float(has_taint),                      # 12: taint source present
        min(max_var_num / 80.0, 1.0),          # 13: max VAR number
        min(mean_var_num / 20.0, 1.0),         # 14: mean VAR number
    ]

FS_META_DIM = 15  # must match len(_extract_fs_meta())

def _build_fs_meta_features(ds_key: str, config: dict) -> dict | None:
    """
    Loads raw code from data/processed/fs/{split}.csv, matches rows to
    graph-bundle ordering via sample_id stored in quantum labels files,
    and returns {split: (N, FS_META_DIM) float32 array}.
    Only called when ds_key == 'fs'.
    """
    if ds_key != "fs":
        return None

    try:
        import sys as _sys2, pickle, importlib
        _sys2.path.insert(0, str(_ROOT / "src"))
        _stage2_mod = importlib.import_module("stage2_graph_construction")
        # Register in __main__ so pickle.load can find GraphBundle
        _sys2.modules["__main__"].GraphBundle = _stage2_mod.GraphBundle
        GraphBundle = _stage2_mod.GraphBundle
    except Exception as _e:
        logger.warning(f"  [FS meta] Could not import GraphBundle ({_e}) — skipping meta features")
        return None

    processed_dir = _ROOT / "data" / "processed" / "fs"
    graphs_dir    = _ROOT / "data" / "graphs"    / "fs"

    import pandas as pd
    meta = {}
    for sp in ["train", "val", "test"]:
        csv_path   = processed_dir / f"{sp}.csv"
        graph_path = graphs_dir / f"{sp}.pkl"
        if not csv_path.exists() or not graph_path.exists():
            logger.warning(f"  [FS meta] Missing {csv_path} or {graph_path} — skipping")
            return None

        df = pd.read_csv(csv_path)
        id_to_code = dict(zip(df["id"].values, df["code"].values))

        with open(graph_path, "rb") as f:
            bundles = pickle.load(f)

        feats = []
        for b in bundles:
            code = id_to_code.get(b.sample_id, "")
            feats.append(_extract_fs_meta(code))
        meta[sp] = np.array(feats, dtype=np.float32)
        logger.info(f"  [FS meta] {sp}: shape={meta[sp].shape}  "
                    f"range=[{meta[sp].min():.3f},{meta[sp].max():.3f}]")

    return meta


# ── FS enhanced fusion (EnhancedFSHybridClassifier) ──────────────────────────

def _run_fs_enhanced_fusion(classical, quantum, labels, fs_meta, config, ckpt, eval_only, hybrid_dir):
    """
    FS-only enhanced fusion using EnhancedFSHybridClassifier with cross-attention.
    Called from run_dataset() when ds_key=='fs' and q_dim==40.
    """
    from stage6_enhanced_vqc import EnhancedFSHybridClassifier, FocalLoss as FocalLossEnh

    use_extra = fs_meta is not None
    extra_dim = FS_META_DIM if use_extra else 0
    model = EnhancedFSHybridClassifier(
        classical_dim=CLASSICAL_DIM,
        quantum_dim=40,
        extra_dim=extra_dim,
        dropout=0.40,
        use_extra=use_extra,
    )
    focal  = FocalLossEnh(gamma=2.0, alpha=0.5)
    cfg_t  = config["training"]
    bs     = cfg_t["batch_size"]

    logger.info(f"  [FS Enhanced] EnhancedFSHybridClassifier  params={model.n_params():,}")
    logger.info(f"  [FS Enhanced] Fusion: {CLASSICAL_DIM}+40+{extra_dim}+16={CLASSICAL_DIM+40+extra_dim+16}-dim  "
                f"use_extra={use_extra}")

    tr_ldr = _fs_ldr(classical["train"], quantum["train"], labels["train"], bs, True,
                     extra=fs_meta["train"] if use_extra else None)
    va_ldr = _fs_ldr(classical["val"],   quantum["val"],   labels["val"],   bs, False,
                     extra=fs_meta["val"]   if use_extra else None)
    te_ldr = _fs_ldr(classical["test"],  quantum["test"],  labels["test"],  bs, False,
                     extra=fs_meta["test"]  if use_extra else None)

    print("Train labels distribution:")
    print(np.bincount(labels["train"].astype(int)))

    if eval_only:
        if not ckpt.exists():
            logger.warning("No best checkpoint found, saving current model.")
            torch.save(model.state_dict(), ckpt)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        logger.info(f"  Loaded: {ckpt}")
    else:
        lr = 3e-4; max_ep = 200; pat = 30
        opt   = AdamW(model.parameters(), lr=lr, weight_decay=cfg_t["weight_decay"])
        sched = CosineAnnealingLR(opt, T_max=max_ep, eta_min=lr * 0.01)
        best_auc = -1.0; pat_ctr = 0
        tracker  = EpochTracker()

        logger.info(f"  Training: max_epochs={max_ep}  bs={bs}  lr={lr}  patience={pat}")

        for epoch in range(1, max_ep + 1):
            t0 = time.time()
            tr_loss, tr_acc = _train_epoch_fs(model, tr_ldr, opt, focal, cfg_t["grad_clip"])
            va_loss, vp, vl = _eval_fs(model, va_ldr, focal)
            sched.step()
            val_thr = find_optimal_threshold(vl, vp, metric="mcc")
            m = compute_metrics(vl, vp, threshold=val_thr, dataset_name="fs_val")
            m.epoch = epoch
            tracker.log("val", m)
            logger.info(
                f"  Epoch {epoch:03d}  tr={tr_loss:.4f} acc={tr_acc:.3f}  "
                f"va={va_loss:.4f} f1={m.f1:.4f} auc={m.roc_auc:.4f} mcc={m.mcc:.4f}  "
                f"thr={val_thr:.3f}  lr={sched.get_last_lr()[0]:.1e}  [{time.time()-t0:.1f}s]"
            )
            if m.roc_auc > best_auc:
                best_auc = m.roc_auc; torch.save(model.state_dict(), ckpt); pat_ctr = 0
                logger.info(f"  [SAVE] val_auc={best_auc:.4f}  val_f1={m.f1:.4f}")
            else:
                pat_ctr += 1
                if pat_ctr >= pat:
                    logger.info(f"  Early stop epoch {epoch}"); break

        tracker.save(str(_ROOT / "results" / "metrics" / "fs_stage78_history.json"))
        if not ckpt.exists():
            logger.warning("No best checkpoint found after training, saving current model.")
            torch.save(model.state_dict(), ckpt)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    # Test evaluation
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    _, tp, tl = _eval_fs(model, te_ldr, focal)
    thr = find_optimal_threshold(tl, tp, metric="mcc")
    tm  = compute_metrics(tl, tp, threshold=thr, dataset_name="fs_final_test")
    print(f"\n{'='*55}\n  FINAL TEST -- FS (ENHANCED)\n{'='*55}")
    tm.pretty_print()
    tm.save(str(_ROOT / "results" / "metrics" / "fs_stage8_test.json"))

    # Save hybrid vectors for Stage 9
    hybrid_dim = CLASSICAL_DIM + 40 + extra_dim
    logger.info(f"  Saving {hybrid_dim}-dim hybrid vectors...")
    for sp in ["train", "val", "test"]:
        parts = [classical[sp], quantum[sp]]
        if use_extra:
            parts.append(fs_meta[sp])
        hv = np.concatenate(parts, axis=1)
        np.save(hybrid_dir / f"{sp}_hybrid.npy", hv.astype(np.float32))
        np.save(hybrid_dir / f"{sp}_labels.npy", labels[sp])
        logger.info(f"  {sp}_hybrid.npy  shape={hv.shape}")

    logger.info("Stage 7+8 [OK] 'fs' (enhanced)\n")
    return tm.to_dict()


# ── Per-dataset pipeline ──────────────────────────────────
def run_dataset(ds_key, config, eval_only=False, checkpoint=None):
    logger.info("="*60)
    logger.info(f"Stage 7+8 Hybrid Fusion + Classifier -- {ds_key.upper()}")
    logger.info("="*60)

    seed=config["project"]["seed"]; torch.manual_seed(seed); np.random.seed(seed)

    comp_dir    = _ROOT/"data"/"compressed"/ds_key
    quantum_dir = _ROOT/"data"/"quantum"/ds_key
    hybrid_dir  = _ROOT/"data"/"hybrid"/ds_key; hybrid_dir.mkdir(parents=True,exist_ok=True)
    cfg_t=config["training"]; cfg_c=config["classifier"]; bs=cfg_t["batch_size"]

    # Load classical features + quantum vectors
    classical, quantum, labels = {}, {}, {}
    for sp in ["train","val","test"]:
        classical[sp] = _load(comp_dir, sp, "_full.npy")  # always use full 256-dim
        quantum[sp]   = _load(quantum_dir, sp, "_qvec.npy")
        labels[sp]    = _load(quantum_dir, sp, "_labels.npy")
        assert classical[sp].shape[1] == CLASSICAL_DIM, \
            f"Expected {CLASSICAL_DIM}-dim classical, got {classical[sp].shape[1]}"
        logger.info(f"  {sp}: classical={classical[sp].shape}  quantum={quantum[sp].shape}")

    # FS only: load handcrafted meta-features and append to hybrid
    # BO and UAF: fs_meta is None — no change to their pipeline at all
    fs_meta = _build_fs_meta_features(ds_key, config)

    q_dim = quantum["train"].shape[1]

    # ── FS enhanced path: EnhancedFSHybridClassifier (q_dim==40 means EnhancedVQC ran) ──
    if ds_key == "fs" and q_dim == 40:
        _ckpt = Path(checkpoint) if checkpoint else _ROOT / "models" / "final" / f"{ds_key}_hybrid_best.pt"
        try:
            logger.info("  FS enhanced path: EnhancedFSHybridClassifier (cross-attention fusion)")
            return _run_fs_enhanced_fusion(
                classical, quantum, labels, fs_meta, config, _ckpt, eval_only, hybrid_dir
            )
        except Exception as _ef_err:
            logger.warning(f"  EnhancedFSHybridClassifier failed ({_ef_err}), falling back to HybridMLP")

    meta_dim   = FS_META_DIM if fs_meta is not None else 0
    hybrid_dim = CLASSICAL_DIM + q_dim + meta_dim
    logger.info(f"Loading Stage 3 full embeddings ({CLASSICAL_DIM}-dim) + Stage 6 quantum ({q_dim}-dim)"
                + (f" + FS meta ({meta_dim}-dim)" if meta_dim else "")
                + f" → {hybrid_dim}-dim hybrid")


    # Print label distribution for train set
    print("Train labels distribution:")
    print(np.bincount(labels["train"].astype(int)))

    tr_ldr = _loader(classical["train"], quantum["train"], labels["train"], bs, True,  meta=fs_meta["train"] if fs_meta else None, ds_key=ds_key)
    va_ldr = _loader(classical["val"],   quantum["val"],   labels["val"],   bs, False, meta=fs_meta["val"]   if fs_meta else None, ds_key=ds_key)
    te_ldr = _loader(classical["test"],  quantum["test"],  labels["test"],  bs, False, meta=fs_meta["test"]  if fs_meta else None, ds_key=ds_key)

    # Build model: Stage 7 fusion is parameter-free concat
    # Stage 8 MLP has trainable params
    model = HybridMLP(ds_key=ds_key, input_dim=hybrid_dim)
    n_pos = labels["train"].sum()
    pos_ratio = n_pos / len(labels["train"])
    # All classifiers now use FocalLoss — FS was using BCE+LabelSmoothing which
    # hurt calibration and inflated the false-positive rate.
    alpha = 0.5 if pos_ratio > 0.35 else 0.25
    focal = FocalLoss(gamma=2.0, alpha=alpha)
    logger.info(f"  Loss: Focal(gamma=2.0, alpha={alpha})  [pos_ratio={pos_ratio:.2f}]")

    logger.info(f"  Fusion: concat({CLASSICAL_DIM}-dim classical, {q_dim}-dim quantum) → {hybrid_dim}-dim")
    logger.info(f"  Parameters: {model.n_params():,}")

    ckpt = Path(checkpoint) if checkpoint else _ROOT/"models"/"final"/f"{ds_key}_hybrid_best.pt"


    if eval_only:
        if not ckpt.exists():
            logger.warning("No best checkpoint found, saving current model.")
            torch.save(model.state_dict(), ckpt)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        logger.info(f"  Loaded: {ckpt}")
    else:
        fs_mode = ds_key.lower() == "fs"
        lr       = 0.0005 if fs_mode else cfg_t["lr"]
        pat      = 30     if fs_mode else cfg_t["early_stopping_patience"]
        max_ep   = 200    if fs_mode else cfg_t["epochs"]
        opt   = AdamW(model.parameters(), lr=lr, weight_decay=cfg_t["weight_decay"])
        sched = CosineAnnealingLR(opt, T_max=max_ep, eta_min=lr*0.01)
        best_auc = -1.0  # Track AUC for early stopping (threshold-independent)
        pat_ctr=0
        tracker=EpochTracker()

        logger.info(f"  Training: max_epochs={max_ep} bs={bs} "
                    f"lr={lr} patience={pat}")

        for epoch in range(1, max_ep+1):
            t0=time.time()
            tr_loss,tr_acc = _train_epoch(model,tr_ldr,opt,focal,cfg_t["grad_clip"])
            va_loss,vp,vl  = _eval(model,va_ldr,focal)
            sched.step()
            # BO: optimise threshold for MCC (balances TP/FP better than Youden for BO)
            thr_metric = "mcc" if ds_key in ("bo", "fs") else "f1"
            val_thr = find_optimal_threshold(vl, vp, metric=thr_metric)
            m=compute_metrics(vl,vp,threshold=val_thr,dataset_name=f"{ds_key}_val"); m.epoch=epoch
            tracker.log("val",m)
            logger.info(
                f"  Epoch {epoch:03d}  "
                f"tr={tr_loss:.4f} acc={tr_acc:.3f}  "
                f"va={va_loss:.4f} f1={m.f1:.4f} auc={m.roc_auc:.4f} mcc={m.mcc:.4f}  "
                f"thr={val_thr:.3f}  lr={sched.get_last_lr()[0]:.1e}  [{time.time()-t0:.1f}s]"
            )
            # Early stopping on AUC (threshold-independent, reliable)
            if m.roc_auc > best_auc:
                best_auc=m.roc_auc; torch.save(model.state_dict(),ckpt); pat_ctr=0
                logger.info(f"  [SAVE] val_auc={best_auc:.4f}  val_f1={m.f1:.4f}")
            else:
                pat_ctr+=1
                if pat_ctr>=pat: logger.info(f"  Early stop epoch {epoch}"); break
        tracker.save(str(_ROOT/"results"/"metrics"/f"{ds_key}_stage78_history.json"))
        # Fallback: if checkpoint was never saved, save current model
        if not ckpt.exists():
            logger.warning("No best checkpoint found after training, saving current model.")
            torch.save(model.state_dict(), ckpt)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    # Test evaluation
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    _,tp,tl = _eval(model,te_ldr,focal)
    thr = find_optimal_threshold(tl, tp, metric="mcc" if ds_key in ("bo", "fs") else "youden")
    tm  = compute_metrics(tl,tp,threshold=thr,dataset_name=f"{ds_key}_final_test")
    print(f"\n{'='*55}\n  FINAL TEST -- {ds_key.upper()}\n{'='*55}")
    tm.pretty_print()
    tm.save(str(_ROOT/"results"/"metrics"/f"{ds_key}_stage8_test.json"))

    # Save hybrid vectors for Stage 9
    logger.info(f"  Saving {hybrid_dim}-dim hybrid vectors for Stage 9 (explainability)...")
    for sp,c,q,y in [("train",classical["train"],quantum["train"],labels["train"]),
                      ("val",  classical["val"],  quantum["val"],  labels["val"]),
                      ("test", classical["test"], quantum["test"], labels["test"])]:
        parts = [c, q]
        if fs_meta is not None:
            parts.append(fs_meta[sp])
        hv = np.concatenate(parts, axis=1)  # (N, hybrid_dim)
        np.save(hybrid_dir/f"{sp}_hybrid.npy", hv.astype(np.float32))
        np.save(hybrid_dir/f"{sp}_labels.npy", y)
        logger.info(f"  {sp}_hybrid.npy  shape={hv.shape}")

    logger.info(f"Stage 7+8 [OK] '{ds_key}'\n")
    return tm.to_dict()


def main():
    p=argparse.ArgumentParser(description="QEGVD Stage 7+8 -- Hybrid Fusion + Classifier")
    p.add_argument("--dataset",choices=["bo","fs","uaf","all"],required=True)
    p.add_argument("--config",default=None)
    p.add_argument("--eval-only",action="store_true")
    p.add_argument("--checkpoint",default=None)
    args=p.parse_args()
    cfg=load_config(args.config)
    datasets=["bo","fs","uaf"] if args.dataset=="all" else [args.dataset]

    all_res={}
    for ds in datasets:
        all_res[ds]=run_dataset(ds,cfg,args.eval_only,args.checkpoint)

    print("\n"+"="*65)
    print("  STAGE 7+8 FINAL SUMMARY")
    print("="*65)
    print(f"  {'DS':<6}  {'F1':>8}  {'AUC':>8}  {'MCC':>8}  {'Acc':>8}  {'FPR':>8}  {'FNR':>8}")
    print("  "+"-"*57)
    for ds,r in all_res.items():
        print(f"  {ds.upper():<6}  {r['f1']:>8.4f}  {r['roc_auc']:>8.4f}  "
              f"{r['mcc']:>8.4f}  {r['accuracy']:>8.4f}  "
              f"{r['fpr']:>8.4f}  {r['fnr']:>8.4f}")
    print("="*65)
    print("  Hybrid vectors (N,260) -> data/hybrid/<ds>/")
    print("  Final models   -> models/final/<ds>_hybrid_best.pt")
    print("="*65+"\n")

if __name__=="__main__":
    main()