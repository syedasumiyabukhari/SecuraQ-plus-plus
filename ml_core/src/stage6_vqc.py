"""
QEGVD -- Stage 6: Variational Quantum Circuit (VQC)
====================================================
Paper Section 7 (exact circuit spec):
  H^4 -> [RY/RZ s1] -> CNOT-ring -> Var_A(x3) -> [RY/RZ s2] -> CNOT-ring -> Var_B(x3) -> <Z>^4

Real quantum simulation via PennyLane 'default.qubit' backend.
N_QUBITS = 4, output dim = 4 (PauliZ expectation values).
Data re-uploading: 8-dim QAFA stage1 output split into two 4-dim encoding rounds.

Usage:
    python src/stage6_vqc.py --dataset bo
    python src/stage6_vqc.py --dataset all
    python src/stage6_vqc.py --dataset all --eval-only
"""

from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np, yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
for _d in ["logs","data/quantum","models/checkpoints","results/metrics"]:
    (_ROOT / _d).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)),
        logging.FileHandler(_ROOT/"logs"/"stage6.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("Stage6")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from utils.metrics import compute_metrics, find_optimal_threshold, EpochTracker

try:
    import pennylane as qml
    _PL_AVAILABLE = True
    logger.info(f"PennyLane {qml.version()} loaded — using real quantum circuit simulation")
except ImportError:
    _PL_AVAILABLE = False
    logger.error("PennyLane not found — install with: pip install pennylane")
    sys.exit(1)

# Paper constants
N_QUBITS    = 4   # circuit width (paper Section 7)
QUANTUM_DIM = N_QUBITS   # output dim = <Z>^4
N_VAR_LAYERS = 3  # variational layers per block (Var_A and Var_B)
BO_N_ROUNDS  = 8  # BO: 8 rounds × 8 features = 64 features, output 8×4=32-dim


# ---------------------------------------------------------------------------
# PennyLane VQC
# ---------------------------------------------------------------------------

class PennyLaneVQC(nn.Module):
    """
    Real variational quantum circuit via PennyLane default.qubit.

    Circuit (paper Section 7):
      H^4 -> RY/RZ(s1) -> CNOT-ring -> Var_A(3 layers) ->
      RY/RZ(s2) -> CNOT-ring -> Var_B(3 layers) -> <Z>^4

    Input:  s1, s2 — each (B, N_QUBITS=4) tensors
    Output: (B, N_QUBITS=4) PauliZ expectation values in [-1, 1]
    Gradients: backprop through PennyLane interface
    """

    def __init__(self, n_var_layers: int = N_VAR_LAYERS):
        super().__init__()
        self.n_qubits    = N_QUBITS
        self.n_var_layers = n_var_layers

        # Variational parameters for block A and block B
        self.params_A = nn.Parameter(
            torch.zeros(n_var_layers, N_QUBITS, 2).uniform_(-np.pi / 8, np.pi / 8))
        self.params_B = nn.Parameter(
            torch.zeros(n_var_layers, N_QUBITS, 2).uniform_(-np.pi / 8, np.pi / 8))

        # Per-qubit input scale (learnable, initialised to pi so inputs mapped to [-pi,pi])
        self.input_scale = nn.Parameter(torch.ones(N_QUBITS) * np.pi)

        # BatchNorm for input stabilisation
        self.bn = nn.BatchNorm1d(N_QUBITS)

        # Build PennyLane device and QNode
        self._dev   = qml.device("default.qubit", wires=N_QUBITS)
        self._qnode = qml.QNode(
            self._circuit, self._dev,
            interface="torch", diff_method="backprop"
        )

    def _circuit(self, s1, s2, params_A, params_B, scale):
        """
        Quantum circuit: data re-uploading with two encoding rounds.
          s1, s2: (B, N_QUBITS) or (N_QUBITS,) tensors
          params_A, params_B: (n_var_layers, N_QUBITS, 2)
          scale: (N_QUBITS,)
        Returns: list of N_QUBITS expectation-value tensors
        """
        n = self.n_qubits

        # --- Hadamard initialisation ---
        for k in range(n):
            qml.Hadamard(wires=k)

        # --- Round 1: encode s1 ---
        for k in range(n):
            qml.RY(s1[..., k] * scale[k], wires=k)
            qml.RZ(s1[..., k] * scale[k], wires=k)

        # CNOT entanglement ring
        for k in range(n):
            qml.CNOT(wires=[k, (k + 1) % n])

        # Variational block A
        for layer in range(self.n_var_layers):
            for k in range(n):
                qml.RY(params_A[layer, k, 0], wires=k)
                qml.RZ(params_A[layer, k, 1], wires=k)

        # --- Round 2: encode s2 (data re-uploading) ---
        for k in range(n):
            qml.RY(s2[..., k] * scale[k], wires=k)
            qml.RZ(s2[..., k] * scale[k], wires=k)

        # CNOT entanglement ring
        for k in range(n):
            qml.CNOT(wires=[k, (k + 1) % n])

        # Variational block B
        for layer in range(self.n_var_layers):
            for k in range(n):
                qml.RY(params_B[layer, k, 0], wires=k)
                qml.RZ(params_B[layer, k, 1], wires=k)

        # Measurement: PauliZ expectation on each qubit
        return [qml.expval(qml.PauliZ(k)) for k in range(n)]

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """
        s1, s2: (B, N_QUBITS) input tensors
        Returns: (B, N_QUBITS) expectation values in [-1, 1], float32
        """
        B = s1.shape[0]
        # Stabilise inputs: tanh → [-1,1], then scale by learnable input_scale
        if B > 1:
            s1 = torch.tanh(self.bn(s1))
        else:
            s1 = torch.tanh(s1)
        s2 = torch.tanh(s2)

        result = self._qnode(s1, s2, self.params_A, self.params_B, self.input_scale)
        # result: list of N_QUBITS tensors each (B,)  (PennyLane broadcasting)
        return torch.stack(result, dim=-1).float()   # (B, N_QUBITS)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Pretraining head
# ---------------------------------------------------------------------------

class VQCPretrainModel(nn.Module):
    """4-dim quantum output → classification head."""
    def __init__(self, vqc: PennyLaneVQC):
        super().__init__()
        self.vqc  = vqc
        self.head = nn.Sequential(
            nn.Linear(QUANTUM_DIM, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1)
        )

    def forward(self, s1, s2):
        return self.head(self.vqc(s1, s2)).squeeze(-1)


class MultiRoundVQCModel(nn.Module):
    """BO: n_rounds × 8-dim QAFA → (n_rounds × 4)-dim quantum → classify.
    Shared VQC weights process each 8-dim chunk (split as s1[:4], s2[4:]).
    """
    def __init__(self, vqc: PennyLaneVQC, n_rounds: int = BO_N_ROUNDS):
        super().__init__()
        self.vqc      = vqc
        self.n_rounds = n_rounds
        q_out = N_QUBITS * n_rounds   # 32 for BO
        self.head = nn.Sequential(
            nn.Linear(q_out, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # x: (B, n_rounds * 8)
        qvecs = []
        for i in range(self.n_rounds):
            chunk = x[:, i * 8:(i + 1) * 8]
            s1 = chunk[:, :N_QUBITS]
            s2 = chunk[:, N_QUBITS:]
            qvecs.append(self.vqc(s1, s2))   # (B, 4)
        return self.head(torch.cat(qvecs, dim=-1)).squeeze(-1)   # (B,)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__(); self.gamma = gamma; self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        at  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (at * (1 - pt) ** self.gamma * bce).mean()


# ---------------------------------------------------------------------------
# Dataset & loaders
# ---------------------------------------------------------------------------

class _DS(Dataset):
    def __init__(self, s1, s2, y):
        self.s1 = torch.from_numpy(s1.astype(np.float32))
        self.s2 = torch.from_numpy(s2.astype(np.float32))
        self.y  = torch.from_numpy(y.astype(np.float32))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.s1[i], self.s2[i], self.y[i]

def _loader(s1, s2, y, bs, shuffle):
    return DataLoader(_DS(s1, s2, y), batch_size=bs, shuffle=shuffle, num_workers=0)


def _load_qafa_split(d: Path, sp: str):
    """
    Load stage1 QAFA features and split the 8-dim vector into two 4-dim halves.
    s1 = features[:4] (first encoding round)
    s2 = features[4:] (second encoding round / data re-uploading)
    """
    stage1 = np.load(d / f"{sp}_stage1.npy")   # (N, 8)
    y      = np.load(d / f"{sp}_labels.npy")
    return stage1[:, :N_QUBITS], stage1[:, N_QUBITS:N_QUBITS*2], y


def _load_qafa_multi_stages(d: Path, sp: str, n_rounds: int):
    """Load n_rounds stage files and concatenate to (N, n_rounds*8)."""
    arrays = [np.load(d / f"{sp}_stage{i+1}.npy") for i in range(n_rounds)]  # each (N, 8)
    y = np.load(d / f"{sp}_labels.npy")
    return np.concatenate(arrays, axis=1), y   # (N, n_rounds*8)


class _DS_multi(Dataset):
    """Dataset for multi-round BO: stores (N, n_rounds*8) input."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ---------------------------------------------------------------------------
# Quantum vector extraction (inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_qvecs(vqc: PennyLaneVQC, s1: np.ndarray, s2: np.ndarray, bs: int = 128):
    vqc.eval()
    out = []
    for i in range(0, len(s1), bs):
        s1b = torch.from_numpy(s1[i:i + bs].astype(np.float32))
        s2b = torch.from_numpy(s2[i:i + bs].astype(np.float32))
        out.append(vqc(s1b, s2b).numpy())
    return np.concatenate(out, axis=0)   # (N, N_QUBITS=4)


@torch.no_grad()
def extract_qvecs_multi(vqc: PennyLaneVQC, all_stages: np.ndarray,
                        n_rounds: int = BO_N_ROUNDS, bs: int = 64):
    """BO: run shared VQC on each of n_rounds 8-dim chunks, concat → (N, n_rounds*4)."""
    vqc.eval()
    out = []
    for i in range(0, len(all_stages), bs):
        x = torch.from_numpy(all_stages[i:i + bs].astype(np.float32))
        qvecs = []
        for r in range(n_rounds):
            chunk = x[:, r * 8:(r + 1) * 8]
            qvecs.append(vqc(chunk[:, :N_QUBITS], chunk[:, N_QUBITS:]))  # (B, 4)
        out.append(torch.cat(qvecs, dim=-1).numpy())
    return np.concatenate(out, axis=0)   # (N, n_rounds*4)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path=None):
    return yaml.safe_load(open(path or _ROOT / "configs" / "config.yaml"))


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def _run_training_loop(model, tr_ldr, va_ldr, y_val, cfg_t, ckpt, ds_key,
                       is_multi_round=False):
    """Shared training loop for both single-round and multi-round VQC models."""
    focal   = FocalLoss(gamma=2.0, alpha=0.25)
    opt     = AdamW(model.parameters(), lr=cfg_t["lr"], weight_decay=cfg_t["weight_decay"])
    sch     = CosineAnnealingLR(opt, T_max=cfg_t["epochs"], eta_min=cfg_t["lr"] * 0.01)
    tracker = EpochTracker()
    best_va  = float("inf")
    best_thr = 0.5
    pat_ctr  = 0

    logger.info(f"  Training: epochs={cfg_t['epochs']}  bs={cfg_t['batch_size']}  patience={cfg_t['early_stopping_patience']}")

    for epoch in range(1, cfg_t["epochs"] + 1):
        t0 = time.time()
        model.train(); ep = n = 0
        for batch in tr_ldr:
            opt.zero_grad()
            if is_multi_round:
                Xb, yb = batch
                lg = model(Xb)
            else:
                s1b, s2b, yb = batch
                lg = model(s1b, s2b)
            loss = focal(lg, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg_t["grad_clip"])
            opt.step()
            ep += loss.item() * len(yb); n += len(yb)
        sch.step()

        model.eval(); va_l = 0.0; vp, vl = [], []
        with torch.no_grad():
            for batch in va_ldr:
                if is_multi_round:
                    Xb, yb = batch
                    lg = model(Xb)
                else:
                    s1b, s2b, yb = batch
                    lg = model(s1b, s2b)
                va_l += focal(lg, yb).item() * len(yb)
                vp.append(torch.sigmoid(lg).numpy()); vl.append(yb.numpy())
        va_l /= len(y_val)
        vp_all = np.concatenate(vp); vl_all = np.concatenate(vl)

        thr = find_optimal_threshold(vl_all, vp_all, metric="youden")
        m   = compute_metrics(vl_all, vp_all, threshold=thr, dataset_name=f"{ds_key}_vqc_val")
        tracker.log("val", m)
        logger.info(
            f"  Epoch {epoch:03d}  tr={ep/n:.4f}  va={va_l:.4f}  "
            f"f1={m.f1:.4f}  acc={m.accuracy:.4f}  [{time.time()-t0:.0f}s]"
        )
        if va_l < best_va:
            best_va = va_l; best_thr = thr
            torch.save(model.state_dict(), ckpt)
            pat_ctr = 0
            logger.info(f"  [SAVE] val_loss={best_va:.4f}  thr={best_thr:.3f}")
        else:
            pat_ctr += 1
            if pat_ctr >= cfg_t["early_stopping_patience"]:
                logger.info(f"  [EARLY STOP] epoch {epoch}"); break

    np.save(_ROOT / "models" / "checkpoints" / f"{ds_key}_vqc_threshold.npy",
            np.array([best_thr]))
    tracker.save(str(_ROOT / "results" / "metrics" / f"{ds_key}_stage6_history.json"))
    return best_thr


# ---------------------------------------------------------------------------
# FS-specific: EnhancedVQC runner
# ---------------------------------------------------------------------------

def _run_fs_enhanced_vqc(enhanced_vqc, ds_key, config, qafa_dir, quantum_dir, ckpt, eval_only):
    """Run the 8-qubit EnhancedVQC for FS. Produces (N, 40) quantum vectors."""
    from stage6_enhanced_vqc import FocalLoss as FSFocalLoss, QUANTUM_OUT_FS
    from torch.utils.data import TensorDataset, DataLoader as _DL

    cfg_t   = config["training"]
    seed    = config["project"]["seed"]
    torch.manual_seed(seed); np.random.seed(seed)

    n_rounds = BO_N_ROUNDS   # 8 rounds of 8-dim input
    X_all, y_all = {}, {}
    for sp in ["train", "val", "test"]:
        X_all[sp], y_all[sp] = _load_qafa_multi_stages(qafa_dir, sp, n_rounds)
    logger.info(f"  FS Enhanced: Train={len(y_all['train'])} Val={len(y_all['val'])} Test={len(y_all['test'])}")

    # Use first 8-dim stage as input to EnhancedVQC (one round at a time)
    # Input: (N, 8) from stage1

    class _FSDS(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X[:, :8].astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.float32))
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i]

    class _FSHead(nn.Module):
        """Quick classification head on top of EnhancedVQC for pretraining."""
        def __init__(self, vqc):
            super().__init__()
            self.vqc  = vqc
            self.head = nn.Sequential(
                nn.Linear(QUANTUM_OUT_FS, 32), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.head(self.vqc(x)).squeeze(-1)

    model = _FSHead(enhanced_vqc)
    focal = FSFocalLoss(gamma=2.0, alpha=0.5)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg_t["lr"],
                              weight_decay=cfg_t["weight_decay"])
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg_t["epochs"], eta_min=cfg_t["lr"] * 0.01)
    tracker = EpochTracker()
    best_va = float("inf"); pat_ctr = 0; best_thr = 0.5

    bs = cfg_t["batch_size"]
    tr_ldr = _DL(_FSDS(X_all["train"], y_all["train"]), bs, shuffle=True,  num_workers=0)
    va_ldr = _DL(_FSDS(X_all["val"],   y_all["val"]),   bs, shuffle=False, num_workers=0)

    if not eval_only:
        logger.info(f"  Training EnhancedVQC for FS …")
        for epoch in range(1, cfg_t["epochs"] + 1):
            t0 = time.time()
            model.train(); ep = n = 0
            for xb, yb in tr_ldr:
                opt.zero_grad()
                loss = focal(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg_t["grad_clip"])
                opt.step()
                ep += loss.item() * len(yb); n += len(yb)
            sch.step()

            model.eval(); va_l = 0; vp, vl = [], []
            with torch.no_grad():
                for xb, yb in va_ldr:
                    lg = model(xb)
                    va_l += focal(lg, yb).item() * len(yb)
                    vp.append(torch.sigmoid(lg).numpy()); vl.append(yb.numpy())
            va_l /= len(y_all["val"])
            vp_all = np.concatenate(vp); vl_all = np.concatenate(vl)
            thr = find_optimal_threshold(vl_all, vp_all, metric="youden")
            m   = compute_metrics(vl_all, vp_all, threshold=thr, dataset_name="fs_evqc_val")
            tracker.log("val", m)
            logger.info(f"  Epoch {epoch:03d}  tr={ep/n:.4f}  va={va_l:.4f}  "
                        f"f1={m.f1:.4f}  acc={m.accuracy:.4f}  [{time.time()-t0:.0f}s]")
            if va_l < best_va:
                best_va = va_l; best_thr = thr
                torch.save(model.state_dict(), ckpt); pat_ctr = 0
                logger.info(f"  [SAVE] val_loss={best_va:.4f}  thr={best_thr:.3f}")
            else:
                pat_ctr += 1
                if pat_ctr >= cfg_t["early_stopping_patience"]:
                    logger.info(f"  [EARLY STOP] epoch {epoch}"); break

        np.save(_ROOT / "models" / "checkpoints" / f"{ds_key}_vqc_threshold.npy",
                np.array([best_thr]))
        tracker.save(str(_ROOT / "results" / "metrics" / f"{ds_key}_stage6_history.json"))

    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    thr_file = _ROOT / "models" / "checkpoints" / f"{ds_key}_vqc_threshold.npy"
    best_thr = float(np.load(thr_file)[0]) if thr_file.exists() else 0.5

    # Extract 40-dim quantum vectors for all splits
    result = {}
    for sp in ["train", "val", "test"]:
        x_sp = torch.from_numpy(X_all[sp][:, :8].astype(np.float32))
        with torch.no_grad():
            qv = model.vqc(x_sp).numpy()   # (N, 40)
        np.save(quantum_dir / f"{sp}_qvec.npy",   qv)
        np.save(quantum_dir / f"{sp}_labels.npy",  y_all[sp])
        logger.info(f"  {sp}: {qv.shape}  range=[{qv.min():.3f}, {qv.max():.3f}]")
        if sp == "test":
            with torch.no_grad():
                lg    = model(torch.from_numpy(X_all[sp][:, :8].astype(np.float32)))
                probs = torch.sigmoid(lg).numpy()
            tm = compute_metrics(y_all[sp], probs, threshold=best_thr,
                                 dataset_name=f"{ds_key}_vqc_test")
            tm.pretty_print()
            tm.save(str(_ROOT / "results" / "metrics" / f"{ds_key}_stage6_test.json"))
            result = tm.to_dict()

    logger.info(f"Stage 6 EnhancedVQC [OK] FS  quantum dim={QUANTUM_OUT_FS}\n")
    return result


def run_dataset(ds_key: str, config: dict, eval_only: bool = False) -> dict:
    logger.info("=" * 60)
    logger.info(f"Stage 6 VQC -- {ds_key.upper()}  (PennyLane real quantum circuit)")
    logger.info("=" * 60)

    seed = config["project"]["seed"]
    torch.manual_seed(seed); np.random.seed(seed)

    qafa_dir    = _ROOT / "data" / "qafa"    / ds_key
    quantum_dir = _ROOT / "data" / "quantum" / ds_key
    quantum_dir.mkdir(parents=True, exist_ok=True)
    cfg_t = config["training"]
    ckpt  = _ROOT / "models" / "checkpoints" / f"{ds_key}_vqc_best.pt"

    for sp in ["train", "val", "test"]:
        if not (qafa_dir / f"{sp}_stage1.npy").exists():
            raise FileNotFoundError(f"Missing {qafa_dir}/{sp}_stage1.npy — run Stage 5 first")

    # ── FS: use EnhancedVQC (8-qubit + ZZ correlators) ───────────────────────
    if ds_key == "fs":
        try:
            from stage6_enhanced_vqc import EnhancedVQC, QUANTUM_OUT_FS
            enhanced_vqc = EnhancedVQC(n_qubits=8, n_var_layers=N_VAR_LAYERS, n_reruns=5)
            logger.info(f"  FS: EnhancedVQC  8-qubit + ZZ correlators  out={QUANTUM_OUT_FS}-dim")
            logger.info(f"  EnhancedVQC trainable params: {enhanced_vqc.n_params()}")
            return _run_fs_enhanced_vqc(
                enhanced_vqc, ds_key, config, qafa_dir, quantum_dir, ckpt, eval_only
            )
        except Exception as _ev_err:
            logger.warning(f"  EnhancedVQC failed ({_ev_err}), falling back to standard VQC")

    logger.info(f"  Circuit: {N_QUBITS} qubits · {N_VAR_LAYERS} var-layers/block")
    logger.info(f"  H^{N_QUBITS} → RY/RZ(s1) → CNOT-ring → Var_A({N_VAR_LAYERS}) → RY/RZ(s2) → CNOT-ring → Var_B({N_VAR_LAYERS}) → <Z>^{N_QUBITS}")
    vqc = PennyLaneVQC(n_var_layers=N_VAR_LAYERS)
    logger.info(f"  VQC trainable params: {vqc.n_params()}")

    # ── BO/FS: multi-round (8 rounds × 8-dim → 32-dim quantum) ───────────────
    if ds_key in ("bo", "fs"):
        n_rounds = BO_N_ROUNDS
        X_all, y_all = {}, {}
        for sp in ["train", "val", "test"]:
            X_all[sp], y_all[sp] = _load_qafa_multi_stages(qafa_dir, sp, n_rounds)
        logger.info(
            f"  Train={len(y_all['train'])}  Val={len(y_all['val'])}  Test={len(y_all['test'])}"
        )
        logger.info(f"  Input: {n_rounds} rounds × 8-dim → {n_rounds * N_QUBITS}-dim quantum output")

        model = MultiRoundVQCModel(vqc, n_rounds=n_rounds)

        if eval_only:
            if not ckpt.exists():
                raise FileNotFoundError(f"No checkpoint: {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            logger.info(f"  [EVAL-ONLY] Loaded: {ckpt}")
            thr_file = _ROOT / "models" / "checkpoints" / f"{ds_key}_vqc_threshold.npy"
            best_thr = float(np.load(thr_file)[0]) if thr_file.exists() else 0.5
        else:
            bs = cfg_t["batch_size"]
            tr_ldr = DataLoader(_DS_multi(X_all["train"], y_all["train"]), bs, shuffle=True,  num_workers=0)
            va_ldr = DataLoader(_DS_multi(X_all["val"],   y_all["val"]),   bs, shuffle=False, num_workers=0)
            best_thr = _run_training_loop(
                model, tr_ldr, va_ldr, y_all["val"], cfg_t, ckpt, ds_key,
                is_multi_round=True
            )

        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        q_out_dim = n_rounds * N_QUBITS
        logger.info(f"  Extracting quantum vectors (N, {q_out_dim}) for all splits...")
        result = {}
        for sp in ["train", "val", "test"]:
            qv = extract_qvecs_multi(model.vqc, X_all[sp], n_rounds)
            np.save(quantum_dir / f"{sp}_qvec.npy",   qv)
            np.save(quantum_dir / f"{sp}_labels.npy",  y_all[sp])
            logger.info(f"  {sp}: {qv.shape}  range=[{qv.min():.3f}, {qv.max():.3f}]")
            if sp == "test":
                with torch.no_grad():
                    x_t = torch.from_numpy(X_all[sp].astype(np.float32))
                    lg  = model(x_t)
                    probs = torch.sigmoid(lg).numpy()
                tm = compute_metrics(y_all[sp], probs, threshold=best_thr,
                                     dataset_name=f"{ds_key}_vqc_test")
                tm.pretty_print()
                tm.save(str(_ROOT / "results" / "metrics" / f"{ds_key}_stage6_test.json"))
                result = tm.to_dict()

    # ── FS / UAF: single-round (8-dim stage1 → 4-dim quantum) ─────────────
    else:
        s1, s2, y = {}, {}, {}
        for sp in ["train", "val", "test"]:
            s1[sp], s2[sp], y[sp] = _load_qafa_split(qafa_dir, sp)
        logger.info(f"  Train={len(y['train'])}  Val={len(y['val'])}  Test={len(y['test'])}")
        logger.info(f"  Input: 8-dim QAFA → s1{s1['train'].shape[1:]}, s2{s2['train'].shape[1:]}")

        model = VQCPretrainModel(vqc)

        if eval_only:
            if not ckpt.exists():
                raise FileNotFoundError(f"No checkpoint: {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            logger.info(f"  [EVAL-ONLY] Loaded: {ckpt}")
            thr_file = _ROOT / "models" / "checkpoints" / f"{ds_key}_vqc_threshold.npy"
            best_thr = float(np.load(thr_file)[0]) if thr_file.exists() else 0.5
        else:
            bs = cfg_t["batch_size"]
            tr_ldr = _loader(s1["train"], s2["train"], y["train"], bs, True)
            va_ldr = _loader(s1["val"],   s2["val"],   y["val"],   bs, False)
            best_thr = _run_training_loop(
                model, tr_ldr, va_ldr, y["val"], cfg_t, ckpt, ds_key,
                is_multi_round=False
            )

        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        thr_file = _ROOT / "models" / "checkpoints" / f"{ds_key}_vqc_threshold.npy"
        best_thr = float(np.load(thr_file)[0]) if thr_file.exists() else 0.5

        logger.info("  Extracting quantum vectors (N, 4) for all splits...")
        result = {}
        for sp in ["train", "val", "test"]:
            qv = extract_qvecs(model.vqc, s1[sp], s2[sp])
            np.save(quantum_dir / f"{sp}_qvec.npy",  qv)
            np.save(quantum_dir / f"{sp}_labels.npy", y[sp])
            logger.info(f"  {sp}: {qv.shape}  range=[{qv.min():.3f}, {qv.max():.3f}]")
            if sp == "test":
                with torch.no_grad():
                    lg    = model.head(torch.from_numpy(qv.astype(np.float32))).squeeze(-1)
                    probs = torch.sigmoid(lg).numpy()
                tm = compute_metrics(y[sp], probs, threshold=best_thr,
                                     dataset_name=f"{ds_key}_vqc_test")
                tm.pretty_print()
                tm.save(str(_ROOT / "results" / "metrics" / f"{ds_key}_stage6_test.json"))
                result = tm.to_dict()

    logger.info(f"Stage 6 [OK] '{ds_key.upper()}'\n")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="QEGVD Stage 6 — VQC (PennyLane)")
    p.add_argument("--dataset", choices=["bo", "fs", "uaf", "all"], required=True)
    p.add_argument("--config",  default=None)
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    cfg      = load_config(args.config)
    datasets = ["bo", "fs", "uaf"] if args.dataset == "all" else [args.dataset]
    t0       = time.time()
    results  = {}

    for ds in datasets:
        results[ds] = run_dataset(ds, cfg, args.eval_only)

    print("\n" + "=" * 60)
    print("  STAGE 6  VQC (PennyLane)  COMPLETE")
    print(f"  Circuit: {N_QUBITS} qubits · {N_VAR_LAYERS} var-layers/block · backprop")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    print(f"  {'DS':<6}  {'F1':>8}  {'AUC':>8}  {'MCC':>8}  {'ACC':>8}")
    print("  " + "-" * 48)
    for ds, r in results.items():
        if r:
            print(f"  {ds.upper():<6}  {r.get('f1', 0):>8.4f}  {r.get('roc_auc', 0):>8.4f}  "
                  f"{r.get('mcc', 0):>8.4f}  {r.get('accuracy', 0):>8.4f}")
    print(f"\n  Quantum vectors (N, {QUANTUM_DIM}) -> data/quantum/<ds>/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
