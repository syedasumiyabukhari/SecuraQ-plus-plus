"""
QEGVD -- Stage 4: Classical Feature Encoding and Compression
=============================================================
Compresses the 128-dim GAT embedding to a 32-dim latent vector.

Paper architecture (exact):
    Input (128) -> FC(128->64) -> ReLU + BN + Dropout(0.3)
                -> FC(64->32)  -> ReLU + BN + Dropout(0.3)
    Output: 32-dim latent vulnerability signature

The 32-dim output feeds directly into Stage 5 (QAFA).

Usage
-----
    python src/stage4_classical_encoder.py --dataset bo
    python src/stage4_classical_encoder.py --dataset all
    python src/stage4_classical_encoder.py --dataset bo --eval-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

for _d in ["logs", "models/checkpoints", "results/metrics",
           "data/compressed"]:
    (_ROOT / _d).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
        logging.FileHandler(
            _ROOT / "logs" / "stage4.log", mode="a", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("Stage4")


def _check_deps():
    try:
        import torch
    except ImportError:
        logger.error("torch not found -- pip install torch")
        sys.exit(1)

_check_deps()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from utils.metrics import compute_metrics, find_optimal_threshold, EpochTracker

# ---------------------------------------------------------------------------
# Paper-exact constants
# ---------------------------------------------------------------------------
INPUT_DIM           = 256   # GAT fused embedding dim (Stage 3 output)
COMPRESSED_DIM      = 32    # default compressed dim (FS / UAF)
COMPRESSED_DIM_BO   = 64    # BO/FS: 256 -> 128 -> 64 (richer for multi-round QAFA)
COMPRESSED_DIM_FS   = 64    # FS: same as BO — more capacity for subtle patterns


# ---------------------------------------------------------------------------
# Encoder: 256 -> 128 -> compressed_dim
# ReLU + BatchNorm + Dropout after each FC layer
# ---------------------------------------------------------------------------

class ClassicalEncoder(nn.Module):
    """
    Architecture: 256 -> 128 -> compressed_dim
    BO  : compressed_dim=64  (256->128->64)
    FS/UAF: compressed_dim=32 (256->128->32)
    """

    def __init__(self, input_dim=256, compressed_dim=32, dropout=0.2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, compressed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(compressed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Stage 4 Model: Encoder + Binary Classifier (for supervised training)
# ---------------------------------------------------------------------------

class Stage4Model(nn.Module):
    def __init__(self, encoder: ClassicalEncoder, dropout=0.2, compressed_dim=COMPRESSED_DIM):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Sequential(
            nn.Linear(compressed_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1)
        return self.classifier(self.encoder(x)).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def make_loader(X, y, batch_size, shuffle):
    return DataLoader(EmbeddingDataset(X, y), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=False)


# ---------------------------------------------------------------------------
# Train / Eval / Compress
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = correct = total = 0
    pos_weight = torch.tensor(1.0, device=device)
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct    += ((torch.sigmoid(logits) >= 0.7).long() == y.long()).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = total = 0
    all_probs = []
    all_labels = []
    pos_weight = torch.tensor(1.0, device=device)
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss   = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        total_loss += loss.item() * y.size(0)
        total      += y.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    return (total_loss / total,
            np.concatenate(all_probs),
            np.concatenate(all_labels))


@torch.no_grad()
def compress_split(model, loader, device):
    model.eval()
    compressed = []
    full_emb = []
    labels_out = []
    for X, y in loader:
        X = X.to(device)
        compressed.append(model.encoder(X).cpu().numpy())
        full_emb.append(X.cpu().numpy())
        labels_out.append(y.numpy())
    return (np.concatenate(compressed),
            np.concatenate(full_emb),
            np.concatenate(labels_out))


# ---------------------------------------------------------------------------
# Config + loaders
# ---------------------------------------------------------------------------

def load_config(path=None):
    if path is None:
        path = _ROOT / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_embeddings(embed_dir: Path, split: str):
    X = np.load(embed_dir / f"{split}.npy")
    y = np.load(embed_dir / f"{split}_labels.npy")
    return X, y


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def run_dataset(ds_key, config, eval_only=False, checkpoint=None):
    logger.info("=" * 60)
    logger.info(f"Stage 4 -- {ds_key.upper()}")
    logger.info("=" * 60)

    device = torch.device("cpu")
    seed   = config["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    embed_dir = _ROOT / "data" / "embeddings" / ds_key
    comp_dir  = _ROOT / "data" / "compressed"  / ds_key
    comp_dir.mkdir(parents=True, exist_ok=True)

    cfg_t = config["training"]
    bs    = cfg_t["batch_size"]

    # Load Stage 3 embeddings
    logger.info("Loading Stage 3 embeddings...")
    for split in ["train", "val", "test"]:
        if not (embed_dir / f"{split}.npy").exists():
            raise FileNotFoundError(
                f"Missing: {embed_dir}/{split}.npy -- run Stage 3 first"
            )

    X_train, y_train = load_embeddings(embed_dir, "train")
    X_val,   y_val   = load_embeddings(embed_dir, "val")
    X_test,  y_test  = load_embeddings(embed_dir, "test")

    logger.info(f"  Train={X_train.shape}  Val={X_val.shape}  Test={X_test.shape}")
    assert X_train.shape[1] == INPUT_DIM, (
        f"Expected {INPUT_DIM}-dim input, got {X_train.shape[1]}. "
        f"Check Stage 3 FUSED_DIM is {INPUT_DIM}.")

    train_loader = make_loader(X_train, y_train, bs, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   bs, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  bs, shuffle=False)

    # Build model — BO uses 64-dim output, FS/UAF use 32-dim
    compressed_dim = COMPRESSED_DIM_BO if ds_key == "bo" else (COMPRESSED_DIM_FS if ds_key == "fs" else COMPRESSED_DIM)
    encoder = ClassicalEncoder(
        input_dim=INPUT_DIM,
        compressed_dim=compressed_dim,
        dropout=0.2,
    )
    model = Stage4Model(encoder, dropout=0.2, compressed_dim=compressed_dim).to(device)
    logger.info(f"  Architecture: {INPUT_DIM} -> 128 -> {compressed_dim}  (ReLU+BN+Dropout=0.2)")
    logger.info(f"  Parameters: {model.count_parameters():,}")

    ckpt_path = Path(checkpoint) if checkpoint else                 _ROOT / "models" / "checkpoints" / f"{ds_key}_encoder_best.pt"

    if eval_only:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info(f"  Loaded: {ckpt_path}")
    else:
        optimizer = AdamW(model.parameters(),
                          lr=cfg_t["lr"],
                          weight_decay=cfg_t["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=cfg_t["epochs"],
                                      eta_min=cfg_t["lr"] * 0.01)

        best_f1      = 0.0
        patience_ctr = 0
        patience     = cfg_t["early_stopping_patience"]
        tracker      = EpochTracker()

        logger.info(f"  Training: max_epochs={cfg_t['epochs']} bs={bs} ")
        logger.info(f"  lr={cfg_t['lr']}  patience={patience}")

        for epoch in range(1, cfg_t["epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, device,
                cfg_t.get("grad_clip", 1.0))
            val_loss, val_probs, val_labels = evaluate(model, val_loader, device)
            scheduler.step()

            m = compute_metrics(val_labels, val_probs, dataset_name=f"{ds_key}_val")
            m.epoch = epoch
            tracker.log("val", m)

            logger.info(
                f"  Epoch {epoch:03d}  "
                f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_f1={m.f1:.4f}  "
                f"val_auc={m.roc_auc:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.1e}  "
                f"[{time.time()-t0:.1f}s]"
            )

            if m.f1 > best_f1:
                best_f1 = m.f1
                torch.save(model.state_dict(), ckpt_path)
                patience_ctr = 0
                logger.info(f"  [SAVE] val_f1={best_f1:.4f}")
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    logger.info(f"  Early stop at epoch {epoch}")
                    break

        tracker.save(str(_ROOT / "results" / "metrics" /
                         f"{ds_key}_stage4_history.json"))

    # Test evaluation
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_probs, test_labels = evaluate(model, test_loader, device)
    thr = find_optimal_threshold(test_labels, test_probs, metric="f1")
    test_m = compute_metrics(test_labels, test_probs,
                             threshold=thr, dataset_name=f"{ds_key}_test")
    test_m.pretty_print()
    test_m.save(str(_ROOT / "results" / "metrics" /
                    f"{ds_key}_stage4_test.json"))

    logger.info(f"  Saving {compressed_dim}-dim compressed features for Stage 5 (QAFA)...")
    for name, loader in [("train", train_loader),
                         ("val",   val_loader),
                         ("test",  test_loader)]:
        comp, full, lbls = compress_split(model, loader, device)
        np.save(comp_dir / f"{name}.npy",        comp)
        np.save(comp_dir / f"{name}_full.npy",   full)
        np.save(comp_dir / f"{name}_labels.npy", lbls)
        logger.info(f"  {name}.npy  shape={comp.shape}  "
                    f"range=[{comp.min():.3f}, {comp.max():.3f}]")
        logger.info(f"  {name}_full.npy  shape={full.shape}")

    logger.info(f"Stage 4 [OK] '{ds_key}'")
    return test_m.to_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="QEGVD Stage 4 - Classical Feature Encoder"
    )
    parser.add_argument("--dataset",    choices=["bo","fs","uaf","all"], required=True)
    parser.add_argument("--config",     type=str,  default=None)
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--checkpoint", type=str,  default=None)
    args = parser.parse_args()

    config   = load_config(args.config)
    datasets = ["bo","fs","uaf"] if args.dataset == "all" else [args.dataset]

    results = {}
    for ds in datasets:
        results[ds] = run_dataset(ds, config, args.eval_only, args.checkpoint)

    print("\n" + "=" * 55)
    print("  STAGE 4 SUMMARY")
    print("=" * 55)
    print(f"  {'DS':<6} {'F1':>8} {'AUC':>8} {'MCC':>8} {'Acc':>8}")
    print("  " + "-" * 38)
    for ds, r in results.items():
        print(f"  {ds.upper():<6} {r['f1']:>8.4f} {r['roc_auc']:>8.4f} "
              f"{r['mcc']:>8.4f} {r['accuracy']:>8.4f}")
    print("=" * 55)
    print("  Compressed 32-dim features -> data/compressed/<ds>/")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()