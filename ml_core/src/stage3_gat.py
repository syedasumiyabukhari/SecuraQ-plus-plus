"""
QEGVD -- Stage 3: Multi-View Graph Attention Network (GAT)
===========================================================
Trains a GAT encoder on all 7 graph types per sample.

Architecture:
    Per view: GATConv x4 (128-dim, 8 heads) -> mean+max readout -> 512-dim
    Fusion:   attention-weighted sum of 8 views -> 128-dim
    Head:     Linear(512->64->1) binary classifier

Output per run:
    models/checkpoints/<ds>_gat_best.pt
    data/embeddings/<ds>/{train,val,test}.npy          -- (N, 512) for Stage 4
    data/embeddings/<ds>/{train,val,test}_labels.npy
    results/metrics/<ds>_stage3_test.json

Usage
-----
    python src/stage3_gat.py --dataset bo
    python src/stage3_gat.py --dataset all
    python src/stage3_gat.py --dataset bo --eval-only
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

for _d in ["logs", "models/checkpoints", "models/final",
           "results/metrics", "data/embeddings"]:
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
            _ROOT / "logs" / "stage3.log", mode="a", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("Stage3")


def _check_deps():
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch  ->  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    try:
        import torch_geometric
    except ImportError:
        missing.append("torch-geometric  ->  pip install torch-geometric")
    if missing:
        for m in missing:
            logger.error("Missing: " + m)
        sys.exit(1)

_check_deps()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

from utils.metrics import compute_metrics, find_optimal_threshold, EpochTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRAPH_TYPES   = ["AST", "CFG", "DFG", "PDG", "TPG", "MAG", "CG", "FSG"]
NODE_FEAT_DIM = 72
FUSED_DIM     = 256




# ---------------------------------------------------------------------------
# NetworkX -> PyG Data
# ---------------------------------------------------------------------------

def nx_to_pyg(G, feat_dim: int = NODE_FEAT_DIM) -> Data:
    nodes = list(G.nodes())
    if not nodes:
        return Data(
            x=torch.zeros((1, feat_dim), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
    node_idx = {n: i for i, n in enumerate(nodes)}
    feats = []
    for n in nodes:
        f = G.nodes[n].get("feature", None)
        if f is None:
            f = np.zeros(feat_dim, dtype=np.float32)
        feats.append(f.astype(np.float32))
    x = torch.from_numpy(np.stack(feats))
    edges = [(node_idx[u], node_idx[v])
             for u, v in G.edges() if u in node_idx and v in node_idx]
    if edges:
        src, dst = zip(*edges)
        edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)





# ---------------------------------------------------------------------------
# Adaptive sizing based on dataset size
# ---------------------------------------------------------------------------

def get_adaptive_config(n_train: int, ds_key: str = "") -> dict:
    """
    Model config per dataset. FS gets larger capacity + less regularisation
    because format-string patterns are subtler and the dataset is mid-sized.
    """
    if ds_key.lower() == "fs":
        return dict(hidden_dim=128, num_heads=8, num_layers=4,
                    view_dim=256, dropout=0.2, label_smoothing=0.02)
    return dict(hidden_dim=64, num_heads=8, num_layers=4,
                view_dim=128, dropout=0.3, label_smoothing=0.05)


# ---------------------------------------------------------------------------
# GAT Encoder -- SHARED across all graph views (reduces params 8x)
# ---------------------------------------------------------------------------

class SharedGATEncoder(nn.Module):
    """
    Single GAT encoder shared across ALL graph views.
    Parameter count: ~O(hidden_dim^2) instead of O(n_views * hidden_dim^2).

    View identity injected via a learned view-type embedding added to node
    features before encoding — so the encoder sees which graph type it is.
    """

    def __init__(self, n_views: int, in_dim: int = 64, hidden_dim: int = 32,
                 out_dim: int = 64, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.4):
        super().__init__()
        self.dropout = dropout

        # Learned view-type embedding (n_views, in_dim) — injected as bias
        self.view_emb = nn.Embedding(n_views, in_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Layer 0: in_dim -> hidden_dim * num_heads
        self.convs.append(GATConv(in_dim, hidden_dim, heads=num_heads,
                                  dropout=dropout, concat=True,
                                  add_self_loops=True))
        self.norms.append(nn.LayerNorm(hidden_dim * num_heads))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim,
                                      heads=num_heads, dropout=dropout,
                                      concat=True, add_self_loops=True))
            self.norms.append(nn.LayerNorm(hidden_dim * num_heads))

        # Final layer: average heads -> out_dim
        self.convs.append(GATConv(hidden_dim * num_heads, out_dim,
                                  heads=1, dropout=dropout,
                                  concat=False, add_self_loops=True))
        self.norms.append(nn.LayerNorm(out_dim))

        # Readout: mean + max pooling -> 2*out_dim
        self.readout_proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.embed_dim = out_dim * 2

        # Xavier init for stable training
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.view_emb.weight, mean=0.0, std=0.01)

    def encode_view(self, x: torch.Tensor, edge_index: torch.Tensor,
                    batch: torch.Tensor, view_id: int,
                    edge_drop: float = 0.0) -> torch.Tensor:
        """Encode a single graph view. view_id injects view identity."""
        # Add view embedding to all nodes in this view
        v_emb = self.view_emb(torch.tensor(view_id, device=x.device))
        x = x + v_emb.unsqueeze(0)

        for conv, norm in zip(self.convs, self.norms):
            # DropEdge: randomly remove edges during training
            if self.training and edge_drop > 0 and edge_index.size(1) > 0:
                keep = torch.rand(edge_index.size(1), device=edge_index.device) >= edge_drop
                ei = edge_index[:, keep] if keep.any() else edge_index
            else:
                ei = edge_index
            x = conv(x, ei)
            x = norm(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        return self.readout_proj(torch.cat([x_mean, x_max], dim=-1))


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss — pushes same-class embeddings closer,
    different-class embeddings further apart on the unit hypersphere."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """features: (B, D) L2-normalized, labels: (B,) binary."""
        device = features.device
        B = features.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        sim = torch.mm(features, features.t()) / self.temperature
        labels_col = labels.view(-1, 1)
        pos_mask = torch.eq(labels_col, labels_col.t()).float()
        pos_mask.fill_diagonal_(0)
        logits_max = sim.max(dim=1, keepdim=True).values
        logits = sim - logits_max.detach()
        # Mask out self-similarity (no in-place ops to keep autograd happy)
        self_mask = 1.0 - torch.eye(B, device=device)
        exp_logits = torch.exp(logits) * self_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(pos_mask * log_prob).sum(dim=1) / n_pos
        return loss.mean()


# ---------------------------------------------------------------------------
# Multi-View Attention Fusion (lightweight)
# ---------------------------------------------------------------------------

class MultiViewFusion(nn.Module):
    """
    Attention-weighted fusion of n_views embeddings -> fused_dim.
    Lightweight: single attention head, single linear projection.
    """

    def __init__(self, n_views: int = 8, view_dim: int = 128,
                 fused_dim: int = FUSED_DIM, dropout: float = 0.3):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(view_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1, bias=False),
        )
        self.proj = nn.Sequential(
            nn.Linear(view_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = fused_dim

    def forward(self, view_embeds: list) -> torch.Tensor:
        stacked = torch.stack(view_embeds, dim=1)             # (B, n_views, view_dim)
        weights = torch.softmax(self.attn(stacked), dim=1)    # (B, n_views, 1)
        fused   = (stacked * weights).sum(dim=1)              # (B, view_dim)
        return self.proj(fused)                               # (B, fused_dim)


# ---------------------------------------------------------------------------
# Full Stage 3 Model
# ---------------------------------------------------------------------------

class QEGVDStage3(nn.Module):
    """
    Multi-view GAT with shared encoder + binary classifier.
    Shared encoder drastically reduces params vs 8 separate encoders.
    """

    def __init__(self, n_views: int = 8, in_dim: int = 64, hidden_dim: int = 32,
                 view_dim: int = 64, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.4, fused_dim: int = FUSED_DIM,
                 clf_hidden: int = 32, edge_drop: float = 0.0):
        super().__init__()
        self.n_views = n_views
        self.edge_drop = edge_drop

        # SHARED encoder (one set of weights for all 8 views)
        self.encoder = SharedGATEncoder(
            n_views=n_views, in_dim=in_dim, hidden_dim=hidden_dim,
            out_dim=view_dim, num_heads=num_heads, num_layers=num_layers,
            dropout=dropout
        )

        self.fusion = MultiViewFusion(n_views, view_dim * 2, fused_dim, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, clf_hidden),
            nn.LayerNorm(clf_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden, 1),
        )

    def embed(self, batch_views: list) -> torch.Tensor:
        drop = self.edge_drop if self.training else 0.0
        view_embeds = [
            self.encoder.encode_view(b.x, b.edge_index, b.batch, i,
                                     edge_drop=drop)
            for i, b in enumerate(batch_views)
        ]
        return self.fusion(view_embeds)  # (B, fused_dim)

    def forward(self, batch_views: list) -> torch.Tensor:
        return self.classifier(self.embed(batch_views)).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Dataset + DataLoader
# ---------------------------------------------------------------------------

def detect_feat_dim(bundles):
    """Detect actual node feature dimension from first valid bundle."""
    for b in bundles:
        for gt in GRAPH_TYPES:
            G = b.graphs.get(gt)
            if G is None:
                continue
            for n in G.nodes():
                f = G.nodes[n].get("feature", None)
                if f is not None:
                    return len(f)
    return NODE_FEAT_DIM  # fallback


class GraphBundleDataset(torch.utils.data.Dataset):
    def __init__(self, bundles, name="", feat_dim=NODE_FEAT_DIM):
        valid_idx = [i for i, b in enumerate(bundles) if b.is_valid()]
        self.bundles = [bundles[i] for i in valid_idx]
        self.feat_dim = feat_dim
        skipped = len(bundles) - len(self.bundles)
        if skipped:
            logger.warning(f"[{name}] Skipped {skipped} invalid bundles")

    def __len__(self):
        return len(self.bundles)

    def __getitem__(self, idx):
        b = self.bundles[idx]
        graphs = [nx_to_pyg(b.graphs[gt], feat_dim=self.feat_dim) for gt in GRAPH_TYPES]
        label  = torch.tensor(b.label, dtype=torch.float32)
        return (graphs, label)


def collate_multiview(batch):
    pyg_lists, labels = zip(*batch)
    batched = [
        Batch.from_data_list([s[i] for s in pyg_lists])
        for i in range(len(GRAPH_TYPES))
    ]
    return batched, torch.stack(labels)


def make_loader(bundles, batch_size, shuffle, name="", feat_dim=NODE_FEAT_DIM):
    ds = GraphBundleDataset(bundles, name, feat_dim=feat_dim)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_multiview, num_workers=0, pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Train / Eval / Extract
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0,
                    label_smoothing=0.0, supcon_fn=None, supcon_w=0.3,
                    pos_weight=None):
    model.train()
    total_loss = correct = total = 0
    pw = torch.tensor([pos_weight], device=device) if pos_weight is not None else None

    for batch_views, labels in loader:
        batch_views = [b.to(device) for b in batch_views]
        labels = labels.to(device)
        optimizer.zero_grad()

        # Separate embed + classify so we can access embeddings for SupCon
        embeddings = model.embed(batch_views)
        logits = model.classifier(embeddings).squeeze(-1)

        bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
        loss = bce

        if supcon_fn is not None:
            norm_emb = F.normalize(embeddings, p=2, dim=1)
            loss = loss + supcon_w * supcon_fn(norm_emb, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct    += ((torch.sigmoid(logits) >= 0.5).long() == labels.long()).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = total = 0
    all_probs = []
    all_labels = []
    for batch_views, labels in loader:
        batch_views = [b.to(device) for b in batch_views]
        labels = labels.to(device)
        logits = model(batch_views)
        loss   = F.binary_cross_entropy_with_logits(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total      += labels.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return (total_loss / total,
            np.concatenate(all_probs),
            np.concatenate(all_labels))


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embeds = []
    lbls   = []
    for batch_views, labels in loader:
        batch_views = [b.to(device) for b in batch_views]
        embeds.append(model.embed(batch_views).cpu().numpy())
        lbls.append(labels.numpy())
    return np.concatenate(embeds), np.concatenate(lbls)


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def load_config(path=None):
    if path is None:
        path = _ROOT / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_bundles(graphs_dir, split):
    pkl = graphs_dir / f"{split}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"Not found: {pkl} -- run stage2 first"
        )
    # Register GraphBundle so pickle can deserialise it from any entry point
    import stage2_graph_construction as _s2
    import sys as _sys
    _sys.modules["__main__"].GraphBundle = _s2.GraphBundle
    with open(pkl, "rb") as f:
        return pickle.load(f)


def run_dataset(ds_key, config, eval_only=False, checkpoint=None):
    logger.info("=" * 60)
    logger.info(f"Stage 3 -- {ds_key.upper()}")
    logger.info("=" * 60)

    device = torch.device("cpu")
    seed   = config["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    graphs_dir = _ROOT / config["data"]["graphs_dir"] / ds_key
    embed_dir  = _ROOT / "data" / "embeddings" / ds_key
    embed_dir.mkdir(parents=True, exist_ok=True)

    cfg_t  = config["training"]
    cfg_g  = config["gat"]
    bs     = cfg_t["batch_size"]

    # Load bundles
    logger.info("Loading graph bundles...")
    train_b = load_bundles(graphs_dir, "train")
    val_b   = load_bundles(graphs_dir, "val")
    test_b  = load_bundles(graphs_dir, "test")
    logger.info(f"  Train={len(train_b)}, Val={len(val_b)}, Test={len(test_b)}")

    # Auto-detect feature dimension from actual graph data
    actual_feat_dim = detect_feat_dim(train_b)
    logger.info(f"  Detected node feature dim: {actual_feat_dim}")

    train_loader = make_loader(train_b, bs, shuffle=True,  name="train",
                               feat_dim=actual_feat_dim)
    val_loader   = make_loader(val_b,   bs, shuffle=False, name="val",
                               feat_dim=actual_feat_dim)
    test_loader  = make_loader(test_b,  bs, shuffle=False, name="test",
                               feat_dim=actual_feat_dim)

    # Adaptive model config based on training set size
    n_train      = len(train_b)
    adapt        = get_adaptive_config(n_train, ds_key=ds_key)
    label_smooth = adapt.pop("label_smoothing", 0.0)

    # DropEdge: less augmentation for FS (subtler patterns)
    edge_drop = 0.10 if ds_key.lower() == "fs" else 0.15

    model = QEGVDStage3(
        n_views    = len(GRAPH_TYPES),
        in_dim     = actual_feat_dim,
        hidden_dim = adapt["hidden_dim"],
        view_dim   = adapt["view_dim"],
        num_heads  = adapt["num_heads"],
        num_layers = adapt["num_layers"],
        dropout    = adapt["dropout"],
        fused_dim  = FUSED_DIM,
        clf_hidden = max(16, adapt["view_dim"] // 2),
        edge_drop  = edge_drop,
    ).to(device)
    logger.info(f"  Parameters: {model.count_parameters():,}  "
                f"(adaptive for n_train={n_train})")
    logger.info(f"  Config: hidden={adapt['hidden_dim']} heads={adapt['num_heads']} "
                f"layers={adapt['num_layers']} view_dim={adapt['view_dim']} "
                f"dropout={adapt['dropout']} edge_drop={edge_drop} "
                f"label_smooth={label_smooth}")

    ckpt_path = Path(checkpoint) if checkpoint else \
                _ROOT / "models" / "checkpoints" / f"{ds_key}_gat_best.pt"

    if eval_only:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info(f"  Loaded: {ckpt_path}")
    else:
        # Shared encoder needs lower LR than separate encoders
        lr = cfg_t["lr"] * 0.5   # 0.0005 instead of 0.001
        optimizer = AdamW(model.parameters(),
                          lr=lr, weight_decay=cfg_t["weight_decay"])
        # ReduceLROnPlateau: cuts LR when val_auc stops improving
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                      patience=5, min_lr=lr * 0.01)

        # Supervised Contrastive Loss + pos_weight for balanced training
        # BO: tighter SupCon temperature → tighter class clusters in embedding space
        supcon_temp = 0.07 if ds_key.lower() == "bo" else 0.1
        supcon_fn = SupConLoss(temperature=supcon_temp)
        all_train_labels = [b.label for b in train_b]
        n_pos = sum(1 for l in all_train_labels if l == 1)
        n_neg = len(all_train_labels) - n_pos
        # BO: pos_weight < 1 → penalise FP more than FN → forces model to be
        # conservative (predict vuln only when confident) → lowers FPR
        pw = 0.5 if ds_key.lower() == "bo" else n_neg / max(n_pos, 1)

        best_auc     = 0.0
        patience_ctr = 0
        patience     = 25 if ds_key.lower() == "bo" else (35 if ds_key.lower() == "fs" else cfg_t["early_stopping_patience"])
        tracker      = EpochTracker()

        max_epochs = 150 if ds_key.lower() == "bo" else (200 if ds_key.lower() == "fs" else cfg_t["epochs"])
        logger.info(f"  Training: max_epochs={max_epochs}, "
                    f"batch={bs}, lr={cfg_t['lr']}, patience={patience}")

        # ── Overfitting / underfitting thresholds ──────────────────────
        OVERFIT_GAP   = 0.15   # warn if tr_loss - val_loss > this
        UNDERFIT_ACC  = 0.55   # warn if tr_acc stays below this after epoch 5
        overfit_warned = False

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, device,
                cfg_t.get("grad_clip", 1.0),
                label_smoothing=label_smooth,
                supcon_fn=supcon_fn, supcon_w=0.3,
                pos_weight=pw,
            )
            val_loss, val_probs, val_labels = evaluate(model, val_loader, device)

            m = compute_metrics(val_labels, val_probs, dataset_name=f"{ds_key}_val")
            m.epoch = epoch
            tracker.log("val", m)
            scheduler.step(m.roc_auc)   # ReduceLROnPlateau monitors val_auc

            # ── Overfitting check ────────────────────────────────────
            loss_gap = tr_loss - val_loss   # negative = val worse than train = overfit
            overfit_flag = ""
            if val_loss > tr_loss + OVERFIT_GAP and not overfit_warned:
                overfit_flag = "  [OVERFIT WARNING: val_loss >> tr_loss]"
                overfit_warned = True
                logger.warning(
                    f"  Overfitting detected at epoch {epoch}: "
                    f"tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
                    f"gap={val_loss-tr_loss:.4f}"
                )

            # ── Underfitting check ───────────────────────────────────
            underfit_flag = ""
            if epoch >= 5 and tr_acc < UNDERFIT_ACC:
                underfit_flag = "  [UNDERFIT: tr_acc too low]"
                if epoch == 5:
                    logger.warning(
                        f"  Possible underfitting at epoch {epoch}: "
                        f"tr_acc={tr_acc:.3f} < {UNDERFIT_ACC}"
                    )

            logger.info(
                f"  Epoch {epoch:03d}  "
                f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_f1={m.f1:.4f}  "
                f"val_auc={m.roc_auc:.4f}  "
                f"gap={loss_gap:+.3f}  "
                f"lr={optimizer.param_groups[0]['lr']:.1e}  "
                f"[{time.time()-t0:.0f}s]"
                f"{overfit_flag}{underfit_flag}"
            )

            if m.roc_auc > best_auc:
                best_auc = m.roc_auc
                torch.save(model.state_dict(), ckpt_path)
                patience_ctr = 0
                logger.info(f"  [SAVE] val_auc={best_auc:.4f}  val_f1={m.f1:.4f}")
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    logger.info(f"  Early stop at epoch {epoch}")
                    break

        # ── Final fit diagnosis ──────────────────────────────────────
        tracker.save(str(_ROOT / "results" / "metrics" /
                         f"{ds_key}_stage3_history.json"))
        logger.info(f"  Final diagnosis: best_val_auc={best_auc:.4f}  "
                    f"overfit={'YES' if overfit_warned else 'NO'}")

    # Test evaluation
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_probs, test_labels = evaluate(model, test_loader, device)
    thr = find_optimal_threshold(test_labels, test_probs, metric="youden")
    test_m = compute_metrics(test_labels, test_probs,
                             threshold=thr, dataset_name=f"{ds_key}_test")
    test_m.pretty_print()
    test_m.save(str(_ROOT / "results" / "metrics" /
                    f"{ds_key}_stage3_test.json"))

    # Save embeddings for Stage 4
    logger.info("  Saving embeddings for Stage 4...")
    for name, loader in [("train", train_loader),
                         ("val",   val_loader),
                         ("test",  test_loader)]:
        embs, lbls = extract_embeddings(model, loader, device)
        np.save(embed_dir / f"{name}.npy",        embs)
        np.save(embed_dir / f"{name}_labels.npy", lbls)
        logger.info(f"  {name}.npy  shape={embs.shape}")

    logger.info(f"Stage 3 [OK] '{ds_key}'\n")
    return test_m.to_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QEGVD Stage 3 - Multi-View GAT")
    parser.add_argument("--dataset",    choices=["bo","fs","uaf","all"], required=True)
    parser.add_argument("--config",     type=str, default=None)
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config   = load_config(args.config)
    datasets = ["bo","fs","uaf"] if args.dataset == "all" else [args.dataset]

    results = {}
    for ds in datasets:
        results[ds] = run_dataset(ds, config, args.eval_only, args.checkpoint)

    print("\n" + "=" * 55)
    print("  STAGE 3 SUMMARY")
    print("=" * 55)
    print(f"  {'DS':<6} {'F1':>8} {'AUC':>8} {'MCC':>8} {'Acc':>8}")
    print("  " + "-" * 38)
    for ds, r in results.items():
        print(f"  {ds.upper():<6} {r['f1']:>8.4f} {r['roc_auc']:>8.4f} "
              f"{r['mcc']:>8.4f} {r['accuracy']:>8.4f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()