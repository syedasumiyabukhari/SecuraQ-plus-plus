"""
QEGVD -- Inference Pipeline (Section 12)
=========================================
End-to-end pipeline for unseen C/C++ source code.
Three binary classifiers (BO / FS / UAF) run in parallel.

Steps (paper Section 12 exact):
    1.  Source code input
    2.  Identifier masking (Stage 1 normalisation reused)
    3.  Static analysis -> 7 graphs (Stage 2)
    4.  GAT encoding -> 128-dim (Stage 3 checkpoint)
    5.  Classical encoder -> 32-dim (Stage 4 checkpoint)
    6.  QAFA selection -> top-16 -> 2x8 angle features (Stage 5 metadata)
    7.  VQC processing -> 4-dim quantum vector (Stage 6 checkpoint)
    8.  Hybrid fusion -> concat(32,4) = 36-dim
    9.  MLP classifier -> P(vulnerable) (Stage 8 checkpoint)
    10. Explainability -> SHAP report

Usage
-----
    python src/inference.py --code "void foo(char *s){char buf[10]; strcpy(buf,s);}"
    python src/inference.py --file path/to/function.c
    python src/inference.py --file func.c --explain
"""

from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np, yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
(_ROOT / "logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)),
        logging.FileHandler(_ROOT/"logs"/"inference.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("Inference")

try:
    import torch
except ImportError:
    logger.error("torch not found"); sys.exit(1)

import torch.nn as nn
from stage7_fusion import HybridMLP, CLASSICAL_DIM, QUANTUM_DIM, HYBRID_DIM

DS_KEYS = ["bo", "fs", "uaf"]
VULN_LABEL = {"bo":"Buffer Overflow","fs":"Format String","uaf":"Use-After-Free"}
THRESHOLD = 0.5


# ── Step 2: Identifier masking ────────────────────────────
def mask_identifiers(code: str) -> str:
    """
    Reuse Stage 1 normalisation: neutralise function names and
    user-defined identifiers to prevent label leakage.
    """
    import re
    # Remove good/bad markers (Stage 1 audit patterns)
    code = re.sub(r'\b(good|bad|Good|Bad|CWE|Juliet)\w*\b', 'FUNC', code)
    code = re.sub(r'//[^\n]*', '', code)   # strip line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', ' ', code).strip()
    return code


# ── Steps 3-4: Graph construction + GAT embedding ─────────
def get_gat_embedding(code: str, config: dict) -> np.ndarray:
    """
    Build 7 graphs via Stage 2, run trained GAT (Stage 3).
    Returns (1, 128) numpy array.
    """
    try:
        import stage2_graph_construction as s2
        import torch
        from stage3_gat import QEGVDStage3, load_config as s3_cfg
        from torch_geometric.data import Batch

        # Build graphs
        bundle = s2.build_graph_bundle(code, sample_id="inference")
        if not bundle.is_valid():
            logger.warning("Graph construction failed -- using zero embedding")
            return np.zeros((1, 128), dtype=np.float32)

        # Load GAT (use first available checkpoint)
        for ds in DS_KEYS:
            ckpt = _ROOT/"models"/"checkpoints"/f"{ds}_gat_best.pt"
            if ckpt.exists():
                gat_config = yaml.safe_load(open(_ROOT/"configs"/"config.yaml"))
                model = QEGVDStage3(gat_config)
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                model.eval()
                with torch.no_grad():
                    emb = model.embed([bundle])  # (1, 128)
                return emb.numpy()

        logger.warning("No GAT checkpoint found -- using zero embedding")
        return np.zeros((1, 128), dtype=np.float32)

    except Exception as e:
        logger.warning(f"GAT embedding failed ({e}) -- using zero embedding")
        return np.zeros((1, 128), dtype=np.float32)


# ── Step 5: Classical encoder ─────────────────────────────
def get_classical_encoding(gat_emb: np.ndarray, ds_key: str) -> np.ndarray:
    """Stage 4 encoder: 128 -> 32."""
    from stage4_classical_encoder import ClassicalEncoder
    ckpt = _ROOT/"models"/"checkpoints"/f"{ds_key}_encoder_best.pt"
    enc  = ClassicalEncoder(input_dim=128, compressed_dim=32, dropout=0.0)
    if ckpt.exists():
        from stage4_classical_encoder import Stage4Model
        full = Stage4Model(enc, dropout=0.0)
        full.load_state_dict(torch.load(ckpt, map_location="cpu"))
        full.eval()
        with torch.no_grad():
            compressed = full.encoder(
                torch.from_numpy(gat_emb.astype(np.float32))
            ).numpy()
    else:
        logger.warning(f"No encoder checkpoint for {ds_key} -- using zeros")
        compressed = np.zeros((1, 32), dtype=np.float32)
    return compressed   # (1, 32)


# ── Step 6: QAFA feature selection and angle encoding ─────
def get_qafa_angles(classical: np.ndarray, ds_key: str) -> tuple:
    """
    Load pre-computed QAFA feature selection metadata,
    apply to new sample, return (stage1_angles, stage2_angles) each (1,8).
    """
    import json
    import numpy as np

    qafa_dir = _ROOT/"data"/"qafa"/ds_key
    idx_path = qafa_dir/"selected_indices.npy"
    if not idx_path.exists():
        logger.warning(f"No QAFA metadata for {ds_key} -- using first 16 features")
        selected = np.arange(16)
    else:
        selected = np.load(idx_path)   # (16,)

    x_sel  = classical[:, selected]           # (1, 16)
    stage1 = np.tanh(x_sel[:, :8]) * np.pi   # (1, 8)
    stage2 = np.tanh(x_sel[:, 8:]) * np.pi   # (1, 8)
    return stage1.astype(np.float32), stage2.astype(np.float32)


# ── Step 7: VQC quantum vector ────────────────────────────
def get_quantum_vector(s1: np.ndarray, s2: np.ndarray,
                       ds_key: str) -> np.ndarray:
    """Stage 6 VQC: 2x8 angles -> (1, 4) quantum vector."""
    from stage6_vqc import VQCLayer, VQCPretrainModel, N_VAR_LAYERS
    ckpt = _ROOT/"models"/"checkpoints"/f"{ds_key}_vqc_best.pt"
    vqc   = VQCLayer(N_VAR_LAYERS)
    model = VQCPretrainModel(vqc)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        qvec = vqc(
            torch.from_numpy(s1),
            torch.from_numpy(s2),
        ).numpy()  # (1, 4)
    return qvec


# ── Steps 8+9: Hybrid fusion + classification ─────────────
def classify(classical: np.ndarray, quantum: np.ndarray,
             ds_key: str, config: dict) -> float:
    """concat(32,4)=36 -> MLP -> probability."""
    hybrid = np.concatenate([classical, quantum], axis=1)  # (1,36)
    model  = HybridMLP(
        hidden  = config["classifier"]["hidden_dim"],
        dropout = 0.0,
    )
    ckpt = _ROOT/"models"/"final"/f"{ds_key}_hybrid_best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(
            model(torch.from_numpy(hybrid.astype(np.float32)))
        ).item()
    return prob


# ── Step 10: Quick SHAP attribution ───────────────────────
def quick_shap(classical, quantum, ds_key, config):
    """
    Minimal SHAP attribution for single-sample inference report.
    Uses gradient*input as a fast proxy.
    """
    from stage9_explainability import (
        ALL_NAMES, _grad_attribution, load_model as load_clf
    )
    import numpy as np
    clf    = load_clf(ds_key, config)
    hybrid = np.concatenate([classical, quantum], axis=1).astype(np.float32)
    attrs  = _grad_attribution(clf, hybrid)[0]   # (36,)
    ranked = sorted(enumerate(attrs), key=lambda x:abs(x[1]), reverse=True)[:4]
    return [(ALL_NAMES[i], round(float(v),4)) for i,v in ranked]


# ── Full inference function ───────────────────────────────
def run_inference(code: str, config: dict, explain: bool = False) -> dict:
    """
    Run the full QEGVD pipeline on a single code function.
    Returns dict with predictions for BO, FS, UAF.
    """
    results = {"predictions": {}, "dominant": None, "code_length": len(code)}

    # Step 2: Identifier masking
    logger.info("Step 2: Masking identifiers...")
    masked = mask_identifiers(code)

    # Step 4: GAT embedding (shared across all 3 classifiers)
    logger.info("Step 4: GAT embedding (128-dim)...")
    gat_emb = get_gat_embedding(masked, config)   # (1, 128)

    best_prob = 0.0
    best_ds   = None

    for ds in DS_KEYS:
        logger.info(f"  Processing classifier: {VULN_LABEL[ds]}")

        # Step 5
        classical = get_classical_encoding(gat_emb, ds)   # (1,32)

        # Step 6
        s1, s2 = get_qafa_angles(classical, ds)           # (1,8),(1,8)

        # Step 7
        quantum = get_quantum_vector(s1, s2, ds)          # (1,4)

        # Steps 8+9
        prob = classify(classical, quantum, ds, config)

        results["predictions"][ds] = {
            "vuln_type":  VULN_LABEL[ds],
            "probability": round(prob, 4),
            "verdict":    "VULNERABLE" if prob >= THRESHOLD else "SAFE",
        }

        if prob > best_prob:
            best_prob = prob; best_ds = ds

        # Step 10: Explainability
        if explain:
            try:
                top_feats = quick_shap(classical, quantum, ds, config)
                results["predictions"][ds]["top_shap"] = top_feats
            except Exception as e:
                logger.warning(f"  SHAP failed for {ds}: {e}")

    results["dominant"] = VULN_LABEL.get(best_ds, "SAFE") if best_prob >= THRESHOLD else "SAFE"
    return results


def print_results(res: dict):
    print("\n" + "="*65)
    print("  QEGVD -- Vulnerability Assessment")
    print("="*65)
    for ds, r in res["predictions"].items():
        verdict_tag = " <-- DETECTED" if r["verdict"]=="VULNERABLE" else ""
        print(f"  P({r['vuln_type']:<20}) = {r['probability']:.4f}  "
              f"->  {r['verdict']}{verdict_tag}")
        if "top_shap" in r:
            print(f"    Top SHAP features:")
            for name, val in r["top_shap"][:3]:
                sign = "+" if val > 0 else ""
                print(f"      {name:<44}  {sign}{val:.4f}")
    print(f"\n  Dominant finding: {res['dominant']}")
    print("="*65+"\n")


def load_config(path=None):
    return yaml.safe_load(open(path or _ROOT/"configs"/"config.yaml"))


def main():
    p = argparse.ArgumentParser(description="QEGVD Inference -- single function analysis")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--code", type=str, help="C/C++ function as string")
    g.add_argument("--file", type=str, help="Path to .c/.cpp file")
    p.add_argument("--config",   default=None)
    p.add_argument("--explain",  action="store_true", help="Include SHAP attribution")
    p.add_argument("--output",   default=None, help="Save JSON results to file")
    args = p.parse_args()

    config = load_config(args.config)

    if args.file:
        code = open(args.file).read()
    else:
        code = args.code

    logger.info(f"Analysing function ({len(code)} chars)...")
    results = run_inference(code, config, explain=args.explain)
    print_results(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved: {args.output}")


if __name__ == "__main__":
    main()
