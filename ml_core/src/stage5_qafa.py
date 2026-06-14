"""
QEGVD -- Stage 5: Quantum-Aware Feature Alignment (QAFA)
=========================================================
Takes the 32-dim compressed features from Stage 4 and selects the
top 16 most vulnerability-relevant features for quantum circuit input.

Paper spec (exact):
    Input:  32-dim latent vector
    Method: Composite ranking over 3 criteria:
              S_i = alpha * MI_norm(i)  +  beta * SHAP_norm(i)
                  + gamma * Centrality_norm(i)
    Select: Top 16 features by composite score S_i
    Encode: 16 features split into 2 groups of 8
            Rescale each group to [-pi, pi] via tanh
    Output: Two 8-dim angle-encoded arrays per sample (Stage 1 + Stage 2)

Output files:
    data/qafa/<ds>/train_{s1,s2}.npy       -- angle-encoded, shape (N, 8)
    data/qafa/<ds>/train_labels.npy
    data/qafa/<ds>/feature_scores.json     -- importance scores for all 32 features
    data/qafa/<ds>/selected_indices.npy    -- top 16 feature indices
    results/metrics/<ds>_stage5_qafa.json

Usage
-----
    python src/stage5_qafa.py --dataset bo
    python src/stage5_qafa.py --dataset all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

for _d in ["logs", "data/qafa", "results/metrics"]:
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
            _ROOT / "logs" / "stage5.log", mode="a", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("Stage5")

# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------
INPUT_DIM    = 32    # Stage 4 default output (FS/UAF); BO uses 64 — inferred at runtime
N_SELECTED   = 16   # top features selected for quantum input
N_QUBITS     = 8    # features per quantum encoding stage
N_STAGES     = 2    # two-stage data re-uploading (8 + 8 = 16)
ALPHA        = 0.40  # MI weight
BETA         = 0.35  # SHAP weight
GAMMA        = 0.25  # centrality weight
PI           = np.pi


# ---------------------------------------------------------------------------
# Criterion 1: Mutual Information
# ---------------------------------------------------------------------------

def compute_mutual_information(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute MI between each feature and the binary label.
    Returns normalised MI scores in [0, 1], shape (n_features,).
    """
    from sklearn.feature_selection import mutual_info_classif

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi_scores = mutual_info_classif(
            X, y.astype(int),
            discrete_features=False,
            n_neighbors=5,
            random_state=42,
        )

    # Normalise to [0, 1]
    mi_max = mi_scores.max()
    if mi_max > 0:
        mi_scores = mi_scores / mi_max
    return mi_scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Criterion 2: SHAP Values
# ---------------------------------------------------------------------------

def compute_shap_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_bg: Optional[np.ndarray] = None,
    n_bg: int = 100,
) -> np.ndarray:
    """
    Train a lightweight LightGBM or RandomForest, then compute SHAP
    feature importances as mean |SHAP value| per feature.
    Returns normalised scores in [0, 1], shape (n_features,).
    Falls back to permutation importance if SHAP/LGBM unavailable.
    """
    n_features = X_train.shape[1]

    # Try tree-based SHAP first
    try:
        import shap
        from sklearn.ensemble import GradientBoostingClassifier

        logger.info("  SHAP: fitting GradientBoostingClassifier...")
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )
        clf.fit(X_train, y_train.astype(int))

        background = X_train[:n_bg] if X_bg is None else X_bg[:n_bg]
        explainer  = shap.TreeExplainer(clf)
        shap_vals  = explainer.shap_values(background)

        # For binary classifier, shap_values may be list[2] or single array
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # class 1

        importance = np.abs(shap_vals).mean(axis=0)
        logger.info("  SHAP: computed via TreeExplainer")

    except Exception as exc:
        logger.warning(f"  SHAP TreeExplainer failed ({exc}), "
                       f"falling back to permutation importance")
        importance = _permutation_importance(X_train, y_train)

    imp_max = importance.max()
    if imp_max > 0:
        importance = importance / imp_max
    return importance.astype(np.float32)


def _permutation_importance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fallback: permutation importance via RandomForest.
    Mean decrease in accuracy when each feature is shuffled.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    clf = RandomForestClassifier(n_estimators=50, max_depth=6,
                                 random_state=42, n_jobs=1)
    clf.fit(X, y.astype(int))

    base_acc = accuracy_score(y, clf.predict(X))
    importances = np.zeros(X.shape[1], dtype=np.float32)
    rng = np.random.default_rng(42)

    for j in range(X.shape[1]):
        X_perm = X.copy()
        X_perm[:, j] = rng.permutation(X_perm[:, j])
        importances[j] = max(0, base_acc - accuracy_score(y, clf.predict(X_perm)))

    return importances


# ---------------------------------------------------------------------------
# Criterion 3: Graph Centrality Scores
# ---------------------------------------------------------------------------

def compute_centrality_scores(
    graphs_dir: Path,
    n_features: int = INPUT_DIM,
    n_samples: int  = 200,
) -> np.ndarray:
    """
    Extract degree, betweenness, and PageRank centrality from program graphs,
    then map to feature-space importance scores.

    Strategy:
        - Sample up to n_samples GraphBundles from the train split
        - Compute centrality per node per graph type
        - Aggregate centrality into n_features bins (matching embedding dims)
        - Average across samples -> per-feature centrality importance

    Returns normalised scores in [0, 1], shape (n_features,).
    """
    import pickle
    import networkx as nx

    pkl_path = graphs_dir / "train.pkl"
    if not pkl_path.exists():
        logger.warning("  Centrality: graph pkl not found, using uniform scores")
        return np.ones(n_features, dtype=np.float32) / n_features

    try:
        import stage2_graph_construction as _s2
        import sys as _sys
        _sys.modules["__main__"].GraphBundle = _s2.GraphBundle

        with open(pkl_path, "rb") as f:
            bundles = pickle.load(f)
    except Exception as exc:
        logger.warning(f"  Centrality: failed to load graphs ({exc}), using uniform")
        return np.ones(n_features, dtype=np.float32) / n_features

    bundles = bundles[:n_samples]
    centrality_acc = np.zeros(n_features, dtype=np.float64)
    count = 0

    graph_types = ["AST", "CFG", "DFG", "PDG", "TPG", "MAG", "CG"]

    for bundle in bundles:
        if not bundle.is_valid():
            continue
        sample_scores = np.zeros(n_features, dtype=np.float64)

        for gt in graph_types:
            G = bundle.graphs.get(gt)
            if G is None or G.number_of_nodes() < 2:
                continue

            G_undir = G.to_undirected()
            nodes   = list(G.nodes())
            n_nodes = len(nodes)

            # Degree centrality (fast, always available)
            deg_cent = nx.degree_centrality(G_undir)

            # Betweenness centrality (subsample for speed)
            try:
                if n_nodes > 50:
                    bet_cent = nx.betweenness_centrality(
                        G_undir, k=min(50, n_nodes), normalized=True
                    )
                else:
                    bet_cent = nx.betweenness_centrality(G_undir, normalized=True)
            except Exception:
                bet_cent = {n: 0.0 for n in nodes}

            # PageRank
            try:
                pr = nx.pagerank(G, alpha=0.85, max_iter=50)
            except Exception:
                pr = {n: 1.0 / n_nodes for n in nodes}

            # Map node centrality -> feature dimension bins
            # Each node contributes its centrality to a feature bin
            # based on its position in the graph (stmt_idx % n_features)
            for node in nodes:
                node_data = G.nodes[node]
                stmt_idx  = node_data.get("stmt_idx", 0)
                if stmt_idx < 0:
                    stmt_idx = 0
                bin_idx = int(stmt_idx) % n_features

                c_deg = deg_cent.get(node, 0.0)
                c_bet = bet_cent.get(node, 0.0)
                c_clo = nx.closeness_centrality(G_undir).get(node, 0.0)
                # New weighting: 0.6 betweenness, 0.2 degree, 0.2 closeness
                combined = 0.2 * c_deg + 0.6 * c_bet + 0.2 * c_clo
                sample_scores[bin_idx] += combined

        # Normalise within sample
        s_max = sample_scores.max()
        if s_max > 0:
            sample_scores /= s_max
        centrality_acc += sample_scores
        count += 1

    if count == 0:
        return np.ones(n_features, dtype=np.float32) / n_features

    result = (centrality_acc / count).astype(np.float32)
    r_max  = result.max()
    if r_max > 0:
        result /= r_max
    logger.info(f"  Centrality: computed over {count} samples")
    return result


# ---------------------------------------------------------------------------
# Composite ranking
# ---------------------------------------------------------------------------

def compute_composite_scores(
    mi_scores:          np.ndarray,
    shap_scores:        np.ndarray,
    centrality_scores:  np.ndarray,
    alpha: float = ALPHA,
    beta:  float = BETA,
    gamma: float = GAMMA,
) -> np.ndarray:
    """
    S_i = alpha * MI_norm(i) + beta * SHAP_norm(i) + gamma * Centrality_norm(i)
    All inputs already normalised to [0,1].
    Returns composite scores, shape (n_features,).
    """
    assert abs(alpha + beta + gamma - 1.0) < 1e-4, \
        f"Weights must sum to 1.0, got {alpha+beta+gamma}"

    return (alpha * mi_scores +
            beta  * shap_scores +
            gamma * centrality_scores).astype(np.float32)


# ---------------------------------------------------------------------------
# Quantum angle encoding
# ---------------------------------------------------------------------------

def angle_encode(features: np.ndarray) -> np.ndarray:
    """
    Rescale features to [-pi, pi] using tanh activation.
    features: (N, k) array
    Returns: (N, k) array in [-pi, pi]
    """
    return (np.tanh(features) * PI).astype(np.float32)


# ---------------------------------------------------------------------------
# Full QAFA pipeline
# ---------------------------------------------------------------------------

def run_qafa(
    X: np.ndarray,
    y: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    graphs_dir: Path,
    split_name: str,
    cached_indices: Optional[np.ndarray] = None,
    cached_scores:  Optional[dict]       = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Run QAFA on a single split.

    Parameters
    ----------
    X, y          : This split's features + labels
    X_train, y_train : Training features (used to compute importance on train only)
    graphs_dir    : Path to graph pkl files
    split_name    : 'train' | 'val' | 'test'
    cached_indices: If provided, reuse pre-computed selected indices
    cached_scores : If provided, reuse pre-computed importance scores

    Returns
    -------
    stage1_features : (N, 8) angle-encoded top-8 features
    stage2_features : (N, 8) angle-encoded next-8 features
    selected_idx    : (16,)  selected feature indices
    scores_dict     : dict with all importance scores
    composite_scores: (32,)  composite importance per feature
    """
    n_features = X.shape[1]   # 64 for BO, 32 for FS/UAF

    # Determine dataset key from graphs_dir (parent folder name)
    ds_key = str(graphs_dir).split("/")[-1].lower()
    if ds_key not in ["bo", "fs", "uaf"]:
        # fallback for Windows paths
        ds_key = str(graphs_dir).split("\\")[-1].lower()

    # ── FS: use EnhancedQAFA (4-component: MI+SHAP+Centrality+MMD) ──────────
    if ds_key == "fs" and cached_indices is None:
        logger.info(f"  [{split_name}] FS: Using EnhancedQAFA (MMD + histogram filter)...")
        try:
            from stage5_enhanced_qafa import EnhancedQAFA
            eqafa = EnhancedQAFA(n_select=64, overlap_threshold=0.90,
                                 alpha=0.30, beta=0.25, gamma=0.20, delta=0.25)
            selected_idx = eqafa.fit_select(X_train, y_train, graphs_dir=graphs_dir)
            scores_dict = eqafa.get_scores_dict()
            composite = np.array(scores_dict["composite"])
            logger.info(f"  [EnhancedQAFA] Selected {len(selected_idx)} features")
        except Exception as _eq_err:
            logger.warning(f"  EnhancedQAFA failed ({_eq_err}), falling back to standard QAFA")
            ds_key = "fs_fallback"   # force fallback path below

    # Always compute feature importance (composite ranking)
    if ds_key == "fs" and cached_indices is not None:
        logger.info(f"  [{split_name}] Reusing cached feature selection")
        selected_idx    = cached_indices
        composite       = np.array(cached_scores["composite"])
        scores_dict     = cached_scores
    elif ds_key not in ("fs",) or cached_indices is not None:
        if cached_indices is not None:
            logger.info(f"  [{split_name}] Reusing cached feature selection")
            selected_idx    = cached_indices
            composite       = np.array(cached_scores["composite"])
            scores_dict     = cached_scores
        else:
            logger.info(f"  [{split_name}] Computing importance scores on train set...")

            # Criterion 1: Mutual Information
            logger.info("  Computing Mutual Information...")
            mi_scores = compute_mutual_information(X_train, y_train)

            # Criterion 2: SHAP
            logger.info("  Computing SHAP values...")
            shap_scores = compute_shap_importance(X_train, y_train)

            # Criterion 3: Graph Centrality
            logger.info("  Computing Graph Centrality...")
            centrality_scores = compute_centrality_scores(graphs_dir, n_features)

            # Composite score — BO gets higher centrality weight; FS and UAF use
            # standard balanced weights (MI + SHAP dominate for FS patterns).
            if ds_key == "bo":
                composite = compute_composite_scores(
                    mi_scores, shap_scores, centrality_scores,
                    alpha=0.30, beta=0.20, gamma=0.50
                )
                alpha_used, beta_used, gamma_used = 0.30, 0.20, 0.50
            else:
                composite = compute_composite_scores(
                    mi_scores, shap_scores, centrality_scores
                )
                alpha_used, beta_used, gamma_used = ALPHA, BETA, GAMMA

            if ds_key == "bo":
                n_select = 64   # 8 rounds × 8 features
            else:
                n_select = 16   # 2 rounds × 8 features
            selected_idx = np.argsort(composite)[::-1][:n_select].astype(np.int32)

            scores_dict = {
                "mi":          mi_scores.tolist(),
                "shap":        shap_scores.tolist(),
                "centrality":  centrality_scores.tolist(),
                "composite":   composite.tolist(),
                "selected_idx": selected_idx.tolist(),
                "alpha": alpha_used, "beta": beta_used, "gamma": gamma_used,
            }

        if ds_key == "bo":
            logger.info(
                f"  Top-64 feature indices (centrality-weighted α={alpha_used} β={beta_used} γ={gamma_used}): "
                f"{selected_idx.tolist()}"
            )
        elif ds_key == "fs":
            logger.info(
                f"  Top-32 feature indices (by composite): {selected_idx.tolist()}"
            )
        else:
            logger.info(
                f"  Top-16 feature indices: {selected_idx[:8].tolist()} | "
                f"{selected_idx[8:].tolist()}"
            )
        logger.info(
            f"  Composite score range: "
            f"[{composite.min():.4f}, {composite.max():.4f}]"
        )

    X_selected = X[:, selected_idx]

    if ds_key == "bo":
        # 8 rounds of 8 features (N,64) -> 8x(N,8)
        stage_encoded = []
        for i in range(8):
            X_stage = X_selected[:, i*8:(i+1)*8]
            stage_encoded.append(angle_encode(X_stage))
        scores_dict["stage_encoded_shapes"] = [s.shape for s in stage_encoded]
        logger.info(f"  [{split_name}] Using TOP 64 features for BO (8x8)")
        return stage_encoded[0], stage_encoded[1], selected_idx, scores_dict, np.array(scores_dict.get("composite", composite)), stage_encoded
    elif ds_key == "fs":
        # 8 rounds of 8 features (N,64) -> 8x(N,8)
        stage_encoded = []
        for i in range(8):
            X_stage = X_selected[:, i*8:(i+1)*8]
            stage_encoded.append(angle_encode(X_stage))
        scores_dict["stage_encoded_shapes"] = [s.shape for s in stage_encoded]
        logger.info(f"  [{split_name}] Using TOP 64 features for FS (8x8)")
        return stage_encoded[0], stage_encoded[1], selected_idx, scores_dict, np.array(scores_dict.get("composite", composite)), stage_encoded
    else:
        # UAF: 2 rounds of 8 features (N,16) -> 2x(N,8)
        X_stage1 = X_selected[:, :N_QUBITS]
        X_stage2 = X_selected[:, N_QUBITS:]
        stage1_encoded = angle_encode(X_stage1)
        stage2_encoded = angle_encode(X_stage2)
        logger.info(
            f"  [{split_name}] Stage1 range: "
            f"[{stage1_encoded.min():.3f}, {stage1_encoded.max():.3f}]  "
            f"Stage2 range: [{stage2_encoded.min():.3f}, {stage2_encoded.max():.3f}]"
        )
        return stage1_encoded, stage2_encoded, selected_idx, scores_dict, np.array(scores_dict.get("composite", composite))


# ---------------------------------------------------------------------------
# Config + loader
# ---------------------------------------------------------------------------

def load_config(path=None):
    if path is None:
        path = _ROOT / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_compressed(comp_dir: Path, split: str):
    X = np.load(comp_dir / f"{split}.npy")
    y = np.load(comp_dir / f"{split}_labels.npy")
    return X, y


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def run_dataset(ds_key: str, config: dict) -> dict:
    logger.info("=" * 60)
    logger.info(f"Stage 5 QAFA -- {ds_key.upper()}")
    logger.info("=" * 60)

    np.random.seed(config["project"]["seed"])

    comp_dir   = _ROOT / "data" / "compressed" / ds_key
    graphs_dir = _ROOT / config["data"]["graphs_dir"] / ds_key
    qafa_dir   = _ROOT / "data" / "qafa" / ds_key
    qafa_dir.mkdir(parents=True, exist_ok=True)

    # Load compressed 32-dim features
    logger.info("Loading Stage 4 compressed features...")
    for split in ["train", "val", "test"]:
        if not (comp_dir / f"{split}.npy").exists():
            raise FileNotFoundError(
                f"Missing: {comp_dir}/{split}.npy -- run Stage 4 first"
            )

    X_train, y_train = load_compressed(comp_dir, "train")
    X_val,   y_val   = load_compressed(comp_dir, "val")
    X_test,  y_test  = load_compressed(comp_dir, "test")
    logger.info(
        f"  Train={X_train.shape}  Val={X_val.shape}  Test={X_test.shape}"
    )


    # Run QAFA on train first (computes importance scores)
    logger.info("\n--- Computing feature importance on TRAIN split ---")
    qafa_train = run_qafa(
        X=X_train, y=y_train,
        X_train=X_train, y_train=y_train,
        graphs_dir=graphs_dir,
        split_name="train",
    )

    # Apply same feature selection to val and test (no data leakage)
    logger.info("\n--- Applying feature selection to VAL split ---")
    qafa_val = run_qafa(
        X=X_val, y=y_val,
        X_train=X_train, y_train=y_train,
        graphs_dir=graphs_dir,
        split_name="val",
        cached_indices=qafa_train[2],
        cached_scores=qafa_train[3],
    )

    logger.info("\n--- Applying feature selection to TEST split ---")
    qafa_test = run_qafa(
        X=X_test, y=y_test,
        X_train=X_train, y_train=y_train,
        graphs_dir=graphs_dir,
        split_name="test",
        cached_indices=qafa_train[2],
        cached_scores=qafa_train[3],
    )

    # Save all outputs
    logger.info("\n  Saving QAFA outputs...")
    if ds_key in ["bo", "fs"]:
        # Save 8 rounds for BO/FS
        for split_name, qafa_out, y_arr in [
            ("train", qafa_train, y_train),
            ("val",   qafa_val,   y_val),
            ("test",  qafa_test,  y_test),
        ]:
            # qafa_out: (stage1, stage2, selected_idx, scores_dict, composite, stage_encoded)
            stage_encoded = qafa_out[5]
            for i, s in enumerate(stage_encoded):
                np.save(qafa_dir / f"{split_name}_stage{i+1}.npy", s)
            np.save(qafa_dir / f"{split_name}_labels.npy", y_arr)
            logger.info(
                f"  {split_name}: " + ", ".join([f"stage{i+1}={s.shape}" for i,s in enumerate(stage_encoded)])
            )
    else:
        # UAF: 2 rounds
        for split_name, qafa_out, y_arr in [
            ("train", qafa_train, y_train),
            ("val",   qafa_val,   y_val),
            ("test",  qafa_test,  y_test),
        ]:
            s1, s2 = qafa_out[0], qafa_out[1]
            np.save(qafa_dir / f"{split_name}_stage1.npy", s1)
            np.save(qafa_dir / f"{split_name}_stage2.npy", s2)
            np.save(qafa_dir / f"{split_name}_labels.npy", y_arr)
            logger.info(
                f"  {split_name}: stage1={s1.shape}  stage2={s2.shape}"
            )

    # Save feature selection metadata
    if ds_key in ["bo", "fs"]:
        selected_idx = qafa_train[2]
        scores_dict = qafa_train[3]
        np.save(qafa_dir / "selected_indices.npy", selected_idx)
        scores_path = qafa_dir / "feature_scores.json"
        with open(scores_path, "w") as f:
            json.dump(scores_dict, f, indent=2)
        logger.info(f"  Feature scores saved: {scores_path}")

        # Print importance table
        print(f"\n{'-'*60}")
        print(f"  QAFA Feature Importance -- {ds_key.upper()}")
        print(f"{'-'*60}")
        print(f"  {'Feat':>5}  {'MI':>8}  {'SHAP':>8}  {'Cent':>8}  {'Score':>8}  {'Sel':>4}")
        print(f"  {'-'*52}")
        sel_set = set(selected_idx.tolist())
        mi_list   = scores_dict["mi"]
        sh_list   = scores_dict["shap"]
        ce_list   = scores_dict["centrality"]
        co_list   = scores_dict["composite"]
        # Print top 20
        top_order = np.argsort(co_list)[::-1][:20]
        for i in top_order:
            sel_marker = "[X]" if i in sel_set else "   "
            print(f"  {i:>5}  {mi_list[i]:>8.4f}  {sh_list[i]:>8.4f}  "
                  f"{ce_list[i]:>8.4f}  {co_list[i]:>8.4f}  {sel_marker}")
        print(f"{'-'*60}")
        n_stages = 8 if ds_key == "fs" else 8  # both bo and fs: 8 rounds
        n_sel = len(selected_idx)
        print(f"  Selected indices (top {n_sel}): {selected_idx.tolist()}")
        for i in range(n_stages):
            print(f"  Stage {i+1} encodes: {selected_idx[i*8:(i+1)*8].tolist()}")
        print(f"{'-'*60}\n")

        # Save summary metrics
        summary = {
            "dataset": ds_key,
            "input_dim": INPUT_DIM,
            "n_selected": n_sel,
            "n_qubits": N_QUBITS,
            "n_stages": n_stages,
            "weights": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA},
            "selected_indices": selected_idx.tolist(),
            "stage_indices": [selected_idx[i*8:(i+1)*8].tolist() for i in range(n_stages)],
            "top5_composite_scores": sorted(
                zip(range(INPUT_DIM), co_list),
                key=lambda x: x[1], reverse=True
            )[:5],
        }
        summary_path = _ROOT / "results" / "metrics" / f"{ds_key}_stage5_qafa.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Stage 5 QAFA [OK] '{ds_key}'\n")
        return summary
    else:
        selected_idx = qafa_train[2]
        scores_dict = qafa_train[3]
        np.save(qafa_dir / "selected_indices.npy", selected_idx)
        scores_path = qafa_dir / "feature_scores.json"
        with open(scores_path, "w") as f:
            json.dump(scores_dict, f, indent=2)
        logger.info(f"  Feature scores saved: {scores_path}")

        # Print importance table
        print(f"\n{'-'*60}")
        print(f"  QAFA Feature Importance -- {ds_key.upper()}")
        print(f"{'-'*60}")
        print(f"  {'Feat':>5}  {'MI':>8}  {'SHAP':>8}  {'Cent':>8}  {'Score':>8}  {'Sel':>4}")
        print(f"  {'-'*52}")
        sel_set = set(selected_idx.tolist())
        mi_list   = scores_dict["mi"]
        sh_list   = scores_dict["shap"]
        ce_list   = scores_dict["centrality"]
        co_list   = scores_dict["composite"]
        # Print top 20
        top_order = np.argsort(co_list)[::-1][:20]
        for i in top_order:
            sel_marker = "[X]" if i in sel_set else "   "
            print(f"  {i:>5}  {mi_list[i]:>8.4f}  {sh_list[i]:>8.4f}  "
                  f"{ce_list[i]:>8.4f}  {co_list[i]:>8.4f}  {sel_marker}")
        print(f"{'-'*60}")
        print(f"  Selected indices (top 16): {selected_idx.tolist()}")
        print(f"  Stage 1 encodes: {selected_idx[:8].tolist()}")
        print(f"  Stage 2 encodes: {selected_idx[8:].tolist()}")
        print(f"{'-'*60}\n")

        # Save summary metrics
        summary = {
            "dataset": ds_key,
            "input_dim": INPUT_DIM,
            "n_selected": N_SELECTED,
            "n_qubits": N_QUBITS,
            "n_stages": N_STAGES,
            "weights": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA},
            "selected_indices": selected_idx.tolist(),
            "stage1_indices": selected_idx[:N_QUBITS].tolist(),
            "stage2_indices": selected_idx[N_QUBITS:].tolist(),
            "top5_composite_scores": sorted(
                zip(range(INPUT_DIM), co_list),
                key=lambda x: x[1], reverse=True
            )[:5],
        }
        summary_path = _ROOT / "results" / "metrics" / f"{ds_key}_stage5_qafa.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Stage 5 QAFA [OK] '{ds_key}'\n")
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="QEGVD Stage 5 - Quantum-Aware Feature Alignment"
    )
    parser.add_argument("--dataset", choices=["bo","fs","uaf","all"], required=True)
    parser.add_argument("--config",  type=str, default=None)
    args = parser.parse_args()

    config   = load_config(args.config)
    datasets = ["bo","fs","uaf"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        run_dataset(ds, config)

    print("=" * 55)
    print("  STAGE 5 QAFA COMPLETE")
    print("  Angle-encoded features -> data/qafa/<ds>/")
    print("  Stage1 (top-8) + Stage2 (next-8) ready for VQC")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
