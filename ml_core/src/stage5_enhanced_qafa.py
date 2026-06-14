"""
stage5_enhanced_qafa.py — Enhanced QAFA for FS classifier
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Improves the standard QAFA pipeline for FS by:

  Stage A — Histogram overlap elimination:
             Drops features whose class-conditional histogram overlap
             exceeds 90% (i.e. near-useless features).

  Stage B — 4-component composite scoring:
             S_i = α·MI_norm + β·SHAP_norm + γ·Centrality_norm + δ·MMD_norm
             where MMD (Maximum Mean Discrepancy) is the 4th criterion.

  Meta features — Appends 15 FS-specific handcrafted signals to extend
                  the 64-dim QAFA output to 79-dim.

Only activated when ds_type == "fs". BO and UAF use the existing QAFA.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

# ── 15 meta-features (appended to 64-dim to produce 79-dim for FS) ────────────
FS_META_FEATURES = [
    "code_len_norm",          # 0: code length / 2000
    "n_unique_vars_norm",     # 1: unique VAR_ / 30
    "n_unique_funcs_norm",    # 2: unique func_ / 15
    "n_str_tokens_norm",      # 3: STR_ count / 10
    "n_null_norm",            # 4: NULL / 5
    "n_loops_norm",           # 5: loops / 5
    "n_ifs_norm",             # 6: ifs / 15
    "n_case_norm",            # 7: case / 5 (more in SAFE)
    "has_switch",             # 8: binary
    "has_default",            # 9: binary
    "fmt_var_direct",         # 10: printf(VAR_) direct
    "fmt_safe_str",           # 11: printf with STR_ (safe)
    "has_taint",              # 12: taint source present
    "max_var_num_norm",       # 13: max VAR number / 80
    "mean_var_num_norm",      # 14: mean VAR number / 20
]
FS_META_DIM = len(FS_META_FEATURES)   # = 15


# ── MMD per feature ────────────────────────────────────────────────────────────

def mmd_per_feature(
    X: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Compute per-feature Maximum Mean Discrepancy (MMD) between the two classes.

    Uses an RBF kernel approximation:
        MMD²(f) = E[k(x,x')] + E[k(z,z')] - 2·E[k(x,z)]
    where x ~ class-1 and z ~ class-0 for feature f.

    Returns normalised MMD² scores in [0, 1], shape (n_features,).
    High MMD = feature is discriminative (good separation between classes).
    """
    X0 = X[y == 0]
    X1 = X[y == 1]

    n_features = X.shape[1]
    mmd_scores = np.zeros(n_features, dtype=np.float32)

    for j in range(n_features):
        x0 = X0[:, j]
        x1 = X1[:, j]

        # Sub-sample for speed if very large
        n_max = 500
        if len(x0) > n_max:
            rng = np.random.default_rng(42)
            x0 = rng.choice(x0, n_max, replace=False)
        if len(x1) > n_max:
            rng = np.random.default_rng(42)
            x1 = rng.choice(x1, n_max, replace=False)

        # Median heuristic for bandwidth if gamma not given
        if gamma is None:
            all_x = np.concatenate([x0, x1])
            dists = np.abs(all_x[:, None] - all_x[None, :])
            gam   = 1.0 / (2 * np.median(dists[dists > 0])**2 + 1e-9)
        else:
            gam = gamma

        def rbf_mean(a, b):
            d = (a[:, None] - b[None, :]) ** 2
            return np.mean(np.exp(-gam * d))

        mmd2 = rbf_mean(x1, x1) + rbf_mean(x0, x0) - 2 * rbf_mean(x1, x0)
        mmd_scores[j] = max(mmd2, 0.0)

    # Normalise to [0, 1]
    m = mmd_scores.max()
    if m > 0:
        mmd_scores /= m
    return mmd_scores


# ── Histogram overlap ─────────────────────────────────────────────────────────

def _histogram_overlap(x0: np.ndarray, x1: np.ndarray, bins: int = 30) -> float:
    """
    Compute the overlap (intersection area) of the two class-conditional
    histograms for a single feature.  Returns a value in [0, 1].
    1.0 = perfect overlap (uninformative), 0.0 = completely separate.
    """
    lo = min(x0.min(), x1.min())
    hi = max(x0.max(), x1.max())
    if hi - lo < 1e-9:
        return 1.0   # constant feature — perfectly overlapping

    edges  = np.linspace(lo, hi, bins + 1)
    h0, _  = np.histogram(x0, bins=edges, density=True)
    h1, _  = np.histogram(x1, bins=edges, density=True)
    # Normalise
    width  = edges[1] - edges[0]
    p0     = h0 * width + 1e-12
    p1     = h1 * width + 1e-12
    p0    /= p0.sum()
    p1    /= p1.sum()
    return float(np.sum(np.minimum(p0, p1)))


# ── Graph centrality (betweenness) ─────────────────────────────────────────────

def graph_centrality_scores(
    graphs_dir,
    n_features: int = 64,
    n_samples: int = 200,
) -> np.ndarray:
    """
    Compute betweenness-centrality based feature importance scores.
    Aggregates betweenness centrality over sampled FS graph bundles,
    bins node centrality values into feature-space bins.

    Returns normalised scores in [0, 1], shape (n_features,).
    Falls back to uniform scores if graphs aren't available.
    """
    import pickle
    from pathlib import Path
    import networkx as nx

    pkl_path = Path(graphs_dir) / "train.pkl"
    if not pkl_path.exists():
        return np.ones(n_features, dtype=np.float32) / n_features

    try:
        import sys
        import importlib
        sys.path.insert(0, str(Path(graphs_dir).parent.parent / "src"))
        s2 = importlib.import_module("stage2_graph_construction")
        sys.modules["__main__"].GraphBundle = s2.GraphBundle
        with open(pkl_path, "rb") as f:
            bundles = pickle.load(f)
    except Exception:
        return np.ones(n_features, dtype=np.float32) / n_features

    bundles  = bundles[:n_samples]
    acc      = np.zeros(n_features, dtype=np.float64)
    count    = 0

    for bundle in bundles:
        if not bundle.is_valid():
            continue
        sample_scores = np.zeros(n_features, dtype=np.float64)

        for gt in ["AST", "CFG", "DFG", "PDG", "TPG", "FSG"]:
            G = bundle.graphs.get(gt)
            if G is None or G.number_of_nodes() < 2:
                continue
            Gu = G.to_undirected()
            nodes = list(G.nodes())
            n_nodes = len(nodes)
            try:
                k = min(50, n_nodes)
                bet = nx.betweenness_centrality(Gu, k=k, normalized=True)
            except Exception:
                bet = {n: 0.0 for n in nodes}
            for node in nodes:
                stmt_idx = G.nodes[node].get("stmt_idx", 0)
                if stmt_idx < 0:
                    stmt_idx = 0
                bin_i = int(stmt_idx) % n_features
                sample_scores[bin_i] += bet.get(node, 0.0)

        smax = sample_scores.max()
        if smax > 0:
            sample_scores /= smax
        acc += sample_scores
        count += 1

    if count == 0:
        return np.ones(n_features, dtype=np.float32) / n_features

    result = (acc / count).astype(np.float32)
    rmax   = result.max()
    if rmax > 0:
        result /= rmax
    return result


# ── EnhancedQAFA ──────────────────────────────────────────────────────────────

class EnhancedQAFA:
    """
    Enhanced QAFA for FS dataset.

    Stage A — Histogram overlap elimination:
      Any feature whose histogram overlap exceeds `overlap_threshold`
      (default 0.90) is marked ineligible for selection.

    Stage B — 4-component composite scoring:
      S_i = α·MI + β·SHAP + γ·Centrality + δ·MMD
      Weights α=0.30, β=0.25, γ=0.20, δ=0.25 (sum to 1.0).

    Parameters
    ----------
    n_select         : number of features to select (default 64)
    overlap_threshold: max allowed histogram overlap (default 0.90)
    alpha / beta / gamma / delta: composite weights
    """

    def __init__(
        self,
        n_select:          int   = 64,
        overlap_threshold: float = 0.90,
        alpha:  float = 0.30,
        beta:   float = 0.25,
        gamma:  float = 0.20,
        delta:  float = 0.25,
    ):
        self.n_select          = n_select
        self.overlap_threshold = overlap_threshold
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta

        assert abs(alpha + beta + gamma + delta - 1.0) < 1e-4, \
            "Weights must sum to 1.0"

    def fit_select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        graphs_dir=None,
    ) -> np.ndarray:
        """
        Fit importance scores on training data and return selected indices.

        Returns
        -------
        selected_idx : (n_select,) array of selected feature column indices
        """
        n_features = X_train.shape[1]
        X0 = X_train[y_train == 0]
        X1 = X_train[y_train == 1]

        # ── Stage A: histogram overlap filter ─────────────────────────────
        eligible = np.ones(n_features, dtype=bool)
        for j in range(n_features):
            overlap = _histogram_overlap(X0[:, j], X1[:, j])
            if overlap > self.overlap_threshold:
                eligible[j] = False

        n_eligible = eligible.sum()
        print(f"  [EQAFA] Stage A: {n_eligible}/{n_features} features pass overlap < {self.overlap_threshold}")

        # ── Stage B: composite scoring ────────────────────────────────────
        # MI
        from sklearn.feature_selection import mutual_info_classif
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi = mutual_info_classif(X_train, y_train.astype(int),
                                     n_neighbors=5, random_state=42)
        mi = _norm01(mi)

        # SHAP
        shap = self._shap_scores(X_train, y_train)

        # Centrality
        if graphs_dir is not None:
            cent = graph_centrality_scores(graphs_dir, n_features=n_features)
        else:
            cent = np.ones(n_features, dtype=np.float32) / n_features
        cent = _norm01(cent)

        # MMD
        mmd = mmd_per_feature(X_train, y_train)

        # Composite
        composite = (self.alpha * mi +
                     self.beta  * shap +
                     self.gamma * cent +
                     self.delta * mmd).astype(np.float32)

        # Zero out ineligible features
        composite[~eligible] = 0.0

        self.composite_scores_ = composite
        self.mi_scores_         = mi
        self.shap_scores_       = shap
        self.centrality_scores_ = cent
        self.mmd_scores_        = mmd
        self.eligible_mask_     = eligible

        # Select top-n
        selected = np.argsort(composite)[::-1][:self.n_select].astype(np.int32)
        self.selected_idx_ = selected
        return selected

    def _shap_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        try:
            import shap
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
            clf.fit(X, y.astype(int))
            bg = X[:min(100, len(X))]
            ex = shap.TreeExplainer(clf)
            sv = ex.shap_values(bg)
            if isinstance(sv, list):
                sv = sv[1]
            importance = np.abs(sv).mean(axis=0)
        except Exception:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X, y.astype(int))
            importance = clf.feature_importances_
        return _norm01(importance.astype(np.float32))

    def get_scores_dict(self) -> dict:
        return {
            "mi":          self.mi_scores_.tolist(),
            "shap":        self.shap_scores_.tolist(),
            "centrality":  self.centrality_scores_.tolist(),
            "mmd":         self.mmd_scores_.tolist(),
            "composite":   self.composite_scores_.tolist(),
            "eligible":    self.eligible_mask_.tolist(),
            "selected_idx": self.selected_idx_.tolist(),
        }


# ── Meta feature appender ─────────────────────────────────────────────────────

def append_meta_features(
    X: np.ndarray,
    codes: list[str],
) -> np.ndarray:
    """
    Append 15 FS-specific meta features to an (N, 64) QAFA output.
    Returns (N, 79) array.

    Parameters
    ----------
    X     : (N, 64) QAFA output array
    codes : list of N raw C code strings
    """
    import re as _re
    meta_rows = []
    for code in codes:
        code_len      = len(code)
        n_unique_vars = len(set(_re.findall(r'VAR_\d+', code)))
        n_unique_funcs= len(set(_re.findall(r'func_\d+', code)))
        n_strs        = len(_re.findall(r'\bSTR_\d+\b', code))
        n_null        = code.count('NULL')
        n_loops       = len(_re.findall(r'\b(?:for|while|do)\b', code))
        n_ifs         = len(_re.findall(r'\bif\b', code))
        n_case        = len(_re.findall(r'\bcase\b', code))
        has_switch    = int('switch' in code)
        has_default   = int('default' in code)
        fmt_var_direct= int(bool(_re.search(r'\bprintf\s*\(\s*VAR_\w+\s*[,)]', code)))
        fmt_safe_str  = int(bool(_re.search(r'\b(?:printf|fprintf|sprintf)\s*\([^)]*STR_', code)))
        has_taint     = int(bool(_re.search(r'\b(?:fgets|gets|fgetws|getenv|recv|scanf)\b', code)))
        var_nums = [int(m) for m in _re.findall(r'VAR_(\d+)', code)]
        max_var_num   = max(var_nums) if var_nums else 0
        mean_var_num  = float(np.mean(var_nums)) if var_nums else 0.0

        row = [
            min(code_len / 2000.0, 1.0),
            min(n_unique_vars / 30.0, 1.0),
            min(n_unique_funcs / 15.0, 1.0),
            min(n_strs / 10.0, 1.0),
            min(n_null / 5.0, 1.0),
            min(n_loops / 5.0, 1.0),
            min(n_ifs / 15.0, 1.0),
            min(n_case / 5.0, 1.0),
            float(has_switch),
            float(has_default),
            float(fmt_var_direct),
            float(fmt_safe_str),
            float(has_taint),
            min(max_var_num / 80.0, 1.0),
            min(mean_var_num / 20.0, 1.0),
        ]
        meta_rows.append(row)

    meta = np.array(meta_rows, dtype=np.float32)   # (N, 15)
    return np.concatenate([X, meta], axis=1)         # (N, 79)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm01(x: np.ndarray) -> np.ndarray:
    xmax = x.max()
    if xmax > 0:
        return (x / xmax).astype(np.float32)
    return x.astype(np.float32)
