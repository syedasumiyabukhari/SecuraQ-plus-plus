"""
stage2_graph_improvements.py — FS-specific graph enhancements
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provides two new graph types for the FS (Format String) classifier:

  VLG — Variable Lineage Graph
        Tracks VAR_ assignment chains: who assigned to whom,
        whether the origin is a taint source, and how many
        hops separate an assignment from a format sink.

  APG — Argument Position Graph
        Encodes the *position* of each argument in format function
        calls (position 0 = format string slot = dangerous if VAR).

Also provides:
  NodeFeatureNormaliser  — group-wise normalisation of node feature vecs
  clip_tpg_edge_weights  — clips TPG edge weights to a safe maximum
  l2_normalise_embeddings — L2 normalises a numpy embedding matrix

These are imported by the integration patch in stage2_graph_construction.py.
They are ONLY activated when ds_type == "fs".
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

# ── VLG node feature dimensionality ───────────────────────────────────────────
# 8 dims per node:
#   [0] is_format_arg_pos0     — is this VAR in the format-string argument slot?
#   [1] is_format_arg_nonpos0  — is this VAR an *extra* argument (non-format)?
#   [2] assigned_from_taint    — was this VAR assigned from a known taint source?
#   [3] hop_distance           — normalised # steps from assignment to use (0→1)
#   [4] reachable_from_source  — transitive taint reachability flag
#   [5] assigned_from_str      — was this VAR assigned from a string literal (safe)?
#   [6] var_number_norm        — VAR_N index normalised (higher = more complex fn)
#   [7] is_function_param      — was this VAR declared in a function signature?
VLG_DIM = 8

# ── APG node dimensionality ────────────────────────────────────────────────────
# Call node (5 dims):
#   [0] one-hot: printf_family   [1] one-hot: fprintf_family
#   [2] one-hot: sprintf_family  [3] is_snprintf (bounded)
#   [4] call_count_norm
# Arg node (4 dims, padded to 5):
#   [0] arg_position_norm       — 0 = format-string slot (dangerous)
#   [1] is_var_arg              — 1 if argument is a VAR token
#   [2] is_str_arg              — 1 if argument is a STR literal
#   [3] is_tainted_arg          — 1 if arg was tainted (heuristic)
APG_DIM = 5

_TAINT_SOURCES = frozenset({
    "fgets", "fgetws", "gets", "scanf", "fscanf", "sscanf",
    "read", "recv", "recvfrom", "fread", "getenv", "getchar", "cin",
})

_FMT_SINKS_1 = frozenset({"printf", "wprintf", "vprintf"})
_FMT_SINKS_2 = frozenset({"fprintf", "sprintf", "vsprintf", "vfprintf",
                           "swprintf", "vswprintf", "syslog"})
_FMT_SINKS_3 = frozenset({"snprintf", "vsnprintf", "wsnprintf"})
_FMT_SINKS_ALL = _FMT_SINKS_1 | _FMT_SINKS_2 | _FMT_SINKS_3

_SNPRINTF = frozenset({"snprintf", "vsnprintf", "wsnprintf"})


# ── Variable Lineage Graph ─────────────────────────────────────────────────────

def build_vlg(code: str, sample_id: int = 0) -> nx.DiGraph:
    """
    Build a Variable Lineage Graph for a C code snippet.

    Nodes: one per unique VAR_N identifier encountered.
    Edges: VAR_src → VAR_dst when VAR_dst is assigned from VAR_src
           (i.e. appears on the RHS of an assignment whose LHS is VAR_dst).
    Node features: VLG_DIM (8) float32 vector.
    """
    G = nx.DiGraph(graph_type="VLG")

    # ── Step 1: Gather all VAR tokens and their contexts ────────────────────
    stmts = _split_stmts(code)
    n_stmts = max(len(stmts), 1)

    # Which variables were assigned from taint sources?
    taint_assigned: set[str] = set()
    # Which variables were assigned from string literals?
    str_assigned: set[str] = set()
    # All variable assignments: LHS → list of RHS vars
    assignments: dict[str, list[str]] = {}
    # Position of var in source
    var_pos: dict[str, int] = {}  # var → stmt_index of first assignment
    # Which vars appear as format arguments (position 0 = dangerous)?
    fmt_pos0: set[str] = set()   # format string slot
    fmt_posN: set[str] = set()   # extra argument slot (not format string)

    func_params: set[str] = set()

    # Detect function parameters (appear in function signature before '{')
    sig_m = re.match(r'[^{]+\(([^)]*)\)', code.strip())
    if sig_m:
        params_text = sig_m.group(1)
        for p in re.findall(r'\bVAR_\d+\b', params_text):
            func_params.add(p)

    for stmt_i, stmt in enumerate(stmts):
        # Assignments: VAR_X = expr
        lhs_m = re.match(r'\s*\*?\s*(VAR_\d+)\s*(?:\[[^\]]*\])?\s*=(?!=)', stmt)
        if lhs_m:
            lhs = lhs_m.group(1)
            rhs = stmt[lhs_m.end():]
            rhs_vars = re.findall(r'\bVAR_\d+\b', rhs)
            if lhs not in var_pos:
                var_pos[lhs] = stmt_i
            # Check taint assignment: VAR = taint_source(...)
            if any(src in rhs for src in _TAINT_SOURCES):
                taint_assigned.add(lhs)
            # Check string assignment: VAR = STR_... or "..."
            if re.search(r'STR_\d+|"[^"]*"', rhs):
                str_assigned.add(lhs)
            if rhs_vars:
                assignments[lhs] = rhs_vars

        # Format function calls — detect argument positions
        for fn in _FMT_SINKS_ALL:
            pat = re.compile(rf'\b{fn}\s*\((.+)\)', re.DOTALL)
            m = pat.search(stmt)
            if not m:
                continue
            args_raw = _split_args(m.group(1))
            # Determine which arg index is the format string
            if fn in _FMT_SINKS_1:
                fmt_idx = 0
            elif fn in _FMT_SINKS_2:
                fmt_idx = 1
            elif fn in _FMT_SINKS_3:
                fmt_idx = 2
            else:
                fmt_idx = 0
            for arg_i, arg in enumerate(args_raw):
                for var in re.findall(r'\bVAR_\d+\b', arg):
                    if arg_i == fmt_idx:
                        fmt_pos0.add(var)
                    else:
                        fmt_posN.add(var)

    # ── Step 2: Propagate taint transitively ────────────────────────────────
    changed = True
    while changed:
        changed = False
        for lhs, rhs_vars in assignments.items():
            if lhs not in taint_assigned:
                if any(v in taint_assigned for v in rhs_vars):
                    taint_assigned.add(lhs)
                    changed = True

    # ── Step 3: Collect all unique VARs ─────────────────────────────────────
    all_vars = set(re.findall(r'\bVAR_\d+\b', code))

    # Extract VAR numbers for normalisation
    all_nums = [int(m) for m in re.findall(r'VAR_(\d+)', code)]
    max_num  = max(all_nums) if all_nums else 1

    # ── Step 4: Build nodes ──────────────────────────────────────────────────
    for var in all_vars:
        v_num = int(re.search(r'VAR_(\d+)', var).group(1))
        stmt_i = var_pos.get(var, 0)

        feat = np.zeros(VLG_DIM, dtype=np.float32)
        feat[0] = float(var in fmt_pos0)
        feat[1] = float(var in fmt_posN and var not in fmt_pos0)
        feat[2] = float(var in taint_assigned)
        feat[3] = stmt_i / max(n_stmts - 1, 1)   # normalised position
        feat[4] = float(var in taint_assigned)     # transitive taint flag
        feat[5] = float(var in str_assigned)
        feat[6] = v_num / max(max_num, 1)
        feat[7] = float(var in func_params)

        G.add_node(var, ntype="VAR_NODE", label=var, feature=feat)

    # ── Step 5: Add assignment edges ─────────────────────────────────────────
    for lhs, rhs_vars in assignments.items():
        for rhs in rhs_vars:
            if rhs in G.nodes and lhs in G.nodes:
                w = 2.0 if (rhs in taint_assigned or lhs in fmt_pos0) else 1.0
                G.add_edge(rhs, lhs, etype="VAR_ASSIGN",
                           tainted=float(rhs in taint_assigned), weight=w)

    # ── Step 6: Add taint-to-fmt edges ──────────────────────────────────────
    for var in taint_assigned:
        if var in fmt_pos0 and var in G.nodes:
            G.add_edge(var, var, etype="TAINT_FMT",
                       weight=4.0)   # self-loop signals direct taint→format

    return G


# ── Argument Position Graph ────────────────────────────────────────────────────

def build_apg(code: str, sample_id: int = 0) -> nx.DiGraph:
    """
    Build an Argument Position Graph.

    For each format function call site:
      - Add a CALL node with 5-dim features encoding which function family.
      - Add one ARG node per argument with 5-dim features (padded from 4).
      - Connect CALL → ARG with an edge encoding the argument position.

    Position 0 (format string slot) with a VAR argument is the key vuln signal.
    """
    G = nx.DiGraph(graph_type="APG")

    stmts = _split_stmts(code)

    # Which variables are taint-assigned (inherit from VLG logic)
    taint_vars: set[str] = set()
    for stmt in stmts:
        if any(src in stmt for src in _TAINT_SOURCES):
            lhs_m = re.match(r'\s*\*?\s*(VAR_\d+)\s*=', stmt)
            if lhs_m:
                taint_vars.add(lhs_m.group(1))

    node_ctr = 0

    for stmt_i, stmt in enumerate(stmts):
        for fn in _FMT_SINKS_ALL:
            m = re.search(rf'\b({fn})\s*\((.+)\)', stmt, re.DOTALL)
            if not m:
                continue
            fn_name   = m.group(1)
            args_raw  = _split_args(m.group(2))

            # CALL node features (5-dim)
            call_feat = np.zeros(APG_DIM, dtype=np.float32)
            call_feat[0] = float(fn_name in _FMT_SINKS_1)
            call_feat[1] = float(fn_name in _FMT_SINKS_2)
            call_feat[2] = float(fn_name in _FMT_SINKS_3)
            call_feat[3] = float(fn_name in _SNPRINTF)
            call_feat[4] = min(len(args_raw) / 6.0, 1.0)

            call_node_id = f"CALL_{fn_name}_{stmt_i}_{node_ctr}"
            G.add_node(call_node_id, ntype="CALL_NODE",
                       label=fn_name, feature=call_feat)
            node_ctr += 1

            # Determine format string position
            if fn_name in _FMT_SINKS_1:
                fmt_idx = 0
            elif fn_name in _FMT_SINKS_2:
                fmt_idx = 1
            elif fn_name in _FMT_SINKS_3:
                fmt_idx = 2
            else:
                fmt_idx = 0

            # ARG nodes (4-dim padded to 5)
            for arg_i, arg in enumerate(args_raw):
                arg_vars = re.findall(r'\bVAR_\d+\b', arg)
                is_var   = int(bool(arg_vars))
                is_str   = int(bool(re.search(r'STR_\d+|"[^"]*"', arg)))
                is_taint = int(any(v in taint_vars for v in arg_vars))
                pos_norm = arg_i / max(len(args_raw) - 1, 1)

                arg_feat = np.zeros(APG_DIM, dtype=np.float32)
                # [0] normalised position (0 = format string slot)
                arg_feat[0] = 1.0 - pos_norm if arg_i == fmt_idx else pos_norm
                # [1] is VAR
                arg_feat[1] = float(is_var)
                # [2] is STR
                arg_feat[2] = float(is_str)
                # [3] is tainted
                arg_feat[3] = float(is_taint)
                # [4] danger signal: format position with VAR (not STR)
                arg_feat[4] = float(arg_i == fmt_idx and is_var and not is_str)

                arg_node_id = f"ARG_{fn_name}_{stmt_i}_{arg_i}_{node_ctr}"
                G.add_node(arg_node_id, ntype="ARG_NODE",
                           label=arg[:40], feature=arg_feat)
                node_ctr += 1

                # Edge from CALL to ARG
                danger_weight = 4.0 if (arg_i == fmt_idx and is_var and not is_str) else 1.0
                G.add_edge(call_node_id, arg_node_id,
                           etype="ARG_EDGE", pos=arg_i, weight=danger_weight)

    return G


# ── NodeFeatureNormaliser ──────────────────────────────────────────────────────

class NodeFeatureNormaliser:
    """
    Normalises node feature vectors from stage2 graphs.

    Group-wise normalisation matching the existing encode_node_features layout:
      [0:35]   one-hot node type   → no scaling
      [35:64]  structural signals  → RobustScaler
      [64:84]  FS-specific signals → MinMaxScaler
    """

    def __init__(self):
        from sklearn.preprocessing import RobustScaler, MinMaxScaler
        self._robust  = RobustScaler()
        self._minmax  = MinMaxScaler()
        self._fitted  = False

    def fit(self, feature_matrix: np.ndarray) -> "NodeFeatureNormaliser":
        """feature_matrix: (N, 84) array of node features from FS graphs."""
        n = feature_matrix.shape[1]
        if n < 84:
            return self   # not enough dims, skip
        self._robust.fit(feature_matrix[:, 35:64])
        self._minmax.fit(feature_matrix[:, 64:min(84, n)])
        self._fitted = True
        return self

    def transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return feature_matrix
        X = feature_matrix.copy()
        n = X.shape[1]
        if n < 64:
            return X
        X[:, 35:64] = self._robust.transform(X[:, 35:64])
        if n >= 84:
            X[:, 64:84] = self._minmax.transform(X[:, 64:84])
        return X


# ── TPG edge weight clipper ────────────────────────────────────────────────────

def clip_tpg_edge_weights(G: nx.DiGraph, max_weight: float = 2.0) -> nx.DiGraph:
    """
    Clip edge weights in a TPG to `max_weight`.
    The original pipeline used 5.0; 2.0 prevents the taint-path signals
    from dominating and destabilising the GAT message passing.
    Only modifies TPG graphs.
    """
    for u, v, data in G.edges(data=True):
        if data.get("weight", 1.0) > max_weight:
            data["weight"] = max_weight
    return G


# ── L2 embedding normaliser ────────────────────────────────────────────────────

def l2_normalise_embeddings(embeddings: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Row-wise L2 normalisation of a (N, D) embedding matrix.
    Prevents a few high-magnitude samples from dominating GAT aggregation.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_stmts(code: str) -> list[str]:
    """Very lightweight statement splitter: split on ';' and '{' / '}'."""
    stmts = re.split(r'[;{}]', code)
    return [s.strip() for s in stmts if s.strip() and len(s.strip()) > 2]


def _split_args(args_str: str) -> list[str]:
    """Split comma-separated function arguments, respecting nested parens."""
    args, depth, buf = [], 0, []
    for ch in args_str:
        if ch == '(':
            depth += 1; buf.append(ch)
        elif ch == ')':
            depth -= 1; buf.append(ch)
        elif ch == ',' and depth == 0:
            args.append("".join(buf).strip()); buf = []
        else:
            buf.append(ch)
    if buf:
        args.append("".join(buf).strip())
    return [a for a in args if a]
