"""
QEGVD - Stage 2: Static Code Analysis & Multi-View Graph Construction
======================================================================
Constructs 7 program analysis graphs per code sample using pure Python
static analysis (no external parser required - fully offline/portable):

  1. AST  - Abstract Syntax Tree (structural decomposition)
  2. CFG  - Control Flow Graph  (execution paths)
  3. DFG  - Data Flow Graph     (def-use chains)
  4. PDG  - Program Dependence Graph (CFG + DFG unified)
  5. TPG  - Taint Propagation Graph (source → sink flows)
  6. MAG  - Memory Access Graph (alloc/free/access relations)
  7. CG   - Call Graph          (caller → callee)

Each graph is stored as a NetworkX DiGraph with:
  - node attributes: type, label, line_idx, feature_vec (dim=64)
  - edge attributes: type, weight

Output: data/graphs/<dataset>/<split>.pkl  (list of GraphBundle objects)
        data/graphs/<dataset>/<split>_stats.json

Usage
-----
    python src/stage2_graph_construction.py --dataset bo
    python src/stage2_graph_construction.py --dataset all
    python src/stage2_graph_construction.py --dataset all --split train
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

(_ROOT / "logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)),
        logging.FileHandler(_ROOT / "logs" / "stage2.log", mode="a"),
    ],
)
logger = logging.getLogger("Stage2")

# ---------------------------------------------------------------------------
# Tree-sitter C parser (optional — falls back to regex if unavailable)
# ---------------------------------------------------------------------------
_TS_PARSER = None
try:
    from tree_sitter import Language as _TSLanguage, Parser as _TSParser
    import tree_sitter_c as _tsc
    _TS_PARSER = _TSParser(_TSLanguage(_tsc.language()))
    logger.info("tree-sitter C parser loaded — using AST-based classification")
except Exception as _ts_err:
    logger.warning(f"tree-sitter unavailable ({_ts_err}), falling back to regex classifier")

# tree-sitter node types that map directly to our types
_TS_DIRECT_MAP: dict[str, str] = {
    "function_definition": "FUNC_DEF",
    "if_statement":        "IF",
    "else_clause":         "ELSE",
    "for_statement":       "FOR",
    "while_statement":     "WHILE",
    "do_statement":        "WHILE",
    "switch_statement":    "SWITCH",
    "case_statement":      "CASE",
    "goto_statement":      "GOTO",
    "return_statement":    "RETURN",
    "labeled_statement":   "LABEL",
    "break_statement":     "EXPR",
    "continue_statement":  "EXPR",
}

# ---------------------------------------------------------------------------
# Vocabulary tables (used for node feature encoding)
# ---------------------------------------------------------------------------

NODE_TYPES = [
    "FUNC_DEF", "PARAM", "VAR_DECL", "ARRAY_DECL", "PTR_DECL",
    "ASSIGN", "CALL", "RETURN", "IF", "ELSE", "FOR", "WHILE",
    "SWITCH", "CASE", "GOTO", "LABEL", "BLOCK_START", "BLOCK_END",
    "EXPR", "LITERAL", "OPERATOR", "CAST",
    # Memory
    "ALLOC", "FREE", "DELETE", "MEM_READ", "MEM_WRITE",
    # Taint
    "TAINT_SOURCE", "TAINT_SINK", "TAINT_PROPAGATE",
    # CG
    "FUNC_CALL", "UNSAFE_API", "ENTRY", "EXIT",
    "FORMAT_STR", "FORMAT_ARG", "FORMAT_FUNC", "FORMAT_SPEC", "MISMATCH", "DATAFLOW",
    "UNKNOWN",
]
NODE_TYPE_IDX = {t: i for i, t in enumerate(NODE_TYPES)}

EDGE_TYPES = [
    "AST_CHILD", "CFG_NEXT", "CFG_TRUE", "CFG_FALSE", "CFG_BACK",
    "DFG_DEF_USE", "PDG_CTRL", "PDG_DATA",
    "TAINT_FLOW", "MEM_ALLOC_USE", "MEM_ALLOC_FREE", "MEM_FREE_USE",
    "CALL_EDGE", "RETURN_EDGE",
]
EDGE_TYPE_IDX = {t: i for i, t in enumerate(EDGE_TYPES)}

# C/C++ API classification
ALLOC_APIS    = {"malloc", "calloc", "realloc", "new", "strdup", "strndup"}
FREE_APIS     = {"free", "delete", "delete[]"}
MEM_OPS       = {"memcpy", "memmove", "memset", "strcpy", "strncpy",
                 "strcat", "strncat", "wmemset", "wmemcpy", "wcscpy",
                 "wcsncpy", "wcscat", "wcsncat", "sprintf", "snprintf",
                 "vsprintf", "vsnprintf", "gets", "fgets", "scanf", "sscanf"}
FORMAT_SINKS  = {"printf", "fprintf", "sprintf", "snprintf", "vsprintf",
                 "vsnprintf", "wprintf", "vprintf", "syslog"}
TAINT_SOURCES = {"argv", "getenv", "fgets", "fscanf", "scanf", "read",
                 "recv", "recvfrom", "fread", "getchar", "cin"}
UNSAFE_APIS   = {"gets", "strcpy", "strcat", "sprintf", "scanf",
                 "vsprintf", "wcscpy", "wcscat"} | FORMAT_SINKS


# ---------------------------------------------------------------------------
# Token / line extraction helpers
# ---------------------------------------------------------------------------

class TokenStream:
    """Lightweight tokenizer for masked C/C++ code."""

    # Token patterns in priority order
    _PATTERNS = [
        ("COMMENT",    r"//[^\n]*|/\*.*?\*/"),
        ("STRING",     r'"[^"]*"'),
        ("CHAR",       r"'[^']*'"),
        ("NUMBER",     r"\b\d+(?:\.\d+)?\b"),
        ("FUNC_CALL",  r"\b(\w+)\s*(?=\()"),
        ("IDENT",      r"\b[A-Za-z_]\w*\b"),
        ("OP_MULTI",   r"->|::|<<|>>|<=|>=|==|!=|\+=|-=|\*=|/=|&&|\|\||\+\+|--"),
        ("OP",         r"[+\-*/%&|^~<>=!,;:.?@]"),
        ("BRACKET",    r"[\[\](){}]"),
        ("WHITESPACE", r"\s+"),
    ]
    _RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _PATTERNS), re.DOTALL)

    @classmethod
    def tokenize(cls, code: str) -> list[dict]:
        tokens = []
        line = 0
        for m in cls._RE.finditer(code):
            kind = m.lastgroup
            val  = m.group()
            if kind == "WHITESPACE":
                line += val.count("\n")
                continue
            if kind == "COMMENT":
                continue
            tokens.append({"kind": kind, "val": val, "line": line})
        return tokens


def split_statements(code: str) -> list[str]:
    """
    Split code into pseudo-statements using brace/semicolon heuristics.
    Returns a list of non-empty statement strings.
    """
    # Normalise newlines
    code = re.sub(r'\r\n?', '\n', code)
    stmts = []
    depth = 0
    buf = []
    for char in code:
        buf.append(char)
        if char in "({":
            depth += 1
        elif char in ")}":
            depth -= 1
            if depth == 0:
                s = "".join(buf).strip()
                if s:
                    stmts.append(s)
                buf = []
        elif char == ";" and depth <= 1:
            s = "".join(buf).strip()
            if s:
                stmts.append(s)
            buf = []
    remainder = "".join(buf).strip()
    if remainder:
        stmts.append(remainder)
    return [s for s in stmts if len(s.strip()) > 1]


# ---------------------------------------------------------------------------
# Tree-sitter based statement splitter (proper per-statement granularity)
# ---------------------------------------------------------------------------

def _ts_decode(node) -> str:
    return node.text.decode("utf-8", errors="replace") if node and node.text else ""


def _ts_walk_compound(node, result: list, depth: int) -> None:
    """Walk a compound_statement and append (text, ts_type, depth) to result."""
    for child in node.children:
        if not child.is_named or child.type == "comment":
            continue
        _ts_walk_stmt(child, result, depth)


def _ts_walk_stmt(node, result: list, depth: int) -> None:
    """Walk a single statement node, recursing into control-flow bodies."""
    nt = node.type

    # --- if_statement ---
    if nt == "if_statement":
        cond = node.child_by_field_name("condition")
        cond_text = f"if {_ts_decode(cond)}" if cond else "if (...)"
        result.append((cond_text, "IF", depth))

        then_body = node.child_by_field_name("consequence")
        if then_body:
            if then_body.type == "compound_statement":
                _ts_walk_compound(then_body, result, depth + 1)
            else:
                _ts_walk_stmt(then_body, result, depth + 1)

        alt = node.child_by_field_name("alternative")
        if alt:
            result.append(("else", "ELSE", depth))
            for c in alt.children:
                if c.is_named and c.type not in ("comment",):
                    if c.type == "compound_statement":
                        _ts_walk_compound(c, result, depth + 1)
                    else:
                        _ts_walk_stmt(c, result, depth + 1)
        return

    # --- for / while / do ---
    if nt in ("for_statement", "while_statement", "do_statement"):
        ts_type = "FOR" if nt == "for_statement" else "WHILE"
        # Build header: everything up to (but not including) the body block
        header_parts = []
        body = None
        for c in node.children:
            if c.type == "compound_statement":
                body = c
                break
            if c.is_named and c.type != "comment":
                header_parts.append(_ts_decode(c))
            elif not c.is_named:
                header_parts.append(_ts_decode(c))
        header = "".join(header_parts).strip().rstrip("{").strip()
        if not header:
            header = _ts_decode(node)[:80]
        result.append((header, ts_type, depth))
        if body:
            _ts_walk_compound(body, result, depth + 1)
        return

    # --- switch ---
    if nt == "switch_statement":
        cond = node.child_by_field_name("condition")
        header = f"switch {_ts_decode(cond)}" if cond else "switch (...)"
        result.append((header, "SWITCH", depth))
        body = node.child_by_field_name("body")
        if body:
            _ts_walk_compound(body, result, depth + 1)
        return

    # --- case / default labels ---
    if nt == "case_statement":
        # The label is the first value, then the body statements follow
        label_text = None
        for c in node.children:
            if c.is_named and c.type not in ("compound_statement",):
                label_text = _ts_decode(c)
                break
        result.append((f"case {label_text}:" if label_text else "case:", "CASE", depth))
        for c in node.children:
            if c.is_named and c != node.children[0]:
                _ts_walk_stmt(c, result, depth + 1)
        return

    # --- function definition (skip — handled at top level) ---
    if nt == "function_definition":
        body = node.child_by_field_name("body")
        if body and body.type == "compound_statement":
            _ts_walk_compound(body, result, depth)
        return

    # --- plain statement ---
    stmt_text = _ts_decode(node).strip()
    if not stmt_text or len(stmt_text) <= 1:
        return
    ts_type = _ts_classify_node(node) or _classify_regex(stmt_text)
    result.append((stmt_text, ts_type, depth))


def _ts_split_statements(code: str) -> list[tuple[str, str, int]]:
    """
    Parse code with tree-sitter and return flat list of (text, ts_type, depth).
    Returns None on any failure so callers can fall back to regex.
    """
    if _TS_PARSER is None:
        return None
    try:
        tree = _TS_PARSER.parse(bytes(code, "utf-8", errors="replace"))
        root = tree.root_node
        result = []
        for child in root.children:
            if child.type == "function_definition":
                # Add the function header as stmt 0
                body = child.child_by_field_name("body")
                # Build header text = everything before the body
                header_parts = []
                for c in child.children:
                    if c == body:
                        break
                    header_parts.append(_ts_decode(c))
                header = "".join(header_parts).strip()
                result.append((header if header else _ts_decode(child)[:80],
                                "FUNC_DEF", 0))
                if body and body.type == "compound_statement":
                    _ts_walk_compound(body, result, 1)
                return result if result else None
            elif child.is_named and child.type != "comment":
                _ts_walk_stmt(child, result, 0)
        return result if result else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Statement classifier
# ---------------------------------------------------------------------------

def _classify_regex(stmt: str) -> str:
    """Original regex-based classifier — used as fallback."""
    s = stmt.strip()
    _QUALS = r"(?:static|inline|extern|const|volatile|register)\s+"
    _TYPES = r"(?:void|int|char|float|double|long|short|unsigned|wchar_t|bool|auto)"
    # Standard-type function definition
    if re.match(rf"(?:{_QUALS})*{_TYPES}\s+\*?\s*\w+\s*\(", s):
        return "FUNC_DEF"
    # Custom-return-type function definition: qualifier + word + word + (
    # e.g. "static VAR_21 func_2(" or "VAR_21 func_2("
    if re.match(rf"(?:{_QUALS})*\w+\s+\w+\s*\(", s) and not re.match(r"\s*(?:if|for|while|switch)\b", s):
        return "FUNC_DEF"
    if re.match(rf"(?:{_QUALS})*{_TYPES}\s+\*?\s*\w+\s*\[", s):
        return "ARRAY_DECL"
    if re.match(rf"(?:{_QUALS})*{_TYPES}\s+\*\s*\w+", s):
        return "PTR_DECL"
    if re.match(rf"(?:{_QUALS})*{_TYPES}\s+\w+", s):
        return "VAR_DECL"
    if re.match(r"\s*if\s*\(", s):
        return "IF"
    if re.match(r"\s*else\b", s):
        return "ELSE"
    if re.match(r"\s*for\s*\(", s):
        return "FOR"
    if re.match(r"\s*while\s*\(", s):
        return "WHILE"
    if re.match(r"\s*switch\s*\(", s):
        return "SWITCH"
    if re.match(r"\s*case\b", s):
        return "CASE"
    if re.match(r"\s*goto\s+\w+", s):
        return "GOTO"
    if re.match(r"\s*return\b", s):
        return "RETURN"
    if re.match(r"\s*\w+\s*:", s) and "?" not in s:
        return "LABEL"
    if re.search(r"\b(malloc|calloc|realloc|new|strdup)\b", s):
        return "ALLOC"
    if re.search(r"\b(free|delete)\b", s):
        return "FREE"
    if re.search(r"\b\w+\s*\(", s):
        call = re.search(r"\b(\w+)\s*\(", s)
        if call and call.group(1) in UNSAFE_APIS:
            return "UNSAFE_API"
        return "CALL"
    if "=" in s and "==" not in s:
        if re.search(r"\w+\s*\[", s):
            return "MEM_WRITE"
        return "ASSIGN"
    if re.search(r"\w+\s*\[", s):
        return "MEM_READ"
    return "EXPR"


def _ts_classify_expr_node(node) -> str:
    """Map a tree-sitter expression node to our statement type."""
    nt = node.type
    if nt == "call_expression":
        fn = node.child_by_field_name("function")
        raw = fn.text.decode("utf-8", errors="replace").strip() if fn and fn.text else ""
        name = re.sub(r"^[*&()\s]+", "", raw)
        if any(a in name for a in ALLOC_APIS):  return "ALLOC"
        if any(a in name for a in FREE_APIS):   return "FREE"
        if name in UNSAFE_APIS:                 return "UNSAFE_API"
        return "CALL"
    if nt in ("assignment_expression", "augmented_assignment_expression"):
        # RHS alloc/free takes priority — e.g. "ptr = malloc(n)" is ALLOC
        right = node.child_by_field_name("right")
        if right and right.type == "call_expression":
            fn = right.child_by_field_name("function")
            rname = fn.text.decode("utf-8", errors="replace").strip() if fn and fn.text else ""
            if any(a in rname for a in ALLOC_APIS): return "ALLOC"
            if any(a in rname for a in FREE_APIS):  return "FREE"
        left = node.child_by_field_name("left")
        if left and left.type == "subscript_expression":
            return "MEM_WRITE"
        return "ASSIGN"
    if nt == "subscript_expression":  return "MEM_READ"
    if nt == "cast_expression":       return "CAST"
    if nt == "update_expression":     return "EXPR"
    return "EXPR"


def _ts_classify_node(node) -> Optional[str]:
    """Map a tree-sitter statement node to our type. Returns None on unknown."""
    nt = node.type
    if nt in _TS_DIRECT_MAP:
        return _TS_DIRECT_MAP[nt]
    if nt == "declaration":
        # Check whether it declares an array, pointer, or plain variable.
        # Works for both standard types AND custom/typedef names (e.g. VAR_5 VAR_3).
        for child in node.children:
            ct = child.type
            if ct in ("array_declarator",):
                return "ARRAY_DECL"
            if ct in ("pointer_declarator",):
                return "PTR_DECL"
            if ct == "init_declarator":
                for sub in child.children:
                    if sub.type == "array_declarator":  return "ARRAY_DECL"
                    if sub.type == "pointer_declarator": return "PTR_DECL"
        return "VAR_DECL"
    if nt == "expression_statement":
        for child in node.children:
            if child.is_named:
                return _ts_classify_expr_node(child)
    return None


def _ts_classify(stmt: str) -> Optional[str]:
    """Parse stmt with tree-sitter and return our node type, or None on failure."""
    if _TS_PARSER is None:
        return None
    try:
        # Wrap in a dummy function so the parser sees a valid translation unit
        wrapped = f"void __qegvd_classify__(void){{\n{stmt}\n}}"
        tree = _TS_PARSER.parse(bytes(wrapped, "utf-8", errors="replace"))
        body = None
        for fn_node in tree.root_node.children:
            if fn_node.type == "function_definition":
                for c in fn_node.children:
                    if c.type == "compound_statement":
                        body = c
                        break
                break
        if body is None:
            return None
        for child in body.children:
            if not child.is_named or child.type == "comment":
                continue
            result = _ts_classify_node(child)
            return result   # only look at the first named statement
    except Exception:
        pass
    return None


@lru_cache(maxsize=8192)
def classify_statement(stmt: str) -> str:
    """
    Classify a C/C++ statement to one of the 35 node types.
    Uses tree-sitter AST for accuracy; falls back to regex.
    LRU-cached — each unique statement text is parsed at most once.
    """
    ts = _ts_classify(stmt)
    if ts is not None:
        return ts
    return _classify_regex(stmt)


# ---------------------------------------------------------------------------
# Feature vector encoder (72-dim)
# ---------------------------------------------------------------------------

FEATURE_DIM    = 72   # default (UAF)
FEATURE_DIM_BO = 84   # BO: 72 + 8 buffer-overflow signals + 4 numeric literal features
FEATURE_DIM_FS = 84   # FS: 72 + 12 format-string taint signals

# BO-specific function sets used in [72:80] features
_UNSAFE_COPY = frozenset({
    'strcpy', 'strcat', 'gets', 'sprintf', 'vsprintf',
    'wcscpy', 'wcscat', 'lstrcpy', 'lstrcpyA',
})
_SAFE_COPY = frozenset({
    'strncpy', 'strncat', 'snprintf', 'vsnprintf',
    'fgets', 'strlcpy', 'strlcat', 'strncpy_s',
})


def encode_node_features(
    node_type: str,
    stmt: str,
    stmt_idx: int,
    total_stmts: int,
    ds_type: str = "all",
    func_stats: dict = None,
) -> np.ndarray:
    """
    Encode a node as a float32 feature vector:
      [0:35]   one-hot node type
      [35:49]  structural signals
      [49:64]  lexical signals
      [64:72]  sink-specific features (format-string vulnerability signals)
      [72:80]  BO-specific buffer-overflow signals (ds_type=="bo" only)
    Returns 72-dim for FS/UAF, 80-dim for BO.
    """
    fdim = FEATURE_DIM_BO if ds_type == "bo" else (FEATURE_DIM_FS if ds_type == "fs" else FEATURE_DIM)
    vec = np.zeros(fdim, dtype=np.float32)

    # One-hot node type (35 types)
    t_idx = NODE_TYPE_IDX.get(node_type, NODE_TYPE_IDX["UNKNOWN"])
    if t_idx < 35:
        vec[t_idx] = 1.0

    # --- Structural signals (14 dims) [35:49] ---
    tokens   = re.findall(r'\b\w+\b', stmt)
    n_tokens = len(tokens)

    vec[35] = min(n_tokens / 30.0, 1.0)                          # token count (normalised)
    vec[36] = stmt.count("(") / max(n_tokens, 1)                  # call density
    vec[37] = stmt.count("[") / max(n_tokens, 1)                  # array access density
    vec[38] = stmt.count("*") / max(n_tokens, 1)                  # pointer density
    vec[39] = stmt.count("=") / max(n_tokens, 1)                  # assignment density
    vec[40] = stmt_idx / max(total_stmts - 1, 1)                  # position in function
    vec[41] = 1.0 if re.search(r'\b(if|for|while|switch)\b', stmt) else 0.0
    vec[42] = 1.0 if re.search(r'\b(return|goto|break|continue)\b', stmt) else 0.0
    vec[43] = 1.0 if re.search(r'\b(malloc|calloc|realloc|new)\b', stmt) else 0.0
    vec[44] = 1.0 if re.search(r'\b(free|delete)\b', stmt) else 0.0
    vec[45] = 1.0 if re.search(r'\b(memcpy|memset|strcpy|strcat|wmemset)\b', stmt) else 0.0
    vec[46] = 1.0 if any(api in stmt for api in FORMAT_SINKS) else 0.0
    vec[47] = 1.0 if re.search(r'\b(NULL|nullptr)\b', stmt) else 0.0
    vec[48] = min(stmt.count("{") + stmt.count("}"), 5) / 5.0    # block depth signal

    # --- Lexical signals (15 dims) [49:64] ---
    # Taint source presence
    vec[49] = 1.0 if any(src in stmt for src in TAINT_SOURCES) else 0.0
    # Unsafe API
    vec[50] = 1.0 if any(api in stmt for api in UNSAFE_APIS) else 0.0
    # Pointer arithmetic
    vec[51] = 1.0 if re.search(r'\w+\s*(\+|-)\s*\d+|\w+\s*\+\+|--\s*\w+', stmt) else 0.0
    # Type cast
    vec[52] = 1.0 if re.search(r'\(\s*(int|char|void|long|float|double)\s*\*?\s*\)', stmt) else 0.0
    # Array with variable index
    vec[53] = 1.0 if re.search(r'\w+\s*\[\s*[A-Za-z_]\w*\s*\]', stmt) else 0.0
    # String literal
    vec[54] = 1.0 if re.search(r'STR_\d+', stmt) else 0.0
    # Multiple assignments
    vec[55] = 1.0 if stmt.count("=") > 2 else 0.0
    # Comparison operators
    vec[56] = 1.0 if re.search(r'[<>]=?|==|!=', stmt) else 0.0
    # Boolean operators
    vec[57] = 1.0 if re.search(r'&&|\|\|', stmt) else 0.0
    # Sizeof
    vec[58] = 1.0 if "sizeof" in stmt else 0.0
    # wchar / wide string ops
    vec[59] = 1.0 if re.search(r'\bw(char|mem|str|printf|scanf)', stmt) else 0.0
    # Function pointer
    vec[60] = 1.0 if re.search(r'\(\s*\*\s*\w+\s*\)', stmt) else 0.0
    # Struct/member access
    vec[61] = 1.0 if re.search(r'\w+\s*(->\s*\w+|\.\s*\w+)', stmt) else 0.0
    # Numeric bounds literal
    vec[62] = 1.0 if re.search(r'\b\d{2,}\b', stmt) else 0.0
    # Conditional expression (ternary)
    vec[63] = 1.0 if "?" in stmt else 0.0

    # --- Sink-specific features (8 dims) [64:72] ---
    # These target the key FS bottleneck: distinguishing tainted vs safe
    # format calls by encoding data-provenance signals.

    has_fmt_sink = any(api in stmt for api in FORMAT_SINKS)
    has_taint    = any(src in stmt for src in TAINT_SOURCES)

    # [64] Taint-sink co-occurrence: taint source AND format sink together
    vec[64] = 1.0 if (has_taint and has_fmt_sink) else 0.0

    # [65] Direct variable as format string (no literal) – key vuln pattern
    #      e.g. printf(buf) instead of printf("hello %s", name)
    if has_fmt_sink:
        # Check if format arg is a variable, not a string literal
        m = re.search(r'\b(?:printf|fprintf|sprintf|snprintf|syslog)\s*\(\s*([^,)]+)', stmt)
        if m:
            fmt_arg = m.group(1).strip()
            # Vulnerable: format arg is a bare variable (no string literal)
            is_literal = bool(re.search(r'STR_\d+|"[^"]*"', fmt_arg))
            vec[65] = 0.0 if is_literal else 1.0

    # [66] Format specifier count (normalized) – more specifiers = riskier
    spec_count = len(re.findall(r'%[-+0-9 #*.(hljztL]*[diouxXeEfgGcspaAn]', stmt))
    vec[66] = min(spec_count / 5.0, 1.0)

    # [67] Arg-specifier mismatch signal
    if has_fmt_sink and spec_count > 0:
        # Count arguments after the format string
        call_m = re.search(r'\b(?:printf|fprintf|sprintf|snprintf)\s*\((.+)\)\s*;?\s*$', stmt)
        if call_m:
            args = [a.strip() for a in re.split(r',(?![^()]*\))', call_m.group(1))]
            # Skip format arg (and stream/buffer arg for fprintf/sprintf)
            func_m = re.search(r'\b(fprintf|sprintf|snprintf)', stmt)
            n_skip = 2 if func_m else 1  # fprintf(stream, fmt, ...) vs printf(fmt, ...)
            n_extra_args = max(len(args) - n_skip, 0)
            if n_extra_args != spec_count:
                vec[67] = 1.0  # mismatch detected

    # [68] Bounded/safer variant (snprintf, vsnprintf have size param)
    vec[68] = 1.0 if re.search(r'\b[v]?snprintf\b', stmt) else 0.0

    # [69] String copy feeding into format call context
    #      strcpy/strcat near format sink = risky buffer manipulation
    vec[69] = 1.0 if re.search(r'\b(strcpy|strcat|strncpy|strncat|sprintf)\b', stmt) else 0.0

    # [70] User-controlled input source (argv, getenv, env variables)
    vec[70] = 1.0 if re.search(r'\b(argv|argc|getenv|getopt|optarg)\b', stmt) else 0.0

    # [71] Sink danger score: weighted by API risk level
    if has_fmt_sink:
        # Unbounded printf/sprintf = high risk; snprintf = lower
        if re.search(r'\b(sprintf|vsprintf)\b', stmt):
            vec[71] = 1.0   # unbounded write + format
        elif re.search(r'\b(printf|fprintf|vprintf|vfprintf|syslog)\b', stmt):
            vec[71] = 0.7   # format but no buffer overflow
        elif re.search(r'\b[v]?snprintf\b', stmt):
            vec[71] = 0.3   # bounded, safer

    # ── BO-specific buffer-overflow signals [72:80] ──────────────────────
    if ds_type == "bo":
        # [72] Unsafe unbounded copy/write (no size parameter)
        vec[72] = 1.0 if any(re.search(r'\b' + fn + r'\s*\(', stmt) for fn in _UNSAFE_COPY) else 0.0

        # [73] Safe bounded copy (has explicit size parameter)
        vec[73] = 1.0 if any(re.search(r'\b' + fn + r'\s*\(', stmt) for fn in _SAFE_COPY) else 0.0

        # [74] Dynamic size via strlen (variable-length → potentially unbounded)
        vec[74] = 1.0 if re.search(r'\bstrlen\s*\(', stmt) else 0.0

        # [75] Buffer/array declared with explicit numeric size
        #      e.g. char buf[20], char *p = malloc(20)
        vec[75] = 1.0 if re.search(
            r'\bchar\s+\w+\s*\[\s*\d+\s*\]'
            r'|\b(malloc|calloc)\s*\(\s*\d+', stmt) else 0.0

        # [76] Bounds check: explicit size comparison before buffer op
        #      e.g. if (len < 20), if (size >= MAX)
        vec[76] = 1.0 if re.search(
            r'\b(len|size|count|num|n|sz)\s*[<>]=?\s*\d+'
            r'|\d+\s*[<>]=?\s*(len|size|count|num|n|sz)\b', stmt) else 0.0

        # [77] memcpy/memmove with a variable (non-literal) size → risky
        vec[77] = 1.0 if re.search(
            r'\b(memcpy|memmove|bcopy)\s*\([^,]+,[^,]+,\s*[A-Za-z_]\w*', stmt) else 0.0

        # [78] sizeof used as size guard — indicates careful programming
        vec[78] = 1.0 if re.search(r'\bsizeof\s*\(', stmt) else 0.0

        # [79] Pointer/index arithmetic that could overflow buffer bounds
        #      e.g. buf + len, ptr + n, data + size
        vec[79] = 1.0 if re.search(
            r'\b\w+\s*\+\s*(len|size|count|num|n\b|idx|index|offset)', stmt) else 0.0

        # ── Numeric literal features [80:84] — critical for BO size analysis ──
        all_literals = [int(m) for m in re.findall(r'\b(\d+)\b', stmt)
                        if 1 <= int(m) <= 100000]
        if all_literals:
            max_lit = max(all_literals)
            # [80] Largest numeric literal normalised (buf/write sizes like 20, 100, 1024)
            vec[80] = min(max_lit / 1000.0, 1.0)
            # [81] Is the literal large (>100)? Large literals in copy ops = high risk
            vec[81] = 1.0 if max_lit > 100 else 0.0
            # [82] Literal appears in a memory operation argument (copy size context)
            vec[82] = 1.0 if (all_literals and re.search(
                r'\b(memcpy|memmove|strncpy|strncat|snprintf|malloc|calloc|realloc)\s*\(', stmt)
            ) else 0.0
            # [83] Literal matches typical small buffer size (4–256): declaration signal
            vec[83] = 1.0 if any(4 <= v <= 256 for v in all_literals) else 0.0

    # ── FS-specific format-string signals [72:84] ──────────────────────────────
    # Key finding: vulnerable FS functions have +23% more taint sources, are more
    # complex (longer, more control flow). Node-level features PLUS function-level
    # context (injected via func_stats) give the GAT the discriminative signal.
    if ds_type == "fs":
        # ── Function-level context features (same for ALL nodes in this function) ──
        # These broadcast the function's global vulnerability indicators to every node,
        # so the GAT has direct access without needing long-range message passing.
        if func_stats:
            # [72] Normalised taint-source count (strongest signal: +23% in vuln)
            vec[72] = min(func_stats.get('n_taint', 0) / 5.0, 1.0)
            # [73] Has at least one taint source (boolean)
            vec[73] = 1.0 if func_stats.get('n_taint', 0) > 0 else 0.0
            # [74] Normalised format-sink count
            vec[74] = min(func_stats.get('n_fmt', 0) / 5.0, 1.0)
            # [75] Taint-to-sink ratio (high ratio = multiple taint sources per call)
            n_fmt = max(func_stats.get('n_fmt', 0), 1)
            vec[75] = min(func_stats.get('n_taint', 0) / n_fmt, 3.0) / 3.0
            # [76] Normalised statement count (longer functions more vuln)
            vec[76] = min(func_stats.get('n_stmts', 0) / 50.0, 1.0)
            # [77] Normalised loop count (+20% in vuln)
            vec[77] = min(func_stats.get('n_loops', 0) / 5.0, 1.0)
            # [78] Normalised if-statement count (+11% in vuln)
            vec[78] = min(func_stats.get('n_ifs', 0) / 15.0, 1.0)
            # [79] Taint source present AND format sink present (co-occurrence)
            vec[79] = 1.0 if (func_stats.get('n_taint', 0) > 0 and
                               func_stats.get('n_fmt', 0) > 0) else 0.0

        # ── Node-level per-statement features ──────────────────────────────────
        # [80] This statement is a taint source (data enters here)
        vec[80] = 1.0 if any(re.search(r'\b' + src + r'\s*\(', stmt)
                              for src in TAINT_SOURCES) else 0.0

        # [81] This statement calls a format sink with a VARIABLE as format arg
        #      (vuln pattern: printf(VAR_N, ...) instead of printf(STR_N, ...))
        _fmt_m = re.search(r'\b(printf|fprintf|vprintf|vfprintf|wprintf)\s*\(', stmt)
        if _fmt_m:
            _body = stmt[_fmt_m.end():]
            if 'fprintf' in _fmt_m.group(0) or 'vfprintf' in _fmt_m.group(0):
                _c = _body.find(',')
                _body = _body[_c+1:].strip() if _c != -1 else _body
            _fmt_tok = re.split(r',', _body)[0].strip().rstrip(')')
            vec[81] = 0.0 if re.search(r'STR_\d+|TYPE_\d+|"[^"]*"', _fmt_tok) else 1.0

        # [82] This statement calls a format sink with a CONSTANT as format arg (safe)
        if _fmt_m:
            _body = stmt[_fmt_m.end():]
            if 'fprintf' in _fmt_m.group(0) or 'vfprintf' in _fmt_m.group(0):
                _c = _body.find(',')
                _body = _body[_c+1:].strip() if _c != -1 else _body
            _fmt_tok = re.split(r',', _body)[0].strip().rstrip(')')
            vec[82] = 1.0 if re.search(r'STR_\d+|TYPE_\d+|"[^"]*"', _fmt_tok) else 0.0

        # [83] This statement uses sprintf/vsprintf (combined buffer+format risk)
        vec[83] = 1.0 if re.search(r'\b(sprintf|vsprintf|swprintf)\s*\(', stmt) else 0.0

    return vec


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

class GraphBuilder:
    """
    Builds all 7 graphs from a single C/C++ function string.
    All graphs are NetworkX DiGraph instances.
    """

    def __init__(self, code: str, sample_id: int = 0, ds_type: str = "all"):
        self.code      = code
        self.sample_id = sample_id
        self.ds_type   = ds_type
        # Try tree-sitter split first; fall back to regex split
        ts_result = _ts_split_statements(code)
        if ts_result:
            self.stmts      = [t  for t,  _, _ in ts_result]
            self._ts_types  = [ty for _, ty, _ in ts_result]
            self._ts_depths = [d  for _, _,  d in ts_result]
        else:
            self.stmts      = split_statements(code)
            self._ts_types  = None
            self._ts_depths = None
        self.n         = len(self.stmts)
        self._var_defs: dict[str, list[int]] = {}   # var → stmt indices where defined
        self._calls:    list[tuple[int, str]] = []  # (stmt_idx, callee_name)
        self._alloc_nodes:  list[int] = []
        self._free_nodes:   list[int] = []
        self._access_nodes: list[int] = []
        self._taint_src:    list[int] = []
        self._taint_snk:    list[int] = []
        # Pre-compute function-level stats for FS (injected into every node)
        self._fs_func_stats: dict = self._compute_fs_func_stats() if ds_type == "fs" else {}
        self._analyse()

    def _compute_fs_func_stats(self) -> dict:
        """Function-level statistics for FS — broadcast to all nodes as context."""
        _TAINT = {'fscanf','scanf','fgets','getenv','argv','recv','recvfrom',
                  'read','fread','getchar','cin','gets','sscanf'}
        _FMT   = {'printf','fprintf','sprintf','snprintf','vsprintf','vsnprintf',
                  'wprintf','vprintf','syslog','vfprintf','swprintf'}
        full = self.code
        n_taint = sum(1 for s in _TAINT if re.search(r'\b' + s + r'\b', full))
        n_fmt   = sum(1 for s in _FMT   if re.search(r'\b' + s + r'\b', full))
        n_stmts = full.count(';')
        n_ifs   = len(re.findall(r'\bif\s*\(', full))
        n_loops = len(re.findall(r'\b(for|while|do)\s*[({]', full))
        return dict(n_taint=n_taint, n_fmt=n_fmt, n_stmts=n_stmts,
                    n_ifs=n_ifs, n_loops=n_loops)

    # ----------------------------------------------------------------
    # Pre-analysis pass (populate data structures used by all builders)
    # ----------------------------------------------------------------

    def _analyse(self) -> None:
        for i, stmt in enumerate(self.stmts):
            stype = self._classify(i)

            # Variable definitions
            for var in re.findall(r'\b(VAR_\d+|ARR_\d+|func_\d+)\b', stmt):
                # Mark as def if it's on the LHS of = or in a decl
                if re.search(rf'\b{re.escape(var)}\s*=|\b(int|char|void|wchar_t|float|double|long|short)\b.*\b{re.escape(var)}\b', stmt):
                    self._var_defs.setdefault(var, []).append(i)

            # Function calls
            for call in re.findall(r'\b(\w+)\s*\(', stmt):
                if call not in {"if", "for", "while", "switch", "return"}:
                    self._calls.append((i, call))

            # Memory classification
            if stype == "ALLOC" or re.search(r'\b(malloc|calloc|realloc|new|strdup)\b', stmt):
                self._alloc_nodes.append(i)
            if stype == "FREE" or re.search(r'\b(free|delete)\b', stmt):
                self._free_nodes.append(i)
            if re.search(r'\w+\s*\[|\*\s*\w+|memcpy|memset|strcpy|strcat', stmt):
                self._access_nodes.append(i)

            # Taint sources / sinks
            if any(src in stmt for src in TAINT_SOURCES) or re.search(r'\bVAR_\d+\s*\[', stmt):
                self._taint_src.append(i)
            if any(snk in stmt for snk in FORMAT_SINKS | MEM_OPS):
                self._taint_snk.append(i)

    # ----------------------------------------------------------------
    # Helper: find the statement index AFTER a control-flow block
    # ----------------------------------------------------------------

    def _find_block_end(self, ctrl_i: int) -> int:
        """
        Find the index of the statement AFTER the entire block starting at ctrl_i.
        Returns len(self.stmts) as a sentinel meaning EXIT.
        Uses tree-sitter depth info when available (fast + accurate);
        otherwise falls back to brace counting.
        """
        if self._ts_depths is not None:
            ctrl_depth = self._ts_depths[ctrl_i]
            for j in range(ctrl_i + 1, self.n):
                if self._ts_depths[j] <= ctrl_depth:
                    return j
            return self.n
        # Regex fallback: brace counting
        depth = 0
        for j in range(ctrl_i, self.n):
            s = self.stmts[j]
            depth += s.count("{") - s.count("}")
            if j > ctrl_i and depth <= 0:
                nxt = j + 1
                return nxt if nxt < self.n else self.n
        return self.n   # sentinel → EXIT

    # ----------------------------------------------------------------
    # Helper: classify statement at index i
    # ----------------------------------------------------------------

    def _classify(self, i: int) -> str:
        """Return node type for statement at index i.
        Uses pre-computed tree-sitter types when available; falls back to regex."""
        if self._ts_types is not None and 0 <= i < len(self._ts_types):
            return self._ts_types[i]
        return classify_statement(self.stmts[i])

    # ----------------------------------------------------------------
    # Helper: add a node with full attributes
    # ----------------------------------------------------------------

    def _add_node(
        self,
        G: nx.DiGraph,
        node_id: int,
        node_type: str,
        stmt: str,
        stmt_idx: int,
    ) -> None:
        G.add_node(
            node_id,
            ntype=node_type,
            label=stmt[:80],                          # truncate for storage
            stmt_idx=stmt_idx,
            feature=encode_node_features(node_type, stmt, stmt_idx, self.n, self.ds_type, self._fs_func_stats),
        )

    # ----------------------------------------------------------------
    # 1. AST - Abstract Syntax Tree
    # ----------------------------------------------------------------

    def build_ast(self) -> nx.DiGraph:
        """
        Hierarchical decomposition of the function.
        Root → function def → statements → sub-expressions.
        When tree-sitter depths are available, uses depth-based parent
        tracking for accurate hierarchy; otherwise falls back to brace counting.
        """
        G = nx.DiGraph(graph_type="AST")
        root_id = -1
        G.add_node(root_id, ntype="ENTRY", label="FUNC_ROOT",
                   stmt_idx=-1, feature=np.zeros(FEATURE_DIM_BO if self.ds_type == "bo" else (FEATURE_DIM_FS if self.ds_type == "fs" else FEATURE_DIM), np.float32))

        if self._ts_depths is not None:
            # Depth-based parent tracking: depth_to_parent[d] = last node at depth d
            depth_to_parent: dict[int, int] = {-1: root_id}
            for i, stmt in enumerate(self.stmts):
                stype = self._classify(i)
                self._add_node(G, i, stype, stmt, i)
                d = self._ts_depths[i]
                parent = depth_to_parent.get(d - 1, root_id)
                G.add_edge(parent, i, etype="AST_CHILD", weight=1.0)
                depth_to_parent[d] = i
                # Prune stale deeper entries
                for k in [k for k in depth_to_parent if k > d]:
                    del depth_to_parent[k]
        else:
            # Fallback: brace-depth block stack
            block_stack = [root_id]
            for i, stmt in enumerate(self.stmts):
                stype = self._classify(i)
                self._add_node(G, i, stype, stmt, i)
                parent = block_stack[-1]
                G.add_edge(parent, i, etype="AST_CHILD", weight=1.0)
                if stype in ("IF", "FOR", "WHILE", "SWITCH", "FUNC_DEF"):
                    block_stack.append(i)
                elif stype == "ELSE" and len(block_stack) > 1:
                    block_stack.pop()
                    block_stack.append(i)
                elif stmt.strip() == "}" and len(block_stack) > 1:
                    block_stack.pop()

        # Sub-expression edges (array accesses + call sites) for all stmts
        for i, stmt in enumerate(self.stmts):
            for arr in re.findall(r'\b(\w+)\s*\[', stmt):
                sub_id = (i, f"ARR_{arr}")
                G.add_node(sub_id, ntype="MEM_READ",
                           label=f"{arr}[...]", stmt_idx=i,
                           feature=encode_node_features("MEM_READ", stmt, i, self.n, self.ds_type, self._fs_func_stats))
                G.add_edge(i, sub_id, etype="AST_CHILD", weight=0.8)

            for _, callee in [(ci, cn) for ci, cn in self._calls if ci == i]:
                sub_id = (i, f"CALL_{callee}")
                ntype  = "UNSAFE_API" if callee in UNSAFE_APIS else "FUNC_CALL"
                G.add_node(sub_id, ntype=ntype,
                           label=f"{callee}(...)", stmt_idx=i,
                           feature=encode_node_features(ntype, stmt, i, self.n, self.ds_type, self._fs_func_stats))
                G.add_edge(i, sub_id, etype="AST_CHILD", weight=0.9)

        return G

    # ----------------------------------------------------------------
    # 2. CFG - Control Flow Graph
    # ----------------------------------------------------------------

    def build_cfg(self) -> nx.DiGraph:
        """
        Basic-block level CFG.
        Each statement is a node; edges represent possible next-statement flow.
        Conditional statements create true/false branches.
        """
        G = nx.DiGraph(graph_type="CFG")

        entry_id = "ENTRY"
        exit_id  = "EXIT"
        G.add_node(entry_id, ntype="ENTRY", label="ENTRY",
                   stmt_idx=-1, feature=np.zeros(FEATURE_DIM_BO if self.ds_type == "bo" else (FEATURE_DIM_FS if self.ds_type == "fs" else FEATURE_DIM), np.float32))
        G.add_node(exit_id, ntype="EXIT", label="EXIT",
                   stmt_idx=self.n, feature=np.zeros(FEATURE_DIM_BO if self.ds_type == "bo" else (FEATURE_DIM_FS if self.ds_type == "fs" else FEATURE_DIM), np.float32))

        for i, stmt in enumerate(self.stmts):
            stype = self._classify(i)
            self._add_node(G, i, stype, stmt, i)

        # Entry → first statement
        if self.n > 0:
            G.add_edge(entry_id, 0, etype="CFG_NEXT", weight=1.0)

        # Build sequential + branching edges
        label_map: dict[str, int] = {}
        for i, stmt in enumerate(self.stmts):
            stype = self._classify(i)

            # Collect goto labels
            if stype == "LABEL":
                lname = re.match(r'\s*(\w+)\s*:', stmt)
                if lname:
                    label_map[lname.group(1)] = i

        for i, stmt in enumerate(self.stmts):
            stype = self._classify(i)
            next_i = i + 1

            if stype == "RETURN":
                G.add_edge(i, exit_id, etype="CFG_NEXT", weight=1.0)

            elif stype == "GOTO":
                target = re.search(r'goto\s+(\w+)', stmt)
                if target and target.group(1) in label_map:
                    t = label_map[target.group(1)]
                    G.add_edge(i, t, etype="CFG_BACK" if t <= i else "CFG_NEXT", weight=1.0)
                elif next_i < self.n:
                    G.add_edge(i, next_i, etype="CFG_NEXT", weight=1.0)

            elif stype in ("IF",):
                # True branch: next statement (first stmt in the if-body)
                if next_i < self.n:
                    G.add_edge(i, next_i, etype="CFG_TRUE", weight=1.0)
                # False branch: skip past the ENTIRE if-block using brace counting
                false_idx = self._find_block_end(i)
                false_target = false_idx if false_idx < self.n else exit_id
                G.add_edge(i, false_target, etype="CFG_FALSE", weight=1.0)

            elif stype in ("FOR", "WHILE"):
                # Body edge
                if next_i < self.n:
                    G.add_edge(i, next_i, etype="CFG_TRUE", weight=1.0)
                # Loop back + exit
                G.add_edge(i, i, etype="CFG_BACK", weight=0.5)   # self-loop = possible re-entry
                G.add_edge(i, exit_id, etype="CFG_FALSE", weight=1.0)

            else:
                if next_i < self.n:
                    G.add_edge(i, next_i, etype="CFG_NEXT", weight=1.0)
                elif stype != "RETURN":
                    G.add_edge(i, exit_id, etype="CFG_NEXT", weight=1.0)

        return G

    # ----------------------------------------------------------------
    # 3. DFG - Data Flow Graph (def-use chains)
    # ----------------------------------------------------------------

    def build_dfg(self) -> nx.DiGraph:
        """
        Nodes = statements; edges = def → use for each variable.
        A variable defined at stmt i and used at stmt j creates edge i→j.
        """
        G = nx.DiGraph(graph_type="DFG")

        for i, stmt in enumerate(self.stmts):
            stype = self._classify(i)
            self._add_node(G, i, stype, stmt, i)

        # For every var, connect each definition to each subsequent use
        all_vars = set(self._var_defs.keys())
        for var in all_vars:
            def_sites = self._var_defs.get(var, [])
            for i, stmt in enumerate(self.stmts):
                if re.search(rf'\b{re.escape(var)}\b', stmt):
                    for def_i in def_sites:
                        if def_i != i:
                            G.add_edge(
                                def_i, i,
                                etype="DFG_DEF_USE",
                                var=var,
                                weight=1.0,
                            )

        # Inter-statement data flow via function arguments
        for call_i, callee in self._calls:
            # Arguments of a call depend on prior assignments
            call_stmt = self.stmts[call_i]
            arg_vars  = re.findall(r'\b(VAR_\d+|ARR_\d+)\b', call_stmt)
            for av in arg_vars:
                for def_i in self._var_defs.get(av, []):
                    if def_i < call_i:
                        G.add_edge(def_i, call_i, etype="DFG_DEF_USE",
                                   var=av, weight=0.9)

        return G

    # ----------------------------------------------------------------
    # 4. PDG - Program Dependence Graph (CFG + DFG)
    # ----------------------------------------------------------------

    def build_pdg(self) -> nx.DiGraph:
        """
        Unified graph: control dependency edges (PDG_CTRL)
        + data dependency edges (PDG_DATA).
        """
        G = nx.DiGraph(graph_type="PDG")

        for i, stmt in enumerate(self.stmts):
            stype = self._classify(i)
            self._add_node(G, i, stype, stmt, i)

        # Control dependencies: if/for/while → dominated statements
        ctrl_nodes = [
            i for i in range(self.n)
            if self._classify(i) in ("IF", "FOR", "WHILE", "SWITCH")
        ]
        for ci in ctrl_nodes:
            for j in range(ci + 1, min(ci + 6, self.n)):
                G.add_edge(ci, j, etype="PDG_CTRL", weight=1.0)

        # Data dependencies (same as DFG)
        for var, def_sites in self._var_defs.items():
            for i, stmt in enumerate(self.stmts):
                if re.search(rf'\b{re.escape(var)}\b', stmt):
                    for def_i in def_sites:
                        if def_i != i:
                            G.add_edge(def_i, i, etype="PDG_DATA",
                                       var=var, weight=1.0)

        return G

    # ----------------------------------------------------------------
    # 5. TPG - Taint Propagation Graph
    # ----------------------------------------------------------------

    def build_tpg(self) -> nx.DiGraph:
        """
        Tracks untrusted data from taint sources through assignments
        to dangerous sinks (format/memory functions).
        """
        G = nx.DiGraph(graph_type="TPG")

        for i, stmt in enumerate(self.stmts):
            stype = self._classify(i)
            # Override type for taint nodes
            if i in self._taint_src:
                stype = "TAINT_SOURCE"
            elif i in self._taint_snk:
                stype = "TAINT_SINK"
            self._add_node(G, i, stype, stmt, i)

        # Propagate taint: source → intermediate → sink
        tainted_vars: set[str] = set()

        for i, stmt in enumerate(self.stmts):
            # Any var assigned from a taint source becomes tainted
            if i in self._taint_src:
                new_vars = re.findall(r'\b(VAR_\d+|ARR_\d+)\b', stmt)
                tainted_vars.update(new_vars)

            # Propagation through assignments
            if "=" in stmt and "==" not in stmt:
                lhs_vars = re.findall(r'\b(VAR_\d+|ARR_\d+)\b', stmt.split("=")[0])
                rhs_vars = re.findall(r'\b(VAR_\d+|ARR_\d+)\b', stmt.split("=", 1)[1])
                if tainted_vars & set(rhs_vars):
                    tainted_vars.update(lhs_vars)
                    # Find source statements and add propagation edges
                    for src_i in self._taint_src:
                        G.add_edge(src_i, i, etype="TAINT_FLOW", weight=1.0)

            # Check if tainted var reaches a sink
            if i in self._taint_snk:
                sink_vars = re.findall(r'\b(VAR_\d+|ARR_\d+)\b', stmt)
                if tainted_vars & set(sink_vars):
                    for j in range(max(0, i - 5), i):
                        if j in self._taint_src or G.has_edge(j, i - 1):
                            G.add_edge(j, i, etype="TAINT_FLOW", weight=1.5)
                    for src_i in self._taint_src:
                        G.add_edge(src_i, i, etype="TAINT_FLOW", weight=2.0)

        # Connect adjacent propagation nodes
        for i in range(self.n - 1):
            if not G.has_edge(i, i + 1):
                s_i = self._classify(i)
                if s_i in ("ASSIGN", "CALL", "TAINT_PROPAGATE"):
                    G.add_edge(i, i + 1, etype="TAINT_FLOW", weight=0.5)

        return G

    # ----------------------------------------------------------------
    # 6. MAG - Memory Access Graph
    # ----------------------------------------------------------------

    def build_mag(self, dataset_key=None) -> nx.DiGraph:
        """
        Enhanced: For FS, if MAG is empty, attempt to infer and add edges between alloc/free/realloc for the same variable.
        """
        G = nx.DiGraph(graph_type="MAG")
        for i, stmt in enumerate(self.stmts):
            stype = self._classify(i)
            self._add_node(G, i, stype, stmt, i)

        # ALLOC → ACCESS edges
        for a_i in self._alloc_nodes:
            for acc_i in self._access_nodes:
                if acc_i > a_i:
                    a_vars   = set(re.findall(r'\b(VAR_\d+|ARR_\d+)\b', self.stmts[a_i]))
                    acc_vars = set(re.findall(r'\b(VAR_\d+|ARR_\d+)\b', self.stmts[acc_i]))
                    if a_vars & acc_vars:
                        G.add_edge(a_i, acc_i, etype="MEM_ALLOC_USE", weight=1.0)

        # ALLOC → FREE edges
        for a_i in self._alloc_nodes:
            for f_i in self._free_nodes:
                if f_i > a_i:
                    a_vars = set(re.findall(r'\b(VAR_\d+|ARR_\d+)\b', self.stmts[a_i]))
                    f_vars = set(re.findall(r'\b(VAR_\d+|ARR_\d+)\b', self.stmts[f_i]))
                    if a_vars & f_vars:
                        G.add_edge(a_i, f_i, etype="MEM_ALLOC_FREE", weight=1.0)

        # FREE → ACCESS edges (USE-AFTER-FREE pattern - critical signal)
        for f_i in self._free_nodes:
            for acc_i in self._access_nodes:
                if acc_i > f_i:
                    f_vars   = set(re.findall(r'\b(VAR_\d+|ARR_\d+)\b', self.stmts[f_i]))
                    acc_vars = set(re.findall(r'\b(VAR_\d+|ARR_\d+)\b', self.stmts[acc_i]))
                    if f_vars & acc_vars:
                        G.add_edge(f_i, acc_i, etype="MEM_FREE_USE", weight=2.0)

        # --- Enhancement for FS: If MAG is empty, try to infer edges ---
        if dataset_key == "fs" and G.number_of_edges() == 0:
            # Try to add edges between alloc/free for same variable
            for a_i in self._alloc_nodes:
                a_vars = set(re.findall(r'\b(VAR_\d+|ARR_\d+)\b', self.stmts[a_i]))
                for f_i in self._free_nodes:
                    f_vars = set(re.findall(r'\b(VAR_\d+|ARR_\d+)\b', self.stmts[f_i]))
                    if a_vars & f_vars:
                        G.add_edge(a_i, f_i, etype="MEM_ALLOC_FREE", weight=0.5)
            # Also try to add alloc→access and free→access for same variable
            for a_i in self._alloc_nodes:
                a_vars = set(re.findall(r'\b(VAR_\d+|ARR_\d+)\b', self.stmts[a_i]))
                for acc_i in self._access_nodes:
                    acc_vars = set(re.findall(r'\b(VAR_\d+|ARR_\d+)\b', self.stmts[acc_i]))
                    if a_vars & acc_vars:
                        G.add_edge(a_i, acc_i, etype="MEM_ALLOC_USE", weight=0.5)
            for f_i in self._free_nodes:
                f_vars = set(re.findall(r'\b(VAR_\d+|ARR_\d+)\b', self.stmts[f_i]))
                for acc_i in self._access_nodes:
                    acc_vars = set(re.findall(r'\b(VAR_\d+|ARR_\d+)\b', self.stmts[acc_i]))
                    if f_vars & acc_vars:
                        G.add_edge(f_i, acc_i, etype="MEM_FREE_USE", weight=1.0)
        return G

    # ----------------------------------------------------------------
    # 7. CG - Call Graph
    # ----------------------------------------------------------------

    def build_cg(self) -> nx.DiGraph:
        """
        Intra-function call graph.
        Nodes: caller statement + callee function.
        Edges: call site → callee (with unsafe API annotation).
        """
        G = nx.DiGraph(graph_type="CG")

        # Add all statements as potential caller nodes
        for i, stmt in enumerate(self.stmts):
            stype = self._classify(i)
            self._add_node(G, i, stype, stmt, i)

        # Add unique callee nodes and call edges
        seen_callees: set[str] = set()
        for call_i, callee in self._calls:
            callee_node_id = f"CALLEE_{callee}"
            if callee_node_id not in seen_callees:
                is_unsafe = callee in UNSAFE_APIS
                G.add_node(
                    callee_node_id,
                    ntype="UNSAFE_API" if is_unsafe else "FUNC_CALL",
                    label=callee,
                    stmt_idx=-1,
                    feature=encode_node_features(
                        "UNSAFE_API" if is_unsafe else "FUNC_CALL",
                        callee, -1, self.n, self.ds_type, self._fs_func_stats
                    ),
                )
                seen_callees.add(callee_node_id)

            G.add_edge(
                call_i,
                callee_node_id,
                etype="CALL_EDGE",
                callee=callee,
                is_unsafe=callee in UNSAFE_APIS,
                weight=2.0 if callee in UNSAFE_APIS else 1.0,
            )

        return G


    def build_fsg(self) -> nx.DiGraph:
        """
        Format String Graph (FSG) -- 8th graph, critical for CWE-134 detection.

        Full data-flow chain modelled:
            taint_source -> variable -> printf (dangerous)
            literal_str  -> printf           (safe)

        Node types:
          FORMAT_FUNC  -- printf/fprintf/sprintf call site
          FORMAT_STR   -- format string argument node
          FORMAT_ARG   -- additional argument passed to format function
          FORMAT_SPEC  -- each %s/%d/etc specifier in the literal
          TAINT_SRC    -- taint source or tainted variable node
          MISMATCH     -- vulnerability signal (direct var as fmt / mismatch)
          DATAFLOW     -- intermediate variable in input->var->printf chain

        Edges:
          CALL_EDGE     : stmt_node  -> FORMAT_STR   (format arg)
          ARG_EDGE      : stmt_node  -> FORMAT_ARG   (extra args)
          SPEC_EDGE     : FORMAT_STR -> FORMAT_SPEC  (specifiers)
          TAINT_EDGE    : TAINT_SRC  -> DATAFLOW -> FORMAT_FUNC  (data-flow chain)
          MISMATCH_EDGE : stmt_node  -> MISMATCH  (vulnerability signal, weight=4)
          DATAFLOW_EDGE : source_stmt -> var_stmt -> fmt_stmt  (inter-stmt data flow)
        """
        import re as _re
        G = nx.DiGraph(graph_type="FSG")

        # Add all statements as background nodes
        for i, stmt in enumerate(self.stmts):
            stype = self._classify(i)
            self._add_node(G, i, stype, stmt, i)

        fmt_func_pattern = _re.compile(
            r'\b(printf|fprintf|sprintf|snprintf|vprintf|vfprintf|vsprintf|'
            r'vsnprintf|wprintf|wfprintf|wsprintf|syslog)\s*\('
        )
        spec_pattern = _re.compile(r'%[-+0-9 #*.(hljztL]*[diouxXeEfgGcspaAn%]')

        # ── Step 1: Build taint map WITH stmt-level tracking ──────────────────
        # Maps variable name -> SET of stmt indices where it was tainted
        # (set prevents exponential duplication during taint propagation)
        taint_origin: dict[str, set[int]] = {}   # var -> {stmt_idx where tainted}
        tainted_vars: set[str] = set()

        for i, stmt in enumerate(self.stmts):
            lhs_vars = _re.findall(r'\b([A-Za-z_]\w*)\s*=', stmt)
            rhs_vars = _re.findall(r'\b([A-Za-z_]\w*)\b', stmt)

            if any(src in stmt for src in TAINT_SOURCES):
                # This statement is a taint source
                for v in lhs_vars:
                    taint_origin.setdefault(v, set()).add(i)
                    tainted_vars.add(v)
                # Parameters directly tainted (e.g. scanf reads into param)
                for v in rhs_vars:
                    taint_origin.setdefault(v, set()).add(i)
                    tainted_vars.add(v)
            else:
                # Propagate taint through assignments
                if tainted_vars & set(rhs_vars):
                    for v in lhs_vars:
                        # Inherit taint origins from rhs (set union — no duplicates)
                        origins = taint_origin.setdefault(v, set())
                        for rv in rhs_vars:
                            if rv in taint_origin:
                                origins |= taint_origin[rv]
                        tainted_vars.add(v)

        node_counter = self.n  # synthetic nodes start after stmt nodes

        # ── Step 2: Process each format function call ─────────────────────────
        for i, stmt in enumerate(self.stmts):
            m = fmt_func_pattern.search(stmt)
            if not m:
                continue

            func_name = m.group(1)

            # Parse call arguments
            call_body_match = _re.search(r'\((.+)\)\s*;?\s*$', stmt)
            if not call_body_match:
                continue
            call_body = call_body_match.group(1)
            args = [a.strip() for a in _re.split(r',(?![^()]*\))', call_body)]

            # Format arg index: 0 for printf/vprintf, 1 for fprintf/sprintf etc
            fmt_arg_idx = 1 if func_name in ('fprintf','sprintf','snprintf',
                                              'vfprintf','vsprintf','vsnprintf',
                                              'wfprintf','wsprintf') else 0
            if len(args) <= fmt_arg_idx:
                continue

            fmt_arg   = args[fmt_arg_idx]
            extra_args = args[fmt_arg_idx + 1:]

            # Detect if format arg is a tainted variable
            fmt_vars = set(_re.findall(r'\b([A-Za-z_]\w*)\b', fmt_arg))
            is_tainted_fmt = bool(tainted_vars & fmt_vars)
            is_direct_var  = (not _re.search(r'STR_\d+|"[^"]*"', fmt_arg) and
                              bool(_re.search(r'\b[A-Za-z_]\w*\b', fmt_arg)))

            # ── Add FORMAT_STR node ──────────────────────────────────
            fmt_ntype   = "TAINT_SRC" if is_tainted_fmt else "FORMAT_STR"
            fmt_node_id = node_counter; node_counter += 1
            G.add_node(fmt_node_id,
                       ntype=fmt_ntype, label=fmt_arg[:60], stmt_idx=i,
                       feature=encode_node_features(fmt_ntype, fmt_arg, i, self.n, self.ds_type, self._fs_func_stats))
            G.add_edge(i, fmt_node_id, etype="CALL_EDGE",
                       weight=3.0 if is_tainted_fmt else 1.0)

            # ── DATA-FLOW CHAIN: source_stmt -> intermediate_stmts -> fmt_stmt ──
            # This is the key improvement: explicitly model input->var->printf
            if is_tainted_fmt:
                for var in (fmt_vars & tainted_vars):
                    if var in taint_origin:
                        for src_stmt_idx in taint_origin[var]:
                            if src_stmt_idx != i:
                                # Add TAINT_SRC node for the origin statement
                                src_node_id = node_counter; node_counter += 1
                                src_stmt = self.stmts[src_stmt_idx]
                                G.add_node(src_node_id,
                                           ntype="TAINT_SRC",
                                           label=f"SRC:{src_stmt[:50]}",
                                           stmt_idx=src_stmt_idx,
                                           feature=encode_node_features(
                                               "TAINT_SRC", src_stmt, src_stmt_idx, self.n, self.ds_type, self._fs_func_stats))

                                # DATAFLOW node representing the variable itself
                                df_node_id = node_counter; node_counter += 1
                                G.add_node(df_node_id,
                                           ntype="DATAFLOW",
                                           label=f"VAR:{var}",
                                           stmt_idx=src_stmt_idx,
                                           feature=encode_node_features(
                                               "TAINT_SRC", var, src_stmt_idx, self.n, self.ds_type, self._fs_func_stats))

                                # Full chain: src_stmt -> DATAFLOW(var) -> fmt_stmt
                                G.add_edge(src_node_id, df_node_id,
                                           etype="DATAFLOW_EDGE", weight=2.0)
                                G.add_edge(df_node_id, i,
                                           etype="TAINT_EDGE", weight=3.0)
                                G.add_edge(src_stmt_idx, i,
                                           etype="DATAFLOW_EDGE", weight=2.0)

                # Bidirectional taint edge on the FORMAT_STR node itself
                G.add_edge(fmt_node_id, i, etype="TAINT_EDGE", weight=3.0)

            # ── Specifier nodes ──────────────────────────────────────
            spec_count = 0
            if _re.search(r'STR_\d+|"[^"]*"', fmt_arg):
                specs = [s for s in spec_pattern.findall(fmt_arg) if s != '%%']
                spec_count = len(specs)
                for spec in specs:
                    sn_id = node_counter; node_counter += 1
                    G.add_node(sn_id, ntype="FORMAT_SPEC", label=spec,
                               stmt_idx=i,
                               feature=encode_node_features("FORMAT_SPEC", spec, i, self.n, self.ds_type, self._fs_func_stats))
                    G.add_edge(fmt_node_id, sn_id, etype="SPEC_EDGE", weight=1.0)

            # ── FORMAT_ARG nodes ─────────────────────────────────────
            arg_count = len(extra_args)
            for arg in extra_args:
                arg_vars    = set(_re.findall(r'\b(\w+)\b', arg))
                arg_tainted = bool(tainted_vars & arg_vars)
                an_id = node_counter; node_counter += 1
                G.add_node(an_id,
                           ntype="TAINT_SRC" if arg_tainted else "FORMAT_ARG",
                           label=arg[:60], stmt_idx=i,
                           feature=encode_node_features(
                               "TAINT_SRC" if arg_tainted else "FORMAT_ARG",
                               arg, i, self.n, self.ds_type, self._fs_func_stats))
                G.add_edge(i, an_id, etype="ARG_EDGE",
                           weight=2.0 if arg_tainted else 1.0)

                # Data-flow edges for tainted args
                if arg_tainted:
                    for var in (arg_vars & tainted_vars):
                        if var in taint_origin:
                            for src_i in taint_origin[var]:
                                if src_i != i:
                                    G.add_edge(src_i, i,
                                               etype="DATAFLOW_EDGE", weight=2.0)

            # ── MISMATCH / vulnerability signal node ─────────────────
            count_mismatch = (spec_count > 0 and arg_count != spec_count)
            is_vuln_signal = is_direct_var or count_mismatch or is_tainted_fmt

            if is_vuln_signal:
                mm_id = node_counter; node_counter += 1
                mm_label = (
                    f"TAINTED_FMT:{func_name}"   if is_tainted_fmt  else
                    f"DIRECT_VAR:{func_name}"    if is_direct_var   else
                    f"MISMATCH:{spec_count}s/{arg_count}a"
                )
                G.add_node(mm_id, ntype="MISMATCH", label=mm_label, stmt_idx=i,
                           feature=encode_node_features("MISMATCH", mm_label, i, self.n, self.ds_type, self._fs_func_stats))
                G.add_edge(i, mm_id, etype="MISMATCH_EDGE",
                           weight=4.0)   # strongest signal in entire pipeline

        return G


    # ----------------------------------------------------------------
    # Public: build all 8 graphs (+ VLG/APG for FS)
    # ----------------------------------------------------------------

    def build_all(self, dataset_key=None) -> dict[str, nx.DiGraph]:
        graphs = {
            "AST": self.build_ast(),
            "CFG": self.build_cfg(),
            "DFG": self.build_dfg(),
            "PDG": self.build_pdg(),
            "TPG": self.build_tpg(),
            "MAG": self.build_mag(dataset_key=dataset_key),
            "CG":  self.build_cg(),
            "FSG": self.build_fsg(),
        }
        # ── FS-only: VLG + APG (Variable Lineage + Argument Position) ──
        if dataset_key == "fs":
            try:
                from stage2_graph_improvements import (
                    build_vlg, build_apg, clip_tpg_edge_weights
                )
                graphs["VLG"] = build_vlg(self.code, self.sample_id)
                graphs["APG"] = build_apg(self.code, self.sample_id)
                # Clip TPG edge weights to prevent GAT instability
                graphs["TPG"] = clip_tpg_edge_weights(graphs["TPG"], max_weight=2.0)
            except Exception as _vlg_err:
                logger.debug(f"VLG/APG build skipped: {_vlg_err}")
        return graphs


# ---------------------------------------------------------------------------
# GraphBundle - container per sample
# ---------------------------------------------------------------------------

@dataclass
class GraphBundle:
    sample_id:  int
    label:      int
    graphs:     dict = field(default_factory=dict)   # graph_type → nx.DiGraph
    n_stmts:    int  = 0
    code_len:   int  = 0
    build_error: Optional[str] = None

    def is_valid(self) -> bool:
        return self.build_error is None and len(self.graphs) >= 8

    def node_counts(self) -> dict[str, int]:
        return {k: G.number_of_nodes() for k, G in self.graphs.items()}

    def edge_counts(self) -> dict[str, int]:
        return {k: G.number_of_edges() for k, G in self.graphs.items()}


# ---------------------------------------------------------------------------
# Process a single split CSV → list[GraphBundle]
# ---------------------------------------------------------------------------

def process_split(
    csv_path: Path,
    split_name: str,
    max_nodes: int = 500,
    chunk_size: int = 500,
) -> tuple[list[GraphBundle], dict]:
    """Process one CSV split, saving graph bundles in RAM-friendly chunks."""
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    logger.info(f"  {split_name}: {total_rows} samples")

    # Temp chunk files written every `chunk_size` samples to keep RAM low
    chunk_dir   = csv_path.parent  # same dir; hidden name
    chunk_paths: list[Path] = []
    current_chunk: list[GraphBundle] = []

    stats = {"total": 0, "valid": 0, "errors": 0,
             "avg_nodes": {g: 0 for g in ["AST","CFG","DFG","PDG","TPG","MAG","CG","FSG"]},
             "avg_edges": {g: 0 for g in ["AST","CFG","DFG","PDG","TPG","MAG","CG","FSG"]},
             "label_0": 0, "label_1": 0}

    for _, row in df.iterrows():
        stats["total"] += 1
        sample_id = int(row["id"])
        label     = int(row["label"])
        code      = str(row["code"])

        bundle = GraphBundle(sample_id=sample_id, label=label,
                             code_len=len(code))
        try:
            ds_key_here = csv_path.parent.name
            builder = GraphBuilder(code, sample_id, ds_type=ds_key_here)
            bundle.n_stmts = builder.n
            graphs  = builder.build_all(dataset_key=ds_key_here)

            # Truncate oversized graphs
            for gtype, G in graphs.items():
                if G.number_of_nodes() > max_nodes:
                    nodes_to_keep = list(G.nodes)[:max_nodes]
                    graphs[gtype] = G.subgraph(nodes_to_keep).copy()

            bundle.graphs = graphs
            stats["valid"] += 1

            for gtype, G in graphs.items():
                stats["avg_nodes"][gtype] = stats["avg_nodes"].get(gtype, 0) + G.number_of_nodes()
                stats["avg_edges"][gtype] = stats["avg_edges"].get(gtype, 0) + G.number_of_edges()

        except Exception as exc:
            bundle.build_error = str(exc)
            stats["errors"] += 1
            logger.debug(f"    Error on sample {sample_id}: {exc}")

        current_chunk.append(bundle)
        if label == 0:
            stats["label_0"] += 1
        else:
            stats["label_1"] += 1

        # Progress log every 100 samples
        if stats["total"] % 100 == 0:
            logger.info(
                f"    [{stats['total']}/{total_rows}] "
                f"valid={stats['valid']}  errors={stats['errors']}"
            )

        # Flush chunk to disk to free RAM
        if len(current_chunk) >= chunk_size:
            cp = chunk_dir / f".{split_name}_chunk{len(chunk_paths)}.pkl"
            with open(cp, "wb") as f:
                pickle.dump(current_chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
            chunk_paths.append(cp)
            current_chunk = []
            logger.info(f"    [chunk {len(chunk_paths)} saved — RAM freed]")

    # Compute averages
    n_valid = max(stats["valid"], 1)
    for gtype in stats["avg_nodes"]:
        stats["avg_nodes"][gtype] = round(stats["avg_nodes"][gtype] / n_valid, 1)
        stats["avg_edges"][gtype] = round(stats["avg_edges"][gtype] / n_valid, 1)

    logger.info(
        f"  {split_name} done: {stats['valid']}/{stats['total']} valid, "
        f"{stats['errors']} errors"
    )

    # Merge chunks + remainder into final list
    if chunk_paths:
        bundles: list[GraphBundle] = []
        for cp in chunk_paths:
            with open(cp, "rb") as f:
                bundles.extend(pickle.load(f))
            cp.unlink()
        bundles.extend(current_chunk)
    else:
        bundles = current_chunk

    return bundles, stats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        config_path = _ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def process_dataset(
    dataset_key: str,
    config: dict,
    splits: list[str],
    dry_run: bool = False,
) -> None:
    proc_dir   = _ROOT / config["data"]["processed_dir"] / dataset_key
    graphs_dir = _ROOT / config["data"]["graphs_dir"]    / dataset_key
    graphs_dir.mkdir(parents=True, exist_ok=True)
    max_nodes  = config["graphs"]["max_nodes_per_graph"]

    logger.info(f"{'='*60}")
    logger.info(f"Graph construction: {dataset_key.upper()}")
    logger.info(f"{'='*60}")

    all_stats = {}

    for split in splits:
        csv_path = proc_dir / f"{split}.csv"
        if not csv_path.exists():
            logger.warning(f"Split file not found: {csv_path}  (run Stage 1 first)")
            continue

        logger.info(f"Building graphs for split: {split}")
        bundles, stats = process_split(csv_path, split, max_nodes)
        all_stats[split] = stats

        if not dry_run:
            # Save bundles
            pkl_path = graphs_dir / f"{split}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(bundles, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"  Saved: {pkl_path}  ({len(bundles)} bundles)")

            # Save stats
            stats_path = graphs_dir / f"{split}_stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'-'*55}")
    print(f"  Graph Stats - {dataset_key.upper()}")
    print(f"{'-'*55}")
    for split, s in all_stats.items():
        print(f"\n  [{split}]  valid={s['valid']}  errors={s['errors']}")
        print(f"  {'Graph':<6}  {'Nodes':>8}  {'Edges':>8}")
        print(f"  {'-'*28}")
        for gtype in ["AST","CFG","DFG","PDG","TPG","MAG","CG","FSG"]:
            print(f"  {gtype:<6}  {s['avg_nodes'][gtype]:>8.1f}  {s['avg_edges'][gtype]:>8.1f}")
    print(f"{'-'*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QEGVD Stage 2 - Graph Construction"
    )
    parser.add_argument("--dataset", choices=["bo","fs","uaf","all"], required=True)
    parser.add_argument("--split",   choices=["train","val","test","all"], default="all")
    parser.add_argument("--config",  type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config   = load_config(args.config)
    datasets = ["bo","fs","uaf"] if args.dataset == "all" else [args.dataset]
    splits   = ["train","val","test"] if args.split == "all" else [args.split]

    for ds in datasets:
        process_dataset(ds, config, splits, args.dry_run)

    print("Stage 2 complete [OK]  ->  data/graphs/")


if __name__ == "__main__":
    main()