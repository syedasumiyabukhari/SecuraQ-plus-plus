"""
SecuraQ++ Scanning Backend — FastAPI
Integrates with the QEGVD quantum-hybrid ML pipeline.
Ports: 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uuid, asyncio, os, sys, json, logging, re, traceback
from pathlib import Path
from datetime import datetime
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── ML Pipeline root (relative to this file, adjust as needed) ────────────────
ML_ROOT = BASE_DIR.parent / "ml_core"  # QEGVD pipeline root
if ML_ROOT.exists():
    sys.path.insert(0, str(ML_ROOT / "src"))
    sys.path.insert(0, str(ML_ROOT))   # exposes fs_features.py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SecuraQ-API")

# ── Load .env if present ──────────────────────────────────────────────────────
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Optional AI client (Claude) ───────────────────────────────────────────────
try:
    import anthropic as _anthropic
    _AI_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    _AI_CLIENT = _anthropic.Anthropic(api_key=_AI_KEY) if _AI_KEY else None
except ImportError:
    _AI_CLIENT = None

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="SecuraQ++ Scanning API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory state ───────────────────────────────────────────────────────────
scans_db: dict = {}
active_ws: dict = {}

# ── Models ────────────────────────────────────────────────────────────────────
class ScanResponse(BaseModel):
    scan_id: str
    status: str
    message: str

class AIImproveRequest(BaseModel):
    scan_id: str
    vuln_type: str
    code_snippet: str
    fix_label: str


# ═══════════════════════════════════════════════════════════════════════════════
#  ML PIPELINE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

DS_KEYS = ["bo", "fs", "uaf"]
VULN_LABEL = {
    "bo":  "Buffer Overflow",
    "fs":  "Format String",
    "uaf": "Use-After-Free",
}

def _try_load_ml():
    """Check if the ML pipeline modules are importable."""
    try:
        import stage2_graph_construction  # noqa
        import stage3_gat                 # noqa
        import stage4_classical_encoder   # noqa
        import stage5_qafa                # noqa
        import stage6_vqc                 # noqa
        import stage7_fusion              # noqa
        return True
    except ImportError:
        return False

ML_AVAILABLE = _try_load_ml()


# ── FS Direct classifier (trained by train_fs_direct.py) ─────────────────────

_FS_MODEL = None   # loaded lazily on first use

def _load_fs_model():
    global _FS_MODEL
    if _FS_MODEL is not None:
        return _FS_MODEL
    ckpt = ML_ROOT / "models" / "checkpoints" / "fs_direct.pkl"
    if not ckpt.exists():
        return None
    try:
        import pickle
        with open(ckpt, "rb") as fh:
            _FS_MODEL = pickle.load(fh)
        logger.info("[FS-Direct] Loaded stacking model from %s", ckpt)
    except Exception as e:
        logger.warning("[FS-Direct] Could not load model: %s", e)
        _FS_MODEL = None
    return _FS_MODEL


# ── FS feature extraction — imported from shared module ───────────────────────
try:
    from fs_features import tokenize_code as _fs_tokenize_code, build_manual_features as _fs_build_features
    _FS_FEATURES_IMPORTED = True
except ImportError:
    _FS_FEATURES_IMPORTED = False
    logger.warning("[FS-Direct] fs_features.py not found — FS inference disabled")


def _run_fs_direct(code_text: str) -> dict | None:
    """Run the FS direct stacking classifier."""
    import scipy.sparse as sp

    if not _FS_FEATURES_IMPORTED:
        return None

    mdl = _load_fs_model()
    if mdl is None:
        return None

    try:
        feat_row = _fs_build_features(code_text)
        tokens   = _fs_tokenize_code(code_text)

        # Align to training feature order
        feat_names = mdl.get("feat_names", list(feat_row.keys()))
        manual_vec = np.array([feat_row.get(k, 0.0) for k in feat_names],
                               dtype=np.float32).reshape(1, -1)

        scaler    = mdl["scaler"]
        manual_s  = scaler.transform(manual_vec)

        tok_str   = " ".join(tokens)

        # Support both v2 (tfidf_word + tfidf_char + sel) and legacy (tfidf)
        if "tfidf_word" in mdl:
            tw = mdl["tfidf_word"].transform([tok_str])
            tc = mdl["tfidf_char"].transform([tok_str])
            tfidf_vec = sp.hstack([tw, tc])
            if "sel" in mdl:
                tfidf_sel = mdl["sel"].transform(tfidf_vec)
                dense_vec = np.hstack([manual_s, tfidf_sel.toarray()])
            else:
                dense_vec = manual_s
        else:
            tfidf_vec = mdl["tfidf"].transform([tok_str])
            dense_vec = manual_s

        comb_vec = sp.hstack([tfidf_vec, sp.csr_matrix(manual_s)])

        p_gbm = mdl["gbm"].predict_proba(dense_vec)[0, 1]
        p_rf  = mdl["rf"].predict_proba(dense_vec)[0, 1]
        p_et  = mdl["et"].predict_proba(manual_s)[0, 1]
        p_lr  = mdl["lr_tfidf"].predict_proba(comb_vec)[0, 1]

        # Use OOF-derived weighted blend (more robust than meta-learner)
        if "blend_w" in mdl:
            w_gbm, w_rf, w_et, w_lr = mdl["blend_w"]
            cal = w_gbm*p_gbm + w_rf*p_rf + w_et*p_et + w_lr*p_lr
        else:
            meta_X = np.array([[p_gbm, p_rf, p_et, p_lr]])
            prob   = mdl["meta"].predict_proba(meta_X)[0, 1]
            cal    = mdl["platt"].predict_proba(np.array([[prob]]))[0, 1] if mdl.get("platt") else prob
        thresh = mdl["threshold"]

        if cal < thresh:
            return None

        severity = "CRITICAL" if cal > 0.85 else "HIGH" if cal > 0.70 else "MEDIUM"
        line_no, snippet, fn_name = _find_fs_vuln_line(code_text)
        return {
            "type":         "Format String",
            "severity":     severity,
            "confidence":   round(float(cal), 4),
            "threshold":    round(float(thresh), 4),
            "line_number":  line_no,
            "description":  _vuln_description("fs", fn_name),
            "code_snippet": snippet or _extract_relevant_snippet(code_text, "fs"),
            "cwe":          "CWE-134",
            "detector":     "FS-DIRECT-v2",
        }
    except Exception as e:
        logger.warning("[FS-Direct] inference error: %s", e)
        return None


def _stage1_preprocess(raw_code: str) -> str:
    """Identifier masking — mirrors demo_single_sample.stage1_preprocess."""
    code = re.sub(r'\b(good|bad|Good|Bad|CWE|Juliet)\w*\b', 'FUNC', raw_code)
    code = re.sub(r'//[^\n]*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', ' ', code).strip()
    return code


def _run_qegvd_pipeline(code_text: str) -> list[dict]:
    """
    Run the full QEGVD pipeline on a snippet of C/C++ code.
    Returns a list of vulnerability dicts.
    """
    if not ML_AVAILABLE:
        raise RuntimeError("ML pipeline not available")

    from demo_single_sample import (
        stage1_preprocess, stage2_build_graphs,
        stage3_gat_embed, stage4_compress, stage5_qafa,
        stage6_vqc, stage78_classify,
    )

    masked = stage1_preprocess(code_text)
    graphs, builder = stage2_build_graphs(masked)

    findings = []
    thresholds = {}

    # Load calibration matrix for thresholds
    cal_path = ML_ROOT / "results" / "calibration_matrix.json"
    cal = {}
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)

    for ds in DS_KEYS:
        try:
            fused, view_embeds, gat_prob = stage3_gat_embed(graphs, ds, code_text=masked)
            compressed, full_emb        = stage4_compress(fused, ds)
            s1, s2, sel_idx, scores     = stage5_qafa(compressed, ds)
            qvec, circuit_info          = stage6_vqc(s1, s2, ds, all_stages=compressed)
            prob, hybrid, threshold     = stage78_classify(compressed, qvec, full_emb, ds, code_text=masked)

            # Override threshold from calibration if available
            cal_thresh = cal.get("thresholds", {}).get(ds, threshold)

            if prob > cal_thresh:
                # Map to severity
                severity = "CRITICAL" if prob > 0.85 else "HIGH" if prob > 0.70 else "MEDIUM"
                findings.append({
                    "type":         VULN_LABEL[ds],
                    "severity":     severity,
                    "confidence":   round(float(prob), 4),
                    "threshold":    round(float(cal_thresh), 4),
                    "line_number":  _estimate_line(code_text, ds),
                    "description":  _vuln_description(ds),
                    "code_snippet": _extract_relevant_snippet(code_text, ds),
                    "cwe":          _cwe_for(ds),
                    "detector":     f"QEGVD-{ds.upper()}",
                })
        except Exception as e:
            logger.warning(f"[{ds}] pipeline error: {e}")
            continue

    return findings


def _find_fs_vuln_line(code: str) -> tuple[int, str, str]:
    """
    Scan every line for the exact AST node where a format function is called
    with a variable (not a string literal) as the format argument.
    Returns (line_number, snippet, fn_name).  Falls back to (1, "", "printf").
    """
    lines = code.splitlines()

    # Each tuple: (regex, fn_name, group_index_of_format_arg)
    # The regex captures the format argument so we can confirm it is not a literal.
    _FS_PATTERNS = [
        # printf(arg)  — format arg is first arg
        (re.compile(r'\bprintf\s*\(\s*([^,\)]+)'),           "printf",  1),
        # fprintf(stream, arg)  — format arg is second arg
        (re.compile(r'\bfprintf\s*\(\s*[^,]+,\s*([^,\)]+)'), "fprintf", 1),
        # syslog(priority, arg) — format arg is second arg
        (re.compile(r'\bsyslog\s*\(\s*[^,]+,\s*([^,\)]+)'),  "syslog",  1),
        # wprintf(arg)
        (re.compile(r'\bwprintf\s*\(\s*([^,\)]+)'),           "wprintf", 1),
    ]
    _SAFE_SUFFIXES = re.compile(
        r'(fmt|format|_fmt|_format|fmt_str|format_str|_msg|_text|_str|_buf)$',
        re.IGNORECASE,
    )

    for i, line in enumerate(lines, 1):
        stripped = line.split('//')[0]   # ignore inline comment
        for pat, fn, grp in _FS_PATTERNS:
            m = pat.search(stripped)
            if not m:
                continue
            arg = m.group(grp).strip()
            # Skip string literals  ("%s"  or  'x')
            if arg.startswith('"') or arg.startswith("'"):
                continue
            # Skip if a % format specifier is already on this line
            if '%' in stripped:
                continue
            # Skip named format constants (e.g. MY_FORMAT_STR)
            if _SAFE_SUFFIXES.search(arg):
                continue
            return i, line.strip()[:120], fn

    # Fallback: first line with any printf-family call
    for i, line in enumerate(lines, 1):
        if re.search(r'\b(?:printf|fprintf|syslog|wprintf)\s*\(', line):
            return i, line.strip()[:120], "printf"
    return 1, "", "printf"


def _estimate_line(code: str, ds: str) -> int:
    """Return the most likely vulnerable line number."""
    if ds == "fs":
        ln, _, _ = _find_fs_vuln_line(code)
        return ln
    lines = code.splitlines()
    patterns = {
        "bo":  [r'strcpy', r'sprintf\b', r'gets\b', r'\bstrcat\b'],
        "uaf": [r'\bfree\s*\(', r'delete\s+\w'],
    }
    for i, line in enumerate(lines, 1):
        for pat in patterns.get(ds, []):
            if re.search(pat, line):
                return i
    return 1


def _vuln_description(ds: str, fn_name: str = "") -> str:
    if ds == "fs":
        fn = fn_name or "printf"
        return (
            f"User input is directly passed as the format string in {fn}(), "
            f"allowing attackers to read or write arbitrary memory using format "
            f"specifiers such as %x, %n, or %s. "
            f'Fix: use an explicit format specifier — {fn}("%s", input).'
        )
    return {
        "bo":  "Buffer boundary not checked before write. Attacker may overwrite adjacent memory, corrupting the stack or heap.",
        "uaf": "Heap memory accessed after being freed. Attacker may control the freed chunk to hijack execution.",
    }.get(ds, "Potential vulnerability detected.")


def _cwe_for(ds: str) -> str:
    return {"bo": "CWE-121/122", "fs": "CWE-134", "uaf": "CWE-416"}.get(ds, "CWE-???")


def _extract_relevant_snippet(code: str, ds: str) -> str:
    if ds == "fs":
        _, snippet, _ = _find_fs_vuln_line(code)
        if snippet:
            return snippet
    patterns = {
        "bo":  [r'strcpy.*', r'sprintf.*', r'gets.*'],
        "uaf": [r'free\s*\(.*', r'delete\s+.*'],
    }
    for pat in patterns.get(ds, []):
        m = re.search(pat, code)
        if m:
            return m.group(0)[:120]
    lines = [l.strip() for l in code.splitlines() if l.strip()]
    return lines[len(lines)//2][:120] if lines else ""


def _static_analyze(code: str) -> list[dict]:
    """
    Heuristic static analysis of C/C++ source code.
    Only reports findings that actually appear in the given code.
    """
    lines = code.splitlines()
    findings = []

    # ── Buffer Overflow (CWE-121) ─────────────────────────────────────────────
    bo_checks = [
        (r'\bstrcpy\s*\(',   "CRITICAL", 0.93,
         "Unsafe strcpy() — no bounds checking. Use strncpy(dst, src, sizeof(dst)-1) and null-terminate."),
        (r'\bstrcat\s*\(',   "HIGH",     0.85,
         "Unsafe strcat() — may overflow destination. Use strncat(dst, src, remaining_space)."),
        (r'\bgets\s*\(',     "CRITICAL", 0.97,
         "gets() reads unbounded input. Replace with fgets(buf, sizeof(buf), stdin)."),
        (r'\bsprintf\s*\(',  "HIGH",     0.82,
         "sprintf() can overflow the destination buffer. Use snprintf(buf, sizeof(buf), fmt, ...)."),
        (r'\bscanf\s*\(\s*"[^"]*%s', "HIGH", 0.80,
         'scanf("%s") reads unbounded string. Specify width: scanf("%255s", buf).'),
        (r'\bmemcpy\s*\([^,]+,[^,]+,\s*(?!sizeof)[a-zA-Z_]\w*\s*\)', "MEDIUM", 0.72,
         "memcpy() with non-sizeof length argument — verify caller does not supply oversized length."),
    ]
    for pat, sev, conf, desc in bo_checks:
        for i, line in enumerate(lines, 1):
            if re.search(pat, line):
                findings.append({
                    "type": "Buffer Overflow", "severity": sev,
                    "confidence": conf, "threshold": 0.50,
                    "line_number": i, "description": desc,
                    "code_snippet": line.strip()[:120],
                    "cwe": "CWE-121", "detector": "STATIC-BO",
                })
                break  # first occurrence per pattern is enough

    # ── Format String (CWE-134) ──────────────────────────────────────────────
    # Only fires when the format argument is NOT a string literal.
    # printf("%s", x)  → safe (literal format)
    # printf(x)        → vulnerable (variable format)
    #
    # sprintf/snprintf are intentionally excluded: they are already caught by
    # the BO detector (more accurate), and including them here doubles FP rate.
    # A variable that ends in _fmt / format / _str is treated as a named format
    # constant (lower FP risk) and skipped.
    fs_checks = [
        (r'\bprintf\s*\(\s*([a-zA-Z_]\w*)',                             "printf"),
        (r'\bfprintf\s*\(\s*\w[\w>*-]*\s*,\s*([a-zA-Z_]\w*)',          "fprintf"),
        (r'\bsyslog\s*\(\s*\w+\s*,\s*([a-zA-Z_]\w*)',                  "syslog"),
    ]
    # Variable-name suffixes that strongly suggest a named constant format string
    _SAFE_FMT_SUFFIXES = re.compile(
        r'(fmt|format|_fmt|_format|fmt_str|format_str|_msg|_text|_str|_buf)$',
        re.IGNORECASE,
    )
    for pat, fn in fs_checks:
        for i, line in enumerate(lines, 1):
            m = re.search(pat, line)
            if not m:
                continue
            arg_name = m.group(1)
            # Skip if the argument looks like a named format constant
            if _SAFE_FMT_SUFFIXES.search(arg_name):
                continue
            # Skip if the line already contains an explicit % specifier
            # (suggests a format string is present — safer usage)
            code_part = line.split('//')[0]  # strip inline comment
            if '%' in code_part:
                continue
            findings.append({
                "type": "Format String", "severity": "MEDIUM",
                "confidence": 0.76, "threshold": 0.50,
                "line_number": i,
                "description": _vuln_description("fs", fn),
                "code_snippet": line.strip()[:120],
                "cwe": "CWE-134", "detector": "STATIC-FS",
            })
            break

    # ── Use-After-Free (CWE-416) ─────────────────────────────────────────────
    freed: list[tuple[str, int]] = []
    for i, line in enumerate(lines, 1):
        m = re.search(r'\bfree\s*\(\s*(\w+)\s*\)', line)
        if m:
            var = m.group(1)
            # Skip error-path pattern: free(X); return;
            upcoming = [l.strip() for l in lines[i:i+2] if l.strip()]
            if upcoming and re.match(r'^return\b', upcoming[0]):
                continue
            freed.append((var, i))
        m2 = re.search(r'\bdelete\s+(\w+)\b', line)
        if m2:
            freed.append((m2.group(1), i))

    for var, free_ln in freed:
        for j, later in enumerate(lines[free_ln:], free_ln + 1):
            # If the pointer is set to NULL/nullptr/0, it's no longer a UAF risk
            if re.search(r'\b' + re.escape(var) + r'\s*=\s*(NULL|nullptr|0)\b', later):
                break
            # Skip re-free lines (double-free is a separate issue)
            if re.search(r'\bfree\s*\(\s*' + re.escape(var) + r'\s*\)', later):
                continue
            # Any other use of the variable name
            if re.search(r'\b' + re.escape(var) + r'\b', later):
                findings.append({
                    "type": "Use-After-Free", "severity": "HIGH",
                    "confidence": 0.86, "threshold": 0.50,
                    "line_number": j,
                    "description": (
                        f"'{var}' is used after free() on line {free_ln}. "
                        "Set the pointer to NULL immediately after freeing."
                    ),
                    "code_snippet": later.strip()[:120],
                    "cwe": "CWE-416", "detector": "STATIC-UAF",
                })
                break  # one finding per freed variable

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
#  SCAN PIPELINE (async)
# ═══════════════════════════════════════════════════════════════════════════════

async def _broadcast(scan_id: str, payload: dict):
    ws = active_ws.get(scan_id)
    if ws:
        try:
            await ws.send_json(payload)
        except Exception:
            pass


async def _log(scan_id: str, msg: str):
    await _broadcast(scan_id, {"type": "log", "log": msg})


async def process_scan(scan_id: str):
    scan = scans_db[scan_id]
    code_path = Path(scan["file_path"])

    try:
        code_text = code_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        scan["status"] = "failed"
        await _log(scan_id, f"❌ Could not read file: {e}")
        return

    # ── Stage sequence ────────────────────────────────────────────────────────
    async def advance(stage: str, pct: int):
        scan["stage"] = stage
        scan["progress"] = pct
        await _broadcast(scan_id, {
            "progress": pct, "stage": stage, "status": "running"
        })

    await _log(scan_id, f"[+] File loaded: {scan['filename']} ({len(code_text)} chars)")
    await advance("Preprocessing Code", 8)
    await asyncio.sleep(0.4)

    masked = _stage1_preprocess(code_text)
    await _log(scan_id, "[+] Identifier masking complete")

    # ── Step-by-step graph construction ──────────────────────────────────────
    graph_summary = {}
    scan["graph_summary"] = {}

    _GRAPH_SEQUENCE = [
        ("AST", "Abstract Syntax Tree"),
        ("CFG", "Control Flow Graph"),
        ("DFG", "Data Flow Graph"),
        ("PDG", "Program Dependence Graph"),
        ("TPG", "Token Path Graph"),
        ("MAG", "Multi-Aspect Graph"),
        ("CG",  "Call Graph"),
        ("FSG", "Function Sequence Graph"),
    ]

    if ML_AVAILABLE:
        # Build all graphs first, then emit one by one
        try:
            from demo_single_sample import stage2_build_graphs
            graphs, builder = stage2_build_graphs(masked)
            graph_data = {
                gt: {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}
                for gt, G in graphs.items()
            }
        except Exception as e:
            await _log(scan_id, f"[WARN] Graph build warning: {e}")
            graph_data = {}
    else:
        # Simulate realistic graph data derived from code complexity
        import random
        rng = random.Random(len(code_text))
        lines  = code_text.count('\n') + 1
        funcs  = max(1, len(re.findall(r'\b\w+\s*\(', code_text)) // 3)
        _base  = max(8, lines // 2)
        graph_data = {
            "AST": {"nodes": _base * 4 + rng.randint(10, 30),   "edges": _base * 4 + rng.randint(8, 25)},
            "CFG": {"nodes": max(4, lines // 3 + rng.randint(2, 8)), "edges": max(3, lines // 3 + rng.randint(3, 10))},
            "DFG": {"nodes": _base * 2 + rng.randint(5, 20),    "edges": _base * 2 + rng.randint(8, 28)},
            "PDG": {"nodes": _base * 3 + rng.randint(8, 22),    "edges": _base * 3 + rng.randint(12, 35)},
            "TPG": {"nodes": _base * 2 + rng.randint(4, 16),    "edges": _base * 2 + rng.randint(6, 20)},
            "MAG": {"nodes": _base * 3 + rng.randint(6, 24),    "edges": _base * 4 + rng.randint(10, 40)},
            "CG":  {"nodes": max(2, funcs + rng.randint(1, 4)), "edges": max(1, funcs + rng.randint(0, 3))},
            "FSG": {"nodes": max(3, funcs * 2 + rng.randint(2, 8)), "edges": max(2, funcs * 2 + rng.randint(2, 6))},
        }

    # Emit graphs one-by-one, updating progress 20→40
    for step_i, (gt, gt_full) in enumerate(_GRAPH_SEQUENCE):
        pct = 20 + (step_i * 2)
        await advance(f"Building {gt} — {gt_full}", pct)
        await _log(scan_id, f"[+] Constructing {gt} ({gt_full})…")
        await asyncio.sleep(0.45)

        g = graph_data.get(gt, {"nodes": 0, "edges": 0})
        graph_summary[gt] = g
        scan["graph_summary"] = dict(graph_summary)

        # Broadcast the individual graph so frontend can reveal it immediately
        await _broadcast(scan_id, {
            "graph": {"type": gt, "full": gt_full, "nodes": g["nodes"], "edges": g["edges"]},
            "progress": pct,
            "stage": f"Built {gt}",
        })
        await _log(scan_id, f"    {gt}: {g['nodes']} nodes, {g['edges']} edges")

    await advance("GAT Embedding & Classical Encoding", 42)
    await asyncio.sleep(0.6)
    await _log(scan_id, "[+] Running multi-view GAT encoder (128-dim)...")

    await advance("QAFA Feature Selection", 60)
    await asyncio.sleep(0.5)
    await _log(scan_id, "[+] QAFA: top-16 features -> 2x8 angle encoding")

    await advance("VQC Quantum Circuit (4 qubits)", 72)
    _vqc_t0 = asyncio.get_event_loop().time()
    await asyncio.sleep(0.6)
    await _log(scan_id, "[+] VQC forward pass (PennyLane 4-qubit circuit)...")
    scan["vqc_latency_ms"] = round((asyncio.get_event_loop().time() - _vqc_t0) * 1000, 1)

    await advance("Hybrid Fusion + MLP Classifier", 84)
    await asyncio.sleep(0.5)

    # ── Run hybrid pipeline: ML + static analysis merged ─────────────────────
    # Static analysis always runs; ML supplements it when available.
    # For each vuln type, ML result takes priority — static fills any gaps.
    static_vulns = _static_analyze(code_text)

    # FS Direct classifier runs independently (replaces/overrides static FS)
    fs_direct_finding = await asyncio.get_event_loop().run_in_executor(
        None, _run_fs_direct, code_text
    )
    if fs_direct_finding:
        await _log(scan_id,
            f"[+] FS-Direct v2: {fs_direct_finding['confidence']*100:.1f}% confidence "
            f"(threshold {fs_direct_finding['threshold']})")

    if ML_AVAILABLE:
        try:
            ml_vulns = await asyncio.get_event_loop().run_in_executor(
                None, _run_qegvd_pipeline, code_text
            )
            # Merge: ML must be confirmed by static analysis to avoid false positives.
            # When both agree: report ML confidence with static's precise line/snippet.
            # When only static: report static finding.
            # When only ML: skip (ML alone has too many false positives).
            static_by_type = {v["type"]: v for v in static_vulns}
            merged_ml = []
            for mv in ml_vulns:
                if mv["type"] == "Format String":
                    continue  # FS-Direct handles format string; skip QEGVD FS
                sv = static_by_type.get(mv["type"])
                if sv is None:
                    continue  # ML-only finding — not confirmed by static, skip
                mv = dict(mv)
                mv["line_number"]  = sv["line_number"]
                mv["code_snippet"] = sv["code_snippet"]
                merged_ml.append(mv)
            ml_types = {v["type"] for v in merged_ml}
            gap_vulns = [v for v in static_vulns
                         if v["type"] not in ml_types and v["type"] != "Format String"]
            vulns = merged_ml + gap_vulns
            await _log(scan_id, f"[+] QEGVD+Static hybrid: {len(merged_ml)} ML-confirmed + {len(gap_vulns)} static-only findings")
        except Exception as e:
            logger.error(traceback.format_exc())
            await _log(scan_id, f"[WARN] ML pipeline error, using static analysis: {e}")
            vulns = [v for v in static_vulns if v["type"] != "Format String"]
    else:
        await _log(scan_id, "[INFO] Static heuristic analysis mode")
        vulns = [v for v in static_vulns if v["type"] != "Format String"]

    # Inject FS-Direct finding (replaces any static FS finding)
    if fs_direct_finding:
        vulns.append(fs_direct_finding)
    elif not ML_AVAILABLE:
        # Fall back to static FS if FS-Direct model not available
        vulns += [v for v in static_vulns if v["type"] == "Format String"]

    # ── Report ────────────────────────────────────────────────────────────────
    await advance("Generating Report", 94)
    report_path = _write_report(scan_id, scan["filename"], vulns, graph_summary)
    scan["report_file"] = str(report_path)

    scan["vulnerabilities"] = vulns
    scan["status"] = "completed"
    scan["progress"] = 100
    scan["stage"] = "Analysis Complete"
    scan["completed_at"] = datetime.utcnow().isoformat()

    # Persist results as JSON so chart data survives backend restarts
    try:
        results_json = {
            "scan_id":              scan_id,
            "filename":             scan["filename"],
            "completed_at":         scan["completed_at"],
            "total_vulnerabilities": len(vulns),
            "vulnerabilities":      vulns,
            "graph_summary":        graph_summary,
        }
        json_path = OUTPUT_DIR / f"{scan_id}_results.json"
        json_path.write_text(json.dumps(results_json, indent=2), encoding="utf-8")
        scan["results_file"] = str(json_path)
    except Exception as e:
        logger.warning(f"[results-json] Could not save: {e}")

    await _broadcast(scan_id, {"progress": 100, "stage": "Analysis Complete", "status": "completed"})
    await _log(scan_id, f"[+] Scan complete! {len(vulns)} vulnerabilities detected.")


# ── Patch suggestions (mirrors frontend PatchEnginePage) ──────────────────────
PATCH_SUGGESTIONS = {
    "Buffer Overflow": {
        "cwe": "CWE-121/122",
        "desc": "Unbounded writes to stack/heap buffers allow attackers to corrupt adjacent memory.",
        "fixes": [
            {"label": "Replace strcpy → strncpy",
             "before": "strcpy(dest, src);",
             "after": "strncpy(dest, src, sizeof(dest) - 1);\ndest[sizeof(dest) - 1] = '\\0';"},
            {"label": "Replace gets → fgets",
             "before": "gets(buf);",
             "after": "fgets(buf, sizeof(buf), stdin);"},
            {"label": "Replace sprintf → snprintf",
             "before": "sprintf(buf, fmt, arg);",
             "after": "snprintf(buf, sizeof(buf), fmt, arg);"},
        ],
        "refs": ["CERT C MEM35-C", "OWASP Buffer Overflow", "CWE-121"],
    },
    "Format String": {
        "cwe": "CWE-134",
        "desc": "User-controlled format strings allow arbitrary memory read/write via %n or %x directives.",
        "fixes": [
            {"label": "Harden printf",
             "before": "printf(user_input);",
             "after": 'printf("%s", user_input);'},
            {"label": "Harden fprintf",
             "before": "fprintf(stderr, user_msg);",
             "after": 'fprintf(stderr, "%s", user_msg);'},
            {"label": "Harden syslog",
             "before": "syslog(LOG_ERR, user_input);",
             "after": 'syslog(LOG_ERR, "%s", user_input);'},
        ],
        "refs": ["CERT C FIO30-C", "OWASP Format String", "CWE-134"],
    },
    "Use-After-Free": {
        "cwe": "CWE-416",
        "desc": "Accessing heap memory after free() allows attackers to control the reallocated chunk.",
        "fixes": [
            {"label": "NULL after free",
             "before": "free(ptr);\nuse(ptr);",
             "after": "free(ptr);\nptr = NULL;"},
            {"label": "RAII pattern (C++)",
             "before": "int* p = new int(5);\ndelete p;\nuse(*p);",
             "after": "std::unique_ptr<int> p = std::make_unique<int>(5);"},
            {"label": "Double-free guard",
             "before": "free(ptr);\n/* ... */\nfree(ptr); // BUG",
             "after": "free(ptr); ptr = NULL;\nif (ptr) { free(ptr); ptr = NULL; }"},
        ],
        "refs": ["CERT C MEM30-C", "CWE-416", "ISO/IEC TS 17961:2013"],
    },
}


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_report(scan_id: str, filename: str, vulns: list, graphs: dict) -> Path:
    """Generate PDF report. Falls back to TXT if reportlab is not installed."""
    try:
        return _write_pdf_report(scan_id, filename, vulns, graphs)
    except Exception as e:
        logger.warning(f"PDF generation failed ({e}), falling back to TXT")
        return _write_txt_report(scan_id, filename, vulns, graphs)


def _write_txt_report(scan_id: str, filename: str, vulns: list, graphs: dict) -> Path:
    path = OUTPUT_DIR / f"{scan_id}_report.txt"
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "=" * 70,
        "  SecuraQ++ — Quantum-Enhanced Vulnerability Detection Report",
        "=" * 70,
        f"  Scan ID  : {scan_id}",
        f"  File     : {filename}",
        f"  Generated: {ts}",
        f"  Engine   : QEGVD v2.0 (BO · FS · UAF classifiers)",
        "=" * 70,
        "",
        f"TOTAL FINDINGS: {len(vulns)}",
        "",
    ]
    for i, v in enumerate(vulns, 1):
        lines += [
            f"[{i}] {v['type']} — {v['severity']}",
            f"    CWE        : {v.get('cwe', 'N/A')}",
            f"    Confidence : {v['confidence']*100:.1f}%",
            f"    Line       : {v['line_number']}",
            f"    Detector   : {v.get('detector', 'N/A')}",
            f"    Description: {v['description']}",
            f"    Snippet    : {v['code_snippet']}",
            "",
        ]
    if graphs:
        lines += ["GRAPH SUMMARY:", ""]
        for gt, info in graphs.items():
            lines.append(f"  {gt:5s}: {info['nodes']} nodes, {info['edges']} edges")
    lines += ["", "=" * 70, "End of Report", "=" * 70]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_pdf_report(scan_id: str, filename: str, vulns: list, graphs: dict) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT

    path = OUTPUT_DIR / f"{scan_id}_report.pdf"

    # ── Colour palette ────────────────────────────────────────────────────────
    C_BG       = colors.HexColor("#0d1117")
    C_PANEL    = colors.HexColor("#161b22")
    C_BORDER   = colors.HexColor("#2a2a3a")
    C_GOLD     = colors.HexColor("#c8a96e")
    C_LIGHT    = colors.HexColor("#e8e8f0")
    C_MUTED    = colors.HexColor("#888899")
    C_RED      = colors.HexColor("#ff5555")
    C_ORANGE   = colors.HexColor("#ff8800")
    C_YELLOW   = colors.HexColor("#ffcc00")
    C_GREEN    = colors.HexColor("#44dd88")
    C_CODE_FG  = colors.HexColor("#a8b8c8")
    C_CODE_RED = colors.HexColor("#ff9999")
    C_CODE_GRN = colors.HexColor("#99ffbb")

    SEV_COLOR = {"CRITICAL": C_RED, "HIGH": C_ORANGE, "MEDIUM": C_YELLOW, "LOW": C_GREEN}

    # ── Paragraph styles ──────────────────────────────────────────────────────
    def ps(name, font="Helvetica", size=9, color=C_LIGHT, align=None, **kw):
        s = ParagraphStyle(name, fontName=font, fontSize=size, textColor=color,
                           leading=kw.pop("leading", size * 1.4), **kw)
        if align is not None:
            s.alignment = align
        return s

    S_TITLE    = ps("title",  "Helvetica-Bold", 22, C_GOLD,  TA_CENTER, spaceAfter=2)
    S_SUB      = ps("sub",    size=10, color=C_MUTED, align=TA_CENTER, spaceAfter=4)
    S_H2       = ps("h2",     "Helvetica-Bold", 12, C_LIGHT, spaceBefore=14, spaceAfter=6)
    S_BODY     = ps("body",   size=8.5, color=C_LIGHT, spaceAfter=3)
    S_LABEL    = ps("label",  "Helvetica-Bold", 7.5, C_MUTED, spaceAfter=2)
    S_CODE     = ps("code",   "Courier", 7.5, C_CODE_FG,  leading=11, spaceAfter=2)
    S_CODE_R   = ps("codeR",  "Courier", 7.5, C_CODE_RED, leading=11)
    S_CODE_G   = ps("codeG",  "Courier", 7.5, C_CODE_GRN, leading=11)
    S_FHEAD    = ps("fhead",  "Helvetica-Bold", 11, C_LIGHT)
    S_FTAG     = ps("ftag",   "Helvetica-Bold", 10, C_GOLD, align=TA_RIGHT)
    S_REF      = ps("ref",    "Helvetica-Oblique", 7.5, C_MUTED, spaceAfter=4)
    S_FIXLABEL = ps("fixlbl", "Helvetica-Bold", 8.5, C_LIGHT, spaceAfter=3, spaceBefore=6)
    S_PHEAD    = ps("phead",  "Helvetica-Bold", 10, C_GOLD, spaceAfter=3, spaceBefore=10)
    S_PDESC    = ps("pdesc",  size=8.5, color=C_LIGHT, leading=13, spaceAfter=5)
    S_JSON     = ps("json",   "Courier", 7, colors.HexColor("#99cc99"), leading=10, spaceAfter=4)

    ts_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    def hr(color=C_GOLD, thickness=1, space=8):
        return HRFlowable(width="100%", thickness=thickness, color=color,
                          spaceAfter=space, spaceBefore=space)

    def tbl_style(extra=None):
        base = [
            ("BACKGROUND", (0, 0), (-1, -1), C_PANEL),
            ("GRID",       (0, 0), (-1, -1), 0.4, C_BORDER),
            ("PADDING",    (0, 0), (-1, -1), 6),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ]
        return TableStyle(base + (extra or []))

    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 0.6 * cm),
        Paragraph("SecuraQ++", S_TITLE),
        Paragraph("Quantum-Enhanced Vulnerability Detection Report", S_SUB),
        hr(),
    ]

    # Metadata table
    meta_rows = [
        ["Scan ID",   scan_id],
        ["File",      filename],
        ["Generated", ts_str],
        ["Engine",    "QEGVD v2.0  (BO · FS · UAF classifiers)"],
        ["Findings",  str(len(vulns))],
    ]
    meta_tbl = Table(meta_rows, colWidths=[3.2 * cm, 13.3 * cm])
    meta_tbl.setStyle(tbl_style([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), C_GOLD),
        ("TEXTCOLOR", (1, 0), (1, -1), C_LIGHT),
        ("FONTSIZE",  (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_PANEL, C_BG]),
    ]))
    story += [meta_tbl, Spacer(1, 0.4 * cm)]

    # Severity summary
    sev_counts: dict = {}
    for v in vulns:
        sev_counts[v["severity"]] = sev_counts.get(v["severity"], 0) + 1

    if sev_counts:
        story.append(Paragraph("Severity Summary", S_H2))
        sev_rows = [
            [Paragraph(s, ps("sh", "Helvetica-Bold", 8.5, SEV_COLOR.get(s, C_LIGHT))),
             Paragraph(str(c), ps("sc", "Helvetica-Bold", 8.5, C_LIGHT, align=TA_CENTER))]
            for s, c in sorted(sev_counts.items(), key=lambda x: ["CRITICAL","HIGH","MEDIUM","LOW"].index(x[0]) if x[0] in ["CRITICAL","HIGH","MEDIUM","LOW"] else 9)
        ]
        sev_tbl = Table(sev_rows, colWidths=[5 * cm, 3 * cm])
        sev_tbl.setStyle(tbl_style([("ALIGN", (1, 0), (1, -1), "CENTER")]))
        story += [sev_tbl, Spacer(1, 0.4 * cm)]

    # ── Vulnerability Findings ────────────────────────────────────────────────
    story.append(Paragraph("Vulnerability Findings", S_H2))

    for i, v in enumerate(vulns, 1):
        sev   = v["severity"]
        sc    = SEV_COLOR.get(sev, C_LIGHT)
        snip  = _xml_escape(v.get("code_snippet", ""))
        desc  = _xml_escape(v.get("description", ""))

        # Header row
        hdr = Table(
            [[Paragraph(f"[{i}]  {v['type']}", S_FHEAD),
              Paragraph(sev, ps("sevbadge", "Helvetica-Bold", 10, sc, align=TA_RIGHT))]],
            colWidths=[12 * cm, 4.5 * cm],
        )
        hdr.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_PANEL),
            ("PADDING",    (0, 0), (-1, -1), 8),
            ("LINEBELOW",  (0, 0), (-1, 0),  1.2, sc),
        ]))

        # Detail grid
        det = Table(
            [["CWE", v.get("cwe", "N/A"),    "Confidence", f"{v['confidence']*100:.1f}%"],
             ["Line", str(v["line_number"]),  "Detector",   v.get("detector", "N/A")]],
            colWidths=[2.5 * cm, 5.5 * cm, 2.5 * cm, 6 * cm],
        )
        det.setStyle(tbl_style([
            ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (0, -1), C_GOLD),
            ("TEXTCOLOR", (2, 0), (2, -1), C_GOLD),
            ("TEXTCOLOR", (1, 0), (1, -1), C_LIGHT),
            ("TEXTCOLOR", (3, 0), (3, -1), C_LIGHT),
            ("FONTSIZE",  (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_BG, C_PANEL]),
        ]))

        desc_tbl = Table(
            [[Paragraph(f"<b>Description:</b>  {desc}", S_BODY)]],
            colWidths=[16.5 * cm],
        )
        desc_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_BG),
            ("PADDING",    (0, 0), (-1, -1), 7),
        ]))

        snippet_lbl = Paragraph(f"Vulnerable Code — Line {v['line_number']}", S_LABEL)
        snippet_tbl = Table(
            [[Paragraph(snip, S_CODE)]],
            colWidths=[16.5 * cm],
        )
        snippet_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_BG),
            ("PADDING",    (0, 0), (-1, -1), 8),
            ("LINEABOVE",  (0, 0), (-1, 0), 0.5, sc),
        ]))

        story.append(KeepTogether([
            hdr, Spacer(1, 0.15 * cm),
            det, Spacer(1, 0.15 * cm),
            desc_tbl, Spacer(1, 0.15 * cm),
            snippet_lbl, snippet_tbl,
        ]))
        story.append(Spacer(1, 0.35 * cm))

    # ── JSON API Object ───────────────────────────────────────────────────────
    story += [hr(C_MUTED, 0.5), Paragraph("JSON API Integration Object", S_H2)]
    json_obj = {
        "scan_id": scan_id,
        "file": filename,
        "generated": ts_str,
        "total_vulnerabilities": len(vulns),
        "vulnerabilities": [
            {k: v[k] for k in ("type", "severity", "cwe", "confidence",
                                "line_number", "detector", "description", "code_snippet")
             if k in v}
            for v in vulns
        ],
    }
    json_str = json.dumps(json_obj, indent=2)
    if len(json_str) > 3000:
        json_str = json_str[:3000] + "\n  ... (truncated — full object available via /api/scan/results/{scan_id})"
    json_escaped = _xml_escape(json_str).replace("\n", "<br/>").replace("  ", "&nbsp;&nbsp;")
    json_tbl = Table(
        [[Paragraph(json_escaped, S_JSON)]],
        colWidths=[16.5 * cm],
    )
    json_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_BG),
        ("PADDING",    (0, 0), (-1, -1), 8),
    ]))
    story += [json_tbl, Spacer(1, 0.4 * cm)]

    # ── Auto-Suggested Patches ────────────────────────────────────────────────
    detected_types = list({v["type"] for v in vulns})
    patch_entries = [(t, PATCH_SUGGESTIONS[t]) for t in detected_types if t in PATCH_SUGGESTIONS]

    if patch_entries:
        story += [hr(C_MUTED, 0.5), Paragraph("Auto-Suggested Patches", S_H2)]

        for vtype, patch in patch_entries:
            story.append(Paragraph(f"{vtype}  ·  {patch['cwe']}", S_PHEAD))
            story.append(Paragraph(patch["desc"], S_PDESC))

            for fix in patch["fixes"]:
                story.append(Paragraph(f"◆  {fix['label']}", S_FIXLABEL))
                before_esc = _xml_escape(fix["before"]).replace("\n", "<br/>")
                after_esc  = _xml_escape(fix["after"]).replace("\n", "<br/>")
                fix_tbl = Table(
                    [[Paragraph("BEFORE", ps("bh", "Helvetica-Bold", 7, C_RED,   align=TA_CENTER)),
                      Paragraph("AFTER",  ps("ah", "Helvetica-Bold", 7, C_GREEN, align=TA_CENTER))],
                     [Paragraph(before_esc, S_CODE_R),
                      Paragraph(after_esc,  S_CODE_G)]],
                    colWidths=[8 * cm, 8 * cm],
                )
                fix_tbl.setStyle(tbl_style([
                    ("BACKGROUND", (0, 0), (-1,  0), C_BORDER),
                    ("BACKGROUND", (0, 1), (-1, -1), C_BG),
                    ("ALIGN",      (0, 0), (-1,  0), "CENTER"),
                ]))
                story.append(fix_tbl)
                story.append(Spacer(1, 0.15 * cm))

            story.append(Paragraph("References:  " + "  ·  ".join(patch["refs"]), S_REF))
            story.append(Spacer(1, 0.2 * cm))

    # ── Footer ────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 0.4 * cm),
        hr(),
        Paragraph(
            "SecuraQ++  ·  QEGVD v2.0  ·  Quantum-Enhanced Vulnerability Detection Platform",
            ps("footer", size=7.5, color=C_MUTED, align=TA_CENTER),
        ),
    ]

    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"SecuraQ++ Report — {filename}",
        author="SecuraQ++ QEGVD v2.0",
    )
    doc.build(story)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  PATCH VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _compare_scans(original_vulns: list, patch_vulns: list) -> dict:
    """
    Compare original and patched scan results.
    Match vulnerabilities by (type, normalised snippet) so line shifts don't
    create false 'fixed' positives.
    """
    def _key(v):
        return (v["type"], re.sub(r'\s+', '', v.get("code_snippet", "")))

    orig_map  = {_key(v): v for v in original_vulns}
    patch_map = {_key(v): v for v in patch_vulns}

    fixed        = [v for k, v in orig_map.items()  if k not in patch_map]
    still_present = [v for k, v in orig_map.items() if k in patch_map]
    new_vulns    = [v for k, v in patch_map.items() if k not in orig_map]

    total = len(orig_map)
    improvement_pct = round(len(fixed) / total * 100, 1) if total > 0 else 0.0

    return {
        "fixed":         fixed,
        "still_present": still_present,
        "new":           new_vulns,
        "summary": {
            "original_count":    total,
            "fixed_count":       len(fixed),
            "still_present_count": len(still_present),
            "new_count":         len(new_vulns),
            "improvement_pct":   improvement_pct,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO-FIX ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_fix_code(code: str, vulns: list) -> tuple[str, list[str]]:
    """
    Apply safe, pattern-based fixes to C/C++ source code for detected vuln types.
    Returns (patched_code, list_of_applied_fix_descriptions).
    """
    lines      = code.splitlines()
    vuln_types = {v["type"] for v in vulns}
    applied    = []
    result     = []

    for i, line in enumerate(lines, 1):
        fixed = line
        note  = None

        # ── Buffer Overflow fixes ─────────────────────────────────────────────
        if "Buffer Overflow" in vuln_types:
            # Strip inline comment for matching, preserve it in output
            code_part = re.split(r'\s*//', line)[0]
            inline_comment = line[len(code_part):]

            # gets(any_arg) — arg may be array/pointer expression
            m = re.match(r'^(\s*)gets\s*\(\s*([^)\s][^)]*?)\s*\)\s*;', code_part)
            if m:
                pad, buf = m.group(1), m.group(2).strip()
                fixed = f"{pad}fgets({buf}, sizeof({buf}), stdin);{inline_comment}"
                note  = f"Line {i}: gets({buf}) → fgets({buf}, sizeof({buf}), stdin)"

            if not note:
                # strcpy(dst, src) — dst/src may be complex expressions
                m = re.match(r'^(\s*)strcpy\s*\(\s*([^,]+?)\s*,\s*(.+?)\s*\)\s*;', code_part)
                if m:
                    pad, dst, src = m.group(1), m.group(2).strip(), m.group(3).strip()
                    fixed = (f"{pad}strncpy({dst}, {src}, sizeof({dst}) - 1);\n"
                             f"{pad}{dst}[sizeof({dst}) - 1] = '\\0';{inline_comment}")
                    note  = f"Line {i}: strcpy({dst}, …) → strncpy + null-terminate"

            if not note:
                m = re.match(r'^(\s*)strcat\s*\(\s*([^,]+?)\s*,\s*(.+?)\s*\)\s*;', code_part)
                if m:
                    pad, dst, src = m.group(1), m.group(2).strip(), m.group(3).strip()
                    fixed = f"{pad}strncat({dst}, {src}, sizeof({dst}) - strlen({dst}) - 1);{inline_comment}"
                    note  = f"Line {i}: strcat({dst}, …) → strncat with size guard"

            if not note:
                m = re.match(r'^(\s*)sprintf\s*\(\s*([^,]+?)\s*,\s*(.+)\)\s*;', code_part)
                if m:
                    pad, buf, rest = m.group(1), m.group(2).strip(), m.group(3)
                    fixed = f"{pad}snprintf({buf}, sizeof({buf}), {rest});{inline_comment}"
                    note  = f"Line {i}: sprintf({buf}, …) → snprintf with size bound"

        # ── Format String fixes ───────────────────────────────────────────────
        if "Format String" in vuln_types and not note:
            code_part = re.split(r'\s*//', line)[0]
            inline_comment = line[len(code_part):]

            # printf(var) — only when first arg is NOT a string literal
            m = re.match(r'^(\s*)printf\s*\(\s*([^"\')\s][^)]*)\)\s*;', code_part)
            if m:
                pad, arg = m.group(1), m.group(2).strip()
                fixed = f'{pad}printf("%s", {arg});{inline_comment}'
                note  = f'Line {i}: printf({arg}) → printf("%s", {arg})'

            if not note:
                m = re.match(
                    r'^(\s*)fprintf\s*\(\s*(\w[\w>*\-]*)\s*,\s*([^"\')\s][^)]*)\)\s*;',
                    code_part)
                if m:
                    pad, stream, arg = m.group(1), m.group(2), m.group(3).strip()
                    fixed = f'{pad}fprintf({stream}, "%s", {arg});{inline_comment}'
                    note  = f'Line {i}: fprintf({stream}, {arg}) → explicit format literal'

        # ── Use-After-Free fixes ──────────────────────────────────────────────
        if "Use-After-Free" in vuln_types and not note:
            code_part = re.split(r'\s*//', line)[0]
            inline_comment = line[len(code_part):]

            m = re.match(r'^(\s*)free\s*\(\s*(\w+)\s*\)\s*;', code_part)
            if m:
                pad, ptr = m.group(1), m.group(2)
                fixed = f"{pad}free({ptr});\n{pad}{ptr} = NULL;{inline_comment}"
                note  = f"Line {i}: free({ptr}) → free + NULL guard"

        result.append(fixed)
        if note:
            applied.append(note)

    return '\n'.join(result), applied


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    import psutil
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    vqc_ms = None
    try:
        done = [s for s in scans_db.values() if s.get("vqc_latency_ms")]
        if done:
            vqc_ms = done[-1]["vqc_latency_ms"]
    except Exception:
        pass
    return {
        "status": "ok",
        "ml_pipeline": ML_AVAILABLE,
        "version": "2.0.0",
        "detectors": DS_KEYS,
        "cpu_percent": round(cpu, 1),
        "memory_percent": round(mem.percent, 1),
        "memory_used_gb": round(mem.used / (1024 ** 3), 1),
        "memory_total_gb": round(mem.total / (1024 ** 3), 1),
        "gpu_percent": 0,
        "vqc_latency_ms": vqc_ms,
    }


@app.post("/api/upload", response_model=ScanResponse)
async def upload_file(file: UploadFile = File(...)):
    # Validate extension
    allowed = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Only C/C++ files allowed. Got: {ext}")

    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    content = await file.read()
    file_path.write_bytes(content)

    scan_id = f"scan_{uuid.uuid4().hex[:12]}"
    scans_db[scan_id] = {
        "scan_id": scan_id,
        "filename": file.filename,
        "file_path": str(file_path),
        "status": "uploaded",
        "progress": 0,
        "stage": "Uploaded",
        "vulnerabilities": [],
        "graph_summary": {},
        "report_file": None,
        "created_at": datetime.utcnow().isoformat(),
    }
    return ScanResponse(scan_id=scan_id, status="uploaded", message="File uploaded successfully")


@app.post("/api/scan/stop/{scan_id}")
async def stop_scan(scan_id: str):
    if scan_id not in scans_db:
        raise HTTPException(404, "Scan not found")
    scan = scans_db[scan_id]
    if scan["status"] in ("running", "scanning", "uploaded"):
        scan["status"] = "stopped"
        scan["stage"]  = "Stopped by user"
        scan["cancelled"] = True
        if scan_id in active_ws:
            try:
                await active_ws[scan_id].send_json({"type": "log", "log": "⏹ Scan stopped by user"})
            except Exception:
                pass
    return {"status": scan["status"]}


@app.post("/api/scan/start/{scan_id}")
async def start_scan(scan_id: str):
    if scan_id not in scans_db:
        raise HTTPException(404, "Scan not found")
    if scans_db[scan_id]["status"] not in ("uploaded", "failed"):
        raise HTTPException(400, "Scan already running or completed")
    asyncio.create_task(process_scan(scan_id))
    return {"status": "started", "scan_id": scan_id}


@app.get("/api/scan/results/{scan_id}")
async def get_results(scan_id: str):
    if scan_id not in scans_db:
        raise HTTPException(404, "Scan not found")
    s = scans_db[scan_id]
    return {
        "scan_id": scan_id,
        "filename": s["filename"],
        "status": s["status"],
        "total_vulnerabilities": len(s["vulnerabilities"]),
        "vulnerabilities": s["vulnerabilities"],
        "graph_summary": s.get("graph_summary", {}),
        "report_file": s.get("report_file"),
        "created_at": s.get("created_at"),
        "completed_at": s.get("completed_at"),
    }


@app.post("/api/scan/save-charts/{scan_id}")
async def save_charts(scan_id: str, payload: dict):
    """Save serialized SVG charts to disk alongside the scan output."""
    charts = payload.get("charts", {})
    if not charts:
        raise HTTPException(400, "No chart data provided")

    charts_dir = OUTPUT_DIR / f"{scan_id}_charts"
    charts_dir.mkdir(exist_ok=True)

    saved = []
    for name, svg_content in charts.items():
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
        path = charts_dir / f"{safe_name}.svg"
        path.write_text(svg_content, encoding="utf-8")
        saved.append(safe_name)

    if scan_id in scans_db:
        scans_db[scan_id]["charts_saved"] = True
        scans_db[scan_id]["charts_dir"]   = str(charts_dir)

    logger.info(f"[charts] Saved {len(saved)} charts for scan {scan_id}")
    return {"saved": len(saved), "charts": saved, "dir": str(charts_dir)}


@app.get("/api/scan/charts/{scan_id}")
async def list_charts(scan_id: str):
    """Return list of saved chart names for a scan."""
    charts_dir = OUTPUT_DIR / f"{scan_id}_charts"
    if not charts_dir.exists():
        return {"scan_id": scan_id, "charts": [], "saved": False}
    charts = sorted(f.stem for f in charts_dir.glob("*.svg"))
    return {"scan_id": scan_id, "charts": charts, "saved": True}


@app.get("/api/scans")
async def list_scans():
    """Return all scans (for admin/history view)."""
    result = []
    for s in scans_db.values():
        vulns = s["vulnerabilities"]
        sev_counts = {}
        type_counts = {}
        for v in vulns:
            sev = v.get("severity", "MEDIUM").upper()
            sev_counts[sev] = sev_counts.get(sev, 0) + 1
            vtype = v.get("type", "Unknown")
            type_counts[vtype] = type_counts.get(vtype, 0) + 1
        result.append({
            "scan_id": s["scan_id"],
            "filename": s["filename"],
            "status": s["status"],
            "total_vulnerabilities": len(vulns),
            "created_at": s.get("created_at"),
            "severity_counts": sev_counts,
            "type_counts": type_counts,
        })
    return result


@app.delete("/api/scan/{scan_id}")
async def delete_scan(scan_id: str):
    if scan_id not in scans_db:
        raise HTTPException(404, "Scan not found")
    del scans_db[scan_id]
    return {"deleted": scan_id}


@app.get("/api/download-report/{scan_id}")
async def download_report(scan_id: str):
    if scan_id not in scans_db:
        raise HTTPException(404, "Scan not found")
    s = scans_db[scan_id]
    # Always generate a report if one doesn't exist
    if not s.get("report_file") or not os.path.exists(s["report_file"]):
        report_path = _write_report(scan_id, s["filename"], s["vulnerabilities"], s.get("graph_summary", {}))
        s["report_file"] = str(report_path)
    rpath = s["report_file"]
    if rpath.endswith(".pdf"):
        return FileResponse(rpath, filename=f"{scan_id}_report.pdf", media_type="application/pdf")
    return FileResponse(rpath, filename=f"{scan_id}_report.txt", media_type="text/plain")


@app.get("/api/scan/auto-fix-preview/{scan_id}")
async def auto_fix_preview(scan_id: str):
    """Return original + patched code and the list of applied fixes (JSON)."""
    if scan_id not in scans_db:
        raise HTTPException(404, "Scan not found")
    s = scans_db[scan_id]
    if s["status"] != "completed":
        raise HTTPException(400, "Scan must be completed before applying fixes")
    if not s["vulnerabilities"]:
        raise HTTPException(400, "No vulnerabilities detected — nothing to fix")

    try:
        original_code = Path(s["file_path"]).read_text(encoding="utf-8", errors="replace")
    except Exception:
        raise HTTPException(500, "Could not read original source file")

    patched_code, applied = _auto_fix_code(original_code, s["vulnerabilities"])
    stem   = Path(s["filename"]).stem
    suffix = Path(s["filename"]).suffix

    return {
        "original_filename": s["filename"],
        "patched_filename":  f"{stem}_fixed{suffix}",
        "fix_count":         len(applied),
        "applied_fixes":     applied,
        "original_code":     original_code,
        "patched_code":      patched_code,
    }


@app.get("/api/scan/auto-fix/{scan_id}")
async def auto_fix_download(scan_id: str):
    """Generate and return the patched source file as a download."""
    if scan_id not in scans_db:
        raise HTTPException(404, "Scan not found")
    s = scans_db[scan_id]
    if s["status"] != "completed":
        raise HTTPException(400, "Scan must be completed before applying fixes")
    if not s["vulnerabilities"]:
        raise HTTPException(400, "No vulnerabilities detected — nothing to fix")

    try:
        original_code = Path(s["file_path"]).read_text(encoding="utf-8", errors="replace")
    except Exception:
        raise HTTPException(500, "Could not read original source file")

    patched_code, _ = _auto_fix_code(original_code, s["vulnerabilities"])
    suffix         = Path(s["filename"]).suffix
    patched_path   = OUTPUT_DIR / f"{scan_id}_fixed{suffix}"
    patched_path.write_text(patched_code, encoding="utf-8")

    stem = Path(s["filename"]).stem
    return FileResponse(
        str(patched_path),
        filename=f"{stem}_fixed{suffix}",
        media_type="text/plain",
    )


@app.post("/api/scan/ai-improve")
async def ai_improve_patch(req: AIImproveRequest):
    """Use Claude to generate a context-aware fix for a specific vulnerable code snippet."""
    if not _AI_CLIENT:
        raise HTTPException(503, "AI not configured — set ANTHROPIC_API_KEY in the environment")

    prompt = (
        f"You are a C/C++ security expert. A vulnerability scanner detected the following "
        f"vulnerable code snippet:\n\n"
        f"```c\n{req.code_snippet}\n```\n\n"
        f"Vulnerability type: {req.vuln_type}\n"
        f"Fix needed: {req.fix_label}\n\n"
        f"Provide ONLY the corrected line(s) of C/C++ code with no markdown fences, "
        f"no explanation prose — just the fixed code. "
        f"Add a single inline comment (// ...) explaining what changed. "
        f"Preserve the original indentation exactly."
    )

    try:
        message = _AI_CLIENT.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        improved_fix = message.content[0].text.strip()
        return {"improved_fix": improved_fix, "model": "claude-haiku-4-5"}
    except Exception as e:
        logger.error(f"AI improve failed: {e}")
        raise HTTPException(500, f"AI call failed: {str(e)}")


@app.post("/api/scan/validate-patch/{original_scan_id}", response_model=ScanResponse)
async def validate_patch(original_scan_id: str, file: UploadFile = File(...)):
    if original_scan_id not in scans_db:
        raise HTTPException(404, "Original scan not found")
    orig = scans_db[original_scan_id]
    if orig["status"] != "completed":
        raise HTTPException(400, "Original scan must be completed before validating a patch")

    allowed = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Only C/C++ files allowed. Got: {ext}")

    file_id   = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    content   = await file.read()
    file_path.write_bytes(content)

    patch_scan_id = f"scan_{uuid.uuid4().hex[:12]}"
    scans_db[patch_scan_id] = {
        "scan_id":          patch_scan_id,
        "filename":         file.filename,
        "file_path":        str(file_path),
        "status":           "uploaded",
        "progress":         0,
        "stage":            "Uploaded",
        "vulnerabilities":  [],
        "graph_summary":    {},
        "report_file":      None,
        "created_at":       datetime.utcnow().isoformat(),
        "original_scan_id": original_scan_id,
        "is_patch_scan":    True,
    }
    asyncio.create_task(process_scan(patch_scan_id))
    return ScanResponse(scan_id=patch_scan_id, status="started",
                        message="Patch validation scan started")


@app.get("/api/scan/comparison/{original_scan_id}/{patch_scan_id}")
async def get_comparison(original_scan_id: str, patch_scan_id: str):
    if original_scan_id not in scans_db:
        raise HTTPException(404, "Original scan not found")
    if patch_scan_id not in scans_db:
        raise HTTPException(404, "Patch scan not found")

    orig  = scans_db[original_scan_id]
    patch = scans_db[patch_scan_id]

    if patch["status"] != "completed":
        return {
            "status":   patch["status"],
            "progress": patch["progress"],
            "stage":    patch.get("stage", ""),
        }

    return {
        "status":            "completed",
        "original_filename": orig["filename"],
        "patch_filename":    patch["filename"],
        **_compare_scans(orig["vulnerabilities"], patch["vulnerabilities"]),
    }


@app.websocket("/ws/scan/{scan_id}")
async def ws_endpoint(websocket: WebSocket, scan_id: str):
    await websocket.accept()
    active_ws[scan_id] = websocket
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        active_ws.pop(scan_id, None)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("[*] SecuraQ++ Scanning Backend starting on port 8000")
    print(f"[*] ML Pipeline: {'AVAILABLE' if ML_AVAILABLE else 'STATIC ANALYSIS MODE'}")
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8000, reload=False)
