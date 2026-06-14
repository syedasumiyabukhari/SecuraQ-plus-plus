"""
stage0_token_normalisation.py — FS feature extraction and normalisation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extracts 48 interpretable features from raw Juliet CWE-134 C code and
fits per-group scalers on the training split only.

Feature groups (48 total):
  [0:10]  Format sink signals    — which fmt functions are present
  [10:18] Taint source signals   — which user input functions appear
  [18:26] Format argument type   — VAR vs STR as format string
  [26:32] Data flow proximity    — taint-to-sink distance
  [32:38] Structural complexity  — code length, depth, loops
  [38:43] Variable lineage       — VAR assignment patterns
  [43:48] Binary safety flags    — has_snprintf, guarded, etc.

Scalers:
  Group [0:35]   — no scaling (one-hot / ratio already [0,1])
  Group [35:43]  — RobustScaler (structural counts, outlier-robust)
  Group [43:48]  — no scaling (binary flags)

Run:
    cd ml_core
    python stage0_token_normalisation.py
"""

import os, re, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import RobustScaler, MinMaxScaler

ROOT     = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw" / "fs_dataset_sanitized.csv"
DATA_OUT = ROOT / "data" / "raw" / "fs_dataset_enriched.csv"
SCALER_OUT = ROOT / "models" / "checkpoints" / "fs_scalers.pkl"
FEAT_IMPORTANCE = ROOT / "results" / "feature_importance.csv"

ROOT.joinpath("models/checkpoints").mkdir(parents=True, exist_ok=True)
ROOT.joinpath("results").mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
FORMAT_SINKS_1 = {"printf", "wprintf", "vprintf"}          # format = arg[0]
FORMAT_SINKS_2 = {"fprintf", "sprintf", "vsprintf",        # format = arg[1]
                   "vfprintf", "swprintf", "vswprintf",
                   "syslog", "err", "warn", "errx", "warnx"}
FORMAT_SINKS_3 = {"snprintf", "vsnprintf", "wsnprintf"}    # format = arg[2]
FORMAT_SINKS_ALL = FORMAT_SINKS_1 | FORMAT_SINKS_2 | FORMAT_SINKS_3

TAINT_SOURCES = {"fgets", "fgetws", "gets", "scanf", "fscanf", "sscanf",
                 "read", "recv", "recvfrom", "fread", "getenv", "getchar",
                 "cin", "getline", "popen"}

FEATURE_NAMES = [
    # [0:10] Format sink signals
    "has_printf", "has_fprintf", "has_sprintf", "has_snprintf",
    "has_wprintf", "has_vprintf", "has_vsprintf", "has_syslog",
    "n_fmt_calls", "fmt_sink_diversity",
    # [10:18] Taint source signals
    "has_fgets", "has_scanf", "has_gets", "has_read",
    "has_recv", "has_getenv", "n_taint_sources", "taint_source_count",
    # [18:26] Format argument type
    "printf_var_fmt", "fprintf_var_fmt", "printf_str_fmt", "fprintf_str_fmt",
    "any_fmt_var", "any_fmt_str", "fmt_var_ratio", "taint_sink_cooccur",
    # [26:32] Data flow proximity
    "taint_to_printf_chain", "var_assigned_from_taint", "fmt_var_then_call",
    "taint_source_before_fmt", "indirect_fmt_call", "n_dataflow_steps",
    # [32:38] Structural complexity
    "n_semicolons", "n_if", "n_loops", "n_func_calls",
    "n_unique_func_calls", "n_return",
    # [38:43] Variable lineage
    "n_var_tokens", "n_str_tokens", "str_var_ratio",
    "max_var_num", "mean_var_num",
    # [43:48] Binary safety flags
    "has_snprintf_guard", "has_sizeof_guard",
    "has_null_check", "is_wide_char", "safe_only_pattern",
]
assert len(FEATURE_NAMES) == 48, f"Expected 48 features, got {len(FEATURE_NAMES)}"


# ── Feature extractor ──────────────────────────────────────────────────────────

class FSFeatureExtractor:
    """Extracts 48 interpretable features from a C code snippet."""

    def extract(self, code: str) -> np.ndarray:
        vec = np.zeros(48, dtype=np.float32)

        # ── Group 0: Format sink signals [0:10] ───────────────────────────────
        has_printf   = int(bool(re.search(r'\bprintf\s*\(', code)))
        has_fprintf  = int(bool(re.search(r'\bfprintf\s*\(', code)))
        has_sprintf  = int(bool(re.search(r'\bsprintf\s*\(', code)))
        has_snprintf = int(bool(re.search(r'\bsnprintf\s*\(', code)))
        has_wprintf  = int(bool(re.search(r'\bw?printf\s*\(', code)))
        has_vprintf  = int(bool(re.search(r'\bv(?:printf|sprintf|fprintf|snprintf)\s*\(', code)))
        has_vsprintf = int(bool(re.search(r'\bvsprintf\s*\(', code)))
        has_syslog   = int(bool(re.search(r'\bsyslog\s*\(', code)))

        n_fmt_calls = sum(len(re.findall(rf'\b{fn}\s*\(', code))
                         for fn in FORMAT_SINKS_ALL)
        n_fmt_types = (has_printf + has_fprintf + has_sprintf + has_snprintf +
                       has_wprintf + has_vprintf + has_vsprintf + has_syslog)

        vec[0]  = has_printf
        vec[1]  = has_fprintf
        vec[2]  = has_sprintf
        vec[3]  = has_snprintf
        vec[4]  = has_wprintf
        vec[5]  = has_vprintf
        vec[6]  = has_vsprintf
        vec[7]  = has_syslog
        vec[8]  = min(n_fmt_calls / 5.0, 1.0)
        vec[9]  = min(n_fmt_types / 4.0, 1.0)

        # ── Group 1: Taint source signals [10:18] ─────────────────────────────
        has_fgets  = int(bool(re.search(r'\bfgets\b', code)))
        has_scanf  = int(bool(re.search(r'\bscanf\b', code)))
        has_gets   = int(bool(re.search(r'\bgets\s*\(', code)))
        has_read   = int(bool(re.search(r'\bread\s*\(', code)))
        has_recv   = int(bool(re.search(r'\brecv(?:from)?\s*\(', code)))
        has_getenv = int(bool(re.search(r'\bgetenv\s*\(', code)))

        n_taint_types = has_fgets + has_scanf + has_gets + has_read + has_recv + has_getenv
        n_taint_calls = sum(len(re.findall(rf'\b{fn}\b', code)) for fn in TAINT_SOURCES)

        vec[10] = has_fgets
        vec[11] = has_scanf
        vec[12] = has_gets
        vec[13] = has_read
        vec[14] = has_recv
        vec[15] = has_getenv
        vec[16] = min(n_taint_types / 3.0, 1.0)
        vec[17] = min(n_taint_calls / 5.0, 1.0)

        # ── Group 2: Format argument type [18:26] ─────────────────────────────
        # printf(VAR...) — variable as format string (dangerous)
        printf_var_fmt = int(bool(re.search(r'\b(?:printf|wprintf|vprintf)\s*\(\s*VAR', code)))
        # fprintf(x, VAR...) — variable as format (dangerous)
        fprintf_var_fmt = int(bool(re.search(r'\bfprintf\s*\([^,\)]{0,30},\s*VAR', code)))
        # printf(STR...) — string literal as format (safe)
        printf_str_fmt = int(bool(re.search(r'\b(?:printf|wprintf|vprintf)\s*\(\s*STR', code)))
        # fprintf(x, STR...) — string literal (safe)
        fprintf_str_fmt = int(bool(re.search(r'\bfprintf\s*\([^,\)]{0,30},\s*STR', code)))

        any_fmt_var = int(printf_var_fmt or fprintf_var_fmt)
        any_fmt_str = int(printf_str_fmt or fprintf_str_fmt)

        # Ratio of unsafe to total format calls
        total_fmt_obs = any_fmt_var + any_fmt_str
        fmt_var_ratio = any_fmt_var / (total_fmt_obs + 1e-9)

        # Taint source co-occurs with format sink
        taint_sink_cooccur = int(n_taint_types > 0 and n_fmt_calls > 0)

        vec[18] = printf_var_fmt
        vec[19] = fprintf_var_fmt
        vec[20] = printf_str_fmt
        vec[21] = fprintf_str_fmt
        vec[22] = any_fmt_var
        vec[23] = any_fmt_str
        vec[24] = fmt_var_ratio
        vec[25] = taint_sink_cooccur

        # ── Group 3: Data flow proximity [26:32] ──────────────────────────────
        # Check for user input → VAR → printf chain
        taint_to_printf_chain = 0
        var_assigned_from_taint = 0
        fmt_var_then_call = 0
        taint_source_before_fmt = 0

        # Check if a variable was assigned from a taint source
        # Then check if that same variable appears as format arg
        taint_vars = set(re.findall(r'VAR_\d+', ' '.join(
            re.findall(r'(?:fgets|scanf|gets|read|recv|getenv)\s*\([^;]*?([A-Z]+_\d+)', code)
        )))

        # Simplified heuristic: taint source present AND format var present
        if n_taint_types > 0 and any_fmt_var:
            taint_to_printf_chain = 1

        # VAR = taint_source(...) pattern
        if re.search(r'VAR\s*=\s*(?:fgets|scanf|gets|recv|read|getenv)\s*\(', code):
            var_assigned_from_taint = 1

        # fmt_var followed by a function call
        if re.search(r'(?:printf|fprintf)\s*\([^;]*VAR[^;]*\)\s*;[^;]*func', code):
            fmt_var_then_call = 1

        # Any taint source appears before format sink in code (rough ordering)
        taint_pos = min((code.find(fn) for fn in TAINT_SOURCES if fn in code), default=len(code))
        fmt_pos   = min((code.find(fn) for fn in FORMAT_SINKS_ALL if fn in code), default=len(code))
        taint_source_before_fmt = int(taint_pos < fmt_pos and taint_pos < len(code))

        # Indirect call: format function inside a non-standard function
        indirect_fmt_call = int(bool(re.search(
            r'\bfunc_\d+\s*\([^)]*VAR[^)]*\)', code
        )) and n_fmt_calls == 0)

        # Number of data flow steps (rough: number of assignments between taint and fmt)
        n_dataflow_steps = len(re.findall(r'VAR\s*=\s*', code)) if n_taint_types > 0 else 0

        vec[26] = taint_to_printf_chain
        vec[27] = var_assigned_from_taint
        vec[28] = fmt_var_then_call
        vec[29] = taint_source_before_fmt
        vec[30] = indirect_fmt_call
        vec[31] = min(n_dataflow_steps / 10.0, 1.0)

        # ── Group 4: Structural complexity [32:38] (RobustScaler) ─────────────
        n_semicolons = code.count(';')
        n_if         = len(re.findall(r'\bif\s*\(', code))
        n_loops      = len(re.findall(r'\b(?:for|while|do)\s*[({]', code))
        n_func_calls = len(re.findall(r'\bfunc_\d+\s*\(', code))
        n_unique_fc  = len(set(re.findall(r'\b(func_\d+)\s*\(', code)))
        n_return     = len(re.findall(r'\breturn\b', code))

        vec[32] = n_semicolons        # scaled later
        vec[33] = n_if
        vec[34] = n_loops
        vec[35] = n_func_calls
        vec[36] = n_unique_fc
        vec[37] = n_return

        # ── Group 5: Variable lineage [38:43] (RobustScaler) ──────────────────
        var_nums    = [int(m) for m in re.findall(r'VAR_(\d+)', code)]
        str_nums    = [int(m) for m in re.findall(r'STR_(\d+)', code)]
        n_var_toks  = len(var_nums)
        n_str_toks  = len(str_nums)
        str_var_r   = n_str_toks / (n_var_toks + 1e-9)
        max_var_n   = max(var_nums) if var_nums else 0
        mean_var_n  = float(np.mean(var_nums)) if var_nums else 0.0

        vec[38] = n_var_toks          # scaled later
        vec[39] = n_str_toks          # scaled later
        vec[40] = str_var_r
        vec[41] = max_var_n           # scaled later
        vec[42] = mean_var_n          # scaled later

        # ── Group 6: Binary safety flags [43:48] ──────────────────────────────
        # snprintf with size guard (safer variant of sprintf)
        has_snprintf_guard = int(bool(re.search(r'\bsnprintf\s*\([^,]+,\s*sizeof', code)))
        # sizeof in copy/format context (bounds checking)
        has_sizeof_guard   = int(bool(re.search(r'\bsizeof\s*\(', code)) and
                                 bool(re.search(r'\b(?:snprintf|strncpy|strncat)\b', code)))
        # NULL pointer check
        has_null_check     = int(bool(re.search(r'!=\s*NULL|NULL\s*!=', code)))
        # Wide character variant (wchar_t functions)
        is_wide_char       = int(bool(re.search(r'\bwchar_t\b|\bwcslen\b|\bwprintf\b', code)))
        # Safe-only pattern: no dangerous format calls (only snprintf / with literal)
        safe_only = int(not any_fmt_var and (any_fmt_str or n_fmt_calls == 0))

        vec[43] = has_snprintf_guard
        vec[44] = has_sizeof_guard
        vec[45] = has_null_check
        vec[46] = is_wide_char
        vec[47] = safe_only

        return vec


# ── Normaliser ─────────────────────────────────────────────────────────────────

class FSNormaliser:
    """
    Per-group scaling:
      [0:32]  no scaling  (binary flags / ratios already in [0,1])
      [32:43] RobustScaler (counts — outlier robust)
      [43:48] no scaling  (binary)
    Fitted on training split only to prevent data leakage.
    """

    def __init__(self):
        self.scaler_struct = RobustScaler()

    def fit(self, X_train: np.ndarray) -> "FSNormaliser":
        self.scaler_struct.fit(X_train[:, 32:43])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[:, 32:43] = self.scaler_struct.transform(X[:, 32:43])
        return X

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run():
    print(f"Loading raw FS dataset: {DATA_RAW}")
    df = pd.read_csv(DATA_RAW)
    print(f"  Total samples: {len(df)}  label balance: {dict(df.label.value_counts())}")

    # Ensure code column exists
    code_col = "code" if "code" in df.columns else df.columns[1]
    df = df.rename(columns={code_col: "code"})

    extractor = FSFeatureExtractor()

    print("Extracting features …")
    feats = np.stack([extractor.extract(str(row["code"])) for _, row in df.iterrows()])
    print(f"  Feature matrix: {feats.shape}")

    # 70/15/15 split (reproducible)
    n = len(df)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    X_train = feats[train_idx]
    X_val   = feats[val_idx]
    X_test  = feats[test_idx]

    # Fit normaliser on training data only
    normaliser = FSNormaliser()
    X_train_n = normaliser.fit_transform(X_train)
    X_val_n   = normaliser.transform(X_val)
    X_test_n  = normaliser.transform(X_test)

    print(f"  Train={X_train_n.shape}  Val={X_val_n.shape}  Test={X_test_n.shape}")

    # Quick discriminative power report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    y_train = df["label"].values[train_idx].astype(int)
    y_val   = df["label"].values[val_idx].astype(int)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_n, y_train)
    acc = accuracy_score(y_val, rf.predict(X_val_n))
    print(f"\n  Quick RF val accuracy on 48 features: {acc:.4f} ({acc*100:.1f}%)")

    # Feature importance table
    importances = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda x: -x[1])
    print("\n  Top 15 discriminative features:")
    for name, imp in importances[:15]:
        i = FEATURE_NAMES.index(name)
        v_mean = X_train_n[y_train == 1, i].mean()
        s_mean = X_train_n[y_train == 0, i].mean()
        print(f"    {name:<35} imp={imp:.4f}  vuln={v_mean:.3f}  safe={s_mean:.3f}")

    # Save enriched CSV
    feat_df = pd.DataFrame(feats, columns=FEATURE_NAMES)
    feat_df.insert(0, "id",    df.get("id",    pd.RangeIndex(len(df))).values)
    feat_df.insert(1, "label", df["label"].values)
    feat_df["code"] = df["code"].values
    feat_df.to_csv(DATA_OUT, index=False)
    print(f"\n  Enriched CSV saved: {DATA_OUT}")

    # Save scalers
    payload = {"normaliser": normaliser, "extractor": extractor, "feature_names": FEATURE_NAMES}
    with open(SCALER_OUT, "wb") as f:
        import pickle as _pk
        _pk.dump(payload, f)
    print(f"  Scalers saved: {SCALER_OUT}")

    # Save feature importance
    imp_df = pd.DataFrame(importances, columns=["feature", "importance"])
    imp_df.to_csv(FEAT_IMPORTANCE, index=False)
    print(f"  Feature importance saved: {FEAT_IMPORTANCE}")

    return acc


if __name__ == "__main__":
    acc = run()
    print(f"\nStage 0 done. RF val accuracy: {acc:.4f}")
