"""
QEGVD - 15-Point Leakage Audit Utility
=======================================
Runs a comprehensive label-leakage audit on any vulnerability dataset.
All 15 checks must pass (zero violations) before a dataset is cleared for use.
"""

import re
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit check definitions
# ---------------------------------------------------------------------------

AUDIT_CHECKS = {
    # 1. CWE numeric tokens
    "cwe_tokens": {
        "pattern": r"CWE\d+",
        "description": "CWE identifiers embedded in code",
        "severity": "CRITICAL",
    },
    # 2. Juliet good/bad function names
    "bad_good_funcs": {
        "pattern": r"\b(bad|good)\s*\(",
        "description": "Juliet bad()/good() function signatures",
        "severity": "CRITICAL",
    },
    # 3. Juliet-specific fingerprint tokens
    "juliet_fingerprints": {
        "pattern": r"\b(goodG2B|badG2B|goodB2G|badB2G|badSink|goodSink|printLine|RAND32|globalTrue|dataBadBuffer)\b",
        "description": "Juliet-specific API / macro tokens",
        "severity": "CRITICAL",
    },
    # 4. Synthetic hardcoded control-flow
    "synthetic_controlflow": {
        "pattern": r"if\s*\(\s*(0|1)\s*\)|switch\s*\(\s*5\s*\)|while\s*\(\s*0\s*\)",
        "description": "Synthetic Juliet control-flow patterns (if(0), if(1), switch(5))",
        "severity": "HIGH",
    },
    # 5. Vulnerability hint comments
    "hint_comments": {
        "pattern": r"(//\s*(FLAW|FIX|VULNERABILITY|OVERFLOW|UNSAFE)|/\*[^*]*?(FLAW|FIX|overflow|vuln|unsafe)[^*]*?\*/)",
        "description": "Vulnerability hint comments (// FLAW, // FIX, etc.)",
        "severity": "CRITICAL",
    },
    # 6. Vulnerability-encoding identifier names
    "vuln_identifiers": {
        "pattern": r"\b(overflow_buffer|unsafe_fmt|vuln_ptr|heap_write|stack_validate|bad_buf|safe_buf)\b",
        "description": "Identifiers that encode vulnerability semantics",
        "severity": "HIGH",
    },
    # 7. CWE strings inside string literals
    "cwe_in_strings": {
        "pattern": r'"[^"]*CWE[^"]*"',
        "description": "CWE references inside string literals",
        "severity": "HIGH",
    },
    # 8. printLine / Juliet output macros
    "juliet_output_macros": {
        "pattern": r"\b(printLine|printLongLongLine|printIntLine|printHexCharLine)\s*\(",
        "description": "Juliet output helper function calls",
        "severity": "HIGH",
    },
    # 9. Residual bad/good tokens as variable/label names
    "residual_bad_good_vars": {
        "pattern": r"\b(isBad|isGood|badData|goodData|badPtr|goodPtr)\b",
        "description": "Residual bad/good variable names",
        "severity": "MEDIUM",
    },
    # 10. Raw numeric label in code (e.g., label = 1 as comment)
    "label_in_comment": {
        "pattern": r"//\s*label\s*[=:]\s*[01]",
        "description": "Explicit label value in code comment",
        "severity": "CRITICAL",
    },
    # 11. Unmasked function names that correlate with CWE
    "unmasked_cwe_functions": {
        "pattern": r"\b(heapWrite|stackWrite|heapRead|stackRead|fmtString|uafVuln)\b",
        "description": "Unmasked function names with CWE correlation",
        "severity": "HIGH",
    },
    # 12. NIST / Juliet namespace markers
    "nist_markers": {
        "pattern": r"\b(NIST|Juliet|testcase|TestCase)\b",
        "description": "NIST/Juliet test-case markers",
        "severity": "MEDIUM",
    },
    # 13. Exploit payload strings
    "exploit_strings": {
        "pattern": r'(%[0-9]+\$[nsdx]|%[0-9]{3,}[sdxn]|\\x[0-9a-fA-F]{2}\\x[0-9a-fA-F]{2})',
        "description": "Exploit payload patterns in string literals",
        "severity": "MEDIUM",
    },
    # 14. Hardcoded source/sink labels in comments
    "source_sink_comments": {
        "pattern": r"//\s*(source|sink|taint|sanitize[dr]?)",
        "description": "Explicit source/sink/taint labels in comments",
        "severity": "HIGH",
    },
    # 15. Double-free / UAF explicit markers
    "explicit_vuln_markers": {
        "pattern": r"//\s*(double.?free|use.?after.?free|buffer.?overflow|format.?string)",
        "description": "Explicit vulnerability type markers in comments",
        "severity": "CRITICAL",
    },
}


# ---------------------------------------------------------------------------
# Dataclass to hold audit results
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    dataset_name: str
    total_rows: int
    passed: bool = True
    violations: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    label_balance: dict = field(default_factory=dict)
    duplicate_count: int = 0
    near_duplicate_count: int = 0

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 60}",
            f"  AUDIT REPORT - {self.dataset_name}",
            f"{'=' * 60}",
            f"  Total rows        : {self.total_rows}",
            f"  Label distribution: {self.label_balance}",
            f"  Exact duplicates  : {self.duplicate_count}",
            f"  Near-duplicates   : {self.near_duplicate_count}",
            f"  Verdict           : {'[OK] PASS' if self.passed else '[FAIL] FAIL'}",
        ]
        if self.violations:
            lines.append(f"\n  Violations ({len(self.violations)} checks failed):")
            for check, info in self.violations.items():
                lines.append(
                    f"    [{info['severity']}] {check}: {info['count']} rows affected"
                )
                lines.append(f"           → {info['description']}")
        if self.warnings:
            lines.append(f"\n  Warnings:")
            for w in self.warnings:
                lines.append(f"    [WARN]  {w}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core audit function
# ---------------------------------------------------------------------------

def run_audit(
    df: pd.DataFrame,
    dataset_name: str,
    code_col: str = "code",
    label_col: str = "label",
    strict: bool = True,
) -> AuditResult:
    """
    Run all 15 leakage checks on a dataframe.

    Parameters
    ----------
    df           : Input dataframe
    dataset_name : Human-readable name for reporting
    code_col     : Column containing source code strings
    label_col    : Column containing binary labels (0/1)
    strict       : If True, any CRITICAL/HIGH violation marks as failed

    Returns
    -------
    AuditResult dataclass with full findings
    """
    result = AuditResult(
        dataset_name=dataset_name,
        total_rows=len(df),
    )

    if code_col not in df.columns:
        raise ValueError(f"Column '{code_col}' not found in dataframe.")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in dataframe.")

    code_series = df[code_col].fillna("").astype(str)

    # --- Label balance check ---
    vc = df[label_col].value_counts().to_dict()
    result.label_balance = {str(k): int(v) for k, v in vc.items()}
    imbalance_pct = (
        abs(vc.get(0, 0) - vc.get(1, 0)) / max(len(df), 1) * 100
    )
    if imbalance_pct > 10:
        result.warnings.append(
            f"Label imbalance detected: {imbalance_pct:.1f}% skew "
            f"(0={vc.get(0,0)}, 1={vc.get(1,0)})"
        )

    # --- Exact duplicate check ---
    result.duplicate_count = int(code_series.duplicated().sum())
    if result.duplicate_count > 0:
        result.warnings.append(
            f"{result.duplicate_count} exact duplicate code entries found"
        )

    # --- Near-duplicate check via MD5 on stripped/normalised code ---
    def _normalise(code: str) -> str:
        code = re.sub(r"\s+", " ", code).strip().lower()
        return code

    norm_hashes = code_series.apply(lambda c: hashlib.md5(_normalise(c).encode()).hexdigest())
    result.near_duplicate_count = int(norm_hashes.duplicated().sum())
    if result.near_duplicate_count > result.duplicate_count:
        result.warnings.append(
            f"{result.near_duplicate_count} near-duplicate entries (after normalisation)"
        )

    # --- Function-name to label correlation check ---
    _check_fname_label_correlation(df, code_col, label_col, result)

    # --- Run all 15 pattern checks ---
    for check_name, check_info in AUDIT_CHECKS.items():
        try:
            hits = code_series.str.contains(
                check_info["pattern"], regex=True, na=False
            ).sum()
        except re.error as exc:
            result.warnings.append(f"Regex error in check '{check_name}': {exc}")
            continue

        if hits > 0:
            result.violations[check_name] = {
                "count": int(hits),
                "severity": check_info["severity"],
                "description": check_info["description"],
            }
            if strict and check_info["severity"] in ("CRITICAL", "HIGH"):
                result.passed = False

    return result


# ---------------------------------------------------------------------------
# Helper: function-name → label correlation (Pearson phi coefficient)
# ---------------------------------------------------------------------------

def _check_fname_label_correlation(
    df: pd.DataFrame,
    code_col: str,
    label_col: str,
    result: AuditResult,
    threshold: float = 0.3,
) -> None:
    """
    Extracts top function names and checks if any single name
    is strongly correlated with the label (phi > threshold).
    """
    fname_pattern = re.compile(r"\b(\w+)\s*\(")

    def extract_first_func(code: str) -> Optional[str]:
        m = fname_pattern.search(code)
        return m.group(1) if m else None

    df_tmp = df.copy()
    df_tmp["_fname"] = df_tmp[code_col].apply(extract_first_func)
    df_tmp = df_tmp.dropna(subset=["_fname"])

    top_names = df_tmp["_fname"].value_counts().head(50).index
    for fname in top_names:
        mask = df_tmp["_fname"] == fname
        if mask.sum() < 10:
            continue
        label_when_match = df_tmp.loc[mask, label_col]
        correlation = abs(label_when_match.mean() - df_tmp[label_col].mean())
        if correlation > threshold:
            result.warnings.append(
                f"Function name '{fname}' shows label correlation "
                f"(delta={correlation:.2f}, n={mask.sum()}). "
                f"Possible residual leakage."
            )


# ---------------------------------------------------------------------------
# Convenience wrapper for a full dataset path
# ---------------------------------------------------------------------------

def audit_file(
    filepath: str,
    dataset_name: Optional[str] = None,
    code_col: str = "code",
    label_col: str = "label",
    strict: bool = True,
    print_report: bool = True,
) -> AuditResult:
    """Load a CSV and run the full audit."""
    name = dataset_name or filepath.split("/")[-1].replace(".csv", "")
    df = pd.read_csv(filepath)
    result = run_audit(df, name, code_col, label_col, strict)
    if print_report:
        print(result.summary())
    return result