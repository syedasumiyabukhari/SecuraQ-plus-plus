"""
QEGVD - Stage 1: Dataset Preprocessing & Stratified Splitting
==============================================================
Responsibilities:
  1. Load sanitized CSV datasets (bo / fs / uaf)
  2. Run the 15-point leakage audit - abort if any CRITICAL/HIGH violation
  3. Apply final normalisation (whitespace, length filtering)
  4. Stratified train / val / test split (70 / 15 / 15)
  5. Save splits as CSV files to data/processed/<dataset>/
  6. Write a split_report.json with statistics for downstream stages
  7. Optionally verify no data overlap between splits

Usage
-----
    # Process a single dataset
    python src/stage1_preprocessing.py --dataset bo

    # Process all three datasets
    python src/stage1_preprocessing.py --dataset all

    # Dry-run: audit only, no files written
    python src/stage1_preprocessing.py --dataset all --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Bootstrap: make src/ importable regardless of working directory
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from utils.audit import audit_file, AuditResult   # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
(_ROOT / "logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)),
        logging.FileHandler(_ROOT / "logs" / "stage1.log", mode="a"),
    ],
)
logger = logging.getLogger("Stage1")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_FILENAMES = {
    "bo":  "bo_dataset_sanitized.csv",
    "fs":  "fs_dataset_sanitized.csv",
    "uaf": "uaf_dataset_sanitized.csv",
}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        config_path = _ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalise_whitespace(code: str) -> str:
    """Collapse tabs and multiple spaces to single space; strip edges."""
    code = code.replace("\t", " ")
    code = re.sub(r"[ ]{2,}", " ", code)
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code.strip()


def filter_by_length(
    df: pd.DataFrame,
    code_col: str,
    min_len: int,
    max_len: int,
) -> tuple[pd.DataFrame, int]:
    """Drop rows whose code length is outside [min_len, max_len]."""
    before = len(df)
    mask = df[code_col].str.len().between(min_len, max_len)
    df = df[mask].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Length filter dropped {dropped} rows "
                       f"(min={min_len}, max={max_len})")
    return df, dropped


def remove_exact_duplicates(
    df: pd.DataFrame,
    code_col: str,
) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates(subset=[code_col]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.warning(f"Removed {removed} exact duplicate code rows")
    return df, removed


def compute_md5(code: str) -> str:
    normalised = re.sub(r"\s+", " ", code).strip().lower()
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()


def remove_near_duplicates(
    df: pd.DataFrame,
    code_col: str,
) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df["_md5"] = df[code_col].apply(compute_md5)
    df = df.drop_duplicates(subset=["_md5"]).drop(columns=["_md5"]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.warning(f"Removed {removed} near-duplicate code rows (MD5 after normalisation)")
    return df, removed


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (train_df, val_df, test_df) with stratified label distribution.
    Split is performed in two passes:
      Pass 1: Split off test set
      Pass 2: Split remaining into train + val
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Split fractions must sum to 1.0"

    # Pass 1: hold out test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    idx_trainval, idx_test = next(sss1.split(df, df[label_col]))

    df_trainval = df.iloc[idx_trainval].reset_index(drop=True)
    df_test     = df.iloc[idx_test].reset_index(drop=True)

    # Pass 2: split trainval into train + val
    val_relative = val_frac / (train_frac + val_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed)
    idx_train, idx_val = next(sss2.split(df_trainval, df_trainval[label_col]))

    df_train = df_trainval.iloc[idx_train].reset_index(drop=True)
    df_val   = df_trainval.iloc[idx_val].reset_index(drop=True)

    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Overlap verification
# ---------------------------------------------------------------------------

def verify_no_overlap(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    code_col: str,
) -> bool:
    """
    Confirm zero code overlap between splits.
    Returns True if clean, raises AssertionError otherwise.
    """
    train_hashes = set(train[code_col].apply(compute_md5))
    val_hashes   = set(val[code_col].apply(compute_md5))
    test_hashes  = set(test[code_col].apply(compute_md5))

    tv = train_hashes & val_hashes
    tt = train_hashes & test_hashes
    vt = val_hashes   & test_hashes

    leaks = []
    if tv: leaks.append(f"Train∩Val: {len(tv)} overlap")
    if tt: leaks.append(f"Train∩Test: {len(tt)} overlap")
    if vt: leaks.append(f"Val∩Test: {len(vt)} overlap")

    if leaks:
        msg = "SPLIT OVERLAP DETECTED: " + " | ".join(leaks)
        logger.error(msg)
        raise AssertionError(msg)

    logger.info("Split overlap check: [OK] CLEAN - no leakage between splits")
    return True


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_split_report(
    dataset_name: str,
    original_rows: int,
    final_rows: int,
    exact_dups_removed: int,
    near_dups_removed: int,
    length_filtered: int,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label_col: str,
    audit_result: AuditResult,
    config: dict,
) -> dict:
    def split_stats(df: pd.DataFrame, split_name: str) -> dict:
        vc = df[label_col].value_counts().to_dict()
        return {
            "rows": len(df),
            "label_0": int(vc.get(0, 0)),
            "label_1": int(vc.get(1, 0)),
            "balance_pct": round(
                min(vc.get(0, 0), vc.get(1, 0)) / max(len(df), 1) * 100, 2
            ),
        }

    return {
        "dataset": dataset_name,
        "audit_passed": audit_result.passed,
        "original_rows": original_rows,
        "exact_duplicates_removed": exact_dups_removed,
        "near_duplicates_removed": near_dups_removed,
        "length_filtered_removed": length_filtered,
        "final_rows": final_rows,
        "splits": {
            "train": split_stats(train, "train"),
            "val":   split_stats(val, "val"),
            "test":  split_stats(test, "test"),
        },
        "config": {
            "seed": config["project"]["seed"],
            "train_frac": config["data"]["splits"]["train"],
            "val_frac":   config["data"]["splits"]["val"],
            "test_frac":  config["data"]["splits"]["test"],
            "min_code_len": config["data"]["min_code_length"],
            "max_code_len": config["data"]["max_code_length"],
        },
        "audit_warnings": audit_result.warnings,
    }


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_key: str,
    config: dict,
    dry_run: bool = False,
) -> dict:
    """
    Full Stage 1 pipeline for a single dataset.

    Parameters
    ----------
    dataset_key : 'bo' | 'fs' | 'uaf'
    config      : Parsed YAML config dict
    dry_run     : If True, skip file writes

    Returns
    -------
    Split report dictionary
    """
    logger.info(f"{'='*60}")
    logger.info(f"Processing dataset: {dataset_key.upper()}")
    logger.info(f"{'='*60}")

    # --- Paths ---
    raw_dir   = _ROOT / config["data"]["raw_dir"]
    proc_dir  = _ROOT / config["data"]["processed_dir"] / dataset_key
    proc_dir.mkdir(parents=True, exist_ok=True)

    filename  = DATASET_FILENAMES[dataset_key]
    raw_path  = raw_dir / filename

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {raw_path}\n"
            f"Place the sanitized CSV in data/raw/"
        )

    # --- Load ---
    logger.info(f"Loading: {raw_path}")
    df = pd.read_csv(raw_path)
    original_rows = len(df)
    logger.info(f"Loaded {original_rows} rows")

    # --- 15-Point Leakage Audit ---
    logger.info("Running 15-point leakage audit ...")
    audit_result = audit_file(
        filepath=str(raw_path),
        dataset_name=dataset_key,
        strict=True,
        print_report=True,
    )
    if not audit_result.passed:
        logger.error(
            f"Dataset '{dataset_key}' FAILED leakage audit. "
            f"Violations: {list(audit_result.violations.keys())}"
        )
        if not dry_run:
            raise RuntimeError(
                f"Aborting Stage 1 - dataset '{dataset_key}' failed audit. "
                f"Fix leakage issues before proceeding."
            )
        logger.warning("Dry-run mode: continuing despite audit failure.")
    else:
        logger.info(f"Audit PASSED [OK] for {dataset_key}")

    # --- Normalise whitespace ---
    logger.info("Normalising whitespace ...")
    df["code"] = df["code"].fillna("").astype(str).apply(normalise_whitespace)

    # --- Length filtering ---
    min_len = config["data"]["min_code_length"]
    max_len = config["data"]["max_code_length"]
    df, length_filtered = filter_by_length(df, "code", min_len, max_len)
    logger.info(f"After length filter: {len(df)} rows")

    # --- Exact duplicate removal ---
    df, exact_dups = remove_exact_duplicates(df, "code")
    logger.info(f"After exact-dedup: {len(df)} rows")

    # --- Near-duplicate removal ---
    df, near_dups = remove_near_duplicates(df, "code")
    logger.info(f"After near-dedup: {len(df)} rows")

    # --- Shuffle with fixed seed ---
    seed = config["project"]["seed"]
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # --- Stratified split ---
    logger.info("Performing stratified split (70/15/15) ...")
    train_df, val_df, test_df = stratified_split(
        df=df,
        label_col="label",
        train_frac=config["data"]["splits"]["train"],
        val_frac=config["data"]["splits"]["val"],
        test_frac=config["data"]["splits"]["test"],
        seed=seed,
    )

    logger.info(
        f"Split sizes - Train: {len(train_df)}, "
        f"Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # --- Overlap verification ---
    verify_no_overlap(train_df, val_df, test_df, "code")

    # --- Log label distributions ---
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        vc = split_df["label"].value_counts().to_dict()
        logger.info(f"  {split_name}: label_0={vc.get(0,0)}, label_1={vc.get(1,0)}")

    # --- Build report ---
    report = build_split_report(
        dataset_name=dataset_key,
        original_rows=original_rows,
        final_rows=len(df),
        exact_dups_removed=exact_dups,
        near_dups_removed=near_dups,
        length_filtered=length_filtered,
        train=train_df,
        val=val_df,
        test=test_df,
        label_col="label",
        audit_result=audit_result,
        config=config,
    )

    # --- Save ---
    if not dry_run:
        for split_name, split_df in [
            ("train", train_df), ("val", val_df), ("test", test_df)
        ]:
            out_path = proc_dir / f"{split_name}.csv"
            split_df.to_csv(out_path, index=False)
            logger.info(f"Saved: {out_path}  ({len(split_df)} rows)")

        report_path = proc_dir / "split_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Split report: {report_path}")
    else:
        logger.info("Dry-run: no files written.")

    logger.info(f"Stage 1 complete for '{dataset_key}' [OK]\n")
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QEGVD Stage 1 - Preprocessing & Splitting"
    )
    parser.add_argument(
        "--dataset",
        choices=["bo", "fs", "uaf", "all"],
        required=True,
        help="Which dataset(s) to process",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run audit and stats only - do not write output files",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Set global seed
    seed = config["project"]["seed"]
    np.random.seed(seed)

    datasets_to_process = (
        ["bo", "fs", "uaf"] if args.dataset == "all" else [args.dataset]
    )

    all_reports = {}
    for ds in datasets_to_process:
        try:
            report = process_dataset(ds, config, dry_run=args.dry_run)
            all_reports[ds] = report
        except (FileNotFoundError, RuntimeError) as exc:
            logger.error(f"FAILED for dataset '{ds}': {exc}")
            sys.exit(1)

    # --- Master summary ---
    print("\n" + "=" * 60)
    print("  STAGE 1 SUMMARY")
    print("=" * 60)
    for ds, report in all_reports.items():
        train_info = report["splits"]["train"]
        val_info   = report["splits"]["val"]
        test_info  = report["splits"]["test"]
        print(f"\n  {ds.upper()}")
        print(f"    Original rows     : {report['original_rows']}")
        print(f"    Final rows        : {report['final_rows']}")
        print(f"    Exact dups removed: {report['exact_duplicates_removed']}")
        print(f"    Near dups removed : {report['near_duplicates_removed']}")
        print(f"    Length filtered   : {report['length_filtered_removed']}")
        print(f"    Train             : {train_info['rows']} "
              f"(0={train_info['label_0']}, 1={train_info['label_1']})")
        print(f"    Val               : {val_info['rows']} "
              f"(0={val_info['label_0']}, 1={val_info['label_1']})")
        print(f"    Test              : {test_info['rows']} "
              f"(0={test_info['label_0']}, 1={test_info['label_1']})")
    print("\n" + "=" * 60)
    print("  All datasets processed. Outputs → data/processed/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()