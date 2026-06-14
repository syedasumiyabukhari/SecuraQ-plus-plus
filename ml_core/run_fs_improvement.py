"""
run_fs_improvement.py — Master runner for FS accuracy improvements
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Orchestrates all FS-specific improvement stages:

  Step 0 : Token normalisation + feature extraction (stage0)
  Step 1 : Analyse discriminative power of 48 features
  Step 2 : Retrain FS classifier using enriched features

Integration notes:
  - VLG / APG graphs are added to stage2_graph_construction.py via
    the FS-only patch (see stage2_graph_improvements.py)
  - EnhancedQAFA is used automatically by stage5_qafa.py for fs
  - EnhancedVQC is used automatically by stage6_vqc.py for fs
  - EnhancedFSHybridClassifier is used by stage7_fusion.py for fs

Run from the ml_core directory:
    python run_fs_improvement.py
"""

import os, sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

print("=" * 65)
print("  FS IMPROVEMENT PIPELINE")
print("=" * 65)

# ── Step 0: Token normalisation ───────────────────────────────────────────────
print("\n[Step 0] Token normalisation & feature extraction …")
t0 = time.time()

try:
    import stage0_token_normalisation as s0
    acc = s0.run()
    print(f"  Stage 0 done in {time.time()-t0:.1f}s  RF val acc={acc:.4f}")
except Exception as e:
    print(f"  [WARN] Stage 0 failed: {e}")
    acc = 0.0

# ── Step 1: Analyse features ──────────────────────────────────────────────────
print("\n[Step 1] Analysing feature discriminative power …")
feat_imp_path = ROOT / "results" / "feature_importance.csv"
if feat_imp_path.exists():
    import pandas as pd
    feat_df = pd.read_csv(feat_imp_path)
    print("  Top 10 most discriminative features:")
    for _, row in feat_df.head(10).iterrows():
        print(f"    {row['feature']:<35} importance={row['importance']:.4f}")
else:
    print("  [WARN] feature_importance.csv not found, skipping analysis")

# ── Step 2: Enhanced direct classifier ───────────────────────────────────────
print("\n[Step 2] Retraining FS direct classifier with enriched features …")
t2 = time.time()

try:
    import importlib, subprocess, sys as _sys
    result = subprocess.run(
        [_sys.executable, str(ROOT / "train_fs_direct.py")],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    # Print last 30 lines of output
    lines = result.stdout.strip().split("\n")
    for line in lines[-30:]:
        print(" ", line)
    if result.returncode != 0:
        print("  STDERR:", result.stderr[-500:])
except Exception as e:
    print(f"  [WARN] Step 2 failed: {e}")

print(f"\n[Done] FS improvement pipeline complete in {time.time()-t0:.1f}s")
print("=" * 65)

# ── Print integration status ──────────────────────────────────────────────────
print("\n  Integration status:")
checks = [
    (ROOT / "src" / "stage2_graph_improvements.py", "VLG + APG graph improvements"),
    (ROOT / "src" / "stage5_enhanced_qafa.py",      "Enhanced QAFA (MMD + 4-component)"),
    (ROOT / "src" / "stage6_enhanced_vqc.py",       "Enhanced VQC (8-qubit + ZZ)"),
    (ROOT / "data" / "raw" / "fs_dataset_enriched.csv", "Enriched FS dataset"),
    (ROOT / "models" / "checkpoints" / "fs_scalers.pkl", "FS scalers"),
    (ROOT / "models" / "checkpoints" / "fs_direct.pkl",  "FS direct classifier"),
]
for path, desc in checks:
    status = "[OK]" if path.exists() else "[MISSING]"
    print(f"    {status}  {desc}")

print()
print("  To run the full QEGVD pipeline improvement (Stage 2-7):")
print("    cd ml_core/src")
print("    python stage2_graph_construction.py --dataset fs")
print("    python stage3_gat.py               --dataset fs")
print("    python stage4_classical_encoder.py --dataset fs")
print("    python stage5_qafa.py              --dataset fs")
print("    python stage6_vqc.py               --dataset fs")
print("    python stage7_fusion.py            --dataset fs")
print()
