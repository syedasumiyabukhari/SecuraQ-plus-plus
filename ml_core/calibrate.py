"""
Quick calibration script - runs all test samples through all 3 classifiers
and prints a probability matrix. No plots, no heavy I/O.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Suppress matplotlib and most logging
import logging
logging.basicConfig(level=logging.WARNING)
import os
os.environ["MPLBACKEND"] = "Agg"

from demo_single_sample import (
    stage1_preprocess, stage2_build_graphs,
    stage3_gat_embed, stage4_compress, stage5_qafa,
    stage6_vqc, stage78_classify
)

SAMPLES = {
    "train_bo_vuln":   ROOT / "test_samples" / "_train_bo_vuln.c",
    "train_bo_safe":   ROOT / "test_samples" / "_train_bo_safe.c",
    "train_fs_vuln":   ROOT / "test_samples" / "_train_fs_vuln.c",
    "train_fs_safe":   ROOT / "test_samples" / "_train_fs_safe.c",
    "train_uaf_vuln":  ROOT / "test_samples" / "_train_uaf_vuln.c",
    "train_uaf_safe":  ROOT / "test_samples" / "_train_uaf_safe.c",
}

CLASSIFIERS = ["bo", "fs", "uaf"]

def run_classifier(graphs, masked_code, ds_key):
    """Run stages 3-8 for one classifier. Returns probability or None."""
    try:
        fused, view_embeds, gat_prob = stage3_gat_embed(graphs, ds_key, code_text=masked_code)
        compressed, full_emb = stage4_compress(fused, ds_key)
        s1, s2, selected_idx, scores = stage5_qafa(compressed, ds_key)
        qvec, circuit_info = stage6_vqc(s1, s2, ds_key)
        prob, hybrid, threshold = stage78_classify(compressed, qvec, full_emb, ds_key)
        return prob, threshold
    except Exception as e:
        return None, None

def main():
    results = {}
    thresholds = {}

    for sample_name, sample_path in SAMPLES.items():
        code = sample_path.read_text(encoding="utf-8")
        masked = stage1_preprocess(code)
        graphs, builder = stage2_build_graphs(masked)

        results[sample_name] = {}
        for ds in CLASSIFIERS:
            prob, thresh = run_classifier(graphs, masked, ds)
            results[sample_name][ds] = prob
            if thresh is not None:
                thresholds[ds] = thresh

    # Print matrix
    print("\n" + "="*70)
    print("PROBABILITY MATRIX  (rows=samples, cols=classifiers)")
    print("="*70)
    header = f"{'Sample':<20} {'BO':>8} {'FS':>8} {'UAF':>8}"
    print(header)
    print("-"*len(header))
    for sample_name in SAMPLES:
        row = results[sample_name]
        vals = []
        for ds in CLASSIFIERS:
            p = row[ds]
            vals.append(f"{p:.4f}" if p is not None else "  N/A ")
        print(f"{sample_name:<20} {'  '.join(vals)}")

    print()
    print("Current thresholds:")
    for ds in CLASSIFIERS:
        t = thresholds.get(ds, "N/A")
        print(f"  {ds.upper()}: {t:.4f}" if isinstance(t, float) else f"  {ds.upper()}: {t}")

    # Save to JSON for later use
    out = {"probabilities": results, "thresholds": thresholds}
    out_path = ROOT / "results" / "calibration_matrix.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert for JSON serialization
    for sname in out["probabilities"]:
        for ds in out["probabilities"][sname]:
            v = out["probabilities"][sname][ds]
            if v is not None:
                out["probabilities"][sname][ds] = round(float(v), 6)
    for ds in out["thresholds"]:
        out["thresholds"][ds] = round(float(out["thresholds"][ds]), 6)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
