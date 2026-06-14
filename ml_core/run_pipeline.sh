#!/usr/bin/env bash
# ============================================================
#  QEGVD — Full Pipeline Runner
#  Usage: bash run_pipeline.sh [bo|fs|uaf|all]
# ============================================================

set -euo pipefail

DATASET=${1:-all}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="python"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  QEGVD — Quantum-Enhanced Graph Vulnerability Det.   ║"
echo "║  Pipeline Runner                                     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Root    : $ROOT_DIR"
echo "  Dataset : $DATASET"
echo "  Python  : $(which $PYTHON)"
echo ""

cd "$ROOT_DIR"

# ── Stage 1 ──────────────────────────────────────────────────
echo "▶  Stage 1: Preprocessing & Stratified Splitting"
$PYTHON src/stage1_preprocessing.py --dataset "$DATASET"
echo "   Stage 1 complete ✅"
echo ""

# ── Stage 2 ──────────────────────────────────────────────────
echo "▶  Stage 2: Static Code Analysis & Graph Construction"
$PYTHON src/stage2_graph_construction.py --dataset "$DATASET"
echo "   Stage 2 complete ✅"
echo ""

# ── Stage 3 ──────────────────────────────────────────────────
echo "▶  Stage 3: Graph Attention Network (GAT) Training"
$PYTHON src/stage3_gat.py --dataset "$DATASET"
echo "   Stage 3 complete ✅"
echo ""

# ── Stage 4 ──────────────────────────────────────────────────
echo "▶  Stage 4: Classical Feature Encoding & Compression"
$PYTHON src/stage4_classical_encoder.py --dataset "$DATASET"
echo "   Stage 4 complete ✅"
echo ""

# ── Stage 5 ──────────────────────────────────────────────────
echo "▶  Stage 5: Quantum-Aware Feature Alignment (QAFA)"
$PYTHON src/stage5_qafa.py --dataset "$DATASET"
echo "   Stage 5 complete ✅"
echo ""

# ── Stage 6 ──────────────────────────────────────────────────
echo "▶  Stage 6: Variational Quantum Circuit (VQC)"
$PYTHON src/stage6_vqc.py --dataset "$DATASET"
echo "   Stage 6 complete ✅"
echo ""

# ── Stage 7+8 ────────────────────────────────────────────────
echo "▶  Stage 7+8: Residual Hybrid Fusion + MLP Classifier"
$PYTHON src/stage7_fusion.py --dataset "$DATASET"
echo "   Stage 7+8 complete [OK]"
echo ""

# ── Stage 9 ──────────────────────────────────────────────────
echo "▶  Stage 9: Explainable Vulnerability Analysis"
$PYTHON src/stage9_explainability.py --dataset "$DATASET"
echo "   Stage 9 complete ✅"
echo ""

echo "╔══════════════════════════════════════════════════════╗"
echo "║  PIPELINE COMPLETE  ✅                               ║"
echo "║  Results → results/metrics/                         ║"
echo "║  Models  → models/final/                            ║"
echo "╚══════════════════════════════════════════════════════╝"