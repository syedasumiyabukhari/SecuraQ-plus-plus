"""
Generate experiment progression charts for defense presentation
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'results/defense'
OUT.mkdir(parents=True, exist_ok=True)

DARK_BG  = '#0f1117'
CARD     = '#1a1d27'
GREEN    = '#00d4aa'
BLUE     = '#4a9eff'
ORANGE   = '#ff7b4a'
PURPLE   = '#c084fc'
YELLOW   = '#fbbf24'
RED      = '#f87171'
GREY     = '#6b7280'
WHITE    = '#f1f5f9'

plt.rcParams.update({
    'figure.facecolor': DARK_BG, 'axes.facecolor': CARD,
    'axes.edgecolor': '#2d3148', 'axes.labelcolor': WHITE,
    'xtick.color': WHITE, 'ytick.color': WHITE,
    'text.color': WHITE, 'grid.color': '#2d3148',
    'grid.linestyle': '--', 'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans',
})

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — BO QAFA Weight Experiments (v1–v9)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('BO — QAFA Weight Allocation Experiments (v1–v9)',
             fontsize=15, fontweight='bold', color=WHITE, y=1.01)

exps   = ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
acc    = [90.54, 90.25, 90.19, 90.19, 90.37, 90.02, 90.13, 90.43]
f1     = [91.17, 90.96, 90.77, 90.81, 90.94, 90.68, 90.42, 91.03]
auc    = [97.23, 97.09, 96.41, 95.41, 95.22, 95.81, 96.05, 97.08]
fnr    = [1.88,  1.41,  3.05,  2.58,  2.82,  2.35,  6.34,  2.35]
colors = [GREEN if e == 'v1' else (RED if e == 'v8' else BLUE) for e in exps]

ax1 = axes[0]
bars = ax1.bar(exps, acc, color=colors, width=0.6, zorder=3)
ax1.set_ylim(89.5, 91.0)
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_title('Accuracy per Experiment', fontsize=12, color=WHITE)
ax1.grid(axis='y', zorder=0)
for bar, val in zip(bars, acc):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=8.5, color=WHITE, fontweight='bold')
ax1.axhline(90.54, color=GREEN, linewidth=1.2, linestyle='--', label='v1 best (90.54%)')
ax1.legend(fontsize=9)

ax2 = axes[1]
x = np.arange(len(exps))
w = 0.28
b1 = ax2.bar(x-w, f1,  width=w, label='F1 (%)',  color=BLUE,   zorder=3)
b2 = ax2.bar(x,   auc, width=w, label='AUC (%)', color=PURPLE,  zorder=3)
b3 = ax2.bar(x+w, fnr, width=w, label='FNR (%)', color=ORANGE,  zorder=3)
ax2.set_xticks(x); ax2.set_xticklabels(exps)
ax2.set_title('F1 · AUC · FNR per Experiment', fontsize=12, color=WHITE)
ax2.set_ylabel('Score (%)', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(axis='y', zorder=0)
ax2.axvline(0, color=GREEN, linewidth=1.5, linestyle='--', alpha=0.6)

patch_v1 = mpatches.Patch(color=GREEN, label='v1 — Best (selected)')
patch_v8 = mpatches.Patch(color=RED,   label='v8 — Anomaly (high FNR)')
patch_bl = mpatches.Patch(color=BLUE,  label='Other configs')
fig.legend(handles=[patch_v1, patch_v8, patch_bl],
           loc='lower center', ncol=3, fontsize=10,
           facecolor=CARD, edgecolor=GREY, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
fig.savefig(OUT/'bo_qafa_experiments.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print('Saved: bo_qafa_experiments.png')

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — FS Step-by-Step Accuracy Progression
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(DARK_BG)

steps = [
    'QEGVD\nPaper\nBaseline',
    'Run 1\nGAT\n32Q',
    'Run 2\nGAT\n64Q',
    'Run 3\nFull\nPipeline',
    'Run 4\nHybrid\nMLP',
    'Run 5\n303-dim\nHybrid',
    'FINAL\nSupervisor\n+XGBoost',
]
accs = [61.2, 64.4, 67.2, 65.1, 67.1, 67.6, 82.0]
cols = [GREY, BLUE, BLUE, BLUE, BLUE, BLUE, ORANGE]

x = np.arange(len(steps))
bars = ax.bar(x, accs, color=cols, width=0.65, zorder=3)

for bar, val in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val}%', ha='center', va='bottom', fontsize=9.5,
            color=WHITE, fontweight='bold')

# Trend line
ax.plot(x, accs, color=YELLOW, linewidth=2, marker='o',
        markersize=6, zorder=4, label='Accuracy trend')

# Annotation for final jump
ax.annotate('★ Supervisor\nguidance\n+XGB ensemble\n+enriched features',
            xy=(6, 82.0), xytext=(4.8, 76),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
            ha='center', fontsize=8, color=GREEN, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(steps, fontsize=8.5)
ax.set_ylim(55, 90)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('FS Format String — Step-by-Step Accuracy Progression\n(QEGVD Baseline 61.2% → SecuraQpp v2 82.0%)',
             fontsize=14, fontweight='bold', color=WHITE, pad=15)
ax.grid(axis='y', zorder=0)

ax.axhline(61.2, color=GREY, linewidth=1.2, linestyle=':', label='Paper baseline (61.2%)')
ax.axhline(82.0, color=ORANGE, linewidth=1.5, linestyle='--', label='Final result (82.0%)')

p1 = mpatches.Patch(color=BLUE,   label='Our improvements')
p2 = mpatches.Patch(color=GREEN,  label='Supervisor guidance')
p3 = mpatches.Patch(color=ORANGE, label='Final result')
p4 = mpatches.Patch(color=GREY,   label='Paper baseline')
ax.legend(handles=[p1,p2,p3,p4], fontsize=9, facecolor=CARD,
          edgecolor=GREY, loc='upper left')

plt.tight_layout()
fig.savefig(OUT/'fs_accuracy_progression.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print('Saved: fs_accuracy_progression.png')

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — All 3 Datasets: Stage-by-Stage Progression
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(DARK_BG)

stage_labels = ['Paper\nBaseline', 'GAT\n(S3)', 'Encoder\n(S4)', 'QAFA\n(S5)', 'VQC\n(S6)', 'Fusion\n(S7+8)']
bo_prog  = [78.6, 90.4, 90.6, 90.2, 89.8, 90.5]
fs_prog  = [61.2, 68.0, 68.2, 67.0, 66.3, 82.0]
uaf_prog = [74.3, 84.3, 88.8, 85.0, 82.8, 91.8]

x = np.arange(len(stage_labels))
ax.plot(x, bo_prog,  color=BLUE,   linewidth=2.5, marker='o', markersize=8, label='BO (Buffer Overflow)')
ax.plot(x, fs_prog,  color=GREEN,  linewidth=2.5, marker='s', markersize=8, label='FS (Format String)')
ax.plot(x, uaf_prog, color=ORANGE, linewidth=2.5, marker='^', markersize=8, label='UAF (Use-After-Free)')

for prog, col in [(bo_prog, BLUE), (fs_prog, GREEN), (uaf_prog, ORANGE)]:
    for xi, yi in zip(x, prog):
        ax.text(xi, yi+0.8, f'{yi}%', ha='center', va='bottom', fontsize=8, color=col, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(stage_labels, fontsize=10)
ax.set_ylim(55, 97)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Accuracy Progression Across Pipeline Stages — All 3 Datasets',
             fontsize=13, fontweight='bold', color=WHITE, pad=12)
ax.grid(zorder=0)
ax.legend(fontsize=10, facecolor=CARD, edgecolor=GREY)

# Annotate fusion jump for FS
ax.annotate('+15% jump\n(Fusion stage)',
            xy=(5, 82.0), xytext=(4.2, 75),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
            fontsize=9, color=GREEN, fontweight='bold')

plt.tight_layout()
fig.savefig(OUT/'all_stage_progression.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print('Saved: all_stage_progression.png')

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — QAFA Weight Sensitivity (Centrality γ impact)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('QAFA Experiment Analysis — What Drives BO Accuracy',
             fontsize=13, fontweight='bold', color=WHITE)

# Left: Encoder size comparison
ax1 = axes[0]
cats   = ['64-dim\n(v1,v2,v9)', '90-dim\n(v4,v5,v6,v7,v8)']
avg_acc= [np.mean([90.54,90.25,90.43]), np.mean([90.19,90.19,90.37,90.02,90.13])]
avg_f1 = [np.mean([91.17,90.96,91.03]), np.mean([90.77,90.81,90.94,90.68,90.42])]
bw = 0.35
xc = np.array([0,1])
ax1.bar(xc-bw/2, avg_acc, width=bw, color=BLUE,   label='Avg Accuracy (%)', zorder=3)
ax1.bar(xc+bw/2, avg_f1,  width=bw, color=GREEN,  label='Avg F1 (%)',       zorder=3)
ax1.set_xticks(xc); ax1.set_xticklabels(cats, fontsize=11)
ax1.set_ylim(89.5, 91.5)
ax1.set_title('64-dim vs 90-dim Encoder', fontsize=11, color=WHITE)
ax1.set_ylabel('Score (%)')
ax1.grid(axis='y', zorder=0)
ax1.legend(fontsize=9)
for xi, a, f in zip(xc, avg_acc, avg_f1):
    ax1.text(xi-bw/2, a+0.05, f'{a:.2f}%', ha='center', fontsize=9, color=WHITE)
    ax1.text(xi+bw/2, f+0.05, f'{f:.2f}%', ha='center', fontsize=9, color=WHITE)

# Right: Centrality weight vs F1
ax2 = axes[1]
cent_vals = [0.50, 0.60, 0.65, 0.70, 0.60, 0.50, 0.50, 0.50]
f1_vals   = [91.17, 90.96, 90.77, 90.81, 90.94, 90.68, 90.42, 91.03]
enc_cols  = [GREEN if e in ['v1','v2','v9'] else BLUE for e in exps]
sc = ax2.scatter(cent_vals, f1_vals, c=[GREEN,BLUE,BLUE,BLUE,BLUE,BLUE,RED,GREEN],
                 s=120, zorder=4)
for e, cx, fy in zip(exps, cent_vals, f1_vals):
    ax2.annotate(e, (cx, fy), textcoords='offset points',
                 xytext=(6,4), fontsize=9, color=WHITE)
ax2.set_xlabel('Centrality Weight (γ)', fontsize=11)
ax2.set_ylabel('F1 Score (%)', fontsize=11)
ax2.set_title('Centrality Weight vs F1 Score', fontsize=11, color=WHITE)
ax2.grid(zorder=0)
ax2.axvline(0.50, color=GREEN, linestyle='--', linewidth=1.2, label='Optimal γ=0.50')
ax2.legend(fontsize=9)

plt.tight_layout()
fig.savefig(OUT/'qafa_sensitivity.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print('Saved: qafa_sensitivity.png')

print('\nAll experiment charts saved to:', OUT)
