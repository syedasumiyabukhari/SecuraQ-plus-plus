import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

BASE   = os.path.dirname(os.path.abspath(__file__))
METRICS= os.path.join(BASE, 'results', 'metrics')
OUT    = os.path.join(BASE, 'results', 'defense')
os.makedirs(OUT, exist_ok=True)

# ── confirmed test results ──────────────────────────────────────────────────
DATA = {
    'BO': {
        'S3':   dict(tn=714, fp=147, fn=18,  tp=834, f1=0.9100, auc=0.9716, mcc=0.8168),
        'S6':   dict(tn=746, fp=115, fn=60,  tp=792, f1=0.9005, auc=0.9672, mcc=0.7974),
        'S7+8': dict(tn=715, fp=146, fn=16,  tp=836, f1=0.9117, auc=0.9723, mcc=0.8205),
    },
    'FS': {
        'S3':   dict(tn=209, fp=194, fn=65,  tp=341, f1=0.7248, auc=0.7040, mcc=0.3788),
        'S6':   dict(tn=211, fp=192, fn=77,  tp=329, f1=0.7098, auc=0.7170, mcc=0.3487),
        'S7+8': dict(tn=313, fp=90,  fn=56,  tp=350, f1=0.8274, auc=0.8900, mcc=0.6412),
    },
    'UAF': {
        'S3':   dict(tn=62,  fp=8,  fn=13,  tp=51,  f1=0.8293, auc=0.9275, mcc=0.6868),
        'S6':   dict(tn=60,  fp=10, fn=13,  tp=51,  f1=0.8160, auc=0.8970, mcc=0.6560),
        'S7+8': dict(tn=58,  fp=12, fn=3,   tp=61,  f1=0.8905, auc=0.9375, mcc=0.7841),
    },
}

def acc(d):  return (d['tp']+d['tn'])/(d['tp']+d['tn']+d['fp']+d['fn'])
def prec(d): return d['tp']/(d['tp']+d['fp']) if (d['tp']+d['fp'])>0 else 0
def rec(d):  return d['tp']/(d['tp']+d['fn']) if (d['tp']+d['fn'])>0 else 0

COLORS = {'BO': '#f59e0b', 'FS': '#3b82f6', 'UAF': '#10b981'}
STAGE_COLORS = {'S3': '#6366f1', 'S6': '#ec4899', 'S7+8': '#14b8a6'}
BG = '#0f0f1a'
PANEL = '#1a1a2e'

plt.rcParams.update({
    'font.family': 'monospace',
    'text.color': '#e2e8f0',
    'axes.labelcolor': '#94a3b8',
    'axes.titlecolor': '#e2e8f0',
    'xtick.color': '#64748b',
    'ytick.color': '#64748b',
    'axes.facecolor': PANEL,
    'figure.facecolor': BG,
    'axes.edgecolor': '#2d2d4e',
    'grid.color': '#1e1e3a',
    'grid.linewidth': 0.6,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────
def make_training_curves():
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), facecolor=BG)
    fig.suptitle('Stage 3 GAT — Validation Training Curves', fontsize=14,
                 fontweight='bold', color='#e2e8f0', y=0.98)

    metrics_map = [('f1', 'F1 Score', '#f59e0b'),
                   ('roc_auc', 'ROC-AUC', '#3b82f6'),
                   ('mcc', 'MCC', '#10b981')]
    vuln_types  = [('BO', 'Buffer Overflow'),
                   ('FS', 'Format String'),
                   ('UAF', 'Use-After-Free')]

    for row, (vt, vname) in enumerate(vuln_types):
        fname = f'{vt.lower()}_stage3_history.json'
        with open(os.path.join(METRICS, fname)) as fp:
            hist = json.load(fp)
        vals = hist.get('val', [])
        epochs = [e['epoch'] for e in vals]

        for col, (mkey, mlabel, mcol) in enumerate(metrics_map):
            ax = axes[row][col]
            values = [e.get(mkey, 0) for e in vals]
            ax.plot(epochs, values, color=mcol, linewidth=1.4, alpha=0.9)
            ax.fill_between(epochs, values, alpha=0.08, color=mcol)

            # rolling max line
            best = max(values) if values else 0
            ax.axhline(best, color=mcol, linewidth=0.7, linestyle='--', alpha=0.5)
            ax.text(epochs[-1]*0.98, best+0.01, f'{best:.3f}',
                    fontsize=7, color=mcol, ha='right', va='bottom')

            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Epoch', fontsize=8)
            if col == 0:
                ax.set_ylabel(vname, fontsize=8, color=COLORS[vt], fontweight='bold')
            if row == 0:
                ax.set_title(mlabel, fontsize=9, color=mcol, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT, 'training_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. PIPELINE PROGRESSION
# ─────────────────────────────────────────────────────────────────────────────
def make_pipeline_progression():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    fig.suptitle('Pipeline Progression: GAT → VQC → Hybrid Fusion',
                 fontsize=13, fontweight='bold', color='#e2e8f0', y=1.02)

    stage_labels = ['S3\n(GAT)', 'S6\n(VQC)', 'S7+8\n(Hybrid)']
    metrics_cfg  = [('f1',  'F1 Score',  '#f59e0b'),
                    ('auc', 'ROC-AUC',   '#3b82f6'),
                    ('mcc', 'MCC',       '#10b981')]

    for mi, (mkey, mlabel, mcol) in enumerate(metrics_cfg):
        ax = axes[mi]
        for vt, vname in [('BO','Buffer Overflow'),('FS','Format String'),('UAF','Use-After-Free')]:
            vals = [DATA[vt][s][mkey] for s in ['S3','S6','S7+8']]
            vcol = COLORS[vt]
            ax.plot([0,1,2], vals, 'o-', color=vcol, linewidth=2,
                    markersize=8, label=vt, markeredgecolor='white', markeredgewidth=0.8)
            for xi, v in enumerate(vals):
                ax.text(xi, v+0.012, f'{v:.3f}', ha='center', fontsize=7.5,
                        color=vcol, fontweight='bold')

        ax.set_xticks([0,1,2])
        ax.set_xticklabels(stage_labels, fontsize=9)
        ax.set_title(mlabel, fontsize=11, color=mcol, fontweight='bold', pad=8)
        ax.set_ylim(0.2, 1.05)
        ax.grid(True, alpha=0.35)
        ax.yaxis.set_tick_params(labelsize=8)
        if mi == 0:
            ax.legend(fontsize=8, loc='lower right',
                      facecolor=PANEL, edgecolor='#2d2d4e', labelcolor='#e2e8f0')

    plt.tight_layout()
    out = os.path.join(OUT, 'pipeline_progression.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. METRICS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def make_metrics_table():
    fig, ax = plt.subplots(figsize=(16, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis('off')
    fig.suptitle('QEGVD — Complete Test Set Results', fontsize=13,
                 fontweight='bold', color='#e2e8f0', y=0.98)

    col_labels = ['Vuln\nType', 'Stage', 'Accuracy', 'Precision', 'Recall',
                  'F1 Score', 'MCC', 'ROC-AUC', 'FPR', 'FNR']
    rows = []
    for vt in ['BO', 'FS', 'UAF']:
        for si, stage in enumerate(['S3', 'S6', 'S7+8']):
            d = DATA[vt][stage]
            a = acc(d); p = prec(d); r = rec(d)
            fpr = d['fp']/(d['fp']+d['tn']) if (d['fp']+d['tn'])>0 else 0
            fnr = d['fn']/(d['fn']+d['tp']) if (d['fn']+d['tp'])>0 else 0
            stage_label = {'S3':'Stage 3 (GAT)','S6':'Stage 6 (VQC)','S7+8':'Stage 7+8 (Hybrid)'}[stage]
            rows.append([
                vt if si == 0 else '',
                stage_label,
                f'{a:.4f}', f'{p:.4f}', f'{r:.4f}',
                f'{d["f1"]:.4f}', f'{d["mcc"]:.4f}', f'{d["auc"]:.4f}',
                f'{fpr:.4f}', f'{fnr:.4f}'
            ])

    cell_text = [r[1:] for r in rows]
    row_labels = [r[0] for r in rows]

    col_widths = [0.09,0.16,0.09,0.09,0.08,0.09,0.08,0.09,0.08,0.08]
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels[1:],
        cellLoc='center', rowLoc='center',
        loc='center',
        colWidths=col_widths
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.1)

    # Style header
    for j in range(len(col_labels)-1):
        cell = table[0, j]
        cell.set_facecolor('#1e1e3a')
        cell.set_text_props(color='#94a3b8', fontweight='bold', fontsize=8)
        cell.set_edgecolor('#2d2d4e')

    # Style row labels
    for i, vt in enumerate(['BO','BO','BO','FS','FS','FS','UAF','UAF','UAF']):
        cell = table[i+1, -1]
        cell.set_facecolor('#1a1a2e')
        cell.set_text_props(color=COLORS[vt], fontweight='bold')
        cell.set_edgecolor('#2d2d4e')

    # Style data cells
    highlight_cols = [4, 5, 6]  # F1, MCC, AUC (0-indexed in cell_text)
    for i in range(9):
        vt = ['BO','BO','BO','FS','FS','FS','UAF','UAF','UAF'][i]
        stage = ['S3','S6','S7+8'][i%3]
        is_final = (stage == 'S7+8')
        for j in range(len(col_labels)-1):
            cell = table[i+1, j]
            if is_final:
                cell.set_facecolor('#16213e')
            else:
                cell.set_facecolor('#1a1a2e' if i%2==0 else '#1c1c30')
            cell.set_edgecolor('#2d2d4e')
            if j in highlight_cols and is_final:
                cell.set_text_props(color=COLORS[vt], fontweight='bold')
            else:
                cell.set_text_props(color='#cbd5e1')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, 'metrics_table.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# 4. METRICS COMPARISON (bar charts)
# ─────────────────────────────────────────────────────────────────────────────
def make_metrics_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=BG)
    fig.suptitle('Stage 7+8 Final Performance — Metrics Comparison',
                 fontsize=13, fontweight='bold', color='#e2e8f0', y=0.99)

    vtypes = ['BO', 'FS', 'UAF']
    metrics_cfg = [
        ('f1',   'F1 Score',   '#f59e0b', axes[0][0]),
        ('auc',  'ROC-AUC',    '#3b82f6', axes[0][1]),
        ('mcc',  'MCC',        '#10b981', axes[1][0]),
        ('acc',  'Accuracy',   '#8b5cf6', axes[1][1]),
    ]

    x = np.arange(3)
    bar_w = 0.5

    for mkey, mlabel, mcol, ax in metrics_cfg:
        vals = []
        for vt in vtypes:
            d = DATA[vt]['S7+8']
            vals.append(acc(d) if mkey == 'acc' else d[mkey])

        vcols = [COLORS[vt] for vt in vtypes]
        bars = ax.bar(x, vals, width=bar_w, color=vcols, alpha=0.85,
                      edgecolor='white', linewidth=0.5)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#e2e8f0')

        ax.set_xticks(x)
        ax.set_xticklabels(vtypes, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.12)
        ax.set_title(mlabel, fontsize=11, color=mcol, fontweight='bold', pad=6)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='x', colors='#e2e8f0')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT, 'metrics_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONFUSION MATRICES (3×3 grid)
# ─────────────────────────────────────────────────────────────────────────────
def make_confusion_matrices():
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), facecolor=BG)
    fig.suptitle('Confusion Matrices — All Stages & Vulnerability Types',
                 fontsize=13, fontweight='bold', color='#e2e8f0', y=0.99)

    stages = ['S3', 'S6', 'S7+8']
    stage_titles = {'S3': 'Stage 3 (GAT)', 'S6': 'Stage 6 (VQC)', 'S7+8': 'Stage 7+8 (Hybrid)'}
    vtypes = ['BO', 'FS', 'UAF']
    vnames = {'BO': 'Buffer Overflow', 'FS': 'Format String', 'UAF': 'Use-After-Free'}

    for row, vt in enumerate(vtypes):
        cmap = LinearSegmentedColormap.from_list('',
            ['#1a1a2e', COLORS[vt]], N=256)
        for col, stage in enumerate(stages):
            ax = axes[row][col]
            d = DATA[vt][stage]
            cm = np.array([[d['tn'], d['fp']], [d['fn'], d['tp']]])
            total = cm.sum()
            cm_norm = cm / total

            im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect='auto')

            labels = [['TN', 'FP'], ['FN', 'TP']]
            for i in range(2):
                for j in range(2):
                    val = cm[i, j]
                    pct = cm_norm[i, j]*100
                    brightness = cm_norm[i, j]
                    txt_col = '#0f0f1a' if brightness > 0.5 else '#e2e8f0'
                    ax.text(j, i, f'{labels[i][j]}\n{val}\n({pct:.1f}%)',
                            ha='center', va='center', fontsize=8.5,
                            color=txt_col, fontweight='bold')

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Non-Vuln', 'Vuln'], fontsize=7.5)
            ax.set_yticklabels(['Non-Vuln', 'Vuln'], fontsize=7.5)
            ax.set_xlabel('Predicted', fontsize=8, color='#94a3b8')
            ax.set_ylabel('Actual', fontsize=8, color='#94a3b8')

            f1 = d['f1']; au = d['auc']
            title_col = COLORS[vt] if col == 0 else '#e2e8f0'
            prefix = f'{vnames[vt]}\n' if col == 0 else ''
            ax.set_title(f'{prefix}{stage_titles[stage]}\nF1={f1:.4f}  AUC={au:.4f}',
                         fontsize=8, color=title_col,
                         fontweight='bold' if col==0 else 'normal', pad=4)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT, 'confusion_matrices.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Delete old PNGs and generate all 5
# ─────────────────────────────────────────────────────────────────────────────
import glob as _glob
old = _glob.glob(os.path.join(OUT, '*.png'))
for f in old:
    os.remove(f)
    print(f'  deleted: {f}')

print('\nGenerating charts...')
make_training_curves()
make_pipeline_progression()
make_metrics_table()
make_metrics_comparison()
make_confusion_matrices()
print('\nDone. All 5 charts saved to:', OUT)
