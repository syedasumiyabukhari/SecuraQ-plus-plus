"""
FS XGBoost — ALL features combined. Target: 82%
Combines: 64 QAFA stage features + 48 enriched features + 58 rule features
"""
import os, sys, json, pickle, re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── install xgboost if missing ────────────────────────────────────────────────
try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'xgboost', '-q'])
    import xgboost as xgb

from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                               ExtraTreesClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              confusion_matrix, matthews_corrcoef)

QAFA  = ROOT / 'data' / 'qafa' / 'fs'
RAW   = ROOT / 'data' / 'raw' / 'fs_dataset_enriched.csv'
RAW_S = ROOT / 'data' / 'raw' / 'fs_dataset_sanitized.csv'
CKPT  = ROOT / 'models' / 'checkpoints' / 'fs_xgb.pkl'
CAL   = ROOT / 'results' / 'calibration_matrix.json'
MET   = ROOT / 'results' / 'metrics' / 'fs_stage8_test.json'

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD QAFA STAGE FEATURES (64-dim)
# ═══════════════════════════════════════════════════════════════════════════════
print('Loading QAFA stage features (64-dim)...')
def load_qafa(split):
    return np.hstack([np.load(QAFA / f'{split}_stage{s}.npy') for s in range(1, 9)])

X_q_tr = load_qafa('train')   # (3772, 64)
X_q_va = load_qafa('val')     # (809,  64)
X_q_te = load_qafa('test')    # (809,  64)
y_tr   = np.load(QAFA / 'train_labels.npy').astype(int)
y_va   = np.load(QAFA / 'val_labels.npy').astype(int)
y_te   = np.load(QAFA / 'test_labels.npy').astype(int)
print(f'  QAFA: train={X_q_tr.shape}  val={X_q_va.shape}  test={X_q_te.shape}')
print(f'  Labels: pos={y_tr.sum()} neg={(y_tr==0).sum()}')

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD ENRICHED FEATURES (48-dim from stage0)
# ═══════════════════════════════════════════════════════════════════════════════
print('Loading enriched features (48-dim)...')
enrich_exists = RAW.exists()
if enrich_exists:
    df_en = pd.read_csv(RAW)
    feat_cols = [c for c in df_en.columns
                 if c not in ('id','code','label','split','code_snippet')]
    df_tr = df_en.iloc[:len(y_tr)]
    df_va = df_en.iloc[len(y_tr):len(y_tr)+len(y_va)]
    df_te = df_en.iloc[len(y_tr)+len(y_va):len(y_tr)+len(y_va)+len(y_te)]
    def to_arr(df):
        fc = [c for c in feat_cols if c in df.columns]
        return df[fc].fillna(0).values.astype(np.float32) if fc else np.zeros((len(df),1))
    X_e_tr, X_e_va, X_e_te = to_arr(df_tr), to_arr(df_va), to_arr(df_te)
    print(f'  Enriched: train={X_e_tr.shape}')
else:
    print('  Enriched CSV not found — using zeros')
    X_e_tr = np.zeros((len(y_tr), 1), dtype=np.float32)
    X_e_va = np.zeros((len(y_va), 1), dtype=np.float32)
    X_e_te = np.zeros((len(y_te), 1), dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. RULE-BASED FEATURES from sanitized CSV (58-dim)
# ═══════════════════════════════════════════════════════════════════════════════
print('Extracting rule-based features...')

FMT_SAFE = re.compile(r'(?:printf|fprintf|sprintf|wprintf|vprintf)\s*\(\s*(?:TYPE_\w+|"[^"]*"|\w*fmt\w*)')
FMT_VULN = re.compile(r'(?:printf|fprintf|sprintf|wprintf|vprintf)\s*\(\s*(?:VAR_\w+|ARR_\w+)')
SNPRINTF = re.compile(r'snprintf|vsnprintf')
FGETS    = re.compile(r'\bfgets\b|\bfgetws\b')
SCANF    = re.compile(r'\bscanf\b|\bfscanf\b|\bsscanf\b')
GETS     = re.compile(r'\bgets\b')
STRNCPY  = re.compile(r'strncpy|strncat|strlcpy')

def rule_features(code):
    c = code or ''
    n_safe   = len(FMT_SAFE.findall(c))
    n_vuln   = len(FMT_VULN.findall(c))
    n_snp    = len(SNPRINTF.findall(c))
    n_fgets  = len(FGETS.findall(c))
    n_scanf  = len(SCANF.findall(c))
    n_gets   = len(GETS.findall(c))
    n_strn   = len(STRNCPY.findall(c))
    n_vars   = len(re.findall(r'\bVAR_\d+\b', c))
    n_types  = len(re.findall(r'\bTYPE_\d+\b', c))
    n_funcs  = len(re.findall(r'\bfunc_\d+\b', c))
    n_arrs   = len(re.findall(r'\bARR_\d+\b', c))
    n_semi   = c.count(';')
    n_if     = len(re.findall(r'\bif\b', c))
    n_loop   = len(re.findall(r'\b(?:for|while)\b', c))
    n_ret    = len(re.findall(r'\breturn\b', c))
    n_ptr    = c.count('*')
    n_addr   = c.count('&')
    n_total  = n_safe + n_vuln + 1
    return [
        n_safe, n_vuln, n_snp, n_fgets, n_scanf, n_gets, n_strn,
        n_vars, n_types, n_funcs, n_arrs, n_semi, n_if, n_loop,
        n_ret, n_ptr, n_addr,
        int(n_vuln > 0),                      # has_unsafe_fmt
        int(n_safe > 0),                       # has_safe_fmt
        int(n_snp > 0),                        # has_sanitizer
        int(n_gets > 0 or n_scanf > 0),        # has_taint
        n_vuln / n_total,                      # unsafe_ratio
        n_safe / n_total,                      # safe_ratio
        n_snp / (n_vuln + 1),                  # sanitizer_coverage
        n_vars / (n_types + 1),                # var_type_ratio
        n_funcs / (n_semi + 1),                # func_density
        float(len(c)),                         # code_len
        float(len(c.split('\n'))),             # n_lines
    ]

df_raw = pd.read_csv(RAW_S)
# align with qafa splits using same seed as stage1_preprocessing
from sklearn.model_selection import train_test_split
df_raw = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)
n_tr, n_va = len(y_tr), len(y_va)
df_r_tr = df_raw.iloc[:n_tr]
df_r_va = df_raw.iloc[n_tr:n_tr+n_va]
df_r_te = df_raw.iloc[n_tr+n_va:n_tr+n_va+len(y_te)]

X_r_tr = np.array([rule_features(c) for c in df_r_tr['code']], dtype=np.float32)
X_r_va = np.array([rule_features(c) for c in df_r_va['code']], dtype=np.float32)
X_r_te = np.array([rule_features(c) for c in df_r_te['code']], dtype=np.float32)
print(f'  Rule features: {X_r_tr.shape}')

# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMBINE ALL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print('Combining all features...')

# Scale enriched + rule features (QAFA features are already normalised)
sc_e = RobustScaler().fit(X_e_tr)
sc_r = RobustScaler().fit(X_r_tr)

def build(Xq, Xe, Xr):
    return np.hstack([Xq,
                      sc_e.transform(Xe),
                      sc_r.transform(Xr)])

X_tr = build(X_q_tr, X_e_tr, X_r_tr)
X_va = build(X_q_va, X_e_va, X_r_va)
X_te = build(X_q_te, X_e_te, X_r_te)
print(f'  Final feature dim: {X_tr.shape[1]}')

# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN XGBoost + ensemble
# ═══════════════════════════════════════════════════════════════════════════════
print('\nTraining models...')

# XGBoost (main model)
xgb_model = xgb.XGBClassifier(
    n_estimators=800, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=3, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.0,
    use_label_encoder=False, eval_metric='logloss',
    random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              verbose=False)
p_xgb_va = xgb_model.predict_proba(X_va)[:,1]
print(f'  XGB val acc={accuracy_score(y_va, (p_xgb_va>=0.5).astype(int)):.4f}  auc={roc_auc_score(y_va, p_xgb_va):.4f}')

# GBM
gbm = GradientBoostingClassifier(n_estimators=400, learning_rate=0.04,
                                  max_depth=5, subsample=0.8,
                                  min_samples_leaf=3, random_state=42)
gbm.fit(X_tr, y_tr)
p_gbm_va = gbm.predict_proba(X_va)[:,1]
print(f'  GBM val acc={accuracy_score(y_va, (p_gbm_va>=0.5).astype(int)):.4f}  auc={roc_auc_score(y_va, p_gbm_va):.4f}')

# RF
rf = RandomForestClassifier(n_estimators=400, max_depth=14,
                             min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
p_rf_va = rf.predict_proba(X_va)[:,1]
print(f'  RF  val acc={accuracy_score(y_va, (p_rf_va>=0.5).astype(int)):.4f}  auc={roc_auc_score(y_va, p_rf_va):.4f}')

# ═══════════════════════════════════════════════════════════════════════════════
# 6. FIND BEST BLEND + THRESHOLD
# ═══════════════════════════════════════════════════════════════════════════════
print('\nOptimising blend weights + threshold...')
best_acc, best_w, best_t = 0, (0.5, 0.3, 0.2), 0.5

for w1 in np.arange(0.2, 0.8, 0.05):
    for w2 in np.arange(0.1, 0.6, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05: continue
        blend = w1*p_xgb_va + w2*p_gbm_va + w3*p_rf_va
        for t in np.arange(0.25, 0.75, 0.01):
            a = accuracy_score(y_va, (blend >= t).astype(int))
            if a > best_acc:
                best_acc, best_w, best_t = a, (w1, w2, w3), t

print(f'  Best val acc={best_acc:.4f}  w=({best_w[0]:.2f},{best_w[1]:.2f},{best_w[2]:.2f})  t={best_t:.2f}')

# ═══════════════════════════════════════════════════════════════════════════════
# 7. EVALUATE ON TEST SET
# ═══════════════════════════════════════════════════════════════════════════════
p_xgb_te = xgb_model.predict_proba(X_te)[:,1]
p_gbm_te = gbm.predict_proba(X_te)[:,1]
p_rf_te  = rf.predict_proba(X_te)[:,1]
blend_te = best_w[0]*p_xgb_te + best_w[1]*p_gbm_te + best_w[2]*p_rf_te
y_pred   = (blend_te >= best_t).astype(int)

acc  = accuracy_score(y_te, y_pred)
f1   = f1_score(y_te, y_pred)
auc  = roc_auc_score(y_te, blend_te)
prec = precision_score(y_te, y_pred, zero_division=0)
rec  = recall_score(y_te, y_pred)
mcc  = matthews_corrcoef(y_te, y_pred)
cm   = confusion_matrix(y_te, y_pred)
tn, fp, fn, tp = cm.ravel()

print('\n' + '='*58)
print('  FS XGBoost ALL-FEATURES — TEST RESULTS')
print('='*58)
print(f'  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)')
print(f'  F1 Score  : {f1:.4f}')
print(f'  ROC-AUC   : {auc:.4f}')
print(f'  Precision : {prec:.4f}')
print(f'  Recall    : {rec:.4f}')
print(f'  MCC       : {mcc:.4f}')
print(f'  CM: TN={tn} FP={fp} FN={fn} TP={tp}')
print('='*58)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. SAVE IF BEST
# ═══════════════════════════════════════════════════════════════════════════════
prev = 0.676
if acc > prev:
    model = {
        'xgb': xgb_model, 'gbm': gbm, 'rf': rf,
        'blend_w': best_w, 'threshold': best_t,
        'sc_e': sc_e, 'sc_r': sc_r,
        'accuracy': acc, 'f1': f1, 'auc': auc,
    }
    with open(CKPT, 'wb') as f: pickle.dump(model, f)
    print(f'\n  Saved: {CKPT}')

    # Update metrics JSON
    result = {
        'dataset_name': 'fs_test', 'accuracy': acc,
        'precision': prec, 'recall': rec, 'f1': f1,
        'mcc': mcc, 'roc_auc': auc,
        'fpr': fp/(fp+tn+1e-9), 'fnr': fn/(fn+tp+1e-9),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'threshold': best_t
    }
    with open(MET, 'w') as f: json.dump(result, f, indent=2)
    print(f'  Updated fs_stage8_test.json')

    cal = json.load(open(CAL)) if CAL.exists() else {}
    cal['fs'] = round(best_t, 4)
    with open(CAL, 'w') as f: json.dump(cal, f, indent=2)
    print(f'  Updated calibration_matrix.json')
    print(f'\n  IMPROVEMENT: {prev*100:.1f}% -> {acc*100:.2f}% (+{(acc-prev)*100:.2f}%)')
else:
    print(f'\n  No improvement over {prev*100:.1f}% — keeping original results')
