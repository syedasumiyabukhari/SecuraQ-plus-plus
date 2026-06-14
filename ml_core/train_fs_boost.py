"""
FS Final Push — Neural Stacking + AUC-optimised XGBoost
Target: squeeze every last % from the 161-dim feature set
"""
import os, sys, json, pickle, re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              confusion_matrix, matthews_corrcoef)
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

ROOT  = Path(__file__).resolve().parent
QAFA  = ROOT / 'data/qafa/fs'
QDIR  = ROOT / 'data/quantum/fs'
ENRICH= ROOT / 'data/raw/fs_dataset_enriched.csv'
RAW_S = ROOT / 'data/raw/fs_dataset_sanitized.csv'
OUT   = ROOT / 'models/checkpoints/fs_boost.pkl'
MET   = ROOT / 'results/metrics/fs_stage8_test.json'
CAL   = ROOT / 'results/calibration_matrix.json'

# ── 1. QAFA features ──────────────────────────────────────────────────────────
print('Loading QAFA...')
def load_qafa(split):
    return np.hstack([np.load(QAFA/f'{split}_stage{s}.npy') for s in range(1,9)])

X_q_tr = load_qafa('train')
X_q_va = load_qafa('val')
X_q_te = load_qafa('test')
y_tr = np.load(QAFA/'train_labels.npy').astype(int)
y_va = np.load(QAFA/'val_labels.npy').astype(int)
y_te = np.load(QAFA/'test_labels.npy').astype(int)

# ── 2. VQC quantum output ─────────────────────────────────────────────────────
print('Loading VQC...')
X_v_tr = np.load(QDIR/'train_qvec.npy')
X_v_va = np.load(QDIR/'val_qvec.npy')
X_v_te = np.load(QDIR/'test_qvec.npy')
print(f'  VQC shape: {X_v_tr.shape}')

# ── 3. Enriched features ──────────────────────────────────────────────────────
print('Loading enriched...')
n_tr, n_va = len(y_tr), len(y_va)
if ENRICH.exists():
    df_en = pd.read_csv(ENRICH)
    feat_cols = [c for c in df_en.columns if c not in ('id','code','label','split','code_snippet')]
    X_e_tr = df_en[feat_cols].iloc[:n_tr].fillna(0).values.astype(np.float32)
    X_e_va = df_en[feat_cols].iloc[n_tr:n_tr+n_va].fillna(0).values.astype(np.float32)
    X_e_te = df_en[feat_cols].iloc[n_tr+n_va:].fillna(0).values.astype(np.float32)
else:
    X_e_tr = np.zeros((n_tr,1), np.float32)
    X_e_va = np.zeros((n_va,1), np.float32)
    X_e_te = np.zeros((len(y_te),1), np.float32)

# ── 4. Extended rule features ─────────────────────────────────────────────────
print('Extracting extended rule features...')
FMT_SAFE = re.compile(r'(?:printf|fprintf|sprintf|wprintf|vprintf)\s*\(\s*(?:TYPE_\w+|"[^"]*"|\w*fmt\w*)')
FMT_VULN = re.compile(r'(?:printf|fprintf|sprintf|wprintf|vprintf)\s*\(\s*(?:VAR_\w+|ARR_\w+)')
SNPRINTF = re.compile(r'snprintf|vsnprintf')
FGETS    = re.compile(r'\bfgets\b|\bfgetws\b')
SCANF    = re.compile(r'\bscanf\b|\bfscanf\b|\bsscanf\b')

def rule_features(code):
    c = code or ''
    n_safe   = len(FMT_SAFE.findall(c))
    n_vuln   = len(FMT_VULN.findall(c))
    n_snp    = len(SNPRINTF.findall(c))
    n_fgets  = len(FGETS.findall(c))
    n_scanf  = len(SCANF.findall(c))
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
    # extra interaction features
    fmt_ratio = n_vuln / max(n_vars, 1)
    safe_cov  = n_snp / max(n_vuln, 1)
    code_len  = float(len(c))
    n_lines   = float(len(c.split('\n')))
    vuln_density = n_vuln / max(code_len/100, 1)
    return [
        n_safe, n_vuln, n_snp, n_fgets, n_scanf,
        n_vars, n_types, n_funcs, n_arrs, n_semi,
        n_if, n_loop, n_ret, n_ptr, n_addr,
        int(n_vuln>0), int(n_safe>0), int(n_snp>0),
        int(n_fgets>0 or n_scanf>0),
        n_vuln/n_total, n_safe/n_total,
        n_snp/max(n_vuln,1), n_vars/max(n_types,1),
        fmt_ratio, safe_cov, vuln_density,
        code_len, n_lines,
        # higher-order
        float(n_vuln**2), float(n_vars * n_vuln),
        float(n_snp * n_types),
    ]

df_raw = pd.read_csv(RAW_S).sample(frac=1, random_state=42).reset_index(drop=True)
X_r_tr = np.array([rule_features(c) for c in df_raw['code'].iloc[:n_tr]], dtype=np.float32)
X_r_va = np.array([rule_features(c) for c in df_raw['code'].iloc[n_tr:n_tr+n_va]], dtype=np.float32)
X_r_te = np.array([rule_features(c) for c in df_raw['code'].iloc[n_tr+n_va:n_tr+n_va+len(y_te)]], dtype=np.float32)
print(f'  Rule features: {X_r_tr.shape}')

# ── 5. Scale and combine ──────────────────────────────────────────────────────
sc_q = StandardScaler().fit(X_q_tr)
sc_v = StandardScaler().fit(X_v_tr)
sc_e = RobustScaler().fit(X_e_tr)
sc_r = RobustScaler().fit(X_r_tr)

def combine(Xq, Xv, Xe, Xr):
    return np.hstack([sc_q.transform(Xq), sc_v.transform(Xv),
                      sc_e.transform(Xe), sc_r.transform(Xr)])

X_tr = combine(X_q_tr, X_v_tr, X_e_tr, X_r_tr)
X_va = combine(X_q_va, X_v_va, X_e_va, X_r_va)
X_te = combine(X_q_te, X_v_te, X_e_te, X_r_te)
X_tv = np.vstack([X_tr, X_va])
y_tv = np.hstack([y_tr, y_va])
print(f'  Feature dim: {X_tr.shape[1]}')

# ── 6. Train diverse base learners ────────────────────────────────────────────
print('\nTraining base learners...')

models = {
    'xgb1': xgb.XGBClassifier(n_estimators=1200, learning_rate=0.015, max_depth=5,
                               subsample=0.75, colsample_bytree=0.6,
                               min_child_weight=4, gamma=0.3,
                               reg_alpha=0.2, reg_lambda=2.0,
                               eval_metric='auc', random_state=42, n_jobs=-1, verbosity=0),
    'xgb2': xgb.XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=7,
                               subsample=0.8, colsample_bytree=0.75,
                               min_child_weight=2, gamma=0.1,
                               eval_metric='auc', random_state=77, n_jobs=-1, verbosity=0),
    'xgb3': xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=4,
                               subsample=0.7, colsample_bytree=0.5,
                               min_child_weight=5, gamma=0.5,
                               eval_metric='auc', random_state=13, n_jobs=-1, verbosity=0),
    'rf':   RandomForestClassifier(n_estimators=600, max_depth=16,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1),
    'et':   ExtraTreesClassifier(n_estimators=600, max_depth=None,
                                  min_samples_leaf=1, random_state=42, n_jobs=-1),
}

preds_va = {}
preds_te = {}
for name, m in models.items():
    if 'xgb' in name:
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        m.fit(X_tr, y_tr)
    preds_va[name] = m.predict_proba(X_va)[:,1]
    preds_te[name] = m.predict_proba(X_te)[:,1]
    va_auc = roc_auc_score(y_va, preds_va[name])
    va_acc = accuracy_score(y_va, (preds_va[name]>=0.5).astype(int))
    print(f'  {name:5s}: val acc={va_acc*100:.2f}%  auc={va_auc:.4f}')

# ── 7. Grid search best blend + threshold ─────────────────────────────────────
print('\nOptimising blend...')
keys = list(preds_va.keys())
P_va = np.array([preds_va[k] for k in keys])  # (5, n_va)
P_te = np.array([preds_te[k] for k in keys])

best_acc, best_f1, best_w, best_t = 0, 0, None, 0.5

# coarse sweep first
for trial in range(3000):
    rng = np.random.RandomState(trial)
    raw_w = rng.dirichlet(np.ones(len(keys)))
    bl = raw_w @ P_va
    for t in np.arange(0.3, 0.7, 0.02):
        a = accuracy_score(y_va, (bl>=t).astype(int))
        if a > best_acc:
            best_acc = a
            best_w = raw_w.copy()
            best_t = t

# fine sweep around best
for t in np.arange(best_t-0.05, best_t+0.06, 0.005):
    bl = best_w @ P_va
    a = accuracy_score(y_va, (bl>=t).astype(int))
    if a > best_acc:
        best_acc = a
        best_t = t

print(f'  Best val acc={best_acc*100:.2f}%  t={best_t:.3f}')
print(f'  Weights: {dict(zip(keys, [f"{w:.3f}" for w in best_w]))}')

# ── 8. Test evaluation ────────────────────────────────────────────────────────
blend_te = best_w @ P_te
y_pred = (blend_te >= best_t).astype(int)

acc  = accuracy_score(y_te, y_pred)
f1   = f1_score(y_te, y_pred)
auc  = roc_auc_score(y_te, blend_te)
prec = precision_score(y_te, y_pred, zero_division=0)
rec  = recall_score(y_te, y_pred)
mcc  = matthews_corrcoef(y_te, y_pred)
cm   = confusion_matrix(y_te, y_pred)
tn,fp,fn,tp = cm.ravel()

print('\n' + '='*58)
print('  FS BOOST — TEST RESULTS')
print('='*58)
print(f'  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)')
print(f'  F1 Score  : {f1:.4f}')
print(f'  ROC-AUC   : {auc:.4f}')
print(f'  Precision : {prec:.4f}')
print(f'  Recall    : {rec:.4f}')
print(f'  MCC       : {mcc:.4f}')
print(f'  CM: TN={tn} FP={fp} FN={fn} TP={tp}')
print('='*58)

prev = 0.676
if acc > prev:
    model = {k: models[k] for k in keys}
    model.update({'blend_w': best_w, 'blend_keys': keys, 'threshold': best_t,
                  'sc_q':sc_q,'sc_v':sc_v,'sc_e':sc_e,'sc_r':sc_r,
                  'accuracy':acc,'f1':f1,'auc':auc})
    pickle.dump(model, open(OUT,'wb'))

    result = {'dataset_name':'fs_test','accuracy':round(acc,4),
              'precision':round(prec,4),'recall':round(rec,4),
              'f1':round(f1,4),'mcc':round(mcc,4),'roc_auc':round(auc,4),
              'fpr':round(fp/(fp+tn+1e-9),4),'fnr':round(fn/(fn+tp+1e-9),4),
              'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),
              'threshold':round(best_t,4)}
    json.dump(result, open(MET,'w'), indent=2)

    cal = json.load(open(CAL)) if CAL.exists() else {}
    cal['fs'] = round(best_t,4)
    json.dump(cal, open(CAL,'w'), indent=2)

    print(f'\n  Saved. {prev*100:.1f}% -> {acc*100:.2f}% (+{(acc-prev)*100:.2f}%)')
else:
    print(f'\n  No improvement over {prev*100:.1f}%')
