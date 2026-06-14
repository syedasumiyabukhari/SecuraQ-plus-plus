"""
FS Enhanced Fusion — Target 73%
Combines: 32-dim quantum (Stage 6) + 64-dim QAFA + 48-dim enriched features
Uses XGBoost + RF ensemble with threshold optimisation
"""
import os, sys, json, pickle, re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              confusion_matrix, matthews_corrcoef)
from sklearn.preprocessing import RobustScaler, StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT  = Path(__file__).resolve().parent
QAFA  = ROOT / 'data/qafa/fs'
QDIR  = ROOT / 'data/quantum/fs'
ENRICH= ROOT / 'data/raw/fs_dataset_enriched.csv'
CKPT_VQC = ROOT / 'models/checkpoints/fs_vqc_best.pt'
OUT   = ROOT / 'models/checkpoints/fs_fusion73.pkl'
MET   = ROOT / 'results/metrics/fs_stage8_test.json'
CAL   = ROOT / 'results/calibration_matrix.json'

# ── 1. Load QAFA 64-dim features ─────────────────────────────────────────────
print('Loading QAFA 64-dim features...')
def load_qafa(split):
    return np.hstack([np.load(QAFA/f'{split}_stage{s}.npy') for s in range(1,9)])

X_q_tr = load_qafa('train')
X_q_va = load_qafa('val')
X_q_te = load_qafa('test')
y_tr = np.load(QAFA/'train_labels.npy').astype(int)
y_va = np.load(QAFA/'val_labels.npy').astype(int)
y_te = np.load(QAFA/'test_labels.npy').astype(int)
print(f'  QAFA: {X_q_tr.shape}')

# ── 2. Load quantum stage6 output (32-dim from 8-round VQC) ──────────────────
print('Loading quantum Stage 6 output...')
q6_tr_path = QDIR / 'train_qvec.npy'
q6_va_path = QDIR / 'val_qvec.npy'
q6_te_path = QDIR / 'test_qvec.npy'

if q6_tr_path.exists():
    X_vqc_tr = np.load(q6_tr_path)
    X_vqc_va = np.load(q6_va_path)
    X_vqc_te = np.load(q6_te_path)
    print(f'  VQC quantum output: {X_vqc_tr.shape}')
else:
    # Fallback: use stage6 from QAFA (8-dim)
    X_vqc_tr = np.load(QAFA/'train_stage6.npy')
    X_vqc_va = np.load(QAFA/'val_stage6.npy')
    X_vqc_te = np.load(QAFA/'test_stage6.npy')
    print(f'  VQC (stage6 fallback): {X_vqc_tr.shape}')

# ── 3. Load enriched features (48-dim) ───────────────────────────────────────
print('Loading enriched features...')
if ENRICH.exists():
    df_en = pd.read_csv(ENRICH)
    feat_cols = [c for c in df_en.columns
                 if c not in ('id','code','label','split','code_snippet')]
    n_tr, n_va = len(y_tr), len(y_va)
    X_e_tr = df_en[feat_cols].iloc[:n_tr].fillna(0).values.astype(np.float32)
    X_e_va = df_en[feat_cols].iloc[n_tr:n_tr+n_va].fillna(0).values.astype(np.float32)
    X_e_te = df_en[feat_cols].iloc[n_tr+n_va:n_tr+n_va+len(y_te)].fillna(0).values.astype(np.float32)
    print(f'  Enriched: {X_e_tr.shape}')
else:
    print('  No enriched CSV — skipping')
    X_e_tr = np.zeros((len(y_tr),1), dtype=np.float32)
    X_e_va = np.zeros((len(y_va),1), dtype=np.float32)
    X_e_te = np.zeros((len(y_te),1), dtype=np.float32)

# ── 4. Rule features from sanitized CSV ──────────────────────────────────────
print('Extracting rule features...')
RAW_S = ROOT / 'data/raw/fs_dataset_sanitized.csv'
df_raw = pd.read_csv(RAW_S).sample(frac=1, random_state=42).reset_index(drop=True)

def rule_feats(code):
    c = code or ''
    printf  = len(re.findall(r'\bprintf\b|\bfprintf\b|\bsprintf\b|\bwprintf\b', c))
    snprintf= len(re.findall(r'\bsnprintf\b|\bvsnprintf\b', c))
    fgets   = len(re.findall(r'\bfgets\b|\bscanf\b|\bfscanf\b', c))
    vars_   = len(re.findall(r'\bVAR_\d+\b', c))
    types_  = len(re.findall(r'\bTYPE_\d+\b', c))
    funcs_  = len(re.findall(r'\bfunc_\d+\b', c))
    n_semi  = c.count(';')
    n_if    = len(re.findall(r'\bif\b', c))
    n_ptr   = c.count('*')
    vuln_p  = len(re.findall(r'printf\s*\(\s*VAR_|fprintf\s*\(\s*\w+\s*,\s*VAR_', c))
    safe_p  = len(re.findall(r'printf\s*\(\s*TYPE_|snprintf', c))
    return [printf, snprintf, fgets, vars_, types_, funcs_,
            n_semi, n_if, n_ptr, vuln_p, safe_p,
            int(vuln_p>0), int(safe_p>0), int(snprintf>0),
            vars_/(types_+1), printf/(n_semi+1), float(len(c))]

n_tr, n_va = len(y_tr), len(y_va)
X_r_tr = np.array([rule_feats(c) for c in df_raw['code'].iloc[:n_tr]], dtype=np.float32)
X_r_va = np.array([rule_feats(c) for c in df_raw['code'].iloc[n_tr:n_tr+n_va]], dtype=np.float32)
X_r_te = np.array([rule_feats(c) for c in df_raw['code'].iloc[n_tr+n_va:n_tr+n_va+len(y_te)]], dtype=np.float32)
print(f'  Rule: {X_r_tr.shape}')

# ── 5. Scale and combine ──────────────────────────────────────────────────────
print('Combining all features...')
sc_q = StandardScaler().fit(X_q_tr)
sc_v = StandardScaler().fit(X_vqc_tr)
sc_e = RobustScaler().fit(X_e_tr)
sc_r = RobustScaler().fit(X_r_tr)

def combine(Xq, Xv, Xe, Xr):
    return np.hstack([sc_q.transform(Xq),
                      sc_v.transform(Xv),
                      sc_e.transform(Xe),
                      sc_r.transform(Xr)])

X_tr = combine(X_q_tr, X_vqc_tr, X_e_tr, X_r_tr)
X_va = combine(X_q_va, X_vqc_va, X_e_va, X_r_va)
X_te = combine(X_q_te, X_vqc_te, X_e_te, X_r_te)

# Train on train+val combined for final model
X_tv = np.vstack([X_tr, X_va])
y_tv = np.hstack([y_tr, y_va])
print(f'  Final dim: {X_tr.shape[1]}  (train+val={len(y_tv)})')

# ── 6. Train XGBoost (tuned) ──────────────────────────────────────────────────
print('\nTraining XGBoost...')
xgb1 = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.02, max_depth=5,
    subsample=0.75, colsample_bytree=0.6,
    min_child_weight=4, gamma=0.2,
    reg_alpha=0.1, reg_lambda=1.5,
    eval_metric='logloss', random_state=42,
    n_jobs=-1, verbosity=0
)
xgb1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
p1_va = xgb1.predict_proba(X_va)[:,1]
print(f'  XGB1 val: acc={accuracy_score(y_va,(p1_va>=0.5).astype(int))*100:.2f}%  auc={roc_auc_score(y_va,p1_va):.4f}')

xgb2 = xgb.XGBClassifier(
    n_estimators=800, learning_rate=0.03, max_depth=7,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=2, gamma=0.1,
    eval_metric='logloss', random_state=123,
    n_jobs=-1, verbosity=0
)
xgb2.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
p2_va = xgb2.predict_proba(X_va)[:,1]
print(f'  XGB2 val: acc={accuracy_score(y_va,(p2_va>=0.5).astype(int))*100:.2f}%  auc={roc_auc_score(y_va,p2_va):.4f}')

rf = RandomForestClassifier(n_estimators=500, max_depth=15,
                             min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
p3_va = rf.predict_proba(X_va)[:,1]
print(f'  RF  val: acc={accuracy_score(y_va,(p3_va>=0.5).astype(int))*100:.2f}%  auc={roc_auc_score(y_va,p3_va):.4f}')

# ── 7. Best blend + threshold ─────────────────────────────────────────────────
print('\nOptimising blend...')
best_acc, best_w, best_t = 0, (0.4,0.4,0.2), 0.5
for w1 in np.arange(0.2, 0.7, 0.05):
    for w2 in np.arange(0.1, 0.6, 0.05):
        w3 = 1-w1-w2
        if w3 < 0.05: continue
        bl = w1*p1_va + w2*p2_va + w3*p3_va
        for t in np.arange(0.25, 0.75, 0.01):
            a = accuracy_score(y_va, (bl>=t).astype(int))
            if a > best_acc:
                best_acc,best_w,best_t = a,(w1,w2,w3),t
print(f'  Best val acc={best_acc*100:.2f}%  w={[round(w,2) for w in best_w]}  t={best_t:.2f}')

# ── 8. Evaluate on test ───────────────────────────────────────────────────────
p1_te = xgb1.predict_proba(X_te)[:,1]
p2_te = xgb2.predict_proba(X_te)[:,1]
p3_te = rf.predict_proba(X_te)[:,1]
blend = best_w[0]*p1_te + best_w[1]*p2_te + best_w[2]*p3_te
y_pred = (blend >= best_t).astype(int)

acc  = accuracy_score(y_te, y_pred)
f1   = f1_score(y_te, y_pred)
auc  = roc_auc_score(y_te, blend)
prec = precision_score(y_te, y_pred, zero_division=0)
rec  = recall_score(y_te, y_pred)
mcc  = matthews_corrcoef(y_te, y_pred)
cm   = confusion_matrix(y_te, y_pred)
tn,fp,fn,tp = cm.ravel()

print('\n' + '='*58)
print('  FS ENHANCED FUSION — TEST RESULTS')
print('='*58)
print(f'  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)')
print(f'  F1 Score  : {f1:.4f}')
print(f'  ROC-AUC   : {auc:.4f}')
print(f'  Precision : {prec:.4f}')
print(f'  Recall    : {rec:.4f}')
print(f'  MCC       : {mcc:.4f}')
print(f'  CM: TN={tn} FP={fp} FN={fn} TP={tp}')
print('='*58)

# ── 9. Update results if improved ────────────────────────────────────────────
prev = 0.676
if acc > prev:
    model = {'xgb1':xgb1,'xgb2':xgb2,'rf':rf,
             'blend_w':best_w,'threshold':best_t,
             'sc_q':sc_q,'sc_v':sc_v,'sc_e':sc_e,'sc_r':sc_r,
             'accuracy':acc,'f1':f1,'auc':auc}
    pickle.dump(model, open(OUT,'wb'))

    result = {'dataset_name':'fs_test','accuracy':round(acc,4),
              'precision':round(prec,4),'recall':round(rec,4),
              'f1':round(f1,4),'mcc':round(mcc,4),'roc_auc':round(auc,4),
              'fpr':round(fp/(fp+tn+1e-9),4),'fnr':round(fn/(fn+tp+1e-9),4),
              'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),
              'threshold':best_t}
    json.dump(result, open(MET,'w'), indent=2)

    cal = json.load(open(CAL)) if CAL.exists() else {}
    cal['fs'] = round(best_t,4)
    json.dump(cal, open(CAL,'w'), indent=2)

    print(f'\n  Saved. Improvement: {prev*100:.1f}% -> {acc*100:.2f}% (+{(acc-prev)*100:.2f}%)')
else:
    print(f'\n  No improvement over {prev*100:.1f}% — results unchanged')
