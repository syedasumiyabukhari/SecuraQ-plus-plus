"""
FS Tokenizer Classifier — Target 82%
Uses clean dataset with preserved function names (printf, scanf, etc.)
Key insight: printf(VAR_N) = vulnerable, printf(TYPE_N) = safe
"""

import re, os, json, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              confusion_matrix, matthews_corrcoef)
from sklearn.model_selection import StratifiedKFold
import scipy.sparse as sp

ROOT  = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(ROOT, 'data', 'raw', 'fs_dataset_clean.csv')
CKPT  = os.path.join(ROOT, 'models', 'checkpoints', 'fs_tokenizer.pkl')
CAL   = os.path.join(ROOT, 'results', 'calibration_matrix.json')

# ── Tokenizer ─────────────────────────────────────────────────────────────────
DANGEROUS = {
    'printf':   'FUNC_PRINTF',
    'sprintf':  'FUNC_SPRINTF',
    'fprintf':  'FUNC_FPRINTF',
    'snprintf': 'FUNC_SNPRINTF',
    'scanf':    'FUNC_SCANF',
    'fscanf':   'FUNC_FSCANF',
    'sscanf':   'FUNC_SSCANF',
    'gets':     'FUNC_GETS',
    'fgets':    'FUNC_FGETS',
    'syslog':   'FUNC_SYSLOG',
    'wprintf':  'FUNC_WPRINTF',
}
KEYWORDS = {'int','char','float','double','return','if','else','for','while',
            'void','static','const','unsigned','size_t','struct','typedef'}

def normalize(code):
    code = re.sub(r'//.*',     '',  code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code

def tokenize(code):
    code = normalize(code)
    tokens = re.findall(r'[A-Za-z_]\w*|==|!=|<=|>=|[\(\)\{\};,\[\]]|[\+\-\*/=<>!&|]+', code)
    out = []
    for t in tokens:
        if t in DANGEROUS:        out.append(DANGEROUS[t])
        elif t in KEYWORDS:       out.append(t.upper())
        elif re.match(r'VAR_',t): out.append('VAR')
        elif re.match(r'ARR_',t): out.append('ARR')
        elif re.match(r'TYPE_',t):out.append('TYPE')
        elif re.match(r'func_',t):out.append('FUNC_UNK')
        elif re.match(r'[A-Za-z_]\w*', t): out.append('ID')
        else:                     out.append(t)
    return out

def token_str(code):
    return ' '.join(tokenize(code))

# ── Rule-based features ────────────────────────────────────────────────────────
def manual_features(code):
    toks = tokenize(code)
    n = len(toks)
    feats = {}

    # Core FS signal: FUNC_PRINTF(VAR → vulnerable pattern
    feats['unsafe_printf'] = 0
    feats['safe_printf']   = 0
    feats['unsafe_scanf']  = 0
    feats['n_printf']      = 0
    feats['n_scanf']       = 0
    feats['n_fgets']       = 0
    feats['n_snprintf']    = 0

    for i, t in enumerate(toks):
        if t == 'FUNC_PRINTF':
            feats['n_printf'] += 1
            # look ahead for ( then first meaningful token
            j = i + 1
            while j < n and toks[j] in ('(', ' '): j += 1
            if j < n:
                next_tok = toks[j] if j < n else ''
                if next_tok == 'VAR':    feats['unsafe_printf'] += 1
                elif next_tok == 'TYPE': feats['safe_printf']   += 1
                elif next_tok == 'ARR':  feats['unsafe_printf'] += 1

        if t in ('FUNC_SCANF', 'FUNC_FSCANF', 'FUNC_SSCANF'):
            feats['n_scanf'] += 1
        if t == 'FUNC_FGETS':
            feats['n_fgets'] += 1
        if t == 'FUNC_SNPRINTF':
            feats['n_snprintf'] += 1

    # Structural
    feats['n_var']    = toks.count('VAR')
    feats['n_type']   = toks.count('TYPE')
    feats['n_arr']    = toks.count('ARR')
    feats['n_tokens'] = n
    feats['n_if']     = toks.count('IF')
    feats['n_return'] = toks.count('RETURN')
    feats['n_func_unk'] = toks.count('FUNC_UNK')

    # Ratios
    feats['var_type_ratio'] = feats['n_var'] / (feats['n_type'] + 1)
    feats['unsafe_ratio']   = feats['unsafe_printf'] / (feats['n_printf'] + 1)
    feats['fmt_density']    = feats['n_printf'] / (n + 1)

    # Has any dangerous call with VAR as first arg
    feats['any_unsafe_fmt'] = int(feats['unsafe_printf'] > 0)

    return list(feats.values())

FEAT_NAMES = ['unsafe_printf','safe_printf','unsafe_scanf','n_printf','n_scanf',
              'n_fgets','n_snprintf','n_var','n_type','n_arr','n_tokens','n_if',
              'n_return','n_func_unk','var_type_ratio','unsafe_ratio','fmt_density',
              'any_unsafe_fmt']

# ── Load & split ──────────────────────────────────────────────────────────────
print('Loading clean FS dataset...')
df = pd.read_csv(CLEAN)
print(f'  Total: {len(df)}  vuln={df.label.sum()}  safe={(df.label==0).sum()}')

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=42)
df_train, df_val  = train_test_split(df_train, test_size=0.12, stratify=df_train['label'], random_state=42)
print(f'  Train={len(df_train)}  Val={len(df_val)}  Test={len(df_test)}')

# ── Extract features ──────────────────────────────────────────────────────────
print('Extracting manual features...')
X_man_tr = np.array([manual_features(c) for c in df_train['code']], dtype=np.float32)
X_man_va = np.array([manual_features(c) for c in df_val['code']],   dtype=np.float32)
X_man_te = np.array([manual_features(c) for c in df_test['code']],  dtype=np.float32)

print('Building TF-IDF features...')
tok_tr = [token_str(c) for c in df_train['code']]
tok_va = [token_str(c) for c in df_val['code']]
tok_te = [token_str(c) for c in df_test['code']]

tfidf_w = TfidfVectorizer(ngram_range=(1,3), max_features=3000, min_df=2)
tfidf_c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=2000, min_df=3)

X_tw_tr = tfidf_w.fit_transform(tok_tr)
X_tw_va = tfidf_w.transform(tok_va)
X_tw_te = tfidf_w.transform(tok_te)

X_tc_tr = tfidf_c.fit_transform(tok_tr)
X_tc_va = tfidf_c.transform(tok_va)
X_tc_te = tfidf_c.transform(tok_te)

scaler = StandardScaler()
X_sc_tr = scaler.fit_transform(X_man_tr)
X_sc_va = scaler.transform(X_man_va)
X_sc_te = scaler.transform(X_man_te)

def combine(Xm, Xtw, Xtc):
    return sp.hstack([sp.csr_matrix(Xm), Xtw, Xtc]).toarray()

X_tr = combine(X_sc_tr, X_tw_tr, X_tc_tr)
X_va = combine(X_sc_va, X_tw_va, X_tc_va)
X_te = combine(X_sc_te, X_tw_te, X_tc_te)

y_tr = df_train['label'].values
y_va = df_val['label'].values
y_te = df_test['label'].values

print(f'  Feature dim: {X_tr.shape[1]}')

# ── Train base models ─────────────────────────────────────────────────────────
print('\nTraining base models...')
gbm = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                  max_depth=5, subsample=0.8, random_state=42)
rf  = RandomForestClassifier(n_estimators=300, max_depth=12,
                              min_samples_leaf=2, random_state=42, n_jobs=-1)
et  = ExtraTreesClassifier(n_estimators=300, max_depth=12,
                            min_samples_leaf=2, random_state=42, n_jobs=-1)
lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

gbm.fit(X_tr, y_tr); print(f'  GBM val acc={accuracy_score(y_va, gbm.predict(X_va)):.4f}')
rf.fit(X_tr, y_tr);  print(f'  RF  val acc={accuracy_score(y_va, rf.predict(X_va)):.4f}')
et.fit(X_tr, y_tr);  print(f'  ET  val acc={accuracy_score(y_va, et.predict(X_va)):.4f}')
lr.fit(X_tr, y_tr);  print(f'  LR  val acc={accuracy_score(y_va, lr.predict(X_va)):.4f}')

# ── Find best blend ───────────────────────────────────────────────────────────
print('\nFinding best ensemble blend...')
p_gbm = gbm.predict_proba(X_va)[:,1]
p_rf  = rf.predict_proba(X_va)[:,1]
p_et  = et.predict_proba(X_va)[:,1]
p_lr  = lr.predict_proba(X_va)[:,1]

best_acc, best_w, best_t = 0, (0.25,0.25,0.25,0.25), 0.5
for w1 in np.arange(0.1, 0.7, 0.1):
    for w2 in np.arange(0.1, 0.7, 0.1):
        for w3 in np.arange(0.1, 0.6, 0.1):
            w4 = 1 - w1 - w2 - w3
            if w4 < 0.05: continue
            blend = w1*p_gbm + w2*p_rf + w3*p_et + w4*p_lr
            for t in np.arange(0.3, 0.7, 0.02):
                acc = accuracy_score(y_va, (blend >= t).astype(int))
                if acc > best_acc:
                    best_acc, best_w, best_t = acc, (w1,w2,w3,w4), t

print(f'  Best val acc={best_acc:.4f}  w={[round(w,2) for w in best_w]}  t={best_t:.2f}')

# ── Evaluate on test ──────────────────────────────────────────────────────────
p_gbm_te = gbm.predict_proba(X_te)[:,1]
p_rf_te  = rf.predict_proba(X_te)[:,1]
p_et_te  = et.predict_proba(X_te)[:,1]
p_lr_te  = lr.predict_proba(X_te)[:,1]

blend_te = best_w[0]*p_gbm_te + best_w[1]*p_rf_te + best_w[2]*p_et_te + best_w[3]*p_lr_te
y_pred   = (blend_te >= best_t).astype(int)

acc  = accuracy_score(y_te, y_pred)
f1   = f1_score(y_te, y_pred)
auc  = roc_auc_score(y_te, blend_te)
prec = precision_score(y_te, y_pred)
rec  = recall_score(y_te, y_pred)
mcc  = matthews_corrcoef(y_te, y_pred)
cm   = confusion_matrix(y_te, y_pred)
tn,fp,fn,tp = cm.ravel()

print('\n' + '='*55)
print('  FS TOKENIZER CLASSIFIER - TEST RESULTS')
print('='*55)
print(f'  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)')
print(f'  F1 Score  : {f1:.4f}')
print(f'  ROC-AUC   : {auc:.4f}')
print(f'  Precision : {prec:.4f}')
print(f'  Recall    : {rec:.4f}')
print(f'  MCC       : {mcc:.4f}')
print(f'  CM: TN={tn} FP={fp} FN={fn} TP={tp}')
print('='*55)

# ── Save model ────────────────────────────────────────────────────────────────
model = {
    'version': 'tokenizer_v1',
    'gbm': gbm, 'rf': rf, 'et': et, 'lr': lr,
    'blend_w': best_w, 'threshold': best_t,
    'tfidf_w': tfidf_w, 'tfidf_c': tfidf_c, 'scaler': scaler,
    'accuracy': acc, 'f1': f1, 'auc': auc,
}
with open(CKPT, 'wb') as f: pickle.dump(model, f)
print(f'\n  Saved: {CKPT}')

# Update result JSON files
test_result = {
    'dataset_name': 'fs_test', 'accuracy': acc, 'precision': prec,
    'recall': rec, 'f1': f1, 'mcc': mcc, 'roc_auc': auc,
    'fpr': fp/(fp+tn), 'fnr': fn/(fn+tp),
    'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    'threshold': best_t
}

metrics_dir = os.path.join(ROOT, 'results', 'metrics')
with open(os.path.join(metrics_dir, 'fs_stage8_test.json'), 'w') as f:
    json.dump(test_result, f, indent=2)
print('  Updated fs_stage8_test.json')

# Update calibration matrix
if os.path.exists(CAL):
    with open(CAL) as f: cal = json.load(f)
else:
    cal = {}
cal['fs'] = round(best_t, 4)
with open(CAL, 'w') as f: json.dump(cal, f, indent=2)
print('  Updated calibration_matrix.json')
