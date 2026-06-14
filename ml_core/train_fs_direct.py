"""
FS Direct Classifier v2  — Maximum Accuracy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bypasses broken QAFA pipeline.
Combines smart tokenisation + TF-IDF n-grams + manual features + stacking.

Run:
    cd ml_core
    python train_fs_direct.py
"""

import json, os, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import (HistGradientBoostingClassifier,
                               RandomForestClassifier,
                               ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              confusion_matrix, matthews_corrcoef)

ROOT   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(ROOT, "data", "processed", "fs")
CKPT   = os.path.join(ROOT, "models", "checkpoints", "fs_direct.pkl")
CAL_IN = os.path.join(ROOT, "results", "calibration_matrix.json")
os.makedirs(os.path.dirname(CKPT), exist_ok=True)

# ── Feature extraction — single source of truth ───────────────────────────────
from fs_features import (
    tokenize_code, token_string,
    build_manual_features, build_manual_matrix,
)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading processed CSV data …")
train_df = pd.read_csv(os.path.join(DATA, "train.csv"))
val_df   = pd.read_csv(os.path.join(DATA, "val.csv"))
test_df  = pd.read_csv(os.path.join(DATA, "test.csv"))
# Merge train + val for final training
full_df  = pd.concat([train_df, val_df], ignore_index=True)

print(f"  train {len(train_df)}  val {len(val_df)}  test {len(test_df)}")
print(f"  label balance  full: {int(full_df.label.sum())}/{len(full_df)}")

# ── Build manual feature matrices ─────────────────────────────────────────────
print("Extracting manual + rule features …")
X_train_m, y_train_m, feat_names = build_manual_matrix(train_df)
X_val_m,   y_val_m,   _          = build_manual_matrix(val_df)
X_test_m,  y_test_m,  _          = build_manual_matrix(test_df)
X_full_m,  y_full_m,  _          = build_manual_matrix(full_df)
print(f"  manual feature dim: {X_train_m.shape[1]}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_m)
X_val_s   = scaler.transform(X_val_m)
X_test_s  = scaler.transform(X_test_m)
X_full_s  = scaler.fit_transform(X_full_m)   # refit on full for deployment

# ── Build TF-IDF token matrices ───────────────────────────────────────────────
print("Building TF-IDF token features (word 1-3-gram + char 3-6-gram) …")
train_tok = [token_string(c) for c in train_df['code']]
val_tok   = [token_string(c) for c in val_df['code']]
test_tok  = [token_string(c) for c in test_df['code']]
full_tok  = [token_string(c) for c in full_df['code']]

# Word n-gram TF-IDF
tfidf_w = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 3),
    max_features=5000, min_df=2, sublinear_tf=True,
)
# Character n-gram TF-IDF (captures micro-patterns in token names)
tfidf_c = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 6),
    max_features=3000, min_df=3, sublinear_tf=True,
)
X_train_tw = tfidf_w.fit_transform(train_tok)
X_train_tc = tfidf_c.fit_transform(train_tok)
X_val_tw   = tfidf_w.transform(val_tok)
X_val_tc   = tfidf_c.transform(val_tok)
X_test_tw  = tfidf_w.transform(test_tok)
X_test_tc  = tfidf_c.transform(test_tok)

X_train_tf = sp.hstack([X_train_tw, X_train_tc])
X_val_tf   = sp.hstack([X_val_tw,   X_val_tc])
X_test_tf  = sp.hstack([X_test_tw,  X_test_tc])
print(f"  TF-IDF dim (word+char): {X_train_tf.shape[1]}")

# Select top-400 TF-IDF features by chi2 (reduces noise for LR)
sel = SelectKBest(chi2, k=min(400, X_train_tf.shape[1]))
X_train_tfs = sel.fit_transform(X_train_tf, y_train_m)
X_val_tfs   = sel.transform(X_val_tf)
X_test_tfs  = sel.transform(X_test_tf)

# Dense combined matrix for HistGBM (manual + selected TF-IDF)
X_train_dense = np.hstack([X_train_s, X_train_tfs.toarray()])
X_val_dense   = np.hstack([X_val_s,   X_val_tfs.toarray()])
X_test_dense  = np.hstack([X_test_s,  X_test_tfs.toarray()])
print(f"  Combined dense dim: {X_train_dense.shape[1]}")

# Combined sparse (for LogisticRegression)
X_train_comb = sp.hstack([X_train_tf, sp.csr_matrix(X_train_s)])
X_val_comb   = sp.hstack([X_val_tf,   sp.csr_matrix(X_val_s)])
X_test_comb  = sp.hstack([X_test_tf,  sp.csr_matrix(X_test_s)])

# ── Base model 1: HistGradientBoosting on manual + TF-IDF (dense) ────────────
print("\nTraining base models …")
gbm = HistGradientBoostingClassifier(
    max_iter=800,
    learning_rate=0.02,
    max_depth=8,
    min_samples_leaf=8,
    l2_regularization=0.05,
    max_bins=127,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,
    scoring="f1",
)
gbm.fit(X_train_dense, y_train_m)
gbm_val_p = gbm.predict_proba(X_val_dense)[:, 1]
print(f"  GBM   val acc={accuracy_score(y_val_m, gbm.predict(X_val_dense)):.4f}  "
      f"auc={roc_auc_score(y_val_m, gbm_val_p):.4f}")

# ── Base model 2: RandomForest on dense combined features ────────────────────
rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=None,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_dense, y_train_m)
rf_val_p = rf.predict_proba(X_val_dense)[:, 1]
print(f"  RF    val acc={accuracy_score(y_val_m, rf.predict(X_val_dense)):.4f}  "
      f"auc={roc_auc_score(y_val_m, rf_val_p):.4f}")

# ── Base model 3: ExtraTrees on manual-only features (adds diversity) ─────────
et = ExtraTreesClassifier(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
et.fit(X_train_s, y_train_m)
et_val_p = et.predict_proba(X_val_s)[:, 1]
print(f"  ET    val acc={accuracy_score(y_val_m, et.predict(X_val_s)):.4f}  "
      f"auc={roc_auc_score(y_val_m, et_val_p):.4f}")

# ── Base model 4: LogisticRegression on TF-IDF + manual (liblinear) ──────────
lr_tfidf = LogisticRegression(C=1.0, solver='liblinear', max_iter=5000,
                               class_weight='balanced', random_state=42)
lr_tfidf.fit(X_train_comb, y_train_m)
lr_val_p = lr_tfidf.predict_proba(X_val_comb)[:, 1]
print(f"  LR    val acc={accuracy_score(y_val_m, lr_tfidf.predict(X_val_comb)):.4f}  "
      f"auc={roc_auc_score(y_val_m, lr_val_p):.4f}")

# ── Stacking: 5-fold OOF meta-features ───────────────────────────────────────
print("\nBuilding stacking meta-features (5-fold OOF) …")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_all_s      = np.vstack([X_train_s,     X_val_s])
X_all_dense  = np.vstack([X_train_dense, X_val_dense])
X_all_tf     = sp.vstack([X_train_tf,    X_val_tf])
y_all        = np.concatenate([y_train_m, y_val_m])

oof_gbm = np.zeros(len(y_all))
oof_rf  = np.zeros(len(y_all))
oof_et  = np.zeros(len(y_all))
oof_lr  = np.zeros(len(y_all))

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_all_s, y_all)):
    gbm_f = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.025, max_depth=7,
        min_samples_leaf=8, l2_regularization=0.05,
        max_bins=127, random_state=42)
    gbm_f.fit(X_all_dense[tr_idx], y_all[tr_idx])
    oof_gbm[vl_idx] = gbm_f.predict_proba(X_all_dense[vl_idx])[:, 1]

    rf_f = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1)
    rf_f.fit(X_all_dense[tr_idx], y_all[tr_idx])
    oof_rf[vl_idx] = rf_f.predict_proba(X_all_dense[vl_idx])[:, 1]

    et_f = ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1)
    et_f.fit(X_all_s[tr_idx], y_all[tr_idx])
    oof_et[vl_idx] = et_f.predict_proba(X_all_s[vl_idx])[:, 1]

    X_cb_tr = sp.hstack([X_all_tf[tr_idx], sp.csr_matrix(X_all_s[tr_idx])])
    X_cb_vl = sp.hstack([X_all_tf[vl_idx], sp.csr_matrix(X_all_s[vl_idx])])
    lr_f = LogisticRegression(C=1.0, solver='liblinear', max_iter=5000,
                               class_weight='balanced', random_state=42)
    lr_f.fit(X_cb_tr, y_all[tr_idx])
    oof_lr[vl_idx] = lr_f.predict_proba(X_cb_vl)[:, 1]

    print(f"  Fold {fold+1}/5 done")

# ── Meta blend: weighted average of OOF probabilities ────────────────────────
# Simple ensemble avoids meta-learner overfitting on small data
print("Finding optimal OOF blend weights …")
# OOF AUC per model
from sklearn.metrics import roc_auc_score as _auc
auc_gbm = _auc(y_all, oof_gbm)
auc_rf  = _auc(y_all, oof_rf)
auc_et  = _auc(y_all, oof_et)
auc_lr  = _auc(y_all, oof_lr)
print(f"  OOF AUC: GBM={auc_gbm:.4f} RF={auc_rf:.4f} ET={auc_et:.4f} LR={auc_lr:.4f}")

# Weights proportional to AUC (higher AUC → more weight)
w_total = auc_gbm + auc_rf + auc_et + auc_lr
w_gbm, w_rf, w_et, w_lr = auc_gbm/w_total, auc_rf/w_total, auc_et/w_total, auc_lr/w_total

oof_blend = w_gbm*oof_gbm + w_rf*oof_rf + w_et*oof_et + w_lr*oof_lr

# Find optimal threshold on OOF blend
best_t_oof, best_oof_score = 0.5, 0.0
for t in np.arange(0.10, 0.90, 0.005):
    pred = (oof_blend > t).astype(int)
    sc   = accuracy_score(y_all, pred)
    if sc > best_oof_score:
        best_oof_score, best_t_oof = sc, t
print(f"  OOF blend acc={best_oof_score:.4f}  threshold={best_t_oof:.3f}")

# Also keep meta-learner for comparison
meta_X = np.column_stack([oof_gbm, oof_rf, oof_et, oof_lr])
meta = LogisticRegression(C=5.0, solver='lbfgs', max_iter=1000, random_state=42)
meta.fit(meta_X, y_all)
print(f"  Meta OOF acc={accuracy_score(y_all, meta.predict(meta_X)):.4f}")

# ── Retrain all base models on full data (train+val) for deployment ───────────
print("\nRetraining base models on train+val …")
tfidf_w_full = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 3),
    max_features=5000, min_df=2, sublinear_tf=True,
)
tfidf_c_full = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 6),
    max_features=3000, min_df=3, sublinear_tf=True,
)
tfidf_w_full.fit(full_tok)
tfidf_c_full.fit(full_tok)
X_full_tw = tfidf_w_full.transform(full_tok)
X_full_tc = tfidf_c_full.transform(full_tok)
X_full_tf  = sp.hstack([X_full_tw, X_full_tc])
X_test_tw2 = tfidf_w_full.transform(test_tok)
X_test_tc2 = tfidf_c_full.transform(test_tok)
X_test_tf2 = sp.hstack([X_test_tw2, X_test_tc2])

# Feature selector re-fit on full data
sel_full = SelectKBest(chi2, k=min(400, X_full_tf.shape[1]))
X_full_tfs  = sel_full.fit_transform(X_full_tf, y_full_m)
X_test_tfs2 = sel_full.transform(X_test_tf2)

X_full_dense  = np.hstack([X_full_s,  X_full_tfs.toarray()])
X_test_dense2 = np.hstack([X_test_s,  X_test_tfs2.toarray()])

X_full_cb   = sp.hstack([X_full_tf,  sp.csr_matrix(X_full_s)])
X_test_cb2  = sp.hstack([X_test_tf2, sp.csr_matrix(X_test_s)])

gbm.fit(X_full_dense, y_full_m)
rf.fit(X_full_dense, y_full_m)
et.fit(X_full_s, y_full_m)
lr_tfidf = LogisticRegression(C=1.0, solver='liblinear', max_iter=5000,
                               class_weight='balanced', random_state=42)
lr_tfidf.fit(X_full_cb, y_full_m)

# ── Test-set evaluation ───────────────────────────────────────────────────────
print("Evaluating on held-out test set …")
test_gbm_p = gbm.predict_proba(X_test_dense2)[:, 1]
test_rf_p  = rf.predict_proba(X_test_dense2)[:, 1]
test_et_p  = et.predict_proba(X_test_s)[:, 1]
test_lr_p  = lr_tfidf.predict_proba(X_test_cb2)[:, 1]

# Weighted blend using OOF-derived weights
test_cal = w_gbm*test_gbm_p + w_rf*test_rf_p + w_et*test_et_p + w_lr*test_lr_p

# Use OOF-derived threshold (no test-set look-ahead)
best_t = best_t_oof
platt  = None   # no Platt needed for simple blend

pred = (test_cal > best_t).astype(int)
acc  = accuracy_score(y_test_m, pred)
prec = precision_score(y_test_m, pred, zero_division=0)
rec  = recall_score(y_test_m, pred, zero_division=0)
f1   = f1_score(y_test_m, pred, zero_division=0)
auc  = roc_auc_score(y_test_m, test_cal)
mcc  = matthews_corrcoef(y_test_m, pred)
cm   = confusion_matrix(y_test_m, pred)
tn, fp, fn, tp = cm.ravel()
fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0

print("\n" + "=" * 60)
print("  FS DIRECT v2 STACKING — TEST RESULTS")
print("=" * 60)
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  MCC       : {mcc:.4f}")
print(f"  FPR       : {fpr:.4f}  ({fpr*100:.1f}%)")
print(f"  Threshold : {best_t:.3f}")
print(f"  CM: TN={tn} FP={fp} FN={fn} TP={tp}")
print("=" * 60)
print(f"  Previous best     : ~67.5%")
print(f"  New accuracy      : {acc*100:.1f}%")
if acc > 0.675:
    print(f"  Improvement       : +{(acc - 0.675)*100:.1f}%  [OK]")
print("=" * 60)

# ── Save ──────────────────────────────────────────────────────────────────────
payload = {
    "version":      "v2_stacking",
    "gbm":          gbm,
    "rf":           rf,
    "et":           et,
    "lr_tfidf":     lr_tfidf,
    "meta":         meta,
    "blend_w":      (w_gbm, w_rf, w_et, w_lr),
    "platt":        None,
    "scaler":       scaler,
    "tfidf_word":   tfidf_w_full,
    "tfidf_char":   tfidf_c_full,
    "tfidf":        tfidf_w_full,   # backward-compat alias
    "sel":          sel_full,
    "threshold":    best_t,
    "accuracy":     float(acc),
    "f1":           float(f1),
    "auc":          float(auc),
    "feat_names":   feat_names,
}
with open(CKPT, "wb") as fh:
    pickle.dump(payload, fh)
print(f"\n  Saved: {CKPT}")

# Update calibration_matrix.json
try:
    with open(CAL_IN) as fh:
        cal_data = json.load(fh)
    old_t = cal_data["thresholds"]["fs"]
    cal_data["thresholds"]["fs"] = round(float(best_t), 6)
    with open(CAL_IN, "w") as fh:
        json.dump(cal_data, fh, indent=2)
    print(f"  Updated calibration_matrix.json: fs {old_t} -> {best_t:.4f}")
except Exception as e:
    print(f"  [warn] Could not update calibration_matrix.json: {e}")

print("\nDone.")
