"""
train_fs_codebert.py — FS Hybrid Classifier (CodeBERT + Rule Features)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Combines:
  - CodeBERT CLS embeddings (768-dim) — deep code understanding
  - Rule + context window features (58-dim) — explicit vuln patterns
  - XGBoost stacking ensemble

Run:
    cd ml_core
    python generate_codebert_embeddings.py   # once
    python train_fs_codebert.py
"""

import os, json, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              confusion_matrix, matthews_corrcoef)
import xgboost as xgb

ROOT  = Path(__file__).resolve().parent
DATA  = ROOT / "data" / "processed" / "fs"
EMB   = DATA / "embeddings"
CKPT  = ROOT / "models" / "checkpoints" / "fs_codebert.pkl"
CAL   = ROOT / "results" / "calibration_matrix.json"
CKPT.parent.mkdir(exist_ok=True)

from fs_features import build_manual_features, build_manual_matrix

# ── Load splits ───────────────────────────────────────────────────────────────
print("Loading data …")
train_df = pd.read_csv(DATA / "train.csv")
val_df   = pd.read_csv(DATA / "val.csv")
test_df  = pd.read_csv(DATA / "test.csv")
full_df  = pd.concat([train_df, val_df], ignore_index=True)
print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

# ── Load CodeBERT embeddings ──────────────────────────────────────────────────
print("Loading CodeBERT embeddings …")
for split in ["train", "val", "test"]:
    p = EMB / f"{split}_codebert.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found.\nRun:  python generate_codebert_embeddings.py"
        )

E_train = np.load(EMB / "train_codebert.npy").astype(np.float32)
E_val   = np.load(EMB / "val_codebert.npy").astype(np.float32)
E_test  = np.load(EMB / "test_codebert.npy").astype(np.float32)
E_full  = np.vstack([E_train, E_val])
print(f"  Embedding shape: {E_train.shape}")

# ── Build rule/manual features ────────────────────────────────────────────────
print("Extracting rule + context features …")
X_train_m, y_train, feat_names = build_manual_matrix(train_df)
X_val_m,   y_val,   _          = build_manual_matrix(val_df)
X_test_m,  y_test,  _          = build_manual_matrix(test_df)
X_full_m,  y_full,  _          = build_manual_matrix(full_df)
print(f"  Manual feature dim: {X_train_m.shape[1]}")

scaler_m = StandardScaler()
X_train_ms = scaler_m.fit_transform(X_train_m)
X_val_ms   = scaler_m.transform(X_val_m)
X_test_ms  = scaler_m.transform(X_test_m)
X_full_ms  = scaler_m.fit_transform(X_full_m)

# Scale embeddings separately
scaler_e = StandardScaler()
E_train_s = scaler_e.fit_transform(E_train)
E_val_s   = scaler_e.transform(E_val)
E_test_s  = scaler_e.transform(E_test)
E_full_s  = scaler_e.fit_transform(E_full)

# ── Combined feature matrix ───────────────────────────────────────────────────
X_train = np.hstack([E_train_s, X_train_ms])   # 768 + 58 = 826 dims
X_val   = np.hstack([E_val_s,   X_val_ms])
X_test  = np.hstack([E_test_s,  X_test_ms])
X_full  = np.hstack([E_full_s,  X_full_ms])
print(f"  Combined feature dim: {X_train.shape[1]}")

y_all = np.concatenate([y_train, y_val])

# ── Base models ───────────────────────────────────────────────────────────────
print("\nTraining base models …")

xgb_model = xgb.XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
xgb_val_p = xgb_model.predict_proba(X_val)[:, 1]
print(f"  XGB   val acc={accuracy_score(y_val, xgb_model.predict(X_val)):.4f}  "
      f"auc={roc_auc_score(y_val, xgb_val_p):.4f}")

rf = RandomForestClassifier(
    n_estimators=600, max_depth=None, min_samples_leaf=3,
    max_features='sqrt', class_weight='balanced',
    random_state=42, n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_val_p = rf.predict_proba(X_val)[:, 1]
print(f"  RF    val acc={accuracy_score(y_val, rf.predict(X_val)):.4f}  "
      f"auc={roc_auc_score(y_val, rf_val_p):.4f}")

# Logistic regression on embeddings only — strong linear baseline
lr = LogisticRegression(C=4.0, solver='lbfgs', max_iter=2000,
                        class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
lr_val_p = lr.predict_proba(X_val)[:, 1]
print(f"  LR    val acc={accuracy_score(y_val, lr.predict(X_val)):.4f}  "
      f"auc={roc_auc_score(y_val, lr_val_p):.4f}")

# Rule-only model — keeps explicit pattern signal
rule_lr = LogisticRegression(C=2.0, solver='lbfgs', max_iter=2000,
                              class_weight='balanced', random_state=42)
rule_lr.fit(X_train_ms, y_train)
rule_val_p = rule_lr.predict_proba(X_val_ms)[:, 1]
print(f"  Rule  val acc={accuracy_score(y_val, rule_lr.predict(X_val_ms)):.4f}  "
      f"auc={roc_auc_score(y_val, rule_val_p):.4f}")

# ── 5-fold OOF stacking ───────────────────────────────────────────────────────
print("\nBuilding OOF meta-features (5-fold) …")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all_full = np.vstack([X_train, X_val])
X_all_rule = np.vstack([X_train_ms, X_val_ms])

oof_xgb  = np.zeros(len(y_all))
oof_rf   = np.zeros(len(y_all))
oof_lr   = np.zeros(len(y_all))
oof_rule = np.zeros(len(y_all))

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_all_full, y_all)):
    # XGB
    m = xgb.XGBClassifier(n_estimators=500, learning_rate=0.04, max_depth=6,
                           subsample=0.8, colsample_bytree=0.7,
                           use_label_encoder=False, eval_metric="logloss",
                           random_state=42, n_jobs=-1)
    m.fit(X_all_full[tr_idx], y_all[tr_idx], verbose=False)
    oof_xgb[vl_idx] = m.predict_proba(X_all_full[vl_idx])[:, 1]

    # RF
    m2 = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    m2.fit(X_all_full[tr_idx], y_all[tr_idx])
    oof_rf[vl_idx] = m2.predict_proba(X_all_full[vl_idx])[:, 1]

    # LR
    m3 = LogisticRegression(C=4.0, solver='lbfgs', max_iter=1000,
                             class_weight='balanced', random_state=42)
    m3.fit(X_all_full[tr_idx], y_all[tr_idx])
    oof_lr[vl_idx] = m3.predict_proba(X_all_full[vl_idx])[:, 1]

    # Rule LR
    m4 = LogisticRegression(C=2.0, solver='lbfgs', max_iter=1000,
                             class_weight='balanced', random_state=42)
    m4.fit(X_all_rule[tr_idx], y_all[tr_idx])
    oof_rule[vl_idx] = m4.predict_proba(X_all_rule[vl_idx])[:, 1]

    print(f"  Fold {fold+1}/5 done")

# ── AUC-weighted blend ────────────────────────────────────────────────────────
from sklearn.metrics import roc_auc_score as _auc
auc_xgb  = _auc(y_all, oof_xgb)
auc_rf   = _auc(y_all, oof_rf)
auc_lr   = _auc(y_all, oof_lr)
auc_rule = _auc(y_all, oof_rule)
print(f"\n  OOF AUC: XGB={auc_xgb:.4f}  RF={auc_rf:.4f}  "
      f"LR={auc_lr:.4f}  Rule={auc_rule:.4f}")

w_total = auc_xgb + auc_rf + auc_lr + auc_rule
w_xgb, w_rf, w_lr, w_rule = (auc_xgb/w_total, auc_rf/w_total,
                               auc_lr/w_total, auc_rule/w_total)

oof_blend = w_xgb*oof_xgb + w_rf*oof_rf + w_lr*oof_lr + w_rule*oof_rule

# Optimal threshold on OOF
best_t, best_sc = 0.5, 0.0
for t in np.arange(0.10, 0.90, 0.005):
    sc = accuracy_score(y_all, (oof_blend > t).astype(int))
    if sc > best_sc:
        best_sc, best_t = sc, t
print(f"  OOF blend acc={best_sc:.4f}  threshold={best_t:.3f}")

# ── Retrain on full data ──────────────────────────────────────────────────────
print("\nRetraining on train+val …")
xgb_model.fit(X_full, y_full, verbose=False)
rf.fit(X_full, y_full)
lr_full = LogisticRegression(C=4.0, solver='lbfgs', max_iter=2000,
                              class_weight='balanced', random_state=42)
lr_full.fit(X_full, y_full)
rule_lr_full = LogisticRegression(C=2.0, solver='lbfgs', max_iter=2000,
                                   class_weight='balanced', random_state=42)
rule_lr_full.fit(X_full_ms, y_full)

# ── Test evaluation ───────────────────────────────────────────────────────────
print("Evaluating on test set …")
p_xgb  = xgb_model.predict_proba(X_test)[:, 1]
p_rf   = rf.predict_proba(X_test)[:, 1]
p_lr   = lr_full.predict_proba(X_test)[:, 1]
p_rule = rule_lr_full.predict_proba(X_test_ms)[:, 1]

test_blend = w_xgb*p_xgb + w_rf*p_rf + w_lr*p_lr + w_rule*p_rule
pred = (test_blend > best_t).astype(int)

acc  = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred, zero_division=0)
rec  = recall_score(y_test, pred, zero_division=0)
f1   = f1_score(y_test, pred, zero_division=0)
auc  = roc_auc_score(y_test, test_blend)
mcc  = matthews_corrcoef(y_test, pred)
cm   = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()
fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0

print("\n" + "=" * 60)
print("  FS CODEBERT HYBRID — TEST RESULTS")
print("=" * 60)
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  MCC       : {mcc:.4f}")
print(f"  FPR       : {fpr:.4f}")
print(f"  Threshold : {best_t:.3f}")
print(f"  CM: TN={tn} FP={fp} FN={fn} TP={tp}")
print("=" * 60)
print(f"  Previous best (feature-based) : 69.2%")
print(f"  CodeBERT hybrid               : {acc*100:.1f}%")
if acc > 0.692:
    print(f"  Improvement                   : +{(acc-0.692)*100:.1f}%  ✓")
print("=" * 60)

# ── Save ──────────────────────────────────────────────────────────────────────
payload = {
    "version":      "codebert_hybrid",
    "xgb":          xgb_model,
    "rf":           rf,
    "lr":           lr_full,
    "rule_lr":      rule_lr_full,
    "blend_w":      (w_xgb, w_rf, w_lr, w_rule),
    "scaler_e":     scaler_e,
    "scaler_m":     scaler_m,
    "feat_names":   feat_names,
    "threshold":    best_t,
    "accuracy":     float(acc),
    "f1":           float(f1),
    "auc":          float(auc),
}
with open(CKPT, "wb") as fh:
    pickle.dump(payload, fh)
print(f"\n  Saved: {CKPT}")

# Update calibration_matrix.json
try:
    with open(CAL) as fh:
        cal = json.load(fh)
    old = cal["thresholds"]["fs"]
    cal["thresholds"]["fs"] = round(float(best_t), 6)
    with open(CAL, "w") as fh:
        json.dump(cal, fh, indent=2)
    print(f"  Updated calibration threshold: {old} -> {best_t:.4f}")
except Exception as e:
    print(f"  [warn] Could not update calibration: {e}")

print("\nDone.")
