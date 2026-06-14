"""
Quick diagnostic for FS dataset:
1. Label balance per split
2. Embedding shape & stats
3. RandomForest baseline on Stage-3 embeddings
4. Feature separability check (mean distance between classes)
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── 1. Load embeddings and labels ──
embed_dir = ROOT / "data" / "embeddings" / "fs"
comp_dir  = ROOT / "data" / "compressed" / "fs"

splits = {}
for sp in ["train", "val", "test"]:
    X = np.load(embed_dir / f"{sp}.npy")
    y = np.load(embed_dir / f"{sp}_labels.npy")
    splits[sp] = (X, y)

X_train, y_train = splits["train"]
X_val, y_val     = splits["val"]
X_test, y_test   = splits["test"]

print("=" * 60)
print("  DIAGNOSTIC: FS Dataset")
print("=" * 60)

# ── 2. Label balance ──
print("\n── Label Balance ──")
for name, (X, y) in splits.items():
    pos = int(y.sum())
    neg = len(y) - pos
    print(f"  {name:>5}: total={len(y):>5}  pos={pos:>5} ({pos/len(y)*100:.1f}%)  neg={neg:>5} ({neg/len(y)*100:.1f}%)")

# ── 3. Embedding shape & stats ──
print("\n── Embedding Shape & Stats (Stage 3, 128-dim) ──")
for name, (X, y) in splits.items():
    print(f"  {name:>5}: shape={X.shape}  min={X.min():.4f}  max={X.max():.4f}  "
          f"mean={X.mean():.4f}  std={X.std():.4f}")

print("\n── First 3 train samples (first 10 dims) ──")
for i in range(min(3, len(X_train))):
    print(f"  [{i}] label={int(y_train[i])}  {X_train[i, :10]}")

# ── 4. Feature separability ──
print("\n── Feature Separability (Stage 3 embeddings) ──")
pos_mask = y_train == 1
neg_mask = y_train == 0
mean_pos = X_train[pos_mask].mean(axis=0)
mean_neg = X_train[neg_mask].mean(axis=0)
centroid_dist = np.linalg.norm(mean_pos - mean_neg)
print(f"  Centroid distance (L2): {centroid_dist:.4f}")
print(f"  Pos mean norm: {np.linalg.norm(mean_pos):.4f}")
print(f"  Neg mean norm: {np.linalg.norm(mean_neg):.4f}")

# Per-feature t-test (top discriminative features)
from scipy import stats
t_vals = []
for i in range(X_train.shape[1]):
    t, p = stats.ttest_ind(X_train[pos_mask, i], X_train[neg_mask, i])
    t_vals.append((i, abs(t), p))
t_vals.sort(key=lambda x: -x[1])
print(f"\n  Top 10 most discriminative features (by |t-stat|):")
print(f"  {'Feat':>5} {'|t|':>8} {'p-value':>12}")
for feat, t, p in t_vals[:10]:
    print(f"  {feat:>5} {t:>8.3f} {p:>12.2e}")

n_sig = sum(1 for _, _, p in t_vals if p < 0.05)
print(f"\n  Features with p < 0.05: {n_sig}/{X_train.shape[1]}")

# ── 5. RandomForest baseline ──
print("\n── RandomForest Baseline (on Stage-3 embeddings) ──")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
    pred = rf.predict(X)
    prob = rf.predict_proba(X)[:, 1]
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    auc = roc_auc_score(y, prob)
    print(f"\n  {name.upper()} set:")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    F1       : {f1:.4f}")
    print(f"    ROC-AUC  : {auc:.4f}")
    print(classification_report(y, pred, target_names=["Safe", "Vuln"], digits=4))

# ── 6. Also try on compressed (Stage 4, 32-dim) ──
if (comp_dir / "train.npy").exists():
    print("\n── RandomForest on Stage-4 compressed (32-dim) ──")
    Xc_train = np.load(comp_dir / "train.npy")
    yc_train = np.load(comp_dir / "train_labels.npy")
    Xc_test  = np.load(comp_dir / "test.npy")
    yc_test  = np.load(comp_dir / "test_labels.npy")
    
    rf2 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf2.fit(Xc_train, yc_train)
    pred2 = rf2.predict(Xc_test)
    prob2 = rf2.predict_proba(Xc_test)[:, 1]
    acc2 = accuracy_score(yc_test, pred2)
    f1_2 = f1_score(yc_test, pred2)
    auc2 = roc_auc_score(yc_test, prob2)
    print(f"  TEST: Acc={acc2:.4f}  F1={f1_2:.4f}  AUC={auc2:.4f}")
    print(classification_report(yc_test, pred2, target_names=["Safe", "Vuln"], digits=4))

print("\n" + "=" * 60)
print("  DIAGNOSIS COMPLETE")
print("=" * 60)
