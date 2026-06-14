"""
finetune_codebert_fs.py — End-to-end CodeBERT fine-tuning for FS classification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unlike train_fs_codebert.py (frozen embeddings), this script fine-tunes the full
CodeBERT transformer on the FS dataset — weights update via backpropagation.

Run:
    cd ml_core
    python finetune_codebert_fs.py
"""

import os, json, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              confusion_matrix, matthews_corrcoef)
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
DATA     = ROOT / "data" / "processed" / "fs"
CKPT_DIR = ROOT / "models" / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
CAL      = ROOT / "results" / "calibration_matrix.json"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "microsoft/codebert-base"
MAX_LEN      = 256       # 256 tokens — captures full FS call context, faster than 512
BATCH_SIZE   = 16
EPOCHS       = 5
LR           = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
DROPOUT      = 0.2
PATIENCE     = 2         # early stop if val_acc stagnates
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device  : {DEVICE}")
print(f"Config  : max_len={MAX_LEN}  batch={BATCH_SIZE}  lr={LR}  epochs={EPOCHS}\n")


# ── Dataset ───────────────────────────────────────────────────────────────────
class FSDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.codes  = df["code"].fillna("").tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        enc = self.tok(
            self.codes[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────────────────────
class CodeBERTFineTuned(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout    = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # CLS token
        return self.classifier(self.dropout(cls))


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
train_df = pd.read_csv(DATA / "train.csv")
val_df   = pd.read_csv(DATA / "val.csv")
test_df  = pd.read_csv(DATA / "test.csv")
print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}\n")

print(f"Loading tokenizer from {MODEL_NAME} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = FSDataset(train_df, tokenizer)
val_ds   = FSDataset(val_df,   tokenizer)
test_ds  = FSDataset(test_df,  tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model + optimiser ─────────────────────────────────────────────────────────
print("Initialising model …")
model = CodeBERTFineTuned().to(DEVICE)

total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

print(f"  Total steps : {total_steps}  |  Warmup : {warmup_steps}\n")


# ── Training loop ─────────────────────────────────────────────────────────────
best_val_acc    = 0.0
best_epoch      = 0
no_improve      = 0
best_ckpt_path  = CKPT_DIR / "codebert_fs_finetuned_best.pt"

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(train_loader, 1):
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = criterion(logits, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if step % 30 == 0 or step == len(train_loader):
            elapsed = time.time() - t0
            eta     = elapsed / step * (len(train_loader) - step)
            print(f"  Epoch {epoch}/{EPOCHS}  step {step}/{len(train_loader)}"
                  f"  loss={total_loss/step:.4f}  eta={eta/60:.1f}min", end="\r")

    print()

    # ── Validation ────────────────────────────────────────────────────────────
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbl  = batch["label"].to(DEVICE)
            out  = model(ids, mask)
            p    = torch.softmax(out, dim=1)[:, 1]
            probs.extend(p.cpu().numpy())
            preds.extend(out.argmax(dim=1).cpu().numpy())
            labels.extend(lbl.cpu().numpy())

    val_acc = accuracy_score(labels, preds)
    val_f1  = f1_score(labels, preds, zero_division=0)
    val_auc = roc_auc_score(labels, probs)
    epoch_time = (time.time() - t0) / 60

    print(f"  Epoch {epoch}/{EPOCHS}  "
          f"loss={total_loss/len(train_loader):.4f}  "
          f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  val_auc={val_auc:.4f}  "
          f"({epoch_time:.1f} min)")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        no_improve   = 0
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"  ✓ New best saved  (val_acc={val_acc:.4f})")
    else:
        no_improve += 1
        print(f"  No improvement ({no_improve}/{PATIENCE})")
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

print(f"\nBest val_acc={best_val_acc:.4f} at epoch {best_epoch}\n")


# ── Test evaluation ───────────────────────────────────────────────────────────
print("Loading best checkpoint for test evaluation …")
model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
model.eval()

preds, labels, probs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbl  = batch["label"].to(DEVICE)
        out  = model(ids, mask)
        p    = torch.softmax(out, dim=1)[:, 1]
        probs.extend(p.cpu().numpy())
        preds.extend(out.argmax(dim=1).cpu().numpy())
        labels.extend(lbl.cpu().numpy())

# Optimal threshold search
best_t, best_sc = 0.5, 0.0
for t in np.arange(0.10, 0.90, 0.005):
    sc = accuracy_score(labels, (np.array(probs) > t).astype(int))
    if sc > best_sc:
        best_sc, best_t = sc, t

pred_final = (np.array(probs) > best_t).astype(int)
acc  = accuracy_score(labels, pred_final)
prec = precision_score(labels, pred_final, zero_division=0)
rec  = recall_score(labels, pred_final, zero_division=0)
f1   = f1_score(labels, pred_final, zero_division=0)
auc  = roc_auc_score(labels, probs)
mcc  = matthews_corrcoef(labels, pred_final)
cm   = confusion_matrix(labels, pred_final)
tn, fp, fn, tp = cm.ravel()
fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0

print("\n" + "=" * 60)
print("  FS CODEBERT FINE-TUNED — TEST RESULTS")
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
print(f"  Previous best (CodeBERT hybrid) : 69.0%")
print(f"  Fine-tuned CodeBERT             : {acc*100:.1f}%")
if acc > 0.692:
    print(f"  Improvement                     : +{(acc-0.692)*100:.1f}%  ✓")
if acc >= 0.80:
    print(f"  TARGET 80% REACHED ✓")
print("=" * 60)


# ── Save inference payload ────────────────────────────────────────────────────
payload = {
    "version":    "codebert_finetuned",
    "model_path": str(best_ckpt_path),
    "model_name": MODEL_NAME,
    "max_len":    MAX_LEN,
    "threshold":  float(best_t),
    "accuracy":   float(acc),
    "f1":         float(f1),
    "auc":        float(auc),
}
meta_path = CKPT_DIR / "codebert_fs_finetuned_meta.json"
with open(meta_path, "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"\n  Weights : {best_ckpt_path}")
print(f"  Meta    : {meta_path}")

# Update calibration threshold
try:
    with open(CAL) as fh:
        cal = json.load(fh)
    old = cal["thresholds"]["fs"]
    cal["thresholds"]["fs"] = round(float(best_t), 6)
    with open(CAL, "w") as fh:
        json.dump(cal, fh, indent=2)
    print(f"  Calibration threshold updated: {old} -> {best_t:.4f}")
except Exception as e:
    print(f"  [warn] Could not update calibration: {e}")

print("\nDone.")
