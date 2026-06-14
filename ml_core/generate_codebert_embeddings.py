"""
generate_codebert_embeddings.py
Precomputes CodeBERT CLS embeddings for all FS splits.
Run once — saves .npy files used by train_fs_codebert.py.

    cd ml_core
    python generate_codebert_embeddings.py
"""

import os, numpy as np, pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch

ROOT  = Path(__file__).resolve().parent
DATA  = ROOT / "data" / "processed" / "fs"
EMB   = ROOT / "data" / "processed" / "fs" / "embeddings"
EMB.mkdir(exist_ok=True)

MODEL_NAME = "microsoft/codebert-base"
BATCH      = 16
MAX_LEN    = 512
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device : {DEVICE}")
print(f"Loading {MODEL_NAME} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("Model loaded.\n")


def embed_batch(texts):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        out = model(**enc)
    # CLS token — shape (batch, 768)
    return out.last_hidden_state[:, 0, :].cpu().numpy()


def embed_split(csv_path, out_path):
    if out_path.exists():
        print(f"  {out_path.name} already exists — skipping")
        return
    df   = pd.read_csv(csv_path)
    n    = len(df)
    embs = np.zeros((n, 768), dtype=np.float32)
    for start in range(0, n, BATCH):
        end   = min(start + BATCH, n)
        batch = df["code"].iloc[start:end].tolist()
        embs[start:end] = embed_batch(batch)
        if (start // BATCH) % 10 == 0:
            print(f"    {end}/{n}", end="\r", flush=True)
    np.save(out_path, embs)
    print(f"  Saved {out_path.name}  shape={embs.shape}")


for split in ["train", "val", "test"]:
    csv = DATA / f"{split}.csv"
    npy = EMB  / f"{split}_codebert.npy"
    print(f"[{split}]")
    embed_split(csv, npy)

print("\nAll embeddings generated.")
