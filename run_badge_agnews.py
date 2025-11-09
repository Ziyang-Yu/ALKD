#!/usr/bin/env python3
"""
BADGE Active Learning on AG News Text Classification
====================================================

Author: Ziyang Yu
License: MIT
"""

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DistilBertModel,
    DistilBertConfig,
)
from torch.optim import AdamW

torch.use_deterministic_algorithms(True)

# --------------------------
# Utils
# --------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Classifier(nn.Module):
    """Lightweight MLP head over the frozen encoder CLS embedding."""
    def __init__(self, hidden_size=768, num_labels=4):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.classifier(x)

@dataclass
class ALIndices:
    labeled: List[int]
    unlabeled: List[int]

def split_initial_pool(n_total: int, init_size: int, seed: int = 42) -> ALIndices:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    init = idx[:init_size].tolist()
    rest = idx[init_size:].tolist()
    return ALIndices(labeled=init, unlabeled=rest)

# --------------------------
# Data
# --------------------------

class HFDatasetWrapper(Dataset):
    """Wrap a HuggingFace split and produce tokenized batches for classification."""
    def __init__(self, hf_dataset, tokenizer, text_key="text", max_length=128):
        self.ds = hf_dataset
        self.text_key = text_key
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        enc = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(item["label"], dtype=torch.long)
        return enc

def collate_fn(batch):
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    return out

# --------------------------
# Training / Eval
# --------------------------

def train_one_round(encoder, classifier, loader, device, lr=5e-5, epochs=2, weight_decay=0.01, max_grad_norm=1.0):
    """Train only the classifier while keeping the encoder frozen."""
    classifier.train()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    num_train_steps = epochs * max(1, len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, num_train_steps // 10),
        num_training_steps=num_train_steps,
    )
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Compute frozen encoder features under no_grad to save memory/compute
            with torch.no_grad():
                encoder_outputs = encoder(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                hidden_states = encoder_outputs.last_hidden_state[:, 0]  # [CLS]

            logits = classifier(hidden_states)
            loss = criterion(logits, batch['labels'])

            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

@torch.no_grad()
def evaluate(encoder, classifier, loader, device) -> float:
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        encoder_outputs = encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        hidden_states = encoder_outputs.last_hidden_state[:, 0]
        logits = classifier(hidden_states)
        preds = logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += preds.numel()

    return correct / max(1, total)

# --------------------------
# BADGE: gradient embeddings
# --------------------------

@torch.no_grad()
def badge_gradient_embeddings(encoder, classifier, loader, device, num_classes: int) -> Tuple[np.ndarray, List[int]]:
    """
    g(x) = flatten((p - e_yhat) ⊗ h), where h is [CLS] feature and p is softmax over classes.
    """
    encoder.eval()
    classifier.eval()
    embs = []
    map_idx = []

    for batch in loader:
        # Keep labels in batch structure, but do not use them for embeddings
        batch = {k: v.to(device) for k, v in batch.items()}

        encoder_outputs = encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        h = encoder_outputs.last_hidden_state[:, 0]      # [B, H]
        logits = classifier(h)                            # [B, C]

        probs = torch.softmax(logits, dim=-1)            # [B, C]
        yhat = probs.argmax(dim=-1)                      # [B]
        onehot = torch.zeros_like(probs)
        onehot[torch.arange(probs.size(0)), yhat] = 1.0
        delta = probs - onehot                           # [B, C]

        g = (delta.unsqueeze(-1) * h.unsqueeze(1)).reshape(probs.size(0), -1)  # [B, C*H]
        embs.append(g.cpu().float().numpy())

        if "idx" in batch:
            map_idx.extend(batch["idx"].cpu().tolist())
        else:
            map_idx.extend([None] * probs.size(0))

    if len(embs) == 0:
        return np.zeros((0, 1), dtype=np.float32), []
    E = np.concatenate(embs, axis=0)

    # Normalize for stability (scale-invariant distances for k-means++)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.maximum(norms, 1e-8)
    return E.astype(np.float32), map_idx

# --------------------------
# k-means++ style selection
# --------------------------

from sklearn.cluster import kmeans_plusplus
import numpy as np

def kmeanspp_select_sklearn(embs: np.ndarray, k: int, rng: np.random.Generator):
    # sklearn 会自己做 D^2 加权采样，返回 (centers, indices)
    _, indices = kmeans_plusplus(
        embs.astype(np.float32, copy=False),
        n_clusters=k,
        random_state=int(rng.integers(0, 2**32 - 1)),
    )
    return list(map(int, indices))


def kmeanspp_select(embs: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    """Minimal k-means++ seeding with robust fallbacks for degenerate cases."""
    n = embs.shape[0]
    if n == 0 or k <= 0:
        return []
    if k > n:
        k = n

    # First center: pick the point with maximum L2 norm
    norms = np.sum(embs * embs, axis=1)
    first = int(np.argmax(norms))
    centers = [first]

    # Squared distances to nearest center
    d2 = np.sum((embs - embs[first]) ** 2, axis=1)

    print("Start to select centers via k-means++...")
    print(f"  Total points: {n}, Selecting k={k} centers.")

    for i in range(1, k):
        # print("Selected {}/{} centers...".format(i, k), end="\r")
        # D^2-weighted sampling; guard against zero/NaN probabilities
        denom = np.maximum(d2.sum(), 1e-12)
        probs = d2 / denom
        if (not np.isfinite(probs).all()) or (probs.sum() <= 0) or np.allclose(d2, 0.0):
            # Degenerate: pick random remaining
            candidates = list(set(range(n)) - set(centers))
            centers.append(int(rng.choice(candidates)))
            d2 = np.minimum(d2, np.sum((embs - embs[centers[-1]]) ** 2, axis=1))
            continue

        next_idx = int(rng.choice(n, p=probs))
        while next_idx in centers:
            next_idx = int(rng.integers(0, n))
        centers.append(next_idx)
        d2 = np.minimum(d2, np.sum((embs - embs[next_idx]) ** 2, axis=1))

    return centers

# --------------------------
# Active Learning Loop
# --------------------------

def make_loader(dataset: Dataset, idxs: List[int], batch_size: int, shuffle: bool, add_indices: bool=False):
    """Create a DataLoader over a subset; indices are not exposed in batches here."""
    if add_indices:
        def collate_with_idx(batch):
            out = collate_fn(batch)
            return out
        return DataLoader(
            Subset(dataset, idxs),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_with_idx
        )
    else:
        return DataLoader(
            Subset(dataset, idxs),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed_size", type=int, default=200)
    parser.add_argument("--query_size", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    # Load AG News and tokenize on-the-fly
    raw = load_dataset("ag_news")
    num_labels = 4
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir="./hf_cache")

    train_ds = HFDatasetWrapper(raw["train"], tokenizer, max_length=args.max_length)
    test_ds  = HFDatasetWrapper(raw["test"],  tokenizer, max_length=args.max_length)

    # Initialize labeled/unlabeled pools for Active Learning
    N = len(train_ds)
    pools = split_initial_pool(N, args.seed_size, seed=args.seed)
    print(f"Total train: {N}. Seed labeled: {len(pools.labeled)}. Unlabeled pool: {len(pools.unlabeled)}.")

    # Encoder (frozen DistilBERT) + small classifier head
    encoder_config = DistilBertConfig.from_pretrained('distilbert-base-uncased', cache_dir="./hf_cache")
    encoder = DistilBertModel.from_pretrained('distilbert-base-uncased', config=encoder_config, cache_dir="./hf_cache")
    for p in encoder.parameters():
        p.requires_grad = False
    encoder = encoder.to(device)

    classifier = Classifier(hidden_size=encoder_config.hidden_size, num_labels=num_labels).to(device)

    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    for r in range(args.rounds + 1):
        print("=" * 80)
        print(f"Round {r}/{args.rounds} — Labeled {len(pools.labeled)} | Unlabeled {len(pools.unlabeled)}")

        # Train classifier on the currently labeled set, then evaluate
        train_loader = make_loader(train_ds, pools.labeled, batch_size=args.batch_size, shuffle=True)
        if len(pools.labeled) > 0:
            train_one_round(encoder, classifier, train_loader, device, lr=args.lr, epochs=args.epochs)
        acc = evaluate(encoder, classifier, test_loader, device)
        print(f"[Eval] Test accuracy: {acc*100:.2f}%")

        if r == args.rounds:
            break

        # BADGE on pool: compute gradient embeddings, then k-means++ select
        pool_loader = make_loader(train_ds, pools.unlabeled, batch_size=128, shuffle=False)
        embs, _ = badge_gradient_embeddings(encoder, classifier, pool_loader, device, num_classes=num_labels)
        n_pool = embs.shape[0]
        k = min(args.query_size, n_pool)
        if k == 0:
            print("No points left to select from the pool. Stopping.")
            break

        print(f"Selecting {k} points via BADGE (k-means++ in gradient space)...")
        # select_rel = kmeanspp_select(embs, k, rng)  # indices relative to pools.unlabeled
        t = time.time()
        print("Using sklearn k-means++ for selection...")
        select_rel = kmeanspp_select_sklearn(embs, k, rng)  # indices relative to pools.unlabeled
        print(f"Selection took {time.time() - t:.2f} seconds.")
        newly_selected = [pools.unlabeled[i] for i in select_rel]

        # Move selected from unlabeled -> labeled
        pools.labeled.extend(newly_selected)
        mask = np.ones(len(pools.unlabeled), dtype=bool)
        mask[select_rel] = False
        pools.unlabeled = [pools.unlabeled[i] for i in range(len(mask)) if mask[i]]

    print("Done.")

if __name__ == "__main__":
    main()
