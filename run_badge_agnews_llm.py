#!/usr/bin/env python3
"""
BADGE Active Learning on AG News with LLM Annotator
===================================================

Author: Ziyang Yu
License: MIT
"""

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AutoModel,
    AutoConfig,
)
from torch.optim import AdamW
from dotenv import load_dotenv

import os, subprocess
load_dotenv()   # reads .env
key = os.getenv("OPENAI_API_KEY")

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
    """Wrap a HuggingFace split and produce tokenized batches for classification.

    Supports optional label overrides via a mapping from original dataset index
    to an integer class id. This allows plugging in LLM annotations at runtime.
    """
    def __init__(self, hf_dataset, tokenizer, text_key="text", max_length=128,
                 label_overrides: Optional[Dict[int, int]] = None):
        self.ds = hf_dataset
        self.text_key = text_key
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_overrides: Dict[int, int] = label_overrides or {}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        enc = self.tokenizer(
            item[self.text_key],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        # Use override if present; else fall back to dataset label
        if i in self.label_overrides:
            label = int(self.label_overrides[i])
        else:
            label = int(item["label"])  # from HF dataset
        enc["labels"] = torch.tensor(label, dtype=torch.long)
        # Always expose the original dataset index to downstream consumers
        enc["idx"] = torch.tensor(i, dtype=torch.long)
        return enc

    def set_label_override(self, i: int, label: int):
        self.label_overrides[int(i)] = int(label)

    def set_many_label_overrides(self, indices: List[int], labels: List[int]):
        for ii, ll in zip(indices, labels):
            self.set_label_override(int(ii), int(ll))

def collate_fn(batch):
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    return out

# --------------------------
# Training / Eval
# --------------------------

def train_one_round(encoder, classifier, loader, device, lr=5e-5, epochs=2, weight_decay=0.01, max_grad_norm=1.0):
    """Train only the classifier while keeping the encoder frozen."""
    # Ensure frozen encoder runs deterministically (no dropout)
    encoder.eval()
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

# --------------------------
# LLM Annotator
# --------------------------

AGNEWS_LABELS = [
    "World",        # 0
    "Sports",       # 1
    "Business",     # 2
    "Sci/Tech",     # 3
]

LABEL_NAME_TO_ID = {name.lower(): idx for idx, name in enumerate(AGNEWS_LABELS)}

def _parse_llm_label(text: str) -> Optional[int]:
    """Extract a single label id from a model response string.

    Accepts either numeric class ids (0-3) or label names from AGNEWS_LABELS.
    Returns None if parsing fails.
    """
    if text is None:
        return None
    s = str(text).strip()
    # Try integer first
    try:
        val = int(s)
        if 0 <= val < len(AGNEWS_LABELS):
            return val
    except Exception:
        pass
    # Try to find label name
    low = s.lower()
    for name, idx in LABEL_NAME_TO_ID.items():
        if name in low:
            return idx
    return None

def annotate_with_llm_openai_single(text: str, model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY",
                                    system_prompt: Optional[str] = None, temperature: float = 0.0,
                                    max_retries: int = 3) -> int:
    """Annotate a single text with one LLM call.

    Returns a single integer label id (0..3). Raises RuntimeError on failure
    after retries so caller can decide how to fallback.
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {api_key_env} for LLM annotator.")

    try:
        from openai import OpenAI  # OpenAI SDK v1
    except Exception as e:
        raise RuntimeError("openai package not installed. pip install openai>=1.0.0") from e

    client = OpenAI(api_key=api_key)

    system = system_prompt or (
        "You are a careful annotation assistant for AG News classification.\n"
        "Return only the label id (0,1,2,3) for the input.\n"
        "Label mapping: 0=World, 1=Sports, 2=Business, 3=Sci/Tech.\n"
    )

    sanitized = text.strip().replace("\n", " ")
    user_prompt = (
        "Classify the following news text into one of: 0=World, 1=Sports, 2=Business, 3=Sci/Tech.\n"
        "Return only one integer (0, 1, 2, or 3).\n\n"
        f"TEXT: {sanitized}"
    )

    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            content = (resp.choices[0].message.content or "").strip()
            label = _parse_llm_label(content)
            if label is None:
                # try permissive parse
                import re
                m = re.search(r"(?<!\d)([0-3])(?!\d)", content)
                if m:
                    return int(m.group(1))
                raise RuntimeError(f"LLM response unparsable: {content[:100]}...")
            return label
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"Failed single LLM annotation after retries: {last_err}")

def annotate_with_llm_openai_parallel(texts: List[str], model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY",
                                      system_prompt: Optional[str] = None, temperature: float = 0.0,
                                      max_retries: int = 3, max_workers: int = 16) -> List[Optional[int]]:
    """Annotate multiple texts by issuing one LLM call per text in parallel.

    Returns a list of labels where failures are represented by None. Caller can
    then selectively fallback only for failed items.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Optional[int]] = [None] * len(texts)

    def _task(idx: int, txt: str) -> Tuple[int, int]:
        label = annotate_with_llm_openai_single(
            txt,
            model=model,
            api_key_env=api_key_env,
            system_prompt=system_prompt,
            temperature=temperature,
            max_retries=max_retries,
        )
        return idx, label

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_idx = {ex.submit(_task, i, t): i for i, t in enumerate(texts)}
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            idx, label = fut.result()
            results[idx] = int(label)
    return results

@torch.no_grad()
def annotate_with_classifier(encoder, classifier, tokenizer, texts: List[str], device) -> List[int]:
    """Fallback: use current model predictions as pseudo-labels when LLM is unavailable."""
    classifier.eval()
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[:, 0]
    logits = classifier(outputs)
    return logits.argmax(dim=-1).cpu().tolist()

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
    # LLM annotator options
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini",
                        help="LLM model name for OpenAI provider.")
    parser.add_argument("--llm_api_key_env", type=str, default="OPENAI_API_KEY",
                        help="Env var name holding LLM API key.")
    parser.add_argument("--llm_batch_size", type=int, default=32,
                        help="(Deprecated for LLM) Previously used for batch requests; ignored when --use_llm_annotator is set.")
    parser.add_argument("--llm_workers", type=int, default=8,
                        help="Number of parallel threads for per-text LLM calls.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Optional Hugging Face cache directory.")

    args = parser.parse_args()

    git_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    print(f"Latest git commit hash: {git_commit_hash}")

    # Reproducibility
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    # Load AG News and tokenize on-the-fly
    raw = load_dataset("ag_news")
    num_labels = 4
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.cache_dir)

    train_ds = HFDatasetWrapper(raw["train"], tokenizer, max_length=args.max_length)
    test_ds  = HFDatasetWrapper(raw["test"],  tokenizer, max_length=args.max_length)

    # Initialize labeled/unlabeled pools for Active Learning
    N = len(train_ds)
    pools = split_initial_pool(N, args.seed_size, seed=args.seed)
    print(f"Total train: {N}. Seed labeled: {len(pools.labeled)}. Unlabeled pool: {len(pools.unlabeled)}.")

    # Encoder (frozen) + small classifier head
    encoder_config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    encoder = AutoModel.from_pretrained(args.model_name, config=encoder_config, cache_dir=args.cache_dir)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder = encoder.to(device)

    classifier = Classifier(hidden_size=encoder_config.hidden_size, num_labels=num_labels).to(device)

    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    acc_list = []

    for r in range(args.rounds):
        print("=" * 80)
        print(f"Round {r}/{args.rounds} — Labeled {len(pools.labeled)} | Unlabeled {len(pools.unlabeled)}")

        # Train classifier on the currently labeled set, then evaluate
        train_loader = make_loader(train_ds, pools.labeled, batch_size=args.batch_size, shuffle=True)
        if len(pools.labeled) > 0:
            train_one_round(encoder, classifier, train_loader, device, lr=args.lr, epochs=args.epochs)
        acc = evaluate(encoder, classifier, test_loader, device)
        print(f"[Eval] Test accuracy: {acc*100:.2f}%")
        acc_list.append(acc*100)

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
        t = time.time()
        print("Using sklearn k-means++ for selection...")
        select_rel = kmeanspp_select_sklearn(embs, k, rng)  # indices relative to pools.unlabeled
        print(f"Selection took {time.time() - t:.2f} seconds.")
        newly_selected = [pools.unlabeled[i] for i in select_rel]

        # Annotate selected indices using LLM (or fallback to model predictions)
        print(f"Annotating {len(newly_selected)} selected samples...")
        selected_texts = [raw["train"][i]["text"] for i in newly_selected]
        llm_labels: List[int] = []


        # Per-text LLM calls executed in parallel threads
        parallel_labels = annotate_with_llm_openai_parallel(
            selected_texts,
            model=args.llm_model,
            api_key_env=args.llm_api_key_env,
            max_workers=max(1, int(args.llm_workers)),
        )
        # Identify failures and fallback only for those
        failed_idx = [i for i, v in enumerate(parallel_labels) if v is None]
        if failed_idx:
            failed_texts = [selected_texts[i] for i in failed_idx]
            print(f"LLM failed on {len(failed_idx)} items; falling back to model for those.")
            fallback_labels = annotate_with_classifier(encoder, classifier, tokenizer, failed_texts, device)
            for rel, lab in zip(failed_idx, fallback_labels):
                parallel_labels[rel] = int(lab)
        llm_labels = [int(x) for x in parallel_labels]  # type: ignore
        print("LLM annotations received (with selective fallback if needed).")

        # Store overrides so subsequent training uses these labels
        train_ds.set_many_label_overrides(newly_selected, llm_labels)

        # Move selected from unlabeled -> labeled
        pools.labeled.extend(newly_selected)
        mask = np.ones(len(pools.unlabeled), dtype=bool)
        mask[select_rel] = False
        pools.unlabeled = [pools.unlabeled[i] for i in range(len(mask)) if mask[i]]



    print("\n=== Final Results ===")

    # save to results/agnews
    import json
    from datetime import datetime
    os.makedirs("results/agnews/llm", exist_ok=True)
    with open(f"results/agnews/llm/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}", "w") as f:
        json.dump(acc_list, f)

    print("Done.")

if __name__ == "__main__":
    main()
