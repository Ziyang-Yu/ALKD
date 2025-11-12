#!/usr/bin/env python3
"""
run_badge_agnews_llm_cbm.py
---------------------------------
Active Learning (BADGE) on AG News with:
- Frozen DistilBERT encoder
- Concept Bottleneck Classifier head (CBM)
- LLM annotator for labels (required) and concepts (optional)
- Sklearn k-means++ for BADGE selection
"""

import os, sys, time, random, math, json, concurrent.futures, subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openai import OpenAI
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.cluster import KMeans

from dotenv import load_dotenv

import os
load_dotenv()   # reads .env
key = os.getenv("OPENAI_API_KEY")

# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# Concepts for AG News
# ----------------------------
AGNEWS_CONCEPTS = [
    "mentions_person", "mentions_org", "mentions_location", "mentions_country",
    "mentions_company", "mentions_stock_or_market", "mentions_trade_or_deal", "mentions_economy",
    "mentions_sport", "mentions_team_or_league", "mentions_match_score", "mentions_player",
    "mentions_technology", "mentions_product_or_device", "mentions_research_or_science", "mentions_internet_or_software",
    "mentions_government_or_policy", "mentions_election_or_vote", "mentions_conflict_or_war", "mentions_disaster_or_crime",
    "mentions_health_or_medicine", "mentions_entertainment", "mentions_energy_or_environment", "mentions_space_or_weather",
]
N_CONCEPTS = len(AGNEWS_CONCEPTS)
AGNEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# ----------------------------
# CBM head
# ----------------------------
class CBMHead(nn.Module):
    def __init__(self, hidden_size=768, num_labels=4, n_concepts=N_CONCEPTS, hidden_task=256, dropout=0.1,
                 concept_l1=1e-4, detach_concepts=False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.concept_linear = nn.Linear(hidden_size, n_concepts, bias=True)
        self.task_head = nn.Sequential(
            nn.Linear(n_concepts, hidden_task, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_task, num_labels, bias=True),
        )
        self.concept_l1 = float(concept_l1)
        self.detach_concepts = bool(detach_concepts)

    def forward(self, feats: torch.Tensor, return_concepts: bool = False):
        h = self.dropout(feats)
        z_c = self.concept_linear(h)  # [B, C]
        c = torch.sigmoid(z_c)        # [B, C]
        if self.detach_concepts:
            c = c.detach()
        logits = self.task_head(c)    # [B, K]
        if return_concepts:
            return logits, c, z_c
        return logits

    def loss(self, logits, y, z_c=None, c_targets=None, alpha_concept=1.0):
        ce = F.cross_entropy(logits, y)
        cl = torch.tensor(0.0, device=logits.device)
        if z_c is not None and c_targets is not None:
            cl = alpha_concept * F.binary_cross_entropy_with_logits(z_c, c_targets)
        l1 = self.concept_l1 * (z_c.abs().mean() if z_c is not None else 0.0)
        return ce + cl + l1

# ----------------------------
# Dataset
# ----------------------------
class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, tokenizer, text_key="text", max_length=128):
        self.ds = hf_dataset
        self.text_key = text_key
        self.tok = tokenizer
        self.max_length = max_length
        self.label_overrides: Dict[int, int] = {}
        self.concept_overrides: Dict[int, List[int]] = {}

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        enc = self.tok(
            item[self.text_key],
            truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        # label
        lab = self.label_overrides.get(i, int(item["label"]))
        enc["labels"] = torch.tensor(lab, dtype=torch.long)
        # concepts (optional)
        if i in self.concept_overrides:
            enc["concepts"] = torch.tensor(self.concept_overrides[i], dtype=torch.float)
        enc["idx"] = torch.tensor(i, dtype=torch.long)
        return enc

    def set_many_label_overrides(self, indices: List[int], labels: List[int]):
        for i, y in zip(indices, labels):
            self.label_overrides[int(i)] = int(y)

    def set_many_concept_overrides(self, indices: List[int], concept_matrix: List[List[int]]):
        for i, cv in zip(indices, concept_matrix):
            self.concept_overrides[int(i)] = [int(x) for x in cv]

def collate_fn(batch):
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    return out

# ----------------------------
# BADGE gradient embeddings
# ----------------------------
@torch.no_grad()
def compute_gradient_embeddings(encoder, head, loader, device) -> Tuple[np.ndarray, List[int]]:
    encoder.eval(); head.eval()
    embs, map_idx = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        h = out.last_hidden_state[:, 0]  # CLS
        logits = head(h)                 # [B, K]
        probs = logits.softmax(-1)
        yhat = probs.argmax(-1)
        onehot = torch.zeros_like(probs)
        onehot[torch.arange(probs.size(0)), yhat] = 1.0
        delta = probs - onehot           # [B, K]
        # Outer product (delta x h) flattened
        g = (delta.unsqueeze(-1) * h.unsqueeze(1)).reshape(probs.size(0), -1)
        embs.append(g.cpu().float().numpy())
        map_idx.extend(batch["idx"].cpu().tolist())
    if not embs:
        return np.zeros((0, 1), dtype=np.float32), []
    return np.concatenate(embs, axis=0), map_idx

def kmeanspp_select_sklearn(embs: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    if embs.shape[0] == 0: return []
    k = min(k, embs.shape[0])
    # KMeans with one iteration + k-means++ init yields centers ~= seeding
    km = KMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=1, random_state=int(rng.integers(1e9)))
    km.fit(embs)
    # pick nearest samples to centers
    from sklearn.metrics import pairwise_distances_argmin_min
    idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, embs, metric="euclidean")
    return idx.tolist()

# ----------------------------
# Train / Evaluate
# ----------------------------
def train_one_round(encoder, head, loader, device, lr=5e-5, epochs=3, weight_decay=0.01, max_grad_norm=1.0,
                    alpha_concept=1.0):
    # Freeze encoder features deterministically (disable dropout etc.)
    encoder.eval()
    head.train()
    
    optimizer = AdamW(head.parameters(), lr=lr)
    steps = epochs * max(1, len(loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, steps//10), num_training_steps=steps)

    for i in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Compute frozen encoder features under no_grad
            with torch.no_grad():
                out = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                h = out.last_hidden_state[:, 0]
            logits, c, z_c = head(h, return_concepts=True)
            c_targets = batch.get("concepts", None)
            loss = head.loss(logits, batch["labels"], z_c=z_c, c_targets=c_targets.float(), alpha_concept=alpha_concept)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)

@torch.no_grad()
def evaluate(encoder, head, loader, device) -> float:
    encoder.eval(); head.eval()
    correct = 0; total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        h = out.last_hidden_state[:, 0]
        logits = head(h)
        preds = logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()
        total += preds.numel()
    return correct / max(1, total)

# ----------------------------
# LLM annotators
# ----------------------------
def _coerce_label_from_text(s: str) -> Optional[int]:
    if s is None: return None
    s = s.strip()
    try:
        v = int(s)
        if 0 <= v < 4: return v
    except Exception: pass
    low = s.lower()
    name2id = {"world":0, "sports":1, "business":2, "sci/tech":3, "science":3, "tech":3}
    for k,v in name2id.items():
        if k in low: return v
    return None

def annotate_labels_llm_parallel(texts: List[str], model="gpt-4o-mini", api_key_env="OPENAI_API_KEY", max_workers=4) -> List[Optional[int]]:
    
    client = OpenAI(api_key=os.getenv(api_key_env))

    sys_prompt = "You are a precise annotator for news classification."
    user_tmpl = (
        "Classify the text into one of: 0=World, 1=Sports, 2=Business, 3=Sci/Tech. "
        "Return only a single integer (0,1,2,3). Text: {text}"
    )

    def _one(t):
        msg = user_tmpl.format(text=t.replace("\\n"," "))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":msg}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        return _coerce_label_from_text(content)

    out = [None]*len(texts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, t): i for i,t in enumerate(texts)}
        for f in concurrent.futures.as_completed(futs):
            out[futs[f]] = f.result()
    return out

def annotate_concepts_llm_parallel(texts: List[str], concepts: List[str], model="gpt-4o-mini", api_key_env="OPENAI_API_KEY", max_workers=4) -> List[Optional[List[int]]]:

    client = OpenAI(api_key=os.getenv(api_key_env))

    sys_prompt = "You are a precise annotator. Tag each concept as 0/1."
    concept_list = "\n".join(f"{i}. {c}" for i,c in enumerate(concepts))
    user_tmpl = (
        "For this news text, return a JSON array of 0/1 flags of length {L} in the given order.\n"
        "Concepts:\n{concept_list}\n\nTEXT: {text}\nReturn ONLY the JSON array."
    )

    def _one(t):
        msg = user_tmpl.format(L=len(concepts), concept_list=concept_list, text=t.replace("\\n"," "))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":msg}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        try:
            arr = json.loads(content)
        except Exception:
            l = content.find("["); r = content.rfind("]")
            arr = json.loads(content[l:r+1])
        arr = [int(1 if int(x)>0 else 0) for x in arr]
        if len(arr) != len(concepts): return None
        return arr

    out = [None]*len(texts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, t): i for i,t in enumerate(texts)}
        for f in concurrent.futures.as_completed(futs):
            out[futs[f]] = f.result()
    return out

@torch.no_grad()
def annotate_with_classifier(encoder, head, tokenizer, texts: List[str], device) -> List[int]:
    out_labels = []
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        enc = {k: v.to(device) for k,v in enc.items()}
        feats = encoder(**enc).last_hidden_state[:,0]
        logits = head(feats)
        out_labels.append(int(logits.softmax(-1).argmax(-1).item()))
    return out_labels

# ----------------------------
# Pools
# ----------------------------
@dataclass
class Pools:
    labeled: List[int]
    unlabeled: List[int]

def init_pools(n_total: int, seed_size: int, seed: int) -> Pools:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total).tolist()
    return Pools(labeled=idx[:seed_size], unlabeled=idx[seed_size:])

# ----------------------------
# Main
# ----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed_size", type=int, default=200)
    parser.add_argument("--query_size", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--alpha_concept", type=float, default=1.0)
    parser.add_argument("--concept_l1", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--llm_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--llm_workers", type=int, default=8)
    args = parser.parse_args()

    print("="*20)
    print("Arguments:")
    print("="*20)
    for k,v in vars(args).items():
        print(f"{k}: {v}")

    # print lastest git commit hash
    git_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    print(f"Latest git commit hash: {git_commit_hash}")

    set_seed(args.seed)
    device = torch.device(args.device)

    raw = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.cache_dir)
    train_ds = HFDatasetWrapper(raw["train"], tokenizer, max_length=args.max_length)
    test_ds  = HFDatasetWrapper(raw["test"],  tokenizer, max_length=args.max_length)

    pools = init_pools(len(train_ds), args.seed_size, args.seed)
    print(f"Seed labeled={len(pools.labeled)}, unlabeled={len(pools.unlabeled)}")

    # Encoder (frozen)
    enc_cfg = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    encoder = AutoModel.from_pretrained(args.model_name, config=enc_cfg, cache_dir=args.cache_dir)
    for p in encoder.parameters(): p.requires_grad = False
    encoder.to(device)

    # CBM head
    head = CBMHead(hidden_size=enc_cfg.hidden_size, num_labels=4, n_concepts=N_CONCEPTS, hidden_task=256, concept_l1=args.concept_l1).to(device)

    # data loaders
    def make_loader(idxs, shuffle):
        subset = torch.utils.data.Subset(train_ds, idxs)
        return DataLoader(subset, batch_size=args.batch_size, shuffle=shuffle, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Annotate concepts for initial labeled pool so concept loss applies from round 1
    if len(pools.labeled) > 0:
        init_texts = [raw["train"][i]["text"] for i in pools.labeled]

        init_concepts_raw = annotate_concepts_llm_parallel(
            init_texts,
            AGNEWS_CONCEPTS,
            model=args.llm_model,
            api_key_env=args.llm_api_key_env,
            max_workers=args.llm_workers,
        )
        # Fallback to zeros for any failures
        fail_count = sum(1 for cv in init_concepts_raw if cv is None)
        init_concepts = [cv if cv is not None else [0] * N_CONCEPTS for cv in init_concepts_raw]
        print(
            f"LLM annotated concepts for initial {len(pools.labeled)} samples, "
            f"with {fail_count} failures."
        )
        train_ds.set_many_concept_overrides(pools.labeled, init_concepts)

    acc_list = []

    for r in range(args.rounds):
        print(f"\n=== Round {r}/{args.rounds} ===")
        # Train on current labeled set
        train_loader = make_loader(pools.labeled, shuffle=True)
        train_one_round(encoder, head, train_loader, device, lr=args.lr, epochs=args.epochs, alpha_concept=args.alpha_concept)
        acc = evaluate(encoder, head, test_loader, device)
        print(f"Test accuracy: {acc*100:.2f}%")
        acc_list.append(acc*100)

        # BADGE selection on a random subset of unlabeled to speed up (optional)
        probe_count = min(20000, len(pools.unlabeled))
        probe_idx = np.random.choice(pools.unlabeled, size=probe_count, replace=False).tolist() if probe_count>0 else []
        probe_loader = make_loader(probe_idx, shuffle=False)
        embs, map_idx = compute_gradient_embeddings(encoder, head, probe_loader, device)
        if embs.shape[0] == 0: break
        k = min(args.query_size, embs.shape[0])

        print(f"Selecting {k} via BADGE...")
        print(f"Gradient embedding shape: {embs.shape}")
        t = time.time()
        rel = kmeanspp_select_sklearn(embs, k, np.random.default_rng(args.seed + r))
        print(f"Selection took {time.time()-t:.2f}s.")
        newly_selected = [probe_idx[i] for i in rel]

        # Annotate labels
        texts = [raw["train"][i]["text"] for i in newly_selected]
        labels = annotate_labels_llm_parallel(texts, model=args.llm_model, api_key_env=args.llm_api_key_env, max_workers=args.llm_workers)
        # fallback to model for any None
        fallback_needed = [i for i,v in enumerate(labels) if v is None]
        if fallback_needed:
            raise RuntimeError("LLM label annotation failed for some samples; no fallback implemented.")
        print(f"LLM annotated labels for {len(newly_selected)} samples, with {sum(1 for x in labels if x is None)} failures (fallbacks applied).")
        labels = [int(x) for x in labels]
        # Concepts
        concept_vecs = annotate_concepts_llm_parallel(texts, AGNEWS_CONCEPTS, model=args.llm_model, api_key_env=args.llm_api_key_env, max_workers=args.llm_workers)
        concept_vecs = [cv if cv is not None else [0]*N_CONCEPTS for cv in concept_vecs]
        print(f"LLM annotated concepts for {len(newly_selected)} samples, with {sum(1 for x in concept_vecs if x is None)} failures.")

        # Store overrides
        train_ds.set_many_label_overrides(newly_selected, labels)
        train_ds.set_many_concept_overrides(newly_selected, concept_vecs)

        # Move from pool
        pools.labeled.extend(newly_selected)
        unl = set(pools.unlabeled)
        unl.difference_update(newly_selected)
        pools.unlabeled = list(unl)

    print("\n=== Final Results ===")

    # save to results/agnews
    from datetime import datetime
    os.makedirs("results/agnews/llm_cbm", exist_ok=True)
    with open(f"results/agnews/llm_cbm/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}", "w") as f:
        json.dump(acc_list, f)



    print("Done.")

if __name__ == "__main__":
    main()
