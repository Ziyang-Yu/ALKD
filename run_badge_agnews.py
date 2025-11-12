#!/usr/bin/env python3
"""
BADGE 主动学习在 AG News 文本分类任务上的应用
====================================================

本脚本实现了 BADGE (Batch Active learning by Diverse Gradient Embeddings) 主动学习算法，
用于 AG News 新闻分类任务。使用冻结的预训练编码器和轻量级分类头，通过 BADGE 算法
选择最有价值的未标注样本进行标注，逐步提升模型性能。

主要特点：
- 使用冻结的 DistilBERT 编码器提取特征
- 轻量级 MLP 分类头进行训练
- BADGE 算法选择样本（基于梯度嵌入的 k-means++）
- 使用真实标签（从数据集获取）

Author: Ziyang Yu
License: MIT
"""

import argparse
import os, subprocess
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
    AutoModel,
    AutoConfig,
)
from torch.optim import AdamW

torch.use_deterministic_algorithms(True)

# --------------------------
# 工具函数
# --------------------------

def set_seed(seed: int = 42):
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Classifier(nn.Module):
    """轻量级 MLP 分类头，基于冻结编码器的 CLS 嵌入"""
    def __init__(self, hidden_size=768, num_labels=4):
        super().__init__()
        self.dropout = nn.Dropout(0.1)  # Dropout 层用于正则化
        self.dense = nn.Linear(hidden_size, hidden_size)  # 全连接层
        self.activation = nn.GELU()  # GELU 激活函数
        self.classifier = nn.Linear(hidden_size, num_labels)  # 最终分类层
    
    def forward(self, x):
        """前向传播：对编码器特征进行分类"""
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.classifier(x)

@dataclass
class ALIndices:
    """主动学习索引池：已标注和未标注样本的索引"""
    labeled: List[int]      # 已标注样本的索引列表
    unlabeled: List[int]    # 未标注样本的索引列表

def split_initial_pool(n_total: int, init_size: int, seed: int = 42) -> ALIndices:
    """初始化主动学习池：随机选择初始标注样本"""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)  # 随机打乱索引
    init = idx[:init_size].tolist()      # 初始标注样本
    rest = idx[init_size:].tolist()     # 剩余未标注样本
    return ALIndices(labeled=init, unlabeled=rest)

# --------------------------
# 数据处理
# --------------------------

class HFDatasetWrapper(Dataset):
    """包装 HuggingFace 数据集，为分类任务生成 tokenized 批次"""
    def __init__(self, hf_dataset, tokenizer, text_key="text", max_length=128):
        self.ds = hf_dataset          # HuggingFace 数据集对象
        self.text_key = text_key      # 文本字段的键名
        self.tokenizer = tokenizer    # 分词器
        self.max_length = max_length  # 最大序列长度

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        """获取单个样本：tokenize 文本并返回标签"""
        item = self.ds[i]
        # 对文本进行 tokenization
        enc = self.tokenizer(
            item["text"],
            truncation=True,           # 截断超长文本
            max_length=self.max_length,
            padding="max_length",      # 填充到最大长度
            return_tensors="pt",       # 返回 PyTorch 张量
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}  # 移除批次维度
        enc["labels"] = torch.tensor(item["label"], dtype=torch.long)  # 添加标签
        return enc

def collate_fn(batch):
    """将批次中的样本堆叠成张量"""
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    return out

# --------------------------
# 训练和评估
# --------------------------

def train_one_round(encoder, classifier, loader, device, lr=5e-5, epochs=2, weight_decay=0.01, max_grad_norm=1.0):
    """训练一轮：只训练分类头，保持编码器冻结"""
    classifier.train()

    # 为不同参数组设置不同的权重衰减（bias 和 LayerNorm 不衰减）
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
    # 线性学习率调度器，带 warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, num_train_steps // 10),
        num_training_steps=num_train_steps,
    )
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # 在 no_grad 下计算冻结编码器的特征，节省内存和计算
            with torch.no_grad():
                encoder_outputs = encoder(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                hidden_states = encoder_outputs.last_hidden_state[:, 0]  # 取 [CLS] token 的嵌入

            logits = classifier(hidden_states)
            loss = criterion(logits, batch['labels'])

            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), max_grad_norm)  # 梯度裁剪
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

@torch.no_grad()
def evaluate(encoder, classifier, loader, device) -> float:
    """评估模型在测试集上的准确率"""
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
        hidden_states = encoder_outputs.last_hidden_state[:, 0]  # [CLS] 嵌入
        logits = classifier(hidden_states)
        preds = logits.argmax(dim=-1)  # 预测类别
        correct += (preds == batch["labels"]).sum().item()  # 统计正确预测数
        total += preds.numel()  # 统计总样本数

    return correct / max(1, total)  # 返回准确率

# --------------------------
# BADGE: 梯度嵌入计算
# --------------------------

@torch.no_grad()
def badge_gradient_embeddings(encoder, classifier, loader, device, num_classes: int) -> Tuple[np.ndarray, List[int]]:
    """
    计算 BADGE 梯度嵌入
    
    BADGE 算法通过计算梯度嵌入来选择样本，公式为：
    g(x) = flatten((p - e_yhat) ⊗ h)
    其中：
    - h 是 [CLS] token 的特征向量
    - p 是类别上的 softmax 概率分布
    - e_yhat 是预测类别的 one-hot 向量
    - ⊗ 表示外积
    
    这个嵌入近似于损失函数对分类头参数的梯度，用于衡量样本的信息量。
    """
    encoder.eval()
    classifier.eval()
    embs = []
    map_idx = []

    for batch in loader:
        # 保留标签在批次结构中，但不用于计算嵌入
        batch = {k: v.to(device) for k, v in batch.items()}

        encoder_outputs = encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        h = encoder_outputs.last_hidden_state[:, 0]      # [B, H] CLS 嵌入
        logits = classifier(h)                            # [B, C] 分类 logits

        probs = torch.softmax(logits, dim=-1)            # [B, C] 类别概率
        yhat = probs.argmax(dim=-1)                      # [B] 预测类别
        onehot = torch.zeros_like(probs)
        onehot[torch.arange(probs.size(0)), yhat] = 1.0  # [B, C] one-hot 向量
        delta = probs - onehot                           # [B, C] 概率与 one-hot 的差

        # 计算梯度嵌入：外积后展平
        g = (delta.unsqueeze(-1) * h.unsqueeze(1)).reshape(probs.size(0), -1)  # [B, C*H]
        embs.append(g.cpu().float().numpy())

        if "idx" in batch:
            map_idx.extend(batch["idx"].cpu().tolist())
        else:
            map_idx.extend([None] * probs.size(0))

    if len(embs) == 0:
        return np.zeros((0, 1), dtype=np.float32), []
    E = np.concatenate(embs, axis=0)

    # 归一化以提高稳定性（k-means++ 需要尺度不变的距离）
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.maximum(norms, 1e-8)
    return E.astype(np.float32), map_idx

# --------------------------
# k-means++ 风格的选择算法
# --------------------------

from sklearn.cluster import kmeans_plusplus
import numpy as np

def kmeanspp_select_sklearn(embs: np.ndarray, k: int, rng: np.random.Generator):
    """
    使用 sklearn 的 k-means++ 算法选择样本
    
    k-means++ 初始化策略：
    1. 随机选择第一个中心点
    2. 对于后续中心点，根据到最近中心的距离平方（D^2）进行加权采样
    这样可以确保选择的样本在梯度嵌入空间中具有多样性。
    """
    # sklearn 会自动进行 D^2 加权采样，返回 (centers, indices)
    _, indices = kmeans_plusplus(
        embs.astype(np.float32, copy=False),
        n_clusters=k,
        random_state=int(rng.integers(0, 2**32 - 1)),
    )
    return list(map(int, indices))


def kmeanspp_select(embs: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    """
    最小化的 k-means++ 初始化算法，包含对退化情况的鲁棒回退
    
    这是手动实现的 k-means++ 算法，用于在梯度嵌入空间中选择多样化的样本。
    """
    n = embs.shape[0]
    if n == 0 or k <= 0:
        return []
    if k > n:
        k = n

    # 第一个中心：选择 L2 范数最大的点
    norms = np.sum(embs * embs, axis=1)
    first = int(np.argmax(norms))
    centers = [first]

    # 到最近中心的距离平方
    d2 = np.sum((embs - embs[first]) ** 2, axis=1)

    print("Start to select centers via k-means++...")
    print(f"  Total points: {n}, Selecting k={k} centers.")

    for i in range(1, k):
        # D^2 加权采样；防止零/NaN 概率
        denom = np.maximum(d2.sum(), 1e-12)
        probs = d2 / denom
        if (not np.isfinite(probs).all()) or (probs.sum() <= 0) or np.allclose(d2, 0.0):
            # 退化情况：随机选择剩余点
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
# 主动学习循环
# --------------------------

def make_loader(dataset: Dataset, idxs: List[int], batch_size: int, shuffle: bool, add_indices: bool=False):
    """为数据集的子集创建 DataLoader"""
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
    """主函数：执行 BADGE 主动学习流程"""
    parser = argparse.ArgumentParser(description="BADGE 主动学习在 AG News 上的应用")
    # 模型参数
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="预训练模型名称")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    # 主动学习参数
    parser.add_argument("--seed_size", type=int, default=200, help="初始标注样本数量")
    parser.add_argument("--query_size", type=int, default=1000, help="每轮查询的样本数量")
    parser.add_argument("--rounds", type=int, default=8, help="主动学习轮数")
    # 训练参数
    parser.add_argument("--epochs", type=int, default=3, help="每轮训练的 epoch 数")
    parser.add_argument("--batch_size", type=int, default=64, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    # 系统参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace 缓存目录")
    args = parser.parse_args()

    # 设置随机种子以确保可复现性
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    # 打印 git commit hash 用于实验追踪
    git_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    print(f"Latest git commit hash: {git_commit_hash}")

    # 加载 AG News 数据集并动态 tokenize
    raw = load_dataset("ag_news")
    num_labels = 4  # AG News 有 4 个类别
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.cache_dir)

    train_ds = HFDatasetWrapper(raw["train"], tokenizer, max_length=args.max_length)
    test_ds  = HFDatasetWrapper(raw["test"],  tokenizer, max_length=args.max_length)

    # 初始化主动学习的标注/未标注池
    N = len(train_ds)
    pools = split_initial_pool(N, args.seed_size, seed=args.seed)
    print(f"Total train: {N}. Seed labeled: {len(pools.labeled)}. Unlabeled pool: {len(pools.unlabeled)}.")

    # 加载编码器（冻结）和分类头
    encoder_config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    encoder = AutoModel.from_pretrained(args.model_name, config=encoder_config, cache_dir=args.cache_dir)
    for p in encoder.parameters():
        p.requires_grad = False  # 冻结编码器参数
    encoder = encoder.to(device)

    classifier = Classifier(hidden_size=encoder_config.hidden_size, num_labels=num_labels).to(device)

    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    # 主动学习主循环
    for r in range(args.rounds + 1):
        print("=" * 80)
        print(f"Round {r}/{args.rounds} — Labeled {len(pools.labeled)} | Unlabeled {len(pools.unlabeled)}")

        # 在当前标注集上训练分类器，然后评估
        train_loader = make_loader(train_ds, pools.labeled, batch_size=args.batch_size, shuffle=True)
        if len(pools.labeled) > 0:
            train_one_round(encoder, classifier, train_loader, device, lr=args.lr, epochs=args.epochs)
        acc = evaluate(encoder, classifier, test_loader, device)
        print(f"[Eval] Test accuracy: {acc*100:.2f}%")

        if r == args.rounds:
            break

        # BADGE 选择：计算梯度嵌入，然后使用 k-means++ 选择
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
        select_rel = kmeanspp_select_sklearn(embs, k, rng)  # 相对于 pools.unlabeled 的索引
        print(f"Selection took {time.time() - t:.2f} seconds.")
        newly_selected = [pools.unlabeled[i] for i in select_rel]

        # 将选中的样本从未标注池移到标注池（使用真实标签）
        pools.labeled.extend(newly_selected)
        mask = np.ones(len(pools.unlabeled), dtype=bool)
        mask[select_rel] = False
        pools.unlabeled = [pools.unlabeled[i] for i in range(len(mask)) if mask[i]]

    print("Done.")

if __name__ == "__main__":
    main()
