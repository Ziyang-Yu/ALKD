#!/usr/bin/env python3
"""
BADGE 主动学习在 AG News 上的应用（使用 LLM 标注器）
===================================================

本脚本实现了 BADGE 主动学习算法，使用 LLM（大语言模型）作为标注器来标注选中的样本。
与基础版本不同，这里使用 LLM 来获取标签，而不是使用数据集的真实标签。

主要特点：
- 使用冻结的 DistilBERT 编码器提取特征
- 轻量级 MLP 分类头进行训练
- BADGE 算法选择样本（基于梯度嵌入的 k-means++）
- 使用 LLM（OpenAI API）进行样本标注
- 支持并行 LLM 调用以提高效率
- 对于 LLM 标注失败的情况，回退到模型预测

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
    """包装 HuggingFace 数据集，为分类任务生成 tokenized 批次
    
    支持通过标签覆盖机制，允许在运行时插入 LLM 标注的标签。
    这样可以在主动学习过程中动态更新样本的标签。
    """
    def __init__(self, hf_dataset, tokenizer, text_key="text", max_length=128,
                 label_overrides: Optional[Dict[int, int]] = None):
        self.ds = hf_dataset          # HuggingFace 数据集对象
        self.text_key = text_key      # 文本字段的键名
        self.tokenizer = tokenizer    # 分词器
        self.max_length = max_length  # 最大序列长度
        self.label_overrides: Dict[int, int] = label_overrides or {}  # 标签覆盖字典

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        """获取单个样本：tokenize 文本并返回标签（优先使用覆盖标签）"""
        item = self.ds[i]
        # 对文本进行 tokenization
        enc = self.tokenizer(
            item[self.text_key],
            truncation=True,           # 截断超长文本
            max_length=self.max_length,
            padding="max_length",      # 填充到最大长度
            return_tensors="pt",       # 返回 PyTorch 张量
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}  # 移除批次维度
        # 如果存在覆盖标签则使用，否则使用数据集原始标签
        if i in self.label_overrides:
            label = int(self.label_overrides[i])
        else:
            label = int(item["label"])  # 从 HuggingFace 数据集获取
        enc["labels"] = torch.tensor(label, dtype=torch.long)
        # 始终暴露原始数据集索引，供下游使用
        enc["idx"] = torch.tensor(i, dtype=torch.long)
        return enc

    def set_label_override(self, i: int, label: int):
        """设置单个样本的标签覆盖"""
        self.label_overrides[int(i)] = int(label)

    def set_many_label_overrides(self, indices: List[int], labels: List[int]):
        """批量设置标签覆盖（用于 LLM 标注结果）"""
        for ii, ll in zip(indices, labels):
            self.set_label_override(int(ii), int(ll))

def collate_fn(batch):
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    return out

# --------------------------
# 训练和评估
# --------------------------

def train_one_round(encoder, classifier, loader, device, lr=5e-5, epochs=2, weight_decay=0.01, max_grad_norm=1.0):
    """训练一轮：只训练分类头，保持编码器冻结"""
    # 确保冻结的编码器以确定性模式运行（无 dropout）
    encoder.eval()
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
# 编码器特征预处理和缓存
# --------------------------
@torch.no_grad()
def preprocess_encoder_features(encoder, dataset, device, cache_path: str, batch_size: int = 2048):
    """
    预处理并保存所有样本的编码器特征（CLS embeddings）
    
    Args:
        encoder: 冻结的编码器模型
        dataset: 数据集
        device: 计算设备
        cache_path: 缓存文件路径
        batch_size: 批处理大小
    
    Returns:
        特征矩阵 [N, H] 和索引列表
    """
    if os.path.exists(cache_path):
        print(f"Loading preprocessed features from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        features = data['features']
        indices = data['indices'].tolist()
        print(f"Loaded {len(features)} preprocessed features.")
        return features, indices
    
    print(f"Preprocessing encoder features for {len(dataset)} samples...")
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    all_features = []
    all_indices = []
    t0 = time.time()
    
    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        encoder_outputs = encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        h = encoder_outputs.last_hidden_state[:, 0]  # [B, H] CLS 嵌入
        all_features.append(h.cpu().float().numpy())
        
        if "idx" in batch:
            all_indices.extend(batch["idx"].cpu().tolist())
        else:
            # 如果没有 idx，使用批次索引
            batch_start = batch_idx * batch_size
            all_indices.extend(range(batch_start, batch_start + h.size(0)))
        
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            progress = (batch_idx + 1) / len(loader) * 100
            print(f"  Progress: {progress:.1f}% ({batch_idx + 1}/{len(loader)} batches, {elapsed:.1f}s)")
    
    features = np.concatenate(all_features, axis=0)
    
    # 保存到文件
    print(f"Saving preprocessed features to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, features=features, indices=np.array(all_indices))
    print(f"Saved {len(features)} features. Total time: {time.time() - t0:.2f}s")
    
    return features, all_indices

# --------------------------
# BADGE: 梯度嵌入计算
# --------------------------

@torch.no_grad()
def badge_gradient_embeddings(encoder, classifier, loader, device, num_classes: int, cached_features: Optional[Dict[int, np.ndarray]] = None) -> Tuple[np.ndarray, List[int]]:
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
    
    Args:
        encoder: 编码器（如果 cached_features 为 None 则使用）
        classifier: 分类器
        loader: 数据加载器
        device: 计算设备
        num_classes: 类别数量
        cached_features: 可选的预计算特征字典 {idx: feature_vector}
    """
    classifier.eval()
    embs = []
    map_idx = []

    if cached_features is not None:
        # 使用预计算的特征
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 获取批次索引（HFDatasetWrapper 总是包含 idx）
            if "idx" in batch:
                batch_indices = batch["idx"].cpu().tolist()
                # 从缓存中获取特征
                h_list = [cached_features[idx] for idx in batch_indices]
                h = torch.tensor(np.stack(h_list), dtype=torch.float32).to(device)  # [B, H]
            else:
                # 如果没有 idx，回退到编码器计算
                encoder.eval()
                encoder_outputs = encoder(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                h = encoder_outputs.last_hidden_state[:, 0]  # [B, H] CLS 嵌入
                batch_indices = [None] * h.size(0)
            
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
                map_idx.extend(batch_indices)
    else:
        # 原始方法：通过编码器计算特征
        encoder.eval()
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

# --------------------------
# LLM 标注器
# --------------------------

# 数据集配置字典（不包含概念，因为此文件不使用 CBM）
DATASET_CONFIGS = {
    "ag_news": {
        "dataset_name": "ag_news",
        "text_key": "text",
        "label_key": "label",
        "labels": ["World", "Sports", "Business", "Sci/Tech"],
        "num_labels": 4,
        "result_dir": "agnews",
    },
    "imdb": {
        "dataset_name": "imdb",
        "text_key": "text",
        "label_key": "label",
        "labels": ["negative", "positive"],
        "num_labels": 2,
        "result_dir": "imdb",
    },
    "amazon_polarity": {
        "dataset_name": "amazon_polarity",
        "text_key": "content",
        "label_key": "label",
        "labels": ["negative", "positive"],
        "num_labels": 2,
        "result_dir": "amazon",
    },
    "yelp_polarity": {
        "dataset_name": "yelp_polarity",
        "text_key": "text",
        "label_key": "label",
        "labels": ["negative", "positive"],
        "num_labels": 2,
        "result_dir": "yelp",
    },
}

def _parse_llm_label(text: str, num_labels: int = 4, labels: Optional[List[str]] = None) -> Optional[int]:
    """从 LLM 响应字符串中提取单个标签 ID
    
    接受数字类别 ID 或标签名称。
    如果解析失败则返回 None。
    
    Args:
        text: LLM 响应字符串
        num_labels: 标签总数
        labels: 标签名称列表（用于映射）
    """
    if text is None:
        return None
    s = str(text).strip()
    # 首先尝试解析为整数
    try:
        val = int(s)
        if 0 <= val < num_labels:
            return val
    except Exception:
        pass
    # 尝试查找标签名称
    if labels:
        label_name_to_id = {name.lower(): idx for idx, name in enumerate(labels)}
        low = s.lower()
        for name, idx in label_name_to_id.items():
            if name in low:
                return idx
    return None

def annotate_with_llm_openai_single(text: str, model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY",
                                    system_prompt: Optional[str] = None, temperature: float = 0.0,
                                    max_retries: int = 3, labels: Optional[List[str]] = None) -> int:
    """使用单次 LLM 调用标注单个文本
    
    返回单个整数标签 ID。如果重试后仍然失败，抛出 RuntimeError，
    以便调用者决定如何回退。
    
    Args:
        text: 待标注的文本
        model: LLM 模型名称
        api_key_env: API 密钥环境变量名
        system_prompt: 系统提示（可选）
        temperature: 温度参数
        max_retries: 最大重试次数
        labels: 标签名称列表（用于生成提示）
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {api_key_env} for LLM annotator.")

    try:
        from openai import OpenAI  # OpenAI SDK v1
    except Exception as e:
        raise RuntimeError("openai package not installed. pip install openai>=1.0.0") from e

    client = OpenAI(api_key=api_key)

    if labels is None:
        labels = ["World", "Sports", "Business", "Sci/Tech"]
    
    label_str = ", ".join([f"{i}={l}" for i, l in enumerate(labels)])
    
    system = system_prompt or (
        f"You are a careful annotation assistant for text classification.\n"
        f"Return only the label id (0-{len(labels)-1}) for the input.\n"
        f"Label mapping: {label_str}.\n"
    )

    sanitized = text.strip().replace("\n", " ")
    user_prompt = (
        f"Classify the following text into one of: {label_str}.\n"
        f"Return only one integer (0-{len(labels)-1}).\n\n"
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
            label = _parse_llm_label(content, num_labels=len(labels), labels=labels)
            if label is None:
                # try permissive parse
                import re
                m = re.search(rf"(?<!\d)([0-{len(labels)-1}])(?!\d)", content)
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
                                      max_retries: int = 3, max_workers: int = 16, labels: Optional[List[str]] = None) -> List[Optional[int]]:
    """并行标注多个文本，每个文本使用一次 LLM 调用
    
    返回标签列表，失败的情况用 None 表示。调用者可以
    选择性地仅对失败的项进行回退。
    
    Args:
        texts: 待标注的文本列表
        model: LLM 模型名称
        api_key_env: API 密钥环境变量名
        system_prompt: 系统提示（可选）
        temperature: 温度参数
        max_retries: 最大重试次数
        max_workers: 并行线程数
        labels: 标签名称列表（用于生成提示）
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
            labels=labels,
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
    """回退方案：当 LLM 不可用时，使用当前模型预测作为伪标签"""
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
    """主函数：执行 BADGE 主动学习流程（使用 LLM 标注）"""
    parser = argparse.ArgumentParser(description="BADGE 主动学习在 AG News 上的应用（LLM 标注）")
    # 模型参数
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="预训练模型名称")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    # 数据集参数
    parser.add_argument("--dataset", type=str, default="ag_news", 
                        choices=["ag_news", "imdb", "amazon_polarity", "yelp_polarity"],
                        help="选择数据集：ag_news, imdb, amazon_polarity, yelp_polarity")
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
    # LLM 标注器选项
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini",
                        help="OpenAI 提供商的 LLM 模型名称")
    parser.add_argument("--llm_api_key_env", type=str, default="OPENAI_API_KEY",
                        help="存储 LLM API 密钥的环境变量名")
    parser.add_argument("--llm_batch_size", type=int, default=32,
                        help="(已弃用) 之前用于批量请求；设置 --use_llm_annotator 时忽略")
    parser.add_argument("--llm_workers", type=int, default=8,
                        help="每个文本 LLM 调用的并行线程数")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="可选的 Hugging Face 缓存目录")

    args = parser.parse_args()

    # 打印 git commit hash 用于实验追踪
    git_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    print(f"Latest git commit hash: {git_commit_hash}")

    # 设置随机种子以确保可复现性
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    # 获取数据集配置
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"不支持的数据集: {args.dataset}。支持的数据集: {list(DATASET_CONFIGS.keys())}")
    config = DATASET_CONFIGS[args.dataset]
    num_labels = config["num_labels"]
    
    print(f"\n使用数据集: {args.dataset}")
    print(f"标签数量: {num_labels}")
    print(f"标签: {config['labels']}")

    # 加载数据集并动态 tokenize
    raw = load_dataset(config["dataset_name"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.cache_dir)

    train_ds = HFDatasetWrapper(raw["train"], tokenizer, text_key=config["text_key"], max_length=args.max_length)
    test_ds  = HFDatasetWrapper(raw["test"],  tokenizer, text_key=config["text_key"], max_length=args.max_length)

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

    # 预处理并缓存编码器特征
    cache_dir = f"cache/{args.dataset}/{args.model_name.replace('/', '_')}"
    cache_path = f"{cache_dir}/encoder_features.npz"
    print(f"\n=== Preprocessing Encoder Features ===")
    t0 = time.time()
    features_array, feature_indices = preprocess_encoder_features(
        encoder, train_ds, device, cache_path, batch_size=2048
    )
    preprocessing_time = time.time() - t0
    print(f"[Time] Total preprocessing time: {preprocessing_time:.2f}s")
    
    # 创建特征字典以便快速查找
    cached_features_dict = {idx: features_array[i] for i, idx in enumerate(feature_indices)}
    print(f"Created feature cache dictionary with {len(cached_features_dict)} entries.")

    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    acc_list = []

    # 主动学习主循环
    for r in range(args.rounds):
        print("=" * 80)
        print(f"Round {r}/{args.rounds} — Labeled {len(pools.labeled)} | Unlabeled {len(pools.unlabeled)}")

        # 在当前标注集上训练分类器，然后评估
        train_loader = make_loader(train_ds, pools.labeled, batch_size=args.batch_size, shuffle=True)
        if len(pools.labeled) > 0:
            train_one_round(encoder, classifier, train_loader, device, lr=args.lr, epochs=args.epochs)
        acc = evaluate(encoder, classifier, test_loader, device)
        print(f"[Eval] Test accuracy: {acc*100:.2f}%")
        acc_list.append(acc*100)

        if r == args.rounds:
            break

        # BADGE 选择：计算梯度嵌入，然后使用 k-means++ 选择
        # 使用更大的 batch size 用于推理任务（H200 GPU 可以处理更大的 batch）
        # 使用预计算的编码器特征，避免重复通过编码器
        pool_loader = make_loader(train_ds, pools.unlabeled, batch_size=2048, shuffle=False)
        embs, _ = badge_gradient_embeddings(encoder, classifier, pool_loader, device, num_classes=num_labels, cached_features=cached_features_dict)
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

        # 使用 LLM 标注选中的样本（失败时回退到模型预测）
        print(f"Annotating {len(newly_selected)} selected samples...")
        selected_texts = [raw["train"][i][config["text_key"]] for i in newly_selected]
        llm_labels: List[int] = []

        # 在并行线程中执行每个文本的 LLM 调用
        parallel_labels = annotate_with_llm_openai_parallel(
            selected_texts,
            model=args.llm_model,
            api_key_env=args.llm_api_key_env,
            max_workers=max(1, int(args.llm_workers)),
            labels=config["labels"],
        )
        # 识别失败项并仅对它们进行回退
        failed_idx = [i for i, v in enumerate(parallel_labels) if v is None]
        if failed_idx:
            failed_texts = [selected_texts[i] for i in failed_idx]
            print(f"LLM failed on {len(failed_idx)} items; falling back to model for those.")
            fallback_labels = annotate_with_classifier(encoder, classifier, tokenizer, failed_texts, device)
            for rel, lab in zip(failed_idx, fallback_labels):
                parallel_labels[rel] = int(lab)
        llm_labels = [int(x) for x in parallel_labels]  # type: ignore
        print("LLM annotations received (with selective fallback if needed).")

        # 存储覆盖标签，以便后续训练使用这些标签
        train_ds.set_many_label_overrides(newly_selected, llm_labels)

        # 将选中的样本从未标注池移到标注池
        pools.labeled.extend(newly_selected)
        mask = np.ones(len(pools.unlabeled), dtype=bool)
        mask[select_rel] = False
        pools.unlabeled = [pools.unlabeled[i] for i in range(len(mask)) if mask[i]]



    print("\n=== Final Results ===")

    # 保存结果
    import json
    from datetime import datetime
    result_dir = f"results/{config['result_dir']}/llm"
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}", "w") as f:
        json.dump(acc_list, f)

    print("Done.")

if __name__ == "__main__":
    main()
