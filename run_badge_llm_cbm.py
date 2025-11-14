#!/usr/bin/env python3
"""
BADGE 主动学习在 AG News 上的应用（使用 LLM 标注器和概念瓶颈模型 CBM）
========================================================================

本脚本实现了 BADGE 主动学习算法，结合概念瓶颈模型（Concept Bottleneck Model, CBM）
和 LLM（大语言模型）标注器。与基础版本不同，这里使用 LLM 来获取标签和概念标注。

主要特点：
- 使用冻结的 DistilBERT 编码器提取特征
- 概念瓶颈模型（CBM）分类头：先预测概念，再基于概念预测标签
- BADGE 算法选择样本（基于梯度嵌入的 k-means++）
- 使用 LLM（OpenAI API）进行样本标签和概念标注
- 支持并行 LLM 调用以提高效率
- 对于 LLM 标注失败的情况，提供回退机制

Author: Ziyang Yu
License: MIT
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
from sklearn.cluster import kmeans_plusplus

from dotenv import load_dotenv

import os
load_dotenv()   # 从 .env 文件读取环境变量
key = os.getenv("OPENAI_API_KEY")

# ----------------------------
# 可复现性设置
# ----------------------------
def set_seed(seed: int):
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# 数据集配置和概念定义
# ----------------------------
# 定义用于概念瓶颈模型的概念集合，这些概念描述了文本的语义特征

# AG News 概念
AGNEWS_CONCEPTS = [
    "mentions_person", "mentions_org", "mentions_location", "mentions_country",
    "mentions_company", "mentions_stock_or_market", "mentions_trade_or_deal", "mentions_economy",
    "mentions_sport", "mentions_team_or_league", "mentions_match_score", "mentions_player",
    "mentions_technology", "mentions_product_or_device", "mentions_research_or_science", "mentions_internet_or_software",
    "mentions_government_or_policy", "mentions_election_or_vote", "mentions_conflict_or_war", "mentions_disaster_or_crime",
    "mentions_health_or_medicine", "mentions_entertainment", "mentions_energy_or_environment", "mentions_space_or_weather",
]

# IMDB 电影评论概念
IMDB_CONCEPTS = [
    "positive_emotion", "negative_emotion", "mentions_plot", "mentions_character",
    "mentions_acting", "mentions_director", "mentions_cinematography", "mentions_music",
    "mentions_dialogue", "mentions_ending", "mentions_genre", "mentions_action",
    "mentions_romance", "mentions_comedy", "mentions_drama", "mentions_horror",
    "mentions_suspense", "mentions_visual_effects", "mentions_setting", "mentions_costume",
    "mentions_sequel", "mentions_remake", "mentions_award", "mentions_box_office",
]

# Amazon Reviews 概念
AMAZON_CONCEPTS = [
    "positive_sentiment", "negative_sentiment", "mentions_price", "mentions_quality",
    "mentions_delivery", "mentions_packaging", "mentions_durability", "mentions_appearance",
    "mentions_functionality", "mentions_ease_of_use", "mentions_value_for_money", "mentions_customer_service",
    "mentions_size", "mentions_color", "mentions_material", "mentions_brand",
    "mentions_warranty", "mentions_instructions", "mentions_assembly", "mentions_cleaning",
    "mentions_storage", "mentions_portability", "mentions_safety", "mentions_compatibility",
]

# Yelp Review 概念
YELP_CONCEPTS = [
    "positive_experience", "negative_experience", "mentions_food_quality", "mentions_service",
    "mentions_atmosphere", "mentions_price", "mentions_location", "mentions_wait_time",
    "mentions_cleanliness", "mentions_portion_size", "mentions_presentation", "mentions_menu_variety",
    "mentions_drinks", "mentions_dessert", "mentions_ambiance", "mentions_staff_attitude",
    "mentions_reservation", "mentions_parking", "mentions_accessibility", "mentions_noise_level",
    "mentions_special_occasion", "mentions_group_dining", "mentions_takeout", "mentions_delivery",
]

# 数据集配置字典
DATASET_CONFIGS = {
    "ag_news": {
        "dataset_name": "ag_news",
        "text_key": "text",
        "label_key": "label",
        "concepts": AGNEWS_CONCEPTS,
        "labels": ["World", "Sports", "Business", "Sci/Tech"],
        "num_labels": 4,
        "result_dir": "agnews",
    },
    "imdb": {
        "dataset_name": "imdb",
        "text_key": "text",
        "label_key": "label",
        "concepts": IMDB_CONCEPTS,
        "labels": ["negative", "positive"],
        "num_labels": 2,
        "result_dir": "imdb",
    },
    "amazon_polarity": {
        "dataset_name": "amazon_polarity",
        "text_key": "content",
        "label_key": "label",
        "concepts": AMAZON_CONCEPTS,
        "labels": ["negative", "positive"],
        "num_labels": 2,
        "result_dir": "amazon",
    },
    "yelp_polarity": {
        "dataset_name": "yelp_polarity",
        "text_key": "text",
        "label_key": "label",
        "concepts": YELP_CONCEPTS,
        "labels": ["negative", "positive"],
        "num_labels": 2,
        "result_dir": "yelp",
    },
}

# ----------------------------
# 概念瓶颈模型（CBM）分类头
# ----------------------------
class CBMHead(nn.Module):
    """
    概念瓶颈模型分类头
    
    CBM 的架构：
    1. 编码器特征 -> 概念预测（concept logits）
    2. 概念预测（sigmoid）-> 任务分类（task logits）
    
    这种设计使得模型首先学习可解释的概念，然后基于这些概念进行分类。
    """
    def __init__(self, hidden_size=768, num_labels=4, n_concepts=N_CONCEPTS, hidden_task=256, dropout=0.1,
                 concept_l1=1e-4, detach_concepts=False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Dropout 层用于正则化
        # 概念预测层：从编码器特征预测概念
        self.concept_linear = nn.Linear(hidden_size, n_concepts, bias=True)
        # 任务分类头：基于概念预测最终标签
        self.task_head = nn.Sequential(
            nn.Linear(n_concepts, hidden_task, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_task, num_labels, bias=True),
        )
        self.concept_l1 = float(concept_l1)  # 概念 L1 正则化系数
        self.detach_concepts = bool(detach_concepts)  # 是否在任务头中分离概念梯度

    def forward(self, feats: torch.Tensor, return_concepts: bool = False):
        """
        前向传播
        
        Args:
            feats: 编码器输出的特征向量 [B, hidden_size]
            return_concepts: 是否返回概念预测（用于损失计算）
        
        Returns:
            如果 return_concepts=True: (logits, concepts, concept_logits)
            否则: logits
        """
        h = self.dropout(feats)
        z_c = self.concept_linear(h)  # [B, C] 概念 logits
        c = torch.sigmoid(z_c)        # [B, C] 概念概率（0-1）
        if self.detach_concepts:
            c = c.detach()  # 分离梯度，只让任务头学习概念到标签的映射
        logits = self.task_head(c)    # [B, K] 任务分类 logits
        if return_concepts:
            return logits, c, z_c
        return logits

    def loss(self, logits, y, z_c=None, c_targets=None, alpha_concept=1.0):
        """
        计算 CBM 的损失函数
        
        损失 = 分类损失 + 概念损失 + L1 正则化
        
        Args:
            logits: 任务分类 logits [B, K]
            y: 真实标签 [B]
            z_c: 概念 logits [B, C]（可选）
            c_targets: 真实概念标签 [B, C]（可选）
            alpha_concept: 概念损失的权重
        
        Returns:
            总损失
        """
        ce = F.cross_entropy(logits, y)  # 分类损失
        cl = torch.tensor(0.0, device=logits.device)
        # 如果有概念标签，计算概念预测损失
        if z_c is not None and c_targets is not None:
            cl = alpha_concept * F.binary_cross_entropy_with_logits(z_c, c_targets)
        # L1 正则化：鼓励稀疏的概念激活
        l1 = self.concept_l1 * (z_c.abs().mean() if z_c is not None else 0.0)
        return ce + cl + l1

# ----------------------------
# 数据集包装器
# ----------------------------
class HFDatasetWrapper(Dataset):
    """
    包装 HuggingFace 数据集，为分类任务生成 tokenized 批次
    
    支持通过标签和概念覆盖机制，允许在运行时插入 LLM 标注的标签和概念。
    这样可以在主动学习过程中动态更新样本的标注。
    """
    def __init__(self, hf_dataset, tokenizer, text_key="text", max_length=128):
        self.ds = hf_dataset  # HuggingFace 数据集对象
        self.text_key = text_key  # 文本字段的键名
        self.tok = tokenizer  # 分词器
        self.max_length = max_length  # 最大序列长度
        self.label_overrides: Dict[int, int] = {}  # 标签覆盖字典
        self.concept_overrides: Dict[int, List[int]] = {}  # 概念覆盖字典

    def __len__(self): 
        return len(self.ds)

    def __getitem__(self, i):
        """获取单个样本：tokenize 文本并返回标签和概念（优先使用覆盖值）"""
        item = self.ds[i]
        # 对文本进行 tokenization
        enc = self.tok(
            item[self.text_key],
            truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}  # 移除批次维度
        # 标签：如果存在覆盖标签则使用，否则使用数据集原始标签
        lab = self.label_overrides.get(i, int(item["label"]))
        enc["labels"] = torch.tensor(lab, dtype=torch.long)
        # 概念（可选）：如果存在覆盖概念则使用
        if i in self.concept_overrides:
            enc["concepts"] = torch.tensor(self.concept_overrides[i], dtype=torch.float)
        enc["idx"] = torch.tensor(i, dtype=torch.long)  # 保存原始索引
        return enc

    def set_many_label_overrides(self, indices: List[int], labels: List[int]):
        """批量设置标签覆盖（用于 LLM 标注结果）"""
        for i, y in zip(indices, labels):
            self.label_overrides[int(i)] = int(y)

    def set_many_concept_overrides(self, indices: List[int], concept_matrix: List[List[int]]):
        """批量设置概念覆盖（用于 LLM 标注结果）"""
        for i, cv in zip(indices, concept_matrix):
            self.concept_overrides[int(i)] = [int(x) for x in cv]

def collate_fn(batch):
    """将批次中的样本堆叠成张量"""
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    return out

# ----------------------------
# BADGE 梯度嵌入计算
# ----------------------------
@torch.no_grad()
def compute_gradient_embeddings(encoder, head, loader, device) -> Tuple[np.ndarray, List[int]]:
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
    head.eval()
    embs, map_idx = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        h = out.last_hidden_state[:, 0]  # [B, H] CLS 嵌入
        logits = head(h)                 # [B, K] 分类 logits
        probs = logits.softmax(-1)        # [B, K] 类别概率
        yhat = probs.argmax(-1)           # [B] 预测类别
        onehot = torch.zeros_like(probs)
        onehot[torch.arange(probs.size(0)), yhat] = 1.0  # [B, K] one-hot 向量
        delta = probs - onehot           # [B, K] 概率与 one-hot 的差
        # 计算梯度嵌入：外积后展平
        g = (delta.unsqueeze(-1) * h.unsqueeze(1)).reshape(probs.size(0), -1)  # [B, K*H]
        embs.append(g.cpu().float().numpy())
        map_idx.extend(batch["idx"].cpu().tolist())
    if not embs:
        return np.zeros((0, 1), dtype=np.float32), []
    return np.concatenate(embs, axis=0), map_idx

# def kmeanspp_select_sklearn(embs: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
#     """
#     使用 sklearn 的 k-means++ 算法选择样本
    
#     k-means++ 初始化策略：
#     1. 随机选择第一个中心点
#     2. 对于后续中心点，根据到最近中心的距离平方（D^2）进行加权采样
#     这样可以确保选择的样本在梯度嵌入空间中具有多样性。
    
#     注意：这里使用 KMeans 的一次迭代 + k-means++ 初始化来近似 k-means++ 选择。
#     """
#     if embs.shape[0] == 0: 
#         return []
#     k = min(k, embs.shape[0])
#     # KMeans with one iteration + k-means++ init yields centers ~= seeding
#     km = KMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=1, random_state=int(rng.integers(1e9)))
#     km.fit(embs)
#     # 选择距离聚类中心最近的样本
#     from sklearn.metrics import pairwise_distances_argmin_min
#     idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, embs, metric="euclidean")
#     return idx.tolist()

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

# ----------------------------
# 训练和评估
# ----------------------------
def train_one_round(encoder, head, loader, device, lr=5e-5, epochs=3, weight_decay=0.01, max_grad_norm=1.0,
                    alpha_concept=1.0):
    """
    训练一轮：只训练 CBM 分类头，保持编码器冻结
    
    Args:
        encoder: 冻结的编码器
        head: CBM 分类头
        loader: 训练数据加载器
        device: 计算设备
        lr: 学习率
        epochs: 训练轮数
        weight_decay: 权重衰减
        max_grad_norm: 梯度裁剪的最大范数
        alpha_concept: 概念损失的权重
    """
    # 确保冻结的编码器以确定性模式运行（无 dropout）
    encoder.eval()
    head.train()
    
    optimizer = AdamW(head.parameters(), lr=lr)
    steps = epochs * max(1, len(loader))
    # 线性学习率调度器，带 warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, steps//10), num_training_steps=steps)

    for i in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # 在 no_grad 下计算冻结编码器的特征，节省内存和计算
            with torch.no_grad():
                out = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                h = out.last_hidden_state[:, 0]  # [CLS] 嵌入
            # 前向传播，返回 logits、概念和概念 logits
            logits, c, z_c = head(h, return_concepts=True)
            c_targets = batch.get("concepts", None)  # 概念标签（如果存在）
            # 计算损失（包括分类损失、概念损失和 L1 正则化）
            loss = head.loss(logits, batch["labels"], z_c=z_c, c_targets=c_targets.float() if c_targets is not None else None, alpha_concept=alpha_concept)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_grad_norm)  # 梯度裁剪
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

@torch.no_grad()
def evaluate(encoder, head, loader, device) -> float:
    """
    评估模型在测试集上的准确率
    
    Args:
        encoder: 编码器
        head: CBM 分类头
        loader: 测试数据加载器
        device: 计算设备
    
    Returns:
        测试集准确率
    """
    encoder.eval()
    head.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        h = out.last_hidden_state[:, 0]  # [CLS] 嵌入
        logits = head(h)  # 分类 logits
        preds = logits.argmax(-1)  # 预测类别
        correct += (preds == batch["labels"]).sum().item()  # 统计正确预测数
        total += preds.numel()  # 统计总样本数
    return correct / max(1, total)  # 返回准确率

# ----------------------------
# LLM 标注器
# ----------------------------
def _coerce_label_from_text(s: str, num_labels: int = 4) -> Optional[int]:
    """
    从 LLM 响应字符串中提取单个标签 ID
    
    接受数字类别 ID 或标签名称。
    如果解析失败则返回 None。
    
    Args:
        s: LLM 响应字符串
        num_labels: 标签总数
    """
    if s is None: 
        return None
    s = s.strip()
    # 首先尝试解析为整数
    try:
        v = int(s)
        if 0 <= v < num_labels: 
            return v
    except Exception: 
        pass
    # 尝试查找标签名称（通用映射）
    low = s.lower()
    name2id = {
        "world": 0, "sports": 1, "business": 2, "sci/tech": 3, "science": 3, "tech": 3,
        "negative": 0, "positive": 1, "neg": 0, "pos": 1,
    }
    for k, v in name2id.items():
        if k in low and v < num_labels: 
            return v
    return None

def annotate_labels_llm_parallel(texts: List[str], model="gpt-4o-mini", api_key_env="OPENAI_API_KEY", 
                                  max_workers=4, labels: Optional[List[str]] = None) -> List[Optional[int]]:
    """
    并行标注多个文本的标签，每个文本使用一次 LLM 调用
    
    返回标签列表，失败的情况用 None 表示。调用者可以
    选择性地仅对失败的项进行回退。
    
    Args:
        texts: 待标注的文本列表
        model: LLM 模型名称
        api_key_env: 存储 API 密钥的环境变量名
        max_workers: 并行线程数
        labels: 标签名称列表（用于生成提示），如果为 None 则使用默认的 AG News 标签
    
    Returns:
        标签列表，失败项为 None
    """
    client = OpenAI(api_key=os.getenv(api_key_env))

    if labels is None:
        labels = ["World", "Sports", "Business", "Sci/Tech"]
    
    label_str = ", ".join([f"{i}={l}" for i, l in enumerate(labels)])
    sys_prompt = "You are a precise annotator for text classification."
    user_tmpl = (
        f"Classify the text into one of: {label_str}. "
        f"Return only a single integer (0-{len(labels)-1}). Text: {{text}}"
    )

    def _one(t):
        """单个文本的标注任务"""
        msg = user_tmpl.format(text=t.replace("\\n"," "))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":msg}],
            temperature=0.0,  # 使用确定性输出
        )
        content = resp.choices[0].message.content.strip()
        return _coerce_label_from_text(content, num_labels=len(labels))

    out = [None] * len(texts)
    # 使用线程池并行执行标注任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, t): i for i, t in enumerate(texts)}
        for f in concurrent.futures.as_completed(futs):
            out[futs[f]] = f.result()
    return out

def annotate_concepts_llm_parallel(texts: List[str], concepts: List[str], model="gpt-4o-mini", api_key_env="OPENAI_API_KEY", max_workers=4) -> List[Optional[List[int]]]:
    """
    并行标注多个文本的概念，每个文本使用一次 LLM 调用
    
    返回概念向量列表，每个向量是一个 0/1 列表，表示该文本是否包含每个概念。
    失败的情况用 None 表示。
    
    Args:
        texts: 待标注的文本列表
        concepts: 概念列表
        model: LLM 模型名称
        api_key_env: 存储 API 密钥的环境变量名
        max_workers: 并行线程数
    
    Returns:
        概念向量列表，失败项为 None
    """
    client = OpenAI(api_key=os.getenv(api_key_env))

    sys_prompt = "You are a precise annotator. Tag each concept as 0/1."
    concept_list = "\n".join(f"{i}. {c}" for i, c in enumerate(concepts))
    user_tmpl = (
        "For this news text, return a JSON array of 0/1 flags of length {L} in the given order.\n"
        "Concepts:\n{concept_list}\n\nTEXT: {text}\nReturn ONLY the JSON array."
    )

    def _one(t):
        """单个文本的概念标注任务"""
        msg = user_tmpl.format(L=len(concepts), concept_list=concept_list, text=t.replace("\\n"," "))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":msg}],
            temperature=0.0,  # 使用确定性输出
        )
        content = resp.choices[0].message.content.strip()
        try:
            arr = json.loads(content)
        except Exception:
            # 如果直接解析失败，尝试提取 JSON 数组部分
            l = content.find("[")
            r = content.rfind("]")
            arr = json.loads(content[l:r+1])
        # 将数组元素转换为 0/1
        arr = [int(1 if int(x) > 0 else 0) for x in arr]
        if len(arr) != len(concepts): 
            return None
        return arr

    out = [None] * len(texts)
    # 使用线程池并行执行标注任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, t): i for i, t in enumerate(texts)}
        for f in concurrent.futures.as_completed(futs):
            out[futs[f]] = f.result()
    return out

@torch.no_grad()
def annotate_with_classifier(encoder, head, tokenizer, texts: List[str], device) -> List[int]:
    """
    回退方案：当 LLM 不可用时，使用当前模型预测作为伪标签
    
    Args:
        encoder: 编码器
        head: CBM 分类头
        tokenizer: 分词器
        texts: 待标注的文本列表
        device: 计算设备
    
    Returns:
        预测标签列表
    """
    out_labels = []
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        feats = encoder(**enc).last_hidden_state[:, 0]  # [CLS] 嵌入
        logits = head(feats)  # 分类 logits
        out_labels.append(int(logits.softmax(-1).argmax(-1).item()))  # 预测类别
    return out_labels

# ----------------------------
# 主动学习池
# ----------------------------
@dataclass
class Pools:
    """主动学习索引池：已标注和未标注样本的索引"""
    labeled: List[int]      # 已标注样本的索引列表
    unlabeled: List[int]    # 未标注样本的索引列表

def init_pools(n_total: int, seed_size: int, seed: int) -> Pools:
    """
    初始化主动学习池：随机选择初始标注样本
    
    Args:
        n_total: 总样本数
        seed_size: 初始标注样本数量
        seed: 随机种子
    
    Returns:
        包含已标注和未标注索引的 Pools 对象
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total).tolist()  # 随机打乱索引
    return Pools(labeled=idx[:seed_size], unlabeled=idx[seed_size:])

# ----------------------------
# 主函数
# ----------------------------
def main():
    """主函数：执行 BADGE 主动学习流程（使用 LLM 标注器和 CBM）"""
    import argparse
    parser = argparse.ArgumentParser(description="BADGE 主动学习在 AG News 上的应用（LLM 标注器 + CBM）")
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
    parser.add_argument("--alpha_concept", type=float, default=1.0, help="概念损失的权重")
    parser.add_argument("--concept_l1", type=float, default=1e-4, help="概念 L1 正则化系数")
    # 系统参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace 缓存目录")
    # LLM 标注器选项
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="OpenAI 提供商的 LLM 模型名称")
    parser.add_argument("--llm_api_key_env", type=str, default="OPENAI_API_KEY", help="存储 LLM API 密钥的环境变量名")
    parser.add_argument("--llm_workers", type=int, default=8, help="每个文本 LLM 调用的并行线程数")
    args = parser.parse_args()

    print("="*20)
    print("Arguments:")
    print("="*20)
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # 打印 git commit hash 用于实验追踪
    git_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    print(f"Latest git commit hash: {git_commit_hash}")

    set_seed(args.seed)
    device = torch.device(args.device)

    # 获取数据集配置
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"不支持的数据集: {args.dataset}。支持的数据集: {list(DATASET_CONFIGS.keys())}")
    config = DATASET_CONFIGS[args.dataset]
    concepts = config["concepts"]
    num_labels = config["num_labels"]
    n_concepts = len(concepts)
    
    print(f"\n使用数据集: {args.dataset}")
    print(f"标签数量: {num_labels}")
    print(f"概念数量: {n_concepts}")
    print(f"标签: {config['labels']}")

    # 加载数据集并动态 tokenize
    raw = load_dataset(config["dataset_name"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.cache_dir)
    train_ds = HFDatasetWrapper(raw["train"], tokenizer, text_key=config["text_key"], max_length=args.max_length)
    test_ds  = HFDatasetWrapper(raw["test"],  tokenizer, text_key=config["text_key"], max_length=args.max_length)

    # 初始化主动学习的标注/未标注池
    pools = init_pools(len(train_ds), args.seed_size, args.seed)
    print(f"Seed labeled={len(pools.labeled)}, unlabeled={len(pools.unlabeled)}")

    # 加载编码器（冻结）和 CBM 分类头
    enc_cfg = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    encoder = AutoModel.from_pretrained(args.model_name, config=enc_cfg, cache_dir=args.cache_dir)
    for p in encoder.parameters(): 
        p.requires_grad = False  # 冻结编码器参数
    encoder.to(device)

    # CBM 分类头
    head = CBMHead(hidden_size=enc_cfg.hidden_size, num_labels=num_labels, n_concepts=n_concepts, hidden_task=256, concept_l1=args.concept_l1).to(device)

    # 数据加载器
    def make_loader(idxs, shuffle):
        """为数据集的子集创建 DataLoader"""
        subset = torch.utils.data.Subset(train_ds, idxs)
        return DataLoader(subset, batch_size=args.batch_size, shuffle=shuffle, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # 为初始标注池标注概念，以便从第一轮开始应用概念损失
    if len(pools.labeled) > 0:
        init_texts = [raw["train"][i][config["text_key"]] for i in pools.labeled]

        init_concepts_raw = annotate_concepts_llm_parallel(
            init_texts,
            concepts,
            model=args.llm_model,
            api_key_env=args.llm_api_key_env,
            max_workers=args.llm_workers,
        )
        # 对于失败的标注，回退到全零向量
        fail_count = sum(1 for cv in init_concepts_raw if cv is None)
        init_concepts = [cv if cv is not None else [0] * n_concepts for cv in init_concepts_raw]
        print(
            f"LLM annotated concepts for initial {len(pools.labeled)} samples, "
            f"with {fail_count} failures."
        )
        train_ds.set_many_concept_overrides(pools.labeled, init_concepts)

    acc_list = []

    # 主动学习主循环
    for r in range(args.rounds):
        print(f"\n=== Round {r}/{args.rounds} ===")
        # 在当前标注集上训练分类器，然后评估
        train_loader = make_loader(pools.labeled, shuffle=True)
        train_one_round(encoder, head, train_loader, device, lr=args.lr, epochs=args.epochs, alpha_concept=args.alpha_concept)
        acc = evaluate(encoder, head, test_loader, device)
        print(f"Test accuracy: {acc*100:.2f}%")
        acc_list.append(acc*100)

        # BADGE 选择：在未标注池的随机子集上计算梯度嵌入（可选，用于加速）
        probe_count = min(20000, len(pools.unlabeled))
        probe_idx = np.random.choice(pools.unlabeled, size=probe_count, replace=False).tolist() if probe_count > 0 else []
        probe_loader = make_loader(probe_idx, shuffle=False)
        embs, map_idx = compute_gradient_embeddings(encoder, head, probe_loader, device)
        if embs.shape[0] == 0: 
            break
        k = min(args.query_size, embs.shape[0])

        print(f"Selecting {k} via BADGE...")
        print(f"Gradient embedding shape: {embs.shape}")
        t = time.time()
        rel = kmeanspp_select_sklearn(embs, k, np.random.default_rng(args.seed + r))
        print(f"Selection took {time.time()-t:.2f}s.")
        newly_selected = [probe_idx[i] for i in rel]

        # 使用 LLM 标注选中的样本的标签
        texts = [raw["train"][i][config["text_key"]] for i in newly_selected]
        labels = annotate_labels_llm_parallel(texts, model=args.llm_model, api_key_env=args.llm_api_key_env, 
                                              max_workers=args.llm_workers, labels=config["labels"])
        # 检查是否有标注失败的情况（当前实现要求所有标注成功）
        fallback_needed = [i for i, v in enumerate(labels) if v is None]
        if fallback_needed:
            raise RuntimeError("LLM label annotation failed for some samples; no fallback implemented.")
        print(f"LLM annotated labels for {len(newly_selected)} samples, with {sum(1 for x in labels if x is None)} failures (fallbacks applied).")
        labels = [int(x) for x in labels]
        
        # 使用 LLM 标注选中的样本的概念
        concept_vecs = annotate_concepts_llm_parallel(texts, concepts, model=args.llm_model, api_key_env=args.llm_api_key_env, max_workers=args.llm_workers)
        # 对于失败的标注，回退到全零向量
        concept_vecs = [cv if cv is not None else [0]*n_concepts for cv in concept_vecs]
        print(f"LLM annotated concepts for {len(newly_selected)} samples, with {sum(1 for x in concept_vecs if x is None)} failures.")

        # 存储覆盖标签和概念，以便后续训练使用这些标注
        train_ds.set_many_label_overrides(newly_selected, labels)
        train_ds.set_many_concept_overrides(newly_selected, concept_vecs)

        # 将选中的样本从未标注池移到标注池
        pools.labeled.extend(newly_selected)
        unl = set(pools.unlabeled)
        unl.difference_update(newly_selected)
        pools.unlabeled = list(unl)

    print("\n=== Final Results ===")

    # 保存结果
    from datetime import datetime
    result_dir = f"results/{config['result_dir']}/llm_cbm"
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}", "w") as f:
        json.dump(acc_list, f)

    print("Done.")

if __name__ == "__main__":
    main()
