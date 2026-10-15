#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型调用和相似度数学逻辑；Gradio 和命令行共用。"""
from __future__ import annotations

import os
import threading
from typing import Sequence

import numpy as np
import torch

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
_MODEL = None
_DEVICE = None
_NOTE = ""
_LOCK = threading.RLock()


def choose_device(requested: str = "auto") -> str:
    """auto: Mac 优先 MPS，也兼容 CUDA/Linux CPU。"""
    requested = requested.lower().strip()
    if requested not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError("EMBEDDING_DEVICE 只能为 auto、mps、cuda 或 cpu")
    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return "mps"
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(device: str):
    from sentence_transformers import SentenceTransformer

    model_source = os.getenv("QWEN_MODEL_PATH", MODEL_ID).strip() or MODEL_ID
    try:
        return SentenceTransformer(
            model_source,
            device=device,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"无法从本地缓存加载模型：{model_source}。\n"
            "请先在当前环境执行：hf download Qwen/Qwen3-Embedding-0.6B\n"
            "模型默认位于 ~/.cache/huggingface/hub/；"
            "并确认 sentence-transformers、transformers、torch 已安装。\n"
            f"原始错误：{exc}"
        ) from exc


def encode_texts(texts: Sequence[str], requested_device: str | None = None) -> tuple[np.ndarray, str]:
    """返回 Float32 单位向量矩阵 (N,1024) 与本次实际运算设备。"""
    global _MODEL, _DEVICE, _NOTE
    values = [str(text).strip() for text in texts]
    if not values or any(not text for text in values):
        raise ValueError("输入不能是空字符串。")

    device = choose_device(requested_device or os.getenv("EMBEDDING_DEVICE", "auto"))
    with _LOCK:
        if _MODEL is None:
            _MODEL = _load_model(device)
            _DEVICE = device
        elif _DEVICE != device:
            _MODEL = _MODEL.to(device)
            _DEVICE = device
        try:
            embeddings = _MODEL.encode(
                values,
                batch_size=min(len(values), 4),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except (RuntimeError, NotImplementedError) as exc:
            if _DEVICE == "cpu":
                raise
            _NOTE = f"设备 {_DEVICE.upper()} 推理失败，已回退 CPU：{str(exc)[:160]}"
            _MODEL = _MODEL.to("cpu")
            _DEVICE = "cpu"
            embeddings = _MODEL.encode(
                values,
                batch_size=min(len(values), 4),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(values):
            raise RuntimeError(f"模型输出形状异常：{matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError("向量含 NaN 或无穷大，无法计算相似度")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise RuntimeError("出现长度为0的向量，无法计算余弦相似度")
        matrix = (matrix / norms).astype(np.float32, copy=False)
        return matrix, _DEVICE


def cosine_pairs(texts: Sequence[str], vectors: np.ndarray) -> list[dict]:
    """列出 n*(n-1)/2 个两两余弦相似度，按输入顺序排列。"""
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != len(texts):
        raise ValueError("词语数量与向量数量不一致")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0):
        raise ValueError("零向量不能计算余弦相似度")
    normalized = array / norms[:, None]
    scores = normalized @ normalized.T
    return [
        {
            "keyword_a": texts[i],
            "keyword_b": texts[j],
            "cosine_similarity": float(np.clip(scores[i, j], -1.0, 1.0)),
        }
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
    ]


def model_note() -> str:
    return _NOTE
