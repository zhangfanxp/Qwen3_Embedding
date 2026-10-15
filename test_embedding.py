#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""沿用原来的命令行实验：打印词语向量，保存所有向量与相似度。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embedding_engine import MODEL_ID, cosine_pairs, encode_texts

DEFAULT_WORDS = ["胡萝卜", "白菜", "飞机", "苹果", "我爱吃胡萝卜"]


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 Qwen3 Embedding 真实向量")
    parser.add_argument("--words", nargs="+", default=DEFAULT_WORDS, help="待向量化的词语或句子")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--full", action="store_true", help="显示全部向量数值")
    parser.add_argument("--output", default="outputs/embedding_vectors.json", help="JSON 保存路径")
    args = parser.parse_args()

    words = list(dict.fromkeys(args.words))
    vectors, device = encode_texts(words, requested_device=args.device)
    print(f"模型：{MODEL_ID} | 设备：{device} | 维度：{vectors.shape[1]}")
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    for word, vector in zip(words, vectors):
        print(f"\n词语：{word}\n向量：{vector if args.full else vector[:10]}")

    pairs = cosine_pairs(words, vectors)
    print("\n两两余弦相似度：")
    for pair in pairs:
        print(f"{pair['keyword_a']} : {pair['keyword_b']} = {pair['cosine_similarity']:.6f}")

    payload = {
        "model": MODEL_ID,
        "device": device,
        "normalized": True,
        "dimensions": vectors.shape[1],
        "vectors": [{"text": word, "values": v.tolist()} for word, v in zip(words, vectors)],
        "pairwise_cosine_similarity": pairs,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整结果：{path.resolve()}")


if __name__ == "__main__":
    main()
