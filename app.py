#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3 Embedding Gradio WebUI。运行：uv run python app.py"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import gradio as gr
import numpy as np

from embedding_engine import MODEL_ID, cosine_pairs, encode_texts, model_note

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_words(*entries: str) -> list[str]:
    words = [str(e or "").strip() for e in entries]
    words = [word for word in words if word]
    if not words:
        raise gr.Error("请至少输入一个关键词。")
    if len(words) > 5:
        raise gr.Error("最多只能输入5个关键词。")
    if any(len(word) > 500 for word in words):
        raise gr.Error("每个关键词或短句请控制在500字以内。")
    return words


def save_json(prefix: str, payload: dict) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = OUTPUT_DIR / f"{prefix}_{stamp}_{uuid4().hex[:8]}.json"
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(target)


def vectorize(*entries: str):
    words = get_words(*entries)
    try:
        vectors, device = encode_texts(words)
    except Exception as exc:
        raise gr.Error(f"向量化失败：{exc}") from exc

    dimension = vectors.shape[1]
    # gr.State 使用基础 Python 数据类型，避免 numpy 数组的序列化问题。
    state = {"words": words, "vectors": vectors.tolist(), "device": device}
    vector_items = [
        {"keyword": word, "dimensions": dimension, "values": vector.tolist()}
        for word, vector in zip(words, vectors)
    ]
    payload = {"model": MODEL_ID, "device": device, "normalized": True, "vectors": vector_items}
    file_path = save_json("vectors", payload)
    previews = [
        [word, dimension, f"{np.linalg.norm(v):.6f}", ", ".join(f"{n:.6f}" for n in v[:12])]
        for word, v in zip(words, vectors)
    ]
    note = f"；{model_note()}" if model_note() else ""
    status = (
        f"✅ 已完成 **{len(words)}** 个关键词的向量化；每条 **{dimension} 维**；"
        f"设备：**{device.upper()}**；已进行 L2 归一化{note}。"
        "\n\n现在可以点击 **「计算相似度」**，查看所有两两组合。"
    )
    return state, status, previews, payload, file_path, "", [], None


def similarity(state: dict | None, *entries: str):
    if not state:
        raise gr.Error("请先点击「向量化」按钮。")
    words = get_words(*entries)
    if words != state.get("words"):
        raise gr.Error("输入内容已经变化，请先重新点击「向量化」，避免使用旧向量计算。")
    if len(words) < 2:
        return "至少需要 **2个关键词** 才能计算两两相似度。", [], None

    pairs = cosine_pairs(words, np.asarray(state["vectors"], dtype=np.float32))
    rows = [[p["keyword_a"], p["keyword_b"], round(p["cosine_similarity"], 6)] for p in pairs]
    best = max(pairs, key=lambda p: p["cosine_similarity"])
    report = (
        f"✅ 计算完成：**{len(words)}** 个词，共 **{len(pairs)}** 组两两组合。"
        f"\n\n其中最高相似度：**{best['keyword_a']} ↔ {best['keyword_b']} = "
        f"{best['cosine_similarity']:.6f}**。"
        "\n\n相似度越接近1表示向量方向越一致；它并不是“两个词意思相同的概率”。"
    )
    payload = {
        "model": MODEL_ID,
        "normalized": True,
        "keywords": words,
        "dimensions": len(state["vectors"][0]),
        "pairwise_cosine_similarity": pairs,
    }
    file_path = save_json("similarities", payload)
    return report, rows, file_path


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Qwen3 Embedding 向量实验室") as demo:
        gr.Markdown(
            "# Qwen3-Embedding 向量实验室\n"
            "使用本机的 **Qwen3-Embedding-0.6B**，查看词语的真实1024维向量，"
            "再计算任意两条向量的**余弦相似度**。"
        )
        gr.Markdown("### ① 输入1～5个关键词（留空的输入框会跳过）")
        with gr.Row():
            with gr.Column():
                word1 = gr.Textbox(label="关键词1", value="胡萝卜", lines=1)
                word2 = gr.Textbox(label="关键词2", value="白菜", lines=1)
                word3 = gr.Textbox(label="关键词3", value="飞机", lines=1)
            with gr.Column():
                word4 = gr.Textbox(label="关键词4", value="苹果", lines=1)
                word5 = gr.Textbox(label="关键词5", placeholder="可留空", lines=1)
        with gr.Row():
            vector_btn = gr.Button("① 向量化", variant="primary")
            similarity_btn = gr.Button("② 计算相似度", variant="secondary")

        session_state = gr.State(value=None)
        gr.Markdown("### ② 向量化结果")
        vector_status = gr.Markdown("点击「向量化」后，将从本地缓存加载模型并计算真实向量。")
        vector_table = gr.Dataframe(
            headers=["关键词", "维度", "L2长度", "前12维数值（预览）"],
            datatype=["str", "number", "str", "str"],
            interactive=False,
            wrap=True,
            label="向量预览（完整数值在下方或 JSON 文件里）",
        )
        with gr.Accordion("展开查看全部1024维数值（JSON）", open=False):
            vector_json = gr.JSON(label="完整向量数据")
        vector_file = gr.File(label="下载完整向量 JSON", interactive=False)

        gr.Markdown("### ③ 两两余弦相似度")
        sim_status = gr.Markdown("请先完成向量化，再点击「计算相似度」。")
        sim_table = gr.Dataframe(
            headers=["关键词 A", "关键词 B", "余弦相似度 (-1～1)"],
            datatype=["str", "str", "number"],
            interactive=False,
            wrap=True,
            label="所有两两组合（最多10组）",
        )
        sim_file = gr.File(label="下载两两相似度 JSON", interactive=False)
        gr.Markdown(
            "**计算原理：** `余弦相似度 = 点积 ÷ (向量A长度 × 向量B长度)`。"
            "本项目已做 L2 归一化，点积与余弦相似度等价（允许浮点误差）。"
            "\n\n**注意：** 语义相关不等于事实相同，例如“喜欢”和“不喜欢”也可能具有较高相似度。"
        )
        inputs = [word1, word2, word3, word4, word5]
        vector_btn.click(
            fn=vectorize,
            inputs=inputs,
            outputs=[session_state, vector_status, vector_table, vector_json,
                     vector_file, sim_status, sim_table, sim_file],
            concurrency_limit=1,
        )
        similarity_btn.click(
            fn=similarity,
            inputs=[session_state, *inputs],
            outputs=[sim_status, sim_table, sim_file],
            concurrency_limit=1,
        )
    return demo


demo = build_ui()

if __name__ == "__main__":
    host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7862"))
    print(f"\n打开浏览器：http://{host}:{port}")
    demo.launch(server_name=host, server_port=port, share=False, show_error=True)
