# Qwen3 Embedding 向量实验室（Gradio / macOS / uv）

在 Apple Silicon Mac 上运行本地 `Qwen/Qwen3-Embedding-0.6B`，输入 **1–5 个关键词或短句**，展示每条文本的真实 **1024 维向量**，并计算任意两条文本的 **余弦相似度**（最多 10 组）。完全本地推理，不需要调用付费 API。

- **模型：** [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)，Apache-2.0 许可
- **设备：** 优先自动检测 Apple Metal / MPS（Linux CUDA 也可），不可用时使用 CPU
- **模型下载位置：** Hugging Face 默认缓存 `~/.cache/huggingface/hub/`
- **WebUI：** 默认 `http://127.0.0.1:7862`
- **Python：** 3.11（也支持 3.12）

## 目录

```text
qwen3-embedding-gradio/
├── app.py                # Gradio 页面、两步按钮逻辑、JSON 导出
├── embedding_engine.py   # 模型加载、MPS/CPU 选择、向量化、余弦计算
├── test_embedding.py     # 命令行实验（沿用之前的脚本用途）
├── requirements.txt      # 给 pip / uv pip 的直接依赖
├── pyproject.toml        # uv 项目定义及依赖版本范围
├── uv.lock               # 在可连接 Python 包索引的环境执行 uv lock 后生成并提交
├── .python-version       # Python 3.11
├── .env.example          # 环境变量示例（默认不需要改）
├── .gitignore
├── tests/
│   └── test_math.py      # 不需要模型的数学测试
└── outputs/              # 生成的 JSON，不提交 Git
```

## 1. Mac 安装 / 创建虚拟环境

先安装 [uv](https://docs.astral.sh/uv/)（已安装可跳过）。打开终端：

```bash
# 推荐先把项目解压到 ~/Documents/ 下
cd ~/Documents/qwen3-embedding-gradio

# 下载安装适合项目的 Python 3.11（如本地已有则复用）
uv python install 3.11

# 如果项目已有提交的 uv.lock：
uv sync --locked

# 如果本次下载的版本暂未附带 uv.lock：
# 先解析依赖并生成真正的锁文件，然后按锁文件安装：
uv lock
uv sync --locked

# 可选：查看当前 Python / PyTorch
uv run python -c 'import sys, torch; print(sys.version); print(torch.__version__); print("MPS:", torch.backends.mps.is_available())'
```

> `uv.lock` 应由 `uv lock` 解析包索引生成，不能手动编造。锁文件生成后应提交 GitHub。之后安装推荐使用 `uv sync --locked`。不要把 `.venv/` 提交上去。

> 如果不使用 uv 项目模式，也可以运行：`uv venv --python 3.11 && source .venv/bin/activate && uv pip install -r requirements.txt`。不过此方式不会自动使用 `uv.lock`。

## 2. 从 Hugging Face 下载模型（默认缓存）

```bash
cd ~/Documents/qwen3-embedding-gradio
uv run hf download Qwen/Qwen3-Embedding-0.6B
```

无需 `--local-dir`，下载后会存在：

```text
~/.cache/huggingface/hub/
└── models--Qwen--Qwen3-Embedding-0.6B/
    ├── blobs/
    ├── refs/
    └── snapshots/
```

**可选：** 网络不能直连 Hugging Face 时，若你有可访问且可信任的镜像端点，可在下载时设置 `HF_ENDPOINT`：

```bash
HF_ENDPOINT=https://hf-mirror.com uv run hf download Qwen/Qwen3-Embedding-0.6B
```

这只是可选镜像示例，镜像的实际可用性需要自行确认。下载完成后程序使用 `local_files_only=True`，正常情况下**不会再次联网下载模型**。

检查模型是否已经缓存：

```bash
uv run hf cache ls | grep Qwen3-Embedding-0.6B
```

## 3. 启动 Gradio 网页

```bash
cd ~/Documents/qwen3-embedding-gradio
uv run python app.py
```

浏览器访问：**http://127.0.0.1:7862**。

**使用顺序：**

1. 在 5 个输入框中填写 1～5 个关键词，空白的会自动忽略。例如：`胡萝卜`、`白菜`、`飞机`、`苹果`。
2. 点击 **「① 向量化」**：本地模型为每个输入生成真实向量、完成 L2 归一化，在页面显示 **前12维、向量长度**。展开 JSON 可查看全部1024维，也可下载完整 JSON。
3. 点击 **「② 计算相似度」**：依次列出所有两两组合的余弦相似度。两个词有 1 组、三个词有 3 组、五个词有 10 组。
4. 修改任何输入后，**必须重新点击向量化**；程序会检查输入是否与保存的向量一致，防止使用旧结果。

模型是**第一次点击向量化时按需加载**；之后复用已经加载的模型。首次运行可能较慢，取决于机器性能和模型是否已缓存。

### 查看完整输出

每次运行生成带时间戳和随机后缀的文件，例如：

```text
outputs/vectors_20261015_093000_2a3b4c5d.json
outputs/similarities_20261015_093005_6e7f8a9b.json
```

向量文件包含输入词、模型 ID、运行设备、每条词语的 1024 个实际浮点数；相似度文件包含所有比较对和分数。项目只在本地保存结果，不会上传到 Hugging Face 或外部 API。

### 数学原理

两个词对应的向量分别为：

\[
X=[x_1,x_2,\ldots,x_n],\quad Y=[y_1,y_2,\ldots,y_n]
\]

余弦相似度：

\[
\operatorname{cosim}(X,Y)=\frac{\sum_{i=1}^{n}x_i y_i}{\sqrt{\sum_{i=1}^{n}x_i^2}\sqrt{\sum_{i=1}^{n}y_i^2}}
\]

程序先做 L2 归一化，所以长度均约等于 1，**点积就近似等于余弦相似度**。一个分数反映的是模型向量之间的方向关系，**不是**“两个词意思相同的概率”，更无法独立判断喜欢/讨厌等否定关系。

## 4. 继续运行命令行测试脚本

```bash
# 默认展示前10维，并打印所有两两相似度
uv run python test_embedding.py

# 自定义词汇
uv run python test_embedding.py --words 胡萝卜 白菜 苹果 飞机

# 打印完整1024维
uv run python test_embedding.py --full

# 不使用 GPU
uv run python test_embedding.py --device cpu
```

所有数据也会写入 `outputs/embedding_vectors.json`。

## 5. 端口、设备和公网访问

程序默认绑定 `127.0.0.1`，只允许本机访问。需要局域网访问（**请先评估访问权限**）：

```bash
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7862 uv run python app.py
```

使用 CPU 运行：

```bash
EMBEDDING_DEVICE=cpu uv run python app.py
```

如模型已经下载到非默认位置：

```bash
QWEN_MODEL_PATH=/absolute/path/to/your/model uv run python app.py
```

不建议将无认证的 Gradio 测试页面直接暴露在公网；如要远程访问，建议通过 Cloudflare Tunnel 配合访问控制。

## 6. 核心源码说明

| 文件 | 作用 |
|---|---|
| `embedding_engine.py` | `choose_device()` 检测 MPS/CUDA/CPU；`encode_texts()` 一次计算多条真实向量并归一化；`cosine_pairs()` 通过点积和 L2 范数得到两两相似度 |
| `app.py` | `get_words()` 处理1～5个输入；`vectorize()` 返回真实向量并导出 JSON；`similarity()` 检查输入没有被修改，再计算所有两两组合；`build_ui()` 创建 Gradio 页面 |
| `test_embedding.py` | 复用同一个模型引擎运行终端对比实验 |
| `pyproject.toml` | 推荐依赖配置，uv 依据它解析跨平台锁文件 |
| `requirements.txt` | 兼容 pip 的直接依赖清单（不是锁文件） |

## 7. 测试

```bash
uv run python -m unittest discover -s tests -v
```

上述单元测试检查：同方向向量相似度为1，互相垂直为0；零向量会被拒绝。测试无需真正下载 Hugging Face 模型。

## 8. 常见问题

**Q：出现“无法从本地缓存加载模型”？**

A：先确认模型是否完整下载：`uv run hf download Qwen/Qwen3-Embedding-0.6B`。检查当前用户的缓存目录；如果是从其他位置下载，可使用 `QWEN_MODEL_PATH` 指向实际的 snapshot 目录。

**Q：报 `KeyError: 'qwen3'`？**

A：transformers 版本偏旧。运行 `uv lock --upgrade-package transformers`、`uv sync`，确保 transformers >= 4.51.0。

**Q：MPS 报不支持的运算？**

A：程序在常见 MPS 推理错误时会尝试把模型移动到 CPU 重试，也可用 `EMBEDDING_DEVICE=cpu` 强制 CPU。

**Q：1个关键词为什么不能计算相似度？**

A：余弦相似度需要两个向量；至少输入两个关键词。

**Q：为什么模型重复生成后小数位可能稍有差异？**

A：GPU/CPU、浮点精度、模型和依赖版本可能导致轻微浮点误差，这不代表向量概念发生变化。

**Q：会保存到 Qdrant 吗？**

A：不会。本项目是 Embedding / 余弦相似度教学实验，不依赖 Mem0 或 Qdrant；JSON 供你观察和下载。将来如需长期检索，再加入 Qdrant。

## 9. GitHub 上传

```bash
git init
git add .
git commit -m "Add Qwen3 Embedding Gradio playground"
```

建议**把 uv.lock 提交到 GitHub**，但**不要提交** `.venv`、模型权重、`outputs/` 中的用户输出。

## 参考资料

- [官方 Qwen3 Embedding 模型页](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Sentence Transformers 文档](https://www.sbert.net/)
- [Gradio 文档](https://www.gradio.app/docs)
- [uv 项目与锁文件说明](https://docs.astral.sh/uv/concepts/projects/lockfiles/)
