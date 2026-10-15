#!/usr/bin/env bash
# 在能访问 Python 包索引的 Mac/Linux 上生成真实 uv.lock；生成后提交 Git。
set -euo pipefail
cd "$(dirname "$0")"
uv python install 3.11
uv lock
uv sync --locked
printf '\n✅ 已生成 uv.lock 并安装环境。\n'
