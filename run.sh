#!/usr/bin/env bash
# GestureMate 一键启动
#
#   ./run.sh                      # 摄像头 + 默认 config.json（./data）
#   ./run.sh --data="./example/data_example"   # 跑计算器 demo
#   ./run.sh --video=some.mp4     # 用视频文件当输入（不需要摄像头）
#   ./run.sh --cvshow             # 开窗口看骨架
#   ./run.sh --complexity=1 --fps=15
#
# 首次运行会自动建 venv 并装依赖（uv 已在 /Users/imac/.local/bin）。
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

if [ ! -x "$PY" ]; then
  echo "[run.sh] 没找到 .venv，正在创建 Python 3.12 环境并安装依赖 ..."
  uv venv --python 3.12 .venv
  uv pip install --python "$PY" -i "$PIP_INDEX" \
    -r requirements.txt "PyAutoGUI==0.9.54" "requests==2.32.4"
fi

exec "$PY" Main.py "$@"
