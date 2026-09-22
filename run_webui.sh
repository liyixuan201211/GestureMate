#!/usr/bin/env bash
# GestureMate WebUI（WebRTC）一键启动
#
#   ./run_webui.sh                                   # 浏览器取流（推荐，本机摄像头）
#   ./run_webui.sh --source file --video tests/fixtures/hands_gestures.mp4
#   ./run_webui.sh --data ./example/socket_example    # 顺便跑一份任务配置
#   ./run_webui.sh --port 8770 --complexity 1
#
# 打开 http://127.0.0.1:8770 ，点「开始」并在弹窗里允许摄像头。
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
PORT="${PORT:-8770}"

if [ ! -x "$PY" ]; then
  echo "[run_webui.sh] 没有 .venv，先建环境 ..."
  uv venv --python 3.12 .venv
  uv pip install --python "$PY" -i "$PIP_INDEX" \
    -r requirements.txt "PyAutoGUI==0.9.54" "requests==2.32.4"
fi

# WebUI 额外依赖：aiortc(WebRTC) + aiohttp(静态页/信令)
if ! "$PY" -c "import aiortc, aiohttp" >/dev/null 2>&1; then
  echo "[run_webui.sh] 安装 WebUI 依赖：aiortc aiohttp ..."
  uv pip install --python "$PY" -i "$PIP_INDEX" aiortc aiohttp
fi

exec "$PY" webui/server.py --port "$PORT" "$@"
