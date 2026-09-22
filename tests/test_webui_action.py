#!/usr/bin/env python3
"""动作识别接进 WebUI 的端到端测试（无头，不需要摄像头/浏览器）。

    # 起服务（默认就是 infer=browser）
    .venv/bin/python webui/server.py --port 8771
    # 再跑本脚本
    .venv/bin/python tests/test_webui_action.py --url ws://127.0.0.1:8771/ws

为什么需要这个文件：动作识别接进 WebUI 之后，真正跑的是**整条链路**

    浏览器(MediaPipe) → DataChannel → parse_landmarks → LandmarkTaskEngine
        → ActionDetector → state → DataChannel → 前端

单元测试（tests/test_action_detect.py）只覆盖 `ActionDetector` 本身，
覆盖不到"接进去"这一段。这里沿用 `tests/rtc_client.py` 那套无头 RTC 客户端
把浏览器那一半演出来，推**合成关键点**上去，断言服务端回传的
`state.action` 里四个动作都对。

⚠️ 这里还钉住了一件**最容易错**的事：左右方向的基准两边**不一样**。

    · WebUI：坐标是「相机原图」（`infer.js` 顶部写明"这里不镜像"），
      所以画面 +x 是玩家的**左边** → 服务端 `mirrored_input=False`
    · 命令行：`TaskController` 先把画面镜像再推理 → 那边是 `True`

所以本测试推一帧"髋中点在画面里往 +x 移"，断言它报的是**「左移」**。
这正是 §5.7 说"合成数据不算数、必须真人验一次"的那一处：
这个测试只能证明"整条链路按约定的基准在跑"，不能证明"约定本身对不对"。
"""
import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from action.synth import make_pose, make_world

#: 推帧间隔（ms）。必须接近真实帧率 —— 引擎只保留最新一帧，
#: 推太快会让标定拿不到足够的样本（见 Calibrator.MIN_SAMPLES）。
FRAME_MS = 33.0


class NullVideoTrack(VideoStreamTrack):
    """一条假视频轨。关键点模式下服务端不看画面，但 SDP 里有条轨更接近真实浏览器。"""

    kind = "video"

    def __init__(self, w=320, h=240, fps=15):
        super().__init__()
        self.w, self.h, self.fps = w, h, fps
        self._frame = np.zeros((h, w, 3), dtype=np.uint8)
        self.sent = 0

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        vf = VideoFrame.from_ndarray(self._frame, format="bgr24")
        vf.pts, vf.time_base = pts, time_base
        self.sent += 1
        await asyncio.sleep(1.0 / self.fps)
        return vf


def as_lms(pts):
    """[[x,y,z]] 保留 4 位小数 —— 与浏览器 infer.js 的 toXYZ 一致。"""
    return [[round(float(x), 4), round(float(y), 4), round(float(z), 4)]
            for (x, y, z) in pts]


async def push(dc, pose, world, ms=FRAME_MS):
    """按真实帧率推一帧关键点。"""
    dc.send(json.dumps({
        "type": "landmarks",
        "frameT": time.time(),
        "face": None,
        "body": as_lms(pose),
        "bodyWorld": as_lms(world) if world is not None else None,
        "leftHand": None,
        "rightHand": None,
    }))
    await asyncio.sleep(ms / 1000.0)


async def play(dc, plan):
    """plan: [(pose, world, ms), ...]"""
    for pose, world, ms in plan:
        n = max(1, int(round(ms / FRAME_MS)))
        for _ in range(n):
            await push(dc, pose, world)


def build_plan():
    """一段"把四个动作各做一遍"的合成关键点。

    顺序：站直(标定) → 站 → 蹲下 → 站起 → 画面+x(玩家左) → 回位
          → 画面−x(玩家右) → 回位 → 跳 → 站定

    世界坐标一起给：这样顺便验证 `bodyWorld` 这条新字段
    （infer.js → parse_landmarks → 引擎 → 膝角）真的通了。
    """
    stand, standW = make_pose(), make_world(180.0)
    squat, squatW = make_pose(knee_deg=110.0), make_world(110.0)
    return [
        (stand, standW, 1400),                       # 标定窗口（服务端 1200ms 收口）
        (stand, standW, 400),
        (squat, squatW, 1200),                       # 蹲下
        (stand, standW, 1200),                       # 站起
        (make_pose(lane=0.6), standW, 500),          # 画面 +x = 玩家的左边
        (stand, standW, 900),                        # 回位
        (make_pose(lane=-0.6), standW, 500),         # 画面 -x = 玩家的右边
        (stand, standW, 900),                        # 回位
        (make_pose(rise=0.30), standW, 400),         # 跳
        (stand, standW, 900),
    ]


def collect_events(states):
    """把各帧 state 里带的 events 汇总去重（保持首次出现的顺序）。"""
    seen, out = set(), []
    for st in states:
        a = (st or {}).get("action") or {}
        for e in (a.get("events") or []):
            key = (e.get("kind"), e.get("dir"), e.get("on"), e.get("wall"))
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
    return out


def label(e):
    k = e.get("kind")
    if k == "SQUAT":
        return "SQUAT:ON" if e.get("on") else "SQUAT:OFF"
    if k == "LANE":
        return "LANE:" + ("L" if e.get("dir") == "left" else "R")
    return k


async def run(args):
    print(f"连接信令 {args.url}")
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(args.url, heartbeat=20) as ws:
            pc = RTCPeerConnection()
            track = NullVideoTrack()
            pc.addTrack(track)
            dc = pc.createDataChannel("state")

            states, hello = [], []
            connected = {"v": False}

            @dc.on("message")
            def _msg(m):
                try:
                    d = json.loads(m)
                except Exception:
                    return
                if d.get("type") == "state":
                    states.append(d.get("state") or {})
                elif d.get("type") == "hello":
                    hello.append(d)

            @pc.on("connectionstatechange")
            def _st():
                if pc.connectionState == "connected":
                    connected["v"] = True

            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            await ws.send_json({"type": "offer", "sdp": pc.localDescription.sdp,
                                "sdpType": pc.localDescription.type})

            answered = False
            deadline = time.time() + 20
            while time.time() < deadline and not answered:
                try:
                    raw = await asyncio.wait_for(ws.receive(), timeout=20)
                except asyncio.TimeoutError:
                    break
                if raw.type != aiohttp.WSMsgType.TEXT:
                    if raw.type in (aiohttp.WSMsgType.CLOSED,
                                    aiohttp.WSMsgType.ERROR):
                        break
                    continue
                data = json.loads(raw.data)
                if data.get("type") == "answer":
                    await pc.setRemoteDescription(RTCSessionDescription(
                        sdp=data["sdp"], type=data["sdpType"]))
                    answered = True
                elif data.get("type") == "candidate":
                    try:
                        await pc.addIceCandidate(data["candidate"])
                    except Exception:
                        pass
            if not answered:
                print("  ✗ 没等到 answer")
                await pc.close()
                return 2

            # 等 DataChannel 打开
            t0 = time.time()
            while dc.readyState != "open" and time.time() - t0 < 10:
                await asyncio.sleep(0.1)
            if dc.readyState != "open":
                print(f"  ✗ DataChannel 没开（{dc.readyState}）")
                await pc.close()
                return 2
            print("  ✓ DataChannel 已打开")
            if hello:
                print(f"  ✓ hello: infer={hello[0].get('infer')}")

            # ── 开始标定，然后按剧本推关键点
            print("  → 发送 actionCalib（1200ms）")
            dc.send(json.dumps({"type": "actionCalib", "ms": 1200}))
            await asyncio.sleep(0.15)

            print("  → 开始推合成关键点（四个动作各一遍）…")
            await play(dc, build_plan())
            await asyncio.sleep(0.6)          # 让最后几帧的 state 回流

            # ── 结果
            last = {}
            for st in reversed(states):
                if st.get("action"):
                    last = st
                    break
            a = last.get("action") or {}
            events = collect_events(states)
            labels = [label(e) for e in events
                      if e.get("kind") in ("SQUAT", "LANE", "JUMP")]

            print("\n================ 结果 ================")
            print(f"连接 / DataChannel : {connected['v']} / {dc.readyState}")
            print(f"收到 state 消息    : {len(states)}")
            print(f"已标定             : {a.get('calibrated')}")
            print(f"block              : {a.get('block')!r}")
            print(f"mirroredInput      : {a.get('mirroredInput')}"
                  f"（WebUI 应为 False）")
            print(f"kneeSource         : {a.get('kneeSource')}"
                  f"（给了 bodyWorld 就该是 3d）")
            print(f"膝角               : {a.get('kneeL')} / {a.get('kneeR')}")
            print(f"动作事件序列       : {labels}")
            print(f"最近一条           : {a.get('events', [{}])[-1].get('text') if a.get('events') else '–'}")
            print("=====================================")

            ok = True

            def check(name, cond, detail=""):
                nonlocal ok
                print(f"  {'✅' if cond else '❌'} {name}{(' — ' + detail) if detail else ''}")
                ok = ok and bool(cond)

            print("\n断言：")
            check("RTC 通路连通", connected["v"] and len(states) > 0)
            check("标定成功", a.get("calibrated") is True,
                  f"block={a.get('block')!r}")
            check("bodyWorld 通到了膝角", a.get("kneeSource") == "3d",
                  f"kneeSource={a.get('kneeSource')!r}")
            check("WebUI 用相机原图基准（mirrored_input=False）",
                  a.get("mirroredInput") is False)
            check("四个动作都命中且方向按 WebUI 基准",
                  labels == ["SQUAT:ON", "SQUAT:OFF", "LANE:L", "LANE:R", "JUMP"],
                  f"实际 {labels}")

            await pc.close()
            print("\n" + ("✅ WebUI 动作识别端到端验证通过"
                          if ok else "❌ 验证未通过，见上面 ❌"))
            return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8771/ws")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
