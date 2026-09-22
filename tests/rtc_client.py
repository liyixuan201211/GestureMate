#!/usr/bin/env python3
"""无头 RTC 客户端 —— 冒充浏览器验证服务端的 WebRTC 通路。

浏览器那侧（getUserMedia）没法在无头环境里跑，但除了「拿摄像头」这一步，
其余全是标准 WebRTC：WebSocket 信令 → SDP 交换 → ICE/DTLS-SRTP →
视频轨上行 → DataChannel 回传状态。这个脚本用 aiortc 把浏览器那一半演出来，
喂测试视频当摄像头，于是可以把服务端整条链路端到端验证一遍。

    # 先起服务（浏览器取流模式）
    .venv/bin/python webui/server.py --source browser --data tests/_e2e_data --port 8770
    # 再跑本脚本
    .venv/bin/python tests/rtc_client.py --url ws://127.0.0.1:8770/ws \
        --video tests/fixtures/hands_gestures.mp4 --seconds 8
"""
import argparse
import asyncio
import json
import sys
import time

import aiohttp
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame


class VideoFileTrack(VideoStreamTrack):
    """把视频文件当作「浏览器摄像头」持续送帧。"""

    kind = "video"

    def __init__(self, path, fps=15):
        super().__init__()
        self.cap = cv2.VideoCapture(path)
        self.fps = fps
        self.sent = 0

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("视频读不出来")
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts, vf.time_base = pts, time_base
        self.sent += 1
        await asyncio.sleep(1.0 / self.fps)
        return vf


async def run(args):
    print(f"连接信令 {args.url}")
    print(f"假装浏览器摄像头：{args.video} @ {args.fps}fps")

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(args.url, heartbeat=20) as ws:
            pc = RTCPeerConnection()
            track = VideoFileTrack(args.video, args.fps)
            pc.addTrack(track)
            dc = pc.createDataChannel("state")

            states, events, hello = [], [], []
            connected_seen = {"v": False}

            @dc.on("open")
            def _open():
                print("  ✓ DataChannel 已打开")

            @dc.on("message")
            def _msg(m):
                try:
                    d = json.loads(m)
                except Exception:
                    return
                if d.get("type") == "state":
                    states.append(d)
                    events.extend(d.get("events") or [])
                elif d.get("type") == "hello":
                    hello.append(d)
                    print(f"  ✓ 收到 hello: {d}")

            @pc.on("connectionstatechange")
            def _st():
                print(f"  · 连接状态 -> {pc.connectionState}")
                if pc.connectionState == "connected":
                    connected_seen["v"] = True

            @pc.on("iceconnectionstatechange")
            def _ice():
                print(f"  · ICE -> {pc.iceConnectionState}")

            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            await ws.send_json({"type": "offer",
                                "sdp": pc.localDescription.sdp,
                                "sdpType": pc.localDescription.type})
            print("  → 已发出 offer")

            answered = False
            deadline = time.time() + 20
            while time.time() < deadline and not answered:
                try:
                    raw = await asyncio.wait_for(ws.receive(), timeout=20)
                except asyncio.TimeoutError:
                    break
                if raw.type != aiohttp.WSMsgType.TEXT:
                    if raw.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    continue
                data = json.loads(raw.data)
                if data.get("type") == "answer":
                    await pc.setRemoteDescription(RTCSessionDescription(
                        sdp=data["sdp"], type=data["sdpType"]))
                    answered = True
                    print("  ✓ 已收到并设置 answer")
                elif data.get("type") == "candidate":
                    try:
                        await pc.addIceCandidate(data["candidate"])
                    except Exception:
                        pass
                elif data.get("type") == "connectionstate":
                    print(f"  · 服务端看到的状态: {data['state']}")

            if not answered:
                print("  ✗ 没等到 answer")
                await pc.close()
                return 2

            # 让媒体跑一会儿
            print(f"\n  收集中（{args.seconds}s）…")
            t0 = time.time()
            while time.time() - t0 < args.seconds:
                await asyncio.sleep(0.5)

            async def growing(seconds, label, settle=0.0):
                """看服务端的 frames 计数在给定时间内是否还在涨。

                settle 用来等「重启管线」完成 —— 重启会新建一条管线，
                frames 计数从 0 重新开始，所以基准必须在重启之后才采样，
                否则会测出一个负增长（那是计数器归零，不是没在处理）。
                """
                if settle:
                    await asyncio.sleep(settle)
                base = (states[-1]["state"].get("frames") or 0) if states else 0
                t = time.time()
                while time.time() - t < seconds:
                    await asyncio.sleep(0.3)
                now = (states[-1]["state"].get("frames") or 0) if states else 0
                grew = now - base
                print(f"  · {label}: frames {base} -> {now} (+{grew})")
                return grew

            toggle_ok = True
            if args.toggle_complexity:
                print("\n  === 回归测试：切换复杂度 ===")
                print("  （曾经的 bug：切换会抛 TypeError 并让管线彻底停止）")
                for v in (2, 0, 1):
                    dc.send(json.dumps({"type": "setComplexity", "value": v}))
                    print(f"  → 已发送 setComplexity={v}")
                    if await growing(4, f"complexity={v} 后", settle=2.0) <= 0:
                        toggle_ok = False
                        print(f"  ✗ complexity={v} 之后管线不再处理帧")

            last = states[-1]["state"] if states else {}
            # 注意：必须用「关闭前」记录的状态来判断，pc.close() 之后
            # connectionState 会变成 closed。
            conn, ice, dcstate = pc.connectionState, pc.iceConnectionState, dc.readyState
            print("\n================ 结果 ================")
            print(f"连接状态        : {conn}（曾经 connected: {connected_seen['v']}）")
            print(f"ICE 状态        : {ice}")
            print(f"DataChannel     : {dcstate}")
            print(f"已发送视频帧    : {track.sent}")
            print(f"收到 state 消息 : {len(states)}")
            # 状态更新率 = 叠加层每秒能动几次。以前固定 10Hz 推送，视频 30fps，
            # 所以骨架明显“跟不上手”；改成事件驱动后应该接近推理帧率。
            if len(states) > 1:
                span = states[-1]["t"] - states[0]["t"]
                if span > 0:
                    print(f"状态更新率      : {(len(states)-1)/span:.1f} /s"
                          f"（以前固定推送封顶 ~10/s）")
            lats = [s["state"].get("latencyMs") for s in states
                    if s["state"].get("latencyMs") is not None]
            if lats:
                print(f"服务端结果延迟  : 中位 {sorted(lats)[len(lats)//2]:.1f} ms "
                      f"/ 最小 {min(lats):.1f} / 最大 {max(lats):.1f}"
                      f"（算完到发出；以前固定轮询平均白等 ~19ms）")
            print(f"服务端已处理帧  : {last.get('frames')}")
            print(f"服务端 FPS      : {last.get('fps')}")
            print(f"服务端推理      : {last.get('inferMs')} ms / 单帧 {last.get('loopMs')} ms")
            print(f"引擎            : {last.get('engine')}  parts={last.get('parts')}")
            h = last.get("hands") or {}
            print(f"手部            : 左 {'✓' if h.get('leftHand') else '✗'} "
                  f"右 {'✓' if h.get('rightHand') else '✗'}")
            print(f"任务            : {list((last.get('tasks') or {}).keys())}")
            # 面部是降频下发的，所以取所有 state 里的最大值来判断「有没有下发过」
            face_len = max((len(s["state"].get("face") or []) for s in states), default=0)
            body_len = max((len(s["state"].get("body") or []) for s in states), default=0)
            print(f"面部 / 身体     : face={face_len} 点, body={body_len} 点")
            ft = last.get("features") or {}
            if ft:
                e = ft.get("eyes") or {}
                m = ft.get("mouth") or {}
                hd = ft.get("head") or {}
                bd = ft.get("body") or {}
                print(f"  眼睛          : 右EAR {e.get('right')} 左EAR {e.get('left')} "
                      f"闭眼(右/左) {e.get('rightClosed')}/{e.get('leftClosed')} "
                      f"眨眼 {ft.get('blinkCount')}")
                print(f"  嘴巴          : {m.get('state')} (MAR {m.get('open')}, "
                      f"宽比 {m.get('widthRatio')})")
                print(f"  头部          : {hd.get('direction')} "
                      f"yaw={hd.get('yaw')} pitch={hd.get('pitch')} roll={hd.get('rollDeg')}°")
                print(f"  身体          : {bd.get('landmarkCount')} 点, "
                      f"举手={bd.get('handsUp')}, 肩线={bd.get('shoulderTiltDeg')}° {bd.get('lean')}")
            print(f"触发事件        : {[e['text'] for e in events][:8]}")
            print(f"管线报错        : {last.get('error')}")
            print("=====================================")

            await pc.close()

            ok = (connected_seen["v"]
                  and (last.get("frames") or 0) > 0
                  and len(states) > 0
                  and toggle_ok)
            if args.toggle_complexity:
                print(f"复杂度切换    : {'✅ 每次切换后管线都继续处理' if toggle_ok else '❌ 有切换把管线搞死了'}")
            print("\n" + ("✅ RTC 通路验证通过："
                          "信令/ICE/DTLS/视频上行/状态回传全部工作"
                          if ok else "❌ 验证未通过，见上面数字"))
            return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8770/ws")
    ap.add_argument("--video", default="tests/fixtures/hands_gestures.mp4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--toggle-complexity", action="store_true",
                    help="中途切换复杂度，回归测试「重启管线」这条路")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
