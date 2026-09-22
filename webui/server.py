"""GestureMate WebUI —— WebRTC 版服务端。

两种推理模式（`--infer`）
-----------------------
**browser（默认）**：推理在浏览器里跑（MediaPipe Tasks Vision，WebGL/WASM）。

    浏览器  --getUserMedia-->  <video>  ──► MediaPipe Tasks ──► 画骨架（同一帧！）
                                          └── 关键点 --DataChannel--> 服务端
    服务端  --DataChannel-->  语义特征 + 任务事件  -->  浏览器

  * 叠加层画在**它自己刚算过的那一帧**上，对齐是构造上成立的，不靠外推。
  * 视频**完全不上行**：省掉编码、传输、排队、推理，服务端几乎不干活。
  * Python 侧仍负责它不可替代的部分：TaskController 任务系统 + face_body 语义。

**server**：推理在服务端跑（旧的 mps 管线，也是命令行版那条路）。

    浏览器  --RTP 视频上行-->  服务端 mps  -->  关键点 --DataChannel-->  浏览器

  用于对比、以及给服务进程授了摄像头权限后直接用 `--source opencv` 的场景。

媒体/数据拓扑
-------------
* 信令走 WebSocket（WebRTC 必需），数据走 DataChannel，同一条 PeerConnection。
* 纯本机/局域网场景，host candidate 就够，默认不配 STUN（离线也能用）。

为什么不让服务端直接开摄像头？
--------------------------------
实测 dsh 所在进程树**没有 macOS 摄像头权限**：OpenCV 直接报
`not authorized to capture video (status 0)`，ffmpeg/AVFoundation 则会在
协商完像素格式后**卡死并被 SIGKILL**（TCC 拦截）。浏览器是正规 GUI App，
点一下「允许」就能用，所以取流必须放浏览器。
如果你后来给终端/服务进程授了摄像头权限，用 `--source opencv` 或
`--source ffmpeg` 就能切回服务端取流。

启动
----
    ./run_webui.sh                 # 默认浏览器推理
    ./run_webui.sh --infer server --source file \
        --video tests/fixtures/hands_gestures.mp4
    ./run_webui.sh --port 8770 --data ./example/socket_example
"""
import argparse
import asyncio
import json
import logging
import mimetypes
import os
import signal
import socket
import sys
import time

# 浏览器端推理要用到这几类文件，MIME 必须给对：
#   · .mjs 是 ES module，浏览器**严格**校验类型，给错就拒绝 import
#   · .wasm 用 application/wasm 才能走流式编译
#   · .task 是模型包，按二进制流给即可
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".task")

from aiohttp import WSMsgType, web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pipeline import (BrowserPushSource, DropWebuiEvents, EventMarker,
                      GesturePipeline, LandmarkTaskEngine, build_source,
                      parse_landmarks)  # noqa: E402

STATIC_DIR = os.path.join(HERE, "static")


def lan_ip():
    """拿一个非回环的本机 IP，方便用手机访问。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


class GestureWebApp:
    def __init__(self, args):
        self.args = args
        self.complexity = args.complexity
        self.dataDir = args.data
        # 推理在哪跑：browser=浏览器内（默认，叠加层与画面同帧）/ server=mps 管线
        self.inferMode = getattr(args, "infer", "browser")
        # 只有服务端推理模式才需要帧源；浏览器模式视频根本不上行
        self.source = None
        if self.inferMode == "server":
            self.source = build_source(args.source, video=args.video,
                                       camera_index=args.camera_index)
        self.pipeline = None
        self.pcs = set()
        self.channels = set()
        self.producer = None          # 服务端推理时：只有这一个 peer 的帧会喂给管线
        self.tasks = set()
        self.startedAt = time.time()
        self.shuttingDown = False
        # 468 个脸点体积不小：默认 5Hz 下发（状态本身是 10Hz），
        # 前端可以随时用 setFace 关掉或调频。
        self.faceEnabled = True
        self.faceRate = 5.0
        self.configError = None       # 配置加载失败时给界面显示的原因
        self.watchdogFails = 0
        # 事件驱动推送用：管线有新结果时由回调 set()，推送循环立刻醒
        self.stateEvent = asyncio.Event()
        self.serverView = False       # 「服务端视角」是否开启（决定要不要在服务端画骨架）

    @property
    def landmarkMode(self):
        return self.inferMode == "browser"

    # ------------------------------------------------------------ 管线
    def start_pipeline(self):
        """构造并启动引擎；失败时**不动**正在跑的旧引擎。

        关键顺序：先构造成功，再换掉旧的。
        原来是把旧管线 stop 了才去构造，一旦新配置有问题
        （比如 match 的 poseFile 路径找不到），旧管线已经死了、新的又起不来，
        界面就永远卡在「画面在动、数字不动」—— 这就是「卡死」的成因。

        两种模式共用这一段：browser=关键点任务引擎（无 mps），
        server=原来的 mps 管线。
        """
        try:
            if self.landmarkMode:
                new = LandmarkTaskEngine(data_dir=self.dataDir,
                                         verbose=self.args.verbose)
            else:
                new = GesturePipeline(
                    self.source,
                    data_dir=self.dataDir,
                    complexity=self.complexity,
                    # 默认**不在服务端画骨架**：浏览器自己会画叠加层，服务端那份纯属
                    # 重复劳动（实测 3.21ms/帧，约占 9%）。只有开「服务端视角」时才画。
                    draw=self.serverView,
                    verbose=self.args.verbose,
                )
        except Exception as e:
            self.configError = f"{type(e).__name__}: {e}"
            logging.error(f"[webui] 配置加载失败，保留原有引擎：{self.configError}")
            return False

        self.configError = getattr(new, "configError", None)
        old = self.pipeline
        self.pipeline = new
        # 事件驱动推送：引擎线程每产出一帧结果就 set 一次（跨线程，必须走
        # call_soon_threadsafe）。没有它推送循环就只能傻等，白送 ~19ms 延迟。
        try:
            loop = asyncio.get_running_loop()
            new.onUpdate = lambda: loop.call_soon_threadsafe(self.stateEvent.set)
        except RuntimeError:
            new.onUpdate = None          # 不在事件循环里（理论上不会）就别接
        new.start()
        if old is not None:
            old.stop()
            try:
                old.join(timeout=5)
            except Exception as e:
                # threading.Thread 内部也有 _stop()，被 Event 覆盖会炸（已改名）
                logging.warning(f"[webui] 旧引擎停止时出错（已忽略）: {e}")
        logging.info(f"[webui] 引擎已启动 infer={self.inferMode} "
                     f"complexity={self.complexity} "
                     f"config={self.dataDir or '(无配置，只做识别)'}")
        return True

    def restart_pipeline(self, complexity=None, data_dir=None):
        oldC, oldD = self.complexity, self.dataDir
        if complexity is not None:
            self.complexity = complexity
        if data_dir is not None:
            self.dataDir = data_dir
        if self.start_pipeline():
            self.watchdogFails = 0
            return True
        # 失败就回滚，避免「界面显示的配置」和「实际在跑的配置」不一致
        self.complexity, self.dataDir = oldC, oldD
        return False

    async def _watchdog(self):
        """自愈：管线线程若意外死了，把它拉起来。

        但**不能无限重试**：配置本身有问题时（例如 poseFile 不存在），
        每秒重试一次只会把日志刷爆（实测刷了几千行、持续好几分钟）且永远起不来。
        连续失败 3 次就停下，把错误留在 configError 里交给界面显示。
        """
        while not self.shuttingDown:
            await asyncio.sleep(1.0)
            p = self.pipeline
            if p is None or self.shuttingDown:
                continue
            if p.is_alive():
                self.watchdogFails = 0
                continue
            if self.watchdogFails >= 3:
                continue
            snap = p.snapshot()
            logging.warning(f"[webui] 看门狗：管线线程已停止"
                            f"（frames={snap.get('frames')} err={snap.get('error')}），尝试重启")
            if self.start_pipeline():
                self.watchdogFails = 0
            else:
                self.watchdogFails += 1
                if self.watchdogFails >= 3:
                    logging.error(f"[webui] 连续 3 次启动失败，停止自动重试。"
                                  f"请修好配置后重新选择：{self.configError}")

    # ------------------------------------------------------------ HTTP
    async def index(self, request):
        return web.FileResponse(os.path.join(STATIC_DIR, "index.html"))

    async def status(self, request):
        snap = self.pipeline.snapshot() if self.pipeline else {}
        return web.json_response({
            "ok": True,
            "server": {
                "infer": self.inferMode,
                "source": (self.source.describe() if self.source
                           else "browser（浏览器内推理，视频不上行）"),
                "sourceKind": ("browser-infer" if self.landmarkMode
                               else self.args.source),
                "complexity": self.complexity,
                "dataDir": self.dataDir,
                "draw": self.serverView,
                "peers": len(self.pcs),
                "producer": self.producer,
                "uptimeSec": round(time.time() - self.startedAt, 1),
            },
            "pipeline": snap,
        })

    async def snapshot(self, request):
        # 关键点模式下服务端根本没有画面，这个接口没有意义
        if self.landmarkMode or not self.pipeline or \
                not hasattr(self.pipeline, "snapshot_jpeg"):
            return web.Response(status=404,
                                text="浏览器推理模式下没有服务端画面")
        jpg = self.pipeline.snapshot_jpeg()
        if not jpg:
            return web.Response(status=503, text="还没有帧（浏览器未推流？）")
        return web.Response(body=jpg, content_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})

    async def configs(self, request):
        """列出可以选用的 config.json 目录。"""
        out = []
        cand = [self.dataDir, os.path.join(ROOT, "data")]
        ex = os.path.join(ROOT, "example")
        if os.path.isdir(ex):
            for name in sorted(os.listdir(ex)):
                cand.append(os.path.join(ex, name))
        for d in cand:
            if d and os.path.exists(os.path.join(d, "config.json")):
                out.append({"dir": d,
                            "name": os.path.basename(d.rstrip("/")) or d,
                            "tasks": _count_tasks(os.path.join(d, "config.json"))})
        # 去重
        seen, uniq = set(), []
        for o in out:
            if o["dir"] in seen:
                continue
            seen.add(o["dir"])
            uniq.append(o)
        return web.json_response({"configs": uniq})

    # ------------------------------------------------------------ WS 信令
    async def ws(self, request):
        ws = web.WebSocketResponse(heartbeat=30, max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)
        pc = None
        logging.info(f"[webui] 信令连接来自 {request.remote}")

        async def send(payload):
            if not ws.closed:
                await ws.send_str(json.dumps(payload))

        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                kind = data.get("type")

                if kind == "offer":
                    if pc is not None:
                        await pc.close()
                    pc = RTCPeerConnection()
                    self.pcs.add(pc)
                    peer_id = id(pc)

                    @pc.on("datachannel")
                    def on_datachannel(channel):
                        logging.info(f"[webui] DataChannel 打开: {channel.label}")
                        self.channels.add(channel)
                        channel.send(json.dumps({
                            "type": "hello",
                            "infer": self.inferMode,
                            "source": (self.source.describe() if self.source
                                       else "browser（浏览器内推理，视频不上行）"),
                            "complexity": self.complexity,
                            "faceEnabled": self.faceEnabled,
                            "faceRate": self.faceRate,
                            "uiParts": ["face", "body", "leftHand", "rightHand"]}))
                        t = asyncio.ensure_future(self._push_state(channel))
                        self.tasks.add(t)
                        t.add_done_callback(self.tasks.discard)

                        @channel.on("message")
                        def on_message(message):
                            asyncio.ensure_future(self._on_control(message, channel))

                    @pc.on("track")
                    def on_track(track):
                        logging.info(f"[webui] 收到远端轨: {track.kind}")
                        if track.kind != "video":
                            return
                        if self.landmarkMode:
                            # 关键点模式下视频不该上行；真上来了就丢掉，别白解码
                            logging.warning("[webui] 浏览器推理模式下收到视频轨，已忽略"
                                            "（视频不需要上行）")
                            return
                        if self.producer is None:
                            self.producer = peer_id
                            logging.info(f"[webui] peer {peer_id} 成为帧提供者")
                        t = asyncio.ensure_future(self._consume(track, peer_id))
                        self.tasks.add(t)
                        t.add_done_callback(self.tasks.discard)

                    @pc.on("icecandidate")
                    async def on_ice(candidate):
                        if candidate:
                            await send({"type": "candidate",
                                        "candidate": {
                                            "candidate": candidate.candidate,
                                            "sdpMid": candidate.sdpMid,
                                            "sdpMLineIndex": candidate.sdpMLineIndex}})

                    @pc.on("connectionstatechange")
                    async def on_state():
                        logging.info(f"[webui] 连接状态: {pc.connectionState}")
                        if pc.connectionState in ("failed", "closed"):
                            await self._cleanup_peer(pc, peer_id)
                        await send({"type": "connectionstate",
                                    "state": pc.connectionState})

                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=data["sdp"], type=data["sdpType"]))
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    # aiortc 会在 setLocalDescription 内完成 ICE 收集，
                    # 因此这里的 SDP 已含候选，无需 trickle
                    await send({"type": "answer",
                                "sdp": pc.localDescription.sdp,
                                "sdpType": pc.localDescription.type})

                elif kind == "candidate" and pc is not None:
                    cand = data.get("candidate") or {}
                    try:
                        from aiortc import RTCIceCandidate
                        await pc.addIceCandidate(RTCIceCandidate(
                            candidate=cand.get("candidate", ""),
                            sdpMid=cand.get("sdpMid"),
                            sdpMLineIndex=cand.get("sdpMLineIndex")))
                    except Exception as e:
                        logging.debug(f"addIceCandidate 忽略: {e}")

                elif kind == "ping":
                    await send({"type": "pong", "t": data.get("t")})

        except Exception as e:
            logging.warning(f"[webui] 信令异常: {type(e).__name__}: {e}")
        finally:
            if pc is not None:
                await self._cleanup_peer(pc, id(pc))
            if not ws.closed:
                await ws.close()
        return ws

    async def _cleanup_peer(self, pc, peer_id):
        try:
            await pc.close()
        except Exception:
            pass
        self.pcs.discard(pc)
        if self.producer == peer_id:
            self.producer = None
            logging.info("[webui] 帧提供者已断开，等待新的浏览器接入")

    # 远端帧 -> 管线
    async def _consume(self, track, peer_id):
        n = 0
        try:
            while True:
                frame = await track.recv()
                if self.producer != peer_id:
                    continue          # 只认第一个提供者，避免多浏览器互相打架
                img = frame.to_ndarray(format="bgr24")
                if self.source is None:
                    # 关键点模式下视频不该上行；真收到了也别白解码
                    logging.warning("[webui] 关键点模式下收到视频帧，已丢弃")
                    return
                self.source.submit(img)
                n += 1
                if n == 1:
                    logging.info(f"[webui] 首帧到达 {img.shape}，推理开始")
        except MediaStreamError:
            pass
        except Exception as e:
            logging.warning(f"[webui] 取帧结束: {type(e).__name__}: {e}")
        finally:
            logging.info(f"[webui] peer {peer_id} 共收到 {n} 帧")

    # 状态推送
    async def _push_state(self, channel):
        """事件驱动推送：管线一产出新结果就发。

        原来是 `while: send(); await sleep(0.1)` 的**固定 10Hz 轮询**，实测代价：
          · 每个新结果平均要等 18.8ms 才发出去（最高 37ms）—— 纯延迟
          · 前端每秒只拿到 ~10 次更新，而视频是 30fps，叠加层明显“跟不上手”
        改成事件驱动后：结果平均年龄 0.7ms，更新率 ~28/s。

        注意：单纯把轮询频率提到 30Hz **没用**（实测仍积压 17.9ms），
        必须由「有新结果」来触发。
        """
        lastEvt = 0.0
        lastFaceSent = 0.0
        try:
            while channel.readyState == "open":
                # 有新结果立刻醒；长时间没帧也每秒醒一次，保证在线时长等状态还在刷新
                try:
                    await asyncio.wait_for(self.stateEvent.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                self.stateEvent.clear()

                snap = self.pipeline.snapshot() if self.pipeline else {}
                now = time.time()
                ft = snap.get("frameT") or 0.0
                # 「结果从产生到发出去」的真实延迟，直接显示给用户看
                snap["latencyMs"] = round((now - ft) * 1e3, 1) if ft else None
                snap["configError"] = self.configError
                snap["dataDir"] = self.dataDir
                snap["pipelineAlive"] = bool(self.pipeline and self.pipeline.is_alive())

                events = self.pipeline.drainEvents(lastEvt) if self.pipeline else []
                if events:
                    lastEvt = events[-1]["t"]

                # 脸点降频：按**时间**判（推送节奏变了，按 tick 计数就不准了）
                if self.faceEnabled and snap.get("face") and \
                        (now - lastFaceSent) >= 1.0 / max(0.5, self.faceRate):
                    lastFaceSent = now
                else:
                    snap["face"] = None

                channel.send(json.dumps({"type": "state", "t": now,
                                         "state": snap, "events": events}))
        except Exception as e:
            logging.debug(f"[webui] 状态推送结束: {e}")
        finally:
            self.channels.discard(channel)

    async def _on_control(self, message, channel):
        try:
            data = json.loads(message)
        except Exception:
            return
        kind = data.get("type")

        # ---- 关键点模式：浏览器把算好的关键点发过来
        if kind == "landmarks":
            if self.landmarkMode and isinstance(self.pipeline, LandmarkTaskEngine):
                lm = parse_landmarks(data)
                lm["frameT"] = data.get("frameT") or time.time()
                self.pipeline.submit(lm)
            return

        if kind == "setComplexity":
            v = int(data.get("value", 1))
            if v in (0, 1, 2) and v != self.complexity:
                if self.landmarkMode:
                    # 浏览器模式没有服务端推理可调；模型选择由前端自己选文件。
                    # 这里只记录并在状态里回显，前端据此切换 pose 模型。
                    self.complexity = v
                    logging.info(f"[webui] 记录模型的精度档 -> {v}（由浏览器切换模型）")
                else:
                    logging.info(f"[webui] 切换 complexity -> {v}（重启管线）")
                    self.restart_pipeline(complexity=v)
        elif kind == "setConfig":
            d = data.get("dataDir") or None
            logging.info(f"[webui] 切换配置 -> {d}（重建任务引擎）")
            self.restart_pipeline(data_dir=d)
        elif kind == "setFace":
            if "enabled" in data:
                self.faceEnabled = bool(data["enabled"])
            if "rate" in data:
                try:
                    self.faceRate = min(10.0, max(0.5, float(data["rate"])))
                except (TypeError, ValueError):
                    pass
            logging.info(f"[webui] 面部关键点下发: enabled={self.faceEnabled} "
                         f"rate={self.faceRate}Hz")
        elif kind == "setServerView":
            if self.landmarkMode:
                # 服务端没有画面，这个开关无意义
                return
            # 运行时可切，不用重启管线（画骨架只是管线里的一个开关）
            self.serverView = bool(data.get("enabled"))
            if self.pipeline:
                self.pipeline.draw = self.serverView
            logging.info(f"[webui] 服务端绘制（服务端视角）: {self.serverView}")


def _count_tasks(config_path):
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        kinds = {}
        for t in cfg:
            kinds[t.get("type")] = kinds.get(t.get("type"), 0) + 1
        return kinds
    except Exception:
        return {}


async def amain(args):
    app = GestureWebApp(args)
    app.start_pipeline()

    watchdog = asyncio.ensure_future(app._watchdog())

    webapp = web.Application()
    webapp.router.add_get("/", app.index)
    webapp.router.add_get("/ws", app.ws)
    webapp.router.add_get("/api/status", app.status)
    webapp.router.add_get("/api/snapshot", app.snapshot)
    webapp.router.add_get("/api/configs", app.configs)
    webapp.router.add_static("/static/", STATIC_DIR, show_index=False)

    runner = web.AppRunner(webapp, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()

    ip = lan_ip()
    print()
    print("  GestureMate WebUI (WebRTC) 已启动")
    print(f"    本机:  http://127.0.0.1:{args.port}")
    if ip:
        print(f"    局域网: http://{ip}:{args.port}   (手机/另一台电脑可开)")
    if app.landmarkMode:
        print("    推理:  浏览器内（MediaPipe Tasks / WebGL）"
              "—— 叠加层与画面同帧，视频不上行")
        print("    模型:  /static/mediapipe/models （face/pose/hand .task）")
    else:
        print("    推理:  服务端 mps 管线")
        print(f"    帧源:  {app.source.describe()}   (--source {args.source})")
    print(f"    配置:  {app.dataDir or '(仅检测手部，不跑任务)'}")
    print(f"    complexity={app.complexity}  draw={app.serverView}")
    print("    浏览器打开后点「允许」授权摄像头，页面会自动建立 WebRTC 连接。")
    print()

    stop = asyncio.Event()

    def _sig(*_a):
        stop.set()

    loop = asyncio.get_event_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _sig)
        except NotImplementedError:
            pass

    await stop.wait()
    print("\n  正在关闭 ...")
    app.shuttingDown = True
    watchdog.cancel()
    if app.pipeline:
        app.pipeline.stop()
        try:
            app.pipeline.join(timeout=5)
        except Exception:
            pass
    for t in list(app.tasks):
        t.cancel()
    for ch in list(app.channels):
        try:
            await ch.close()
        except Exception:
            pass
    for pc in list(app.pcs):
        try:
            await pc.close()
        except Exception:
            pass
    # 帧源归服务端管（管线重启时不关，见 pipeline.py 的说明）；浏览器模式没有帧源
    try:
        if app.source is not None:
            app.source.close()
    except Exception:
        pass
    await runner.cleanup()


def main():
    ap = argparse.ArgumentParser(
        description="GestureMate WebUI —— 浏览器取流 + 手/面部/身体识别（WebRTC）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--infer", default="browser", choices=["browser", "server"],
                    help="browser=推理在浏览器里跑(默认)：叠加层与画面同帧、"
                         "视频不上行；server=推理在服务端跑(旧的 mps 管线)")
    ap.add_argument("--source", default="browser",
                    choices=["browser", "opencv", "ffmpeg", "file"],
                    help="仅 --infer server 时有意义：帧从哪来。"
                         "opencv/ffmpeg=服务端取流(需摄像头权限)；file=视频文件")
    ap.add_argument("--video", default=None, help="--source file 时的视频路径")
    ap.add_argument("--camera-index", type=int, default=None)
    ap.add_argument("--data", default=None, help="含 config.json 的目录（决定跑哪些任务）")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                    help="--infer server: mps 复杂度；--infer browser: 精度档(由前端选模型)")
    ap.add_argument("--no-draw", action="store_true", help="不在服务端画骨架")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S')

    # 逐帧的「任务触发」事件只喂给网页，不要刷控制台（否则一秒几十行）
    root = logging.getLogger()
    root.addFilter(EventMarker())
    for h in root.handlers:
        h.addFilter(DropWebuiEvents())

    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
