"""GestureMate WebUI 的推理管线 —— 与传输层完全解耦。

设计要点
--------
1) **帧源可插拔**。同一个管线既能吃「浏览器用 getUserMedia 抓的本机摄像头」，
   也能吃服务端直接抓的摄像头（OpenCV / ffmpeg），或者一个视频文件。

   为什么默认走浏览器？—— 实测发现 dsh 所在的进程树**没有摄像头权限**：
   OpenCV 报 `not authorized to capture video (status 0)`，ffmpeg 则会在
   协商完像素格式后**卡住并被 macOS SIGKILL**（TCC 拦截）。
   而浏览器是正规 GUI App，弹窗点一下「允许」即可，于是把取流放到浏览器、
   把推理留在 Python 侧，两边各干最擅长的事。

2) **推理沿用已经优化过的 mps 管线**：`LandmarkEngine` + `Utils.extractLandmarks`
   + `TaskController`，所以 WebUI 与命令行版的识别结果、任务语义完全一致。

3) **坐标约定（重要）**：命令行版推理前会把画面水平镜像，再在
   `extractLandmarks` 里把左右手互换来抵消镜像。WebUI 沿用同一套语义，
   但为了避免浏览器端再猜一次，这里统一把关键点的 x 还原到
   **「相机原始（未镜像）」坐标系**再发布，前端就可以直接画在本地视频上。
"""
import json
import logging
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LandmarkEngine import LandmarkEngine, parts_from_config
from Utils import drawLandmarks, extractLandmarks, generateNullLandmarks
from TaskController import TaskController
from face_body import FaceBodyAnalyzer

ALL_PARTS = ("face", "body", "leftHand", "rightHand")
#: WebUI 永远要这几组（画脸/画身体/画手 + 语义识别），与任务配置无关
UI_PARTS = ("face", "body", "leftHand", "rightHand")


# --------------------------------------------------------------------- 帧源
class FrameSource:
    """帧源基类：read() 返回 BGR uint8 (H,W,3)，没有新帧时返回 None。"""

    name = "base"

    def read(self):
        raise NotImplementedError

    def close(self):
        pass

    def describe(self):
        return self.name


class BrowserPushSource(FrameSource):
    """帧由浏览器经 WebRTC 推上来；只保留最新一帧，避免延迟堆积。"""

    name = "browser"

    def __init__(self, maxsize=1):
        self.q = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.received = 0

    def submit(self, frame_bgr):
        self.received += 1
        try:
            self.q.put_nowait(frame_bgr)
        except queue.Full:
            # 丢旧留新：宁可丢帧也不要让端到端延迟越积越大
            try:
                self.q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(frame_bgr)
            except queue.Full:
                pass

    def read(self):
        try:
            return self.q.get(timeout=0.2)
        except queue.Empty:
            return None

    def describe(self):
        return f"browser (收到 {self.received} 帧, 丢弃 {self.dropped})"


class OpenCVSource(FrameSource):
    """服务端直接用 OpenCV 抓摄像头（需要进程有 TCC 摄像头权限）。"""

    name = "opencv"

    def __init__(self, index=0, width=1280, height=720):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        if not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()

    def describe(self):
        return f"opencv index={self.index} opened={self.cap.isOpened()}"


class FFmpegSource(FrameSource):
    """用 ffmpeg/AVFoundation 抓摄像头，绕过 OpenCV 的权限实现。

    注意：在没有 TCC 摄像头权限时，ffmpeg 打开设备会**卡住**，
    因此这里只用它做「有权限时」的备选。
    """

    name = "ffmpeg"

    def __init__(self, index=1, width=1280, height=720, fps=30):
        self.width, self.height = width, height
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation", "-framerate", str(fps),
            "-video_size", f"{width}x{height}", "-i", str(index),
            "-pix_fmt", "bgr24", "-f", "rawvideo", "-",
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL,
                                     bufsize=width * height * 3)
        self.nbytes = width * height * 3

    def read(self):
        buf = self.proc.stdout.read(self.nbytes)
        if not buf or len(buf) < self.nbytes:
            return None
        return np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)

    def close(self):
        try:
            self.proc.kill()
        except Exception:
            pass

    def describe(self):
        return f"ffmpeg avfoundation index={self.index}"


class FileSource(FrameSource):
    """视频文件循环播放（无摄像头时的演示/自测用）。"""

    name = "file"

    def __init__(self, path, loop=True):
        self.path, self.loop = path, loop
        self.cap = cv2.VideoCapture(path)

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()

    def describe(self):
        return f"file {os.path.basename(self.path)}"


# --------------------------------------------------------------------- 事件
class EventLogHandler(logging.Handler):
    """把 Task.process 的 `processing <id>` 抓成 UI 事件流。"""

    def __init__(self, sink, capacity=200):
        super().__init__(level=logging.INFO)
        self.sink = sink
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if msg.startswith("processing "):
            self.sink(msg[len("processing "):])
        elif msg.startswith("[socket]") or msg.startswith("[command]"):
            self.sink(msg, kind="warn")


class EventMarker(logging.Filter):
    """给「任务触发」这类逐帧事件打标记。

    标记之后：UI 侧捕获它们，同时控制台把它们丢掉 —— 否则有手在画面里时
    每秒会刷几十行 `processing left/right`，把服务端日志彻底淹掉。
    """

    PREFIXES = ("processing ", "[socket]", "[command]")

    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if msg.startswith(self.PREFIXES):
            record.webuiEvent = True
        return True


class DropWebuiEvents(logging.Filter):
    """挂在控制台 handler 上，丢掉被 EventMarker 标记过的记录。"""

    def filter(self, record):
        return not getattr(record, "webuiEvent", False)


# --------------------------------------------------------------------- 管线
class GesturePipeline(threading.Thread):
    """抓帧 → mps 推理 → 跑任务 → 发布状态。"""

    def __init__(self, source, data_dir=None, complexity=1, mirror=True,
                 draw=False, want_parts=None, verbose=False, ui_parts=UI_PARTS):
        super().__init__(daemon=True, name="gesture-pipeline")
        self.source = source
        self.complexity = complexity
        self.mirror = mirror
        self.draw = draw
        self.verbose = verbose

        self.controller = TaskController()
        self.need = set(want_parts or ())
        if data_dir:
            cfg = os.path.join(data_dir, "config.json")
            if os.path.exists(cfg):
                self.need |= set(self.controller.readConfig(cfg))
        # UI 要把脸/身体/手都画出来并做语义识别，所以这几组永远取
        self.need |= set(ui_parts or ())

        # 面部/身体的语义分析（有状态：EMA 平滑 + 眨眼计数）
        self.analyzer = FaceBodyAnalyzer()

        self.engine = None
        self.events = deque(maxlen=60)
        self._lock = threading.Lock()
        self._state = {
            "running": False,
            "frames": 0,
            "fps": 0.0,
            "inferMs": 0.0,
            "loopMs": 0.0,
            "hands": {"leftHand": None, "rightHand": None},
            "body": None,
            "face": None,
            "features": None,
            "tasks": {},
            "engine": "-",
            "parts": sorted(self.need),
            "source": source.describe(),
            "error": None,
            "frameT": 0.0,            # 最近一帧结果的产生时间（前端算延迟用）
            "startedAt": time.time(),
        }
        # 注意：**不能**叫 self._stop —— threading.Thread 内部有一个 _stop()
        # 方法，join() 会调用它；用 Event 覆盖掉之后 join() 就会抛
        # TypeError("'Event' object is not callable")。踩过一次，故用 _stopEvent。
        self._stopEvent = threading.Event()
        self._lastDisplay = None      # 最近一帧（带骨架），供 /api/snapshot 用
        #: 每产出一帧新结果就回调一次。服务端用它做**事件驱动推送**：
        #: 有结果立刻发，而不是固定 10Hz 轮询（那样平均白等 19ms）。
        self.onUpdate = None

        self._logHandler = EventLogHandler(self._addEvent)
        logging.getLogger().addHandler(self._logHandler)

    # ---------------------------------------------------------------- 事件
    def _addEvent(self, text, kind="task"):
        with self._lock:
            self.events.append({"t": time.time(), "kind": kind, "text": text})

    def drainEvents(self, since=0.0):
        with self._lock:
            return [e for e in self.events if e["t"] > since]

    # ---------------------------------------------------------------- 状态
    def snapshot(self):
        with self._lock:
            s = dict(self._state)
            s["hands"] = dict(self._state["hands"])
            s["tasks"] = dict(self._state["tasks"])
        s["dropped"] = getattr(self.source, "dropped", 0)
        # 帧源描述必须**实时**算：里面带「收到 N 帧 / 丢弃 M 帧」这类计数，
        # 用构造时缓存的字符串会一直显示 0。
        s["source"] = self.source.describe()
        return s

    def _taskSnapshot(self):
        return {tid: {"type": self.controller.tasks[tid].taskType,
                      "active": bool(self.controller.activate.get(tid))}
                for tid in self.controller.tasks}

    def _publish(self, **kw):
        with self._lock:
            self._state.update(kw)

    def snapshot_jpeg(self, quality=80):
        """把最近一帧编码成 JPEG（给 /api/snapshot 与「服务端视角」用）。"""
        frame = self._lastDisplay
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else None

    # ---------------------------------------------------------------- 主循环
    def run(self):
        try:
            self.engine = LandmarkEngine(model_complexity=self.complexity,
                                         need=self.need, mode="holistic")
        except Exception as e:
            self._publish(error=f"引擎初始化失败: {type(e).__name__}: {e}")
            logging.error(f"[webui] 引擎初始化失败: {e}")
            return

        self._publish(running=True, engine=self.engine.mode)
        # 任务清单要**立刻**发出去：原来只在帧循环里更新，
        # 没接视频流时界面会一直显示「未加载配置」，很容易被误判成配置加载失败。
        self._publish(tasks=self._taskSnapshot())
        logging.info(f"[webui] 管线启动: {self.engine.describe()} | "
                     f"帧源={self.source.describe()} | parts={sorted(self.need)} | "
                     f"任务 {len(self.controller.tasks)} 个")

        lastT = time.time()
        emaInfer = emaLoop = 0.0
        fpsN, fpsT0 = 0, time.time()

        try:
            while not self._stopEvent.is_set():
                frame = self.source.read()
                if frame is None:
                    time.sleep(0.002)
                    continue

                tLoop = time.perf_counter()
                # 镜像：必须用 cv2.flip，别用 numpy 的 f[:, ::-1, :]。
                # 后者是**负步长视图**，ascontiguousarray 得逐元素按步长拷，
                # 实测 1280x720 要 3.93ms；cv2.flip 是 SIMD 优化过的，只要 0.39ms。
                # 一行之差，白省 3.5ms（而且零精度损失）。
                # 顺便：负步长视图还不能直接给 OpenCV 绘图函数写（会报
                # "Layout of the output array img is incompatible with cv::Mat"）。
                inferIn = cv2.flip(frame, 1) if self.mirror else frame

                rgb = cv2.cvtColor(inferIn, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                t0 = time.perf_counter()
                results = self.engine.process(rgb)
                inferMs = (time.perf_counter() - t0) * 1e3

                lm = extractLandmarks(results, self.need)
                if self.mirror:
                    # 还原到「相机原始」坐标系，前端才能直接画在本地视频上
                    lm = _unmirror(lm)

                # 面部/身体语义识别（眼睛开合、眨眼、嘴巴、头朝向、举手…）
                try:
                    feats = self.analyzer.update(lm.get("face"), lm.get("body"),
                                                 (lm.get("leftHand"), lm.get("rightHand")))
                except Exception as e:
                    feats = None
                    logging.debug(f"[webui] 语义分析失败: {e}")

                # 跑任务（沿用命令行版语义）
                try:
                    self.controller.listen(lm)
                except Exception as e:
                    logging.error(f"[webui] 任务执行异常: {type(e).__name__}: {e}")

                if self.draw:
                    # 画骨架失败不应该拖垮整条管线（只影响观感）
                    try:
                        drawLandmarks(inferIn, results)
                    except Exception as e:
                        logging.warning(f"[webui] 绘制骨架失败（已忽略）: {e}")
                        self.draw = False
                self._lastDisplay = inferIn

                # 统计
                loopMs = (time.perf_counter() - tLoop) * 1e3
                emaInfer = inferMs if emaInfer == 0 else emaInfer * 0.9 + inferMs * 0.1
                emaLoop = loopMs if emaLoop == 0 else emaLoop * 0.9 + loopMs * 0.1
                fpsN += 1
                now = time.time()
                fps = 0.0
                if now - fpsT0 >= 0.5:
                    fps = fpsN / (now - fpsT0)
                    fpsN, fpsT0 = 0, now

                tasks = self._taskSnapshot()
                with self._lock:
                    self._state["frames"] += 1
                    self._state["hands"] = {"leftHand": lm.get("leftHand"),
                                            "rightHand": lm.get("rightHand")}
                    self._state["body"] = lm.get("body")
                    self._state["face"] = lm.get("face")
                    self._state["features"] = feats
                    self._state["inferMs"] = round(emaInfer, 2)
                    self._state["loopMs"] = round(emaLoop, 2)
                    self._state["tasks"] = tasks
                    if fps:
                        self._state["fps"] = round(fps, 1)
                    # 这一帧结果产生的时间戳：前端拿它算「结果已经多旧了」，
                    # 用于延迟补偿（外推）以及显示真实延迟。
                    self._state["frameT"] = now
                # 通知推送循环「有新结果了」—— 事件驱动，不再固定 10Hz 轮询。
                # 固定轮询白送 ~19ms 平均延迟（最高 37ms），而且叠加层每秒只更新
                # 10 次、视频 30fps，看起来就是「跟不上手」。
                if self.onUpdate:
                    try:
                        self.onUpdate()
                    except Exception:
                        pass
                lastT = now
        except Exception as e:
            self._publish(error=f"{type(e).__name__}: {e}")
            logging.error(f"[webui] 管线异常: {e}")
        finally:
            self._publish(running=False)
            try:
                self.engine and self.engine.close()
            except Exception:
                pass
            # 注意：**不要**在这里 close 帧源。管线可能被重启（切复杂度/看门狗），
            # 而 opencv/file 帧源一旦 close 就废了，重启后会拿到一个关闭的句柄。
            # 帧源的生命周期归服务端管（见 server.py 的关闭流程）。
            logging.getLogger().removeHandler(self._logHandler)

    def stop(self):
        self._stopEvent.set()


def _unmirror(lm, places=4):
    """把镜像坐标系里的关键点 x 还原：x' = 1 - x。

    顺带把坐标压到 4 位小数：468 个脸点每帧都要序列化成 JSON，
    不压的话光是小数尾巴就能把流量撑大一倍（本地链路也别浪费）。
    """
    out = {}
    for k, v in lm.items():
        if not v:
            out[k] = None
            continue
        out[k] = [[round(1.0 - p[0], places), round(p[1], places),
                   round(p[2], places)] for p in v]
    return out


def build_source(kind, video=None, camera_index=None):
    """按参数构造帧源。kind: browser | opencv | ffmpeg | file"""
    if kind == "file" or video:
        if not video:
            raise SystemExit("--source file 需要同时给 --video")
        return FileSource(video)
    if kind == "opencv":
        return OpenCVSource(camera_index or 0)
    if kind == "ffmpeg":
        return FFmpegSource(camera_index if camera_index is not None else 1)
    return BrowserPushSource()


#: 各组关键点的上限（人脸带虹膜是 478，MediaPipe Tasks 默认给 468）
LANDMARK_LIMITS = {"face": 478, "body": 33, "leftHand": 21, "rightHand": 21}


def parse_landmarks(msg):
    """校验并规整浏览器发来的关键点。

    浏览器算是「半个信任边界」：同机同源，但依然是外部输入。这里做三件事：
      1) 按上限截断点数
      2) 强制转 float、丢掉非法项
      3) 坐标夹到 [0,1]（z 允许为负，它是相对深度）
    少任何一步，一个 NaN 就能顺着 TaskController 把整条任务链搞崩。
    """
    out = {}
    for key, limit in LANDMARK_LIMITS.items():
        v = msg.get(key)
        if not isinstance(v, list):
            out[key] = None
            continue
        pts = []
        for p in v[:limit]:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                x, y = float(p[0]), float(p[1])
                z = float(p[2]) if len(p) > 2 else 0.0
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            pts.append([min(1.0, max(0.0, x)), min(1.0, max(0.0, y)), z])
        out[key] = pts or None
    return out


class LandmarkTaskEngine(threading.Thread):
    """关键点模式：推理在浏览器里跑，服务端只负责「看懂 + 执行」。

    和 `GesturePipeline` 的区别：没有 mps、没有帧源、没有图像 —— 浏览器算完
    关键点经 DataChannel 发过来，这里只做两件很便宜的事：

        1) `FaceBodyAnalyzer` 语义分析（眼睛开合/眨眼/嘴巴/头朝向/举手）~0.03ms
        2) `TaskController.listen` 跑任务系统

    加起来远低于 1ms，但**仍然放在独立线程**里：任务里的 command / keypress /
    socket 都可能阻塞（模拟按键、跑子进程），绝不能让它们卡住 asyncio 事件循环。

    为什么要独立线程而不是直接在 DataChannel 回调里同步跑？见上一段 —— 阻塞。
    """
    #: 服务端视角在这个模式下没有意义（服务端根本没有画面）
    draw = False

    def __init__(self, data_dir=None, verbose=False):
        super().__init__(daemon=True, name="landmark-tasks")
        self.verbose = verbose
        self.dataDir = data_dir
        self.controller = TaskController()
        self.configError = None
        self.need = set()
        self.configError = None
        if data_dir:
            cfg = os.path.join(data_dir, "config.json")
            if os.path.exists(cfg):
                try:
                    self.need |= set(self.controller.readConfig(cfg))
                except Exception as e:
                    # 配置坏了也不能让整条路挂掉：识别照旧，任务为空
                    self.configError = f"{type(e).__name__}: {e}"
                    logging.error(f"[webui] 任务配置加载失败（识别不受影响）: {self.configError}")
        self.need |= set(ALL_PARTS)

        self.analyzer = FaceBodyAnalyzer()
        self.events = []
        self._lock = threading.Lock()
        self._stopEvent = threading.Event()
        self._cv = threading.Condition()
        self._pending = None
        self.onUpdate = None          # 有新结果时回调（事件驱动推送）
        self._lastMs = 0.0            # 上一帧关键点的到达时间（算「输入帧率」）
        self._inMs = 0.0
        self._state = {
            "running": False,
            "frames": 0,
            "fps": 0.0,
            "inferMs": 0.0,           # 本模式下=收到关键点的间隔
            "loopMs": 0.0,            # 服务端处理耗时（应当极小）
            "hands": {"leftHand": None, "rightHand": None},
            "body": None,
            "face": None,
            "features": None,
            "tasks": {},
            "engine": "browser",
            "parts": sorted(self.need),
            "source": "browser (MediaPipe Tasks，浏览器内推理，视频不上行)",
            "error": None,
            "frameT": 0.0,
            "startedAt": time.time(),
            "infer": "browser",
        }
        self._logHandler = EventLogHandler(self._addEvent)
        logging.getLogger().addHandler(self._logHandler)

    # ---------------------------------------------------------------- 事件
    def _addEvent(self, text, kind="task"):
        with self._lock:
            self.events.append({"t": time.time(), "kind": kind, "text": text})

    def drainEvents(self, since=0.0):
        with self._lock:
            return [e for e in self.events if e["t"] > since]

    # ---------------------------------------------------------------- 状态
    def snapshot(self):
        with self._lock:
            s = dict(self._state)
            s["hands"] = dict(self._state["hands"])
            s["tasks"] = dict(self._state["tasks"])
        s["configError"] = self.configError
        return s

    def _taskSnapshot(self):
        return {tid: {"type": self.controller.tasks[tid].taskType,
                      "active": bool(self.controller.activate.get(tid))}
                for tid in self.controller.tasks}

    def _publish(self, **kw):
        with self._lock:
            self._state.update(kw)

    # ---------------------------------------------------------------- 输入
    def submit(self, lm):
        """浏览器送来一帧关键点。只保留最新一帧（丢旧留新）。

        和帧源那边一样：宁可丢帧也不能积压，否则延迟会越堆越长。
        """
        with self._cv:
            self._pending = lm
            self._cv.notify()

    # ---------------------------------------------------------------- 主循环
    def run(self):
        self._publish(running=True, engine="browser",
                      tasks=self._taskSnapshot(),
                      parts=sorted(self.need))
        logging.info(f"[webui] 任务引擎启动（关键点模式）："
                     f"tasks={len(self.controller.tasks)} parts={sorted(self.need)}")
        if self.onUpdate:
            try:
                self.onUpdate()
            except Exception:
                pass
        emaLoop = 0.0
        emaGap = 0.0
        fpsN, fpsT0 = 0, time.time()
        try:
            while not self._stopEvent.is_set():
                with self._cv:
                    while self._pending is None and not self._stopEvent.is_set():
                        self._cv.wait(0.5)
                    lm = self._pending
                    self._pending = None
                if lm is None:
                    continue
                t0 = time.perf_counter()
                arrived = time.time()
                if self._lastMs:
                    gap = (arrived - self._lastMs) * 1e3
                    emaGap = gap if emaGap == 0 else emaGap * 0.9 + gap * 0.1
                self._lastMs = arrived

                try:
                    feats = self.analyzer.update(lm.get("face"), lm.get("body"),
                                                 (lm.get("leftHand"), lm.get("rightHand")))
                except Exception as e:
                    feats = None
                    logging.debug(f"[webui] 语义分析失败: {e}")

                try:
                    self.controller.listen(lm)
                except Exception as e:
                    logging.error(f"[webui] 任务执行异常: {type(e).__name__}: {e}")

                loopMs = (time.perf_counter() - t0) * 1e3
                emaLoop = loopMs if emaLoop == 0 else emaLoop * 0.9 + loopMs * 0.1
                fpsN += 1
                now = time.time()
                fps = 0.0
                if now - fpsT0 >= 0.5:
                    fps = fpsN / (now - fpsT0)
                    fpsN, fpsT0 = 0, now

                with self._lock:
                    self._state["frames"] += 1
                    self._state["hands"] = {"leftHand": lm.get("leftHand"),
                                            "rightHand": lm.get("rightHand")}
                    self._state["body"] = lm.get("body")
                    self._state["face"] = lm.get("face")
                    self._state["features"] = feats
                    self._state["inferMs"] = round(emaGap, 2)   # 输入间隔
                    self._state["loopMs"] = round(emaLoop, 2)   # 服务端处理
                    self._state["tasks"] = self._taskSnapshot()
                    self._state["frameT"] = now
                    if fps:
                        self._state["fps"] = round(fps, 1)
                if self.onUpdate:
                    try:
                        self.onUpdate()
                    except Exception:
                        pass
        except Exception as e:
            self._publish(error=f"{type(e).__name__}: {e}")
            logging.error(f"[webui] 任务引擎异常: {e}")
        finally:
            self._publish(running=False)
            logging.getLogger().removeHandler(self._logHandler)

    def stop(self):
        self._stopEvent.set()
        with self._cv:
            self._cv.notify()
