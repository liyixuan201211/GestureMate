from tasks import *
import json
import os
import cv2 as cv
from Utils import drawLandmarks, extractLandmarks, generateNullLandmarks
import time
import sys
from collections import deque
import math
import logging

from LandmarkEngine import LandmarkEngine, parts_from_config


class TaskController:
    FPS_COUNT_FRAME = 10

    def __init__(self):
        self.tasks = {}
        self.activate = {}
        self.usedParts = None
        self.consoleUI = False
        #: 基准/诊断用途：最近一次 startListen 的分段耗时
        self.bench_stats = None

    def listen(self, x):
        # 原实现每帧都 print 清屏转义序列并强制 flush stdout；只有在真的挂着
        # 终端、又不开 cvShow 时才需要它。
        if self.consoleUI:
            print("\033[H\033[J")

        for i in self.tasks.keys():
            if self.activate[i]:
                self.tasks[i].listen(x)

    def activateTask(self, id: str, x):
        if not id in self.activate.keys():
            logging.error(f"Unknown task id {id}")
            return
        self.activate[id] = True
        self.tasks[id].activate(x)

    def deactivateTask(self, id: str, x):
        if not id in self.activate.keys():
            logging.error(f"Unknown task id {id}")
            return
        self.activate[id] = False
        self.tasks[id].deactivate(x)

    def removeTask(self, id: str):
        self.activate.pop(id)

    def addTask(self, task: Task):
        self.tasks[task.id] = task
        self.activate[task.id] = False
        if task.start:
            self.activateTask(task.id, generateNullLandmarks())

    def clear(self):
        self.tasks = {}
        self.Activate = {}

    def readConfig(self, path: str, base_dir: str = None):
        """读取 config.json 并建好所有任务。

        base_dir 用来解析配置里那些**相对路径**（主要是 match 的 poseFile）。
        配置里的 `"./hand/hand0.json"` 是相对**配置所在目录**写的 ——
        命令行版靠 `os.chdir(dataDir)` 才能找到；WebUI 不能 chdir
        （那是全局进程状态，会把别的相对路径一起搞乱），所以这里显式传入。
        默认取 path 所在目录，因此命令行版（path="config.json" 且已 chdir）
        行为完全不变。
        """
        with open(path, "r") as f:
            config = json.loads(f.read())

        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(path))

        def _resolve(p):
            return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))

        # 先算出这份配置真正引用到的部位，决定后面要跑哪些模型
        self.usedParts = parts_from_config(config)
        for task in config:
            taskType = task.get('type')
            if taskType == "command":
                taskObject = CommandTask(self, task['id'], task['command'],
                                         task.get('timeout',[]),
                                         task.get('nextTasks', []),
                                         task.get('start', False))
            elif taskType == "keypress":
                taskObject = KeyTask(self, task['id'], task['keys'],
                                     task.get('nextTasks', []),
                                     task.get('start', False))
            elif taskType == "detect":
                taskObject = DetectTask(self, task['id'], task['bodyPart'],
                                        task['frames'],
                                        task.get('nextTasks', []),
                                        task.get('start', False))
            elif taskType == "match":
                poseFiles = task['poseFile']
                if isinstance(poseFiles, str):
                    poseFiles = [poseFiles]
                taskObject = MatchTask(self, task['id'], task['bodyPart'],
                                       [_resolve(p) for p in poseFiles],
                                       task['sensetive'], task['frames'],
                                       task.get('nextTasks', []),
                                       task.get('start', False))
            elif taskType == "timeout":
                taskObject = TimeoutTask(self, task['id'], task['timeout'],
                                         task.get('nextTasks', []),
                                         task.get('start', False))
            elif taskType == "socketsend":
                taskObject = SocketSendTask(self, task['id'],
                                            task.get('ip', "127.0.0.1"),
                                            task['port'],
                                            task.get('extra', None),
                                            task.get('nextTasks', []),
                                            task.get('start', False))
            elif taskType == "request":
                taskObject = RequestTask(self, task['id'],
                                            task['url'],
                                            task['port'],
                                            task.get('data', {}),
                                            task.get('headers', {}),
                                            task.get('cookies', {}),
                                            task.get('nextTasks', []),
                                            task.get('start', False))
            else:
                # 原来这里会直接落到 addTask(taskObject) 上，
                # 抛一个莫名其妙的 UnboundLocalError。给个能看懂的错误。
                raise ValueError(
                    f"不认识的 task type: {taskType!r} "
                    f"(id={task.get('id')!r})，支持 command/keypress/detect/"
                    f"match/timeout/socketsend/request")
            self.addTask(taskObject)
        return self.usedParts

    # ------------------------------------------------------------ 主循环
    def startListen(self, targetFPS, modelComplexity, cvShow,
                    video=None, procSize=0, consoleUI=None, engine="holistic"):
        """采集 → MediaPipe 推理 → 分发任务。

        video     : None/0 用摄像头，否则读视频文件（无相机也能复现与压测）
        procSize  : 推理前把长边缩到这个尺寸，0/None 表示不缩放。默认 0 ——
                    实测缩到 640 只快约 2%，却可能影响小目标手部检出，不值。
        consoleUI : 是否使用「每帧清屏」的控制台界面。默认只在挂着终端且
                    没开 cvShow 时启用。
        engine    : holistic | minimal | auto，见 LandmarkEngine。
                    默认 holistic —— 实测比「只跑手」更快也更准。
        """
        if consoleUI is None:
            consoleUI = (not cvShow) and sys.stdout.isatty()
        self.consoleUI = consoleUI

        source = 0 if video in (None, "", "camera", "0", 0) else video
        camera = cv.VideoCapture(source)
        if source == 0:
            camera.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
            camera.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
            camera.set(cv.CAP_PROP_FPS, 60)
        if not camera.isOpened():
            logging.error(f"打不开视频源: {source!r}")
            return

        isFile = source != 0
        w = int(camera.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
        logging.info(f"视频源: {'摄像头 0' if not isFile else source}"
                     f"{f' ({w}x{h})' if w else ''}")

        need = self.usedParts or parts_from_config([])
        engineObj = LandmarkEngine(model_complexity=modelComplexity, need=need,
                                   mode=engine)
        logging.info(f"[mps] {engineObj.describe()}  procSize={procSize or 'off'}")

        q = deque([], self.FPS_COUNT_FRAME + 10)
        framecnt = 0
        waitTime = 1
        engine_ms = []
        loop_ms = []
        try:
            while camera.isOpened():
                ret, frame = camera.read()
                if not ret:
                    if isFile:
                        break          # 视频读完正常收工
                    time.sleep(0.005)
                    continue

                tLoop = time.perf_counter()
                start = time.time() * 1e3

                frame = frame[:, ::-1, :]
                if procSize:
                    fh, fw = frame.shape[:2]
                    longSide = max(fh, fw)
                    if longSide > procSize:
                        s = procSize / longSide
                        infer = cv.resize(
                            frame,
                            (max(1, int(round(fw * s))), max(1, int(round(fh * s)))),
                            interpolation=cv.INTER_AREA)
                    else:
                        infer = frame
                else:
                    infer = frame

                image = cv.cvtColor(infer, cv.COLOR_BGR2RGB)
                image.flags.writeable = False
                tInfer = time.perf_counter()
                results = engineObj.process(image)
                engine_ms.append((time.perf_counter() - tInfer) * 1e3)

                if cvShow:
                    # 只有真的要显示时才需要转回 BGR，也只有这时才值得画骨架
                    image.flags.writeable = True
                    display = cv.cvtColor(image, cv.COLOR_RGB2BGR)
                    drawLandmarks(display, results)

                if framecnt >= self.FPS_COUNT_FRAME * 3:
                    self.listen(extractLandmarks(results, need))

                now = time.time() * 1e3
                if framecnt >= self.FPS_COUNT_FRAME:
                    last = q.pop()
                    fps = self.FPS_COUNT_FRAME * 1e3 / (now - last)
                    logging.debug(f"FPS: {fps:.3f}")
                    if cvShow:
                        cv.putText(display, f"FPS: {fps:.3f}", (10, 30),
                                   cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0),
                                   bottomLeftOrigin=False)
                    if targetFPS != 0:
                        waitTime = max(1,
                                       math.floor(1e3 / targetFPS - (now - start)) - 3)
                    else:
                        waitTime = 1
                q.appendleft(now)

                if cvShow:
                    cv.imshow('OpenCV Feed', display)
                framecnt += 1
                loop_ms.append((time.perf_counter() - tLoop) * 1e3)

                if cvShow:
                    if cv.waitKey(waitTime) & 0xFF == ord('q'):
                        for task in self.tasks.keys():
                            self.deactivateTask(task, generateNullLandmarks())
                            break
                else:
                    time.sleep(waitTime / 1000)
        except KeyboardInterrupt:
            for task in self.tasks.keys():
                self.deactivateTask(task, generateNullLandmarks())
                break
        finally:
            camera.release()
            engineObj.close()
            if cvShow:
                cv.destroyAllWindows()

            def mean(xs):
                return sum(xs) / len(xs) if xs else 0.0

            self.bench_stats = {
                "frames": framecnt,
                "engine_ms": mean(engine_ms),
                "loop_ms": mean(loop_ms),
                "engine": engineObj.mode,
                "parts": sorted(need),
                "procSize": procSize,
            }
            logging.info(
                f"[mps] 处理 {framecnt} 帧 | 推理均值 {mean(engine_ms):.1f} ms "
                f"| 单帧全流程 {mean(loop_ms):.1f} ms "
                f"| 理论上限 {1000 / mean(loop_ms) if loop_ms else 0:.1f} FPS")
