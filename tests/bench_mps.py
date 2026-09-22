#!/usr/bin/env python3
"""GestureMate · mps (mediapipe.python.solutions) 管线基准测试

用一个**确定性**的帧源（MediaPipe 官方测试数据里的真实手势照片）驱动
*真正的* TaskController.startListen()，从而在不同代码版本 / 不同开关之间
做可复现的对比。

    # 完整消融：原版 vs 优化后（逐帧日志、无条件绘制各自贡献多少）
    .venv/bin/python tests/bench_mps.py --srcdir . --baseline tests/baseline --matrix

    # 只看默认配置
    .venv/bin/python tests/bench_mps.py --srcdir .

测什么:
    * 端到端吞吐（帧/秒）—— 版本无关、客观
    * 每帧写出的 logging 记录条数（原代码逐帧写 stdout + error.log）
    * 优化版额外给出「纯推理均值 ms」，用于区分引擎收益与外围开销

不做的事: 不碰摄像头、不显示窗口、不执行任何 command 任务。
"""
import argparse
import importlib
import inspect
import json
import logging
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, "fixtures", "hands")


# ----------------------------------------------------------------- 帧源
def load_fixture_frames(width, height):
    """把真实手势照片读成 (H,W,3) BGR 帧列表（居中贴到目标画幅）。"""
    names = sorted(n for n in os.listdir(FIXTURE_DIR)
                   if n.lower().endswith((".jpg", ".jpeg", ".png")))
    if not names:
        raise SystemExit(f"没有找到测试图片: {FIXTURE_DIR}")
    frames = []
    for n in names:
        img = cv2.imread(os.path.join(FIXTURE_DIR, n))
        if img is None:
            continue
        h, w = img.shape[:2]
        s = min(width / w, height / h)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        y = (height - img.shape[0]) // 2
        x = (width - img.shape[1]) // 2
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        frames.append(canvas)
    return frames


class FakeCapture:
    """冒充 cv2.VideoCapture(0)，吐完固定数量的帧后自行「结束」。"""

    def __init__(self, frames, total):
        self.frames = frames
        self.total = total
        self.i = 0

    def isOpened(self):
        return self.i < self.total

    def read(self):
        if self.i >= self.total:
            return False, None
        f = self.frames[self.i % len(self.frames)]
        self.i += 1
        return True, f.copy()

    def set(self, *a, **k):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.frames[0].shape[1]
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frames[0].shape[0]
        return 0.0

    def release(self):
        pass


class CountingHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.count = 0

    def emit(self, record):
        self.count += 1


def make_bench_config(data_dir, hands_dir):
    """只含 detect / match 的配置——没有任何副作用任务。"""
    cfg = [
        {"type": "detect", "id": "dl", "bodyPart": ["leftHand"], "frames": 3,
         "start": True, "nextTasks": [{"operate": "start", "id": "dl"}]},
        {"type": "detect", "id": "dr", "bodyPart": ["rightHand"], "frames": 3,
         "start": True, "nextTasks": [{"operate": "start", "id": "dr"}]},
    ]
    hands = sorted(n for n in os.listdir(hands_dir) if n.endswith(".json"))
    if hands:
        # 加一个 match 任务，更接近真实负载（也覆盖连续帧计数逻辑）
        cfg.append({
            "type": "match", "id": "m1",
            "bodyPart": [["leftHand"]],
            "poseFile": [os.path.join(hands_dir, hands[0])],
            "sensetive": [0.5], "frames": [2],
            "start": True, "nextTasks": [{"operate": "start", "id": "m1"}],
        })
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def reset_logging(log_path, level="info", mirror=True):
    """复刻 Main.py 的 logging 配置（stdout + 文件），外加一个计数器。"""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(getattr(logging, level.upper()))
    counter = CountingHandler()
    root.addHandler(counter)
    if mirror:
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        sh = logging.StreamHandler(open(os.devnull, "w"))
        sh.setFormatter(fmt)
        root.addHandler(sh)
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return counter


def run_once(srcdir, config_path, frames, n_frames, complexity, width, height,
             cvshow, quiet, proc_size, log_level="info", engine="holistic"):
    """在 srcdir 版本的代码上跑一次；异常不抛出，记录在 error 里。"""
    srcdir = os.path.abspath(srcdir)
    # 注意要删掉 `tasks` **包本身**（不只是 tasks.*），否则第二次运行会继续
    # 复用上一次导入的包，对比就串味了。
    for m in [m for m in list(sys.modules)
              if m in ("TaskController", "Utils", "LandmarkEngine", "tasks")
              or m.startswith("tasks.")]:
        del sys.modules[m]
    sys.path.insert(0, srcdir)

    capture = FakeCapture(frames, n_frames)
    orig = (cv2.VideoCapture, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows)
    cv2.VideoCapture = lambda *a, **k: capture
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: -1
    cv2.destroyAllWindows = lambda *a, **k: None

    log_path = os.path.join(HERE, "_bench_error.log")
    counter = reset_logging(log_path, level=log_level, mirror=not quiet)

    error = None
    stats = None
    c = None
    t0 = time.perf_counter()
    try:
        mod = importlib.import_module("TaskController")
        c = mod.TaskController()
        c.readConfig(config_path)
        params = inspect.signature(c.startListen).parameters
        kwargs = {}
        if "procSize" in params:
            kwargs["procSize"] = proc_size
        if "video" in params:
            kwargs["video"] = None
        if "engine" in params:
            kwargs["engine"] = engine
        c.startListen(0, complexity, cvshow, **kwargs)
    except BaseException as e:          # 原版会抛 TypeError，记下来而不是中断
        error = f"{type(e).__name__}: {e}"
    finally:
        dt = time.perf_counter() - t0
        cv2.VideoCapture, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows = orig
        if srcdir in sys.path:
            sys.path.remove(srcdir)
        if c is not None:
            stats = getattr(c, "bench_stats", None)

    done = capture.i
    return {
        "frames": done,
        "seconds": dt,
        "fps": done / dt if dt > 0 else 0.0,
        "log_records": counter.count,
        "error": error,
        "stats": stats,
        "srcdir": srcdir,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcdir", default=".")
    ap.add_argument("--baseline", default=None, help="优化前代码快照目录")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--proc-size", type=int, default=0)
    ap.add_argument("--engine", default="holistic",
                    choices=["holistic", "minimal", "auto"])
    ap.add_argument("--quiet", action="store_true", help="不挂 stdout/文件 handler")
    ap.add_argument("--matrix", action="store_true",
                    help="跑完整消融：日志×绘制×引擎")
    args = ap.parse_args()

    frames = load_fixture_frames(args.width, args.height)
    print(f"帧源: {len(frames)} 张真实手势照片 -> 画幅 {args.width}x{args.height}")
    print(f"      共 {args.frames} 帧, complexity={args.complexity}")
    print()

    data_dir = os.path.join(HERE, "_bench_data")
    hands_dir = os.path.join(os.path.dirname(HERE), "example", "data_example", "hand")
    if not os.path.isdir(hands_dir):
        hands_dir = os.path.join(HERE, "fixtures")
    config_path = make_bench_config(data_dir, hands_dir)

    if args.matrix:
        variants = []
        if args.baseline:
            # 原版：逐帧 INFO + 无条件绘制，两者都还在
            variants.append(("优化前 原版(逐帧日志+绘制)", args.baseline,
                             "info", True, 0, "holistic"))
        # 用优化后的代码，把两个开关重新打开，逐一量化各自的开销
        variants += [
            ("A 逐帧日志开+绘制开", args.srcdir, "debug", True, 0, "holistic"),
            ("B 逐帧日志关+绘制开", args.srcdir, "info", True, 0, "holistic"),
            ("C 逐帧日志关+绘制关", args.srcdir, "info", False, 0, "holistic"),
            ("D = C + 只跑手模型", args.srcdir, "info", False, 0, "minimal"),
            ("E = C + procSize=640", args.srcdir, "info", False, 640, "holistic"),
        ]
    else:
        variants = [("优化后(默认)", args.srcdir, "info", False, args.proc_size,
                     args.engine)]

    rows = []
    print(f"{'方案':<26}{'帧':>5}{'耗时':>9}{'FPS':>9}{'日志条数':>10}"
          f"{'推理ms':>10}  引擎")
    print("-" * 88)
    for label, srcdir, lvl, draw, ps, eng in variants:
        r = run_once(srcdir, config_path, frames, args.frames, args.complexity,
                     args.width, args.height, draw, args.quiet, ps, lvl, eng)
        if r["error"]:
            print(f"{label:<26}{r['frames']:>5}{'崩':>9}  "
                  f"{r['error'].splitlines()[0][:44]}")
        else:
            st = r["stats"] or {}
            print(f"{label:<26}{r['frames']:>5}{r['seconds']:>8.2f}s"
                  f"{r['fps']:>9.1f}{r['log_records']:>10}"
                  f"{st.get('engine_ms', 0):>10.1f}  {st.get('engine', '?')}")
        rows.append((label, r))
    print("-" * 88)

    by = {l: r for l, r in rows}
    a = by.get("A 逐帧日志开+绘制开")
    b = by.get("B 逐帧日志关+绘制开")
    c = by.get("C 逐帧日志关+绘制关")
    if a and b and c and not any(x["error"] for x in (a, b, c)):
        print(f"A→B 只关逐帧日志: {a['fps']:.1f} -> {b['fps']:.1f} FPS "
              f"({b['fps']/a['fps']:.2f}x, 日志 {a['log_records']}->{b['log_records']})")
        print(f"B→C 再关无条件绘制: {b['fps']:.1f} -> {c['fps']:.1f} FPS "
              f"({c['fps']/b['fps']:.2f}x)")
        print(f"A→C 合计: {a['fps']:.1f} -> {c['fps']:.1f} FPS "
              f"({c['fps']/a['fps']:.2f}x)")


if __name__ == "__main__":
    main()
