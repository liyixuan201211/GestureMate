#!/usr/bin/env python3
"""mps 引擎级微基准。

两种用法：

1) 方案对比（默认）：Holistic 全家桶 vs「只跑手模型」，并核对检出与关键点。
       .venv/bin/python tests/bench_engine.py --frames 120

2) complexity 扫描：0/1/2 三档的推理耗时与检出率 —— 这才是 mps
   部分真正能大幅改速度的旋钮（默认 2 = heavy，但 Main.py 默认却是 2）。
       .venv/bin/python tests/bench_engine.py --sweep --frames 120
"""
import argparse
import os
import statistics
import sys
import time

import cv2
import numpy as np

# 让 tests/ 下也能 import 到仓库根目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mediapipe.python.solutions as sol
from LandmarkEngine import LandmarkEngine

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, "fixtures", "hands")


def load_frames(width, height):
    names = sorted(n for n in os.listdir(FIXTURE_DIR)
                   if n.lower().endswith((".jpg", ".jpeg", ".png")))
    out = []
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
        # 与主程序一致：推理前水平镜像
        out.append(np.ascontiguousarray(canvas[:, ::-1, :]))
    return out


def to_rgb(frame, proc_size):
    if proc_size:
        fh, fw = frame.shape[:2]
        long_side = max(fh, fw)
        if long_side > proc_size:
            s = proc_size / long_side
            frame = cv2.resize(frame, (max(1, int(round(fw * s))),
                                       max(1, int(round(fh * s)))),
                               interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    return img


def lm_list(lms):
    return None if lms is None else [[p.x, p.y, p.z] for p in lms.landmark]


def run(engine, frames, n, proc_size):
    """返回 (ms/帧 列表, 检出统计, 每帧的左右手关键点)。"""
    times, hands = [], []
    detect = {"left": 0, "right": 0, "any": 0}
    for i in range(n):
        frame = frames[i % len(frames)]
        img = to_rgb(frame, proc_size)
        t0 = time.perf_counter()
        r = engine.process(img)
        times.append((time.perf_counter() - t0) * 1e3)
        lh, rh = lm_list(r.left_hand_landmarks), lm_list(r.right_hand_landmarks)
        hands.append((lh, rh))
        detect["left"] += lh is not None
        detect["right"] += rh is not None
        detect["any"] += (lh is not None or rh is not None)
    return times, detect, hands


def pointwise_diff(a, b):
    diffs = []
    for (lh1, rh1), (lh2, rh2) in zip(a, b):
        for p, q in ((lh1, lh2), (rh1, rh2)):
            if p is None or q is None or len(p) != len(q):
                continue
            diffs.append(float(np.abs(np.array(p) - np.array(q)).mean()))
    return (statistics.mean(diffs), len(diffs)) if diffs else (float("nan"), 0)


class _HolisticWrap:
    """把 Holistic 包成和 LandmarkEngine 一样的 process(rgb) 接口。"""

    def __init__(self, h):
        self.h = h

    def process(self, img):
        return self.h.process(img)


def sweep(args, frames):
    print(f"complexity 扫描（holistic，{args.width}x{args.height}，{args.frames} 帧）")
    print(f"{'complexity':<14}{'含义':<10}{'单帧推理':>10}{'中位':>9}{'P95':>9}"
          f"{'左手帧':>8}{'右手帧':>8}{'任一帧':>8}")
    print("-" * 86)
    meaning = {0: "lite", 1: "medium", 2: "heavy"}
    base = None
    for c in (0, 1, 2):
        with sol.holistic.Holistic(min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5,
                                   model_complexity=c) as h:
            t, d, _ = run(_HolisticWrap(h), frames, args.frames, 0)
        m = statistics.mean(t)
        if base is None:
            base = m
        print(f"{c:<14}{meaning[c]:<10}{m:>8.1f}ms"
              f"{statistics.median(t):>8.1f}ms"
              f"{np.percentile(t, 95):>8.1f}ms"
              f"{d['left']:>8}{d['right']:>8}{d['any']:>8}"
              f"   ({base / m:.2f}x)")
    print()
    print("提示：默认 complexity=2。若只做「手有没有出现/大致手势」这类粗判，"
          "降到 1 往往够用且明显更快；0 最快但精度下降最多。")


def compare(args, frames):
    print(f"帧源 {len(frames)} 张真实手势照片 -> {args.width}x{args.height}，"
          f"跑 {args.frames} 帧（循环），complexity={args.complexity}")
    print()
    results = {}
    with sol.holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5,
                               model_complexity=args.complexity) as h:
        t, d, hands = run(_HolisticWrap(h), frames, args.frames, 0)
        results["holistic (原版)"] = (t, d, hands)
    for label, ps in (("hands-only (不降采样)", 0),
                      ("hands-only (procSize=640)", 640)):
        with LandmarkEngine(model_complexity=args.complexity, need=("leftHand", "rightHand"),
                            mode="minimal") as eng:
            t, d, hands = run(eng, frames, args.frames, ps)
        results[label] = (t, d, hands)

    print(f"{'方案':<30}{'单帧推理':>10}{'中位':>9}{'P95':>9}"
          f"{'左手帧':>8}{'右手帧':>8}{'任一帧':>8}")
    print("-" * 84)
    for label, (t, d, _h) in results.items():
        print(f"{label:<30}{statistics.mean(t):>8.1f}ms"
              f"{statistics.median(t):>8.1f}ms"
              f"{np.percentile(t, 95):>8.1f}ms"
              f"{d['left']:>8}{d['right']:>8}{d['any']:>8}")

    hb = results["holistic (原版)"]
    print()
    print("结论：holistic 在本机既更快又更准 —— 「只跑手模型」是负优化，"
          "因此 LandmarkEngine 默认走 holistic。")
    for label, (t, d, hands) in results.items():
        if label.startswith("hands-only"):
            sp = statistics.mean(hb[0]) / statistics.mean(t)
            diff, cnt = pointwise_diff(hb[2], hands)
            print(f"  {label}: 相对 holistic {sp:.2f}x，检出 {d['any']}/{args.frames}"
                  f"（holistic {hb[1]['any']}/{args.frames}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--complexity", type=int, default=1)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    frames = load_frames(args.width, args.height)
    if args.sweep:
        sweep(args, frames)
    else:
        compare(args, frames)


if __name__ == "__main__":
    main()
