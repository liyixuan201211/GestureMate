#!/usr/bin/env python3
"""延迟剖析：把 WebUI 一帧的时间拆开，看清瓶颈在哪。

    .venv/bin/python tests/bench_latency.py            # 分段耗时
    .venv/bin/python tests/bench_latency.py --sweep     # 输入尺寸 × 复杂度的推理耗时

只测「服务端这一侧」的真实开销：推理 / 关键点提取 / 坐标还原 / 语义分析 / 绘制。
摄像头采集与浏览器编码那一侧量不到，所以结论里会单独说明。
"""
import argparse
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "webui"))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from LandmarkEngine import LandmarkEngine
from Utils import drawLandmarks, extractLandmarks
from face_body import FaceBodyAnalyzer
from pipeline import _unmirror

ALL = ("face", "body", "leftHand", "rightHand")


def load_frames(path, n):
    cap = cv2.VideoCapture(path)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = cap.read()
            if not ok:
                break
        out.append(f)
    cap.release()
    return out


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def breakdown(args):
    frames = load_frames(args.video, args.frames)
    if not frames:
        print("读不到视频帧"); return
    print(f"素材 {os.path.basename(args.video)}：{len(frames)} 帧 "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}")
    print(f"复杂度 {args.complexity}\n")

    tInfer, tExtract, tUnmirror, tFeat, tDraw, tMirror, tTotal = \
        [], [], [], [], [], [], []
    analyzer = FaceBodyAnalyzer()
    with LandmarkEngine(model_complexity=args.complexity, need=ALL) as eng:
        for frame in frames:
            t0 = time.perf_counter()
            # 与 pipeline 保持一致：cv2.flip（numpy 的 f[:, ::-1] 要慢 10 倍）
            t = time.perf_counter()
            mir = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(mir, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            tMirror.append((time.perf_counter() - t) * 1e3)

            t = time.perf_counter()
            res = eng.process(rgb)
            tInfer.append((time.perf_counter() - t) * 1e3)

            t = time.perf_counter()
            lm = extractLandmarks(res, ALL)
            tExtract.append((time.perf_counter() - t) * 1e3)

            t = time.perf_counter()
            lm = _unmirror(lm)
            tUnmirror.append((time.perf_counter() - t) * 1e3)

            t = time.perf_counter()
            analyzer.update(lm.get("face"), lm.get("body"))
            tFeat.append((time.perf_counter() - t) * 1e3)

            t = time.perf_counter()
            drawLandmarks(mir, res)
            tDraw.append((time.perf_counter() - t) * 1e3)

            tTotal.append((time.perf_counter() - t0) * 1e3)

    rows = [("镜像+转 RGB", tMirror, True), ("推理 mps", tInfer, True),
            ("提取关键点", tExtract, True), ("坐标还原", tUnmirror, True),
            ("语义分析", tFeat, True), ("服务端绘制", tDraw, True)]
    print(f"{'阶段':<14}{'中位':>9}{'p90':>9}{'最小':>9}{'最大':>9}")
    print("-" * 52)
    for name, xs, _ in rows:
        print(f"{name:<14}{statistics.median(xs):>8.2f}m{pct(xs,0.9):>8.2f}m"
              f"{min(xs):>8.2f}m{max(xs):>8.2f}m")

    def med(xs):
        return statistics.median(xs)
    # WebUI 默认不画骨架（浏览器自己画），所以「实际每帧」要把绘制扣掉
    default = med(tTotal) - med(tDraw)
    print(f"\n单帧合计（带服务端绘制）{med(tTotal):.2f} ms -> {1000/med(tTotal):.1f} FPS")
    print(f"单帧合计（WebUI 默认，不画）{default:.2f} ms -> {1000/default:.1f} FPS  "
          f"← 省掉 {med(tDraw):.2f} ms")
    print(f"\n其中镜像那一步如果用 numpy 的 f[:, ::-1] 会多花 ~3.5ms"
          f"（负步长视图逐元素拷贝）")

    # 每帧都要序列化成 JSON 发给前端，这个开销也别忽略
    import json
    payload = {"face": lm.get("face"), "body": lm.get("body"), "hands": lm.get("hands")}
    t = time.perf_counter()
    s = json.dumps(payload)
    dt = (time.perf_counter() - t) * 1e3
    print(f"状态 JSON 序列化 {dt:.2f} ms（{len(s)/1024:.1f} KB，含 468 脸点）")
    print(f"  · 不含脸点时 ", end="")
    payload2 = {"body": lm.get("body"), "hands": lm.get("hands")}
    t = time.perf_counter()
    s2 = json.dumps(payload2)
    dt2 = (time.perf_counter() - t) * 1e3
    print(f"{dt2:.2f} ms（{len(s2)/1024:.1f} KB）")


def sweep(args):
    """输入尺寸 × 复杂度：推理耗时与「三组都检出」的比例。"""
    frames = load_frames(args.video, args.frames)
    if not frames:
        print("读不到视频帧"); return
    sizes = [(1280, 720), (960, 540), (640, 480), (480, 360)]
    print(f"素材 {len(frames)} 帧；每格测 {len(frames)} 次推理\n")
    print(f"{'输入尺寸':<12}{'复杂度':>7}{'推理中位':>11}{'p90':>9}{'脸':>7}{'身体':>7}{'手':>7}")
    print("-" * 62)
    for (w, h) in sizes:
        for c in (1, 0):
            small = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]
            ts, nf, nb, nh = [], 0, 0, 0
            with LandmarkEngine(model_complexity=c, need=ALL) as eng:
                for f in small:
                    mir = np.ascontiguousarray(f[:, ::-1, :])
                    rgb = cv2.cvtColor(mir, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    t = time.perf_counter()
                    res = eng.process(rgb)
                    ts.append((time.perf_counter() - t) * 1e3)
                    lm = extractLandmarks(res, ALL)
                    nf += bool(lm["face"]); nb += bool(lm["body"])
                    nh += bool(lm["leftHand"] or lm["rightHand"])
            n = len(ts)
            print(f"{w}x{h}".ljust(12) + f"{c:>7}{statistics.median(ts):>10.2f}m"
                  f"{pct(ts,0.9):>8.2f}m{nf*100//n:>6}%{nb*100//n:>6}%{nh*100//n:>6}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(ROOT, "tests/fixtures/hands_gestures.mp4"))
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--complexity", type=int, default=1)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    (sweep if args.sweep else breakdown)(args)


if __name__ == "__main__":
    main()
