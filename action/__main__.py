"""动作识别的可跑入口：`python -m action`。

三种跑法
--------
    .venv/bin/python -m action --demo
        合成关键点，把四个动作各做一遍。**不用摄像头、不用模型权重**，
        用来确认"判定 → 事件 → 序列化"这条链路是通的。

    .venv/bin/python -m action --video clip.mp4
        用项目自己的 LandmarkEngine 真跑一段视频。这条是**真链路**：
        取流（含 TaskController 那一步水平镜像）→ MediaPipe → 特征 → 状态机 → 事件。

    .venv/bin/python -m action --video clip.mp4 --record lm.jsonl
    .venv/bin/python -m action --replay lm.jsonl
        把关键点录下来，之后离线回放。这正是 §5.9 要的回归方式
        （"录制真人关键点序列 → 离线回放 → 断言事件序列完全一致"），
        好处是阈值重调时不必反复对着摄像头做动作。

为什么没有"直接开摄像头"这一档
------------------------------
服务端进程在 macOS 上拿不到摄像头权限（TCC），`cv2.VideoCapture(0)` 会报
`not authorized to capture video`。取流必须在浏览器或 GUI App 里做 ——
这条结论写在 GestureMate 的交接文档 §6.1 与 BoxingMate 的 README 里。
所以命令行这一档只读**文件**；真机取流请走 WebUI，或先录成视频再喂进来。

⚠️ 左右方向（镜像）**必须真人验证一次**（§5.7）。真人往左移一下，
   屏幕上该报「左移」；报反了就加 `--no-mirror`，只有这一处开关。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):                      # 允许直接 python action/__main__.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action import (KIND_JUMP, KIND_LANE, KIND_PLAYER, KIND_READY, KIND_SQUAT,
                    ActionConfig, ActionDetector, describe_event)
from action.synth import demo_sequence


# ────────────────────────── 帧源 ──────────────────────────

def _to_list(lms):
    """MediaPipe 关键点列表 → [[x,y,z], ...]（None 原样返回）。"""
    if lms is None:
        return None
    return [[p.x, p.y, p.z] for p in lms.landmark]


def frames_demo(lead_ms=None):
    """合成序列。产出 (t_ms, body, world)。

    `lead_ms` = 开头"站直"的时长，由调用方按**标定窗口**给（见 main）。
    给短了就会标到动作上去 —— 那会被正确判成"人没站住"并拒绝标定，
    于是 demo 直接失败。这不是 bug，是那道门在正常工作。
    """
    for (t, body, _world) in demo_sequence(calib_ms=lead_ms or 3000.0):
        yield (t, body, None)


def frames_video(path, use_world=True, mirror_frame=True, complexity=1):
    """真链路：视频文件 → 项目自己的 LandmarkEngine → 关键点。"""
    import cv2
    from LandmarkEngine import LandmarkEngine

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1000.0 / fps if fps > 0 else 33.0
    eng = LandmarkEngine(model_complexity=complexity, need=("body",))
    print(f"视频源 {path}  {fps:.1f} fps  → {eng.describe()}"
          f"  world={'on' if use_world else 'off'}"
          f"  frame_mirror={'on' if mirror_frame else 'off'}")
    t = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if mirror_frame:
                # 与 TaskController.startListen 完全一致（自拍视角）。
                # 这就是 mirrored_input 默认 True 的原因。
                frame = frame[:, ::-1, :]
            r = eng.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            world = None
            if use_world:
                world = _to_list(getattr(r, "pose_world_landmarks", None))
            t += dt
            yield (t, _to_list(r.pose_landmarks), world)
    finally:
        cap.release()
        eng.close()


def frames_replay(path):
    """回放录制好的关键点 JSONL（每行 {"t":..,"body":..,"world":..}）。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield (float(row.get("t", 0.0)), row.get("body"), row.get("world"))


# ────────────────────────── 主流程 ──────────────────────────

def run(frames, args, title):
    cfg = ActionConfig(requireKnees=args.require_knees,
                       allowUnstableCalibration=args.allow_unstable_calib)
    det = ActionDetector(cfg, mirrored_input=args.mirror)

    print(f"\n=== {title} ===")
    print(f"镜像开关 mirrored_input={args.mirror}"
          f"{'（与自拍镜像管线一致）' if args.mirror else '（原始画面）'}"
          f"　require_knees={args.require_knees}")
    print(f"① 标定：前 {args.calib_ms:.0f}ms 请站直别动…")

    fh = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None
    record = open(args.record, "w", encoding="utf-8") if args.record else None
    det.begin_calibration()
    events, n, t_last = [], 0, 0.0
    calibrated = False
    try:
        for (t, body, world) in frames:
            n += 1
            t_last = t
            if record is not None:
                record.write(json.dumps({"t": t, "body": body, "world": world},
                                        ensure_ascii=False) + "\n")
            if not calibrated:
                det.update(t, body)                 # 标定期只累积基线
                if t >= args.calib_ms:
                    evs = det.finish_calibration(t)
                    if not evs:
                        print(f"   ✗ 标定失败：{det.block}")
                        return 1
                    b = det.basis
                    print(f"   标定完成：{b.samples} 帧　躯干 {b.torso:.4f}　"
                          f"抖动 {b.hip_spread:.4f}　roll {b.roll_deg:+.1f}°"
                          f"{'　⚠️ 基线不稳' if not b.stable else ''}")
                    calibrated = True
                    events += evs
                    _emit(evs, fh)
                    print("② 开始识别…（每个动作报一行）")
                continue
            evs = det.update(t, body, world)
            events += evs
            _emit(evs, fh)
    finally:
        if fh:
            fh.close()
        if record:
            record.close()

    if not calibrated:
        print(f"✗ 素材在 {args.calib_ms:.0f}ms 内就结束了，没能完成标定"
              f"（共 {n} 帧）")
        return 1

    # ── 汇总
    counts = {}
    for e in events:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    st = det.state
    print("\n── 汇总 ──")
    print(f"  帧数 {n}　平均帧间隔 {st['dtAvgMs']:.0f}ms"
          f"（≈{1000.0 / max(st['dtAvgMs'], 1e-6):.0f} fps）")
    for k in (KIND_LANE, KIND_JUMP, KIND_SQUAT, KIND_PLAYER, KIND_READY):
        if counts.get(k):
            print(f"  {k:<6} × {counts[k]}")
    if not counts.get(KIND_LANE) and not counts.get(KIND_JUMP) \
            and not counts.get(KIND_SQUAT):
        print("  （没有识别到任何动作）")
    if st["lowFps"]:
        print("  ⚠️ 帧率偏低（平均间隔 > 100ms）。这些判据是按「持续时间」设计的，"
              "帧率太低时会明显变钝 —— 30fps 以上最稳。")
    if st["kneeWarn"]:
        print("  ⚠️ requireKnees 开着，但只有 2D 膝角：正面机位下 2D 膝角几乎不变，"
              "这道门基本过不去。要么关掉它，要么喂世界坐标（--world）。")
    if st["block"]:
        print(f"  最后卡在：{st['block']}")
    if args.jsonl:
        print(f"  事件已写入 {args.jsonl}")
    if args.record:
        print(f"  关键点已写入 {args.record}（可用 --replay 回放）")
    return 0


def _emit(evs, fh):
    for e in evs:
        print(f"  [{e.t:8.0f} ms] {describe_event(e)}")
        if fh:
            fh.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m action",
        description="动作识别：左移 / 右移 / 跳 / 蹲（《向星而行》§5）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("三种跑法")[1] if "三种跑法" in __doc__ else None)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--demo", action="store_true",
                     help="跑合成序列（不用摄像头/模型，默认）")
    src.add_argument("--video", metavar="FILE", help="用 LandmarkEngine 真跑一段视频")
    src.add_argument("--replay", metavar="FILE", help="回放录制好的关键点 JSONL")

    p.add_argument("--calib-ms", type=float, default=2000.0,
                   help="标定阶段时长（毫秒），这段时间请站直 [默认 2000]")
    p.add_argument("--no-mirror", dest="mirror", action="store_false",
                   help="关掉镜像开关（真机上左右判反了才用，见 §5.7）")
    p.add_argument("--no-world", dest="world", action="store_false",
                   help="不喂世界坐标（膝角退化成 2D 投影角）")
    p.add_argument("--no-frame-mirror", dest="frame_mirror", action="store_false",
                   help="--video 时不把画面水平翻转（默认翻转，与 TaskController 一致）")
    p.add_argument("--require-knees", action="store_true",
                   help="蹲下必须两膝 <130°（§5.4a）。⚠️ 只有 2D 膝角时正面蹲会全漏，"
                        "建议配合世界坐标使用")
    p.add_argument("--allow-unstable-calib", action="store_true",
                   help="允许用「人没站住」的基线（仅调试；会明显更容易误触发）")
    p.add_argument("--complexity", type=int, default=1, choices=(0, 1, 2),
                   help="MediaPipe pose 模型复杂度 [默认 1]")
    p.add_argument("--jsonl", metavar="FILE", help="把事件流写成一文件一行 JSON")
    p.add_argument("--record", metavar="FILE",
                   help="（配 --video）把每帧关键点写成 JSONL，供 --replay 离线回放")
    p.set_defaults(mirror=True, world=True, frame_mirror=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.record and not args.video:
        print("--record 只能配 --video 用（要录的是真实关键点）")
        return 2

    if args.video:
        frames = frames_video(args.video, use_world=args.world,
                              mirror_frame=args.frame_mirror,
                              complexity=args.complexity)
        title = f"真链路：{args.video}"
    elif args.replay:
        frames = frames_replay(args.replay)
        title = f"离线回放：{args.replay}"
    else:
        # demo 开头"站直"要比标定窗口多 0.5s，保证标定期里人真的没动
        frames = frames_demo(args.calib_ms + 500.0)
        title = "合成序列（--demo）—— 只证明链路通，不证明真人准"
    return run(frames, args, title)


if __name__ == "__main__":
    sys.exit(main())
