"""命令行：把一个文件夹里的全景帧，跑成「跟着人 + 对该人做动作检测」的结果。

    # 用 x5-link README 里那批等距柱状帧
    .venv/bin/python -m pano --frames /Users/imac/南客松-v2/3dgs-lab/proj_erp/frames \
        --out /tmp/pano_out --limit 12

    # 也可以直接吃一个等距柱状 mp4
    .venv/bin/python -m pano --video pano.mp4 --out /tmp/pano_out

    # 只跟随、不做动作检测（快，用来调跟随即不稳）
    .venv/bin/python -m pano --frames DIR --out OUT --no-detect

输出：`OUT/pano_00000.png`（抽出来的透视视角 + 骨架标注）和 `OUT/tracks.json`
（每帧的 yaw/pitch/fov/score 与动作特征）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "webui"))

from pano.follow import PanoFollower, annotate   # noqa: E402


def iter_frames(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f
        cap.release()
        return
    pats = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG")
    files: list[str] = []
    for p in pats:
        files += glob.glob(os.path.join(args.frames, p))
    for path in sorted(files)[: args.limit or None]:
        img = cv2.imread(path)
        if img is not None:
            yield img


def main() -> int:
    ap = argparse.ArgumentParser(description="全景里跟随人物 + 动作检测")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", help="等距柱状图片所在文件夹")
    src.add_argument("--video", help="等距柱状视频")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少帧（0=全部）")
    ap.add_argument("--no-detect", action="store_true", help="只跟随，不做动作检测")
    ap.add_argument("--sweep", type=int, default=8, help="找人时绕一圈抽几个方位（yaw）")
    ap.add_argument("--pitches", type=float, nargs="+", default=None,
                    help="找人时采样的俯仰角（默认 -40 5 50，覆盖约 -85°..+95°）。"
                         "X5 平放在桌上时人是俯身入镜的，只扫地平线会找不到人")
    ap.add_argument("--pitch-limit", type=float, default=75.0,
                    help="锁定/跟随允许的最大俯仰角（超过 ~60° 透视视角开始病态）")
    ap.add_argument("--smooth", type=float, default=0.45, help="跟随平滑（大=跟得紧）")
    ap.add_argument("--fov", type=float, default=70.0, help="初始视场角")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                    help="动作检测用的 mps 复杂度")
    ap.add_argument("--save-all", action="store_true", help="每帧都存图（默认只存有人的）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tracks: list[dict] = []
    n = n_found = 0
    t0 = time.time()

    kw = dict(fov=args.fov, sweep_n=args.sweep, smooth=args.smooth,
              complexity=args.complexity, pitch_limit=args.pitch_limit)
    if args.pitches:
        kw["sweep_pitches"] = tuple(args.pitches)
    with PanoFollower(**kw) as fol:
        for i, erp in enumerate(iter_frames(args)):
            n += 1
            r = fol.step(erp, detect=not args.no_detect)
            f = r.fix
            n_found += int(f.locked)
            rec = {
                "frame": i, "locked": f.locked, "yaw": round(f.yaw, 2),
                "pitch": round(f.pitch, 2), "fov": round(f.fov, 1),
                "score": round(f.score, 3), "joints": f.n_joints,
                "acquired": f.acquired, "misses": f.misses,
                "t_locate_ms": round(r.t_locate, 2), "t_detect_ms": round(r.t_detect, 2),
            }
            if r.features:
                fy = r.features
                rec["features"] = {
                    "landmarks": fy.get("landmarkCount"),
                    "blinks": fy.get("blinkCount"),
                    "handsUp": (fy.get("body") or {}).get("handsUp"),
                }
            tracks.append(rec)
            print(f"  #{i:<4} {'锁定' if f.locked else '丢失'} "
                  f"yaw={f.yaw:+7.2f} pitch={f.pitch:+6.2f} fov={f.fov:5.1f} "
                  f"score={f.score:.2f} 定位={r.t_locate:.0f}ms 检测={r.t_detect:.0f}ms"
                  + (f"  {rec.get('features')}" if rec.get("features") else ""))

            if r.view is not None and (args.save_all or f.locked):
                img = annotate(r) if not args.no_detect else r.view
                cv2.imwrite(os.path.join(args.out, f"pano_{i:05d}.png"), img)

    with open(os.path.join(args.out, "tracks.json"), "w") as fh:
        json.dump(tracks, fh, ensure_ascii=False, indent=1)

    dt = time.time() - t0
    print(f"\n处理 {n} 帧，锁定 {n_found} 帧（{100*n_found/max(1,n):.0f}%），"
          f"共 {dt:.1f}s（{n/max(dt,1e-9):.1f} FPS）")
    print(f"结果写到 {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
