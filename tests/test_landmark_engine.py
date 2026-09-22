#!/usr/bin/env python3
"""`LandmarkTaskEngine` 的回归测试 —— 浏览器推理模式的服务端那一半。

    .venv/bin/python tests/test_landmark_engine.py

浏览器推理模式下，服务端不再跑 mps，只做两件事：语义分析 + 任务系统。
这个文件用**真实检出**的关键点（拿 Python 管线跑 fixture 视频得到）灌进引擎，
验证整条链：关键点 -> face_body 语义 -> TaskController 任务 -> 事件。

顺带覆盖 `parse_landmarks` 这个「半个信任边界」的输入校验。
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "webui"))
sys.path.insert(0, ROOT)

import cv2

from LandmarkEngine import LandmarkEngine
from Utils import extractLandmarks
from pipeline import LandmarkTaskEngine, parse_landmarks
from pipeline import _unmirror

ALL = ("face", "body", "leftHand", "rightHand")


def real_landmarks(n=40):
    """用真实管线在 fixture 视频上取 n 帧关键点（服务端推理那条路，用于造数据）。"""
    path = os.path.join(ROOT, "tests/fixtures/hands_gestures.mp4")
    cap = cv2.VideoCapture(path)
    out = []
    with LandmarkEngine(model_complexity=1, need=ALL) as eng:
        while len(out) < n:
            ok, frame = cap.read()
            if not ok:
                break
            mir = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(mir, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            out.append(_unmirror(extractLandmarks(eng.process(rgb), ALL)))
    cap.release()
    return out


def test_parse_landmarks():
    """外部输入必须被规整：截断、丢非法、夹取范围、挡 NaN。"""
    assert parse_landmarks({}) == {"face": None, "body": None,
                                  "leftHand": None, "rightHand": None}

    r = parse_landmarks({"face": [[0.1, 0.2, 0.3]] * 999, "body": [[0.5, 0.5]] * 50})
    assert len(r["face"]) == 478, "脸点数应被截到上限"
    assert len(r["body"]) == 33, "身体点数应被截到上限"

    r = parse_landmarks({"leftHand": [[float("nan"), 0.1], [0.5, float("inf")],
                                      [0.3, 0.4, 0.5]]})
    assert r["leftHand"] == [[0.3, 0.4, 0.5]], f"NaN/Inf 应被丢掉：{r['leftHand']}"

    r = parse_landmarks({"rightHand": [[-5.0, 9.0, 0.1]]})
    assert r["rightHand"] == [[0.0, 1.0, 0.1]], "越界坐标应被夹到 [0,1]"

    r = parse_landmarks({"body": ["x", [1], [0.1, 0.2], None, {}]})
    assert r["body"] == [[0.1, 0.2, 0.0]], f"垃圾项应被丢掉：{r['body']}"

    print("PASS 1: parse_landmarks 截断/丢非法/夹范围/挡 NaN 全部正确")


def test_engine_with_real_landmarks():
    """真检出灌进引擎：要出语义特征，也要能触发任务事件。"""
    frames = real_landmarks(40)
    assert frames, "取不到 fixture 关键点"
    # 引擎的 state 只保留**最新一帧**，而 fixture 末尾几帧人是空手的，
    # 所以要断言「最终状态有手」，就得喂有手的那几帧（否则测的是「最后一帧空手」）
    hands_frames = [f for f in frames if f["leftHand"] or f["rightHand"]]
    assert hands_frames, "fixture 里应当能检出手"
    print(f"      （fixture {len(frames)} 帧里有 {len(hands_frames)} 帧检出到手）")
    frames = hands_frames

    eng = LandmarkTaskEngine(data_dir=os.path.join(ROOT, "tests/_e2e_data"))
    assert eng.configError is None, eng.configError
    assert eng.controller.tasks, "应当加载出任务"
    eng.start()
    try:
        for f in frames:
            eng.submit(dict(f))
            time.sleep(0.03)          # 模拟浏览器的 30Hz 上行
        time.sleep(0.6)
    finally:
        eng.stop()
        eng.join(timeout=3)

    snap = eng.snapshot()
    assert snap["frames"] > 0, f"引擎没有处理任何关键点：{snap['frames']}"
    assert snap["hands"]["leftHand"] or snap["hands"]["rightHand"], "手部关键点没被保存"
    assert snap["body"], "身体关键点没被保存"

    feats = snap["features"] or {}
    for k in ("eyes", "mouth", "head", "body", "blinkCount"):
        assert k in feats, f"语义特征缺少 {k}：{sorted(feats)}"

    # 服务端只做语义 + 任务，单帧开销应当是「亚毫秒」级（这是这个模式的意义）
    assert snap["loopMs"] < 5.0, f"单帧开销异常：{snap['loopMs']}ms"

    ev = eng.drainEvents(0)
    print(f"PASS 2: 真实关键点跑通整条链 "
          f"（{len(frames)} 帧 -> features={sorted(feats.keys())}，"
          f"单帧 {snap['loopMs']}ms，事件 {len(ev)} 条）")
    return snap


def test_engine_without_config():
    """没配置也要能跑：只做识别，不跑任务。"""
    eng = LandmarkTaskEngine(data_dir=None)
    assert eng.controller.tasks == {}
    assert sorted(eng.need) == sorted(ALL)
    eng.start()
    eng.submit({"leftHand": [[0.1, 0.2, 0.3]] * 21, "body": [[0.5, 0.5, 0.0]] * 33})
    time.sleep(0.4)
    eng.stop()
    eng.join(timeout=3)
    snap = eng.snapshot()
    assert snap["frames"] == 1, snap["frames"]
    print("PASS 3: 无配置时只做识别、不跑任务，且不报错")


def test_bad_config_does_not_kill_recognition():
    """配置坏了：识别必须照旧，只把错误记下来给界面显示。"""
    import tempfile
    import json
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump([{"type": "match", "id": "m", "bodyPart": [["leftHand"]],
                        "poseFile": ["./nope/missing.json"], "sensetive": [0.5],
                        "frames": [1], "start": True}], f)
        eng = LandmarkTaskEngine(data_dir=d)
        assert eng.configError and "missing.json" in eng.configError, eng.configError
        eng.start()
        eng.submit({"leftHand": [[0.1, 0.2, 0.3]] * 21})
        time.sleep(0.4)
        eng.stop()
        eng.join(timeout=3)
        assert eng.snapshot()["frames"] == 1, "配置坏了也不该影响识别"
    print("PASS 4: 配置加载失败时识别照常，错误留给界面显示")


if __name__ == "__main__":
    test_parse_landmarks()
    snap = test_engine_with_real_landmarks()
    test_engine_without_config()
    test_bad_config_does_not_kill_recognition()
    print("\n全部通过。")
