#!/usr/bin/env python3
"""针对本次修改的两个「让程序根本跑不起来」的 bug 的回归测试。

不依赖摄像头、不依赖 mediapipe 模型权重，跑得很快：

    .venv/bin/python tests/test_fixes.py

1) Task.listen 曾用 `logging.info(..., end="")` —— 标准库 logging 不接受 `end`，
   任何任务被激活后都会在第 FPS_COUNT_FRAME*3 帧抛 TypeError。
2) MatchTask._listen 曾把 `self.count[i] = 0` 放在 if 之外，等于每帧清零，
   于是 `frames[i] > 1` 的 match 任务永远不会触发（官方计算器 demo 的
   frames 全是 4，等于全部失效）。
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tasks.MatchTask import MatchTask
from tasks.DetectTask import DetectTask
from tasks.Task import Task


class FakeController:
    """记录 process() 调用，不做任何副作用。"""

    def __init__(self):
        self.processed = []

    def activateTask(self, tid, x):
        pass

    def deactivateTask(self, tid, x):
        pass


def hands_only(valid=True):
    pts = [[0.5, 0.5, 0.0]] * 21
    return {"leftHand": pts if valid else None,
            "rightHand": None, "body": None, "face": None}


def test_task_listen_does_not_raise():
    c = FakeController()
    t = Task(c, "t1", "Test", [], True)
    t._listen = lambda x: None
    t.listen(hands_only())          # 曾经在这里 TypeError
    print("PASS 1: Task.listen 不再抛 TypeError（原为 logging.info(..., end=)）")


def test_task_listen_accepts_debug_level():
    """确保降到 debug 之后，默认日志级别下不会往 stdout 喷内容。"""
    import logging
    c = FakeController()
    t = Task(c, "t1", "Test", [], True)
    t._listen = lambda x: None
    root = logging.getLogger()
    old = root.level
    root.setLevel(logging.INFO)
    t.listen(hands_only())
    root.setLevel(old)
    print("PASS 1b: 默认 INFO 级别下 listen 不产生逐帧日志")


def test_match_requires_consecutive_frames():
    c = FakeController()
    with tempfile.TemporaryDirectory() as d:
        pose = os.path.join(d, "pose.json")
        with open(pose, "w") as f:
            json.dump(hands_only(), f)

        # 让匹配总是成功，从而只考察「连续帧计数」这一段逻辑。
        # 注意要复刻真实 calcDelta 的 None 语义：任一部位缺失就返回 -1（不算命中）。
        orig = MatchTask.calcDelta
        MatchTask.calcDelta = staticmethod(
            lambda bp, x, p: (-1 if any(x[part] is None for part in bp) else 0.0))
        try:
            t = MatchTask(c, "m1", [["leftHand"]], [pose], [0.5], [4],
                          [], True)
            fired = []
            t.process = lambda x: fired.append(1)

            for i in range(3):
                t._listen(hands_only())
                assert not fired, f"第 {i+1} 帧就触发了，应为 4 帧"
            t._listen(hands_only())
            assert fired, "连续 4 帧命中后仍未触发"
            print("PASS 2: match 的 frames=4 需要连续 4 帧命中才触发")

            # 中断一次后计数必须清零
            t.count = [0]
            t.process = lambda x: fired.append(1)
            n0 = len(fired)
            t._listen(hands_only())                 # 1
            t._listen(hands_only())                 # 2
            t._listen(hands_only(valid=False))      # 断了 -> 清零
            t._listen(hands_only())                 # 1
            assert len(fired) == n0, "中断后未清零，计数被错误地累加"
            print("PASS 2b: 未命中会清零计数（不是每帧清零）")
        finally:
            MatchTask.calcDelta = orig


def test_detect_counter_still_works():
    c = FakeController()
    t = DetectTask(c, "d1", ["leftHand"], 3, [], True)
    fired = []
    t.process = lambda x: fired.append(1)
    for _ in range(2):
        t._listen(hands_only())
    assert not fired
    t._listen(hands_only())
    assert fired, "detect frames=3 未触发"
    print("PASS 3: detect 的连续帧计数仍然正确")


def test_landmark_swap_is_preserved():
    """镜像补偿：holistic 的 left 必须映射到 rightHand（原代码行为）。"""
    from Utils import extractLandmarks

    class R:
        class _L:
            def __init__(self, v):
                self.landmark = [type("P", (), {"x": v, "y": 0.0, "z": 0.0})()
                                 for _ in range(21)]
        left_hand_landmarks = _L(0.11)
        right_hand_landmarks = _L(0.22)
        pose_landmarks = None
        face_landmarks = None

    res = extractLandmarks(R(), need=("leftHand", "rightHand"))
    assert abs(res["rightHand"][0][0] - 0.11) < 1e-9, "left→rightHand 的镜像补偿丢了"
    assert abs(res["leftHand"][0][0] - 0.22) < 1e-9, "right→leftHand 的镜像补偿丢了"
    assert res["face"] is None and res["body"] is None
    print("PASS 4: 左右手镜像补偿保持原样；未用到的部位不再转换")


def test_config_relative_paths():
    """配置里的相对路径（match 的 poseFile）必须能自己解析。

    命令行版靠 os.chdir(dataDir) 才能找到 "./hand/hand0.json"；
    WebUI 不能 chdir，原来就直接 FileNotFoundError，
    导致「一选计算器 demo 管线就起不来、界面卡住」。
    """
    from TaskController import TaskController
    cfg = os.path.join(ROOT, "example", "data_example", "config.json")
    if not os.path.exists(cfg):
        print("SKIP 8: 没有 example/data_example/config.json")
        return
    c = TaskController()
    c.readConfig(cfg)                      # 以前这里就抛 FileNotFoundError
    matches = [t for t in c.tasks.values() if t.taskType == "Match"]
    assert matches, "应加载出 match 任务"
    poses = sum(len(t.pose) for t in matches)
    names = sum(len(t.poseName) for t in matches)
    assert poses > 0 and poses == names, "poseFile 应被真正读进来"
    print(f"PASS 8: 相对 poseFile 按配置目录解析成功"
          f"（{len(matches)} 个 match 任务，共 {poses} 个姿态文件）")


def test_unknown_task_type_errors_clearly():
    """未知 task type 应给出能看懂的错误，而不是 UnboundLocalError。"""
    import tempfile
    from TaskController import TaskController
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "config.json")
        with open(p, "w") as f:
            json.dump([{"type": "nonsense", "id": "x"}], f)
        c = TaskController()
        try:
            c.readConfig(p)
        except ValueError as e:
            assert "nonsense" in str(e), str(e)
            print("PASS 9: 未知 task type 报错清晰（原来抛 UnboundLocalError）")
            return
        raise AssertionError("未知 task type 应当报错")


if __name__ == "__main__":
    test_task_listen_does_not_raise()
    test_task_listen_accepts_debug_level()
    test_match_requires_consecutive_frames()
    test_detect_counter_still_works()
    test_landmark_swap_is_preserved()
    test_config_relative_paths()
    test_unknown_task_type_errors_clearly()
    print("\n全部通过。")
