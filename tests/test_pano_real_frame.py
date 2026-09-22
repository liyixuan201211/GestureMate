#!/usr/bin/env python3
"""真机画面的回归测试 —— 用 **X5 实拍的等距柱状帧** 验证"跟随"。

    .venv/bin/python tests/test_pano_real_frame.py

为什么要有这个文件
------------------
`tests/test_pano_follow.py` 用的是"把一张透视图贴进素色全景"的**合成**数据，
它能证明"人在任意方位都能找回来"，但证明不了真机可用 —— 交接文档里也写着
「没验证到的：真人站在全景里的实时跟随」。

补验之后，真实 X5 画面暴露了合成数据**结构上测不出**的三类问题（都是真机才有的）：

1. 探针复用了有状态的视频跟踪模式检出器 -> 分数取决于"上一次探的是哪个方向"。
   合成数据每次只贴一个人、方位也干净，测不到这个。
2. 只扫地平线 -> 但 X5 平放在桌上时人是**俯身从上方入镜**的，两个人都在
   pitch +50°..+80°，地平线上根本没有躯干。
3. 俯仰被硬夹到 ±35° -> 就算找到了 +50° 的人也会被夹回地平线。

这个文件就是拿一张真实帧把这三条钉死。帧本身是 X5 Webcam 模式（2880x1440）
抓下来的，存在 `pano/demo/real_x5/0_X5实机原始帧_2880x1440.jpg`。

画面内容：相机平放在桌上，一位女士（正面、能检到脸）和一位俯身的男士入镜。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "webui"))

import cv2

from pano.follow import PanoFollower

FRAME = os.path.join(ROOT, "pano/demo/real_x5/0_X5实机原始帧_2880x1440.jpg")


def _frame():
    if not os.path.exists(FRAME):
        print(f"SKIP: 找不到真机帧 {FRAME}")
        print("      （它应该在仓库里；丢了就从 X5 的 /snapshot.jpg 再抓一张）")
        return None
    img = cv2.imread(FRAME)
    assert img is not None, f"读不出 {FRAME}"
    return img


def test_probe_is_stateless():
    """同一个视角、同一个输入，分数不能取决于"之前探过哪里"。

    这是真机上最隐蔽的一条：原来的 `_probe` 用一个
    `pose.Pose(static_image_mode=False)`（视频跟踪模式）去扫一堆**互不相关**的
    视角，于是跟踪状态会从上一次探的方向漏进来。实测同一张输入，作为第 1 次调用
    score=0.26、作为第 13 次调用 score=0.77 —— 依靠分数排序的"找人"因此不可靠。

    这个断言是确定性的：无状态检出器对同一输入必须给出一模一样的分数。
    """
    erp = _frame()
    if erp is None:
        return

    fol = PanoFollower()
    target = (120.0, 60.0)          # 一个真机里"有人"的方位附近
    first = fol._probe(erp, target[0], target[1], 224, 224, 90.0)

    # 拿一堆互不相关的视角把检出器"污染"一遍
    for yaw in (-180.0, -90.0, 0.0, 45.0, 150.0):
        for pitch in (-60.0, 0.0, 75.0):
            fol._probe(erp, yaw, pitch, 224, 224, 90.0)

    again = fol._probe(erp, target[0], target[1], 224, 224, 90.0)
    assert abs(first[0] - again[0]) < 1e-9, (
        f"同一输入两次探针分数不一致：{first[0]:.4f} vs {again[0]:.4f} —— "
        f"检出器又变成有状态的了")
    assert first[3] == again[3], f"关节数不一致：{first[3]} vs {again[3]}"

    # 别让这个测试变成"两边都是 0"的空断言
    scores = [fol._probe(erp, y, p, 224, 224, 90.0)[0]
              for y in (-180.0, -60.0, 60.0, 120.0) for p in (25.0, 60.0)]
    assert max(scores) > 0.0, "整张帧一个方位都没检出人，这个测试失去意义"
    print(f"PASS 1: 探针无状态（同一视角两次 {first[0]:.3f} == {again[0]:.3f}，"
          f"且帧内确有可检出方位 max={max(scores):.2f}）")


def test_real_frame_locks_and_detects():
    """真机帧上必须锁得住，并且能出脸 + 身体的动作检测结果。"""
    erp = _frame()
    if erp is None:
        return

    with PanoFollower() as fol:
        r = fol.step(erp, detect=True)

    f = r.fix
    assert f.locked, (
        f"真机帧没锁住（score={f.score:.2f}）—— 真机上是存在两个人的，"
        f"锁不住就是跟随坏了")
    assert f.score >= fol.min_score, f"锁定分数低于阈值：{f.score}"

    lm = r.landmarks or {}
    n_face = len(lm.get("face") or [])
    n_body = len(lm.get("body") or [])
    assert n_body > 0, "锁定了却检不到身体"
    assert n_face > 0, (
        "锁定了却检不到脸 —— 锁定的视角不是下游能用的视角"
        "（这曾经真的发生过：验证用的视角和实际锁的视角差 2.4°）")

    fy = r.features or {}
    assert fy.get("available"), f"语义特征没出来：{sorted(fy)}"
    assert "eyes" in fy and "body" in fy, f"特征不完整：{sorted(fy)}"

    print(f"PASS 2: 真机帧锁定 yaw={f.yaw:+.1f}° pitch={f.pitch:+.1f}° "
          f"fov={f.fov:.0f}° score={f.score:.2f}；脸 {n_face} 点 + 身体 {n_body} 点，"
          f"定位 {r.t_locate:.0f}ms / 检测 {r.t_detect:.0f}ms")


def test_people_are_off_horizon():
    """钉死"只扫地平线"这个错误：真机里人是**俯身入镜**的，不在 pitch=0。

    如果哪天有人把 sweep_pitches 改回只有地平线，这条会失败 —— 因为那时
    acquire 根本看不到这两个人。
    """
    erp = _frame()
    if erp is None:
        return

    with PanoFollower() as fol:
        ok = fol.acquire(erp)
    assert ok, "球面扫描没能找到人"
    assert abs(fol.fix.pitch) > 35.0, (
        f"真机里找到的人在 pitch={fol.fix.pitch:+.1f}°，本该远离地平线；"
        f"如果这里回到 0 附近，说明测试数据或视角约定变了")
    print(f"PASS 3: 找到的人在 pitch={fol.fix.pitch:+.1f}°（|pitch|>35°）—— "
          f"证实「只扫地平线」必然找不到他")


def test_tracking_holds_the_lock():
    """跟住之后应当继续保持锁定，而且比"重新找人"便宜得多。"""
    erp = _frame()
    if erp is None:
        return

    with PanoFollower() as fol:
        r1 = fol.step(erp, detect=True)
        assert r1.found, "第一帧就没锁住"
        r2 = fol.step(erp, detect=True)

    assert r2.found, "第二帧跟丢了"
    assert not r2.fix.acquired, "第二帧不该重新找人"
    assert r2.t_locate < r1.t_locate, (
        f"跟住的一帧（{r2.t_locate:.0f}ms）不该比重新找人（{r1.t_locate:.0f}ms）慢")
    assert (r2.features or {}).get("available"), "跟住的那一帧语义特征丢了"
    print(f"PASS 4: 跟住不掉锁，且常态开销 {r2.t_locate:.0f}ms << "
          f"重新找人 {r1.t_locate:.0f}ms（检测 {r2.t_detect:.0f}ms）")


if __name__ == "__main__":
    test_probe_is_stateless()
    test_real_frame_locks_and_detects()
    test_people_are_off_horizon()
    test_tracking_holds_the_lock()
    print("\n全部通过（真机帧）。")
