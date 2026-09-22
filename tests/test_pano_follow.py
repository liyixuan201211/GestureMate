#!/usr/bin/env python3
"""「全景里跟随人物」的回归测试 —— 用**已知答案**验证，而不是靠眼睛看。

    .venv/bin/python tests/test_pano_follow.py

为什么需要一个"造数据"的环节：跟随准不准，如果只拿真实全景肉眼看，是没有
ground truth 的（你不知道人在第几度）。所以这里反过来做 —— 把一张**确实能被
检测出人**的普通透视照片，用 `paint_perspective_into_equirect()` 贴进全景的
**指定方位**，再断言跟随能不能把那个方位找回来。

这同时也验证了需求里那句"我可能出现在全景的任意位置"：测试会贴在
+70 / -110 / 0 / 155 这几个方位上，包括正后方。
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "webui"))

import cv2
import numpy as np

from pano.projection import (equirect_to_perspective, pano_yaw_pitch,
                             paint_perspective_into_equirect, view_point_to_pano)
from pano.follow import PanoFollower, _circular_smooth

#: 合成全景的尺寸。**别调小**：这个尺寸决定了"贴进去的人"在抽出视角里有多少真实像素。
#: 1024x512 时，fov 70° 的一屏只覆盖 1024*70/360 ≈ 199px，而跟随要抽出 512px 的视角
#: —— 等于把脸放大 2.6 倍，人脸网格检测就卡在临界点上（插值差 1 个灰度值就能翻转
#: 检不检出）。2048x1024 时同样一屏覆盖 ≈398px，放大 1.3 倍，测的才是"抽出的视角
#: 能不能跑动作检测"，而不是"mediapipe 能不能认出一张 2.6 倍糊掉的脸"。
ERP_W, ERP_H = 2048, 1024
ALL = ("face", "body", "leftHand", "rightHand")


def _source_person():
    """从 fixture 视频里挑一帧**确实能检出人**的普通透视图。"""
    import mediapipe.python.solutions as sol
    path = os.path.join(ROOT, "tests/fixtures/hands_gestures.mp4")
    cap = cv2.VideoCapture(path)
    best, best_v = None, 0.0
    with sol.pose.Pose(static_image_mode=True, model_complexity=0) as p:
        for _ in range(60):
            ok, f = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            r = p.process(rgb)
            if not r.pose_landmarks:
                continue
            vis = [l.visibility for l in r.pose_landmarks.landmark]
            torso = float(np.mean([vis[i] for i in (11, 12, 23, 24)]))
            if torso > best_v:
                best, best_v = f, torso
            if torso > 0.9:
                break
    cap.release()
    return best, best_v


def test_projection_roundtrip():
    """正反投影必须一致：视角里的点 -> 全景角度 -> 再投影回应在同一点。"""
    w, h, fov, yaw0, pitch0 = 512, 512, 70.0, 33.0, -12.0
    for (px, py) in [(0, 0), (256, 256), (511, 511), (100, 400), (400, 80)]:
        y, p = view_point_to_pano(px, py, w, h, fov, yaw0, pitch0)
        y = float(np.asarray(y).reshape(-1)[0])
        p = float(np.asarray(p).reshape(-1)[0])
        # 反过来：给定角度，该方向在视角里的像素位置
        f = (w * 0.5) / math.tan(math.radians(fov) / 2)
        from pano.projection import view_basis
        fwd, rgt, up = view_basis(yaw0, pitch0)
        d = np.array([math.sin(math.radians(y)) * math.cos(math.radians(p)),
                      math.sin(math.radians(p)),
                      math.cos(math.radians(y)) * math.cos(math.radians(p))])
        z = float(d @ fwd)
        x = float(d @ rgt) / z
        yy = float(d @ up) / z
        bx, by = x * f + w * 0.5, -(yy) * f + h * 0.5
        if abs(bx - px) > 1e-6 or abs(by - py) > 1e-6:
            raise AssertionError(f"往返不一致 ({px},{py}) -> ({bx:.6f},{by:.6f})")
    print("PASS 1: 视角<->全景 投影往返一致（5 个点，误差 < 1e-6）")


def test_pano_pixel_angles():
    """全景像素 -> 角度的约定：中心的 yaw 是 0°，左右边缘是 ∓180°，上方是 +90°。

    注意中心落在**半个像素**上：x = (lon/2pi+0.5)*(w-1)，lon=0 -> (w-1)/2 = 511.5。
    整数像素 512 其实偏了 0.176°，第一版测试就是在这里假失败。
    """
    y, _ = pano_yaw_pitch([0, (ERP_W - 1) / 2.0, ERP_W - 1],
                          [(ERP_H - 1) / 2.0] * 3, ERP_W, ERP_H)
    assert abs(y[0] + 180) < 1e-9, f"左边缘应为 -180°，实得 {y[0]}"
    assert abs(y[1]) < 1e-9, f"中心应为 0°，实得 {y[1]}"
    assert abs(y[2] - 180) < 1e-9, f"右边缘应为 +180°，实得 {y[2]}"
    _, pt = pano_yaw_pitch([(ERP_W - 1) / 2.0] * 2, [0, ERP_H - 1], ERP_W, ERP_H)
    assert abs(pt[0] - 90) < 1e-9 and abs(pt[1] + 90) < 1e-9
    print("PASS 2: 全景像素->角度 约定正确（中心 0°、左 -180°、上 +90°）")


def test_circular_smooth():
    """角度平滑必须走圆周：359° 和 1° 之间不能平均出 180°。"""
    got = _circular_smooth(359.0, 1.0, 0.5)
    assert min(abs(got - 0.0), abs(abs(got) - 360.0)) < 1e-6, f"跨 ±180 平滑出错：{got}"
    mid = _circular_smooth(0.0, 90.0, 0.5)
    assert abs(mid - 45.0) < 1e-6, mid
    print("PASS 3: 角度按圆周平滑（359°/1° -> 0°，0°/90° -> 45°）")


def test_follow_at_known_yaws():
    """核心：把人贴在指定方位，看能不能找回来 —— 含正后方。"""
    person, torso = _source_person()
    assert person is not None, "fixture 里找不到可检出的帧"
    print(f"      源图：{person.shape[1]}x{person.shape[0]}，躯干可见度 {torso:.2f}")

    base = np.full((ERP_H, ERP_W, 3), 96, np.uint8)     # 素色底，别干扰检测
    erro = []
    for want_yaw in (0.0, 70.0, -110.0, 155.0):
        erp = paint_perspective_into_equirect(base.copy(), person,
                                              yaw_deg=want_yaw, pitch_deg=0.0,
                                              fov_deg=70.0)
        with PanoFollower(sweep_n=12) as fol:
            ok = fol.acquire(erp)
            got = fol.fix.yaw
        if not ok:
            raise AssertionError(f"方位 {want_yaw:+.0f}° 没找到人")
        # 方位是圆周量，误差也要按圆周算
        d = abs((got - want_yaw + 180) % 360 - 180)
        erro.append(d)
        print(f"      贴 {want_yaw:+7.1f}° -> 找到 {got:+7.1f}°  误差 {d:5.1f}°  "
              f"score={fol.fix.score:.2f}")
        assert d < 25.0, f"方位误差过大：{want_yaw} -> {got}（差 {d:.1f}°）"
    print(f"PASS 4: 任意方位都能找到（4 个方位，最大误差 {max(erro):.1f}°）")


def test_no_false_positive():
    """空全景（没有人）时必须老实说没找到，不能瞎锁一个方位。"""
    base = np.full((ERP_H, ERP_W, 3), 96, np.uint8)
    with PanoFollower(sweep_n=12) as fol:
        ok = fol.acquire(base)
    assert not ok, f"空场景不该锁定，却锁在 yaw={fol.fix.yaw}"
    assert not fol.fix.locked
    print("PASS 5: 空场景不误锁（没有人就说没有）")


def test_detect_on_extracted_view():
    """第二步：抽出来的透视画面上能跑通 GestureMate 的动作检测。"""
    person, _ = _source_person()
    base = np.full((ERP_H, ERP_W, 3), 96, np.uint8)
    erp = paint_perspective_into_equirect(base, person, yaw_deg=70.0, fov_deg=70.0)
    with PanoFollower(sweep_n=12) as fol:
        r = fol.step(erp, detect=True)
    assert r.found, "没锁定"
    assert r.view is not None and r.view.shape[2] == 3
    fy = r.features or {}
    assert "eyes" in fy and "body" in fy, f"特征不完整：{sorted(fy)}"
    body = fy.get("body") or {}
    print(f"      yaw={r.fix.yaw:+.1f}° 视角 {r.view.shape[1]}x{r.view.shape[0]}，"
          f"关键点 {fy.get('landmarkCount')}，举手={body.get('handsUp')}，"
          f"定位 {r.t_locate:.0f}ms / 检测 {r.t_detect:.0f}ms")
    assert (fy.get("landmarkCount") or 0) > 0, "抽出来的画面上没检测到关键点"
    print("PASS 6: 在抽出的透视视角上跑通了动作检测（脸+身+手 + 语义特征）")


if __name__ == "__main__":
    test_projection_roundtrip()
    test_pano_pixel_angles()
    test_circular_smooth()
    test_follow_at_known_yaws()
    test_no_false_positive()
    test_detect_on_extracted_view()
    print("\n全部通过。")
