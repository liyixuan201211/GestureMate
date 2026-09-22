#!/usr/bin/env python3
"""面部/身体语义分析的回归测试（纯合成关键点，不需要模型权重）。

    .venv/bin/python tests/test_face_body.py

这里专门盯住几个「看图看不出来、错了很难发现」的地方：
  * 左右眼是按**交付坐标的 x 顺序**判定的，不是按 MediaPipe 的编号
  * roll 必须归一化到 (-90, 90]，不能出现 ±180
  * 眨眼要用**原始** EAR 判定（平滑值会把单帧眨眼抹掉）
  * yaw 的符号约定：>0 = 用户把头转向自己的右边
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "webui"))
sys.path.insert(0, ROOT)

from face_body import (EYE_GROUP_A, EYE_GROUP_B, FaceBodyAnalyzer, _normalize_pm90,
                       eye_aspect_ratio, eye_groups, head_pose, mouth_aspect_ratio)

N = 478


def blank_face():
    return [[0.5, 0.5, 0.0] for _ in range(N)]


def set_eye(face, group, cx, cy, openness=1.0):
    """在 group 的 6 个点上摆一个眼睛：p1/p4 是左右眼角，p2/p3/p6/p5 是上下眼睑。

    openness=1.0 -> 睁眼（睑距 = 0.6 * 眼宽）；0.0 -> 完全闭合。
    """
    p1, p2, p3, p4, p5, p6 = group
    half = 0.02                       # 眼角半宽
    lid = 0.6 * half * openness       # 眼睑张开距离
    face[p1] = [cx - half, cy, 0.0]
    face[p4] = [cx + half, cy, 0.0]
    face[p2] = [cx - half / 2, cy - lid, 0.0]
    face[p3] = [cx + half / 2, cy - lid, 0.0]
    face[p6] = [cx - half / 2, cy + lid, 0.0]
    face[p5] = [cx + half / 2, cy + lid, 0.0]


def make_face(right_cx=0.45, left_cx=0.55, cy=0.40, openness=1.0,
              nose=(0.50, 0.55), roll_deg=0.0, mouth_open=0.02):
    """构造一张「正脸」，并可按需加滚转 / 张开嘴。

    约定：right_cx < left_cx 表示右眼在画面左侧（也就是正常的正脸）。
    """
    face = blank_face()
    set_eye(face, EYE_GROUP_A, right_cx, cy, openness)
    set_eye(face, EYE_GROUP_B, left_cx, cy, openness)
    face[1] = [nose[0], nose[1], 0.0]                       # 鼻尖
    face[10] = [0.50, 0.30, 0.0]                            # 额头
    face[152] = [0.50, 0.70, 0.0]                           # 下巴
    # 嘴
    w = 0.06
    face[61] = [0.50 - w, 0.62, 0.0]                        # 左嘴角
    face[291] = [0.50 + w, 0.62, 0.0]                       # 右嘴角
    face[13] = [0.50, 0.62 - mouth_open / 2, 0.0]           # 上唇内缘
    face[14] = [0.50, 0.62 + mouth_open / 2, 0.0]           # 下唇内缘
    face[0] = [0.50, 0.62 - mouth_open, 0.0]
    face[17] = [0.50, 0.62 + mouth_open, 0.0]

    if roll_deg:
        # 绕眼中心整体旋转（只转 x/y）
        cx, cyy = (right_cx + left_cx) / 2, cy
        th = math.radians(roll_deg)
        c, s = math.cos(th), math.sin(th)
        for p in face:
            dx, dy = p[0] - cx, p[1] - cyy
            p[0] = cx + dx * c - dy * s
            p[1] = cyy + dx * s + dy * c
    return face


def test_eye_groups_by_x():
    # 正常正脸：A 组在画面左 -> A 是右眼
    f = make_face(right_cx=0.45, left_cx=0.55)
    r, l = eye_groups(f)
    assert r == EYE_GROUP_A and l == EYE_GROUP_B, "正脸时 A 组应为右眼"

    # 把两组左右对调（模拟 MediaPipe 手性反过来）：仍应按 x 判定
    f2 = make_face(right_cx=0.55, left_cx=0.45)
    r2, l2 = eye_groups(f2)
    assert r2 == EYE_GROUP_B and l2 == EYE_GROUP_A, "A 组到画面右侧后应判为左眼"
    print("PASS 1: 左右眼按交付坐标 x 顺序判定，不依赖 MediaPipe 编号")


def test_roll_normalized():
    assert _normalize_pm90(180.0) == 0.0 or abs(_normalize_pm90(180.0)) == 0.0
    assert -90 < _normalize_pm90(-179.0) <= 90
    assert -90 < _normalize_pm90(179.0) <= 90

    level = head_pose(make_face())
    assert level is not None
    assert abs(level[2]) < 1.0, f"正脸 roll 应接近 0，实际 {level[2]}"

    # 右眼抬高 -> 眼睛连线倾斜
    f = make_face()
    _, _, r10 = head_pose(f)
    f2 = make_face()
    set_eye(f2, EYE_GROUP_A, 0.45, 0.35)     # 右眼（画面左）抬高
    _, _, r_up = head_pose(f2)
    assert abs(r_up) > 1.0, "明显倾斜时 roll 不应为 0"
    assert -90 < r_up <= 90, f"roll 必须落在 (-90,90]，实际 {r_up}"
    print(f"PASS 2: roll 归一化正常（正脸 {r10:.2f}°，倾斜 {r_up:.2f}°）")


def test_yaw_sign():
    """yaw 的符号约定：>0 表示「在**镜像显示**里脸朝右」。

    注意别搞反：交付坐标是相机原图，网页上视频被 CSS 镜像过
    （x_disp = 1 - x_delivered），所以
        交付里鼻子偏画面右(0.56)  ->  显示上看起来偏左  ->  yaw<0  ->  朝左
        交付里鼻子偏画面左(0.44)  ->  显示上看起来偏右  ->  yaw>0  ->  朝右
    这样标签永远和用户屏幕上看到的一致。
    """
    f = make_face(nose=(0.50, 0.55))
    y0 = head_pose(f)[0]
    assert abs(y0) < 1e-6, f"正脸 yaw 应为 0，实际 {y0}"

    f2 = make_face(nose=(0.56, 0.55))       # 交付里偏画面右 -> 显示偏左
    y_right_img = head_pose(f2)[0]
    assert y_right_img < 0, f"应 <0（显示朝左），实际 {y_right_img}"

    f3 = make_face(nose=(0.44, 0.55))       # 交付里偏画面左 -> 显示偏右
    y_left_img = head_pose(f3)[0]
    assert y_left_img > 0, f"应 >0（显示朝右），实际 {y_left_img}"

    # 端到端确认标签
    from face_body import _direction
    assert _direction(0.5, 1.0) == "朝右"
    assert _direction(-0.5, 1.0) == "朝左"
    assert _direction(0.0, 1.0) == "正对"
    print(f"PASS 3: yaw 符号约定正确（正脸 {y0:.3f}，交付右偏 {y_right_img:.3f}→朝左，"
          f"交付左偏 {y_left_img:.3f}→朝右）")


def test_blink_uses_raw_ear():
    """单帧闭眼必须能数出一次眨眼 —— 这正是「用平滑值判定」会漏掉的情况。"""
    an = FaceBodyAnalyzer(smoothing=0.4)
    open_f = make_face(openness=1.0)
    shut_f = make_face(openness=0.05)

    e_open = eye_aspect_ratio(open_f, EYE_GROUP_A)
    e_shut = eye_aspect_ratio(shut_f, EYE_GROUP_A)
    assert e_open > an.ear_open, f"睁眼 EAR 应 > ear_open，实际 {e_open}"
    assert e_shut < an.ear_closed, f"闭眼 EAR 应 < ear_closed，实际 {e_shut}"

    an.update(open_f, None)
    assert an.blinkCount == 0
    an.update(shut_f, None)                  # 单帧闭眼
    assert an.blinkCount == 0, "还没睁开，不该计数"
    an.update(open_f, None)                  # 睁开 -> 计一次
    assert an.blinkCount == 1, f"单帧眨眼应被数到，实际 {an.blinkCount}"

    # 连眨三次
    for _ in range(3):
        an.update(shut_f, None)
        an.update(open_f, None)
    assert an.blinkCount == 4, f"应为 4 次，实际 {an.blinkCount}"
    print(f"PASS 4: 单帧眨眼可检出（原始 EAR 睁 {e_open:.3f} / 闭 {e_shut:.3f}），"
          f"计数 {an.blinkCount}")


def test_mouth_states():
    an = FaceBodyAnalyzer()
    an.update(make_face(mouth_open=0.004), None)
    closed = an.update(make_face(mouth_open=0.004), None)["mouth"]["state"]
    assert closed == "闭合", f"实际 {closed}"

    an2 = FaceBodyAnalyzer()
    mid = an2.update(make_face(mouth_open=0.03), None)["mouth"]["state"]
    assert mid in ("微张", "张开"), f"实际 {mid}"

    an3 = FaceBodyAnalyzer()
    wide = an3.update(make_face(mouth_open=0.09), None)["mouth"]["state"]
    assert wide == "张开", f"实际 {wide}"
    print(f"PASS 5: 嘴巴状态分级正确（{closed} / {mid} / {wide}）")


def test_body_hands_up():
    an = FaceBodyAnalyzer()
    body = [[0.5, 0.5, 0.0] for _ in range(33)]
    body[11] = [0.4, 0.6, 0.0]      # 左肩
    body[12] = [0.6, 0.6, 0.0]      # 右肩
    body[15] = [0.4, 0.4, 0.0]      # 左手腕 高于肩 -> 举手
    body[16] = [0.6, 0.8, 0.0]      # 右手腕 低于肩 -> 没举
    out = an.update(None, body)["body"]
    assert out["handsUp"]["left"] is True, "左手腕高于肩应判为举手"
    assert out["handsUp"]["right"] is False, "右手腕低于肩不应判为举手"
    assert out["landmarkCount"] == 33
    print(f"PASS 6: 举手判定正确（左 {out['handsUp']['left']} / 右 {out['handsUp']['right']}，"
          f"肩线 {out['shoulderTiltDeg']}° {out['lean']}）")


def test_no_face_is_safe():
    an = FaceBodyAnalyzer()
    out = an.update(None, None, None)
    assert out["available"] is False
    assert "eyes" not in out and "mouth" not in out and "body" not in out
    print("PASS 7: 没有人脸/身体时不报错、不产出假数据")


if __name__ == "__main__":
    test_eye_groups_by_x()
    test_roll_normalized()
    test_yaw_sign()
    test_blink_uses_raw_ear()
    test_mouth_states()
    test_body_hands_up()
    test_no_face_is_safe()
    print("\n全部通过。")
