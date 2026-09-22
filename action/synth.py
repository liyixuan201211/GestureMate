"""合成关键点 —— **测试与 `--demo` 共用同一套几何**。

为什么要单独一个模块：本项目的 `README`/`交接文档` 反复记着同一条教训 ——
"测试替身跟真实接口不一致"是最难查的一类 bug（BoxingMate 就栽在这上面：
测试和合成数据都用对象，而真实链路发的是数组，于是整套测试通过、真机全废）。
所以合成姿势只写**一份**，测试和 demo 都从这里取；
`tests/test_action_detect.py` 里还有一条专门测这个替身自洽性的用例。

⚠️ 合成数据只能证明**逻辑与门控**通不通，**证明不了**真人身上准不准，
更证明不了左右方向对不对（§5.7：那一件事只能真人往左跳一次来确认）。
"""
from __future__ import annotations

import math

#: 合成姿势里"直立时的躯干长度"（画面归一化坐标）
TORSO = 0.20
#: 直立时髋中点的 y
HIP_Y = 0.62
#: 地面线（踝站在这上面）
GROUND = 1.02
#: 默认帧间隔（ms）= 30fps
DT = 33.0

__all__ = ["TORSO", "HIP_Y", "GROUND", "DT", "make_pose", "make_world",
           "rotate_pose", "scale_pose", "demo_sequence"]


def make_pose(knee_deg=180.0, lane=0.0, rise=0.0, hands_up=False, torso=TORSO,
              frontal=False, base_x=0.50, ground=GROUND, half_hip=0.04,
              foot_sep=0.04):
    """造一帧 33 点（图像坐标，y 向下）。

    knee_deg : 膝关节角（180=伸直）。**踝钉在地面上**，所以屈膝会自然把髋压低
               —— sink 与膝角在这个模型里是同一件事的两面（物理自洽）。
    lane     : 髋中点横向位移，单位 = 躯干（+x = 画面右）
    rise     : 整体竖直平移，单位 = 躯干（跳：整个人离地）
    frontal  : True 时把膝盖放在髋-踝连线上（h=0）。这模拟的是**正面机位**：
               腿在矢状面里弯、在画面上投影成一条直线，所以 2D 膝角恒 ≈180°
               —— 也就是"正面下蹲为什么看不出膝弯"那件事。
    """
    L = torso
    d = 2.0 * L * math.sin(math.radians(knee_deg) / 2.0)     # 髋-踝距离
    h = 0.0 if frontal else math.sqrt(max(0.0, L * L - (d / 2.0) ** 2))
    hx = base_x + lane * torso
    hip_y = ground - d - rise * torso
    ank_y = ground - rise * torso
    sx, sy = hx, hip_y - torso                               # 肩中点（无前倾）

    pts = [(0.5, 0.5, 0.0) for _ in range(33)]
    pts[0] = (sx, sy - 0.10, 0.0)                            # 鼻
    pts[11] = (sx - 0.09, sy, 0.0)                           # 左肩
    pts[12] = (sx + 0.09, sy, 0.0)
    pts[23] = (hx - half_hip, hip_y, 0.0)                    # 左髋
    pts[24] = (hx + half_hip, hip_y, 0.0)
    mid_y = (hip_y + ank_y) / 2.0
    pts[25] = (hx - half_hip - h, mid_y, 0.0)                # 左膝
    pts[26] = (hx + half_hip + h, mid_y, 0.0)
    pts[27] = (hx - foot_sep, ank_y, 0.0)                    # 左踝
    pts[28] = (hx + foot_sep, ank_y, 0.0)
    wy = sy - 0.12 if hands_up else sy + 0.22
    pts[13] = (sx - 0.13, sy + 0.10, 0.0)                    # 左肘
    pts[14] = (sx + 0.13, sy + 0.10, 0.0)
    pts[15] = (sx - 0.13, wy, 0.0)                           # 左腕
    pts[16] = (sx + 0.13, wy, 0.0)
    return pts


def make_world(knee_deg=180.0, leg=0.40):
    """造一帧 33 点**世界坐标**（米制 3D），腿在 z 方向弯。

    关键点：它的 (x, y) 投影是**共线**的 —— 正面机位看到的就长这样：
    2D 膝角恒 180°，而 3D 膝角是真实的弯曲量。§5.1 让开 `outputWorldLandmarks`
    就是为了拿到这个量。
    """
    d = 2.0 * leg * math.sin(math.radians(knee_deg) / 2.0)
    h = math.sqrt(max(0.0, leg * leg - (d / 2.0) ** 2))
    pts = [(0.0, 0.0, 0.0) for _ in range(33)]
    for hip, knee, ank, sgn in ((23, 25, 27, -1), (24, 26, 28, +1)):
        pts[hip] = (sgn * 0.10, 0.0, 0.0)
        pts[knee] = (sgn * 0.10, -d / 2.0, -h)
        pts[ank] = (sgn * 0.10, -d, 0.0)
    return pts


def rotate_pose(pose, deg, cx=0.50, cy=HIP_Y):
    """把整帧绕髋中点转 deg 度 —— 模拟**相机横滚**（§5.3.3 的 rollResidual）。"""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return [(cx + (x - cx) * c - (y - cy) * s,
             cy + (x - cx) * s + (y - cy) * c, z) for (x, y, z) in pose]


def scale_pose(pose, k, cx=0.50, cy=HIP_Y):
    """把整帧以髋中点为中心缩放 k 倍 —— 模拟人走近 / 走远。"""
    return [(cx + (x - cx) * k, cy + (y - cy) * k, z) for (x, y, z) in pose]


def demo_sequence(dt=DT, calib_ms=3000.0, stand_ms=400.0, hold_ms=1200.0):
    """合成一段"四个动作各做一遍"的关键点流。

    产出 `[(t_ms, body, world), ...]`：先站直一段（给标定用），
    然后依次 蹲下→站起→右移→回位→左移→回位→跳→站定。

    `calib_ms` 是开头"站直"的时长，**必须大于调用方的标定窗口**，
    否则标定会盖到动作上、被判成"人没站住"而拒绝（那是正确行为，
    但 demo 会因此跑不起来）。`action/__main__.py` 会按需把它调大。

    ⚠️ 合成数据。它证明的是"链路通、事件对、序列化没问题"，
    **不证明**真人身上准，也**不证明**左右方向（§5.7）。
    """
    frames = []
    t = 0.0

    def add(body, ms, world=None):
        nonlocal t
        n = max(1, int(round(ms / dt)))
        for _ in range(n):
            frames.append((t, body, world))
            t += dt

    add(make_pose(), calib_ms)                       # 标定：站直
    add(make_pose(), stand_ms)
    add(make_pose(knee_deg=110.0), hold_ms)          # 蹲下（并保持）
    add(make_pose(), hold_ms)                        # 站起
    add(make_pose(), stand_ms)
    add(make_pose(lane=0.6), 500.0)                  # 右移
    add(make_pose(), 900.0)                          # 回位
    add(make_pose(lane=-0.6), 500.0)                 # 左移
    add(make_pose(), 900.0)                          # 回位
    add(make_pose(rise=0.30), 400.0)                 # 跳
    add(make_pose(), 900.0)
    return frames
