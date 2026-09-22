"""关键点 → 动作特征：纯函数 + 少量有状态滤波器。

本模块**只依赖标准库**（不 import mediapipe / cv2 / numpy），
所以 tests/test_action_detect.py 能用合成关键点直接跑，不需要模型权重。

坐标系与归一化（照《向星而行-UE实现设计.md》§5.1）
================================================

MediaPipe Pose 的 33 点，图像坐标：x 向右、y **向下**，取值 0..1（归一化到画面）。

    原点 = 髋中点 (23,24)
    尺度 = |肩中点 − 髋中点|
    图像 y 轴向下 → 所有「高度」取负号后使用

**尺度 = 当前帧的躯干长度**（就是 §5.1 的写法），但夹在标定中性躯干的
[0.35, 3.0] 倍之间。为什么要夹，是量出来的：

  * 同一段 120 帧真实素材里，躯干长度在 **0.168 ~ 0.478** 之间变（差 2.8 倍）
    —— 人往前走两步，画面里的躯干就大一倍；
  * 若改用**固定的**中性躯干当尺度，"往镜头走近"会直接变成一次假横移/假下蹲；
  * 若完全不夹（纯用当前帧），人侧身、或关键点崩掉时躯干接近 0，
    特征会被放大到天上。

顺带纠正一个一开始想岔的地方：用当前帧躯干**不会**让下蹲形成正反馈。
sink 变大并不会反过来改变尺度；只是前倾时躯干投影略短（约 −13%）、
sink 稍微敏感一点 —— 方向是对的，不是失稳。

相机歪斜（§5.3.3 的 rollResidual）
==================================
标定时记下「肩中点→髋中点」这个轴作为**重力反方向**，并据此构造一组正交基：

    up   = normalize(肩中点 − 髋中点)     // 图像坐标里"向上"
    side = (−up.y, up.x)                  // 直立时 = (1, 0)，指向画面 +x

之后所有方向性特征都是**位移在这组基上的投影**，因此相机歪了（横滚）不影响方向
语义 —— 这正是 §5.3.3 想用 rollResidual 扣掉的那个量。用投影实现更直接，而且基是
标定时刻定死的，**不会因为玩家自己前倾而跟着转**。

镜像（§5.7，最容易写反的一处）
==============================
`TaskController.startListen()` 在推理前把画面水平翻转（自拍视角，第 194 行
`frame = frame[:, ::-1, :]`）。在**翻转过的**画面里：

    玩家的右手边 → 画面 +x（x 变大）

本模块把这件事交给**唯一一个开关** `mirrored_input`（检测器上的参数，默认 True，
与上面的管线一致），位置固定在「特征提取之后、状态机之前」，与 §5.7 的要求一致：

    mirrored_input=True    玩家右移 → laneDx > 0
    mirrored_input=False   玩家右移 → laneDx < 0

⚠️ 这一条**必须真人验证一次**（§5.7 原话：合成数据不算数）。CLI 上有
`--mirror / --no-mirror`，真人往左移一次就能确认。

好消息：这里**不会**踩「镜像后 MediaPipe 解剖学左右标签也跟着反」那个坑 ——
横向位移只用**髋中点**，而髋中点是左右两点的均值，与左右标注无关。

特征清单（§5.2 里与本项目四个动作相关的部分）
============================================
    laneDx   (髋中点横向位移) / 躯干        → 左移 / 右移
    rise     (当前髋高 − 中性髋高) / 躯干    → 跳
    sink     −rise                          → 蹲（同一个轴的另一侧）
    kneeL/R  髋-膝-踝夹角（度）              → 蹲（区分"下蹲"与"只是前倾"）
    handsUp  双腕都高于各自同侧肩            → 调试 / 仲裁用（§5.2 的 wristAbove）

⚠️ `sink` 与 `rise` 是**同一个量的两个符号**：§5.2 把它们列成两行，但
(中性髋高 − 当前髋高) 与 (当前髋高 − 中性髋高) 互为相反数。这里两个都给出，
免得用的人自己取错符号 —— 这是本项目最容易犯的一类错误。

未实现的（本次不需要，留位）：`ankleLiftL/R`（抬腿）、`spread/dSpread`（开合跳）。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "MIN_LANDMARKS",
    "P_NOSE", "P_L_SHOULDER", "P_R_SHOULDER", "P_L_ELBOW", "P_R_ELBOW",
    "P_L_WRIST", "P_R_WRIST", "P_L_HIP", "P_R_HIP", "P_L_KNEE", "P_R_KNEE",
    "P_L_ANKLE", "P_R_ANKLE",
    "KEY_INDICES",
    "parse_body", "hip_mid", "shoulder_mid", "torso_length", "knee_angle",
    "NeutralBasis", "Calibrator", "compute_features",
    "MedianFilter", "OneEuroFilter", "FeatureSmoother", "SMOOTHED_SIGNALS",
]

# ── MediaPipe Pose 33 点里本模块用到的那些 ────────────────────────────────────
P_NOSE = 0
P_L_SHOULDER, P_R_SHOULDER = 11, 12
P_L_ELBOW, P_R_ELBOW = 13, 14
P_L_WRIST, P_R_WRIST = 15, 16
P_L_HIP, P_R_HIP = 23, 24
P_L_KNEE, P_R_KNEE = 25, 26
P_L_ANKLE, P_R_ANKLE = 27, 28

#: 用到 ankle(28) 就至少要 29 个点。完整 Pose 是 33 个。
MIN_LANDMARKS = 29

#: 做"人在不在画面里"/质量自检时看的点
KEY_INDICES = (P_L_SHOULDER, P_R_SHOULDER, P_L_WRIST, P_R_WRIST,
               P_L_HIP, P_R_HIP, P_L_KNEE, P_R_KNEE, P_L_ANKLE, P_R_ANKLE)


# ────────────────────────── 关键点访问器 ──────────────────────────

def _xyz(p):
    """把一个关键点读成 (x, y, z)；读不出来返回 None。

    ⚠️ 必须**三种形状都吃**，这是从隔壁 BoxingMate 踩出来的血泪教训：
    真实链路（`Utils.extractLandmarks` → `_toList`）交给下游的是**数组**
    `[[x,y,z], ...]`，而 MediaPipe 原始输出是带 `.x/.y/.z` 的**对象**。
    只认一种，就会出现"合成数据能跑、真机全废"或者反过来 ——
    而这种 bug 整套单元测试都抓不到，因为测试替身和真实接口不一致。
    所以这里对象 / 字典 / 序列都支持，并且测试里**真的喂一次数组**。
    """
    if p is None:
        return None
    if isinstance(p, dict):
        x, y, z = p.get("x"), p.get("y"), p.get("z", 0.0)
    elif hasattr(p, "x") and hasattr(p, "y"):
        x, y, z = p.x, p.y, getattr(p, "z", 0.0)
    elif isinstance(p, (list, tuple)):
        if len(p) < 2:
            return None
        x, y = p[0], p[1]
        z = p[2] if len(p) > 2 else 0.0
    else:
        return None
    try:
        x, y, z = float(x), float(y), float(z)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return (x, y, z)


def parse_body(body) -> Optional[list]:
    """把一帧的 body 关键点读成 [(x,y,z), ...]；不可用返回 None。

    可接受：`[[x,y,z], ...]`（本项目链路的真实格式）、`[{'x':..}, ...]`、
    `[NormalizedLandmark, ...]`、以及带 `.landmark` 的 `NormalizedLandmarkList`。

    **不做置信度过滤**：本项目链路里 `_toList()` 只保留 x/y/z，MediaPipe 的
    `visibility/presence` 在这一层已经丢了。所以"人在不在"只能靠几何自检
    （见 compute_features 的 quality），这一点在文档里写明了。
    """
    if body is None:
        return None
    inner = getattr(body, "landmark", None)
    if inner is not None:
        body = inner
    try:
        n = len(body)
    except TypeError:
        return None
    if n < MIN_LANDMARKS:
        return None
    pts = []
    for p in body:
        q = _xyz(p)
        if q is None:
            return None
        pts.append(q)
    return pts


# ────────────────────────── 几何 ──────────────────────────

def _mean2(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def hip_mid(pts):
    """髋中点（图像坐标）。**只用它做横向位移** —— 它是左右两点的均值，
    与 MediaPipe 的左右标注无关，所以镜像翻转不会影响它。"""
    return _mean2(pts[P_L_HIP], pts[P_R_HIP])


def shoulder_mid(pts):
    return _mean2(pts[P_L_SHOULDER], pts[P_R_SHOULDER])


def torso_length(pts) -> float:
    """|肩中点 − 髋中点|（图像坐标下的二维距离）。"""
    s, h = shoulder_mid(pts), hip_mid(pts)
    return math.hypot(s[0] - h[0], s[1] - h[1])


def knee_angle(hip, knee, ankle) -> Optional[float]:
    """髋-膝-踝夹角，单位**度**。腿伸直 ≈ 180°，深蹲 ≈ 70~120°。

    取的是「膝→髋」与「膝→踝」两个向量的夹角：
    伸直时髋和踝在膝的两侧，两向量反向，夹角 = 180°。
    （与 BoxingMate 的 elbowExtension 同一个几何，只是那里返回 0..1。）

    **传 2 维点算 2D 投影角，传 3 维点算真三维角。** 这一条很关键：
    正面机位下蹲时，髋-膝-踝在画面里几乎共线，2D 投影角始终 ≈180°，
    **根本看不出蹲**；只有世界坐标（米制 3D）才能算出真实膝弯（§5.1 原话：
    "世界坐标是米制，能让蹲下/抬腿的角度判据基本与机位无关"）。
    """
    d = min(len(hip), len(knee), len(ankle), 3)
    a = [hip[i] - knee[i] for i in range(d)]
    b = [ankle[i] - knee[i] for i in range(d)]
    na = math.sqrt(sum(v * v for v in a))
    nb = math.sqrt(sum(v * v for v in b))
    if na < 1e-9 or nb < 1e-9:
        return None
    c = sum(a[i] * b[i] for i in range(d)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# ────────────────────────── 中性基线（标定） ──────────────────────────

@dataclass
class NeutralBasis:
    """一次标定得到的"站立中性"参照系。

    hip / torso 都是**图像坐标**下的量（不是归一化后的），
    因为位移要拿它们当原点与尺度。
    """
    hip_x: float
    hip_y: float
    torso: float
    #: 重力反方向（图像坐标），直立时 ≈ (0, -1)
    up_x: float
    up_y: float
    #: 与 up 正交；直立时 ≈ (1, 0) = 画面 +x
    side_x: float
    side_y: float
    samples: int = 0
    #: 标定期间髋中点的抖动（躯干为单位）。太大说明"站直 2 秒"时人没站住。
    hip_spread: float = 0.0
    #: §5.3.3 的 rollResidual：躯干轴偏离竖直的角度（度），仅供诊断
    roll_deg: float = 0.0

    @property
    def stable(self) -> bool:
        """标定期抖动是否小到可以信。阈值 0.06 躯干 ≈ 站直时髋部晃不到 3cm。"""
        return self.hip_spread <= 0.06


class Calibrator:
    """采集"站直"的若干帧，算出 NeutralBasis。

    用法（见 action/__main__.py 的 --calib-ms）：
        cal = Calibrator()
        cal.add(pts)            # 每帧喂一次
        basis = cal.finalize()  # 不够帧数 / 数据太差 → None（**不编数字**）
    """

    #: 至少要这么多帧才敢标定。§5.3 说"站直 2 秒"，所以正常会远超它；
    #: 这条下限是为了挡住"一帧就标定"这种用法。
    MIN_SAMPLES = 8
    MAX_SAMPLES = 2000

    def __init__(self, min_samples: int = MIN_SAMPLES):
        self.min_samples = min_samples
        self.reset()

    def reset(self):
        self._hips = []
        self._torsos = []
        self._ups = []
        self.rejected = 0

    @property
    def count(self) -> int:
        return len(self._hips)

    def add(self, pts) -> bool:
        """加一帧。`pts` 可以是原始 body（数组/对象/字典都行）或已解析的点。

        返回是否被采纳（读不出来、或躯干长度不合理的帧会被丢掉）。
        """
        pts = parse_body(pts)
        if pts is None:
            self.rejected += 1
            return False
        t = torso_length(pts)
        # 躯干短到 0.01 基本就是"人侧对镜头/关键点崩了"，不要污染基线
        if not (t > 0.01) or not math.isfinite(t):
            self.rejected += 1
            return False
        s, h = shoulder_mid(pts), hip_mid(pts)
        ux, uy = s[0] - h[0], s[1] - h[1]
        n = math.hypot(ux, uy)
        if n < 1e-9:
            self.rejected += 1
            return False
        if len(self._hips) >= self.MAX_SAMPLES:
            return False
        self._hips.append(h)
        self._torsos.append(t)
        self._ups.append((ux / n, uy / n))
        return True

    def finalize(self) -> Optional[NeutralBasis]:
        """算出基线。样本不足返回 None —— 宁可不标定，也不用假基线。"""
        n = len(self._hips)
        if n < self.min_samples:
            return None
        hx = sum(p[0] for p in self._hips) / n
        hy = sum(p[1] for p in self._hips) / n
        torso = sum(self._torsos) / n
        if not (torso > 0.01) or not math.isfinite(torso):
            return None
        ux = sum(p[0] for p in self._ups) / n
        uy = sum(p[1] for p in self._ups) / n
        norm = math.hypot(ux, uy)
        if norm < 1e-9:
            return None
        ux, uy = ux / norm, uy / norm
        # side = (−up.y, up.x)：直立时 up=(0,−1) → side=(1,0) 指向画面 +x
        sx, sy = -uy, ux
        spread = math.sqrt(
            sum((p[0] - hx) ** 2 + (p[1] - hy) ** 2 for p in self._hips) / n
        ) / torso
        roll = math.degrees(math.atan2(ux, -uy))
        return NeutralBasis(hip_x=hx, hip_y=hy, torso=torso,
                            up_x=ux, up_y=uy, side_x=sx, side_y=sy,
                            samples=n, hip_spread=spread, roll_deg=roll)


# ────────────────────────── 特征提取 ──────────────────────────

def compute_features(basis: NeutralBasis, pts, world=None) -> Optional[dict]:
    """一帧关键点 → 原始特征字典（**未滤波、未过镜像开关**）。

    `pts` 可以是原始 body（数组/对象/字典都行）或已解析的点 —— 内部统一
    过一遍 parse_body，所以调用方不必先自己转格式（本项目最大的 bug 来源
    就是"某一层只认一种格式"，这里从接口上把它堵死）。

    返回 None 表示这一帧不可用（点不够 / 躯干崩了）。
    `laneDx` 在这里还是**图像坐标**下的横向位移；把它翻成"玩家的左右"
    是检测器的事（唯一那个 mirrored_input 开关），见模块头注释。
    """
    pts = parse_body(pts)
    if pts is None or basis is None:
        return None
    # 世界坐标（米制 3D）只用来算**角度**：正面机位下蹲时髋-膝-踝在画面里
    # 近乎共线，2D 投影角始终 ≈180°，根本看不出蹲。见 knee_angle 的注释。
    wpts = parse_body(world) if world is not None else None
    h = hip_mid(pts)
    dx, dy = h[0] - basis.hip_x, h[1] - basis.hip_y

    # 位移投影到标定时刻定下的正交基上（= §5.3.3 扣掉 rollResidual）
    #   rise > 0 = 髋部比中性更高（跳）
    #   lane   > 0 = 髋部朝画面 +x 移动
    up = dx * basis.up_x + dy * basis.up_y
    lane = dx * basis.side_x + dy * basis.side_y

    # 尺度 = **当前帧**的躯干长度（§5.1 的写法），但夹在标定中性躯干的
    # [0.35, 3.0] 倍之间。为什么要夹：真实素材里躯干会随人前后走动大幅变化
    # （本机实测同一段 120 帧素材里 0.168 ~ 0.478，差 2.8 倍），若改用固定的
    # 中性躯干，"往镜头走近一步"就会被直接读成一次横移/下蹲；
    # 而完全不夹又会在人侧身、或关键点崩掉（躯干接近 0）时把特征放大到天上。
    torso_now = torso_length(pts)
    if not (torso_now > 1e-9) or not math.isfinite(torso_now):
        torso_now = basis.torso
    scale = max(basis.torso * 0.35, min(basis.torso * 3.0, torso_now))
    if not (scale > 1e-9):
        return None
    rise = up / scale
    laneDx = lane / scale

    if wpts is not None:
        knee_l = knee_angle(wpts[P_L_HIP], wpts[P_L_KNEE], wpts[P_L_ANKLE])
        knee_r = knee_angle(wpts[P_R_HIP], wpts[P_R_KNEE], wpts[P_R_ANKLE])
        knee_src = "3d"
    else:
        knee_l = knee_angle(pts[P_L_HIP], pts[P_L_KNEE], pts[P_L_ANKLE])
        knee_r = knee_angle(pts[P_R_HIP], pts[P_R_KNEE], pts[P_R_ANKLE])
        knee_src = "2d"

    # §5.2 的 wristAbove：双腕都高于各自同侧肩（图像 y 越小越高）
    l_up = pts[P_L_WRIST][1] < pts[P_L_SHOULDER][1]
    r_up = pts[P_R_WRIST][1] < pts[P_R_SHOULDER][1]
    hands_up_count = int(l_up) + int(r_up)

    # 质量自检：关键点是否大体落在画面里（宽松一点，0..1 之外也允许一点）
    inside = 0
    for i in KEY_INDICES:
        x, y = pts[i][0], pts[i][1]
        if -0.35 <= x <= 1.35 and -0.35 <= y <= 1.35:
            inside += 1
    quality = inside / len(KEY_INDICES)

    return {
        "laneDx": laneDx,          # 图像坐标语义，符号由检测器的 mirror 开关决定
        "rise": rise,              # >0 高于中性
        "sink": -rise,             # §5.2 的同一根轴，另一侧；两行都给，免得取错符号
        "kneeL": knee_l,
        "kneeR": knee_r,
        # "3d" = 用了世界坐标（可信，可开 requireKnees）；"2d" = 画面投影角
        "kneeSource": knee_src,
        "handsUp": bool(l_up and r_up),
        "handsUpCount": hands_up_count,
        "quality": quality,
        "torso": torso_now,              # 当前帧躯干（诊断用）
        "scaleUsed": scale,              # 真正用于归一化的尺度（诊断用）
        "torsoRatio": torso_now / basis.torso,
    }


# ────────────────────────── 滤波（§5.1） ──────────────────────────

#: 参与平滑的信号。sink 是 rise 的相反数，不用单独滤。
SMOOTHED_SIGNALS = ("laneDx", "rise", "kneeL", "kneeR")


class MedianFilter:
    """滑动中值。专杀单帧尖峰（跟踪丢失重捕造成的"瞬移"）。

    少于 3 个样本时直接透传 —— 那时中值没有意义，硬算反而引入偏差。
    """

    def __init__(self, window: int = 5):
        self.window = max(1, int(window))
        self._buf = deque(maxlen=self.window)

    def reset(self):
        self._buf.clear()

    def __call__(self, x: float) -> float:
        self._buf.append(x)
        if len(self._buf) < 3:
            return x
        return sorted(self._buf)[len(self._buf) // 2]


class OneEuroFilter:
    """1€ 滤波器（Casiez et al. 2012）。

    "抖动抑制"与"快速动作跟随"之间的平衡最好，已被单目 BlazePose 动捕工作验证
    （§5.1 引用的 ARC 论文）。默认 `fc=1.0, beta=0.007` 就是 §5.1 给的值。

    原理一句话：截止频率随**速度**自适应 —— 慢速时重滤波（压抖动），
    快速时轻滤波（不拖尾）。
    """

    def __init__(self, fc: float = 1.0, beta: float = 0.007, dcutoff: float = 1.0):
        self.fc = fc
        self.beta = beta
        self.dcutoff = dcutoff
        self.reset()

    def reset(self):
        self._x = None
        self._dx = 0.0
        self._t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: float) -> float:
        """t 单位**秒**。时间不前进（或首帧）时原样返回并把状态对齐。"""
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._t, self._dx = x, t, 0.0
            return x
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.dcutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.fc + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        self._t = t
        return self._x


class FeatureSmoother:
    """把 §5.1 的两级滤波套到特征上：中值（窗口 5）→ 1€。

    为什么滤在**特征**上而不是滤关键点：状态机看的量就是这几个，
    滤在这一层，行为和阈值的关系最直接、也最好测。
    """

    def __init__(self, median_window: int = 5, fc: float = 1.0,
                 beta: float = 0.007):
        self.median_window = median_window
        self.fc = fc
        self.beta = beta
        self.reset()

    def reset(self):
        self._med = {}
        self._eur = {}

    def update(self, t_sec: float, feats: Optional[dict]) -> Optional[dict]:
        """返回平滑后的特征副本；`feats` 为 None 时原样返回 None。"""
        if feats is None:
            return None
        out = dict(feats)
        for k in SMOOTHED_SIGNALS:
            v = feats.get(k)
            if v is None or not isinstance(v, (int, float)):
                continue
            med = self._med.get(k)
            if med is None:
                med = self._med[k] = MedianFilter(self.median_window)
            eur = self._eur.get(k)
            if eur is None:
                eur = self._eur[k] = OneEuroFilter(self.fc, self.beta)
            out[k] = eur(t_sec, med(float(v)))
        return out
