"""跟着全景里的人，并对这个人做动作检测。

两步（对应需求原文）
--------------------
1. **跟随我**：我可能出现在全景的任意方位。所以先在球面上「找人」，锁定之后
   只在当前方位附近抽一个视角「跟住」；跟丢若干帧才重新找人。
2. **做我的动作检测**：把锁定方位的那一小块抽成正常透视画面，
   喂给 GestureMate 原有的管线（holistic 关键点 + face_body 语义），
   于是手势 / 眨眼 / 举手 / 身体姿态全部照旧可用。

为什么要两级
------------
"找人"要便宜、要能扫一圈；"做动作检测"要准、要细。两者矛盾，所以分开：
  · 找人/跟随：`pose.Pose(complexity=0)`，小图（默认 224），快
  · 动作检测：`LandmarkEngine`（holistic，脸+身+手），在锁定后抽的大图上跑一次
找人是"偶尔"发生的（只在跟丢时全扫），所以每帧的常态开销只是一次小图 pose + 一次大图 holistic。

关于"任意位置"
--------------
直接把整张等距柱状图丢给 pose 是不行的：全景左右两端被拉伸得最厉害，
人在那儿会检不出来。所以本模块**永远只把人形检测器喂给"已经矫正过的透视视角"**，
全景本身只用来抽视角。这也是 x5-link README 里那句
「从这张全景图里按朝向抽透视视角即可」的落地方式。

真机（X5 实机画面）暴露的三个问题 —— 本文件已修
------------------------------------------------
2024 年那次交接里写着「没验证到的：真人站在全景里的实时跟随」。补验之后，
在真实 X5 画面（2880x1440，相机平放在桌上、人从上方俯身入镜）上，
原来的实现对**真实存在的两个人一帧都锁不住**（`locked=False`，score 0.26 < 0.35）。
原因有三条，都是可复现的：

1. **探针检出器是有状态的**。原来 `_probe` 复用同一个
   `pose.Pose(static_image_mode=False)`（视频跟踪模式），而"绕一圈找人"喂进去的
   是一堆**互不相关**的视角。实测同一张输入（yaw=-180, pitch=0, fov=90, 224px），
   作为第 1 次调用 score=0.26，作为第 13 次调用 score=0.77 —— 分数取决于"上一次
   探的是哪个方向"。找人因此不可靠。→ 现在**找人/跟随一律用独立的无状态检出器**
   （`static_image_mode=True`），分数只由这一帧、这个视角决定。
2. **只扫地平线**。原来 `acquire()` 只采 pitch=0。可 X5 平放在桌上时，人是**俯身
   从上方入镜**的 —— 实测这一帧里两个人分别在 pitch≈+52° 和 pitch≈+78°，
   地平线上根本没有躯干。→ 现在扫**球面**（yaw x pitch 2D，默认覆盖 -90°..+105°）。
3. **俯仰被硬夹到 ±35°、锁定时还乘了 0.6**。就算找到了 pitch=78° 的人，也会被
   夹回 35°，等于没锁。→ 现在可配 `pitch_limit`（默认 75°）。

另外两条同源的坑：

* **近天顶视角是病态的**。`view_basis()` 里 `right = forward x world_up`，pitch→±90°
  时 `right` 长度→0，抽出来的透视画面会**乱滚**、检测器检不出。实测同一个方位
  (yaw+131, pitch+77)：fov=90 能检出脸 468 + 身 33，fov=70 冷启动检出 0。
  → 所以锁定时不只是"挑分数最高"，还要**用最终那个视角去验一遍**（见下）。
* **分数最高的方位未必是最适合检测的方位**。实测最高分 0.996 落在 pitch+77°
  那个"被大手臂占满画面"的视角上，而真正框得正的是 pitch≈+52° 的女士。
  → 现在对近天顶方位按比例降权，并且**逐个候选拿最终视角验证**，验不过就试下一个。

锁定的做法：粗扫 -> 细化 -> 验证
--------------------------------
    coarse:  球面上 yaw x pitch 粗扫（默认 12 x 4 = 48 个视角，224px，约 1.4s）
    nms:     把落在同一个人身上的重复候选合并，让候选**覆盖不同的人**
    refine:  在候选附近 ±15° 再采一轮，收紧方位
    verify:  在"真正要用的那个视角"（view_size + 选定 fov）上确认能看见躯干；
             fov 依次试「按占比拟合出来的」-> 90 -> 110 -> 70，取第一个验过的
    只有验过的候选才锁。全部验不过 = 老实说没找到（空场景不误锁）。

代价：跟丢时略慢是可以接受的 —— 那是"重新找人"，本来就是偶发事件。
跟住之后每帧的常态开销仍然是 一次小图 pose + 一次大图 holistic。
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # GestureMate 根
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "webui"))

from pano.projection import (equirect_to_perspective, pano_yaw_pitch,  # noqa: E402
                             view_point_to_pano)

ALL_PARTS = ("face", "body", "leftHand", "rightHand")

#: 判断"这个视角里有没有人"时，取这些关节的可见度
_TORSO = [11, 12, 23, 24]          # 双肩 + 双髋：人要能被跟住，躯干必须看得见

#: 从 holistic 的身体关键点反推方位时用这些关节（鼻/双肩/双髋/双膝）。
#: 它们都在画面内就足够定出人相对相机的方向，不必要求全身。
_ANCHORS = [0, 11, 12, 23, 24, 25, 26]

#: 球面粗扫的默认俯仰采样。3 个采样配 fov 90° 覆盖约 -85°..+95°（即除正下方极点
#: 之外整个球面），而 12x3=36 次探针已经够用 —— 实测把采样从 48 次降到 24 次，
#: 冷启动找人从 ~2.1-2.7s 降到 ~1.5s，而且锁到的是**同一个人**、分数还更高
#: （0.96 vs 0.85）。别再退回"只扫地平线"（理由见文件头）。
DEFAULT_SWEEP_PITCHES = (-40.0, 5.0, 50.0)

#: 锁定时依次尝试的 fov 候选（第一项由"人在画面里占多高"拟合得到）。
#: 顺序是"由紧到松"：能框紧就别用大广角，人越大下游检测越稳。
_FOV_FALLBACKS = (70.0, 90.0, 110.0)

#: 超过这个俯仰角，透视视角开始病态（right 向量变短、画面乱滚）
_DEGENERATE_PITCH = 60.0


def _circular_smooth(prev_deg: float, new_deg: float, alpha: float) -> float:
    """角度按圆周平滑。

    直接对角度做 EMA 是错的：359° 和 1° 只差 2°，平均却给 180°。
    所以在 (sin,cos) 上平滑再 atan2 回来。
    """
    a = math.radians(prev_deg)
    b = math.radians(new_deg)
    s = (1 - alpha) * math.sin(a) + alpha * math.sin(b)
    c = (1 - alpha) * math.cos(a) + alpha * math.cos(b)
    return math.degrees(math.atan2(s, c))


def _angle_diff(a: float, b: float) -> float:
    """两个方位角之间的最短圆周距离（0..180）。"""
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass
class Fix:
    """一次"人在哪"的结果。"""
    yaw: float = 0.0
    pitch: float = 0.0
    fov: float = 70.0
    score: float = 0.0                 # 0..1，越大越可信
    n_joints: int = 0
    locked: bool = False
    acquired: bool = False             # 这一帧是不是刚重新找到人
    misses: int = 0


@dataclass
class PanoResult:
    """一帧的完整结果：定位 + 抽出来的画面 + 动作检测。"""
    fix: Fix = field(default_factory=Fix)
    view: Optional[np.ndarray] = None       # 抽出来的透视画面（就是"玩家视角"）
    landmarks: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    t_locate: float = 0.0
    t_detect: float = 0.0
    #: 这一帧**真的跑了**动作检测吗？`detect_every > 1` 时，跳过的那些帧
    #: 复用上一帧的 landmarks/features（画面照抽、方位照走），这里就是 False，
    #: 免得把"上一帧的结果"当成"这一帧的结果"报出去。
    detected: bool = True

    @property
    def found(self) -> bool:
        return self.fix.locked


@dataclass
class _Candidate:
    """粗扫出来的一个"这里可能有人"。"""
    score: float
    yaw: float          # 人在全景里的方位（已反投影，不是探针的朝向）
    pitch: float
    n_joints: int


class PanoFollower:
    """在全景里找一个人并跟住，同时对他做动作检测。"""

    def __init__(self, fov: float = 70.0, view_size=(384, 384),
                 sweep_n: int = 8,
                 sweep_pitches: Sequence[float] = DEFAULT_SWEEP_PITCHES,
                 sweep_size=(224, 224), sweep_fov: float = 90.0,
                 smooth: float = 0.45, fov_smooth: float = 0.85,
                 max_misses: int = 4, min_score: float = 0.35,
                 target_height: float = 0.62, fov_range=(38.0, 95.0),
                 complexity: int = 1, want_face: bool = True,
                 pose_complexity: int = 0,
                 pitch_limit: float = 75.0,
                 verify_top_k: int = 3,
                 verify_fovs: Sequence[float] = _FOV_FALLBACKS,
                 relock_span: float = 20.0,
                 track_from_detect: bool = True,
                 detect_every: int = 1):
        self.fov = float(fov)
        self.view_w, self.view_h = view_size
        self.sweep_n = int(sweep_n)
        self.sweep_pitches = tuple(float(p) for p in sweep_pitches) or (0.0,)
        self.sweep_w, self.sweep_h = sweep_size
        self.sweep_fov = float(sweep_fov)
        self.smooth = float(smooth)
        self.fov_smooth = float(fov_smooth)
        self.max_misses = int(max_misses)
        self.min_score = float(min_score)
        self.target_height = float(target_height)
        self.fov_min, self.fov_max = fov_range
        self.complexity = complexity
        self.want_face = want_face
        self.pose_complexity = pose_complexity
        self.pitch_limit = float(pitch_limit)
        self.verify_top_k = int(verify_top_k)
        self.verify_fovs = tuple(float(f) for f in verify_fovs)
        #: 上次锁定过的大致方位。"跟丢"绝大多数只是抖动/短暂遮挡，先在这一圈
        #: 附近找（约 0.5s），比把整个球面重扫一遍（约 3s）便宜 6 倍。
        self.relock_span = float(relock_span)
        #: 跟住的时候，直接用下游引擎检出的身体关键点来闭环，省掉一次独立的
        #: pose 探针（实测 512px 的一次**成功**检出要约 116ms，是整帧最大的一笔）。
        self.track_from_detect = bool(track_from_detect)

        self.fix = Fix()
        self._last: Optional[tuple[float, float]] = None
        #: 抽视角/画图每帧都做，但**动作检测可以每 N 帧才跑一次**。holistic 是现在
        #: 整条流水线的地板（实测 40–70ms，占一帧预算的大头），跳过一帧就省一帧。
        #: 默认 1（每帧都检）；设 2 大约能把帧率再翻一倍，代价是语义（眨眼/举手）
        #: 以一半的频率刷新、且跳过的帧不再闭环校正方位。
        self.detect_every = max(1, int(detect_every))
        self._n = 0
        self._last_lm: dict = {}
        self._last_features: dict = {}
        self._detector = None
        self._engine = None
        self._analyzer = None

    # ------------------------------------------------------------ 生命周期
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        for obj in (self._detector, self._engine):
            try:
                obj and obj.close()
            except Exception:
                pass
        self._detector = self._engine = None

    # ------------------------------------------------------------ 懒加载
    @property
    def detector(self):
        """**无状态**的人形检出器，专门给"找人/跟随"用。

        必须是 `static_image_mode=True`：找人是拿一堆**互不相关**的视角去问
        "这里有没有人"，而视频跟踪模式（=False）会把上一次的跟踪状态带进来 ——
        实测同一张输入，作为第 1 次调用 score=0.26、作为第 13 次调用 score=0.77。
        跟随的分数绝不能再依赖"上一个探的是哪个方向"。
        """
        if self._detector is None:
            import mediapipe.python.solutions as sol
            self._detector = sol.pose.Pose(
                static_image_mode=True, model_complexity=self.pose_complexity,
                enable_segmentation=False,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        return self._detector

    @property
    def engine(self):
        if self._engine is None:
            from LandmarkEngine import LandmarkEngine
            self._engine = LandmarkEngine(model_complexity=self.complexity,
                                          need=ALL_PARTS)
        return self._engine

    @property
    def analyzer(self):
        if self._analyzer is None:
            from face_body import FaceBodyAnalyzer
            self._analyzer = FaceBodyAnalyzer()
        return self._analyzer

    # ------------------------------------------------------------ 打分
    @staticmethod
    def _orientation_penalty(person_pitch: float) -> float:
        """近天顶的方位要降权。

        `view_basis()` 里 `right = forward x world_up`，pitch 越接近 ±90° 这个
        叉积越短，抽出来的透视画面会乱滚、检测器检不出（实测 fov=70 时对
        pitch+77° 的画面冷启动检出 0，而 fov=90 能检出）。所以"人在天顶"这件事
        本身就要扣分，否则会锁到一个下游根本检不出的视角上。
        """
        a = abs(person_pitch)
        if a <= _DEGENERATE_PITCH:
            return 1.0
        return float(max(0.25, 1.0 - 0.5 * (a - _DEGENERATE_PITCH) / 30.0))

    # ------------------------------------------------------------ 找人
    def _probe(self, erp, yaw, pitch, w, h, fov):
        """在一个候选方位抽小图跑 pose，返回 (score, pano_yaw, pano_pitch, n_joints)。

        score 综合考虑「躯干可见度」和「人在画面里的占比」：
        只看可见度的话，一个贴在画面边缘、只露出半个身子的人也会得高分。
        最后再乘一个近天顶降权项（见 `_orientation_penalty`）。
        """
        view = equirect_to_perspective(erp, w, h, fov, yaw, pitch)
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.detector.process(rgb)
        lm = getattr(res, "pose_landmarks", None)
        if not lm:
            return 0.0, yaw, pitch, 0

        pts = np.array([[p.x, p.y, p.visibility] for p in lm.landmark])
        vis = pts[:, 2]
        n_vis = int((vis > 0.5).sum())
        torso = float(np.mean(vis[_TORSO]))
        if torso < 0.3:
            # 只有上半身在画面里时（fov 收紧或人靠得很近），髋部本来就看不见，
            # 不该因此判成"没有人"。给双肩一个机会，但降一点权 —— 这是"跟住"
            # 能和"找人"用同一把尺子的前提，否则跟踪会一直报 miss 并反复重找。
            shoulders = float(np.mean(vis[[11, 12]]))
            if shoulders < 0.5:
                return 0.0, yaw, pitch, n_vis
            torso = shoulders * 0.8

        # 只用**画面内**的关节。MediaPipe 会把关节外推到画面外（实测髋部 y=1.66），
        # 而画面外的点有两个坏处：span 被撑大导致"占比"项饱和（分数全挤在 1.0 附近），
        # 反投影回全景时的方向也已经不成立。
        inside = ((pts[:, 0] > -0.05) & (pts[:, 0] < 1.05)
                  & (pts[:, 1] > -0.05) & (pts[:, 1] < 1.05))
        m = (vis > 0.4) & inside
        if int(m.sum()) < 4:
            return 0.0, yaw, pitch, n_vis

        # 人在画面里占多高（用可见关节的纵向跨度近似），太小/贴边都降分
        ys = pts[m, 1]
        span = float(ys.max() - ys.min())
        if span <= 0.02:
            return 0.0, yaw, pitch, n_vis
        fill = min(1.0, span / 0.6)

        score = float(np.clip(torso, 0, 1) * (0.35 + 0.65 * fill)) * min(1.0, n_vis / 12.0)
        if score <= 0:
            return 0.0, yaw, pitch, n_vis

        # 反推这个人在全景里的方位：取可见且可信的关节做中位数（抗单点抖动）
        py_, pp_ = view_point_to_pano(pts[m, 0] * w, pts[m, 1] * h, w, h, fov, yaw, pitch)
        # 中位数要在圆周上取：直接用 nanmedian 在 ±180 交界会翻车
        rad = np.radians(py_)
        myaw = math.degrees(math.atan2(np.median(np.sin(rad)), np.median(np.cos(rad))))
        mpitch = float(np.median(pp_))

        score *= self._orientation_penalty(mpitch)
        return float(score), myaw, mpitch, n_vis

    def _sweep(self, erp) -> list[_Candidate]:
        """在**球面**上粗扫找人（yaw x pitch），而不是只扫地平线。

        X5 平放在桌上时人是俯身从上方入镜的（实测 pitch +52° / +78°），
        只扫 pitch=0 会一个人都看不到。
        """
        out: list[_Candidate] = []
        for pitch in self.sweep_pitches:
            for i in range(self.sweep_n):
                yaw = -180.0 + 360.0 * i / self.sweep_n
                s, y, p, n = self._probe(erp, yaw, pitch, self.sweep_w,
                                         self.sweep_h, self.sweep_fov)
                if s > 0:
                    out.append(_Candidate(s, y, p, n))
        return out

    def _nms(self, cands: list[_Candidate], yaw_tol: float = 25.0,
             pitch_tol: float = 25.0) -> list[_Candidate]:
        """把落在同一个人身上的重复候选合并，让候选覆盖**不同的人**。

        一个 90° 视场、步进 30° 的粗扫会把同一个人报到 3 个格子里。不合并的话
        "验证前 k 个"就全花在同一个人身上，一个更好的第二个人永远轮不到。
        """
        kept: list[_Candidate] = []
        for c in sorted(cands, key=lambda c: -c.score):
            if any(_angle_diff(c.yaw, k.yaw) < yaw_tol
                   and abs(c.pitch - k.pitch) < pitch_tol for k in kept):
                continue
            kept.append(c)
        return kept

    def _refine(self, erp, yaw: float, pitch: float,
                step: float = 15.0) -> _Candidate:
        """在候选附近再采一轮，收紧方位。"""
        best = _Candidate(0.0, yaw, pitch, 0)
        for dp in (-step, 0.0, step):
            for dy in (-step, 0.0, step):
                s, y, p, n = self._probe(erp, yaw + dy, pitch + dp,
                                         self.sweep_w, self.sweep_h, self.sweep_fov)
                if s > best.score:
                    best = _Candidate(s, y, p, n)
        return best

    def _fit_fov(self, erp, yaw: float, pitch: float) -> Optional[float]:
        """按"人在画面里占多高"估一个 fov，让远近都差不多大。

        返回 None 表示**这个 fov 下压根没检到他** —— 原来这里直接返回原 fov，
        于是"检不到"被当成了"fov 正合适"，会把一个坏视角锁死。
        """
        view = equirect_to_perspective(erp, self.sweep_w, self.sweep_h, self.fov,
                                       yaw, pitch)
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.detector.process(rgb)
        lm = getattr(res, "pose_landmarks", None)
        if not lm:
            return None
        pts = np.array([[p.x, p.y, p.visibility] for p in lm.landmark])
        vis = pts[:, 2]
        m = vis > 0.4
        if int(m.sum()) < 4:
            return None
        # **被画面裁到就不要拟合尺寸**。可见跨度只有在人整个在画面里时才代表他的
        # 真实角尺寸；被裁之后用它调 fov 会两头翻车（两个方向都实测到了）：
        #   · 只用画面内的关节 -> 收紧 -> 裁更多 -> 跨度更小 -> 一路崩到 fov_min（实测 51.8）
        #   · 用上画面外的关节 -> 越界点把跨度撑爆 -> 一路顶到 fov_max（实测 95.0）
        # 所以裁到就返回 None（"这次不调"），fov 保持不变。等人退到画面里再自动收敛。
        if ((pts[m, 0] < -0.02) | (pts[m, 0] > 1.02)
                | (pts[m, 1] < -0.02) | (pts[m, 1] > 1.02)).any():
            return None
        ys = pts[m, 1]
        span = float(ys.max() - ys.min())
        if span <= 0.02:
            return None
        # 想让 span -> target_height。视场跨度在小角度下近似与 fov 成正比，
        # 一阶近似就够了 —— span 本身就是个粗略量，没必要解 tan 方程。
        return float(np.clip(self.fov * (span / self.target_height),
                             self.fov_min, self.fov_max))

    def _verify(self, erp, yaw: float, pitch: float, deep: bool = False):
        """用**真正要用的那个视角**确认能不能看见他，并顺便定 fov。

        这一步是关键：分数最高的方位不一定是检测器能用的方位（实测最高分
        0.996 落在被大手臂占满的近天顶视角上，fov=70 冷启动检出 0）。
        所以候选要拿最终的 view_size + fov 验一遍，验不过就换下一个 fov、
        再换下一个候选。

        `deep=True` 时再用**下游真正要跑的那个引擎**（holistic 全家桶）复核一遍。
        依据是实测：同一个视角，pose 说"有人"，而 holistic 冷启动可能一个点都
        检不出来（引擎内部是有状态的跟踪器）。锁定的视角必须是**下游能用的**
        视角，不是 pose 觉得能用的视角。顺带这也把引擎预热了。

        返回 (fov, score, n_joints, pano_yaw, pano_pitch)；全试完都没过 -> None。
        """
        pitch = float(np.clip(pitch, -self.pitch_limit, self.pitch_limit))
        fit = self._fit_fov(erp, yaw, pitch)
        tried: list[float] = []
        body_only = []
        for f in ((fit,) if fit is not None else ()) + self.verify_fovs:
            f = float(np.clip(f, self.fov_min, self.fov_max))
            if any(abs(f - t) < 1e-6 for t in tried):
                continue
            tried.append(f)
            s, py_, pp_, n = self._probe(erp, yaw, pitch, self.view_w,
                                         self.view_h, f)
            if s < self.min_score:
                continue
            pp_ = float(np.clip(pp_, -self.pitch_limit, self.pitch_limit))
            if not deep:
                return f, s, n, py_, pp_
            # ⚠️ 要在**真正会锁的那个视角**（py_, pp_）上验，而不是候选视角。
            # 锁定用的是 probe 反推出来的"人在哪"，和候选视角能差几度 —— 实测差
            # 2.4° 就足以让有状态的下游引擎从"有脸"变成"没脸"，于是 acquire 说
            # 验过了、紧接着 step() 的检测却一个脸部点都出不来。
            seen = self._engine_sees(erp, py_, pp_, f)
            if seen == "face":
                return f, s, n, py_, pp_
            if seen == "body":
                body_only.append((f, s, n, py_, pp_))
        if body_only:
            # 只检到身体（人背对镜头/低头）也接受 —— "跟随"不该因为看不见脸就放弃。
            # 但要先把所有候选试完，能拿到脸的优先，因为脸才有眨眼/头朝向那些语义。
            f, s, n, py_, pp_ = body_only[0]
            pp_ = float(np.clip(pp_, -self.pitch_limit, self.pitch_limit))
            return f, s, n, py_, pp_
        return None

    def _engine_sees(self, erp, yaw: float, pitch: float, fov: float):
        """下游 holistic 引擎在这个视角上能检到什么：'face' / 'body' / None。"""
        view = equirect_to_perspective(erp, self.view_w, self.view_h, fov, yaw, pitch)
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            res = self.engine.process(rgb)
            if getattr(res, "face_landmarks", None):
                return "face"
            if getattr(res, "pose_landmarks", None):
                return "body"
            return None
        except Exception:
            return None

    # ------------------------------------------------------------ 用下游检出的身体闭环
    def _fix_from_body(self, body, view_w: int, view_h: int, fov: float):
        """从 holistic 检出的身体关键点反推"人在全景的哪、占多大"。

        `extractLandmarks` 给的是 [[x,y,z], ...]（归一化画面坐标，**没有 visibility**），
        所以能不能用只看"在不在画面里"。返回 (yaw, pitch, span) 或 None。
        """
        if body is None:
            return None
        p = np.asarray(body, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] < 27:
            return None
        p = p[:, :2]
        inside = ((p[:, 0] > -0.02) & (p[:, 0] < 1.02)
                  & (p[:, 1] > -0.02) & (p[:, 1] < 1.02))
        anchor = np.zeros(len(p), dtype=bool)
        anchor[_ANCHORS] = True
        m = inside & anchor
        # 至少 4 个锚点：3 个太松，空旷但杂乱的画面里 pose 会"看出"一个人来
        # （实测在没有人物的合成全景上，3 个锚点就能让跟随锁住一个不存在的人）。
        if int(m.sum()) < 4:
            return None
        ys, ps = view_point_to_pano(p[m, 0] * view_w, p[m, 1] * view_h,
                                    view_w, view_h, fov,
                                    self.fix.yaw, self.fix.pitch)
        rad = np.radians(ys)
        pyaw = math.degrees(math.atan2(np.median(np.sin(rad)), np.median(np.cos(rad))))
        ppitch = float(np.median(ps))
        # 尺寸：整条身体的纵向跨度。**被画面裁到就不算** —— 理由同 _fit_fov，
        # 用裁过的跨度去调 fov 会正反馈。
        span = None
        if int(inside.sum()) >= 4 and not (
                (p[:, 0] < -0.02) | (p[:, 0] > 1.02)
                | (p[:, 1] < -0.02) | (p[:, 1] > 1.02)).any():
            s = float(p[:, 1].max() - p[:, 1].min())
            span = s if s > 0.02 else None
        return pyaw, ppitch, span, int(m.sum())

    def _absorb_detect(self, lm, torso_vis: Optional[float] = None) -> bool:
        """把这一帧的检测结果吸收进跟随状态。成功返回 True。

        `torso_vis` 是下游 holistic 对该身体的**躯干可见度**（`extractLandmarks` 会把
        visibility 丢掉，所以在 step() 里从原始结果单独取出来传进来）。它必须参与打分：
        只用"锚点关节有几个在画面内"来当分数是**骗人的** —— 实测在一个**铺好的床**上，
        holistic 会把被子的褶子检成一具 33 点的"身体"，而且 7/7 锚点全在画面内，
        于是报出 score 1.00。那不是"很确定"，那是"全都框进来了"。
        """
        if torso_vis is not None and torso_vis < 0.3:
            return False
        r = self._fix_from_body(lm.get("body"), self.view_w, self.view_h, self.fix.fov)
        if r is None:
            return False
        pyaw, ppitch, span, n_used = r
        self.fix.acquired = False
        self.fix.yaw = _circular_smooth(self.fix.yaw, pyaw, self.smooth)
        a = min(1.0, self.smooth + 0.2)
        ppitch = float(np.clip(ppitch, -self.pitch_limit, self.pitch_limit))
        self.fix.pitch = self.fix.pitch + (ppitch - self.fix.pitch) * a
        self.fix.pitch = float(np.clip(self.fix.pitch, -self.pitch_limit,
                                       self.pitch_limit))
        conf = min(1.0, n_used / float(len(_ANCHORS)))
        if torso_vis is not None:
            conf *= float(np.clip(torso_vis, 0.0, 1.0))
        self.fix.score = conf
        self.fix.n_joints = n_used
        # 分数掉到阈值以下就当成"这一帧没跟上"，别硬撑着说还在跟。
        if conf < self.min_score:
            self.fix.misses += 1
            if self.fix.misses > self.max_misses:
                self.fix.locked = False
            return True
        self.fix.misses = 0
        self._last = (self.fix.yaw, self.fix.pitch)
        if span is not None:
            want = float(np.clip(self.fix.fov * (span / self.target_height),
                                 self.fov_min, self.fov_max))
            self.fov = self.fov * (1 - self.fov_smooth) + want * self.fov_smooth
        return True

    def _lock(self, erp, yaw: float, pitch: float, deep: bool) -> bool:
        """在这个候选方位附近细化 + 验证，过了就锁定。"""
        r = self._refine(erp, yaw, pitch)
        if r.score < self.min_score:
            return False
        v = self._verify(erp, r.yaw, r.pitch, deep=deep)
        if v is None:
            return False
        fov, score, n, py_, pp_ = v
        self.fix = Fix(yaw=py_, pitch=pp_, fov=fov, score=score,
                       n_joints=n or r.n_joints, locked=True, acquired=True)
        self._last = (py_, pp_)
        return True

    def acquire(self, erp, deep: bool = False) -> bool:
        """找人。先试上次锁定的附近，不行才全球面粗扫。

        顺序是有意的：跟丢绝大多数只是画面抖动或短暂遮挡，人还在原地方位附近。
        实测"局部 3x3 @224px"约 0.48s、能拿回 score 0.93；而"全球面 48 个视角 + 细化
        + 验证"约 3.0s。先局部就轮到的那次，恢复时间直接少 6 倍。
        """
        if self._last is not None and self._lock(erp, self._last[0], self._last[1], deep):
            return True

        cands = self._sweep(erp)
        good = [c for c in cands if c.score >= self.min_score]
        if not good:
            best = max((c.score for c in cands), default=0.0)
            self.fix = Fix(score=best, locked=False, misses=self.fix.misses + 1)
            return False

        for c in self._nms(good)[: self.verify_top_k]:
            if self._lock(erp, c.yaw, c.pitch, deep):
                return True

        self.fix = Fix(score=good[0].score, locked=False, misses=self.fix.misses + 1)
        return False

    # ------------------------------------------------------------ 跟住
    def track(self, erp) -> bool:
        """在当前方位附近抽一个视角跟住它。返回是否仍然锁定。"""
        if not self.fix.locked:
            return False
        # 这一帧是"跟住"而不是"重新找到"：acquired 必须清掉，否则它会一直是
        # 第一次锁定时的 True（原来就是漏了这一步，调用方永远分不清这两种状态）。
        self.fix.acquired = False
        # 用 **view_size** 探，而不是粗扫用的 sweep_size：这一帧的意义是"我马上要交给
        # 下游引擎的那个视角里还有人吗"。用 224 的小图去替 512 的大图做判断，
        # 会在人偏小时判成"没人" —— 于是每帧都记一次 miss，攒够 8 次就白白重新
        # 找人（实测真机帧上 score 掉成 0.00 就是这个）。两种尺寸的耗时几乎一样
        # （224px≈29ms、512px≈35ms），没有省的必要。
        s, y, p, n = self._probe(erp, self.fix.yaw, self.fix.pitch,
                                 self.view_w, self.view_h, self.fov)
        if s < self.min_score:
            self.fix.misses += 1
            self.fix.score = s
            self.fix.n_joints = n
            if self.fix.misses > self.max_misses:
                self.fix.locked = False
            return False

        # 注意 smooth 语义：alpha 越大越"跟得紧"，越小越稳
        self.fix.yaw = _circular_smooth(self.fix.yaw, y, self.smooth)
        # pitch 是 [-90,90] 上的连续量，不需要圆周平滑；但要夹在可用范围内 ——
        # 夹到 ±35° 是错的，X5 平放时人本来就在 +50°..+80°。
        a = min(1.0, self.smooth + 0.2)
        self.fix.pitch = float(np.clip(self.fix.pitch + (p - self.fix.pitch) * a,
                                       -self.pitch_limit, self.pitch_limit))
        self.fix.score = s
        self.fix.n_joints = n
        self.fix.misses = 0
        self._last = (self.fix.yaw, self.fix.pitch)
        self._update_fov(erp)
        return True

    def _update_fov(self, erp):
        """平滑地把 fov 调整到"他占画面 target_height"。检不到就不动。"""
        want = self._fit_fov(erp, self.fix.yaw, self.fix.pitch)
        if want is None:
            return
        self.fov = self.fov * (1 - self.fov_smooth) + want * self.fov_smooth

    # ------------------------------------------------------------ 主入口
    def step(self, erp, detect=True) -> PanoResult:
        """处理一帧全景。detect=False 时只做跟随（省掉 holistic）。

        延迟要点：`detect=True` 且 `track_from_detect=True`（默认）时，**跟住的那一帧
        不做独立的 pose 探针** —— 下游引擎已经在这一帧检出了身体，直接拿它反推方位与
        尺寸来闭环。于是每帧只剩一次检测，实测稳态从 ~292ms 降到 ~90ms。
        检不到身体时才退回探针那条路（保稳）。
        """
        t0 = time.perf_counter()

        if not self.fix.locked:
            self.acquire(erp, deep=detect)
        elif not (detect and self.track_from_detect):
            self.track(erp)
            if not self.fix.locked:
                self.acquire(erp, deep=detect)   # 刚跟丢，立刻重找
        # else: 跟住 + 有下游客 —— 不需要探针，闭环在下面 _absorb_detect

        if not self.fix.locked:
            return PanoResult(fix=replace(self.fix),
                              t_locate=(time.perf_counter() - t0) * 1e3)

        view = equirect_to_perspective(erp, self.view_w, self.view_h, self.fix.fov,
                                       self.fix.yaw, self.fix.pitch)
        # out.fix 是**抽出这个 view 时用的那一份**，这样"标在全景平面上的圈"和
        # "右边那张跟随视角"永远是同一帧、同一个方位；内部状态随后才被闭环更新。
        out = PanoResult(fix=replace(self.fix), view=view,
                         t_locate=(time.perf_counter() - t0) * 1e3)
        if not detect:
            return out

        # 到点才跑检测；没到点的帧复用上一帧结果，并老实标 detected=False。
        run_detect = (self.detect_every <= 1) or (self._n % self.detect_every == 0)
        self._n += 1
        if not run_detect:
            out.landmarks = self._last_lm
            out.features = self._last_features
            out.detected = False
            return out

        t1 = time.perf_counter()
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.engine.process(rgb)
        # holistic 的躯干可见度：extractLandmarks 会把它丢掉，这里单独取出来，
        # 给 _absorb_detect 当"这真的是个人吗"的置信度（否则分数会骗人）。
        torso_vis = None
        pose_lm = getattr(res, "pose_landmarks", None)
        if pose_lm is not None:
            try:
                torso_vis = float(np.mean([pose_lm.landmark[i].visibility
                                           for i in _TORSO]))
            except Exception:
                torso_vis = None
        from Utils import extractLandmarks
        lm = extractLandmarks(res, ALL_PARTS)
        out.landmarks = lm
        try:
            out.features = self.analyzer.update(
                lm.get("face"), lm.get("body"),
                (lm.get("leftHand"), lm.get("rightHand"))) or {}
        except Exception:
            out.features = {}
        out.t_detect = (time.perf_counter() - t1) * 1e3
        self._last_lm = lm
        self._last_features = out.features

        if self.track_from_detect:
            if not self._absorb_detect(lm, torso_vis):
                # 这一帧下游没给出身体（遮挡/转身/走出画面）-> 用探针兜一次，
                # 再不行就记 miss，攒够 max_misses 自然解锁、下一帧去重找。
                self.track(erp)
                if not self.fix.locked:
                    self.fix.misses += 1
            # 分数/关节数/跟丢计数要跟**这一帧的 landmarks 同一来源**，否则会出现
            # "锁定 score=0.58，但脸/身体都是 0"这种自相矛盾的面板（快照里装的是
            # 上一帧的分数）。yaw/pitch/fov 仍然保持"抽出这个 view 时用的那一份"。
            out.fix.score = self.fix.score
            out.fix.n_joints = self.fix.n_joints
            out.fix.misses = self.fix.misses
            out.fix.locked = self.fix.locked
        return out


# ─────────────────────────────────────────────────────────── 画个便于人看的结果

def annotate(result: PanoResult) -> Optional[np.ndarray]:
    """在抽出来的画面上画出检测到的东西（脸/身体/手），用于肉眼验收。"""
    if result.view is None:
        return None
    img = result.view.copy()
    h, w = img.shape[:2]
    lm = result.landmarks

    def draw(pts, color, conns=None, r=2):
        if not pts:
            return
        P = [(int(p[0] * w), int(p[1] * h)) for p in pts]
        if conns:
            for a, b in conns:
                if a < len(P) and b < len(P):
                    cv2.line(img, P[a], P[b], color, 1, cv2.LINE_AA)
        for p in P:
            cv2.circle(img, p, r, color, -1, cv2.LINE_AA)

    if lm.get("body"):
        import mediapipe.python.solutions as sol
        draw(lm["body"], (255, 140, 250), sol.pose.POSE_CONNECTIONS, 2)
    if lm.get("face"):
        draw(lm["face"], (255, 235, 160), None, 1)
    for key, color in (("leftHand", (110, 220, 130)), ("rightHand", (255, 160, 80))):
        if lm.get(key):
            import mediapipe.python.solutions as sol
            draw(lm[key], color, sol.hands.HAND_CONNECTIONS, 2)

    f = result.fix
    txt = f"yaw {f.yaw:+.1f}  pitch {f.pitch:+.1f}  fov {f.fov:.0f}  score {f.score:.2f}"
    cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    fy = result.features or {}
    if fy.get("landmarkCount") is not None:
        cv2.putText(img, f"LM {fy.get('landmarkCount')}  blinks {fy.get('blinkCount')}",
                    (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 255, 200), 1,
                    cv2.LINE_AA)
    return img
