"""四个动作的状态机：左移 / 右移 / 跳 / 蹲 → ActionEvent 流。

对应《向星而行-UE实现设计.md》§5「动作识别与门控设计」。
与 §5 一致的那条总约定，照抄一遍免得走样（`向星而行-任务拆分.md` §1.1）：

    识别侧保证「一次真实动作 = 恰好一个事件」。
    游戏侧不做消歧、不做去抖、不做节流、不做方向反转。

所以迟滞、边沿触发、不应期、优先级仲裁、回位门控**全部在这里实现**。
这是本模块存在的全部理由，也是它必须能被离线断言的原因。

事件（v1.0 契约的子集 —— 本次只要这四个动作）
=============================================
    READY                     标定完成，可以开始
    SQUAT  { on: bool }       锁存态边沿：true=蹲下进入, false=站起
    LANE   { dir: l|r }       换道 / 左移 / 右移（已过镜像与回位门控）
    JUMP                      一次起跳
    PLAYER { present: bool }  人在不在画面里

``PLAYER``/``READY`` 是契约里就有的（游戏侧要用来做暂停和"重新确认站位"），
顺手一起发；`JACK`（开合跳）与 `LEG`（抬腿）本次不做 —— 用户要的是四个动作，
多做一个就多一份真人重调的成本。

四个动作怎么分（同一批下肢关键点，靠仲裁而不是靠四个独立检测器）
================================================================
    rise = (当前髋高 − 中性髋高)/躯干      >0 高于中性
    sink = −rise                            >0 低于中性（§5.2 的同一根轴）
    laneDx = 髋中点横向位移/躯干            >0 = 玩家向自己的右边

    · 跳   ：rise 够大，且**横向没怎么动**（|laneDx| 小）
    · 左移/右移：|laneDx| 够大（方向由符号定）
    · 蹲   ：sink 够大 **且两膝都弯** —— 只看 sink 会把"前倾/后撤"也判成蹲，
             加上膝角才把"下蹲"和"只是身体前倾"分开（§5.2 把 kneeL/R 列出来就是这用）

仲裁与互斥（§5.5，按顺序短路）
==============================
  1. **Crouch 期间屏蔽一切瞬时动作**（蹲下时腿部姿态极端，误触发率高）
  2. **跳要求横向没动**：|laneDx| ≥ laneOn 的一律算换道，不算原地跳
     （所以 LANE 和 JUMP 由构造互斥：jumpMaxLane < laneOn）
  3. **全局回位门控 NeutralGate**：一次瞬时动作之后必须回到中性才能触发下一个
     —— 这是 §5.5 第 4 条，也是"不应期"的统一实现
  4. `laneBlockWhenHandsUp`：§5.5 第 1/2 条要求"双手过肩时锁死换道"，用来把
     开合跳的张腿从横移里摘出去。**本实现默认关**，理由见该字段注释 ——
     我们用的 `laneDx` 是**髋中点**位移，左右腿对称张开时髋中点根本不动，
     所以那条锁死对我们没有收益，只会漏掉"举着手往旁边跳"。

⚠️ 阈值全部是 §5.6 的**首版值**，那张表自己写着"必须真人重调"。
凡是我改动过 §5.6 默认值的地方，都在字段注释里写明了为什么改。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .features import (Calibrator, FeatureSmoother, NeutralBasis,
                       compute_features, parse_body)

__all__ = [
    "KIND_READY", "KIND_SQUAT", "KIND_LANE", "KIND_JUMP", "KIND_PLAYER",
    "POSTURE_STAND", "POSTURE_CROUCH",
    "ActionEvent", "ActionConfig", "Sustained", "ActionDetector",
    "describe_event",
]

KIND_READY = "READY"
KIND_SQUAT = "SQUAT"
KIND_LANE = "LANE"
KIND_JUMP = "JUMP"
KIND_PLAYER = "PLAYER"

POSTURE_STAND = "stand"
POSTURE_CROUCH = "crouch"


def _clamp(v, a, b):
    return a if v < a else (b if v > b else v)


@dataclass
class ActionEvent:
    """一条动作事件。字段随 kind 取用，其余为 None（to_dict 会剔掉）。"""

    kind: str
    t: float                      # 单调毫秒，与调用方传进来的 t 同一个时钟
    conf: float = 1.0             # 0..1，**仅供调试显示**，游戏逻辑不许依赖
    dir: Optional[str] = None     # LANE: "left" | "right"
    on: Optional[bool] = None     # SQUAT: True=蹲下进入, False=站起
    present: Optional[bool] = None  # PLAYER

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "t": round(float(self.t), 1),
             "conf": round(float(self.conf), 3)}
        if self.dir is not None:
            d["dir"] = self.dir
        if self.on is not None:
            d["on"] = bool(self.on)
        if self.present is not None:
            d["present"] = bool(self.present)
        return d


def describe_event(ev: ActionEvent) -> str:
    """给人看的一行（CLI / HUD 用）。"""
    if ev.kind == KIND_SQUAT:
        what = "蹲下" if ev.on else "站起"
    elif ev.kind == KIND_LANE:
        what = "左移" if ev.dir == "left" else "右移"
    elif ev.kind == KIND_JUMP:
        what = "跳"
    elif ev.kind == KIND_PLAYER:
        what = "人在画面里" if ev.present else "人离开画面"
    else:
        what = "标定完成，可以开始"
    return f"{ev.kind:<6} {what}   conf={ev.conf:.2f}"


@dataclass
class ActionConfig:
    """全部阈值集中在这里，方便真人现场重调（§5.6）。"""

    # ── 标定
    calibMinSamples: int = 8
    #: 标定期间髋中点的允许抖动（躯干为单位）。超了就认为"人没站住"，
    #: **拒绝这次标定** —— 一个乱动的基线会让所有位移特征一起失真。
    #: 实测（tests/fixtures 的真人手部特写素材）：标出来 spread=0.37，
    #: 拿它去跑必然误报动作。
    calibMaxSpread: float = 0.06
    #: 显式允许使用"不可信"的基线（仅供调试比对，会明显更容易误触发）。
    allowUnstableCalibration: bool = False
    # ── 滤波（§5.1：中值窗口 5 → One-Euro fc≈1.0, beta≈0.007）
    medianWindow: int = 5
    oneEuroFc: float = 1.0
    oneEuroBeta: float = 0.007

    # ── 在场（§5.4e）
    qualityMin: float = 0.6
    presenceOffMs: float = 500.0
    presenceOnMs: float = 500.0

    # ── 蹲下（§5.4a / §5.6）
    sinkOn: float = 0.28          # 进入：髋部比中性低这么多（躯干为单位）
    sinkOff: float = 0.15         # 退出：迟滞下沿（§5.6 迟滞比 ≈1.9）
    kneeOnDeg: float = 130.0      # 两膝都必须弯过这个角度
    squatDwellMs: float = 250.0   # 持续这么久才算蹲下
    squatReleaseMs: float = 200.0 # 站起也要持续这么久（防临界抖动）
    requireKnees: bool = False
    #: ⚠️ 默认是 **False**，这是与 §5.4a/§5.6「两膝都要 <130°」的一处**有意偏离**。
    #: 理由是量出来的（不是猜的）：正面机位下蹲时，髋-膝-踝在**画面**里几乎共线，
    #: 2D 投影角始终停在 ≈180°，**"膝角 <130°"这条门在正对镜头时永远不可能满足**
    #: —— 打开它会把所有正面下蹲全挡掉。
    #: §5.1 自己写了这条路：要算可靠的下肢角度就得开 `outputWorldLandmarks`
    #: （米制 3D）。所以：
    #:   · 只有 2D 关键点（当前管线默认）→ 保持 False，靠 sink（髋部下沉）判蹲；
    #:   · 拿得到世界坐标（`--world` / 引擎回传 pose_world_landmarks）→ 打开它，
    #:     就是 §5.4a 的原语义，能区分"真下蹲"和"只是身体前倾"。
    #: tests 里两条路都钉了断言（含"正面蹲在 requireKnees=True 下必须不触发"）。

    # ── 左移 / 右移（§5.4c / §5.6）
    laneOn: float = 0.35          # 横向位移阈值
    laneNeutral: float = 0.12     # "回到中性"的判据（回位门控）
    laneDwellMs: float = 80.0     # 位移维持多久才算（§5.6 没给，取短；见 minSamples）
    laneMaxMs: float = 600.0      # 超过这么久才越阈 = 慢速侧移，不算换道（§5.4c）
    laneRearmMs: float = 300.0    # 回到中性维持这么久才重新武装
    laneRequireAirborne: bool = False
    #: §5.6 的「腾空 rise > 0.06」本来是给"左右跳"当辅助证据的。
    #: 用户把"跳"单列成一个动作，所以左右移默认不要求腾空（走路横移也算）。
    #: 想要 §5.4c 那种"左右**跳**"的语义，把它打开。
    laneAirborneMin: float = 0.06
    #: §5.5 第 1/2 条：双手过肩时锁死换道。**默认关** ——
    #: 那条规则的目的是"别把开合跳的张腿读成横移"，而开合跳是**左右腿对称**张开的，
    #: 髋中点（我们用的量）几乎不动，所以这条锁死对本实现没有收益，
    #: 副作用却是漏判"举着手往旁边跳"。想严格复刻 §5.5 就打开。
    laneBlockWhenHandsUp: bool = False

    # ── 跳
    #: ⚠️ §5.6 写的是 rise > 0.06。那个值对"辅助证据"够用，但**单独当"跳"的
    #: 触发门太低**：站起来时重心一晃、原地小颠一下都能到 0.06。
    #: 一次认真的起跳，髋部要抬 15cm 以上，躯干约 0.5m → 0.3 左右。
    #: 所以这里从 0.12 起步（仍是保守的首版值，务必真人重调）。
    riseOn: float = 0.12
    riseOff: float = 0.04         # 迟滞下沿：落回这个高度以下才重新武装
    jumpDwellMs: float = 120.0    # 腾空本来只有 200~400ms，别设太长
    jumpMaxLane: float = 0.20     # 横向超过它就算换道，不算原地跳（< laneOn）

    # ── 通用
    #: 条件必须**连续成立的样本数**。低帧率下时间判据会退化成"单帧就算过"，
    #: 这一条是兜底（BoxingMate 里同样的教训：15fps 时窗口里只有 2 个样本，
    #: 抖动直接变成"动作"）。
    minSamples: int = 2
    neutralRise: float = 0.04     # |rise| 小于它算"高度回到中性"
    neutralDwellMs: float = 120.0 # 中性维持这么久，瞬时动作才重新武装


class Sustained:
    """「条件连续成立够久、且样本数够」才为真。

    时间判据按**样本数**自适应拉长：低帧率下 `dwell_ms` 可能只覆盖 1 个样本，
    这时要求 `(min_samples-1)*dt` 的时长 —— 拉长窗口不会漏掉真动作
    （真动作本来就持续 200ms 以上），却能挡住单帧尖峰。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.since = None
        self.n = 0

    def update(self, cond: bool) -> bool:
        if cond:
            if self.since is None:
                self.n = 1
            else:
                self.n += 1
            return True
        self.since = None
        self.n = 0
        return False

    def mark(self, t: float):
        """条件为真时调用，记录起点（在 update 之后）。"""
        if self.since is None:
            self.since = t

    def ok(self, t: float, dwell_ms: float, min_samples: int,
           dt_avg_ms: float) -> bool:
        if self.since is None or self.n < min_samples:
            return False
        need = max(float(dwell_ms), (min_samples - 1) * max(dt_avg_ms, 0.0))
        return (t - self.since) >= need


def _margin_conf(value: float, threshold: float) -> float:
    """越阈多少 → 0..1 的调试置信度：刚好压线 0.5，1.5 倍阈值 1.0。

    只用于显示。**游戏逻辑不许依赖它** —— 契约里 `conf` 就是这么写的。
    """
    if threshold <= 0:
        return 1.0
    return _clamp(0.5 + 0.5 * (abs(value) / threshold - 1.0) / 0.5, 0.0, 1.0)


class ActionDetector:
    """关键点流 → 动作事件流。

    典型用法（见 action/__main__.py）：

        det = ActionDetector(mirrored_input=True)
        det.begin_calibration()
        ...             # 玩家站直，喂若干帧
        events = det.finish_calibration(t_ms)   # 成功则含一条 READY
        while frames:
            events = det.update(t_ms, body)     # body 可为 None

    `mirrored_input` 是**唯一**的镜像开关，默认 True = 与 TaskController 的
    自拍镜像管线一致（见 features.py 模块头的推导）。真人验证左右反了，
    只需把它改成 False 或换用 `--no-mirror`。
    """

    def __init__(self, config: Optional[ActionConfig] = None,
                 mirrored_input: bool = True):
        self.cfg = config or ActionConfig()
        self.mirrored_input = bool(mirrored_input)
        self._cal = Calibrator(self.cfg.calibMinSamples)
        self.basis: Optional[NeutralBasis] = None
        self.calibrating = False
        self.reset()

    # ------------------------------------------------------------ 生命周期
    def reset(self):
        """清空所有状态机（**不清 basis**）。"""
        cfg = self.cfg
        self._smoother = FeatureSmoother(cfg.medianWindow, cfg.oneEuroFc,
                                         cfg.oneEuroBeta)
        self.last_t = None
        self._dt_avg = 33.0
        # 姿态（锁存）
        self.posture = POSTURE_STAND
        self._crouch_hold = Sustained()
        self._stand_hold = Sustained()
        # 瞬时动作
        self._lane_hold = Sustained()
        self._jump_hold = Sustained()
        self._transient_armed = True
        self._neutral_hold = Sustained()
        self._lane_since = None          # |laneDx| 第一次越过 laneNeutral 的时刻
        self._lane_slow = False          # 本次外移被判为"慢速侧移"
        # 在场
        self.present = False
        self._last_good_t = None
        self._absent_since = None
        self._present_hold = Sustained()
        self.features = None             # 最近一帧的平滑特征（HUD 用）
        self.block = "未标定"

    def begin_calibration(self):
        """开始标定：之后的 update() 只累积基线，不产生动作事件。"""
        self._cal.reset()
        self.calibrating = True
        self.basis = None
        self.reset()
        self.block = "标定中"

    def finish_calibration(self, t: float):
        """结束标定。成功返回 [READY]，失败返回 []（**不编数字**）。"""
        self.calibrating = False
        basis = self._cal.finalize()
        if basis is None:
            self.block = (f"标定失败：有效帧只有 {self._cal.count} 帧"
                          f"（需要 ≥{self.cfg.calibMinSamples}）")
            return []
        if basis.hip_spread > self.cfg.calibMaxSpread \
                and not self.cfg.allowUnstableCalibration:
            # "宁可不标定，也不用假基线"（ActionPredict 里同一条）：
            # 基线抖 0.37 躯干的时候，"相对中性的位移"这个量本身就没有意义了。
            self.block = (f"标定不可信：站直期间髋部抖动 {basis.hip_spread:.3f} 躯干"
                          f"（上限 {self.cfg.calibMaxSpread}）—— 请站定再标定；"
                          f"确要用它请显式开 allowUnstableCalibration")
            return []
        self.basis = basis
        # 标定刚结束，滤波器从零开始，免得把标定期的抖动带进来
        self._smoother.reset()
        # ⚠️ 而且**第一次瞬时动作也必须先回到中性**（NeutralGate 的第一段）。
        # 否则标定一结束、只要当时的位置离中性够远，下一帧就会直接兑现成一个
        # 假动作 —— 实测就是这样：用一份不可信基线跑真实素材，标定后 800ms
        # 误报了一次「跳」。基线可信、人站在中性时，这道门 120ms 内就放行，
        # 不影响正常使用。
        self._reset_transients()
        self.block = ""
        return [ActionEvent(KIND_READY, t, 1.0)]

    @property
    def calibrated(self) -> bool:
        return self.basis is not None

    @property
    def calibration_count(self) -> int:
        return self._cal.count

    # ------------------------------------------------------------ 主入口
    def update(self, t: float, body, world=None):
        """喂一帧。返回本帧产生的事件列表（通常为空）。

        body  : 33 个**图像**关键点（`[[x,y,z], ...]` / 对象 / 字典都行）；
                None 表示这一帧没检到人。
        world : 可选，33 个**世界坐标**（米制 3D）。给了就用它算膝角 ——
                正面机位下 2D 膝角看不出蹲，见 ActionConfig.requireKnees。
        """
        cfg = self.cfg
        events = []

        # 时间步长（用于低帧率自适应）。同一时刻重复调用时不动它。
        if self.last_t is not None:
            dt = t - self.last_t
            if dt > 0:
                self._dt_avg = 0.85 * self._dt_avg + 0.15 * dt
        self.last_t = t

        if self.calibrating:
            self._cal.add(body)
            return events

        pts = parse_body(body)
        raw = (compute_features(self.basis, pts, world)
               if (pts is not None and self.basis) else None)
        feats = self._smoother.update(t / 1000.0, raw)

        # ── 在场判定（§5.4e）
        # 本链路拿不到 MediaPipe 的 visibility/presence（_toList 只留 x/y/z），
        # 所以只能靠几何自检：点齐不齐 + 关键点在不在画面里。
        good = feats is not None and feats["quality"] >= cfg.qualityMin
        if good:
            self._last_good_t = t
            self._absent_since = None
        else:
            if self._absent_since is None:
                self._absent_since = t
            # ── 跟踪断了就当"不连续"处理（这条是实测补上的）──────────────
            # 关键点缺失的那几帧里，滤波器仍留着旧样本；重新捕获到人时，
            # 位置可能已经差了很多，滤波/斜率会把它读成一次极快的位移。
            # 实测：真实素材上正是这样误报了一次"跳"（断 2 帧后 rise 一步到 +0.72）。
            # 所以断帧要①清滤波状态②把所有瞬时计时器清掉并重新要求回中性。
            self._smoother.reset()
            self._reset_transients()

        if not self.present:
            self._present_hold.update(good)
            if good:
                self._present_hold.mark(t)
            if (good and self._present_hold.ok(t, cfg.presenceOnMs, 2,
                                               self._dt_avg)):
                self.present = True
                self._present_hold.reset()
                events.append(ActionEvent(KIND_PLAYER, t, 1.0, present=True))
        else:
            if self._absent_since is not None and \
                    (t - self._absent_since) >= cfg.presenceOffMs:
                self.present = False
                self._absent_since = None
                events.append(ActionEvent(KIND_PLAYER, t, 1.0, present=False))
                # 重新出现时要求先回到中性，且丢掉所有计时器 ——
                # 否则"离开前那一瞬间的计时器"会在回来后立刻兑现成一个假动作。
                self._reset_transients()

        if not self.present or feats is None:
            self.features = feats
            if not self.calibrated:
                self.block = "未标定"
            elif not good:
                self.block = "看不到人（关键点不全或跑出画面）"
            return events

        # ── 镜像开关：唯一一处把"画面 +x"翻成"玩家的左右"（§5.7）
        lane_img = feats["laneDx"]
        lane = lane_img if self.mirrored_input else -lane_img
        rise = feats["rise"]
        sink = -rise
        knee_l, knee_r = feats["kneeL"], feats["kneeR"]
        # 把玩家坐标系的 laneDx 写回特征，HUD/下游看到的就是最终语义
        feats = dict(feats, laneDx=lane, laneDxImage=lane_img)
        self.features = feats

        # ── 1) 姿态：站立 / 蹲下（锁存态，§5.4a）
        knees_ok = True
        if cfg.requireKnees:
            knees = [k for k in (knee_l, knee_r) if k is not None]
            # 两膝都要弯：只给一个膝角时也要求它弯（不假设另一条腿）
            knees_ok = bool(knees) and all(k < cfg.kneeOnDeg for k in knees)
        want_crouch = (sink >= cfg.sinkOn) and knees_ok
        want_stand = sink <= cfg.sinkOff
        if self.posture == POSTURE_STAND:
            self._crouch_hold.update(want_crouch)
            self._stand_hold.reset()
            if want_crouch:
                self._crouch_hold.mark(t)
            if self._crouch_hold.ok(t, cfg.squatDwellMs, cfg.minSamples,
                                    self._dt_avg):
                self.posture = POSTURE_CROUCH
                self._crouch_hold.reset()
                events.append(ActionEvent(
                    KIND_SQUAT, t, _margin_conf(sink, cfg.sinkOn), on=True))
        else:
            self._stand_hold.update(want_stand)
            self._crouch_hold.reset()
            if want_stand:
                self._stand_hold.mark(t)
            if self._stand_hold.ok(t, cfg.squatReleaseMs, cfg.minSamples,
                                   self._dt_avg):
                self.posture = POSTURE_STAND
                self._stand_hold.reset()
                events.append(ActionEvent(
                    KIND_SQUAT, t, _margin_conf(rise, cfg.sinkOff), on=False))

        # ── 2) 全局回位门控（§5.5 第 4 条）
        neutral_now = (self.posture == POSTURE_STAND
                       and abs(lane) <= cfg.laneNeutral
                       and abs(rise) <= cfg.neutralRise)
        self._neutral_hold.update(neutral_now)
        if neutral_now:
            self._neutral_hold.mark(t)
        if not self._transient_armed and self._neutral_hold.ok(
                t, cfg.neutralDwellMs, cfg.minSamples, self._dt_avg):
            self._transient_armed = True
            self._neutral_hold.reset()

        # 横向"这次外移是什么时候开始的" —— 用来排除慢速侧移（§5.4c）
        if abs(lane) <= cfg.laneNeutral:
            self._lane_since = None
            self._lane_slow = False
        elif self._lane_since is None:
            self._lane_since = t

        # ── 3) 瞬时动作。Crouch 期间一律屏蔽（§5.5 第 3 条）
        if self.posture == POSTURE_CROUCH:
            self._lane_hold.reset()
            self._jump_hold.reset()
            self.block = "蹲下期间不产生瞬时动作"
            return events

        hands_up = bool(feats["handsUp"])
        # 与换道有关的横向量：把开合跳的对称张腿从横移里摘出去（§5.5 第 1/2 条，
        # 默认关，见 ActionConfig.laneBlockWhenHandsUp）
        lane_for_trigger = 0.0 if (cfg.laneBlockWhenHandsUp and hands_up) else lane

        airborne_ok = (not cfg.laneRequireAirborne) or (rise >= cfg.laneAirborneMin)
        lane_cond = (self._transient_armed
                     and not self._lane_slow
                     and abs(lane_for_trigger) >= cfg.laneOn
                     and airborne_ok)
        self._lane_hold.update(lane_cond)
        if lane_cond:
            self._lane_hold.mark(t)

        # 跳：rise 够大 **且横向没怎么动**（否则那是换道，不是原地跳）
        jump_cond = (self._transient_armed
                     and rise >= cfg.riseOn
                     and abs(lane) < cfg.jumpMaxLane)
        self._jump_hold.update(jump_cond)
        if jump_cond:
            self._jump_hold.mark(t)

        fired = None
        if self._lane_hold.ok(t, cfg.laneDwellMs, cfg.minSamples, self._dt_avg):
            # 慢速侧移：从离开中性到越阈花了太久 → 不是"换道"，是走过去/漂过去
            if (self._lane_since is not None
                    and (t - self._lane_since) > cfg.laneMaxMs):
                self._lane_slow = True
                self._lane_hold.reset()
            else:
                direction = "right" if lane_for_trigger > 0 else "left"
                fired = ActionEvent(KIND_LANE, t,
                                    _margin_conf(lane_for_trigger, cfg.laneOn),
                                    dir=direction)
        if fired is None and self._jump_hold.ok(t, cfg.jumpDwellMs,
                                                cfg.minSamples, self._dt_avg):
            fired = ActionEvent(KIND_JUMP, t,
                                _margin_conf(rise, cfg.riseOn))

        if fired is not None:
            events.append(fired)
            self._transient_armed = False
            self._neutral_hold.reset()
            self._lane_hold.reset()
            self._jump_hold.reset()

        self.block = self._block_reason(lane, rise, lane_cond, jump_cond,
                                        hands_up, knees_ok)
        return events

    # ------------------------------------------------------------ 内部
    def _reset_transients(self):
        """丢掉所有瞬时动作的计时器与武装状态（人在场变化时用）。"""
        self._lane_hold.reset()
        self._jump_hold.reset()
        self._neutral_hold.reset()
        self._transient_armed = False
        self._lane_since = None
        self._lane_slow = False

    def _block_reason(self, lane, rise, lane_cond, jump_cond, hands_up,
                      knees_ok) -> str:
        """HUD 用：现在到底卡在哪道门。别让人对着黑盒发呆。"""
        cfg = self.cfg
        if not self._transient_armed:
            return "回位门控中（要回到中性才算完一次）"
        near_lane = abs(lane) >= cfg.laneOn * 0.6
        near_jump = rise >= cfg.riseOn * 0.6
        if near_lane and not lane_cond:
            if self._lane_slow:
                return "移动太慢（判为走过去，不是换道）"
            if cfg.laneBlockWhenHandsUp and hands_up:
                return "双手过肩，换道被锁死"
            if cfg.laneRequireAirborne and rise < cfg.laneAirborneMin:
                return "横向够了但没有腾空"
            return "横向位移还不够"
        if near_jump and not jump_cond:
            if abs(lane) >= cfg.jumpMaxLane:
                return "横向挪太多（算换道，不算原地跳）"
            return "起跳高度还不够"
        return ""

    # ------------------------------------------------------------ 诊断
    @property
    def state(self) -> dict:
        """给 HUD / 调试用的一帧快照（纯读，无副作用）。"""
        cfg = self.cfg
        f = self.features or {}
        return {
            "present": self.present,
            "calibrated": self.calibrated,
            "posture": self.posture,
            "armed": self._transient_armed,
            "block": self.block,
            "dtAvgMs": round(self._dt_avg, 1),
            # 帧率太低时识别的下限就被摸到了（BoxingMate 同一条教训）。
            # GestureMate 默认 target fps 是 8，那个帧率下这四个动作都会很不稳。
            "lowFps": self._dt_avg > 100.0,
            "laneDx": round(f.get("laneDx", 0.0), 3),
            "laneDxImage": round(f.get("laneDxImage", 0.0), 3),
            "rise": round(f.get("rise", 0.0), 3),
            "sink": round(-f.get("rise", 0.0), 3),
            "kneeL": None if f.get("kneeL") is None else round(f["kneeL"], 1),
            "kneeR": None if f.get("kneeR") is None else round(f["kneeR"], 1),
            "kneeSource": f.get("kneeSource"),
            # 开了 requireKnees 却只有 2D 膝角 —— 那这道门几乎不可能过，
            # 界面上必须说出来，否则表现为"怎么蹲都不触发"，无从排查。
            "kneeWarn": bool(cfg.requireKnees and f.get("kneeSource") == "2d"),
            "handsUp": bool(f.get("handsUp", False)),
            "quality": round(f.get("quality", 0.0), 2),
            "mirroredInput": self.mirrored_input,
        }
