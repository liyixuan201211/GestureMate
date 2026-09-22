#!/usr/bin/env python3
"""动作识别（左移 / 右移 / 跳 / 蹲）的回归测试 —— 合成关键点，不需要模型权重。

    .venv/bin/python tests/test_action_detect.py           # 合成用例（快，确定性）
    .venv/bin/python tests/test_action_detect.py --real     # 附带真实素材的事实核查

为什么值得单独一个文件：这四个动作的判定**全是状态机 + 阈值**，
"看着像对"完全说明不了问题。所以这里把每一道门都造出来单独打：

  * 静止 3 秒不许产生任何动作（误触发是最要命的失败模式）
  * 蹲下是**锁存态**：进一次出一个事件，保持期间不许重复报
  * 蹲下期间屏蔽瞬时动作（§5.5 第 3 条）
  * 一次换道只出一个事件；**没回到中性不许再触发**（回位门控，§5.4c）
  * 慢速侧移不算换道（§5.4c 的"持续 < 0.6s"）
  * 横向跳算换道、不算原地跳（两个动作不能互相抢）
  * 单帧瞬移（跟踪丢失重捕）不许触发 —— 低帧率下最容易出的一类假动作
  * 镜像开关一翻，左右必须互换
  * 相机横滚 15° 不许把方向语义带歪（§5.3.3 的 rollResidual）
  * 正面下蹲时 2D 膝角恒为 180°：所以 `requireKnees` 默认关，
    而给了世界坐标（3D）之后同一道门又必须能过 —— 这是 §5.1 要求开
    `outputWorldLandmarks` 的**直接证据**，两条路都钉住

⚠️ 合成数据证明不了"真人身上准不准"（§5.6 的教训：拿 fixture 图得出"两个实现
完全一致"就下结论，而参考基准本身是错的）。这个文件钉的是**逻辑与门控**；
阈值合不合手只能真人在场调（见 README 的"待真人验证"一节）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from action import (KIND_JUMP, KIND_LANE, KIND_PLAYER, KIND_READY, KIND_SQUAT,
                    POSTURE_CROUCH, POSTURE_STAND, ActionConfig, ActionDetector,
                    ActionEvent, Calibrator, MedianFilter, OneEuroFilter,
                    compute_features, describe_event, knee_angle, parse_body,
                    torso_length)
# 只引测试真正用到的关节常量（其余的在 action 包里，不必全引）
from action.features import (P_L_ANKLE, P_L_HIP, P_L_KNEE, P_R_ANKLE, P_R_HIP,
                             P_R_KNEE)

# 合成几何**只写一份** —— 测试与 `python -m action --demo` 共用 action/synth.py。
# 本项目的教训：测试替身与真实实现各写一份，最后两边都不可信
# （BoxingMate 那句："测试和合成数据都用对象，而真实链路发的是数组"。）
from action.synth import (DT, TORSO, make_pose, make_world, rotate_pose,
                          scale_pose)


def hold(pose, ms, dt=DT):
    """把一个姿势（或返回姿势的函数）保持 ms 毫秒，返回逐帧列表。"""
    n = max(1, int(round(ms / dt)))
    return [pose() if callable(pose) else pose for _ in range(n)]


def feed(det, seq, t0=0.0, dt=DT):
    """逐帧喂给检测器。seq 的元素可以是 body，也可以是 (body, world)。"""
    out = []
    t = t0
    for item in seq:
        if isinstance(item, tuple):
            body, world = item
        else:
            body, world = item, None
        for ev in det.update(t, body, world):
            out.append((t, ev))
        t += dt
    return out


def actions(events):
    """只留四个动作的事件（PLAYER/READY 不算）。"""
    want = (KIND_SQUAT, KIND_LANE, KIND_JUMP)
    return [(t, ev) for (t, ev) in events if ev.kind in want]


def label(events):
    """把事件压成好读的短标签，断言失败时一眼能看出序列错在哪。"""
    out = []
    for _, ev in events:
        if ev.kind == KIND_SQUAT:
            out.append("SQUAT:ON" if ev.on else "SQUAT:OFF")
        elif ev.kind == KIND_LANE:
            out.append("LANE:" + ("R" if ev.dir == "right" else "L"))
        else:
            out.append(ev.kind)
    return out


def calibrate_basis(**kw):
    cal = Calibrator()
    for _ in range(20):
        cal.add(make_pose(**kw))
    return cal.finalize()


def make_detector(mirrored_input=True, warmup_ms=900.0, **cfg):
    """标定好的检测器。返回 (det, t)，t 是"可以开始做动作"的时刻。

    默认先喂一段"站直"，把**在场**和**回位门控**都放到就绪态。这一步不是可有可无：
    标定结束后在场还是 False，要连续 good 满 `presenceOnMs`(500ms) 才置位；
    而回位门控又要求"先在中性待够 120ms"。两步加起来约 620ms，
    这段时间里任何瞬时动作都不会触发 —— 不做预热的话，测试开头那段
    "站直"会被吃掉，表现为"第一个动作莫名其妙不报"。

    （这条规则本身是对的、也是实测要的：标定刚结束时不许拿当前位置直接兑现成
      一个动作。真机上标定期间人本来就站着，所以自然满足。）
    """
    det = ActionDetector(ActionConfig(**cfg), mirrored_input=mirrored_input)
    det.begin_calibration()
    t = 0.0
    for _ in range(40):
        det.update(t, make_pose())
        t += DT
    evs = det.finish_calibration(t)
    assert [e.kind for e in evs] == [KIND_READY], f"标定应产出 1 条 READY，实际 {evs}"
    if warmup_ms > 0:
        n = max(1, int(round(warmup_ms / DT)))
        feed(det, [make_pose()] * n, t0=t)
        t += n * DT
    return det, t


# ────────────────────────── 1. 测试替身本身 ──────────────────────────

def test_fixture_is_self_consistent():
    """先测"测试替身"——BoxingMate 就是栽在替身与真实接口不一致上。

    如果这个合成姿势的几何不是自洽的（膝角对不上、lane 的刻度不对），
    后面所有断言都只是在验证一个错误的模型。
    """
    # 躯干长度精确等于 TORSO（后面的归一化刻度全靠它）
    assert abs(torso_length(make_pose()) - TORSO) < 1e-9

    # 想摆多少度膝角，量出来就该是多少度（两条腿各查一遍）
    for deg in (180.0, 150.0, 120.0, 90.0):
        p = make_pose(knee_deg=deg)
        got_l = knee_angle(p[P_L_HIP], p[P_L_KNEE], p[P_L_ANKLE])
        got_r = knee_angle(p[P_R_HIP], p[P_R_KNEE], p[P_R_ANKLE])
        assert abs(got_l - deg) < 0.5 and abs(got_r - deg) < 0.5, \
            f"膝角摆 {deg}° 量出 {got_l:.1f}/{got_r:.1f}"
    print("PASS 1: 合成姿势自洽（躯干=%.2f，膝角 180/150/120/90 都能精确还原）"
          % TORSO)


def test_fixture_scale_is_physical():
    """lane / rise 的刻度，以及"屈膝必然压低髋"这条物理关系。"""
    b = calibrate_basis()
    f = compute_features(b, make_pose(lane=0.5))
    assert abs(f["laneDx"] - 0.5) < 1e-6, f"laneDx={f['laneDx']}"
    f = compute_features(b, make_pose(rise=0.3))
    assert abs(f["rise"] - 0.3) < 1e-6, f"rise={f['rise']}"

    # 屈膝 = 髋变低 = sink > 0，而且 sink 与 rise 必须严格反号（§5.2 是同一根轴）
    f = compute_features(b, make_pose(knee_deg=110.0))
    assert f["sink"] > 0.28, f"屈膝到 110° 应产生 sink>0.28，实际 {f['sink']:.3f}"
    assert abs(f["sink"] + f["rise"]) < 1e-9, "sink 必须是 rise 的相反数"
    print("PASS 2: lane/rise 刻度正确；屈膝 110° → sink=%.3f（= −rise）"
          % f["sink"])


# ────────────────────────── 2. 关键点格式 ──────────────────────────

def test_parse_body_accepts_every_real_format():
    """对象 / 字典 / 数组 / NormalizedLandmarkList 四种都要吃。

    本项目最大的 bug 来源就是"某一层只认一种格式"（BoxingMate 的
    "接上真摄像头一拳都认不出来"）。**数组是链路的真实格式**，必须能过。
    """
    base = make_pose()
    assert len(parse_body(base)) == 33                       # [[x,y,z], ...] 真实格式
    assert len(parse_body([[x, y, z] for (x, y, z) in base])) == 33

    class Obj:
        def __init__(self, t):
            self.x, self.y, self.z = t

    assert len(parse_body([Obj(t) for t in base])) == 33      # MediaPipe 原始对象
    assert len(parse_body([{"x": x, "y": y, "z": z}
                           for (x, y, z) in base])) == 33     # 字典

    class Wrap:
        landmark = base

    assert len(parse_body(Wrap())) == 33                      # NormalizedLandmarkList
    assert parse_body(None) is None
    assert parse_body([]) is None
    assert parse_body(base[:28]) is None, "少一个踝(28)就该判不可用"
    bad = list(base)
    bad[0] = (float("nan"), 0.5, 0.0)
    assert parse_body(bad) is None, "NaN 必须当不可用，不能悄悄传下去"
    # 对象也可以直接喂给 compute_features（不必先自己转格式）
    b = calibrate_basis()
    assert compute_features(b, [Obj(t) for t in base]) is not None
    print("PASS 3: 关键点四种格式都能读；NaN / 点数不足被挡住")


# ────────────────────────── 3. 几何：2D vs 3D 膝角 ──────────────────────────

def test_knee_angle_2d_vs_3d():
    """正面蹲：2D 投影角恒 180°，3D 才看得出弯 —— 这是 requireKnees 默认关的根据。"""
    w = make_world(110.0)
    k3 = knee_angle(w[P_L_HIP], w[P_L_KNEE], w[P_L_ANKLE])
    assert abs(k3 - 110.0) < 0.5, f"3D 膝角应为 110°，实际 {k3:.1f}"

    xy = [(x, y) for (x, y, _z) in w]
    k2 = knee_angle(xy[P_L_HIP], xy[P_L_KNEE], xy[P_L_ANKLE])
    assert abs(k2 - 180.0) < 0.5, \
        f"正面蹲的 2D 投影本来就该共线(=180°)，实际 {k2:.1f}"

    # 退化输入不能炸
    assert knee_angle((0, 0), (0, 0), (0, 1)) is None
    print(f"PASS 4: 同一记正面下蹲 —— 3D 膝角 {k3:.1f}°，2D 投影角 {k2:.1f}°")


# ────────────────────────── 4. 相机横滚 / 远近 ──────────────────────────

def test_camera_roll_does_not_leak_into_lane():
    """相机歪 15°：横向/竖直语义不许跟着歪（§5.3.3 的 rollResidual）。"""
    cal = Calibrator()
    for _ in range(20):
        cal.add(rotate_pose(make_pose(), 15.0))
    b = cal.finalize()
    assert abs(abs(b.roll_deg) - 15.0) < 1.0, f"应记下 15° 歪斜，实际 {b.roll_deg:.2f}"

    f = compute_features(b, rotate_pose(make_pose(lane=0.5), 15.0))
    assert abs(f["laneDx"] - 0.5) < 0.02, f"横移被歪斜污染：{f['laneDx']:.3f}"
    assert abs(f["rise"]) < 0.02, f"横移不该串到竖直：rise={f['rise']:.3f}"
    print(f"PASS 5: 相机横滚 15°（rollResidual={b.roll_deg:+.1f}°）后 "
          f"laneDx={f['laneDx']:.3f}、rise={f['rise']:+.3f}，语义没被带歪")


def test_scale_tracks_distance_but_is_clamped():
    """走近走远不该改变归一化特征；关键点崩掉时也要被夹住。"""
    b = calibrate_basis()
    f = compute_features(b, scale_pose(make_pose(lane=0.4), 1.5))
    assert abs(f["laneDx"] - 0.4) < 0.02, f"走近 1.5 倍后 laneDx={f['laneDx']:.3f}"

    # 极端：整帧缩到 1/5（等价于关键点崩了）——尺度必须被夹在下限
    f2 = compute_features(b, scale_pose(make_pose(lane=0.4), 0.2))
    assert f2["scaleUsed"] >= b.torso * 0.35 - 1e-9, \
        f"尺度没被夹住：{f2['scaleUsed']:.4f} < {b.torso * 0.35:.4f}"
    print(f"PASS 6: 走近 1.5 倍 laneDx 仍是 {f['laneDx']:.3f}；"
          f"缩到 1/5 时尺度被夹在 {f2['scaleUsed']:.4f}")


# ────────────────────────── 5. 滤波器 ──────────────────────────

def test_median_filter_kills_single_spike():
    mf = MedianFilter(5)
    out = [mf(v) for v in (0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0)]
    assert max(abs(v) for v in out) == 0.0, f"单帧尖峰没被干掉：{out}"
    print("PASS 7: 中值滤波把单帧尖峰 5.0 完全抹掉")


def test_one_euro_smooths_jitter_but_follows_fast_moves():
    """1€ 的核心卖点：慢时重滤（压抖）、快时轻滤（跟手）。两半都要钉住。"""
    f = OneEuroFilter(fc=1.0, beta=0.007)
    t, outs = 0.0, []
    for i in range(90):
        outs.append(f(t, 0.01 if i % 2 else -0.01))   # 输入峰峰值 0.02
        t += 1.0 / 30.0
    tail = outs[30:]
    jitter = max(tail) - min(tail)
    assert jitter < 0.01, f"±1cm 抖动应被压到 0.01 以内，实际 {jitter:.4f}"

    g = OneEuroFilter(fc=1.0, beta=0.007)
    t = 0.0
    for _ in range(30):
        g(t, 0.0)
        t += 1.0 / 30.0
    y = 0.0
    for i in range(15):                                # 0 -> 1.0 的快速位移
        y = g(t, min(1.0, (i + 1) / 5.0))
        t += 1.0 / 30.0
    assert y > 0.6, f"快速动作必须跟得上，实际只到 {y:.3f}"
    print(f"PASS 8: 1€ 把 ±1cm 抖动压到 {jitter:.4f}，同时快速位移能跟到 {y:.2f}")


# ────────────────────────── 6. 标定 ──────────────────────────

def test_calibration_refuses_to_invent_numbers():
    """样本不够 / 数据太差时返回 None —— 宁可不标定，也不编一个基线。"""
    cal = Calibrator(min_samples=8)
    for _ in range(3):
        cal.add(make_pose())
    assert cal.finalize() is None, "只有 3 帧就该拒绝标定"

    cal = Calibrator(min_samples=8)
    for _ in range(20):
        cal.add(None)
    assert cal.finalize() is None and cal.rejected == 20

    # 标定期间人没站住 → 标记为不可信
    cal = Calibrator(min_samples=8)
    for i in range(20):
        cal.add(make_pose(lane=0.25 * (i % 2)))        # 左右乱晃
    b = cal.finalize()
    assert b is not None and not b.stable, \
        f"标定时乱动应标为不可信，实际 spread={b.hip_spread:.3f}"

    # 站得稳 → 可信
    b2 = calibrate_basis()
    assert b2.stable and b2.samples == 20
    print(f"PASS 9: 标定拒绝编数字（3 帧 → None；乱动 spread={b.hip_spread:.3f} "
          f"不可信；站稳 spread={b2.hip_spread:.3f} 可信）")


# ────────────────────────── 7. 四个动作 ──────────────────────────

def test_idle_produces_no_actions():
    """最重要的一条：站着不动 3 秒，一个动作都不许报。"""
    det, t = make_detector()
    evs = actions(feed(det, hold(make_pose(), 3000), t0=t))
    assert evs == [], f"静止却报了 {label(evs)}"
    print("PASS 10: 静止 3 秒（约 90 帧）零动作 —— 没有误触发")


def test_squat_is_latched_and_edge_triggered():
    """蹲下是锁存态：进一次出一个事件，保持期间不许重复报（R6）。"""
    det, t = make_detector()
    seq = hold(make_pose(), 400)
    seq += hold(lambda: make_pose(knee_deg=110.0), 1500)   # 蹲下并保持
    seq += hold(make_pose(), 1500)                          # 站起来
    evs = actions(feed(det, seq, t0=t))
    assert label(evs) == ["SQUAT:ON", "SQUAT:OFF"], f"实际 {label(evs)}"
    print(f"PASS 11: 蹲下锁存正确 —— {label(evs)}（保持 1.5s 期间没有重复报）")


def test_crouch_blocks_transient_actions():
    """§5.5 第 3 条：蹲下期间屏蔽一切瞬时动作。"""
    det, t = make_detector()
    deep = lambda **kw: make_pose(knee_deg=89.0, **kw)     # sink≈0.6 的深蹲
    seq = hold(deep, 1200)                                  # 先蹲下（锁存）
    seq += hold(lambda: deep(lane=0.9), 500)                # 蹲着往旁边挪
    seq += hold(lambda: deep(rise=0.20), 500)               # 蹲着向上窜（sink 仍够）
    evs = actions(feed(det, seq, t0=t))
    assert label(evs) == ["SQUAT:ON"], f"蹲下期间不该有瞬时动作，实际 {label(evs)}"
    print(f"PASS 12: 蹲下期间横移与上窜都被屏蔽 —— 只有 {label(evs)}")


def test_lane_left_and_right():
    """左右方向，以及"一次换道 = 一个事件"。"""
    det, t = make_detector()
    seq = hold(make_pose(), 400)
    seq += hold(lambda: make_pose(lane=0.6), 500)          # 画面 +x
    seq += hold(make_pose(), 800)                           # 回位
    seq += hold(lambda: make_pose(lane=-0.6), 500)          # 画面 −x
    seq += hold(make_pose(), 800)
    evs = actions(feed(det, seq, t0=t))
    assert label(evs) == ["LANE:R", "LANE:L"], f"实际 {label(evs)}"
    print(f"PASS 13: 左移/右移各命中一次 —— {label(evs)}")


def test_mirror_switch_flips_direction():
    """镜像开关一翻，左右必须互换。真人验证左右反了，就靠它一处修正（§5.7）。"""
    got = {}
    for mir in (True, False):
        det, t = make_detector(mirrored_input=mir)
        seq = hold(make_pose(), 400) + hold(lambda: make_pose(lane=0.6), 500)
        evs = actions(feed(det, seq, t0=t))
        assert len(evs) == 1, f"mirror={mir} 应恰好命中一次，实际 {label(evs)}"
        got[mir] = evs[0][1].dir
    assert got[True] != got[False], f"镜像开关没起作用：{got}"
    assert (got[True], got[False]) == ("right", "left"), got
    print(f"PASS 14: mirror=True → {got[True]}，mirror=False → {got[False]}（互换）")


def test_lane_needs_return_to_neutral():
    """§5.4c 的回位门控：没回到中性，不许出第二次换道。"""
    det, t = make_detector()
    seq = hold(make_pose(), 400)
    seq += hold(lambda: make_pose(lane=0.6), 600)           # 第一次换道
    seq += hold(lambda: make_pose(lane=0.75), 1500)         # 一直待在右边、还更远
    evs = actions(feed(det, seq, t0=t))
    assert label(evs) == ["LANE:R"], f"没回中性却报了 {label(evs)}"
    print("PASS 15: 没回到中性就不重复触发（回位门控生效）")


def test_slow_drift_is_not_a_lane_change():
    """§5.4c 的"持续 < 0.6s"：慢慢挪过去算走路，不算换道。"""
    det, t = make_detector()
    seq = hold(make_pose(), 400)
    seq += [make_pose(lane=0.36 * (i + 1) / 40) for i in range(40)]  # 1.3s 慢慢挪
    seq += hold(lambda: make_pose(lane=0.36), 500)          # 保持住：dwell 够了
    evs = actions(feed(det, seq, t0=t))
    assert evs == [], f"慢速侧移不该算换道，实际 {label(evs)}"
    print("PASS 16: 1.3 秒的慢速侧移被识别为「走过去」而不是换道")


def test_jump_fires_once():
    det, t = make_detector()
    seq = hold(make_pose(), 400)
    seq += hold(lambda: make_pose(rise=0.30), 400)          # 起跳
    seq += hold(make_pose(), 900)                            # 落地
    evs = actions(feed(det, seq, t0=t))
    assert label(evs) == ["JUMP"], f"实际 {label(evs)}"
    print(f"PASS 17: 起跳命中一次 —— {label(evs)}")


def test_lateral_jump_reports_lane_not_jump():
    """横向跳：算换道，不算原地跳（两个动作不能互相抢）。"""
    det, t = make_detector()
    seq = hold(make_pose(), 400)
    seq += hold(lambda: make_pose(lane=0.8, rise=0.30), 500)
    seq += hold(make_pose(), 900)
    evs = actions(feed(det, seq, t0=t))
    assert label(evs) == ["LANE:R"], f"横向跳应为 LANE，实际 {label(evs)}"
    print(f"PASS 18: 横向起跳报成换道而不是跳 —— {label(evs)}")


def test_single_frame_spike_does_not_fire():
    """单帧瞬移（跟踪丢失重捕/画面切人）不许触发 —— 低帧率下最常见的假动作。"""
    det, t = make_detector()
    seq = hold(make_pose(), 400)
    seq += [make_pose(lane=3.0)]                            # 一帧瞬移 3 个躯干
    seq += hold(make_pose(), 900)
    evs = actions(feed(det, seq, t0=t))
    assert evs == [], f"单帧跳变不该触发，实际 {label(evs)}"
    print("PASS 19: 单帧瞬移 3 个躯干被挡掉（中值 + minSamples 两道门）")


def test_frontal_squat_knee_gate():
    """正面下蹲：`requireKnees` 默认必须关；给了世界坐标才该打开。

    这是本次实现里唯一一处**有意偏离** §5.4a/§5.6 的地方，所以钉得最细。
    """
    b = calibrate_basis()
    frontal = make_pose(knee_deg=110.0, frontal=True)
    f = compute_features(b, frontal)
    assert f["sink"] > 0.28, f"正面蹲的髋确实沉下去了：sink={f['sink']:.3f}"
    assert abs(f["kneeL"] - 180.0) < 1.0, \
        f"正面蹲的 2D 膝角本来就该是 180°，实际 {f['kneeL']:.1f}"

    # (a) 默认 requireKnees=False + 只有 2D → 判得出来
    det, t = make_detector()
    seq = hold(make_pose(), 400) + hold(lambda: frontal, 1200)
    evs = actions(feed(det, seq, t0=t))
    assert label(evs) == ["SQUAT:ON"], f"默认配置应判出蹲，实际 {label(evs)}"

    # (b) requireKnees=True + 只有 2D → **判不出来**（这就是它默认关的原因）
    det2, t2 = make_detector(requireKnees=True)
    evs2 = actions(feed(det2, hold(make_pose(), 400) + hold(lambda: frontal, 1200),
                        t0=t2))
    assert evs2 == [], f"2D 下 knee 门不该过，实际 {label(evs2)}"

    # (c) requireKnees=True + 世界坐标(3D) → 判得出来（§5.1 推荐的那条路）
    det3, t3 = make_detector(requireKnees=True)
    seq3 = hold(make_pose(), 400)
    seq3 += hold(lambda: (frontal, make_world(110.0)), 1200)
    evs3 = actions(feed(det3, seq3, t0=t3))
    assert label(evs3) == ["SQUAT:ON"], f"有 3D 膝角时应判出蹲，实际 {label(evs3)}"

    st = det3.state
    assert st["kneeSource"] == "3d" and st["kneeWarn"] is False
    print("PASS 20: 正面蹲 —— 2D 靠 sink 可判(默认)；requireKnees 在 2D 下判不出；"
          "开世界坐标后同一条门能过（kneeSource=3d）")


# ────────────────────────── 8. 在场 / 契约 / 不变式 ──────────────────────────

def test_presence_events():
    # 不预热：这条用例要看的就是"在场"从无到有的那几次跳变
    det, t = make_detector(warmup_ms=0)
    seq = hold(make_pose(), 700)          # 人出现
    seq += hold(None, 900)                # 人走了
    seq += hold(make_pose(), 900)         # 人回来
    evs = feed(det, seq, t0=t)
    got = [(e.kind, e.present) for _, e in evs if e.kind == KIND_PLAYER]
    assert got == [(KIND_PLAYER, True), (KIND_PLAYER, False),
                   (KIND_PLAYER, True)], got
    print("PASS 21: 在场事件正确 —— 出现/离开/回来各一次")


def test_no_actions_before_calibration():
    """没标定就一个动作都不许出（避免"用一个假的基线"乱报）。"""
    det = ActionDetector(ActionConfig())
    evs = actions(feed(det, hold(lambda: make_pose(lane=2.0), 800), t0=0.0))
    assert evs == [], f"未标定却报了 {label(evs)}"
    assert det.state["block"] == "未标定"
    print("PASS 22: 未标定时零动作，且 block 说明是「未标定」")


def test_event_dict_and_text():
    """事件序列化/描述的形状（OSC 与 JSONL 都吃它）。"""
    e = ActionEvent(KIND_LANE, 1234.5, 0.75, dir="left")
    d = e.to_dict()
    assert d["kind"] == KIND_LANE and d["t"] == 1234.5 and d["conf"] == 0.75
    assert d["dir"] == "left" and "on" not in d and "present" not in d
    assert "左移" in describe_event(e)

    s = ActionEvent(KIND_SQUAT, 10.0, 1.0, on=False)
    assert s.to_dict()["on"] is False and "站起" in describe_event(s)
    j = ActionEvent(KIND_JUMP, 10.0, 1.0)
    assert "跳" in describe_event(j)
    assert "on" not in j.to_dict()
    print("PASS 23: 事件 to_dict/describe_event 形状正确（None 字段会被剔掉）")


def test_threshold_invariants():
    """几条必须成立的关系 —— 破了就说明阈值被改出了逻辑矛盾。"""
    c = ActionConfig()
    assert c.jumpMaxLane < c.laneOn, \
        "跳的横向上限必须小于换道阈值，否则两个动作会互相抢"
    assert c.laneNeutral < c.laneOn, "回中性的判据必须低于触发阈值"
    assert c.sinkOff < c.sinkOn, "蹲下退出阈值必须低于进入阈值（迟滞）"
    assert c.riseOff < c.riseOn, "跳的迟滞下沿必须低于上沿"
    assert c.laneAirborneMin <= c.riseOn
    assert c.minSamples >= 2, "至少 2 个样本，否则单帧尖峰能直接触发"
    print("PASS 24: 阈值不变式成立（迟滞方向、互斥关系、minSamples≥2）")


# ────────────────────────── 9. 真实素材（可选） ──────────────────────────

def test_unstable_calibration_is_refused():
    """标定时人没站住 → 必须**拒绝**标定，而不是拿一个乱动的基线去判定。

    实测抓到的缺陷：第一版算了 `basis.stable` 却没用它，于是接受了
    spread=0.37 躯干的脏基线（"中性"本身都没有意义了），跑真实素材立刻误报。
    这里把"宁可不标定，也不用假基线"钉住。
    """
    det = ActionDetector(ActionConfig())
    det.begin_calibration()
    t = 0.0
    for i in range(40):
        det.update(t, make_pose(lane=0.25 * (i % 2)))     # 标定期间左右乱晃
        t += DT
    evs = det.finish_calibration(t)
    assert evs == [] and not det.calibrated, "乱动的标定必须被拒绝"
    assert "标定不可信" in det.block, f"拒绝的理由没说出来：{det.block!r}"

    # 显式开关可以强行放行（仅供调试比对）
    det2 = ActionDetector(ActionConfig(allowUnstableCalibration=True))
    det2.begin_calibration()
    t = 0.0
    for i in range(40):
        det2.update(t, make_pose(lane=0.25 * (i % 2)))
        t += DT
    assert [e.kind for e in det2.finish_calibration(t)] == [KIND_READY], \
        "显式开关应该放行"
    print("PASS 25: 乱动的标定默认被拒绝（block 说明原因）；显式开关才放行")


def test_tracking_gap_is_not_an_action():
    """关键点断了几帧之后重新捕获，不许被读成一次动作。

    实测抓到的缺陷：断帧期间滤波器仍留着旧样本，重新捕获时位置已经差了很多，
    于是被读成一次极快的位移 —— 真实素材上就是这样误报了一次「跳」
    （断 2 帧后 rise 一帧跳到 +0.72）。
    """
    det, t = make_detector()
    seq = hold(make_pose(), 400)
    seq += hold(None, 3)                                    # 断 3 帧
    seq += hold(lambda: make_pose(rise=0.5), 700)           # 回来时人"高了一截"
    evs = actions(feed(det, seq, t0=t))
    assert evs == [], f"断帧后重新捕获不该算动作，实际 {label(evs)}"
    print("PASS 26: 断 3 帧后重新捕获（位置差 0.5 躯干）不产生动作")


def test_webui_state_contract():
    """WebUI 前端从 `state.action` 里读的每个键，服务端都必须真的给。

    这条是被**抓出来**的：前端读 `a.calibrating` 来显示标定倒计时，而
    `ActionDetector.state` 里当时根本没有这个键 —— 界面上标定期间会静默地
    错显示成「未标定」。JS 读不存在的字段不报错、只变 undefined，
    所以只有做这种交叉检查才抓得到（截图也未必看得出）。
    """
    import re
    js_path = os.path.join(ROOT, "webui/static/app.js")
    if not os.path.exists(js_path):
        print("SKIP: 没有 webui/static/app.js")
        return
    js = open(js_path, encoding="utf-8").read()
    m = re.search(r"function updateActionPanel\(s\)\s*\{(.*?)\n\}", js, re.S)
    assert m, "app.js 里找不到 updateActionPanel（改名了？）"
    read = set(re.findall(r"\ba\.([A-Za-z_][A-Za-z0-9_]*)", m.group(1)))
    provided = set(ActionDetector(ActionConfig()).state.keys())
    provided |= {"events", "calibLeftMs"}       # 引擎额外塞进去的两个
    missing = sorted(read - provided)
    assert not missing, f"前端读了、但服务端 action state 里没有的键：{missing}"
    print(f"PASS 28: WebUI 状态契约成立 —— 前端读的 {len(read)} 个键服务端全都提供")


def test_real_footage_no_false_actions():
    """真实素材：① 脏标定必须被拒绝；② 就算强行用脏标定，也不许误报动作。

    `tests/fixtures/hands_gestures.mp4` 是**真人 + 硬切镜头 + 手部特写**，
    全身关键点抖得厉害（实测：标定期髋部抖动 0.37 躯干、躯干长度在
    0.17~0.48 之间变、髋部竖直位移折合 1.57 个躯干）—— 正是误触发最容易
    发生的那类素材。这条用例**是真的抓出了两个缺陷**，不是设想的：

      · 缺陷一：算了 `basis.stable` 却没用 → 接受了 spread=0.37 的脏基线；
      · 缺陷二：关键点缺失的帧没清滤波状态 → 重新捕获时误报了一次「跳」。

    两个都修了（见 detector.py 的 calibMaxSpread 与断帧处理），这里两条都钉住。
    """
    import cv2

    from LandmarkEngine import LandmarkEngine

    vid = os.path.join(ROOT, "tests/fixtures/hands_gestures.mp4")
    if not os.path.exists(vid):
        print("SKIP: 没有 tests/fixtures/hands_gestures.mp4")
        return

    def as_lists(lms):
        return None if lms is None else [[p.x, p.y, p.z] for p in lms.landmark]

    # ⚠️ 必须用**和 CLI 完全相同的引擎与帧率**（LandmarkEngine + 真实 fps），
    # 否则测的就不是真实路径 —— "测试替身与真实实现不一致"是本项目的头号教训。
    # （这一条是被打脸补上的：初版这里用 raw holistic/complexity=0/固定 33ms，
    #   于是"强制脏基线零误报"通过了；换成 CLI 的引擎后同素材仍会误报一次。）
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1000.0 / fps if fps > 0 else 33.0
    frames = []
    eng = LandmarkEngine(model_complexity=1, need=("body",))
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:, ::-1, :]                 # 与 TaskController 一致
            r = eng.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append((as_lists(r.pose_landmarks),
                           as_lists(getattr(r, "pose_world_landmarks", None))))
    finally:
        cap.release()
        eng.close()
    assert len(frames) > 40, f"素材太短（{len(frames)} 帧）"

    def replay(cfg):
        det = ActionDetector(cfg)
        det.begin_calibration()
        t, acts = 0.0, []
        for i, (pose, world) in enumerate(frames):
            t += dt
            if i < 30:
                det.update(t, pose)
                if i == 29:
                    det.finish_calibration(t)
                continue
            if not det.calibrated:
                continue
            for ev in det.update(t, pose, world):
                if ev.kind in (KIND_SQUAT, KIND_LANE, KIND_JUMP):
                    acts.append(ev)
        return det, acts

    # ① 这段素材根本标不出可信基线 → 必须拒绝，并说清原因
    det, acts = replay(ActionConfig())
    assert det.basis is None, "这种抖动的素材不该标定成功"
    assert acts == [], f"没标定却报了动作：{[describe_event(e) for e in acts]}"
    assert "标定不可信" in det.block, f"拒绝的理由没说出来：{det.block!r}"

    # ② 就算强行用脏基线，也不许误报
    det2, acts2 = replay(ActionConfig(allowUnstableCalibration=True))
    assert det2.calibrated, "强制开关应让它标定成功（用来验证第②条）"
    assert acts2 == [], (f"强用脏基线仍不该误报，实际 "
                         f"{[describe_event(e) for e in acts2]}")
    print(f"PASS 27: 真实素材 {len(frames)} 帧@{fps:.0f}fps（真人+硬切+特写，"
          f"与 CLI 同引擎）—— 脏标定被拒绝且零动作；强用脏基线也零动作")


# ────────────────────────── 主入口 ──────────────────────────

if __name__ == "__main__":
    test_fixture_is_self_consistent()
    test_fixture_scale_is_physical()
    test_parse_body_accepts_every_real_format()
    test_knee_angle_2d_vs_3d()
    test_camera_roll_does_not_leak_into_lane()
    test_scale_tracks_distance_but_is_clamped()
    test_median_filter_kills_single_spike()
    test_one_euro_smooths_jitter_but_follows_fast_moves()
    test_calibration_refuses_to_invent_numbers()
    test_idle_produces_no_actions()
    test_squat_is_latched_and_edge_triggered()
    test_crouch_blocks_transient_actions()
    test_lane_left_and_right()
    test_mirror_switch_flips_direction()
    test_lane_needs_return_to_neutral()
    test_slow_drift_is_not_a_lane_change()
    test_jump_fires_once()
    test_lateral_jump_reports_lane_not_jump()
    test_single_frame_spike_does_not_fire()
    test_frontal_squat_knee_gate()
    test_presence_events()
    test_no_actions_before_calibration()
    test_event_dict_and_text()
    test_threshold_invariants()
    test_unstable_calibration_is_refused()
    test_tracking_gap_is_not_an_action()
    if "--real" in sys.argv:
        test_real_footage_no_false_actions()
    else:
        print("（真实素材核查未跑；加 --real 打开）")
    test_webui_state_contract()
    print("\n全部通过。")
