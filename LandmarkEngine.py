"""地标引擎封装 —— 统一 Holistic 与「按需组合」两条路线。

背景
----
GestureMate 原本固定使用 `mediapipe.python.solutions.holistic.Holistic`。
Holistic 是一个「全家桶」图，每帧都会跑人脸检测/FaceMesh(468 点)、
BlazePose(33 点)、手掌检测与手部 21 点 x2。

一个很自然的优化想法是：官方两个 demo 只声明了 `leftHand` / `rightHand`，
既然用不到 face/body，那就别跑，只留手部模型 —— 于是本模块提供了
`mode="minimal"`。

**但实测把这条路否掉了**（M2 Max / mediapipe 0.10.14，见 tests/bench_engine.py）：

    方案                单帧推理    左右手任一检出
    holistic (原版)      28.6 ms     93 / 120
    只跑 hands           32.1 ms     84 / 120     ← 更慢，而且检出更少

原因：Holistic 会用**人体姿态**推出手部 ROI 再做手部关键点，而独立的
`hands.Hands` 每帧都要在整张图上跑一遍手掌检测。少了人脸/人体，却多了
全图手掌检测，净亏。

因此本模块的默认 `mode` 是 **"holistic"**（与上游行为一致），
"minimal" 只作为可选项保留，方便在别的机型/别的 mediapipe 版本上复测 ——
换环境后请先跑 tests/bench_engine.py 再决定。

对外它返回一个与 Holistic 结果**同构**的对象
（`face_landmarks` / `pose_landmarks` / `left_hand_landmarks` / `right_hand_landmarks`），
所以 `Utils.drawLandmarks`、`Utils.extractLandmarks` 以及全部 Task 都不用改。

镜像与左右手
------------
`TaskController` 在推理前把画面水平翻转（自拍视角）。MediaPipe 的手性判定
按「输入图里看起来是哪只手」给出标签，翻转后标签与真手相反。原代码正是靠
`extractLandmarks` 里的 left↔right 互换来抵消这件事。本模块沿用完全相同的
约定：把 MediaPipe 标为 `Left` 的手放进 `left_hand_landmarks`（随后由
`extractLandmarks` 换到 `rightHand`），因此**最终语义与原来一致**。
"""
from types import SimpleNamespace

import mediapipe.python.solutions as sol

FACE = "face"
BODY = "body"
LEFT = "leftHand"
RIGHT = "rightHand"
ALL_PARTS = (FACE, BODY, LEFT, RIGHT)

DEFAULT_DETECTION_CONFIDENCE = 0.5
DEFAULT_TRACKING_CONFIDENCE = 0.5


class LandmarkResult(SimpleNamespace):
    """与 mediapipe holistic 结果同构的容器（只带四个属性）。"""


class LandmarkEngine:
    """只实例化配置真正需要的 MediaPipe 模型。

    参数
    ----
    model_complexity : 0/1/2，与原来的 `--complexity` 同义。
    need             : 需要的部位集合，取值见 ALL_PARTS；为空表示「全都要」。
    """

    def __init__(self,
                 model_complexity=1,
                 need=(),
                 mode="auto",
                 min_detection_confidence=DEFAULT_DETECTION_CONFIDENCE,
                 min_tracking_confidence=DEFAULT_TRACKING_CONFIDENCE):
        need = set(need) if need else set(ALL_PARTS)
        self.need = need
        self.need_face = FACE in need
        self.need_body = BODY in need
        self.need_hands = bool({LEFT, RIGHT} & need)
        self.model_complexity = model_complexity

        # mode:
        #   "holistic" —— 一律用 holistic 全家桶（**默认，实测最快最准**）
        #   "minimal"  —— 只加载需要的模型（省掉人脸/人体）
        #   "auto"     —— 需要 face 时用 holistic，否则用 minimal
        #
        # 为什么要默认 holistic？见本文件顶部以及 tests/bench_engine.py 的实测：
        # 在 M2 Max + mediapipe 0.10.14 上，holistic 28.6ms/帧、左右手任一检出
        # 93/120；只跑 hands 反而 32.1ms/帧、检出 84/120。原因是 holistic 会用
        # 人体姿态推出双手 ROI，而独立的 hands 方案每帧都要在全图上跑手掌检测。
        # 也就是说「少跑人脸/人体」在这台机器上是**负优化**，因此默认关闭。
        self.mode_requested = mode
        if mode == "auto":
            mode = "holistic" if self.need_face else "minimal"
        self._use_holistic = (mode == "holistic")
        self.mode = "holistic" if self._use_holistic else "composed"

        self._holistic = None
        self._pose = None
        self._hands = None

        if self._use_holistic:
            self._holistic = sol.holistic.Holistic(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=model_complexity)
        else:
            if self.need_body:
                self._pose = sol.pose.Pose(
                    static_image_mode=False,
                    model_complexity=model_complexity,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence)
            if self.need_hands:
                # holistic 的手模型等价参数：model_complexity 只能是 0/1
                self._hands = sol.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=min(int(model_complexity), 1),
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence)

    # ------------------------------------------------------------ 推理
    def process(self, rgb):
        """rgb: HxWx3 uint8 RGB。返回 LandmarkResult。"""
        if self._holistic is not None:
            r = self._holistic.process(rgb)
            return LandmarkResult(face_landmarks=r.face_landmarks,
                                  pose_landmarks=r.pose_landmarks,
                                  left_hand_landmarks=r.left_hand_landmarks,
                                  right_hand_landmarks=r.right_hand_landmarks)

        face = None
        pose = None
        if self._pose is not None:
            pose = self._pose.process(rgb).pose_landmarks
        left = right = None
        if self._hands is not None:
            left, right = self._split_hands(self._hands.process(rgb))
        return LandmarkResult(face_landmarks=face,
                              pose_landmarks=pose,
                              left_hand_landmarks=left,
                              right_hand_landmarks=right)

    @staticmethod
    def _split_hands(hand_result):
        """把 multi_hand_landmarks 按 handedness 拆成 (left, right)。

        与 holistic 的输出约定保持一致：标签 `Left` 即 holistic 的
        `left_hand_landmarks`。若模型没给出 handedness，则退化为按出现顺序分配。
        """
        hands = getattr(hand_result, "multi_hand_landmarks", None)
        if not hands:
            return None, None
        handedness = getattr(hand_result, "multi_handedness", None) or []
        left = right = None
        for i, lm in enumerate(hands):
            label = ""
            if i < len(handedness):
                cls = getattr(handedness[i], "classification", None)
                if cls:
                    label = cls[0].label or ""
            if label == "Left" and left is None:
                left = lm
            elif label == "Right" and right is None:
                right = lm
        if left is None and right is None:
            left = hands[0]
            if len(hands) > 1:
                right = hands[1]
        return left, right

    # ------------------------------------------------------------ 生命周期
    def close(self):
        for obj in (self._holistic, self._pose, self._hands):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._holistic = self._pose = self._hands = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ 描述
    def describe(self):
        return (f"engine={self.mode} "
                f"(face={'on' if self.need_face else 'off'}, "
                f"body={'on' if self.need_body else 'off'}, "
                f"hands={'on' if self.need_hands else 'off'}, "
                f"complexity={self.model_complexity})")


def parts_from_config(config):
    """扫描任务配置，返回真正被引用的部位集合。

    支持 detect 的 `bodyPart: ["leftHand"]` 与 match 的
    `bodyPart: [["leftHand"], ...]` 两种写法。若配置里一个部位都没写，
    退回 ALL_PARTS（保证不会因为解析不到而偷懒少给数据）。
    """
    used = set()
    for task in config or []:
        parts = task.get("bodyPart") if isinstance(task, dict) else None
        if not parts:
            continue
        for p in parts:
            if isinstance(p, (list, tuple)):
                used.update(x for x in p if x in ALL_PARTS)
            elif p in ALL_PARTS:
                used.add(p)
    return used or set(ALL_PARTS)
