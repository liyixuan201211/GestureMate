"""动作识别：左移 / 右移 / 跳 / 蹲（《向星而行-UE实现设计.md》§5 的落地）。

对外只有两样东西：

    ActionDetector   关键点流 → ActionEvent 流（门控全在里面）
    ActionConfig     全部阈值（真人现场重调用）

再加一组纯函数/滤波器（features），以及 `python -m action` 这个可跑入口。

设计要点、坐标约定、镜像开关的位置、以及**哪些是改过 §5 默认值的**，
都写在 detector.py 的模块头与各字段注释里 —— 这里不重复，免得两处说法走样。
"""
from .detector import (KIND_JUMP, KIND_LANE, KIND_PLAYER, KIND_READY,
                       KIND_SQUAT, POSTURE_CROUCH, POSTURE_STAND, ActionConfig,
                       ActionDetector, ActionEvent, Sustained, describe_event)
from .features import (KEY_INDICES, MIN_LANDMARKS, P_L_ANKLE, P_L_ELBOW,
                       P_L_HIP, P_L_KNEE, P_L_SHOULDER, P_L_WRIST, P_NOSE,
                       P_R_ANKLE, P_R_ELBOW, P_R_HIP, P_R_KNEE, P_R_SHOULDER,
                       P_R_WRIST, Calibrator, FeatureSmoother, MedianFilter,
                       NeutralBasis, OneEuroFilter, compute_features, hip_mid,
                       knee_angle, parse_body, shoulder_mid, torso_length)

__all__ = [
    # 核心
    "ActionDetector", "ActionConfig", "ActionEvent", "Sustained",
    "describe_event",
    # 事件种类 / 姿态
    "KIND_READY", "KIND_SQUAT", "KIND_LANE", "KIND_JUMP", "KIND_PLAYER",
    "POSTURE_STAND", "POSTURE_CROUCH",
    # 特征层
    "Calibrator", "NeutralBasis", "compute_features", "parse_body",
    "hip_mid", "shoulder_mid", "torso_length", "knee_angle",
    "MedianFilter", "OneEuroFilter", "FeatureSmoother",
    # 常量
    "MIN_LANDMARKS", "KEY_INDICES",
    "P_NOSE", "P_L_SHOULDER", "P_R_SHOULDER", "P_L_ELBOW", "P_R_ELBOW",
    "P_L_WRIST", "P_R_WRIST", "P_L_HIP", "P_R_HIP", "P_L_KNEE", "P_R_KNEE",
    "P_L_ANKLE", "P_R_ANKLE",
]
