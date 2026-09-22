import subprocess
import logging

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


def drawLandmarks(image, results):
    import mediapipe.python.solutions as sol
    # Draw face connections
    sol.drawing_utils.draw_landmarks(
        image,
        results.face_landmarks,
        sol.holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=sol.drawing_styles.
        get_default_face_mesh_tesselation_style())
    sol.drawing_utils.draw_landmarks(
        image,
        results.face_landmarks,
        sol.holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=sol.drawing_styles.
        get_default_face_mesh_contours_style())
    # Draw pose connections
    sol.drawing_utils.draw_landmarks(
        image, results.pose_landmarks, sol.holistic.POSE_CONNECTIONS,
        sol.drawing_styles.get_default_pose_landmarks_style())
    # Draw left hand connections
    sol.drawing_utils.draw_landmarks(
        image, results.left_hand_landmarks, sol.holistic.HAND_CONNECTIONS,
        sol.drawing_styles.get_default_hand_landmarks_style(),
        sol.drawing_styles.get_default_hand_connections_style())
    # Draw right hand connections
    sol.drawing_utils.draw_landmarks(
        image, results.right_hand_landmarks, sol.holistic.HAND_CONNECTIONS,
        sol.drawing_styles.get_default_hand_landmarks_style(),
        sol.drawing_styles.get_default_hand_connections_style())


def _toList(landmarks):
    """NormalizedLandmarkList -> [[x,y,z], ...]；None 原样返回。"""
    if landmarks is None:
        return None
    return [[p.x, p.y, p.z] for p in landmarks.landmark]


def extractLandmarks(x, need=None):
    """把 MediaPipe 结果转成任务层使用的 dict。

    need: 只转换真正被配置引用的部位（None = 全部都要）。
          跳过 face 就省下每帧 468 次属性访问和列表构造 —— 官方两个 demo
          都只用左右手，所以这一项经常能省掉九成以上的转换量。

    注意 left↔right 的互换是**有意为之**：推理前画面被水平镜像过，
    MediaPipe 的手性标签与真手相反（详见 LandmarkEngine 模块注释）。
    """
    want = set(need) if need else {"body", "leftHand", "rightHand", "face"}
    res = {}
    res['body'] = _toList(x.pose_landmarks) if 'body' in want else None
    res['rightHand'] = _toList(x.left_hand_landmarks) if 'rightHand' in want else None
    res['leftHand'] = _toList(x.right_hand_landmarks) if 'leftHand' in want else None
    res['face'] = _toList(x.face_landmarks) if 'face' in want else None
    return res


def generateNullLandmarks():
    res = {}
    res['body'] = None
    res['leftHand'] = None
    res['rightHand'] = None
    res['face'] = None
    return res
