"""面部 / 身体的关键点语义分析 —— 把 468+33 个点变成人看得懂的结论。

放在服务端算的原因：逻辑可单测、前后端数值一致，前端只管画和显示。

坐标前提（很重要）
------------------
本模块接收的是 **pipeline 交付坐标**，也就是「相机原图」坐标系：
推理前画面被水平镜像过，关键点在 `pipeline._unmirror()` 里已经还原（x' = 1-x）。

左右眼的判定**不看 MediaPipe 的编号，只看交付坐标里 x 的大小**，原因：
实测 FaceMesh 的手性是按「画面哪一边」定的 —— 把镜像图喂进去，它照样把
idx33 放在那张图的左侧（原图 33=0.485 / 镜像图 33=0.481，几乎没变），
而不是跟着解剖学换到另一只眼。于是我们再 unmirror 一次之后，
idx33 反而落到了交付画面的**右侧**。

所以这里统一按几何约定：
    交付坐标里在**画面左侧**的那只眼 = 被摄者的**右眼**
    （镜头正对时，画面左就是对方右侧，这是标准约定）
这样无论 MediaPipe 内部怎么编号，标签都和用户看到的画面对得上。

眼睛开合度 EAR（eye aspect ratio）
--------------------------------
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
p1/p4 是内外眼角，p2/p3 与 p6/p5 是上下眼睑。睁眼约 0.25~0.5，闭眼掉到 0.25 以下。
它是**比值**，所以对远近/分辨率不敏感。

嘴巴开合度 MAR
-------------
    MAR = |上唇内缘 - 下唇内缘| / |左嘴角 - 右嘴角|

头部朝向
--------
没有做 solvePnP（需要相机内参与 3D 人脸模型），这里给的是**归一化比例**：
以眼角距为尺度，量鼻尖相对眼中心偏移了多少。yaw/roll 已经换算到
**镜像显示坐标系**（也就是用户在网页上看到的那一面），所以
「朝右」= 用户把头转向自己的右边 = 屏幕上脸朝右看。
roll 归一化到 (-90, 90]，避免出现 ±180° 这种「反了个向」的读数。
老实说：这只是**粗略估计**，不是角度。
"""
import math

# ---------------------------------------------------------------- 关键点索引
# 两只眼睛（EAR 用的 6 点，顺序：外眼角, 上睑1, 上睑2, 内眼角, 下睑2, 下睑1）。
# 名字里的 A/B 只是分组编号，不代表左右 —— 左右由运行时的 x 顺序决定。
EYE_GROUP_A = (33, 160, 158, 133, 153, 144)
EYE_GROUP_B = (362, 385, 387, 263, 373, 380)

NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10

MOUTH_UPPER_INNER = 13
MOUTH_LOWER_INNER = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_UPPER_OUTER = 0
MOUTH_LOWER_OUTER = 17

# 身体（MediaPipe Pose 33 点里的常用几个）
P_NOSE = 0
P_L_SHOULDER, P_R_SHOULDER = 11, 12
P_L_ELBOW, P_R_ELBOW = 13, 14
P_L_WRIST, P_R_WRIST = 15, 16
P_L_HIP, P_R_HIP = 23, 24

#: 归一化「开合度」用的满量程（EAR 到这个值算全开）
EAR_FULL = 0.42


def _d(a, b):
    """两个归一化关键点的欧氏距离（只用 x,y）。"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mid(a, b):
    return [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0,
            ((a[2] if len(a) > 2 else 0) + (b[2] if len(b) > 2 else 0)) / 2.0]


def _mean_x(face, idx):
    if not face or len(face) <= max(idx):
        return None
    return sum(face[i][0] for i in idx) / len(idx)


def eye_aspect_ratio(face, idx):
    """6 点 EAR。点缺失时返回 None。"""
    if not face or len(face) <= max(idx):
        return None
    p1, p2, p3, p4, p5, p6 = (face[i] for i in idx)
    denom = 2.0 * _d(p1, p4)
    if denom < 1e-9:
        return None
    return (_d(p2, p6) + _d(p3, p5)) / denom


def eye_groups(face):
    """返回 (右眼组, 左眼组)；按交付坐标里 x 的大小判定。

    画面左 = 被摄者右眼（镜头正对的标准约定），与 MediaPipe 的编号无关。
    """
    xa, xb = _mean_x(face, EYE_GROUP_A), _mean_x(face, EYE_GROUP_B)
    if xa is None or xb is None:
        return None, None
    if xa < xb:                       # A 在画面左侧 -> A 是右眼
        return EYE_GROUP_A, EYE_GROUP_B
    return EYE_GROUP_B, EYE_GROUP_A


def mouth_aspect_ratio(face):
    if not face or len(face) <= max(MOUTH_UPPER_INNER, MOUTH_LOWER_INNER,
                                    MOUTH_LEFT, MOUTH_RIGHT):
        return None
    width = _d(face[MOUTH_LEFT], face[MOUTH_RIGHT])
    if width < 1e-9:
        return None
    return _d(face[MOUTH_UPPER_INNER], face[MOUTH_LOWER_INNER]) / width


def mouth_width_ratio(face):
    """嘴角张开宽度 / 眼距 —— 咧嘴时会变大。"""
    if not face or len(face) <= max(MOUTH_LEFT, MOUTH_RIGHT):
        return None
    face_w = _d(face[EYE_GROUP_A[0]], face[EYE_GROUP_B[0]])
    if face_w < 1e-9:
        return None
    return _d(face[MOUTH_LEFT], face[MOUTH_RIGHT]) / face_w


def _normalize_pm90(deg):
    """把角度折到 (-90, 90]，避免出现 ±180° 这种「反了个向」的读数。"""
    while deg > 90:
        deg -= 180
    while deg <= -90:
        deg += 180
    return deg


def head_pose(face):
    """返回 (yaw, pitch, rollDeg)。

    yaw/pitch 是以眼距归一化的比例（不是角度）；yaw 已换算到镜像显示坐标系，
    yaw>0 表示用户把头转向自己的右边（屏幕上脸朝右）。
    rollDeg 是眼睛连线的倾角，归一化到 (-90, 90]。
    """
    right, left = eye_groups(face)
    if not right or not left:
        return None
    er, el = _mid(face[right[0]], face[right[3]]), _mid(face[left[0]], face[left[3]])
    eye_dist = _d(er, el)
    if eye_dist < 1e-9:
        return None
    center = _mid(er, el)
    nose = face[NOSE_TIP]
    # 交付坐标 -> 显示坐标是 x_disp = 1-x，后者只改符号，
    # 因此 yaw_disp = (眼中心x - 鼻尖x) / 眼距
    yaw = (center[0] - nose[0]) / eye_dist
    pitch = (nose[1] - center[1]) / eye_dist
    # roll：从右眼（画面左）指向左眼（画面右）的向量
    roll = _normalize_pm90(math.degrees(math.atan2(el[1] - er[1], el[0] - er[0])))
    return yaw, pitch, roll


def _direction(yaw, pitch):
    """按阈值给个方向标签。阈值是拿 40 张真实人像量出来的经验值。

    俯仰（抬头/低头）是三项里最不可靠的 —— 没做 PnP，只靠鼻尖相对眼线的
    偏移，而且人跟人的脸型差异会直接体现在这个比值上。所以阈值取得很保守：
    只在明显仰头/低头时才给标签，中间大片区域都算「正对」。
    （实测该比值：最低 0.58，中位 1.02，最高 1.37）
    """
    parts = []
    if yaw > 0.13:
        parts.append("朝右")
    elif yaw < -0.13:
        parts.append("朝左")
    if pitch < 0.72:
        parts.append("抬头")
    elif pitch > 1.32:
        parts.append("低头")
    return "·".join(parts) if parts else "正对"


class FaceBodyAnalyzer:
    """把逐帧关键点变成带平滑与状态的识别结果。

    有状态（EMA 平滑 + 眨眼计数），所以每路视频要各持一个实例。
    """

    def __init__(self, ear_closed=0.26, ear_open=0.30, ear_full=EAR_FULL,
                 mar_closed=0.08, mar_open=0.25, smoothing=0.4):
        """
        阈值是在 tests/fixtures/hands 那 40 张真实人像上量出来的经验值：
            EAR  中位 ≈ 0.45，最低 ≈ 0.23（那一帧接近闭眼）
            MAR  中位 ≈ 0.03，最大 ≈ 0.27（张嘴）
        EAR 是比值，与远近/分辨率无关，但**因人而异**（眼型差异很大），
        所以真实使用中如果「睁闭」判反了，调 ear_closed/ear_open 即可。
        """
        self.ear_closed = ear_closed
        self.ear_open = ear_open
        self.ear_full = ear_full
        self.mar_closed = mar_closed
        self.mar_open = mar_open
        self.alpha = smoothing          # EMA 系数，越大越跟手
        self.blinkCount = 0
        self._ear = {"left": None, "right": None}
        self._mar = None
        self._eyesClosed = {"left": False, "right": False}

    # ------------------------------------------------------------ 内部
    def _emaEar(self, side, value):
        if value is None:
            return self._ear.get(side)
        old = self._ear.get(side)
        new = value if old is None else old * (1 - self.alpha) + value * self.alpha
        self._ear[side] = new
        return new

    def _update_eye(self, side, ear):
        """更新单只眼的睁/闭状态；返回是否发生「闭 -> 睁」的跳变。

        注意：返回值由调用方**合并**成一次眨眼 —— 两只眼通常同时眨，
        若在这里各自 +1，一次眨眼会被数成 2 次（踩过）。
        """
        if ear is None:
            return False
        if not self._eyesClosed[side] and ear < self.ear_closed:
            self._eyesClosed[side] = True
            return False
        if self._eyesClosed[side] and ear > self.ear_open:
            self._eyesClosed[side] = False
            return True
        return False

    # ------------------------------------------------------------ 主入口
    def update(self, face, body, hands=None):
        out = {"available": bool(face), "blinkCount": self.blinkCount}

        # ---- 面部
        if face:
            right, left = eye_groups(face)
            raw_r = eye_aspect_ratio(face, right) if right else None
            raw_l = eye_aspect_ratio(face, left) if left else None
            # 判「睁/闭」必须用**原始值**：EMA 平滑会把单帧的眨眼抹掉
            # （实测平滑后最低只到 0.34，永远碰不到 ear_closed），
            # 所以平滑值只拿来显示「开合度」。
            jumped_r = self._update_eye("right", raw_r)
            jumped_l = self._update_eye("left", raw_l)
            if jumped_r or jumped_l:
                self.blinkCount += 1          # 双眼同时睁开只算一次
            ear_r = self._emaEar("right", raw_r)
            ear_l = self._emaEar("left", raw_l)
            closed_r = self._eyesClosed["right"]
            closed_l = self._eyesClosed["left"]

            mar = mouth_aspect_ratio(face)
            # 注意不能用 `mar or self._mar` —— mar 合法地等于 0.0 时会被误判成缺失
            if mar is not None:
                self._mar = mar if self._mar is None else \
                    self._mar * (1 - self.alpha) + mar * self.alpha
            mwr = mouth_width_ratio(face)
            pose = head_pose(face)

            full = self.ear_full
            out["eyes"] = {
                "right": None if ear_r is None else round(ear_r, 3),
                "left": None if ear_l is None else round(ear_l, 3),
                "rightOpen": None if ear_r is None else round(min(1.0, ear_r / full), 3),
                "leftOpen": None if ear_l is None else round(min(1.0, ear_l / full), 3),
                "rightClosed": closed_r,
                "leftClosed": closed_l,
            }
            mar_v = None if self._mar is None else round(self._mar, 3)
            if mar_v is None:
                mouth_state = "未知"
            elif mar_v < self.mar_closed:
                mouth_state = "闭合"
            elif mar_v < self.mar_open:
                mouth_state = "微张"
            else:
                mouth_state = "张开"
            out["mouth"] = {"open": mar_v, "state": mouth_state,
                            "widthRatio": None if mwr is None else round(mwr, 3)}
            if pose:
                yaw, pitch, roll = pose
                out["head"] = {
                    "yaw": round(yaw, 3), "pitch": round(pitch, 3),
                    "rollDeg": round(roll, 1),
                    "direction": _direction(yaw, pitch),
                }
            out["landmarkCount"] = len(face)

        # ---- 身体
        if body and len(body) > max(P_L_WRIST, P_R_WRIST):
            ls, rs = body[P_L_SHOULDER], body[P_R_SHOULDER]
            lw, rw = body[P_L_WRIST], body[P_R_WRIST]
            # 图像坐标 y 越小越高
            hands_up = {
                "left": lw[1] < ls[1],
                "right": rw[1] < rs[1],
            }
            tilt = math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0]))
            out["body"] = {
                "handsUp": hands_up,
                "shoulderTiltDeg": round(tilt, 1),
                "shoulderWidth": round(_d(ls, rs), 3),
                "landmarkCount": len(body),
                "lean": "左倾" if tilt > 7 else ("右倾" if tilt < -7 else "正"),
            }
        return out
