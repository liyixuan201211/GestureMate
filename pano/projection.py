"""等距柱状（经纬图）全景 <-> 透视视角 的投影数学。

来源与出处
----------
核心投影公式**直接取自** `南客松-v2/3dgs-lab/x360/projection.py`（已数值验证过），
按原样复制到这里而不是跨项目 import：跨项目依赖很脆（对方改个函数名这边就断），
而这段数学是稳定、自包含、无第三方依赖的。原文件头部注释说明了坐标系约定：

    pano 坐标系：+X = 右, +Y = 上, +Z = 前（yaw=0 时视线朝向）
    yaw   : 绕 +Y 旋转，0 = 看向 +Z，**正值 = 向右转**
    pitch : 绕视线右轴旋转，正值 = 抬头
    lon = atan2(d.x, d.z),  lat = asin(d.y)
    xs  = (lon/2pi + 0.5) * (w-1)
    ys  = (0.5 - lat/pi)  * (h-1)

本文件在原公式之外**补了两个反向映射**（"跟随"必须要它们）：
`pano_yaw_pitch()` 把全景像素坐标换成角度，`view_point_to_pano()` 把某个
透视视角里的点换回全景角度。这两个是反着推出来的，我加了往返一致性的测试盯着。

X5 的 Webcam 模式吐的就是 2880x1440 的等距柱状图（机内已拼接+防抖）。
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "view_basis",
    "equirect_to_perspective",
    "pano_yaw_pitch",
    "view_point_to_pano",
    "paint_perspective_into_equirect",
]


# ─────────────────────────────────────────────────────────── 原 x360 的实现（照抄）

def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """在 float 坐标 xs,ys 处对 (H,W,C) 做双线性采样；越界返回 0。

    ⚠️ 这里必须交给 `cv2.remap`，不要用 numpy 花式索引。原来那版是
    `src = img.astype(np.float32)` + 4 次 262k 点采集 —— **每抽一个 512 视角就要把
    整张 2880×1440×3 的源图转成 float**（约 149MB 临时内存），实测 ~76ms，
    比跑一次 holistic（~35ms）还贵一倍，是整条实时流水线最大的一笔开销。
    `cv2.remap` 在 C++ 里直接吃 uint8、定点插值，同一件事只要几毫秒。
    """
    map_x = np.ascontiguousarray(xs, dtype=np.float32)
    map_y = np.ascontiguousarray(ys, dtype=np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def view_basis(yaw_deg: float, pitch_deg: float):
    """返回该视角的 (forward, right, up) 单位向量（pano 坐标系）。"""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    forward = np.array([sy * cp, sp, cy * cp], dtype=np.float64)
    forward /= np.linalg.norm(forward)

    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:          # 正上/正下看
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return forward, right, up


def _view_rays(out_w: int, out_h: int, fov_deg: float,
               yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """透视视角每个像素对应的单位视线方向 (out_h, out_w, 3)。"""
    forward, right, up = view_basis(yaw_deg, pitch_deg)
    f = (out_w * 0.5) / np.tan(np.radians(fov_deg) * 0.5)

    us, vs = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x = (us - out_w * 0.5) / f
    y = -(vs - out_h * 0.5) / f            # 图像 y 向下

    d = (forward[None, None, :] + x[..., None] * right[None, None, :]
         + y[..., None] * up[None, None, :])
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d


def equirect_to_perspective(erp: np.ndarray, out_w: int = 512, out_h: int = 512,
                            fov_deg: float = 90.0, yaw_deg: float = 0.0,
                            pitch_deg: float = 0.0) -> np.ndarray:
    """从等距柱状全景里抽一个透视视角（无畸变）。"""
    d = _view_rays(out_w, out_h, fov_deg, yaw_deg, pitch_deg)

    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))

    h, w = erp.shape[:2]
    xs = (lon / (2 * np.pi) + 0.5) * (w - 1)
    ys = (0.5 - lat / np.pi) * (h - 1)
    return _bilinear(erp, xs, ys)          # cv2.remap 出来已经是 uint8


# ─────────────────────────────────────────────────────────── 反向映射（本文件新增）

def pano_yaw_pitch(px, py, w: int, h: int):
    """全景像素坐标 -> (yaw_deg, pitch_deg)。上面 xs/ys 公式的反解。"""
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    lon = (px / (w - 1) - 0.5) * (2 * np.pi)
    lat = (0.5 - py / (h - 1)) * np.pi
    return np.degrees(lon), np.degrees(lat)


def view_point_to_pano(px, py, view_w: int, view_h: int,
                       fov_deg: float, yaw_deg: float, pitch_deg: float):
    """把某个透视视角里的像素点换回全景角度 (yaw_deg, pitch_deg)。

    「跟随」的核心：在透视视角里检出人的关键点 -> 反推它落在全景的哪个方位。
    透视视角里 x 向右、y 向下：x = (px - W/2)/f, y = -(py - H/2)/f，
    其中 f = (W/2)/tan(fov/2)，与 `_view_rays` 完全一致。
    """
    forward, right, up = view_basis(yaw_deg, pitch_deg)
    f = (view_w * 0.5) / np.tan(np.radians(fov_deg) * 0.5)
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    x = (px - view_w * 0.5) / f
    y = -(py - view_h * 0.5) / f

    d = (forward[None, :] + x[..., None] * right[None, :]
         + y[..., None] * up[None, :])
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    yaw = np.degrees(np.arctan2(d[..., 0], d[..., 2]))
    pitch = np.degrees(np.arcsin(np.clip(d[..., 1], -1.0, 1.0)))
    return yaw, pitch


def paint_perspective_into_equirect(erp: np.ndarray, img: np.ndarray,
                                    yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                                    fov_deg: float = 70.0, feather: int = 8) -> np.ndarray:
    """把一张普通透视图「贴」进全景里，像它本来就在那个方位被拍到一样。

    用途：**造带已知答案的测试数据**。我拿一段真有人的视频帧贴到全景的指定
    方位，就能反过来断言「跟随」有没有把方位找对 —— 否则没有 ground truth，
    跟随准不准只能靠眼睛看。

    做法是标准针孔重投影：对全景每个像素求视线方向，换到该视角的相机坐标
    (x=d·right, y=d·up, z=d·forward)，z>0 的投影到像平面取色。
    """
    out = erp.copy()
    h, w = out.shape[:2]
    forward, right, up = view_basis(yaw_deg, pitch_deg)
    f = (img.shape[1] * 0.5) / np.tan(np.radians(fov_deg) * 0.5)

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    lon = (us / (w - 1) - 0.5) * (2 * np.pi)
    lat = (0.5 - vs / (h - 1)) * np.pi
    d = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat),
                  np.cos(lat) * np.cos(lon)], axis=-1)

    x = d @ right
    y = d @ up
    z = d @ forward
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = np.where(z > 1e-6, (x / np.maximum(z, 1e-9)) * f + img.shape[1] * 0.5, -1)
        sy = np.where(z > 1e-6, -(y / np.maximum(z, 1e-9)) * f + img.shape[0] * 0.5, -1)

    m = (sx >= 0) & (sx <= img.shape[1] - 1) & (sy >= 0) & (sy <= img.shape[0] - 1)
    if not np.any(m):
        return out
    sampled = _bilinear(img, sx, sy)

    # 边缘羽化：不留一圈硬切口（硬边会被人形检测器当成奇怪的物体）
    a = m.astype(np.float32)
    if feather > 0:
        k = feather | 1
        for _ in range(2):
            p = np.pad(a, 1, mode="edge")
            a = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
    a = np.clip(a, 0, 1)[..., None]

    return np.clip(out.astype(np.float32) * (1 - a) + sampled * a, 0, 255).astype(np.uint8)
