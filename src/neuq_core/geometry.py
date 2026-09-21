"""底层几何 primitives：单应求解、齐次变换、四边形与多边形判据。

从 neuq_vision_calib.py 原样搬来，常量、函数体、docstring 与注释逐字未改。依赖方向
单向且最浅：只用 numpy，不导入 cv2，不依赖 neuq_core.config，更不导入 facade，
因此可以独立 import、独立测试。

刻意**没有**搬进来的：
  valid_fov_polygon     直接读 DEN_EPS / MAX_RANGE_CM / MAX_LATERAL_CM，后两个是 CLI 在
                        apply_options() 里用 global 重新赋值的运行时状态。搬它就得为这些
                        值做一份跨模块镜像，那是新增状态复制，不是这一刀的范围。它留在
                        facade，继续调用这里搬出来的 clip_polygon_halfplane / apply_homography。
  MAX_RANGE_CM / MAX_LATERAL_CM
                        运行时状态，权威留在 facade。
  IpmCalibrator         交互标定类，依赖 cv2 窗口与大量运行时开关，不属于这一层。
"""

from typing import List, Optional, Tuple

import numpy as np

LINE_PARALLEL_EPS = 1e-8    # 两直线求交的行列式下限，小于此值视为平行
DEGENERATE_EPS = 1e-12      # DLT 归一化的平均距离下限，小于此值视为点集退化
DEN_EPS = 1e-3          # 地平线裁剪余量，|den| 小于此值视为映射到无穷远


def normalize_points_for_dlt(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley 归一化：平移到重心、缩放到平均距离 sqrt(2)，返回归一化点与变换矩阵。"""
    points = np.asarray(pts, dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center
    mean_distance = float(np.mean(np.linalg.norm(centered, axis=1)))
    if mean_distance < DEGENERATE_EPS:
        raise ValueError('点集退化。')

    scale = np.sqrt(2.0) / mean_distance
    T = np.array([
        [scale, 0.0, -scale * center[0]],
        [0.0, scale, -scale * center[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))
    normalized = (T @ homogeneous.T).T
    return normalized[:, :2], T


def compute_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """四点单应，归一化 DLT，float64 全程。

    这里刻意不用 OpenCV 的现成接口：cv2.getPerspectiveTransform 只接受 CV_32F 输入
    （传 float64 直接抛 checkVector 断言），cv2.findHomography 虽接受 float64 但内部
    同样降精度。实测同一组亚像素四点的重投影误差——getPerspectiveTransform 1.2e-6 px、
    findHomography 1.3e-6 px、本实现 1.4e-14 px，差 8 个数量级。H0 的误差会经
    T·S·R 放大后进入两套查找表，因此这里保留自研实现。
    """
    src, T_src = normalize_points_for_dlt(src_pts)
    dst, T_dst = normalize_points_for_dlt(dst_pts)
    if src.shape[0] != 4 or dst.shape[0] != 4:
        raise ValueError('compute_homography 只接受 4 点。')

    A = np.zeros((src.shape[0] * 2, 9), dtype=np.float64)
    for i, ((x, y), (u, v)) in enumerate(zip(src, dst, strict=True)):
        A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]

    _, _, vh = np.linalg.svd(A)
    H = np.linalg.inv(T_dst) @ vh[-1].reshape(3, 3) @ T_src

    if not np.isfinite(H).all():
        raise ValueError('单应矩阵含非有限值。')
    if abs(float(H[2, 2])) <= np.finfo(np.float64).eps * float(np.linalg.norm(H)):
        raise ValueError('单应矩阵退化（H[2,2] 接近 0）。')
    return H / H[2, 2]


def physical_rect(phys_w: float, phys_h: float) -> np.ndarray:
    """标定矩形的物理角点，顺序 TL,TR,BL,BR，中心为原点，y 向下即朝向车辆。"""
    hw, hh = phys_w / 2.0, phys_h / 2.0
    return np.array([
        [-hw, -hh],
        [+hw, -hh],
        [-hw, +hh],
        [+hw, +hh],
    ], dtype=np.float64)


def rotation_matrix(deg: float) -> np.ndarray:
    """cm 坐标系内绕原点旋转的齐次矩阵（y 向下，故正角为顺时针）。"""
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def scale_matrix(k: float) -> np.ndarray:
    """cm -> 像素的等比缩放齐次矩阵，k 的单位是 px/cm。"""
    return np.array([[k, 0.0, 0.0], [0.0, k, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def translation_matrix(tx: float, ty: float) -> np.ndarray:
    """平移齐次矩阵。"""
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """对 (N,2) 点集应用单应，返回 (N,2)。

    刻意不用 cv2.perspectiveTransform：后者在齐次分量接近 0 时把结果置 0，
    而本项目依赖 isfinite 判据剔除映射到无穷远的点，需要 inf 如实传播。
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack((p, np.ones(p.shape[0])))
    out = (H @ hom.T).T
    with np.errstate(divide='ignore', invalid='ignore'):
        out = out[:, :2] / out[:, 2:3]
    return out


def homography_denominator(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """单应的齐次分母 H[2,0]*x + H[2,1]*y + H[2,2]，符号区分地平线两侧。

    非有限的输入点先归零：它们已被调用方的 isfinite 判据剔除，这里只是避免
    inf 参与乘法产生溢出告警。
    """
    p = np.nan_to_num(np.asarray(pts, dtype=np.float64).reshape(-1, 2),
                      nan=0.0, posinf=0.0, neginf=0.0)
    return H[2, 0] * p[:, 0] + H[2, 1] * p[:, 1] + H[2, 2]


def horizon_sign(H0: np.ndarray, ground_pts: np.ndarray) -> float:
    """判定"地面有限侧"对应的分母符号；参考点与单应必须全部有限，否则报错。

    必须用标定四点作参考——它们一定落在地面上、映射到有限的 cm 坐标。
    若拿图像中心当参考，当地平线落在图像中心以下（相机上仰、或四条线被拖到
    异常位形）时会选中天空侧，max_scale 随之完全错误。

    非有限输入一律 ValueError，不过滤、不"用剩下的点继续投票"。因为这个函数的
    语义不是"从一堆可能有效的点里投票"，而是"用确定属于地面的参考点判断有限侧"：
    四个可信参考点掉成三个，大概率还能得出同一个符号，但那等于把一次上游几何
    错误吞掉，事后只会表现成某些查找表的有效区莫名其妙。

    尤其不能依赖 homography_denominator 的 nan_to_num：那个 helper 是为 LUT 的
    向量化路径设计的（无效点已由调用方的 mask 判掉，归零只为避免 inf 运算告警），
    在这里会把 inf 参考点悄悄变成 (0,0)，算出一个看着正常的分母、返回一个看着
    正常的 ±1。契约是：要么返回可信的 +1/-1，要么明确失败，绝不把 NaN 翻译成
    某个方向。
    """
    H = np.asarray(H0, dtype=np.float64)
    pts = np.asarray(ground_pts, dtype=np.float64).reshape(-1, 2)

    if not np.isfinite(H).all():
        raise ValueError('单应矩阵含非有限值，无法判断地平线有限侧。')
    if not np.isfinite(pts).all():
        raise ValueError('地面参考点含非有限值，无法判断地平线有限侧。')

    den = homography_denominator(H, pts)
    # 即便 H 与 pts 都有限，极端数值下乘法仍可能溢出。这道检查成本几乎为零，
    # 是"要么可信要么失败"这条契约的最后一环。
    if not np.isfinite(den).all():
        raise ValueError('地平线分母含非有限值，无法判断有限侧。')

    return 1.0 if float(np.mean(den)) >= 0 else -1.0



def line_intersection(a1, a2, b1, b2) -> Optional[np.ndarray]:
    """两直线交点，平行或退化时返回 None。"""
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(float(den)) < LINE_PARALLEL_EPS:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py], dtype=np.float64)


def order_corners_tl_tr_bl_br(pts: np.ndarray) -> np.ndarray:
    """按 y 再按 x 把四点排成 TL,TR,BL,BR。"""
    a = np.asarray(pts, dtype=np.float64)
    idxs = np.argsort(a[:, 1])
    top = a[idxs[:2]]
    bottom = a[idxs[2:]]
    tl, tr = (top[0], top[1]) if top[0, 0] < top[1, 0] else (top[1], top[0])
    bl, br = (bottom[0], bottom[1]) if bottom[0, 0] < bottom[1, 0] else (bottom[1], bottom[0])
    return np.array([tl, tr, bl, br], dtype=np.float64)


def quad_area(pts: np.ndarray) -> float:
    """TL,TR,BL,BR 四点围成的四边形面积（shoelace 绝对值）。"""
    p = np.asarray(pts, dtype=np.float64)[[0, 1, 3, 2]]
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def is_convex_quad(pts: np.ndarray) -> bool:
    """判断 TL,TR,BL,BR 四点是否构成凸四边形（非自交）。

    面积阈值挡不住自交：蝴蝶形（bowtie）的 shoelace 绝对值同样可以很大。
    自交时 H0 仍有解但几何完全错误，必须单独判。
    """
    p = np.asarray(pts, dtype=np.float64)[[0, 1, 3, 2]]
    edges = np.roll(p, -1, axis=0) - p
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    return bool(np.all(cross > 0) or np.all(cross < 0))


def clip_polygon_halfplane(poly: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """保留满足 a*x + b*y + c >= 0 的部分（Sutherland-Hodgman 单边裁剪）。"""
    if poly.shape[0] == 0:
        return poly
    out: List[np.ndarray] = []
    n = poly.shape[0]
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        d_cur = a * cur[0] + b * cur[1] + c
        d_nxt = a * nxt[0] + b * nxt[1] + c
        if d_cur >= 0:
            out.append(cur)
        if (d_cur >= 0) != (d_nxt >= 0):
            t = d_cur / (d_cur - d_nxt)
            out.append(cur + t * (nxt - cur))
    return np.array(out, dtype=np.float64) if out else np.zeros((0, 2), dtype=np.float64)


def max_scale_for_fov(poly_cm: np.ndarray, anchor_px: Tuple[float, float],
                      out_size: Tuple[int, int]) -> float:
    """不裁切有效视野时允许的最大 scale（px/cm）。

    约束是 anchor + k*q 落在输出图内，k 线性出现，故有闭式解。
    """
    if poly_cm.shape[0] < 3:
        return 0.0
    ax, ay = anchor_px
    w, h = out_size
    if not (0.0 <= ax <= w - 1 and 0.0 <= ay <= h - 1):
        return 0.0

    qx_min, qy_min = poly_cm.min(axis=0)
    qx_max, qy_max = poly_cm.max(axis=0)

    bounds: List[float] = []
    if qx_max > 0:
        bounds.append((w - 1 - ax) / qx_max)
    if qx_min < 0:
        bounds.append(-ax / qx_min)
    if qy_max > 0:
        bounds.append((h - 1 - ay) / qy_max)
    if qy_min < 0:
        bounds.append(-ay / qy_min)
    if not bounds:
        return 0.0
    return max(0.0, float(min(bounds)))


__all__ = [
    'DEGENERATE_EPS',
    'DEN_EPS',
    'LINE_PARALLEL_EPS',
    'apply_homography',
    'clip_polygon_halfplane',
    'compute_homography',
    'homography_denominator',
    'horizon_sign',
    'is_convex_quad',
    'line_intersection',
    'max_scale_for_fov',
    'normalize_points_for_dlt',
    'order_corners_tl_tr_bl_br',
    'physical_rect',
    'quad_area',
    'rotation_matrix',
    'scale_matrix',
    'translation_matrix',
]
