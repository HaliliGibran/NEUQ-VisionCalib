"""逆透视所需的几何积木：单应、齐次变换、地平线与多边形裁剪。

本模块只回答数学问题，不读取工程配置，也不打开 OpenCV 窗口。最重要的坐标约定是：

* 图像点是 ``(x, y)`` 像素，x 向右、y 向下；
* :func:`physical_rect` 生成的地面点以标定矩形中心为原点，单位是 cm；
* 最终 BirdView 仍是 x 向右、y 向下，公制点经比例尺变成像素点；
* 3×3 矩阵接收 ``[x, y, 1]`` 形式的齐次坐标（homogeneous coordinates）。

这里的函数刻意保持无状态，方便单独验证公式。需要读取运行时范围的
``valid_fov_polygon``，以及负责交互的 ``IpmCalibrator``，留在应用编排层。
完整直觉与推导见 ``docs/KNOWLEDGE_GUIDE.md`` 的第 2、7～11 节。
"""

from typing import List, Optional, Tuple

import numpy as np

LINE_PARALLEL_EPS = 1e-8    # 两直线求交的行列式下限，小于此值视为平行
DEGENERATE_EPS = 1e-12      # DLT 归一化的平均距离下限，小于此值视为点集退化
DEN_EPS = 1e-3          # 地平线裁剪余量，|den| 小于此值视为映射到无穷远
FIT_TOLERANCE_PX = 1e-6     # 自动布局自检的越界容差（像素），只为吸收闭式解的舍入


def normalize_points_for_dlt(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """为直接线性变换（DLT）整理二维点，返回归一化点与 3×3 变换 ``T``。

    ``pts`` 是任意二维坐标系里的 ``(N, 2)`` 点。先补成 ``[x, y, 1]``，就能把
    “减去重心”和“统一缩放”合在矩阵 ``T`` 中；输出前两列则回到普通二维坐标。

    归一化不改变点之间的投影关系，只把数值搬到原点附近、把平均距离调到
    ``sqrt(2)``。否则像素坐标里的上千和齐次常数 1 同时进入 DLT 方程，奇异值分解
    容易被量级差放大舍入误差。所有点几乎重合时没有可用的几何尺度，因此明确报退化。
    """
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
    """由四对平面点求单应（Homography）``dst ~ H @ src``。

    输入、输出都是按 ``TL, TR, BL, BR`` 对应的 ``(4, 2)`` 点；两边可以采用不同
    单位。本项目通常是“去畸变图像素 → 标定矩形 cm”。符号 ``~`` 表示齐次比例
    等价：``[x, y, w]`` 与 ``[kx, ky, kw]`` 表示同一个二维点，最后都除以第三项。

    H 有 9 个数，但整体乘任意非零倍数都不改变结果，所以只有 8 个自由度；每对点
    提供 x/y 两条约束，四对点刚好给出 8 条。DLT 把这些约束写成 ``A h = 0``，
    SVD 取最接近零空间的向量，再撤销两边的 Hartley 归一化。四点共线、重合或极度
    挤在一起时约束会退化；上游还会用面积与凸性判据把这类交互位形挡住。

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
    """返回标定矩形坐标系中的四个物理角点，单位 cm。

    顺序是 ``TL, TR, BL, BR``，原点在矩形中心，x 向右、y 向下（朝向车辆）。
    这是用四点建立 ``H0`` 的标定矩形坐标系（代码中的 marker frame），不是最终
    逆透视坐标系；后者的原点由 :func:`ground_reference_origin` 另行定义。
    两个“原点”不能混用。
    """
    hw, hh = phys_w / 2.0, phys_h / 2.0
    return np.array([
        [-hw, -hh],
        [+hw, -hh],
        [-hw, +hh],
        [+hw, +hh],
    ], dtype=np.float64)


def rotation_matrix(deg: float) -> np.ndarray:
    """返回地面 cm 坐标绕原点旋转的齐次矩阵；y 向下，所以正角为顺时针。"""
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def scale_matrix(k: float) -> np.ndarray:
    """返回“地面 cm → BirdView 像素”的等比缩放矩阵，``k`` 的单位是 px/cm。"""
    return np.array([[k, 0.0, 0.0], [0.0, k, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def translation_matrix(tx: float, ty: float) -> np.ndarray:
    """返回二维平移的齐次矩阵；3×3 形式让平移也能参加矩阵连乘。"""
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """把 ``(N, 2)`` 点补成 ``[x, y, 1]`` 后应用单应，再返回普通二维点。

    若 ``H @ [x, y, 1]`` 得到 ``[u, v, w]``，真正坐标是 ``[u/w, v/w]``。
    与普通仿射变换相比，这个随位置变化的分母正是透视效果的来源；``w`` 接近 0
    时点趋向投影意义上的无穷远，也就是地平线附近。

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
    """返回单应的齐次分母 ``H[2] · [x, y, 1]``。

    分母为 0 的直线会被映到无穷远，即图像中的地平线；正负号区分它的两侧。
    只看绝对值会把天空侧也当成合法地面，因此调用方还要配合 :func:`horizon_sign`。

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
    """求 ``a1-a2`` 与 ``b1-b2`` 两条无限直线的像素交点；平行或退化时返回 None。"""
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
    """按图像坐标把四点排成 ``TL, TR, BL, BR``，供像素点与物理角点一一对应。"""
    a = np.asarray(pts, dtype=np.float64)
    idxs = np.argsort(a[:, 1])
    top = a[idxs[:2]]
    bottom = a[idxs[2:]]
    tl, tr = (top[0], top[1]) if top[0, 0] < top[1, 0] else (top[1], top[0])
    bl, br = (bottom[0], bottom[1]) if bottom[0, 0] < bottom[1, 0] else (bottom[1], bottom[0])
    return np.array([tl, tr, bl, br], dtype=np.float64)


def quad_area(pts: np.ndarray) -> float:
    """返回 ``TL,TR,BL,BR`` 四点围成的面积（px²），用鞋带公式计算。"""
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
    """用半平面 ``a*x + b*y + c >= 0`` 裁剪二维多边形。

    沿多边形边界逐段走：两端同在保留侧就留下终点，一内一外就补上与边界线的交点。
    这正是 Sutherland–Hodgman 裁剪的一步；连续对地平线、横向/纵向范围执行，就能
    得到既在相机有效地面侧、又没有延伸到数值无穷远的有效区域。

    poly 必须是 (N, 2)；返回值始终是新建的 float64 (M, 2) 数组。
    """
    p = np.asarray(poly, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f'多边形点集必须是 (N, 2)，收到形状 {p.shape}。')

    if p.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    out: List[np.ndarray] = []
    n = p.shape[0]
    for i in range(n):
        cur = p[i]
        nxt = p[(i + 1) % n]
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
    """给定锚点后，求不裁切 ``poly_cm`` 的最大比例尺（px/cm）。

    每个地面点 ``q`` 的输出位置都是 ``anchor + k*q``。``k`` 在上下左右四条边界
    约束里都只以一次项出现，因此无需试探或迭代：分别算出允许的上限，取最小值
    就是闭式解。输入多边形与锚点分别使用 cm 和 px，返回值负责把两者连接起来。
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


def ground_reference_origin(H0: np.ndarray, src_size: Tuple[int, int],
                            sign: float) -> Optional[np.ndarray]:
    """去畸变图**底边中点**经 H0 映射到地平面得到的点（标定矩形坐标系，cm）。

    这个点是**逆透视坐标系的参考原点**，人为选定，语义仅此而已：

      它不代表相机光心，不代表车辆几何中心，也不代表保险杠位置。

    为什么必须单列一条：源图底边中点 ((W-1)/2, H-1) 对应的是"从光心出发、穿过这个
    位置的视线与地面的交点"，通常落在车前一小段距离处。像素中心采用 0-based 坐标，
    偶数宽图像的几何中点位于两个中间像素之间，例如 W=1280 时 x=639.5。相机光心
    本身根本没有有限的像素坐标（它是所有成像射线的起点）。所以"画面最底下就是车"
    这个**构图直觉**是对的，
    "底边中点就是摄像头位置"这个**物理解释**是错的，两者极易混。

    也不能拿标定矩形中心当业务原点：那只是为了求 H0、表达 45x45 尺度而设的
    标定矩形坐标系 原点，凭什么代表车？真要拿到车辆位置得用 solvePnP 恢复相机位姿再
    取光心的地面投影，再叠一个安装偏置——那属于车辆安装参数，不该让标定矩形承担。
    本工具刻意不走那条路，只要一个**说得清**的坐标参考点。

    返回 None 的情形：底边中点落在地平线的无穷远侧（sign*den < DEN_EPS），此时没有
    有限的地面交点。调用方应当退回 标定矩形坐标系 原点并把这件事落盘说明，而不是
    硬取一个 1e15 量级的坐标。
    """
    H = np.asarray(H0, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f'H0 必须是 3x3，收到形状 {H.shape}。')
    if not np.isfinite(H).all():
        raise ValueError('H0 含非有限值，无法求地面参考原点。')
    w, h = int(src_size[0]), int(src_size[1])
    if w <= 0 or h <= 0:
        raise ValueError(f'源图尺寸必须为正，收到 {src_size}。')
    if not np.isfinite(sign) or sign == 0:
        raise ValueError(f'horizon_sign 必须是非零有限值，收到 {sign}。')

    px = reference_origin_px((w, h))
    den = float(H[2, 0] * px[0] + H[2, 1] * px[1] + H[2, 2])
    if not np.isfinite(den) or float(sign) * den < DEN_EPS:
        return None
    out = np.array([(H[0, 0] * px[0] + H[0, 1] * px[1] + H[0, 2]) / den,
                    (H[1, 0] * px[0] + H[1, 1] * px[1] + H[1, 2]) / den],
                   dtype=np.float64)
    return out if np.isfinite(out).all() else None


def reference_origin_px(src_size: Tuple[int, int]) -> np.ndarray:
    """逆透视坐标参考原点在去畸变图上的像素位置：底边中点。

    单独成函数是为了让"底边中点"只有一个定义处——UI 上要画这个点、落盘要记它、
    校验脚本要拿它验"H 把它映到 anchor"，三处各写一遍 (w-1)/2 迟早会漂。
    """
    w, h = int(src_size[0]), int(src_size[1])
    return np.array([(w - 1) / 2.0, float(h - 1)], dtype=np.float64)


def target_window_cm(width_cm: float, forward_cm: float) -> np.ndarray:
    """目标地面范围：以逆透视坐标参考原点为原点，x 对称 width_cm、向前 forward_cm。

    坐标系是**已经平移到参考原点、并按 heading 旋转后**的 cm 坐标，所以窗口就是

        x in [-width/2, +width/2]
        y in [-forward, 0]

    y 向下即朝向车辆，故 y=0 这条边（近边）正是参考原点所在的那条。标定矩形只负责
    建立公制映射，不负责决定最终构图；这里因此只接收真正的业务目标宽度与前向深度。

    这是**构图目标**，与 valid_fov_polygon 给出的**有效性边界**是两件不同的事。
    MAX_RANGE_CM / MAX_LATERAL_CM 存在的理由只是"地平线附近映射到无穷远，必须截
    断"，属于数学边界；把它当取景目标会让 ±300 cm 的地面被塞进一张 1280x720，
    实测结果是 48% 的输出像素来自不到 0.04 个源像素——整幅图是放射状拉丝。
    """
    if not (np.isfinite(width_cm) and width_cm > 0):
        raise ValueError(f'目标地面范围的横向宽度必须为正有限值，收到 {width_cm}。')
    if not (np.isfinite(forward_cm) and forward_cm > 0):
        raise ValueError(f'目标地面范围的前向深度必须为正有限值，收到 {forward_cm}。')

    hw = float(width_cm) / 2.0
    far = -float(forward_cm)
    return np.array([
        [-hw, far],
        [+hw, far],
        [-hw, 0.0],
        [+hw, 0.0],
    ], dtype=np.float64)


def fit_bottom_aligned(
    poly_cm: np.ndarray,
    anchor_x_px: float,
    out_size: Tuple[int, int],
    margin_px: float = 1.0,
) -> Optional[Tuple[float, float]]:
    """返回 (anchor_y_px, scale)；输入合法但无可行布局时返回 None。

    要解决的问题与 max_scale_for_fov 不同：那个函数里 anchor 是给定的，只求
    "不裁切的 scale 上限"；这里 anchor_y 也是自由量，求的是"近边贴住输出图底边、
    同时把给定多边形放到最大且一点不裁切"的那一组 (anchor_y, scale)。

    对 poly_cm 是什么刻意不作假设——它只是“这一块 cm 区域要完整装进画布”。调用方
    传目标地面范围（target_window_cm）就是为构图服务；传 valid_fov_polygon 则表示
    想完整容纳数学上的有效视野。函数名不带 fov，正是为了不混淆这两种目标。

    物理 y 向下即朝向车辆，所以多边形的 qy_max 是最近处、qy_min 是最远处。把近边
    钉在 B = H-1-margin 上（这才是"贴底"），scale 就只受三个上界约束：纵向总高
    装得下、横向左右两侧装得下。三者取 min，至少有一个是紧的，因此结果是最大的。

    anchor_x 不在本函数的决策范围内：它由用户或调用方给定，越界时直接说"没有可行
    布局"，绝不偷偷把它挪回画布内——那会让界面上的横向位置莫名其妙地跳。

    失败分两类，刻意不混在一起：
      输入本身非法（形状不是 (N,2)、含非有限值、out_size 非正、margin 为负）
        -> ValueError，与 clip_polygon_halfplane / horizon_sign 的先例一致；
      输入合法但当前几何没有正的可行布局（多边形退化、anchor_x 越界、画布太小）
        -> None。绝不用 (0, 0) 之类的魔法值表达"无解"。
    """
    poly = np.asarray(poly_cm, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError(f'多边形点集必须是 (N, 2)，收到形状 {poly.shape}。')
    if not np.isfinite(poly).all():
        raise ValueError('多边形点集含非有限值，无法求布局。')
    if not np.isfinite(anchor_x_px):
        raise ValueError('anchor_x_px 含非有限值，无法求布局。')
    if not np.isfinite(margin_px) or margin_px < 0:
        raise ValueError(f'margin_px 必须是非负有限值，收到 {margin_px}。')

    w, h = int(out_size[0]), int(out_size[1])
    if w <= 0 or h <= 0:
        raise ValueError(f'输出尺寸必须为正，收到 {out_size}。')

    if poly.shape[0] < 3:
        return None

    ax = float(anchor_x_px)
    margin = float(margin_px)
    if not (margin <= ax <= w - 1 - margin):
        return None

    qx_min, qy_min = (float(v) for v in poly.min(axis=0))
    qx_max, qy_max = (float(v) for v in poly.max(axis=0))
    if qy_max - qy_min <= 0:
        return None

    bounds: List[float] = [(h - 1 - 2 * margin) / (qy_max - qy_min)]
    if qx_max > 0:
        bounds.append((w - 1 - margin - ax) / qx_max)
    if qx_min < 0:
        bounds.append((ax - margin) / (-qx_min))

    scale = float(min(bounds))
    anchor_y = float(h - 1 - margin) - scale * qy_max

    # 自检：上面的闭式解在数值上是否真的落在图内。不满足就老实说无解。
    if not (np.isfinite(scale) and scale > 0):
        return None
    if not np.isfinite(anchor_y) or not (0.0 <= anchor_y <= h - 1):
        return None
    out = np.column_stack((ax + scale * poly[:, 0], anchor_y + scale * poly[:, 1]))
    if not (out[:, 0].min() >= margin - FIT_TOLERANCE_PX
            and out[:, 0].max() <= w - 1 - margin + FIT_TOLERANCE_PX
            and out[:, 1].min() >= margin - FIT_TOLERANCE_PX
            and out[:, 1].max() <= h - 1 - margin + FIT_TOLERANCE_PX):
        return None
    return anchor_y, scale


__all__ = [
    'DEGENERATE_EPS',
    'DEN_EPS',
    'FIT_TOLERANCE_PX',
    'LINE_PARALLEL_EPS',
    'apply_homography',
    'clip_polygon_halfplane',
    'compute_homography',
    'fit_bottom_aligned',
    'ground_reference_origin',
    'homography_denominator',
    'horizon_sign',
    'is_convex_quad',
    'line_intersection',
    'max_scale_for_fov',
    'normalize_points_for_dlt',
    'order_corners_tl_tr_bl_br',
    'physical_rect',
    'quad_area',
    'reference_origin_px',
    'rotation_matrix',
    'scale_matrix',
    'target_window_cm',
    'translation_matrix',
]
