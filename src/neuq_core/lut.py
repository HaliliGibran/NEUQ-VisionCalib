"""查找表的数学核：网格生成、无效点标定、重采样与复合反向映射。

从 neuq_vision_calib.py 原样搬来，函数体、docstring 与注释逐字未改。依赖方向单向：
本模块只用 cv2/numpy 与 neuq_core.geometry，不导入 neuq_core.config，更不导入
facade，因此可以独立 import、独立测试。

分界线是"要不要自己去读运行时开关"：resample_map_pair 属于"给我目标 size 我就做"
的数学操作，所以搬；prepare_map_pair 属于"自己去读 TABLE_SIZE / TABLE_FORMAT /
TABLE_FIXED_POINT"的策略编排，所以留。

刻意**没有**搬进来的：
  quantize_table / dequantize_table / prepare_map_pair
  write_table_txt / write_table_bin / write_table_c / format_int16_array
  clear_stale_tables / serialize_map_pair
  load_table / _load_c_header / load_map_pair
  table_convention_text / report_table_size
  export_undistort_tables / export_composite_tables / load_exported_reverse_pair
  BIN_SENTINEL / TABLE_FORMAT / TABLE_FIXED_POINT / TABLE_SIZE
                        这批直接读 TABLE_FORMAT / TABLE_FIXED_POINT / TABLE_SIZE，
                        而这三个是 CLI 在 apply_options() 里用 global 重新赋值的
                        运行时状态。硬搬就得再加一个 bind_table_options() 的镜像。
                        尤其 dequantize_table(..., fixed_point=None) 里
                        `fp = TABLE_FIXED_POINT if fixed_point is None else ...`
                        这个设计本身就是为了让运行时修改生效，搬进没有 runtime
                        context 的模块会直接破坏该契约。
  valid_fov_polygon / MAX_RANGE_CM / MAX_LATERAL_CM
                        同为运行时状态，权威留在 facade（见 geometry 的说明）。
  IpmCalibrator         交互标定类，依赖 cv2 窗口与大量运行时开关，不属于这一层。
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .geometry import (
    DEN_EPS,
    apply_homography,
    homography_denominator,
)


def downsample_factor(source_size: Tuple[int, int],
                      table_size: Optional[Tuple[int, int]]) -> int:
    """校验查找表网格是源尺寸的整数倍等比缩小，返回倍率 n；不合法则抛 ValueError。

    为什么必须**等比**：BirdView 的公制尺度是各向同性的（scale px/cm 对 x 与 y 同一个
    值）。1280x720 降成 320x240 时 x 缩 4 倍、y 缩 3 倍，于是小图里
    x 是 scale/4、y 是 scale/3 —— 一个物理上严格 45x45 cm 的正方形会变成
    53.8 x 71.7 px 的竖长矩形，宽高比 0.75。C 端若直接拿这张图找线、算角度、
    曲率、横向误差，赛道会被横向压缩 25%，全部失真。而 batch_test() 输出的正是
    这张真实的小图，所以风险不是理论上的。

    等比之后 C 端的坐标换算也从两个 step 简化成一个 n：
        full_x = (small_x + 0.5) * n - 0.5
        full_y = (small_y + 0.5) * n - 0.5

    table_size 为 None（不降采样）时返回 1。
    """
    sw, sh = int(source_size[0]), int(source_size[1])
    if sw <= 0 or sh <= 0:
        raise ValueError(f'源图尺寸必须为正，收到 {source_size}。')
    if table_size is None:
        return 1
    tw, th = int(table_size[0]), int(table_size[1])
    if tw <= 0 or th <= 0:
        raise ValueError(f'查找表网格必须为正，收到 {table_size}。')
    if tw > sw or th > sh:
        raise ValueError(f'查找表网格不能大于源图尺寸：源图 {sw}x{sh}，收到 {tw}x{th}。')

    if sw % tw == 0 and sh % th == 0 and sw // tw == sh // th:
        return sw // tw

    allowed = ', '.join(f'{sw // n}x{sh // n}' for n in table_grid_factors(sw, sh)
                        if n > 1)
    raise ValueError(
        '查找表只能按整数倍等比例降采样。\n'
        f'当前源图 {sw}x{sh}，{tw}x{th} 分别对应 {sw / tw:g}x 与 {sh / th:g}x，'
        '会破坏 BirdView 公制纵横比。\n'
        + (f'例如可使用 {allowed}。' if allowed else '当前源图尺寸没有可用的等比倍率。'))


def table_grid_factors(width: int, height: int, min_width: int = 80,
                       min_height: int = 45) -> list:
    """能同时整除宽高、且结果网格不至于小到没用的所有倍率（升序，含 1）。

    界面直接拿它生成"降采样倍率"下拉项，所以允许倍率只有这一处定义——让用户自己填
    宽高就一定会有人填出 320x240 那种 4x/3x 的组合。
    """
    limit = math.gcd(int(width), int(height))
    return [n for n in range(1, limit + 1)
            if limit % n == 0 and width // n >= min_width and height // n >= min_height]


def resample_map_pair(map_x: np.ndarray, map_y: np.ndarray,
                      size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """把一组映射表重采样到目标网格。

    只在所有贡献像素都有效时才输出有效值：地平线两侧的坐标相差极大，跨越边界插值
    会算出根本不存在的采样点，宁可把有效边界收缩一格。
    """
    h, w = map_x.shape
    tw, th = size
    if (w, h) == (tw, th):
        return map_x, map_y

    valid = (np.isfinite(map_x) & np.isfinite(map_y)
             & (map_x >= 0.0) & (map_y >= 0.0))
    interp = cv2.INTER_AREA if (tw < w or th < h) else cv2.INTER_LINEAR
    keep = cv2.resize(valid.astype(np.float32), (tw, th), interpolation=interp) >= 0.999

    out = []
    for m in (map_x, map_y):
        vals = cv2.resize(np.where(valid, m, 0.0).astype(np.float32), (tw, th),
                          interpolation=interp).astype(np.float64)
        out.append(np.where(keep, vals, np.nan))
    return out[0], out[1]


@dataclass
class MapPair:
    """一份"最终交付"的映射表。

    这是整条打表链路的分水岭。重采样、无效哨兵、定点量化只在这里做一次，
    之后序列化与批量测试消费的是同一个对象——而不是"写文件走一条链、
    验证走另一条链"，那正是导出的表与验过的表其实是两份数据的老毛病。

    `x`/`y` 里放的是 C 端重建出来的坐标（bin/c 已按 Q 定点还原），
    所以拿它跑批量测试等价于拿真实交付物跑。
    """

    x: np.ndarray                      # (H, W) 输出像素 -> 源图采样坐标 X
    y: np.ndarray                      # 同上，Y
    source_size: Tuple[int, int]       # 采样坐标所在的原图分辨率 (W, H)
    fixed_point: Optional[int] = None  # bin/c 的定点位数；文本格式为 None
    qx: Optional[np.ndarray] = None    # 已量化的 int16 载荷（仅 bin/c）
    qy: Optional[np.ndarray] = None

    @property
    def size(self) -> Tuple[int, int]:
        """当前映射表的网格尺寸 (W, H)，直接取 x/y 数组的实际形状。

        上层若对映射表做过重采样，这里自然反映重采样后的网格。
        """
        return self.x.shape[1], self.x.shape[0]

    @property
    def invalid(self) -> np.ndarray:
        """无效点掩码；-1 是两套表约定的哨兵，任一分量落到图上即视为无效。"""
        return (self.x < 0) | (self.y < 0)


def table_spaces(tag: str) -> dict:
    """一套表的索引空间与取值空间。

    四种表的语义各不相同，不能统一写一个 coordinate_space：
      undistort reverse    : 索引 = 去畸变图像素，取值 = 原图采样坐标
      undistort forward    : 索引 = 原图像素，    取值 = 去畸变图落点
      undistort_ipm reverse: 索引 = BirdView 像素，取值 = 原图采样坐标
      undistort_ipm forward: 索引 = 原图像素，    取值 = BirdView 落点
    把 forward 表也标成 raw_distorted_px 会让 C 端直接理解反。
    """
    is_ipm = tag.startswith('undistort_ipm')
    is_reverse = tag.endswith('reverse')
    out_space = 'birdview_px' if is_ipm else 'undistorted_px'
    if is_reverse:
        return {'mapping_direction': 'dst_to_src',
                'index_space': out_space,
                'value_space': 'raw_distorted_px',
                'index_desc': f'{out_space}（输出图像素）',
                'value_desc': '原始畸变图上的采样坐标'}
    return {'mapping_direction': 'src_to_dst',
            'index_space': 'raw_distorted_px',
            'value_space': out_space,
            'index_desc': '原始畸变图像素',
            'value_desc': f'{out_space}（输出图上的落点）'}


def pixel_grid(size: Tuple[int, int]) -> np.ndarray:
    """返回 (H*W, 2) 的像素坐标点集，按行优先展开。"""
    w, h = size
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    return np.column_stack((xs.ravel(), ys.ravel()))


def mask_out_of_range(pts: np.ndarray, size: Tuple[int, int],
                      extra_valid: Optional[np.ndarray] = None) -> np.ndarray:
    """把越界、非有限或未通过附加判据的点置 -1。

    -1 是交付给 C 端的无效哨兵。越界坐标若原样写出去，C 端会当成合法采样点用，
    所以必须和非有限值一样统一标掉。
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2).copy()
    w, h = size
    valid = np.isfinite(p).all(axis=1)
    valid &= (p[:, 0] >= 0.0) & (p[:, 0] <= w - 1.0)
    valid &= (p[:, 1] >= 0.0) & (p[:, 1] <= h - 1.0)
    if extra_valid is not None:
        valid &= extra_valid
    p[~valid] = -1.0
    return p


def undistorted_grid(K: np.ndarray, D: np.ndarray, Knew: np.ndarray,
                     size: Tuple[int, int]) -> np.ndarray:
    """每个原始畸变像素对应的去畸变坐标，(H*W, 2)。"""
    grid = pixel_grid(size).reshape(-1, 1, 2)
    return cv2.undistortPoints(grid, K, D, P=Knew).reshape(-1, 2)


def distort_points(pts_undist: np.ndarray, K: np.ndarray, D: np.ndarray,
                   Knew: np.ndarray) -> np.ndarray:
    """去畸变像素 -> 原始畸变像素。畸变正向模型是闭式的，无需迭代。"""
    norm = apply_homography(np.linalg.inv(Knew), pts_undist)
    obj = np.column_stack((norm, np.ones(norm.shape[0])))
    zeros = np.zeros(3, dtype=np.float64)
    proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), zeros, zeros, K, D)
    return proj.reshape(-1, 2)


def build_composite_reverse_map(K: np.ndarray, D: np.ndarray, Knew: np.ndarray,
                                H: np.ndarray, H0: np.ndarray, sign: float,
                                size: Tuple[int, int]
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """BirdView 每个输出像素 -> 原始畸变图采样坐标，无效点标 -1。

    两道判据都不能省：
    1. 中间的去畸变坐标必须落在地平线的地面侧。地平线另一侧（相机后方）的输出像素
       反投后同样会落在图内，坐标有限、下游无法识别，采样出来是镜像鬼影。
    2. 中间的去畸变坐标必须落在图内。畸变多项式只在有效成像域内可逆，域外的点会被
       径向项折返到图内某个无关位置。
    """
    w, h = size
    und = apply_homography(np.linalg.inv(H), pixel_grid(size))

    valid = np.isfinite(und).all(axis=1)
    valid &= sign * homography_denominator(H0, und) >= DEN_EPS
    valid &= (und[:, 0] >= 0.0) & (und[:, 0] <= w - 1.0)
    valid &= (und[:, 1] >= 0.0) & (und[:, 1] <= h - 1.0)

    dist = np.full_like(und, np.nan)
    if valid.any():
        dist[valid] = distort_points(und[valid], K, D, Knew)

    dist = mask_out_of_range(dist, size, valid)
    return dist[:, 0].reshape(h, w), dist[:, 1].reshape(h, w)


__all__ = [
    'MapPair',
    'build_composite_reverse_map',
    'distort_points',
    'mask_out_of_range',
    'pixel_grid',
    'resample_map_pair',
    'table_spaces',
    'undistorted_grid',
]
