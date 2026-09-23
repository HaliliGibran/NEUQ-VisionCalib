"""把一次导出当成小型验收项目：检查矩阵、图像效果与以下导出物：
两类查找表（去畸变、去畸变+逆透视），每类各含正向/反向，共 4 组 LUT。

交付给嵌入式 C 端的是预先算好的坐标；方向写反、坐标系错一层或量化溢出，往往仍能
生成一张“有画面”的图，所以只看文件存在远远不够。本脚本按由几何到字节的顺序检查：

  1. ``H`` 的四个几何不变量：尺寸、邻边垂直、朝向、参考原点 → 锚点
  2. LUT 网格是否为源图的整数倍等比缩小
  3. ``undistort/reverse`` 重建图是否等于 OpenCV ``undistort``
  4. ``undistort_ipm/reverse`` 重建图是否等于 OpenCV 两步参考链
  5. IPM forward/reverse 往返误差（降采样时只作诊断）
  6. 导出 LUT 的 X/Y 无效哨兵是否同步、值是否统一
  7. 落盘 forward/reverse 是否分别等于流水线应生成的最终数组

这里故意使用两类 oracle（判定参照）：第 3、4 项走 OpenCV 的独立图像 API，回答
“数学方向和组合结果对不对”；第 7 项按本项目流水线重建数组，回答“重采样、量化和
序列化有没有忠实落盘”。两个结果互相一致不等于两者都正确，因此不能只让 forward
与 reverse 互相证明。第 5 项在全分辨率下能补充检验方向，在降采样后则不能当硬门槛：
栅格化、面积重采样与定点量化都不可逆，严格互逆本来就不再是数学不变量。

第 6 项只检查哨兵的成对和值域约定，不判断有效区是否连通。无效区域是否出现在正确
位置，由第 7 项对流水线重算与落盘表的 mask 逐点比较；这比用连通性作 hard gate
更直接。每项 docstring 还说明了失败通常指向哪一层。

落盘格式自动识别：逗号分隔文本（MapW.txt）、int16 定点二进制（MapW.bin）、
C 头文件（Map.h）。读取统一走主脚本的 core.load_map_pair，这样"脚本怎么读表"
与"程序怎么读表"永远是同一套语义。

用法:
    python tools/verify_outputs.py [工程根目录]
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# 本脚本位于 <工程根>/tools/，主脚本在 <工程根>/src/，先把 src 挂到 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'src'))

import neuq_vision_calib as core  # noqa: E402

VISIBLE_FRAC = 0.005  # 允许的"肉眼可见差异"像素占比（灰度差 > 8）

# 全分辨率下 forward(reverse(p)) 的判据。刻意用分位数而不是最大值：
# 逆透视在远场是强非线性的，双线性插一格的残差在那里天然可以到几个像素，
# 用 max 当判据等于让极少数近地平线的点决定结论。这两个数是标定出来的——
# 一对正确的表实测 p50≈0.14 / p95≈0.71（留 2~3 倍余量），而"最近邻取整"那个
# 老实现 p50≈4.57，会在这里差一个数量级地失败。
ROUNDTRIP_P50_PX = 0.5
ROUNDTRIP_P95_PX = 2.0

BIN_SENTINEL = core.BIN_SENTINEL

# 由 matrices.json 填入，bin/c 还原坐标时要用
FIXED_POINT = core.TABLE_FIXED_POINT

# bin 是裸数组，文件里没有维度信息，必须靠 matrices.json 与标定分辨率还原
TABLE_SHAPE: tuple[int, int] | None = None   # (W, H)
IMAGE_SIZE: tuple[int, int] = (0, 0)         # 源图尺寸，来自 calib.json
KNEW: np.ndarray | None = None               # 去畸变输出矩阵，来自 matrices.json
IS_TEXT_TABLE = True                         # 表格式是否为 %.2f 文本，影响量化容差
SKIPPED = 0                                  # 因降采样而跳过的检查条数
DIAGNOSED = 0                                # 只报告、不判定通过/失败的条数


def undistort_reference(src: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """按导出时用的同一个 Knew 做去畸变，作为比对参考。

    表的坐标是 build_composite_reverse_map 用 Knew 反投影出来的；若这里用 K
    去构造参考，只在 UNDIST_ALPHA 为 None（Knew == K）时才对得上。
    """
    return cv2.undistort(src, K, D, None, K if KNEW is None else KNEW)



def load_pair(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取一个方向的 MapW/MapH。

    直接复用主脚本的读取实现（core.load_map_pair），这样"脚本怎么读这张表"
    与"程序怎么读这张表"永远是一套语义，不会各自漂移。
    """
    pair = core.load_map_pair(folder, IMAGE_SIZE, TABLE_SHAPE, FIXED_POINT)
    return pair.x, pair.y


def describe_pair(folder: Path) -> str:
    """描述一个方向的落盘格式与网格，用于打印。"""
    if (folder / 'MapW.txt').is_file():
        m = np.loadtxt(folder / 'MapW.txt', delimiter=',')
        return f'逗号分隔文本，网格 {m.shape[1]}x{m.shape[0]}'
    if (folder / 'MapW.bin').is_file():
        grid = f'，网格 {TABLE_SHAPE[0]}x{TABLE_SHAPE[1]}' if TABLE_SHAPE else ''
        return f'int16 定点二进制 Q{FIXED_POINT}{grid}'
    header = folder / 'Map.h'
    if header.is_file():
        text = header.read_text(encoding='utf-8')
        mm = re.search(r'网格:\s*(\d+)\s*x\s*(\d+)', text)
        return f'C 头文件，网格 {mm.group(1)}x{mm.group(2)}' if mm else 'C 头文件'
    return '未知格式'


def apply_homography(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """对 ``(N,2)`` 点集应用单应；本地实现让验收脚本不依赖主流程的同名 helper。"""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack((p, np.ones(p.shape[0])))
    out = (M @ hom.T).T
    with np.errstate(divide='ignore', invalid='ignore'):
        out = out[:, :2] / out[:, 2:3]
    return out


def skip(name: str, why: str) -> bool:
    """打印一条"已跳过"的检查。

    计为通过，但用 [跳过] 而不是 [通过] 标出来——跳过不等于验证过，
    混淆这两者正是这类自检脚本最容易被误信的地方。
    """
    global SKIPPED
    SKIPPED += 1
    print(f'  [跳过] {name}: {why}')
    return True


def report(name: str, ok: bool, detail: str) -> bool:
    """打印一条检查结果。"""
    print(f'  [{"通过" if ok else "失败"}] {name}: {detail}')
    return ok


def diagnose(name: str, detail: str) -> bool:
    """打印一条只报告、不判定的观测。

    与 skip() 的区别：skip 是"这条没法验"，diagnose 是"这条验了，但它不该当
    hard gate"。降采样表的互逆性就属于后者——两张离散表分别经过栅格化、
    INTER_AREA 重采样与定点量化，互逆只是近似成立，没有可靠的固定阈值。
    """
    global DIAGNOSED
    DIAGNOSED += 1
    print(f'  [诊断] {name}: {detail}')
    return True


def grid_from_full(v: np.ndarray, step: float) -> np.ndarray:
    """全分辨率坐标 -> 降采样网格坐标（像素中心对齐，与 cv2.resize 同一约定）。

    不能写成 v / step。cv2.resize 的语义是像素**中心**对应，正变换是
    full = (small + 0.5) * step - 0.5，所以反变换必须带上这半个像素：
    small = (full + 0.5) / step - 0.5。320x240 下这个偏差是 x 方向约 1.5、
    y 方向约 1.0 个全分辨率像素，足以混进"互逆误差"里假装成表的问题。
    """
    return (v + 0.5) / step - 0.5


def full_from_grid(i: np.ndarray, step: float) -> np.ndarray:
    """降采样网格索引 -> 它代表的全分辨率坐标。grid_from_full 的逆。"""
    return (i + 0.5) * step - 0.5


def bilinear_at(a: np.ndarray, b: np.ndarray, gx: np.ndarray, gy: np.ndarray):
    """在 (a, b) 两张同形表上做双线性采样，返回 (va, vb, 可用掩码)。

    四邻域必须**全部**有效才给值：无效点是哨兵 -1，插进来会得到一个既不是
    坐标也不是哨兵的中间数。这与 core.resample_map_pair 的判据一致——
    宁可把有效边界收缩一格，也不输出不存在的采样点。
    """
    th, tw = a.shape
    inside = (gx >= 0) & (gx <= tw - 1) & (gy >= 0) & (gy <= th - 1)
    x0 = np.clip(np.floor(np.where(inside, gx, 0)), 0, tw - 2).astype(np.int64)
    y0 = np.clip(np.floor(np.where(inside, gy, 0)), 0, th - 2).astype(np.int64)
    wx = np.clip(gx - x0, 0.0, 1.0)
    wy = np.clip(gy - y0, 0.0, 1.0)

    def valid(j, i):
        return (a[j, i] >= 0) & (b[j, i] >= 0)

    usable = (inside & valid(y0, x0) & valid(y0, x0 + 1)
              & valid(y0 + 1, x0) & valid(y0 + 1, x0 + 1))

    def lerp(m):
        top = (1 - wx) * m[y0, x0] + wx * m[y0, x0 + 1]
        bot = (1 - wx) * m[y0 + 1, x0] + wx * m[y0 + 1, x0 + 1]
        return (1 - wy) * top + wy * bot

    return lerp(a), lerp(b), usable


def full_resolution() -> bool:
    """表是否与源图同尺寸。不同尺寸时逐像素比对无意义。"""
    return TABLE_SHAPE is None or TABLE_SHAPE == IMAGE_SIZE


def pipeline_composite_maps(calib: dict, matrices: dict) -> dict:
    """按流水线重算 undistort_ipm 的正反两张表（重采样、量化之前）。

    这是“落盘忠实度”的流水线 oracle：复刻 core.export_composite_tables 里的生成步骤，
    逐点比对落盘结果。**正反都要算** —— 以前只重算 reverse，forward 的数学
    错了（方向反了、无效判据不一致）在这里是看不出来的，而 forward 恰恰是
    "互逆"那一条唯一能间接碰到的地方，那条又天然带着插值误差。
    """
    K = np.asarray(calib['camera_matrix'], dtype=np.float64).reshape(3, 3)
    D = np.asarray(calib['dist_coeffs'], dtype=np.float64).ravel()
    Knew = np.asarray(matrices['Knew'], dtype=np.float64).reshape(3, 3)
    H = np.asarray(matrices['H'], dtype=np.float64).reshape(3, 3)
    H0 = np.asarray(matrices['H0'], dtype=np.float64).reshape(3, 3)
    sign = float(matrices['horizon_sign'])
    w, h = IMAGE_SIZE

    rev = core.build_composite_reverse_map(K, D, Knew, H, H0, sign, IMAGE_SIZE)

    und = core.undistorted_grid(K, D, Knew, IMAGE_SIZE)
    valid = np.isfinite(und).all(axis=1)
    valid &= sign * core.homography_denominator(H0, und) >= core.DEN_EPS
    fwd = core.mask_out_of_range(core.apply_homography(H, und), IMAGE_SIZE, valid)
    return {
        'reverse': rev,
        'forward': (fwd[:, 0].reshape(h, w), fwd[:, 1].reshape(h, w)),
    }


def compare_to_pipeline(root: Path, direction: str,
                        expect: tuple) -> bool:
    """把一个方向的落盘表与流水线重算结果逐点比对。

    无效掩码不一致通常指向边界/哨兵处理；掩码一致但数值超容差，通常指向重采样、
    Q 格式解释或序列化。forward 与 reverse 分开比较，避免一侧的错误被另一侧掩盖。
    """
    fx, fy = expect
    if IMAGE_SIZE != TABLE_SHAPE:
        fx, fy = core.resample_map_pair(fx, fy, TABLE_SHAPE)
    ax, ay = load_pair(root / 'lookup_table' / 'undistort_ipm' / direction)

    name = f'落盘 {direction} == 流水线重算'
    if ax.shape != fx.shape:
        return report(name, False,
                      f'网格不符: 落盘 {ax.shape[1]}x{ax.shape[0]}，'
                      f'重算 {fx.shape[1]}x{fx.shape[0]}')

    # NaN 既不满 >=0 也不满 <0，所以"无效"必须显式写成 ~isfinite | <0，
    # 只用 <0 判断会把重算侧的无效点漏掉一大半。
    bad_expect = ~np.isfinite(fx) | (fx < 0.0) | ~np.isfinite(fy) | (fy < 0.0)
    bad_actual = (ax < 0.0) | (ay < 0.0)

    mismatch = int(np.count_nonzero(bad_expect != bad_actual))
    good = ~bad_expect
    err = np.hypot(ax[good] - fx[good], ay[good] - fy[good])
    max_err = float(err.max()) if err.size else 0.0
    tol = quant_tolerance()

    ok = mismatch == 0 and max_err <= tol
    return report(name, ok,
                  f'最大差 {max_err:.4f} px（容差 {tol:.4f}），'
                  f'有效点 {int(good.sum())}，无效判定不一致 {mismatch} 个')


def check_grid_isotropy() -> bool:
    """检查 2：表网格必须是源图的**整数倍等比**缩小。

    这一条是交付级的公制几何门槛，不是格式挑剔。1280x720 降成 320x240 时 x 缩 4 倍、
    y 缩 3 倍，小图里的公制尺度就变成 x=scale/4、y=scale/3——一个物理上 45x45 cm 的
    正方形会成为 53.8 x 71.7 px、宽高比 0.75 的竖长矩形。C 端若直接拿这张图找线、
    算角度、曲率、横向误差，赛道被横向压缩 25%，全部失真。而 batch_test() 输出的
    正是这张真实的小图，所以风险不是理论上的。

    等比之后 C 端只需要一个倍率 n：full = (small + 0.5) * n - 0.5。
    失败通常不是镜头标定问题，而是导出尺寸/metadata 违反了公制纵横比约定。
    """
    sw, sh = IMAGE_SIZE
    tw, th = (TABLE_SHAPE if TABLE_SHAPE else IMAGE_SIZE)
    try:
        n = core.downsample_factor(IMAGE_SIZE, (tw, th))
    except ValueError as exc:
        return report('表网格为整数倍等比', False, str(exc).replace('\n', ' '))
    return report('表网格为整数倍等比',
                  True,
                  f'源图 {sw}x{sh} -> 网格 {tw}x{th}，倍率 {n}x（x/y 同一倍率）；'
                  f'C 端换算 full = (small + 0.5) * {n} - 0.5')


def check_export_fidelity(root: Path, calib: dict, matrices: dict) -> bool:
    """检查 7：落盘的正反两张表是否忠实等于流水线重算的结果。

    覆盖重采样与定点量化两步。检查 3/4 验证的是数学（只有全分辨率下才可比），
    这一项验证的是"算出来的东西有没有原样写进文件"，任何网格、任何格式都成立，
    因此它才是降采样交付物的主判据。失败通常落在文件格式、网格重采样、Q 位数或
    哨兵序列化；它与流水线共享数学 helper，所以不能替代第 3、4 项的独立 OpenCV oracle。
    """
    # core 已在模块顶部导入（那时就把 src/ 挂上了 sys.path），这里只需按用户给的
    # 工程根重绑定路径常量，并把表格格式同步过去，好让 core.load_map_pair 读得对。
    core.configure_paths(root=root)
    core.TABLE_SIZE = TABLE_SHAPE
    core.TABLE_FIXED_POINT = FIXED_POINT

    expect = pipeline_composite_maps(calib, matrices)
    results = [compare_to_pipeline(root, d, expect[d]) for d in ('reverse', 'forward')]
    return all(results)


def quant_tolerance() -> float:
    """落盘量化的理论最大误差。

    单轴最多差半个量化步长（文本是 %.2f 即 0.005，定点是 0.5/2^N）；
    两轴合成后是二维欧氏距离，最大 sqrt(2) 倍。以前写成 `0.01 + step`，
    对文本表偏松约 7 倍、对 Q0 偏松到 1.01 px，几乎没有判别力。以 Q4 为例，
    单轴上限 ``0.5/16`` px，二维上限 ``sqrt(2)*0.5/16 ≈ 0.0442`` px。
    """
    half_axis = 0.005 if IS_TEXT_TABLE else 0.5 / (1 << FIXED_POINT)
    return float(np.sqrt(2.0) * half_axis + 1e-6)


_REF_CACHE: dict = {}


def load_reference_image(root: Path, size: tuple[int, int]) -> np.ndarray:
    """取一张可用于比对的源图。

    必须是标定时用的同一张。ipm_input/ 里通常躺着多张候选，随手挑一张会让
    "表重建 vs 参考实现"的比较彻底失去意义——两张不同的图本来就该不一样，
    差异率会直接飙到几十个百分点，看起来像表坏了。

    判定顺序（越往下越弱）：
      1. ipm_state.json 记的相对路径 + src_sha256 都对上  → 这就是当时那张
      2. 路径失效但同名图哈希能对上                        → 依然是当时那张
      3. 只有老状态文件（没记哈希）                        → 按路径/文件名取，并明确警告
      4. 都找不到                                          → 报错退出，不猜
    """
    key = (str(root), size)
    if key in _REF_CACHE:
        return _REF_CACHE[key]

    candidates: list[Path] = []
    recorded: str | None = None
    want_sha: str | None = None
    state_path = root / 'matrix' / 'ipm_state.json'
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            state = {}
        recorded = state.get('src_image')
        want_sha = state.get('src_sha256')

    if recorded:
        p = Path(recorded)
        recorded_path = p if p.is_absolute() else (root / p)
        if recorded_path.is_file():
            candidates.append(recorded_path)
        by_name = root / 'ipm_input' / p.name
        if by_name.is_file() and by_name not in candidates:
            candidates.append(by_name)

    preferred = root / 'ipm_input' / 'UnInverseImage.jpg'
    if preferred.is_file() and preferred not in candidates:
        candidates.append(preferred)
    for extra in sorted((root / 'ipm_input').glob('*.jpg')):
        if extra not in candidates:
            candidates.append(extra)

    if not candidates:
        raise SystemExit(f'{root / "ipm_input"} 下没有可用于比对的图片。')

    chosen = None
    if want_sha:
        # 有哈希就认哈希：只有内容完全一致的才算"当时那张"
        for c in candidates:
            if _sha256(c) == want_sha:
                chosen = c
                break
        if chosen is None:
            raise SystemExit(
                '找不到与 ipm_state.json 记录的内容哈希一致的原图，无法做有意义的比对。\n'
                f'  期望 sha256 = {want_sha}\n'
                f'  已试过: {", ".join(c.name for c in candidates)}\n'
                '把标定时用的那张原图放回 ipm_input/ 再跑，或重做逆透视标定。')
        if chosen.name != Path(recorded).name:
            print(f'  提示: 记录的路径已失效，按内容哈希找到同一张图 {chosen.name}。')
    else:
        chosen = candidates[0]
        print('  警告: ipm_state.json 里没有内容哈希（旧版本写的），'
              f'只能按路径/文件名取 {chosen.name}，无法证明是当时那张。')

    src = core.safe_imread(chosen)
    if src is None:
        raise SystemExit(f'无法读取 {chosen}')
    print(f'  比对源图: {chosen.name}')
    src = core.normalize_camera_image(src, size, f'验证参考图 {chosen.name}')
    _REF_CACHE[key] = src
    return src


def _sha256(path: Path) -> str:
    """文件的 SHA-256，用于确认"是不是当时那张原图"。"""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def compare_images(a: np.ndarray, b: np.ndarray,
                   invalid: np.ndarray | None) -> tuple[float, float, float]:
    """比较两张图，返回 (可见差异占比, p99.9 差, 最大差)。

    表里的坐标按 %.2f 量化，边缘像素会因亚像素舍入产生跳变，用"最大差"当判据会被
    极少数边缘点绑架。真正有意义的是"有多少像素差到肉眼可见"（灰度差 > 8）。
    """
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if invalid is not None and invalid.any():
        diff[invalid] = 0.0
    per_pixel = diff.max(axis=2) if diff.ndim == 3 else diff
    visible = float((per_pixel > 8).mean())
    return visible, float(np.percentile(per_pixel, 99.9)), float(per_pixel.max())


def check_homography(matrices: dict) -> bool:
    """检查 1: H 把源四点映射成矩形，且尺寸、旋转、参考原点都与标定参数一致。

    H = T_canvas @ S(scale) @ R(heading) @ T(-ref) @ H0，所以只要 heading != 0，输出
    矩形本来就是旋转的——用"轴对齐"当判据在非零 heading 下必然误判。这里验证四个对
    任意旋转都成立的不变量：

      1. |TL->TR| == 物理宽 × scale，|TL->BL| == 物理高 × scale
      2. 邻边垂直：(TL->TR) · (TL->BL) == 0
      3. atan2(TR-TL) == heading（矩形对边平行，角度在模 180° 意义下比较）
      4. **去畸变图底边中点** 经 H 之后正好落在 anchor 上

    上述第 4 个不变量不能写成“标定矩形中心 == anchor”：H0 之后还有
    ``T(-ref)``，落在 anchor 上的是**逆透视坐标参考原点**（底边中点对应的
    地面点），不是标定矩形中心。
    任一项失败通常说明 H 的组合顺序、角点对应、物理尺寸/比例尺或参考原点语义错了。
    """
    H = np.asarray(matrices['H'], dtype=np.float64).reshape(3, 3)
    quad = np.asarray(matrices['src_quad_tl_tr_bl_br'], dtype=np.float64).reshape(4, 2)
    scale = float(matrices['scale_px_per_cm'])
    phys_w = float(matrices['phys_w_cm'])
    phys_h = float(matrices['phys_h_cm'])
    heading = float(matrices.get('heading_deg') or 0.0)
    anchor_x = float(matrices.get('anchor_x') if matrices.get('anchor_x') is not None else 0.5)
    anchor_y = float(matrices.get('anchor_y') if matrices.get('anchor_y') is not None else 0.5)

    out = apply_homography(H, quad)
    tl, tr, bl = out[0], out[1], out[2]

    w_px = float(np.linalg.norm(tr - tl))
    h_px = float(np.linalg.norm(bl - tl))
    expect_w, expect_h = phys_w * scale, phys_h * scale
    size_ok = abs(w_px - expect_w) < 0.5 and abs(h_px - expect_h) < 0.5

    e1, e2 = tr - tl, bl - tl
    cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    ortho_ok = abs(cos) < 1e-3

    rot_deg = float(np.degrees(np.arctan2(e1[1], e1[0])))
    delta = (rot_deg - heading + 90.0) % 180.0 - 90.0
    rot_ok = abs(delta) < 0.05

    canvas_w, canvas_h = IMAGE_SIZE
    want_cx, want_cy = anchor_x * (canvas_w - 1), anchor_y * (canvas_h - 1)
    # 参考原点的像素位置优先取 matrices.json 记录的那一份；老文件没有就按约定算。
    px = matrices.get('ground_origin_image_px')
    ref_px = (np.asarray(px, dtype=np.float64).reshape(1, 2) if px
              else np.array([[(canvas_w - 1) / 2.0, float(canvas_h - 1)]]))
    ox, oy = apply_homography(H, ref_px)[0]
    origin_ok = abs(ox - want_cx) < 0.5 and abs(oy - want_cy) < 0.5

    return report(
        'H 映射源四点成矩形（尺寸/垂直/旋转/参考原点）',
        size_ok and ortho_ok and rot_ok and origin_ok,
        f'尺寸 {w_px:.1f}x{h_px:.1f} 期望 {expect_w:.1f}x{expect_h:.1f} '
        f'{"OK" if size_ok else "不符"}；'
        f'邻边 |cos|={abs(cos):.2e} {"OK" if ortho_ok else "不垂直"}；'
        f'旋转 {rot_deg:.2f}° 期望 {heading:.2f}°（偏差 {delta:+.3f}°）'
        f'{"OK" if rot_ok else "不符"}；'
        f'参考原点 ({ox:.1f},{oy:.1f}) 期望 anchor ({want_cx:.1f},{want_cy:.1f})'
        f'{"OK" if origin_ok else "不符"}')


def check_undistort_tables(root: Path, calib: dict) -> bool:
    """检查 3：去畸变 reverse 表重建图是否等于独立的 ``cv2.undistort``。

    这项想证明 destination → source 方向、K/D/Knew 与双线性取样都一致。失败通常
    指向 reverse 表方向写反、Knew 选错、表值坐标系错误，或落盘精度超出预期。
    降采样后输出尺寸不同，逐像素图像比较失去同位语义，因此明确跳过而不伪装成通过。
    """
    if not full_resolution():
        return skip('undistort/reverse 表 == cv2.undistort',
                    f'表已降采样到 {TABLE_SHAPE[0]}x{TABLE_SHAPE[1]}，'
                    '输出尺寸与源图不同，逐像素比对无意义（改由"落盘表 == 流水线重算"覆盖）')

    K = np.asarray(calib['camera_matrix'], dtype=np.float64).reshape(3, 3)
    D = np.asarray(calib['dist_coeffs'], dtype=np.float64).ravel()
    size = (int(calib['image_width']), int(calib['image_height']))
    src = load_reference_image(root, size)

    map_x, map_y = load_pair(root / 'lookup_table' / 'undistort' / 'reverse')
    invalid = (map_x < 0) | (map_y < 0)
    mx = np.where(invalid, -1e6, map_x).astype(np.float32)
    my = np.where(invalid, -1e6, map_y).astype(np.float32)
    via_table = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    reference = undistort_reference(src, K, D)

    visible, p999, mx_diff = compare_images(via_table, reference, invalid)
    return report('undistort/reverse 表 == cv2.undistort', visible <= VISIBLE_FRAC,
                  f'可见差异像素 {100 * visible:.4f}%，p99.9={p999:.1f}，'
                  f'最大 {mx_diff:.0f}，无效点 {100 * invalid.mean():.1f}%')


def check_composite_tables(root: Path, calib: dict, matrices: dict) -> bool:
    """检查 4：复合 reverse 表是否等于 OpenCV 的“去畸变 → warpPerspective”。

    这是端到端的独立图像 oracle：一侧用一次 LUT remap，另一侧调用两套 OpenCV API。
    失败通常说明畸变与 H 的复合方向、地平线有效区、Knew 或插值约定不一致。
    """
    if not full_resolution():
        return skip('undistort_ipm/reverse 表 == warpPerspective',
                    f'表已降采样到 {TABLE_SHAPE[0]}x{TABLE_SHAPE[1]}，'
                    '输出尺寸与源图不同，逐像素比对无意义（改由"落盘表 == 流水线重算"覆盖）')

    K = np.asarray(calib['camera_matrix'], dtype=np.float64).reshape(3, 3)
    D = np.asarray(calib['dist_coeffs'], dtype=np.float64).ravel()
    H = np.asarray(matrices['H'], dtype=np.float64).reshape(3, 3)
    size = (int(calib['image_width']), int(calib['image_height']))
    src = load_reference_image(root, size)

    map_x, map_y = load_pair(root / 'lookup_table' / 'undistort_ipm' / 'reverse')
    invalid = (map_x < 0) | (map_y < 0)
    mx = np.where(invalid, -1e6, map_x).astype(np.float32)
    my = np.where(invalid, -1e6, map_y).astype(np.float32)
    via_table = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    undist = undistort_reference(src, K, D)
    reference = cv2.warpPerspective(undist, H, size, flags=cv2.INTER_LINEAR)

    visible, p999, mx_diff = compare_images(via_table, reference, invalid)
    ok = report('undistort_ipm/reverse 表 == warpPerspective', visible <= VISIBLE_FRAC,
                f'可见差异像素 {100 * visible:.4f}%，p99.9={p999:.1f}，'
                f'最大 {mx_diff:.0f}，有效区 {100 * (~invalid).mean():.1f}%')

    # 顺带确认落盘的结果图确实由这张表生成。必须在有效区内比：无效区里表是黑的，
    # 而 warpPerspective 会照常填上标定矩形以外的画面，不屏蔽就会得出误导性的差异率。
    result_path = root / 'ipm_output' / 'UnDistortionInverseImage.jpg'
    if result_path.is_file():
        produced = core.safe_imread(result_path)
        if produced is not None and produced.shape[:2] == via_table.shape[:2]:
            vis2, _p, m2 = compare_images(produced, via_table, invalid)
            print(f'         结果图 vs 表重建（限有效区）: 可见差异 {100 * vis2:.4f}%，'
                  f'最大 {m2:.0f}（JPEG 有损压缩会带来边缘抖动）')
    return ok


def check_forward_reverse(root: Path) -> bool:
    """检查 5：逆透视的 forward 与 reverse 是否近似互为逆映射。

    做法：reverse 给出连续的原图坐标 -> 换算成 forward 表的连续网格坐标 ->
    **双线性**采样 forward -> 应当回到出发的那个输出像素。

    以前这里是 `rint(reverse / step)` 再整数索引 forward，那验的其实是
    "reverse -> 最近邻源网格 -> forward"。逆透视的 forward 局部放大率中位数
    实测 24 倍，0.2 个源像素的取整残差会被放大成约 5 个输出像素；实测误差与
    "取整残差 x 局部雅可比"的相关系数 0.9917，在放大 <1.5 的区域往返误差
    p99 只有 0.547 px —— 表是互逆的，是判据自己引入了误差。改成双线性后
    同一对表的中位误差从 4.57 px 降到 0.14 px。

    判定分两档：
      全分辨率表  -> hard gate（分位判据，见 ROUNDTRIP_* 的注释）
      降采样表    -> 只报告。两张表分别经过栅格化、INTER_AREA 重采样与定点
                     量化，这三步都不可逆，"严格互逆"不是必须成立的数学不变量，
                     硬卡阈值只会得到一个随网格与 scale 漂移的假失败。这一档的
                     正确性由检查 7（落盘 == 流水线重算，正反两张都比）保证。
    """
    rev_x, rev_y = load_pair(root / 'lookup_table' / 'undistort_ipm' / 'reverse')
    fwd_x, fwd_y = load_pair(root / 'lookup_table' / 'undistort_ipm' / 'forward')
    th, tw = rev_x.shape
    if fwd_x.shape != (th, tw):
        return report('forward/reverse 互逆', False,
                      f'尺寸不一致: reverse {rev_x.shape} vs forward {fwd_x.shape}')

    step_x = IMAGE_SIZE[0] / tw
    step_y = IMAGE_SIZE[1] / th

    ys, xs = np.mgrid[0:th, 0:tw]
    seed = (rev_x >= 0) & (rev_y >= 0)
    if not seed.any():
        return report('forward/reverse 互逆', False, '反向表没有有效点')

    # 出发点：网格索引代表的那个全分辨率输出像素
    out_x = full_from_grid(xs[seed].astype(np.float64), step_x)
    out_y = full_from_grid(ys[seed].astype(np.float64), step_y)
    # reverse 的取值是全分辨率原图坐标，要先换成 forward 表的网格坐标
    gx = grid_from_full(rev_x[seed], step_x)
    gy = grid_from_full(rev_y[seed], step_y)

    back_x, back_y, usable = bilinear_at(fwd_x, fwd_y, gx, gy)
    if not usable.any():
        return report('forward/reverse 互逆', False, 'forward 在这些源点上全是无效值')

    err = np.hypot(back_x[usable] - out_x[usable], back_y[usable] - out_y[usable])
    p50 = float(np.median(err))
    p95 = float(np.percentile(err, 95))
    mx = float(err.max())
    detail = (f'往返误差 p50={p50:.3f} p95={p95:.3f} max={mx:.3f} px，'
              f'可判定覆盖率 {100 * usable.mean():.1f}%'
              f'（四邻域全有效才算），网格 {tw}x{th}')

    if not full_resolution():
        return diagnose('forward/reverse 互逆（降采样，仅诊断）', detail)
    return report('forward/reverse 互逆', p50 <= ROUNDTRIP_P50_PX and p95 <= ROUNDTRIP_P95_PX,
                  detail + f'；判据 p50<={ROUNDTRIP_P50_PX} p95<={ROUNDTRIP_P95_PX}')


def check_sentinel(root: Path) -> bool:
    """检查 6：每组表的 X/Y 无效掩码同步，且解码后统一使用 -1。

    X 无效而 Y 有效会让 C 端拼出不存在的二维坐标；出现其它负值则说明定点哨兵还原
    或文本序列化约定漂移。这项不判断无效区域的几何形状或连通性；无效区域是否出现
    在正确位置，由检查 7 的 bad_expect != bad_actual 逐点 mask 比较负责。
    """
    ok = True
    for name in ('undistort', 'undistort_ipm'):
        for direction in ('reverse', 'forward'):
            map_x, map_y = load_pair(root / 'lookup_table' / name / direction)
            neg_x, neg_y = map_x < 0, map_y < 0
            mismatch = int(np.count_nonzero(neg_x != neg_y))
            only_neg1 = bool(np.all(map_x[neg_x] == -1.0) and np.all(map_y[neg_y] == -1.0))
            frac = 100 * (neg_x | neg_y).mean()
            ok &= (mismatch == 0) and only_neg1
            report(f'{name}/{direction} 哨兵', mismatch == 0 and only_neg1,
                   f'无效点 {frac:.1f}%，XY 不同步 {mismatch} 个，'
                   f'哨兵值{"全为 -1" if only_neg1 else "含 -1 以外的负值"}')
    return ok


def print_table_inventory(root: Path) -> None:
    """打印各组 LUT 的落盘格式、网格与体积。

    不是通过/失败判据，而是让"交付给 C 端的到底是什么"一目了然——
    换过 --table-format 或 --table-size 之后最容易搞混的就是这个。
    """
    total = 0
    for name in ('undistort', 'undistort_ipm'):
        for direction in ('reverse', 'forward'):
            folder = root / 'lookup_table' / name / direction
            if not folder.is_dir():
                continue
            size = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
            total += size
            print(f'  {name}/{direction:<8} {describe_pair(folder):<28} {size / 1024:>9.1f} KB')
    print(f'  {"合计":<20} {"":<28} {total / 1024:>9.1f} KB')


def main() -> int:
    """入口。"""
    global FIXED_POINT, TABLE_SHAPE, IMAGE_SIZE, SKIPPED, IS_TEXT_TABLE, DIAGNOSED

    root = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1
            else Path(__file__).resolve().parent.parent)

    calib_path = root / 'calib_data' / 'calib.json'
    matrix_path = root / 'matrix' / 'matrices.json'
    for p in (calib_path, matrix_path):
        if not p.is_file():
            raise SystemExit(f'缺少 {p}，请先跑完标定与打表。')

    calib = json.loads(calib_path.read_text(encoding='utf-8'))
    matrices = json.loads(matrix_path.read_text(encoding='utf-8'))

    fmt = matrices.get('table_format', 'txt')
    # Q0 是合法配置，不能用 `or 4` 兜底——0 在这里会被当成假值，静默变成 Q4
    _fp = matrices.get('table_fixed_point')
    FIXED_POINT = 4 if _fp is None else int(_fp)
    IS_TEXT_TABLE = fmt == 'txt'

    # 表被重采样过就用重采样后的网格，否则与标定分辨率同尺寸
    IMAGE_SIZE = (int(calib['image_width']), int(calib['image_height']))
    size = matrices.get('table_size_wh')
    TABLE_SHAPE = (int(size[0]), int(size[1])) if size else IMAGE_SIZE
    # 去畸变输出用的相机矩阵。UNDIST_ALPHA 非 None 时 Knew != K，
    # 拿 K 当参考就会用错的去畸变模型去比表，得出满屏假差异。
    KNEW = (np.asarray(matrices['Knew'], dtype=np.float64).reshape(3, 3)
            if matrices.get('Knew') is not None else None)
    SKIPPED = 0
    DIAGNOSED = 0

    print(f'校验 {root}')
    print(f'表格式 {fmt}' + (f'（Q{FIXED_POINT} 定点）' if fmt != 'txt' else '')
          + f'，源图 {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}，表网格 {TABLE_SHAPE[0]}x{TABLE_SHAPE[1]}')
    print(f'去畸变输出矩阵 Knew: {"来自 matrices.json" if KNEW is not None else "缺省，回退用 K"}')

    print('查找表清单:')
    print_table_inventory(root)
    print()

    # 按“几何语义 → 网格契约 → 独立图像 oracle → 双向诊断 → 字节忠实度”排列。
    # 保留逐项结果而不是遇首错就退出：一次运行可以同时告诉同学问题落在哪几层。
    results = [
        # 1. H 自己是否仍表达实测矩形、朝向与参考原点。
        check_homography(matrices),
        # 2. 降采样是否保持 x/y 同一个公制倍率。
        check_grid_isotropy(),
        # 3. 只去畸变的 reverse LUT 对照独立 OpenCV 实现。
        check_undistort_tables(root, calib),
        # 4. 去畸变 + IPM 的复合 reverse LUT 对照 OpenCV 两步链。
        check_composite_tables(root, calib, matrices),
        # 5. forward/reverse 往返；降采样时明确降级为诊断。
        check_forward_reverse(root),
        # 6. 各组表的无效哨兵在 X/Y 两分量上是否成对。
        check_sentinel(root),
        # 7. 正反两张落盘数组各自是否忠实于生成流水线。
        check_export_fidelity(root, calib, matrices),
    ]

    tail = []
    if SKIPPED:
        tail.append(f'{SKIPPED} 项因降采样跳过')
    if DIAGNOSED:
        tail.append(f'{DIAGNOSED} 项只作诊断不判定')
    suffix = f'（其中 {"，".join(tail)}）' if tail else ''
    print(f'\n{sum(results)}/{len(results)} 项通过{suffix}。')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
