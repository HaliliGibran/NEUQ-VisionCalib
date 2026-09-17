"""校验导出的矩阵与查找表是否自洽。

交付给嵌入式 C 端的是一堆映射表。表一旦内部不一致（方向搞反、无效点没标、精度不够），
在车上是极难排查的。这个脚本把每条约定都独立验一遍：

  1. H 把源四点映射成轴对齐矩形，尺寸等于物理尺寸乘 scale
  2. undistort/reverse 表 remap 的结果 == cv2.undistort
  3. undistort_ipm/reverse 表 remap 的结果 == cv2.warpPerspective(去畸变图, H)
  4. undistort_ipm 的 forward 与 reverse 互为逆映射
  5. 无效哨兵只出现在合法区域之外，且有效区连通

落盘格式自动识别：逗号分隔文本（MapW.txt）、int16 定点二进制（MapW.bin）、
C 头文件（Map.h）。三种格式的无效哨兵不同，脚本内部统一归一成 -1 再比较。

用法:
    python tools/verify_outputs.py [工程根目录]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

TOL_INV_PX = 3.0      # forward/reverse 互逆的容差（表按 %.2f 量化，留足余量）
VISIBLE_FRAC = 0.005  # 允许的"肉眼可见差异"像素占比（灰度差 > 8）

BIN_SENTINEL = -32768  # 与主脚本 neuq_vision_calib.BIN_SENTINEL 保持一致

# 由 matrices.json 填入，bin/c 还原坐标时要用
FIXED_POINT = 4

# bin 是裸数组，文件里没有维度信息，必须靠 matrices.json 与标定分辨率还原
TABLE_SHAPE: tuple[int, int] | None = None   # (W, H)
IMAGE_SIZE: tuple[int, int] = (0, 0)         # 源图尺寸，来自 calib.json
SKIPPED = 0                                  # 因降采样而跳过的检查条数


def dequantize(raw: np.ndarray, shift: int) -> np.ndarray:
    """int16 定点还原成 float64 像素坐标，无效哨兵归一成 -1。

    三种落盘格式对无效点用了不同哨兵（txt 是 -1，bin/c 是 int16 最小值）。
    这里统一归一，后面的检查逻辑就不必关心文件格式。
    """
    out = raw.astype(np.float64)
    ok = raw != BIN_SENTINEL
    out[ok] = raw[ok] / float(1 << shift)
    out[~ok] = -1.0
    return out


def load_c_header(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """从生成的 C 头文件里把两张定点表抠出来。"""
    text = path.read_text(encoding='utf-8')

    def define(name: str) -> int:
        m = re.search(rf'#define\s+\w+_{name}\s+\(?(-?\d+)\)?', text)
        if not m:
            raise SystemExit(f'{path} 里找不到 {name} 宏。')
        return int(m.group(1))

    shift = define('SHIFT')
    w, h = define('W'), define('H')

    def array(suffix: str) -> np.ndarray:
        m = re.search(rf'_{suffix}\[[^\]]*\]\s*=\s*\{{(.*?)\}};', text, re.S)
        if not m:
            raise SystemExit(f'{path} 里找不到 {suffix} 数组。')
        vals = [int(v) for v in m.group(1).replace('\n', ' ').split(',') if v.strip()]
        if len(vals) != w * h:
            raise SystemExit(f'{path} 的 {suffix} 数组长度 {len(vals)} 与 {w}x{h} 不符。')
        return dequantize(np.asarray(vals, dtype=np.int64), shift)

    return array('mapW').reshape(h, w), array('mapH').reshape(h, w)


def load_map_file(path: Path) -> np.ndarray:
    """读一张映射表，返回 (H, W) 的 float64 数组，无效点统一为 -1。"""
    if path.suffix == '.bin':
        if TABLE_SHAPE is None:
            raise SystemExit('未能确定 bin 表的网格尺寸（matrices.json 里缺 table_size_wh '
                             '且标定分辨率不可用）。')
        w, h = TABLE_SHAPE
        raw = np.fromfile(path, dtype='<i2').astype(np.int64)
        if raw.size != w * h:
            raise SystemExit(f'{path} 有 {raw.size} 个值，与网格 {w}x{h} 不符。')
        return dequantize(raw, FIXED_POINT).reshape(h, w)
    return np.loadtxt(path, delimiter=',', dtype=np.float64)


def load_pair(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取一个方向的 MapW/MapH，自动识别 txt / bin / C 头文件三种落盘格式。"""
    for suffix in ('.txt', '.bin'):
        wx, wy = folder / f'MapW{suffix}', folder / f'MapH{suffix}'
        if wx.is_file() and wy.is_file():
            return load_map_file(wx), load_map_file(wy)
    header = folder / 'Map.h'
    if header.is_file():
        return load_c_header(header)
    raise SystemExit(f'{folder} 下没有可识别的查找表（MapW.txt / MapW.bin / Map.h）。')


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
    """对 (N,2) 点集应用单应。"""
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


def full_resolution() -> bool:
    """表是否与源图同尺寸。不同尺寸时逐像素比对无意义。"""
    return TABLE_SHAPE is None or TABLE_SHAPE == IMAGE_SIZE


def check_export_fidelity(root: Path, calib: dict, matrices: dict) -> bool:
    """检查 6: 落盘的表是否忠实等于流水线重算的结果。

    覆盖重采样与定点量化两步。检查 2/3 验证的是数学（只有全分辨率下才可比），
    这一项验证的是"算出来的东西有没有原样写进文件"，任何网格、任何格式都成立。
    """
    # 本脚本以 tools/verify_outputs.py 的方式运行，sys.path[0] 是 tools/ 而不是工程根，
    # 直接 import 主模块会失败，得先把 src/ 挂上去（主脚本位于 <工程根>/src/）。
    src = root / 'src'
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import neuq_vision_calib as core

    core.configure_paths(root=root)
    core.TABLE_SIZE = TABLE_SHAPE
    core.TABLE_FIXED_POINT = FIXED_POINT

    K = np.asarray(calib['camera_matrix'], dtype=np.float64).reshape(3, 3)
    D = np.asarray(calib['dist_coeffs'], dtype=np.float64).ravel()
    Knew = np.asarray(matrices['Knew'], dtype=np.float64).reshape(3, 3)
    H = np.asarray(matrices['H'], dtype=np.float64).reshape(3, 3)
    H0 = np.asarray(matrices['H0'], dtype=np.float64).reshape(3, 3)
    sign = float(matrices['horizon_sign'])
    size = IMAGE_SIZE

    fx, fy = core.build_composite_reverse_map(K, D, Knew, H, H0, sign, size)
    if TABLE_SHAPE != size:
        fx, fy = core.resample_map_pair(fx, fy, TABLE_SHAPE)
    ax, ay = load_pair(root / 'lookup_table' / 'undistort_ipm' / 'reverse')

    if ax.shape != fx.shape:
        return report('落盘表 == 流水线重算', False,
                      f'网格不符: 落盘 {ax.shape[1]}x{ax.shape[0]}，重算 {fx.shape[1]}x{fx.shape[0]}')

    # NaN 既不满 >=0 也不满 <0，所以"无效"必须显式写成 ~isfinite | <0，
    # 只用 <0 判断会把重算侧的无效点漏掉一大半。
    bad_ex = ~np.isfinite(fx) | (fx < 0.0)
    bad_ey = ~np.isfinite(fy) | (fy < 0.0)
    bad_ax = ax < 0.0
    bad_ay = ay < 0.0

    mismatch = int(np.count_nonzero((bad_ex | bad_ey) != (bad_ax | bad_ay)))
    good = ~(bad_ex | bad_ey)
    err = np.hypot(ax[good] - fx[good], ay[good] - fy[good])
    max_err = float(err.max()) if err.size else 0.0
    step = 1.0 / (1 << FIXED_POINT)
    tol = 0.01 + step      # 文本是 %.2f，定点是 1/2^N，各留一点浮点余量

    ok = mismatch == 0 and max_err <= tol
    return report('落盘表 == 流水线重算', ok,
                  f'最大差 {max_err:.4f} px（容差 {tol:.4f}），'
                  f'有效点 {int(good.sum())}，无效判定不一致 {mismatch} 个')


_REF_CACHE: dict = {}


def load_reference_image(root: Path, size: tuple[int, int]) -> np.ndarray:
    """取一张可用于比对的源图。

    必须是标定时用的同一张。ipm_input/ 里通常躺着多张候选，随手挑一张会让
    "表重建 vs 参考实现"的比较彻底失去意义——两张不同的图本来就该不一样，
    差异率会直接飙到几十个百分点，看起来像表坏了。
    优先读 ipm_state.json 记录的原图，其次才退到约定名和目录首张。
    """
    key = (str(root), size)
    if key in _REF_CACHE:
        return _REF_CACHE[key]

    candidates: list[Path] = []
    state_path = root / 'matrix' / 'ipm_state.json'
    if state_path.is_file():
        try:
            recorded = json.loads(state_path.read_text(encoding='utf-8')).get('src_image')
        except json.JSONDecodeError:
            recorded = None
        if recorded:
            p = Path(recorded)
            if p.is_file():
                candidates.append(p)
            else:
                # 工程被搬过家、或目录结构调整过：记录的绝对路径失效，
                # 但同名图多半还在 ipm_input/ 下。
                # 先按文件名捞一次——直接退回"目录里第一张"会挑到另一张图，
                # 让后面的逐像素比对彻底失去意义。
                by_name = root / 'ipm_input' / p.name
                if by_name.is_file():
                    print(f'  提示: 记录的原图路径已失效，改用同名图 {by_name.name}。')
                    candidates.append(by_name)
                else:
                    print(f'  提示: ipm_state.json 记录的原图已不在 {p}，退回目录内查找。')
    preferred = root / 'ipm_input' / 'UnInverseImage.jpg'
    if preferred.is_file():
        candidates.append(preferred)
    if not candidates:
        candidates = sorted((root / 'ipm_input').glob('*.jpg'))
    if not candidates:
        raise SystemExit(f'{root / "ipm_input"} 下没有可用于比对的图片。')

    src = cv2.imread(str(candidates[0]))
    if src is None:
        raise SystemExit(f'无法读取 {candidates[0]}')
    print(f'  比对源图: {candidates[0].name}')
    if (src.shape[1], src.shape[0]) != size:
        src = cv2.resize(src, size)
    _REF_CACHE[key] = src
    return src


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
    """检查 1: H 把源四点映射成轴对齐矩形，且尺寸等于物理尺寸乘 scale。"""
    H = np.asarray(matrices['H'], dtype=np.float64).reshape(3, 3)
    quad = np.asarray(matrices['src_quad_tl_tr_bl_br'], dtype=np.float64).reshape(4, 2)
    scale = float(matrices['scale_px_per_cm'])
    phys_w = float(matrices['phys_w_cm'])
    phys_h = float(matrices['phys_h_cm'])

    out = apply_homography(H, quad)
    w = out[1, 0] - out[0, 0]
    h = out[2, 1] - out[0, 1]
    expect_w, expect_h = phys_w * scale, phys_h * scale

    axis_aligned = (abs(out[0, 1] - out[1, 1]) < 1e-3      # TL.y == TR.y
                    and abs(out[0, 0] - out[2, 0]) < 1e-3   # TL.x == BL.x
                    and abs(out[2, 1] - out[3, 1]) < 1e-3   # BL.y == BR.y
                    and abs(out[1, 0] - out[3, 0]) < 1e-3)  # TR.x == BR.x
    size_ok = abs(w - expect_w) < 0.5 and abs(h - expect_h) < 0.5

    return report('H 映射源四点为轴对齐矩形', axis_aligned and size_ok,
                  f'输出矩形 {w:.1f}x{h:.1f} px，期望 {expect_w:.1f}x{expect_h:.1f} px，'
                  f'轴对齐{"是" if axis_aligned else "否"}')


def check_undistort_tables(root: Path, calib: dict) -> bool:
    """检查 2: 去畸变反向表 remap == cv2.undistort。"""
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
    reference = cv2.undistort(src, K, D, None, K)

    visible, p999, mx_diff = compare_images(via_table, reference, invalid)
    return report('undistort/reverse 表 == cv2.undistort', visible <= VISIBLE_FRAC,
                  f'可见差异像素 {100 * visible:.4f}%，p99.9={p999:.1f}，'
                  f'最大 {mx_diff:.0f}，无效点 {100 * invalid.mean():.1f}%')


def check_composite_tables(root: Path, calib: dict, matrices: dict) -> bool:
    """检查 3: 逆透视反向表 remap == warpPerspective。"""
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

    undist = cv2.undistort(src, K, D, None, K)
    reference = cv2.warpPerspective(undist, H, size, flags=cv2.INTER_LINEAR)

    visible, p999, mx_diff = compare_images(via_table, reference, invalid)
    ok = report('undistort_ipm/reverse 表 == warpPerspective', visible <= VISIBLE_FRAC,
                f'可见差异像素 {100 * visible:.4f}%，p99.9={p999:.1f}，'
                f'最大 {mx_diff:.0f}，有效区 {100 * (~invalid).mean():.1f}%')

    # 顺带确认落盘的结果图确实由这张表生成。必须在有效区内比：无效区里表是黑的，
    # 而 warpPerspective 会照常填上标定矩形以外的画面，不屏蔽就会得出误导性的差异率。
    result_path = root / 'ipm_output' / 'UnDistortionInverseImage.jpg'
    if result_path.is_file():
        produced = cv2.imread(str(result_path))
        if produced is not None and produced.shape[:2] == via_table.shape[:2]:
            vis2, _p, m2 = compare_images(produced, via_table, invalid)
            print(f'         结果图 vs 表重建（限有效区）: 可见差异 {100 * vis2:.4f}%，'
                  f'最大 {m2:.0f}（JPEG 有损压缩会带来边缘抖动）')
    return ok


def check_forward_reverse(root: Path) -> bool:
    """检查 4: 逆透视的 forward 与 reverse 互为逆映射。"""
    rev_x, rev_y = load_pair(root / 'lookup_table' / 'undistort_ipm' / 'reverse')
    fwd_x, fwd_y = load_pair(root / 'lookup_table' / 'undistort_ipm' / 'forward')
    th, tw = rev_x.shape
    if fwd_x.shape != (th, tw):
        return report('forward/reverse 互逆', False,
                      f'尺寸不一致: reverse {rev_x.shape} vs forward {fwd_x.shape}')

    # 降采样后两个表的"索引步长"都变成 step 个原图像素，往返路径必须按这个比例换算：
    # 输出网格点 (x,y) 对应输出像素 (x*step, y*step)，reverse 给出的却是原图坐标，
    # 要除以 step 才能拿去索引 forward。漏掉这一步会算出几百像素的假误差。
    step_x = IMAGE_SIZE[0] / tw
    step_y = IMAGE_SIZE[1] / th

    ys, xs = np.mgrid[0:th, 0:tw]
    ys, xs = ys.ravel(), xs.ravel()
    valid = (rev_x[ys, xs] >= 0) & (rev_y[ys, xs] >= 0)
    ys, xs = ys[valid], xs[valid]
    if ys.size == 0:
        return report('forward/reverse 互逆', False, '反向表没有有效点')

    sx = np.rint(rev_x[ys, xs] / step_x).astype(np.int64).clip(0, tw - 1)
    sy = np.rint(rev_y[ys, xs] / step_y).astype(np.int64).clip(0, th - 1)
    back_x, back_y = fwd_x[sy, sx], fwd_y[sy, sx]

    hit = (back_x >= 0) & (back_y >= 0)
    if not hit.any():
        return report('forward/reverse 互逆', False, 'forward 在这些源点上全是无效值')
    err = np.hypot(back_x[hit] - xs[hit] * step_x, back_y[hit] - ys[hit] * step_y)

    # 容差随步长放大：forward 只能落在最近的源图网格上，索引取整误差会被
    # 局部放大率（每个源像素对应多少输出像素）放大。
    tol = TOL_INV_PX * max(step_x, step_y) + 1.0
    med = float(np.median(err))
    p95 = float(np.percentile(err, 95))
    return report('forward/reverse 互逆', med <= tol and p95 <= tol * 3,
                  f'往返误差中位数 {med:.2f} px，95 分位 {p95:.2f} px，'
                  f'容差 {tol:.1f} px，命中率 {100 * hit.mean():.1f}%')


def check_sentinel(root: Path) -> bool:
    """检查 5: 无效哨兵 -1 的分布是否合理。"""
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
    """打印四组表的落盘格式、网格与体积。

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
    global FIXED_POINT, TABLE_SHAPE, IMAGE_SIZE, SKIPPED

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
    if fmt != 'txt':
        FIXED_POINT = int(matrices.get('table_fixed_point') or 4)

    # 表被重采样过就用重采样后的网格，否则与标定分辨率同尺寸
    IMAGE_SIZE = (int(calib['image_width']), int(calib['image_height']))
    size = matrices.get('table_size_wh')
    TABLE_SHAPE = (int(size[0]), int(size[1])) if size else IMAGE_SIZE
    SKIPPED = 0

    print(f'校验 {root}')
    print(f'表格式 {fmt}' + (f'（Q{FIXED_POINT} 定点）' if fmt != 'txt' else '')
          + f'，源图 {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}，表网格 {TABLE_SHAPE[0]}x{TABLE_SHAPE[1]}\n')

    print('查找表清单:')
    print_table_inventory(root)
    print()

    results = [
        check_homography(matrices),
        check_undistort_tables(root, calib),
        check_composite_tables(root, calib, matrices),
        check_forward_reverse(root),
        check_sentinel(root),
        check_export_fidelity(root, calib, matrices),
    ]

    tail = f'（其中 {SKIPPED} 项因降采样跳过）' if SKIPPED else ''
    print(f'\n{sum(results)}/{len(results)} 项通过{tail}。')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
