"""校验导出的矩阵与查找表是否自洽。

交付给嵌入式 C 端的是一堆映射表。表一旦内部不一致（方向搞反、无效点没标、精度不够），
在车上是极难排查的。这个脚本把每条约定都独立验一遍：

  1. H 把源四点映射成矩形，且尺寸/邻边垂直/旋转角/锚点四项不变量成立
  2. undistort/reverse 表 remap 的结果 == cv2.undistort
  3. undistort_ipm/reverse 表 remap 的结果 == cv2.warpPerspective(去畸变图, H)
  4. undistort_ipm 的 forward 与 reverse 互为逆映射
  5. 无效哨兵只出现在合法区域之外，且有效区连通

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

TOL_INV_PX = 3.0      # forward/reverse 互逆的容差（表按 %.2f 量化，留足余量）
VISIBLE_FRAC = 0.005  # 允许的"肉眼可见差异"像素占比（灰度差 > 8）

BIN_SENTINEL = core.BIN_SENTINEL

# 由 matrices.json 填入，bin/c 还原坐标时要用
FIXED_POINT = core.TABLE_FIXED_POINT

# bin 是裸数组，文件里没有维度信息，必须靠 matrices.json 与标定分辨率还原
TABLE_SHAPE: tuple[int, int] | None = None   # (W, H)
IMAGE_SIZE: tuple[int, int] = (0, 0)         # 源图尺寸，来自 calib.json
KNEW: np.ndarray | None = None               # 去畸变输出矩阵，来自 matrices.json
IS_TEXT_TABLE = True                         # 表格式是否为 %.2f 文本，影响量化容差
SKIPPED = 0                                  # 因降采样而跳过的检查条数


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
    # core 已在模块顶部导入（那时就把 src/ 挂上了 sys.path），这里只需按用户给的
    # 工程根重绑定路径常量，并把表格格式同步过去，好让 core.load_map_pair 读得对。
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
    tol = quant_tolerance()

    ok = mismatch == 0 and max_err <= tol
    return report('落盘表 == 流水线重算', ok,
                  f'最大差 {max_err:.4f} px（容差 {tol:.4f}），'
                  f'有效点 {int(good.sum())}，无效判定不一致 {mismatch} 个')


def quant_tolerance() -> float:
    """落盘量化的理论最大误差。

    单轴最多差半个量化步长（文本是 %.2f 即 0.005，定点是 0.5/2^N）；
    两轴合成后是二维欧氏距离，最大 sqrt(2) 倍。以前写成 `0.01 + step`，
    对文本表偏松约 7 倍、对 Q0 偏松到 1.01 px，几乎没有判别力。
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
    """检查 1: H 把源四点映射成矩形，且尺寸、旋转、锚点都与标定参数一致。

    H = T(anchor) @ S(scale) @ R(heading) @ H0，所以只要 heading != 0，输出矩形
    本来就是旋转的——用"轴对齐"当判据在非零 heading 下必然误判（本项目实测
    heading=-1.6°，旧判据一直报失败）。这里改为验证四个对任意旋转都成立的不变量：

      1. 中心 == anchor（归一化锚点换算到像素）
      2. |TL->TR| == 物理宽 × scale，|TL->BL| == 物理高 × scale
      3. 邻边垂直：(TL->TR) · (TL->BL) == 0
      4. atan2(TR-TL) == heading（矩形对边平行，角度在模 180° 意义下比较）
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
    cx, cy = out.mean(axis=0)
    anchor_ok = abs(cx - want_cx) < 0.5 and abs(cy - want_cy) < 0.5

    return report(
        'H 映射源四点成矩形（尺寸/垂直/旋转/锚点）',
        size_ok and ortho_ok and rot_ok and anchor_ok,
        f'尺寸 {w_px:.1f}x{h_px:.1f} 期望 {expect_w:.1f}x{expect_h:.1f} '
        f'{"OK" if size_ok else "不符"}；'
        f'邻边 |cos|={abs(cos):.2e} {"OK" if ortho_ok else "不垂直"}；'
        f'旋转 {rot_deg:.2f}° 期望 {heading:.2f}°（偏差 {delta:+.3f}°）'
        f'{"OK" if rot_ok else "不符"}；'
        f'中心 ({cx:.1f},{cy:.1f}) 期望 ({want_cx:.1f},{want_cy:.1f})'
        f'{"OK" if anchor_ok else "不符"}')


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
    reference = undistort_reference(src, K, D)

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
    global FIXED_POINT, TABLE_SHAPE, IMAGE_SIZE, SKIPPED, IS_TEXT_TABLE

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

    print(f'校验 {root}')
    print(f'表格式 {fmt}' + (f'（Q{FIXED_POINT} 定点）' if fmt != 'txt' else '')
          + f'，源图 {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}，表网格 {TABLE_SHAPE[0]}x{TABLE_SHAPE[1]}')
    print(f'去畸变输出矩阵 Knew: {"来自 matrices.json" if KNEW is not None else "缺省，回退用 K"}')

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
