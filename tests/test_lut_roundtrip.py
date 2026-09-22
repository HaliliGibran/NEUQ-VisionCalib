"""打表链路的参数化回归测试。

    python tests/test_lut_roundtrip.py

为什么不用 pytest：本工程唯一装了 opencv 的解释器往往是系统 Python，
为了跑测试往里装包不划算。这里用一个十行的迷你参数化框架，零依赖，
拿到任何有 cv2 的解释器上都能直接跑。

覆盖的是**参数组合**触发的缺陷——单看某一种配置永远测不出来：
  format   : txt / bin / c
  size     : 原尺寸 / 320x240 / 160x120
  Q 位数   : 0 / 4        （bin 与 c）
  heading  : 0° / -1.6°
  Knew     : 与 K 相同 / alpha=0.5 算出的不同矩阵

历史缺陷（本文件针对它们而写）：
  1. C 头文件把头文件的网格尺寸当成了源图分辨率，降采样后 batch_test
     会拿缩到 160x120 的图去采 800 多的坐标；
  2. bin 是裸二进制，重启后若没从 matrices.json 恢复 Q 位数，
     Q2 导出的表会被按默认 Q4 解码，坐标整体差 4 倍；
  3. 写文件与验证消费两份不同的 map（已由 MapPair 收敛）。
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import neuq_vision_calib as core  # noqa: E402

SRC_SIZE = (1280, 720)
CASES: list[tuple] = []
for fmt in ('txt', 'bin', 'c'):
    # 只能取整数倍等比网格。这里原来写的是 (320,240) 与 (160,120)，它们对 1280x720
    # 分别是 4x/3x 与 8x/6x —— 非等比，会把 16:9 的 BirdView 压成 4:3，物理上正方的
    # 标定块在小图里变成竖长矩形。T1 之后 prepare_map_pair 直接拒绝这类网格。
    for size in (None, (320, 180), (160, 90)):
        for fp in ((0, 4) if fmt != 'txt' else (None,)):
            for heading in (0.0, -1.6):
                for alpha in (None, 0.5):
                    CASES.append((fmt, size, fp, heading, alpha))


def synth_camera(size=SRC_SIZE):
    """合成一套相机参数：畸变明显一点，好让无效区真实存在。"""
    w, h = size
    K = np.array([[600.0, 0.0, w / 2 - 4], [0.0, 598.0, h / 2 + 3], [0.0, 0.0, 1.0]])
    D = np.array([0.11, -0.048, 0.0012, -0.0009, 0.0035])
    return K, D


def new_camera_matrix(K, D, alpha, size=SRC_SIZE):
    """算出生产代码真正会用的 Knew。

    resolve_new_camera_matrix() 的第三个参数是 img_size，而 alpha 读的是模块
    全局 UNDIST_ALPHA——所以必须真的把全局设过去，否则 alpha=None 与 alpha=0.5
    两组会拿到完全相同的 Knew，Knew 这一维等于没测。
    """
    old = core.UNDIST_ALPHA
    try:
        core.UNDIST_ALPHA = alpha
        return np.asarray(core.resolve_new_camera_matrix(K, D, size), dtype=np.float64)
    finally:
        core.UNDIST_ALPHA = old


def synth_ipm(K, D, Knew, heading, size=SRC_SIZE):
    """合成一组自洽的逆透视标定：H = T(anchor) @ S(scale) @ R(heading) @ H0。"""
    w, h = size
    quad = np.array([[w * 0.30, h * 0.72], [w * 0.70, h * 0.72],
                     [w * 0.08, h * 0.97], [w * 0.92, h * 0.97]], dtype=np.float64)
    phys_w = phys_h = 80.0
    H0 = core.compute_homography(quad, core.physical_rect(phys_w, phys_h))
    scale = 4.0
    anchor = (0.5 * (w - 1), 0.62 * (h - 1))
    H = (core.translation_matrix(*anchor) @ core.scale_matrix(scale)
         @ core.rotation_matrix(heading) @ H0)
    sign = core.horizon_sign(H0, quad)
    return H, H0, sign, quad, phys_w, phys_h, anchor


def check(cond, label, detail=''):
    if cond:
        print(f'    [OK] {label}' + (f'  {detail}' if detail else ''))
    else:
        print(f'    [!!] {label}  {detail}')
    return bool(cond)


def run_case(fmt, table_size, fp, heading, alpha, tmp: Path) -> bool:
    tag = (f'fmt={fmt} size={table_size} Q={fp} heading={heading} '
           f'alpha={alpha}')
    print(f'\n--- {tag}')
    core.TABLE_FORMAT = fmt
    core.TABLE_SIZE = table_size
    if fp is not None:
        core.TABLE_FIXED_POINT = fp

    K, D = synth_camera()
    Knew = new_camera_matrix(K, D, alpha)
    H, H0, sign, quad, phys_w, phys_h, anchor = synth_ipm(K, D, Knew, heading)

    ok = True
    ok &= check(np.array_equal(Knew, K) == (alpha is None),
                'Knew 维度真的生效（alpha 非 None 时 Knew != K）',
                f'alpha={alpha}')
    table_root = tmp / 'tables'
    pair = core.export_composite_tables(K, D, Knew, H, H0, sign, SRC_SIZE,
                                        table_root=table_root)
    expect_grid = table_size or SRC_SIZE

    ok &= check(pair.size == expect_grid, '返回网格 == 期望网格',
                f'{pair.size} vs {expect_grid}')
    ok &= check(pair.source_size == SRC_SIZE, '源图分辨率保持为标定分辨率',
                f'{pair.source_size}')

    # ---- 落盘后读回（模拟重启：只用 matrices.json 里能查到的元信息）
    meta = {'table_size_wh': list(table_size) if table_size else None,
            'table_fixed_point': fp}
    grid = tuple(meta['table_size_wh']) if meta['table_size_wh'] else None
    back = core.load_map_pair(table_root / 'undistort_ipm' / 'reverse', SRC_SIZE,
                              grid, meta['table_fixed_point'])

    ok &= check(back.size == pair.size, '读回网格一致',
                f'{back.size} vs {pair.size}')
    ok &= check(back.source_size == SRC_SIZE,
                '读回源图分辨率 == 标定分辨率（C 格式曾错写成网格尺寸）',
                f'{back.source_size}')
    ok &= check(np.array_equal(pair.x, back.x),
                '读回 x 与导出值逐值一致',
                f'max|d|={np.abs(pair.x - back.x).max():.3g}')
    ok &= check(np.array_equal(pair.invalid, back.invalid), '无效点掩码一致')

    if fmt != 'txt':
        step = 1.0 / (1 << fp)
        v = pair.x[pair.x >= 0]
        ok &= check(float(np.abs(v / step - np.rint(v / step)).max()) < 1e-9,
                    f'有效坐标落在 Q{fp} 网格上')

    # ---- batch_test：输出网格必须是最终表的网格，而不是源图尺寸
    test_in, test_out = tmp / 'in', tmp / 'out'
    test_in.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (SRC_SIZE[1], SRC_SIZE[0], 3), dtype=np.uint8)
    core.safe_imwrite(test_in / 'a.jpg', img)
    core.DIR_TEST_IN, core.DIR_TEST_OUT = test_in, test_out
    core.batch_test(pair)
    outs = sorted(test_out.glob('*.jpg'))
    ok &= check(len(outs) == 1, '批量测试产出 1 张')
    if outs:
        got = core.safe_imread(outs[0])
        ow, oh = pair.size
        ok &= check((got.shape[1], got.shape[0]) == (ow, oh),
                    '输出网格 == 最终表网格', f'{got.shape[1]}x{got.shape[0]} vs {ow}x{oh}')

    return ok


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix='neuq_lut_test_'))
    print(f'参数组合共 {len(CASES)} 组，工作目录 {tmp}')
    failed = []
    try:
        for args in CASES:
            case_dir = tmp / '_'.join(str(a) for a in args).replace('/', 'x')
            case_dir.mkdir(parents=True, exist_ok=True)
            try:
                if not run_case(*args, case_dir):
                    failed.append(args)
            except Exception:
                print('    [!!] 抛异常:')
                print('    ' + traceback.format_exc().replace('\n', '\n    '))
                failed.append(args)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        core.DIR_TEST_IN = core.SCRIPT_DIR / 'test_input'
        core.DIR_TEST_OUT = core.SCRIPT_DIR / 'test_output'

    print()
    if failed:
        print(f'{len(failed)}/{len(CASES)} 组失败:')
        for f in failed:
            print('   ', f)
        return 1
    print(f'全部 {len(CASES)} 组通过。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
