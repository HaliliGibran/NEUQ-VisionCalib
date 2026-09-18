"""标定板规格的回归测试。

    python tests/test_board_spec.py

为什么这个必须有测试：棋盘规格不只喂给相机标定，**素材导入时判断"这张是棋盘照
还是地面照"用的也是它**。规格填错，第一步分拣就已经错了，而错误会一路带到
逆透视候选里，最后表现为"莫名其妙混进来一张棋盘照"。

覆盖：
  A. 规格对象本身：换算、校验、局部子网格候选的推导
  B. 命令行解析：--board-squares / --board-corners 互斥
  C. 检测行为：同一张合成棋盘，规格对能检出、规格错检不出
  D. calib.json 往返：写入含 board 段，读取兼容老格式
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import neuq_vision_calib as core  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=''):
    print(('  [OK] ' if cond else '  [!!] ') + label + (f'  {detail}' if detail else ''))
    if not cond:
        FAILED.append(label)
    return bool(cond)


def synth_board_image(spec, margin=60, px=45):
    """按规格画一张棋盘照片（12x9 方格 → 11x8 内角点）。"""
    w = spec.squares_x * px + margin * 2
    h = spec.squares_y * px + margin * 2
    img = np.full((h, w), 255, np.uint8)
    for i in range(spec.squares_x):
        for j in range(spec.squares_y):
            if (i + j) % 2 == 0:
                x0, y0 = margin + i * px, margin + j * px
                img[y0:y0 + px, x0:x0 + px] = 20
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main() -> int:
    print('[A] 规格对象')
    d = core.DEFAULT_BOARD
    check((d.squares_x, d.squares_y, d.square_size_mm) == (12, 9, 20.0),
          '默认规格是 12x9 / 20 mm', f'{d.squares_x}x{d.squares_y}')
    check(d.corners == (11, 8), '换算出的内角点是 11x8', f'{d.corners}')
    check(d.partial_grids == ((9, 6), (7, 5), (6, 4), (5, 3)),
          '默认规格的局部子网格候选与改造前一致', f'{d.partial_grids}')
    small = core.CheckerboardSpec(8, 6, 25.0)
    check(all(g[0] < 7 and g[1] < 5 and g[0] >= 4 and g[1] >= 3
              for g in small.partial_grids),
          '小棋盘的候选随之缩小且不小于 4x3', f'{small.partial_grids}')
    check(small.partial_grids != d.partial_grids, '候选不再写死，随规格变化')
    for bad, why in (((4, 9, 20.0), '方格数太小'), ((12, 9, 0.0), '边长为 0'),
                     ((12, 9, -1.0), '边长为负')):
        try:
            core.CheckerboardSpec(*bad)
            check(False, f'拒绝非法规格（{why}）')
        except ValueError:
            check(True, f'拒绝非法规格（{why}）')
    check('内角点' in d.label, 'label 里明确写出内角点，避免误解', d.label)

    print('\n[B] 命令行解析')
    got = core.resolve_board([12, 9], None, 20.0)
    check(got.corners == (11, 8), '--board-squares 12 9 → 内角点 11x8')
    got2 = core.resolve_board(None, [11, 8], 20.0)
    check(got2 == got, '--board-corners 11 8 与之等价')
    try:
        core.resolve_board([12, 9], [11, 8], None)
        check(False, '同时给两种参数会报错')
    except SystemExit as exc:
        check('只能给一个' in str(exc), '同时给两种参数会报错', str(exc)[:36])

    print('\n[C] 检测行为（合成棋盘，规格必须影响结果）')
    spec = core.CheckerboardSpec(12, 9, 20.0)
    img = synth_board_image(spec)
    gray = core.to_gray(img)
    old = core.BOARD
    try:
        core.configure_board(spec)
        found, corners = core.detect_chessboard(gray)
        check(found, '规格正确时检出棋盘',
              f'角点 {None if corners is None else len(corners)}')
        wrong = core.CheckerboardSpec(8, 6, 20.0)
        found_w, _ = core.detect_chessboard(gray, board=wrong)
        check(not found_w, '规格错误时检不出（说明规格真的在起作用）',
              f'错规格 {wrong.squares_x}x{wrong.squares_y}')
        core.configure_board(wrong)
        found_w2, _ = core.detect_chessboard(gray)
        check(not found_w2, '全局规格被改错时同样检不出')
        core.configure_board(spec)
        found3, _ = core.detect_chessboard(gray)
        check(found3, '改回正确规格后又能检出')
    finally:
        core.configure_board(old)

    print('\n[D] calib.json 往返')
    tmp = Path(tempfile.mkdtemp(prefix='neuq_board_'))
    old_root = core.SCRIPT_DIR
    try:
        core.configure_paths(root=tmp)
        core.configure_board(core.CheckerboardSpec(9, 7, 25.0))
        (core.DIR_CALIB_DATA).mkdir(parents=True, exist_ok=True)
        K = np.eye(3)
        core.save_calibration(K, np.zeros(5), (1280, 720))
        raw = json.loads(core.CALIB_JSON.read_text(encoding='utf-8'))
        check(raw.get('schema_version') == core.SCHEMA_VERSION,
              'calib.json 带 schema_version', str(raw.get('schema_version')))
        check(raw.get('tool_version') == core.TOOL_VERSION, '带 tool_version')
        b = raw.get('board') or {}
        check((b.get('squares_x'), b.get('squares_y')) == (9, 7),
              'board 段记下方格数', f'{b.get("squares_x")}x{b.get("squares_y")}')
        check((b.get('corners_x'), b.get('corners_y')) == (8, 6),
              'board 段同时记下内角点数')
        check(raw.get('chessboard_corners') == [8, 6],
              '老字段 chessboard_corners 仍然保留（兼容外部脚本）')
        meta = core.calib_board_meta()
        check((meta or {}).get('squares_x') == 9, 'calib_board_meta 能读回规格',
              str(meta and meta.get('squares_x')))

        # 老格式：只有 chessboard_corners
        legacy = {k: v for k, v in raw.items()
                  if k not in ('board', 'schema_version', 'tool_version')}
        core.CALIB_JSON.write_text(json.dumps(legacy), encoding='utf-8')
        meta2 = core.calib_board_meta()
        check((meta2 or {}).get('squares_x') == 9,
              '只有内角点的老 calib.json 也能换算回方格数',
              str(meta2 and meta2.get('squares_x')))

        spec2 = core.CheckerboardSpec.from_dict({'corners': [11, 8],
                                                 'square_size_mm': 20})
        check((spec2.squares_x, spec2.squares_y) == (12, 9),
              'CheckerboardSpec.from_dict 兼容老格式')
    finally:
        core.configure_paths(root=old_root)
        core.configure_board(core.DEFAULT_BOARD)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print('失败项:')
        for f in FAILED:
            print('   ', f)
        return 1
    print('全部通过。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
