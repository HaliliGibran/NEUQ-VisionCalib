"""扫描一个目录下的图片，报告分辨率分布与棋盘格检出情况。

用途：在正式跑标定前先摸清素材底细，避免把时间浪费在注定失败的批次上。

用法:
    python tools/scan_dataset.py <图片目录> --board-squares 12 9

棋盘规格与主程序共用同一套参数与同一份检测实现（core.detect_chessboard），
不再各留一套常量——否则核心改了规格，这个工具会给出对不上的结论。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'src'))

import neuq_vision_calib as core  # noqa: E402


def main() -> int:
    """入口。"""
    ap = argparse.ArgumentParser(
        description='扫描图片目录的棋盘格检出情况',
        epilog='棋盘规格与主程序共用同一套参数：推荐用 --board-squares 填方格数。')
    ap.add_argument('folder', type=Path, help='待扫描的图片目录')
    ap.add_argument('--board-squares', nargs=2, type=int, metavar=('X', 'Y'),
                    help='棋盘方格数，例如 12 9（默认）')
    ap.add_argument('--board-corners', nargs=2, type=int, metavar=('COLS', 'ROWS'),
                    help='兼容用法：直接给内角点数，如 11 8。与 --board-squares 互斥')
    ap.add_argument('--square-size-mm', type=float, metavar='MM',
                    help='单格边长毫米，默认 20')
    args = ap.parse_args()

    # 与主程序共用规格模型与检测实现，避免"核心一套参数、工具又一套"
    try:
        board = core.resolve_board(args.board_squares, args.board_corners,
                                   args.square_size_mm)
    except (SystemExit, ValueError) as exc:
        print(exc)
        return 2

    files = core.list_images(args.folder)
    if not files:
        print(f'{args.folder} 下没有图片。')
        return 1

    corners = board.corners
    sizes = Counter()
    ok_list, fail_list = [], []

    print(f'扫描 {args.folder}（{len(files)} 张）')
    print(f'标定板 {board.label}\n')
    for path in files:
        img = core.safe_imread(path)
        if img is None:
            fail_list.append((path.name, '无法读取', None))
            continue
        gray = core.to_gray(img)
        sizes[(gray.shape[1], gray.shape[0])] += 1
        found, pts = core.detect_chessboard(gray, board=board)
        if found:
            spread = float(np.linalg.norm(pts[-1] - pts[0]))
            ok_list.append((path.name, (gray.shape[1], gray.shape[0]), spread))
        else:
            fail_list.append((path.name, '未检出', (gray.shape[1], gray.shape[0])))

    print('分辨率分布:')
    for (w, h), n in sizes.most_common():
        print(f'  {w}x{h}  {n} 张')

    print(f'\n检出成功 {len(ok_list)} 张:')
    for name, size, spread in ok_list:
        print(f'  {name:<40} {size[0]}x{size[1]}  对角跨度 {spread:.0f} px')

    print(f'\n未检出 {len(fail_list)} 张:')
    for name, why, size in fail_list:
        dim = f'{size[0]}x{size[1]}' if size else '-'
        print(f'  {name:<40} {why}  {dim}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
