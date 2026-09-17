"""扫描一个目录下的图片，报告分辨率分布与棋盘格检出情况。

用途：在正式跑标定前先摸清素材底细，避免把时间浪费在注定失败的批次上。

用法:
    python tools/scan_dataset.py <图片目录> [--corners 11 8]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp')


def list_images(folder: Path):
    """按文件名排序列出目录下的图片。"""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def to_gray(img):
    """转灰度，已是单通道则原样返回。"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def detect(gray, corners, fast=False):
    """检出棋盘内角点，先 SB 后经典算法。返回 (是否检出, 角点或 None)。"""
    if hasattr(cv2, 'findChessboardCornersSB'):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if not fast:
            flags |= cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, pts = cv2.findChessboardCornersSB(gray, corners, flags)
        if found:
            return True, pts
    found, pts = cv2.findChessboardCorners(
        gray, corners,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return False, None
    pts = cv2.cornerSubPix(gray, pts, (11, 11), (-1, -1),
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    return True, pts


def main() -> int:
    """入口。"""
    ap = argparse.ArgumentParser(description='扫描图片目录的棋盘格检出情况')
    ap.add_argument('folder', type=Path, help='待扫描的图片目录')
    ap.add_argument('--corners', nargs=2, type=int, default=[11, 8],
                    metavar=('COLS', 'ROWS'), help='棋盘内角点数，默认 11 8')
    args = ap.parse_args()

    files = list_images(args.folder)
    if not files:
        print(f'{args.folder} 下没有图片。')
        return 1

    corners = (args.corners[0], args.corners[1])
    sizes = Counter()
    ok_list, fail_list = [], []

    print(f'扫描 {args.folder}（{len(files)} 张），棋盘内角点 {corners[0]}x{corners[1]}\n')
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            fail_list.append((path.name, '无法读取', None))
            continue
        gray = to_gray(img)
        sizes[(gray.shape[1], gray.shape[0])] += 1
        found, pts = detect(gray, corners)
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
