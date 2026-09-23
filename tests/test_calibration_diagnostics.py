"""采集质量诊断启发式的纯函数回归测试。

    python tests/test_calibration_diagnostics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from neuq_core.calibration import diagnose_calibration_views  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    print(('  [OK] ' if cond else '  [!!] ') + label)
    if not cond:
        FAILED.append(label)


def test_per_view_heatmap_weighting_and_repetition():
    # 每张图有很多落在同一格的点，但该图对此格最多贡献 1 次。
    image_points = [np.repeat([[10.0, 10.0]], 100, axis=0).astype(np.float32)
                    for _ in range(5)]
    object_points = [np.array([[0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0]],
                              dtype=np.float32) for _ in image_points]
    rvecs = [np.zeros((3, 1), dtype=np.float64) for _ in image_points]
    tvecs = [np.array([[0.0], [0.0], [500.0]]) for _ in image_points]
    diagnostics = diagnose_calibration_views(
        image_points, object_points, rvecs, tvecs,
        np.array([[800.0, 0, 50], [0, 800.0, 40], [0, 0, 1]]),
        np.array([1.0, 1.0, 1.0, 1.0]), (100, 80),
        [f'view_{i}.jpg' for i in range(5)], [0.5] * 5)

    heatmap = diagnostics['heatmap']
    check(heatmap['max_count'] == 5,
          '单张照片在同一网格的许多角点只记作一次，重复度按照片数累计')
    check(heatmap['observed_once_ratio'] < 0.1
          and diagnostics['metrics'][2]['value'] == '较多',
          '热力图区分未覆盖区域，并识别重复拍摄的棋盘中心位置')
    check(any('重复出现较多' in text for text in diagnostics['recommendations']),
          '重复位置诊断给出继续拍相似位置帮助有限的行动建议')
    check(len(diagnostics['std_intrinsics']) == 4
          and 'fx/fy/cx/cy' in diagnostics['std_intrinsics_note']
          and '不包含畸变参数' in diagnostics['std_intrinsics_note'],
          '内参标准差明确限于 fx/fy/cx/cy，不暗示畸变参数也稳定')
    check('score' not in diagnostics and '合格标准' in diagnostics['heuristics_note'],
          '诊断不生成伪精确总分，并明确不是合格判据')
    check('priority' not in diagnostics
          and all('：' in text for text in diagnostics['recommendations']),
          '建议逐项标注主题，不生成固定顺序的“最需要改善项”')
    try:
        json.dumps(diagnostics, ensure_ascii=False)
        serializable = True
    except (TypeError, ValueError):
        serializable = False
    check(serializable, '诊断结果可以作为非持久化 API 字段序列化')


def test_pose_scale_and_edge_diagnostics():
    width, height = 1000, 800
    centers = [(0.15, 0.15), (0.15, 0.15), (0.85, 0.15), (0.85, 0.15),
               (0.15, 0.85), (0.15, 0.85), (0.85, 0.85), (0.85, 0.85)]
    widths = [0.15, 0.28, 0.15, 0.28, 0.15, 0.28, 0.15, 0.28]
    image_points = []
    object_points = []
    for (cx, cy), span in zip(centers, widths, strict=True):
        xs = np.linspace((cx - span / 2) * (width - 1),
                         (cx + span / 2) * (width - 1), 8)
        ys = np.linspace((cy - span / 2) * (height - 1),
                         (cy + span / 2) * (height - 1), 6)
        image_points.append(np.array([[x, y] for y in ys for x in xs], dtype=np.float32))
        object_points.append(np.zeros((48, 3), dtype=np.float32))
    angles = [0.0, 0.35, -0.35, 0.5, -0.5, 0.8, 0.25, -0.25]
    rvecs = [np.array([[angle], [0.0], [0.0]]) for angle in angles]
    tvecs = [np.array([[0.0], [0.0], [distance]])
             for distance in (900, 700, 500, 350, 250, 600, 800, 450)]
    diagnostics = diagnose_calibration_views(
        image_points, object_points, rvecs, tvecs,
        np.array([[800.0, 0, 500], [0, 800.0, 400], [0, 0, 1]]),
        np.array([2.0, 2.0, 2.0, 2.0]), (width, height),
        [f'view_{i}.jpg' for i in range(len(image_points))], [0.5] * len(image_points))
    metrics = {item['key']: item for item in diagnostics['metrics']}

    check(metrics['pose_diversity']['value'] == '良好',
          '姿态变化由每张棋盘平面法线的倾角分布评估')
    check(metrics['scale_diversity']['value'] == '良好',
          '尺度变化由每张棋盘角点凸包的投影面积评估')
    check(metrics['edge_coverage']['value'] == '一般'
          and len(diagnostics['corners']) == 4,
          f"边缘与四角分别累计跨照片的有效观测：{metrics['edge_coverage']['detail']}，"
          f"四角={diagnostics['corners']}")
    check(all(view['distance_cm'] is not None for view in diagnostics['views']),
          '详细视图数据同时包含由 rvec/tvec 估计的棋盘中心距离')


def test_distinct_positions_in_small_batch_are_not_repeated():
    width, height = 100, 80
    centers = [(0.125, 1 / 6), (0.375, 1 / 6), (0.625, 1 / 6),
               (0.875, 1 / 6), (0.125, 0.5)]
    image_points = [np.array([[x * (width - 1), y * (height - 1)]], dtype=np.float32)
                    for x, y in centers]
    object_points = [np.zeros((1, 3), dtype=np.float32) for _ in centers]
    rvecs = [np.zeros((3, 1), dtype=np.float64) for _ in centers]
    tvecs = [np.array([[0.0], [0.0], [500.0]]) for _ in centers]
    diagnostics = diagnose_calibration_views(
        image_points, object_points, rvecs, tvecs,
        np.array([[800.0, 0, 50], [0, 800.0, 40], [0, 0, 1]]),
        np.array([1.0, 1.0, 1.0, 1.0]), (width, height),
        [f'view_{i}.jpg' for i in range(len(centers))], [0.5] * len(centers))
    repetition = next(item for item in diagnostics['metrics']
                      if item['key'] == 'position_repetition')

    check(repetition['value'] == '较少',
          '5 张照片各落在不同的位置网格时不提示位置重复')


if __name__ == '__main__':
    print('逐图覆盖与位置重复')
    test_per_view_heatmap_weighting_and_repetition()
    print('姿态、尺度与边缘覆盖')
    test_pose_scale_and_edge_diagnostics()
    print('小批次的无重复位置')
    test_distinct_positions_in_small_batch_are_not_repeated()
    sys.exit(1 if FAILED else 0)
