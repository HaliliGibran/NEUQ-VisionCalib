"""相机标定首轮高误差建议与逐视图迭代筛选回归测试。

    python tests/test_calibration_threshold.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import neuq_vision_calib as core  # noqa: E402
from neuq_core.calibration import recommend_reprojection_threshold  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    print(('  [OK] ' if cond else '  [!!] ') + label)
    if not cond:
        FAILED.append(label)


def test_threshold_recommendation():
    normal = recommend_reprojection_threshold([1.0, 1.05, 1.1, 1.15, 1.2])
    check(normal['threshold'] is None and normal['status'] == 'no_outliers',
          '正常分布不产生筛选阈值')

    single = recommend_reprojection_threshold([1.0, 1.05, 1.1, 1.15, 8.0])
    check(single['status'] == 'recommended' and single['outlier_count'] == 1
          and 1.15 < single['threshold'] < 8.0,
          '单个离群视图的建议值严格位于两组之间')

    multiple = recommend_reprojection_threshold(
        [1.0, 1.05, 1.1, 1.15, 1.2, 7.0, 8.0])
    check(multiple['status'] == 'recommended' and multiple['outlier_count'] == 2
          and 1.2 < multiple['threshold'] < 7.0,
          '多个离群视图被识别，建议值位于正常组与异常组之间')

    degenerate = recommend_reprojection_threshold([1.0, 1.0, 1.0, 1.0, 8.0])
    check(degenerate['threshold'] is None and degenerate['status'] == 'mad_degenerate',
          'MAD 为零时不强行建议剔除值')

    too_few = recommend_reprojection_threshold([1.0, 1.05, 1.1, 8.0])
    check(too_few['threshold'] is None and too_few['status'] == 'too_few_samples',
          '样本少于 5 张时不产生推荐值')


def fake_calibration(error_sets, calls):
    def calibrate(obj_points, _img_points, _img_size):
        errors = error_sets[len(calls)]
        calls.append(len(obj_points))
        count = len(obj_points)
        return (0.5, np.eye(3), np.zeros(5), [None] * count, [None] * count,
                np.array([]), np.asarray(errors, dtype=np.float64))
    return calibrate


def test_iterative_single_view_removal():
    calls = []
    error_sets = ([1.0, 9.0, 8.0, 1.0, 1.0], [1.0, 8.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    points = [np.zeros((1, 1, 3), dtype=np.float32) for _ in range(5)]
    images = [np.zeros((1, 1, 2), dtype=np.float32) for _ in range(5)]
    names = [Path(f'view_{i}.jpg') for i in range(5)]
    with (patch.object(core, 'MAX_REPROJ_ERR', 2.0),
          patch.object(core, '_calibrate_once', side_effect=fake_calibration(error_sets, calls))):
        result = core.fit_camera(points, images, names, (640, 480))

    check(calls == [5, 4, 3], '每次完整标定后只移除一张，再重新标定')
    check([name for name, _error in result[10]] == ['view_1.jpg', 'view_2.jpg'],
          '每轮移除当轮 RMS 最大的超限视图')
    check(result[12] is None and len(result[9]) == 3,
          '误差满足阈值时正常结束，最终保留 3 张')


def test_three_view_minimum():
    calls = []
    points = [np.zeros((1, 1, 3), dtype=np.float32) for _ in range(3)]
    images = [np.zeros((1, 1, 2), dtype=np.float32) for _ in range(3)]
    names = [Path(f'view_{i}.jpg') for i in range(3)]
    with (patch.object(core, 'MAX_REPROJ_ERR', 2.0),
          patch.object(core, '_calibrate_once', side_effect=fake_calibration(([9, 8, 7],), calls))):
        result = core.fit_camera(points, images, names, (640, 480))

    check(calls == [3] and not result[10], '达到 3 张数学下限后不再剔除')
    check(result[12] is not None and '仍有 3 张' in result[12],
          '下限阻止继续剔除时返回明确的超限原因')


if __name__ == '__main__':
    print('推荐阈值统计')
    test_threshold_recommendation()
    print('逐视图迭代剔除')
    test_iterative_single_view_removal()
    test_three_view_minimum()
    sys.exit(1 if FAILED else 0)
