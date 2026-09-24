"""LOOCV 自动筛选建议的隔离、选择规则与取消行为测试。"""
from __future__ import annotations

import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import neuq_vision_calib as core  # noqa: E402
from neuq_core.calibration import (  # noqa: E402
    calibration_filter_geometry_regressions,
    select_calibration_filter_candidate,
)

FAILED: list[str] = []


def check(cond, label):
    print(('  [OK] ' if cond else '  [!!] ') + label)
    if not cond:
        FAILED.append(label)


def _diagnostics(levels=None):
    levels = levels or {}
    keys = ('image_coverage', 'edge_coverage', 'pose_diversity', 'scale_diversity')
    return {'metrics': [
        {'key': key, 'label': key, 'value': '良好',
         'level': levels.get(key, 'good'), 'detail': ''}
        for key in keys
    ]}


def test_one_standard_error_prefers_more_views():
    candidates = [
        {'threshold': None, 'retained_count': 10, 'geometry_safe': True,
         'fold_errors': [0.99] * 5},
        {'threshold': 2.0, 'retained_count': 8, 'geometry_safe': True,
         'fold_errors': [0.80, 0.90, 0.90, 0.90, 0.90]},
        {'threshold': 1.5, 'retained_count': 6, 'geometry_safe': True,
         'fold_errors': [0.80, 0.90, 0.90, 0.90, 0.90]},
        {'threshold': 1.0, 'retained_count': 4, 'geometry_safe': False,
         'fold_errors': [0.1] * 5},
    ]
    chosen = select_calibration_filter_candidate(candidates, 5)
    check(chosen['near_best'][0]['retained_count'] == 8,
          '一标准误范围内优先保留更多照片，且排除几何退化候选')

    keep_all = select_calibration_filter_candidate([
        {'threshold': None, 'retained_count': 5, 'geometry_safe': True,
         'fold_errors': [0.9] * 5},
        {'threshold': 1.0, 'retained_count': 3, 'geometry_safe': True,
         'fold_errors': [0.89, 0.91, 0.9, 0.9, 0.9]},
    ], 5)
    check(keep_all['near_best'][0]['threshold'] is None,
          '不剔除基线处于一标准误范围内时优先不剔除')


def test_geometry_gate():
    baseline = _diagnostics()
    candidate = _diagnostics({'edge_coverage': 'fair', 'scale_diversity': 'weak'})
    regressions = calibration_filter_geometry_regressions(baseline, candidate)
    check(regressions == ['edge_coverage', 'scale_diversity'],
          '画面覆盖、边缘、姿态、尺度按现有诊断等级阻止退化')


def test_loocv_replays_each_candidate_without_heldout_view():
    count = 5
    object_points = [np.array([[[i, 0, 0]], [[i, 1, 0]], [[i, 0, 1]]], dtype=np.float32)
                     for i in range(count)]
    image_points = [np.zeros((3, 1, 2), dtype=np.float32) for _ in range(count)]
    used = [Path(f'view_{i}.jpg') for i in range(count)]
    active_fold = None
    fold_training_calls = defaultdict(list)
    fold_starts = set()
    fold_thresholds = {}
    full_data_calls = []
    formal_replay_active = False
    force_formal_replay_failure = False

    def fake_calibrate(train_obj, _train_img, _img_size):
        ids = [int(np.asarray(points).reshape(-1, 3)[0, 0]) for points in train_obj]
        if active_fold is None:
            full_data_calls.append(ids)
        else:
            fold_training_calls[active_fold].append(ids)
        errors = (np.ones(len(ids)) if formal_replay_active and force_formal_replay_failure
                  else np.asarray([4.0 if i == 4 else 3.0 if i == 3 else 1.0
                                  for i in ids]))
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 0] = 60.0 if 4 in ids else 50.0
        rotations = [np.zeros((3, 1), dtype=np.float64) for _ in ids]
        translations = [np.zeros((3, 1), dtype=np.float64) for _ in ids]
        return 0.5, matrix, np.zeros(5), rotations, translations, np.array([]), errors

    def on_progress(item):
        nonlocal active_fold, formal_replay_active
        if item['stage'] == 'fold_paths':
            active_fold = None
            formal_replay_active = False
        elif item['stage'] == 'cross_validation' and item.get('fit_round') == 0:
            active_fold = (item['candidate_index'], item['fold_index'])
            fold_starts.add(active_fold)
            fold_thresholds[active_fold] = item.get('threshold')
        elif item['stage'] == 'full_path':
            active_fold = None
            formal_replay_active = False
        elif item['stage'] == 'formal_replay':
            active_fold = None
            formal_replay_active = True

    def track_progress(item):
        progress.append(item)
        on_progress(item)

    def fake_solve_pnp(_obj, _img, matrix, _dist, flags):
        assert flags == core.cv2.SOLVEPNP_ITERATIVE
        error = 2.0 if matrix[0, 0] > 55 else 1.0
        return True, np.array([[error], [0.0], [0.0]]), np.zeros((3, 1))

    def fake_project_points(obj, rvec, _tvec, _matrix, _dist):
        return np.tile(np.array([[[rvec[0, 0], 0.0]]]), (len(obj), 1, 1)), None

    with (patch.object(core, '_calibrate_once', side_effect=fake_calibrate),
          patch.object(core, 'diagnose_calibration_views', side_effect=lambda *a: _diagnostics()),
          patch.object(core.cv2, 'solvePnP', side_effect=fake_solve_pnp),
          patch.object(core.cv2, 'projectPoints', side_effect=fake_project_points)):
        progress = []
        result = core.assess_calibration_filter(
            object_points, image_points, used, (640, 480),
            progress_callback=track_progress)

    expected_fold_count = result['candidate_count'] * count
    check(len(fold_starts) == expected_fold_count,
          '每个候选阈值都在每张留出视图上启动独立训练折')
    excluded = all(all(key[1] not in ids for ids in calls)
                   for key, calls in fold_training_calls.items())
    check(excluded and len(fold_training_calls) == expected_fold_count,
          '留出视图从未进入该折标定或逐张删帧决策')
    one_at_a_time = True
    for calls in fold_training_calls.values():
        counts = [len(ids) for ids in calls]
        one_at_a_time &= all(a - b == 1 for a, b in pairwise(counts))
    check(one_at_a_time, '训练折每轮至多剔除一张，并在下一轮重新完整拟合')
    check(result['status'] == 'recommended' and result['retained_count'] == 4
          and result['dropped_names'] == ['view_4.jpg']
          and abs(result['recommended_threshold'] - 3.5) < 1e-9,
          'LOOCV 改善且正式全数据路径复核成功后才返回阈值')
    check(result['selected_cv_rms'] < result['baseline_cv_rms']
          and any(item['stage'] == 'cross_validation' for item in progress),
          '比较留出 RMS 而非训练 RMS，并报告交叉验证进度')
    filtered_fold_thresholds = [value for (candidate, _fold), value in fold_thresholds.items()
                                if candidate == 2]
    check(len(filtered_fold_thresholds) == count
          and len(set(filtered_fold_thresholds)) > 1,
          '每折阈值只由该折训练照片生成，不从留出照片的误差派生')

    fold_training_calls.clear()
    fold_starts.clear()
    fold_thresholds.clear()
    full_data_calls.clear()
    progress.clear()
    force_formal_replay_failure = True
    with (patch.object(core, '_calibrate_once', side_effect=fake_calibrate),
          patch.object(core, 'diagnose_calibration_views', side_effect=lambda *a: _diagnostics()),
          patch.object(core.cv2, 'solvePnP', side_effect=fake_solve_pnp),
          patch.object(core.cv2, 'projectPoints', side_effect=fake_project_points)):
        try:
            core.assess_calibration_filter(
                object_points, image_points, used, (640, 480),
                progress_callback=track_progress)
        except ValueError as exc:
            replay_rejected = '未通过完整数据正式筛选复核' in str(exc)
        else:
            replay_rejected = False
    check(replay_rejected, '完整数据正式筛选无法复现候选集合时不返回建议阈值')


def test_cancellation():
    event = Event()
    event.set()
    points = [np.zeros((1, 1, 3), dtype=np.float32) for _ in range(4)]
    images = [np.zeros((1, 1, 2), dtype=np.float32) for _ in range(4)]
    names = [Path(f'view_{i}.jpg') for i in range(4)]
    with patch.object(core, '_calibrate_once') as calibrate:
        try:
            core.assess_calibration_filter(points, images, names, (640, 480),
                                           cancel_event=event)
        except core.CalibrationCancelled:
            cancelled = True
        else:
            cancelled = False
    check(cancelled and not calibrate.called,
          '取消信号在新的 OpenCV 标定开始前立即生效')


def test_webui_progress_and_cancel_bypass_compute_lock():
    import webui.server as server

    def cancelled_collection(*_args, cancel_event=None, **_kwargs):
        if cancel_event is not None and cancel_event.is_set():
            raise server.core.CalibrationCancelled('测试取消。')
        raise AssertionError('该任务应在检测前被取消。')

    with patch.object(server.core, 'collect_calibration_views',
                      side_effect=cancelled_collection):
        server.LOCK.acquire()
        try:
            started = server.api_calibration_selection_start({})
            status_while_locked = server.calibration_selection_status()
            cancelled = server.api_calibration_selection_cancel({})
        finally:
            server.LOCK.release()

        deadline = monotonic() + 2.0
        while monotonic() < deadline:
            final = server.calibration_selection_status()
            if final['status'] == 'cancelled':
                break
            sleep(0.01)
        server.invalidate_calibration_selection()

    check(started['status'] == 'running'
          and status_while_locked['status'] == 'running',
          '后台评估启动后可在计算锁占用期间读取进度状态')
    check(cancelled['status'] == 'cancelling' and final['status'] == 'cancelled',
          '取消接口不等待计算锁，并在任务安全检查点停止后台评估')


if __name__ == '__main__':
    print('LOOCV 筛选选择规则')
    test_one_standard_error_prefers_more_views()
    test_geometry_gate()
    print('留一视图训练隔离与逐张重放')
    test_loocv_replays_each_candidate_without_heldout_view()
    print('进度与取消')
    test_cancellation()
    test_webui_progress_and_cancel_bypass_compute_lock()
    sys.exit(1 if FAILED else 0)
