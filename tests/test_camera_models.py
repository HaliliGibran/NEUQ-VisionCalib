"""End-to-end coverage for standard and fisheye camera-model routing."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import neuq_vision_calib as core  # noqa: E402
from neuq_core import camera_model  # noqa: E402

FAILED: list[str] = []


def check(condition, label, detail=''):
    print(('  [OK] ' if condition else '  [!!] ') + label
          + (f' — {detail}' if detail else ''))
    if not condition:
        FAILED.append(label)


def synthetic_views():
    size = (640, 480)
    K = np.array([[300.0, 0.0, 320.0],
                  [0.0, 305.0, 240.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([-0.05, 0.01, -0.002, 0.0005], dtype=np.float64)
    board = np.array([[x * 25.0, y * 25.0, 0.0]
                      for y in range(6) for x in range(9)], dtype=np.float64)
    poses = [
        (0.10, 0.20, 0.00, -70.0, -40.0, 700.0),
        (-0.15, 0.10, 0.05, 10.0, -25.0, 850.0),
        (0.20, -0.12, -0.10, -30.0, 20.0, 780.0),
        (-0.10, -0.22, 0.10, 40.0, 25.0, 920.0),
        (0.25, 0.15, 0.05, -50.0, 10.0, 1000.0),
        (-0.22, -0.10, -0.04, 50.0, -10.0, 730.0),
        (0.08, 0.26, 0.08, 5.0, 30.0, 880.0),
        (-0.25, 0.03, -0.08, -20.0, -15.0, 820.0),
    ]
    object_points = []
    image_points = []
    for ax, ay, az, tx, ty, tz in poses:
        rvec = np.array([[ax], [ay], [az]], dtype=np.float64)
        tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)
        projected, _jacobian = camera_model.project_points(
            board.reshape(-1, 1, 3), rvec, tvec, K, D, camera_model.FISHEYE)
        object_points.append(board.reshape(-1, 1, 3))
        image_points.append(projected)
    return size, K, D, object_points, image_points


def test_fisheye_calibration_and_geometry():
    size, expected_K, expected_D, objects, images = synthetic_views()
    names = [Path(f'view_{i}.jpg') for i in range(len(objects))]
    result = core.fit_camera(objects, images, names, size,
                             threshold=None, camera_model='fisheye', verbose=False)
    K, D, errors = result[1], result[2], result[5]
    check(D.size == 4 and len(errors) == len(objects),
          '鱼眼标定产出 4 个畸变系数与逐视图 RMS', f'D={D.size}, RMS={len(errors)}')
    check(np.max(np.abs(K - expected_K)) < 1e-3
          and np.max(np.abs(D - expected_D)) < 1e-3,
          'OpenCV 鱼眼标定能恢复合成相机参数', f'RMS={result[0]:.2e}')

    success, heldout_rvec, heldout_tvec = camera_model.solve_pnp(
        objects[0], images[0], K, D, camera_model.FISHEYE)
    projected, _jacobian = camera_model.project_points(
        objects[0], heldout_rvec, heldout_tvec, K, D, camera_model.FISHEYE)
    pose_error = np.sqrt(np.mean(np.sum(
        (projected.reshape(-1, 2) - images[0].reshape(-1, 2)) ** 2, axis=1)))
    check(success and pose_error < 1e-4,
          'LOOCV 留出视图可用鱼眼模型独立求姿态与验证', f'{pose_error:.2e} px')

    fish_Knew = camera_model.estimate_new_camera_matrix(
        expected_K, expected_D, size, 0.5, camera_model.FISHEYE)
    raw_points = images[0].reshape(-1, 1, 2)
    ideal = camera_model.undistort_points(
        raw_points, K, D, fish_Knew, camera_model.FISHEYE)
    redistorted = core.distort_points(
        ideal, K, D, fish_Knew, camera_model.FISHEYE)
    roundtrip = np.max(np.linalg.norm(redistorted - raw_points.reshape(-1, 2), axis=1))
    check(roundtrip < 1e-4, '鱼眼去畸变点与正向畸变闭环一致', f'{roundtrip:.2e} px')

    map_x, map_y = camera_model.init_undistort_rectify_map(
        K, D, fish_Knew, size, camera_model.FISHEYE)
    image = np.full((size[1], size[0], 3), 127, dtype=np.uint8)
    undistorted = camera_model.undistort_image(
        image, K, D, fish_Knew, camera_model.FISHEYE)
    check(map_x.shape == (size[1], size[0]) and map_y.shape == map_x.shape
          and undistorted.shape == image.shape,
          '鱼眼图像去畸变与 reverse LUT 使用同尺寸映射')

    standard_D = np.zeros(5, dtype=np.float64)
    standard_image = np.arange(size[0] * size[1], dtype=np.uint8).reshape(
        size[1], size[0])
    expected = cv2.undistort(standard_image, expected_K, standard_D,
                             None, expected_K)
    actual = camera_model.undistort_image(
        standard_image, expected_K, standard_D, expected_K, camera_model.STANDARD)
    check(np.array_equal(actual, expected),
          '标准模型图像去畸变与既有 OpenCV 路径逐像素一致')

    paths = [Path(f'view_{i}.jpg') for i in range(len(objects))]
    fold = core._evaluate_calibration_fold({
        'fold_index': 0,
        'fold_name': paths[0].name,
        'total_views': len(paths),
        'candidate_counts': list(range(len(paths), 3, -1)),
        'train_obj': objects[1:],
        'train_img': images[1:],
        'train_used': paths[1:],
        'img_size': size,
        'heldout_obj': objects[0],
        'heldout_img': images[0],
        'camera_model': camera_model.FISHEYE,
    })
    fold_errors = [row['error'] for row in fold['candidate_results']]
    check(len(fold_errors) == len(paths) - 3 and fold_errors[0] is not None
          and all(error is not None for error in fold_errors),
          '鱼眼 LOOCV 折复用逐张剔除路径上的候选模型', str(fold_errors))


def test_model_persistence_and_basis(tmp: Path):
    old_root = core.SCRIPT_DIR
    try:
        core.configure_paths(root=tmp)
        size, K, D, _objects, _images = synthetic_views()
        core.DIR_CALIB_DATA.mkdir(parents=True, exist_ok=True)
        legacy_D = np.zeros(5, dtype=np.float64)
        canonical = json.dumps({
            'K': K.tolist(), 'D': legacy_D.tolist(), 'Knew': K.tolist(),
            'image_size': list(size),
        }, sort_keys=True, separators=(',', ':'))
        legacy = {
            'camera_matrix': K.tolist(), 'dist_coeffs': legacy_D.tolist(),
            'image_width': size[0], 'image_height': size[1],
            'calibration_basis_hash': hashlib.sha256(
                canonical.encode('utf-8')).hexdigest(),
        }
        core.CALIB_JSON.write_text(json.dumps(legacy), encoding='utf-8')
        loaded = core.load_calibration_with_model()
        check(loaded is not None and loaded[3] == 'standard'
              and core.restore_camera_model() == 'standard',
              '缺少 camera_model 的旧 calib.json 等价于 standard')
        check(core.calibration_basis_stale(legacy['calibration_basis_hash']) is None,
              '旧标准模型 basis hash 在迁移后仍能通过校验')

        core.configure_camera_model('fisheye', persist=True)
        core.configure_paths(root=tmp)
        check(core.CAMERA_MODEL == 'fisheye'
              and core.load_project_config().get('camera_model') == 'fisheye',
              '鱼眼模型选择写入并从 project.json 恢复')
        mismatch_rejected = False
        try:
            core.load_or_run_calibration_with_model()
        except SystemExit as exc:
            mismatch_rejected = '已有 calib.json 使用 standard 模型' in str(exc)
        check(mismatch_rejected,
              '选择模型与现有标定不同时拒绝静默复用')

        _rms, fitted_K, fitted_D, _rvecs, _tvecs, _std, _per_view = (
            core._calibrate_once(*synthetic_views()[3:], size, camera_model='fisheye'))
        core.commit_calibration(fitted_K, fitted_D, size, [], camera_model='fisheye')
        loaded = core.load_calibration_with_model()
        check(loaded is not None and loaded[3] == 'fisheye' and loaded[1].size == 4,
              'calib.json 保存并恢复鱼眼模型 ID')
        reused = core.load_or_run_calibration_with_model()
        check(reused[4] == 'fisheye',
              '相同模型的 calib.json 可通过标准复用入口恢复')
        fish_hash = core.calibration_basis_hash(fitted_K, fitted_D, fitted_K,
                                                size, 'fisheye')
        standard_hash = core.calibration_basis_hash(fitted_K, fitted_D, fitted_K,
                                                    size, 'standard')
        check(fish_hash != standard_hash,
              'basis hash 把 camera_model 纳入标定基准')
        check(core.calibration_basis_stale(standard_hash) is not None,
              '鱼眼标定不会接受同参数数组下的标准模型产物')

        fallback_root = tmp / 'calib_model_fallback'
        core.configure_paths(root=fallback_root)
        core.DIR_CALIB_DATA.mkdir(parents=True, exist_ok=True)
        core.CALIB_JSON.write_text(json.dumps({
            'camera_matrix': fitted_K.tolist(), 'dist_coeffs': fitted_D.tolist(),
            'image_width': size[0], 'image_height': size[1],
            'camera_model': 'fisheye',
        }), encoding='utf-8')
        core.configure_paths(root=fallback_root)
        check(core.CAMERA_MODEL == 'fisheye'
              and core.load_project_config().get('camera_model') == 'fisheye',
              '项目未记录模型时从鱼眼 calib.json 恢复并迁移设置')
        core.configure_paths(root=tmp)
    finally:
        core.configure_paths(root=old_root)


def test_fisheye_export(tmp: Path):
    size, K, D, _objects, _images = synthetic_views()
    scale = 0.1
    small_size = (int(size[0] * scale), int(size[1] * scale))
    small_K = K.copy()
    small_K[0, :] *= scale
    small_K[1, :] *= scale
    small_K[2, 2] = 1.0
    core.configure_paths(root=tmp)
    core.configure_camera_model('fisheye')
    core.TABLE_FORMAT = 'txt'
    core.TABLE_SIZE = None
    core.export_all(small_K, D, small_K, np.eye(3), np.eye(3), 1.0,
                    {}, small_size, camera_model='fisheye')
    matrices = json.loads((core.DIR_MATRIX / 'matrices.json').read_text(encoding='utf-8'))
    metadata = [json.loads(path.read_text(encoding='utf-8'))['camera_model']
                for path in core.DIR_TABLE.rglob('metadata.json')]
    check(matrices.get('camera_model') == 'fisheye'
          and len(metadata) == 4 and set(metadata) == {'fisheye'},
          '鱼眼模型贯穿 matrices.json 与四组 LUT 元数据', str(metadata))


def main() -> int:
    temp = Path(tempfile.mkdtemp(prefix='neuq_camera_models_'))
    try:
        print('[1] 鱼眼标定、姿态、去畸变与点映射')
        test_fisheye_calibration_and_geometry()
        print('[2] 相机模型持久化与标定基准')
        test_model_persistence_and_basis(temp / 'persistence')
        print('[3] 鱼眼矩阵与 LUT 导出')
        test_fisheye_export(temp / 'export')
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main())
