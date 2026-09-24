"""相机标定的数学内核：检测棋盘角点，并把多张观测交给 OpenCV 拟合。

一张棋盘照片提供两组彼此对应的点：棋盘平面上已知的 ``object points``（mm）和
照片中检测到的 ``image points``（px）。多张不同距离、倾角和画面位置的照片共同
约束相机内参 ``K``、所选模型的畸变参数 ``D``，以及每张照片各自的姿态 ``rvec/tvec``。
标准模型使用 ``D=[k1,k2,p1,p2,k3]``；鱼眼模型使用独立的四参数模型。只拍一张正对镜头的棋盘，
许多参数会彼此“冒充”，数值看似能拟合，换到画面边缘却不可信。

本模块只保留不依赖目录与界面的计算。素材读取、异常帧剔除和结果提交由
``neuq_vision_calib.py`` 编排。背景知识见 ``KNOWLEDGE_GUIDE.md`` 第 3～6 节。
"""

from math import isfinite, sqrt
from statistics import stdev
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from . import camera_model as _camera_model
from . import config

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

# 采集质量诊断的启发式，不是相机标定合格标准。口径与 README 的“标定数据诊断”一致。
CALIBRATION_DIAGNOSTIC_HEURISTICS = {
    'heatmap_cols': 8,
    'heatmap_rows': 6,
    'cell_min_views': 2,
    'coverage_good_ratio': 0.50,
    'coverage_fair_ratio': 0.25,
    'edge_good_ratio': 0.55,
    'edge_fair_ratio': 0.30,
    'corner_min_views': 2,
    'corner_good_regions': 3,
    'corner_fair_regions': 2,
    'position_cols': 4,
    'position_rows': 3,
    'position_repeat_min_views': 3,
    'position_repeat_fraction': 0.35,
    'position_some_min_views': 2,
    'position_some_fraction': 0.20,
    'pose_tilt_deg': 12.0,
    'pose_spread_good_deg': 25.0,
    'pose_spread_fair_deg': 12.0,
    'pose_good_fraction': 0.30,
    'pose_fair_fraction': 0.15,
    'scale_good_ratio': 1.8,
    'scale_fair_ratio': 1.3,
    'focal_sigma_good_ratio': 0.02,
    'focal_sigma_fair_ratio': 0.05,
    'principal_sigma_good_ratio': 0.02,
    'principal_sigma_fair_ratio': 0.05,
    'suggested_region_min_views': 2,
    'suggested_region_max_views': 3,
}


def detect_chessboard(gray: np.ndarray, fast: bool = False,
                      board: Optional[config.CheckerboardSpec] = None
                      ) -> Tuple[bool, Optional[np.ndarray]]:
    """检出棋盘内角点，全部检出才算成功。

    ``board.corners`` 指相邻方格交界处的**内角点数**：例如 12×9 个方格只有
    11×8 个内角点。OpenCV 的棋盘检测是全有全无的：拓扑推断要求整块棋盘完整可见，
    检不到就返回 False，
    不存在"只返回一部分角点"的中间态。所以这里的策略是尽量提高检出率：
    先用 SB（sector-based）算法，它对低分辨率、运动模糊、光照不均和大透视畸变明显更鲁棒，
    且自带亚像素精度；失败再退回经典算法配 cornerSubPix。

    真正需要「部分可见也能用」的场合（棋盘被裁切或遮挡），普通棋盘做不到，
    必须换成 ChArUco 板——每个格子有唯一编码，才能把局部角点对上正确的物理坐标。

    fast=True 用于实时预览，跳过耗时的 EXHAUSTIVE/ACCURACY 搜索。
    board 不给时使用当前工程的标定板规格。
    """
    spec = config.BOARD if board is None else board
    if hasattr(cv2, 'findChessboardCornersSB'):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if not fast:
            flags |= cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, spec.corners, flags)
        if found:
            return True, corners

    found, corners = cv2.findChessboardCorners(
        gray, spec.corners,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return False, None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
    return True, corners


def detect_chessboard_partial(gray: np.ndarray,
                              board: Optional[config.CheckerboardSpec] = None
                              ) -> Optional[Tuple[int, int]]:
    """用更小的内角点阵去匹配画面里的局部棋盘，返回命中的尺寸，全不中返回 None。

    用途是把"棋盘没拍全的照片"和"压根没有棋盘的地面照"分开。完整棋盘要求整块可见，
    拍不全就一律检不出；但这两类图在流程里的去处完全不同——前者属于素材没拍好，
    后者才是真正的逆透视候选。只靠"完整板检没检出"这一个二值判断，会把拍不全的
    棋盘照误当成地面照塞进 ipm_input/，让用户在挑原图时莫名其妙看到一张棋盘。

    候选尺寸由 board.partial_grids 按当前规格动态生成（见 CheckerboardSpec），
    不能写死：换一块小棋盘之后，固定候选就全不合理了。

    实测（38 张现场素材）：8 张地面照连 5x3 子网格都检不出，唯一那张拍不全的棋盘照
    稳定命中 5x3，区分度是干净的。
    """
    spec = config.BOARD if board is None else board
    if not hasattr(cv2, 'findChessboardCornersSB'):
        return None
    flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
             | cv2.CALIB_CB_ACCURACY)
    for size in spec.partial_grids:
        found, _corners = cv2.findChessboardCornersSB(gray, size, flags)
        if found:
            return size
    return None


def _calibrate_once(obj_points, img_points, img_size, *, camera_model='standard'):
    """用当前全部观测跑一次标定，返回 ``(rms, K, D, rvecs, tvecs, ...)``。

    ``obj_points`` 是每张图对应的棋盘平面点（mm），``img_points`` 是同序角点（px），
    ``img_size`` 是图像 ``(W, H)``。``K`` 把归一化相机坐标变成像素坐标；``D`` 中
    ``k1/k2/k3`` 描述随半径变化的径向畸变，``p1/p2`` 描述镜头装配偏心造成的切向
    畸变。每张图的 ``rvec/tvec`` 则回答“这块棋盘相对相机在哪里”，它们不是一套
    可以跨照片共用的相机内参。

    优先用 calibrateCameraExtended 以拿到每张图的 RMS 与内参标准差；旧版 OpenCV
    没有这个接口时退回 calibrateCamera，此时这两个量以空数组代替。
    """
    model = _camera_model.normalize_camera_model(camera_model)
    if model == _camera_model.FISHEYE:
        return _camera_model.calibrate_fisheye(obj_points, img_points, img_size)

    if hasattr(cv2, 'calibrateCameraExtended'):
        rms, K, D, rvecs, tvecs, std_int, _std_ext, per_view = cv2.calibrateCameraExtended(
            obj_points, img_points, img_size, None, None)
        per_view = np.asarray(per_view, dtype=np.float64).ravel()
        std_int = np.asarray(std_int, dtype=np.float64).ravel()
    else:
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None)
        per_view = np.array([], dtype=np.float64)
        std_int = np.array([], dtype=np.float64)
    return (rms, np.asarray(K, dtype=np.float64),
            np.asarray(D, dtype=np.float64).ravel(),
            rvecs, tvecs, std_int, per_view)


def report_reprojection_error(obj_points, img_points, rvecs, tvecs,
                              K: np.ndarray, D: np.ndarray,
                              camera_model: str = 'standard') -> float:
    """返回所有角点的平均欧氏重投影误差（px）。

    做法是把已知棋盘点按拟合出的 ``K/D/rvec/tvec`` 重新投回照片，再量预测点与实测
    角点的距离。它回答“一个角点平均偏了多少像素”。OpenCV 返回的 RMS 先平方误差、
    求平均再开方，对少数大误差更敏感；两个数口径不同，不能互相替代。

    RMS 或均值很小也不代表画面边缘一定可信：如果角点都集中在中央，边缘畸变参数
    仍缺少数据约束。这里单独计算均值，是为了与 MATLAB cameraCalibrator 的
    Overall Mean Error 采用同一口径，便于对比。
    """
    total_err = 0.0
    total_pts = 0
    for objp, imgp, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs, strict=True):
        proj, _ = _camera_model.project_points(
            objp, rvec, tvec, K, D, camera_model)
        diff = proj.reshape(-1, 2) - imgp.reshape(-1, 2)
        total_err += float(np.sum(np.linalg.norm(diff, axis=1)))
        total_pts += diff.shape[0]
    return total_err / max(1, total_pts)


def recommend_reprojection_threshold(errors: Sequence[float]) -> dict:
    """根据未经筛除的首轮逐视图 RMS，给出可选的高误差筛选阈值建议。

    使用 median + 3 * (1.4826 * MAD) 识别明显高误差视图，再把建议值放在
    最大正常值与最小异常值之间。少于 5 个有效视图、MAD 退化或没有异常时不
    提供数值建议；该结果仅供用户参考，不会自动应用。

    返回 ``threshold``、``outlier_count``、``sample_count`` 和 ``status``，其中
    status 为 ``recommended``、``too_few_samples``、``mad_degenerate`` 或
    ``no_outliers``。
    """
    values = np.asarray(errors, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    sample_count = int(values.size)
    result = {
        'threshold': None,
        'outlier_count': 0,
        'sample_count': sample_count,
        'status': 'too_few_samples',
    }
    if sample_count < 5:
        return result

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= np.finfo(np.float64).eps * max(1.0, abs(median)):
        result['status'] = 'mad_degenerate'
        return result

    robust_sigma = 1.4826 * mad
    abnormal = values > median + 3.0 * robust_sigma
    outlier_count = int(np.count_nonzero(abnormal))
    if not outlier_count:
        result['status'] = 'no_outliers'
        return result

    max_normal = float(np.max(values[~abnormal]))
    min_abnormal = float(np.min(values[abnormal]))
    midpoint = (max_normal + min_abnormal) / 2.0
    threshold = midpoint
    # 通常精确到 0.01 px；区间很窄时提高精度，确保舍入后仍严格位于两组之间。
    for decimals in range(2, 10):
        candidate = round(midpoint, decimals)
        if max_normal < candidate < min_abnormal:
            threshold = candidate
            break

    result.update(threshold=float(threshold), outlier_count=outlier_count,
                  status='recommended')
    return result


def calibration_filter_geometry_regressions(baseline: dict, candidate: dict) -> list[str]:
    """列出相对“不剔除”基线明显降级的采集几何维度。

    只比较现有诊断给出的等级，不引入新的覆盖、姿态或尺度阈值。
    """
    keys = ('image_coverage', 'edge_coverage', 'pose_diversity', 'scale_diversity')
    rank = {'weak': 0, 'fair': 1, 'good': 2}
    base_metrics = {item.get('key'): item for item in baseline.get('metrics', [])}
    next_metrics = {item.get('key'): item for item in candidate.get('metrics', [])}
    regressions = []
    for key in keys:
        base = base_metrics.get(key)
        current = next_metrics.get(key)
        if (base is None or current is None
                or base.get('level') not in rank or current.get('level') not in rank
                or rank[current['level']] < rank[base['level']]):
            label = (current or base or {}).get('label', key)
            regressions.append(str(label))
    return regressions


def select_calibration_filter_candidate(candidates: Sequence[dict], fold_count: int) -> dict:
    """按 LOOCV 平方误差的一标准误规则选候选，并在接近最优时保留更多照片。"""
    valid = []
    for candidate in candidates:
        errors = candidate.get('fold_errors', [])
        try:
            finite_errors = [float(value) for value in errors]
        except (TypeError, ValueError):
            continue
        if (not candidate.get('geometry_safe', False) or len(errors) != fold_count
                or any(not isfinite(value) or value < 0 for value in finite_errors)):
            continue
        losses = [value ** 2 for value in finite_errors]
        mean_loss = sum(losses) / fold_count
        valid.append({**candidate, 'fold_losses': losses,
                      'mean_squared_error': mean_loss,
                      'cv_rms': sqrt(mean_loss)})
    if not valid:
        return {'best_cv_rms': None, 'best_retained_count': None,
                'one_se_limit_mse': None, 'near_best': []}

    best = min(valid, key=lambda item: item['mean_squared_error'])
    best_losses = best['fold_losses']
    standard_error = stdev(best_losses) / sqrt(fold_count) if fold_count > 1 else 0.0
    one_se_limit = best['mean_squared_error'] + standard_error
    near_best = [item for item in valid
                 if item['mean_squared_error'] <= one_se_limit + 1e-12]
    near_best.sort(key=lambda item: (-int(item['retained_count']),
                                     item['mean_squared_error']))
    return {
        'best_cv_rms': float(best['cv_rms']),
        'best_retained_count': int(best['retained_count']),
        'one_se_limit_mse': float(one_se_limit),
        'near_best': near_best,
    }


def diagnose_calibration_views(image_points, object_points, rvecs, tvecs,
                               K: np.ndarray, std_int: np.ndarray,
                               img_size: Tuple[int, int], names: Sequence[str],
                               rms_errors: Sequence[float]) -> dict:
    """生成采集质量提示：逐图网格贡献、位置重复、姿态/尺度多样性及内参不确定性。

    每张照片在一个热力图格子里最多贡献一次，避免同一张棋盘的许多角点把重复观测数
    人为放大。状态阈值集中在 ``CALIBRATION_DIAGNOSTIC_HEURISTICS``；所有结论都只是
    采集提示，不是标定是否合格的判据，也不会据此自动剔除照片。
    """
    width, height = map(int, img_size)
    if width < 2 or height < 2:
        raise ValueError('图像尺寸必须至少为 2×2。')

    rules = CALIBRATION_DIAGNOSTIC_HEURISTICS
    cols = rules['heatmap_cols']
    rows = rules['heatmap_rows']
    cell_counts = np.zeros((rows, cols), dtype=np.int32)
    position_counts = np.zeros((rules['position_rows'], rules['position_cols']),
                               dtype=np.int32)
    corner_names = ('左上', '右上', '左下', '右下')
    corner_view_counts = np.zeros(4, dtype=np.int32)
    corner_cells = (
        (range(0, 2), range(0, 2)),
        (range(0, 2), range(cols - 2, cols)),
        (range(rows - 2, rows), range(0, 2)),
        (range(rows - 2, rows), range(cols - 2, cols)),
    )
    edge_mask = np.zeros((rows, cols), dtype=bool)
    edge_mask[0, :] = edge_mask[-1, :] = True
    edge_mask[:, 0] = edge_mask[:, -1] = True

    views = []
    areas = []
    normals = []
    tilts = []
    n_views = len(image_points)
    pos_cols = rules['position_cols']
    pos_rows = rules['position_rows']
    for index, raw_points in enumerate(image_points):
        points = np.asarray(raw_points, dtype=np.float64).reshape(-1, 2)
        inside = (np.isfinite(points).all(axis=1)
                  & (points[:, 0] >= 0) & (points[:, 0] <= width - 1)
                  & (points[:, 1] >= 0) & (points[:, 1] <= height - 1))
        visible = points[inside]
        cells = set()
        center_x = center_y = area_fraction = 0.0
        position_region = '未知区域'
        if visible.size:
            normalized = visible / np.array([width - 1, height - 1], dtype=np.float64)
            cell_x = np.minimum((normalized[:, 0] * cols).astype(int), cols - 1)
            cell_y = np.minimum((normalized[:, 1] * rows).astype(int), rows - 1)
            cells = set(zip(cell_y.tolist(), cell_x.tolist(), strict=True))
            for row, col in cells:
                cell_counts[row, col] += 1
            center_x, center_y = np.mean(normalized, axis=0).tolist()
            pos_x = min(int(center_x * pos_cols), pos_cols - 1)
            pos_y = min(int(center_y * pos_rows), pos_rows - 1)
            position_counts[pos_y, pos_x] += 1
            position_region = _diagnostic_position_region(pos_x, pos_y, pos_cols, pos_rows)

            if len(visible) >= 3:
                hull = cv2.convexHull(visible.astype(np.float32))
                area_fraction = float(cv2.contourArea(hull) / ((width - 1) * (height - 1)))
            areas.append(area_fraction)

            for corner_index, (corner_rows, corner_cols) in enumerate(corner_cells):
                if any(row in corner_rows and col in corner_cols for row, col in cells):
                    corner_view_counts[corner_index] += 1

        rvec = rvecs[index] if index < len(rvecs) else None
        tvec = tvecs[index] if index < len(tvecs) else None
        tilt_x = tilt_y = tilt = distance_cm = None
        if rvec is not None:
            try:
                rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
                normal = rotation[:, 2].copy()
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                nz = abs(float(normal[2]))
                tilt = float(np.degrees(np.arccos(np.clip(nz, 0.0, 1.0))))
                tilt_x = float(np.degrees(np.arctan2(normal[0], max(nz, 1e-12))))
                tilt_y = float(np.degrees(np.arctan2(normal[1], max(nz, 1e-12))))
                normals.append(normal)
                tilts.append(tilt)
                if tvec is not None and index < len(object_points):
                    board_center = np.mean(
                        np.asarray(object_points[index], dtype=np.float64).reshape(-1, 3), axis=0)
                    camera_center = rotation @ board_center + np.asarray(
                        tvec, dtype=np.float64).reshape(3)
                    distance_cm = float(np.linalg.norm(camera_center) / 10.0)
            except (cv2.error, ValueError):
                pass

        error = float(rms_errors[index]) if index < len(rms_errors) else None
        views.append({
            'name': str(names[index]) if index < len(names) else f'view_{index + 1}',
            'center_x': float(center_x),
            'center_y': float(center_y),
            'area_fraction': float(area_fraction),
            'occupied_cells': int(len(cells)),
            'position_region': position_region,
            'tilt_x_deg': tilt_x,
            'tilt_y_deg': tilt_y,
            'tilt_deg': tilt,
            'distance_cm': distance_cm,
            'rms_px': error,
        })

    min_cell_views = rules['cell_min_views']
    occupied_once = cell_counts >= 1
    sufficiently_observed = cell_counts >= min_cell_views
    coverage_ratio = float(np.count_nonzero(sufficiently_observed) / cell_counts.size)
    edge_ratio = float(np.count_nonzero(sufficiently_observed & edge_mask)
                        / np.count_nonzero(edge_mask))
    good_regions = int(np.count_nonzero(corner_view_counts >= rules['corner_min_views']))
    coverage_label, coverage_level = _diagnostic_ratio_label(
        coverage_ratio, rules['coverage_good_ratio'], rules['coverage_fair_ratio'])
    if (edge_ratio >= rules['edge_good_ratio']
            and good_regions >= rules['corner_good_regions']):
        edge_label, edge_level = '良好', 'good'
    elif (edge_ratio >= rules['edge_fair_ratio']
          and good_regions >= rules['corner_fair_regions']):
        edge_label, edge_level = '一般', 'fair'
    else:
        edge_label, edge_level = '较弱', 'weak'

    pos_y, pos_x = np.unravel_index(np.argmax(position_counts), position_counts.shape)
    repeated_count = int(position_counts[pos_y, pos_x]) if n_views else 0
    repeated_fraction = repeated_count / max(1, n_views)
    if (repeated_count >= rules['position_repeat_min_views']
            and repeated_fraction >= rules['position_repeat_fraction']):
        repetition_label, repetition_level = '较多', 'weak'
    elif (repeated_count >= rules['position_some_min_views']
          and repeated_fraction >= rules['position_some_fraction']):
        repetition_label, repetition_level = '一般', 'fair'
    else:
        repetition_label, repetition_level = '较少', 'good'
    repeated_region = _diagnostic_position_region(pos_x, pos_y, pos_cols, pos_rows)

    pose_spread = 0.0
    if len(normals) > 1:
        pose_spread = max(
            float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0))))
            for i, a in enumerate(normals) for b in normals[i + 1:])
    varied_fraction = (sum(value >= rules['pose_tilt_deg'] for value in tilts)
                       / max(1, len(tilts)))
    if (len(tilts) >= 3 and pose_spread >= rules['pose_spread_good_deg']
            and varied_fraction >= rules['pose_good_fraction']):
        pose_label, pose_level = '良好', 'good'
    elif (len(tilts) >= 3 and pose_spread >= rules['pose_spread_fair_deg']
          and varied_fraction >= rules['pose_fair_fraction']):
        pose_label, pose_level = '一般', 'fair'
    else:
        pose_label, pose_level = '较弱', 'weak'

    positive_areas = [value for value in areas if value > 0 and np.isfinite(value)]
    scale_ratio = (float(np.sqrt(max(positive_areas) / min(positive_areas)))
                   if positive_areas else 1.0)
    if scale_ratio >= rules['scale_good_ratio']:
        scale_label, scale_level = '良好', 'good'
    elif scale_ratio >= rules['scale_fair_ratio']:
        scale_label, scale_level = '一般', 'fair'
    else:
        scale_label, scale_level = '较弱', 'weak'

    std_values = np.asarray(std_int, dtype=np.float64).ravel()
    intrinsic_names = ('fx', 'fy', 'cx', 'cy')
    intrinsic_values = (float(K[0, 0]), float(K[1, 1]),
                        float(K[0, 2]), float(K[1, 2]))
    intrinsic_bases = (abs(intrinsic_values[0]), abs(intrinsic_values[1]),
                       float(width), float(height))
    stability_rows = []
    stability_available = len(std_values) >= 4
    relative_uncertainties = []
    if stability_available:
        for index, name in enumerate(intrinsic_names):
            sigma = float(std_values[index])
            value = intrinsic_values[index]
            relative = sigma / intrinsic_bases[index] if intrinsic_bases[index] else np.inf
            if not (np.isfinite(sigma) and sigma >= 0 and np.isfinite(value)
                    and np.isfinite(relative)):
                stability_available = False
            stability_rows.append({
                'name': name,
                'value_px': value if np.isfinite(value) else None,
                'sigma_px': sigma if np.isfinite(sigma) else None,
                'relative_uncertainty': float(relative) if np.isfinite(relative) else None,
            })
            relative_uncertainties.append((float(relative), index))
    if not stability_available:
        stability_label, stability_level = '不可用', 'unknown'
        stability_max_ratio = None
    else:
        stability_max_ratio = max(value for value, _ in relative_uncertainties)
        normalized_limits = [
            rules['focal_sigma_good_ratio'], rules['focal_sigma_good_ratio'],
            rules['principal_sigma_good_ratio'], rules['principal_sigma_good_ratio'],
        ]
        fair_limits = [
            rules['focal_sigma_fair_ratio'], rules['focal_sigma_fair_ratio'],
            rules['principal_sigma_fair_ratio'], rules['principal_sigma_fair_ratio'],
        ]
        if all(value <= normalized_limits[index]
               for value, index in relative_uncertainties):
            stability_label, stability_level = '良好', 'good'
        elif all(value <= fair_limits[index] for value, index in relative_uncertainties):
            stability_label, stability_level = '一般', 'fair'
        else:
            stability_label, stability_level = '较弱', 'weak'

    coverage_pct = coverage_ratio * 100.0
    edge_pct = edge_ratio * 100.0
    metrics = [
        {'key': 'image_coverage', 'label': '画面覆盖', 'value': coverage_label,
         'level': coverage_level,
         'detail': (f'{np.count_nonzero(sufficiently_observed)} / {cell_counts.size} 个网格'
                    f'由至少 {min_cell_views} 张照片覆盖（{coverage_pct:.0f}%）')},
        {'key': 'edge_coverage', 'label': '边缘覆盖', 'value': edge_label,
         'level': edge_level,
         'detail': (f'外圈网格至少 {min_cell_views} 张观测的比例 {edge_pct:.0f}%；'
                    f'四角中 {good_regions} 个有重复观测')},
        {'key': 'position_repetition', 'label': '位置重复', 'value': repetition_label,
         'level': repetition_level,
         'detail': (f'{repeated_region}的棋盘中心出现在 {repeated_count} / {n_views} 张照片'
                    if n_views else '没有可用视图')},
        {'key': 'pose_diversity', 'label': '姿态变化', 'value': pose_label,
         'level': pose_level,
         'detail': (f'{sum(value >= rules["pose_tilt_deg"] for value in tilts)} / {len(tilts)} 张'
                    f'倾角至少 {rules["pose_tilt_deg"]:.0f}°；棋盘法线最大夹角 {pose_spread:.1f}°')},
        {'key': 'scale_diversity', 'label': '尺度变化', 'value': scale_label,
         'level': scale_level,
         'detail': f'棋盘投影面积的等效线性尺寸最大 / 最小约 {scale_ratio:.2f} 倍'},
        {'key': 'parameter_stability', 'label': '内参稳定性', 'value': stability_label,
         'level': stability_level,
         'detail': ('依据 fx / fy / cx / cy 标准差；不包括畸变参数的不确定性，也不代表正确性保证。'
                    + (f'最大相对不确定性约 {stability_max_ratio * 100:.2f}%。'
                       if stability_max_ratio is not None else '当前 OpenCV 未提供可用标准差。'))},
    ]

    recommendations = []
    missing_corners = [corner_names[i] for i, count in enumerate(corner_view_counts)
                       if count < rules['corner_min_views']]
    if missing_corners:
        low = rules['suggested_region_min_views']
        high = rules['suggested_region_max_views']
        corner_list = '、'.join(missing_corners)
        recommendations.append({
            'topic': '边缘覆盖',
            'text': (f'在{corner_list}附近各补拍 {low}～{high} 张；让棋盘角点延伸到图像边缘/角区，'
                     '不要只移动棋盘中心，并保持棋盘完整、清晰可见。'),
        })
    elif edge_level == 'weak':
        recommendations.append({
            'topic': '边缘覆盖',
            'text': '把棋盘移到热力图中观测较少的四边和四角补拍；让角点实际进入边缘区域，'
                    '不要只移动棋盘中心，并保持棋盘完整、清晰可见。',
        })
    if coverage_level == 'weak':
        recommendations.append({
            'topic': '画面覆盖',
            'text': '优先让角点进入热力图中空白或只出现过一次的网格区域。',
        })
    if repetition_level == 'weak':
        recommendations.append({
            'topic': '位置重复',
            'text': (f'{repeated_region}重复出现较多（{repeated_count} / {n_views} 张），'
                     '相似位置新增的几何信息有限；后续把棋盘中心移到热力图中观测较少的区域，'
                     '并同时改变距离和倾角。已有照片不会仅因位置重复而自动删除。'),
        })
    if pose_level == 'weak':
        recommendations.append({
            'topic': '姿态变化',
            'text': '增加几张明显左右、上下倾斜的完整棋盘照片，避免几乎都正对镜头。',
        })
    if scale_level == 'weak':
        recommendations.append({
            'topic': '尺度变化',
            'text': '增加几张距离明显不同的完整棋盘照片，让棋盘在画面中的投影尺寸有变化。',
        })
    if stability_level == 'weak':
        recommendations.append({
            'topic': '内参稳定性',
            'text': '核心内参不确定性参考偏高；可补充不同画面位置、倾角和距离的观测，再比较变化。',
        })
    if not recommendations:
        recommendations.append({
            'topic': '检查说明',
            'text': '这些启发式检查未发现突出的采集缺口；仍需结合重投影误差、去畸变预览和实际成像检查。',
        })

    return {
        'view_count': int(n_views),
        'metrics': metrics,
        'recommendations': [f"{item['topic']}：{item['text']}" for item in recommendations],
        'heatmap': {
            'cols': int(cols), 'rows': int(rows), 'counts': cell_counts.tolist(),
            'max_count': int(np.max(cell_counts)) if cell_counts.size else 0,
            'observed_once_ratio': float(np.count_nonzero(occupied_once) / cell_counts.size),
            'repeated_ratio': coverage_ratio,
        },
        'corners': [{'name': corner_names[i], 'view_count': int(count)}
                    for i, count in enumerate(corner_view_counts)],
        'std_intrinsics': stability_rows if stability_available else [],
        'std_intrinsics_note': ('仅表示当前模型和当前数据下 fx/fy/cx/cy 的估计不确定性参考；'
                                '不包含畸变参数不确定性，也不能单独证明标定正确。'),
        'heuristics_note': ('以下状态和逐项补拍建议是采集质量启发式提示，不是相机标定合格标准。'
                            '位置重复不等同于坏照片：重复视图可降低该区域的随机观测噪声，'
                            '但新增几何信息会递减；分布失衡可能削弱参数可辨识性和边缘泛化。'
                            '程序不会自动删除位置重复或低信息量视图。'),
        'views': views,
    }


def _diagnostic_ratio_label(value: float, good: float, fair: float):
    if value >= good:
        return '良好', 'good'
    if value >= fair:
        return '一般', 'fair'
    return '较弱', 'weak'


def _diagnostic_position_region(x: int, y: int, cols: int, rows: int) -> str:
    if y == rows // 2 and x in (cols // 2 - 1, cols // 2):
        return '中央区域'
    horizontal = '左侧' if x == 0 else ('右侧' if x == cols - 1 else '')
    vertical = '上方' if y == 0 else ('下方' if y == rows - 1 else '')
    if horizontal and vertical:
        return f'{horizontal[0]}{vertical[0]}区域'
    if horizontal:
        return f'{horizontal}区域'
    if vertical:
        return f'{vertical}区域'
    return '中央区域'


__all__ = [
    'CALIBRATION_DIAGNOSTIC_HEURISTICS',
    'SUBPIX_CRITERIA',
    '_calibrate_once',
    'calibration_filter_geometry_regressions',
    'detect_chessboard',
    'detect_chessboard_partial',
    'diagnose_calibration_views',
    'recommend_reprojection_threshold',
    'report_reprojection_error',
    'select_calibration_filter_candidate',
]

