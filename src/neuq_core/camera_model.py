"""Thin adapters for the supported OpenCV camera models."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

STANDARD = 'standard'
FISHEYE = 'fisheye'
CAMERA_MODELS = (STANDARD, FISHEYE)


def normalize_camera_model(value: Optional[str] = None) -> str:
    """Normalize a model ID; absent legacy values mean the standard model."""
    model = STANDARD if value in (None, '') else str(value).strip().lower()
    if model not in CAMERA_MODELS:
        raise ValueError(f'不支持的相机模型 {value!r}；可选 standard 或 fisheye。')
    return model


def _fisheye_coefficients(D: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(D, dtype=np.float64).reshape(-1)
    if coeffs.size != 4:
        raise ValueError(f'鱼眼模型需要 4 个畸变系数，收到 {coeffs.size} 个。')
    return coeffs.reshape(4, 1)


def project_points(object_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                   K: np.ndarray, D: np.ndarray,
                   camera_model: Optional[str] = None):
    """Project 3D points with the selected camera model, matching OpenCV's return shape."""
    model = normalize_camera_model(camera_model)
    if model == FISHEYE:
        points = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
        return cv2.fisheye.projectPoints(
            points, rvec, tvec, K, _fisheye_coefficients(D))
    return cv2.projectPoints(object_points, rvec, tvec, K, D)


def undistort_points(points: np.ndarray, K: np.ndarray, D: np.ndarray,
                     Knew: Optional[np.ndarray] = None,
                     camera_model: Optional[str] = None) -> np.ndarray:
    """Map distorted pixels to the chosen model's ideal pixel coordinates."""
    model = normalize_camera_model(camera_model)
    if model == FISHEYE:
        return cv2.fisheye.undistortPoints(
            np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
            K, _fisheye_coefficients(D), P=Knew).reshape(-1, 2)
    return cv2.undistortPoints(points, K, D, P=Knew).reshape(-1, 2)


def init_undistort_rectify_map(K: np.ndarray, D: np.ndarray, Knew: np.ndarray,
                               size: Tuple[int, int],
                               camera_model: Optional[str] = None):
    """Build remap grids for a distorted image and the selected camera model."""
    model = normalize_camera_model(camera_model)
    if model == FISHEYE:
        return cv2.fisheye.initUndistortRectifyMap(
            K, _fisheye_coefficients(D), np.eye(3), Knew, size, cv2.CV_32FC1)
    return cv2.initUndistortRectifyMap(K, D, None, Knew, size, cv2.CV_32FC1)


def undistort_image(image: np.ndarray, K: np.ndarray, D: np.ndarray,
                    Knew: np.ndarray, camera_model: Optional[str] = None) -> np.ndarray:
    """Undistort one image without changing its output dimensions."""
    model = normalize_camera_model(camera_model)
    if model == STANDARD:
        return cv2.undistort(image, K, D, None, Knew)
    size = (int(image.shape[1]), int(image.shape[0]))
    map_x, map_y = init_undistort_rectify_map(K, D, Knew, size, model)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


def estimate_new_camera_matrix(K: np.ndarray, D: np.ndarray,
                               size: Tuple[int, int], alpha: Optional[float],
                               camera_model: Optional[str] = None) -> np.ndarray:
    """Resolve the output intrinsics while preserving the existing same-view default."""
    model = normalize_camera_model(camera_model)
    if alpha is None:
        return np.asarray(K, dtype=np.float64).copy()
    if model == FISHEYE:
        return np.asarray(cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, _fisheye_coefficients(D), size, np.eye(3), balance=float(alpha),
            new_size=size), dtype=np.float64)
    Knew, _roi = cv2.getOptimalNewCameraMatrix(K, D, size, float(alpha), size)
    return np.asarray(Knew, dtype=np.float64)


def solve_pnp(object_points: np.ndarray, image_points: np.ndarray,
              K: np.ndarray, D: np.ndarray,
              camera_model: Optional[str] = None):
    """Estimate a view pose using the same lens model used for calibration."""
    model = normalize_camera_model(camera_model)
    if model == FISHEYE:
        return cv2.fisheye.solvePnP(
            np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3),
            np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2),
            K, _fisheye_coefficients(D), flags=cv2.SOLVEPNP_ITERATIVE)
    return cv2.solvePnP(object_points, image_points, K, D,
                        flags=cv2.SOLVEPNP_ITERATIVE)


def calibrate_fisheye(obj_points: Sequence[np.ndarray],
                      img_points: Sequence[np.ndarray],
                      img_size: Tuple[int, int]):
    """Fit OpenCV's four-coefficient fisheye model and return per-view RMS values."""
    object_points = [np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)
                     for points in obj_points]
    image_points = [np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
                    for points in img_points]
    K = np.eye(3, dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        object_points, image_points, img_size, K, D, flags=flags, criteria=criteria)

    per_view = []
    for objp, imgp, rvec, tvec in zip(
            object_points, image_points, rvecs, tvecs, strict=True):
        projected, _jacobian = project_points(objp, rvec, tvec, K, D, FISHEYE)
        delta = projected.reshape(-1, 2) - imgp.reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return (float(rms), np.asarray(K, dtype=np.float64),
            np.asarray(D, dtype=np.float64).ravel(), rvecs, tvecs,
            np.array([], dtype=np.float64), np.asarray(per_view, dtype=np.float64))


__all__ = [
    'CAMERA_MODELS',
    'FISHEYE',
    'STANDARD',
    'calibrate_fisheye',
    'estimate_new_camera_matrix',
    'init_undistort_rectify_map',
    'normalize_camera_model',
    'project_points',
    'solve_pnp',
    'undistort_image',
    'undistort_points',
]
