"""相机标定的数学内核：检测棋盘角点，并把多张观测交给 OpenCV 拟合。

一张棋盘照片提供两组彼此对应的点：棋盘平面上已知的 ``object points``（mm）和
照片中检测到的 ``image points``（px）。多张不同距离、倾角和画面位置的照片共同
约束相机内参 ``K``、五个畸变参数 ``D=[k1,k2,p1,p2,k3]``，以及每张照片各自的
姿态 ``rvec/tvec``。只拍一张正对镜头的棋盘，许多参数会彼此“冒充”，数值看似能拟合，
换到画面边缘却不可信。

本模块只保留不依赖目录与界面的计算。素材读取、异常帧剔除和结果提交由
``neuq_vision_calib.py`` 编排。背景知识见 ``docs/KNOWLEDGE_GUIDE.md`` 第 3～6 节。
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from . import config

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


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


def _calibrate_once(obj_points, img_points, img_size):
    """用当前全部观测跑一次标定，返回 ``(rms, K, D, rvecs, tvecs, ...)``。

    ``obj_points`` 是每张图对应的棋盘平面点（mm），``img_points`` 是同序角点（px），
    ``img_size`` 是图像 ``(W, H)``。``K`` 把归一化相机坐标变成像素坐标；``D`` 中
    ``k1/k2/k3`` 描述随半径变化的径向畸变，``p1/p2`` 描述镜头装配偏心造成的切向
    畸变。每张图的 ``rvec/tvec`` 则回答“这块棋盘相对相机在哪里”，它们不是一套
    可以跨照片共用的相机内参。

    优先用 calibrateCameraExtended 以拿到每张图的 RMS 与内参标准差；旧版 OpenCV
    没有这个接口时退回 calibrateCamera，此时这两个量以空数组代替。
    """
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
                              K: np.ndarray, D: np.ndarray) -> float:
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
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        diff = proj.reshape(-1, 2) - imgp.reshape(-1, 2)
        total_err += float(np.sum(np.linalg.norm(diff, axis=1)))
        total_pts += diff.shape[0]
    return total_err / max(1, total_pts)


__all__ = [
    'SUBPIX_CRITERIA',
    '_calibrate_once',
    'detect_chessboard',
    'detect_chessboard_partial',
    'report_reprojection_error',
]

