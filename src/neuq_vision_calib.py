"""NEUQ 智能车视觉标定一体化工具。

目录约定：各工作目录平铺在工程根；data/ 只装两类不属于流水线产物的东西——
导入前的原始素材（data/import/）与备份压缩包（data/backups/），两者都不入库。

流程：
  1. 相机标定（读 calib_input/ 已有照片，或在线拍摄自动存入）
     → calib_data/calib.json
  2. 标定照片去畸变效果 → calib_preview/
  3. 逆透视标定原图去畸变 → ipm_output/UnDistortionImage.jpg
  4. 交互标定逆透视：四条线定义地平面几何约束，三自由度调节目标 ROI
  5. 六矩阵导出 → matrix/
  6. 两套打表（各含正向/反向）→ lookup_table/undistort/、lookup_table/undistort_ipm/
  7. 批量测试 test_input/ → test_output/

坐标约定：全程 OpenCV 0-based 像素坐标。物理坐标单位 cm，x 向右、y 向下（朝向车辆），
标定矩形中心为原点；BirdView 图正上方为车辆前进方向。

单应分解：H = T(anchor) @ S(scale) @ R(heading) @ H0
  H0    : 源图（去畸变）像素 -> 物理 cm 坐标，仅由四点与物理长宽决定
  R     : 绕原点旋转 heading_offset，物理标定区相对车辆前进坐标系的转角
  S     : cm -> BirdView 像素，scale 的单位是 px/cm
  T     : 平移到 anchor 指定的输出图位置
T/S/R 第三行均为 [0,0,1]，故 H[2,:] 恒等于 H0[2,:]，地平线只随四点变化。

查找表约定（交付给嵌入式 C 端）：
  reverse/ : 对每个输出像素，给出源图（原始畸变图）的采样坐标
  forward/ : 对每个源图像素，给出它在输出图中的落点
  两套表的无效点统一写 -1（映射到无穷远、落在地平线另一侧、或超出图像范围）。

命令行（在工程根目录下执行）：
  python src/neuq_vision_calib.py --list                 查看各目录现状
  python src/neuq_vision_calib.py --import-dir <混合目录> 按"能否检出棋盘"拆分素材入库
  python src/neuq_vision_calib.py --stage calib          只跑相机标定
  python src/neuq_vision_calib.py                        跑全流程（交互标定逆透视）
  python src/neuq_vision_calib.py --stage tables --quad ...
                                                         无 GUI 跑完整链路
  python src/webui/server.py                             打开浏览器控制台（推荐）
不带任何参数时的行为与改造前一致：从 calib_input/ 标定，
用 ipm_input/ 的原图交互标定。
"""

import argparse
import contextlib
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from neuq_core import config as _config
from neuq_core.calibration import (  # noqa: F401
    SUBPIX_CRITERIA,
    _calibrate_once,
    detect_chessboard,
    detect_chessboard_partial,
    report_reprojection_error,
)
from neuq_core.config import (  # noqa: F401
    BACKUP_SUFFIX,
    DEFAULT_BOARD,
    IMAGE_SUFFIXES,
    INCOMPLETE_SUBDIR,
    MATERIAL_PENDING_KEY,
    PROJECT_JSON_NAME,
    SCHEMA_VERSION,
    STAGING_SUFFIX,
    TOOL_VERSION,
    CheckerboardSpec,
    MaterialImportTransaction,
    check_schema,
    clear_material_board,
    in_material_library,
    list_images,
    load_project_config,
    material_basis_unknown_reason,
    material_board,
    material_has_images,
    material_import_pending,
    material_stage_pairs,
    material_stale_reason,
    material_transaction_residue,
    material_transaction_state,
    material_transaction_unsafe_reason,
    project_config_path,
    require_material_basis,
    require_no_material_residue,
    resolve_board,
    set_material_board,
    spec_from_meta,
    update_project_config,
    write_json_atomic,
    write_project_config_strict,
)
from neuq_core.fs_transaction import commit_dirs
from neuq_core.geometry import (  # noqa: F401
    DEGENERATE_EPS,
    DEN_EPS,
    LINE_PARALLEL_EPS,
    apply_homography,
    clip_polygon_halfplane,
    compute_homography,
    fit_fov_bottom_aligned,
    homography_denominator,
    horizon_sign,
    is_convex_quad,
    line_intersection,
    max_scale_for_fov,
    normalize_points_for_dlt,
    order_corners_tl_tr_bl_br,
    physical_rect,
    quad_area,
    rotation_matrix,
    scale_matrix,
    translation_matrix,
)
from neuq_core.io import (
    safe_imread,
    safe_imwrite,
    to_gray,
)
from neuq_core.lut import (  # noqa: F401
    MapPair,
    build_composite_reverse_map,
    distort_points,
    mask_out_of_range,
    pixel_grid,
    resample_map_pair,
    table_spaces,
    undistorted_grid,
)

# ---------------------------------------------------------------- 配置

# 当前工程的标定板规格。命令行 --board-squares / 网页上的规格卡片都会改它，
# 之后素材导入、在线拍摄、相机标定、结果落盘全部读这一份，不再各留一套常量。
BOARD: CheckerboardSpec = DEFAULT_BOARD


def restore_board() -> CheckerboardSpec:
    """决定启动时该用哪块棋盘：project.json → calib.json → 默认。

    顺序不能反。project.json 是用户显式设过的；calib.json 只是"上次标定时用的"，
    作为迁移来源；都没有才退回仓库自带的 12x9/20mm。
    """
    cfg = load_project_config()
    saved = spec_from_meta(cfg.get('board'))
    if saved is not None:
        return saved
    from_calib = spec_from_meta(calib_board_meta())
    if from_calib is not None:
        # 只在 project.json 没有 board 段时补写一次，不去覆盖用户显式设过的规格。
        # 缺了这一步，calib.json 就只是个"临时 fallback"：用户点一次「备份并清空」
        # 把它删掉，规格便无声退回仓库默认的 12x9/20mm，而素材库还留在原地。
        if 'board' not in cfg:
            update_project_config(board=from_calib.to_dict())
            print(f'标定板规格已从 calib.json 迁移进 {PROJECT_JSON_NAME}: '
                  f'{from_calib.label}')
        return from_calib
    return DEFAULT_BOARD


def configure_board(spec: CheckerboardSpec,
                    persist: bool = False) -> CheckerboardSpec:
    """设置当前工程的标定板规格。

    persist=True 时写进 project.json —— 用户在界面/命令行上显式设定的场合才用它；
    启动时的自动恢复不要写回去，否则会把用户的设置覆盖成默认值。
    """
    global BOARD
    changed = spec != BOARD
    BOARD = spec
    _config.bind_board(spec)
    if persist:
        update_project_config(board=spec.to_dict())
    if changed:
        warning = material_stale_reason(spec)
        if warning:
            print(f'注意: {warning}')
    return BOARD


# 在线拍摄：置 True 时打开摄像头，空格存图到 calib_input/，回车结束采集。
CAPTURE_ONLINE = False
CAMERA_INDEX = 0
# (W, H)；None 表示不设置，用摄像头原生分辨率。
# 原值 (320, 240) 是 160x120 时代的遗留配置，若摄像头实际是 1280x720 而强行设成
# 320x240，采集到的照片会与标定/逆透视链路的分辨率全部对不上。
CAPTURE_SIZE = None

# 去畸变输出视角。None 等价 MATLAB 的 'OutputView','same'（Knew = K）；
# 传 0.0~1.0 则用 getOptimalNewCameraMatrix，0 裁掉全部黑边、1 保留全部像素。
UNDIST_ALPHA = None

# 地面逆透视标定矩形的真实物理尺寸（cm）。运行时可交互覆盖。
# 注意这与"相机标定板规格"（棋盘格内角点数 + 方格边长，见 CheckerboardSpec）
# 是两件不同的事：这里量的是地面上那个矩形的外框边长。
# 网页、命令行、API 三个入口共用这一组 fallback —— 各自写一份默认值的话，
# 同一次标定从不同入口进去会得到不同的 px/cm。
PHYS_W_CM = 45.0
PHYS_H_CM = 45.0

# 目标 ROI 三自由度初值
ANCHOR_X = 0.5      # 归一化，ROI 中心在输出图的横向位置
ANCHOR_Y = 0.708    # 归一化，沿用原 160x120 方案中下部的布局比例
HEADING_DEG = 0.0   # 物理标定区相对车辆前进坐标系的转角，正值顺时针

# 有效视野的前向距离上限（cm）。地平线附近像素映射到无穷远，必须截断，
# 否则"不裁切完整视野"会把最大 scale 逼到 0。
MAX_RANGE_CM = 300.0
MAX_LATERAL_CM = 300.0

DISPLAY_SCALE = 2.0     # 交互窗口放大倍数，仅影响显示与拾取
PICK_RADIUS = 12        # 端点拾取半径（显示坐标下的像素）

MIN_QUAD_AREA_PX = 10.0     # 交互标定里判四条线退化重合的面积阈值（业务阈值，不属于 quad_area）
INIT_MARGIN_RATIO = 0.20    # 四条线初始位置距图像边缘的比例
INIT_SCALE_RATIO = 0.8      # 初始 scale 取 max_scale 的比例，留出裕量
MIN_SCALE = 0.05            # scale 下限（px/cm），防止退化为 0

# trackbar 只支持整数刻度，以下是整数刻度到物理量的换算
ANCHOR_TICKS = 1000            # anchor = pos / 1000，覆盖 0.000~1.000
HEADING_TICKS = 3600           # heading_deg = pos / 10 - 180，覆盖 -180.0~+180.0
SCALE_TICKS = 2000             # scale = pos / SCALE_TICKS_PER_UNIT
SCALE_TICKS_PER_UNIT = 20.0    # 即 0.05 px/cm 一档，量程 0.05~100 px/cm

WINDOW_PREFIX = 'NEUQ'


def default_project_root() -> Path:
    """工程根目录 —— 代码、静态资源与用户数据共同的安装位置。

    打包成 exe 之后不能再拿 `__file__` 当准绳：PyInstaller 会把脚本解包到一个临时
    目录，`__file__` 指向那里，于是所有输入输出都会跑到 Temp 下去，用户双击 exe
    之后会发现"素材放进去了但程序说没有"。冻结状态下要取 exe 自己所在的目录。

    源码方式运行时本文件位于 <工程根>/src/ 下，所以要再上一级才是工程根。

    目录约定：各工作目录（calib_input/、matrix/、lookup_table/ …）直接放在工程根，
    与改造前一致；只有 data/ 是个例外，它只装两类"不属于流水线产物"的东西——
    导入前的原始素材（data/import/）和备份压缩包（data/backups/）。
    """
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


SCRIPT_DIR = default_project_root()
# 原始素材与备份归档单独收在 data/ 下，其余产物目录平铺在工程根
DIR_IMPORT = SCRIPT_DIR / 'data' / 'import'
DIR_BACKUP = SCRIPT_DIR / 'data' / 'backups'
DIR_CALIB_IN = SCRIPT_DIR / 'calib_input'
DIR_CALIB_PREVIEW = SCRIPT_DIR / 'calib_preview'
DIR_CALIB_DATA = SCRIPT_DIR / 'calib_data'
DIR_IPM_IN = SCRIPT_DIR / 'ipm_input'
DIR_IPM_OUT = SCRIPT_DIR / 'ipm_output'
DIR_MATRIX = SCRIPT_DIR / 'matrix'
DIR_TABLE = SCRIPT_DIR / 'lookup_table'
DIR_TEST_IN = SCRIPT_DIR / 'test_input'
DIR_TEST_OUT = SCRIPT_DIR / 'test_output'

CALIB_JSON = DIR_CALIB_DATA / 'calib.json'
IPM_SOURCE = DIR_IPM_IN / 'UnInverseImage.jpg'
UNDIST_RESULT = DIR_IPM_OUT / 'UnDistortionImage.jpg'
IPM_RESULT = DIR_IPM_OUT / 'UnDistortionInverseImage.jpg'
MATRIX_JSON = DIR_MATRIX / 'matrices.json'

# 模块导入期先同步一次镜像：不调 configure_paths() / configure_board() 的场合，
# neuq_core.config 里那几个按裸全局读路径与规格的函数也要看到正确的值。
_config.bind_paths(root=SCRIPT_DIR, calib_in=DIR_CALIB_IN, ipm_in=DIR_IPM_IN)
_config.bind_board(BOARD)

# 运行期可被命令行覆盖的量，默认 None 表示沿用上面的模块级配置。
MAX_REPROJ_ERR: Optional[float] = None   # 标定迭代剔除的误差上限（px），None 不剔除

# 查找表导出。默认值保证与历史产物完全一致（1280x720 的逗号分隔文本）。
# 全尺寸文本表在 720p 下约 55 MB，单片机放不下，所以另给了定点二进制与 C 头文件两条路。
TABLE_FORMAT = 'txt'                      # txt | bin | c
TABLE_FIXED_POINT = 4                     # bin/c 的定点小数位数，Q4 即 1/16 px
TABLE_SIZE: Optional[Tuple[int, int]] = None   # (W, H)，None 表示与图像同尺寸
# int16 最小值。合法坐标恒为非负，用它与任何真实坐标都不会撞。
BIN_SENTINEL = -32768


def resolve_user_path(value, fallback_dir: Path) -> Path:
    """把命令行给的文件路径解析成真实路径。

    只写文件名（如 floor.jpg）时按 fallback_dir 查找，符合"照片就在那个目录里"的
    直觉；写相对路径时先看当前工作目录，再退回 fallback_dir。命令行给的是绝对路径
    时原样采用，不做存在性检查——不存在要让下游报出明确的错误，而不是在这里静默改道。
    """
    p = Path(value).expanduser()
    if p.is_absolute() or p.is_file():
        return p.resolve()
    candidate = fallback_dir / p
    return candidate.resolve() if candidate.is_file() else p.resolve()


def resolve_import_dir(value) -> Path:
    """把用户给的素材目录解析成真实目录。

    只认一个基准不够用：前端"选择文件夹"只能拿到文件夹名，命令行里又常常
    随手敲一个相对路径，而素材可能放在工程根，也可能是整理工程时收进
    data/import/ 的那批原始素材。这里按可能性依次试探，全都找不到时把试过
    的路径列全——比干巴巴一句"不是目录"有用得多。
    """
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        if raw.is_dir():
            return raw.resolve()
        raise SystemExit(f'--import-dir 不是目录: {raw}')

    tried: List[Path] = []
    for base in (Path.cwd(), SCRIPT_DIR, DIR_IMPORT):
        p = (base / raw).resolve()
        if p in tried:
            continue
        tried.append(p)
        if p.is_dir():
            return p

    raise SystemExit(
        f'找不到素材目录 {raw}。已依次尝试：\n  '
        + '\n  '.join(str(p) for p in tried)
        + '\n改用绝对路径，或把素材目录放进 data/import/ 下再填文件夹名。')


def configure_paths(root: Optional[Path] = None,
                    calib_dir: Optional[Path] = None,
                    ipm_dir: Optional[Path] = None,
                    test_dir: Optional[Path] = None,
                    ipm_source: Optional[Path] = None) -> None:
    """重绑定各目录常量，供命令行覆盖默认布局。

    这些常量在模块级被各函数直接引用，这里用 global 重写而不是层层传参：
    改动面最小，且所有调用点都不必知道路径是从哪来的。
    """
    global SCRIPT_DIR, DIR_IMPORT, DIR_BACKUP
    global DIR_CALIB_IN, DIR_CALIB_PREVIEW, DIR_CALIB_DATA
    global DIR_IPM_IN, DIR_IPM_OUT, DIR_MATRIX, DIR_TABLE, DIR_TEST_IN, DIR_TEST_OUT
    global CALIB_JSON, IPM_SOURCE, UNDIST_RESULT, IPM_RESULT, MATRIX_JSON

    if root is not None:
        SCRIPT_DIR = Path(root).expanduser().resolve()
    DIR_IMPORT = SCRIPT_DIR / 'data' / 'import'
    DIR_BACKUP = SCRIPT_DIR / 'data' / 'backups'

    DIR_CALIB_IN = (Path(calib_dir).expanduser().resolve() if calib_dir
                    else SCRIPT_DIR / 'calib_input')
    DIR_IPM_IN = (Path(ipm_dir).expanduser().resolve() if ipm_dir
                  else SCRIPT_DIR / 'ipm_input')
    DIR_TEST_IN = (Path(test_dir).expanduser().resolve() if test_dir
                   else SCRIPT_DIR / 'test_input')
    DIR_CALIB_PREVIEW = SCRIPT_DIR / 'calib_preview'
    DIR_CALIB_DATA = SCRIPT_DIR / 'calib_data'
    DIR_IPM_OUT = SCRIPT_DIR / 'ipm_output'
    DIR_MATRIX = SCRIPT_DIR / 'matrix'
    DIR_TABLE = SCRIPT_DIR / 'lookup_table'
    DIR_TEST_OUT = SCRIPT_DIR / 'test_output'

    CALIB_JSON = DIR_CALIB_DATA / 'calib.json'
    IPM_SOURCE = (resolve_user_path(ipm_source, DIR_IPM_IN) if ipm_source
                  else DIR_IPM_IN / 'UnInverseImage.jpg')
    UNDIST_RESULT = DIR_IPM_OUT / 'UnDistortionImage.jpg'
    IPM_RESULT = DIR_IPM_OUT / 'UnDistortionInverseImage.jpg'
    MATRIX_JSON = DIR_MATRIX / 'matrices.json'

    _config.bind_paths(root=SCRIPT_DIR, calib_in=DIR_CALIB_IN, ipm_in=DIR_IPM_IN)

    # 棋盘规格属于项目级配置，随工程根一起重新解析。
    # 不写回盘：这是"读配置"，不是"用户改了设置"。
    configure_board(restore_board())


# ---------------------------------------------------------------- 1. 相机标定



def next_capture_path() -> Path:
    """给在线拍摄分配不会覆盖已有文件的路径。

    不能用"目录里的图片数量"当编号：目录里若只剩 calib_005.jpg，
    计数为 1 时会生成 calib_001.jpg，再拍到第 5 张就把已有照片覆盖掉。
    """
    used = set()
    for p in DIR_CALIB_IN.glob('calib_*.jpg'):
        stem = p.stem[len('calib_'):]
        if stem.isdigit():
            used.add(int(stem))
    index = max(used) + 1 if used else 1
    return DIR_CALIB_IN / f'calib_{index:03d}.jpg'


def capture_calibration_images() -> None:
    """在线拍摄标定照片，实时叠加角点检测结果，空格存盘、回车结束。"""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f'无法打开摄像头 index={CAMERA_INDEX}')
    if CAPTURE_SIZE is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])

    DIR_CALIB_IN.mkdir(parents=True, exist_ok=True)
    saved = 0
    win = WINDOW_PREFIX + '_capture'
    print('在线采集：空格保存当前帧，回车结束采集。绿色角点表示这一帧可用。')

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('读取摄像头帧失败，结束采集。')
                break

            found, corners = detect_chessboard(to_gray(frame), fast=True)
            view = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(view, BOARD.corners, corners, found)
            tip = f'detected={found}  saved={saved}  [space]save [enter]done'
            # 把当前规格显示出来：换了棋盘却忘了改参数时，画面会一直 detected=False，
            # 用户很容易去怀疑相机或代码，其实只是规格没切过来。
            cv2.putText(view, f'board {BOARD.squares_x}x{BOARD.squares_y} squares',
                        (8, 34), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0), 1)
            cv2.putText(view, tip, (8, 18), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (0, 255, 0) if found else (0, 0, 255), 1)
            cv2.imshow(win, view)

            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):
                break
            if key == ord(' '):
                if not found:
                    print('当前帧未检出完整棋盘，未保存。')
                    continue
                path = next_capture_path()
                safe_imwrite(path, frame)
                saved += 1
                print('已保存:', path.name)
    finally:
        cap.release()
        cv2.destroyWindow(win)


def fit_camera(obj_points, img_points, used, img_size):
    """标定，并在 MAX_REPROJ_ERR 给定时迭代剔除高误差视图。

    棋盘照片是手持拍的，个别帧的运动模糊会把整体误差抬高。剔除必须逐轮做：
    每轮用当前参数算各视图 RMS，丢掉超限的再重标定。一次性剔除会误杀那些
    仅仅因为初始参数还不准才显得误差大的好帧。

    返回值末尾追加两个量：
      dropped    — [(name, error), ...]，按剔除顺序记录每一轮被丢掉的照片及其误差
      first_pass — (names, errors)，**第一轮、剔除之前**全体视图的误差。

    前端要做一个"改阈值就能预览会删多少张"的联动，只有 dropped 不够用：被剔的照片
    在最终一轮里根本不存在，拿不到误差。所以额外把第一轮的全体误差带出去，
    任意阈值下都能精确数出超限张数。
    """
    dropped: List[Tuple[str, float]] = []
    first_pass: Optional[Tuple[List[str], np.ndarray]] = None
    while True:
        rms, K, D, rvecs, tvecs, std_int, per_view = _calibrate_once(
            obj_points, img_points, img_size)
        if first_pass is None:
            first_pass = ([p.name for p in used], np.asarray(per_view, dtype=np.float64))
        if MAX_REPROJ_ERR is None or per_view.size == 0:
            break
        keep = [i for i, e in enumerate(per_view) if e <= MAX_REPROJ_ERR]
        if len(keep) < 3 or len(keep) == len(obj_points):
            break
        drop_set = set(range(len(obj_points))) - set(keep)
        names = [used[i].name for i in sorted(drop_set)]
        # 记录这一轮被剔的照片及其误差，供前端画图用
        for i in sorted(drop_set):
            dropped.append((used[i].name, float(per_view[i])))
        print(f'  剔除 {len(names)} 张误差超过 {MAX_REPROJ_ERR:.2f} px 的视图后重新标定: '
              + ', '.join(names[:6]) + (' ...' if len(names) > 6 else ''))
        obj_points = [obj_points[i] for i in keep]
        img_points = [img_points[i] for i in keep]
        used = [used[i] for i in keep]
    return (rms, K, D, rvecs, tvecs, per_view, std_int,
            obj_points, img_points, used, dropped, first_pass)


def calibrate_camera(board: Optional[CheckerboardSpec] = None
                     ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], dict]:
    """棋盘标定，返回 (K, D, (W, H), fit_result)，并把每张图的重投影误差与去畸变预览一并输出。

    board 不给时用当前工程的规格（core.BOARD）；显式传入便于测试与将来拆分。

    素材库是按另一块棋盘分拣的时候不允许开标：calib_input/ 里那批图是按旧规格的
    判定标准挑进来的，用新规格逐张重检会悄悄剔掉一大半，剩下两三张也标得出参数，
    只是精度和 provenance 都对不上。要换板子，就得按新规格重新导入。
    """
    spec = BOARD if board is None else board
    require_material_basis('相机标定', block_unknown=False)
    files = list_images(DIR_CALIB_IN)
    if len(files) < 3:
        raise SystemExit(f'{DIR_CALIB_IN} 中标定图不足（当前 {len(files)} 张），'
                         '至少需要 3 张，建议 15 张以上。')

    cols, rows = spec.corners
    # OpenCV 的 calibrateCamera 要求 objectPoints 为 Point3f，必须是 float32。
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= spec.square_size_mm

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    used: List[Path] = []
    failed: List[str] = []

    # 先统计分辨率分布，取众数作为标定分辨率。
    # 不能拿"第一张图"的分辨率当准绳：目录里只要混进一张旧分辨率的图（例如历史遗留的
    # 160x120），而它按文件名恰好排在最前，后面所有正常图片都会被判成"分辨率不一致"
    # 而全部剔除，最后以 0 张有效收场，报错信息还完全指向错误的方向。
    size_votes: Dict[Tuple[int, int], int] = {}
    for path in files:
        img = safe_imread(path)
        if img is None:
            failed.append(f'{path.name}（无法读取）')
            continue
        key = (img.shape[1], img.shape[0])
        size_votes[key] = size_votes.get(key, 0) + 1
    if not size_votes:
        raise SystemExit(f'{DIR_CALIB_IN} 中没有任何可读图片。')

    # 票数相同时取面积大的，避免 2 张小图与 2 张大图打平后选中了没用的那张。
    img_size: Tuple[int, int] = max(size_votes, key=lambda k: (size_votes[k], k[0] * k[1]))
    if len(size_votes) > 1:
        others = ', '.join(f'{w}x{h}（{n} 张）' for (w, h), n
                           in sorted(size_votes.items(), key=lambda kv: -kv[1])
                           if (w, h) != img_size)
        print(f'注意: calib_input/ 存在多种分辨率，按众数取 '
              f'{img_size[0]}x{img_size[1]}，以下将被剔除: {others}')

    for path in files:
        img = safe_imread(path)
        if img is None:
            continue
        gray = to_gray(img)
        if (gray.shape[1], gray.shape[0]) != img_size:
            failed.append(f'{path.name}（分辨率 {gray.shape[1]}x{gray.shape[0]} '
                          f'与 {img_size[0]}x{img_size[1]} 不一致）')
            continue

        found, corners = detect_chessboard(gray, board=spec)
        if not found or corners is None:
            failed.append(f'{path.name}（未检出完整 {spec.corners[0]}x'
                          f'{spec.corners[1]} 内角点，当前规格 {spec.label}）')
            continue
        obj_points.append(objp.copy())
        img_points.append(np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2))
        used.append(path)

    if failed:
        print(f'\n以下 {len(failed)} 张未参与标定:')
        for item in failed:
            print('  ', item)
        if len(failed) > len(files) // 2:
            px_per_square = min(img_size[0] / (spec.corners[0] + 1),
                                img_size[1] / (spec.corners[1] + 1))
            print(f'  过半图片检出失败。当前 {img_size[0]}x{img_size[1]} 下 '
                  f'{spec.corners[0]}x{spec.corners[1]} 内角点，'
                  f'每格满屏时也只有约 {px_per_square:.0f} px；'
                  '低于 20 px 就很难稳定检出，建议换更粗的棋盘（更少角点、更大方格）重拍。')

    if len(obj_points) < 3:
        raise SystemExit(f'成功检出棋盘的图片只有 {len(obj_points)} 张，不足以标定。')

    (rms, K, D, rvecs, tvecs, per_view, std_int,
     obj_points, img_points, used, dropped, first_pass) = fit_camera(
        obj_points, img_points, used, img_size)

    first_names, first_errors = first_pass if first_pass else ([], np.array([]))
    fit_result = {
        'rms': float(rms),
        'per_view': per_view.tolist() if per_view.size else [],
        'std_int': std_int.tolist() if std_int.size else [],
        'used_names': [p.name for p in used],
        'dropped': [{'name': n, 'error': float(e)} for n, e in dropped],
        'threshold': float(MAX_REPROJ_ERR) if MAX_REPROJ_ERR is not None else None,
        # 剔除前的全体视图误差，供前端做"改阈值即时预览会删多少张"
        'all_names': list(first_names),
        'all_errors': np.asarray(first_errors, dtype=np.float64).tolist(),
    }

    print(f'\n标定完成：{len(used)}/{len(files)} 张有效，分辨率 {img_size[0]}x{img_size[1]}')
    print(f'整体重投影误差 RMS = {rms:.4f} px')
    print('（注意：OpenCV 报的是 RMS，MATLAB cameraCalibrator 报的是平均欧氏距离，前者数值偏大）')

    mean_err = report_reprojection_error(obj_points, img_points, rvecs, tvecs, K, D)
    print(f'平均欧氏重投影误差 = {mean_err:.4f} px（这个才和 MATLAB 的口径一致）')

    if per_view.size:
        order = np.argsort(-per_view)
        print('每张图误差（RMS，由大到小前 5 张）:')
        for i in order[:5]:
            print(f'  {used[i].name:<28} {per_view[i]:.4f} px')
    if std_int.size >= 4:
        print(f'内参标准差: fx={std_int[0]:.3f} fy={std_int[1]:.3f} '
              f'cx={std_int[2]:.3f} cy={std_int[3]:.3f}')

    export_undistort_previews(used, K, D, img_size)

    # 返回值从 3 元组扩到 4 元组：前三个保留旧签名给命令行路径用，
    # 第四个 fit_result 是给 Web 端画柱状图用的逐帧数据。
    return K, D, img_size, fit_result


def export_undistort_previews(files: List[Path], K: np.ndarray, D: np.ndarray,
                              img_size: Tuple[int, int]) -> None:
    """把参与标定的每张图去畸变后存盘，等价 cameraCalibrator 的 Show Undistorted。"""
    DIR_CALIB_PREVIEW.mkdir(parents=True, exist_ok=True)
    Knew = resolve_new_camera_matrix(K, D, img_size)
    for path in files:
        img = safe_imread(path)
        if img is None:
            continue
        undist = cv2.undistort(img, K, D, None, Knew)
        safe_imwrite(DIR_CALIB_PREVIEW / f'{path.stem}_undist.jpg', undist)
    print(f'标定图去畸变效果已写入: {DIR_CALIB_PREVIEW}')


def resolve_new_camera_matrix(K: np.ndarray, D: np.ndarray,
                              img_size: Tuple[int, int]) -> np.ndarray:
    """按 UNDIST_ALPHA 决定去畸变输出的相机矩阵 Knew。"""
    if UNDIST_ALPHA is None:
        return K.copy()
    Knew, _roi = cv2.getOptimalNewCameraMatrix(K, D, img_size, float(UNDIST_ALPHA), img_size)
    return np.asarray(Knew, dtype=np.float64)


def save_calibration(K: np.ndarray, D: np.ndarray, img_size: Tuple[int, int],
                     board: Optional[CheckerboardSpec] = None) -> None:
    """把内参、畸变系数与标定分辨率写入 calib_data/calib.json。

    board 显式传入时以它为准：provenance 必须跟着"实际用于标定的那块板"走，
    不能只读当时的全局状态——否则 calibrate_camera(board=A) 之后再改全局，
    落盘的就会是 A 的参数配 B 的规格。
    """
    spec = BOARD if board is None else board
    DIR_CALIB_DATA.mkdir(parents=True, exist_ok=True)
    payload = {
        'image_width': int(img_size[0]),
        'image_height': int(img_size[1]),
        'camera_matrix': K.tolist(),
        'dist_coeffs': D.tolist(),
        'schema_version': SCHEMA_VERSION,
        'tool_version': TOOL_VERSION,
        'board': spec.to_dict(),
        # 老字段保留一轮：外部脚本或旧版网页可能还在读这两个键
        'chessboard_corners': list(spec.corners),
        'square_size_mm': spec.square_size_mm,
    }
    write_json_atomic(CALIB_JSON, payload)
    print('标定数据已保存:', CALIB_JSON)


def calib_board_meta() -> Optional[dict]:
    """读 calib.json 里记录的标定板规格，供界面回显。

    没有它的话，过一阵子看到一份 calib.json 就只能猜"这是哪块棋盘算出来的"。
    老文件只存了内角点，这里顺手换算成方格数，界面不必分两种格式显示。

    只读一个字段也要先过 schema 闸：restore_board() 会把这里读到的 board 迁进
    project.json，绕过闸门等于让更新版本的 calib.json 反向污染旧工程配置——
    而同一份文件走 load_calibration() 是会被拒绝的，两条路径必须一致。
    """
    if not CALIB_JSON.is_file():
        return None
    try:
        data = json.loads(CALIB_JSON.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    check_schema(data, str(CALIB_JSON))
    board = data.get('board')
    if board:
        return board
    corners = data.get('chessboard_corners')
    if isinstance(corners, (list, tuple)) and len(corners) == 2:
        return {
            'type': 'checkerboard',
            'squares_x': int(corners[0]) + 1,
            'squares_y': int(corners[1]) + 1,
            'corners_x': int(corners[0]),
            'corners_y': int(corners[1]),
            'square_size_mm': data.get('square_size_mm'),
        }
    return None


def load_calibration() -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
    """载入 calib.json，文件不存在时返回 None。

    这个文件允许用户手工从 MATLAB 导出，所以不能假定字段齐全：缺键或形状不对时
    给出明确提示而不是抛 KeyError/ValueError 崩栈。
    """
    if not CALIB_JSON.is_file():
        legacy = list(DIR_CALIB_DATA.glob('*.mat')) + list(DIR_CALIB_DATA.glob('*/*.mat'))
        if legacy:
            print(f'提示: 只找到 MATLAB 标定文件 {legacy[0].name}，本工具无法解析 '
                  f'cameraParameters 对象。请重新标定，或从 MATLAB 导出 calib.json。')
        return None

    try:
        data = json.loads(CALIB_JSON.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise SystemExit(f'{CALIB_JSON} 不是合法 JSON: {exc}') from None
    if not isinstance(data, dict):
        raise SystemExit(f'{CALIB_JSON} 顶层应为 JSON 对象。')
    check_schema(data, str(CALIB_JSON))

    required = ('camera_matrix', 'dist_coeffs', 'image_width', 'image_height')
    missing = [k for k in required if k not in data]
    if missing:
        raise SystemExit(
            f'{CALIB_JSON} 缺少字段: {", ".join(missing)}。\n'
            '期望格式: camera_matrix 为 3x3 嵌套数组、dist_coeffs 为一维数组、'
            'image_width/image_height 为正整数（可参照本工具标定后生成的文件）。')

    try:
        K = np.asarray(data['camera_matrix'], dtype=np.float64).reshape(3, 3)
        D = np.asarray(data['dist_coeffs'], dtype=np.float64).ravel()
        img_size = (int(data['image_width']), int(data['image_height']))
    except (ValueError, TypeError) as exc:
        raise SystemExit(f'{CALIB_JSON} 字段格式不正确: {exc}') from None

    if D.size == 0:
        raise SystemExit(f'{CALIB_JSON} 的 dist_coeffs 为空。')
    if img_size[0] <= 0 or img_size[1] <= 0:
        raise SystemExit(f'{CALIB_JSON} 的分辨率非法: {img_size[0]}x{img_size[1]}')

    print(f'已载入标定数据: {CALIB_JSON}（{img_size[0]}x{img_size[1]}）')
    return K, D, img_size


# ---------------------------------------------------------------- 单应与几何

def valid_fov_polygon(H0: np.ndarray, R: np.ndarray, src_size: Tuple[int, int],
                      sign: float) -> np.ndarray:
    """源图有效视野在旋转后物理坐标系下的多边形。

    先在源图裁掉地平线另一侧（那里映射到无穷远），再映射到 cm 坐标，
    最后按前向/横向距离上限截断。sign 由 horizon_sign 给出。
    """
    w, h = src_size
    poly = np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
                    dtype=np.float64)

    a, b, c = float(H0[2, 0]), float(H0[2, 1]), float(H0[2, 2])
    poly = clip_polygon_halfplane(poly, sign * a, sign * b, sign * c - DEN_EPS)
    if poly.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float64)

    poly_cm = apply_homography(R @ H0, poly)
    if not np.isfinite(poly_cm).all():
        return np.zeros((0, 2), dtype=np.float64)

    # 车辆前进为 -y，故前向上限是 y >= -MAX_RANGE_CM。
    poly_cm = clip_polygon_halfplane(poly_cm, 0.0, 1.0, MAX_RANGE_CM)
    poly_cm = clip_polygon_halfplane(poly_cm, 0.0, -1.0, MAX_RANGE_CM)
    poly_cm = clip_polygon_halfplane(poly_cm, 1.0, 0.0, MAX_LATERAL_CM)
    poly_cm = clip_polygon_halfplane(poly_cm, -1.0, 0.0, MAX_LATERAL_CM)
    return poly_cm


# ---------------------------------------------------------------- 4. 交互标定

class IpmCalibrator:
    """四条线定义地平面约束，三自由度实时调节目标 ROI。"""

    def __init__(self, img: np.ndarray, phys_w: float, phys_h: float):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.out_size = (self.w, self.h)
        self.phys_w = phys_w
        self.phys_h = phys_h
        self.phys_pts = physical_rect(phys_w, phys_h)

        self.display_scale = DISPLAY_SCALE
        self.raw_win = WINDOW_PREFIX + '_raw'
        self.preview_win = WINDOW_PREFIX + '_birdview'

        mx = int(round(self.w * INIT_MARGIN_RATIO))
        my = int(round(self.h * INIT_MARGIN_RATIO))
        self.line_points = np.array([
            [[mx, my], [self.w - mx, my]],
            [[mx, self.h - my], [self.w - mx, self.h - my]],
            [[mx, my], [mx, self.h - my]],
            [[self.w - mx, my], [self.w - mx, self.h - my]],
        ], dtype=np.float64)
        self.line_default = self.line_points.copy()
        self.line_colors = [(0, 255, 255), (255, 180, 0), (0, 255, 0), (255, 0, 255)]
        self.line_names = ['TOP', 'BOTTOM', 'LEFT', 'RIGHT']

        self.anchor_x = ANCHOR_X
        self.anchor_y = ANCHOR_Y
        self.heading = HEADING_DEG
        self.scale = 1.0
        self.scale_initialized = False

        self.drag: Optional[Tuple[int, int]] = None
        self.dirty = True
        self.trackbars_ready = False
        self.trackbar_warned = False

        self.H0: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray] = None
        self.birdview: Optional[np.ndarray] = None
        self.corners: Optional[np.ndarray] = None
        self.sign = 1.0
        self.max_scale = 0.0

    # ---- 几何

    def build_corners(self) -> Optional[np.ndarray]:
        """由四条线的交点得到 TL,TR,BL,BR 四角点，几何非法时返回 None。

        非法包含三种：任意两线平行取不到交点；四边形面积过小（四条线退化重合）；
        按 y/x 重排后的顺序与"上下左右交点"给出的身份不一致（说明用户把 TOP 拖到了
        BOTTOM 下方之类），或四边形自交/非凸。后两种若不拦住，H0 会得到镜像或扭曲解。
        """
        top, bottom, left, right = self.line_points
        tl = line_intersection(top[0], top[1], left[0], left[1])
        tr = line_intersection(top[0], top[1], right[0], right[1])
        bl = line_intersection(bottom[0], bottom[1], left[0], left[1])
        br = line_intersection(bottom[0], bottom[1], right[0], right[1])
        if tl is None or tr is None or bl is None or br is None:
            return None

        pts = np.array([tl, tr, bl, br], dtype=np.float64)
        if not np.isfinite(pts).all():
            return None
        if quad_area(pts) < MIN_QUAD_AREA_PX:
            return None
        if not np.allclose(order_corners_tl_tr_bl_br(pts), pts):
            return None
        if not is_convex_quad(pts):
            return None
        return pts

    def anchor_px(self) -> Tuple[float, float]:
        """anchor 的归一化坐标换算到输出图像素坐标。"""
        return self.anchor_x * (self.w - 1), self.anchor_y * (self.h - 1)

    def compose(self, H0: np.ndarray) -> np.ndarray:
        """按当前三自由度组合出完整单应 H = T @ S @ R @ H0。"""
        ax, ay = self.anchor_px()
        return (translation_matrix(ax, ay) @ scale_matrix(self.scale)
                @ rotation_matrix(self.heading) @ H0)

    def recompute(self) -> None:
        """重算 H0、有效视野、max_scale、H 与 BirdView 预览。"""
        self.corners = self.build_corners()
        if self.corners is None:
            self.H0 = self.H = self.birdview = None
            self.max_scale = 0.0
            return

        try:
            self.H0 = compute_homography(self.corners, self.phys_pts)
        except ValueError:
            self.H0 = self.H = self.birdview = None
            self.max_scale = 0.0
            return

        self.sign = horizon_sign(self.H0, self.corners)
        poly = valid_fov_polygon(self.H0, rotation_matrix(self.heading),
                                 (self.w, self.h), self.sign)
        self.max_scale = max_scale_for_fov(poly, self.anchor_px(), self.out_size)

        if not self.scale_initialized:
            self.scale = (max(MIN_SCALE, self.max_scale * INIT_SCALE_RATIO)
                          if self.max_scale > 0 else 1.0)
            self.scale_initialized = True
            self.sync_scale_trackbar()

        self.H = self.compose(self.H0)
        self.birdview = cv2.warpPerspective(self.img, self.H, self.out_size,
                                            flags=cv2.INTER_LINEAR)

    def is_over_crop(self) -> bool:
        """当前 scale 是否已经超过不裁切视野的上限。"""
        return self.max_scale > 0 and self.scale > self.max_scale

    # ---- 交互

    def pick_endpoint(self, dx: int, dy: int) -> Optional[Tuple[int, int]]:
        """把显示坐标下的点击换算到原图，返回命中的 (线序号, 端点序号)。"""
        px, py = dx / self.display_scale, dy / self.display_scale
        threshold = PICK_RADIUS / self.display_scale
        best = None
        best_dist = float('inf')
        for li in range(4):
            for ei in range(2):
                ex, ey = self.line_points[li, ei]
                d = float(np.hypot(ex - px, ey - py))
                if d < best_dist and d <= threshold:
                    best, best_dist = (li, ei), d
        return best

    def on_mouse(self, event, x, y, flags, _param) -> None:
        """鼠标回调：拖动端点。只置脏标记，实际重算交给主循环。"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = self.pick_endpoint(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self.drag is not None:
                li, ei = self.drag
                ox = min(max(x / self.display_scale, 0.0), self.w - 1.0)
                oy = min(max(y / self.display_scale, 0.0), self.h - 1.0)
                self.line_points[li, ei] = (ox, oy)
                self.dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None

    def create_trackbars(self) -> None:
        """创建位置、转角、尺度三组 trackbar。"""
        cv2.createTrackbar('anchor_x x1000', self.raw_win,
                           int(round(self.anchor_x * ANCHOR_TICKS)),
                           ANCHOR_TICKS, self.on_anchor_x)
        cv2.createTrackbar('anchor_y x1000', self.raw_win,
                           int(round(self.anchor_y * ANCHOR_TICKS)),
                           ANCHOR_TICKS, self.on_anchor_y)
        cv2.createTrackbar('heading +180 x10', self.raw_win,
                           int(round((self.heading + 180.0) * 10)),
                           HEADING_TICKS, self.on_heading)
        cv2.createTrackbar('scale x20', self.raw_win,
                           int(round(self.scale * SCALE_TICKS_PER_UNIT)),
                           SCALE_TICKS, self.on_scale)
        self.trackbars_ready = True

    def sync_scale_trackbar(self) -> None:
        """把自动算出的 scale 回写到 trackbar 位置。

        trackbar 尚未创建时（首次 recompute 早于 create_trackbars）直接跳过；
        其余 cv2 故障打印一次告警，不静默吞掉。
        """
        if not self.trackbars_ready:
            return
        pos = int(round(min(float(SCALE_TICKS),
                            max(1.0, self.scale * SCALE_TICKS_PER_UNIT))))
        try:
            cv2.setTrackbarPos('scale x20', self.raw_win, pos)
        except cv2.error as exc:
            if not self.trackbar_warned:
                self.trackbar_warned = True
                print('scale trackbar 同步失败，滑块位置可能与实际值不一致:', exc)

    def on_anchor_x(self, v: int) -> None:
        """anchor_x trackbar 回调。"""
        self.anchor_x = v / ANCHOR_TICKS
        self.dirty = True

    def on_anchor_y(self, v: int) -> None:
        """anchor_y trackbar 回调。"""
        self.anchor_y = v / ANCHOR_TICKS
        self.dirty = True

    def on_heading(self, v: int) -> None:
        """heading trackbar 回调。"""
        self.heading = v / 10.0 - 180.0
        self.dirty = True

    def on_scale(self, v: int) -> None:
        """scale trackbar 回调。"""
        self.scale = max(MIN_SCALE, v / SCALE_TICKS_PER_UNIT)
        self.dirty = True

    # ---- 绘制

    def render(self) -> None:
        """重算并刷新原图窗口与 BirdView 预览窗口。"""
        self.recompute()
        disp = cv2.resize(self.img, (0, 0), fx=self.display_scale, fy=self.display_scale,
                          interpolation=cv2.INTER_LINEAR)
        thickness = max(1, int(round(self.display_scale / 2)))

        for li in range(4):
            p1, p2 = self.line_points[li]
            d1 = (int(round(p1[0] * self.display_scale)), int(round(p1[1] * self.display_scale)))
            d2 = (int(round(p2[0] * self.display_scale)), int(round(p2[1] * self.display_scale)))
            cv2.line(disp, d1, d2, self.line_colors[li], thickness)
            cv2.circle(disp, d1, max(3, thickness + 2), (0, 0, 255), -1)
            cv2.circle(disp, d2, max(3, thickness + 2), (0, 0, 255), -1)
            cv2.putText(disp, self.line_names[li], (d1[0] + 6, d1[1] - 6),
                        cv2.FONT_HERSHEY_PLAIN, max(1.0, self.display_scale / 2),
                        self.line_colors[li], 1)

        if self.corners is not None:
            for i, p in enumerate(self.corners):
                d = (int(round(p[0] * self.display_scale)), int(round(p[1] * self.display_scale)))
                cv2.circle(disp, d, max(3, thickness + 3), (255, 255, 0), -1)
                cv2.putText(disp, f'C{i + 1}', (d[0] + 6, d[1] - 6), cv2.FONT_HERSHEY_PLAIN,
                            max(1.0, self.display_scale / 2), (255, 255, 0), 1)

        cv2.putText(disp, 'drag red endpoints | f fit-scale | r reset | q save&exit',
                    (8, 16), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 0), 1)
        cv2.imshow(self.raw_win, disp)
        self.render_preview()

    def render_preview(self) -> None:
        """刷新 BirdView 预览：叠加目标 ROI、anchor、前进方向与 scale 状态。"""
        if self.birdview is None or self.corners is None or self.H is None:
            blank = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            cv2.putText(blank, 'invalid quad (parallel/crossed/concave)', (8, 20),
                        cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 255), 1)
            cv2.imshow(self.preview_win, blank)
            return

        view = self.birdview.copy()
        if view.ndim == 2:
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)

        roi = apply_homography(self.H, self.corners)
        cv2.polylines(view, [roi[[0, 1, 3, 2]].astype(np.int32)], True, (0, 255, 0), 1)
        ax, ay = self.anchor_px()
        cv2.drawMarker(view, (int(round(ax)), int(round(ay))), (0, 128, 255),
                       cv2.MARKER_CROSS, 10, 1)
        cv2.arrowedLine(view, (self.w // 2, self.h - 6), (self.w // 2, self.h - 26),
                        (255, 255, 255), 1, tipLength=0.4)

        over = ' OVER-CROP' if self.is_over_crop() else ''
        cv2.putText(view, f'scale={self.scale:.2f} max={self.max_scale:.2f}{over}',
                    (6, 14), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    (0, 0, 255) if over else (0, 255, 0), 1)
        cv2.putText(view, f'anchor=({self.anchor_x:.3f},{self.anchor_y:.3f}) '
                          f'head={self.heading:.1f}deg',
                    (6, 28), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1)
        cv2.imshow(self.preview_win, view)

    # ---- 主循环

    def window_alive(self) -> bool:
        """原图窗口是否还存在（用户点标题栏 X 关闭后返回 False）。"""
        try:
            return cv2.getWindowProperty(self.raw_win, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def run(self) -> Optional[np.ndarray]:
        """打开交互窗口，返回退出时的单应 H；几何非法则返回 None。

        所有重算都集中在主循环里按脏标记触发，回调只改参数，避免在 GUI 回调中
        跑完整 warp；窗口与回调在 finally 中统一释放，异常路径也不残留窗口。
        """
        cv2.namedWindow(self.raw_win, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.preview_win, cv2.WINDOW_NORMAL)
        try:
            cv2.setMouseCallback(self.raw_win, self.on_mouse)
            self.create_trackbars()
            self.dirty = True
            self.render()
            cv2.resizeWindow(self.raw_win, int(self.w * self.display_scale),
                             int(self.h * self.display_scale))

            print(f'\n交互标定：物理标定矩形 {self.phys_w:g} x {self.phys_h:g} cm')
            print('  拖动红色端点调整四条线 -> 交点即地平面几何约束')
            print('  trackbar: anchor_x/anchor_y 位置, heading 转角, scale 物理->像素尺度')
            print('  f = scale 吸附到不裁切视野的最大值, r = 复位四条线, q = 保存退出')

            while True:
                key = cv2.waitKey(20) & 0xFF
                if not self.window_alive():
                    print('窗口已关闭，按未保存处理。')
                    return None

                if key == ord('q') or key == 27:
                    break
                elif key == ord('r'):
                    self.line_points[:] = self.line_default
                    self.dirty = True
                elif key == ord('f'):
                    if self.max_scale > 0:
                        self.scale = self.max_scale
                        self.sync_scale_trackbar()
                        self.dirty = True
                    else:
                        print('当前几何下无有效视野，无法吸附 scale。')
                elif key in (ord('+'), ord('=')):
                    self.display_scale = min(self.display_scale * 1.25, 20.0)
                    self.dirty = True
                elif key == ord('-'):
                    self.display_scale = max(self.display_scale / 1.25, 1.0)
                    self.dirty = True

                if self.dirty:
                    self.dirty = False
                    self.render()

            return self.H
        finally:
            # 窗口可能已经被用户关掉了，setMouseCallback 会抛 cv2.error；
            # 这里只关心"无论如何把窗口收干净"，直接抑制掉最清楚。
            with contextlib.suppress(cv2.error):
                cv2.setMouseCallback(self.raw_win, lambda *_a: None)
            cv2.destroyWindow(self.raw_win)
            cv2.destroyWindow(self.preview_win)


# ---------------------------------------------------------------- 5. 矩阵导出

def table_convention_text() -> str:
    """把当前生效的表格式约定写成一句话，随矩阵一起导出。"""
    base = ('reverse/ = 对每个输出像素给出源图（原始畸变图）采样坐标；'
            'forward/ = 对每个源图像素给出输出图落点。')
    if TABLE_FORMAT == 'txt':
        return (base + ' 无效点一律为 -1（映射到无穷远、位于地平线另一侧、或超出图像范围）。'
                'MapW.txt 为 x 方向，MapH.txt 为 y 方向，均为 0-based 像素坐标，%.2f 精度。')
    unit = 1 << TABLE_FIXED_POINT
    files = 'Map.h' if TABLE_FORMAT == 'c' else 'MapW.bin / MapH.bin'
    return (base + f' 无效点一律为 {BIN_SENTINEL}。'
            f'MapW / MapH 为 int16 定点，Q{TABLE_FIXED_POINT}，'
            f'实际像素坐标 = 表值 / {unit}.0f，0-based。'
            f'文件形式: {files}。')


def export_matrices(K: np.ndarray, D: np.ndarray, Knew: np.ndarray,
                    H: np.ndarray, extra: dict,
                    matrix_root: Optional[Path] = None) -> None:
    """导出六矩阵与畸变系数，同时写一份高精度文本便于粘贴。"""
    out_dir = Path(matrix_root) if matrix_root is not None else DIR_MATRIX
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema_version': SCHEMA_VERSION,
        'tool_version': TOOL_VERSION,
        # 这份矩阵是哪块棋盘标出来的 —— 以 calib.json 记录的为准，
        # 而不是当次的全局变量：provenance 要跟着标定结果走。
        'board': (spec_from_meta(calib_board_meta()) or BOARD).to_dict(),
        'note': (
            '去畸变的非线性部分由 dist_coeffs 承载，无法写成矩阵；'
            f'因此原图->BirdView 的复合变换只以查表形式给出，见 {DIR_TABLE.name}/undistort/ '
            f'与 {DIR_TABLE.name}/undistort_ipm/。'
        ),
        'table_convention': table_convention_text(),
        'table_format': TABLE_FORMAT,
        'table_fixed_point': (None if TABLE_FORMAT == 'txt' else TABLE_FIXED_POINT),
        'table_size_wh': list(TABLE_SIZE) if TABLE_SIZE else None,
        'K': K.tolist(),
        'K_inv': np.linalg.inv(K).tolist(),
        'Knew': Knew.tolist(),
        'Knew_inv': np.linalg.inv(Knew).tolist(),
        'H': H.tolist(),
        'H_inv': np.linalg.inv(H).tolist(),
        'dist_coeffs': D.tolist(),
    }
    payload.update(extra)
    (out_dir / 'matrices.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = []
    for name in ('K', 'K_inv', 'Knew', 'Knew_inv', 'H', 'H_inv'):
        m = np.asarray(payload[name], dtype=np.float64)
        body = ';\n    '.join(', '.join(f'{v:.17e}' for v in row) for row in m)
        lines.append(f'{name} = [\n    {body}\n];')
    d = np.asarray(D, dtype=np.float64).ravel()
    lines.append('D = [' + ', '.join(f'{v:.17e}' for v in d) + '];')
    (out_dir / 'matrices.txt').write_text('\n\n'.join(lines) + '\n', encoding='utf-8')
    print('六矩阵与畸变系数已保存:', out_dir / 'matrices.json')


def assert_invertible(name: str, m: np.ndarray) -> None:
    """导出前校验矩阵可逆，避免写表写到一半才抛 LinAlgError 留下半套产物。"""
    try:
        np.linalg.inv(m)
    except np.linalg.LinAlgError as exc:
        raise SystemExit(f'{name} 不可逆，无法导出: {exc}') from None


# ---------------------------------------------------------------- 6. 打表

def quantize_table(mat: np.ndarray) -> np.ndarray:
    """把浮点映射表量化成 int16 定点，无效点写 BIN_SENTINEL。

    定点位数受量程限制：Q4（1/16 px）在 int16 下最大表示 2047.9 px，够覆盖 1280 宽的源图；
    再高一位就只剩 1023.9 px，反而装不下。要更细的精度只能换 int32。
    溢出时报错而不是静默截断——被截断的坐标在车上是乱采样的鬼影，极难倒查。
    """
    m = np.asarray(mat, dtype=np.float64)
    scale = float(1 << TABLE_FIXED_POINT)
    limit = (1 << 15) - 1
    out = np.full(m.shape, BIN_SENTINEL, dtype=np.int16)

    valid = np.isfinite(m) & (m >= 0.0)
    if not valid.any():
        return out
    q = np.rint(m[valid] * scale)
    overflow = int(np.count_nonzero(q > limit))
    if overflow:
        raise SystemExit(
            f'打表溢出: {overflow} 个坐标超出 Q{TABLE_FIXED_POINT} 定点量程 '
            f'（{limit / scale:.1f} px）。请降低 --table-fixed-point，'
            '或缩小 --table-size。')
    out[valid] = q.astype(np.int16)
    return out


def write_table_txt(path: Path, mat: np.ndarray) -> None:
    """逗号分隔文本，非有限值写 -1。与历史产物格式完全一致。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.asarray(mat, dtype=np.float64)
    m = np.where(np.isfinite(m), m, -1.0)
    np.savetxt(path, m, fmt='%.2f', delimiter=', ')


def write_table_bin(path: Path, table: np.ndarray) -> None:
    """int16 小端裸二进制，无效点写 BIN_SENTINEL。

    入参必须是 quantize_table() 的输出，这里不再二次量化：落盘的字节要和
    batch_test、校验脚本消费的那一份严格同源。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(table, dtype='<i2').tofile(path)


def dequantize_table(q: np.ndarray,
                     fixed_point: Optional[int] = None) -> np.ndarray:
    """定点表 -> 像素坐标，无效哨兵还原成 -1。

    这正是 C 端的算法（表值 / 2^shift），所以它就是"交付内容的浮点视图"：
    拿它去跑批量测试，等于拿 C 端真正会用的坐标去跑。

    注意 fixed_point 默认值必须写成 None 再在函数内取 TABLE_FIXED_POINT：
    写成 `fixed_point=TABLE_FIXED_POINT` 会在**函数定义时**就把当时的值绑死，
    之后运行时改全局（--table-fixed-point、Web 上的格式选择）都不会生效。
    """
    fp = TABLE_FIXED_POINT if fixed_point is None else int(fixed_point)
    arr = np.asarray(q, dtype=np.float64) / float(1 << fp)
    arr[np.asarray(q) == BIN_SENTINEL] = -1.0
    return arr


def format_int16_array(values: np.ndarray, per_line: int = 12) -> str:
    """把 int16 数组排成 C 字面量，每行 per_line 个。"""
    flat = values.ravel()
    rows = ['    ' + ', '.join(str(int(v)) for v in flat[i:i + per_line]) + ','
            for i in range(0, flat.size, per_line)]
    return '\n'.join(rows)


def write_table_c(out_dir: Path, tag: str, desc: str,
                  qx: np.ndarray, qy: np.ndarray,
                  source_size: Tuple[int, int]) -> None:
    """写一个自包含的 C 头文件，含两张 int16 定点表。

    直接把维度、定点位数、无效哨兵和用法都写进注释与宏，C 端拿到就能用，
    不必再回头翻本工程的文档去猜表的排布和单位。

    **网格尺寸与源图分辨率必须分开写**：表被 --table-size 降采样后，网格是
    160x120，而表里的采样坐标仍然活在 1280x720 的原图上。只写一个模糊的
    W/H，读回来的人（包括本工程自己的 load_map_pair）就会把网格当成原图尺寸，
    拿缩到 160x120 的图去采 800 多的坐标。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    qx, qy = np.asarray(qx, dtype=np.int16), np.asarray(qy, dtype=np.int16)
    th, tw = qx.shape
    sw, sh = source_size
    safe = ''.join(c if c.isalnum() else '_' for c in tag).upper()
    guard = f'NEUQ_MAP_{safe}_H'
    unit = 1 << TABLE_FIXED_POINT
    sp = table_spaces(tag)
    # forward 表的取值不是"去哪张源图采样"，而是"落到目标图哪里"。
    # 注释不分方向地写"源图 W/H"，拿 forward 表的人一定会理解反。
    use_note = (f'index {sp["index_desc"]}\n'
                f' * value {sp["value_desc"]}\n'
                f' * 用法: int16_t v = {safe}_mapW[y * {safe}_GRID_W + x];\n'
                f' *       if (v == {safe}_INVALID) 丢弃该点，'
                f'否则取坐标 = v / {unit}.0f（{sp["value_desc"]}）')

    body = f"""/* NEUQ 视觉标定工具自动生成，请勿手工修改。
 *
 * 映射: {desc}
 * 方向: {sp['mapping_direction']}
 * 网格: {tw} x {th}（宽 x 高），按行优先，索引 = y * {tw} + x
 * 完整图尺寸: {sw} x {sh}
 * 定点: Q{TABLE_FIXED_POINT}，实际坐标 = 表值 / {unit}.0f
 * 无效: {BIN_SENTINEL}，表示该点映射到无穷远、地平线另一侧或超出图像范围
 *
 * {use_note}
 */
#ifndef {guard}
#define {guard}

#include <stdint.h>

#define {safe}_GRID_W  {tw}
#define {safe}_GRID_H  {th}
#define {safe}_SRC_W   {sw}
#define {safe}_SRC_H   {sh}
#define {safe}_SHIFT   {TABLE_FIXED_POINT}
#define {safe}_INVALID ({BIN_SENTINEL})

static const int16_t {safe}_mapW[{th * tw}] = {{
{format_int16_array(qx)}
}};

static const int16_t {safe}_mapH[{th * tw}] = {{
{format_int16_array(qy)}
}};

#endif /* {guard} */
"""
    (out_dir / 'Map.h').write_text(body, encoding='utf-8')


# 打表会产出的全部文件名。换格式重跑时要按这个清单清理上一版。
TABLE_FILENAMES = ('MapW.txt', 'MapH.txt', 'MapW.bin', 'MapH.bin', 'Map.h')


def clear_stale_tables(out_dir: Path) -> None:
    """清掉该目录里上一版格式的表。

    换 --table-format 重跑时，如果不清，目录里会同时躺着 MapW.txt 和 MapW.bin。
    下游按文件名猜格式就会拿到过期的那一份，而且两边都"看着像对的"，极难发现。
    """
    if not out_dir.is_dir():
        return
    for name in TABLE_FILENAMES:
        path = out_dir / name
        if path.is_file():
            path.unlink()


def prepare_map_pair(map_x: np.ndarray, map_y: np.ndarray,
                     source_size: Tuple[int, int]) -> MapPair:
    """把数学上算出的映射表加工成最终要交付的那一份。

    加工顺序刻意与落盘顺序一致：先按 TABLE_SIZE 重采样，再统一无效哨兵，
    最后按 TABLE_FORMAT 量化。走完这一步，磁盘上的字节与内存里的 `x`/`y`
    就是同一份数据的两种表示——批量测试与校验脚本都只认后者。
    """
    if TABLE_SIZE is not None:
        map_x, map_y = resample_map_pair(map_x, map_y, TABLE_SIZE)

    ok = np.isfinite(map_x) & np.isfinite(map_y) & (map_x >= 0.0) & (map_y >= 0.0)
    invalid = ~ok

    if TABLE_FORMAT in ('bin', 'c'):
        qx = quantize_table(np.where(invalid, -1.0, map_x))
        qy = quantize_table(np.where(invalid, -1.0, map_y))
        return MapPair(x=dequantize_table(qx), y=dequantize_table(qy),
                       source_size=source_size,
                       fixed_point=TABLE_FIXED_POINT, qx=qx, qy=qy)

    # 文本表按 %.2f 落盘，内存里也先舍到同一位；否则测到的精度比交付的高
    x = np.where(invalid, -1.0, np.round(map_x, 2))
    y = np.where(invalid, -1.0, np.round(map_y, 2))
    return MapPair(x=x, y=y, source_size=source_size)


def serialize_map_pair(pair: MapPair, out_dir: Path, tag: str, desc: str) -> None:
    """把 MapPair 写成文件，并附一份自描述的 metadata.json。

    只做序列化，不再改动任何数值。

    metadata.json 的意义在于让每套表**自带解释**：bin 是裸二进制，文件里既没有
    网格也没有 Q 位数，只能反过来依赖 matrices.json——那份元数据一旦损坏或没跟着
    搬走，同一串字节就会被按错误的方式解读（而且不报错）。自描述之后，
    交付给 C 端也说得清楚。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_tables(out_dir)

    if TABLE_FORMAT == 'c':
        write_table_c(out_dir, tag, desc, pair.qx, pair.qy, pair.source_size)
    elif TABLE_FORMAT == 'bin':
        write_table_bin(out_dir / 'MapW.bin', pair.qx)
        write_table_bin(out_dir / 'MapH.bin', pair.qy)
    else:
        write_table_txt(out_dir / 'MapW.txt', pair.x)
        write_table_txt(out_dir / 'MapH.txt', pair.y)

    meta = {
        'schema_version': SCHEMA_VERSION,
        'tool_version': TOOL_VERSION,
        'format': TABLE_FORMAT,
        'tag': tag,
        'description': desc,
        'grid_size': list(pair.size),
        'source_size': list(pair.source_size),
        'fixed_point': pair.fixed_point,
        'invalid_sentinel': (-1.0 if TABLE_FORMAT == 'txt' else BIN_SENTINEL),
    }
    meta.update(table_spaces(tag))
    # 索引网格可能被 --table-size 降采样，但两侧的完整图像尺寸仍是标定分辨率
    meta['index_full_size'] = list(pair.source_size)
    meta['value_full_size'] = list(pair.source_size)
    (out_dir / 'metadata.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')


def load_table(path: Path, fixed_point: Optional[int] = None,
               grid: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """读回一张已落盘的表，返回 (H, W) 的 float64，无效点统一成 -1。

    fixed_point 同样走 None 占位，避免默认值在定义时被绑死。
    """
    fp = TABLE_FIXED_POINT if fixed_point is None else int(fixed_point)
    if path.suffix == '.bin':
        if grid is None:
            raise SystemExit(f'读取 {path.name} 需要先确定网格尺寸。')
        w, h = grid
        raw = np.fromfile(path, dtype='<i2').astype(np.int64)
        if raw.size != w * h:
            raise SystemExit(f'{path} 有 {raw.size} 个值，与网格 {w}x{h} 不符。')
        return dequantize_table(raw, fp).reshape(h, w)

    arr = np.loadtxt(path, delimiter=',', dtype=np.float64)
    return np.atleast_2d(arr)


def _load_c_header(path: Path, source_size: Tuple[int, int]) -> MapPair:
    """从生成的 C 头文件里把两张定点表与网格抠出来。

    网格尺寸（GRID_W/GRID_H）与源图分辨率（SRC_W/SRC_H）是两回事：降采样后
    网格是 160x120，采样坐标却仍在 1280x720 的原图上。老版本的头文件只有
    模糊的 W/H，读回来时只能拿调用方知道的 source_size 兜底。
    """
    text = path.read_text(encoding='utf-8')

    def macro(name: str) -> Optional[int]:
        m = re.search(rf'#define\s+\w+_{name}\s+\(?(-?\d+)\)?', text)
        return int(m.group(1)) if m else None

    shift = macro('SHIFT')
    if shift is None:
        raise SystemExit(f'{path} 里找不到 SHIFT 宏。')
    w = macro('GRID_W') or macro('W')
    h = macro('GRID_H') or macro('H')
    if w is None or h is None:
        raise SystemExit(f'{path} 里找不到网格尺寸宏（GRID_W/GRID_H）。')
    src = (macro('SRC_W'), macro('SRC_H'))
    real_src = (src[0], src[1]) if src[0] and src[1] else source_size

    qs = []
    for suffix in ('mapW', 'mapH'):
        m = re.search(rf'_{suffix}\[[^\]]*\]\s*=\s*\{{(.*?)\}};', text, re.S)
        if not m:
            raise SystemExit(f'{path} 里找不到 {suffix} 数组。')
        vals = [int(v) for v in m.group(1).replace('\n', ' ').split(',') if v.strip()]
        if len(vals) != w * h:
            raise SystemExit(f'{path} 的 {suffix} 数组长度 {len(vals)} 与 {w}x{h} 不符。')
        qs.append(np.asarray(vals, dtype=np.int16).reshape(h, w))
    return MapPair(x=dequantize_table(qs[0], shift), y=dequantize_table(qs[1], shift),
                   source_size=real_src, fixed_point=shift, qx=qs[0], qy=qs[1])


def load_map_pair(folder: Path, source_size: Tuple[int, int],
                  grid: Optional[Tuple[int, int]] = None,
                  fixed_point: Optional[int] = None) -> MapPair:
    """读回一组已落盘的表，还原成 MapPair（按 txt / bin / C 三种格式自动识别）。

    优先用同目录的 metadata.json：它是和表一起写的自描述，网格与 Q 位数直接写在
    里面，比调用方从别处推断可靠得多。老表没有 metadata 时才回退到传入参数。

    批量测试与校验脚本都走这条路，验证的就是"真正交出去的那一份"，
    而不是重算出来的数学结果。
    """
    folder = Path(folder)
    meta: dict = {}
    meta_path = folder / 'metadata.json'
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            meta = {}
    if meta:
        check_schema(meta, str(meta_path))
    if meta.get('grid_size'):
        grid = tuple(int(v) for v in meta['grid_size'])
    if meta.get('fixed_point') is not None:
        fixed_point = int(meta['fixed_point'])
    if meta.get('source_size'):
        meta_src = tuple(int(v) for v in meta['source_size'])
        if meta_src != tuple(source_size):
            print(f'注意: {folder} 记录的原图分辨率 {meta_src} 与当前标定的 '
                  f'{tuple(source_size)} 不一致，按表里记录的为准。')
        source_size = meta_src

    fp = TABLE_FIXED_POINT if fixed_point is None else int(fixed_point)
    for suffix in ('.txt', '.bin'):
        wx, wy = folder / f'MapW{suffix}', folder / f'MapH{suffix}'
        if wx.is_file() and wy.is_file():
            if suffix == '.txt':
                x, y = load_table(wx, fp, grid), load_table(wy, fp, grid)
                return MapPair(x=x, y=y, source_size=source_size)
            # bin 是裸数组，文件里没有维度。没指定 --table-size 时，输出网格
            # 就是源图尺寸——这正是默认配置，不能因为"查不到网格"就报错。
            g = tuple(grid) if grid else tuple(source_size)
            x, y = load_table(wx, fp, g), load_table(wy, fp, g)
            return MapPair(x=x, y=y, source_size=source_size, fixed_point=fp)
    header = folder / 'Map.h'
    if header.is_file():
        return _load_c_header(header, source_size)
    raise SystemExit(f'{folder} 下没有可识别的查找表'
                     '（MapW.txt / MapW.bin / Map.h 都没有）。')


def report_table_size(out_dir: Path, shape: Tuple[int, int]) -> None:
    """打印一组表的体积与格式，便于判断能不能塞进目标平台。"""
    files = [p for p in out_dir.rglob('*') if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    fmt = {'txt': '逗号分隔文本', 'bin': 'int16 定点二进制',
           'c': 'C 头文件'}.get(TABLE_FORMAT, TABLE_FORMAT)
    note = '' if TABLE_FORMAT == 'txt' else f'（Q{TABLE_FIXED_POINT}）'
    print(f'  格式 {fmt}{note}，网格 {shape[0]}x{shape[1]}，'
          f'体积 {total / 1024:.1f} KB（{len(files)} 个文件）')


def export_undistort_tables(K: np.ndarray, D: np.ndarray, Knew: np.ndarray,
                            size: Tuple[int, int],
                            table_root: Optional[Path] = None) -> MapPair:
    """原图 <-> 去畸变图，正反两套表。返回反向表的最终交付版本。"""
    out_dir = (Path(table_root) if table_root is not None else DIR_TABLE) / 'undistort'
    w, h = size

    map_x, map_y = cv2.initUndistortRectifyMap(K, D, None, Knew, size, cv2.CV_32FC1)
    rev = mask_out_of_range(np.column_stack((map_x.ravel(), map_y.ravel())), size)
    fwd = mask_out_of_range(undistorted_grid(K, D, Knew, size), size)

    # 先加工成最终交付的那一份，再序列化：写盘与后续测试同源
    rev_pair = prepare_map_pair(rev[:, 0].reshape(h, w), rev[:, 1].reshape(h, w), size)
    fwd_pair = prepare_map_pair(fwd[:, 0].reshape(h, w), fwd[:, 1].reshape(h, w), size)
    serialize_map_pair(rev_pair, out_dir / 'reverse', 'undistort_reverse',
                       '去畸变 反向表（去畸变图像素 -> 原始畸变图采样坐标）')
    serialize_map_pair(fwd_pair, out_dir / 'forward', 'undistort_forward',
                       '去畸变 正向表（原始畸变图像素 -> 去畸变图落点）')

    print('去畸变表已写入:', out_dir)
    report_table_size(out_dir, rev_pair.size)
    return rev_pair


def export_composite_tables(K: np.ndarray, D: np.ndarray, Knew: np.ndarray,
                            H: np.ndarray, H0: np.ndarray, sign: float,
                            size: Tuple[int, int],
                            table_root: Optional[Path] = None) -> MapPair:
    """原图 <-> 去畸变逆透视图，正反两套表。返回反向表的最终交付版本。"""
    out_dir = (Path(table_root) if table_root is not None else DIR_TABLE) / 'undistort_ipm'
    w, h = size

    map_x, map_y = build_composite_reverse_map(K, D, Knew, H, H0, sign, size)
    und = undistorted_grid(K, D, Knew, size)
    # 与反向表同一判据：地平线另一侧的源图像素经 H 会得到符号翻转的镜像落点，
    # 地平线附近则得到 1e15 量级的坐标，两者都必须标成无效而不是原样写出去。
    valid = np.isfinite(und).all(axis=1)
    valid &= sign * homography_denominator(H0, und) >= DEN_EPS
    fwd = mask_out_of_range(apply_homography(H, und), size, valid)

    rev_pair = prepare_map_pair(map_x, map_y, size)
    fwd_pair = prepare_map_pair(fwd[:, 0].reshape(h, w), fwd[:, 1].reshape(h, w), size)
    serialize_map_pair(rev_pair, out_dir / 'reverse', 'undistort_ipm_reverse',
                       '逆透视 反向表（BirdView 输出像素 -> 原始畸变图采样坐标）')
    serialize_map_pair(fwd_pair, out_dir / 'forward', 'undistort_ipm_forward',
                       '逆透视 正向表（原始畸变图像素 -> BirdView 落点）')

    print('去畸变逆透视表已写入:', out_dir)
    report_table_size(out_dir, rev_pair.size)
    return rev_pair


def export_all(K: np.ndarray, D: np.ndarray, Knew: np.ndarray,
               H: np.ndarray, H0: np.ndarray, sign: float, extra: dict,
               size: Tuple[int, int],
               ipm_state: Optional[dict] = None) -> MapPair:
    """事务式导出：六矩阵 + 逆透视状态 + 两套表，全部成功才落到正式目录。

    原先的顺序是"先写 matrix/，再写两套表"。中间任何一步失败（打表溢出、
    磁盘满、被 Ctrl+C）都会留下"新矩阵配旧表"的组合，而两边的文件单看都正常，
    几乎无法察觉。这里改成：

      1) 全部写进 *.staging/（暂存区先用正式目录内容播种——否则整体替换会
         删掉本轮不生成的文件，比如用户自己放进去的参考资料）
      2) 两个正式目录一起挪到 *.old
      3) 两个暂存目录一起就位
      4) 全部成功才删掉 *.old

    第 2、3 步刻意跨目录成对执行：matrix 与 lookup_table 必须作为一套一起换，
    不能一个换成了新的、另一个还留着旧的。

    `ipm_state` 若给出，会写进 matrix 的暂存区，随事务一起提交；这样就不必
    事先单独保存状态文件，也就不会出现"新状态 + 旧矩阵"的中间态。
    """
    table_stage = DIR_TABLE.with_name(DIR_TABLE.name + '.staging')
    matrix_stage = DIR_MATRIX.with_name(DIR_MATRIX.name + '.staging')
    for stage, target in ((table_stage, DIR_TABLE), (matrix_stage, DIR_MATRIX)):
        shutil.rmtree(stage, ignore_errors=True)
        if target.is_dir():
            shutil.copytree(target, stage, dirs_exist_ok=True)

    try:
        export_matrices(K, D, Knew, H, extra, matrix_root=matrix_stage)
        if ipm_state is not None:
            (matrix_stage / 'ipm_state.json').write_text(
                json.dumps(ipm_state, indent=2, ensure_ascii=False), encoding='utf-8')
        export_undistort_tables(K, D, Knew, size, table_root=table_stage)
        pair = export_composite_tables(K, D, Knew, H, H0, sign, size,
                                       table_root=table_stage)
    except BaseException:
        # 任何一步出问题：丢掉暂存区，正式目录一个字节都没动过
        shutil.rmtree(table_stage, ignore_errors=True)
        shutil.rmtree(matrix_stage, ignore_errors=True)
        raise

    _commit_dirs(((table_stage, DIR_TABLE), (matrix_stage, DIR_MATRIX)))
    print(f'导出完成: {DIR_MATRIX} 与 {DIR_TABLE}')
    return pair


def _commit_dirs(pairs: Sequence[Tuple[Path, Path]], what: str = 'matrix 与 lookup_table',
                 keep_staging: bool = False) -> None:
    """把一批暂存目录整体换成正式目录；任何一步失败则全部回滚。

    实现已下沉到 neuq_core.fs_transaction：那里的 DirectorySwapTransaction 把
    install / finalize 拆开，好让"换装完、元数据还没落盘"这段窗口仍然可以回滚
    （素材覆盖导入要的就是这个）。这里保留名字，转发给一次性的 commit_dirs，
    既有调用点与外部引用的行为一个字不变。
    """
    commit_dirs(pairs, what=what, keep_staging=keep_staging)



# ---------------------------------------------------------------- 7. 批量测试

def same_aspect(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """两个分辨率是否同宽高比。

    容差 0.2%：只够容纳"编码器把奇数边裁掉一两个像素"这类差别
    （720p 上约 2~3 px）。放到 1% 就能容忍好几像素的裁剪甚至换了个 FOV，
    那就失去把关的意义了。
    """
    return abs(a[0] / a[1] - b[0] / b[1]) <= 0.002 * (b[0] / b[1])


def normalize_camera_image(img: np.ndarray, want_size: Tuple[int, int],
                           what: str = '图像', strict: bool = True
                           ) -> Optional[np.ndarray]:
    """把一张相机原图统一到标定分辨率，统一入口。

    规则刻意收紧到只有"同宽高比"才允许缩放：
      - 尺寸相同          → 原样返回
      - 尺寸不同但同比例  → 缩放（几何等价，但前提是拍摄时的 FOV 与裁剪模式没变，
                            所以会打印一条明确的警告）
      - 宽高比不同        → 拒绝。硬缩会把透视几何揉变形，而结果看上去"正常"，
                            是这一类里最难查的错。

    strict=True 时拒绝会抛 SystemExit（CLI 与 Web 都把它当作用户输入错误）；
    strict=False 时返回 None，交给调用方按"跳过这一张"处理（批量测试用）。
    """
    w, h = img.shape[1], img.shape[0]
    if (w, h) == want_size:
        return img
    if not same_aspect((w, h), want_size):
        msg = (f'{what}的分辨率 {w}x{h} 与标定时的 {want_size[0]}x{want_size[1]} '
               '宽高比不同，直接缩放会让透视几何失真。'
               '请换同比例的图片，或按正确分辨率重新标定。')
        if strict:
            raise SystemExit(msg)
        print(f'跳过：{msg}')
        return None
    print(f'提示: {what}是 {w}x{h}，与标定分辨率 {want_size[0]}x{want_size[1]} 同比例，'
          f'已缩放后使用（前提是拍摄时的视场与裁剪模式没有变化）。')
    return cv2.resize(img, want_size, interpolation=cv2.INTER_AREA)


def batch_test(pair: MapPair) -> None:
    """用一份"最终交付"的映射表批量处理 test_input/ 的图片，结果写入 test_output/。

    关键点是入参 MapPair：它已经过重采样与量化，与落盘的字节同源。走表而不是
    重算，测试才真正验证了导出的那份表；无效点按 C 端语义显式置黑。
    """
    files = list_images(DIR_TEST_IN)
    if not files:
        print(f'{DIR_TEST_IN} 中没有测试图，跳过批量测试。')
        return
    DIR_TEST_OUT.mkdir(parents=True, exist_ok=True)

    mx32 = pair.x.astype(np.float32)
    my32 = pair.y.astype(np.float32)
    invalid = pair.invalid
    ow, oh = pair.size

    for path in files:
        img = safe_imread(path)
        if img is None:
            print(f'跳过无法读取的文件: {path.name}')
            continue
        img = normalize_camera_image(img, pair.source_size, f'{path.name}',
                                     strict=False)
        if img is None:
            continue
        out = cv2.remap(img, mx32, my32, cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        out[invalid] = 0
        safe_imwrite(DIR_TEST_OUT / f'{path.stem}_birdview.jpg', out)
    print(f'批量测试完成（网格 {ow}x{oh}，来源是导出后的表）: {DIR_TEST_OUT}')


# ---------------------------------------------------------------- 8. 命令行与素材管理

STAGES = ('all', 'calib', 'ipm', 'tables', 'test', 'import', 'list')


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    ap = argparse.ArgumentParser(
        prog='neuq_vision_calib',
        description='NEUQ 智能车视觉标定一体化工具：相机标定 -> 逆透视标定 -> 打表 -> 批量测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '示例:\n'
            '  python neuq_vision_calib.py --list\n'
            '  python neuq_vision_calib.py --import-dir WIN_20260903_17_22_10_Pro\n'
            '  python neuq_vision_calib.py --stage calib --max-reproj-err 1.5\n'
            '  python neuq_vision_calib.py --ipm-source ipm_input/floor.jpg --phys-w 60 --phys-h 40\n'
            '  python neuq_vision_calib.py --quad 300,300,980,300,120,700,1160,700 --headless\n'
            '  python neuq_vision_calib.py --stage tables   # 复用上次交互标定的结果\n'
        ))
    ap.add_argument('--stage', choices=STAGES, default=None,
                    help='跑到哪个阶段为止。不给时：普通运行=all（全流程），'
                         '带 --import-dir 时=calib（导入后直接标定）')
    ap.add_argument('--list', action='store_true', help='打印各目录现状后退出')
    ap.add_argument('--import-dir', type=Path, metavar='DIR',
                    help='把混合素材目录按"能否检出棋盘"拆分进 calib_input/ 与 ipm_input/')
    ap.add_argument('--import-mode', choices=('add', 'replace'), default='add',
                    help='增量添加（按文件名去重，默认，要求规格与现有素材一致）/ '
                         '覆盖（新素材先备进暂存目录，全部成功后整体换装；换规格只能走这个）')
    ap.add_argument('--move', action='store_true',
                    help='--import-dir 时用移动代替复制')

    path_g = ap.add_argument_group('路径')
    path_g.add_argument('--root', type=Path, help='工程根目录，默认取脚本所在目录')
    path_g.add_argument('--calib-dir', type=Path, help='标定照片目录')
    path_g.add_argument('--ipm-dir', type=Path, help='逆透视原图目录')
    path_g.add_argument('--test-dir', type=Path, help='批量测试输入目录')
    path_g.add_argument('--ipm-source', type=Path, help='直接指定逆透视标定原图')

    board_g = ap.add_argument_group('标定板规格')
    board_g.add_argument('--board-squares', nargs=2, type=int, metavar=('X', 'Y'),
                         help='棋盘方格数，例如 12 9；不给则沿用当前工程里存的规格。'
                              '内角点数由程序换算（12x9 → 11x8），不用自己减 1')
    board_g.add_argument('--board-corners', nargs=2, type=int, metavar=('X', 'Y'),
                         help='兼容用法：直接给内角点数（如 11 8）。'
                              '与 --board-squares 只能给一个')
    board_g.add_argument('--square-size-mm', type=float, metavar='MM',
                         help='单格边长毫米；不给则沿用当前工程里存的规格')

    cal_g = ap.add_argument_group('标定参数')
    cal_g.add_argument('--force-calib', action='store_true',
                       help='忽略已有的 calib.json，重新标定（换了标定板时必须）')
    cal_g.add_argument('--undist-alpha', type=float,
                       help='去畸变输出视角 0~1；不给等价 MATLAB 的 OutputView=same（Knew=K）')
    cal_g.add_argument('--max-reproj-err', type=float, metavar='PX',
                       help='迭代剔除重投影误差超过该值的视图，例如 1.5')
    cal_g.add_argument('--capture-size', nargs=2, type=int, metavar=('W', 'H'),
                       help='在线拍摄分辨率；不给则用摄像头原生分辨率')

    ipm_g = ap.add_argument_group('逆透视参数')
    ipm_g.add_argument('--quad', metavar='X1,Y1,...,X4,Y4',
                       help='直接给出源图上标定矩形的四个角点（TL,TR,BL,BR），跳过交互')
    ipm_g.add_argument('--phys-w', type=float, help='标定矩形真实宽度（cm）')
    ipm_g.add_argument('--phys-h', type=float, help='标定矩形真实高度（cm）')
    ipm_g.add_argument('--scale', type=float, help='cm -> 输出像素的尺度（px/cm）')
    ipm_g.add_argument('--anchor-x', type=float, help='ROI 中心的归一化横向位置 0~1')
    ipm_g.add_argument('--anchor-y', type=float, help='ROI 中心的归一化纵向位置 0~1')
    ipm_g.add_argument('--heading', type=float, help='标定区相对车辆前进方向的转角（度）')
    ipm_g.add_argument('--max-range-cm', type=float, help='前向有效距离上限（cm）')
    ipm_g.add_argument('--max-lateral-cm', type=float, help='横向有效距离上限（cm）')

    tbl_g = ap.add_argument_group('查找表导出')
    tbl_g.add_argument('--table-format', choices=('txt', 'bin', 'c'), default=None,
                       help='txt（默认，逗号分隔文本，与历史产物一致）/ '
                            'bin（int16 定点裸二进制）/ c（自包含 C 头文件）')
    tbl_g.add_argument('--table-size', nargs=2, type=int, metavar=('W', 'H'),
                       help='把表重采样到该网格，例如 320 240；不给则与图像同尺寸')
    tbl_g.add_argument('--table-fixed-point', type=int, metavar='N',
                       help='bin/c 的定点小数位数，默认 4（即 1/16 px）')

    misc_g = ap.add_argument_group('其他')
    misc_g.add_argument('--headless', action='store_true', help='禁止打开任何窗口')
    misc_g.add_argument('-y', '--yes', action='store_true', help='所有交互提问一律取默认值')
    return ap


def apply_options(args: argparse.Namespace) -> None:
    """把命令行参数写回模块级配置。

    这些量在模块级被各函数直接引用，用 global 重写而不是层层传参，改动面最小。
    """
    global UNDIST_ALPHA, MAX_RANGE_CM, MAX_LATERAL_CM, MAX_REPROJ_ERR
    global ANCHOR_X, ANCHOR_Y, HEADING_DEG, CAPTURE_SIZE, CAPTURE_ONLINE
    global TABLE_FORMAT, TABLE_SIZE, TABLE_FIXED_POINT

    # 棋盘规格要在最前面定下来：素材导入是否把一张图判成棋盘照，用的就是它。
    # 只有用户显式给了参数才落盘——否则每次启动都会把 project.json 覆盖成默认值。
    if (args.board_squares is not None or args.board_corners is not None
            or args.square_size_mm is not None):
        try:
            board = resolve_board(args.board_squares, args.board_corners,
                                  args.square_size_mm)
        except ValueError as exc:
            raise SystemExit(f'标定板规格不合法: {exc}') from None
        configure_board(board, persist=True)
        print(f'标定板规格: {board.label}')

    if args.undist_alpha is not None:
        UNDIST_ALPHA = args.undist_alpha
    if args.max_range_cm is not None:
        MAX_RANGE_CM = args.max_range_cm
    if args.max_lateral_cm is not None:
        MAX_LATERAL_CM = args.max_lateral_cm
    if args.max_reproj_err is not None:
        MAX_REPROJ_ERR = args.max_reproj_err
    if args.anchor_x is not None:
        ANCHOR_X = args.anchor_x
    if args.anchor_y is not None:
        ANCHOR_Y = args.anchor_y
    if args.heading is not None:
        HEADING_DEG = args.heading
    if args.capture_size is not None:
        CAPTURE_SIZE = (args.capture_size[0], args.capture_size[1])
    if args.table_format is not None:
        TABLE_FORMAT = args.table_format
    if args.table_size is not None:
        if args.table_size[0] <= 0 or args.table_size[1] <= 0:
            raise SystemExit('--table-size 的两个值都必须是正整数。')
        TABLE_SIZE = (args.table_size[0], args.table_size[1])
    if args.table_fixed_point is not None:
        if not 0 <= args.table_fixed_point <= 15:
            raise SystemExit('--table-fixed-point 需在 0~15 之间。')
        TABLE_FIXED_POINT = args.table_fixed_point
    if args.headless:
        # 无 GUI 环境下 cv2 的任何窗口调用都会直接抛错，提前把在线拍摄关掉。
        CAPTURE_ONLINE = False


def parse_quad(text: str) -> np.ndarray:
    """解析 "x1,y1,x2,y2,x3,y3,x4,y4" 为 TL,TR,BL,BR 四点。"""
    try:
        vals = [float(v) for v in text.replace(';', ',').split(',') if v.strip()]
    except ValueError as exc:
        raise SystemExit(f'--quad 含无法解析的数值: {exc}') from None
    if len(vals) != 8:
        raise SystemExit(f'--quad 需要 8 个数字（4 个点的 x,y），收到 {len(vals)} 个。')
    quad = np.asarray(vals, dtype=np.float64).reshape(4, 2)
    ordered = order_corners_tl_tr_bl_br(quad)
    if not np.allclose(ordered, quad):
        print('提示: --quad 四点顺序不是 TL,TR,BL,BR，已自动重排。')
    if not is_convex_quad(ordered):
        raise SystemExit('--quad 四点构成自交或非凸四边形，无法定义地平面矩形。')
    return ordered


def unique_destination(path: Path) -> Path:
    """目标已存在时追加 _1/_2 后缀，绝不覆盖已有素材。"""
    if not path.exists():
        return path
    for i in range(1, 1000):
        cand = path.with_name(f'{path.stem}_{i}{path.suffix}')
        if not cand.exists():
            return cand
    raise SystemExit(f'无法为 {path} 找到不冲突的目标名。')


def library_has(name: str) -> bool:
    """文件名是否已存在于素材库的任一目录。

    去重必须查全库，不能只查"这次要放进去的那个目录"：同一张照片在不同轮次里
    可能被判到不同目录（比如判定逻辑升级后，原来在 ipm_input/ 的棋盘照改判进了
    _incomplete/），只查目标目录就会把它再复制一份，素材库越滚越大。
    """
    return any((d / name).is_file() for d in
               (DIR_CALIB_IN, DIR_CALIB_IN / INCOMPLETE_SUBDIR, DIR_IPM_IN))


def import_dataset(src_dir: Path, move: bool = False,
                   mode: str = 'add') -> dict:
    """把混合素材目录按"能否检出棋盘"拆进 calib_input/ 与 ipm_input/。

    mode:
      add (默认) — 增量导入，按文件名去重：素材库里已有同名文件就跳过，
                  没冲突的才复制/移入。重复点同一份素材时不会堆出 _1/_2 后缀。
                  要求当前规格与这套素材的分类依据一致，否则直接拒绝
                  （两套规格的分拣结果混在一个库里，事后无从分辨）。
                  遇到坏图跳过、其余照常入库（best effort：它只追加，不删东西）。
      replace — 覆盖导入：先把新素材完整地准备进 *.staging/，全部成功后才
                  整体换装。中途失败（磁盘满、权限、坏图）时正式的素材库
                  一个文件都不会少 —— 坏图同样算失败，整批放弃（all-or-nothing），
                  否则"19 张好图 + 1 张坏图"会把旧库整套换成那 19 张。
                  换装与"这套素材按哪块棋盘分拣"由 MaterialImportTransaction 一起
                  成交：写 pending → 换装（留 .old）→ 一次写完 material_set 与
                  last_import → 删 .old，任何一步失败都连目录一起回滚。
                  只有这个模式允许换棋盘规格。


    同名文件在 add 模式下跳过（不去重才追加 _1/_2 后缀）；move 控制是搬过来还是拷贝过来。
    返回 {"imported_calib","imported_ipm","skipped_duplicates","skipped_unreadable","cleared"}。
    """
    src = Path(src_dir).expanduser()
    if not src.is_dir():
        raise SystemExit(f'--import-dir 不是目录: {src}')
    files = list_images(src)
    if not files:
        raise SystemExit(f'{src} 下没有图片。')
    if mode not in ('add', 'replace'):
        raise SystemExit(f'未知的 --import-mode {mode!r}，可选 add / replace。')

    # 分拣目的地：正式目录 -> 本轮实际写入的目录
    incomplete = DIR_CALIB_IN / INCOMPLETE_SUBDIR
    if mode == 'replace':
        # 必须在创建暂存区之前检查：下面那句 rmtree(stage) 会毫不留情地
        # 删掉上一轮失败时特意保住的 *.staging，而 --move 时那可能是唯一副本。
        require_no_material_residue()
        pairs = material_stage_pairs()
        dest_of = {
            DIR_CALIB_IN: pairs[0][0],
            incomplete: pairs[0][0] / INCOMPLETE_SUBDIR,
            DIR_IPM_IN: pairs[1][0],
        }
        for stage, _ in pairs:
            shutil.rmtree(stage, ignore_errors=True)
            stage.mkdir(parents=True, exist_ok=True)
    else:
        require_material_basis('增量导入')
        pairs = []
        dest_of = {DIR_CALIB_IN: DIR_CALIB_IN,
                   incomplete: incomplete,
                   DIR_IPM_IN: DIR_IPM_IN}
    for folder in dest_of.values():
        folder.mkdir(parents=True, exist_ok=True)
    action = shutil.move if move else shutil.copy2

    print(f'导入 {src}')
    print(f'  标定板 {BOARD.label}，'
          f'共 {len(files)} 张，模式 {mode}，{"移动" if move else "复制"}到工程目录\n')
    calib_n = ipm_n = skip_n = dup_n = partial_n = 0
    try:
        for path in files:
            img = safe_imread(path)
            if img is None:
                print(f'  [跳过] {path.name}（无法读取）')
                skip_n += 1
                continue

            gray = to_gray(img)
            found, _corners = detect_chessboard(gray)
            partial = None
            if found:
                final_dir, tag = DIR_CALIB_IN, '棋盘'
            else:
                # 完整板没检出，再退一步找局部子网格：命中说明是棋盘照没拍全，
                # 而不是地面照。归到 _incomplete/ 子目录，避免污染 ipm_input/ 的候选列表。
                partial = detect_chessboard_partial(gray)
                if partial is not None:
                    final_dir = incomplete
                    tag = f'棋盘不全({partial[0]}x{partial[1]})'
                else:
                    final_dir, tag = DIR_IPM_IN, '地面'
            dest_dir = dest_of[final_dir]

            target = dest_dir / path.name
            if mode == 'add' and library_has(path.name):
                print(f'  [跳过] {path.name}（素材库里已存在）')
                dup_n += 1
                continue
            dest = unique_destination(target)
            action(str(path), str(dest))
            print(f'  [{tag}] {path.name} -> {dest_dir.relative_to(SCRIPT_DIR)}/'
                  + ('' if dest.name == path.name else f'（重命名为 {dest.name}）'))
            if found:
                calib_n += 1
            elif partial is not None:
                partial_n += 1
            else:
                ipm_n += 1

        if mode == 'replace' and skip_n:
            # 覆盖导入是 all-or-nothing：坏图与磁盘满、权限错一样都算失败。
            # 放过去的话，"19 张好图 + 1 张坏图"会把旧素材库整套换成那 19 张，
            # 而 README 承诺的是"中途失败时正式素材库一个文件都不会少"。
            # add 模式仍是 best effort：它只往库里追加，不会删掉任何既有素材。
            raise SystemExit(f'有 {skip_n} 张图片无法读取，已放弃覆盖，'
                             '原素材库保持不变。\n'
                             '请剔除坏图后重试；只想尽力导入能读的那些，用增量导入。')
    except BaseException:

        if mode == 'replace':
            print('\n覆盖导入中断：正式的素材库未做任何改动，仍是原来那一套。')
            for stage, _ in pairs:
                print(f'  本轮已准备好的新素材停在 {stage}，确认后可自行删除。')
        raise

    if mode == 'replace':
        if calib_n + partial_n + ipm_n == 0:
            # 一张都没进来就不要真的去覆盖：否则换装把旧素材连同 .old 一起删了，
            # 用户拿到的是一句"导入完成"和一个空库。
            for stage, _ in pairs:
                shutil.rmtree(stage, ignore_errors=True)
            raise SystemExit('没有一张图成功入库，已放弃覆盖，原素材库保持不变。')
        replaced = sum(1 for d in (DIR_CALIB_IN, DIR_IPM_IN) if d.is_dir()
                       for p in d.rglob('*') if p.is_file())
        # 换装与"这套素材按哪块棋盘分拣"必须一起成交。事务先严格写下 pending、
        # 再换装（保留 .old）、再一次写完 material_set + last_import，最后才删 .old；
        # 中间任何一步失败都会连目录一起回滚，不会留下"目录是新的、配置是旧的"。
        txn = MaterialImportTransaction(pairs, BOARD, {
            'at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mode': mode,
            'counts': {'calib': calib_n, 'partial': partial_n, 'ipm': ipm_n},
        })
        txn.install()
        txn.commit()
        print(f'覆盖模式：旧素材库（{replaced} 个文件）已被这一轮替换。')


    parts = [f'棋盘 {calib_n} 张']
    if partial_n:
        parts.append(f'棋盘不全 {partial_n} 张')
    parts.append(f'地面 {ipm_n} 张')
    if dup_n: parts.append(f'重复跳过 {dup_n} 张')
    if skip_n: parts.append(f'无法读取 {skip_n} 张')
    parts.append('移动完成' if move else '复制完成')
    print('\n完成: ' + '，'.join(parts) + '。')
    if partial_n:
        print(f'  「棋盘不全」指检不出完整 {BOARD.corners[0]}x{BOARD.corners[1]} '
              f'但能匹配到局部子网格的照片，已归到 '
              f'{(DIR_CALIB_IN / INCOMPLETE_SUBDIR).relative_to(SCRIPT_DIR)}/。'
              '它们无法参与标定（标定要求整块棋盘可见），只是从地面候选里摘出来。')
    if ipm_n > 1:
        print(f'{DIR_IPM_IN.name}/ 里有 {ipm_n} 张地面候选，'
              '需人工挑出真正用于逆透视标定的那一张，再用 --ipm-source 指定。')

    # 这套素材现在整体是按哪块棋盘分拣的，是**工程级事实**，必须单独记：
    # 记在"最近一次导入"上的话，一次增量导入就能把混合状态洗白。
    # replace 的这两段已经由上面的事务一次写完（它必须与换装同生共死）；
    # add 不做破坏性换装，沿用原来的两次独立写：最坏情况是"库里有素材但依据未知"，
    # 现有的 material_basis_unknown_reason 闸门下一次会把它挡住。
    if mode == 'add':
        set_material_board(BOARD)
        update_project_config(last_import={
            'at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mode': mode,
            'counts': {'calib': calib_n, 'partial': partial_n, 'ipm': ipm_n},
        })


    return {'mode': mode, 'imported_calib': calib_n,
            'imported_partial': partial_n, 'imported_ipm': ipm_n,
            'skipped_duplicates': dup_n, 'skipped_unreadable': skip_n,
            'cleared': mode == 'replace'}



def print_inventory() -> None:
    """打印各目录现状，用于确认素材是否就位。"""
    # 最后一项是 recursive 标志：calib_input 下面挂着 _incomplete/ 子目录，
    # 而那个子目录单独占一行，父目录统计时必须只看直属文件，否则两边重复计数。
    rows = (
        ('calib_input', DIR_CALIB_IN, '相机标定照片（≥3 张，建议 15+）', False),
        ('calib_input/_incomplete', DIR_CALIB_IN / INCOMPLETE_SUBDIR,
         '检不出完整棋盘的标定照（不参与标定）', False),
        ('calib_data', DIR_CALIB_DATA, '标定结果 calib.json', False),
        ('calib_preview', DIR_CALIB_PREVIEW, '标定图去畸变验收', False),
        ('ipm_input', DIR_IPM_IN, '逆透视标定原图', False),
        ('ipm_output', DIR_IPM_OUT, '去畸变图 + BirdView 结果', False),
        ('matrix', DIR_MATRIX, '六矩阵与逆透视状态', False),
        ('lookup_table', DIR_TABLE, '两套查找表（正/反向）', True),
        ('test_input', DIR_TEST_IN, '批量测试输入', False),
        ('test_output', DIR_TEST_OUT, '批量测试输出', False),
        ('backups', DIR_BACKUP, '备份并清空产生的归档', True),
    )
    print(f'工程根目录: {SCRIPT_DIR}\n')
    basis = material_board()
    print(f'  当前标定板           {BOARD.label}')
    print('  素材库的分类依据     '
          + (basis.label if basis is not None else '（无记录）') + '\n')
    for name, path, desc, recursive in rows:
        if not path.is_dir():
            state = '缺失'
        else:
            files = ([p for p in path.rglob('*') if p.is_file()] if recursive
                     else [p for p in path.iterdir() if p.is_file()])
            n_img = sum(1 for p in files if p.suffix.lower() in IMAGE_SUFFIXES)
            if not files:
                state = '空'
            elif n_img == len(files):
                state = f'{n_img} 张图'
            else:
                state = f'{len(files)} 个文件'
        print(f'  {name:<24} {state:<12} {desc}')

    print('\n下一步:')
    if not list_images(DIR_CALIB_IN):
        print('  calib_input/ 是空的，先用 --import-dir <素材目录> 导入棋盘照片。')
    elif not CALIB_JSON.is_file():
        print('  运行 --stage calib 生成 calib.json。')
    elif not list_images(DIR_IPM_IN):
        print('  ipm_input/ 是空的，放入一张地面标定原图。')
    else:
        print('  直接运行（不带参数）开始交互标定逆透视。')


def ipm_state_path() -> Path:
    """逆透视交互结果的中转文件，让后续阶段不必重跑交互。"""
    return DIR_MATRIX / 'ipm_state.json'


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """算文件的 SHA-256。用来证明"就是当时那张原图"，而不是"名字碰巧一样"。"""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def recorded_source(path: Path) -> dict:
    """把原图记成"工程内相对路径 + 内容哈希"。

    不记绝对路径的原因有三：工程一搬家就失效；备份压缩包会把
    C:\\Users\\某人\\... 一起带走；发给别人等于顺手泄漏本机目录结构。
    哈希则让校验方能够**证明**用的是同一张图，而不是靠文件名猜。
    """
    p = Path(path).resolve()
    rel = None
    try:
        rel = p.relative_to(SCRIPT_DIR).as_posix()
    except ValueError:
        rel = None                      # 不在工程内，只能靠名字 + 哈希
    out = {'src_image': rel or p.name,
           'src_image_in_project': rel is not None}
    try:
        out['src_sha256'] = file_sha256(p)
    except OSError:
        out['src_sha256'] = None
    if rel is None:
        out['src_image_external'] = True
    return out


def board_conflict(meta: Optional[dict],
                   spec: Optional[CheckerboardSpec] = None) -> Optional[str]:
    """已有 calib.json 与当前规格不一致时，返回一句给人看的说明；一致返回 None。

    为什么要拦：复用旧 calib.json 时，K/D 来自当时那块棋盘，而界面/命令行上显示的
    是现在这块。不拦的话会出现"界面写着 12x9、运行标定成功"，实际复用的却是
    9x7 的旧参数——数学没变，但 provenance 被写错了，事后完全说不清。
    """
    saved = spec_from_meta(meta)
    now = BOARD if spec is None else spec
    if saved is None or saved == now:
        return None
    return (f'当前标定板规格与已有 calib.json 不一致：\n'
            f'  当前：{now.label}\n'
            f'  已有：{saved.label}\n'
            '请改回已有规格，或勾选"强制重新标定"（命令行加 --force-calib）重标一次。')


def build_ipm_state(cal: 'IpmCalibrator', phys_w: float, phys_h: float,
                    img_size: Tuple[int, int],
                    src_path: Optional[Path] = None) -> dict:
    """组装逆透视标定状态。

    与"落盘"分开是为了让它能并进导出事务：矩阵、查找表、状态三者要么一起
    换成新的，要么一起保持旧的。否则导出中途失败会留下"新状态配旧矩阵"。
    """
    payload = {
        'schema_version': SCHEMA_VERSION,
        'tool_version': TOOL_VERSION,
        'H': cal.H.tolist(),
        'H0': cal.H0.tolist(),
        'horizon_sign': cal.sign,
        'scale_px_per_cm': cal.scale,
        'max_scale_px_per_cm': cal.max_scale,
        'anchor_x': cal.anchor_x,
        'anchor_y': cal.anchor_y,
        'heading_deg': cal.heading,
        'phys_w_cm': phys_w,
        'phys_h_cm': phys_h,
        'src_quad_tl_tr_bl_br': cal.corners.tolist(),
        'src_image_width': int(img_size[0]),
        'src_image_height': int(img_size[1]),
    }
    if src_path is not None:
        payload.update(recorded_source(src_path))
    return payload


def save_ipm_state(cal: 'IpmCalibrator', phys_w: float, phys_h: float,
                   img_size: Tuple[int, int], src_path: Optional[Path] = None) -> None:
    """把逆透视标定的结果落盘。

    交互标定是全流程里唯一需要人工介入的一步，如果只存在于内存里，
    后面想单独重跑打表或批量测试就得把四条线再拖一遍。

    同时记录用的是哪张原图：ipm_input/ 里通常躺着多张候选，不记下来的话，
    后续校验工具只能靠猜，猜错就会拿另一张图去对表，得出满屏假差异。

    只有"到此为止、不导出"的 --stage ipm 才走这里；要导出的话应该把
    build_ipm_state() 的结果交给 export_all()，随事务一起提交。
    """
    payload = build_ipm_state(cal, phys_w, phys_h, img_size, src_path)
    p = ipm_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print('逆透视标定状态已保存:', p)


def load_ipm_state() -> Optional[dict]:
    """读取逆透视标定状态，文件不存在返回 None。"""
    p = ipm_state_path()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise SystemExit(f'{p} 不是合法 JSON: {exc}') from None
    if not isinstance(data, dict):
        return None
    check_schema(data, str(p))
    return data


def make_calibrator_from_quad(img: np.ndarray, quad: np.ndarray, phys_w: float,
                              phys_h: float, scale: Optional[float] = None
                              ) -> 'IpmCalibrator':
    """由四个角点反推四条线，复用 IpmCalibrator 的几何与合法性校验。

    四条线就是四边形的四条边：TOP=TL->TR、BOTTOM=BL->BR、LEFT=TL->BL、RIGHT=TR->BR。
    这样无 GUI 路径与交互路径走的是同一套 build_corners/compute_homography，
    不会出现两条链路结果不一致的问题。
    """
    tl, tr, bl, br = (np.asarray(p, dtype=np.float64) for p in quad)
    cal = IpmCalibrator(img, phys_w, phys_h)
    cal.line_points = np.array([[tl, tr], [bl, br], [tl, bl], [tr, br]], dtype=np.float64)
    cal.line_default = cal.line_points.copy()
    if scale is not None:
        cal.scale = max(MIN_SCALE, float(scale))
        cal.scale_initialized = True   # 阻止 recompute 用 max_scale 比例覆盖指定值
    cal.recompute()
    return cal


# ---------------------------------------------------------------- 主流程

def ask_physical_size(default_w: float = PHYS_W_CM,
                      default_h: float = PHYS_H_CM) -> Tuple[float, float]:
    """交互询问标定矩形的真实物理尺寸（cm），非法输入回落到默认值。"""
    def ask(label: str, default: float) -> float:
        try:
            raw = input(f'{label}（cm，回车用默认 {default:g}）: ').strip()
        except EOFError:
            print(f'无输入流，用默认值 {default:g}。')
            return default
        if not raw:
            return default
        try:
            v = float(raw)
        except ValueError:
            print(f'无法解析 "{raw}"，用默认值 {default:g}。')
            return default
        if v <= 0:
            print(f'必须为正数，用默认值 {default:g}。')
            return default
        return v

    print('\n输入标定矩形的真实物理尺寸，用于确定目标 ROI 的长宽比与物理尺度。')
    return ask('标定矩形宽度', default_w), ask('标定矩形高度', default_h)


def resolve_physical_size(args: argparse.Namespace) -> Tuple[float, float]:
    """确定标定矩形的物理尺寸：命令行优先，其次交互询问，最后默认值。"""
    if args.phys_w is not None and args.phys_h is not None:
        if args.phys_w <= 0 or args.phys_h <= 0:
            raise SystemExit('--phys-w/--phys-h 必须为正数。')
        print(f'标定矩形物理尺寸: {args.phys_w:g} x {args.phys_h:g} cm（来自命令行）')
        return float(args.phys_w), float(args.phys_h)
    if args.yes or args.headless:
        print(f'标定矩形物理尺寸: {PHYS_W_CM:g} x {PHYS_H_CM:g} cm（默认值）')
        return PHYS_W_CM, PHYS_H_CM
    return ask_physical_size(args.phys_w if args.phys_w else PHYS_W_CM,
                             args.phys_h if args.phys_h else PHYS_H_CM)


def discover_ipm_source(explicit: Optional[Path] = None) -> Optional[Path]:
    """确定逆透视标定用的原图。

    改造前这里写死 UnInverseImage.jpg：只要照片换个名字丢进 ipm_input/，流程就会以
    "无法读取"结束，而目录里明明躺着可用素材。现在的顺序是
    显式指定 -> 约定名 -> 目录内唯一一张 -> 列出候选并要求显式指定。

    显式指定的文件分两种：位于 ipm_input/ 之内的，仍要过素材依据这一关——它躺在
    那里本身就是旧规格分拣的结果（"这张检不出棋盘，算地面照"），用户写出文件名
    并不能让它重新可信；只有素材库之外的外部图才真正绕过闸门，它不依赖任何分拣结果。
    自动取图同理必须过闸，且依据未知时也要拦：文件被放进 ipm_input/ 就是过去分类
    的产物，依据未知就无法证明它真是地面照。
    """
    if explicit is not None:
        p = resolve_user_path(explicit, DIR_IPM_IN)
        if not p.is_file():
            raise SystemExit(f'--ipm-source 指向的文件不存在: {p}\n'
                             f'（只写文件名时会到 {DIR_IPM_IN} 下查找）')
        if in_material_library(p):
            require_material_basis('逆透视标定', block_unknown=True)
        return p
    require_material_basis('逆透视标定', block_unknown=True)

    if IPM_SOURCE.is_file():
        return IPM_SOURCE
    candidates = list_images(DIR_IPM_IN)
    if len(candidates) == 1:
        print(f'提示: ipm_input/ 中没有 {IPM_SOURCE.name}，'
              f'改用唯一候选 {candidates[0].name}。')
        return candidates[0]
    if not candidates:
        print(f'{DIR_IPM_IN} 中没有图片。')
        return None
    print(f'{DIR_IPM_IN} 中有 {len(candidates)} 张候选，无法自动确定逆透视原图：')
    for p in candidates:
        print('   ', p.name)
    print('  请用 --ipm-source <文件名> 指定其中一张。')
    return None


def run_ipm_calibration(src: np.ndarray, img_size: Tuple[int, int],
                        args: argparse.Namespace, quad: Optional[np.ndarray]
                        ) -> Optional[Tuple['IpmCalibrator', float, float]]:
    """执行逆透视标定，返回 (标定器, 物理宽, 物理高)。

    给了 quad 就走无 GUI 路径，否则打开交互窗口。两条路径产出的 H/H0 结构一致，
    下游的矩阵导出与打表完全共用，不会出现两套结果。
    """
    phys_w, phys_h = resolve_physical_size(args)
    if quad is not None:
        print('无 GUI 模式：使用命令行指定的四点')
        for name, p in zip(('TL', 'TR', 'BL', 'BR'), quad, strict=True):
            print(f'  {name} = ({p[0]:.1f}, {p[1]:.1f})')
        cal = make_calibrator_from_quad(src, quad, phys_w, phys_h, scale=args.scale)
        if cal.H0 is None or cal.H is None:
            print('指定的四点无法构成有效四边形（退化/平行/自交），未产生结果。')
            return None
        return cal, phys_w, phys_h

    if args.headless:
        raise SystemExit('--headless 下无法拖拽四条线，请同时给出 --quad。')

    cal = IpmCalibrator(src, phys_w, phys_h)
    H = cal.run()
    if H is None or cal.birdview is None or cal.corners is None or cal.H0 is None:
        print('四条线未构成有效四边形，未保存任何结果。')
        return None
    return cal, phys_w, phys_h


def rebuild_reverse_map_from_state(K: np.ndarray, D: np.ndarray, Knew: np.ndarray,
                                   img_size: Tuple[int, int]) -> MapPair:
    """从 ipm_state.json 重建 BirdView 反向表（已加工成最终交付形态）。"""
    state = load_ipm_state()
    if state is None:
        raise SystemExit(f'未找到 {ipm_state_path()}。'
                         '请先跑一次 --stage ipm（或带 --quad 的完整流程）。')
    saved_size = (state.get('src_image_width'), state.get('src_image_height'))
    if saved_size != img_size:
        raise SystemExit(f'逆透视状态记录的分辨率 {saved_size} 与标定分辨率 {img_size} '
                         '不一致，请重做逆透视标定。')
    H = np.asarray(state['H'], dtype=np.float64).reshape(3, 3)
    H0 = np.asarray(state['H0'], dtype=np.float64).reshape(3, 3)
    sign = float(state.get('horizon_sign', 1.0))
    map_x, map_y = build_composite_reverse_map(K, D, Knew, H, H0, sign, img_size)
    return prepare_map_pair(map_x, map_y, img_size)


def load_exported_reverse_pair(img_size: Tuple[int, int]) -> MapPair:
    """读回已经导出的 BirdView 反向表。

    批量测试优先走这条路：验证的是真正落盘、将来要烧进车里的那份表，
    而不是重算出来的数学结果。

    网格尺寸与定点位数都必须从 matrices.json 恢复。bin 是裸二进制，文件里
    既没有维度也没有 Q 位数：导出时用了 Q2、重启后按程序默认的 Q4 去读，
    同一串字节会解出 4 倍大的坐标，而且看起来"读成功了"。
    """
    folder = DIR_TABLE / 'undistort_ipm' / 'reverse'
    grid: Optional[Tuple[int, int]] = None
    fixed_point: Optional[int] = None
    if MATRIX_JSON.is_file():
        try:
            meta = json.loads(MATRIX_JSON.read_text(encoding='utf-8'))
            check_schema(meta, str(MATRIX_JSON))   # 契约要覆盖所有读它的地方
            size = meta.get('table_size_wh')
            grid = (int(size[0]), int(size[1])) if size else None
            fp = meta.get('table_fixed_point')
            fixed_point = None if fp is None else int(fp)
        except (json.JSONDecodeError, TypeError, ValueError):
            grid = fixed_point = None
    return load_map_pair(folder, img_size, grid, fixed_point)


def load_or_run_calibration(force: bool = False
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """取得 (K, D, Knew, img_size)：有 calib.json 就直接用，没有则现场标定并落盘。

    复用之前会先核对标定板规格：已有 calib.json 若是用另一块棋盘标的，
    直接复用会让后面写出的 matrices.json 记上错误的 provenance
    （K/D 来自旧棋盘，board 段却写着新棋盘）。这种情况一律要求显式重标。
    """
    calib = None if force else load_calibration()
    if calib is not None:
        conflict = board_conflict(calib_board_meta())
        if conflict:
            raise SystemExit(conflict)
        K, D, img_size = calib
    else:
        if force:
            print('按要求忽略已有 calib.json，重新标定。')
        else:
            print('未找到 calib.json，开始从标定照片重新标定。')
        spec = BOARD          # 显式捕获：provenance 跟着本次实际用的板走
        K, D, img_size, _fit = calibrate_camera(board=spec)
        save_calibration(K, D, img_size, board=spec)
    Knew = resolve_new_camera_matrix(K, D, img_size)
    assert_invertible('K', K)
    assert_invertible('Knew', Knew)
    return K, D, Knew, img_size


def main(argv: Optional[Sequence[str]] = None) -> None:
    """按命令行选择的阶段执行。不带参数时等价于改造前的全流程。"""
    args = build_arg_parser().parse_args(argv)
    configure_paths(root=args.root, calib_dir=args.calib_dir, ipm_dir=args.ipm_dir,
                    test_dir=args.test_dir, ipm_source=args.ipm_source)
    apply_options(args)

    # 只给 --import-dir 时默认停在标定阶段：导入完素材紧接着标定是最自然的下一步，
    # 而逆透视标定必须人工挑图、拖线，不该被自动带过去。
    stage = args.stage or ('calib' if args.import_dir is not None else 'all')

    # 四点先校验一次：参数写错了要立刻报，而不是等标定、去畸变跑完才说。
    quad = parse_quad(args.quad) if args.quad is not None else None

    if args.list or stage == 'list':
        print_inventory()
        return

    if args.import_dir is not None:
        import_dataset(resolve_import_dir(args.import_dir),
                       move=args.move, mode=args.import_mode)
        if stage == 'import':
            return
        print()
    elif stage == 'import':
        raise SystemExit('--stage import 需要同时给出 --import-dir <目录>。')

    if CAPTURE_ONLINE:
        capture_calibration_images()

    K, D, Knew, img_size = load_or_run_calibration(force=args.force_calib)

    if stage == 'test':
        # 优先用真正落盘的那份表；表不在（或被删了）才退回按状态重算
        try:
            pair = load_exported_reverse_pair(img_size)
            print(f'批量测试使用已导出的表: {DIR_TABLE / "undistort_ipm" / "reverse"}')
        except SystemExit as exc:
            print(f'未找到可用的导出表（{exc}），改为按 ipm_state.json 重算。')
            pair = rebuild_reverse_map_from_state(K, D, Knew, img_size)
        batch_test(pair)
        print('\n批量测试完成。')
        return

    if stage == 'calib':
        print('\n相机标定阶段完成。')
        return

    # ---- 逆透视标定：ipm / tables / all

    # 只有 tables 阶段允许完全复用上次交互的结果；带 --quad 时一律按新参数重算。
    state = None if quad is not None else load_ipm_state()
    if stage == 'tables' and state is not None:
        print(f'\n复用已有的逆透视标定结果: {ipm_state_path()}')
        H = np.asarray(state['H'], dtype=np.float64).reshape(3, 3)
        H0 = np.asarray(state['H0'], dtype=np.float64).reshape(3, 3)
        sign = float(state.get('horizon_sign', 1.0))
        scale = float(state.get('scale_px_per_cm', 1.0))
        max_scale = float(state.get('max_scale_px_per_cm', 0.0))
        over_crop = max_scale > 0 and scale > max_scale
        if over_crop:
            print(f'警告: scale={scale:.3f} 超过不裁切视野的上限 {max_scale:.3f} px/cm，'
                  '导出的表已裁掉部分有效视野。')
        pair = export_all(K, D, Knew, H, H0, sign,
                          dict(state, over_crop=bool(over_crop),
                               max_range_cm=MAX_RANGE_CM,
                               max_lateral_cm=MAX_LATERAL_CM),
                          img_size)
        batch_test(pair)
        print('\n全部完成（复用已有逆透视标定）。')
        return

    src_path = discover_ipm_source(args.ipm_source)
    if src_path is None:
        if stage == 'tables':
            raise SystemExit('--stage tables 需要 --quad，或先跑一次 --stage ipm。')
        raise SystemExit('未找到逆透视标定原图。把地面标定照放进 ipm_input/，'
                         '或用 --ipm-source 指定。')
    src = safe_imread(src_path)
    if src is None:
        raise SystemExit(f'无法读取逆透视标定原图: {src_path}')
    print('逆透视标定原图:', src_path)
    src = normalize_camera_image(src, img_size, '逆透视标定原图')

    undist = cv2.undistort(src, K, D, None, Knew)
    safe_imwrite(UNDIST_RESULT, undist)
    print('去畸变图已保存:', UNDIST_RESULT)

    result = run_ipm_calibration(undist, img_size, args, quad)
    if result is None:
        return
    calibrator, phys_w, phys_h = result
    H = calibrator.H
    assert_invertible('H', H)

    # 结果图与状态都不在这里写：它们是这一轮产物的一部分，要和矩阵、查找表
    # 一起在导出成功后落盘，否则导出失败会留下"新结果图配旧矩阵"的组合——
    # 不影响 C 端，但会让人以为导出成了。只有 --stage ipm 不导出时才就地写。
    ipm_state = build_ipm_state(calibrator, phys_w, phys_h, img_size, src_path)

    over_crop = calibrator.is_over_crop()
    if over_crop:
        print(f'警告: scale={calibrator.scale:.3f} 超过不裁切视野的上限 '
              f'{calibrator.max_scale:.3f} px/cm，导出的表已裁掉部分有效视野。'
              '如需保留完整视野，重跑并按 f 吸附到 max_scale。')

    if stage == 'ipm':
        safe_imwrite(IPM_RESULT, calibrator.birdview)
        print('去畸变逆透视结果图已保存:', IPM_RESULT)
        save_ipm_state(calibrator, phys_w, phys_h, img_size, src_path)
        print('\n逆透视标定阶段完成。运行 --stage tables 可继续导出矩阵与查找表。')
        return

    pair = export_all(K, D, Knew, H, calibrator.H0, calibrator.sign, {
        'H0': calibrator.H0.tolist(),
        'phys_w_cm': phys_w,
        'phys_h_cm': phys_h,
        'scale_px_per_cm': calibrator.scale,
        'max_scale_px_per_cm': calibrator.max_scale,
        'over_crop': bool(over_crop),
        'anchor_x': calibrator.anchor_x,
        'anchor_y': calibrator.anchor_y,
        'heading_deg': calibrator.heading,
        'src_quad_tl_tr_bl_br': calibrator.corners.tolist(),
        'horizon_sign': calibrator.sign,
        'max_range_cm': MAX_RANGE_CM,
        'max_lateral_cm': MAX_LATERAL_CM,
    }, img_size, ipm_state=ipm_state)

    # 导出已经成功，这时才写结果图：它和矩阵、查找表属于同一批产物
    safe_imwrite(IPM_RESULT, calibrator.birdview)
    print('去畸变逆透视结果图已保存:', IPM_RESULT)
    batch_test(pair)

    print('\n全部完成。')


if __name__ == '__main__':
    try:
        main()
    finally:
        cv2.destroyAllWindows()
