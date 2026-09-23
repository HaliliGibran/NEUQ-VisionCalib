"""NEUQ 视觉标定工具的本地 Web 控制台——主算法外面的一层薄壳。

它把 neuq_vision_calib.py 的能力接到浏览器：素材导入、相机标定、逆透视四点拖拽、
实时 BirdView 预览、矩阵与查找表导出、批量测试。HTTP 层只负责校验请求、保存短期
交互状态和整理响应；几何、事务与文件格式仍直接调用主脚本。这样 Web 和 CLI 只是
同一条流水线的两个入口，不会各自产生一套“看起来差不多”的 H 或 LUT。

启动（在工程根目录下执行）:
    python src/webui/server.py            # 默认 http://127.0.0.1:8770
    python src/webui/server.py --port 9000 --no-browser

也可以直接双击工程根目录下的 start_webui.bat，或运行 python app.py。
用户数据统一读写 <工程根>/data/ 下的各个子目录。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import shutil
import sys
import threading
import time
import traceback
import webbrowser
import zipfile
from contextlib import redirect_stdout, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

FROZEN = getattr(sys, 'frozen', False)

# 让 server.py 能直接以脚本方式运行，同时找到上一级目录里的主脚本。
# 打包后主脚本已经作为模块并进 exe，这个 sys.path 补丁既没必要也可能指错地方。
if not FROZEN:
    # 本文件位于 <工程根>/src/webui/，把 src/ 挂上去才能 import 到主脚本。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 依赖检查放在最前面，且给出可操作的指引。用错 Python 解释器是这套工具最常见的
# 启动失败原因，默认的 ImportError traceback 对新用户毫无帮助。
try:
    import cv2
    import numpy as np

    import neuq_vision_calib as core
except ImportError as exc:
    raise SystemExit(
        f'\n启动失败: 缺少依赖 ({exc})\n'
        f'当前解释器: {sys.executable}\n\n'
        '这通常意味着用了一个没装 opencv-python 的 Python。两种解决办法：\n\n'
        '  1) 换成已装好依赖的解释器运行：\n'
        '     <装了依赖的 python> src\\webui\\server.py\n\n'
        '  2) 或者给当前解释器装上依赖：\n'
        f'     "{sys.executable}" -m pip install -r requirements.txt\n\n'
        '也可以直接双击工程根目录下的 start_webui.bat，它会自动挑解释器并补依赖。\n') from None


def static_dir() -> Path:
    """前端静态文件所在目录。

    打包后这些文件被 PyInstaller 解包到 sys._MEIPASS 下，和 exe 不在一起，
    所以资源目录和用户数据目录（core.SCRIPT_DIR）必须分开取。
    """
    bundle = getattr(sys, '_MEIPASS', None)
    if bundle:
        return Path(bundle) / 'webui' / 'static'
    return Path(__file__).resolve().parent / 'static'


STATIC_DIR = static_dir()

# 服务端只有一份进程内状态：工具只监听本机、面向单人使用，用锁串行化计算请求即可。
# 各字段寿命刻意不同：
#   K/D/img_size       从 calib.json 载入或重标得到；Knew 再由 K/D/UNDIST_ALPHA 现场计算；
#                      四者缓存到服务退出或再次标定
#   src_*              只属于当前选中的 IPM 原图，换图或重标就整体清空；
#   ipm_results        只属于本进程里最近一次“导出后自动批测”，不拿磁盘旧文件冒充。
# 真正需要跨进程复用的事实都在 calib.json / ipm_state.json / matrices.json，不依赖 STATE。
LOCK = threading.Lock()
STATE: dict = {
    'K': None,          # 相机内参：归一化相机坐标 -> 原始像素
    'D': None,          # 原始镜头的畸变系数
    'Knew': None,       # 去畸变输出所采用的像素坐标系
    'img_size': None,   # 标定分辨率 (W, H)
    'src_path': None,   # 当前 IPM 原图路径
    'src_raw': None,    # 当前原始畸变图
    'src_undist': None, # 当前 Knew 去畸变图；四点在这个坐标系中拖拽
    # 上一次导出后自动批测的成果记录：[{name, role, raw, birdview}]，由 batch_test
    # 的**实际返回**填充（见 record_ipm_results）。逆透视成果画廊只读它，不去扫目录
    # 按 stem 反查配对——那正是标定画廊踩过的坑。
    'ipm_results': [],
}

# 当前 HTTP 服务实例，供"退出"接口调用 shutdown()（见 api_shutdown）
HTTPD = None


# ---------------------------------------------------------------- 工具

def encode_jpeg(img: np.ndarray, quality: int = 88) -> str:
    """把 BGR 图像编码成 data URL，供前端直接塞进 <img>/canvas。"""
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError('JPEG 编码失败')
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')


def capture(func, *args, **kwargs):
    """执行主脚本的函数并捕获它打印的日志，返回 (结果, 日志文本)。"""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()


def safe_rel(path: Path) -> str:
    """把绝对路径转成相对工程根目录的字符串，便于前端展示。"""
    try:
        return str(path.relative_to(core.SCRIPT_DIR))
    except ValueError:
        return str(path)


def api_shutdown() -> dict:
    """让服务主动退出。

    Windows 控制台里按 Ctrl+C，信号会同时打到 python 和等着它的 cmd.exe，
    后者会追问一句 "Terminate batch job (Y/N)?" —— 双击 start_webui.bat 的
    用户经常被这句卡住。给浏览器留个体面的退出按钮，就不必再碰 Ctrl+C 了。
    """
    def stop() -> None:
        # 必须等响应发完再关，否则前端收不到回包，看起来像崩了
        if HTTPD is not None:
            HTTPD.shutdown()

    threading.Timer(0.5, stop).start()
    return {'message': '服务即将退出'}


def ensure_calibration() -> None:
    """确保内存里有 K/D/Knew；K/D 可读 calib.json，Knew 始终按当前视角参数计算。"""
    if STATE['K'] is not None:
        return
    K, D, Knew, img_size = core.load_or_run_calibration()
    STATE.update(K=K, D=D, Knew=Knew, img_size=img_size)


def make_calibrator(quad, phys_w, phys_h, anchor_x, anchor_y, heading, scale=None,
                    target_width=None, target_forward=None):
    """按给定参数构造标定器并算好 H 与 BirdView。

    直接复用主脚本的 IpmCalibrator，四条线由四点反推，因此网页上拖出来的结果与
    交互窗口里拖出来的走的是同一套 build_corners / compute_homography。
    """
    tl, tr, bl, br = (np.asarray(p, dtype=np.float64) for p in quad)
    cal = core.IpmCalibrator(STATE['src_undist'], phys_w, phys_h)
    cal.line_points = np.array([[tl, tr], [bl, br], [tl, bl], [tr, br]], dtype=np.float64)
    cal.line_default = cal.line_points.copy()
    # __init__ 会从模块级常量取初值，这里按请求覆盖掉
    cal.anchor_x = float(anchor_x)
    cal.anchor_y = float(anchor_y)
    cal.heading = float(heading)
    # 目标地面范围要在 recompute 之前设好：recompute 里的自动布局就是按它解的
    if target_width is not None:
        cal.target_width_cm = float(target_width)
    if target_forward is not None:
        cal.target_forward_cm = float(target_forward)
    if scale is not None:
        cal.scale = max(core.MIN_SCALE, float(scale))
        cal.scale_initialized = True
        cal.layout_mode = 'manual'
    cal.recompute()
    return cal


def guess_quad(img: np.ndarray) -> list:
    """给四点拖拽提供一个粗略起点。

    当前智能车赛道通常是深色底面上的浅色赛道，十字路口区域往往属于画面中
    较大的高亮连通区域，因此这里用 Otsu 二值化 + 最大亮连通域的凸包估计四角。

    这只是交互初值，不负责自动完成精确标定；最终四个角点仍由用户根据
    十字路口的实际边界手动确认。
    """
    w, h = img.shape[1], img.shape[0]
    fallback = [[w * 0.12, h * 0.35], [w * 0.88, h * 0.32],
                [w * 0.02, h * 0.97], [w * 0.98, h * 0.97]]
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((25, 25), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return fallback
        hull = cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2)
        if hull.shape[0] < 4:
            return fallback
        pts = hull.astype(np.float64)
        s, d = pts[:, 0] + pts[:, 1], pts[:, 0] - pts[:, 1]
        quad = np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                         pts[np.argmin(d)], pts[np.argmax(s)]])
        quad = core.order_corners_tl_tr_bl_br(quad)
        if core.quad_area(quad) < 0.02 * w * h or not core.is_convex_quad(quad):
            return fallback
        return [[float(p[0]), float(p[1])] for p in quad]
    except cv2.error:
        return fallback


# ---------------------------------------------------------------- 各接口实现

def _state_pair(key_a: str, key_b: str,
                default: tuple[float, float]) -> tuple[float, float]:
    """从已保存的 ipm_state 里取一对正数；取不到或非正就用默认值。

    优先级只在服务端判一次。若让前端自己"有历史就用历史、否则用 HTML 里的 value"，
    默认值一改就会出现两套语义：Python 侧 fallback 与页面上的 value 各说一套，
    而用户上次实测/调好的那一组还可能被新默认顶掉。
    """
    state = core.load_ipm_state() or {}
    try:
        a = float(state[key_a])
        b = float(state[key_b])
    except (KeyError, TypeError, ValueError):
        return default
    if a <= 0 or b <= 0:
        return default
    return a, b


def table_grid_options() -> list:
    """查找表可选的降采样倍率清单，供界面渲染下拉项。

    尺寸基准取标定分辨率：本轮进程里的 `STATE['img_size']` 优先，否则读 `calib.json`。
    两者都拿不到就返回空列表——可用倍率**只能**由源图尺寸决定，凭空猜一个 1280x720
    会让界面列出一批对当前工程并不成立的选项。标定完成后状态会刷新，届时自然有值。

    只给"倍率 + 结果网格"两项，页面不做任何除法：1280x720 下用户能看到的就是
    1x/2x/4x/5x/8x/10x/16x，压根没有 320x240 这个选项。
    """
    size = STATE['img_size']
    if not size:
        try:
            loaded = core.load_calibration()
            size = loaded[2] if loaded else None
        except (SystemExit, OSError, ValueError, KeyError, TypeError, IndexError):
            size = None
    if not size:
        return []
    w, h = int(size[0]), int(size[1])
    return [{'factor': n, 'width': w // n, 'height': h // n}
            for n in core.table_grid_factors(w, h)]


def initial_phys_size() -> tuple[float, float]:
    """网页上「物理尺寸」两个输入框的初值：已保存的 ipm_state 优先于默认值。

    ipm_state 里的尺寸是上一次真的量过、并且已经出表的那一组，只要它是正数就采信。
    """
    return _state_pair('phys_w_cm', 'phys_h_cm', (core.PHYS_W_CM, core.PHYS_H_CM))


def initial_target_window() -> tuple[float, float]:
    """网页上「目标地面范围」两个输入框的初值：(横向宽度, 前向深度)。

    与物理尺寸同一套口径。早期 ipm_state 没有这两个键，取不到就退回模块默认，
    不从其它尺寸字段猜测。
    """
    return _state_pair('target_width_cm', 'target_forward_cm',
                       (core.TARGET_WIDTH_CM, core.TARGET_FORWARD_CM))


def api_status() -> dict:
    """返回前端完整状态快照：目录、标定、来源闸门、IPM 与导出物。

    前端的 ``refreshStatus()`` 把这里当作唯一同步入口。按钮动作完成后重新读取快照，
    而不是在浏览器里猜“后端大概改了什么”；刷新页面与刚完成一次操作因此会得到
    同一种状态表达。
    """
    rows = (
        ('calib_input', core.DIR_CALIB_IN, '相机标定照片'),
        ('calib_data', core.DIR_CALIB_DATA, '标定结果 calib.json'),
        ('calib_preview', core.DIR_CALIB_PREVIEW, '标定图去畸变验收'),
        ('ipm_input', core.DIR_IPM_IN, '逆透视标定原图'),
        ('ipm_output', core.DIR_IPM_OUT, '去畸变图 + 俯视图结果'),
        ('matrix', core.DIR_MATRIX, '六矩阵与逆透视状态'),
        ('lookup_table', core.DIR_TABLE,
         '两类查找表（去畸变、去畸变+逆透视），每类各含正向/反向，共 4 组 LUT'),
        ('test_input', core.DIR_TEST_IN, '批量测试输入'),
        ('test_output', core.DIR_TEST_OUT, '批量测试输出'),
    )
    dirs = []
    for name, path, desc in rows:
        files = [p for p in path.rglob('*') if p.is_file()] if path.is_dir() else []
        dirs.append({
            'name': name,
            'desc': desc,
            'exists': path.is_dir(),
            'files': len(files),
            'images': sum(1 for p in files if p.suffix.lower() in core.IMAGE_SUFFIXES),
        })

    calib = None
    if core.CALIB_JSON.is_file():
        try:
            data = json.loads(core.CALIB_JSON.read_text(encoding='utf-8'))
            K = np.asarray(data['camera_matrix'], dtype=np.float64).reshape(3, 3)
            calib = {
                'width': int(data['image_width']),
                'height': int(data['image_height']),
                'fx': float(K[0, 0]), 'fy': float(K[1, 1]),
                'cx': float(K[0, 2]), 'cy': float(K[1, 2]),
                'dist': [float(v) for v in np.asarray(data['dist_coeffs']).ravel()],
                'hfov': float(2 * np.degrees(np.arctan2(data['image_width'] / 2.0, K[0, 0]))),
            }
        except (KeyError, ValueError, json.JSONDecodeError):
            calib = None

    # 三种格式成对校验：txt/bin 必须 W 和 H 同时在，C 格式要有 Map.h。
    # 只看"文件名出现过"会把只写了半张表（MapW.txt 有、MapH.txt 没写成功）
    # 的目录误判成已完成。
    def pair_ready(folder) -> bool:
        if (folder / 'MapW.txt').is_file() and (folder / 'MapH.txt').is_file():
            return True
        if (folder / 'MapW.bin').is_file() and (folder / 'MapH.bin').is_file():
            return True
        return (folder / 'Map.h').is_file()

    tables_ready = all(
        pair_ready(core.DIR_TABLE / name / direction)
        for name in ('undistort', 'undistort_ipm')
        for direction in ('forward', 'reverse'))

    phys_w, phys_h = initial_phys_size()
    target_w, target_f = initial_target_window()

    # 产物是否还属于当前标定。只判定不删文件——与素材库那套闸门同风格。
    saved_ipm = core.load_ipm_state()
    ipm_basis_stale = (core.calibration_basis_stale(saved_ipm.get('calibration_basis_hash'))
                       if saved_ipm else None)
    tables_basis_stale = (core.calibration_basis_stale(core.exported_basis_hash())
                          if tables_ready else None)


    # 报告上次导出用的表格式，让"交付给 C 端的到底是什么"一眼可见
    last_table = None
    if core.MATRIX_JSON.is_file():
        try:
            data = json.loads(core.MATRIX_JSON.read_text(encoding='utf-8'))
            last_table = {
                'format': data.get('table_format', 'txt'),
                'fixed_point': data.get('table_fixed_point'),
                'size_wh': data.get('table_size_wh'),
            }
        except (json.JSONDecodeError, OSError):
            last_table = None

    return {
        'root': str(core.SCRIPT_DIR),
        'dirs': dirs,
        'calib': calib,
        'calib_board': core.calib_board_meta(),
        'board': board_payload(),
        # 素材是否按旧规格分拣的，必须常驻可见：只在"应用规格"那一刻回显一次的话，
        # 刷新页面或重启服务之后警告就消失了，而脏状态还原样留着。
        'material_stale': material_warning(),
        # 素材导入事务的残留：与后端闸门读同一份事实（core.material_transaction_state）。
        # 刻意不缩成一个 material_pending 布尔——界面要分得清"不可判定，连标定都拒绝"
        # 和"只剩暂存目录，旧库完好、只是不能再 replace"这两档。
        'material_transaction': core.material_transaction_state(),

        'ipm_state': saved_ipm,
        # 这两条不是“文件在不在”，而是“它还属于当前标定吗”。stale 时前端不恢复
        # 旧四点，后端也拒绝拿旧表跑批测——产物本身一个都不删。
        'ipm_basis_stale': ipm_basis_stale,
        'tables_basis_stale': tables_basis_stale,
        # 物理尺寸初值：优先级（历史 > 默认）在服务端判完再给前端，页面直接照填
        'phys_init': {'w': phys_w, 'h': phys_h},
        # 目标地面范围初值，同一套优先级。这是**构图目标**，不是有效性边界。
        'target_init': {'width_cm': target_w, 'forward_cm': target_f},
        # 查找表允许的降采样倍率。由服务端按标定分辨率算好，页面只负责渲染下拉项——
        # 前端自己拼宽高就会出现 320x240 那种非等比组合，而那会破坏公制纵横比。
        'table_grid_options': table_grid_options(),

        'ipm_candidates': [p.name for p in core.list_images(core.DIR_IPM_IN)],
        'ipm_source': safe_rel(STATE['src_path']) if STATE['src_path'] else None,
        'tables_ready': tables_ready,
        'last_table': last_table,
        'photo_counts': api_import_status(),
        'preview_count': len(core.list_images(core.DIR_CALIB_PREVIEW)),
        # 第 3 步「查看逆透视成果」按钮的启用条件与张数。只认本轮进程里真实生成过的
        # 配对记录：没有记录时按钮保持 disabled，绝不让界面凭目录里的文件猜成果。
        'ipm_result_count': len(STATE['ipm_results']),
    }


def parse_multipart(headers, body: bytes):
    """解析 multipart/form-data，返回 (普通字段, [(字段名, 文件名, 字节])。

    标准库里已经没有现成的 multipart 解析器了（cgi 在 3.13 被移除），
    为一个上传接口引入第三方依赖又不值当。这里只处理浏览器 FormData
    实际会生成的那种格式，够用即可。
    """
    ctype = headers.get('Content-Type') or ''
    if not ctype.lower().startswith('multipart/form-data'):
        raise ValueError('上传需要 multipart/form-data 请求')
    boundary = None
    for seg in ctype.split(';'):
        part = seg.strip()          # 不覆盖循环变量本身，避免读起来像在改迭代状态
        if part.startswith('boundary='):
            boundary = part[len('boundary='):].strip('"').encode('utf-8')
    if not boundary:
        raise ValueError('请求头里缺少 boundary')

    fields: dict = {}
    files = []
    for chunk in body.split(b'--' + boundary):
        piece = chunk.strip(b'\r\n')
        if not piece or piece == b'--':
            continue
        head, sep, data = piece.partition(b'\r\n\r\n')
        if not sep:
            continue
        head_s = head.decode('utf-8', 'replace')
        name_m = re.search(r'name="([^"]*)"', head_s)
        file_m = re.search(r'filename="([^"]*)"', head_s)
        name = name_m.group(1) if name_m else ''
        if file_m:
            files.append((name, file_m.group(1), data))
        else:
            fields[name] = data.decode('utf-8', 'replace')
    return fields, files


def api_upload_import(fields: dict, files: list) -> dict:
    """把浏览器选中的素材原样上传到 data/import/<文件夹名>/ 并导入。

    浏览器出于安全只肯给出文件内容和相对路径，绝不给真实磁盘路径，所以
    "选了文件夹"这件事只能靠把文件真的传上来完成——让服务端拿一个文件夹名
    去磁盘上猜位置，改个名、挪个地方就必然失败。
    """
    _apply_board_if_given(fields)
    if not files:
        raise ValueError('没有选中任何文件。')

    first_rel = files[0][1].replace('\\', '/')
    folder = (fields.get('name') or '').strip() or first_rel.split('/')[0]
    folder = folder.replace('\\', '/').strip('/').split('/')[0]
    if not folder or folder in ('.', '..') or any(c in folder for c in ':*?"<>|'):
        raise ValueError(f'素材文件夹名不合法: {folder!r}')

    dest = core.DIR_IMPORT / folder
    dest.mkdir(parents=True, exist_ok=True)

    written = 0
    for _field, rel, data in files:
        parts = [p for p in rel.replace('\\', '/').split('/')
                 if p not in ('', '.', '..')]
        if parts and parts[0] == folder:
            parts = parts[1:]           # 去掉最外层那个与文件夹同名的目录
        if not parts:
            continue
        target = dest.joinpath(*parts).resolve()
        # 目录穿越防护：落点必须在 dest 之内
        if dest.resolve() not in target.parents:
            raise ValueError(f'非法的相对路径: {rel}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written += 1

    if not written:
        raise ValueError('没有可写入的图片文件（只支持 jpg/jpeg/png/bmp）。')

    # mode 必须跟着走：之前上传分支固定写死 'add'，于是"覆盖整个素材库"
    # 只在填路径导入时生效，走上传时静默变回增量添加，界面和实际行为不一致。
    mode = (fields.get('mode') or 'add').strip()
    if mode not in ('add', 'replace'):
        raise ValueError(f'导入模式需为 add 或 replace，收到 {mode!r}')

    summary, log = capture(core.import_dataset, dest, False, mode)
    if not isinstance(summary, dict):
        summary = {}
    return {'log': f'已上传 {written} 个文件 → data/import/{folder}/'
                   f'（模式 {mode}）\n' + log,
            'summary': summary, 'folder': folder, 'uploaded': written}


def material_warning() -> str | None:
    """界面上要常驻显示的素材库告警：分拣依据与当前规格不符，或根本无从判断。

    core 里这两条判据分别拦在 stale 检查与增量导入闸门上，界面不必区分——
    对用户来说都是"这批素材不能再往下用了，请重新导入"。
    """
    return core.material_stale_reason() or core.material_basis_unknown_reason()


def board_payload() -> dict:
    """当前生效的标定板规格，供界面回显。"""
    b = core.BOARD
    return {
        'squares_x': b.squares_x,
        'squares_y': b.squares_y,
        'square_size_mm': b.square_size_mm,
        'corners_x': b.corners[0],
        'corners_y': b.corners[1],
        'label': b.label,
        'is_default': b == core.DEFAULT_BOARD,
    }


def board_from_request(body: dict) -> dict | None:
    """从请求里取出标定板规格，两种形状都认。

    JSON 接口传 {"board": {...}}，而 multipart 上传没法嵌套，只能扁平传
    squares_x/squares_y。不归一的话就会出现"填路径导入时规格生效、点上传时
    静默用旧规格"——这条漂移真的太隐蔽了，所以在入口处一次性抹平。
    """
    nested = body.get('board')
    if nested:
        return nested
    if body.get('squares_x') is not None and body.get('squares_y') is not None:
        return body
    return None


def apply_board(body: dict) -> dict:
    """设置标定板规格。

    规格决定"一张图算棋盘照还是地面照"，所以必须能在导入之前设定；
    导入/标定接口也会带上它，避免用户改了网页却没生效。
    """
    if body.get('squares_x') in (None, '') or body.get('squares_y') in (None, ''):
        raise ValueError('需要给出 squares_x 与 squares_y。')
    # 边长不能用 `or 默认值` 兜底：显式传 0 会被静默换成 20，
    # 错误就被吞了。让 0 走到规格校验里去，报明确的错。
    raw_mm = body.get('square_size_mm')
    size_mm = (core.DEFAULT_BOARD.square_size_mm
               if raw_mm in (None, '') else float(raw_mm))
    try:
        spec = core.CheckerboardSpec(int(body['squares_x']),
                                     int(body['squares_y']), size_mm)
    except ValueError as exc:
        raise ValueError(str(exc)) from None
    # persist=True：规格是项目级配置，必须落盘，否则重启后又回到默认，
    # 接着导入的新照片就会按另一套规格分类。
    core.configure_board(spec, persist=True)
    payload = board_payload()
    payload['material_stale'] = material_warning()
    # 事务告警也要跟着返回：改规格这一步不刷 /api/status，少了它界面上那条常驻
    # 红/黄条幅就会停在改规格之前的状态，直到下一次轮询才更正。
    payload['material_transaction'] = core.material_transaction_state()
    return payload


def api_board(body: dict) -> dict:
    """设置规格的接口包装。"""
    return apply_board(body)


def _apply_board_if_given(body: dict) -> None:
    """导入/标定请求里带了 board 就先应用，保证该轮操作用的是同一份规格。"""
    spec = board_from_request(body)
    if spec:
        apply_board(spec)


def api_import(body: dict) -> dict:
    """导入混合素材目录。"""
    _apply_board_if_given(body)
    raw = (body.get('dir') or '').strip()
    if not raw:
        raise ValueError('请填写素材目录。')
    mode = body.get('mode') or 'add'
    if mode not in ('add', 'replace'):
        raise ValueError(f'--import-mode 需为 add 或 replace，收到 {mode!r}')
    # 交给 core 统一解析：素材可能在工程根，也可能在 data/import/ 下
    src = core.resolve_import_dir(raw)
    summary, log = capture(core.import_dataset, src, bool(body.get('move')), mode)
    if not isinstance(summary, dict):
        # 兼容旧版 import_dataset 返回 None 的情况
        summary = {}
    return {'log': log, 'summary': summary}


def api_import_status() -> dict:
    """统计 calib_input 下三类照片的数量，供前端显示导入后的分布。"""
    def count(folder: Path) -> int:
        return len(core.list_images(folder)) if folder.is_dir() else 0

    return {
        'calib': count(core.DIR_CALIB_IN),
        'incomplete': count(core.DIR_CALIB_IN / core.INCOMPLETE_SUBDIR),
        'ipm': count(core.DIR_IPM_IN),
    }


def dir_backup() -> Path:
    """备份目录 —— 必须每次现取，不能在 import 时抄一份。

    `--root` 是在模块导入**之后**才通过 core.configure_paths() 改写的，那时
    抄下来的目录还指着工程自带的位置：备份会写进旧根，而清空的是新根的目录，
    越界检查也跟着放行。core.DIR_IMPORT 一直是现取的，这里保持一致。
    """
    return core.DIR_BACKUP

# 会被打包 + 清空的目录（均直接位于工程根）。
# 刻意不含 assets/checkerboard/（参考靶标，不是产物）、tools/、src/、主脚本本身，
# 也不含 data/import/ 下用户自己的原始素材。
BACKUP_FOLDERS = (
    'calib_input', 'ipm_input', 'test_input',              # 输入
    'calib_data', 'calib_preview', 'ipm_output',           # 标定产物
    'matrix', 'lookup_table', 'test_output',               # 导出产物
)

# 这些文件是用户提供的参考资料 / 原始数据，不是本工具生成的产物，
# 备份里会存一份，但清空时保留——它们无法由本工具重新生成。
PRESERVE_ON_CLEAR = (
    'matrix/legacy_matlab_reference.json',   # 历史 MATLAB 逆透视矩阵参照
    'calib_data/calibrationSession.mat',     # 用户的 MATLAB 标定会话原始文件
)


def _clear_project_folders() -> tuple:
    """清空输入/输出目录，返回 (清除数, 保留数, 各目录明细)。

    「备份并清空」和「仅清空」共用这一段：两者的清空语义必须完全一致，
    各写一份迟早会漂移（比如一边保留了参考资料、另一边忘了）。
    """
    present = [f for f in BACKUP_FOLDERS if (core.SCRIPT_DIR / f).is_dir()]
    keep = {str((core.SCRIPT_DIR / p).resolve()) for p in PRESERVE_ON_CLEAR}
    kept, removed, entries = 0, 0, []

    for folder in present:
        base = core.SCRIPT_DIR / folder
        n = 0
        for p in sorted(base.rglob('*'), reverse=True):
            if p.is_file():
                if str(p.resolve()) in keep:
                    kept += 1
                    continue
                p.unlink()
                removed += 1
                n += 1
            elif p.is_dir():
                # 目录里可能还有被保留的文件，删不掉就跳过
                with suppress(OSError):
                    p.rmdir()
        base.mkdir(parents=True, exist_ok=True)
        entries.append({'folder': folder, 'files': n})

    # 内存里的标定与逆透视状态一并作废，否则界面还显示着已经被清掉的标定
    STATE.update(K=None, D=None, Knew=None, img_size=None,
                 src_path=None, src_raw=None, src_undist=None, ipm_results=[])
    # 素材库空了，"这套素材按哪块棋盘分拣"这个事实也就不存在了。
    # 不清的话，下一次增量导入会拿一条对不上号的旧依据去判 stale。
    core.clear_material_board()
    return removed, kept, entries


def api_clear_all(body: dict) -> dict:
    """不备份，直接清空输入/输出目录。

    必须显式传 confirm=true 才执行。这个接口没有回退余地，光靠前端弹窗挡不住
    误触（脚本、重放、手滑的请求都可能直接打过来），所以服务端也要一道闸。
    """
    if not body.get('confirm'):
        raise ValueError('危险操作：需要在请求里显式带上 confirm=true 才会执行。')
    removed, kept, entries = _clear_project_folders()
    return {'cleared_files': removed, 'kept_files': kept, 'folders': entries}


def api_backup_clear(body: dict) -> dict:
    """把输入/输出产物打包成一个备份，然后清空它们，等待新素材导入。

    顺序不能反：**先打包、校验压缩包确实落盘且非空，再清空**。
    反过来的话一旦打包失败，用户的东西就没了。任何一步出问题都直接中止，不动原目录。
    """
    raw_name = (body.get('name') or '').strip()
    # 名字里禁止路径分隔符和上跳，避免写到 backups/ 之外
    if raw_name and (any(c in raw_name for c in '\\/:*?"<>|') or '..' in raw_name):
        raise ValueError('备份名不能含 \\ / : * ? " < > | 等字符。')
    stamp = time.strftime('%Y%m%d_%H%M%S')
    name = raw_name or stamp

    backup_root = dir_backup()
    target = (backup_root / name).resolve()
    if not target.is_relative_to(backup_root.resolve()):
        raise ValueError('备份路径越界。')
    if target.exists():
        raise ValueError(f'备份 {name} 已存在，换个名字或留空用时间戳。')

    present = [f for f in BACKUP_FOLDERS if (core.SCRIPT_DIR / f).is_dir()]
    if not present:
        raise ValueError('没有可备份的目录。')

    target.mkdir(parents=True, exist_ok=True)
    archive = target / 'data.zip'

    entries = []
    total = 0
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder in present:
            base = core.SCRIPT_DIR / folder
            files = [p for p in base.rglob('*') if p.is_file()]
            for p in files:
                zf.write(p, p.relative_to(core.SCRIPT_DIR).as_posix())
                total += p.stat().st_size
            entries.append({'folder': folder, 'files': len(files)})

    # 打包完先自检：压缩包存在、非空、能打开
    if not archive.is_file() or archive.stat().st_size == 0:
        raise SystemExit(f'打包失败：{archive} 没有正常写出，已中止，原目录未改动。')
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise SystemExit('打包失败：压缩包损坏，已中止，原目录未改动。')

    # 必须在清空之前取：这份快照回答的是"这批素材当时是按什么规格分类的"，
    # 而那件事只活在 project.json 里。光留着活的 project.json 不够 ——
    # 下一轮导入会把 material_set 改写掉，这一批的依据就永久查不到了。
    manifest = {
        'name': name,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'project_root': str(core.SCRIPT_DIR),
        'project_config': core.load_project_config(),
        'archive': archive.name,
        'archive_bytes': archive.stat().st_size,
        'source_bytes': total,
        'folders': entries,
    }
    (target / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    # ---- 到这里才真正清空
    removed, kept, entries = _clear_project_folders()

    return {'name': name, 'path': str(target), 'archive': str(archive),
            'archive_bytes': archive.stat().st_size, 'source_bytes': total,
            'cleared_files': removed, 'kept_files': kept, 'folders': entries,
            'manifest': manifest}


def api_backup_list() -> dict:
    """列出已有的备份，供界面回显。"""
    items = []
    backup_root = dir_backup()
    if backup_root.is_dir():
        for d in sorted(backup_root.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            man = d / 'manifest.json'
            info = {'name': d.name}
            if man.is_file():
                try:
                    m = json.loads(man.read_text(encoding='utf-8'))
                    info.update({'created': m.get('created'),
                                 'archive_bytes': m.get('archive_bytes'),
                                 'source_bytes': m.get('source_bytes')})
                except json.JSONDecodeError:
                    pass
            items.append(info)
    return {'items': items, 'count': len(items)}


def image_url(path: Path) -> str:
    """拼出 /api/image 的取图地址。

    路径统一用正斜杠再整体转义：serve_image 拿到后是 core.SCRIPT_DIR / rel，
    两种分隔符在 Windows 上都能拼对，但只有正斜杠在 URL 里读得懂。
    调用方可以再追加 &max=<像素> 限制最大边（缩略图与大图共用同一条地址）。
    """
    return '/api/image?rel=' + quote(safe_rel(path).replace('\\', '/'))


def pair_preview_images() -> list[dict]:
    """把 calib_input/ 的原图与 calib_preview/ 的去畸变图配成一组一组。

    配对只认磁盘上真实存在的文件：以原图为准建槽位，再拿预览图去填。
    命名规则（{原图名}_undist.jpg，见 export_undistort_previews）只作为"这张
    预览图属于谁"的线索，认不出来的、或者对应槽位已经被占了的，一律单独成组、
    原图侧留空——绝不允许一张预览图挤掉别人的位置，那会让界面把 A 的原图和
    B 的去畸变图摆在一起，而这正是"并排对照"最不能出的错。
    """
    suffix = '_undist'
    slots: dict[str, dict] = {}
    for p in core.list_images(core.DIR_CALIB_IN):
        slots[p.stem] = {'name': p.stem, 'raw': p, 'undistorted': None}

    previews = core.list_images(core.DIR_CALIB_PREVIEW)
    # 带 _undist 后缀的先认领：calib_preview/ 里万一混进一份与原图同名的拷贝，
    # 不能让它抢在真正的去畸变图前面占掉槽位。
    for q in sorted(previews, key=lambda q: (not q.stem.endswith(suffix), q.stem)):
        stem = q.stem[:-len(suffix)] if q.stem.endswith(suffix) else q.stem
        slot = slots.get(stem)
        if slot is not None and slot['undistorted'] is None:
            slot['undistorted'] = q
            continue
        # 配不上任何原图（原图被删了/改名了），或那张原图已经有去畸变图了：
        # 用文件名另开一组，键名带扩展名以免和原图槽位撞车。名字仍给去掉
        # _undist 的那个照片名——这一组的原图侧是空的，不存在配错的风险。
        slots[q.name] = {'name': stem, 'raw': None, 'undistorted': q}

    return [slots[key] for key in sorted(slots, key=lambda k: (slots[k]['name'], k))]


def api_preview_gallery() -> dict:
    """列出"原图 vs 去畸变"对照组，配对关系由服务端给定。

    前端不再自己拼 calib_preview/xxx_undist.jpg 去猜谁配谁：名字规则一旦不成立
    （改过名、扩展名不同、少了一张），前端猜出来的地址要么 404 要么张冠李戴。
    这里直接给出两侧的取图地址，缺哪侧就是 null，让界面显示空态。
    """
    items = [{
        'name': slot['name'],
        'raw_url': image_url(slot['raw']) if slot['raw'] is not None else None,
        'undistorted_url': (image_url(slot['undistorted'])
                            if slot['undistorted'] is not None else None),
    } for slot in pair_preview_images()]
    return {
        'items': items,
        'count': len(items),
        'paired': sum(1 for it in items if it['raw_url'] and it['undistorted_url']),
        'missing_raw': sum(1 for it in items if not it['raw_url']),
        'missing_undistorted': sum(1 for it in items if not it['undistorted_url']),
    }


def api_ipm_result_gallery() -> dict:
    """列出逆透视成果对照组（原图 ↔ BirdView），配对关系由服务端给定。

    数据源是 STATE['ipm_results'] —— 上一次导出成功后自动批测**实际生成**的那批
    配对，不是去扫 test_output/_auto_ipm/ 再按文件名反查。差别在于：被跳过的图
    （读不出来、宽高比不同）在这里根本不会出现，而按目录反查一定会把"少了一张
    结果"猜成别人的结果，或者让整组凭空消失（见 pair_preview_images 的注释）。

    第一项是真正参与四点标定的那张基准图（role=calibration，右侧就是本次导出的
    IPM_RESULT）；其后是没参与标点的测试图（role=test）。
    """
    items = [{
        'name': rec['name'],
        'role': rec['role'],
        'raw_url': image_url(rec['raw']) if rec['raw'] is not None else None,
        'birdview_url': (image_url(rec['birdview'])
                         if rec['birdview'] is not None else None),
    } for rec in STATE['ipm_results']]
    return {'items': items, 'count': len(items)}


def api_calibrate(body: dict) -> dict:
    """跑相机标定；不勾选"强制重新标定"时优先复用已有的 calib.json。

    返回 fit_result 是给前端画柱状图用的——逐帧误差、接受/被剔除的清单、整体 RMS。
    """
    _apply_board_if_given(body)
    err = body.get('max_reproj_err')
    core.MAX_REPROJ_ERR = float(err) if err not in (None, '') else None

    # 复用旧标定前先对规格：已有 calib.json 是用另一块棋盘标的话，界面会显示
    # "当前 12x9、运行标定成功"，实际却复用了 9x7 的旧参数。数学没错，
    # 但 provenance 被写错，事后完全说不清。这里明确拒绝，不静默复用。
    if not body.get('force') and core.CALIB_JSON.is_file():
        conflict = core.board_conflict(core.calib_board_meta())
        if conflict:
            raise ValueError(conflict)

    fit = None
    buf = io.StringIO()
    with redirect_stdout(buf):
        if body.get('force') or not core.CALIB_JSON.is_file():
            spec = core.BOARD     # 显式捕获：provenance 跟着本次实际用的板走
            K, D, img_size, fit = core.calibrate_camera(board=spec)
            # calib.json 与 calib_preview/ 由 calibrate_camera 内部作为一个事务提交；
            # 这里再单独写一次 calib.json 就会留下"新预览 + 旧参数"的窗口。
            # 重标之后旧成果一律作废：那些 BirdView 是用**上一版** K/D 与上一份 LUT
            # 算的，与新标定已经不是一套。宁可按钮回到 disabled。
            STATE['ipm_results'] = []
        else:
            print(f'复用已有标定文件 {core.CALIB_JSON.name}'
                  '（勾选"强制重新标定"可从头再算一遍）。')
            K, D, img_size = core.load_calibration()
        Knew = core.resolve_new_camera_matrix(K, D, img_size)
        core.assert_invertible('K', K)
        core.assert_invertible('Knew', Knew)
    STATE.update(K=K, D=D, Knew=Knew, img_size=img_size,
                 src_path=None, src_raw=None, src_undist=None)

    response = {'log': buf.getvalue(), 'size': list(img_size)}
    if fit is not None:
        response['fit'] = fit
    return response


def api_source(body: dict) -> dict:
    """选定 IPM 原图，统一到标定分辨率并去畸变，再返回一个可拖拽起点。

    四点坐标属于 ``Knew`` 去畸变图，而不是原始畸变图。若素材依据已经 stale 或未知，
    函数在读图前就拒绝：候选列表本身是旧棋盘规格分拣出的结论，点名其中一张并不会
    让它自动重新可信。
    """
    # 界面上的候选列表就是 ipm_input/ 的分拣结果，规格换了它整体不可信：
    # 里面可能混着"新规格能认出是棋盘、旧规格认成地面"的照片。只在页面上飘一行
    # 红字是不够的，用户照样能点下去一路导出。core 侧只拦了自动取图，这里
    # 连"用户点选了某一张"也一并拦住，而且拦在标定检查之前——素材本身不能用了，
    # 先补标定也没有意义。
    # block_unknown=True：这里取的图必然来自 ipm_input/，而"它在这个目录里"本身
    # 就是过去某次分拣的结论。依据未知时无法证明它真是地面照，和 stale 一样要拦。
    # （相机标定那边可以放行，因为它逐张重新检棋盘，检不出会被剔除并报错。）
    core.require_material_basis('逆透视标定', block_unknown=True)
    ensure_calibration()
    name = (body.get('name') or '').strip()


    path = core.resolve_user_path(name, core.DIR_IPM_IN) if name else None
    if path is None or not path.is_file():
        path = core.discover_ipm_source()
    if path is None or not path.is_file():
        raise ValueError('没有可用的逆透视标定原图，请先在 ipm_input/ 放入照片。')

    raw = core.safe_imread(path)
    if raw is None:
        raise ValueError(f'无法读取 {path}')
    # 与命令行共用同一套尺寸契约：宽高比不同直接报错，不静默硬缩
    raw = core.normalize_camera_image(raw, STATE['img_size'], f'{path.name}')
    undist = cv2.undistort(raw, STATE['K'], STATE['D'], None, STATE['Knew'])
    STATE.update(src_path=path, src_raw=raw, src_undist=undist)

    return {
        'name': path.name,
        'rel': safe_rel(path),
        'width': int(undist.shape[1]),
        'height': int(undist.shape[0]),
        'image': encode_jpeg(undist, quality=92),
        'quad': guess_quad(undist),
    }


def _preview_inputs(body: dict):
    """从请求体里取出预览所需的全部参数。"""
    if STATE['src_undist'] is None:
        raise ValueError('请先选择一张逆透视标定原图。')
    quad = np.asarray(body.get('quad'), dtype=np.float64).reshape(4, 2)
    quad = core.order_corners_tl_tr_bl_br(quad)
    if not core.is_convex_quad(quad) or core.quad_area(quad) < core.MIN_QUAD_AREA_PX:
        raise ValueError('四点构成自交/非凸/退化的四边形，请调整。')
    phys_w = float(body.get('phys_w', core.PHYS_W_CM))
    phys_h = float(body.get('phys_h', core.PHYS_H_CM))
    if phys_w <= 0 or phys_h <= 0:
        raise ValueError('标定矩形的物理尺寸必须为正数。')
    target_w = float(body.get('target_width', core.TARGET_WIDTH_CM))
    target_f = float(body.get('target_forward', core.TARGET_FORWARD_CM))
    if target_w <= 0 or target_f <= 0:
        raise ValueError('目标地面范围的宽度与前向深度必须为正数。')
    scale = body.get('scale')
    return {
        'quad': quad,
        'phys_w': phys_w,
        'phys_h': phys_h,
        'target_width': target_w,
        'target_forward': target_f,
        'anchor_x': float(body.get('anchor_x', core.ANCHOR_X)),
        'anchor_y': float(body.get('anchor_y', core.ANCHOR_Y)),
        'heading': float(body.get('heading', core.HEADING_DEG)),
        'scale': float(scale) if scale not in (None, '') else None,
    }


def recommended_layout(cal) -> dict:
    """当前几何下的"自动推荐值"：anchor_y 与 scale 各一个，外加它是怎么来的。

    存在的理由有两个：一是给界面上的「重置」一个**间接**入口（前端只管"问一次、
    采用"，绝不把常量抄进 JS）；二是自动布局模式下 preview 与 commit 都从这里取
    参数，眼睛看到的 BirdView 与最终导出的 H/LUT 因此同源。

    口径是 IpmCalibrator.target_layout()：以逆透视坐标参考原点为 (0,0)，**目标地面
    范围**向前 target_forward_cm、横向关于参考原点对称且总宽 target_width_cm，贴住
    输出图底边并最大化装入，anchor_y 与 scale 一起解出来。若误把整幅 valid FOV 当
    构图目标，会把 ±300 cm 的地面压进 1280x720，48% 的输出像素来自不到 0.04 个
    源像素——整幅图是放射状拉丝。

    source 字段区分两种来源：
      'target_window'  算出了可行布局，anchor_y 与 scale 都是它给的；
      'fallback'       当前几何没有可行布局（窗口比画布还大、anchor_x 越界等）。这时
                       **保留当前 anchor_y**，只把 scale 退回旧的固定-anchor 口径：
                       full_fov_fit_scale 的 INIT_SCALE_RATIO 倍（为 0 时退回 1.0）。
                       语义是"联合布局求不出来，我不再擅自动你的纵向布局"，而不是
                       "顺便把 anchor_y 抹回出厂值"——无解的原因可能只是 anchor_x
                       靠边或某个临界几何，都推不出 anchor_y 必须等于 ANCHOR_Y。
                       之所以要有 fallback：预览不能因为"自动布局不可用"整个崩掉。
    """
    if cal.H0 is not None:
        fit = cal.target_layout()
        if fit is not None:
            ay_px, scale = fit
            return {'anchor_y': float(ay_px / (cal.h - 1)),
                    'scale': float(max(core.MIN_SCALE, scale)),
                    'source': 'target_window'}

    cap = cal.full_fov_fit_scale
    scale = max(core.MIN_SCALE, cap * core.INIT_SCALE_RATIO) if cap > 0 else 1.0
    # fallback 保留当前 anchor_y，不回退到模块常量。full_fov_fit_scale 本来就是按当前
    # anchor_y 算出来的，若返回时把 anchor 改成 ANCHOR_Y，这个 scale 就不再对应
    # 刚才那个诊断值了——那是契约错误，也正是"再跑一次 fallback 得到另一组值"
    # 这个不幂等现象的根源。保留 anchor_y 之后 recompute 不改变它，
    # 重复调用天然幂等。
    return {'anchor_y': float(cal.anchor_y), 'scale': float(scale),
            'source': 'fallback'}


def apply_auto_layout(cal) -> dict:
    """把推荐布局真正写进标定器并重算，返回用了哪一组值。

    preview 与 commit 共用这一个入口：只改 preview 会让"看到的 BirdView"和
    "导出的 H/LUT"分家，那比不做自动布局更糟。
    """
    rec = recommended_layout(cal)
    cal.anchor_y = rec['anchor_y']
    cal.scale = rec['scale']
    cal.scale_initialized = True
    cal.layout_mode = rec['source']
    cal.recompute()
    return rec


def api_preview(body: dict) -> dict:

    """实时预览：按当前四点与俯视图布局四自由度算 H，返回俯视图与标定矩形位置。"""
    p = _preview_inputs(body)
    cal = make_calibrator(p['quad'], p['phys_w'], p['phys_h'],
                          p['anchor_x'], p['anchor_y'], p['heading'], p['scale'],
                          p['target_width'], p['target_forward'])
    if cal.H is None or cal.birdview is None or cal.H0 is None:
        raise ValueError('当前四点无法构成有效单应，请调整。')

    # scale 为 None 就是自动布局模式：anchor_y 与 scale 都由服务端定，
    # 并且与 api_commit 走同一个 apply_auto_layout。
    rec = apply_auto_layout(cal) if p['scale'] is None else recommended_layout(cal)

    view = cal.birdview.copy()
    if view.ndim == 2:
        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    rect = core.apply_homography(cal.H, cal.corners)
    cv2.polylines(view, [rect[[0, 1, 3, 2]].astype(np.int32)], True, (0, 255, 0), 2)
    # anchor 十字正好就是逆透视坐标参考原点（去畸变图底边中点对应的地面点）：
    # 完整 H 会把那个点精确映到 anchor，所以不用再单独画一个标记。
    ax, ay = cal.anchor_px()
    cv2.drawMarker(view, (int(round(ax)), int(round(ay))), (0, 140, 255),
                   cv2.MARKER_CROSS, 18, 2)

    origin = core.ground_origin_meta(cal)
    return {
        'birdview': encode_jpeg(view, quality=85),
        'scale': float(cal.scale),
        'anchor_y': float(cal.anchor_y),
        # 诊断量，不是上限：完整容纳整幅有效视野所需的 scale。目标地面范围远小于有效视野
        # 时 scale 会明显大于它，那是**故意**裁掉远处与侧面，不是错误。
        'full_fov_fit_scale': float(cal.full_fov_fit_scale),
        'layout_mode': str(cal.layout_mode),
        'target_window': {'width_cm': float(cal.target_width_cm),
                          'forward_cm': float(cal.target_forward_cm)},
        'ground_origin': origin,
        'recommended': rec,

        'horizon_sign': float(cal.sign),
        'rect': [[float(v) for v in pt] for pt in rect],
        'valid_ratio': float(np.isfinite(rect).all()),
    }



def apply_table_options(body: dict) -> None:
    """把打表选项写回主脚本的模块级配置。

    这些量在命令行里是全局参数，网页上按次请求给出，所以每次导出前都要重设一遍，
    不能只设一次就放着——否则上一次的降采样尺寸会粘到下一次。
    """
    fmt = body.get('table_format')
    if fmt in ('txt', 'bin', 'c'):
        core.TABLE_FORMAT = fmt

    # 网页只给"降采样倍率"，不给宽高。让用户自己填两个数，就一定会出现 320x240
    # 这种 x 缩 4 倍、y 缩 3 倍的组合——那会把 16:9 的 BirdView 非等比压成 4:3，
    # 一个物理上 45x45 cm 的正方形在小图里变成宽高比 0.75 的竖长矩形，C 端拿它算
    # 角度、曲率、横向误差全部失真。倍率这种表示法连表达非法组合的能力都没有。
    factor = body.get('table_factor')
    if factor in (None, '', 1, '1'):
        core.TABLE_SIZE = None
    else:
        try:
            n = int(factor)
        except (TypeError, ValueError):
            raise ValueError('降采样倍率必须是整数。') from None
        w, h = STATE['img_size']
        if n not in core.table_grid_factors(w, h):
            allowed = '、'.join(f'{n2}x（{w // n2}x{h // n2}）'
                               for n2 in core.table_grid_factors(w, h))
            raise ValueError(f'{n}x 不是 {w}x{h} 的可用等比倍率。可选：{allowed}。')
        core.TABLE_SIZE = (w // n, h // n)

    fp = body.get('table_fixed_point')
    if fp not in (None, ''):
        try:
            fp = int(fp)
        except (TypeError, ValueError):
            raise ValueError('定点位数必须是整数。') from None
        if not 0 <= fp <= 15:
            raise ValueError('定点位数需在 0~15 之间。')
        core.TABLE_FIXED_POINT = fp


# 自动测试集的专属子目录名。它是程序生成、可整体刷新的缓存，所以必须待在自己的
# 子目录里：test_input/ 与 test_output/ 的根目录属于用户，本工具在那里不删不改不读。
AUTO_TEST_SUBDIR = '_auto_ipm'


def auto_test_dirs() -> tuple[Path, Path, Path, Path]:
    """自动测试集的四个目录：(输入, 输入暂存区, 输出, 输出暂存区)。现取，跟着工程根走。"""
    return (core.DIR_TEST_IN / AUTO_TEST_SUBDIR,
            core.DIR_TEST_IN / (AUTO_TEST_SUBDIR + '.staging'),
            core.DIR_TEST_OUT / AUTO_TEST_SUBDIR,
            core.DIR_TEST_OUT / (AUTO_TEST_SUBDIR + '.staging'))


def prepare_auto_test_input(exclude: Path | None) -> list[str]:
    """把 ipm_input/ 里除标点原图之外的候选刷进 test_input/_auto_ipm/。

    选中做四点标定的那张不复制：它是"已经拿来拟合"的图，用它验证等于自证。

    "先准备完整，再整体替换"：全部复制进 _auto_ipm.staging/ 成功之后才换装。
    复制到一半磁盘满的话暂存区被整个丢掉，旧的 _auto_ipm/ 一个字节都没动过——
    绝不会留下半套正式的自动测试集。换装直接复用导出那条路上的 core.commit_dirs
    （keep_staging=False：暂存区里都是拷贝，没有 --move 原件的顾虑）。
    """
    target, stage, _out, _out_stage = auto_test_dirs()
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    skip = exclude.resolve() if exclude is not None else None

    names: list[str] = []
    try:
        for p in core.list_images(core.DIR_IPM_IN):
            if skip is not None and p.resolve() == skip:
                continue
            shutil.copy2(p, stage / p.name)
            names.append(p.name)
    except BaseException:
        # 没能全部复制成功：暂存区作废，正式目录还没被碰过
        shutil.rmtree(stage, ignore_errors=True)
        raise

    core.commit_dirs(((stage, target),), what='自动逆透视测试集')
    return names


def record_ipm_results(done) -> list[dict]:
    """把 batch_test 的实际返回整理成画廊记录，并存进 STATE。

    第一项固定是参与四点标定的那张基准图：它的 BirdView 不在 test_output/ 里，
    而是随本次导出一起提交的 IPM_RESULT。其余各项严格来自 done 里的配对，
    被跳过的图不会出现——画廊因此不可能虚构出一个"成果"。
    """
    records: list[dict] = []
    src = STATE['src_path']
    if src is not None and Path(src).is_file() and core.IPM_RESULT.is_file():
        records.append({'name': Path(src).name, 'role': 'calibration',
                        'raw': Path(src), 'birdview': core.IPM_RESULT})
    for raw, birdview in done or []:
        records.append({'name': Path(raw).name, 'role': 'test',
                        'raw': Path(raw), 'birdview': Path(birdview)})
    STATE['ipm_results'] = records
    return records


def run_auto_batch(pair) -> dict:
    """刷新自动测试集并用刚导出的那份表跑一遍，返回 {ok, generated, error}。

    只能在导出事务提交成功之后调用（见 api_commit 里的顺序）：在那之前动测试集，
    导出一旦失败就会留下一套"对不上任何已交付表"的结果。

    输出侧和输入侧一样走"先写暂存区、全部完成再整体换装"，而不是往正式目录里
    增量写。否则上一轮成功 7 张、这一轮某张因比例不符被跳过时，那张旧的
    _birdview.jpg 会孤零零留在正式目录里——画廊看 done 不会展示它，但目录计数和
    用户手动翻文件时会看到一个对不上本轮 LUT 的旧结果。换装之后正式目录里就是
    本轮实际成果的精确集合。

    batch_test 返回的输出路径指向暂存区，换装后要改写成正式目录下的路径，
    再交给 record_ipm_results——否则画廊指向的是已经不存在的 .staging。
    """
    input_dir, _stage, output_dir, out_stage = auto_test_dirs()
    copied = prepare_auto_test_input(STATE['src_path'])
    print(f'自动测试集已刷新（{len(copied)} 张，不含标点原图）: {safe_rel(input_dir)}')

    shutil.rmtree(out_stage, ignore_errors=True)
    out_stage.mkdir(parents=True, exist_ok=True)
    try:
        done = core.batch_test(pair, input_dir=input_dir, output_dir=out_stage)
    except BaseException:
        # 本轮没跑完：暂存区作废，正式目录里上一轮的成果一个字节都没动过
        shutil.rmtree(out_stage, ignore_errors=True)
        raise
    core.commit_dirs(((out_stage, output_dir),), what='自动逆透视成果')

    done = [(raw, output_dir / Path(bv).name) for raw, bv in done or []]
    record_ipm_results(done)
    return {'ok': True, 'generated': len(done), 'error': None}


def api_commit(body: dict) -> dict:
    """落盘并批量测试：导出六矩阵及两类查找表（去畸变、去畸变+逆透视），
    每类各含正向/反向，共 4 组 LUT。

    前置条件是当前四点能构成 H、表网格合法，且相机参数与选中原图已经在 STATE 中。
    回包里 export_ok 与 batch 是两件事，必须分开看。自动批测是**交付之后**的验证，
    不属于导出事务：它失败了矩阵与查找表照样已经提交成功（绝不回滚），所以它的
    异常不能冒出去——否则前端只会看到一句"导出失败"，而表其实已经在盘上了。
    """
    p = _preview_inputs(body)
    apply_table_options(body)
    cal = make_calibrator(p['quad'], p['phys_w'], p['phys_h'],
                          p['anchor_x'], p['anchor_y'], p['heading'], p['scale'],
                          p['target_width'], p['target_forward'])
    if cal.H is None or cal.birdview is None or cal.H0 is None:
        raise ValueError('当前四点无法构成有效单应，无法导出。')

    # 与 api_preview 同一条路：自动布局模式下导出的 H/LUT 必须是预览里看到的那一组。
    if p['scale'] is None:
        apply_auto_layout(cal)

    K, D, Knew, img_size = STATE['K'], STATE['D'], STATE['Knew'], STATE['img_size']
    H = cal.H
    core.assert_invertible('H', H)

    buf = io.StringIO()
    batch = {'ok': False, 'generated': 0, 'error': None}
    with redirect_stdout(buf):
        origin = core.ground_origin_meta(cal)
        layout = ('目标地面范围自动适配' if cal.layout_mode == 'target_window'
                  else ('自动布局不可用，已采用兼容布局'
                        if cal.layout_mode == 'fallback' else '手动指定比例尺'))
        print(f'目标地面范围 {cal.target_width_cm:g} x {cal.target_forward_cm:g} cm'
              f'，比例尺 {cal.scale:.3f} px/cm，布局方式：{layout}')
        print('逆透视坐标参考原点 = 去畸变图底边中点对应的地面点 '
              f"{origin['ground_origin_marker_cm']} cm（标定矩形坐标系）；"
              '仅用于定义逆透视坐标，不代表摄像头或车辆实际位置')

        # 状态文件与结果图都不单独保存，随 export_all 的事务一起提交，
        # 免得导出失败后留下"新状态/新结果图 + 旧矩阵"。
        ipm_state = core.build_ipm_state(cal, p['phys_w'], p['phys_h'],
                                         img_size, STATE['src_path'])

        # 事务式导出，并把同一份最终表交给批量测试——写盘与验证同源
        pair = core.export_all(K, D, Knew, H, cal.H0, cal.sign, dict({
            'H0': cal.H0.tolist(),
            'phys_w_cm': p['phys_w'],
            'phys_h_cm': p['phys_h'],
            'scale_px_per_cm': cal.scale,
            'layout_mode': cal.layout_mode,
            'target_width_cm': cal.target_width_cm,
            'target_forward_cm': cal.target_forward_cm,
            'full_fov_fit_scale_px_per_cm': cal.full_fov_fit_scale,
            'anchor_x': cal.anchor_x,
            'anchor_y': cal.anchor_y,
            'heading_deg': cal.heading,
            'src_quad_tl_tr_bl_br': cal.corners.tolist(),
            'horizon_sign': cal.sign,
            'max_range_cm': core.MAX_RANGE_CM,
            'max_lateral_cm': core.MAX_LATERAL_CM,
            'calibration_basis_hash': core.current_calibration_basis(),
        }, **origin), img_size, ipm_state=ipm_state)
        # 导出已成功，这时才写结果图（与矩阵、查找表同属一批产物）
        core.safe_imwrite(core.IPM_RESULT, cal.birdview)
        print('去畸变逆透视结果图已保存:', core.IPM_RESULT)
        # 到这里矩阵、查找表、结果图都已落盘。下面是交付后的验证：失败不回滚任何产物，
        # 也不许把异常放出去，只把结论写进 batch 让调用方分两行显示。
        #
        # 先作废上一轮的成果记录，再开始本轮批测。顺序很重要：不清空的话本轮批测一旦
        # 失败，画廊会继续展示上一轮的配对——而那些 BirdView 是用**上一份** LUT 算的，
        # 与刚刚导出的这一份已经对不上了。宁可按钮回到 disabled，也不能展示错的成果。
        STATE['ipm_results'] = []
        try:
            batch = run_auto_batch(pair)
            print('\n全部完成。')
        except (Exception, SystemExit) as exc:
            batch = {'ok': False, 'generated': 0, 'error': str(exc)}
            print(f'\n矩阵与查找表已导出成功，但自动批量测试失败: {exc}')
    log = buf.getvalue()

    files = []
    for rel in ('matrix/matrices.json', 'matrix/matrices.txt', 'matrix/ipm_state.json'):
        pth = core.SCRIPT_DIR / rel
        if pth.is_file():
            files.append({'name': rel, 'size': pth.stat().st_size})
    return {'log': log, 'files': files, 'export_ok': True, 'batch': batch}


def api_batch() -> dict:
    """只重跑批量测试。

    语义与命令行 `--stage test` 一致：跑的是 **test_input/ 根目录**里用户自己放的
    那批图，写到 test_output/ 根目录。导出时的自动测试集在 _auto_ipm/ 子目录里，
    两条路互不干扰（list_images 不递归子目录）。

    优先消费已经导出的那套表——这才是"交付给 C 端的表能不能用"的直接验证；
    表不在时才退回按 ipm_state.json 重算。

    两条路都先过标定基准闸门。重标之后磁盘上的 matrix/lookup_table/ipm_state
    仍是上一版的，拿它们跑出来的图看上去完全正常却不属于当前标定——这种"看起来对"
    的产物必须拦住，而不是删掉。
    """
    ensure_calibration()
    try:
        core.require_calibration_basis(core.exported_basis_hash(), '已导出的查找表不可用')
        pair = core.load_exported_reverse_pair(STATE['img_size'])
        prefix = '使用已导出的查找表。\n'
    except SystemExit as exc:
        state = core.load_ipm_state() or {}
        core.require_calibration_basis(state.get('calibration_basis_hash'),
                                       'ipm_state.json 也不可用')
        pair = core.rebuild_reverse_map_from_state(
            STATE['K'], STATE['D'], STATE['Knew'], STATE['img_size'])
        prefix = f'未使用已导出的表（{exc}），改为按 ipm_state.json 重算。\n'
    _r, log = capture(core.batch_test, pair)
    return {'log': prefix + log}


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    """把上面的 api_* 函数暴露成 HTTP 接口。"""

    server_version = 'NEUQCalibUI/1.0'

    def log_message(self, fmt, *args):
        """静音默认的逐请求日志，只保留真正的错误。"""
        if str(args[1] if len(args) > 1 else '').startswith(('4', '5')):
            sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))

    # ---- 输出辅助

    def send_json(self, obj, status: int = 200) -> None:
        """发送 JSON 响应。"""
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, data: bytes, ctype: str) -> None:
        """发送二进制响应。"""
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_body(self) -> dict:
        """读取并解析 JSON 请求体。"""
        length = int(self.headers.get('Content-Length') or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode('utf-8') or '{}')

    # ---- 路由

    def do_GET(self) -> None:
        """处理 GET：静态资源与图片。"""
        parsed = urlparse(self.path)
        try:
            if parsed.path in ('/', '/index.html'):
                return self.serve_static('index.html')
            if parsed.path.startswith('/static/'):
                return self.serve_static(parsed.path[len('/static/'):])
            if parsed.path == '/api/status':
                with LOCK:
                    return self.send_json(api_status())
            if parsed.path == '/api/preview_gallery':
                with LOCK:
                    return self.send_json(api_preview_gallery())
            if parsed.path == '/api/ipm_result_gallery':
                with LOCK:
                    return self.send_json(api_ipm_result_gallery())
            if parsed.path == '/api/backup_list':
                with LOCK:
                    return self.send_json(api_backup_list())
            if parsed.path == '/api/image':
                return self.serve_image(parse_qs(parsed.query))
            self.send_json({'error': f'未知路径 {parsed.path}'}, 404)
        except Exception as exc:
            self.send_json({'error': str(exc), 'trace': traceback.format_exc()}, 500)

    def do_POST(self) -> None:
        """处理 POST：所有计算型接口都走这里，加锁串行执行。"""
        parsed = urlparse(self.path)
        if parsed.path == '/api/upload_import':
            # 上传是 multipart，不是 JSON，单独走一条分支
            return self.handle_upload()
        routes = {
            '/api/import': api_import,
            '/api/calibrate': api_calibrate,
            '/api/source': api_source,
            '/api/preview': api_preview,
            '/api/commit': api_commit,
            '/api/batch': lambda b: api_batch(),          # 签名一致化：body 未使用
            '/api/backup_clear': api_backup_clear,
            '/api/clear_all': api_clear_all,
            '/api/board': api_board,
            '/api/shutdown': lambda b: api_shutdown(),    # 同上
        }
        handler = routes.get(parsed.path)
        if handler is None:
            return self.send_json({'error': f'未知接口 {parsed.path}'}, 404)
        try:
            body = self.read_body()
            with LOCK:
                result = handler(body)
            self.send_json({'ok': True, **result})
        except (SystemExit, ValueError) as exc:
            # 参数问题（四点退化、尺寸非法、文件不存在）属于用户可修正的输入错误，
            # 用 400 而不是 500，前端才能把它和真正的服务端故障区分开。
            self.send_json({'error': str(exc)}, 400)
        except Exception as exc:
            self.send_json({'error': str(exc), 'trace': traceback.format_exc()}, 500)

    def handle_upload(self) -> None:
        """接收浏览器上传的素材文件。"""
        length = int(self.headers.get('Content-Length') or 0)
        if not length:
            return self.send_json({'error': '空请求'}, 400)
        body = self.rfile.read(length)
        try:
            fields, files = parse_multipart(self.headers, body)
            with LOCK:
                result = api_upload_import(fields, files)
            self.send_json({'ok': True, **result})
        except (SystemExit, ValueError) as exc:
            self.send_json({'error': str(exc)}, 400)
        except Exception as exc:
            self.send_json({'error': str(exc), 'trace': traceback.format_exc()}, 500)

    # ---- 静态与图片

    def serve_static(self, rel: str) -> None:
        """提供 webui/static 下的文件。"""
        path = (STATIC_DIR / rel).resolve()
        if not path.is_relative_to(STATIC_DIR) or not path.is_file():
            return self.send_json({'error': '静态资源不存在'}, 404)
        ctypes = {'.html': 'text/html; charset=utf-8', '.js': 'application/javascript; charset=utf-8',
                  '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png'}
        self.send_bytes(path.read_bytes(), ctypes.get(path.suffix, 'application/octet-stream'))

    def serve_image(self, query: dict) -> None:
        """按相对路径返回工程内的图片，可限制最大边以省带宽。"""
        rel = (query.get('rel') or [''])[0]
        path = (core.SCRIPT_DIR / rel).resolve()
        if not path.is_relative_to(core.SCRIPT_DIR) or not path.is_file():
            return self.send_json({'error': '图片不存在'}, 404)
        img = core.safe_imread(path)
        if img is None:
            return self.send_json({'error': '无法读取图片'}, 404)
        max_edge = int((query.get('max') or ['0'])[0] or 0)
        if max_edge and max(img.shape[:2]) > max_edge:
            k = max_edge / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)))
        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return self.send_json({'error': '编码失败'}, 500)
        self.send_bytes(buf.tobytes(), 'image/jpeg')


def bind_server(host: str, port: int, attempts: int = 12):
    """绑定监听端口，被占用就顺延试下一个，返回 (httpd, 实际端口)。

    端口被占用是最常见的启动失败原因（上次没退干净、或同时开了两个控制台），
    直接报错让用户自己去查端口很不友好，顺延一位继续试更省事。
    """
    last_exc = None
    for offset in range(attempts):
        try:
            return ThreadingHTTPServer((host, port + offset), Handler), port + offset
        except OSError as exc:
            last_exc = exc
            if offset < attempts - 1:
                print(f'端口 {port + offset} 被占用，试 {port + offset + 1}…')
    raise SystemExit(f'端口 {port}~{port + attempts - 1} 全被占用，无法启动。\n'
                     f'最后一次错误: {last_exc}\n'
                     f'请用 --port 指定别的端口，例如 --port 9000。')


def ensure_data_folders() -> None:
    """只保证 data/import/ 与 data/backups/ 存在。

    刻意不去预建 calib_input/、matrix/、lookup_table/ 这些工作目录：
    它们都是"跑起来才会有的产物"，全部提前铺一遍空文件夹只会让工程根变得
    杂乱，用户也分不清哪些目录是自己该往里放东西的。真正写入时各函数自己
    会 mkdir(parents=True)，不存在就建、不写就不建。
    """
    for folder in (core.DIR_IMPORT, core.DIR_BACKUP):
        folder.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """解析参数并启动服务。"""
    global HTTPD

    ap = argparse.ArgumentParser(description='NEUQ 视觉标定工具 Web 控制台')
    ap.add_argument('--port', type=int, default=8770, help='监听端口，默认 8770')
    ap.add_argument('--host', default='127.0.0.1', help='监听地址，默认仅本机')
    ap.add_argument('--root', type=Path, help='工程根目录，默认取主脚本所在目录')
    ap.add_argument('--no-browser', action='store_true', help='启动后不自动打开浏览器')
    args = ap.parse_args()

    core.configure_paths(root=args.root)
    ensure_data_folders()

    httpd, port = bind_server(args.host, args.port)
    HTTPD = httpd
    url = f'http://{args.host}:{port}/'
    print('=' * 56)
    print('  NEUQ 视觉标定控制台已启动')
    print(f'  请在浏览器打开:  {url}')
    print(f'  工程根目录:      {core.SCRIPT_DIR}')
    print('  退出: 点页面右上角「退出」按钮；或按 Ctrl+C；或直接关窗口')
    print('=' * 56)

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n已停止。')
    else:
        # 走的 /api/shutdown：属于正常退出，不该当成异常
        print('\n已按退出请求停止，可以关掉这个窗口了。')
    finally:
        httpd.server_close()
        HTTPD = None


if __name__ == '__main__':
    main()
