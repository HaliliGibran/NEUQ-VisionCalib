"""图像读写的无状态工具：让中文路径与失败处理在全工程保持一致。

OpenCV 的图像编解码没有问题，容易出错的是把 Windows 路径直接交给 ``imread`` /
``imwrite``。这里改由 Python 负责 Unicode 路径和文件字节，OpenCV 只负责编解码；
上层因此不用在每个读写点重复兼容逻辑。
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    """转灰度；已是单通道则原样返回。"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def safe_imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """读图；读不出来返回 None。

    cv2.imread 在 Windows 上按 ANSI 代码页解释路径，中文用户名、中文工程目录
    一律读不到——而且不报错，只是返回 None，排查起来很费劲。绕开的办法是自己
    把字节读进来再解码，这样路径只经过 Python 的 Unicode 文件 API。
    """
    try:
        buf = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def safe_imwrite(path: Path, img: np.ndarray, quality: int = 95) -> None:
    """写图并校验结果。

    cv2.imwrite 失败时只返回 False 而不抛异常，静默失败会让用户看到"已保存"
    却什么都没写；和 imread 一样，它也不可靠地支持非 ASCII 路径。统一走
    imencode + Path.write_bytes，路径不再经过 OpenCV 的编码转换。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or '.png'
    params = ([int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
              if ext in ('.jpg', '.jpeg') else [])
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise SystemExit(f'编码图片失败: {path}')
    path.write_bytes(buf.tobytes())


__all__ = ['safe_imread', 'safe_imwrite', 'to_gray']
