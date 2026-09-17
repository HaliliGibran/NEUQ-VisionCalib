"""NEUQ 视觉标定工具 —— 桌面入口。

双击 exe（或 `python app.py`）后：在程序旁边补齐数据目录 → 起本地服务 →
自动打开浏览器。命令行参数原样透传给 server，例如：

    app.exe --port 9000         换端口
    app.exe --no-browser        不自动开浏览器
    app.exe --root D:\\我的素材  指定工程根目录
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """启动控制台。"""
    # 源码方式运行时把 src/ 挂到 sys.path，好让 `webui` 包与主脚本能被找到；
    # 打包成 exe 后主脚本已经并进可执行文件，这个补丁既没必要也可能指错地方。
    if not getattr(sys, 'frozen', False):
        src = str(Path(__file__).resolve().parent / 'src')
        if src not in sys.path:
            sys.path.insert(0, src)

    from webui import server
    server.main()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # 直接 Ctrl+C 时别甩一串 traceback 给用户
        print('\n已退出。')
