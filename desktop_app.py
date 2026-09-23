"""桌面壳实验原型：把现有本地 Web 控制台套进一个 WebView2 窗口。

    python desktop_app.py                    开窗口（关掉窗口即退出）
    python desktop_app.py --selftest 6       开窗口 6 秒后自动关闭，用于验证可行性
    python desktop_app.py --browser          不套壳，直接走原来的浏览器模式

这是**实验代码**，不是当前的交付入口。交付入口仍然是 app.py：
exe → 本地 HTTP server → 系统浏览器。这里只回答一个问题：
"同一套前端塞进 WebView2 能不能跑、要付多少代价"，结论写在 Release 报告里。

依赖：pywebview（Windows 上走 EdgeChromium 后端，需要系统装有 WebView2 Runtime；
Win10 2004+ / Win11 一般已随 Edge 预装）。系统缺 WebView2 时自动退回浏览器模式。

    python -m pip install pywebview
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path

TITLE = 'NEUQ 视觉标定控制台'

# WebView2 Runtime 的注册表位置。x64 系统上它装在 32 位视图里，所以要查 WOW6432Node；
# 也查一遍 HKCU，用户级安装（部分企业环境）会落在那儿。
WV2_CLIENT = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
WV2_KEYS = (
    (r'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\\' + WV2_CLIENT,
     'HKEY_LOCAL_MACHINE'),
    (r'SOFTWARE\Microsoft\EdgeUpdate\Clients\\' + WV2_CLIENT, 'HKEY_LOCAL_MACHINE'),
    (r'SOFTWARE\Microsoft\EdgeUpdate\Clients\\' + WV2_CLIENT, 'HKEY_CURRENT_USER'),
)


def webview2_version() -> str | None:
    """返回 WebView2 Runtime 的版本号；没装则返回 None。"""
    try:
        import winreg
    except ImportError:      # 非 Windows
        return None
    hives = {'HKEY_LOCAL_MACHINE': winreg.HKEY_LOCAL_MACHINE,
             'HKEY_CURRENT_USER': winreg.HKEY_CURRENT_USER}
    for path, hive_name in WV2_KEYS:
        with suppress(OSError):
            with winreg.OpenKey(hives[hive_name], path) as key:
                version, _ = winreg.QueryValueEx(key, 'pv')
                if version:
                    return str(version)
    return None


def free_port() -> int:
    """要一个空闲端口，避免和已经开着的浏览器版撞车。"""
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def start_server(port: int):
    """在后台线程里起现有的 HTTP server，返回 server 模块。"""
    if not getattr(sys, 'frozen', False):
        src = str(Path(__file__).resolve().parent / 'src')
        if src not in sys.path:
            sys.path.insert(0, src)
    from webui import server

    # server.main() 自己解析 argv：复用它才能保证桌面壳和浏览器模式跑的是同一条启动路径
    sys.argv = [sys.argv[0], '--port', str(port), '--no-browser']
    threading.Thread(target=server.main, daemon=True).start()
    return server


def wait_ready(url: str, timeout: float = 30) -> None:
    """等首页能取到为止。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    raise SystemExit(f'{timeout:.0f}s 内没能打开 {url}，桌面壳启动失败。')


def run_browser_fallback(port: int, server, reason: str) -> int:
    """退回原来的浏览器模式：用系统浏览器打开，Ctrl+C 退出。

    这条路径与 app.py 的行为一致，桌面壳不可用时用户拿到的仍是熟悉的那套界面。
    """
    print(f'  桌面壳不可用（{reason}），已退回浏览器模式。')
    url = f'http://127.0.0.1:{port}/'
    import webbrowser
    webbrowser.open(url)
    print(f'  已在浏览器打开 {url}，按 Ctrl+C 退出。')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if server.HTTPD is not None:
            server.HTTPD.shutdown()
    return 0


def main() -> int:
    """入口。"""
    ap = argparse.ArgumentParser(description='桌面壳实验原型')
    ap.add_argument('--selftest', type=float, metavar='秒',
                    help='开窗口 N 秒后自动关闭（可行性验证用）')
    ap.add_argument('--browser', action='store_true',
                    help='不套壳，直接用系统浏览器打开（等同 app.py）')
    args = ap.parse_args()

    port = free_port()
    server = start_server(port)
    url = f'http://127.0.0.1:{port}/'
    wait_ready(url)

    if args.browser:
        return run_browser_fallback(port, server, '命令行指定 --browser')

    version = webview2_version()
    try:
        import webview
    except ImportError:
        return run_browser_fallback(port, server, '未安装 pywebview')
    if version is None:
        return run_browser_fallback(port, server, '系统未安装 WebView2 Runtime')

    print(f'  WebView2 Runtime: {version}')
    window = webview.create_window(TITLE, url, width=1280, height=860)

    def on_closed() -> None:
        """窗口关闭 → 收掉 server → 立刻结束进程。

        pywebview 在 webview.start() 返回之前还要做十几秒的 WebView2 / .NET 运行时
        清理：窗口其实早就消失了，用户看不到任何东西，却留下一个僵尸进程。实测这段
        时间在 Windows 11 + pywebview 6.2.1 上稳定在 14~16 秒。所以在 closed 事件里
        直接结束进程，用户感知就是"关窗即退出"。

        代价：Chromium 会往 stderr 打一行
        "Failed to unregister class Chrome_WidgetWin_0" —— 窗口类没来得及注销，
        纯属噪音；实测 WebView2 子进程与 python 进程都没有残留。
        """
        if server.HTTPD is not None:
            server.HTTPD.shutdown()
        with suppress(ValueError, OSError):
            sys.stdout.flush()
            sys.stderr.flush()
        os._exit(0)

    window.events.closed += on_closed
    if args.selftest:
        threading.Timer(args.selftest, window.destroy).start()

    webview.start()          # 正常路径下走不到下一行（on_closed 里已 os._exit）

    # 兜底：万一 closed 事件没触发，仍然把 server 收干净
    if server.HTTPD is not None:
        server.HTTPD.shutdown()
    print(f'桌面壳已退出（后端 {url}）。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
