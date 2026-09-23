"""Release 打包流水线的检查逻辑测试。

    python tests/test_release_packaging.py

只测 tools/build_release.py 里的纯函数：发行目录验收、ZIP 往返、SHA256、版本号。
真正跑一遍 PyInstaller 要几分钟、还依赖打包环境，不适合放进回归测试；但"验收规则
本身会不会漏报"必须钉住——否则流水线绿了也说明不了任何事。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import build_release as br  # noqa: E402 必须先把 tools/ 挂上 sys.path

FAILED: list[str] = []


def check(ok: bool, label: str) -> None:
    """记录一条断言。"""
    print(f'  [{"OK" if ok else "!!"}] {label}')
    if not ok:
        FAILED.append(label)


def make_payload(base: Path) -> Path:
    """造一份最小但"合法"的假发行目录。"""
    root = base / br.NAME
    for rel in br.REQUIRED_FILES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x' * 16)
    for rel in br.REQUIRED_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / '_internal' / 'python311.dll').write_bytes(b'dll')
    # onedir 里真实有几十个文件，check_zip 会据此判断"不是只放了个 exe"
    for i in range(40):
        (root / '_internal' / f'lib{i}.dll').write_bytes(b'lib')
    return root


def test_payload_checks(base: Path) -> None:
    """发行目录验收：该过的过，该报的报。"""
    print('发行目录验收:')
    root = make_payload(base / 'ok')
    check(br.check_payload(root) == [], '完整发行目录通过验收')

    root = make_payload(base / 'no-readme')
    (root / 'README.md').unlink()
    check(any('README.md' in p for p in br.check_payload(root)), '缺 README.md 被报出')

    root = make_payload(base / 'no-guide')
    (root / 'KNOWLEDGE_GUIDE.md').unlink()
    check(any('KNOWLEDGE_GUIDE.md' in p for p in br.check_payload(root)),
          '缺 KNOWLEDGE_GUIDE.md 被报出')

    root = make_payload(base / 'no-emblem')
    (root / '_internal/webui/static/assets/neuq-emblem.png').unlink()
    check(any('neuq-emblem.png' in p for p in br.check_payload(root)), '缺校徽被报出')

    root = make_payload(base / 'no-checker')
    shutil.rmtree(root / 'assets')
    check(any('checkerboard' in p for p in br.check_payload(root)),
          '缺 checkerboard 参考素材被报出')

    root = make_payload(base / 'no-dll')
    (root / '_internal' / 'python311.dll').unlink()
    check(any('onedir 运行时不完整' in p for p in br.check_payload(root)),
          '缺解释器 DLL 被报成 onedir 不完整')

    root = make_payload(base / 'no-data')
    shutil.rmtree(root / 'data')
    problems = br.check_payload(root)
    check(sum('data/' in p for p in problems) >= 2, 'data/import 与 data/backups 缺失都被报出')

    root = make_payload(base / 'dev-dirs')
    for name in ('calib_data', 'calib_preview', 'ipm_input', 'ipm_output',
                 'matrix', 'lookup_table'):
        (root / name).mkdir()
    problems = br.check_payload(root)
    check(sum('混入了开发目录' in p for p in problems) == 6, '六个开发目录全部被报出')

    root = make_payload(base / 'state-file')
    (root / 'project.json').write_text('{}', encoding='utf-8')
    check(any('project.json' in p for p in br.check_payload(root)),
          '混入 project.json 被报出')

    root = make_payload(base / 'user-backup')
    (root / 'data/backups/backup_20260101.zip').write_bytes(b'zip')
    (root / 'data/import/my_photos').mkdir()
    problems = br.check_payload(root)
    check(sum('必须为空' in p for p in problems) == 2, '用户备份与用户素材都被报出')

    root = make_payload(base / 'internal-matrix')
    (root / '_internal' / 'numpy' / 'matrix').mkdir(parents=True)
    check(br.check_payload(root) == [], '_internal/ 里的同名目录不误报')


def test_zip_roundtrip(base: Path) -> None:
    """ZIP：完整 onedir 往返一致，只塞个 exe 必须被拦住。"""
    print()
    print('ZIP 验收:')
    root = make_payload(base / 'zip')
    zip_path = base / 'out' / 'demo.zip'
    count = br.zip_release(root, zip_path)
    check(count == len([p for p in root.rglob('*') if p.is_file()]), '所有文件都写进了 zip')
    check(br.check_zip(zip_path, root) == [], '完整 onedir 的 zip 通过验收')

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    check(f'{br.NAME}/{br.EXE}' in names, 'zip 里有 exe')
    check(f'{br.NAME}/data/import/' in names, '空的 data/import/ 也在 zip 里')

    only_exe = base / 'out' / 'only-exe.zip'
    with zipfile.ZipFile(only_exe, 'w') as zf:
        zf.writestr(f'{br.NAME}/{br.EXE}', 'x')
    problems = br.check_zip(only_exe, root)
    check(any('少了' in p for p in problems) and any('onedir' in p for p in problems),
          '只放了 exe 的 zip 被拦住')


def test_hash_and_version(base: Path) -> None:
    """SHA256 与版本号校验。"""
    print()
    print('SHA256 与版本号:')
    blob = base / 'blob.bin'
    payload = b'neuq' * 5000
    blob.write_bytes(payload)
    check(br.sha256_file(blob) == hashlib.sha256(payload).hexdigest(), 'SHA256 与标准库一致')

    ok = [br.VERSION_RE.match(v) is not None for v in ('1.0.0', '0.9.12', '1.0.0-rc1')]
    bad = [br.VERSION_RE.match(v) is None for v in ('v1.0.0', '1.0', '1.0.0.0', '')]
    check(all(ok), '合法版本号被接受')
    check(all(bad), '不合法版本号被拒绝')

    check(br.mentions_repo(str(ROOT).encode('utf-8')), '输出里的仓库路径能被识别（UTF-8）')
    check(br.mentions_repo(str(ROOT).encode('gbk', 'replace')),
          '输出里的仓库路径能被识别（GBK）')
    check(not br.mentions_repo(b'C:\\Users\\someone\\AppData\\Local\\Temp\\neuq-release-x'),
          '仓库外路径不会被误判')

    root = make_payload(base / 'keys')
    check([item['path'] for item in br.key_files(root)] == list(br.REQUIRED_FILES),
          'manifest 的关键文件清单覆盖全部必需文件')


def test_wait_for_url(base: Path) -> None:
    """wait_for_url 的两条路径：端口兜底，以及 exe 提前退出必须立刻失败。

    背景：冻结后的 exe 在 stdout 接管道时是块缓冲的，启动横幅可能一直留在缓冲区里
    出不来。流水线如果只认 stdout 就会干等到超时——所以加了端口探测兜底，这里把它
    钉住，免得以后被人"顺手简化"掉。
    """
    print()
    print('wait_for_url 与端口探测:')

    class Handler(BaseHTTPRequestHandler):
        """冒充我们的后端：/api/status 返回带 root/dirs 的 JSON。"""

        def do_GET(self) -> None:  # noqa: N802 - 框架要求的名字
            body = json.dumps({'root': 'x', 'dirs': []}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            """测试里不需要访问日志。"""

    check(not br.probe_status(1), '未监听的端口探测为 False')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        check(br.probe_status(port), '自家的服务被正确识别')

        class LiveProc:
            """stdout 一直沉默、但进程还活着的假 exe。"""

            returncode = None

            @staticmethod
            def poll():
                """返回 None 表示仍在运行。"""
                return None

        url = br.wait_for_url(LiveProc(), bytearray(), timeout=5, fallback_port=port)
        check(url == f'http://127.0.0.1:{port}/', f'stdout 静默时靠端口兜底拿到 {url}')
    finally:
        server.shutdown()
        server.server_close()

    check(not br.probe_status(port), '服务关掉后探测为 False')

    class DeadProc:
        """一启动就退出的假 exe。"""

        returncode = 3

        @staticmethod
        def poll():
            """返回非 None 表示已退出。"""
            return 3

    try:
        br.wait_for_url(DeadProc(), bytearray(), timeout=5)
        check(False, 'exe 提前退出应该抛 ReleaseError')
    except br.ReleaseError:
        check(True, 'exe 提前退出被立刻拦下（不会白等到超时）')

    try:
        br.wait_for_url(LiveProc(), bytearray(b'no url here'), timeout=1)
        check(False, '既没 stdout 又没端口兜底时应该超时报错')
    except br.ReleaseError as exc:
        check('没等到 server 起来' in str(exc), '超时报错带上了诊断信息')


def main() -> int:
    """入口。"""
    base = Path(tempfile.mkdtemp(prefix='neuq-release-test-'))
    try:
        test_payload_checks(base)
        test_zip_roundtrip(base)
        test_hash_and_version(base)
        test_wait_for_url(base)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    print()
    if FAILED:
        print(f'失败 {len(FAILED)} 项:')
        for label in FAILED:
            print('  -', label)
        return 1
    print('Release 打包验收逻辑全部通过。')
    return 0


if __name__ == '__main__':
    sys.exit(main())


