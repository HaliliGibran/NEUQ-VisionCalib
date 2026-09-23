"""把便携版打成一份可以直接发出去的 Windows Release，并当场验收。

用法:
    python tools/build_release.py --version 1.0.0
    python tools/build_release.py --version 1.0.0 --skip-build   复用现有 dist/（调试用）
    python tools/build_release.py --version 1.0.0 --keep-temp    保留仓库外的冒烟目录

一条命令走完：clean build → 发行目录验收 → 复制到仓库外临时目录 → 启动 exe →
smoke test → shutdown → ZIP → SHA256 → manifest。任何一步不过就抛
ReleaseError 并返回非零退出码；这里刻意没有"打印一条 warning 然后仍然说成功"的路径。

产物:
    release/NEUQ-VisionCalib-v<版本>-Windows-x64.zip
    release/NEUQ-VisionCalib-v<版本>-Windows-x64.zip.sha256
    release/release-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = 'NEUQ-VisionCalib'
EXE = NAME + '.exe'
RELEASE_DIR = ROOT / 'release'
MANIFEST_NAME = 'release-manifest.json'
VERSION_RE = re.compile(r'^\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?$')

# 发行包里必须存在的文件。前四项是用户直接要用的；_internal/ 那几项是
# PyInstaller onedir 的运行时与前端资源——少任何一个，exe 拷到别的机器就废。
REQUIRED_FILES = (
    EXE,
    'README.md',
    'KNOWLEDGE_GUIDE.md',
    'assets/checkerboard/Checkerboard_12_9_20mm.png',
    'assets/checkerboard/Checkerboard_12_9_20mm.pdf',
    '_internal/base_library.zip',
    '_internal/webui/static/index.html',
    '_internal/webui/static/app.js',
    '_internal/webui/static/style.css',
    '_internal/webui/static/assets/neuq-emblem.png',
)

REQUIRED_DIRS = ('data/import', 'data/backups', '_internal/cv2', '_internal/numpy')

# onedir 运行时的硬指标：解释器 DLL 必须在 _internal/ 里
RUNTIME_GLOBS = ('_internal/python3*.dll',)

# 开发者本机的标定链路目录：一个都不许混进发行包
FORBIDDEN_DIRS = ('calib_input', 'calib_data', 'calib_preview', 'ipm_input', 'ipm_output',
                  'matrix', 'lookup_table', 'test_input', 'test_output')

# 本机工作状态文件，同样不许混进发行包
FORBIDDEN_FILES = ('project.json', 'calib.json', 'matrices.json', 'ipm_state.json')

# 必须随包发出、但必须是空的：用户自己的素材与备份不属于发行内容
MUST_BE_EMPTY = ('data/import', 'data/backups')


class ReleaseError(RuntimeError):
    """Release 流水线的失败：一律直接终止，不降级成警告。"""


# ---------------------------------------------------------------- 发行目录验收

def check_payload(root: Path) -> list[str]:
    """检查发行目录的完整性与洁净度，返回问题清单（空表示通过）。

    只做静态检查，不启动 exe。之所以把"缺资源"和"混进开发数据"放在同一个函数里：
    两类问题都只能在打包完、还没压缩之前发现，分开写会漏掉其中一类。
    """
    problems: list[str] = []
    if not root.is_dir():
        return [f'发行目录不存在: {root}']

    for rel in REQUIRED_FILES:
        if not (root / rel).is_file():
            problems.append(f'缺少必需文件: {rel}')
    for rel in REQUIRED_DIRS:
        if not (root / rel).is_dir():
            problems.append(f'缺少必需目录: {rel}')
    for pattern in RUNTIME_GLOBS:
        if not list(root.glob(pattern)):
            problems.append(f'onedir 运行时不完整: 找不到 {pattern}')

    # 开发数据只会落在发行目录根部这一层级的工作目录里；_internal/ 是
    # PyInstaller 自己的地盘（numpy/cv2 里也可能有同名子目录），不参与这项检查。
    for path in root.rglob('*'):
        rel = path.relative_to(root)
        if rel.parts[0] == '_internal':
            continue
        if path.is_dir() and path.name in FORBIDDEN_DIRS:
            problems.append(f'混入了开发目录: {rel.as_posix()}/')
        elif path.is_file() and path.name in FORBIDDEN_FILES:
            problems.append(f'混入了本机状态文件: {rel.as_posix()}')

    for rel in MUST_BE_EMPTY:
        folder = root / rel
        leftovers = sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []
        if leftovers:
            problems.append(f'{rel}/ 必须为空，实际有 {len(leftovers)} 项: '
                            f'{", ".join(leftovers[:5])}')
    return problems


def dir_size(root: Path) -> int:
    """目录内所有文件的字节数之和。"""
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def sha256_file(path: Path) -> str:
    """文件的 SHA256（十六进制小写）。"""
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- ZIP

def zip_release(src: Path, zip_path: Path, arc_root: str = NAME) -> int:
    """把整个发行目录压进 zip，顶层套一层 arc_root/，返回写入的文件数。

    套一层目录是刻意的：用户解压时不会把几十个 DLL 直接摊到下载目录里。
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.rglob('*') if p.is_file())
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, f'{arc_root}/{path.relative_to(src).as_posix()}')
        # 空目录（data/import、data/backups）要显式写进去，否则解压后不存在
        for path in sorted(p for p in src.rglob('*') if p.is_dir()):
            if not any(path.iterdir()):
                zf.writestr(f'{arc_root}/{path.relative_to(src).as_posix()}/', '')
    return len(files)


def check_zip(zip_path: Path, src: Path, arc_root: str = NAME) -> list[str]:
    """检查 zip 是不是一份完整的 onedir，返回问题清单。

    只压一个 exe 进去是最容易犯、也最致命的错（用户解压后必然启动失败），
    所以这里逐个比对磁盘上的文件名，而不是只看 exe 在不在。
    """
    problems: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            problems.append(f'zip 内容校验失败: {bad}')
        names = set(zf.namelist())
    expected = {f'{arc_root}/{p.relative_to(src).as_posix()}'
                for p in src.rglob('*') if p.is_file()}
    missing = sorted(expected - names)
    if missing:
        problems.append(f'zip 少了 {len(missing)} 个文件，例如: {", ".join(missing[:5])}')
    for rel in REQUIRED_FILES:
        if f'{arc_root}/{rel}' not in names:
            problems.append(f'zip 里缺少必需文件: {rel}')
    internal = [n for n in names if n.startswith(f'{arc_root}/_internal/')]
    if len(internal) < 30:
        problems.append(f'zip 里的 _internal/ 只有 {len(internal)} 项，不像完整 onedir')
    return problems


# ---------------------------------------------------------------- 仓库外冒烟

# (说明, 路径, 期望的 Content-Type 前缀)
SMOKE_TARGETS = (
    ('首页 HTML', '/', 'text/html'),
    ('样式表 CSS', '/static/style.css', 'text/css'),
    ('前端脚本 JS', '/static/app.js', 'application/javascript'),
    ('NEUQ 校徽 PNG', '/static/assets/neuq-emblem.png', 'image/png'),
    ('状态接口 /api/status', '/api/status', 'application/json'),
)

URL_RE = re.compile(r'https?://[\d.]+:\d+/')


def free_port() -> int:
    """要一个当前空闲的端口。真正用哪个端口以 exe 自己打印的为准。"""
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def launch_exe(exe: Path, port: int) -> tuple[subprocess.Popen, bytearray]:
    """启动发行版 exe，返回 (进程, 持续累积的 stdout 原始字节)。"""
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    proc = subprocess.Popen(
        [str(exe), '--no-browser', '--port', str(port)],
        cwd=str(exe.parent), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    raw = bytearray()

    def pump() -> None:
        for chunk in iter(lambda: proc.stdout.read(1), b''):
            raw.extend(chunk)
        proc.stdout.close()

    threading.Thread(target=pump, daemon=True).start()
    return proc, raw


def probe_status(port: int) -> bool:
    """探测某个端口是不是我们的服务：/api/status 必须返回预期结构。

    只看"端口通不通"是不够的——机器上任何程序都可能占着那个端口，认错了对象后面
    整套冒烟测试都白做。所以要求响应里带上 status 接口特有的字段。
    """
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/status',
                                    timeout=3) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read())
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and 'root' in data and 'dirs' in data


def wait_for_url(proc: subprocess.Popen, raw: bytearray, timeout: float = 120,
                 fallback_port: int | None = None) -> str:
    """等 exe 打印出监听地址，返回形如 http://127.0.0.1:8770/ 的基址。

    优先解析 stdout：那是 exe 自己报出来的真实端口，即使发生端口顺延也不会认错。
    但冻结后的 exe 在 stdout 接管道时是块缓冲的，横幅有可能一直留在缓冲区里出不来，
    光等 stdout 就是白等 120 秒。所以同时用我们传进去的那个端口做兜底探测。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        match = URL_RE.search(bytes(raw).decode('utf-8', 'replace'))
        if match:
            return match.group(0)
        if proc.poll() is not None:
            raise ReleaseError(f'exe 启动即退出（退出码 {proc.returncode}）：\n'
                               f'{bytes(raw).decode("utf-8", "replace")}')
        if fallback_port is not None and probe_status(fallback_port):
            return f'http://127.0.0.1:{fallback_port}/'
        time.sleep(0.2)
    captured = bytes(raw).decode('utf-8', 'replace')
    if not captured:
        hint = ('（exe 输出为空——说明启动横幅没 flush 出来，'
                '检查 server.force_line_buffering 是否生效）')
    else:
        hint = captured[-600:]
    raise ReleaseError(
        f'{timeout:.0f}s 内没等到 server 起来。\n'
        f'  exe 输出（{len(captured)} 字节）: {hint}')


def probe(url: str, timeout: float = 20) -> tuple[int, str, int]:
    """GET 一个地址，返回 (状态码, Content-Type, 响应体字节数)。"""
    req = urllib.request.Request(url, headers={'Accept': '*/*'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:

            body = resp.read()
            return resp.status, resp.headers.get('Content-Type', ''), len(body)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get('Content-Type', ''), 0
    except OSError as exc:
        raise ReleaseError(f'请求 {url} 失败: {exc}') from exc


def run_smoke(base: str) -> list[dict]:
    """逐项请求 SMOKE_TARGETS，任何一项不是 200 或 MIME 不对就失败。"""
    results: list[dict] = []
    problems: list[str] = []
    for label, path, ctype_prefix in SMOKE_TARGETS:
        status, ctype, size = probe(base.rstrip('/') + path)
        results.append({'label': label, 'path': path, 'status': status,
                        'content_type': ctype, 'bytes': size})
        if status != 200:
            problems.append(f'{label} ({path}) 返回 {status}')
        elif not ctype.startswith(ctype_prefix):
            problems.append(f'{label} ({path}) 的 Content-Type 是 {ctype!r}，'
                            f'期望 {ctype_prefix!r}')
        elif size == 0:
            problems.append(f'{label} ({path}) 返回了空响应体')
    if problems:
        raise ReleaseError('冒烟测试未通过：\n  - ' + '\n  - '.join(problems))
    return results


def shutdown_and_wait(base: str, proc: subprocess.Popen, port: int) -> None:
    """走程序自己的 /api/shutdown 退出，并确认进程与端口都真的放开了。"""
    req = urllib.request.Request(base.rstrip('/') + '/api/shutdown', data=b'{}',
                                 headers={'Content-Type': 'application/json'},
                                 method='POST')
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:

            if resp.status != 200:
                raise ReleaseError(f'/api/shutdown 返回 {resp.status}')
    except OSError as exc:
        raise ReleaseError(f'调用 /api/shutdown 失败: {exc}') from exc

    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise ReleaseError('调用 shutdown 后 exe 进程 30s 内没有退出，已强制结束。'
                           '这属于 Release 阻塞问题：用户点"退出"会留下僵尸进程。') from exc
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise ReleaseError(f'exe 已退出，但端口 {port} 仍在监听，说明还有残留进程。')


def mentions_repo(raw: bytes) -> bool:
    """判断 exe 的输出里有没有提到仓库路径。

    工程根指到源码目录 = 发行版偷偷依赖了开发者的工作区，拷给别人必然出问题。
    仓库路径含中文，冻结后的 exe 用什么编码打印不确定，所以三种编码都比一遍。
    """
    text = str(ROOT)
    for encoding in ('utf-8', 'gbk', 'mbcs'):
        try:
            if text.encode(encoding, 'replace') in raw:
                return True
        except LookupError:
            continue
    return False


def standalone_smoke(payload: Path, keep_temp: bool) -> dict:
    """把发行目录整份复制到仓库外，在那里启动 exe 做冒烟，返回结果摘要。"""
    temp_root = Path(tempfile.mkdtemp(prefix='neuq-release-'))
    if temp_root.is_relative_to(ROOT):
        shutil.rmtree(temp_root, ignore_errors=True)
        raise ReleaseError(f'临时目录 {temp_root} 落在仓库内，冒烟测试没有意义。')
    target = temp_root / NAME
    shutil.copytree(payload, target)
    before = {p.relative_to(target).as_posix() for p in target.rglob('*')}

    port = free_port()
    proc, raw = launch_exe(target / EXE, port)
    try:
        url = wait_for_url(proc, raw, fallback_port=port)
        actual_port = int(url.rstrip('/').rsplit(':', 1)[1])
        checks = run_smoke(url)
        shutdown_and_wait(url, proc, actual_port)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    created = sorted({p.relative_to(target).as_posix() for p in target.rglob('*')} - before)
    out = {
        'temp_dir': str(target),
        'url': url,
        'port': actual_port,
        'exit_code': proc.returncode,
        'checks': checks,
        'created_at_runtime': created,
        'mentions_repo_path': mentions_repo(bytes(raw)),
        'stdout_tail': bytes(raw).decode('utf-8', 'replace')[-800:],
    }
    if out['mentions_repo_path']:
        raise ReleaseError('发行版 exe 在仓库外运行时仍然提到了仓库路径，'
                           '说明它隐式依赖源码目录。')
    if not keep_temp:
        shutil.rmtree(temp_root, ignore_errors=True)
    return out


# ---------------------------------------------------------------- 构建与 manifest

def running_instances(exe: Path) -> list[str]:
    """列出正在运行、且映像就是这个 exe 的进程，形如 ['PID 38228  <路径>']。

    拿不到进程表就返回空列表：这只是为了把诊断说清楚，不该让流水线本身失败。
    """
    script = ('Get-CimInstance Win32_Process -Filter "Name=\'%s\'" | '
              'ForEach-Object { "$($_.ProcessId)`t$($_.ExecutablePath)" }' % EXE)
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', script],
                             capture_output=True, text=True, timeout=30,
                             check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in out.splitlines():
        pid, _, path = line.partition('\t')
        if path.strip().lower() == str(exe).lower():
            found.append(f'PID {pid.strip()}  {path.strip()}')
    return found


def preflight_dist(payload: Path) -> None:
    """clean build 之前先确认 dist/ 没被占用。

    真实踩过的坑：上一版便携版 exe 还在后台跑着，PyInstaller 删不掉 dist/ 只能甩一个
    PermissionError: [WinError 5]，光看那串栈根本想不到"去把旧实例关掉"。所以这里
    提前诊断、把 PID 和路径直接报出来。刻意不自动杀进程——那是用户自己的程序。
    """
    exe = payload / EXE
    if not exe.is_file():
        return
    try:
        with exe.open('ab'):
            return
    except OSError as exc:
        procs = running_instances(exe) or ['（没能列出占用进程，可能是别的程序打开了它）']
        raise ReleaseError(
            f'{exe} 正被占用，clean build 无法删除它：{exc}\n'
            + '\n'.join(f'  占用者: {item}' for item in procs)
            + '\n  请先退出旧的便携版实例（网页右上角「退出」按钮，或关掉它的控制台窗口），'
              '再重新执行本命令。') from exc


def git_info() -> dict:

    """当前 commit 与工作区是否干净；拿不到就记 unknown，但不让流水线失败。"""
    info = {'commit': 'unknown', 'dirty': None}
    try:
        info['commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                        capture_output=True, text=True,
                                        check=True).stdout.strip()
        status = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout
        info['dirty'] = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return info


def run_build(skip: bool) -> Path:
    """调 build_exe.py 做一次 clean build，返回发行目录。"""
    payload = ROOT / 'dist' / NAME
    preflight_dist(payload)
    if skip:

        if not payload.is_dir():
            raise ReleaseError(f'--skip-build 需要现成的 {payload}，但它不存在。')
        print(f'  跳过打包，直接用 {payload}')
        return payload
    result = subprocess.run([sys.executable, str(ROOT / 'build_exe.py'), '--clean'],
                            cwd=ROOT, check=False)
    if result.returncode != 0:
        raise ReleaseError(f'PyInstaller 打包失败（build_exe.py 退出码 '
                           f'{result.returncode}）。')
    if not payload.is_dir():
        raise ReleaseError(f'打包命令成功返回，但 {payload} 不存在。')
    return payload


def key_files(payload: Path) -> list[dict]:
    """manifest 里记录的关键文件清单（名字 + 字节数）。"""
    out = []
    for rel in REQUIRED_FILES:
        path = payload / rel
        out.append({'path': rel, 'bytes': path.stat().st_size})
    return out


def step(index: int, title: str) -> None:
    """打印流水线步骤标题。"""
    print()
    print(f'[{index}/7] {title}')
    print('-' * 56)


def release(version: str, skip_build: bool, keep_temp: bool) -> dict:
    """跑完整条流水线，返回 manifest。任何一步不过都抛 ReleaseError。"""
    if not VERSION_RE.match(version):
        raise ReleaseError(f'版本号 {version!r} 不合法，应形如 1.0.0 或 1.0.0-rc1。')

    step(1, '打包（clean build）')
    payload = run_build(skip_build)
    unpacked = dir_size(payload)
    print(f'  发行目录: {payload}（{unpacked / 1024 / 1024:.1f} MB）')

    step(2, '发行目录验收')
    problems = check_payload(payload)
    if problems:
        raise ReleaseError('发行目录验收未通过：\n  - ' + '\n  - '.join(problems))
    print(f'  必需文件 {len(REQUIRED_FILES)} 项、必需目录 {len(REQUIRED_DIRS)} 项齐全；'
          '未发现开发数据混入。')

    step(3, '复制到仓库外并启动 exe 冒烟')
    smoke = standalone_smoke(payload, keep_temp)
    print(f'  临时目录: {smoke["temp_dir"]}')
    print(f'  监听地址: {smoke["url"]}')
    for check in smoke['checks']:
        print(f'  [OK] {check["label"]}: {check["status"]} '
              f'{check["content_type"]} {check["bytes"]}B')
    print(f'  shutdown 后进程退出码: {smoke["exit_code"]}')

    step(4, '打 ZIP')
    archive = f'{NAME}-v{version}-Windows-x64.zip'
    zip_path = RELEASE_DIR / archive
    if zip_path.exists():
        zip_path.unlink()
    count = zip_release(payload, zip_path)
    print(f'  {zip_path.name}: {count} 个文件，{zip_path.stat().st_size / 1024 / 1024:.1f} MB')

    step(5, 'ZIP 内容验收')
    problems = check_zip(zip_path, payload)
    if problems:
        raise ReleaseError('ZIP 验收未通过：\n  - ' + '\n  - '.join(problems))
    print('  ZIP 内是完整 onedir（逐文件比对通过）。')

    step(6, 'SHA256')
    digest = sha256_file(zip_path)
    (RELEASE_DIR / (archive + '.sha256')).write_text(f'{digest} *{archive}\n',
                                                     encoding='utf-8')
    print(f'  {digest}')

    step(7, '写 manifest')
    info = git_info()
    manifest = {
        'name': NAME,
        'release_version': version,
        'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git_commit': info['commit'],
        'git_dirty': info['dirty'],
        'platform': {
            'os': f'{platform.system()} {platform.release()}',
            'machine': platform.machine(),
            'python': platform.python_version(),
            'target': 'Windows-x64',
        },
        'archive': archive,
        'sha256': digest,
        'archive_bytes': zip_path.stat().st_size,
        'unpacked_bytes': unpacked,
        'file_count': count,
        'key_files': key_files(payload),
        'smoke_test': {
            'ran_outside_repo': True,
            'temp_dir': smoke['temp_dir'],
            'url': smoke['url'],
            'exit_code': smoke['exit_code'],
            'checks': smoke['checks'],
            'created_at_runtime': smoke['created_at_runtime'],
            'mentions_repo_path': smoke['mentions_repo_path'],
        },
    }
    manifest_path = RELEASE_DIR / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                             encoding='utf-8')
    print(f'  {manifest_path}')
    return manifest


def main() -> int:
    """入口。"""
    ap = argparse.ArgumentParser(description='构建并验收正式 Release')
    ap.add_argument('--version', required=True, help='版本号，例如 1.0.0')
    ap.add_argument('--skip-build', action='store_true',
                    help='复用现有 dist/（只调试后续步骤时用，不要用于正式发版）')
    ap.add_argument('--keep-temp', action='store_true', help='保留仓库外的冒烟目录')
    args = ap.parse_args()

    try:
        manifest = release(args.version, args.skip_build, args.keep_temp)
    except ReleaseError as exc:
        print()
        print('=' * 56)
        print(f'Release 失败: {exc}')
        print('=' * 56)
        return 1
    print()
    print('=' * 56)
    print(f'Release 完成: {RELEASE_DIR / manifest["archive"]}')
    print(f'  SHA256: {manifest["sha256"]}')
    print(f'  ZIP:    {manifest["archive_bytes"] / 1024 / 1024:.1f} MB'
          f'（解压后 {manifest["unpacked_bytes"] / 1024 / 1024:.1f} MB）')
    print('=' * 56)
    return 0


if __name__ == '__main__':
    sys.exit(main())







