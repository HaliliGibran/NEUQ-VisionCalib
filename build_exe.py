"""把 NEUQ-VisionCalib 智能车视觉标定工具打包成 Windows 可执行文件。

用法:
    python build_exe.py            打包
    python build_exe.py --clean    先清掉 build/ 与 dist/ 再打包

产物在 dist/NEUQ-VisionCalib/。目录预置 data/import/、data/backups/ 与
assets/checkerboard/，其余工作目录按需创建；发行目录根部同时复制 README.md 与
KNOWLEDGE_GUIDE.md。整个文件夹拷给别人即可运行（不需要装 Python）。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = 'NEUQ-VisionCalib'
SEP = ';' if sys.platform == 'win32' else ':'

# 这些库本工程用不到，排掉能显著缩小体积
EXCLUDES = (
    'matplotlib', 'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'scipy', 'pandas', 'IPython', 'pytest', 'setuptools', 'pip',
    'notebook', 'jupyter', 'sqlite3', 'pydoc',
)

# 预置两个用户素材目录；其余工作目录由程序按需创建。
IMPORT_FOLDERS = ('data/import', 'data/backups')


def build(clean: bool) -> Path:
    """调用 PyInstaller 打包，返回产物目录。"""
    if clean:
        for d in ('build', 'dist'):
            shutil.rmtree(ROOT / d, ignore_errors=True)

    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--noconfirm', '--clean',
        '--name', NAME,
        '--onedir',              # onedir 启动比 onefile 快得多，也不每次解压
        '--console',             # 保留控制台：用户能看到网址，也能 Ctrl+C 干净退出
        '--paths', str(ROOT / 'src'),          # 主脚本与 webui 包都在 src/ 下
        '--add-data', f'{ROOT / "src" / "webui" / "static"}{SEP}webui/static',
        '--hidden-import', 'neuq_vision_calib',
    ]
    for module in EXCLUDES:
        cmd += ['--exclude-module', module]
    cmd.append('app.py')

    print('执行:', ' '.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

    out = ROOT / 'dist' / NAME
    if not out.is_dir():
        raise SystemExit(f'打包似乎没成功：{out} 不存在。')

    for name in IMPORT_FOLDERS:
        (out / name).mkdir(parents=True, exist_ok=True)

    # 棋盘靶标是参考资料，不是运行产物；复制到发行目录的原位置。
    checker = ROOT / 'assets' / 'checkerboard'
    if checker.is_dir():
        shutil.copytree(checker, out / 'assets' / 'checkerboard', dirs_exist_ok=True)

    # 把使用说明放在发行目录根部，便携版用户解压后即可查阅。
    for filename in ('README.md', 'KNOWLEDGE_GUIDE.md'):
        shutil.copy2(ROOT / filename, out / filename)

    return out


def main() -> None:
    """入口。"""
    ap = argparse.ArgumentParser(description='打包成 Windows 可执行文件')
    ap.add_argument('--clean', action='store_true', help='先清掉 build/ 与 dist/')
    args = ap.parse_args()

    out = build(args.clean)
    total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    print()
    print(f'打包完成: {out}')
    print(f'  体积: {total / 1024 / 1024:.1f} MB')
    print(f'  可执行文件: {out / (NAME + ".exe")}')
    print('  整个文件夹拷走即可运行，目标机器不需要装 Python。')


if __name__ == '__main__':
    main()
