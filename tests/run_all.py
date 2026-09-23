"""跑全部测试。

    python tests/run_all.py

零依赖：只要有 cv2 和 numpy 就行，不需要 pytest。
退出码非 0 表示有失败，可以直接用在 CI 里。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TESTS = ('test_static_names.py', 'test_board_spec.py', 'test_calibration_threshold.py',
         'test_calibration_diagnostics.py', 'test_webui_state.py',
         'test_lut_roundtrip.py', 'test_export_transaction.py',
         'test_release_packaging.py')


def main() -> int:
    here = Path(__file__).resolve().parent
    failed = []
    for name in TESTS:
        print('=' * 60)
        print(f'>> {name}')
        print('=' * 60)
        rc = subprocess.call([sys.executable, str(here / name)])
        if rc != 0:
            failed.append(name)
        print()
    if failed:
        print('失败:', ', '.join(failed))
        return 1
    print(f'全部通过（{len(TESTS)} 个文件）。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
