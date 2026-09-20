"""Web 控制台这一侧的状态可见性回归测试。

    python tests/test_webui_state.py

为什么单独一个文件：状态层（core）自己判得对还不够，这一轮的问题恰恰出在
"判出来了但界面看不到"——stale 只在点「应用规格」的那一瞬间回显一次，刷新
页面就没了；而「备份并清空」打包了所有产物目录，唯独没带 project.json，
于是"这批素材当时按什么规格分类"这个事实，备份里查不到。

覆盖：
  A. /api/status 常驻暴露 material_stale，replace 之后消失
  B. 备份 manifest 带 project_config 快照，且清空后素材库依据作废
  C. 备份目录跟着 --root 现取（不能停在 import 时抄下来的旧根）
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import neuq_vision_calib as core  # noqa: E402
from webui import server  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=''):
    print(('  [OK] ' if cond else '  [!!] ') + label + (f'  {detail}' if detail else ''))
    if not cond:
        FAILED.append(label)
    return bool(cond)


def floors(folder: Path, n: int) -> Path:
    """造 n 张纯灰图：任何规格都检不出棋盘，一律归进 ipm_input/。"""
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        core.safe_imwrite(folder / f'floor_{i}.jpg',
                          np.full((80, 80, 3), 128, np.uint8))
    return folder


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix='neuq_webui_'))
    old_root, old_board = core.SCRIPT_DIR, core.BOARD
    spec_a = core.CheckerboardSpec(12, 9, 20.0)
    spec_b = core.CheckerboardSpec(9, 7, 25.0)
    try:
        proj = tmp / 'proj'
        core.configure_paths(root=proj)
        core.configure_board(spec_a)

        # ---- A. 状态接口要一直说得出"素材是按旧规格分的"
        print('[A] /api/status 常驻暴露 material_stale')
        server.api_import({'dir': str(floors(tmp / 'batch_a', 2)), 'mode': 'add'})
        check(server.api_status()['material_stale'] is None, '规格一致时无告警')

        applied = server.api_board({'squares_x': 9, 'squares_y': 7,
                                    'square_size_mm': 25})
        check(applied['material_stale'] is not None,
              '「应用规格」的返回里带告警（原有行为保留）')
        stale = server.api_status()['material_stale']
        check(stale is not None and '不一致' in stale,
              '改规格后 /api/status 就带上告警（刷新页面也还在）',
              (stale or '').splitlines()[0][:40])
        check(spec_a.label in stale and spec_b.label in stale,
              '告警里同时给出"按什么分类的"和"现在是什么"')

        # 增量导入被拒之后，告警必须还在——这正是老模型会被洗白的那条路径
        try:
            server.api_import({'dir': str(floors(tmp / 'batch_b', 2)), 'mode': 'add'})
            check(False, 'stale 时 api_import(add) 被拒')
        except SystemExit as exc:
            check('增量导入已中止' in str(exc), 'stale 时 api_import(add) 被拒',
                  str(exc).splitlines()[0])
        check(server.api_status()['material_stale'] is not None,
              '被拒的增量导入没有把告警洗掉')

        # 覆盖导入才允许换规格，成功后告警消失
        server.api_import({'dir': str(tmp / 'batch_b'), 'mode': 'replace'})
        check(server.api_status()['material_stale'] is None,
              '覆盖导入后告警消除', str(core.material_board()))

        # ---- B. 备份要把"这批素材按什么规格分类"一起存下来
        print('\n[B] 备份 manifest 带 project_config 快照')
        result = server.api_backup_clear({'name': 'round1'})
        manifest = json.loads(
            (server.dir_backup() / 'round1' / 'manifest.json').read_text(encoding='utf-8'))
        snap = manifest.get('project_config') or {}
        check(bool(snap), 'manifest 里有 project_config', str(sorted(snap)))
        check((snap.get('material_set') or {}).get('board', {}).get('squares_x') == 9,
              '快照里记下了素材的分拣依据 9x7',
              str((snap.get('material_set') or {}).get('board')))
        check((snap.get('board') or {}).get('square_size_mm') == 25.0,
              '快照里也记下了当时的工程规格 25mm', str(snap.get('board')))
        check(manifest['project_config'] != core.load_project_config(),
              '取的是清空前的值，不是事后那一份')
        check(core.material_board() is None,
              '清空后素材库依据作废', str(core.material_board()))
        check(result['cleared_files'] > 0, '确实清掉了文件', str(result['cleared_files']))

        # 空的活配置不影响下一次增量导入：依据为空 = 首次导入
        server.api_import({'dir': str(floors(tmp / 'batch_c', 1)), 'mode': 'add'})
        check(core.material_board() == spec_b, '清空后按当前规格重新确立依据')

        # ---- C. 备份目录必须跟着工程根走
        print('\n[C] 备份目录跟着 --root 现取')
        other = tmp / 'other_root'
        core.configure_paths(root=other)
        check(server.dir_backup() == core.DIR_BACKUP,
              'dir_backup() 与 core 当前根一致', str(server.dir_backup()))
        check(other in server.dir_backup().parents,
              '换根之后备份落在新根下', str(server.dir_backup()))
    finally:
        core.configure_paths(root=old_root)
        core.configure_board(old_board)
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print('失败项:')
        for f in FAILED:
            print('   ', f)
        return 1
    print('全部通过。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
