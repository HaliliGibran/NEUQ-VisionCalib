"""Web 控制台这一侧的状态可见性回归测试。

    python tests/test_webui_state.py

为什么单独一个文件：状态层（core）自己判得对还不够，这一轮的问题恰恰出在
"判出来了但界面看不到"——stale 只在点「应用规格」的那一瞬间回显一次，刷新
页面就没了；而「备份并清空」打包了所有产物目录，唯独没带 project.json，
于是"这批素材当时按什么规格分类"这个事实，备份里查不到。

覆盖：
  A. /api/status 常驻暴露 material_stale，replace 之后消失；stale 时 api_source 被拒
  A2. /api/status 常驻暴露 material_transaction：只剩 *.staging 是"可继续用"的一档，
      pending / .old 是"不可判定"的一档，后者连 api_source 都要被拒
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

        # 光看得见不够：stale 时 ipm_input/ 整体不可信（"这张检不出棋盘" 是旧规格
        # 判的），用户照样能点一张原图一路做到导出，所以选原图这一步也要拦
        try:
            server.api_source({'name': 'floor_0.jpg'})
            check(False, 'stale 时 api_source 被拒')
        except SystemExit as exc:
            check('逆透视标定已中止' in str(exc), 'stale 时 api_source 被拒',
                  str(exc).splitlines()[0])

        # 覆盖导入才允许换规格，成功后告警消失
        server.api_import({'dir': str(tmp / 'batch_b'), 'mode': 'replace'})
        check(server.api_status()['material_stale'] is None,
              '覆盖导入后告警消除', str(core.material_board()))

        # ---- A'. 事务残留也要常驻可见，而且分得清两档
        # 后端已经据此 fail closed（pending / .old 时连标定和选图都拒绝），界面要能
        # 提前把这个事实摆出来；否则用户只会在点下一步时莫名吃一个报错。
        print('\n[A2] /api/status 暴露 material_transaction')
        txn = server.api_status()['material_transaction']
        check(txn == {'pending': False, 'residue': [], 'unsafe': False, 'message': None},
              '干净状态下四个字段都是空的（前端据此不显示任何条幅）', str(txn))
        stage_only = core.material_stage_pairs()[0][0]
        stage_only.mkdir(parents=True, exist_ok=True)
        txn = server.api_status()['material_transaction']
        check(txn['residue'] == [stage_only.name] and txn['unsafe'] is False
              and txn['message'] is not None,
              '只剩 *.staging：黄色一档（residue 有、unsafe 为假）', str(txn))
        core.update_project_config(
            **{core.MATERIAL_PENDING_KEY: {'operation': 'replace',
                                           'board': spec_b.to_dict(),
                                           'last_import': {'mode': 'replace'}}})
        txn = server.api_status()['material_transaction']
        check(txn['pending'] is True and txn['unsafe'] is True
              and '上次导入未完成' in (txn['message'] or ''),
              'pending：红色一档（unsafe 为真，消息点明上次导入未完成）',
              (txn['message'] or '').splitlines()[0][:40])
        try:
            server.api_source({'name': 'floor_0.jpg'})
            check(False, 'unsafe 时 api_source 被后端拒绝（界面提示之外仍有真闸门）')
        except SystemExit as exc:
            check('上次导入未完成' in str(exc),
                  'unsafe 时 api_source 被后端拒绝（界面提示之外仍有真闸门）',
                  str(exc).splitlines()[0])
        core.update_project_config(**{core.MATERIAL_PENDING_KEY: None})
        shutil.rmtree(stage_only, ignore_errors=True)
        check(server.api_status()['material_transaction']['message'] is None,
              '残留处置干净后提示自动消失')

        # ---- A3. 改规格这一步不刷 /api/status，所以 /api/board 也得带上事务状态，
        #          否则那条常驻条幅会停在改规格之前的样子，直到下一次轮询才更正。
        print('\n[A3] /api/board 同步 material_transaction')
        b = server.api_board({'squares_x': 9, 'squares_y': 7, 'square_size_mm': 25.0})
        check('material_transaction' in b, 'api_board 返回里带 material_transaction',
              str(sorted(k for k in b if k.startswith('material'))))
        check(b['material_transaction']['unsafe'] is False,
              '干净状态下 unsafe 为假', str(b['material_transaction']))

        stage_only.mkdir(parents=True, exist_ok=True)
        b = server.api_board({'squares_x': 12, 'squares_y': 9, 'square_size_mm': 20.0})
        check(b['material_transaction']['residue'] == [stage_only.name]
              and b['material_transaction']['unsafe'] is False,
              '只有 staging 时返回 residue 且 unsafe 为假（黄色一档）',
              str(b['material_transaction']))

        core.update_project_config(
            **{core.MATERIAL_PENDING_KEY: {'operation': 'replace',
                                           'board': spec_b.to_dict(),
                                           'last_import': {'mode': 'replace'}}})
        b = server.api_board({'squares_x': 9, 'squares_y': 7, 'square_size_mm': 25.0})
        check(b['material_transaction']['unsafe'] is True
              and '上次导入未完成' in (b['material_transaction']['message'] or ''),
              'unsafe 状态下 api_board 仍原样返回 unsafe=true',
              (b['material_transaction']['message'] or '').splitlines()[0][:36])
        core.update_project_config(**{core.MATERIAL_PENDING_KEY: None})
        shutil.rmtree(stage_only, ignore_errors=True)

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
