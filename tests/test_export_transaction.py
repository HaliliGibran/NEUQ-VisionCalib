"""导出事务的回归测试。

    python tests/test_export_transaction.py

matrix/ 与 lookup_table/ 必须作为**一套**一起换：不能一个变成新的、
另一个还留着旧的。这个脚本从三个角度验证：

  A. 正常导出成功，且不留 *.staging / *.old
  B. 导出阶段失败（打表溢出）→ 两个正式目录一字节未变
  C. 换装阶段失败（第 N 次 rename 报错）→ 两个目录一起回滚

C 是重点。老实现是"逐个目录 _swap_dir"，第二个目录换装失败时会留下
"新 LUT + 旧 matrix"的组合，A/B 都测不出来。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))

import neuq_vision_calib as core  # noqa: E402
from test_lut_roundtrip import SRC_SIZE, synth_camera, synth_ipm  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=''):
    print(('  [OK] ' if cond else '  [!!] ') + label + (f'  {detail}' if detail else ''))
    if not cond:
        FAILED.append(label)
    return bool(cond)


def tree_hash(root: Path) -> str:
    if not root.exists():
        return '(不存在)'
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode('utf-8'))
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def leftovers(root: Path) -> list[str]:
    return sorted(p.name for p in root.glob('*.staging')) + \
           sorted(p.name for p in root.glob('*.old'))


def failing_rename(nth: int):
    """让第 nth 次 Path.rename 抛错，其余放行；返回还原函数。"""
    orig = pathlib.Path.rename
    state = {'n': 0}

    def fake(self, target, *a, **kw):
        state['n'] += 1
        if state['n'] == nth:
            raise OSError(f'模拟第 {nth} 次 rename 失败')
        return orig(self, target, *a, **kw)

    pathlib.Path.rename = fake
    return lambda: setattr(pathlib.Path, 'rename', orig)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix='neuq_txn_test_'))
    root = tmp / 'proj'
    root.mkdir()
    core.configure_paths(root=root)
    core.TABLE_FORMAT = 'txt'
    core.TABLE_SIZE = None
    core.TABLE_FIXED_POINT = 4

    K, D = synth_camera()
    Knew = K
    H, H0, sign, _quad, _pw, _ph, _anchor = synth_ipm(K, D, Knew, -1.6)
    extra = {'H0': H0.tolist(), 'heading_deg': -1.6, 'scale_px_per_cm': 4.0,
             'phys_w_cm': 80.0, 'phys_h_cm': 80.0, 'anchor_x': 0.5,
             'anchor_y': 0.62, 'src_quad_tl_tr_bl_br': [[0, 0]] * 4,
             'horizon_sign': sign, 'phys_w_cm_': 0}
    ipm_state = {'H': H.tolist(), 'H0': H0.tolist(), 'heading_deg': -1.6,
                 'src_image': 'ipm_input/fake.jpg'}

    try:
        # ---------- A. 首次导出成功
        print('\n[A] 正常导出')
        core.export_all(K, D, Knew, H, H0, sign, extra, SRC_SIZE,
                        ipm_state=ipm_state)
        hashes = (tree_hash(core.DIR_TABLE), tree_hash(core.DIR_MATRIX))
        check(all(p.is_file() for p in
                  (core.DIR_MATRIX / 'matrices.json', core.DIR_MATRIX / 'matrices.txt')),
              '矩阵已写出')
        check((core.DIR_MATRIX / 'ipm_state.json').is_file(), '状态随事务一起写出')
        check(tree_hash(core.DIR_TABLE) != '(不存在)', '查找表已写出')
        check(leftovers(root) == [], '无 .staging/.old 残留', str(leftovers(root)))

        # ---------- B. 导出阶段失败
        print('\n[B] 导出阶段失败（Q15 溢出）')
        core.TABLE_FORMAT = 'bin'
        core.TABLE_FIXED_POINT = 15
        raised = False
        try:
            core.export_all(K, D, Knew, H, H0, sign, extra, SRC_SIZE,
                            ipm_state={'H': 0, 'H0': 0})
        except SystemExit:
            raised = True
        check(raised, '如预期抛错')
        check((tree_hash(core.DIR_TABLE), tree_hash(core.DIR_MATRIX)) == hashes,
              '两个正式目录均未改动')
        check(leftovers(root) == [], '无 .staging/.old 残留', str(leftovers(root)))

        # ---------- C. 换装阶段失败：这是老实现会留下半套的地方
        core.TABLE_FORMAT = 'txt'
        core.TABLE_FIXED_POINT = 4
        # _commit_dirs 会做 2 次"挪走旧的"+ 2 次"放上新的"，共 4 次 rename。
        # nth=3 表示两个旧目录都已挪走、第一个新目录还没来得及就位时炸掉。
        for nth, label in ((2, '第二次"挪走旧的"时失败（跨目录中途）'),
                           (3, '第一次"放上新的"时失败（两个旧目录都已挪走）'),
                           (4, '第二次"放上新的"时失败（第一个已就位）')):
            print(f'\n[C] 换装阶段失败：{label}')
            restore = failing_rename(nth)
            raised = False
            try:
                core.export_all(K, D, Knew, H, H0, sign, extra, SRC_SIZE,
                                ipm_state=ipm_state)
            except OSError:
                raised = True
            finally:
                restore()
            check(raised, f'第 {nth} 次 rename 触发了回滚')
            check((tree_hash(core.DIR_TABLE), tree_hash(core.DIR_MATRIX)) == hashes,
                  '两个目录一起回到原样（没有新 LUT + 旧 matrix）')
            check(leftovers(root) == [], '无 .staging/.old 残留', str(leftovers(root)))
        # ---------- D. 模拟"昨天导出、关程序、今天打开直接批量测试"
        # 这一条专治"重启后读表"这一类缺陷：bin 的 Q 位数、网格尺寸、
        # 源图分辨率都必须从落盘产物里恢复出来，而不是靠进程内的残留全局值。
        print('\n[D] 重启后读回并批量测试（bin + Q2 + 320x240）')
        core.TABLE_FORMAT = 'bin'
        core.TABLE_FIXED_POINT = 2
        core.TABLE_SIZE = (320, 240)
        core.export_all(K, D, Knew, H, H0, sign, extra, SRC_SIZE,
                        ipm_state=ipm_state)
        hashes = (tree_hash(core.DIR_TABLE), tree_hash(core.DIR_MATRIX))

        # 模拟程序重启：全局回到默认值，进程里不保留任何上一轮的信息
        core.TABLE_FORMAT = 'txt'
        core.TABLE_FIXED_POINT = 4
        core.TABLE_SIZE = None

        pair = core.load_exported_reverse_pair(SRC_SIZE)
        check(pair.size == (320, 240), '读回的输出网格 == 导出时的网格',
              f'{pair.size}')
        check(pair.fixed_point == 2, '读回的定点位数 == 导出时的 Q2',
              f'{pair.fixed_point}')
        check(pair.source_size == SRC_SIZE, '读回的源图分辨率 == 标定分辨率',
              f'{pair.source_size}')

        test_in, test_out = root / 'test_input', root / 'test_output'
        test_in.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(11)
        core.safe_imwrite(test_in / 'r.jpg',
                          rng.integers(0, 255, (SRC_SIZE[1], SRC_SIZE[0], 3),
                                       dtype=np.uint8))
        core.DIR_TEST_IN, core.DIR_TEST_OUT = test_in, test_out
        core.batch_test(pair)
        outs = sorted(test_out.glob('*.jpg'))
        check(len(outs) == 1, '批量测试产出 1 张')
        if outs:
            got = core.safe_imread(outs[0])
            check((got.shape[1], got.shape[0]) == (320, 240),
                  '输出网格 == 最终表网格 320x240',
                  f'{got.shape[1]}x{got.shape[0]}')
        check(leftovers(root) == [], '无 .staging/.old 残留', str(leftovers(root)))

        # ---------- E. metadata 自描述：把 matrices.json 拿掉也要能读
        print('\n[E] 删掉 matrices.json 后仍能读 bin 表（靠 metadata.json 自描述）')
        (core.DIR_MATRIX / 'matrices.json').rename(root / 'matrices.json.bak')
        try:
            pair2 = core.load_exported_reverse_pair(SRC_SIZE)
            check(pair2.size == (320, 240) and pair2.fixed_point == 2,
                  '无 matrices.json 也能正确解码',
                  f'{pair2.size} Q{pair2.fixed_point}')
        except SystemExit as exc:
            check(False, '无 matrices.json 也能正确解码', str(exc))
        finally:
            (root / 'matrices.json.bak').rename(core.DIR_MATRIX / 'matrices.json')

        # 收尾：把这轮改过的全局还原，免得影响后续段落
        core.TABLE_FORMAT = 'txt'
        core.TABLE_FIXED_POINT = 4
        core.TABLE_SIZE = None
        core.DIR_TEST_IN = core.SCRIPT_DIR / 'test_input'
        core.DIR_TEST_OUT = core.SCRIPT_DIR / 'test_output'
    except Exception:  # noqa: BLE001
        print(traceback.format_exc())
        FAILED.append('未捕获异常')
    finally:
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
