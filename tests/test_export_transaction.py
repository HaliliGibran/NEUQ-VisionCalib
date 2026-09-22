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

from test_lut_roundtrip import SRC_SIZE, synth_camera, synth_ipm  # noqa: E402

import neuq_vision_calib as core  # noqa: E402

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
        print('\n[D] 重启后读回并批量测试（bin + Q2 + 320x180）')
        core.TABLE_FORMAT = 'bin'
        core.TABLE_FIXED_POINT = 2
        core.TABLE_SIZE = (320, 180)
        core.export_all(K, D, Knew, H, H0, sign, extra, SRC_SIZE,
                        ipm_state=ipm_state)
        hashes = (tree_hash(core.DIR_TABLE), tree_hash(core.DIR_MATRIX))

        # 模拟程序重启：全局回到默认值，进程里不保留任何上一轮的信息
        core.TABLE_FORMAT = 'txt'
        core.TABLE_FIXED_POINT = 4
        core.TABLE_SIZE = None

        pair = core.load_exported_reverse_pair(SRC_SIZE)
        check(pair.size == (320, 180), '读回的输出网格 == 导出时的网格',
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
            check((got.shape[1], got.shape[0]) == (320, 180),
                  '输出网格 == 最终表网格 320x180',
                  f'{got.shape[1]}x{got.shape[0]}')
        check(leftovers(root) == [], '无 .staging/.old 残留', str(leftovers(root)))

        # ---------- E. metadata 自描述：把 matrices.json 拿掉也要能读
        print('\n[E] 删掉 matrices.json 后仍能读 bin 表（靠 metadata.json 自描述）')
        (core.DIR_MATRIX / 'matrices.json').rename(root / 'matrices.json.bak')
        try:
            pair2 = core.load_exported_reverse_pair(SRC_SIZE)
            check(pair2.size == (320, 180) and pair2.fixed_point == 2,
                  '无 matrices.json 也能正确解码',
                  f'{pair2.size} Q{pair2.fixed_point}')
        except SystemExit as exc:
            check(False, '无 matrices.json 也能正确解码', str(exc))
        finally:
            (root / 'matrices.json.bak').rename(core.DIR_MATRIX / 'matrices.json')

        # ---------- F. T1：查找表只能整数倍等比降采样
        print('\n[F] T1 非等比网格必须被拒，而且拒在任何写盘之前')
        core.TABLE_FORMAT = 'bin'
        core.TABLE_FIXED_POINT = 4
        before = (tree_hash(core.DIR_TABLE), tree_hash(core.DIR_MATRIX))
        for bad in ((320, 240), (160, 120), (640, 480), (1281, 720)):
            core.TABLE_SIZE = bad
            try:
                core.export_all(K, D, Knew, H, H0, sign, extra, SRC_SIZE,
                                ipm_state=ipm_state)
                check(False, f'{bad[0]}x{bad[1]} 必须被拒绝', '却导出成功了')
            except ValueError as exc:
                first = str(exc).splitlines()[0]
                check('等比' in first or '不能大于' in first,
                      f'{bad[0]}x{bad[1]} -> ValueError', first)
        check((tree_hash(core.DIR_TABLE), tree_hash(core.DIR_MATRIX)) == before,
              '四次被拒之后两个正式目录一字节未变')
        check(leftovers(root) == [], '被拒之后无 .staging/.old 残留', str(leftovers(root)))

        for good, want in (((640, 360), 2), ((320, 180), 4), ((256, 144), 5),
                           ((160, 90), 8), (None, 1)):
            check(core.downsample_factor(SRC_SIZE, good) == want,
                  f'{good} 的倍率 == {want}x')
        check(core.table_grid_factors(1280, 720) == [1, 2, 4, 5, 8, 10, 16],
              '1280x720 的可选倍率清单', str(core.table_grid_factors(1280, 720)))

        core.TABLE_SIZE = (320, 180)
        core.export_all(K, D, Knew, H, H0, sign, extra, SRC_SIZE,
                        ipm_state=ipm_state)
        meta_f = json.loads((core.DIR_TABLE / 'undistort_ipm' / 'reverse'
                             / 'metadata.json').read_text(encoding='utf-8'))
        check(meta_f.get('downsample_factor') == 4
              and meta_f.get('grid_size') == [320, 180]
              and meta_f.get('index_full_size') == [1280, 720],
              'metadata 记下单一倍率，C 端不必再分 step_x/step_y',
              f"factor={meta_f.get('downsample_factor')} grid={meta_f.get('grid_size')}")

        # ---------- G. CAL1：calib_preview 整体换装，不留上一轮的残留
        print('\n[G] CAL1 预览目录整体换装')
        rng2 = np.random.default_rng(7)
        raws = []
        core.DIR_CALIB_IN.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            p = core.DIR_CALIB_IN / f'shot_{i}.jpg'
            core.safe_imwrite(p, rng2.integers(0, 255, (SRC_SIZE[1], SRC_SIZE[0], 3),
                                               dtype=np.uint8))
            raws.append(p)

        core.export_undistort_previews(raws, K, D, SRC_SIZE)
        names4 = sorted(p.name for p in core.DIR_CALIB_PREVIEW.glob('*.jpg'))
        check(len(names4) == 4, '第一轮 4 张预览落盘', str(names4))

        # 第二轮只用前 2 张（等价"人工剔掉误差大的两张后重标"）
        core.export_undistort_previews(raws[:2], K, D, SRC_SIZE)
        names2 = sorted(p.name for p in core.DIR_CALIB_PREVIEW.glob('*.jpg'))
        check(names2 == ['shot_0_undist.jpg', 'shot_1_undist.jpg'],
              '第二轮只剩 2 张——被剔掉那两张的旧预览没有残留', str(names2))
        check(leftovers(root) == [], '换装后无 .staging/.old 残留', str(leftovers(root)))

        # 中途失败：旧预览必须完好，暂存区清干净
        real_imwrite = core.safe_imwrite
        calls = {'n': 0}

        def imwrite_boom(path, img):
            calls['n'] += 1
            if calls['n'] == 2:
                raise OSError('写第 2 张预览时磁盘满（故障注入）')
            return real_imwrite(path, img)

        core.safe_imwrite = imwrite_boom
        try:
            core.export_undistort_previews(raws, K, D, SRC_SIZE)
            check(False, '预览写到一半故障应当抛出', '却正常返回')
        except OSError as exc:
            check(True, '预览写到一半故障抛出', str(exc))
        finally:
            core.safe_imwrite = real_imwrite
        check(sorted(p.name for p in core.DIR_CALIB_PREVIEW.glob('*.jpg')) == names2,
              '中途故障后旧预览完整保留（换装前一个字节都没动）')
        check(leftovers(root) == [], '故障后暂存区已清理', str(leftovers(root)))

        # ---------- H. CAL1：calib.json 落 provenance
        print('\n[H] CAL1 标定 provenance')
        fit = {'rms': 1.053, 'mean_reprojection_error': 0.880, 'threshold': 1.5,
               'used_names': [f'u{i}.jpg' for i in range(21)],
               'dropped': [{'name': 'bad0.jpg', 'error': 2.355},
                           {'name': 'bad1.jpg', 'error': 2.202}],
               'opencv_version': '4.10.0'}
        core.save_calibration(K, D, SRC_SIZE, fit_result=fit)
        cj = json.loads(core.CALIB_JSON.read_text(encoding='utf-8'))
        check(cj.get('rms_px') == 1.053
              and cj.get('mean_reprojection_error_px') == 0.880
              and cj.get('max_reproj_err') == 1.5
              and len(cj.get('used_images') or []) == 21
              and cj.get('dropped_images') == [{'name': 'bad0.jpg', 'rms_px': 2.355},
                                               {'name': 'bad1.jpg', 'rms_px': 2.202}]
              and cj.get('opencv_version') == '4.10.0',
              '"全量→剔 N→用剩下"这段事实完整落盘，不必再靠重算反推',
              f"rms={cj.get('rms_px')} used={len(cj.get('used_images') or [])} "
              f"dropped={len(cj.get('dropped_images') or [])}")

        core.save_calibration(K, D, SRC_SIZE)
        cj2 = json.loads(core.CALIB_JSON.read_text(encoding='utf-8'))
        check(not any(k in cj2 for k in
                      ('rms_px', 'used_images', 'dropped_images', 'opencv_version')),
              '没有 fit_result 时干脆不写这些键，而不是写一堆 null',
              '"字段缺失"与"标定时没记"是两件事，后者会被读成"真的没有被剔的图"')

        # 收尾：把这轮改过的全局还原，免得影响后续段落
        core.TABLE_FORMAT = 'txt'
        core.TABLE_FIXED_POINT = 4
        core.TABLE_SIZE = None
        core.DIR_TEST_IN = core.SCRIPT_DIR / 'test_input'
        core.DIR_TEST_OUT = core.SCRIPT_DIR / 'test_output'
    except Exception:
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
