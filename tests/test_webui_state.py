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
  D. /api/preview_gallery 的配对由服务端按真实文件给出，缺哪一侧就是 null，
     且绝不把别的照片的图串到空出来的那一侧
  E. 物理尺寸初值的优先级（已保存的 ipm_state > 新默认 45x45），以及 /api/preview
     回包里的"推荐值"——重置按钮靠它做成间接层，前端不抄常量
  F. 自动布局（近场贴底 + 最大不裁切）在 preview 与 commit 两侧同源：看到的 BirdView
     与导出的 H 必须是同一组 (anchor_y, scale)
  G. 导出成功之后的自动逆透视测试集（test_input/_auto_ipm）与成果画廊：
     选中的标点图不进测试集、用户自己的文件字节不动、准备过程中途失败不留半套、
     配对来自 batch_test 的实际返回、批测失败不影响已导出的矩阵与查找表
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote

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


def rel_of(url: str | None) -> str | None:
    """从 /api/image?rel=... 里取回相对路径，用来核对两侧指的是哪张文件。"""
    if not url:
        return None
    return unquote(url.split('rel=', 1)[1])


def calib_pairs(stems: list[str]) -> None:
    """在当前工程根下造出成对的标定原图与去畸变图。"""
    core.DIR_CALIB_IN.mkdir(parents=True, exist_ok=True)
    core.DIR_CALIB_PREVIEW.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(stems):
        core.safe_imwrite(core.DIR_CALIB_IN / f'{stem}.jpg',
                          np.full((16, 16, 3), 20 + 10 * i, np.uint8))
        core.safe_imwrite(core.DIR_CALIB_PREVIEW / f'{stem}_undist.jpg',
                          np.full((16, 16, 3), 21 + 10 * i, np.uint8))


def ipm_candidates(stems: list[str], size=(80, 80)) -> None:
    """在 ipm_input/ 造若干张地面候选图。"""
    core.DIR_IPM_IN.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(stems):
        core.safe_imwrite(core.DIR_IPM_IN / f'{stem}.jpg',
                          np.full((size[1], size[0], 3), 40 + 5 * i, np.uint8))


def identity_pair(size=(80, 80)):
    """一份恒等映射的 MapPair：batch_test 只用到 x/y/size/source_size。"""
    w, h = size
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    return core.MapPair(x=xs, y=ys, source_size=(w, h))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def names_in(folder: Path) -> list[str]:
    return [p.name for p in core.list_images(folder)]


def rel_str(path: Path) -> str:
    return server.safe_rel(path).replace('\\', '/')


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

        # ---- D. 成果图画廊：配对是服务端按真实文件算的，缺一侧只能是空态
        # 前端原来自己拼 calib_preview/<名字>_undist.jpg 去猜谁配谁，一旦某一侧
        # 少了文件，"并排对照"就可能把 A 的原图和 B 的去畸变图摆在一起——
        # 那比不显示更糟：验收的人会据此判断畸变矫正做对了。
        print('\n[D] /api/preview_gallery 配对与缺失空态')
        core.configure_paths(root=tmp / 'gallery_root')
        stems = ['shot_a', 'shot_b', 'shot_c']
        calib_pairs(stems)

        g = server.api_preview_gallery()
        check(g['count'] == 3 and g['paired'] == 3
              and g['missing_raw'] == 0 and g['missing_undistorted'] == 0,
              '三组齐全时 count/paired 都是 3',
              f"count={g['count']} paired={g['paired']}")
        check([it['name'] for it in g['items']] == stems,
              '每组有一个名字，顺序按名字排',
              str([it['name'] for it in g['items']]))
        check(all(rel_of(it['raw_url']) == f"calib_input/{it['name']}.jpg"
                  and rel_of(it['undistorted_url'])
                  == f"calib_preview/{it['name']}_undist.jpg"
                  for it in g['items']),
              '两个地址指向同一张照片的两个版本（服务端给的配对）',
              f"{rel_of(g['items'][0]['raw_url'])} | "
              f"{rel_of(g['items'][0]['undistorted_url'])}")

        # 去畸变图被改名：那一组的 undistorted_url 必须是 null，raw_url 照旧
        gone = core.DIR_CALIB_PREVIEW / 'shot_b_undist.jpg'
        gone.rename(gone.parent / (gone.name + '.bak'))
        g2 = server.api_preview_gallery()
        b = next(it for it in g2['items'] if it['name'] == 'shot_b')
        check(g2['count'] == 3 and g2['missing_undistorted'] == 1
              and b['undistorted_url'] is None
              and rel_of(b['raw_url']) == 'calib_input/shot_b.jpg',
              '缺去畸变图时该组只报空态，原图仍是自己的那张', str(b))
        check(all(rel_of(it['undistorted_url']) == f"calib_preview/{it['name']}_undist.jpg"
                  for it in g2['items'] if it['name'] != 'shot_b'),
              '缺的那一侧没有串到别的照片上（其余两组配对不变）',
              str([rel_of(it['undistorted_url']) for it in g2['items']]))
        gone.parent.joinpath(gone.name + '.bak').rename(gone)

        # 反过来：原图被删，去畸变图还在，那一组的 raw_url 为 null
        (core.DIR_CALIB_IN / 'shot_c.jpg').unlink()
        g3 = server.api_preview_gallery()
        c = next(it for it in g3['items'] if it['name'] == 'shot_c')
        check(g3['count'] == 3 and g3['missing_raw'] == 1 and c['raw_url'] is None
              and rel_of(c['undistorted_url']) == 'calib_preview/shot_c_undist.jpg',
              '缺原图时该组仍在列，只是原图侧为空', str(c))

        # ---- E. 物理尺寸的初值与"推荐值"这个间接层
        # 默认值统一成 45x45 之后最危险的回归是"新默认把用户上次实测的尺寸顶掉"，
        # 所以这条优先级要有断言钉住；推荐值则必须由服务端给，前端不许抄常量。
        print('\n[E] 物理尺寸初值的优先级与 /api/preview 的推荐值')
        check((core.PHYS_W_CM, core.PHYS_H_CM) == (45.0, 45.0),
              'Python 侧 fallback 就是 45x45（网页与 CLI/API 同一组默认值）',
              f'{core.PHYS_W_CM} x {core.PHYS_H_CM}')

        core.configure_paths(root=tmp / 'phys_root')
        check(server.initial_phys_size() == (45.0, 45.0),
              '没有 ipm_state 时用默认值', str(server.initial_phys_size()))
        hist = core.ipm_state_path()
        hist.parent.mkdir(parents=True, exist_ok=True)
        hist.write_text(json.dumps({'schema_version': 1, 'phys_w_cm': 33.0,
                                    'phys_h_cm': 22.0}), encoding='utf-8')
        check(server.initial_phys_size() == (33.0, 22.0),
              '有 ipm_state 时历史尺寸优先，新默认不得覆盖',
              str(server.initial_phys_size()))
        check(server.api_status()['phys_init'] == {'w': 33.0, 'h': 22.0},
              '/api/status 把这份初值直接交给前端（页面不再自己判优先级）',
              str(server.api_status()['phys_init']))
        hist.write_text(json.dumps({'schema_version': 1, 'phys_w_cm': 0,
                                    'phys_h_cm': 22.0}), encoding='utf-8')
        check(server.initial_phys_size() == (45.0, 45.0),
              '历史里是非正数（脏数据）时退回默认，不把 0 当尺寸用',
              str(server.initial_phys_size()))

        # 推荐值：服务端算、随预览一起回，口径就是自动布局（近场贴底 + 最大不裁切）
        old_undist = server.STATE['src_undist']
        server.STATE['src_undist'] = np.full((240, 320, 3), 128, np.uint8)
        try:
            pv = server.api_preview({
                'quad': [[90, 80], [230, 80], [30, 200], [290, 200]],
                'phys_w': 45.0, 'phys_h': 45.0})
        finally:
            server.STATE['src_undist'] = old_undist
        rec = pv.get('recommended')
        check(isinstance(rec, dict) and sorted(rec) == ['anchor_y', 'scale', 'source'],
              '/api/preview 回包里有 recommended{anchor_y, scale, source}', str(rec))
        check(rec['source'] in ('auto', 'fallback'),
              '回包能区分"自动布局给的"还是 fallback', str(rec['source']))
        check(abs(pv['scale'] - rec['scale']) < 1e-9,
              '未指定 scale 时预览用的就是推荐值', f"{pv['scale']:.6f}")
        check(abs(pv['anchor_y'] - rec['anchor_y']) < 1e-9,
              '未指定 scale 时 anchor_y 也由推荐值定（自动布局同时定两项）',
              f"{pv['anchor_y']:.6f}")

        # ---- F. 自动布局：preview 与 commit 必须同源
        # U3 留下的 recommended_layout 只被两个重置按钮读，预览与导出各自还在用
        # "当前 anchor_y + 0.8 x max_scale"。那样界面上显示的推荐值和真正生效的 H
        # 是两回事；下面这几条把"显示、预览、导出三者同一组参数"钉住。
        print('\n[F] 自动布局在 preview / commit 两侧同源')
        core.configure_paths(root=tmp / 'auto_root')
        quad = [[90, 80], [230, 80], [30, 200], [290, 200]]
        quad_arr = core.order_corners_tl_tr_bl_br(np.asarray(quad, dtype=np.float64))
        auto = {'quad': quad, 'phys_w': 45.0, 'phys_h': 45.0, 'anchor_x': 0.5,
                'anchor_y': 0.3, 'heading': 0.0, 'scale': None}
        old_state = {k: server.STATE[k] for k in
                     ('K', 'D', 'Knew', 'img_size', 'src_path', 'src_undist')}
        originals = {name: getattr(core, name) for name in
                     ('export_all', 'build_ipm_state', 'batch_test', 'safe_imwrite')}
        recorded: dict = {}

        def fake_export_all(K, D, Knew, H, H0, sign, extra, size, ipm_state=None):
            """只记下导出用的 H 与参数：导出事务本身另有专门的测试文件。"""
            recorded['H'] = np.asarray(H, dtype=np.float64).copy()
            recorded['extra'] = dict(extra)
            return None

        try:
            server.STATE.update(src_undist=np.full((240, 320, 3), 128, np.uint8),
                                K=np.eye(3), D=np.zeros((4, 1)), Knew=np.eye(3),
                                img_size=(320, 240), src_path=None)
            core.export_all = fake_export_all
            core.build_ipm_state = lambda *a, **kw: {}
            # 签名跟着 batch_test 的新形状走（可指定目录、返回实际生成的配对）
            core.batch_test = lambda pair, input_dir=None, output_dir=None: []
            core.safe_imwrite = lambda *a, **kw: None

            pv = server.api_preview(dict(auto))
            rec = pv['recommended']
            check(rec['source'] == 'auto',
                  '这组几何下拿到的是自动布局解（不是 fallback）', str(rec))
            check(abs(pv['anchor_y'] - rec['anchor_y']) < 1e-12
                  and abs(pv['scale'] - rec['scale']) < 1e-12,
                  'preview 的 BirdView/H 真正采用了 recommended 的 anchor_y + scale',
                  f"用了 ay={pv['anchor_y']:.6f} k={pv['scale']:.6f}")
            check(abs(pv['anchor_y'] - 0.3) > 1e-6,
                  '自动模式下 anchor_y 由服务端决定（旧实现只改 scale，会停在 0.3）',
                  f"{pv['anchor_y']:.6f}")
            legacy = max(core.MIN_SCALE, pv['max_scale'] * core.INIT_SCALE_RATIO)
            check(abs(pv['scale'] - legacy) > 1e-6,
                  '用的不是旧的 0.8 x max_scale',
                  f"新={pv['scale']:.6f} 旧口径={legacy:.6f}")

            # 重置按钮读的还是同一个 recommended_layout，且与当前 anchor_y/scale 无关
            cal = server.make_calibrator(quad_arr, 45.0, 45.0, 0.5, 0.11, 0.0, 3.0)
            rec2 = server.recommended_layout(cal)
            check(rec2['source'] == rec['source']
                  and abs(rec2['anchor_y'] - rec['anchor_y']) < 1e-12
                  and abs(rec2['scale'] - rec['scale']) < 1e-12,
                  '两个重置按钮读的 recommended_layout 与预览同一个来源', str(rec2))

            # 最重要的一条：自动模式下 commit 导出的 H 必须是预览那一组参数算出来的
            server.api_commit({**auto, 'table_format': 'txt'})
            ref = server.make_calibrator(quad_arr, 45.0, 45.0, 0.5,
                                         rec['anchor_y'], 0.0, rec['scale'])
            check(np.allclose(recorded['H'], ref.H, rtol=0, atol=1e-12),
                  '自动 commit 导出的 H == 用预览那一组 (anchor_y, scale) 算出的 H',
                  f"最大差 {np.abs(recorded['H'] - ref.H).max():.3e}")
            check(abs(recorded['extra']['anchor_y'] - rec['anchor_y']) < 1e-12
                  and abs(recorded['extra']['scale_px_per_cm'] - rec['scale']) < 1e-12,
                  '落盘的 anchor_y / scale 也是自动布局那一组',
                  f"{recorded['extra']['anchor_y']:.6f} "
                  f"{recorded['extra']['scale_px_per_cm']:.6f}")

            # 手动模式：显式给的值一个字都不许改
            manual = {**auto, 'anchor_y': 0.42, 'scale': 2.0}
            mv = server.api_preview(dict(manual))
            check(abs(mv['scale'] - 2.0) < 1e-9 and abs(mv['anchor_y'] - 0.42) < 1e-9,
                  '手动模式的 preview 完全不覆盖用户值',
                  f"ay={mv['anchor_y']} k={mv['scale']}")
            recorded.clear()
            server.api_commit({**manual, 'table_format': 'txt'})
            check(abs(recorded['extra']['anchor_y'] - 0.42) < 1e-12
                  and abs(recorded['extra']['scale_px_per_cm'] - 2.0) < 1e-12,
                  '手动模式的 commit 也不覆盖用户值',
                  f"{recorded['extra']['anchor_y']} "
                  f"{recorded['extra']['scale_px_per_cm']}")

            # 无可行布局时退回旧口径，但预览不许因此崩掉，且回包标明是 fallback。
            # anchor_y 必须**保留用户当前值**：cal.max_scale 就是按这个 anchor_y 算的，
            # 若返回时改成 ANCHOR_Y，那个 scale 就不再对应刚才的上限。
            fb = server.api_preview({**auto, 'anchor_x': 2.0, 'anchor_y': 0.63})
            check(fb['recommended']['source'] == 'fallback',
                  '无可行布局时标明是 fallback，预览仍出图', str(fb['recommended']))
            check(abs(fb['recommended']['anchor_y'] - 0.63) < 1e-12,
                  'fallback 保留当前 anchor_y，不抹回模块常量 ANCHOR_Y',
                  f"{fb['recommended']['anchor_y']} (ANCHOR_Y={core.ANCHOR_Y})")
            check(abs(fb['anchor_y'] - 0.63) < 1e-12,
                  'fallback 下实际生效的 anchor_y 也是用户那一个', str(fb['anchor_y']))

            # 幂等：保留 anchor_y 之后 recompute 不改变 max_scale，重复走 fallback
            # 必须得到同一组值。改回 ANCHOR_Y 的旧实现在这里会漂。
            fb2 = server.api_preview({**auto, 'anchor_x': 2.0, 'anchor_y': 0.63})
            check(abs(fb2['recommended']['anchor_y'] - fb['recommended']['anchor_y']) < 1e-12
                  and abs(fb2['recommended']['scale'] - fb['recommended']['scale']) < 1e-12,
                  'fallback 幂等：再求一次得到同一组 (anchor_y, scale)',
                  f"{fb['recommended']['scale']:.9f} -> {fb2['recommended']['scale']:.9f}")
        finally:
            for name, fn in originals.items():
                setattr(core, name, fn)
            server.STATE.update(**old_state)

        # ---- G. 导出成功之后：自动逆透视测试集 + 成果画廊
        # 这一刀的风险全在"数据所有权"上：_auto_ipm/ 是工具的缓存可以整体刷新，
        # test_input/ 根目录是用户的东西，一个字节都不能碰。再加上"配对必须来自
        # batch 的实际返回"——按目录 stem 反猜是标定画廊已经踩过的坑。
        print('\n[G] 自动测试集 _auto_ipm 与逆透视成果画廊')
        core.configure_paths(root=tmp / 'auto_ipm_root')
        old_state = {k: server.STATE[k] for k in
                     ('K', 'D', 'Knew', 'img_size', 'src_path', 'src_undist',
                      'ipm_results')}
        originals = {name: getattr(core, name) for name in
                     ('export_all', 'build_ipm_state', 'batch_test',
                      'load_exported_reverse_pair')}
        real_batch = core.batch_test
        try:
            ipm_candidates([f'cand_{i}' for i in range(8)])
            picked = core.DIR_IPM_IN / 'cand_3.jpg'

            # 用户自己放在 test_input/ 根目录的东西：图片、说明文件，还有上次的结果
            core.DIR_TEST_IN.mkdir(parents=True, exist_ok=True)
            core.DIR_TEST_OUT.mkdir(parents=True, exist_ok=True)
            user_img = core.DIR_TEST_IN / 'user_own.jpg'
            core.safe_imwrite(user_img, np.full((80, 80, 3), 7, np.uint8))
            user_txt = core.DIR_TEST_IN / 'readme.txt'
            user_txt.write_text('我自己的测试集说明', encoding='utf-8')
            user_out = core.DIR_TEST_OUT / 'old_result.jpg'
            core.safe_imwrite(user_out, np.full((8, 8, 3), 9, np.uint8))
            before = {p: sha(p) for p in (user_img, user_txt, user_out)}

            # 导出那一步写的 BirdView 结果图（画廊第一项的右侧就是它）
            core.safe_imwrite(core.IPM_RESULT, np.full((40, 40, 3), 200, np.uint8))
            server.STATE.update(src_path=picked, ipm_results=[])

            auto_in, stage, auto_out, out_stage = server.auto_test_dirs()
            res = server.run_auto_batch(identity_pair())
            check(len(names_in(auto_in)) == 7,
                  'ipm_input 有 8 张、选 1 张 → _auto_ipm 正好 7 张',
                  f'{len(names_in(auto_in))} 张: {names_in(auto_in)}')
            check('cand_3.jpg' not in names_in(auto_in),
                  '被选中的标点图绝不进入自动测试集', str(names_in(auto_in)))
            check(all(sha(p) == h for p, h in before.items()),
                  'test_input / test_output 根目录里用户自己的文件字节级不变',
                  ', '.join(p.name for p in before))
            check(not stage.exists(), '换装成功后暂存区不留残留', str(stage.name))
            check(res == {'ok': True, 'generated': 7, 'error': None},
                  'run_auto_batch 报告 7 张成果', str(res))
            check(len(names_in(auto_out)) == 7,
                  '自动结果实际落在 test_output/_auto_ipm',
                  f'{rel_str(auto_out)}: {len(names_in(auto_out))} 张')

            g = server.api_ipm_result_gallery()
            first = g['items'][0]
            check(g['count'] == 8 and first['role'] == 'calibration'
                  and first['name'] == 'cand_3.jpg',
                  '画廊第一项是参与四点标定的那张基准图',
                  f"{first['name']} / {first['role']}")
            check(rel_of(first['raw_url']) == 'ipm_input/cand_3.jpg'
                  and rel_of(first['birdview_url'])
                  == 'ipm_output/UnDistortionInverseImage.jpg',
                  '基准图那一组是「原图 ↔ 当前最终 BirdView（IPM_RESULT）」',
                  f"{rel_of(first['raw_url'])} | {rel_of(first['birdview_url'])}")
            rest = g['items'][1:]
            check(all(it['role'] == 'test' for it in rest)
                  and 'cand_3.jpg' not in [it['name'] for it in rest],
                  '其后各项都是没参与标点的测试图',
                  str([it['name'] for it in rest]))
            check(all(rel_of(it['raw_url']) == f"test_input/_auto_ipm/{it['name']}"
                      and rel_of(it['birdview_url'])
                      == f"test_output/_auto_ipm/{Path(it['name']).stem}_birdview.jpg"
                      for it in rest),
                  'Gallery 的 raw ↔ birdview 配对来自实际 batch 返回记录',
                  f"{rel_of(rest[0]['raw_url'])} | {rel_of(rest[0]['birdview_url'])}")

            # 第二次换标点图：原来那张要进来、新选的那张要出去
            server.STATE.update(src_path=core.DIR_IPM_IN / 'cand_5.jpg')
            server.run_auto_batch(identity_pair())
            names2 = names_in(auto_in)
            check('cand_3.jpg' in names2 and 'cand_5.jpg' not in names2
                  and len(names2) == 7,
                  '第二次换标点图 → 自动测试集正确刷新', str(names2))

            # 准备阶段中途失败：旧的自动测试集必须完好，绝不留半套
            before_names = names_in(auto_in)
            real_copy2 = shutil.copy2
            calls = {'n': 0}

            def flaky_copy2(src, dst, *a, **kw):
                calls['n'] += 1
                if calls['n'] == 4:
                    raise OSError('磁盘已满（故障注入）')
                return real_copy2(src, dst, *a, **kw)

            shutil.copy2 = flaky_copy2
            try:
                server.run_auto_batch(identity_pair())
                check(False, '复制第 4 张失败时整条准备流程中止')
            except OSError as exc:
                check('故障注入' in str(exc), '复制第 4 张失败时整条准备流程中止',
                      str(exc))
            finally:
                shutil.copy2 = real_copy2
            check(names_in(auto_in) == before_names,
                  '自动输入准备中途失败 → 旧自动集不变、无半套', str(names_in(auto_in)))
            check(not stage.exists(), '失败后暂存区被清掉', str(stage.name))

            # 坏图与宽高比不符的图：被 batch 跳过之后画廊不能虚构出成果
            (core.DIR_IPM_IN / 'broken.jpg').write_bytes(b'not an image at all')
            core.safe_imwrite(core.DIR_IPM_IN / 'wide.jpg',
                              np.full((40, 200, 3), 60, np.uint8))
            server.STATE.update(src_path=picked)
            res = server.run_auto_batch(identity_pair())
            check(len(names_in(auto_in)) == 9 and res['generated'] == 7,
                  '坏图/比例不符的图进了输入，但没有生成结果',
                  f"输入 {len(names_in(auto_in))} 张 → 成果 {res['generated']} 张")
            g = server.api_ipm_result_gallery()
            shown = [it['name'] for it in g['items']]
            check('broken.jpg' not in shown and 'wide.jpg' not in shown
                  and g['count'] == 8,
                  '坏图/比例不符被跳过后 gallery 不虚构结果', str(shown))
            (core.DIR_IPM_IN / 'broken.jpg').unlink()
            (core.DIR_IPM_IN / 'wide.jpg').unlink()

            # batch_test 的默认参数一个字没变：不给目录就读写 test_input/test_output 根
            done = core.batch_test(identity_pair())
            check([(rel_str(a), rel_str(b)) for a, b in done]
                  == [('test_input/user_own.jpg', 'test_output/user_own_birdview.jpg')],
                  'batch_test 不带 input_dir/output_dir 时仍是 test_input/ 根目录语义',
                  str([(rel_str(a), rel_str(b)) for a, b in done]))

            # ---- api_commit：batch 必须用 export_all 返回的那个 pair
            sentinel = identity_pair()
            seen: dict = {}

            def fake_export_all(K, D, Knew, H, H0, sign, extra, size, ipm_state=None):
                """只落两个产物标记：导出事务本身另有专门的测试文件。"""
                core.DIR_MATRIX.mkdir(parents=True, exist_ok=True)
                (core.DIR_MATRIX / 'matrices.json').write_text('{}', encoding='utf-8')
                lut = core.DIR_TABLE / 'undistort_ipm' / 'reverse'
                lut.mkdir(parents=True, exist_ok=True)
                (lut / 'MapW.txt').write_text('0', encoding='utf-8')
                return sentinel

            def spy_batch(pair, input_dir=None, output_dir=None):
                seen.update(pair=pair, input_dir=input_dir, output_dir=output_dir)
                return real_batch(pair, input_dir=input_dir, output_dir=output_dir)

            core.export_all = fake_export_all
            core.build_ipm_state = lambda *a, **kw: {}
            core.batch_test = spy_batch
            server.STATE.update(src_undist=np.full((240, 320, 3), 128, np.uint8),
                                K=np.eye(3), D=np.zeros((4, 1)), Knew=np.eye(3),
                                img_size=(320, 240), src_path=picked, ipm_results=[])
            body = {'quad': [[90, 80], [230, 80], [30, 200], [290, 200]],
                    'phys_w': 45.0, 'phys_h': 45.0, 'anchor_x': 0.5,
                    'anchor_y': 0.3, 'heading': 0.0, 'scale': 3.0,
                    'table_format': 'txt'}
            out = server.api_commit(dict(body))
            check(seen.get('pair') is sentinel,
                  'batch 用的是 export_all 返回的同一个最终 MapPair（不是重新算的）',
                  f"同一对象: {seen.get('pair') is sentinel}")
            # P1.1 之后 batch_test 写的是输出暂存区，全部完成才换装成 _auto_ipm；
            # 输入侧仍是 _auto_ipm。两侧都不碰 test_input / test_output 根目录。
            check(seen.get('input_dir') == auto_in
                  and seen.get('output_dir') == out_stage,
                  '自动批测读 _auto_ipm、写输出暂存区，都不碰根目录',
                  f"{rel_str(seen['input_dir'])} → {rel_str(seen['output_dir'])}")
            check(out['export_ok'] is True and out['batch']['ok'] is True
                  and out['batch']['generated'] == 7,
                  'api_commit 分别报告 export_ok 与 batch', str(out['batch']))

            # ---- batch 炸了：矩阵与查找表照样在盘上，接口要说清楚是两件事
            def boom_batch(pair, input_dir=None, output_dir=None):
                raise RuntimeError('批量测试炸了（故障注入）')

            core.batch_test = boom_batch
            out = server.api_commit(dict(body))
            check(out['export_ok'] is True and out['batch']['ok'] is False
                  and '故障注入' in (out['batch']['error'] or ''),
                  'batch 失败时接口明确报告 export_ok=true + batch.ok=false',
                  str(out['batch']))
            check((core.DIR_MATRIX / 'matrices.json').is_file()
                  and (core.DIR_TABLE / 'undistort_ipm' / 'reverse' / 'MapW.txt').is_file(),
                  'batch 失败后 matrices/LUT 仍然存在（批测不属于导出事务）')
            check('矩阵与查找表已导出成功' in out['log'],
                  '日志里也把两件事分开说，而不是一句"导出失败"',
                  out['log'].strip().splitlines()[-1][:40])

            # ---- 单独点「只重跑批量测试」：仍是 test_input 根目录语义
            core.batch_test = spy_batch
            core.load_exported_reverse_pair = lambda size: sentinel
            seen.clear()
            server.api_batch()
            check(seen.get('input_dir') is None and seen.get('output_dir') is None,
                  'api_batch 不传目录 → 保持原来 test_input 根目录的语义',
                  f"input_dir={seen.get('input_dir')}")

            # ---- P1.1：成果必须对应刚刚那一份 LUT，异常路径也不许例外
            # 先跑一轮正常的，让磁盘与 STATE 都留下"上一轮"的成果
            server.STATE.update(src_path=picked, ipm_results=[])
            prev = server.run_auto_batch(identity_pair())
            prev_out = names_in(auto_out)
            check(prev['generated'] == 7 and len(prev_out) == 7,
                  '上一轮：7 张成果落盘并进了画廊记录', f"{prev['generated']} / {prev_out}")

            # 本轮 batch 故障：画廊必须回到 0，绝不能继续展示上一轮的配对
            boom = core.batch_test

            def batch_boom(*_a, **_k):
                raise OSError('本轮批测炸了（故障注入）')

            core.batch_test = batch_boom
            try:
                res_fail = server.api_commit(dict(body))
            finally:
                core.batch_test = boom
            check(res_fail['export_ok'] is True and res_fail['batch']['ok'] is False,
                  '本轮 batch 故障：仍然 export_ok=true', str(res_fail['batch']))
            check(server.api_ipm_result_gallery()['count'] == 0,
                  '本轮 batch 故障 → 画廊 count=0，不展示上一轮的旧配对',
                  str(server.api_ipm_result_gallery()['count']))
            check(len(server.STATE['ipm_results']) == 0,
                  '成果记录在本轮开始批测前就已作废')

            # 上轮 7 张、本轮只有 6 张成功 → 正式 output 恰好 6 张，无孤儿旧文件
            server.STATE.update(src_path=picked, ipm_results=[])
            server.run_auto_batch(identity_pair())
            check(len(names_in(auto_out)) == 7, '重新跑通后又回到 7 张')
            # 从**候选来源**删掉一张。删 _auto_ipm 里的那份没用：run_auto_batch 每轮都会
            # 从 ipm_input/ 重新刷新输入集，删了会被原样补回来。
            (core.DIR_IPM_IN / 'cand_0.jpg').unlink()
            res6 = server.run_auto_batch(identity_pair())
            final = names_in(auto_out)
            check(res6['generated'] == 6 and len(final) == 6
                  and 'cand_0_birdview.jpg' not in final,
                  '上轮 7 张、本轮 6 张成功 → 正式 output 正好 6 张，没有孤儿旧文件',
                  f'{len(final)} 张: {final}')

            # 写到第 N 张故障 → 旧正式 output 完整保留，输出暂存区清理干净
            before_out = names_in(auto_out)
            real_write = core.safe_imwrite
            calls = [0]

            def flaky_write(path, img, *a, **k):
                if '_birdview' in Path(path).name:
                    calls[0] += 1
                    if calls[0] == 3:
                        raise OSError('写第 3 张成果时磁盘满（故障注入）')
                return real_write(path, img, *a, **k)

            core.safe_imwrite = flaky_write
            try:
                server.run_auto_batch(identity_pair())
                check(False, 'batch 写盘故障应当抛出来')
            except OSError as exc:
                check('磁盘满' in str(exc), 'batch 写到第 3 张时故障抛出', str(exc))
            finally:
                core.safe_imwrite = real_write
            check(names_in(auto_out) == before_out,
                  'batch 中途故障 → 旧正式 output 完整保留（换装前一个字节都没动）',
                  f'{before_out} -> {names_in(auto_out)}')
            check(not out_stage.exists() or not any(out_stage.iterdir()),
                  '故障后输出暂存区已清理', out_stage.name)
        finally:
            for name, fn in originals.items():
                setattr(core, name, fn)
            server.STATE.update(**old_state)
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
