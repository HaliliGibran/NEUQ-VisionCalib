"""标定板规格的回归测试。

    python tests/test_board_spec.py

为什么这个必须有测试：棋盘规格不只喂给相机标定，**素材导入时判断"这张是棋盘照
还是地面照"用的也是它**。规格填错，第一步分拣就已经错了，而错误会一路带到
逆透视候选里，最后表现为"莫名其妙混进来一张棋盘照"。

覆盖：
  A. 规格对象本身：换算、校验、局部子网格候选的推导
  B. 命令行解析：--board-squares / --board-corners 互斥、部分参数只改那一项
  C. 检测行为：同一张合成棋盘，规格对能检出、规格错检不出
  D. calib.json 往返：写入含 board 段，读取兼容老格式
  E. 真的调用一次相机标定
  F. project.json 跨重启、原子写
  G. 素材库规格状态机：material_set 不被增量导入洗白、覆盖导入的事务性
  H. 状态层边界：_incomplete/ 也算素材、stale 拦住逆透视取图、replace 遇坏图整批
     放弃、project.json 读不动时 fail closed、库内显式指定仍要过闸、
     replace --move 换装中途失败时原图一张不丢
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import neuq_vision_calib as core  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=''):
    print(('  [OK] ' if cond else '  [!!] ') + label + (f'  {detail}' if detail else ''))
    if not cond:
        FAILED.append(label)
    return bool(cond)


def synth_board_image(spec, margin=60, px=45):
    """按规格画一张棋盘照片（12x9 方格 → 11x8 内角点）。"""
    w = spec.squares_x * px + margin * 2
    h = spec.squares_y * px + margin * 2
    img = np.full((h, w), 255, np.uint8)
    for i in range(spec.squares_x):
        for j in range(spec.squares_y):
            if (i + j) % 2 == 0:
                x0, y0 = margin + i * px, margin + j * px
                img[y0:y0 + px, x0:x0 + px] = 20
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def synth_pose(base, seed):
    """把棋盘图做一点透视扰动，模拟"换角度多拍几张"。"""
    h, w = base.shape[:2]
    rng = np.random.default_rng(seed)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + (rng.uniform(-0.05, 0.05, size=(4, 2)) * np.float32([w, h])).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(base, M, (w, h), borderValue=(255, 255, 255))


def tree_fingerprint(roots) -> str:
    """一组目录的内容指纹（相对路径 + 字节），用来断言"一个字节都没动"。"""
    import hashlib
    h = hashlib.sha256()
    for root in sorted(roots, key=str):
        h.update(str(root).encode('utf-8'))
        for p in sorted(root.rglob('*'), key=str):
            if p.is_file():
                h.update(str(p.relative_to(root)).encode('utf-8'))
                h.update(p.read_bytes())
    return h.hexdigest()[:16]


def failing_rename_into(dest: Path):
    """让"把某个 *.staging 换装成 dest"这一步失败，其余 rename 放行。

    刻意按目标目录匹配而不是数第几次 rename：中间夹进一次 project.json 的原子写
    就会把计数错位，而这个测试关心的恰恰是"第二个目录换装失败"这一个精确时刻。
    返回还原函数。
    """
    import pathlib
    orig = pathlib.Path.rename

    def fake(self, target, *a, **kw):
        if Path(target).name == dest.name and str(self).endswith('.staging'):
            raise OSError(f'模拟把 {Path(self).name} 换装成 {dest.name} 时失败')
        return orig(self, target, *a, **kw)

    pathlib.Path.rename = fake
    return lambda: setattr(pathlib.Path, 'rename', orig)


def scenario_calibration() -> None:
    """端到端真的跑一次相机标定。

    这一条是补课：上一轮把 CHESSBOARD_CORNERS 换成 board.corners 时，
    calibrate_camera() 里根本没有 board 这个变量，"运行相机标定"直接 NameError。
    当时测试把规格对象、检测、JSON 都测了，唯独没按下那个按钮。
    """
    tmp = Path(tempfile.mkdtemp(prefix='neuq_calib_'))
    old_root = core.SCRIPT_DIR
    old_board = core.BOARD
    try:
        spec = core.CheckerboardSpec(12, 9, 20.0)
        core.configure_paths(root=tmp)
        core.configure_board(spec)
        core.DIR_CALIB_IN.mkdir(parents=True, exist_ok=True)
        base = synth_board_image(spec, margin=50)
        for i in range(6):
            core.safe_imwrite(core.DIR_CALIB_IN / f'view_{i:02d}.jpg',
                              synth_pose(base, i))
        K, D, size, fit = core.calibrate_camera()
        check(K.shape == (3, 3), '标定返回 3x3 内参矩阵', f'{K.shape}')
        check(size == (base.shape[1], base.shape[0]),
              '标定分辨率等于图片分辨率', f'{size}')
        check(float(K[0, 0]) > 0 and float(K[1, 1]) > 0,
              '焦距为正', f'fx={float(K[0, 0]):.1f}')
        rms = float(fit.get('rms', -1)) if isinstance(fit, dict) else -1.0
        check(0 <= rms < 3.0, '重投影 RMS 在合理范围', f'{rms:.3f} px')

        # 规格不对时必须明确失败，而不是"标定成功但参数没意义"
        core.configure_board(core.CheckerboardSpec(8, 6, 20.0))
        try:
            core.calibrate_camera()
            check(False, '规格与照片不符时报错而不是硬标')
        except SystemExit:
            check(True, '规格与照片不符时报错而不是硬标')
        core.configure_board(spec)

        # provenance：已有 calib.json 的规格与当前不一致时要拦住复用
        core.save_calibration(K, D, size)
        core.configure_board(core.CheckerboardSpec(9, 7, 25.0))
        conflict = core.board_conflict(core.calib_board_meta())
        check(conflict is not None and '不一致' in conflict,
              '规格与已有 calib.json 不符时给出明确冲突说明',
              (conflict or '').splitlines()[0] if conflict else '')
        try:
            core.load_or_run_calibration(force=False)
            check(False, '规格冲突时拒绝静默复用旧标定')
        except SystemExit as exc:
            check('不一致' in str(exc), '规格冲突时拒绝静默复用旧标定')
        K2, _D2, _Knew2, size2 = core.load_or_run_calibration(force=True)
        check(K2.shape == (3, 3) and size2 == size,
              'force=True 时重新标定并可继续', f'{size2}')
    finally:
        core.configure_paths(root=old_root)
        core.configure_board(old_board)
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


def scenario_persistence() -> None:
    """棋盘规格必须能跨重启存活。

    只放在进程全局里的话：设好 9x7 → 导入素材 → 关掉程序，
    第二天重开又变成 12x9，接着导入的新照片就会按另一套规格分类。
    """
    tmp = Path(tempfile.mkdtemp(prefix='neuq_proj_'))
    old_root = core.SCRIPT_DIR
    old_board = core.BOARD
    try:
        core.configure_paths(root=tmp)
        check(core.BOARD == core.DEFAULT_BOARD,
              '新工程默认用 12x9 / 20mm', str(core.BOARD.squares_x))

        spec = core.CheckerboardSpec(9, 7, 25.0)
        core.configure_board(spec, persist=True)
        cfg = core.load_project_config()
        check((cfg.get('board') or {}).get('squares_x') == 9,
              '设置后立即写进 project.json',
              str((cfg.get('board') or {}).get('squares_x')))
        check(cfg.get('schema_version') == core.SCHEMA_VERSION,
              'project.json 带 schema_version')

        # 模拟重启：清空进程状态，走启动路径重新解析工程根
        core.BOARD = core.DEFAULT_BOARD
        core.configure_paths(root=tmp)
        check(spec == core.BOARD, '重启后规格被恢复',
              f'{core.BOARD.squares_x}x{core.BOARD.squares_y}/{core.BOARD.square_size_mm}')

        # 未显式给参数时，启动不得把 project.json 覆盖回默认值
        core.BOARD = core.DEFAULT_BOARD
        core.configure_paths(root=tmp)
        check((core.load_project_config().get('board') or {}).get('squares_x') == 9,
              '启动不会把已保存的规格冲掉')

        # 素材按旧规格分类后改规格 → 必须提示
        core.DIR_CALIB_IN.mkdir(parents=True, exist_ok=True)
        core.safe_imwrite(core.DIR_CALIB_IN / 'a.jpg',
                          np.zeros((40, 40, 3), dtype=np.uint8))
        core.set_material_board(spec)
        other = core.CheckerboardSpec(12, 9, 20.0)
        stale = core.material_stale_reason(other)
        check(stale is not None and '不一致' in stale,
              '改规格后提示素材已按旧规格分类',
              (stale or '').splitlines()[0] if stale else '')
        check(core.material_stale_reason(spec) is None,
              '规格没变则不报警')

        # project.json 原子写：不留 .tmp，读回来是完整的
        names = [p.name for p in tmp.iterdir() if p.is_file()]
        check(not any(n.endswith('.tmp') for n in names),
              '写配置不留 .tmp 中间文件', str(names))

        # 备份并清空会作废素材依据：否则下一次增量导入拿着一条对不上号的旧记录判 stale
        core.clear_material_board()
        check(core.load_project_config().get('material_set') is None,
              '清空素材后 material_set 一并删除')
        check(core.material_stale_reason(other) is None,
              '没有依据时不再凭空报警')
    finally:
        core.configure_paths(root=old_root)
        core.configure_board(old_board, persist=False)
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


def scenario_material_set() -> None:
    """素材库的规格是"整套素材"的属性，不是"最近一次导入"的属性。

    老模型把规格记在 last_import 上，于是这条路径能把脏状态洗白：
      按 A 导入 → 改成 B → 增量导入一批 B → last_import.board = B → 系统认为
      整个库都是 B，而磁盘上实际是「旧 A 素材 + 新 B 素材」。
    """
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix='neuq_matset_'))
    old_root, old_board = core.SCRIPT_DIR, core.BOARD
    spec_a = core.CheckerboardSpec(12, 9, 20.0)
    spec_b = core.CheckerboardSpec(9, 7, 25.0)

    def add_floors(folder: Path, n: int) -> Path:
        """造几张纯灰图：任何规格都检不出棋盘，一律归到 ipm_input/。"""
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            core.safe_imwrite(folder / f'{folder.name}_{i}.jpg',
                              np.full((80, 80, 3), 128, np.uint8))
        return folder

    try:
        core.configure_paths(root=tmp / 'proj')
        core.configure_board(spec_a)
        src1 = add_floors(tmp / 'batch_a', 2)

        # ---- A. 空库首次导入：任何规格都可以，并确立依据
        got = core.import_dataset(src1, mode='add')
        check(core.material_board() == spec_a, '首次导入确立 material_set.board',
              str(core.material_board()))
        check(got['imported_ipm'] == 2, '两张地面照入库', str(got['imported_ipm']))

        # ---- B. 换规格 → stale，且增量导入不能洗白
        core.configure_board(spec_b, persist=False)
        check(core.material_stale_reason() is not None, '换规格后判为 stale')
        src2 = add_floors(tmp / 'batch_b', 2)
        try:
            core.import_dataset(src2, mode='add')
            check(False, 'stale 时拒绝增量导入')
        except SystemExit as exc:
            check('增量导入已中止' in str(exc), 'stale 时拒绝增量导入',
                  str(exc).splitlines()[0])
        check(core.material_board() == spec_a, '被拒绝的导入没有改写依据',
              str(core.material_board()))
        check(len(core.list_images(core.DIR_IPM_IN)) == 2, '被拒绝的导入没往库里加东西')
        # 光警告不够：stale 时相机标定也要拦得住。旧标准挑进来的那批图，用新规格
        # 逐张重检会悄悄剔掉一大半，剩下两三张照样标得出参数，只是没人会去怀疑。
        try:
            core.calibrate_camera()
            check(False, 'stale 时拒绝相机标定')
        except SystemExit as exc:
            check('相机标定已中止' in str(exc), 'stale 时拒绝相机标定',
                  str(exc).splitlines()[0])

        # ---- C. 只有 replace 允许换规格，成功后才改写依据
        core.import_dataset(src2, mode='replace')
        check(core.material_board() == spec_b, '覆盖导入后才换依据',
              str(core.material_board()))
        check(core.material_stale_reason() is None, '覆盖导入后不再 stale')
        check(not any(p.is_dir() and ('.staging' in p.name or '.old' in p.name)
                      for p in (tmp / 'proj').iterdir()),
              '成功后不留 .staging / .old',
              str([p.name for p in (tmp / 'proj').iterdir() if p.is_dir()]))

        # ---- D. 覆盖导入中途失败：正式素材库必须完全不动
        before = sorted(p.name for p in core.list_images(core.DIR_IPM_IN))
        real_copy = shutil.copy2
        calls = {'n': 0}

        def flaky_copy(a, b, *args, **kwargs):
            calls['n'] += 1
            if calls['n'] == 2:
                raise OSError('模拟磁盘已满')
            return real_copy(a, b, *args, **kwargs)

        src3 = add_floors(tmp / 'batch_c', 3)
        core.shutil.copy2 = flaky_copy
        try:
            core.import_dataset(src3, mode='replace')
            check(False, '复制失败时应当抛出来')
        except OSError:
            check(True, '复制失败时抛出来（不静默吞掉）')
        finally:
            core.shutil.copy2 = real_copy
        check(sorted(p.name for p in core.list_images(core.DIR_IPM_IN)) == before,
              '覆盖导入中途失败：原素材库一个文件都没少',
              f'{before} -> {[p.name for p in core.list_images(core.DIR_IPM_IN)]}')
        check(core.material_board() == spec_b, '中途失败也不会改写依据')
        # 准备阶段失败同样会留下 *.staging（里面 --move 时可能是唯一原件），
        # 所以下一次 replace 会被残留闸拦住。模拟用户按提示处置掉残留再继续。
        check(len(core.material_transaction_residue()) > 0,
              '准备阶段失败也留下残留供人工处置',
              str([p.name for p in core.material_transaction_residue()]))
        for p in core.material_transaction_residue():
            shutil.rmtree(p, ignore_errors=True)

        # ---- E. 一张都没进来时不许覆盖：否则等于把素材库清空了
        empty = tmp / 'unreadable'
        empty.mkdir()
        (empty / 'bad.jpg').write_bytes(b'not an image at all')
        try:
            core.import_dataset(empty, mode='replace')
            check(False, '全部无法读取时拒绝覆盖')
        except SystemExit as exc:
            check('原素材库保持不变' in str(exc), '全部无法读取时拒绝覆盖',
                  str(exc)[:40])
        check(sorted(p.name for p in core.list_images(core.DIR_IPM_IN)) == before,
              '上一条的拒绝确实没动素材库')

        # ---- F. 老工程只有 last_import.board：刻意不采信，落到"依据未知"
        # 早期版本允许"A 规格导一批 → 改成 B → 再增量导一批"，此时
        # last_import.board = B 而库里其实是 A+B 混合。拿它当依据就是把最该
        # 报警的混合状态认成 B，所以宁可判成未知，逼一次覆盖导入重建依据。
        legacy = tmp / 'legacy'
        legacy.mkdir(parents=True)
        core.configure_paths(root=legacy)
        core.DIR_IPM_IN.mkdir(parents=True, exist_ok=True)
        core.safe_imwrite(core.DIR_IPM_IN / 'floor.jpg',
                          np.full((60, 60, 3), 128, np.uint8))
        core.update_project_config(last_import={'board': spec_a.to_dict()})
        check(core.material_board() is None,
              '老配置的 last_import.board 不当依据', str(core.material_board()))
        reason = core.material_basis_unknown_reason()
        check(reason is not None and '最近一次导入' in reason,
              '老配置落到"依据未知"并说明为何不采信',
              (reason or '').splitlines()[-1][:36])
        check(core.material_stale_reason(spec_b) is None,
              '依据未知时不再谈 stale（改由 unknown 这条路拦）')

        # ---- G. 库里已有素材但从没记过依据：增量导入要拦，标定不拦
        orphan = tmp / 'orphan'
        core.configure_paths(root=orphan)
        core.DIR_IPM_IN.mkdir(parents=True, exist_ok=True)
        core.safe_imwrite(core.DIR_IPM_IN / 'floor.jpg',
                          np.full((60, 60, 3), 128, np.uint8))
        check(core.material_basis_unknown_reason() is not None,
              '素材库非空且无依据时给出说明')
        try:
            core.import_dataset(src1, mode='add')
            check(False, '依据未知时拒绝增量导入')
        except SystemExit as exc:
            check('没记下' in str(exc), '依据未知时拒绝增量导入',
                  str(exc).splitlines()[-1][:34])
        core.clear_material_board()
        core.clear_material_board()      # 幂等：重复删不报错
        check(core.material_board() is None, 'clear 之后依据为空')
    finally:
        core.configure_paths(root=old_root)
        core.configure_board(old_board)
        shutil.rmtree(tmp, ignore_errors=True)



def scenario_state_guards() -> None:
    """状态层的四条边界，全部是"看起来已经拦住了、其实有旁路"那一类。

      1. `_incomplete/` 也是分拣产物：漏算它，"全是拍不全的棋盘照"这批素材会
         被当成空库，换规格不报 stale，增量导入直接混两套规格。
      2. stale 时 ipm_input/ 同样不可信 —— 里面装的是"旧规格判它检不出棋盘"的照片。
      3. replace 遇坏图必须整批放弃，否则"19 张好图 + 1 张坏图"照样换掉旧库。
      4. 未来版本 / 损坏的 project.json 不许被当成空配置，更不许被覆盖降级。
      5. 显式指定库内的图不算"绕过分拣结果"，仍要过闸；只有库外文件才放行。
      6. replace --move 换装到一半失败时，被搬走的原图必须仍在暂存区 —— 这是
         唯一一条会真的丢用户原始素材的路径。
    """
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix='neuq_guard_'))
    old_root, old_board = core.SCRIPT_DIR, core.BOARD
    spec_a = core.CheckerboardSpec(12, 9, 20.0)
    spec_b = core.CheckerboardSpec(9, 7, 25.0)
    floor = np.full((80, 80, 3), 128, np.uint8)
    try:
        # ---- A. 只有 _incomplete/ 里有图：仍然算"库非空"
        core.configure_paths(root=tmp / 'incomplete_only')
        core.configure_board(spec_a)
        inc = core.DIR_CALIB_IN / core.INCOMPLETE_SUBDIR
        inc.mkdir(parents=True, exist_ok=True)
        core.safe_imwrite(inc / 'half_board.jpg', floor)
        core.set_material_board(spec_a)
        check(core.material_has_images(), '_incomplete/ 里的图算素材库非空')
        core.configure_board(spec_b)
        check(core.material_stale_reason() is not None,
              '只有 _incomplete/ 有图时，换规格照样判 stale')
        src = tmp / 'batch'
        src.mkdir()
        core.safe_imwrite(src / 'floor.jpg', floor)
        try:
            core.import_dataset(src, mode='add')
            check(False, '只有 _incomplete/ 时增量导入也被拦住')
        except SystemExit as exc:
            check('增量导入已中止' in str(exc), '只有 _incomplete/ 时增量导入也被拦住',
                  str(exc).splitlines()[0])

        # ---- B. stale 时不许继续做逆透视
        core.DIR_IPM_IN.mkdir(parents=True, exist_ok=True)
        core.safe_imwrite(core.DIR_IPM_IN / 'ground.jpg', floor)
        try:
            core.discover_ipm_source()
            check(False, 'stale 时拒绝自动从 ipm_input/ 取图')
        except SystemExit as exc:
            check('逆透视标定已中止' in str(exc), 'stale 时拒绝自动从 ipm_input/ 取图',
                  str(exc).splitlines()[0])
        outside = tmp / 'outside.jpg'
        core.safe_imwrite(outside, floor)
        picked = core.discover_ipm_source(outside)
        check(picked is not None and picked.name == 'outside.jpg',
              '显式指定的外部图放行（它不依赖分拣结果）', str(picked))

        # ---- C. replace 遇坏图：整批放弃（all-or-nothing），add 仍是 best effort
        mixed = tmp / 'mixed'
        mixed.mkdir()
        core.safe_imwrite(mixed / 'ok_1.jpg', floor)
        core.safe_imwrite(mixed / 'ok_2.jpg', floor)
        (mixed / 'broken.jpg').write_bytes(b'not an image at all')
        before = sorted(p.name for p in core.list_images(core.DIR_IPM_IN))
        try:
            core.import_dataset(mixed, mode='replace')
            check(False, 'replace 里混进坏图时整批放弃')
        except SystemExit as exc:
            check('原素材库保持不变' in str(exc), 'replace 里混进坏图时整批放弃',
                  str(exc).splitlines()[0][:30])
        check(sorted(p.name for p in core.list_images(core.DIR_IPM_IN)) == before,
              '被放弃的覆盖导入没动正式素材库', str(before))
        check(core.material_board() == spec_a, '被放弃的覆盖导入没改写依据')
        core.configure_board(spec_a)          # 回到一致状态，才轮得到坏图这条路径
        got = core.import_dataset(mixed, mode='add')
        check(got['imported_ipm'] == 2 and got['skipped_unreadable'] == 1,
              'add 模式跳过坏图、其余照常入库', str(got))

        # ---- D. project.json 读不动时 fail closed，绝不覆盖
        core.configure_paths(root=tmp / 'future')
        payload = {'schema_version': core.SCHEMA_VERSION + 1,
                   'board': spec_b.to_dict(), 'future_only': 'keep me'}
        core.project_config_path().parent.mkdir(parents=True, exist_ok=True)
        core.project_config_path().write_text(json.dumps(payload), encoding='utf-8')
        try:
            core.load_project_config()
            check(False, '未来版本的 project.json 拒绝读取')
        except SystemExit as exc:
            check('结构版本' in str(exc), '未来版本的 project.json 拒绝读取',
                  str(exc).splitlines()[0][:40])
        try:
            core.configure_board(spec_a, persist=True)
            check(False, '未来版本的 project.json 不会被降级覆盖')
        except SystemExit:
            check(True, '未来版本的 project.json 不会被降级覆盖')
        raw = json.loads(core.project_config_path().read_text(encoding='utf-8'))
        check(raw == payload, '文件内容原样保留', str(raw.get('schema_version')))

        core.project_config_path().write_text('{ 这不是 JSON', encoding='utf-8')
        try:
            core.load_project_config()
            check(False, '损坏的 project.json 不当空配置用')
        except SystemExit as exc:
            check('无法读取' in str(exc), '损坏的 project.json 不当空配置用',
                  str(exc).splitlines()[0][:40])

        # ---- E. 显式指定"库内"的图仍要过闸，只有库外的外部图才放行
        # 写得出文件名并不能让它重新可信：它躺在 ipm_input/ 里本身就是旧规格
        # 分拣的结论。只放行库外文件，语义才和"不依赖分拣结果"对得上。
        core.configure_paths(root=tmp / 'explicit')
        core.configure_board(spec_a)
        core.DIR_IPM_IN.mkdir(parents=True, exist_ok=True)
        core.safe_imwrite(core.DIR_IPM_IN / 'ground.jpg', floor)
        core.set_material_board(spec_a)
        core.configure_board(spec_b)                     # 换规格 → 整库 stale
        for label, arg in (('只写文件名', Path('ground.jpg')),
                           ('写库内绝对路径', core.DIR_IPM_IN / 'ground.jpg')):
            try:
                core.discover_ipm_source(arg)
                check(False, f'stale 时{label}指定库内图也被拦住')
            except SystemExit as exc:
                check('逆透视标定已中止' in str(exc),
                      f'stale 时{label}指定库内图也被拦住', str(exc).splitlines()[0])
        ext = tmp / 'explicit_outside.jpg'
        core.safe_imwrite(ext, floor)
        got_ext = core.discover_ipm_source(ext)
        check(got_ext is not None and got_ext.name == ext.name,
              '库外的外部图才真正放行', str(got_ext))
        core.configure_board(spec_a)
        core.clear_material_board()                      # 依据未知，不是 stale
        try:
            core.discover_ipm_source(Path('ground.jpg'))
            check(False, '依据未知时库内图同样被拦住')
        except SystemExit as exc:
            check('没记下' in str(exc), '依据未知时库内图同样被拦住',
                  str(exc).splitlines()[-1][:30])

        # ---- F. replace + --move，第二个新目录换装失败：原图一张都不能丢
        # 这是 keep_staging 最该生效的场合，而老实现恰好在这里失效：第一个 stage
        # 已经 rename 成正式目录，回滚时按"撤掉新目录"把它 rmtree 掉，而那里面是
        # --move 搬进来的仅存原件——用户目录和暂存区都已经没有了。
        core.configure_paths(root=tmp / 'move_fail')
        core.configure_board(spec_a)
        core.DIR_CALIB_IN.mkdir(parents=True, exist_ok=True)
        core.DIR_IPM_IN.mkdir(parents=True, exist_ok=True)
        core.safe_imwrite(core.DIR_CALIB_IN / 'old_board.jpg', synth_board_image(spec_a))
        core.safe_imwrite(core.DIR_IPM_IN / 'old_floor.jpg', floor)
        core.set_material_board(spec_a)
        old_calib = sorted(p.name for p in core.list_images(core.DIR_CALIB_IN))
        old_ipm = sorted(p.name for p in core.list_images(core.DIR_IPM_IN))

        user_dir = tmp / 'user_photos'                   # --move 之后这里会被清空
        user_dir.mkdir()
        base = synth_board_image(spec_a)
        for i in range(3):
            core.safe_imwrite(user_dir / f'shot_{i}.jpg', synth_pose(base, 100 + i))
        core.safe_imwrite(user_dir / 'floor_new.jpg', floor)
        sent = sorted(p.name for p in core.list_images(user_dir))

        stage_calib, stage_ipm = (s for s, _t in core.material_stage_pairs())
        restore = failing_rename_into(core.DIR_IPM_IN)
        try:
            core.import_dataset(user_dir, mode='replace', move=True)
            check(False, 'ipm_input 换装失败时抛错回滚')
        except OSError:
            check(True, 'ipm_input 换装失败时抛错回滚')
        finally:
            restore()

        check(sorted(p.name for p in core.list_images(core.DIR_CALIB_IN)) == old_calib,
              '旧 calib_input 完整恢复',
              str(sorted(p.name for p in core.list_images(core.DIR_CALIB_IN))))
        check(sorted(p.name for p in core.list_images(core.DIR_IPM_IN)) == old_ipm,
              '旧 ipm_input 完整恢复',
              str(sorted(p.name for p in core.list_images(core.DIR_IPM_IN))))
        staged = sorted(p.name for p in
                        list(stage_calib.rglob('*')) + list(stage_ipm.rglob('*'))
                        if p.is_file())
        check(staged == sent, '被 --move 搬走的原图全部仍在 *.staging 里',
              f'staging={staged}')
        check(not core.list_images(user_dir),
              '源目录确实已被 move 清空（所以 staging 是仅存的那一份）')
        check(core.material_board() == spec_a, '失败的覆盖导入没改写依据')

        # 失败 → 修好问题 → 再点一次"覆盖导入"：必须立刻拒绝。
        # 少了这道闸，import_dataset 开头的 rmtree(stage) 会把上一轮保住的
        # 唯一原件删掉 —— 这一次没丢，重试时丢了。
        residue_before = tree_fingerprint(core.material_transaction_residue())
        check(len(core.material_transaction_residue()) == 2,
              '残留检测报出两个 *.staging',
              str([p.name for p in core.material_transaction_residue()]))
        core.safe_imwrite(user_dir / 'retry_shot.jpg', synth_pose(base, 200))
        try:
            core.import_dataset(user_dir, mode='replace', move=True)
            check(False, '有残留时第二次覆盖导入被拒绝')
        except SystemExit as exc:
            check('未完成的素材导入残留' in str(exc), '有残留时第二次覆盖导入被拒绝',
                  str(exc).splitlines()[0])
        check(tree_fingerprint(core.material_transaction_residue()) == residue_before,
              '被拒绝的重试没动过 *.staging 里的任何一个字节')
        check((user_dir / 'retry_shot.jpg').is_file(),
              '被拒绝的重试也没搬走新素材')

        # .old 残留同样要拦（上一轮"搬不回 staging"的极端分支会留下它）
        core.configure_paths(root=tmp / 'old_residue')
        core.configure_board(spec_a)
        core.DIR_CALIB_IN.mkdir(parents=True, exist_ok=True)
        core.safe_imwrite(core.DIR_CALIB_IN / 'live.jpg', synth_board_image(spec_a))
        stale_old = core.DIR_IPM_IN.with_name(core.DIR_IPM_IN.name + core.BACKUP_SUFFIX)
        stale_old.mkdir(parents=True, exist_ok=True)
        core.safe_imwrite(stale_old / 'rescued.jpg', floor)
        core.set_material_board(spec_a)
        src2 = tmp / 'src2'
        src2.mkdir()
        core.safe_imwrite(src2 / 'x.jpg', floor)
        try:
            core.import_dataset(src2, mode='replace')
            check(False, '.old 残留时也拒绝覆盖导入')
        except SystemExit as exc:
            check('未完成的素材导入残留' in str(exc), '.old 残留时也拒绝覆盖导入',
                  str(exc).splitlines()[0])
        check((stale_old / 'rescued.jpg').is_file(), '.old 里被救下的素材没被删')
    finally:
        core.configure_paths(root=old_root)
        core.configure_board(old_board)
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print('[A] 规格对象')
    d = core.DEFAULT_BOARD
    check((d.squares_x, d.squares_y, d.square_size_mm) == (12, 9, 20.0),
          '默认规格是 12x9 / 20 mm', f'{d.squares_x}x{d.squares_y}')
    check(d.corners == (11, 8), '换算出的内角点是 11x8', f'{d.corners}')
    check(d.partial_grids == ((9, 6), (7, 5), (6, 4), (5, 3)),
          '默认规格的局部子网格候选与改造前一致', f'{d.partial_grids}')
    small = core.CheckerboardSpec(8, 6, 25.0)
    check(all(min(g) >= 3 and max(g) >= 4 for g in small.partial_grids),
          '小棋盘的候选随之缩小且不小于 3x4', f'{small.partial_grids}')
    check(small.partial_grids != d.partial_grids, '候选不再写死，随规格变化')
    # 校验不假设方向：4x5 与 5x4 只是转了 90°，应当一视同仁地接受
    for sq, why in (((4, 5, 20.0), '4x5 方格'), ((5, 4, 20.0), '5x4 方格')):
        try:
            core.CheckerboardSpec(*sq)
            check(True, f'接受 {why}（内角点 3x4 / 4x3 等价）')
        except ValueError as exc:
            check(False, f'接受 {why}', str(exc))
    for bad, why in (((4, 4, 20.0), '内角点长边不足 4'), ((3, 8, 20.0), '内角点短边不足 3'),
                     ((12, 9, 0.0), '边长为 0'), ((12, 9, -1.0), '边长为负')):
        try:
            core.CheckerboardSpec(*bad)
            check(False, f'拒绝非法规格（{why}）')
        except ValueError:
            check(True, f'拒绝非法规格（{why}）')
    check('内角点' in d.label, 'label 里明确写出内角点，避免误解', d.label)

    print('\n[B] 命令行解析')
    got = core.resolve_board([12, 9], None, 20.0)
    check(got.corners == (11, 8), '--board-squares 12 9 → 内角点 11x8')
    got2 = core.resolve_board(None, [11, 8], 20.0)
    check(got2 == got, '--board-corners 11 8 与之等价')
    try:
        core.resolve_board([12, 9], [11, 8], None)
        check(False, '同时给两种参数会报错')
    except SystemExit as exc:
        check('只能给一个' in str(exc), '同时给两种参数会报错', str(exc)[:36])

    # 部分覆盖：只给一项时，没给的字段必须沿用**当前工程**的规格。
    # 拿仓库默认值兜底的话，"只想把边长改成 30" 会顺手把方格数冲回 12x9。
    saved = core.BOARD
    try:
        cases = (
            ((None, None, 30.0), (9, 7, 30.0), '只给边长 → 方格数沿用工程的 9x7'),
            (([10, 8], None, None), (10, 8, 25.0), '只给方格数 → 边长沿用工程的 25'),
            (([10, 8], None, 30.0), (10, 8, 30.0), '两项都给 → 都按命令行'),
            ((None, [8, 6], None), (9, 7, 25.0), '给内角点 8x6 → 换算成方格 9x7'),
            ((None, None, None), (9, 7, 25.0), '一项都不给 → 原样'),
        )
        for given, want, why in cases:
            core.configure_board(core.CheckerboardSpec(9, 7, 25.0))
            r = core.resolve_board(*given)
            check((r.squares_x, r.squares_y, r.square_size_mm) == want,
                  f'当前工程 9x7/25 时 {why}',
                  f'→ {r.squares_x}x{r.squares_y}/{r.square_size_mm:g}')
    finally:
        core.configure_board(saved)

    print('\n[C] 检测行为（合成棋盘，规格必须影响结果）')
    spec = core.CheckerboardSpec(12, 9, 20.0)
    img = synth_board_image(spec)
    gray = core.to_gray(img)
    old = core.BOARD
    try:
        core.configure_board(spec)
        found, corners = core.detect_chessboard(gray)
        check(found, '规格正确时检出棋盘',
              f'角点 {None if corners is None else len(corners)}')
        wrong = core.CheckerboardSpec(8, 6, 20.0)
        found_w, _ = core.detect_chessboard(gray, board=wrong)
        check(not found_w, '规格错误时检不出（说明规格真的在起作用）',
              f'错规格 {wrong.squares_x}x{wrong.squares_y}')
        core.configure_board(wrong)
        found_w2, _ = core.detect_chessboard(gray)
        check(not found_w2, '全局规格被改错时同样检不出')
        core.configure_board(spec)
        found3, _ = core.detect_chessboard(gray)
        check(found3, '改回正确规格后又能检出')
    finally:
        core.configure_board(old)

    print('\n[D] calib.json 往返')
    tmp = Path(tempfile.mkdtemp(prefix='neuq_board_'))
    old_root = core.SCRIPT_DIR
    try:
        core.configure_paths(root=tmp)
        core.configure_board(core.CheckerboardSpec(9, 7, 25.0))
        (core.DIR_CALIB_DATA).mkdir(parents=True, exist_ok=True)
        K = np.eye(3)
        core.save_calibration(K, np.zeros(5), (1280, 720))
        raw = json.loads(core.CALIB_JSON.read_text(encoding='utf-8'))
        check(raw.get('schema_version') == core.SCHEMA_VERSION,
              'calib.json 带 schema_version', str(raw.get('schema_version')))
        check(raw.get('tool_version') == core.TOOL_VERSION, '带 tool_version')
        b = raw.get('board') or {}
        check((b.get('squares_x'), b.get('squares_y')) == (9, 7),
              'board 段记下方格数', f'{b.get("squares_x")}x{b.get("squares_y")}')
        check((b.get('corners_x'), b.get('corners_y')) == (8, 6),
              'board 段同时记下内角点数')
        check(raw.get('chessboard_corners') == [8, 6],
              '老字段 chessboard_corners 仍然保留（兼容外部脚本）')
        meta = core.calib_board_meta()
        check((meta or {}).get('squares_x') == 9, 'calib_board_meta 能读回规格',
              str(meta and meta.get('squares_x')))

        # 老格式：只有 chessboard_corners
        legacy = {k: v for k, v in raw.items()
                  if k not in ('board', 'schema_version', 'tool_version')}
        core.CALIB_JSON.write_text(json.dumps(legacy), encoding='utf-8')
        meta2 = core.calib_board_meta()
        check((meta2 or {}).get('squares_x') == 9,
              '只有内角点的老 calib.json 也能换算回方格数',
              str(meta2 and meta2.get('squares_x')))

        spec2 = core.CheckerboardSpec.from_dict({'corners': [11, 8],
                                                 'square_size_mm': 20})
        check((spec2.squares_x, spec2.squares_y) == (12, 9),
              'CheckerboardSpec.from_dict 兼容老格式')
    finally:
        core.configure_paths(root=old_root)
        core.configure_board(core.DEFAULT_BOARD)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print('\n[E] 端到端：真的调用一次相机标定')
    scenario_calibration()

    print('\n[F] 持久化：project.json 记住项目用的棋盘')
    scenario_persistence()

    print('\n[G] 素材库的规格状态机 + 覆盖导入事务')
    scenario_material_set()

    print('\n[H] 状态层边界：_incomplete / 逆透视闸门 / 坏图 / 配置 fail closed')
    scenario_state_guards()

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
