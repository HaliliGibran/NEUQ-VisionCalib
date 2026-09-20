"""一组目录的换装事务：rename / backup / rollback / finalize。

从 neuq_vision_calib.py 的 `_commit_dirs` 下沉而来，并拆成两阶段：install 之后
**不立刻删 .old**，由调用方确认"该换装依赖的元数据也已落盘"再 finalize。这样
"目录已换新、配置还是旧的"这段窗口里仍然可以整体回滚。

两个消费者：
  素材覆盖导入   calib_input / ipm_input 一起换，且 keep_staging=True
                （暂存区里可能是 --move 搬进来的唯一原件，失败时绝不能删）
  矩阵与表导出   matrix / lookup_table 一起换，一次性 commit_dirs 即可

依赖方向只出不进：只用标准库，不导入 neuq_core.config，更不导入 facade，
所以可以独立 import、独立测试。BACKUP_SUFFIX 定义在这里而不是 config：
它是"换装怎么做"的一部分，config 的残留检查从这里拿同一个常量。
"""

import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

# 换装时旧正式目录的暂时落脚点。必须是常量：事务残留检查要和换装看同一组路径，
# 写死两份字面量迟早对不上。
BACKUP_SUFFIX = '.old'


class DirectorySwapTransaction:
    """把一批暂存目录整体换成正式目录，可回滚。

    install() 先统一"挪走旧的"，再统一"放上新的"。这样一旦中途出错，两个目录
    都还躺在 .old 里，可以一起复原——不会留下"一个已是新版、另一个还是旧版"的
    中间态。两轮 rename 的顺序是外部契约（回归测试按第 N 次 rename 注入故障），
    不要调整。

    keep_staging=True 时回滚会保留暂存目录：素材导入的暂存目录里可能是用 --move
    从用户目录搬过来的原件，删了就等于把源文件也删了。

    "保留"必须覆盖到**已经就位**的那几个目录。它们的 stage 已经 rename 成正式
    目录，stage 本身不复存在；此时若按"撤掉新目录"去 rmtree，删掉的正是那批
    原件——用户目录里已经没有了，暂存区也没有了。所以已就位的要先搬回 stage，
    绝不能删。
    """

    def __init__(self, pairs: Sequence[Tuple[Path, Path]],
                 what: str = 'matrix 与 lookup_table',
                 keep_staging: bool = False) -> None:
        self.pairs: List[Tuple[Path, Path]] = list(pairs)
        self.what = what
        self.keep_staging = keep_staging
        self.moved: List[Tuple[Path, Path, Path]] = []   # (target, backup, stage)
        self.installed: List[Tuple[Path, Path]] = []     # (target, stage)，rename 真的成功过

    def install(self) -> None:
        """全部 target -> .old，再全部 stage -> target；**不删 .old**。

        失败时不自动回滚：调用方可能还要在 rollback 之外做别的收尾（比如把
        project.json 恢复成事务开始前的样子），由它决定调用顺序。
        """
        for stage, target in self.pairs:
            backup = target.with_name(target.name + BACKUP_SUFFIX)
            if self.keep_staging and backup.exists():
                # 素材事务：.old 已经存在说明上一轮换装留下了残留，里面可能是
                # 用户仅存的旧素材。宁可什么都不做，也不能顺手删掉它腾地方。
                raise SystemExit(
                    f'检测到上一次未完成的素材换装残留: {backup}\n'
                    '它可能是上一轮特意保住的旧素材库。为避免覆盖，本次不做任何改动。\n'
                    '请先确认其中内容并手工处置，再重试。')
            shutil.rmtree(backup, ignore_errors=True)
            if target.exists():
                target.rename(backup)
            self.moved.append((target, backup, stage))
        for target, _backup, stage in self.moved:
            stage.rename(target)
            self.installed.append((target, stage))

    def finalize(self) -> None:
        """事务真的成功了：删掉 .old。这一步之后不再可能回滚。"""
        for _target, backup, _stage in self.moved:
            shutil.rmtree(backup, ignore_errors=True)

    def rollback(self) -> None:
        """回到 install 之前的状态；install 成功之后同样可以调用（.old 还在）。"""
        stranded: List[Path] = []
        # 撤掉已经就位的新目录。keep_staging 时搬回暂存区而不是删除
        for target, stage in reversed(self.installed):
            if not self.keep_staging:
                shutil.rmtree(target, ignore_errors=True)
                continue
            try:
                shutil.rmtree(stage, ignore_errors=True)
                target.rename(stage)
            except OSError as exc:
                # 搬不回去也绝不删：宁可留一个占位的正式目录要人工处置，
                # 也不能让 --move 进来的原件在这里消失
                stranded.append(target)
                print(f'  警告: {target} 里是本轮的新内容（--move 导入时可能是仅存的原件），'
                      f'无法搬回 {stage}（{exc}），已原样保留，请手工处置后重试。')
        # 再把 .old 放回正式位置
        for target, backup, _stage in self.moved:
            if target in stranded:
                print(f'  {target} 仍被新内容占用，改动前的内容保留在 {backup}。')
                continue
            shutil.rmtree(target, ignore_errors=True)
            if backup.exists():
                backup.rename(target)
        for stage, _target in self.pairs:
            if not stage.exists():
                continue                  # 已就位又搬不回来的，上面已单独提示过
            if self.keep_staging:
                print(f'  未能就位的新内容仍留在 {stage}，确认后可自行删除。')
            else:
                # 暂存区里是这一轮没能提交的新内容：留着只会让人误以为已经生成了
                shutil.rmtree(stage, ignore_errors=True)
        print(f'换装失败，已回滚到改动前的状态（{self.what} 一并复原）。')
        # 回滚过的目录不该再被 finalize 删掉：.old 已经搬回正式位置了
        self.moved = []
        self.installed = []


def commit_dirs(pairs: Sequence[Tuple[Path, Path]],
                what: str = 'matrix 与 lookup_table',
                keep_staging: bool = False) -> None:
    """一次性换装：install 失败则 rollback 并抛出，成功则立刻 finalize。

    给"换装本身就是整个事务"的调用方用（矩阵与查找表导出）。需要在换装与元数据
    落盘之间留一个可回滚窗口的，直接用 DirectorySwapTransaction。
    """
    tx = DirectorySwapTransaction(pairs, what=what, keep_staging=keep_staging)
    try:
        tx.install()
    except BaseException:
        tx.rollback()
        raise
    tx.finalize()


__all__ = ['BACKUP_SUFFIX', 'DirectorySwapTransaction', 'commit_dirs']
