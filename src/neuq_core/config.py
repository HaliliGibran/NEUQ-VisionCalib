"""配置与工程状态：标定板规格、project.json、素材库的分类依据。

从 neuq_vision_calib.py 原样搬来，函数体逐字未改。依赖方向是单向的：
neuq_vision_calib.py 导入本模块，本模块不导入它；本模块只向下依赖
neuq_core.fs_transaction（目录换装的机械动作），那一层不知道 project.json 的存在。
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from neuq_core.fs_transaction import BACKUP_SUFFIX, DirectorySwapTransaction

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp')

TOOL_VERSION = '0.1.0'
# 落盘 JSON（calib.json / matrices.json / ipm_state.json / 查找表 metadata.json）的
# 结构版本。改字段语义时 +1，别让后人拿着旧文件跑新程序、字段都在但含义变了。
SCHEMA_VERSION = 1

# 项目级配置。棋盘规格属于"这个项目用什么板"的前置事实，必须落盘：
# 只放在进程全局里的话，设好规格、导完素材、关掉程序，第二天重开又变回默认，
# 接着导入的新照片就会按另一套规格分类。
PROJECT_JSON_NAME = 'project.json'

# 拍不全的棋盘照的归置目录（calib_input/ 的子目录）。放在子目录里是因为
# list_images 不递归，这样它们既不会被误当成地面照，也不会进标定数据集被反复剔除。
INCOMPLETE_SUBDIR = '_incomplete'

# 覆盖导入的暂存目录后缀，与导出事务同一套路：新素材全部就位后一次性换装。
STAGING_SUFFIX = '.staging'

# 换装时旧正式目录的暂时落脚点（BACKUP_SUFFIX）定义在 neuq_core.fs_transaction：
# 事务残留检查要和换装看同一组路径，写死两份字面量迟早对不上。

# 覆盖导入进行中的标记。写在 project.json 里，含义是"目录可能正处于中间态"：
# 素材换装与 material_set / last_import 的落盘不可能是同一个原子操作，所以先
# 声明"有事务正在进行"，全部成功后再删掉它。它还在，就说明上一次导入没走完。
MATERIAL_PENDING_KEY = 'material_import_pending'



def list_images(folder: Path) -> List[Path]:
    """按文件名排序列出目录下的图片，目录不存在时返回空列表。"""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def check_schema(data: dict, what: str) -> None:
    """落盘文件的结构版本比本程序新时直接拒绝。

    只写一个 schema_version 字段而没有这道闸，它就只是"文件里有个数字"；
    它真正的用途是防止新版格式被旧程序误读（字段都在、含义已经变了，
    比报错难查得多）。老文件没有这个字段则放行，按当前格式尽力读。
    """
    version = data.get('schema_version')
    if version is None:
        return
    try:
        version = int(version)
    except (TypeError, ValueError):
        raise SystemExit(f'{what} 的 schema_version 不是整数: {version!r}') from None
    if version > SCHEMA_VERSION:
        raise SystemExit(
            f'{what} 的结构版本是 {version}，本程序只认到 {SCHEMA_VERSION}。\n'
            '这份文件是更新版本的程序生成的，请升级后再读，'
            '不要用旧版强行解析——字段可能都在，但含义已经变了。')


def write_json_atomic(path: Path, payload) -> None:
    """先写同名 .tmp、fsync 落盘、再整体替换。

    状态文件（project.json / calib.json）要能撑得起"重启之后照原样继续"，
    而 `path.write_text()` 中途被打断会留下半截 JSON：下一次启动读到的是
    JSONDecodeError，规格、素材分类依据这些事实就永久丢了。
    """
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


@dataclass(frozen=True)
class CheckerboardSpec:
    """标定板规格。

    入口刻意用"方格数"而不是"内角点数"：用户手上拿的是 12x9 个方格的棋盘，
    而 OpenCV 要的是 11x8 个内角点。"我明明打印的是 12x9，程序为什么让我填 11x8"
    是这一环最经典的填错来源，所以界面上只问方格数，内角点数由程序换算并显示。

    这个规格不只用于相机标定：素材导入时区分"棋盘照 / 地面照"用的也是它，
    所以它属于**当前标定工程的前置配置**，必须在导入之前就确定下来。
    """

    squares_x: int = 12
    squares_y: int = 9
    square_size_mm: float = 20.0

    def __post_init__(self) -> None:
        # 内角点短边至少 3、长边至少 4 才有稳定的标定意义。
        # 刻意不假设"横向一定多于纵向"：4x5 与 5x4 方格只是转了 90°，
        # 前者内角点 3x4、后者 4x3，应当一视同仁地接受或拒绝。
        cx, cy = self.squares_x - 1, self.squares_y - 1
        if min(cx, cy) < 3 or max(cx, cy) < 4:
            raise ValueError(
                f'内角点至少 3x4（即方格至少 4x5），收到 '
                f'{self.squares_x}x{self.squares_y}（内角点 {cx}x{cy}）。')
        if not self.square_size_mm > 0:
            raise ValueError(f'单格边长必须为正数，收到 {self.square_size_mm}。')

    @property
    def corners(self) -> Tuple[int, int]:
        """OpenCV 需要的内角点数 (cols, rows)。"""
        return self.squares_x - 1, self.squares_y - 1

    @property
    def label(self) -> str:
        """给人和日志看的一行描述。"""
        return (f'{self.squares_x}x{self.squares_y} 方格'
                f'（内角点 {self.corners[0]}x{self.corners[1]}），'
                f'单格 {self.square_size_mm:g} mm')

    def to_dict(self) -> dict:
        """完整写进 calib.json，好让日后能追溯"这份内参是哪块棋盘算出来的"。"""
        return {
            'type': 'checkerboard',
            'squares_x': int(self.squares_x),
            'squares_y': int(self.squares_y),
            'corners_x': int(self.corners[0]),
            'corners_y': int(self.corners[1]),
            'square_size_mm': float(self.square_size_mm),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CheckerboardSpec':
        """从 calib.json 的 board 段还原；三种历史写法都能认。

        注意边长不能用 `x or 20` 兜底：那样写下去，显式记录的 0 会被静默换成 20，
        错误就被吞掉了。这里让 0 正常进入校验并明确报错。
        """
        raw = data.get('square_size_mm')
        size = DEFAULT_BOARD.square_size_mm if raw is None else float(raw)
        if data.get('squares_x') is not None:
            return cls(int(data['squares_x']), int(data['squares_y']), size)
        corners = data.get('corners')
        if corners:
            return cls(int(corners[0]) + 1, int(corners[1]) + 1, size)
        if data.get('corners_x') is not None:
            return cls(int(data['corners_x']) + 1, int(data['corners_y']) + 1, size)
        return cls()

    @property
    def partial_grids(self) -> Tuple[Tuple[int, int], ...]:
        """子网格搜索用的内角点阵，从大到小。

        棋盘被裁切时完整板检不出，但局部往往仍是一个规则子网格——用它区分
        "拍不全的棋盘照"和"根本没有棋盘的地面照"。候选必须随规格走：
        固定写死 (9,6)(7,5)... 只对 11x8 内角点成立，换块小棋盘就全不合理了。
        判据同样不假设方向，与 __post_init__ 的校验保持一致。
        """
        cols, rows = self.corners
        out: List[Tuple[int, int]] = []
        for dx, dy in ((2, 2), (4, 3), (5, 4), (6, 5)):
            g = (cols - dx, rows - dy)
            if (min(g) >= 3 and max(g) >= 4
                    and g not in out and g != self.corners):
                out.append(g)
        return tuple(out)


DEFAULT_BOARD = CheckerboardSpec()


def spec_from_meta(meta: Optional[dict]) -> Optional[CheckerboardSpec]:
    """把 calib.json 里的 board 段还原成规格对象；读不出来返回 None。"""
    if not meta:
        return None
    try:
        return CheckerboardSpec.from_dict(meta)
    except (ValueError, TypeError, KeyError):
        return None


# 路径与当前规格的权威副本仍在 neuq_vision_calib.py（那边几十处函数按裸全局读它们）。
# 这里只持一份镜像，由 configure_paths() / configure_board() 这两个唯一写入点同步过来。
# 拆包完成后会统一收敛成显式 context，这一刀先保证行为零变化。
SCRIPT_DIR: Path = Path(__file__).resolve().parent.parent
DIR_CALIB_IN: Path = SCRIPT_DIR / 'calib_input'
DIR_IPM_IN: Path = SCRIPT_DIR / 'ipm_input'
BOARD: CheckerboardSpec = DEFAULT_BOARD


def bind_paths(*, root: Path, calib_in: Path, ipm_in: Path) -> None:
    """由 facade 的 configure_paths() 调用，同步路径镜像。"""
    global SCRIPT_DIR, DIR_CALIB_IN, DIR_IPM_IN
    SCRIPT_DIR, DIR_CALIB_IN, DIR_IPM_IN = root, calib_in, ipm_in


def bind_board(spec: CheckerboardSpec) -> None:
    """由 facade 的 configure_board() 调用，同步当前规格镜像。"""
    global BOARD
    BOARD = spec


def project_config_path() -> Path:
    """项目配置文件路径。"""
    return SCRIPT_DIR / PROJECT_JSON_NAME


def load_project_config() -> dict:
    """读取 project.json；不存在时返回空 dict，读不动则直接中止。

    这里刻意**不吞异常**。project.json 已经不是可选缓存，它承载工程状态
    （棋盘规格、素材库的分类依据）。一旦把"读失败"翻译成"空配置"，
    update_project_config() 紧接着就会拿这份空配置去覆盖写：

      新版程序写出 schema_version=2 → 旧版读到、报错、吞掉、返回 {}
      → 用户点一次「应用棋盘规格」→ project.json 被降级成 schema=1，
        material_set 等新字段全部丢失。

    甚至不用点：启动时 restore_board() 从 calib.json 迁移规格也会写一次。
    那正好与加 schema 闸门的初衷相反，所以未来版本与损坏文件都 fail closed，
    由用户自己决定是修、是改名还是删除 —— 程序绝不自动覆盖它。
    """
    p = project_config_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f'{p} 无法读取（{exc}）。\n'
                         '它记录着本工程的棋盘规格与素材库的分类依据，'
                         '为避免被覆盖，程序不会绕过它继续运行：'
                         '请修好这个文件，或把它改名/删除后重新设置规格。') from None
    if not isinstance(data, dict):
        raise SystemExit(f'{p} 的内容不是一个 JSON 对象，无法当项目配置读。\n'
                         '请修好它，或把它改名/删除后重新设置规格。')
    check_schema(data, str(p))          # 未来版本的配置：拒绝读，更不能写回去
    return data


def update_project_config(**sections) -> dict:
    """合并若干段落到 project.json 并落盘，返回合并后的完整内容。

    某个段落传 None 表示删除它 —— material_set 在素材库清空后就该消失，
    留着一条过期的"分类依据"比没有更糟。
    """
    data = load_project_config()
    for key, value in sections.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    data['schema_version'] = SCHEMA_VERSION
    data['tool_version'] = TOOL_VERSION
    p = project_config_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(p, data)
    except OSError as exc:
        print(f'警告: 无法写入 {p}（{exc}），本次设置重启后不会保留。')
    return data


def write_project_config_strict(data: dict) -> None:
    """把一份**完整的** project.json 内容写下去，写不成功就抛出来。

    与 update_project_config 的区别只有一条：不吞 OSError。普通设置（棋盘规格之类）
    写失败只是"这次设置重启后不保留"，打个警告继续是合理的；但事务提交不行——
    "磁盘上已经是新素材库、配置里还是旧的" 是没人能事后分辨的脏状态，必须让调用方
    知道写失败了、好去回滚目录。

    刻意不给 update_project_config 加 strict= 参数：那个开关会被几十个普通调用点
    误传，而它们本来就该宽容。两条路分开，误用不了。
    """
    payload = dict(data)
    payload['schema_version'] = SCHEMA_VERSION
    payload['tool_version'] = TOOL_VERSION
    p = project_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(p, payload)



def resolve_board(squares: Optional[Sequence[int]] = None,
                  corners: Optional[Sequence[int]] = None,
                  square_size_mm: Optional[float] = None) -> CheckerboardSpec:
    """把命令行上的棋盘参数解析成规格对象。

    --board-squares 是推荐的入口（和用户数棋盘的方式一致）；
    --board-corners 只作为兼容通道保留。两者同时给直接报错，不猜。

    没给出的字段沿用**当前工程**的规格，而不是仓库默认值：项目已存 9x7/25 时
    `--square-size-mm 30` 的含义是"只改边长"→ 9x7/30，而不是把方格数悄悄冲回
    12x9。否则命令行上想调一项，就得先把其它几项全部抄一遍。
    """
    if squares is not None and corners is not None:
        raise SystemExit('--board-squares 与 --board-corners 只能给一个（两者语义相同，'
                         '同时给会互相矛盾）。')
    size = (square_size_mm if square_size_mm is not None
            else BOARD.square_size_mm)
    if corners is not None:
        return CheckerboardSpec(int(corners[0]) + 1, int(corners[1]) + 1, size)
    if squares is not None:
        return CheckerboardSpec(int(squares[0]), int(squares[1]), size)
    return CheckerboardSpec(BOARD.squares_x, BOARD.squares_y, size)


def material_has_images() -> bool:
    """素材库里是否已经有内容（三个分拣结果目录，不含暂存与备份）。

    `_incomplete/` 必须算进来：它同样是按规格分拣出来的产物。漏掉它就存在这条
    洗白路径 —— 一批棋盘照全都只匹配到局部子网格，于是 calib_input/ 与 ipm_input/
    都是空的，库被判成"空"，换规格不报 stale，增量导入直接把两套规格混进一个库。
    """
    return any(list_images(d) for d in
               (DIR_CALIB_IN, DIR_CALIB_IN / INCOMPLETE_SUBDIR, DIR_IPM_IN))


def in_material_library(path: Path) -> bool:
    """这个文件是不是素材库里的产物（而不是用户自己挑的外部图）。

    用 is_relative_to 而非字符串前缀：后者会把 ipm_input2/ 之类的同名兄弟目录
    误判成库内。
    """
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for d in (DIR_CALIB_IN, DIR_IPM_IN):
        try:
            if resolved.is_relative_to(d.resolve()):
                return True
        except OSError:
            continue
    return False


def material_board() -> Optional[CheckerboardSpec]:
    """当前这套素材是按哪块棋盘分拣出来的；未知时 None。

    刻意不记在"最近一次导入"上：那样一次增量导入就能把它改写掉，于是
    「旧规格导的一批 + 新规格导的一批」会被认成整套都是新规格 —— 混合状态
    被洗白，而它恰恰是最该报警的情况。

    只有 material_set 算依据。老工程的 last_import.board 刻意**不**采信：
    那时增量导入还没被拦，"A 规格导一批 → 改成 B → 再导一批"会留下
    last_import.board = B，而库里其实是 A+B 混合。拿它当依据等于把混合状态
    认成 B，正是最该报警的情况被洗白。宁可判成"依据未知"，逼一次覆盖导入
    重新确立 material_set —— 老工程麻烦一次，换来干净的状态模型。
    """
    cfg = load_project_config()
    return spec_from_meta((cfg.get('material_set') or {}).get('board'))


def set_material_board(spec: CheckerboardSpec) -> None:
    """记下"现在这套素材是按 spec 分拣的"。"""
    update_project_config(material_set={'board': spec.to_dict()})


def clear_material_board() -> None:
    """素材库清空了：分类依据随之作废，下一次导入重新确立。"""
    update_project_config(material_set=None)


def material_import_pending() -> Optional[dict]:
    """有没有一次没走完的覆盖导入；没有则 None。

    返回的记录形如
      {'operation': 'replace', 'board': {...}, 'last_import': {...}}
    也就是"事务打算提交成什么样"。刻意不改写同一份配置里的 material_set：
    最终提交之前，project.json 始终仍然描述**旧的**正式库，这条记录只是额外
    声明"目录可能正处于中间态"。这样读配置的其它代码不需要认识 pending 也不会
    被误导，而下一次覆盖导入据它 fail closed。
    """
    value = load_project_config().get(MATERIAL_PENDING_KEY)
    return value if isinstance(value, dict) else None



def material_stale_reason(spec: Optional[CheckerboardSpec] = None) -> Optional[str]:
    """素材是否是按「另一块棋盘」分类的；是则返回说明，否则 None。

    规格不只用于标定，导入时"这张算棋盘照还是地面照"用的也是它。改了规格却不
    重新分类，calib_input/ 与 ipm_input/ 里躺着的还是旧标准的产物，
    后面选逆透视原图时会莫名其妙看到混进来的棋盘照。
    """
    now = BOARD if spec is None else spec
    used = material_board()
    if used is None or used == now:
        return None
    if not material_has_images():
        return None                       # 素材库是空的，无所谓
    return (f'当前素材是按「{used.label}」分类导入的，与现在的「{now.label}」不一致。\n'
            '规格参与"棋盘照 / 地面照"的判定，请重新分类素材：'
            '清空素材库后按新规格重新导入，或用「覆盖整个素材库」导入。')


def material_basis_unknown_reason() -> Optional[str]:
    """素材库里已经有东西，但没记下它是按哪块棋盘分拣的。

    这种场合不能再走增量导入：新导一批会把两套规格的产物混在一起，而系统对此
    一无所知。相机标定倒不必拦 —— 它逐张重新检棋盘，检不出自然会被剔除并报错。

    老工程（只有 last_import.board、没有 material_set）也落到这里：那个字段
    证明不了整库的分类依据，见 material_board 的说明。
    """
    if material_board() is not None or not material_has_images():
        return None
    legacy = spec_from_meta((load_project_config().get('last_import') or {}).get('board'))
    hint = ''
    if legacy is not None:
        hint = (f'\n（工程配置里只记着"最近一次导入用的是 {legacy.label}"，'
                '它证明不了整库都按这块棋盘分拣——早期版本允许换规格后增量导入，'
                '所以这个值不作为依据。）')
    return (f'calib_input/（含 _incomplete/）与 ipm_input/ 里已有素材，'
            f'但没记下它们是按哪块棋盘（当前是 {BOARD.label}）分拣的。\n'
            '为避免混进两套规格的产物，请改用「覆盖整个素材库」导入，或先清空素材库。'
            + hint)


def require_material_basis(what: str, block_unknown: bool = True) -> None:
    """当前规格下这套素材还能不能用；不能用就中止 what 这件事。"""
    reason = material_stale_reason()
    if reason is None and block_unknown:
        reason = material_basis_unknown_reason()
    if reason is not None:
        raise SystemExit(f'{what}已中止：\n{reason}')


def material_stage_pairs() -> List[Tuple[Path, Path]]:
    """覆盖导入用的 (暂存目录, 正式目录) 对照。

    暂存目录与正式目录同级、同名加后缀，所以换装是同目录内的 rename，
    不会跨盘。_incomplete/ 挂在 calib_input 下面，跟着父目录一起换。
    """
    return [(d.with_name(d.name + STAGING_SUFFIX), d)
            for d in (DIR_CALIB_IN, DIR_IPM_IN)]


def material_transaction_residue() -> List[Path]:
    """上一次素材导入事务留下的残留目录（*.staging / *.old）。

    这些目录只会在换装失败时留下，而 --move 导入的暂存区里可能是用户照片的
    **唯一副本**（原位置已经没有了）。所以新的覆盖导入必须先让路：
    不猜"这是 copy 模式留下的、删了也没事"，一律要求人工确认。

    只列**文件系统上的**残留；project.json 里的 pending 标记同样算残留，但它
    不是一个路径，由 require_no_material_residue() 和它一起判。
    """
    found: List[Path] = []
    for stage, target in material_stage_pairs():
        for p in (stage, target.with_name(target.name + BACKUP_SUFFIX)):
            if p.exists():
                found.append(p)
    return found


def require_no_material_residue() -> None:
    """有事务残留就拒绝启动新的覆盖导入。

    残留有两种：磁盘上的 *.staging / *.old，和 project.json 里没被清掉的 pending
    标记。后者是进程被杀 / 断电 / 回滚本身失败时唯一留下的线索，所以它也要拦——
    否则一次崩溃之后的重试会直接在一个中间态上再做一次破坏性换装。
    """
    residue = material_transaction_residue()
    pending = material_import_pending()
    if not residue and pending is None:
        return
    items = [f'  - {p}' for p in residue]
    extra = ''
    if pending is not None:
        items.append(f'  - {project_config_path()} 里的 {MATERIAL_PENDING_KEY} 标记'
                     '（上次覆盖导入没有走完）')
        extra = ('上次导入未完成，素材库可能处于中间态：目录可能已经换成新的，'
                 '也可能还是旧的。\n'
                 '程序不自动猜恢复方向——两种猜法都可能把用户仅存的原件删掉。\n')
    lines = '\n'.join(items)
    raise SystemExit(
        '检测到上一次未完成的素材导入残留：\n'
        f'{lines}\n'
        + extra +
        '这些目录可能包含通过 --move 搬入、已经不在原位置的唯一原件。\n'
        '为避免数据丢失，本次不会自动删除或覆盖它们。\n'
        '请先检查并恢复/备份其中内容，手工删除残留目录后再重试。')


class MaterialImportTransaction:
    """覆盖导入的事务闭环：pending → 换装（保留 .old）→ 最终配置 → 删 .old。

    要堵的洞：老实现是"换装成功立刻删 .old，然后才写 material_set / last_import"。
    后两步写 project.json 失败就变成「磁盘=新素材库、旧素材库已删、配置仍描述旧的」，
    而且再也回不去了。

    状态机：
      install()  严格写 pending（失败：正式目录一个字节都没动）
                 → 目录换装但保留 .old（失败：用 .old 回滚，--move 的新素材回到
                   *.staging，再把 project.json 恢复成事务开始前的样子）
      commit()   严格写最终 project.json：material_set=新规格 + last_import=本次记录，
                 pending 随之消失（失败：.old 还在，目录一并回滚，配置留在旧
                 material_set）
                 → 成功才删 .old

    两次写都基于构造时读到的同一份 base 算出**完整** dict，所以"最终 JSON 里
    material_set 与 last_import 同时出现"是天然成立的，不存在两次独立写之间的窗口。
    """

    def __init__(self, pairs: Sequence[Tuple[Path, Path]],
                 spec: CheckerboardSpec, last_import: dict) -> None:
        self.base = {k: v for k, v in load_project_config().items()
                     if k != MATERIAL_PENDING_KEY}
        self.spec = spec
        self.last_import = last_import
        self.tx = DirectorySwapTransaction(pairs, what='calib_input 与 ipm_input',
                                           keep_staging=True)

    def install(self) -> None:
        """声明事务开始，然后换装目录；.old 保留着，随时还能回滚。"""
        write_project_config_strict({
            **self.base,
            MATERIAL_PENDING_KEY: {'operation': 'replace',
                                   'board': self.spec.to_dict(),
                                   'last_import': self.last_import},
        })
        try:
            self.tx.install()
        except BaseException:
            self.tx.rollback()
            self._revert_config()
            raise

    def commit(self) -> None:
        """一次写完新的分类依据与导入记录，成功后才删 .old。"""
        try:
            write_project_config_strict({
                **self.base,
                'material_set': {'board': self.spec.to_dict()},
                'last_import': self.last_import,
            })
        except BaseException:
            self.tx.rollback()
            self._revert_config()
            raise
        self.tx.finalize()

    def _revert_config(self) -> None:
        """尽力把 project.json 恢复成事务开始前的样子（pending 随之消失）。

        这一步也失败就让 pending 留着：它是 fail closed 的凭据，下一次覆盖导入
        会据此拒绝，而不是在一个说不清的状态上继续做破坏性换装。
        """
        try:
            write_project_config_strict(self.base)
        except OSError as exc:
            print(f'警告: 无法清除 {project_config_path()} 里的 {MATERIAL_PENDING_KEY} '
                  f'标记（{exc}）。目录已回滚，但下一次覆盖导入会因这条标记被拒绝，'
                  '请确认素材库无误后手工删掉它。')

