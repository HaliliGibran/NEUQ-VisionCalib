# NEUQ 相机标定程序（NEUQ-VisionCalib）

智能车视觉标定一体化工具：相机内参标定 → 畸变矫正 → 逆透视（BirdView）标定 →
导出 **6 个线性矩阵 + 畸变系数 + 两套去畸变/复合查找表（LUT）** → 批量验证，
全部流程可在本地 Web 控制台中完成。

> 说明：去畸变是非线性变换，无法写成矩阵。交付给 C 端的是「6 个矩阵 + 畸变系数
> （`dist_coeffs`）」，原图到 BirdView 的复合变换则以查找表形式给出。

核心几何逻辑只实现一份（`src/neuq_vision_calib.py`），命令行与网页调用同一批函数，
因此网页上看到的结果与命令行跑出来的完全一致。

---

## 目录结构

```
.
├── app.py                  桌面入口（双击/打包用），等价于启动 Web 控制台
├── build_exe.py            用 PyInstaller 打包成 dist/NEUQ-VisionCalib/
├── start_webui.bat         Windows 一键启动脚本（自动挑解释器、自动补依赖）
├── src/
│   ├── neuq_vision_calib.py   核心算法 + 命令行入口（唯一权威实现）
│   └── webui/
│       ├── server.py          本地 HTTP 控制台（无第三方 Web 框架依赖）
│       └── static/            前端页面（index.html / app.js / style.css）
├── tools/
│   ├── scan_dataset.py        标定前摸清素材：分辨率分布 + 棋盘检出率
│   └── verify_outputs.py      校验导出的矩阵与查找表是否自洽
├── assets/
│   └── checkerboard/          12x9 方格、20mm 棋盘靶标（PDF/PNG/DOCX，打印用）
├── data/                      只放两类东西（均不入库）
│   ├── import/                导入前的原始素材（网页上传的落点）
│   └── backups/               「备份并清空」产生的归档
├── calib_input/               相机标定照片，_incomplete/ 存检不出完整棋盘的
├── calib_preview/             标定图去畸变验收图
├── calib_data/                calib.json —— 相机内参与畸变系数
├── ipm_input/                 逆透视标定原图
├── ipm_output/                去畸变图 + BirdView 结果
├── matrix/                    六矩阵、逆透视状态
├── lookup_table/              undistort/ 与 undistort_ipm/ 两套正反向查找表
├── test_input/                批量测试输入
├── test_output/               批量测试输出
└── dist/                      打包产物（不入库）
```

**约定**：工作目录平铺在工程根，跑起来才会产生、不写就不建（写入时各函数自己
`mkdir(parents=True)`，不需要提前铺空文件夹）。`data/` 是唯一的例外，它只装
"不属于流水线产物"的两样东西：导入前的原始素材和备份压缩包。

素材导入有两种方式：在网页里用「选择文件夹…」**直接上传**（推荐，文件夹改名、
放在哪个盘都无所谓），或在输入框里填路径让服务端按 `工程根 → data/import/` 查找。

### 代码分层

`neuq_core/` 是可复用底层，`neuq_vision_calib.py` 是应用编排层（CLI 入口 + 兼容
facade）。外部调用方式始终是 `import neuq_vision_calib as core`，底层模块拆分
不影响它——facade 把底层符号原名转发出来，`core.compute_homography`、
`core.safe_imread` 这些名字指向的就是底层模块里的同一个函数对象，不是副本。

```
neuq_core/
├── io.py            图像读写（绕开 OpenCV 的非 ASCII 路径坑）
├── config.py        标定板规格、project.json、素材库状态机
├── geometry.py      单应、多边形裁剪、有效视野尺度的纯几何运算
├── calibration.py   棋盘检出与 calibrateCamera 的算法核
└── lut.py           查找表的数学核（BirdView 像素 → 原图采样坐标）

依赖方向（单向，无环）：
  io / config / geometry -> 无项目内依赖
  calibration            -> config
  lut                    -> geometry
  neuq_vision_calib.py   -> 以上全部
```

留在编排层的是**目录读写、运行时参数、用户交互、CLI、落盘、事务**这几类——它们
本来就属于顶层，不是"还没拆完"。判据是"有没有粘着会被运行时改写的全局"：
`configure_paths()` / `configure_board()` / `apply_options()` 会重新赋值 17 个路径
全局和 13 个运行时开关（`BOARD`、`UNDIST_ALPHA`、`MAX_REPROJ_ERR`、`TABLE_FORMAT`
等），读这些全局的函数一旦搬进底层模块，就得为它们再做一份镜像——那会把"一份
权威状态"变成"到处都是镜像"，所以刻意停在这里。

过渡期有一个已知妥协：`config.py` 持有 `SCRIPT_DIR` / `DIR_CALIB_IN` /
`DIR_IPM_IN` / `BOARD` 的**镜像**，由 `configure_paths()` 与 `configure_board()`
这两个唯一写入点同步过去。因此正式 API 只认这两个函数——直接 `core.BOARD = spec`
赋值不会同步到底层模块，那是过渡形态的固有弱点，不是可依赖的用法。

---

## 环境要求

- Python 3.10 及以上
- `opencv-python`、`numpy`

```bash
python -m venv .venv
".venv/Scripts/python.exe" -m pip install -r requirements.txt   # Windows
source .venv/bin/activate && pip install -r requirements.txt     # macOS/Linux
```

---

## 运行

### 方式一：Web 控制台（推荐）

Windows 直接双击 `start_webui.bat`，或：

```bash
python app.py                    # 桌面入口
python src/webui/server.py       # 等价，默认 http://127.0.0.1:8770
python src/webui/server.py --port 9000 --no-browser
```

浏览器会自动打开。功能：素材导入、相机标定、逆透视四点拖拽、实时 BirdView 预览、
矩阵与查找表导出、批量测试、备份并清空。

### 方式二：命令行

```bash
python src/neuq_vision_calib.py --list                  # 查看各目录现状
python src/neuq_vision_calib.py --import-dir <混合目录>  # 按能否检出棋盘拆分素材入库
python src/neuq_vision_calib.py --stage calib           # 只跑相机标定
python src/neuq_vision_calib.py                         # 跑全流程（含交互标定逆透视）
python src/neuq_vision_calib.py --stage tables --quad "<四点>"   # 无 GUI 跑完整链路
```

---

## 标定流程

1. 打印 `assets/checkerboard/` 里的棋盘靶标（12x9 方格，边长 20mm）
2. 从多个角度拍摄 15 张以上棋盘照片，放入 `calib_input/`
3. 拍摄一张地面（车道）照片放入 `ipm_input/`
4. 打开控制台，依次执行「相机标定 → 逆透视标定 → 导出」
5. 产物落在 `matrix/`（矩阵）与 `lookup_table/`（查找表）

---

## 标定板规格

导入素材的第一步就是判断"这张是棋盘照还是地面照"，而这一步用的正是棋盘规格。
所以规格属于**当前标定工程的前置配置**，必须在导入之前设好。

界面（以及命令行）只问**方格数**，内角点数由程序换算——"我打印的是 12×9，
程序为什么让我填 11×8"是这一环最经典的填错来源。默认是仓库自带的 12×9 / 20 mm。

```bash
# 命令行（推荐用方格数）
python src/neuq_vision_calib.py --board-squares 12 9 --square-size-mm 20
python src/neuq_vision_calib.py --board-corners 11 8      # 兼容用法，与上面互斥

# 扫描素材用的也是同一套参数与同一份检测实现
python tools/scan_dataset.py <图片目录> --board-squares 12 9
```

在线拍摄的预览窗口顶部会一直显示当前规格——换了棋盘却忘了改参数时，
画面会持续 `detected=False`，用户很容易去怀疑相机或代码，其实只是规格没切过来。

规格会完整写进 `calib.json` 的 `board` 段，事后能追溯"这份内参是哪块棋盘算出来的"；
旧版只写了 `chessboard_corners` 的文件仍可读，会自动换算回方格数。

**换了棋盘必须重标**：复用已有 `calib.json` 前会先核对规格，不一致时直接拒绝并说明
（否则会出现"界面写着 12×9、运行标定成功"，实际复用 9×7 旧参数，
provenance 被写错且事后说不清）。确实要换板就用 `--force-calib`，
或在网页上勾选"强制重新标定"。

规格存在工程根的 **`project.json`** 里，属于项目级配置：设好一次，关掉程序再打开
仍然是它。启动时的恢复顺序是 `project.json` → `calib.json` → 仓库自带的 12×9/20mm；
从 `calib.json` 恢复出来时会顺手把规格**写进** `project.json`（只在它没有 `board` 段时做，
不覆盖用户显式设过的值），否则"备份并清空"删掉 `calib.json` 之后规格就无声退回默认值了。

命令行上只给其中一项时，**没给的字段沿用当前工程的规格**，不是仓库默认值：

```bash
# 工程里存的是 9×7 / 25 mm
python src/neuq_vision_calib.py --square-size-mm 30   # → 9×7 / 30 mm（只改边长）
python src/neuq_vision_calib.py --board-squares 10 8  # → 10×8 / 25 mm（只改方格数）
```

### 素材库按哪块棋盘分类

规格还参与"这张是棋盘照还是地面照"的判定，所以 `calib_input/`（含 `_incomplete/`）
+ `ipm_input/` 这**一整套**素材必然是在某一块棋盘规格下分拣出来的。这个事实单独记在
`project.json` 的 `material_set.board` 里，与"最近一次导入"（`last_import`）分开——
记在后者上的话，一次增量导入就能把它改写掉，混合了两套规格的素材库会被认成
整套都是新规格，脏状态被洗白。

| 操作 | 规则 |
|---|---|
| 素材库为空 | 任何规格都可以，导入后 `material_set.board` = 当前规格 |
| 增量添加 `add` | 要求当前规格 == `material_set.board`，否则**直接拒绝** |
| 覆盖导入 `replace` | 允许换规格，全部成功后才改写 `material_set.board` |
| 备份并清空 / 仅清空 | `material_set` 一并作废，下一次导入重新确立 |

"素材库为空"的判定必须把 `calib_input/_incomplete/` 也算进去：一批棋盘照全都只匹配到
局部子网格时，`calib_input/` 与 `ipm_input/` 恰好都是空的，而库里明明躺着按旧规格分拣
出来的东西。

规格与素材库不一致（stale）时，除了界面上两张卡片常驻显示醒目告警，**相机标定与逆透视
取图都会被中止**：`calib_input/` 里那批图是按旧标准挑进来的，用新规格逐张重检会悄悄
剔掉一大半，剩下两三张也标得出参数，只是精度和 provenance 都对不上；`ipm_input/` 同样
不可信——"这张检不出棋盘，算地面照"本身就是旧规格判出来的结论。要换板子，就得按新规格
重新导入。`--ipm-source` 的放行范围只到**素材库之外**：指定 `ipm_input/` 里的某一张
（包括只写文件名、它会被解析到 `ipm_input/` 下）仍要过素材状态检查——文件躺在那个目录里
本身就是过去分拣的结论，写得出文件名并不能让它重新可信；真正不依赖分拣结果的外部文件
才放行。同理，素材依据未知（没有 `material_set`）时也拦：那时同样证明不了它是地面照。
相机标定不受这一条约束，它逐张重检，检不出会被剔除并报错。

覆盖导入走**暂存目录事务**，与矩阵/查找表导出同一套做法：新素材先完整地准备进
`calib_input.staging/`、`ipm_input.staging/`，全部分拣复制成功后才整体换装。中途磁盘满、
权限出错、坏图时正式的素材库一个文件都不会少（`--move` 时暂存目录会被保留下来，
因为里面是从用户目录搬走的原件）。

保留下来的暂存目录还要**防住下一次覆盖导入**：新的 `replace` 开始前会检查
`*.staging` 与 `*.old` 残留，只要存在就直接拒绝并列出路径，不自动删除、不猜"这是
copy 模式留下的删了也没事"。少了这道闸，"失败 → 修好问题 → 再点一次覆盖导入"
会在开头把上一轮保住的唯一原件 rmtree 掉——等于这一次没丢、重试时丢了。

两种模式对坏图的态度刻意不同：

- `add` = best effort，坏图跳过、其余照常入库（它只往库里追加，不删任何既有素材）
- `replace` = all-or-nothing，**有一张读不出就整批放弃**。放过去的话，"19 张好图 +
  1 张坏图"会把旧素材库整套换成那 19 张，与上面那句承诺直接矛盾

`project.json` 读不动时（JSON 损坏、或 `schema_version` 比本程序新）程序**直接中止**，
不会把它当空配置绕过去：那样下一次"应用规格"就会拿空配置覆盖写，把新版文件降级、
把 `material_set` 抹掉，正好与加 `schema_version` 闸门的目的相反。

「备份并清空」的 `manifest.json` 里带一份 `project_config` 快照，记下这批素材当时是按
什么规格分类的——活的 `project.json` 会被下一轮导入改写，只有备份里那份才是永久记录。

---

## 代码检查

```bash
# 装一次（独立 venv，不污染装 opencv 的那个解释器）
python -m venv .venv-lint && .venv-lint/Scripts/python -m pip install -r requirements-dev.txt

.venv-lint/Scripts/ruff check .        # 静态检查
python tests/run_all.py                # 回归测试（需要 cv2）
```

两条链是分开的：ruff 不需要导入 OpenCV，所以可以装在任何解释器里；
测试仍然跑在装了 cv2/numpy 的那个上。

`pyproject.toml` 里的规则集是挑过的，只留"真的可能出错"的规则
（F/E/W/I/SIM/B/PLW/RUF/DTZ）。刻意没开 pyupgrade 的注解现代化——那会把 130 多处
`Optional[X]` 一次性改成 `X | None`，属于独立的一次风格重构。
另外关掉了 RUF001-003：本项目是中文代码库，注释里的全角标点是正确的。

---

## 测试

```bash
python tests/run_all.py              # 跑全部
python tests/test_static_names.py    # 静态检查：不允许出现未定义的名字
python tests/test_board_spec.py      # 标定板规格 + 真的跑一次相机标定
python tests/test_lut_roundtrip.py   # 打表链路，60 组参数组合
python tests/test_export_transaction.py   # 导出事务、回滚、重启后读表
```

零依赖，不需要 pytest——只要有 cv2 和 numpy 就行。四组测试都只用**合成数据**
（合成相机参数、合成单应、程序画出来的棋盘），不需要任何真实照片。

`test_static_names.py` 值得单独说一句：本项目真的出过一次
"把常量批量替换成 `board.corners`，而那个函数里没有 `board`，
于是运行相机标定直接 NameError"的事故——`compileall` 只查语法不查名字，
单元测试又恰好没调用那个函数，两个都拦不住。这个检查用标准库 `ast`
按作用域链解析每个名字的来源，把这类问题在提交前就拦下来。
它上线时立刻又抓出第二处：校验脚本里 `re` 被误删了导入却仍在使用。

`test_lut_roundtrip` 的参数矩阵是刻意铺开的，因为本项目的缺陷几乎都藏在组合里：

| 维度 | 取值 |
|---|---|
| 表格式 | txt / bin / c |
| 网格 | 原尺寸 / 320×240 / 160×120 |
| 定点位数 | Q0 / Q4 |
| heading | 0° / −1.6° |
| Knew | 与 K 相同 / alpha=0.5 算出的不同矩阵 |

每个组合都验证：导出网格 == 落盘网格、采样坐标空间仍是源图分辨率、
读回值与导出值逐值一致、`batch_test` 的输出网格等于最终表网格。

---

## 验证产物

```bash
python tools/verify_outputs.py            # 默认校验本工程
python tools/verify_outputs.py <工程根目录>
```

逐项检查：单应矩阵正确性、查找表与 `cv2.undistort` / `cv2.warpPerspective` 的一致性、
正反向表互逆性、无效哨兵分布。全通过才会输出 `N/N 项通过`。

标定前想先摸清素材质量：

```bash
python tools/scan_dataset.py <图片目录> --board-squares 12 9
```

---

## 打包成 exe

```bash
pip install -r requirements-dev.txt
python build_exe.py             # 或 python build_exe.py --clean
```

产物在 `dist/NEUQ-VisionCalib/`，已预置 `data/` 下的空目录与 `assets/checkerboard/`。
整个文件夹拷走即可运行，目标机器不需要装 Python。

---

## 坐标与约定

- 全程 OpenCV 0-based 像素坐标；物理坐标单位 cm，x 向右、y 向下（朝向车辆），
  标定矩形中心为原点，BirdView 图正上方为车辆前进方向。
- 单应分解 `H = T(anchor) @ S(scale) @ R(heading) @ H0`。
- 查找表：`reverse/` 给出每个输出像素对应的源图采样坐标；`forward/` 反之；
  无效点统一写 `-1`（映射到无穷远、落在地平线另一侧或超出图像范围）。
- 查找表支持三种落盘格式：逗号分隔文本、int16 定点二进制、C 头文件。
