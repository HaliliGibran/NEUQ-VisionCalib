# NEUQ 相机标定程序（NEUQ-VisionCalib）

智能车视觉标定一体化工具：相机内参标定 → 畸变矫正 → 逆透视（BirdView）标定 →
六矩阵与查找表导出 → 批量验证，全部流程可在本地 Web 控制台中完成。

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
├── data/                      运行数据（自动创建，**不入库**，见 .gitignore）
│   ├── calib_input/           相机标定照片，_incomplete/ 存检不出完整棋盘的
│   ├── calib_preview/         标定图去畸变验收图
│   ├── calib_data/            calib.json —— 相机内参与畸变系数
│   ├── ipm_input/             逆透视标定原图
│   ├── ipm_output/            去畸变图 + BirdView 结果
│   ├── matrix/                六矩阵、逆透视状态
│   ├── lookup_table/          undistort/ 与 undistort_ipm/ 两套正反向查找表
│   ├── test_input/            批量测试输入
│   ├── test_output/           批量测试输出
│   ├── backups/               「备份并清空」产生的归档
│   └── import/                待整理的原始素材
└── dist/                      打包产物（不入库）
```

**约定**：所有输入输出集中在 `data/` 下，由 `src/neuq_vision_calib.py` 的
`DATA_ROOT` 统一定义；`--root` 参数可整体改指向别的工程目录。

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
2. 从多个角度拍摄 15 张以上棋盘照片，放入 `data/calib_input/`
3. 拍摄一张地面（车道）照片放入 `data/ipm_input/`
4. 打开控制台，依次执行「相机标定 → 逆透视标定 → 导出」
5. 产物落在 `data/matrix/`（矩阵）与 `data/lookup_table/`（查找表）

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
python tools/scan_dataset.py <图片目录> [--corners 11 8]
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
