# 从一张相机图到可量距离的俯视图

## NEUQ-VisionCalib 背后的数学与图像知识

这篇导读写给第一次接触计算机视觉的同学。你只需要会高中解析几何，见过矩阵乘法，
再愿意耐心区分几套坐标系，就能顺着它走完本项目的主线。

### 推荐先看

如果你此前对矩阵变换、齐次坐标和透视投影比较陌生，建议先看下面两个视频。
不要求一遍就把所有推导完全看懂，先建立“矩阵如何描述空间变换”以及
“为什么透视投影需要齐次坐标”的直觉，再往下读会轻松很多。

1. [《无所不能的矩阵 - 三维图形变换》](https://www.bilibili.com/video/BV1b34y1y7nF)
2. [《探秘三维透视投影 - 齐次坐标的妙用》](https://www.bilibili.com/video/BV1LS4y1b7xZ)

前一个可以帮助理解后文反复出现的平移、旋转、缩放与矩阵组合；
后一个与齐次坐标、透视投影以及后面的单应矩阵关系更直接。

看完后不必急着记公式，只要先建立一个印象：

> **矩阵不只是一张数字表，它可以表示“一个点经过怎样的空间变换，最终到了哪里”。**

后面的相机内参、去畸变、单应矩阵和逆透视，本质上都是在逐步回答这个问题，
只是其中还会加入镜头畸变这样的非线性环节。

先约定两个阅读习惯：

1. 看到一个点，先问“它属于哪个坐标系、单位是什么、x/y 朝哪边”。
2. 看到一张映射表，先问“谁是索引、谁是表值、方向是 source → destination
   还是 destination → source”。

这两问看起来朴素，却能避开本项目里大多数方向错误。

---

## 0. 先看全局：程序到底在做什么

我们的目标不是单纯把画面“拉正”，而是得到一张有公制意义的俯视图：图上相同数量的
像素，应当代表相同数量的厘米。这样下游才能讨论赛道宽度、横向偏差和转弯半径。

~~~text
世界中的地面
     │  摄像头成像：三维世界压到二维，并叠加镜头畸变
     ▼
带畸变透视图（原始相机图）
     │  使用 K、D 和 Knew 去畸变
     ▼
理想针孔图像（去畸变图）
     │  使用平面单应 H 做逆透视
     ▼
可量距离的地面俯视图（BirdView）
     │  把逐像素计算提前完成
     ▼
LUT：每个输出像素应该去原图哪里取颜色
     │
     ▼
嵌入式端（C 代码）实时查表 + 插值
~~~

这里其实有两个时间尺度：

- **标定与导出阶段**比较慢，但只需在相机安装、镜头或参数变化后重做。它负责求
  K、D、H，再把复合映射预计算成 LUT。
- **车辆运行阶段**每帧都要执行。它不再解方程，只按 LUT 去原图取颜色。

所以 LUT 不是另一个算法，而是把已经确定的几何答案提前存下来，用空间换时间。

> 在本项目里对应：
>
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的
>   calibrate_camera、IpmCalibrator、export_all、batch_test
> - [src/neuq_core/lut.py](src/neuq_core/lut.py) 的
>   build_composite_reverse_map

---

## 1. 坐标系：先搞清“一个点到底在哪”

同一个 [100, 50]，如果不知道坐标系和单位，它什么都不是。它可能是原图第 100 列、
第 50 行，也可能是地面上向右 100 cm、向前 50 cm。数值相同不代表几何意义相同。

本项目会遇到这些坐标系：

| 名称 | 一个点表示什么 | 单位与方向 |
|---|---|---|
| 世界坐标 | 现实空间中的点 | 自选单位和轴向；标定棋盘通常以 mm 表示 |
| 相机坐标 | 以相机光心为原点的三维点 (Xc, Yc, Zc) | 通常 Zc 沿光轴向前 |
| 归一化相机坐标 | (x, y) = (Xc/Zc, Yc/Zc) | 无量纲，还不是像素 |
| 原始畸变像素 | 传感器实际拍到的 (u, v) | px，x 向右、y 向下 |
| Knew 去畸变像素 | 理想针孔图上的 (u, v) | px，x 向右、y 向下 |
| 标定矩形坐标系 | 四个实测地面点所在的平面 | cm，原点在矩形中心，x 向右、y 向下 |
| 逆透视坐标系 | 从标定矩形坐标系平移、旋转后的地面 | cm，逆透视坐标参考原点取去畸变图底边中点对应的地面点 |
| 俯视图像素 | 最终俯视画布上的点 | px，x 向右、y 向下 |

“标定矩形坐标系”和“逆透视坐标系”尤其容易混。前者的原点只是实测矩形中心，
方便四点建立公制关系；后者才是最终构图用的坐标系。本项目把去畸变图**底边中点**
投到地面，得到**逆透视坐标参考原点**。它不是相机光心，也不自动等于车头或保险杠位置。

去畸变图宽高记作 W、H 时，底边中点的精确像素坐标是：

~~~text
((W - 1) / 2, H - 1)
~~~

像素中心采用 0-based 坐标。偶数宽图像没有某一个像素正好位于几何中心，因此这个点
会落在两个中间像素之间；例如 W=1280 时，横坐标是 639.5，而不是 640。

还有一条容易忽略的边界：像素坐标 0 表示第 0 个像素的中心，不是图像左边界外侧。
宽 W 的图像，其像素中心横坐标是 0 到 W-1。

> 在本项目里对应：
>
> - [src/neuq_core/geometry.py](src/neuq_core/geometry.py) 的
>   physical_rect、ground_reference_origin、reference_origin_px、target_window_cm
> - [src/neuq_core/lut.py](src/neuq_core/lut.py) 的
>   table_spaces、pixel_grid
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的
>   IpmCalibrator.ground_transform

---

## 2. 矩阵到底在这里干什么

先看普通二维变换：

~~~text
x' = a x + b y + c
y' = d x + e y + f
~~~

左上角的 a、b、d、e 可以旋转、缩放或错切；c、f 负责平移。若只写 2×2 矩阵，
常数 c、f 没有地方放。于是我们给二维点补一个 1：

~~~text
[x']   [a b c] [x]
[y'] = [d e f] [y]
[ 1]   [0 0 1] [1]
~~~

[x, y, 1] 叫作**齐次坐标（homogeneous coordinates）**。多出来的一维不表示现实里
多了一条轴，它是一种记账方法：让旋转、缩放、平移甚至透视都能写成 3×3 矩阵，
并通过矩阵连乘组合。

齐次坐标还有一条关键规则：

~~~text
[x, y, w]、[k x, k y, k w]（k ≠ 0）表示同一个二维点。
普通坐标 = [x / w, y / w]
~~~

例如 [2, 4, 2] 和 [1, 2, 1] 都表示二维点 (1, 2)。后文单应公式里的“约等于”
符号，表达的正是这种比例等价。

矩阵组合要**从右往左读**。若 H = A · B · C，那么点先经过 C，再经过 B，最后经过
A。很多“旋转中心怎么跑了”的错误，本质上都是把这条顺序读反。

> 在本项目里对应：
>
> - [src/neuq_core/geometry.py](src/neuq_core/geometry.py) 的
>   translation_matrix、rotation_matrix、scale_matrix、apply_homography
> - apply_homography 最后除以第三个齐次分量，完成“去齐次化”

---

## 3. 针孔相机模型与内参 K

把相机想成一个理想小孔。空间点离相机越远，在成像平面上看起来越靠近中心。
相似三角形给出：

~~~text
x = Xc / Zc
y = Yc / Zc
~~~

(x, y) 是归一化相机坐标。再把它换成像素：

~~~text
u = fx · x + cx
v = fy · y + cy
~~~

合起来就是针孔模型：

~~~text
                 [fx  0 cx]
s [u v 1]^T = K [Xc Yc Zc]^T,   K = [ 0 fy cy]
                 [ 0  0  1]
~~~

- **fx、fy** 是横向和纵向的焦距，单位是像素。
- **cx、cy** 是主点，直觉上接近光轴落在传感器的位置，也通常接近图像中心。
- **s** 是齐次比例，实际就是这里的深度 Zc。

焦距为什么以“像素”为单位？真实焦距可以用毫米表示，但投到数字图像还要除以单个
像素的物理尺寸。把这两步合起来，便得到“一个单位的归一化坐标对应多少像素”，
也就是 fx、fy。

K 只描述相机内部成像尺度，所以叫**内参**。每张棋盘照片还有自己的相机姿态：
旋转 rvec 和平移 tvec 把棋盘的世界坐标变到相机坐标。它们叫**外参**。相机没动而
棋盘换了姿态时，K 不变，每张图的 rvec/tvec 会变。

> 在本项目里对应：
>
> - [src/neuq_core/calibration.py](src/neuq_core/calibration.py) 的
>   _calibrate_once、report_reprojection_error
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的 fit_camera、
>   calibration_payload、resolve_new_camera_matrix
> - fit_camera 返回每张图的 rvecs、tvecs；calib_data/calib.json 持久化共享的
>   camera_matrix 与 dist_coeffs

---

## 4. 为什么真实镜头不是针孔：畸变

真实镜头要收集更多光线，不可能是没有厚度的小孔。广角镜头尤其明显：直线在画面边缘
会向外鼓或向内收。OpenCV 当前这条标定链使用五参数模型：

~~~text
D = [k1, k2, p1, p2, k3]
~~~

注意，下面的 x、y 是**归一化相机坐标**，不是像素：

~~~text
r² = x² + y²
radial = 1 + k1 r² + k2 r⁴ + k3 r⁶

xd = x · radial + 2 p1 x y + p2 (r² + 2 x²)
yd = y · radial + p1 (r² + 2 y²) + 2 p2 x y
~~~

- k1、k2、k3 描述**径向畸变（radial distortion）**。r 越大，点离主点越远，
  高次项影响越明显。
- p1、p2 描述**切向畸变（tangential distortion）**，主要补偿镜片与传感器没有
  完全同轴带来的偏斜。

得到 (xd, yd) 后，再用 K 换回原始畸变像素。这里最容易踩坑的是把像素坐标直接代入
r²。像素值动辄上千，模型定义却是在归一化坐标上；少了 K⁻¹ 这一步，量纲就错了。

为什么棋盘角点要覆盖画面边缘？如果所有数据都挤在中心，r 很小，r⁴、r⁶ 更小，
优化器几乎看不到 k2、k3 的作用。即使中心区域重投影误差很漂亮，也不能证明边缘畸变
可信。数学只能从数据中学习约束，不能替你补出没拍到的边缘。

> 在本项目里对应：
>
> - [src/neuq_core/lut.py](src/neuq_core/lut.py) 的 distort_points：
>   Knew 像素 → 归一化坐标 → 按 K、D 重新施加畸变
> - [src/neuq_core/calibration.py](src/neuq_core/calibration.py) 返回的 dist
> - calib_data/calib.json 中的 dist_coeffs

---

## 5. 相机标定到底是在“求什么”

棋盘给了我们一组已知真实几何。假设每格边长是 20 mm，就能在棋盘平面上写出每个
**内角点**的 object points，例如 (0,0,0)、(20,0,0)、(40,0,0)……；照片里的同一批
角点由检测器给出 image points，单位是像素。

先区分两个容易混的数量：

- “12×9 方格”说的是黑白小方格数量。
- 相邻方格之间才有内角点，因此对应 11×8 个内角点。

程序用 findChessboardCornersSB 寻找完整角点阵列。SB 版本对大视角和噪声通常更稳，
但“检出”并不表示照片就足够好：运动模糊、反光、所有姿态过于相似仍会削弱标定。

标定要共同求：

~~~text
一套全局 K、D
+
每张照片各自的 rvec、tvec
~~~

算法拿这些参数预测每个 object point 应该落到哪个像素，再与检测到的 image point
比较。所有点的误差平方加总并尽量压小，就是**非线性最小二乘**的直觉。你不必先会
完整的 Zhang 标定法，也能把它理解成“反复调参数，让模型投影尽量贴住观测”。

多张不同姿态照片很重要，因为单张正对棋盘的照片会让某些参数彼此冒充：改变焦距、
改变距离，可能得到相似的图像。倾斜、远近、位置不同的视角，会给优化器更丰富的约束。

### 重投影误差怎么看

对第 i 个角点，预测位置与实测位置的二维距离记为 eᵢ。

~~~text
RMS = sqrt(mean(eᵢ²))
mean Euclidean error = mean(eᵢ)
~~~

平方会让大误差点在 RMS 中权重更高，所以两个指标数值口径不同。OpenCV 标定接口报告
RMS；一些工具更常报告平均欧氏距离，不能只看数字大小就断言谁更好。

高误差帧被删除后为什么必须重新标？因为 K、D 是用“所有保留帧共同拟合”的。
删掉一张图只是改变了数据集，旧参数仍是旧数据集的最优解；必须在新集合上重新优化。

先别急着看 RMS。一个很小的全局 RMS 仍可能掩盖这些问题：

- 角点只覆盖中心，边缘 D 没有约束；
- 相机姿态太相似，参数可辨识性差；
- 棋盘不平、格子尺寸量错，模型从起点就带偏差；
- 某一张边缘照片很差，却被大量中心点平均掉。

> 在本项目里对应：
>
> - [src/neuq_core/config.py](src/neuq_core/config.py) 的 CheckerboardSpec
> - [src/neuq_core/calibration.py](src/neuq_core/calibration.py) 的
>   detect_chessboard、_calibrate_once、report_reprojection_error
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的
>   fit_camera、calibrate_camera、commit_calibration

---

## 6. 去畸变为什么不能用一张 3×3 矩阵

3×3 投影矩阵能统一表达一次线性/投影变换；它对所有点使用同一组系数。镜头畸变却含
r²、r⁴、r⁶：点离主点多远，偏移量就有多大。这是**非线性**关系，不能塞进一张固定
3×3 矩阵。

因此去畸变通常按像素做：

1. 决定理想输出图上的某个像素。
2. 把它还原成理想归一化射线。
3. 按畸变公式算出这条射线在原图落到哪里。
4. 从原图那个小数坐标取色。

这里还会出现 **Knew**。K 描述原始相机；Knew 描述我们选择怎样把理想射线排到去畸变
输出画布上。调整 Knew 可以在“保留更多边缘视野但出现黑边”和“裁掉一些边缘以填满
画布”之间折中。它不是新相机真的换了焦距，而是输出画布的投影约定变了。

这也解释了为什么标定基准（calibration basis）指纹必须包含 Knew：逆透视四点是在 Knew 定义的
去畸变图上选的。哪怕 K、D 没变，只要 Knew 变了，同一像素就代表不同射线，旧四点与
旧 H0 不能静默复用。

这里还要分清“计算”和“保存”：calib.json 保存 K、D、标定分辨率与来源追溯信息（provenance），
Knew 根据 K、D 和 UNDIST_ALPHA 现场计算。导出阶段才把 Knew 本身写入
matrix/matrices.json；calib.json 中保存的 calibration_basis_hash 已包含 Knew，
所以 Knew 改变仍会让旧 IPM/LUT 被判为过期（stale）。

cv2.undistortPoints 的语义是“给我畸变点，求理想点”，适合 source → destination 的
forward 表。reverse 渲染问的却是“给我理想输出点，原图去哪取色”，方向相反。
本项目用 Knew⁻¹ 和 cv2.projectPoints 显式施加正向畸变，避免把 API 名字看顺眼就硬反用。

> 在本项目里对应：
>
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的
>   resolve_new_camera_matrix、calibration_basis_hash、calibration_basis_stale
> - [src/neuq_core/lut.py](src/neuq_core/lut.py) 的
>   undistorted_grid、distort_points

---

## 7. 什么是单应 Homography

如果一批点都在同一个平面上，那么这个平面从一个视角到另一个视角的投影关系，可以用
一张 3×3 矩阵描述。这张矩阵叫**单应（Homography）**：

~~~text
q ~ H p
~~~

波浪号不是“差不多”，而是“齐次比例等价”。把二维点 p = (u, v) 展开：

~~~text
x' = h11 u + h12 v + h13
y' = h21 u + h22 v + h23
w' = h31 u + h32 v + h33

q = (x'/w', y'/w')
~~~

分母 w' 是投影变换与普通仿射变换最不同的地方。仿射矩阵的最后一行固定为 [0,0,1]，
分母永远是 1；单应的分母随位置变化，于是平行线可以在远处相交、近处和远处尺度不同。

单应只对**同一平面**成立。地面上的赛道线可以用它拉成俯视图，离地的锥桶顶端、车身
或人的头不会落在同一平面上，拉正后仍会歪。这不是实现出错，而是模型边界。

本项目有两张相关矩阵：

- H0：Knew 去畸变像素 → 标定矩形坐标系，单位从 px 变为 cm。
- H：Knew 去畸变像素 → 最终俯视图，单位从 px 变为输出 px。

> 在本项目里对应：
>
> - [src/neuq_core/geometry.py](src/neuq_core/geometry.py) 的
>   compute_homography、apply_homography、homography_denominator
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的
>   IpmCalibrator.recompute、IpmCalibrator.compose

---

## 8. 为什么四个点就够了：自由度、退化与 DLT

H 有 9 个数，但整体乘以任意非零倍数，投影结果不变。因此它实际上只有 8 个自由度。
一对对应点提供输出 x、y 两条独立约束：

~~~text
四对点 × 每对两条约束 = 8 条约束
~~~

刚好可以确定一张一般的平面单应。这就是“四点够用”的自由度解释。

**直接线性变换（Direct Linear Transform, DLT）**把每对点的关系整理成 A h = 0，
h 是把 H 的 9 个元素排成的向量。奇异值分解（SVD）寻找最接近 A 零空间的向量，
再把它还原成 H。

实际计算前还要做 Hartley 归一化：

1. 把点集平移到重心附近；
2. 统一缩放，使平均距离约为 √2；
3. 在数值较温和的坐标上解 DLT；
4. 最后撤销两边的归一化。

这不是改变几何，而是避免“像素上千”和“齐次常数 1”同时进入方程，让浮点舍入把
小量吞掉。

为什么共线不行？若四点都在一条线上，我们只知道这条线如何映射，不知道平面离开这条
线后怎样变化；二维信息退化成了一维。四点重合、几乎重合或挤在很小区域也会让方程
病态：理论上也许有解，微小点击误差却会让结果剧烈变化。自交四边形则可能给出一张
数学上可算、物理上镜像扭曲的 H。

所以“有 4 个数组”不是充分条件。点要有面积、对应顺序一致，并构成正常凸四边形。

> 在本项目里对应：
>
> - [src/neuq_core/geometry.py](src/neuq_core/geometry.py) 的
>   normalize_points_for_dlt、compute_homography、quad_area、is_convex_quad、
>   order_corners_tl_tr_bl_br
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的
>   IpmCalibrator.build_corners

---

## 9. H0 与最终 H：测量几何和画面构图是两回事

本项目的真实组合公式是：

~~~text
H = T(anchor)
  · S(scale)
  · R(heading)
  · T(-reference_origin)
  · H0
~~~

矩阵从右往左作用，逐步看最清楚。

### 9.1 H0：先把像素变成厘米

H0 由去畸变图上的四点与 physical_rect 给出的实测矩形四角建立：

~~~text
Knew 去畸变像素 --H0--> 标定矩形坐标系 cm
~~~

physical_rect 的原点在标定矩形中心。这个选择让四角坐标对称，方便建立 H0，但它只是
“量尺的中心”，不是车辆业务原点。

### 9.2 T(-reference_origin)：换一个真正用于构图的原点

reference_origin 是去畸变图底边中点经 H0 落到地面的点，仍以标定矩形坐标系的 cm
表示。减掉它之后，这个地面点变为 (0,0)。

标定矩形中心 ≠ 逆透视坐标参考原点。前者随你把标定矩形摆在哪里而变；后者固定对应
当前画面底边中点的地面射线。把二者混为一谈，锚点的物理含义就会漂。

### 9.3 R(heading)：绕参考原点应用朝向偏移

先平移、后旋转非常重要。写成 R · T(-ref)，点先被移到参考原点，再绕 (0,0) 旋转。
如果顺序反过来，reference_origin 会先绕标定矩形中心旋转，之后再平移，最终不再落到
锚点。由于图像 y 向下，本项目里的正角在屏幕上看是顺时针。

### 9.4 S(scale)：用比例尺把厘米变成像素

比例尺 scale 的单位是 px/cm。x、y 使用相同倍率，才保留公制纵横比。地面上 45 cm 的正方形
在俯视图里仍应是正方形。

### 9.5 T(anchor)：用锚点把参考原点摆到画布上

横向锚点 anchor_x、纵向锚点 anchor_y 是 0 到 1 的归一化位置，实际像素为：

~~~text
anchor_px = (anchor_x · (W-1), anchor_y · (H-1))
~~~

因此最终 H 应把去畸变图底边中点准确送到锚点 anchor_px。它是一条很有价值的验收不变量。

> 在本项目里对应：
>
> - [src/neuq_core/geometry.py](src/neuq_core/geometry.py) 的
>   physical_rect、ground_reference_origin、rotation_matrix、scale_matrix、
>   translation_matrix
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的
>   IpmCalibrator.anchor_px、ground_transform、compose、recompute

---

## 10. 地平线为什么会跑到“无穷远”

对 H0 来说，齐次分母是：

~~~text
w = h31 x + h32 y + h33
~~~

当 w 接近 0：

~~~text
x_ground = x' / w
y_ground = y' / w
~~~

会变得极大。几何上，这条 w = 0 的直线就是地平线：与地面平行的视线不会在有限距离
内碰到地面，其交点位于**投影无穷远（projective infinity）**。

这里有三个工程后果：

1. 不能因为计算机还能给出一个 10¹⁵ 量级的数，就把它当成真实地面坐标。
2. 地平线两侧的 w 符号不同，其中一侧是相机能看到的有限地面，另一侧不是合法地面。
3. 即使不恰好等于 0，太接近 0 的位置也会把一点像素噪声放大成巨大的地面误差。

horizon_sign 用标定四点判断哪一侧是可信地面侧，因为这四点明确属于有限地面。
随后用 sign · w ≥ DEN_EPS 留出安全余量。只检查 |w| 不够：它会把地平线另一侧同样
“离 0 很远”的点也收进来。

valid_fov_polygon 先在去畸变图像素平面用半平面裁剪去掉错误一侧，再映到地面 cm，
最后截断过远的前向和横向范围。半平面裁剪逐边判断“端点在内还是在外”，跨边界时补
交点；它保留的是一个几何多边形，而不是仅凭四角猜测。

> 在本项目里对应：
>
> - [src/neuq_core/geometry.py](src/neuq_core/geometry.py) 的
>   homography_denominator、horizon_sign、clip_polygon_halfplane、DEN_EPS
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的 valid_fov_polygon
> - [src/neuq_core/lut.py](src/neuq_core/lut.py) 的
>   build_composite_reverse_map 对地平线有限侧的再次检查

---

## 11. 为什么还需要“目标地面范围”

**valid region** 回答：“从投影数学看，哪些地面仍可由相机图像解释？”

**目标地面范围（代码中的 target window）**回答：“为了当前任务，我们希望俯视图
重点装下哪一块地面？”

它们不是同一件事。数学上能映射到 6 米宽，不等于应该把 6 米全塞进 1280×720。
如果强行容纳所有远处与侧面，有用的近场会被压得很小；反过来把远处大幅放大，也只是
把少量源像素摊成很多输出像素，画面会糊，并不会产生新信息。

本项目的目标地面范围以逆透视坐标参考原点为 (0,0)：

~~~text
x ∈ [-width/2, +width/2]
y ∈ [-forward, 0]
~~~

图像 y 向下，所以“车辆前方”取负 y；y=0 是靠近车辆的那条边。自动布局让近边贴近
画布底部，并求能完整容纳这个范围的最大比例尺。

为什么 fit_bottom_aligned 可以闭式求解，而不用一点点试比例尺？对地面点 q：

~~~text
output = anchor + scale · q
~~~

在横向锚点 anchor_x 和贴底位置确定后，上、左、右边界对比例尺都只给出一次不等式
上限。把“纵向装得下、左侧装得下、右侧装得下”的上限取最小值，就是最大可行比例尺。

full_fov_fit_scale 的含义不同：它是“若想把整个数学有效视野都装下，需要多小的
比例尺”。它是诊断量，不是用户比例尺的上限。目标地面范围更小，所以正常比例尺完全
可能大于它。

> 在本项目里对应：
>
> - [src/neuq_core/geometry.py](src/neuq_core/geometry.py) 的
>   target_window_cm、fit_bottom_aligned、max_scale_for_fov
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的
>   IpmCalibrator.target_window、target_layout、apply_target_layout、
>   full_fov_fit_scale

---

## 12. 比例尺 px/cm 是怎么来的

假设地面上一条真实边量得 45 cm，它在最终俯视图中应该占 215.1 px，那么：

~~~text
scale = 215.1 px / 45 cm
      = 4.78 px/cm
~~~

含义是俯视图中每前进 4.78 个像素，地面距离前进 1 cm。反过来：

~~~text
1 px = 1 / 4.78 cm ≈ 0.209 cm
~~~

比例尺不是“看起来舒服”的缩放倍率，而是连接现实长度与图像长度的物理量。它依赖
标定矩形的真实尺寸：若实际是 45 cm，却误填成 50 cm，程序仍能把四点拉成漂亮矩形，
但所有距离都会带约 11% 的系统误差。画面看正了，不代表量尺就准了。

比例尺对 x/y 必须相同。若横向用 4.78 px/cm、纵向用另一个数，虽然可以把画布填满，
却会把圆变成椭圆、把角度和曲率一起改变。

实际检查可以拿标定矩形回代：TL→TR 的像素长度应等于 phys_w_cm × scale，
TL→BL 应等于 phys_h_cm × scale，且两条边仍应垂直。

> 在本项目里对应：
>
> - [src/neuq_core/geometry.py](src/neuq_core/geometry.py) 的 scale_matrix
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的
>   IpmCalibrator.scale、compose
> - [tools/verify_outputs.py](tools/verify_outputs.py) 的 check_homography

---

## 13. 什么是 LUT

LUT 是 **Look-Up Table，查找表**。本项目导出两类查找表（去畸变、去畸变+逆透视），每类各含正向/反向，共 4 组 LUT。
若嵌入式端每帧都对每个像素计算 r²、r⁴、r⁶、
矩阵乘法和除法，代价很高；相机安装和输出几何不变时，这些答案其实每帧都一样。
于是 PC 端提前算好：

~~~text
这个格子的答案是什么？——直接查表。
~~~

本项目导出四种语义：

| 表 | 索引是谁 | 表值是谁 |
|---|---|---|
| undistort/reverse | Knew 去畸变输出像素 | 原始畸变图采样坐标 |
| undistort/forward | 原始畸变图像素 | Knew 去畸变图落点 |
| undistort_ipm/reverse | 俯视图输出像素 | 原始畸变图采样坐标 |
| undistort_ipm/forward | 原始畸变图像素 | 俯视图落点 |

如果只记住一句话：

> reverse 表是“输出图这个像素，要去源图哪里取颜色”。

复合 reverse LUT 尤其重要。对俯视图的每个输出像素，它先用 H⁻¹ 回到 Knew
去畸变图，再施加畸变回到原始图。嵌入式端一次 remap，就同时完成“去畸变 + IPM”，
无需生成中间图。

LUT 有两个尺寸概念：

- **索引网格尺寸**：表有多少行、多少列。
- **source_size**：表值所引用的原图分辨率。

降采样后这两个数不同，后文会专门解释。

> 在本项目里对应：
>
> - [src/neuq_core/lut.py](src/neuq_core/lut.py) 的
>   MapPair、table_spaces、undistorted_grid、build_composite_reverse_map
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的
>   export_undistort_tables、export_composite_tables、export_all

---

## 14. 为什么实际渲染更喜欢 reverse map

设想拿着一张空白目标画布逐格填色。

### forward：从源图向外“撒”

对每个源像素，算它最终落到目标图哪里。这适合追踪点，但用于生成稠密图像会遇到：

- 落点是小数，取整后多个源像素可能撞进同一目标格；
- 相邻源像素经放大后，中间可能隔开多个目标格，留下空洞；
- 谁覆盖谁、空洞怎么补，都要额外制定规则。

### reverse：从目标图向回“取”

遍历每个目标像素，反问它对应源图的哪个位置。这样目标画布的每一格天然都有一次查询，
没有“忘了填某格”的问题。小数采样位置再用插值处理即可。这叫 gather；forward
逐点向外写常被称为 scatter。

当然，reverse 坐标也可能落出原图、跨到地平线另一侧，或根本没有有限对应点。
这些位置必须标成无效，而不是硬夹到最近边界，否则边缘会被拖成一条假纹理。

forward 表仍然有价值：它可以回答“原图某点最终会去哪”，也能与 reverse 做往返诊断。
但若目标是把整幅图稳定渲染出来，reverse 是更自然的主表。

> 在本项目里对应：
>
> - [src/neuq_core/lut.py](src/neuq_core/lut.py) 的
>   build_composite_reverse_map、mask_out_of_range
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的 batch_test 使用
>   reverse MapPair 重建实际交付图

---

## 15. 双线性插值：小数坐标怎样取颜色

LUT 表值可以是 (123.25, 80.6) 这样的源图小数坐标。它落在四个像素中心之间。
若直接四舍五入到最近像素，画面移动时会一格一格跳，斜线也容易出现锯齿。

设：

~~~text
x0 = floor(x),  x1 = x0 + 1,  a = x - x0
y0 = floor(y),  y1 = y0 + 1,  b = y - y0
~~~

四个邻居的权重是：

~~~text
I(x, y) =
    (1-a)(1-b) I(x0, y0)
  + a(1-b)     I(x1, y0)
  + (1-a)b     I(x0, y1)
  + ab         I(x1, y1)
~~~

可以先在上边两点间按 a 插值，再在下边两点间按 a 插值，最后在两条结果间按 b 插值。
四个权重都非负且总和为 1，所以结果是四邻居颜色的平滑加权平均。

这就是**双线性插值（bilinear interpolation）**。它不会创造源图没有的细节，
只是比最近邻更平滑地估计像素之间的连续值。

边界还有一个严格条件：若四邻居中有人无效，往返诊断不能装作有完整插值依据。
本项目只在四邻域都有效时把该点计入可判定覆盖率。

> 在本项目里对应：
>
> - [tools/verify_outputs.py](tools/verify_outputs.py) 的 bilinear_at
> - OpenCV 的 cv2.remap(..., INTER_LINEAR) 用同类思想按 reverse 表取色
> - [src/neuq_core/lut.py](src/neuq_core/lut.py) 的 resample_map_pair

---

## 16. LUT 为什么可以降采样

完整 1280×720 网格有 921,600 个位置；x/y 各存一份，表可能占用不少存储。
映射在大多数区域变化平滑，因此可以只保存更稀的索引网格，让嵌入式端恢复或直接生成
较小输出图。

但一定要分清：

~~~text
LUT 的索引网格变小了
≠
LUT 里面存的源坐标也变小
~~~

例如 1280×720 的 reverse 表降到 320×180，索引只剩四分之一宽和高；表值仍应是
原始 1280×720 图上的采样坐标。若某项原来写 1000.5，降采样后它附近的表值仍在
1000.5 左右，不能再除以 4。否则它会跑到原图完全不同的位置。

### 像素中心为什么有 +0.5 / -0.5

倍率为 n 时，小图第 i 格覆盖原图中的 n 格。把像素看成有面积的小方格：

~~~text
full = (small + 0.5) · n - 0.5
~~~

以 n=4、小图 small=0 为例：

~~~text
full = (0 + 0.5) · 4 - 0.5 = 1.5
~~~

它正好是原图第 0、1、2、3 四个像素中心的平均位置。若简单写 full=small·n，
第一个中心会错放到 0，整张网格产生半像素系的偏移。

### 为什么必须整数倍等比

1280×720 → 320×180，x/y 都缩 4 倍，几何比例不变。

1280×720 → 320×240，x 缩 4 倍、y 缩 3 倍。一个俯视图正方形会变成长方形，
角度、曲率、横向距离全部失真。后者不是“少量画质损失”，而是公制几何被破坏。

要求整数倍还让网格中心对应关系、内存布局和嵌入式端倍率都更明确。非整数比例并非数学上
绝对不能插值，但它会增加一套没有必要的坐标约定，本项目明确不接受。

> 在本项目里对应：
>
> - [src/neuq_core/lut.py](src/neuq_core/lut.py) 的
>   downsample_factor、table_grid_factors、resample_map_pair
> - [tools/verify_outputs.py](tools/verify_outputs.py) 的
>   grid_from_full、full_from_grid、check_grid_isotropy

---

## 17. 什么是 Q4 定点数

许多 MCU 没有充裕的浮点算力或内存。定点数的办法是：约定整数的低 N 位代表小数。
Q4 表示乘以 2⁴ = 16 后存成整数。

~~~text
x = 12.375
q = round(x · 16) = 198
恢复时 x ≈ q / 16 = 12.375
~~~

Q4 的步长是 1/16 px。四舍五入后，单轴最大量化误差为半个步长：

~~~text
单轴误差 ≤ 0.5 / 16 = 0.03125 px
二维欧氏误差 ≤ √2 · 0.5 / 16 ≈ 0.0442 px
~~~

这正是验收脚本使用的理论容差，而不是拍脑袋给一个“差不多 1 px”。

### int16 为什么会溢出

有符号 int16 范围是 -32768 到 32767。对 1280 宽原图，最大合法横坐标是 1279：

~~~text
Q4: 1279 × 16 = 20464   ✓ 可表示
Q5: 1279 × 32 = 40928   ✗ 超出 int16
~~~

小数位越多，精度越高，但可表示的整数范围越小。这是一场范围与精度的交换。

### sentinel 为什么取 -32768

有些输出像素没有合法源坐标，需要一个不会与正常坐标混淆的特殊值。
-32768 是 int16 最小值，正常像素坐标及其定点值都非负，因此适合作为
**哨兵（sentinel）**。读回后统一解码为 -1，便于 Python 侧判断。

X/Y 必须同时无效。若 X 是 sentinel、Y 却合法，嵌入式端会拼出一个不存在的二维点。
量化前也必须先识别无效值，不能让 NaN 或越界数在强制转整数时变成看似合法的载荷。

> 在本项目里对应：
>
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的
>   quantize_table、dequantize_table、prepare_map_pair、BIN_SENTINEL
> - [tools/verify_outputs.py](tools/verify_outputs.py) 的
>   quant_tolerance、check_sentinel、compare_to_pipeline

---

## 18. Jacobian 与“远场为什么糊”

想象源图远处只有一条 2 px 宽的赛道线。逆透视把远处拉开后，它也许要占俯视图
二十几个像素。算法可以平滑地填满这二十几个像素，却不知道原来 2 px 之间真实世界的
纹理。它不是在恢复被压缩的细节，只是在放大已有信息。

为了描述“某个位置附近放大多少”，可以看局部 **Jacobian（雅可比矩阵）**：

~~~text
    [ ∂x'/∂x  ∂x'/∂y ]
J = [                   ]
    [ ∂y'/∂x  ∂y'/∂y ]
~~~

它可以理解为局部的一张 2×2 线性近似：

~~~text
源图里一个很小的位移 Δp
经过映射后约变成 J · Δp
~~~

若 J 的局部放大很大，源图 0.2 px 的坐标误差，到了俯视图可能变成几 px。
这解释了为什么早期“reverse → 最近邻源网格 → forward”的往返检查会在远场出现大
误差：最近邻先引入不到半像素的源图取整误差，Jacobian 再把它放大。那不一定说明两张
表方向错了，也可能是检查方法自己制造了误差。

改成对 forward 表做双线性采样，可以显著减小最近邻栅格误差；但降采样、面积重采样和
定点量化本身仍不可逆，所以降采样表的往返结果只作为诊断，不应拿一个固定阈值硬判。

如果只记住一句话：**远场模糊不是算法凭空制造的失败，而是源图远处本来就没有足够
采样信息。** 改善它通常要从相机俯角、分辨率、焦距、目标地面范围和比例尺的取舍入手。

> 在本项目里对应：
>
> - [tools/verify_outputs.py](tools/verify_outputs.py) 的
>   check_forward_reverse、bilinear_at
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的
>   full_fov_fit_scale 与目标地面范围布局，帮助识别过度拉伸

---

## 19. 验证脚本为什么这样验

运行 tools/verify_outputs.py，不只是看最后一行“通过”。七项检查各自守住一层契约。
把它们分层，是为了失败时知道该查几何、表方向、量化，还是落盘格式。

### 19.1 H 的四个不变量

**验什么：**

1. 标定矩形两条边长度分别等于实物宽/高 × 比例尺；
2. 邻边仍垂直；
3. 上边方向等于朝向偏移 heading（按 180° 周期比较）；
4. 去畸变图底边中点最终落到锚点 anchor。

**为什么这样验：** 朝向偏移不为 0 时，矩形本来就不轴对齐，拿“边必须水平/竖直”
作判据会误报。长度、垂直、朝向、参考原点对任意旋转都成立。

**失败通常意味着：** H 的组合顺序、四角对应、物理尺寸、比例尺、朝向偏移或
逆透视坐标参考原点语义有误。

### 19.2 LUT grid 必须整数倍等比

**验什么：** 源图到表网格的 x/y 缩小倍率相同，且是整数。

**为什么这样验：** 俯视图的公制尺度应各向同性；320×240 不能冒充 1280×720 的
等比小图，320×180 才可以。

**失败通常意味着：** 导出尺寸或 metadata 违反网格契约，不是相机内参突然坏了。

### 19.3 去畸变 reverse 表对 OpenCV 独立参考

**验什么：** 用 LUT + remap 重建的去畸变图，是否与 cv2.undistort 的结果一致。

**为什么这样验：** 这是独立实现的判定参照（oracle），可以发现 reverse 方向写反、Knew 用错、
坐标系错误或插值约定漂移。全分辨率下才逐像素比较；降采样后输出尺寸不同，硬比没有
同位语义，因此明确跳过。

**失败通常意味着：** K/D/Knew、destination → source 方向或表值坐标有问题。

### 19.4 复合 reverse 表对 OpenCV 两步参考

**验什么：** 一次复合 LUT remap，是否等于 OpenCV 的“先 undistort，再
warpPerspective”。

**为什么这样验：** 两条路径的代码结构不同。一边直接查复合表，另一边调用两个成熟
API，能对端到端方向和复合顺序形成独立约束。

**失败通常意味着：** 畸变与 H 的组合方向、H⁻¹、Knew、地平线有效区或插值不一致。

### 19.5 forward/reverse 往返

**验什么：** reverse 找到源图坐标后，在 forward 表上双线性采样，能否回到出发的
输出像素。

**为什么这样验：** 它检查两种方向是否近似互逆。全分辨率表用 p50、p95 守门，比最大
误差更不容易被少数边界点绑架；降采样表经历不可逆重采样与量化，只报告诊断。

**失败通常意味着：** forward/reverse 方向、两表有效域或坐标换算不一致。若只在高
放大远场变差，还要结合 Jacobian 判断，不能见到大 max 就直接判表坏。

### 19.6 sentinel 一致性

**验什么：** 各组表的 X/Y 无效掩码是否同步，解码后无效值是否统一为 -1。

**为什么这样验：** 一个二维采样点必须两个分量一起有效或一起无效，否则嵌入式端会拼出
假坐标。

**失败通常意味着：** 定点哨兵还原、文本/二进制序列化或 X/Y 配对有问题。

这项只验证哨兵的**表示一致性**，不判断无效区域的几何形状或连通性。无效区域是否
出现在正确位置，由下一项“落盘表 == 流水线重算”对 expected/actual mask 逐点比较；
这比把“看起来连通”设为硬性判据（hard gate）更直接。测试知道自己没验什么，和知道自己验了什么
同样重要。

### 19.7 落盘表与流水线重算

**验什么：** 重新按当前流水线生成 composite forward/reverse，经过同样网格重采样后，
是否与真正落盘的交付表逐值一致；容差来自文本小数位或 Q 格式理论量化误差。

**为什么这样验：** 前几项偏重数学语义，这一项守住“算对的结果有没有忠实写进文件”。
它对全分辨率和降采样、文本和定点格式都适用。

**失败通常意味着：** 重采样、Q 位数、文件读写、metadata 或哨兵序列化出了问题。

### 19.8 为什么需要两类判定参照

“两个结果互相一致”不等于“两者都正确”。如果生成表和验证表调用同一个错误 helper，
它们会非常一致地一起错。因此：

- OpenCV 图像参考是**独立判定参照**，重点检验数学方向和结果图；
- 流水线重算是**交付忠实度判定参照**，重点检验正反两张表、重采样、量化和落盘。

两者不是重复，而是互补。前者独立但在降采样时无法逐像素同位比较；后者覆盖所有格式，
却与生产链共享数学 helper，不能单独证明算法正确。

### 19.9 数学正确之外：来源追溯信息与事务

产物还必须回答“它是用哪一版输入生成的”。本项目把 K、D、Knew、图像尺寸等组成
标定基准指纹；产物记录这个指纹。当前基准变化后，旧四点、H 或 LUT 会被
判为过期状态，程序宁可要求重做，也不猜它们“可能还能用”。来源图再记录
路径与 SHA-256，验证时才能确认拿到的是当时那张，而不是同名的另一张。

写文件则采用事务思路。可以把它想成换桌布：新桌布先在旁边完整铺好（staging），
确认所有文件都成功后，再把旧桌布暂存为 old 并换上新的；全部完成才 finalize 删除
old。中途失败就 rollback。pending 和 residue 表示上次换装可能没完成，程序保留现场
让人判断，不擅自删除唯一原始资料。

这种保守不是“程序不够聪明”，而是数据价值不对称：多留一份残留通常还能人工处理，
误删唯一素材却无法恢复。事务保证的是“一组相关文件要么全是旧版，要么全是新版”，
避免新 calib.json 配旧预览、新矩阵配旧 LUT 这种比直接报错更难发现的混搭状态。

> 在本项目里对应：
>
> - [tools/verify_outputs.py](tools/verify_outputs.py) 的
>   check_homography、check_grid_isotropy、check_undistort_tables、
>   check_composite_tables、check_forward_reverse、check_sentinel、
>   check_export_fidelity
> - [src/neuq_core/fs_transaction.py](src/neuq_core/fs_transaction.py) 的
>   DirectorySwapTransaction、commit_dirs
> - [src/neuq_core/config.py](src/neuq_core/config.py) 的
>   MaterialImportTransaction、material_transaction_state、
>   material_stale_reason、require_no_material_residue
> - [src/neuq_vision_calib.py](src/neuq_vision_calib.py) 的
>   calibration_basis_hash、require_calibration_basis、recorded_source

---

## 20. 最后给一张知识地图

~~~text
线性代数
├─ 矩阵与矩阵连乘
├─ 逆矩阵
├─ 齐次坐标
├─ SVD
└─ 最小二乘
       │
       ▼
投影几何
├─ 针孔相机模型
├─ 内参与外参
├─ Homography
├─ DLT
└─ 地平线与投影无穷远
       │
       ▼
图像与数值计算
├─ 畸变模型
├─ reverse remap
├─ 双线性插值
├─ Jacobian 与局部放大
├─ 浮点/量化误差
└─ Q 格式定点数
       │
       ▼
工程交付
├─ LUT 与 sentinel
├─ metadata 与 coordinate space
├─ 来源追溯信息与基准指纹
├─ transaction / rollback
└─ 独立判定参照 / 验证
~~~

如果想继续系统学习，可以按这个顺序：

1. **线性代数课**：矩阵变换、秩、零空间、特征值与 SVD。目标是能读懂 A h = 0
   和“退化为什么意味着约束不够”。
2. **数字图像处理课**：采样、卷积、插值、混叠、图像金字塔。目标是理解 remap、
   降采样和“放大不能创造细节”。
3. **计算机视觉基础课**：针孔模型、相机标定、多视几何、单应、PnP。目标是把
   K、D、rvec/tvec、H 放进同一套投影语言。
4. **数值计算课**：条件数、最小二乘、浮点误差、优化。目标是理解为什么同样的公式，
   点分布不同会有截然不同的稳定性。
5. **嵌入式与软件工程**：定点数、内存布局、文件格式、哈希、原子写入与事务。目标是
   把“在电脑上算对一次”变成“设备上长期可靠地用对”。

最后把整条链再压缩成五句话：

1. 棋盘标定用已知几何反求 K、D，但质量取决于照片给了多少真实约束。
2. 去畸变是随半径变化的非线性映射，不是一张 3×3 矩阵。
3. 地面是平面，所以四对可靠点可以用 DLT 求 H0，再用参考原点、朝向、比例尺和锚点
   组成最终 H。
4. 实时渲染最适合使用“输出像素 → 原图采样坐标”的复合 reverse LUT。
5. 正确交付不只靠公式，还要靠坐标元数据、来源指纹、文件事务和独立验收共同守住。

读到这里，再回头看 IpmCalibrator.compose 或 build_composite_reverse_map，你看到的
应该不再是一串陌生矩阵，而是一条每一步都有坐标系、有单位、有理由的流水线。
