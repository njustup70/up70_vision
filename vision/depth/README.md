# `depth/`：深度相机 -> 抓取目标

纯几何抓取检测：给一张深度图和内参，找出桌面上凸起的物块，输出「吸附点 + 法向 + 可吸性分数」。**不依赖 YOLO，也不依赖彩色图。**

## 文件

| 文件 | 放什么 |
|---|---|
| `types.py` | `Intrinsics` / `SuctionParams` / `GraspTarget`。3D 类型只在这里，与根级 `types.py` 互不 import |
| `primitives.py` | 与吸盘无关的基本运算：反投影、体素降采样、RANSAC 平面、深度图连通域 |
| `core.py` | 抓取业务：顶面拟合、可吸性打分、编排、门面 `detect_grasp_targets()` |
| `timing.py` | RGB-D 整数时间窗口与双流覆盖检查；ROS 外壳与离线工具共用，不读文件、不依赖 ROS |
| `surface.py` | 框内平面候选、可见重心、面内尺寸及对应诊断绘制，独立于顶面抓取 |
| `rgbd.py` | 彩色到对齐深度网格的校正、映射缓存及尺寸/光学 frame 检查 |
| `__init__.py` | 对外只暴露 `detect_grasp_targets` / `analyze` 和三个类型 |

## 文件归属判据

只有一条：**描述这段代码时，需不需要提到「吸盘 / 抓取 / 吸附面」？**

| 不需要 → `primitives.py` | 需要 → `core.py` |
|---|---|
| `backproject`：像素 + 深度 → 三维点 | `fit_top_face`：一团点 → **物块顶面的吸附点** |
| `voxel_downsample`：一堆点 → 更少的点 | `score_suction`：**吸盘能不能吸住** |
| `fit_plane_ransac`：一堆点 → 平面方程 | `analyze` / `detect_grasp_targets`：→ 抓取目标 |
| `label_clusters`：深度图 → 每个像素属于第几团 | |

这条判据的意义是防止 `core.py` 继续膨胀。反投影、降采样、平面拟合、连通域这四件事
将来做深度-彩色对齐、手眼标定、多目标分割时都要再用，它们不该被绑在"抓取"里。
`fit_top_face` / `score_suction` 虽然也在算几何，但它们的存在理由就是吸盘，所以留在 core。

读文件、解析 bag、真值对比、命令行参数一律在 `lab/`；ROS 消息转换一律在
`ros2_ws/src/vision_depth/`。

## 对外接口

```python
from vision.depth import Intrinsics, detect_grasp_targets

k = Intrinsics(fx=409.5, fy=409.5, cx=424.33, cy=241.33, width=848, height=480)
targets = detect_grasp_targets(depth_mm, k)      # depth_mm: uint16，单位毫米
for t in targets:                                # 按可吸性分数降序
    print(t.to_dict())
```

调试用 `analyze()`，它会额外返回平面方程、每个簇的点和限制项，能分清「分割错了」还是「打分了」。

## 不 import ROS

`rclpy` 只能 apt 装、pip 装不了，混进来会让 `pip install -e .` 直接失败。ROS 外壳在 `ros2_ws/src/vision_depth/`，那边只做收消息、调这里、发消息。

`tests/test_architecture.py` 有断言守着这条，不靠人记。

## 参数从哪来

`SuctionParams` 的默认值不是拍的：

- `plane_thresh_mm = 6.0`：A4 实测墙面/桌面本身有 2~5mm 起伏，按 3σ（0.5m 处约 1mm）取会把真实桌面点判成外点
- `leaf_mm = 4.0`：比实测噪声（0.4~1.7mm）大一个量级，不会把噪声当结构
- 可吸性五条：计划书 3.3 的初值，仍需实测调整

## 现在的位置

后续改进路线见 [深度相机识别改进方案](docs/深度相机识别改进方案.md)，涵盖真实数据结论、2D 与深度融合、视觉侧职责及接口验收；其中待实施内容不代表当前已实现。

侧面离线工具与实时窗口现共用纯计算接口 `vision.depth.surface` 和 `vision.depth.rgbd`；原 `lab.side_rgbd_measure` 入口保留。实时 ROS 外壳不导入 lab，不发布抓取话题，启动与坐标口径见 [实时侧面查看器](#实时侧面查看器)。这不改变纯深度 `fuse_node` 的输入或输出。

同步 RGB-D 的采集与静态像素对齐冒烟检查单独走一条线，计划与门槛见
[同步 RGB-D 采集与对齐改进计划](docs/同步RGB-D采集与对齐改进计划.md)。主要入口：

```bash
ros2 launch vision_depth rgbd_capture.launch.py mode:=aligned   # raw / sync / aligned
ros2 run vision_depth rgbd_capture --ros-args -p output_dir:=... -p duration_s:=10.0 \
  -p session_config:=...
python -m lab.rgbd_check inspect|analyze|annotate ...
```

新增 `summarize` 汇总三场景、`compare-replay` 比较回放重录的 header/内容/配对、
`grid-check` 用棋盘格核查彩色与对齐深度是否同一像素网格。
实际覆盖、几何前提、窗口复查按键及旧包更正见
[2026-10-28 验收修正说明](docs/RGB-D验收修正与补采说明-20261028.md)。
新录包的独立复算和空间边界限制见 [重采核查](docs/RGB-D重采核查-20261028.md)。

报告落在 `output/rgbd/<session>/`（`report.html`、`pairs.csv`、`summary.json`、
`spatial_edges.csv`、`overlays/`、`annotations.json`），本阶段默认不入库。
注意：本阶段的「对齐深度」不接入 `fuse_node`——纯深度基线先用 raw 话题。

算法先在 `lab/proto_depth.py` 里跑通并调好参数，再提炼到这里（ROS2 接入方案 10.2 的四步落地）。`lab/` 那个脚本保留作为离线调试工具，读 npy/bag 看中间结果；ROS 侧的 `fuse_node` 调的是这里。

进度、实测数字、未完成项与踩过的坑见 [`docs/progress.md`](docs/progress.md)。改参数或做精度测试前先看一遍，尤其是「未完成与下一步」和「踩过的坑」两节。

## 已知的坑

- **深度是 z 深度，不是欧氏距离。** 反投影是 `x=(u-cx)z/fx` 的直接代入，误当欧氏距离会让点云边缘整体外扩。
- **ROI 裁剪后主点必须跟着平移**，否则反投影整体错位，表现为「物块位置偏了几厘米」，很难联想到是内参没对齐。
- **RANSAC 阈值单位是毫米。** 按米填（0.006）会让每个点都算外点，报的是「找不到桌面」。
- **平面法向方向无约定**，`n` 和 `-n` 都合法。模块内部统一成「朝相机」，算夹角仍取绝对值。
- **顶面拟合不能拿整簇。** 物块是立体的，整簇拟合出的是斜平面，中心还落在方块内部。只取离桌面最远的 20%。
- **聚类不是真正的欧式聚类。** 利用深度图是组织化点云这一特性，在深度跳变处切断连通域，结果等价且快两个量级（容器里没有 scipy，手写 KD 树太慢）。

## 实时侧面查看器

活文档，2026-10-30。窗口显示 YOLO 框内的平面候选及**可见平面内点重心**，不发布机械臂测量或抓取话题。一个框包含多个合格平面时显示 AMBIGUOUS，没有中心点；唯一平面也仅为 CANDIDATE，不保证它属于一个箱子的侧面。

实时外壳只保留三个模块：`rgbd_side_viewer.py` 管节点与主线程窗口，`side_pipeline.py` 管图对解码、测量、绘制和快照，`side_sync.py` 管配对与单槽后台任务。共享平面计算及其诊断绘制放在 `surface.py`，像素网格校正单独放在 `rgbd.py`；离线工具复用它们，不复制算法或依赖 ROS。

### 环境与启动

使用能导入 ROS 2 Jazzy、项目 `vision` 包、NumPy 和 OpenCV >=4.10 的同一个 Python。系统 OpenCV 4.6 不满足项目要求；本轮隔离验证采用 OpenCV 4.11。现有 2D Docker 镜像没有 ROS，不直接用它运行此查看器。

首次准备环境示例（在仓库根执行，虚拟环境在用户目录）：

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 -m venv --system-site-packages ~/.venvs/up70-rgbd
source ~/.venvs/up70-rgbd/bin/activate
python -m pip install -e .
cd ros2_ws
python -m colcon build --packages-select vision_depth --symlink-install
source install/setup.bash
cd ..
ros2 launch vision_depth rgbd_side_viewer.launch.py \
  weights:=$PWD/knowledge/models/yolov8n_mix2.onnx
```

后续终端仍需 source ROS、该虚拟环境和 `ros2_ws/install/setup.bash`，从仓库根启动。`ros2 run` 可执行文件的 Python 应与构建环境一致。权重和同名 `.names.json` 留在本地，不入 Git；没有文件就报错，不自动下载其他模型。默认加载本机的 mix2：0=block、1=top，640 输入，置信度 0.45，NMS IoU 0.45，保留全部类别。模型类名不证明平面的物理含义。

launch 默认包含现有 aligned 相机启动：848×480@30、帧同步、SW D2C 到 COLOR、毫米深度、仅 SHM，**不会启动录包器**。已有相机占用时用 `start_camera:=false`，并保证该相机确实发布对齐深度。查看器不能仅凭话题名证明驱动开启了 D2C 或毫米换算。

```bash
# 已有相机或 bag 回放；只启动查看器。
ros2 launch vision_depth rgbd_side_viewer.launch.py start_camera:=false

# 无窗口的定时验证；max_runtime_s 为秒，0.0 表示持续运行。
ros2 launch vision_depth rgbd_side_viewer.launch.py \
  start_camera:=false headless:=true max_runtime_s:=60.0
```

参数来源为 `config/rgbd_side_viewer.yaml`；显式 launch 参数覆盖 YAML，未指定时保留 YAML。`config_file` 指向查看器配置，`camera_config` 指向既有相机配置。改源码配置后重新构建，确保安装副本同步。

### 同步、时效与失败状态

订阅两路图像和两路 CameraInfo，图像回调只缓存，工作线程解码/推理/拟合，OpenCV GUI 在主线程。色彩编码 bgr8/rgb8，深度编码 16UC1/mono16 且驱动必须已换算为毫米；解码处理行填充和大小端。两路图像与 CameraInfo 的尺寸/frame 必须一致；仅支持 plumb_bob 彩色及零畸变对齐深度，不支持 CameraInfo 的裁剪/合并像素配置。

- 默认配对容差 ±10 ms，每流最多 10 帧，最多等待 50 ms 后取最近配对，等距选较早深度。深度不复用，顺序不倒退。
- 零时间戳、重复、溢出和未配对分别计数；时间倒退或内参变化使旧缓存和执行中的旧结果失效。
- 仅一个执行任务和一个待处理槽；新任务替换过时的待处理图对，记录 ProcessDrop，不不断积压。
- 每次窗口刷新前先有界处理就绪回调（最多 32 次、额外不超过约 4 ms），避免每次只接一条消息造成四路输入积压；单个回调仍不做推理。
- 结果使用自己的那张校正图绘制。ReceiveAge 是较早一张图本地收到后经过的单调时钟时间，**不是曝光到屏幕的延迟**。
- ReceiveAge 超过默认 500 ms 时显示 STALE，撤去框、平面、中心与坐标，仅保留对应原图；不会把旧框叠到最新彩色图。
- 无框显示 NO_DETECTION；点不足或退化为 NO_PLANE；多平面为 AMBIGUOUS；错误框为 INVALID_BOX。一个框失败不影响其他框。

`target_classes: [-1]` 为全部类别，或填写非负编号；`class_names: ['']` 表示自动读取完整类名表。`max_detections` 为单帧检测数量上限。当前只支持 8/16/32 三尺度的密集输出 `(1,4+nc,N)` 或 `(1,N,4+nc)`，N 按输入边长计算（640 时为 8400）。六列 NMS 结果 `(1,300,6)` 即使与两类模型通道数相同也会被拒绝；其他导出格式需适配后再用。

### 看窗口与保存证据

- 窗口从左到右显示融合图、与其同一图对的对齐深度伪彩色图和信息区；深度图黑色为无效值、蓝色偏近、红色偏远，色标固定 0–2000 mm，超过上限饱和。它没有插值补洞。按 D 可隐藏或恢复深度图；过期图会标 STALE，不显示旧的框和中心。
- 粗绿框加黑色描边：检测区域；模型类名、置信度和从零开始的框号按当前帧显示，框号不是跟踪 ID。
- 半透明黄色：平面内点，可按 M 开关；多个平面不会被自动选成一个中心。
- 洋红十字配黑白描边：唯一候选内点的三维重心投影，不是检测框中心像素的单点深度。
- CENTROID X/Y/Z：光学坐标毫米，X 向右、Y 向下、Z 向前；Z 与光心到平面垂距是不同量。
- 输入 FPS 按最近接收间隔计算；Process FPS(avg) 是启动至今平均完成任务数，包含等待期。各帧显示推理/几何耗时与 PairDt。
- Q/Esc 或关闭窗口退出；S 保存当前显示对应图对的诊断快照。过期快照会标为 stale；双图开启时 overlay.png 包含两张图，快照记录 depth_visible，原始深度始终保留在 pair.npz。

每次运行排他新建 `output/realtime_side/run_<时间>_<随机号>/`。退出保存 `summary.json`：同步/处理计数、错误/过期数、各耗时 P50/P95 和模型 SHA-256。统计历史最多保留最近 2000 项，计数为整个运行期。

快照位于运行目录的 snapshots：原始 BGR/深度 `pair.npz`、校正图、叠加图、候选掩码，以及 CameraInfo、原始整数时间戳、参数、模型标识和状态。用户标注与复核字段默认 false，按 S 不会生成真值。重心可能因检测框、遮挡和缺失深度变化，显示稳定也不代表绝对精度通过。

### bag 回放复现

回放端与查看器必须使用相同 ROS_DOMAIN_ID、RMW 和 SHM 配置。以下在两个已经准备好环境的终端执行相同 export，然后分别启动查看器和回放：

```bash
export ROS_DOMAIN_ID=73
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/ros2_ws/src/vision_depth/config/fastdds_shm_only.xml
export OPENBLAS_NUM_THREADS=1

# 终端 A
ros2 launch vision_depth rgbd_side_viewer.launch.py \
  start_camera:=false headless:=true max_runtime_s:=60.0

# 终端 B：读取原包，不修改数据。
ros2 bag play bags/raw_036 --loop --disable-keyboard-controls
```

历史回放性能、窗口基准及验证范围统一见 [`docs/progress.md`](docs/progress.md)。回放和显示稳定不代表空间对齐、实例语义或绝对定位精度通过；当前模型仍可能将两个共面箱子框成一个目标。
