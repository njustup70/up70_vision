# 里程碑实测记录（存档）

> 本文件是 `knowledge/docs/progress.md` 第三节「已完成」在 2026-10-22 的归档（仓库瘦身 P1-3）。
> 里程碑的**当前状态**看 `knowledge/docs/progress.md` 第二节的表格；这里保留逐条实测数字与当时的命令，供复查「这个数是哪来的」。

---

### M0 仓库初始化

- 分层骨架 `vision/`：`config` / `core` / `core.detectors` / `io` / `tools` / `utils` / `viz` 共 8 个包。依赖方向单向（`utils`/`config` → `core` → `io` → `viz` → `cli` → `scripts`），每个 `__init__.py` 写明了该层放什么、不放什么、未来第一件事做什么
- 工程元文件：`pyproject.toml`（PEP 621 + src layout）、`requirements.txt`、`.gitignore`、`.gitattributes`、`.editorconfig`、`.github/PULL_REQUEST_TEMPLATE.md`
- 配置层目标参数面：`configs/default.yaml`、`configs/color_thresholds.yaml`

### M1 环境标准化

- `.devcontainer/Dockerfile`：底座 `ubuntu:22.04` + 系统 Python 3.10 + numpy / OpenCV / PyYAML，`pip install -e .` 可编辑安装
- `.devcontainer/docker-compose.yml`（CPU）与 `docker-compose-gpu.yml`（GPU 透传，用 `extends` 复用 CPU 定义，避免两份配置漂移）
- `.devcontainer/devcontainer.json`：VS Code Reopen in Container 一键进环境
- `.dockerignore`：排除 `.git`、缓存、运行产物、维护者个人笔记
- `scripts/demo.py`：OpenCV 冒烟测试（造图 → BGR2HSV → inRange → findContours → 叠加 → 写图）。**故意不 import `vision`**，避免将来形成第二份「检测 → 判色」实现
- `scripts/check_env.py`：环境自检（Python 版本、opencv 发行包计数、依赖版本区间、图像读写往返、torch 可选提示），零 `src` 依赖

### M2 核心算法实现（本次提交）

按依赖顺序自底向上建，文件与职责：

| 文件 | 职责 |
|---|---|
| `utils/logging.py` | 日志配置（全包挂在 `vision` 父 logger 下，进程调一次）；等级非法时报错并列出可用值 |
| `utils/image_io.py` | 图像读写：把 `cv2.imread` 返回 None、`cv2.imwrite` 返回 False 这两个"静默失败"接口管起来并抛明确异常 |
| `core/types.py` | `Frame` / `Detection` / **`ArucoMarker`** 三种数据结构、`iou`、`clip_bbox`、`bbox_from_xywh`、`is_marker`；bbox 一律绝对像素 + **半开上界**。`Detection` 是判色族（中心 = bbox 中心），`ArucoMarker` 是 ArUco（中心 = 四角算术平均），两者不混用 |
| `core/color.py` | HSV 判色：`parse_thresholds`（校验）、`mask_for_segments`（多段并集，红色跨 0 环绕）、`valid_pixel_mask`、`color_ratios`、`judge_color` |
| `core/detectors/base.py` | `Detector` 抽象基类 + `build_detector` 工厂（不 import config，避免包级循环依赖）；工厂认识 `blob` / `stub_aruco` |
| `core/detectors/blob.py` | 占位检测器：饱和度/亮度分割 + 形态学 + 轮廓提取；`class_name="unknown"`、`class_id=-1` |
| `core/detectors/stub_aruco.py` | 按契约返回**假** `ArucoMarker` 的 stub（`marker_id` 升序、`center_px` = 四角算术平均、float32）；⚠️ 每帧必须恰好调一次 `detect`（它靠内部帧计数器决定码转到哪） |
| `core/pipeline.py` | `ColorConfig`、`FrameResult`、`process_frame(s)`（检测→判色→与真值匹配计分）、`summarize`；判色只对 `Detection` 做，码跳过 |
| `io/base.py` | `FrameSource` 接口（`read() -> Frame \| None`）+ `build_source` 工厂；`camera` 走 `io/camera.py`（MVS SDK），SDK 缺失时在**构造期**报一句带处理办法的 `RuntimeError` |
| `io/synthetic.py` | 合成色块源：格子布局保证 6 块不重叠，带 ground truth，同 seed 逐像素可复现 |
| `io/replay.py` | 回放源：图片目录或视频文件，支持 `frame_stride` / `max_frames` |
| `viz/draw.py` | 画框与标签（**ASCII 英文颜色名**）、HUD、拼图；不改变检测结果 |
| `viz/writer.py` | 落盘：`frame_XXXX.png` / `results.json` / `detections.csv` / `summary.json` |
| `config/schema.py` | 配置 dataclass + 范围校验 + `ConfigError`（消息带字段路径与可用值）；不引 pydantic |
| `config/loader.py` | YAML 读取、`--set a.b=c` 点分覆盖（按现有值类型转换、错键报错）、阈值表按配置目录解析 |
| `cli.py` | 运行逻辑唯一实现处：读配置 → 建源/检测器 → 逐帧 → 汇总 → 自检判定 |
| `scripts/run.py` | 薄入口，只转交 argv |

**落地的硬约定**（README「团队约定」）：BGR + 颜色空间后缀命名；H 0-179；红色两段取并集；判色分母是**有效像素**而非 ROI 总像素；bbox 半开上界；阈值全部外置 YAML；`core/` 不 import ROS / 不读文件 / 不开窗口；`scripts/` 保持薄入口。

### M3 回归测试

| 文件 | 项数 | 覆盖 |
|---|---|---|
| `tests/test_types.py` | 9 | IoU 具体数值（同框 1.0；半重叠 50/150）；半开上界与面积/中心；裁剪；`to_dict` 可 JSON 序列化；**`ArucoMarker` 的契约**（float32 归一、四角均值 ≠ bbox 中心的具体数值、3 类坏输入被拒） |
| `tests/test_color.py` | 7 | 阈值校验的 5 类非法输入；红色在 H=178 与 H=5 都命中、H=90 不命中；**有效像素分母**（一半黑：有效 50，命中 50 → 命中率 1.0 而非 0.5）；有效像素过少/命中率不足两种拒绝原因 |
| `tests/test_pipeline.py` | 11 | 合成源同 seed 逐像素可复现；每帧 6 块且颜色不重复；blob 检出 6 个；`match_to_truth` 具体数值；**端到端 18 个色块、判色 18/18**；配置校验；**stub_aruco 契约与运动**（三个码中心 x = 16/160/304、每帧转 12° 的具体坐标）；**管线对码跳过判色**（`color_counts` 为空、`uncertain` 为 0） |
| `tests/test_config.py` | 12 | 默认值来自 YAML；8 类非法值都报 `ConfigError` 且消息含字段路径；`--set` 类型转换（float/int/bool/列表）；错键报错列出可用键；阈值表按配置文件目录解析；**`detector.name=stub_aruco` 的默认值与 4 类非法参数** |

| `tests/test_calibration.py` | 12 | 合成数据反推：已知 K/畸变投影出角点再标定，检查 fx/fy/cx/cy/k1 是否找回；板子参数、角点个数不符、去畸变、YAML 往返；换分辨率时内参等比缩放（视场角不变）；打印件的 PNG DPI 与 PDF 物理尺寸 |
| `tests/test_aruco.py` | 14 | **真 ArUco**：多码/旋转/无码图对真值（中心与真值完全相等、四角 ≤ 0.25 px）；角点顺序（上边倾角 0°/−30°/−60°）；45° 单码；契约形状（ArucoMarker / float32 / 四角均值）；坏输入被拒；目录里 9 张图逐张对真值 |
| `tests/test_pose.py` | 13 | **位姿**：已知 (R,t) 反解精度、尺度与边长一致性、滚转只改 yaw、相邻角点交换与共线角点被拒、180° 循环错位只由 z 符号暴露、完全对称输入不泄漏 NaN、歧义门槛由参数决定、去畸变角点配 `dist=None` |

**合计 98 项**（39 + 14 + 12 + 13 + 20）。

**双环境验证记录（2026-10-16，复核整改后复测）**

| 环境 | cv2 / numpy | `check_env.py` | `demo.py` | 50 项测试 | 端到端自检 |
|---|---|---|---|---|---|
| 容器 `up70-vision:dev`（标准环境） | 4.14.0 / 2.2.6 | 结果：全部通过 | PASS | **36/36**（ArUco 14 项见 M8） | 18/18，判色全对 |
| 宿主 Ubuntu 22.04（apt） | 4.5.4 / 1.21.5 | 报 4 项版本不符（见坑 23） | PASS | 36/36 | 18/18，判色全对 |

> 镜像里**没有装 pytest**，所以 4 个测试文件都同时支持 `python tests/test_xxx.py` 独立运行（内置 `_run_standalone`）。加 pytest 属于改依赖，留待确认。

### M6 相机接入（已实现）

**接入链路**（四件套）：

| 环节 | 落点 | 说明 |
|---|---|---|
| 相机源 | `vision/io/camera.py` | 实现 `FrameSource`；走 MVS `MV_CC_GetImageForBGR`（SDK 直接给 BGR，绕开 Bayer 命名坑）。**MvImport 的 import 放在 `__init__` 里**，所以没装 MVS 的机器照样能 import 本包，只在真接相机时报一句带处理办法的人话 |
| SDK 进镜像 | `.devcontainer/pkgs/` + `Dockerfile` | MVS 是 deb 包、写不进 `pyproject.toml`，由 Dockerfile 单独一层安装；同时装 `libusb-1.0-0`（MVS 的库**运行时 dlopen** 它，基础镜像没有）。没放 deb 也能 build 成功，只是跳过相机 |
| 设备透传 | `.devcontainer/docker-compose.yml` | `devices: ["/dev/bus/usb:/dev/bus/usb"]`。传的是 USB 总线，**不是 `/dev/video0`** |
| 文档纠错 | `README.md`、`io/__init__.py` | 原文写的"摄像头（UVC）／用户要在 `video` 组／传 `--device=/dev/video0`"与事实不符，已改成 USB3 Vision + 宿主机 udev 规则 + MVS |

**配置面**：`configs/default.yaml` 新增 `input.camera:`（`index` / `exposure_us` / `gain_db` / `timeout_ms` / `max_frames`），默认 80 ms + 10 dB —— 本机实测能看清桌面的值。

**跑法**（容器内；相机要插在宿主机上）：

```bash
docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision \
  python scripts/run.py --set input.source=camera --limit 10
```

**容器内实测（2026-10-14）**：重建镜像后（MVS 与 libusb 进镜像、compose 透传 `/dev/bus/usb`），
`input.source=camera` 在容器内跑通 3 帧并正常落盘（叠加图 / JSON / CSV），图上 HUD 显示
`camera:mvs:MV-CU013-A0UC`。当次取景对着明亮窗口：**34.3% 像素 >250（过曝）**、饱和度均值 26（画面基本无彩色），
于是 9~10 个候选框全部判成 `unknown` —— 这属**预期**：默认阈值是按合成色块定的，
真实场地必须先用 `tools/hsv_tuner.py` 重调（并先把曝光降下来）。
退出时 MVS 会打一行 `*******XOpenDisplay Fail*******`，无显示环境下的正常现象，不影响结果（坑 28）。
另外 MVS 会在**运行目录**下自动生成 `MvSdkLog/` 日志目录，已加进 `.gitignore`（跑完可随手删）。

**为什么必须用 MVS**：Aravis 0.8.20 与自建 0.8.33 都取不到帧、官方 `arv-camera-test-0.8` 同样失败（坑 18）；换 MVS 立刻正常 —— **28.1 fps @ 20 ms 曝光**、7.3 fps @ 80 ms，曝光/增益可读写（20 ms/0 dB → 灰度均值 20；80 ms/10 dB → 180；200 ms/16 dB → 244）。

### M6 相机摸底（实测数据，保留备查）

- **相机**：海康 MV-CU013-A0UC；1280×1024、BayerGB8、USB3 Vision（GenICam）、全局快门、C 口（由内参反推镜头约 6mm）
- **Aravis 路线不可用**（详见坑 17、18）：Aravis 0.8.20 与自建 0.8.33 都取不到帧，官方 `arv-camera-test-0.8` 同样失败
- **MVS 路线可用**：官方 SDK 5.1.0（`libMvCameraControl` 4.8.2.1）实测 **28.1 fps @ 20ms 曝光**、7.3 fps @ 80ms；曝光/增益可读写（20ms/0dB → 均值 20；80ms/10dB → 均值 180；200ms/16dB → 244）
- **产出**：`~/vision_work/` 下的 Aravis 诊断脚本（`hik_probe*.py`、`hik_diag.py`）、MVS 取帧探针与实时判色工具（两者都在仓库外，不进版本库）；`~/MVS_installer/` 留了 MVS 的 deb
- **待办**：见第四节第 1 条（MVS 进镜像 + 设备透传 + 文档纠错）

### 本地验证记录（2026-10-14，M1 遗留项已清）

- `check_env.py` → 容器内**全部通过**；`demo.py` → PASS，退出码 0
- 镜像 `up70-vision:dev` **已真实 build 成功**（862 MB）
- 36 项回归测试在宿主与容器双环境全部通过；端到端自检 18/18
- **2026-10-16 补 ArUco 14 项** -> 合计 50 项；`aruco_check` 9/9 张、四角最大误差 0.23 px（见 M8）

### M8 真 ArUco 检测核心（飞书任务一）

对应飞书《图像处理预备任务》里属于**任务一**的那几栏，也对应仓库任务单「搭档 Owner：检测核心」。

| 飞书原文 | 落点 | 状态 |
|---|---|---|
| 任务一.2 用 OpenCV 生成多个不同 ID 的码图 | `tools/gen_aruco.py`（`cv2.aruco.generateImageMarker`）-> `tests/assets/aruco/` | ✅ 3 个单码 + 多码 2 张 + 旋转 2 张 + 无码 2 张 |
| 任务一.3 读取含码图片并识别 | `tools/aruco_check.py`、`tests/test_aruco.py`；命令行 `--set detector.name=aruco --set input.source=replay --set input.replay.path=<图片或目录>`（单张图片路径也能用） | ✅ |
| 任务一.4 取 ID / 四角像素坐标 / 中心点 | `core/detectors/aruco.py` 的 `detect() -> list[ArucoMarker]`，契约见 `vision/docs/interface.md` 2.2 / 3.2 | ✅ |
| 任务一.5 画边框 + 标注 ID 和中心点 | `viz/draw.py`（Aruco 走 `corners_px` 画四边形，旋转的码也贴合；标签 `id=N`；中心红十字） | ✅ |

**自检命令与数字**（协作规范第 4 条要求"具体数字"）：

```bash
python scripts/check_aruco.py            # 薄入口，自带 src 路径兜底
# 期望：通过 9/9；中心最大误差 0.00 px；四角最大误差 0.23 px
```

整条管线也跑过（真检测器 + 回放图组）：4 帧共 9 个码、id `[0..5]`、退出码 0。

**真值是怎么算准的**：`gen_aruco.py` 贴图时用的仿射矩阵**就是**算真值四角与中心的那个矩阵，
所以误差只来自插值（实测 ≤ 0.23 px）。同一参数重跑**逐字节一致**（`diff -r` 验过）。

### M9 相机标定（飞书「尝试做」第 1 条）

**为什么做**：任务一的角点、任务二的框都还是"像素"。要拿到毫米和位姿，先得把相机内参与畸变解出来。

| 文件 | 作用 |
|---|---|
| `core/calibration.py` | 纯算法：`BoardSpec` / `CameraIntrinsics` / `find_board_corners()` / `calibrate()` / `undistort_image()` / `undistort_points()`。不读文件、不开窗口（契约第 5 节） |
| `tools/gen_chessboard.py` | 按「毫米 + DPI」画可打印棋盘（内角点个数 + 静区白边），并打印"打印后必须量边长"的提示 |
| `tools/calibrate_camera.py` | 两种入口：`--images DIR`（对现成图标定）与 `--capture N`（现场采图）；结果写 `configs/camera_calibration.yaml`，`--from` 可读回复核 |
| `tests/test_calibration.py` | 9 项：用**已知 K 与畸变**投影出角点再反推，检查能不能把答案找回来 |

**合成数据反推的实测数字**（答案已知，所以能判断对错）：

| 场景 | 反解 fx / fy | 反解 cx / cy | 反解 k1 | RMS |
|---|---|---|---|---|
| 纯透视（无畸变）30 张 | 797.6 / 797.5 | 639.9 / 400.1 | −0.012 | 0.394 px |
| 带 k1=−0.28 的桶形畸变 30 张 | 796.9 / 796.7 | 640.2 / 400.8 | **−0.296** | 0.539 px |

真值是 `fx=fy=800、cx=640、cy=400、k1=−0.28`。工具链三条通路都跑通：`--images`、`--capture`（用回放图组
模拟现场采图）、`--from`（读回 YAML）。

**2026-10-18 真机标定（MV-CU013-A0UC，23 张手持，格子实测 23 mm）**：

| 项 | 值 |
|---|---|
| 内参 | `fx=1321.0`、`fy=1324.6`、`cx=642.8`、`cy=532.3` |
| 畸变 | `k1=-0.0588`、`k2=+0.0845`、`p1/p2≈0`、`k3=0`（默认固定） |
| 重投影 RMS | **0.507 px**（23 张剔离群后 19 张；最初 15 张是 1.87 px） |
| 留出验证 | 奇偶两折交叉：**0.453 / 0.470 px**，两折 fx 差 0.43% → 没过拟合 |
| 与老仓库对照 | 老仓库同型号 `fx=1322.8 / fy=1338.4`，我们 `1328.1 / 1329.4`，**差 0.4%** |
| 结论 | `fx≈fy`（0.28%）、主点接近中心 → **可用**；要"好"（<0.5 px）需重新对焦并补四角姿势 |

采图过程中补了三道工程闸（都写进工具了）：**站稳才拍**（连续 3 帧姿态稳定）、**太像不收**
（位置/格距/角度接近已收的就不收）、**剔离群重标**（按单张 RMS 挑掉坏图）。原因见坑 43/44。
