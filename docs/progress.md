# 项目进度

这份文档让每个成员随时知道：项目走到哪一步了、已经有什么、还缺什么、踩过哪些坑。

**维护约定**：阶段状态变化时更新本文；新踩的坑往第五节加一条（写清现象 / 原因 / 处理）；改动跟着当次提交一起进去。
提交信息只写结论（改了什么 / 怎么验），实测数据、过程与未验证项统一记在本文对应小节。

> 最后更新：2026-10-16

---

## 一、当前阶段

**M2 核心算法、M3 回归测试、M6 相机接入均已实现；接口契约按 `docs/interface.md` 落地（含 ArUco 的 `ArucoMarker` 与 `stub_aruco`）；M6 只剩场地标定。**

- `src/up70_vision/` 的 `core/`（判色与管线、`Detection`/`ArucoMarker` 两种检出物）、`io/`（合成图 / 回放 / **真相机 MVS 源**）、`viz/`（画图与落盘）、`config/`（加载与校验）、`cli.py`（编排）、`tools/hsv_tuner.py`（交互调阈值）均已实现
- 回归测试 **61 项**（判色/几何/配置 36 + ArUco 14 + 相机标定 11）；容器与宿主 venv 都通过（宿主 apt 版 `check_env.py` 会报 4 项版本不符，属预期）
- 端到端自检：合成图 3 帧 × 6 块 = **18 个色块，实得 18 个、判色 18/18 全对、平均命中率 1.0**
- 真相机已可跑：`input.source=camera` 走厂商 **MVS SDK**（deb 放 `.devcontainer/pkgs/` 由 Dockerfile 装进镜像；compose 已透传 `/dev/bus/usb`）
- **2026-10-16 复核整改**：PR #3 的复核意见（针对 `a06df7f`）逐条处理 —— `--limit` 丢结果、相机半初始化句柄泄漏、白底白字标签、画框多 1px、中文滑条名、`writer.close` 只在成功路径；整改记录见第三节末
- **2026-10-16 真 ArUco 落地**（飞书任务一）：`core/detectors/aruco.py`（DICT_4X4_50，同图多码 / 旋转可识别）+ `tools/gen_aruco.py`（造测试图组 + 真值）+ `tools/aruco_check.py`（对真值自检）+ `assets/aruco/`（9 张图 + 真值）；按契约返回 `list[ArucoMarker]`，实测 **9/9 张通过、中心最大误差 0.00 px、四角 0.23 px**
- **2026-10-18 相机标定落地**（飞书「尝试做」第 1 条）：`core/calibration.py`（内参 + 畸变 + 去畸变，
  纯算法）+ `tools/calibrate_camera.py`（采图 / 解算 / 写 YAML）+ `tools/gen_chessboard.py`（造可打印板）
  + `docs/相机标定.md` + 9 项测试；合成数据反推实测：**fx 796.9（真值 800）、k1 −0.296（真值 −0.28）、RMS 0.54 px**
- **还没做**：场地 HSV 阈值重调（等场地与物块）；位姿估计（`solvePnP`，下一步）

---

## 二、里程碑总览

| 里程碑 | 状态 | 内容 | 阻塞条件 |
|---|---|---|---|
| M0 仓库初始化 | ✅ 完成 | 目录规范、工程元文件、配置层参数面 | — |
| M1 环境标准化 | ✅ 完成 | Docker 化：镜像 + compose + devcontainer；镜像已真实 build 成功 | — |
| M2 核心算法实现 | ✅ 完成（摄像头源除外） | `core/` 判色与管线、`io/` 合成图与回放、`viz/` 落盘、`config/` 加载、`cli/` 编排 | 摄像头接入见 M6 |
| M3 回归测试 | ✅ 完成 | `tests/`：**36 项**（2026-10-16 复核整改后由 28 项增至 36）、断言带具体数字、双环境通过 | — |
| M7 契约落地 | ✅ 完成 | `docs/interface.md` 定稿（`Frame`/`Detection`/`ArucoMarker`/`Detector`/`FrameSource`）；新增 `stub_aruco.py` 按契约返回假 `ArucoMarker`；真实现 `aruco.py` 由搭档交付 | 真实现见 ArUco 分支 |
| M8 真 ArUco（飞书任务一） | ✅ 完成 | `core/detectors/aruco.py` + `tools/gen_aruco.py` + `tools/aruco_check.py` + `assets/aruco/`（9 张图 + 真值）+ 14 项测试；按契约返回 `list[ArucoMarker]` | 位姿估计（内参 + solvePnP，附加题）顺延阶段二 |
| M9 相机标定（尝试做 1） | ✅ 完成 | `core/calibration.py` + `tools/calibrate_camera.py` + `tools/gen_chessboard.py` + `docs/相机标定.md` + 9 项测试；输出内参与畸变系数 YAML | 真机标定要打印一块棋盘（工具已就绪） |
| M4 ROS 2 对接 | ⬜ 阻塞 | 话题适配器 + 发布节点 + 自定义消息 | 机械臂组 / 电控组定话题名与消息字段 |
| M5 模型接入 | ⬜ 阻塞 | YOLO 检测器 + 训练流水线 | 没有标注数据集、没有确定物块本体 |
| M6 真相机与场地标定 | 🔵 部分完成 | 相机接入（✅ 已实现）+ 相机标定 + HSV 阈值按场地重调 | 阈值重调与标定都要等**物块与场地确定**；相机侧已无阻塞 |

---

## 三、已完成

### M0 仓库初始化

- 分层骨架 `src/up70_vision/`：`config` / `core` / `core.detectors` / `io` / `tools` / `utils` / `viz` 共 8 个包。依赖方向单向（`utils`/`config` → `core` → `io` → `viz` → `cli` → `scripts`），每个 `__init__.py` 写明了该层放什么、不放什么、未来第一件事做什么
- 工程元文件：`pyproject.toml`（PEP 621 + src layout）、`requirements.txt`、`.gitignore`、`.gitattributes`、`.editorconfig`、`.github/PULL_REQUEST_TEMPLATE.md`
- 配置层目标参数面：`configs/default.yaml`、`configs/color_thresholds.yaml`

### M1 环境标准化

- `.devcontainer/Dockerfile`：底座 `ubuntu:22.04` + 系统 Python 3.10 + numpy / OpenCV / PyYAML，`pip install -e .` 可编辑安装
- `.devcontainer/docker-compose.yml`（CPU）与 `docker-compose-gpu.yml`（GPU 透传，用 `extends` 复用 CPU 定义，避免两份配置漂移）
- `.devcontainer/devcontainer.json`：VS Code Reopen in Container 一键进环境
- `.dockerignore`：排除 `.git`、缓存、运行产物、维护者个人笔记
- `scripts/demo.py`：OpenCV 冒烟测试（造图 → BGR2HSV → inRange → findContours → 叠加 → 写图）。**故意不 import `up70_vision`**，避免将来形成第二份「检测 → 判色」实现
- `scripts/check_env.py`：环境自检（Python 版本、opencv 发行包计数、依赖版本区间、图像读写往返、torch 可选提示），零 `src` 依赖

### M2 核心算法实现（本次提交）

按依赖顺序自底向上建，文件与职责：

| 文件 | 职责 |
|---|---|
| `utils/logging.py` | 日志配置（全包挂在 `up70_vision` 父 logger 下，进程调一次）；等级非法时报错并列出可用值 |
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
| `tests/test_config.py` | 9 | 默认值来自 YAML；8 类非法值都报 `ConfigError` 且消息含字段路径；`--set` 类型转换（float/int/bool/列表）；错键报错列出可用键；阈值表按配置文件目录解析；**`detector.name=stub_aruco` 的默认值与 4 类非法参数** |

| `tests/test_calibration.py` | 11 | 合成数据反推：已知 K/畸变投影出角点再标定，检查 fx/fy/cx/cy/k1 是否找回；板子参数、角点个数不符、去畸变、YAML 往返 |
| `tests/test_aruco.py` | 14 | **真 ArUco**：多码/旋转/无码图对真值（中心与真值完全相等、四角 ≤ 0.25 px）；角点顺序（上边倾角 0°/−30°/−60°）；45° 单码；契约形状（ArucoMarker / float32 / 四角均值）；坏输入被拒；目录里 9 张图逐张对真值 |

**合计 61 项**（36 + 14 + 11）。

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
| 相机源 | `src/up70_vision/io/camera.py` | 实现 `FrameSource`；走 MVS `MV_CC_GetImageForBGR`（SDK 直接给 BGR，绕开 Bayer 命名坑）。**MvImport 的 import 放在 `__init__` 里**，所以没装 MVS 的机器照样能 import 本包，只在真接相机时报一句带处理办法的人话 |
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
| 任务一.2 用 OpenCV 生成多个不同 ID 的码图 | `tools/gen_aruco.py`（`cv2.aruco.generateImageMarker`）-> `assets/aruco/` | ✅ 3 个单码 + 多码 2 张 + 旋转 2 张 + 无码 2 张 |
| 任务一.3 读取含码图片并识别 | `tools/aruco_check.py`、`tests/test_aruco.py`；命令行 `--set detector.name=aruco --set input.source=replay --set input.replay.path=<图片或目录>`（单张图片路径也能用） | ✅ |
| 任务一.4 取 ID / 四角像素坐标 / 中心点 | `core/detectors/aruco.py` 的 `detect() -> list[ArucoMarker]`，契约见 `docs/interface.md` 2.2 / 3.2 | ✅ |
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

---

## 四、进行中 / 下一步

**已清的第 1 步：把相机接进来（M6，2026-10-14 完成）** —— 留档备查，方案与实测见第三节。
下面 5 条当时是待办，现已全部落地（`io/camera.py`、MVS 进镜像、compose 透传、文档纠错、`input.camera:` 配置面）：

1. `src/up70_vision/io/camera.py`：实现 `FrameSource`（`read() -> Frame | None`），走 MVS `MV_CC_GetImageForBGR`（SDK 直接给 BGR，绕开 Bayer 命名坑）；**MVS 不是 pip 包**，属于环境改动
2. **MVS 怎么进镜像**（三选一，推荐 A）：
   - A：deb 放 `.devcontainer/pkgs/`（加进 `.gitignore`）+ Dockerfile `COPY` 后 `apt install`
   - B：镜像内下载——官方要登录，不可靠
   - C：挂载宿主机 `/opt/MVS` 进容器——省事但破坏"环境一律 Docker"
3. `.devcontainer/docker-compose.yml` 加 `devices: ["/dev/bus/usb:/dev/bus/usb"]`（USB3 Vision 走 libusb，**不是 `/dev/video0`**）；udev 规则必须在宿主机
4. 文档纠错：`README.md`（第 60 行附近）与 `src/up70_vision/io/__init__.py` 都写的"摄像头（UVC）"，与事实不符
5. 配置面：`configs/default.yaml` 新增 `input.camera:`（index / 曝光 / 增益 / 超时）

**第 2 步：相机标定（像素 → 毫米）**
棋盘格标定拿到内参；可与老仓库 `vision_opencv-main/src/spear_vision/config/camera.yaml` 里那套（1280×1024、fx=1322.8、fy=1338.4、RMS 0.85px）相互对照。

**第 3 步：场地阈值重调**（等物块与场地确定）
用 `configs/color_thresholds.yaml` 逐色调；曝光是这颗相机的关键——现场照度不足时要么加光、要么开大光圈、要么接受低帧率与拖影。

**后续（阻塞中）**：M4 ROS 2 对接、M5 YOLO 接入。

**实现顺序参考（已完成部分）**：`utils/logging.py` → `utils/image_io.py` → `core/types.py` → `core/color.py` → `core/pipeline.py` → `core/detectors/base.py` → `blob.py` → `io/base.py` → `synthetic.py` → `replay.py` → `viz/draw.py` → `writer.py` → `config/schema.py` → `loader.py` → `cli.py` → `tests/`

---

## 五、已知的坑

按「现象 / 原因 / 处理」记录，新踩到的往这里加。

### 环境类

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 1 | `ImportError: libGL.so.1: cannot open shared object file` | opencv-python 的非 headless wheel 需要 libGL，Ubuntu 默认不装 | `sudo apt-get install -y libgl1 libglib2.0-0t64`；镜像里已装好 |
| 2 | `cv2.imshow` 属性检查存在，调用时才报错 | 环境里同时有 `opencv-python` 与 `opencv-python-headless`（或 conda 的 opencv），两者顶层模块都是 `cv2`，谁后装谁覆盖 | 只保留一个：`pip uninstall -y opencv-python-headless`；`check_env.py` 有专项检查 |
| 3 | `CondaToSNonInteractiveError: Terms of Service have not been accepted` | 新版 Miniconda 的 `conda create` 要求先接受 Anaconda 服务条款 | 现已全面 Docker 化，本机不再需要 conda |
| 4 | `ModuleNotFoundError: No module named 'tomllib'` | `tomllib` 是 Python 3.11+ 才有 | 用 `setuptools.config.pyprojecttoml.read_configuration`，或第三方 `tomli` |
| 17 | 海康 USB3 相机插上后 `ls /dev/video*` 没有任何新设备 | 它是 **USB3 Vision（GenICam）**，不是 UVC：数据走 libusb 的 `/dev/bus/usb/...`，根本不产生 v4l2 节点 | 别用 `cv2.VideoCapture` / 别往 `video` 组上靠；用厂商 SDK（MVS）或 Aravis 这类 USB3 Vision 库 |
| 18 | 相机能被 Aravis 识别、参数可读写，但**一帧都取不到**（`timeout_pop_buffer` 立即返回 None） | 该相机 U3V 的 `SIRM_PAYLOAD_SIZE` 恒为 1 MB 且不随 `SIRM_REQ_PAYLOAD_SIZE` 更新（实测把 ROI 调小也一样）；Aravis 按 1 MB 组帧，与真实 1.25 MB 帧长不符，帧被丢弃。Aravis 0.8.20 与自建 0.8.33、官方 `arv-camera-test-0.8` 均失败 | **改用官方 MVS SDK**（实测 28 fps 正常出帧）；Aravis 路线已排除 |
| 19 | 装完 MVS 后新终端里 Python 才能 import 成功 | `MvImport` 通过环境变量 `MVCAM_COMMON_RUNENV` 定位 `.so`，变量由 MVS 装到 `~/.bashrc` | 新开终端；或在脚本里 `os.environ.setdefault("MVCAM_COMMON_RUNENV", "/opt/MVS/lib")` |
| 20 | 别的程序打不开相机（`LIBUSB_ERROR_BUSY`） | 相机是**独占**资源：MVS 客户端 / 探针 / 实时窗口同时只能开一个 | 先关掉占用者再开 |
| 21 | 用 `get_trigger_source()` 看到 `Line0` 就以为"卡在触发模式" | `TriggerSource` 是保留值，`TriggerMode=Off` 时它照样有值 | 判断是否触发模式要看 **`TriggerMode`**，不是 `TriggerSource` |
| 22 | 想按"依赖唯一真源是 pyproject.toml"把 MVS 写进依赖 | MVS 是 **deb 安装包**，不是 pip 包，`dependencies` 表达不了 | 环境类依赖走 `.devcontainer/Dockerfile`；README 该补一句例外说明 |
| 23 | 宿主直接跑 `check_env.py` 报 4 项版本不符（numpy/cv2/yaml） | 宿主用的是 apt 版（numpy 1.21.5 / cv2 4.5.4 / yaml 5.4.1），而 pyproject 钉的是 pip 版区间 | 这属**预期**：标准环境是容器。要在宿主验证就 `docker compose run --rm up70-vision`；本项目代码已实测在两种版本下都正常 |
| 24 | 容器里跑相机源，MVS 报"没找到相机"，但宿主 `lsusb` 明明看得到 | 容器默认看不到宿主 USB 总线 | compose 已配 `devices: ["/dev/bus/usb:/dev/bus/usb"]`；权限（udev 规则）必须在宿主机给 |
| 25 | 在受限开发沙箱 / 精简容器里 MVS 枚举不到相机，可 `lsusb` 又列得出设备 | 沙箱的 `/dev` 是精简 devfs（**没有 `/dev/bus/usb`**），而 `lsusb` 走 sysfs 照样列设备——两者结论会打架 | 判断"相机到底可用吗"必须在**能看到 `/dev/bus/usb` 的环境**里跑（容器加 `--device`，或宿主机终端），别信 `lsusb` 单方面 |
| 26 | 镜像里相机打不开：枚举不到设备 / 权限错 | `libMvCameraControl.so` 的直接依赖只有 glibc 与 libstdc++，**libusb 是运行时 dlopen 的**，而 `ubuntu:22.04` 基础镜像没装它 | Dockerfile 里显式 `apt-get install libusb-1.0-0`（与 MVS 同一层） |
| 27 | MVS 装不进依赖声明 | 它是 **deb 包**，不是 pip 包，`pyproject.toml` 的 `dependencies` 表达不了 | 走镜像层：deb 放 `.devcontainer/pkgs/`（已被 `.gitignore` 排除，184 MB 不进版本库），Dockerfile `COPY` + `apt-get install`；`check_env.py` 不检查它 |
| 28 | 手举 ArUco 码**晃动或快速移动时识别不出来**，静置却稳定 | **运动模糊**：80 ms 曝光的整个期间，码在画面里移动的距离都会被拖成一条影。手以 20 cm/s 挥动、1 m 距离时拖影约 20 px，而 4x4 码的单格也只有 15~20 px，格子边界直接糊掉（2026-10-17 实测）。帧率低（11.6 fps）是同一原因的另一种表现，**不是检测算法慢** | 降曝光优先于提增益：40 ms 下拖影减半、帧率上限 25 fps，实测静置仍稳定识别；增益只换噪声。**首选加光**（补光灯/台灯），既不丢帧率也不加噪声。这是物理限制，任何视觉系统快速运动都会丢，验收要求是"停下来能立刻恢复"，不是"每帧必检出" |
| 28 | 相机程序退出时打印 `*******XOpenDisplay Fail*******` | MVS 的库在收尾时会尝试连 X 显示，容器 / 无显示环境里必然失败 | 无害，可忽略：退出码仍为 0，取帧与落盘都不受影响 |
| 34 | `AttributeError: module 'cv2.aruco' has no attribute 'DetectorParameters_create'`（或 `detectMarkers` 用法对不上） | ArUco 自 OpenCV **4.7** 起进主包并改版 API：老教程的 `DetectorParameters_create()` + `cv2.aruco.detectMarkers(...)` 换成 `DetectorParameters()` + `cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)` | 只认新 API（`pyproject` 钉 `opencv-python>=4.10`）；**不要**为了迁就老教程装 `opencv-contrib-python`（会与 `opencv-python` 争 `cv2` 顶层模块）。`aruco.py` 在旧版本上抛一句带处理办法的 `RuntimeError`，`tests/test_aruco.py` 自动跳过 |

### 代码类

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 5 | `TypeError: can only concatenate list (not "tuple") to list` | `list + tuple` 在任何 Python 版本都非法（`check_env.py` 曾踩） | 用 `list + [0] * n`，或 `tuple(...) + (0,) * n` |
| 6 | `cv2.imwrite` 返回 `False` 但不抛异常 | OpenCV 少数"静默失败"接口之一 | 必须显式检查返回值，别假设成功；`utils/image_io.py` 已封装 |
| 7 | `cv2.imread` 读不到文件返回 `None` 不抛异常 | 同上 | 显式判空再往下走；`utils/image_io.py` 已封装 |
| 8 | 判色结果"看起来不太准"但不报错 | BGR 当 RGB 用不会抛异常，只会让红蓝全互换 | 图像变量一律带颜色空间后缀（`frame_bgr`）；判色前 `cvtColor(BGR2HSV)` |
| 9 | 红色判不出来 | HSV 的 H 范围是 0-179 不是 0-360；且色相在 0/179 处环绕，红色必须写两段阈值取并集 | 阈值表里红色写 `[0,10]` 与 `[170,179]` 两段 |
| 10 | 框内判色命中率偏低 | 判色分母用了 ROI 总像素，框里混入的地面背景稀释了命中率 | 分母用「有效像素」（饱和度/亮度达标的像素）；`core/color.py` 已按此实现 |
| 11 | `cv2.putText` 画出问号且不报错 | Hershey 矢量字体不含汉字字形 | 图上一律用 ASCII 英文颜色名（`viz/draw.py` 还会把非 ASCII 替换掉）；中文说明只进日志 |
| 30 | `--limit N` 一旦真正生效就丢结果：打印「❌ 一帧都没处理」并返回退出码 1，可帧确实处理了、图也落了盘 | `cli.py` 的 `on_result` 里额外 `raise StopIteration` 想断流，而 `process_frames` 是普通函数，异常直接冒泡到外层 `except StopIteration: results = []`，把算好的结果清空；限帧其实已由 `limited()` 生成器做过一遍 | 限帧只留 `limited()` 一处，删掉 `on_result` 里的 raise 与外层 except（复核意见一.1）。**触发条件是 `limit ≤ 可用帧数`**，所以相机那条无限流命令 `--limit 10` 必中 |
| 31 | 相机在某次失败后"打不开"了（`LIBUSB_ERROR_BUSY` / 枚举不到），要重启程序甚至重插 | `io/camera.py` 里 `_opened` 原本只在**全部成功**后才置位；`MV_CC_OpenDevice` 成功后若 `MV_CC_StartGrabbing` 失败就抛异常，此时设备已被**独占**打开，而 `close()` 开头 `if not self._opened: return` 直接返回 → 句柄泄漏 | `_opened` 语义改为「句柄已创建」（CreateHandle 成功即置位），构造函数里 `try: self._open() except: self.close(); raise`；`close()` 做成幂等，且 `CloseDevice` 只在设备确实打开过时调用（复核意见二.1 / 契约欠账二） |
| 32 | 叠加图上"什么都没画"：ID/颜色标签一个也看不见，左上角 HUD 也几乎读不出 | `put_text` 默认写**白字**、HUD 用亮黄字，落在**白底**素材上等于没画——不报错、就是看不见（与坑 11"画问号"同一类"看不出问题的问题"） | `put_text` 加 `outline=True`：先用黑色厚一档描边再写字，白底/黑底/彩色底都读得清（复核意见二.2）。教训：**画完要真的打开图看**，只看 `count/ids` 会以为画对了 |
| 35 | ArUco 四角误差稳定卡在 ~0.55 px，怎么调都下不去 | 测试图组的真值把码边界当成了「最外侧黑像素的**中心**」（`0` 与 `side-1`）；而检测器报的是黑框**最外沿**，正好差半个像素 | 真值改用像素边界 `[-0.5, side-0.5]`：误差立刻降到 0.14~0.23 px。教训：**真值必须和贴图用同一个仿射矩阵算**，别事后目测标点 |
| 36 | 用 `--limit 300` 测实时循环，实际只跑了 **3 帧**就结束 | `--limit` 只限制**循环消费**多少帧，供给端是 `input.source`：默认 `synthetic` 只配了 `num_frames: 3`，源自己 3 帧就结束了，`--limit` 根本轮不到 | 跑长序列要同时放大源帧数：`--set input.synthetic.num_frames=300`；相机源天然无限，不受影响。**验收第 6 条（无码不异常退出）必须跑满上百帧才有说服力**——3 帧只能证明"没崩"，证明不了"持续运行" |

### 工具与工程类

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 12 | `docker build` 报找不到 Dockerfile | compose 的 `dockerfile` 路径是**相对 `context`** 解析的，不是相对 compose 文件所在目录 | Dockerfile 移目录后必须同步改这行 |
| 13 | `.dockerignore` 形同虚设 | Docker 规定它必须在构建上下文根部，放子目录会失效 | 保持在仓库根 |
| 14 | `fish: Unsupported use of '='` | fish 不支持 `VAR=value cmd` 语法 | 用 `set VAR value`；`for` 循环里也别用 `$(...)` 赋值 |
| 15 | `git add -A` 把个人笔记提交了 | 只想本地排除却写进了 `.gitignore`——它会向全队广播规则，且队友写同名文件会被静默忽略 | 个人文件用 `.git/info/exclude`，团队规则才用 `.gitignore` |
| 16 | 删了模块但 `__pycache__` 里还有它的 `.pyc` | Python 不会自动清理已删源文件的字节码缓存 | `find src -name __pycache__ -type d -exec rm -rf {} +` |
| 17 | `*.egg-info/SOURCES.txt` 列着已删文件 | `pip install -e .` 生成的元数据不会随源码变更自动同步 | 删掉 `src/up70_vision.egg-info` 后重装；该目录已被 gitignore |
| 33 | `cv2.createTrackbar` 报 `NULL window handler`（或滑条看不见） | 滑条名 / 窗口标题传了中文：OpenCV 的 highgui 在某些后端上不接受非 ASCII 名字 | 滑条名与窗口标题一律 ASCII（`tools/mvs_color_measure.py` 原来有个「面积下限」，已改 `min_area`，复核意见二.4） |
| 37 | `python -m up70_vision.tools.aruco_check` 报 `ModuleNotFoundError: up70_vision` | 在宿主 venv 里跑但没 `pip install -e .` | 走薄入口 `python scripts/check_aruco.py`（自带 `src` 路径兜底），或 `PYTHONPATH=src python -m up70_vision.tools.aruco_check` |
| 38 | `docker compose build` 重建镜像很慢（几分钟到十几分钟，几乎全卡在 `pip` 下载） | Dockerfile 默认 `PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/`，**2026-10-17 实测只有 304 KB/s**：`numpy`（16.8 MB）用了 56 s，`opencv-python`（**77.4 MB**）按这个速度要 4 分钟以上；`apt` 那层走 Ubuntu 官方源，国内同样慢 | **换清华源重建**：`docker compose -f .devcontainer/docker-compose.yml build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`（README 第 91 行给的是中科大源，同样可用）。注意两种改动的代价完全不同：改这个 build-arg 只让 pip 层失效重下、前面 apt/MVS 层仍走缓存；**改 Dockerfile 里的 apt 层会让它后面所有层重跑**——所以新增系统库请追加到 Dockerfile 最后一层 |
| 39 | 容器里 `cv2.imshow` 报 `Could not load the Qt platform plugin "xcb" ... even though it was found`，而 `--headless` 落盘那条路完全正常 | `opencv-python` 的 GUI 走 Qt，`libqxcb.so` 运行时 dlopen `libSM.so.6` 与 `libICE.so.6`，`ubuntu:22.04` 基础镜像里没有；**报错完全不提缺哪个库**，极易误判成"OpenCV 没编 GUI 支持"而去重装 opencv | 镜像补 `apt-get install -y --no-install-recommends libsm6 libice6`（已加进 Dockerfile 最后一层）；宿主侧还需 `xhost +local:root`、`-e DISPLAY`、挂载 `/tmp/.X11-unix`。排查命令：`ldd $CV2PATH/qt/plugins/platforms/libqxcb.so \| grep "not found"` |
| 40 | 标定出来的**距离**整体偏（我们这台偏 53%），但内参与畸变看着完全正常 | 打印机把标称 15 mm 的格子打成了 **23 mm**；`--square-mm` 填的是标称值。内参不受板子尺度影响（尺度信息被位姿吸收），所以从 RMS 上完全看不出来，只有量过才知道 | **打印完用尺子量一个格子**；工具现在把 `--square-mm` 改成**必填**，不给就直接报错（示例：标称 15 mm、实量 23 mm 就填 23） |
| 41 | 标定 RMS 3~4 px、`fx` 偏 10% 以上，但每张图都「成功找到 54 个角点」 | 板子离太远，135 mm 的棋盘在图上只有 ~120 px，54 个角点挤在一起，亚像素精化失效，检出的点已不是格子阵（实测：角点间距 13/16/5/27 px 乱跳） | 让板子占画面 **1/3 以上**：`tvec.z` 从 1000 mm 改到 220~620 mm 后，同一套流程 RMS 从 3.8 px 降到 0.39 px |
| 42 | 畸变系数 `k2/k3` 跑来跑去、每次标定差别很大 | 纸卷边 / 板子没贴平 / 强反光：镜头看到的是「弯的板」，程序只能把这份弯曲也学进畸变系数 | 棋盘贴硬纸板、用哑光纸、光源偏离正面；`k3` 不稳时可以 `cv2.CALIB_FIX_K3` 固定它 |
| 43 | 标定结果离谱：`fx` 与 `fy` 差 7%、主点偏 200 px、`k2=0.98` `k3=-1.38`，但每张图都「成功找到 54 个角点」 | 快速连拍收进了 7 张**几乎同一姿势**的图（中心差 ≤12 px、格距差 0.1 px）：重复观测把焦距与畸变的耦合解当成真解 | 采图时加「太像就不收」的闸（`DEDUP_*` 阈值），重复姿势会被提示 `too similar to #N - change pose`；已收的图可用 `--images` 重算 |
| 44 | 标定 RMS 1.9 px、单张误差 5.2 px，但参数看着正常 | 手持拍摄时**运动模糊** + 没对实焦（实测锐度 Laplacian 只有 87~152，清晰图通常 300+），角点抖动直接进重投影误差 | 捕图加「站稳才拍」（连续 3 帧姿态稳定才收）、窗口显示 `sharp` 值；标定后自动**剔离群重标**；板子贴硬纸板、摆好停半秒 |

> 注：上表编号 17 与「环境类」新增的 17~23 重复，下次整理时统一重排。

---

### 2026-10-16 复核整改记录（PR #3 复核意见 → 本分支）

复核对象是 `a06df7f`（当时推送的 PR head）。逐条落地情况：

| 复核条目 | 结论 |
|---|---|
| 一.1 `--limit` 丢结果返回 1（必须改） | ✅ 已修，见坑 30；实测 `--limit 3` 退出码 0 |
| 一.2 描述与推送的代码对不上（必须改） | ✅ 描述按最终状态重写；并明确 PR head 以推送后的实际提交为准 |
| 二.1 `io/camera.py` 半初始化句柄泄漏 | ✅ 已修，见坑 31 |
| 二.2 `viz/draw.py` 白底白字 | ✅ 已修，见坑 32；用纯白图 + stub 复现并看图确认 |
| 二.3 `viz/draw.py` 画框比半开上界多 1px | ✅ 改为 `(x2-1, y2-1)` |
| 二.4 `tools/mvs_color_measure.py` 中文滑条名 | ✅ 改为 ASCII `min_area`，见坑 33 |
| 二.5 `writer.close(summary)` 只在成功路径 | ✅ 移到 `finally`，中途出错也能落盘 |
| 二.6 `raise StopIteration` 作控制流 | ✅ 随一.1 一并清掉 |
| 三.1 测试项数各处不一致（27/28） | ✅ 统一为 **36**（复核整改后由 28 增至 36），README / 本文 / 双环境表同步 |
| 三.2 `progress.md` 仍写 `camera` 抛 `NotImplementedError` | ✅ 已改（相机走 `io/camera.py`，SDK 缺失时构造期报 `RuntimeError`） |
| 三.3 第四节仍把 M6 已完成的事列为待办 | ✅ 已把第 1 步标为已清 |
| 三.4 Dockerfile 注释里残留 `io/camera_mvs.py` | ✅ 改为 `io/camera.py` |
| 三.5 `io/__init__.py` 写"依赖 config"，与契约第 5 节矛盾 | ✅ 已改（契约欠账三） |
| 三.6 `configs/README.md` 仍写"src 还是空骨架" | ✅ 已改 |
| 三.7 `configs/README.md` 的 `hsv_tuner` 示例参数不对 | ✅ 改为真实的 `--colors red` / `--source synthetic` |
| 三.8 `configs/default.yaml` 注释说"改了不会有任何效果" | ✅ 已改 |
| 四（相机相关没能验证的部分） | ⏳ 仍需在**接了相机 + 装了 MVS + 透传 `/dev/bus/usb`** 的环境里实跑；本机没有相机，只能读代码 |
| 契约（`docs/interface.md` 新版） | ✅ 已按搭档那版落地：新增 `ArucoMarker` 与 `stub_aruco`，判色族 `Detection` 保留；三条欠账照登记（其中欠账二已修，欠账一按契约"要改先问人"**没动**） |

---

## 六、环境状态

| 项 | 值 |
|---|---|
| 团队标准环境 | Docker 镜像：`ubuntu:22.04` + 系统 Python 3.10；镜像名 `up70-vision:dev`，**已 build 成功**（862 MB） |
| 镜像内容 | numpy / opencv-python / PyYAML，`pip install -e .`；实测 cv2 4.14.0 / numpy 2.2.6 / yaml 6.0.3 |
| 底座选择理由 | 战队标准发行版是 22.04；ROS 2 Humble 官方只支持 22.04 且自带 Python 3.10 |
| 代码生效方式 | compose 把仓库 bind mount 到 `/opt/up70_vision`，改宿主机代码容器里立即生效；**注意镜像内容是 build 时的快照**，跑新代码要么重新 build、要么用 bind mount |
| 本机开发环境 | Ubuntu 22.04.5 LTS（与镜像底座一致）；宿主 apt 版 cv2 4.5.4 / numpy 1.21.5 / PyYAML 5.4.1 |
| Docker | `/usr/bin/docker`，Compose v5.5.1，免 sudo |
| 相机 | 海康 **MV-CU013-A0UC**：1280×1024、BayerGB8、USB3 Vision、全局快门、C 口（镜头约 6mm）；序列号 DA4976553 |
| 相机 SDK | 官方 **MVS 5.1.0**（`libMvCameraControl` 4.8.2.1，SDK 版本 0x04080202）。宿主机装在 `/opt/MVS`，安装包备份在 `~/MVS_installer/`；容器里由 Dockerfile 从 `.devcontainer/pkgs/*.deb` 装入，同一层还装 `libusb-1.0-0`（MVS 运行时 dlopen 它） |
| 相机权限 | 宿主机 `/etc/udev/rules.d/99-hikrobot.rules`：`SUBSYSTEM=="usb", ATTRS{idVendor}=="2bdf", MODE="0666", SYMLINK+="hik_camera"`（udev 必须在宿主机配，容器内改不了） |
| 容器设备 | compose 已配 `devices: ["/dev/bus/usb:/dev/bus/usb"]` —— 传 USB 总线（USB3 Vision 走 libusb），**不是 `/dev/video0`** |
| 镜像体积 | 不含 MVS：~862 MB；含 MVS（相机开发需要）：**约 2.36 GB**。不接相机的人不放 deb 即可，`build` 会自动跳过（见坑 27） |

---

## 七、明确不做（及原因）

| 任务 | 阻塞原因 |
|---|---|
| ROS 2 节点与话题 | 话题名与消息字段取决于机械臂组、电控组需求，先写必然返工 |
| YOLO 接入与训练流水线 | 没有标注数据集，没有确定物块本体（尺寸/材质/颜色会影响输入分辨率与增广策略） |
| 工业相机接入 | ✅ **已完成**（M6）：型号 MV-CU013-A0UC，走厂商 MVS SDK + 容器透传 `/dev/bus/usb`，实测 28.1 fps @20ms；标定与阈值重调见第四节 |
| 场上光照标定 | 等物块与场地确定后才能调 HSV 阈值 |
