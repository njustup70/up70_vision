# 项目进度

这份文档让每个成员随时知道：项目走到哪一步了、已经有什么、还缺什么、踩过哪些坑。

**维护约定**：阶段状态变化时更新本文；**新踩的坑进 `docs/archive/pitfalls.md`**（写清现象 / 原因 / 处理，协作规范第 5 条）；每阶段结束把已完成明细与历史段归档进 `docs/archive/`，本文只留当前进度与下一步，控制在 10 KB 以内。
提交信息只写结论（改了什么 / 怎么验），实测数据、过程与未验证项统一记在本文对应小节。

> 最后更新：2026-10-22
>
> 2026-10-22 拆分：第三节目录明细、第五节全部坑、第六节环境表已归档到 `docs/archive/`（仓库瘦身清单 P1-3），本文由 48 KB 降到 9 KB 以内。

---

## 一、当前阶段

**M2 核心算法、M3 回归测试、M6 相机接入均已实现；接口契约按 `docs/interface.md` 落地（含 ArUco 的 `ArucoMarker` 与 `stub_aruco`）；M6 只剩场地标定。**

- `vision/` 的根级算法模块（判色 `color.py` 与管线 `pipeline.py`、`Detection`/`ArucoMarker` 两种检出物）、`io/`（合成图 / 回放 / **真相机 MVS 源**）、`viz/`（画图与落盘）、`config/`（加载与校验）、`cli/`（编排，已拆成 parsers/run/live/report 四个文件）、`tools/`（生成器与自检工具）、`api.py`（库调用门面）均已实现
- 回归测试 **103 项**（判色/几何/配置 39 + ArUco 14 + 相机标定 12 + 位姿 13 + YOLO 20 + api 门面 5，全部通过）；容器与宿主 venv 都通过（宿主 apt 版 `check_env.py` 会报 4 项版本不符，属预期）
- 端到端自检：合成图 3 帧 × 6 块 = **18 个色块，实得 18 个、判色 18/18 全对、平均命中率 1.0**
- 真相机已可跑：`input.source=camera` 走厂商 **MVS SDK**（deb 放 `.devcontainer/pkgs/` 由 Dockerfile 装进镜像；compose 已透传 `/dev/bus/usb`）
- **2026-10-16 复核整改**：PR #3 的复核意见（针对 `a06df7f`）逐条处理 —— `--limit` 丢结果、相机半初始化句柄泄漏、白底白字标签、画框多 1px、中文滑条名、`writer.close` 只在成功路径；整改记录见第三节末
- **2026-10-16 真 ArUco 落地**（飞书任务一）：`detectors/aruco.py`（DICT_4X4_50，同图多码 / 旋转可识别）+ `tools/aruco/gen_scenes.py`（造测试图组 + 真值）+ `tools/aruco/check.py`（对真值自检）+ `assets/aruco/`（9 张图 + 真值）；按契约返回 `list[ArucoMarker]`，实测 **9/9 张通过、中心最大误差 0.00 px、四角 0.23 px**
- **2026-10-18 相机标定落地**（飞书「尝试做」第 1 条）：`calibration.py`（内参 + 畸变 + 去畸变，
  纯算法）+ `tools/calibration/calibrate.py`（采图 / 解算 / 写 YAML）+ `tools/calibration/gen_board.py`（造可打印板）
  + `docs/相机标定.md` + 9 项测试；合成数据反推实测：**fx 796.9（真值 800）、k1 −0.296（真值 −0.28）、RMS 0.54 px**
- **2026-10-18 位姿估计落地**（飞书任务单第 2、3 条）：`pose.py`（`solvePnP` + 重投影与前向性两条硬校验）
  + `viz.draw_pose_axes` + 实时入口按码输出 `d/tilt/yaw` + `tools/pose/gen_scenes.py`（带真值合成图）
  + `tools/pose/`（逐图核对）+ `pose-check --sweep`（七组扫描、CSV/MD/8 张图）+ 11 项测试。
  合成图（23 mm 码、真机内参 fx=1321）实测：**距离误差 0.11–1.1%（300–1500 mm）、2000 mm 处 3.9%**；
  倾角在真值 ≥15° 时误差 0.05–0.71°，**但接近正视时 4–16°（该自由度弱可观，不是 bug）**；
  过曝 3.0 倍未检出，运动模糊 9 px 起距离偏 9.9%，640×480 下每格 5.5 px 仍可用
- **2026-10-22 结构与入口精简**：入口从 19 个收敛为 `python -m vision`（**14 个功能子命令 + clean**），
  `scripts/` 整体删除；包内去掉 `core/`/`utils/` 中间层（算法模块平铺包根）、`detectors/` 提升为顶层、
  `cli/report.py` 合并进 `run.py`/`live.py`、`tools/` 按功能分七组；回归仍 **103 项**
- **还没做**：场地 HSV 阈值重调（等场地与物块）；位姿的**真机**抽测（需内参文件与实测码边长）

---

## 二、里程碑总览

| 里程碑 | 状态 | 内容 | 阻塞条件 |
|---|---|---|---|
| M0 仓库初始化 | ✅ 完成 | 目录规范、工程元文件、配置层参数面 | — |
| M1 环境标准化 | ✅ 完成 | Docker 化：镜像 + compose + devcontainer；镜像已真实 build 成功 | — |
| M2 核心算法实现 | ✅ 完成（摄像头源除外） | 根级算法模块判色与管线、`io/` 合成图与回放、`viz/` 落盘、`config/` 加载、`cli/` 编排 | 摄像头接入见 M6 |
| M3 回归测试 | ✅ 完成 | `tests/`：**36 项**（2026-10-16 复核整改后由 28 项增至 36）、断言带具体数字、双环境通过 | — |
| M7 契约落地 | ✅ 完成 | `docs/interface.md` 定稿（`Frame`/`Detection`/`ArucoMarker`/`Detector`/`FrameSource`）；新增 `stub_aruco.py` 按契约返回假 `ArucoMarker`；真实现 `aruco.py` 由搭档交付 | 真实现见 ArUco 分支 |
| M8 真 ArUco（飞书任务一） | ✅ 完成 | `detectors/aruco.py` + `tools/aruco/gen_scenes.py` + `tools/aruco/check.py` + `assets/aruco/`（9 张图 + 真值）+ 14 项测试；按契约返回 `list[ArucoMarker]` | 位姿估计（内参 + solvePnP，附加题）顺延阶段二 |
| M9 相机标定（尝试做 1） | ✅ 完成 | `calibration.py` + `tools/calibration/calibrate.py` + `tools/calibration/gen_board.py` + `docs/相机标定.md` + 9 项测试；输出内参与畸变系数 YAML | 真机标定要打印一块棋盘（工具已就绪） |
| M10 位姿估计（任务单第 2、3 条） | ✅ 完成（合成数据） | `pose.py` + `viz.draw_pose_axes` + 实时入口输出 + `tools/pose/gen_scenes.py` / `pose_check.py` / `pose_sweep.py` + `docs/位姿估计.md` + 11 项测试 | 真机精度要等内参文件与实测码边长；合成图不含镜头畸变 |
| M4 ROS 2 对接 | ⬜ 阻塞 | 话题适配器 + 发布节点 + 自定义消息 | 机械臂组 / 电控组定话题名与消息字段 |
| M5 模型接入 | 🟡 接口已接（见 `docs/YOLO接入.md`） | YOLO 检测器（ONNX + cv2.dnn）+ 导出脚本 | **训练仍阻塞**：没有标注数据集、没有确定物块本体。预训练权重只认 COCO 80 类 |
| M6 真相机与场地标定 | 🔵 部分完成 | 相机接入（✅ 已实现）+ 相机标定 + HSV 阈值按场地重调 | 阈值重调与标定都要等**物块与场地确定**；相机侧已无阻塞 |

---

## 三、已完成（明细已归档）

逐条实测记录（每次跑了什么命令、得到什么数字）在 [`archive/milestones.md`](archive/milestones.md)。
里程碑的**当前状态**见上一节的表格；本节只留索引：
M0 仓库初始化、M1 环境标准化、M2 核心算法、M3 回归测试、M6 相机接入、
M8 真 ArUco、M9 相机标定、M10 位姿估计。

---

## 四、进行中 / 下一步

**已清的第 1 步：把相机接进来（M6，2026-10-14 完成）** —— 留档备查，方案与实测见第三节。
下面 5 条当时是待办，现已全部落地（`io/camera.py`、MVS 进镜像、compose 透传、文档纠错、`input.camera:` 配置面）：

1. `vision/io/camera.py`：实现 `FrameSource`（`read() -> Frame | None`），走 MVS `MV_CC_GetImageForBGR`（SDK 直接给 BGR，绕开 Bayer 命名坑）；**MVS 不是 pip 包**，属于环境改动
2. **MVS 怎么进镜像**（三选一，推荐 A）：
   - A：deb 放 `.devcontainer/pkgs/`（加进 `.gitignore`）+ Dockerfile `COPY` 后 `apt install`
   - B：镜像内下载——官方要登录，不可靠
   - C：挂载宿主机 `/opt/MVS` 进容器——省事但破坏"环境一律 Docker"
3. `.devcontainer/docker-compose.yml` 加 `devices: ["/dev/bus/usb:/dev/bus/usb"]`（USB3 Vision 走 libusb，**不是 `/dev/video0`**）；udev 规则必须在宿主机
4. 文档纠错：`README.md`（第 60 行附近）与 `vision/io/__init__.py` 都写的"摄像头（UVC）"，与事实不符
5. 配置面：`configs/default.yaml` 新增 `input.camera:`（index / 曝光 / 增益 / 超时）

**第 2 步：相机标定（像素 → 毫米）**
棋盘格标定拿到内参；可与老仓库 `vision_opencv-main/src/spear_vision/config/camera.yaml` 里那套（1280×1024、fx=1322.8、fy=1338.4、RMS 0.85px）相互对照。

**第 3 步：场地阈值重调**（等物块与场地确定）
用 `configs/color_thresholds.yaml` 逐色调；曝光是这颗相机的关键——现场照度不足时要么加光、要么开大光圈、要么接受低帧率与拖影。

**后续（阻塞中）**：M4 ROS 2 对接、M5 YOLO 接入。

**实现顺序参考（已完成部分）**：`logging.py` → `image_io.py` → `types.py` → `color.py` → `pipeline.py` → `detectors/base.py` → `blob.py` → `io/base.py` → `synthetic.py` → `replay.py` → `viz/draw.py` → `writer.py` → `config/schema.py` → `loader.py` → `cli.py` → `tests/`

---

## 五、踩过的坑（已归档，共 40 余条）

整篇已搬到 [`archive/pitfalls.md`](archive/pitfalls.md)：分环境类 / 代码类 / 工具与工程类三组，
每条都是「现象 / 原因 / 处理」三栏。**改代码前先扫一遍那一篇**——
里面多数是"不报错但结果不对"的坑，比报错难查得多。

新增的坑往那一篇末尾加（坑的编号连续，代码注释里的「坑 40」仍能对上）。

---

## 六、环境状态（已归档）

镜像底座、依赖版本、相机型号与 udev 规则、镜像体积的整张表在
[`archive/env-state.md`](archive/env-state.md)。要复现环境按那一份来；
日常怎么跑看仓库根目录的 `README.md`。

---

## 七、明确不做（及原因）

| 任务 | 阻塞原因 |
|---|---|
| ROS 2 节点与话题 | 话题名与消息字段取决于机械臂组、电控组需求，先写必然返工 |
| YOLO 接入与训练流水线 | 没有标注数据集，没有确定物块本体（尺寸/材质/颜色会影响输入分辨率与增广策略） |
| 工业相机接入 | ✅ **已完成**（M6）：型号 MV-CU013-A0UC，走厂商 MVS SDK + 容器透传 `/dev/bus/usb`，实测 28.1 fps @20ms；标定与阈值重调见第四节 |
| 场上光照标定 | 等物块与场地确定后才能调 HSV 阈值 |
