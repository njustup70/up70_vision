# up70-vision

UP70 战队视觉组仓库：把 2D 视觉链路跑通——**从图像输入到目标检出、位姿、绘制与落盘**，
为机械臂与 ROS 2 打底。能力三条：**判色**（HSV 判色 + 候选框筛选）、
**ArUco**（检测 → 位姿 → 实时循环）、**YOLO**（可选，ONNX + `cv2.dnn`，不需要 torch）。

输入源三选一：`camera`（海康 MV-CU013-A0UC，走 MVS SDK）/ `replay`（图片或视频）/ `synthetic`（自造图）。
**真机链路已跑通**（1280×1024、20 ms 曝光 28 fps）。

---

## 快速开始

统一用 Docker，不在本机装依赖。镜像底座 `ubuntu:22.04` + Python 3.10（对齐 ROS 2 Humble）。

```bash
# 1) 构建镜像（只需一次，约 2~3 分钟）
docker compose -f .devcontainer/docker-compose.yml build

# 2) 环境自检：末行应是「结果：全部通过」
docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision python -m vision doctor

# 3) OpenCV 冒烟：应看到「OpenCV 冒烟测试 -> PASS」
docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision python -m vision demo
```

**一条命令版**：`./demo.sh` 把上面三步再加一步「跑一批合成图」串起来，
结束后 `output/demo/` 里有 4 种产物；哪一步挂了它直接报出来。
跑完可以 `python -m vision clean` 清掉 `output/` 与 `__pycache__/`。

源码是 bind mount，改宿主机代码立即生效；只有改 `pyproject.toml` 才需要重新 build。
正文里的命令都只写 `python -m vision ...`，在容器外执行时加前缀
`docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision`。

---

## 从哪读起

- **要懂原理**（判色 / 位姿 / 标定 / YOLO / 契约）→ [`knowledge/docs/README.md`](knowledge/docs/README.md)（索引，标明哪篇该先读、哪篇是存档）
- **要接 ROS 2** → [`vision/docs/interface.md`](vision/docs/interface.md)（`Frame` / `Detection` / `ArucoMarker` / `MarkerPose` 的唯一契约来源）
- **要写代码** → [`knowledge/docs/视觉组协作规范.md`](knowledge/docs/视觉组协作规范.md)（6 条）与 [`knowledge/docs/代码注释规范.md`](knowledge/docs/代码注释规范.md)

---

## 真机跑起来

### 1. 接相机（三步，缺一步都打不开）

海康 MV-CU013-A0UC 走 **USB3 Vision（GenICam），不是 UVC**：宿主机没有 `/dev/video*`，
`cv2.VideoCapture` 打不开它，报错也不会提示原因。

1. 宿主机 udev 给权限：`/etc/udev/rules.d/99-hikrobot.rules` 写
   `SUBSYSTEM=="usb", ATTRS{idVendor}=="2bdf", MODE="0666", SYMLINK+="hik_camera"`，
   然后 `sudo udevadm control --reload-rules && sudo udevadm trigger`，再拔插相机。
2. MVS 的 deb 放进 `.devcontainer/pkgs/`，重新 `build` 镜像（deb 写不进 `pyproject.toml`）。
3. compose 已透传 `/dev/bus/usb`。相机是**独占资源**：MVS 客户端、调参窗口、本仓库三者互斥，打不开先关掉占用者。

### 2. 实时识别

容器里能开窗口，需要把 X11 挂进去（无显示器就加 `--headless`，见第 4 节）：

```bash
xhost +local:root
docker compose -f .devcontainer/docker-compose.yml run --rm \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix up70-vision \
  python -m vision live --set detector.name=aruco --set input.source=camera \
  --set input.camera.exposure_us=40000 --set input.camera.gain_db=15
```

窗口里按 `q` / `ESC` 退出。终端每 0.5 s 一行，**码增减或 ID 变化立即补一行**，无码也打
`markers=0`——"活着但没检测到"和"卡死了"要能分清。

### 3. 加位姿：画面多三根轴，终端多三个数

只差两个参数：`--set pose.enabled=true --set pose.marker_size_mm=82.5`。
边长必须**卡尺量黑框外沿**（不含白边）；填错不会让重投影误差变大，只让距离按比例错。

```bash
xhost +local:root
docker compose -f .devcontainer/docker-compose.yml run --rm \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix up70-vision \
  python -m vision live --set detector.name=aruco --set input.source=camera \
  --set pose.enabled=true --set pose.marker_size_mm=82.5 \
  --set input.camera.exposure_us=40000 --set input.camera.gain_db=15
```

```text
t=0.03s | frames=1 | fps=48.2 | markers=1 | id=0 center=(642.3, 532.5) d=317mm tilt=37.6 yaw=-144.0
```

真机实拍（82.5 mm 码，40 ms / 15 dB）：绿框是四角，三根轴贴在码面上，HUD 第四行是这一帧的
`d / tilt / yaw`。

![真机位姿：四角、三根坐标轴与 HUD 里的 d/tilt/yaw](knowledge/images/live_pose_demo.jpg)

- `d` 斜距（毫米）、`tilt` 码面与光轴夹角（0 = 正对）、`yaw` 面内转角
- 红/绿/蓝三根轴是码的 x/y/z；正视时蓝轴缩成一个点，这是对的
- **正视附近 `tilt` 不可信**：0° 附近弱可观（实测可散到 10°+），倾斜 15° 以上才降到 1° 内
- 没有内参时位姿自动关闭并打警告（`calibration.required=false`），设 `true` 则直接失败

### 4. 没有显示器：换成 `--headless --save-every 30 --limit 300`

命令同上，去掉 X11 相关参数即可；叠加图落在 `output/aruco/frame_*.png`。
批处理（读完再汇总）用 `python -m vision run --set input.source=camera --limit 10` → `output/demo/`。

---

## 标定（换相机或换分辨率时才做）

```bash
# 打印棋盘：出 PNG + PDF，打印时选「实际大小」
python lab/gen_chessboard.py --cols 9 --rows 6 --square-mm 15

# 现场采 15 张图并解算，结果写 configs/camera_calibration.yaml
python lab/calibrate_camera.py --capture 15 --board 9x6 --square-mm 23
```

**两个尺寸别混**（都"不报错只让数字错"）：

| 尺寸 | 值 | 用在哪 |
|---|---|---|
| 棋盘格方格边长 | **23 mm** | 标定 `--square-mm`（本机标称 15、实量 23；填错内参整体错） |
| ArUco 码边长 | **82.5 mm** | 位姿 `pose.marker_size_mm` / `--marker-mm`（填错距离按比例错） |

仓库强制带了一份内参快照（2026-10-19 标定，33 视图，RMS 0.473 px，1280×1024，fx≈1321.67），
新 clone 直接能跑位姿；重标定会覆盖同一路径，提交前确认是不是全队约定的那台相机。
细节见 `knowledge/docs/相机标定.md`。

---

## 配置

参数都在 `configs/default.yaml`，**判色阈值与判色门槛都在 `configs/color_thresholds.yaml`**
（调颜色只开这一个文件，理由见该文件的注释）。命令行用 `--set` 按点分路径覆盖；只能改已存在的字段，
名字写错启动即报错（返回 2）。启动时会把实际生效的配置整份打印一遍。

```bash
--set detector.name=aruco            # blob / stub_aruco / aruco / yolo
--set input.source=camera            # camera / replay / synthetic
--set input.camera.exposure_us=40000 # 相机曝光，微秒；动目标用 40 ms
--set input.camera.gain_db=15        # 增益，dB
--set pose.enabled=true              # 打开位姿估计（默认关闭）
--set pose.marker_size_mm=82.5       # 码边长，卡尺量黑框外沿
--set color.hit_ratio_min=0.6        # 判色命中率门槛（在阈值表里，也能这样临时盖）
```

曝光按场地重调：20 ms 太暗（灰度均值 20）、80 ms + 10 dB 可行但手举码会因拖影丢码
（曝光期内移动约 20 px ≈ 一个格子），动目标用 **40 ms**；照度不够优先加光 / 开大光圈。

---

## 已知限制

- **位姿精度只在合成图上验收**：真机只验证到链路（1280×1024、20 ms 28 fps），距离/倾角逐项精度
  尚未用卷尺对表，现场需先解决光照（40 ms 下画面均值低于 30 时角点会飘）。
- **位姿只做单码相对位姿**：多码融合、场地坐标系下的绝对位姿都没做。
- **正视时倾角不可信**：`tilt` 在 0° 附近弱可观（可散到 10°+），15° 以上才准。
- **相机需要 MVS SDK**：镜像不带 deb，需手动投放后重建；MVS 客户端与本仓库互斥。
- **ROS 2 未接入**：入口按「薄壳 + 单一实现」留好位置，接入时复用 `vision.cli`。
- **判色阈值依赖光照**：换场地要重调 `color_thresholds.yaml`，`hsv-tuner` 为此准备。

---

## 附录 A · 目录导览

```text
vision/                      # 运行时代码：要 import、要测、要评审、有架构约束
├── __init__.py              包声明与依赖方向说明
├── __main__.py              进程入口：python -m vision <子命令>（6 条）
├── api.py                   库门面：detect(image_bgr, detector)
├── types.py                 唯一共用件：Frame / Detection / 几何函数
├── logging.py               日志配置（进程内调一次）
├── cli/                     命令适配层：parsers / run / live / output（落盘 + HUD）
├── config/                  YAML 加载、字段校验、--set 覆盖
├── sources/                 输入源：camera（MVS）/ replay / synthetic + read_image_bgr
├── aruco/                   功能包：detector / pose / calibration / draw / types
├── yolo/                    功能包：detector / candidates / color_check / pipeline / stub / draw
└── depth/                   深度相机占位（只有 README，没有 .py；D11）

lab/                        一次性工具：check_aruco / pose_check / calibrate / gen_*（零架构约束）
configs/                    场地参数（lab 与 vision 都读）
knowledge/                  知识资产：docs（文档）、calibration（棋盘）、models（权重）、images（截图）
tests/                      回归测试 107 项 + 测试夹具（assets/aruco）
knowledge/docs/                       契约 / 规范 / 教程 / 进度 / archive
.devcontainer/              Dockerfile、CPU/GPU compose、pkgs
data/  output/              数据集与运行产物（不进版本库）
```

依赖方向是硬约束，下层不知道上层存在：

```text
config ┐
       ├─> 根级算法模块（types/color/pipeline/calibration/pose/detectors）─> io ─> viz ─> cli ─> __main__.py
image_io/logging ┘                                                                          └─> api.py
```

根级算法模块不 import ROS、不读文件、不开窗口；`viz/` 只画和写文件，不改检测结果；
`cli/` 是编排层，`api.py` 与 `__main__.py` 是它之上的两个门面（库调用 / 命令行）。

---

## 附录 B · 命令全表

**主入口只有 6 条**（`knowledge/docs/迁移计划书` D3）：`python -m vision` 不带参数就是这个表。
一年用不到三次的功能全部移到 `lab/`，直接 `python lab/xxx.py` 跑。

| 主入口 | 干什么 |
|---|---|
| `run` | 判色/检测批处理：读完所有帧再汇总 |
| `live` | 实时 ArUco：逐帧检测 + 窗口/落盘 |
| `doctor` | 环境自检（实现 `lab/check_env.py`） |
| `demo` | OpenCV 冒烟，末行 `PASS`（实现 `lab/demo.py`） |
| `hsv-tuner` | 拖滑条调 HSV 阈值（实现 `lab/hsv_tuner.py`） |
| `clean` | 清 `output/` 内容（留 `.gitkeep`）与所有 `__pycache__/` |

`lab/` 下的工具（不进主入口）：

| 工具 | 干什么 |
|---|---|
| `lab/check_aruco.py` | 按真值核对 ArUco 图组：ID / 四角 / 中心 |
| `lab/pose_check/`（`python -m lab.pose_check`） | `--dir` 逐图核对 / `--sweep` 批量扫描 |
| `lab/calibrate_camera.py` | 相机标定：采棋盘图 → 内参 + 畸变 |
| `lab/gen_chessboard.py` | 生成棋盘 PNG + PDF |
| `lab/gen_aruco.py` | 生成 ArUco 测试图组 + 真值 JSON |
| `lab/pose_check/gen_scenes.py` | 生成带真值的位姿测试图 |
| `lab/export_yolo.py` | 把 YOLO `.pt` 导出成 ONNX + 类名表 |
| `lab/mvs_color_measure.py` | 海康 MVS 相机实时判色窗口 |
| `lab/gen_html.py` | 教程 md → 单文件 html（需要 pandoc） |

旧的 `scripts/*.py` 薄壳已删除；CI 会拦住 `scripts/` 目录再次出现。

**测试与自检**（容器内）：

| 项 | 命令 | 期望 |
|---|---|---|
| 环境 / 冒烟 | `python -m vision doctor`、`python -m vision demo` | 末行「结果：全部通过」/ `PASS` |
| ArUco 图组 | `python lab/check_aruco.py` | `通过 9/9`，中心 0.00 px，四角 ≤ 0.23 px |
| 真机链路 | `python -m vision run --set input.source=camera --limit 10` | 取到 10 帧、`output/demo/` 落盘、退出码 0 |
| 回归测试 | `for t in tests/test_*.py; do python "$t"; done`（或 `pytest tests/ -q`） | **103 项全过** |

约定：

- 镜像里没有 pytest，`tests/` 每个文件都能用 `python` 直接跑（内置独立运行器）；CI 里用 `pytest`。
- `test_config` 里那条绝对路径断言已按平台规整（`os.path.normpath`），Windows 与容器都是 103/103。
- 断言必须带具体数字（如 `9/9`、`≤ 0.23 px`），不写 `assert x > 0`。
- 改代码前先跑 `python -m vision doctor`；碰 ArUco 跑 `python lab/check_aruco.py`，碰位姿跑一次真机实时。
- 合成图上的精度基线（82.5 mm 码：距离 0.02%–0.42%、倾角 15°–75° 误差 0.05–0.42°）用
  `python lab/pose_check/gen_scenes.py --suite` 造图后由
  `python -m lab.pose_check --dir <目录>`（逐图）或 `--sweep`（出 CSV/MD/8 张曲线）复现，口径与失败分析见 `knowledge/docs/位姿估计.md`。
- 在别的代码里用一次检测：`from vision.api import detect; detect(img, "aruco")`
  ——不必自己走配置加载与工厂。

---

## 附录 C · 阅读须知（新人 / AI）

下面这些是**生成物或本地素材**，不在版本库里，分析代码、翻目录时请直接忽略：

| 目录 | 是什么 | 怎么办 |
|---|---|---|
| `output/` | 运行产物：叠加图、`results.json`、`detections.csv`、位姿扫描曲线 | `python -m vision clean` 清掉 |
| `__pycache__/`、`*.pyc` | Python 字节码缓存 | 同上，一并清掉 |
| `data/` | 数据集、回放素材、标定原图（几十 MB 到几 GB） | 按 `data/README.md` 各自本地准备 |
| `knowledge/docs/*.html` | 教程的构建产物 | 要就 `python lab/gen_html.py` 自己生成 |
| `knowledge/models/*.onnx`、`*.pt` | 模型权重 | `python lab/export_yolo.py` 自己导出 |
| `configs/camera_calibration*.yaml` | 每台相机一份的内参 | 向做过标定的队友要，或自己标 |

`git ls-files` 里只有一百来个文件——**"仓库臃肿"的实感全部来自上面这些本地生成物**，
不是版本库本身。清理后目录会回到"刚 clone 完"的样子。
