# up70-vision

UP70 战队视觉组仓库：把 2D 视觉链路跑通——**从图像输入到目标检出、位姿、绘制与落盘**，
为机械臂与 ROS 2 打底。

| 能力 | 做什么 |
|---|---|
| 判色 | HSV 判色 + 候选框筛选，给出色块位置 / 颜色 / 命中率 |
| ArUco | 检测 → 位姿（距离 / 倾角 / 面内角）→ 实时循环 → 终端与 HUD 报数 |
| YOLO（可选） | ONNX + `cv2.dnn` 推理，不需要 torch；`--set detector.name=yolo` |

输入源三选一：`camera`（海康 MV-CU013-A0UC，走 MVS SDK）/ `replay`（图片或视频）/ `synthetic`（自造图）。
**真机链路已跑通**（1280×1024、20 ms 曝光 28 fps），下文以真机为主。

---

## 快速开始

统一用 Docker，不在本机装依赖。镜像底座 `ubuntu:22.04` + Python 3.10（对齐 ROS 2 Humble）。

```bash
# 1) 构建镜像（只需一次，约 2~3 分钟）
docker compose -f .devcontainer/docker-compose.yml build

# 2) 环境自检：末行应是「结果：全部通过」
docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision python scripts/check_env.py

# 3) OpenCV 冒烟：应看到「OpenCV 冒烟测试 -> PASS」
docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision python scripts/demo.py
```

源码是 bind mount，改宿主机代码立即生效；只有改 `pyproject.toml` 才需要重新 build。
下文命令都在容器内执行，容器外只需加前缀 `docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision`。

---

## 目录导览

```text
up70_vision/
├── src/up70_vision/
│   ├── core/            纯算法层：不读文件、不开窗口
│   │   ├── detectors/   blob 判色 / stub_aruco 假码 / aruco 真码 / yolo
│   │   ├── pose.py      solvePnP + 两条硬校验（重投影 RMS、z 前向性）
│   │   ├── calibration.py 内参、去畸变、按分辨率等比缩放
│   │   └── types.py     Detection / ArucoMarker / MarkerPose
│   ├── io/              输入源：camera（MVS）/ replay / synthetic
│   ├── viz/             画框、坐标轴、HUD、落盘（PNG / JSON / CSV）
│   ├── config/          YAML 加载、字段校验、--set 覆盖
│   ├── utils/           图像读写、日志、控制台编码
│   ├── tools/           gen_* / *_check / calibrate_camera / hsv_tuner / mvs_color_measure
│   └── cli.py           唯一业务实现：run() 与 run_aruco_live()
├── scripts/             薄入口（11 个，只转交 argv）
├── configs/             default.yaml、color_thresholds.yaml、内参快照
├── assets/              aruco/（9 张真值图）、calib/（棋盘 PNG+PDF）、models/、readme/
├── tests/               回归测试 98 项，可 python tests/test_xxx.py 独立跑
├── docs/                进度 / 契约 / 三个规范 / 标定、位姿与 YOLO 说明 / 教程
├── .devcontainer/       Dockerfile、CPU/GPU compose、pkgs/
├── data/                数据集与回放素材（不进版本库）
└── output/              运行产物：demo/、aruco/、pose_*/（不进版本库）
```

依赖方向是硬约束，下层不知道上层存在：

```text
utils ┐
      ├─> core ─> io ─> viz ─> cli ─> scripts/
config ┘
```

`core/` 不 import ROS、不读文件、不开窗口；`viz/` 只画和写文件，不改检测结果；`scripts/` 保持薄入口。

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
  python scripts/run_aruco_live.py --set detector.name=aruco --set input.source=camera \
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
  python scripts/run_aruco_live.py --set detector.name=aruco --set input.source=camera \
  --set pose.enabled=true --set pose.marker_size_mm=82.5 \
  --set input.camera.exposure_us=40000 --set input.camera.gain_db=15
```

```text
t=0.03s | frames=1 | fps=48.2 | markers=1 | id=0 center=(642.3, 532.5) d=317mm tilt=37.6 yaw=-144.0
```

真机实拍（82.5 mm 码，40 ms / 15 dB）：绿框是四角，三根轴贴在码面上，HUD 第四行是这一帧的
`d / tilt / yaw`。

![真机位姿：四角、三根坐标轴与 HUD 里的 d/tilt/yaw](assets/readme/live_pose_demo.jpg)

- `d` 斜距（毫米）、`tilt` 码面与光轴夹角（0 = 正对）、`yaw` 面内转角
- 红/绿/蓝三根轴是码的 x/y/z；正视时蓝轴缩成一个点，这是对的
- **正视附近 `tilt` 不可信**：0° 附近弱可观（实测可散到 10°+），倾斜 15° 以上才降到 1° 内
- 没有内参时位姿自动关闭并打警告（`calibration.required=false`），设 `true` 则直接失败

### 4. 没有显示器：换成 `--headless --save-every 30 --limit 300`

命令同上，去掉 X11 相关参数即可；叠加图落在 `output/aruco/frame_*.png`。
批处理（读完再汇总）用 `python scripts/run.py --set input.source=camera --limit 10` → `output/demo/`。

---

## 标定（换相机或换分辨率时才做）

```bash
# 打印棋盘：出 PNG + PDF，打印时选「实际大小」
python scripts/gen_chessboard.py --cols 9 --rows 6 --square-mm 15

# 现场采 15 张图并解算，结果写 configs/camera_calibration.yaml
python scripts/calibrate_camera.py --capture 15 --board 9x6 --square-mm 23
```

**两个尺寸别混**（都"不报错只让数字错"）：

| 尺寸 | 值 | 用在哪 |
|---|---|---|
| 棋盘格方格边长 | **23 mm** | 标定 `--square-mm`（本机标称 15、实量 23；填错内参整体错） |
| ArUco 码边长 | **82.5 mm** | 位姿 `pose.marker_size_mm` / `--marker-mm`（填错距离按比例错） |

仓库强制带了一份内参快照（2026-10-19 标定，33 视图，RMS 0.473 px，1280×1024，fx≈1321.67），
新 clone 直接能跑位姿；重标定会覆盖同一路径，提交前确认是不是全队约定的那台相机。

---

## 配置

参数都在 `configs/default.yaml`，命令行用 `--set` 按点分路径覆盖；只能改已存在的字段，
名字写错启动即报错（返回 2）。启动时会把实际生效的配置整份打印一遍。

```bash
--set detector.name=aruco            # blob / stub_aruco / aruco / yolo
--set input.source=camera            # camera / replay / synthetic
--set input.camera.exposure_us=40000 # 相机曝光，微秒；动目标用 40 ms
--set input.camera.gain_db=15        # 增益，dB
--set pose.enabled=true              # 打开位姿估计（默认关闭）
--set pose.marker_size_mm=82.5       # 码边长，卡尺量黑框外沿
```

曝光按场地重调：20 ms 太暗（灰度均值 20）、80 ms + 10 dB 可行但手举码会因拖影丢码
（曝光期内移动约 20 px ≈ 一个格子），动目标用 **40 ms**；照度不够优先加光 / 开大光圈。

---

## 测试与自检

| 项 | 命令（容器内） | 期望 |
|---|---|---|
| 环境 / 冒烟 | `python scripts/check_env.py`、`python scripts/demo.py` | 末行「结果：全部通过」/ `PASS` |
| ArUco 图组 | `python scripts/check_aruco.py` | `通过 9/9`，中心 0.00 px，四角 ≤ 0.23 px |
| 真机链路 | `python scripts/run.py --set input.source=camera --limit 10` | 取到 10 帧、`output/demo/` 落盘、退出码 0 |
| 回归测试 | `for t in tests/test_*.py; do python "$t"; done` | **98 项：97 通过 / 1 失败**（`test_config` 的 POSIX 路径断言，`origin/main` 上就有，与业务无关） |

约定：

- 镜像里没有 pytest，`tests/` 每个文件都能用 `python` 直接跑（内置独立运行器）。
- 断言必须带具体数字（如 `9/9`、`≤ 0.23 px`），不写 `assert x > 0`。
- 改 `src/` 前先跑 `check_env.py` 与 `demo.py`；碰 ArUco 跑 `check_aruco.py`，碰位姿跑一次真机实时。
- 合成图上的精度基线（82.5 mm 码：距离 0.02%–0.42%、倾角 15°–75° 误差 0.05–0.42°）用
  `scripts/gen_pose_scenes.py` 造图后由 `scripts/pose_check.py`（逐图）或
  `scripts/pose_check.py --sweep`（出 CSV/MD/8 张曲线）复现，口径与失败分析见 `docs/位姿估计.md`。

---

## 文档

| 文件 | 内容 |
|---|---|
| `docs/progress.md` | 进度、实测数字、踩过的坑（改代码前先扫第五节） |
| `docs/interface.md` | 契约（`Frame` / `Detection` / `ArucoMarker` / `MarkerPose`） |
| `docs/位姿估计.md` | PnP 口径、两批实测（23 mm / 82.5 mm）、失败排查 |
| `docs/相机标定.md` | 内参怎么标、怎么读、填错会怎样 |
| `docs/YOLO接入.md` | YOLO 怎么用、为什么不用 torch、为什么不训练 |
| `docs/ArUco码原理与识别教程.md` | 编码方式、字典与 ID 入门 |
| `docs/视觉组协作规范.md` / `docs/代码注释规范.md` / `docs/提交信息规范.md` | 三个规范 |

---

## 已知限制

- **位姿精度只在合成图上验收**：真机只验证到链路（1280×1024、20 ms 28 fps），距离/倾角逐项精度
  尚未用卷尺对表，现场需先解决光照（40 ms 下画面均值低于 30 时角点会飘）。
- **位姿只做单码相对位姿**：多码融合、场地坐标系下的绝对位姿都没做。
- **正视时倾角不可信**：`tilt` 在 0° 附近弱可观（可散到 10°+），15° 以上才准。
- **相机需要 MVS SDK**：镜像不带 deb，需手动投放后重建；MVS 客户端与本仓库互斥。
- **ROS 2 未接入**：入口按「薄壳 + 单一实现」留好位置，接入时复用 `cli.py`。
- **判色阈值依赖光照**：换场地要重调 `color_thresholds.yaml`，`hsv_tuner.py` 为此准备。
