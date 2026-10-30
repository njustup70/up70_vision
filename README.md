# up70-vision

UP70 战队视觉组仓库：把 2D 视觉链路跑通——**图像输入 → 目标检出 → 位姿 → 绘制与落盘**，
为机械臂与 ROS 2 打底。

三条 2D 能力：**ArUco**（检测 → 位姿 → 实时循环，主力）、**YOLO**（可选，ONNX + `cv2.dnn`，不需要 torch）、
**判色**（HSV，YOLO 不可用时的轻量方案）。输入源三选一：`camera`（海康 MV-CU013-A0UC，MVS SDK）/
`replay`（图片或视频）/ `synthetic`（自造图，**没有相机也能跑全流程**）。

3D 抓取（深度相机）另有一条线：算法在 `vision/depth/`，ROS 2 外壳在 `ros2_ws/src/vision_depth/`，
由深度侧同学维护；本 README 的坑与命令主要针对 2D。

---

## 快速开始（容器，不用在本机装依赖）

```bash
# 只需一次：构建镜像（约 2~3 分钟）
docker compose -f .devcontainer/docker-compose.yml build

# 三条自检，依次应看到「全部通过」「PASS」「✅ 运行成功」
docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision python -m vision doctor
docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision python -m vision demo
docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision python -m vision run
```

`./demo.sh` 把这三步串起来，跑完 `output/demo/` 里有叠加图 + JSON + CSV。
源码是 bind mount，改代码立即生效；只有改 `pyproject.toml` 才需要重新 build。

---

## 怎么用

**只有 6 条主入口**：`python -m vision [run|live|doctor|demo|hsv-tuner|clean]`
（不带参数会打印这张表）。其余一次性工具在 `lab/`，直接 `python lab/xxx.py`。

```bash
# 实时 ArUco（窗口；无显示器加 --headless --save-every 30 --limit 300）
python -m vision live --set detector.name=aruco --set input.source=camera --set input.camera.exposure_us=40000

# 加位姿：画面多三根坐标轴；窗口 HUD 第 4 行直接显示 d/tilt/yaw（终端也同步打）
python -m vision live --set detector.name=aruco --set input.source=camera \
  --set pose.enabled=true --set pose.marker_size_mm=82.5

# 批处理（读完再汇总）：判色 18 块 / ArUco / YOLO
python -m vision run --set input.source=synthetic
python -m vision run --set detector.name=yolo --set input.source=replay --set input.replay.path=<图片目录>

# 造合成的 YOLO 训练集（350 mm 方块 / 200 mm 半红半蓝塔顶，自动标注）
python lab/gen_yolo_dataset.py --out data/yolo_synth --num 300

# 补实拍数据：开窗口给人摆位，每 1 秒存一张（远景/遮挡/空场景负样本都用它）
python lab/capture_dataset.py --tag g6_far --exposure-us 20000 --interval 1.0
python lab/capture_dataset.py --tag neg --exposure-us 10000 --frames 30   # 拍 30 张自动停

# 常用自检：ArUco 图组 9/9、位姿逐图、相机标定
python lab/check_aruco.py
python -m lab.pose_check --dir <图组目录>          # 或 --sweep 做七组精度扫描
python lab/calibrate_camera.py --capture 15 --board 9x6 --square-mm 23
```

在别的代码里用一次检测：`from vision.api import detect; detect(img, "aruco")`。

参数都在 `configs/default.yaml`（判色阈值在 `configs/color_thresholds.yaml`），
`--set 点分路径=值` 临时覆盖，只能改已存在的字段，写错启动即报错。

![真机位姿：四角、三根坐标轴与 HUD 里的 d/tilt/yaw](knowledge/images/live_pose_demo.jpg)

---

## 目录结构

```
vision/        运行时代码（要 import / 要测 / 有架构约束）
├── __main__.py  进程入口：python -m vision <子命令>
├── api.py       库门面：detect(image_bgr, detector)
├── types.py     唯一共用件：Frame / Detection / 几何
├── cli/         命令适配层：parsers / run / live / output
├── config/      YAML 加载、字段校验、--set 覆盖
├── sources/     输入源：camera / replay / synthetic（+ read_image_bgr）
├── aruco/       功能包：detector / pose / calibration / draw / types / stub
├── yolo/        功能包：detector / candidates / color_check / pipeline / draw
└── depth/       深度相机算法（纯 Python，可 pytest；ROS 外壳在 ros2_ws/）

ros2_ws/       ROS 2 工作空间（只有深度侧的 vision_depth 包，2D 不进 ROS）
lab/           一次性工具：check_aruco / pose_check / calibrate / gen_*（零架构约束，可互相 import）
configs/       场地参数；knowledge/ docs·calibration·models·images；tests/ 回归 121 项 + 夹具
```

依赖方向是硬约束：**功能包互不 import，功能包不碰 cli/config，`vision/` 不 import `lab/`**
（`tests/test_architecture.py` 会检查）。`lab/` 可以 import `vision/`，反之不行。

---

## 一定会踩的坑

1. **相机不是 UVC**：海康 MV-CU013-A0UC 走 USB3 Vision，宿主机没有 `/dev/video*`，`cv2.VideoCapture`
   打不开且不提示原因。三步缺一不可：udev 规则给权限（`idVendor=2bdf`）、把 MVS 的 deb 放进
   `.devcontainer/pkgs/` 后**重新 build**、compose 透传 `/dev/bus/usb`。
2. **相机是独占资源**：MVS 客户端、调参窗口、本仓库互斥，打不开先关占用者。
3. **容器里开窗口**要三件事：`xhost +local:root`、`-e DISPLAY=$DISPLAY`、挂 `/tmp/.X11-unix`；
   旧镜像会报 `xcb` 加载失败——镜像需含 `libsm6 libice6`，旧的重新 build。
4. **别装 `opencv-contrib-python` / `-headless`**：三者顶层模块都叫 `cv2`，互相覆盖且不报错。
5. **尺寸别混**：棋盘格 `--square-mm 23`（标称 15、实量 23）、ArUco 码 `pose.marker_size_mm`（实测黑框外沿，
   我们那批是 84 mm）。填错**不会让重投影误差变大**，只会让距离按比例错——用手机屏幕显示码时，
   填的是**屏幕上量到的实际尺寸**，不是文件里的设计值。
6. **位姿正视附近 `tilt` 不可信**（0° 附近可散到 10°+，倾斜 15° 以上才准）。
7. **曝光必须按场景调**：同一台相机在不同光照下要 3 ms ~ 40 ms 之间换（差 13 倍）。过曝会让白格并成一片、
   角点外扩，`d` 系统性偏大（实测偏 5%）；先用 3/10/20/40 ms 各拍一张，白格灰度 <250 再开跑。
8. **码必须贴平**：纸一弯（手捏着拍）平面假设就失效，重投影 RMS 会从 0.4 跳到 2.2 px，位姿被硬校验拒绝。
   用硬板/手机屏幕，四边都贴住。
9. **生成物不进版本库**：`output/`（运行产物）、`data/`（数据集）、`runs/`（训练产物）、
   `knowledge/models/*.onnx`、`configs/camera_calibration*.yaml` 都被 gitignore。
   新 clone 要位姿就先自己标定，要用 YOLO 就先 `python lab/export_yolo.py` 导出权重。
10. **镜像里没有 pytest**：`tests/` 每个文件都能 `python tests/test_x.py` 直接跑。
11. **`live` 只跑 ArUco**：判色/YOLO 出的是矩形框，走 `python -m vision run`；入口会直接提示，不会静默错。

---

## 已知限制

- 位姿已过**真机验收**：位移差分误差 ≤0.2%（200/400 mm），倾角 30°/50°/60° 误差 ≤1.2°；
  但仍只有**单码相对位姿**，多码融合 / 场地坐标没做。绝对距离要比卷尺读数大 ≈12 mm（镜头前端面 ≠ 光心，属基准差）。
- ROS 2 只接了**深度侧**（`ros2_ws/src/vision_depth`）；2D 侧未接（不打算让 `vision/` 依赖 rclpy）。
- YOLO 目前是 COCO 预训练权重 + 合成数据训练的 2 类模型（`block`/`top`），**真图训练仍卡在物块与标注**。
- 判色阈值依赖光照，换场地要重调（`hsv-tuner` 就是干这个的）。

---

## 文档去哪看

- **文档索引**（该先读哪篇、哪篇是存档）：[`knowledge/docs/README.md`](knowledge/docs/README.md)
- **契约**（`Frame` / `Detection` / `ArucoMarker` / `MarkerPose` 的唯一来源）：[`vision/docs/interface.md`](vision/docs/interface.md)
- **写代码前**：[`knowledge/docs/视觉组协作规范.md`](knowledge/docs/视觉组协作规范.md) 与 [`代码注释规范`](knowledge/docs/代码注释规范.md)
- **位姿真机数据与坑**：[`knowledge/docs/位姿估计真机测试计划.md`](knowledge/docs/位姿估计真机测试计划.md)（8.1–8.5 实测记录 + 15 条真机坑）
