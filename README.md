# RGB-D 主逻辑集成交付

本分支只保留 RGB-D 识别运行必需的代码、配置和只读接入文件，面向 `driverLogic` 仓库的 **`driverLogic_2027`** 分支（参考 `e5b2a2c`）。不需要再运行导出工具。

这是专用交付分支；完整视觉工程保留在原 main，勿将此分支的目录裁剪整体合回完整工程。

## 目录

| 目录 | 内容 |
|---|---|
| `vision/` | YOLO、HSV 判色、彩深校正、平面拟合及公共类型 |
| `ros_packages/` | 识别 ROS 包和消息接口包 |
| `configs/` | HSV 阈值 |
| `mainlogic/` | 主逻辑只读适配器和测试入口 |
| `models/` | 模型放置位置和校验和；模型文件单独交付 |

`pyproject.toml` 和 `requirements.txt` 用于安装 Python 包。根目录的 `COLCON_IGNORE` 防止主工作空间重复发现交付目录内的 ROS 包。

## 1. 放入主逻辑仓库

下载**此分支**的 ZIP，解压为主工作空间的 `third_party/up70_rgbd/`。下面操作在主逻辑现有的 **Humble / Python 3.10 / Zenoh 容器**内执行：

```bash
export LOGIC_ROOT="$HOME/ros2_ws"
export RGBD_ROOT="$LOGIC_ROOT/third_party/up70_rgbd"
```

确认 `$RGBD_ROOT` 下直接就是本 README 和 `vision/`，而不是又套了一层目录。首次接入时执行整段；同名文件已存在则先核对，不覆盖：

```bash
(
  set -eu
  test -f "$LOGIC_ROOT/src/MainLogic/link/ros_bridge_node.py"
  test ! -e "$LOGIC_ROOT/src/vision_depth"
  test ! -e "$LOGIC_ROOT/src/vision_interfaces"
  test ! -e "$LOGIC_ROOT/src/MainLogic/adapters/rgbd.py"
  test ! -e "$LOGIC_ROOT/src/MainLogic/MAIN/rgbd_readonly.py"
  cp -a "$RGBD_ROOT/ros_packages/vision_depth" "$LOGIC_ROOT/src/"
  cp -a "$RGBD_ROOT/ros_packages/vision_interfaces" "$LOGIC_ROOT/src/"
  mkdir -p "$LOGIC_ROOT/src/MainLogic/adapters"
  cp "$RGBD_ROOT/mainlogic/MainLogic/adapters/rgbd.py" "$LOGIC_ROOT/src/MainLogic/adapters/"
  cp -n "$RGBD_ROOT/mainlogic/MainLogic/adapters/__init__.py" "$LOGIC_ROOT/src/MainLogic/adapters/"
  cp "$RGBD_ROOT/mainlogic/MainLogic/MAIN/rgbd_readonly.py" "$LOGIC_ROOT/src/MainLogic/MAIN/"
)
```

随后维护主工作空间 `src/` 中的 ROS 和主逻辑接入文件；Python 算法仍在 `third_party/up70_rgbd/vision/`。

## 2. 模型与依赖

把配套的 `yolov8n_mix4.onnx` 和 `yolov8n_mix4.names.json` 放到 `$RGBD_ROOT/models/`。模型不随 Git 下载，不可只复制其中一个。核对文件：

```bash
cd "$RGBD_ROOT/models"
sha256sum -c SHA256SUMS
```

加载主容器原环境，保留原有 Zenoh 路由和 domain。检查结果应为 Humble、Python 3.10、`rmw_zenoh_cpp`；所有终端须一致：

```bash
source /opt/ros/humble/setup.bash
printenv ROS_DISTRO RMW_IMPLEMENTATION ROS_DOMAIN_ID
python3 --version
ros2 pkg prefix rmw_zenoh_cpp
```

补齐依赖；root 用户去掉 `sudo`。安装内容也要记入主工程的镜像构建流程：

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-colcon-common-extensions \
  ros-humble-vision-msgs ros-humble-rosidl-default-generators ros-humble-rviz2 \
  'ros-humble-orbbec-camera=2.9.3-*'
python3 -m pip install -r "$RGBD_ROOT/requirements.txt"
python3 -m pip install --no-deps -e "$RGBD_ROOT"
```

主框架自己的依赖沿用原环境。不要并装多个 pip OpenCV 变体；已有其他版本时先统一。已有 Orbbec 源码工作空间不得遮住目标驱动版本 2.9.3。

## 3. 编译与启动

```bash
cd "$LOGIC_ROOT"
python3 -m colcon --log-base log-rgbd build --base-paths src \
  --packages-up-to vision_depth --build-base build-rgbd --install-base install-rgbd
source install-rgbd/setup.bash
export PYTHONPATH="$LOGIC_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
```

终端 A：关闭其他占用 Gemini 的程序，然后启动识别；USB 设备权限由宿主机配置。

```bash
cd "$RGBD_ROOT"
ros2 launch vision_depth rgbd_side_viewer.launch.py transport:=inherit headless:=true
```

该命令启动一份 Gemini 驱动，使用同步、D2C、毫米深度配置。不要同时启动主工程旧的 `orbbec_setup.launch.py`。已有驱动只有在提供匹配的对齐彩深和两路 CameraInfo 时，才能加 `start_camera:=false`。

终端 B：进入同一个主逻辑容器，重新设置 `LOGIC_ROOT`、`RGBD_ROOT`，source 同一套 ROS/主环境和 `install-rgbd/setup.bash`，设置上面的 `PYTHONPATH` 后执行：

```bash
ros2 topic info /rgbd_side/candidates --verbose
python3 -m MainLogic.Main --main-module rgbd_readonly --main-func async_main
```

应看到 `RGBD status=...`；有有效目标时打印类别、颜色、米制位置及法向。**不要用默认 R2n_Main 验收视觉，它会启动串口并发底盘测试速度。** 只读入口不启动串口或运动。结束时先停只读入口，再停视觉 launch。

## 4. 接入业务

在现有入口的 `async_main()` 中，使用已经初始化的桥接节点：

```python
import asyncio
from MainLogic.adapters.rgbd import RgbdAdapter
from MainLogic.link.ros_bridge_node import RosBridgeNodeInstance

adapter = RgbdAdapter(RosBridgeNodeInstance)
try:
    while True:
        snapshot = adapter.snapshot()
        for item in snapshot.candidates:
            print(item.class_id, item.color_id, item.position_m, item.normal)
        await asyncio.sleep(0.02)
finally:
    adapter.close()
```

每个主节点建立一个适配器供业务共享，不重复初始化 ROS 或 spin。由业务决定使用哪个候选。

- `position_m`：相机光学系可见面重心，单位米，X 向右、Y 向下、Z 向前。
- `normal`：朝相机的单位法向；不是完整物体姿态。
- `class_id`：0=block、1=top；`color_id`：0=未知、1=红、2=蓝、3=红蓝混色。
- `score`：YOLO 置信度；`stamp_ns/frame_id` 保留采集时间和坐标系。

空数组立即清空；超过默认 0.5 秒本地接收有效期，再次读取返回空候选。每次决策前重新读取，旧快照不会自行改变；`age_s` 仅为本地接收后年龄，不是总链路延迟。没有跨帧 ID，不自动选择抓取点或转换基座坐标。

ROS 话题为 `/rgbd_side/candidates`，类型 `vision_interfaces/msg/RgbdCandidateArray`，可靠、易失、深度 1。原始位置和四元数位于 `candidates[i].detection.results[0].pose.pose`；适配器已将四元数的局部 +Z 换算为法向。机械臂使用前仍需手眼标定/TF，不能只改 frame 名字。

## 5. 核对与配置

联调确认：收到有效候选；空数组立即撤销；停掉视觉后读取结果超时清空；重启后恢复；接收过程不触发串口或运动。

配置只需看：主工作空间 `src/vision_depth/config/` 下两份 RGB-D YAML，以及 `$RGBD_ROOT/configs/color_thresholds.yaml`。ROS 配置修改后重新编译并重启，HSV 配置修改后重启识别。默认滤波开启，深度含后处理结果。

收不到消息先检查 RMW、domain、路由和 source 路径，并确认传了 `transport:=inherit`；默认模式仍是 Fast DDS/SHM。没有候选再检查模型、有效深度和图像/CameraInfo 是否匹配。

软件接入通过不等于真机空间、绝对精度或连续稳定性验收。负责人 Zenoh 容器中的实际联调仍须现场完成。

本精简版软件检查：Humble 中干净安装、两个 ROS 包构建及 **95 项 RGB-D/判色/适配器检查通过**；1 项旧 mix2 模型检查因未交付该模型而跳过。运行依赖检查确认不包含 ArUco。未新增真机或 Zenoh 现场验收。
