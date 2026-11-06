# RGB-D → driverLogic_2027 交接上手手册

活文档，2026-11-06。接收仓库为 `njustup70/driverLogic` 的 **`driverLogic_2027`** 分支，参考提交 `e5b2a2c`。环境为主逻辑现有的 Humble / Python 3.10 / Zenoh；视觉和主逻辑在同一容器运行。

交付内容：RGB-D 运行源码、两个 ROS 包、配置、只读适配器和测试入口。模型单独提供。接入完成后，业务层读取候选快照，不执行推理，也不拼 ROS 消息。

## 1. 取得交接包

视觉维护者从交付分支导出（输出目录必须不存在）：

```bash
git fetch origin
git switch codex/rgbd-driverlogic-handoff
python3 lab/export_driverlogic.py --out /tmp/rgbd-handoff \
  --model-dir /实际路径/knowledge/models
```

把整个 `rgbd-handoff/` 交给负责人；其中 `runtime/` 是运行模块，`mainlogic/` 是接入文件，`model_assets/` 是模型。没有模型时可省略 `--model-dir`，运行前另行取得配套 mix4 ONNX 与 `.names.json`。`MANIFEST.json` 记录版本及实际文件哈希；模型不提交 Git。

## 2. 放入主工作空间

以下操作在**主逻辑容器**里完成。先把交接包放到容器可见目录，修改第一行路径。主逻辑工作空间默认 `~/ros2_ws`：

```bash
export HANDOFF=/实际路径/rgbd-handoff
export LOGIC_ROOT="$HOME/ros2_ws"
cd "$HANDOFF"
sha256sum -c SHA256SUMS
```

确认当前主仓库是 `driverLogic_2027` 或负责人基于它创建的集成分支。首次迁入时执行下列整段；目标模块已有同名文件会停止，需先核对版本，不能混装：

```bash
(
  set -eu
  test -f "$LOGIC_ROOT/src/MainLogic/link/ros_bridge_node.py"
  test ! -e "$LOGIC_ROOT/third_party/up70_rgbd"
  test ! -e "$LOGIC_ROOT/src/vision_depth"
  test ! -e "$LOGIC_ROOT/src/vision_interfaces"
  test ! -e "$LOGIC_ROOT/src/MainLogic/adapters/rgbd.py"
  test ! -e "$LOGIC_ROOT/src/MainLogic/MAIN/rgbd_readonly.py"
  mkdir -p "$LOGIC_ROOT/third_party" "$LOGIC_ROOT/src/MainLogic/adapters"
  cp -a "$HANDOFF/runtime" "$LOGIC_ROOT/third_party/up70_rgbd"
  cp -a "$HANDOFF/runtime/ros2_ws/src/vision_depth" "$LOGIC_ROOT/src/"
  cp -a "$HANDOFF/runtime/ros2_ws/src/vision_interfaces" "$LOGIC_ROOT/src/"
  cp "$HANDOFF/mainlogic/MainLogic/adapters/rgbd.py" "$LOGIC_ROOT/src/MainLogic/adapters/"
  cp -n "$HANDOFF/mainlogic/MainLogic/adapters/__init__.py" "$LOGIC_ROOT/src/MainLogic/adapters/"
  cp "$HANDOFF/mainlogic/MainLogic/MAIN/rgbd_readonly.py" "$LOGIC_ROOT/src/MainLogic/MAIN/"
  if test -d "$HANDOFF/model_assets"; then
    cp "$HANDOFF/model_assets/"* "$LOGIC_ROOT/third_party/up70_rgbd/knowledge/models/"
  fi
)
export RGBD_ROOT="$LOGIC_ROOT/third_party/up70_rgbd"
```

完整保留 Python 包是为了保留间接依赖；无需把 ArUco 等功能启动起来。`runtime/COLCON_IGNORE` 避免工作空间发现重复 ROS 包。后续只修改并编译 `src/vision_depth`、`src/vision_interfaces` 中的 ROS 副本，Python 算法及 HSV 配置位于 `third_party/up70_rgbd`。

## 3. 安装与编译

先停止现有 Gemini 驱动。保留主容器原有 Zenoh 配置、路由进程和 domain，不启动第二份路由。加载 Humble 后确认环境：

```bash
source /opt/ros/humble/setup.bash
printenv ROS_DISTRO RMW_IMPLEMENTATION ROS_DOMAIN_ID
ros2 pkg prefix rmw_zenoh_cpp
python3 --version
```

预期 Humble、Python 3.10、`rmw_zenoh_cpp`。domain 可以为空（默认域），但视觉、主逻辑和检查终端必须一致。如果结果不同，先与负责人统一环境。

在主镜像中补充以下依赖；非 root 用户使用 `sudo`。依赖安装后应同步到主工程的镜像构建流程，避免删除容器后丢失：

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-colcon-common-extensions \
  ros-humble-vision-msgs ros-humble-tf2-ros ros-humble-rviz2 \
  ros-humble-rosidl-default-generators ros-humble-rosbag2-storage-mcap \
  'ros-humble-orbbec-camera=2.9.3-*'
python3 -m pip install -r "$RGBD_ROOT/requirements-humble.txt"
python3 -m pip install --no-deps -e "$RGBD_ROOT"
```

只保留一种 pip OpenCV 包；若已有 contrib/headless 变体，先按主镜像依赖安排统一，避免覆盖 `cv2`。现有驱动若来自另一个已 source 的工作空间，确认它没有遮住新版本：

```bash
python3 -c "from ament_index_python.packages import get_package_share_directory as p; import xml.etree.ElementTree as E; print(p('orbbec_camera')); print(E.parse(p('orbbec_camera')+'/package.xml').findtext('version'))"
python3 -c "import vision, cv2, numpy, rclpy; print(vision.__file__); print(cv2.__version__, numpy.__version__)"
```

驱动应为 2.9.3；算法路径应指向 `$RGBD_ROOT/vision`。然后仅编译视觉相关包，独立产物目录不覆盖主逻辑旧构建：

```bash
cd "$LOGIC_ROOT"
python3 -m colcon --log-base log-rgbd build --base-paths src \
  --packages-up-to vision_depth --build-base build-rgbd --install-base install-rgbd
source install-rgbd/setup.bash
export PYTHONPATH="$LOGIC_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ros2 interface show vision_interfaces/msg/RgbdCandidateArray
```

## 4. 启动识别和只读接收

每个新终端都要处于**同一主逻辑容器**，重新设置 `LOGIC_ROOT`、`RGBD_ROOT`，source Humble、主工程原环境及 `install-rgbd/setup.bash`，并设置上面的 `PYTHONPATH`。不要从旧视觉示例复制 Fast DDS 的 export；这里继承主容器 Zenoh。

终端 A（先确认模型两文件位于 `$RGBD_ROOT/knowledge/models/`）：

```bash
cd "$RGBD_ROOT"
ros2 launch vision_depth rgbd_side_viewer.launch.py transport:=inherit headless:=true
```

这会启动一份 Gemini 驱动，使用配套同步、D2C 和毫米深度。主工程原 `orbbec_setup.launch.py` 不要同时启动。只有已有驱动已提供配套对齐彩深和两路 CameraInfo 时才加 `start_camera:=false`；仅话题名相同不代表符合输入要求。

终端 B：

```bash
ros2 topic info /rgbd_side/candidates --verbose
python3 -m MainLogic.Main --main-module rgbd_readonly --main-func async_main
```

应打印 `RGBD status=...`；有可用候选时显示类别、颜色、位置和法向。**不要用默认 `R2n_Main` 测试视觉：该入口会启动串口并发送底盘速度。** 交付的 `rgbd_readonly` 不导入该入口，也不启动运动。

## 5. 接入业务

负责人在实际入口的 `async_main()` 中，待 `Main.py` 已初始化桥接节点后使用：

```python
from MainLogic.adapters.rgbd import RgbdAdapter
from MainLogic.link.ros_bridge_node import RosBridgeNodeInstance

adapter = RgbdAdapter(RosBridgeNodeInstance)
try:
    while True:
        snapshot = adapter.snapshot()
        if snapshot.candidates:
            # 由业务决定使用哪个候选；这里不默认取第一个。
            for item in snapshot.candidates:
                print(item.class_id, item.color_id, item.position_m, item.normal)
        await asyncio.sleep(0.02)
finally:
    adapter.close()
```

原入口通常已经 import `asyncio`；没有时添加 `import asyncio`。每个主节点只建立一个适配器；业务模块共享该实例即可，不调用第二次 `rclpy.init()` 或 `spin()`。适配器直接使用节点的订阅接口，绕开当前桥接帮助函数写死的深度 10，采用消息契约的可靠、易失、深度 1。

`position_m` 为相机光学系可见面重心，单位米；`normal` 为朝相机单位法向。`class_id` 为 0=block、1=top；`color_id` 为 0=未知、1=红、2=蓝、3=红蓝混色；`score` 为 YOLO 置信度。

快照 `status` 为 `waiting/active/empty/stale/invalid/closed`，只有 active 有候选。收到空数组立即清空；超过默认 0.5 秒本地接收有效期，再次读取返回空候选。旧快照不会自行改变，每次决策前重新读取。`stamp_ns/frame_id` 保留采集时间与坐标系；`age_s` 只是本地接收后年龄，不能代替相机到控制器的总延迟。

不提供跨帧 ID、不自动选择目标、不做基座变换或抓取动作。机器人使用前还需要有效手眼标定和 TF；位置做旋转与平移，法向只旋转。完整 ROS 字段语义见交接包 `docs/interface.md`。

## 6. 联调确认与排错

| 检查 | 通过条件 |
|---|---|
| 接口 | 类型为 `vision_interfaces/msg/RgbdCandidateArray`，只读入口实际收到消息 |
| 正常输入 | 与视觉相同场景读到类别、颜色、米制坐标和法向 |
| 撤销 | 无候选的空数组立即清空；停止视觉后，读取快照在 0.5 秒后为空 |
| 重启 | 重启视觉后可重新接收，不沿用上一进程候选 |
| 主逻辑 | 不依赖默认速度测试入口，不因读取视觉而发送串口命令 |

模型找不到：确认从 `$RGBD_ROOT` 启动且模型与类名表配套。消息收不到：核对 RMW、domain、路由、消息类型及 source 路径，确认使用了 `transport:=inherit`。没有候选：检查模型检测框、有效深度、选面状态和配套 CameraInfo，不能只凭空数组判定通信失败。

配置入口为主工作空间 `src/vision_depth/config/` 下两份 RGB-D YAML，以及 `$RGBD_ROOT/configs/color_thresholds.yaml`。前者改后重新编译并重启，后者改后重启识别。当前滤波开启，深度包含后处理结果；详细参数见交接包 `docs/depth-README.md` 和运行目录 `configs/README.md`，手册不重复参数表。

这些检查只证明软件与数据接入。空间对齐、绝对精度及连续稳定性仍需独立验收；不把候选当作已验证抓取点。结束时先 Ctrl+C 停止只读入口，再停止视觉 launch。

升级时重新交付源码、消息与模型并核对 manifest；不可只换一份 `.msg` 或算法文件。此手册及适配器针对 `driverLogic_2027@e5b2a2c`，主框架继续重构时重新核对 `MainLogic.link` 路径。

## 本轮软件核查

2026-11-06：在隔离 Humble 容器中整库 **319 passed、2 skipped**，跳过的是未交付的 mix2 与独立 COCO 模型；本次 mix4 真实模型检查通过。干净导出包的 Python 安装、校验和及两个 ROS 包构建通过；真实 ROS 消息验证了接收、空数组、断流过期和恢复。对 `driverLogic_2027@e5b2a2c` 的临时源码副本加入交接模块后，原 `MainLogic.Main` 入口只读测试通过，未启动串口节点。该测试另装了主框架所需 SciPy 1.14.1，不属于视觉算法的新依赖。

实际消息测试使用现有 Humble/Fast DDS；Zenoh 环境继承已有 launch 回归，但本机没有负责人的 Zenoh 容器，**Zenoh 实际联调、相机真机和精度验收尚未执行**。迁入后按第 4～6 节完成目标环境验收。
