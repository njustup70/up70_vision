# 共用相机容器

活文档，2026-11-04。一份 `Dockerfile`、一份 `docker-compose.yml`，共用 `vision` 服务和
`up70-vision:humble` 镜像。底座为 Ubuntu 22.04 / ROS 2 Humble / x86-64，默认支持
Gemini 336L，海康 MVS SDK 通过构建参数按需安装。使用 Docker Compose v2 或更高版本及 BuildKit。

## 构建和进入容器

从仓库根目录运行；默认构建不需要 MVS deb，即使本地已有 deb 也不会自动安装：

```bash
docker compose -f .devcontainer/docker-compose.yml build
docker compose -f .devcontainer/docker-compose.yml run --rm vision
```

默认进入终端，不自动启动相机；Humble 和项目 ROS 安装空间已加载。
需要海康时，先将一个 MVS Linux x86-64 deb 放入 `.devcontainer/pkgs/`，再构建同一镜像：

```bash
WITH_MVS=1 docker compose -f .devcontainer/docker-compose.yml build
docker compose -f .devcontainer/docker-compose.yml run --rm vision
```

`WITH_MVS` 只接受 0 或 1；启用时缺少或存在多个 deb 均报错，不跳过安装错误。
安装包通过构建时只读挂载提供，不复制到镜像层。启用后，同一容器同时具备两台相机的依赖，
具体启动哪台由命令决定。再次默认构建会替换镜像为无 MVS 版本；已运行容器不会随镜像重建更新，
应退出并重新进入。安装包来源及体积说明见[SDK 说明](pkgs/README.md)。

## 启动 Gemini 或海康

Gemini 先从搭档处取得 `yolov8n_mix4.onnx` 及同名 `.names.json`，放入仓库的
`knowledge/models/`。模型不进入镜像，通过源码挂载提供。确认 USB、模型和窗口就绪后，在容器内运行：

```bash
python -m vision doctor
ros2 launch vision_depth rgbd_side_viewer.launch.py rviz:=true
# 无显示环境：
# ros2 launch vision_depth rgbd_side_viewer.launch.py headless:=true
```

海康在启用 MVS 构建后使用原有 2D 入口；`input.source=camera` 指海康，Gemini 通过 ROS launch 启动：

```bash
python -m vision live --set detector.name=aruco --set input.source=camera \
  --set input.camera.exposure_us=40000
```

其他 YOLO/判色命令见[根 README](../README.md)。位姿尺寸和标定参数须匹配实物。
候选消息包含类别、颜色编号和法向，收发两侧均需构建 `vision_interfaces`；
正式字段及边界以[接口契约](../vision/docs/interface.md)为准。

## USB、窗口和同机通信

Compose 已包含 USB 总线、DISPLAY 和 X11 socket，无需额外覆盖文件。
USB 访问权限由宿主机 udev 决定；检查 USB 3 连接，关闭占用同一相机的其他进程。
海康宿主规则示例（写入 `/etc/udev/rules.d/99-hikrobot.rules` 后重载规则并重新插拔）：

```text
SUBSYSTEM=="usb", ATTRS{idVendor}=="2bdf", MODE="0666", SYMLINK+="hik_camera"
```

Gemini 的 udev 规则按 Orbbec 2.9.3 驱动说明在宿主安装。两台相机均使用 USB 总线，
不额外映射猜测的 `/dev/video0`。

容器默认 root；宿主 X11 会话开窗口前执行 `xhost +si:localuser:root`，结束后用
`xhost -si:localuser:root` 撤销。未设置 DISPLAY 时使用对应命令的 headless 参数。
Wayland 会话需可用的 XWayland；预置配置和动态库检查不等于实际窗口验收。

服务使用宿主网络和 IPC，默认 `ROS_DOMAIN_ID=97`，可在宿主设置环境变量覆盖。
现有 launch 强制 Fast DDS 仅共享内存，适用于同机通信，尚未验证跨机部署。

## 代码更新与软件检查

Python 源码挂载到 `/opt/vision`，改动即时生效。修改依赖、Dockerfile、SDK 或
**`ros2_ws/src/`** 后重新构建；已启用海康的环境重建时继续指定 `WITH_MVS=1`。
ROS 包（包括 `vision_interfaces` 和 `vision_depth`）安装在镜像内 `/opt/up70-ros/install`，
不会读取宿主 Jazzy 的 `ros2_ws/install`，进入容器时也不自动构建。
VS Code Dev Container 直接复用主 Compose。

镜像固定 Humble 基础镜像 digest、pip 24.0、NumPy 1.26.4 和 OpenCV 4.11.0.86；
Orbbec ROS 包要求 `2.9.3-*`，源中没有该版本则构建失败。其余 apt/pip 依赖未全部锁定。
`--build-arg PIP_INDEX_URL=<镜像源地址>` 可以覆盖默认 PyPI 下载源，例如：

```bash
docker compose -f .devcontainer/docker-compose.yml build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
docker image inspect up70-vision:humble --format '{{.Id}}'
docker compose -f .devcontainer/docker-compose.yml run --rm vision \
  bash -c 'python -m pip freeze; dpkg-query -W "ros-humble-*"'
```

直接覆盖默认终端命令即可运行自检和 OpenCV 冒烟：

```bash
docker compose -f .devcontainer/docker-compose.yml run --rm vision python -m vision doctor
docker compose -f .devcontainer/docker-compose.yml run --rm vision python -m vision demo
```

镜像含 pytest。完整回归须在**不含本地实验 output 的隔离源码副本**中运行
`python -m pytest tests/ -q`，部分测试会重建固定输出目录。容器默认 root，写挂载目录可能产生
root 所有的文件。软件和 bag 回放检查不替代真机、空间对齐、绝对精度、连续稳定性或跨机验收。

## Jazzy bag 在 Humble 中回放

Jazzy 录制的 bag 元数据可能是 Humble 无法解析的新格式。不要在原包上运行 reindex。
若 MCAP 内容和消息类型可被 Humble 读取，可在容器能访问的临时目录建立指向原 MCAP
的链接，在该目录执行 `ros2 bag reindex <临时目录> --storage mcap`，再比较重建前后的
消息数、话题和时间范围。原 MCAP 应只读挂载；重建索引不证明所有跨版本消息兼容。
本次 `raw_023` 重建出的 QoS 字符串仍包含 Humble 无法解析的枚举：验证时仅在临时索引清空该字段，并在播放时显式覆盖 QoS（各相机话题可靠、易失，`/tf_static` 可靠、瞬态本地，均 keep-last 深度 5）。原记录的队列深度为 unknown，因此不能宣称完整 QoS 等价。具体结果见深度进度记录；其他 bag 的原 QoS 必须逐项核查，不可盲目套用。
