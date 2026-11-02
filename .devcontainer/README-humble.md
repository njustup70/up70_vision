# ROS 2 Humble 视觉容器

本配置为 Ubuntu 22.04、ROS 2 Humble、x86-64 的独立环境。既有
`docker-compose.yml` 和通用镜像不变。基础镜像不带相机驱动或设备映射；海康和
Gemini 分别使用独立构建目标。同一镜像不意味着两种相机已经在同一台机器联调。

## 构建和纯软件检查

以下均在仓库根目录运行，使用 Docker Compose v2：

```bash
docker compose -f .devcontainer/docker-compose-humble.yml build humble
docker compose -f .devcontainer/docker-compose-humble.yml run --rm humble
docker compose -f .devcontainer/docker-compose-humble.yml run --rm humble \
  python3 -c 'import sys, rclpy, cv2, numpy, vision; print(sys.executable, cv2.__version__, numpy.__version__)'
```

项目目录以 `/opt/vision` 挂载；Python 源码改动即时生效。镜像内 ROS 包按构建时
源码安装在 `/opt/up70-ros/install`：改动 `ros2_ws/src/` 后应重新构建镜像，
不能误以为挂载后安装空间也会自动更新。容器默认 root，写入挂载目录可能产生
root 文件；实验数据请明确指定输出路径。测试会改写 `output/` 的固定子目录，
完整测试须在不含本地实验输出的隔离源码副本中执行：

```bash
docker compose -f .devcontainer/docker-compose-humble.yml run --rm humble \
  python3 -m pytest tests/ -q
```

基础镜像固定 `ros:humble-ros-base` 的 digest，Python 固定 NumPy 1.26.4、
OpenCV 4.11.0.86 和 pip 24.0；其余 apt/pip 依赖取构建时的源，并未全部锁定。
2026-11-02 软件验证时另有 PyYAML 6.0.3、rosbags 0.11.9、matplotlib 3.10.9、pytest 6.2.5，Gemini 的 ROS 包是 `2.9.3-2jammy.20260915.210236`；这些值并非全部固定。
因此重建可能得到不同的间接依赖，需保存镜像 digest 与下列版本记录，不能将
Dockerfile 本身视为完全可复现的依赖快照：

```bash
docker image inspect up70-vision:humble --format '{{.Id}}'
docker compose -f .devcontainer/docker-compose-humble.yml run --rm humble \
  bash -c 'python3 -m pip freeze; dpkg-query -W "ros-humble-*"'
```

## 相机（分别使用，默认不自动启动）

- **海康**：将搭档使用的单个 MVS Linux x86-64 deb 放在 `.devcontainer/pkgs/`，
  再运行 `docker compose -f .devcontainer/docker-compose-humble.yml build hik`。
  缺少或存在多个 deb 时构建失败。运行使用
  `docker compose -f .devcontainer/docker-compose-humble.yml run --rm hik bash`，
  在容器内按项目现有 `python3 -m vision run` 和配置使用海康入口。
- **Gemini 336L**：执行 `docker compose -f .devcontainer/docker-compose-humble.yml build gemini`，
  然后以 `docker compose -f .devcontainer/docker-compose-humble.yml run --rm gemini bash`
  进入。镜像要求 ROS 包 `orbbec_camera` 2.9.3；软件源没有该版本时构建失败，不自动换版本。
  先用 `ros2 launch orbbec_camera gemini_330_series.launch.py --show-args` 核对驱动参数，
  按 [深度 README](../vision/depth/README.md) 核实后再启动相机或现有 launch。

2026-11-02 海康镜像安装检查使用本机 `MVS-5.1.0_x86_64_20260909.deb`（包版本 `2025-07-11`），未接真机；搭档的实际 SDK 版本仍需确认。

两种相机容器按需映射 `/dev/bus/usb`，实际设备权限由宿主机 udev 决定；检查 USB 3
连接和设备访问权限。海康已有规则见[根 README](../README.md)，Orbbec 规则以所用
驱动版本为准。若设备还需要视频节点，先核对驱动和宿主实际节点，再按需映射；不能
仅凭 USB 可见判断图像可用。关闭占用相机的其他进程；不要同时让两个程序争用同一相机。

窗口为可选项：带上覆盖文件
`-f .devcontainer/docker-compose-humble-gui.yml`，并在宿主 X11 会话按需授权容器
访问显示服务、结束后撤销授权。此覆盖只提供 DISPLAY 和 X11 socket，图形显示还依赖
宿主会话；无显示环境使用 headless。两种 GUI 和真实设备均需分别实测。

Gemini 服务使用宿主网络及 IPC，配合现有 launch 的 Fast DDS 仅 SHM 配置供同机使用；
这不是跨机配置。不同容器/主机之间需确认 ROS_DOMAIN_ID、RMW、发现方式、IPC、QoS
与网络；不要把仅 SHM 配置永久写入通用环境。

## Jazzy bag 在 Humble 中回放

Jazzy 录制的 bag 元数据可能是 Humble 无法解析的新格式。不要在原包上运行 reindex。
若 MCAP 内容和消息类型可被 Humble 读取，可在容器能访问的临时目录建立指向原 MCAP
的链接，在该目录执行 `ros2 bag reindex <临时目录> --storage mcap`，再比较重建前后的
消息数、话题和时间范围。原 MCAP 应只读挂载；重建索引不证明所有跨版本消息兼容。
本次 `raw_023` 重建出的 QoS 字符串仍包含 Humble 无法解析的枚举：验证时仅在临时索引清空该字段，并在播放时显式覆盖 QoS（各相机话题可靠、易失，`/tf_static` 可靠、瞬态本地，均 keep-last 深度 5）。原记录的队列深度为 unknown，因此不能宣称完整 QoS 等价。具体结果见深度进度记录；其他 bag 的原 QoS 必须逐项核查，不可盲目套用。
