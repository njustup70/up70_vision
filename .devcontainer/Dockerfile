# 共用视觉环境：Ubuntu 22.04 / Humble，默认 Gemini，WITH_MVS=1 加装海康 SDK。
FROM ros:humble-ros-base@sha256:1813d3c85d7f96ff7d3012d865204583255740182db5d0065f8f8cd029a83138

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-colcon-common-extensions python3-rosdep build-essential cmake \
    libgl1 libglib2.0-0 libsm6 libice6 libusb-1.0-0 \
    ros-humble-vision-msgs ros-humble-tf2-ros ros-humble-rviz2 ros-humble-rosidl-default-generators \
    ros-humble-rosbag2-storage-mcap 'ros-humble-orbbec-camera=2.9.3-*' \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

ARG PIP_INDEX_URL=https://pypi.org/simple
# NumPy 1.x 与 Humble 的系统 ROS Python 扩展保持兼容；所有入口均使用系统 Python 3.10。
RUN python3 -m pip install --no-cache-dir -i "$PIP_INDEX_URL" 'pip==24.0' \
    && python3 -m pip install --no-cache-dir -i "$PIP_INDEX_URL" \
       'numpy==1.26.4' 'opencv-python==4.11.0.86'

WORKDIR /opt/vision
COPY pyproject.toml README.md requirements.txt ./
COPY vision/ vision/
COPY lab/ lab/
COPY configs/ configs/
COPY knowledge/ knowledge/
COPY ros2_ws/src/ ros2_ws/src/
RUN python3 -m pip install --no-cache-dir -i "$PIP_INDEX_URL" -e '.[lab]' pytest \
    && . /opt/ros/humble/setup.sh \
    && colcon --log-base /tmp/up70-colcon-log build \
       --base-paths /opt/vision/ros2_ws/src \
       --build-base /tmp/up70-colcon-build --install-base /opt/up70-ros/install \
    && rm -rf /tmp/up70-colcon-build /tmp/up70-colcon-log

COPY tests/ tests/
COPY .devcontainer/gemini-entrypoint.sh /usr/local/bin/up70-gemini-entrypoint
ENTRYPOINT ["/usr/local/bin/up70-gemini-entrypoint"]
# SDK 只在本层安装；构建时只读挂载 deb，不把安装包写入镜像层。
# 参数放在公共依赖与 ROS 构建之后，切换相机能力时可复用前面的构建缓存。
ARG WITH_MVS=0
RUN --mount=type=bind,source=.devcontainer/pkgs,target=/mnt/mvs-pkgs,readonly \
    set -eu; \
    case "$WITH_MVS" in \
      0) echo '[MVS] disabled (WITH_MVS=0)' ;; \
      1) set -- /mnt/mvs-pkgs/*.deb; \
         if [ ! -f "$1" ] || [ "$#" -ne 1 ]; then \
           echo 'WITH_MVS=1 需要在 .devcontainer/pkgs/ 放入且仅放一个 MVS deb' >&2; exit 1; \
         fi; \
         dpkg -i "$1" ;; \
      *) echo 'WITH_MVS must be 0 or 1' >&2; exit 1 ;; \
    esac

# 无 SDK 时这些路径不触发加载；真正打开海康相机时由 MvImport 使用。
ENV MVCAM_SDK_PATH=/opt/MVS \
    MVCAM_COMMON_RUNENV=/opt/MVS/lib \
    MVCAM_GENICAM_CLPROTOCOL=/opt/MVS/lib/CLProtocol \
    ALLUSERSPROFILE=/opt/MVS/MVFG \
    LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib/32

# 进入后手动启动相机；运行 doctor 或离线工具时可直接覆盖命令。
CMD ["bash"]
