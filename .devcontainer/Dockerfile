# up70-vision 开发/运行环境镜像。
#
# 为什么底座是 ubuntu:22.04：战队标准发行版就是 22.04；且未来 ROS 2 Humble
# 官方只支持 22.04，其自带 Python 3.10。现在把底座定下来，到 ROS 2 阶段只需
# 在同一底座上加层，不用整体换镜像。
#
# 本镜像只放「当前阶段真正用得到」的东西：Python 3.10 + numpy + OpenCV + PyYAML。
# 没有 CUDA、没有 PyTorch、没有 ROS 2——等真需要时再加，避免现在就背一个十几 GB 的镜像。

FROM ubuntu:22.04

# 关掉交互式提示（典型如 tzdata 问时区），否则 build 会卡在等待输入
ENV DEBIAN_FRONTEND=noninteractive

# pip 源可用 build-arg 覆盖。海外或队外同事这样建：
#   docker compose -f .devcontainer/docker-compose.yml build --build-arg PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

# ---- 1. 系统层 ----
# libgl1 / libglib2.0-0：opencv-python 的非 headless wheel 需要 libGL.so.1。
#   缺了它 import cv2 直接报 ImportError，这是 Ubuntu 上最常见的坑，所以在镜像里装好。
# ca-certificates：https 下载所必需。
# --no-install-recommends：不拉推荐包，镜像能小几十 MB。
# rm -rf /var/lib/apt/lists/*：不留 apt 索引，同样为了瘦身。
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      libgl1 \
      libglib2.0-0 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ---- 1.5 相机 SDK 层（可选）----
# 海康 MVS：只有「相机接入」阶段用得到。它是 deb 包而不是 pip 包（写不进
# pyproject.toml 的 dependencies），所以单独占一层。
#   libusb-1.0-0：libMvCameraControl.so 直接依赖里只有 glibc/libstdc++，
#     libusb 是**运行时 dlopen** 的，缺了它相机根本打不开（基础镜像里没有）。
#   deb 放 .devcontainer/pkgs/：184 MB，已被 .gitignore 排除，每个开发者各自下载一份。
#   没放 deb 也能 build 成功，只是 input.source=camera 会报「MVS 未安装」；
#   不接相机的开发（合成图 / 回放 / 判色调参 / 单测）完全不受影响。
# 放在 COPY . . 之前：这一层很少变，放前面可以让改代码时不重建它。
COPY .devcontainer/pkgs/ /tmp/mvs-pkgs/
RUN if ls /tmp/mvs-pkgs/*.deb > /dev/null 2>&1; then \
      apt-get update \
      && apt-get install -y --no-install-recommends libusb-1.0-0 \
      && apt-get install -y --no-install-recommends /tmp/mvs-pkgs/*.deb \
      && rm -rf /var/lib/apt/lists/*; \
    else \
      echo "[MVS] .devcontainer/pkgs/ 下没有 deb —— 跳过相机 SDK，input.source=camera 将不可用"; \
    fi \
 && rm -rf /tmp/mvs-pkgs

# MVS 的运行时环境变量。MVS 的安装脚本会写进 ~/.bashrc，但非交互式 shell 不读它，
# 所以显式写进镜像 ENV（src/up70_vision/io/camera.py 里也做了 setdefault 兜底）。
ENV MVCAM_SDK_PATH=/opt/MVS \
    MVCAM_COMMON_RUNENV=/opt/MVS/lib \
    MVCAM_GENICAM_CLPROTOCOL=/opt/MVS/lib/CLProtocol \
    ALLUSERSPROFILE=/opt/MVS/MVFG \
    LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib/32

# ---- 2. 应用与依赖层 ----
# 源码放到固定路径 /opt/up70_vision。compose 与 devcontainer 都把宿主仓库
# bind mount 到这个路径，所以改宿主机代码在容器里立即生效，不必重建镜像。
# -e 可编辑安装：import up70_vision 拿到的是挂载进来的源码，不是构建时那份拷贝。
# --no-cache-dir：不把 pip 下载缓存留在镜像层里。
WORKDIR /opt/up70_vision
COPY . .
RUN pip3 install --no-cache-dir --upgrade pip -i "$PIP_INDEX_URL" \
 && pip3 install --no-cache-dir -e ".[lab]" -i "$PIP_INDEX_URL"

# ---- 3. 运行时约定 ----
# 仓库文档里的命令统一写 python，补一个软链接，免得在容器里要记 python3
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# ---- 4. 图形界面（容器内开窗口用）----
# cv2.imshow 走 Qt 的 xcb 插件，它 dlopen 这两个 X11 会话库：
#   缺 libSM.so.6 / libICE.so.6 时，报的是
#   "Could not load the Qt platform plugin xcb ... even though it was found"
#   —— 报错不提缺哪个库，很容易误判成"OpenCV 没编 GUI 支持"。
# 装了这两个包 + 宿主 xhost +local:root + 挂 /tmp/.X11-unix 与 DISPLAY，
# 就能在容器里 imshow；不装也不影响 --headless（落盘看图）那条路。
# 放在最后一层：前面的 pip 层缓存不受影响，重建只花几秒。
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsm6 libice6 \
 && rm -rf /var/lib/apt/lists/*

# 默认跑环境自检：docker run 不带 command 时就知道环境是否正常
CMD ["python", "scripts/check_env.py"]
