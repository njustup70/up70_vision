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

# ---- 2. 应用与依赖层 ----
# 源码放到固定路径 /opt/up70_vision。compose 与 devcontainer 都把宿主仓库
# bind mount 到这个路径，所以改宿主机代码在容器里立即生效，不必重建镜像。
# -e 可编辑安装：import up70_vision 拿到的是挂载进来的源码，不是构建时那份拷贝。
# --no-cache-dir：不把 pip 下载缓存留在镜像层里。
WORKDIR /opt/up70_vision
COPY . .
RUN pip3 install --no-cache-dir --upgrade pip -i "$PIP_INDEX_URL" \
 && pip3 install --no-cache-dir -e . -i "$PIP_INDEX_URL"

# ---- 3. 运行时约定 ----
# 仓库文档里的命令统一写 python，补一个软链接，免得在容器里要记 python3
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# 默认跑环境自检：docker run 不带 command 时就知道环境是否正常
CMD ["python", "scripts/check_env.py"]
