# 环境状态（存档快照）

> 本文件是 `knowledge/docs/progress.md` 第六节在 2026-10-22 的归档（仓库瘦身 P1-3）。
> 镜像底座、依赖版本、相机型号、udev 规则、镜像体积这些值改了就更新本篇；日常怎么跑看仓库 `README.md`。

---

| 项 | 值 |
|---|---|
| 团队标准环境 | Docker 镜像：`ubuntu:22.04` + 系统 Python 3.10；镜像名 `up70-vision:dev`，**已 build 成功**（862 MB） |
| 镜像内容 | numpy / opencv-python / PyYAML，`pip install -e .`；实测 cv2 4.14.0 / numpy 2.2.6 / yaml 6.0.3 |
| 底座选择理由 | 战队标准发行版是 22.04；ROS 2 Humble 官方只支持 22.04 且自带 Python 3.10 |
| 代码生效方式 | compose 把仓库 bind mount 到 `/opt/vision`，改宿主机代码容器里立即生效；**注意镜像内容是 build 时的快照**，跑新代码要么重新 build、要么用 bind mount |
| 本机开发环境 | Ubuntu 22.04.5 LTS（与镜像底座一致）；宿主 apt 版 cv2 4.5.4 / numpy 1.21.5 / PyYAML 5.4.1 |
| Docker | `/usr/bin/docker`，Compose v5.5.1，免 sudo |
| 相机 | 海康 **MV-CU013-A0UC**：1280×1024、BayerGB8、USB3 Vision、全局快门、C 口（镜头约 6mm）；序列号 DA4976553 |
| 相机 SDK | 官方 **MVS 5.1.0**（`libMvCameraControl` 4.8.2.1，SDK 版本 0x04080202）。宿主机装在 `/opt/MVS`，安装包备份在 `~/MVS_installer/`；容器里由 Dockerfile 从 `.devcontainer/pkgs/*.deb` 装入，同一层还装 `libusb-1.0-0`（MVS 运行时 dlopen 它） |
| 相机权限 | 宿主机 `/etc/udev/rules.d/99-hikrobot.rules`：`SUBSYSTEM=="usb", ATTRS{idVendor}=="2bdf", MODE="0666", SYMLINK+="hik_camera"`（udev 必须在宿主机配，容器内改不了） |
| 容器设备 | compose 已配 `devices: ["/dev/bus/usb:/dev/bus/usb"]` —— 传 USB 总线（USB3 Vision 走 libusb），**不是 `/dev/video0`** |
| 镜像体积 | 不含 MVS：~862 MB；含 MVS（相机开发需要）：**约 2.36 GB**。不接相机的人不放 deb 即可，`build` 会自动跳过（见坑 27） |
