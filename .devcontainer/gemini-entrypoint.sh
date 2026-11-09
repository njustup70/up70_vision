#!/usr/bin/env bash
# 在同一解释器与 ROS 环境中运行；传入的命令直接接收信号。
set -e
source /opt/ros/humble/setup.bash
source /opt/up70-ros/install/setup.bash
exec "$@"
