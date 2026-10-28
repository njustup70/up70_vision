"""深度相机功能包：深度图 -> 抓取目标（吸附点 + 法向 + 可吸性分数）。

对外只暴露两件事：
    `detect_grasp_targets(...)`  正常使用走这个，返回 `GraspTarget` 列表
    `analyze(...)`               调试用，额外返回平面方程与每个簇的明细

这里**不再**导出 `backproject` / `voxel_downsample` / `fit_plane_ransac` /
`label_clusters` / `fit_top_face` / `score_suction`。它们是实现细节：
    · 纯几何运算住在 `vision.depth.primitives`，需要时显式 import
    · 抓取业务内部步骤留在 `vision.depth.core`
导出它们会让「这个包对外承诺了什么」变模糊，也会诱使外部代码直接调内部步骤，
以后改签名时牵连一片。

不 import ROS：`rclpy` 只能 apt 装，混进来会让 `pip install -e .` 直接失败。
ROS 侧（`ros2_ws/src/vision_depth`）只做收消息、调这里、发消息。
"""

from .core import analyze, detect_grasp_targets
from .types import GraspTarget, Intrinsics, SuctionParams

__all__ = [
    "GraspTarget",
    "Intrinsics",
    "SuctionParams",
    "analyze",
    "detect_grasp_targets",
]
