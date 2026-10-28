"""深度相机功能包：深度图 -> 抓取目标（吸附点 + 法向 + 可吸性分数）。

对外只暴露两件事：
    `detect_grasp_targets(...)`  正常使用走这个，返回 `GraspTarget` 列表
    `analyze(...)`               调试用，额外返回平面方程与每个簇的明细

不 import ROS：`rclpy` 只能 apt 装，混进来会让 `pip install -e .` 直接失败。
ROS 侧（`ros2_ws/src/vision_depth`）只做收消息、调这里、发消息。
"""

from .core import (
    analyze,
    backproject,
    detect_grasp_targets,
    fit_plane_ransac,
    fit_top_face,
    label_clusters,
    score_suction,
    voxel_downsample,
)
from .types import GraspTarget, Intrinsics, SuctionParams

__all__ = [
    "GraspTarget",
    "Intrinsics",
    "SuctionParams",
    "analyze",
    "backproject",
    "detect_grasp_targets",
    "fit_plane_ransac",
    "fit_top_face",
    "label_clusters",
    "score_suction",
    "voxel_downsample",
]
