"""框内平面反投影所需内参；深度使用毫米、内参使用像素。"""
from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)
class Intrinsics:
    """深度相机内参。像素单位，深度单位毫米。

    `width` / `height` 是**内参对应的图像尺寸**，不是当前这帧的尺寸。
    两者不一致时反投影会错位，所以图对测量时会拿它做一次校验。
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("fx", "fy"):
            value = float(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} 必须为正，收到 {value}；焦距填错会让点云整体缩放")
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError(f"图像尺寸必须为正，收到 {self.width}x{self.height}")

    @classmethod
    def from_camera_info(cls, k: list, width: int, height: int) -> "Intrinsics":
        """从 sensor_msgs/CameraInfo 的 k 数组构造。

        ROS 的 k 是行优先 3x3：[fx, 0, cx, 0, fy, cy, 0, 0, 1]。
        这里显式按下标取，避免有人误以为它和 OpenCV 的 Mat 一样是二维。
        """
        if len(k) < 6:
            raise ValueError(f"camera_info.k 至少要有 6 个元素，收到 {len(k)}")
        return cls(fx=float(k[0]), fy=float(k[4]), cx=float(k[2]), cy=float(k[5]),
                   width=int(width), height=int(height))
