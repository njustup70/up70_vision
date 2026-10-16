"""按契约返回假 ArUco 码的 stub（任务单第一节第 4 条：让两边互不阻塞）。

为什么需要它
    "输入源 + 可视化 + 编排"不该等真 ArUco 实现写完才开始。stub 按
    docs/interface.md 第 2.2 / 3.2 节的契约返回假的 `ArucoMarker`，
    这样相机、画框、入口、落盘都能先跑通；真实现（`aruco.py`）就位后直接换掉。

契约（原文见 docs/interface.md 第 2.2 与 3.2 节）
    · 返回 `list[ArucoMarker]`，**无码时 `[]`**（不是 None）
    · 按 `marker_id` **升序**
    · `center_px` = 四角**算术平均**（不是 bbox 中心）
    · `corners_px` 顺时针、起点是码自身坐标系的左上角，dtype float32
    · `detect(image_bgr)` 形参名固定

⚠️ **调用方每帧必须恰好调用一次 `detect`**
    本 stub 用**自身的帧计数器**决定"现在是第几帧、码转到哪儿"（签名里没有帧号）。
    隔帧调用、或一帧调两次，都会让运动错位——症状是"画面里码的位置不对"，**不报错**。

行为
    · 默认造 3 个码，id 固定 0/1/2，横排居中
    · moving=True（默认）时第一个码绕自己的基准位置转圈 + 自转，
      角点是真的转过角度的四边形，用来验证"码移动 / 旋转时持续更新"
"""

import math

import numpy as np

from ..types import ArucoMarker
from .base import Detector


class StubArucoDetector(Detector):
    name = "stub_aruco"

    def __init__(self,
                 ids=(0, 1, 2),
                 side_px: int = 90,
                 moving: bool = True,
                 radius_px: int = 60,
                 deg_per_frame: float = 12.0):
        """
        :param ids: 要伪造的码 ID 列表（顺序即从左到右）
        :param side_px: 码边长（像素）
        :param moving: 第一个码是否动起来（转圈 + 自转）
        :param radius_px: 圆周运动半径
        :param deg_per_frame: 每帧转过的角度（决定运动速度）
        """
        self.ids = [int(i) for i in ids]
        self.side_px = int(side_px)
        self.moving = bool(moving)
        self.radius_px = int(radius_px)
        self.deg_per_frame = float(deg_per_frame)
        self.frame_count = 0
        if not self.ids:
            raise ValueError("ids 不能为空（stub 至少要伪造一个码）")
        if self.side_px <= 0:
            raise ValueError(f"side_px 必须为正：{self.side_px}")
        if self.radius_px < 0:
            raise ValueError(f"radius_px 不能为负：{self.radius_px}")

    def detect(self, image_bgr: np.ndarray) -> list:
        height, width = image_bgr.shape[:2]
        gap = self.side_px * 1.6
        total = gap * (len(self.ids) - 1)
        x_start = width / 2.0 - total / 2.0
        y_mid = height / 2.0
        angle = (self.frame_count * self.deg_per_frame) if self.moving else 0.0

        markers = []
        for index, marker_id in enumerate(self.ids):
            cx = x_start + index * gap
            cy = y_mid
            spin = 0.0
            if index == 0 and self.moving:
                rad = math.radians(angle)
                cx += self.radius_px * math.cos(rad)
                cy += self.radius_px * math.sin(rad)
                spin = angle                      # 第一个码同时自转，角点不再是轴对齐
            markers.append(self._marker(marker_id, cx, cy, spin))

        markers.sort(key=lambda m: m.marker_id)   # 契约 2.2：按 marker_id 升序
        self.frame_count += 1
        return markers

    def _marker(self, marker_id: int, cx: float, cy: float, spin_deg: float) -> ArucoMarker:
        """按 中心 + 边长 + 自转角 造四角点，顺序 TL → TR → BR → BL（码自身坐标系）。"""
        half = self.side_px / 2.0
        base = np.array([[-half, -half], [half, -half], [half, half], [-half, half]],
                        dtype=np.float64)
        rad = math.radians(spin_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
        corners = (base @ rot.T) + np.array([cx, cy], dtype=np.float64)
        # 契约 2.2 第 3 条：center_px 是四角算术平均
        center = corners.mean(axis=0)
        return ArucoMarker(marker_id=int(marker_id),
                           corners_px=corners.astype(np.float32),
                           center_px=(float(center[0]), float(center[1])))
