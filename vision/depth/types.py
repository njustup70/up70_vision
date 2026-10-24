"""深度相机自己的数据类型：内参、抓取目标、算法参数。

根级 `types.py` 只放跨功能共用件（`Frame` / `Detection` / 2D 几何）；这里的
东西只有深度相机用，所以住在 `depth/` 里，与根级 `types.py` **互不 import**
（`vision/depth/README.md` 的约定）。

为什么单独定义 `Intrinsics` 而不是直接用 numpy 的 3x3 矩阵
    反投影只用到 fx/fy/cx/cy 四个数。传矩阵会让调用方去猜「这是 K 还是 P」
    「要不要先做畸变校正」，而这两个问题一旦猜错，点云整体比例就是错的。
    用具名 dataclass，读代码的人不需要知道 OpenCV 的矩阵排布。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """深度相机内参。像素单位，深度单位毫米。

    `width` / `height` 是**内参对应的图像尺寸**，不是当前这帧的尺寸。
    两者不一致时反投影会错位，所以 `core` 里会拿它做一次校验。
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


@dataclass(frozen=True)
class SuctionParams:
    """算法参数与可吸性判据。

    判据的六条初值来自计划书 3.3，都需要实测调整——尤其 `plane_thresh_mm`：
    A4 实测桌面本身就有 2~5mm 起伏，取 3σ（0.5m 处约 1mm）会把真实桌面点
    判成外点。这里给 6mm 是按实测留的余量。

    放 `vision/` 而不是 ROS 包，是因为 `core` 要读它；放进 ROS 包会让算法
    反向依赖 ROS，`pip install -e .` 之后非 ROS 环境直接不可用。
    """

    # ---- 预处理 ----
    roi_ratio: float = 0.6
    leaf_mm: float = 4.0
    ransac_iters: int = 200
    plane_thresh_mm: float = 6.0
    cluster_jump_mm: float = 18.0
    top_ratio: float = 0.2
    min_cluster_points: int = 300

    # ---- 可吸性判据（计划书 3.3）----
    plate_mm: float = 20.0
    min_surface_points: int = 120
    flat_distance_mm: float = 3.0
    flat_ratio_min: float = 0.8
    tilt_max_deg: float = 45.0
    hole_ratio_max: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 < self.roi_ratio <= 1.0:
            raise ValueError(f"roi_ratio 必须在 (0, 1]，收到 {self.roi_ratio}")
        if self.leaf_mm <= 0:
            raise ValueError(f"leaf_mm 必须为正，收到 {self.leaf_mm}")
        if not 0.0 < self.top_ratio <= 1.0:
            raise ValueError(f"top_ratio 必须在 (0, 1]，收到 {self.top_ratio}")


@dataclass(frozen=True)
class GraspTarget:
    """一个可吸附目标。

    约定（与 ROS 侧 `Detection3D` 的映射关系一并写死，避免两边各理解一套）
        `position_mm` 是**吸附点**——物块顶面中心，不是包围盒中心，也不是质心。
            包围盒中心会落在方块内部（含侧面），直接拿去吸会撞。
        `normal` 是顶面法向的单位向量，统一朝向相机一侧（与桌面法向同向）。
            方向不约定的话，下游算预吸附位时会往桌面里退。
        `bbox_size_mm` 是簇的包围盒尺寸，只用来做筛选和展示，不参与定位。
        `score` 是可吸性 0~1，取五条判据的最小值，不是加权平均。
    """

    position_mm: np.ndarray
    normal: np.ndarray
    score: float
    bbox_size_mm: np.ndarray
    points: int
    target_id: int = 0

    def __post_init__(self) -> None:
        position = np.asarray(self.position_mm, dtype=np.float64).reshape(-1)
        normal = np.asarray(self.normal, dtype=np.float64).reshape(-1)
        size = np.asarray(self.bbox_size_mm, dtype=np.float64).reshape(-1)
        if position.shape != (3,):
            raise ValueError(f"position_mm 必须是 3 元素，收到 {position.shape}")
        if normal.shape != (3,):
            raise ValueError(f"normal 必须是 3 元素，收到 {normal.shape}")
        if size.shape != (3,):
            raise ValueError(f"bbox_size_mm 必须是 3 元素，收到 {size.shape}")

        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            raise ValueError("normal 是零向量，无法确定朝向")
        normal = normal / norm

        score = float(self.score)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score 必须在 [0, 1]，收到 {score}")

        # frozen dataclass 改字段只能走 object.__setattr__
        object.__setattr__(self, "position_mm", position)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "bbox_size_mm", size)
        object.__setattr__(self, "score", score)

    def to_dict(self) -> dict:
        """转成可 JSON 序列化的 dict（落盘与日志用）。"""
        return {
            "id": int(self.target_id),
            "position_mm": [round(float(v), 2) for v in self.position_mm],
            "normal": [round(float(v), 4) for v in self.normal],
            "score": round(float(self.score), 3),
            "bbox_size_mm": [round(float(v), 1) for v in self.bbox_size_mm],
            "points": int(self.points),
        }
