"""核心数据结构：Frame、Detection、ArucoMarker，以及 IoU 等纯函数。

这一层是整个包的地基，必须能被单独验证——没有相机、没有文件、不开窗口，
给它 numpy 数组就能算出结果。

两个"检出物"结构（别混用，契约见 vision/docs/interface.md 第 2 节）
    · `Detection`    判色族的产出：轴对齐 `bbox_xyxy`，中心 = bbox 中心，带颜色字段
    · `ArucoMarker`  ArUco 码：四角点 `corners_px`，中心 = 四角算术平均，没有颜色

约定（README「团队约定」）
    1. 图像一律 BGR 顺序，变量名带颜色空间后缀（image_bgr）
    2. 检测框用绝对像素坐标，右/下边界是**半开上界**：bbox=(x1,y1,x2,y2) 覆盖
       x1..x2-1、y1..y2-1，因此宽 = x2-x1，高 = y2-y1
"""

from dataclasses import dataclass, field

import numpy as np

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Frame:
    """一帧图像 + 它的元信息。

    :param image_bgr: HxWx3 uint8 BGR 数组
    :param index: 帧序号，从 0 开始
    :param source: 来源标识（"synthetic" / "replay:xx.mp4" / "camera:mvs"）
    :param stamp: 采集时间戳（秒）；合成/回放没有真实时间戳时为 0.0
    :param ground_truth: 仅合成源提供，格式 ((color_name, (x1,y1,x2,y2)), ...)
    """

    image_bgr: np.ndarray
    index: int = 0
    source: str = ""
    stamp: float = 0.0
    ground_truth: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if self.image_bgr is None or self.image_bgr.ndim != 3 or self.image_bgr.shape[2] != 3:
            shape = None if self.image_bgr is None else self.image_bgr.shape
            raise ValueError(f"Frame.image_bgr 必须是 HxWx3 的 BGR 图，收到 {shape}")

    @property
    def height(self) -> int:
        return int(self.image_bgr.shape[0])

    @property
    def width(self) -> int:
        return int(self.image_bgr.shape[1])

    @property
    def size(self) -> tuple[int, int]:
        """(宽, 高)"""
        return self.width, self.height

    def has_ground_truth(self) -> bool:
        return len(self.ground_truth) > 0


@dataclass
class Detection:
    """一个候选框 + 它的判色结果。

    bbox 恒为绝对像素坐标、半开上界；class_name/class_id 在占位检测器里
    固定为 "unknown" / -1（"是什么颜色"由 color.py 填）。
    """

    bbox_xyxy: BBox
    class_name: str = "unknown"
    class_id: int = -1
    score: float = 1.0
    color: str | None = None
    color_ratio: float = 0.0
    valid_pixels: int = 0

    @property
    def width(self) -> int:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> int:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def to_dict(self) -> dict:
        """转成可 JSON 序列化的 dict（下游/落盘用）。"""
        cx, cy = self.center
        return {
            "bbox_xyxy": [int(v) for v in self.bbox_xyxy],
            "class_name": self.class_name,
            "class_id": int(self.class_id),
            "score": round(float(self.score), 4),
            "color": self.color,
            "color_ratio": round(float(self.color_ratio), 4),
            "valid_pixels": int(self.valid_pixels),
            "center": [round(cx, 2), round(cy, 2)],
        }


def iou(a: BBox, b: BBox) -> float:
    """两个半开上界框的交并比。

    完美重合 = 1.0，不相交 = 0.0，面积为零的框一律按 0.0 处理（避免除零）。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    """把框裁到图像范围内（半开上界：x2 最大为 width）。"""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    return (x1, y1, x2, y2)


def bbox_from_xywh(x: int, y: int, w: int, h: int) -> BBox:
    """(x, y, w, h) -> (x1, y1, x2, y2)，半开上界。"""
    return (int(x), int(y), int(x + w), int(y + h))


def is_marker(det) -> bool:
    """判断一个检出物是不是 ArUco 码（门面与 CLI 都用它分流两种产出）。

    用鸭子类型（有 `marker_id` 就是码）而不是 isinstance：
    根级 `types.py` 是叶子，不能 import 功能包；而 `ArucoMarker` 住在 `aruco/types.py`。
    """
    return hasattr(det, "marker_id")
