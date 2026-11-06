"""RGB-D 单帧工作流中的 YOLO 检测与 HSV 判色。

彩色图只转换一次 HSV；block 普通判色，top 按红蓝分布判断颜色。
"""
from dataclasses import dataclass, field
import logging

from ..types import Detection, Frame
from .color_check import DEFAULT_MIN_VALID_PIXELS, judge_color, judge_top_color

log = logging.getLogger("vision.pipeline")


@dataclass
class ColorConfig:
    """判色参数（来自 configs/color_thresholds.yaml 的 color 段）。"""

    min_saturation: int = 40
    min_value: int = 40
    hit_ratio_min: float = 0.50
    ignore_border_px: int = 0
    min_valid_pixels: int = DEFAULT_MIN_VALID_PIXELS

    def validate(self) -> None:
        if not 0 <= int(self.min_saturation) <= 255:
            raise ValueError(f"color.min_saturation 必须在 0~255：{self.min_saturation}")
        if not 0 <= int(self.min_value) <= 255:
            raise ValueError(f"color.min_value 必须在 0~255：{self.min_value}")
        if not 0.0 < float(self.hit_ratio_min) <= 1.0:
            raise ValueError(
                f"color.hit_ratio_min 必须在 (0, 1]（比例不是百分比）：{self.hit_ratio_min}")
        if int(self.ignore_border_px) < 0:
            raise ValueError(f"color.ignore_border_px 不能为负：{self.ignore_border_px}")
        if int(self.min_valid_pixels) < 1:
            raise ValueError(f"color.min_valid_pixels 至少为 1：{self.min_valid_pixels}")



@dataclass
class FrameResult:
    """同一帧的检测与判色结果；保留原框、模型类别和置信度。"""

    index: int
    source: str
    detections: list = field(default_factory=list)


def process_frame(frame: Frame, detector, thresholds: dict,
                  color_cfg: ColorConfig | None = None) -> FrameResult:
    """运行检测并给各框判色，返回同一张校正图上的结果。

    :param frame: BGR 彩色帧，须与后续深度使用同一个像素网格
    :param detector: 提供 detect(image_bgr) 的 YOLO 检测器
    :param thresholds: 已校验的 HSV 阈值表
    :param color_cfg: 可选有效像素和命中率门槛
    :return: FrameResult，无目标时 detections 为空列表
    """
    cfg = color_cfg or ColorConfig()
    detections = detector.detect(frame.image_bgr)

    color_targets = [d for d in detections if isinstance(d, Detection)]
    if color_targets:
        import cv2  # 局部 import：保持模块导入期不做重活
        hsv = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2HSV)
        for det in color_targets:
            # 塔顶是 红蓝上下叠放 的半砖：整框判色会被两色稀释，改判上半块
            is_top = str(getattr(det, "class_name", "")) == "top"
            kwargs = dict(
                min_saturation=cfg.min_saturation, min_value=cfg.min_value,
                hit_ratio_min=cfg.hit_ratio_min, ignore_border_px=cfg.ignore_border_px,
                min_valid_pixels=cfg.min_valid_pixels)
            # 塔顶红蓝上下叠放：先判上半块；判不出再看整框红蓝各占多少（mixed）
            verdict = (judge_top_color(hsv, det.bbox_xyxy, thresholds, **kwargs)
                       if is_top else judge_color(hsv, det.bbox_xyxy, thresholds, **kwargs))
            det.color = verdict.name
            det.color_ratio = verdict.ratio
            det.valid_pixels = verdict.valid_pixels
            if verdict.name is None:
                log.debug("帧%d 框%s 未判出颜色：%s（最高命中率 %.2f）",
                          frame.index, det.bbox_xyxy, verdict.reason, verdict.ratio)

    return FrameResult(index=frame.index, source=frame.source, detections=detections)
