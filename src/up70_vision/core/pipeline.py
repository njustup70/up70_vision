"""逐帧管线：检测 -> 判色 -> 匹配计分。

依赖谁
    只依赖 core.types / core.color / core.detectors。**绝不** import io / viz / cli，
    也不 import config（上层把配置拆成参数传进来）。
    这样"判色对不对"能在没有相机、没有机器人的情况下几秒钟验完。

匹配计分是干什么的
    合成源会附带 ground truth（每个色块的真实颜色与位置）。把检测结果与它对上，
    就能算出"检出几个、颜色判对几个"，这是 selfcheck 与回归测试的数字来源。
    IoU 门槛默认 0.5。
"""

from dataclasses import dataclass, field

from ..utils.logging import get_logger
from .color import DEFAULT_MIN_VALID_PIXELS, judge_color
from .types import ArucoMarker, Detection, Frame, iou

log = get_logger("core.pipeline")

DEFAULT_IOU_MIN = 0.5


@dataclass
class ColorConfig:
    """判色参数（来自 configs/default.yaml 的 color 段）。"""

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
    """一帧的处理结果。"""

    index: int
    source: str
    detections: list = field(default_factory=list)
    truth_total: int = 0
    matched: int = 0
    color_correct: int = 0

    @property
    def uncertain(self) -> int:
        """没判出颜色的框数——这些要报"不确定"，不能当已知颜色用。

        只数判色族的 `Detection`：ArUco 码是黑白图形、没有颜色语义，
        跳过判色也不算"不确定"（契约 docs/interface.md 第 2.2 / 3.2 节）。
        """
        return sum(1 for d in self.detections if isinstance(d, Detection) and d.color is None)

    @property
    def ids(self) -> list:
        """本帧检出的 ArUco 码 ID（升序）。判色族的框没有 ID，返回 `[]`。"""
        return sorted(int(d.marker_id) for d in self.detections if isinstance(d, ArucoMarker))

    def color_counts(self) -> dict:
        counts = {}
        for d in self.detections:
            if not isinstance(d, Detection):
                continue                      # 码没有颜色语义，不进颜色分布
            key = d.color if d.color is not None else "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "source": self.source,
            "count": len(self.detections),
            "ids": self.ids,
            "truth_total": self.truth_total,
            "matched": self.matched,
            "color_correct": self.color_correct,
            "uncertain": self.uncertain,
            "color_counts": self.color_counts(),
            "detections": [d.to_dict() for d in self.detections],
        }


def match_to_truth(detections, ground_truth, iou_min: float = DEFAULT_IOU_MIN):
    """把检测框与 ground truth 一对一匹配（贪心，按 IoU 从大到小）。

    :param ground_truth: ((color_name, bbox), ...)
    :return: (匹配对数, 其中颜色判对的个数)
    """
    if not detections or not ground_truth:
        return 0, 0
    pairs = []
    for di, det in enumerate(detections):
        if not isinstance(det, Detection):
            continue                     # 真值是色块的框；ArUco 码不参与匹配
        for gi, (color, bbox) in enumerate(ground_truth):
            value = iou(det.bbox_xyxy, bbox)
            if value >= float(iou_min):
                pairs.append((value, di, gi))
    pairs.sort(reverse=True)

    used_det, used_gt, matched, correct = set(), set(), 0, 0
    for value, di, gi in pairs:
        if di in used_det or gi in used_gt:
            continue
        used_det.add(di)
        used_gt.add(gi)
        matched += 1
        if detections[di].color == ground_truth[gi][0]:
            correct += 1
    return matched, correct


def process_frame(frame: Frame,
                  detector,
                  thresholds: dict,
                  color_cfg: ColorConfig | None = None,
                  iou_min: float = DEFAULT_IOU_MIN) -> FrameResult:
    """检测 -> 判色 -> 与 ground truth 匹配计分。

    判色只对判色族的 `Detection` 做：ArUco 码是黑白图形、没有颜色语义，
    跳过它（契约 docs/interface.md 第 2.2 / 3.2 节）。因此接 ArUco 检测器时
    不需要阈值表也能跑，`color_counts()` 会是空的。
    """
    cfg = color_cfg or ColorConfig()
    detections = detector.detect(frame.image_bgr)

    color_targets = [d for d in detections if isinstance(d, Detection)]
    if color_targets:
        import cv2  # 局部 import：保持模块导入期不做重活
        hsv = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2HSV)
        for det in color_targets:
            verdict = judge_color(
                hsv, det.bbox_xyxy, thresholds,
                min_saturation=cfg.min_saturation,
                min_value=cfg.min_value,
                hit_ratio_min=cfg.hit_ratio_min,
                ignore_border_px=cfg.ignore_border_px,
                min_valid_pixels=cfg.min_valid_pixels)
            det.color = verdict.name
            det.color_ratio = verdict.ratio
            det.valid_pixels = verdict.valid_pixels
            if verdict.name is None:
                log.debug("帧%d 框%s 未判出颜色：%s（最高命中率 %.2f）",
                          frame.index, det.bbox_xyxy, verdict.reason, verdict.ratio)

    result = FrameResult(index=frame.index, source=frame.source, detections=detections)
    if frame.has_ground_truth():
        result.truth_total = len(frame.ground_truth)
        result.matched, result.color_correct = match_to_truth(
            detections, frame.ground_truth, iou_min=iou_min)
    log.debug("帧%d 检出%d 个（未判定 %d）", frame.index, len(detections), result.uncertain)
    return result


def process_frames(frames,
                   detector,
                   thresholds: dict,
                   color_cfg: ColorConfig | None = None,
                   iou_min: float = DEFAULT_IOU_MIN,
                   on_result=None) -> list:
    """对一串 Frame 跑完整管线。

    :param frames: 可迭代的 Frame（由 io 层提供；本模块不认识 FrameSource）
    :param on_result: 可选回调 process_frame 之后调用，用于边跑边画图/落盘
    """
    results = []
    for frame in frames:
        result = process_frame(frame, detector, thresholds, color_cfg, iou_min=iou_min)
        results.append(result)
        if on_result is not None:
            on_result(frame, result)
    return results


def summarize(results) -> dict:
    """汇总多帧结果：总检出、id 集合、颜色分布、未判定数、ground truth 命中情况。"""
    total = sum(len(r.detections) for r in results)
    uncertain = sum(r.uncertain for r in results)
    counts: dict = {}
    ratios = []
    ids = set()
    for r in results:
        ids.update(r.ids)
        for d in r.detections:
            if not isinstance(d, Detection):
                continue                     # ArUco 码没有颜色，不进颜色分布
            key = d.color if d.color is not None else "unknown"
            counts[key] = counts.get(key, 0) + 1
            ratios.append(d.color_ratio)
    truth_total = sum(r.truth_total for r in results)
    matched = sum(r.matched for r in results)
    color_correct = sum(r.color_correct for r in results)
    return {
        "frames": len(results),
        "detections": total,
        "ids": sorted(ids),
        "uncertain": uncertain,
        "color_counts": counts,
        "mean_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
        "truth_total": truth_total,
        "matched": matched,
        "color_correct": color_correct,
        "color_accuracy": (round(color_correct / matched, 4) if matched else None),
    }
