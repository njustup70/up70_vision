"""HSV 判色：给定候选框，判断框里是什么颜色、有多可信。

三条必须遵守的规则（README「团队约定」）
    1. H 范围是 0-179，不是 0-360
    2. 红色必须写两段阈值再取并集——色相在 0/179 处环绕
    3. 判色的分母是「有效像素」（饱和度/亮度达标的像素），不是 ROI 总像素；
       否则框里混进的地面背景会把命中率稀释掉

本模块是纯函数：输入 numpy 数组与阈值表，输出判色结论。
不读文件、不开窗口、不 import ROS——阈值表由 config 层读好后传进来。
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ..types import BBox, clip_bbox

# 一段阈值的六个分量：h_lo, s_lo, v_lo, h_hi, s_hi, v_hi
Segment = tuple[int, int, int, int, int, int]
H_MAX = 179
SV_MAX = 255

# 有效像素太少时不下结论（"看不清就报不确定"）
DEFAULT_MIN_VALID_PIXELS = 25


@dataclass
class ColorVerdict:
    """一个框的判色结论。

    :param name:  命中的颜色名；None 表示未命中（可能原因见 reason）
    :param ratio: 命中率 = 命中像素 / 有效像素
    :param valid_pixels: 有效像素数（分母）
    :param ratios: 每种颜色的命中率，便于调阈值时看"差多少"
    :param reason: 未命中/不确定的原因，便于日志排查
    """

    name: str | None
    ratio: float
    valid_pixels: int
    ratios: dict
    reason: str = "ok"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ratio": round(float(self.ratio), 4),
            "valid_pixels": int(self.valid_pixels),
            "ratios": {k: round(float(v), 4) for k, v in self.ratios.items()},
            "reason": self.reason,
        }


def parse_thresholds(raw: dict) -> dict:
    """把 YAML 里的阈值表转成内部结构，并做范围校验。

    输入形如::

        {"red": [[0,120,70,10,255,255], [170,120,70,179,255,255]],
         "green": [[45,120,70,75,255,255]]}

    输出::

        {"red": [(0,120,70,10,255,255), (170,120,70,179,255,255)], ...}

    :raises ValueError: 段数不是 1 或 2、分量个数不对、H/S/V 越界、h_lo > h_hi
    """
    if not isinstance(raw, dict) or not raw:
        raise ValueError("阈值表必须是非空 dict（颜色名 -> 段列表）")
    parsed = {}
    for name, segments in raw.items():
        if not isinstance(segments, (list, tuple)) or not (1 <= len(segments) <= 2):
            raise ValueError(
                f"颜色 {name!r} 的段数必须是 1 或 2（红色需要两段跨色相环绕），收到 {segments!r}")
        out = []
        for seg in segments:
            if not isinstance(seg, (list, tuple)) or len(seg) != 6:
                raise ValueError(
                    f"颜色 {name!r} 的每段必须是 6 个数 [h_lo,s_lo,v_lo,h_hi,s_hi,v_hi]，收到 {seg!r}")
            h_lo, s_lo, v_lo, h_hi, s_hi, v_hi = (int(v) for v in seg)
            if not (0 <= h_lo <= H_MAX and 0 <= h_hi <= H_MAX):
                raise ValueError(f"颜色 {name!r} 的 H 必须在 0~{H_MAX}（不是 0~360），收到 {seg!r}")
            for v in (s_lo, s_hi, v_lo, v_hi):
                if not 0 <= v <= SV_MAX:
                    raise ValueError(f"颜色 {name!r} 的 S/V 必须在 0~{SV_MAX}，收到 {seg!r}")
            if h_lo > h_hi:
                raise ValueError(
                    f"颜色 {name!r} 的 h_lo({h_lo}) 不能大于 h_hi({h_hi})；"
                    f"红色要跨环绕就写成两段（如 [170,...,179] 与 [0,...,10]）")
            out.append((h_lo, s_lo, v_lo, h_hi, s_hi, v_hi))
        parsed[str(name)] = tuple(out)
    return parsed


def mask_for_segments(hsv: np.ndarray, segments) -> np.ndarray:
    """单色掩膜：把该颜色的所有段取并集，返回 0/255 uint8 图。"""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h_lo, s_lo, v_lo, h_hi, s_hi, v_hi in segments:
        lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return mask


def valid_pixel_mask(hsv: np.ndarray, min_saturation: int, min_value: int) -> np.ndarray:
    """有效像素掩膜：饱和度与亮度都达标的像素才算"有颜色"。

    灰、黑、白、背景都会落在这之外，因此它们不参与命中率统计。
    """
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((s >= int(min_saturation)) & (v >= int(min_value))).astype(np.uint8) * 255
    return mask


def color_ratios(hsv: np.ndarray,
                 bbox: BBox,
                 thresholds: dict,
                 min_saturation: int = 40,
                 min_value: int = 40,
                 ignore_border_px: int = 0) -> tuple[dict, int]:
    """算框内每种颜色的命中率。

    :return: (每种颜色的命中率, 有效像素数)
             命中率 = 该颜色命中像素 / 有效像素；有效像素为 0 时全部返回 0.0
    """
    h, w = hsv.shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox, w, h)
    if ignore_border_px > 0:
        b = int(ignore_border_px)
        x1, y1, x2, y2 = x1 + b, y1 + b, x2 - b, y2 - b
    if x2 <= x1 or y2 <= y1:
        return {name: 0.0 for name in thresholds}, 0

    roi = hsv[y1:y2, x1:x2]
    valid = valid_pixel_mask(roi, min_saturation, min_value) > 0
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return {name: 0.0 for name in thresholds}, 0

    ratios = {}
    for name, segments in thresholds.items():
        # 命中必须同时是"有效像素"：否则颜色掩膜比有效像素更宽时（如蓝 V>=30 而
        # 有效像素 V>=40），命中数会大于分母，比例可能 >1，判色结果不可信。
        hits = int(np.count_nonzero((mask_for_segments(roi, segments) > 0) & valid))
        ratios[name] = hits / float(n_valid)
    return ratios, n_valid


def upper_part_bbox(bbox, fraction: float = 0.5):
    """取框的上半部分（`fraction` 为高度比例），用于"塔顶颜色看上面那半砖"。

    为什么需要：`top` 是 200 mm 半砖，红、蓝两块**上下叠放**；整框判色会因为两色各占
    一半而判成"未命中"或随视角漂。取上部（默认上半）判色，才对应"塔顶是红的还是蓝的"。
    向下兼容：`fraction=1.0` 就是原框。

    :param bbox: `(x1, y1, x2, y2)` 像素，半开上界
    :param fraction: 取框高的多少（0.5 = 上半）
    :return: 新的 bbox，y2 上移到 `y1 + h*fraction`
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction 必须在 (0, 1]：{fraction}")
    return (x1, y1, x2, y1 + (y2 - y1) * fraction)


#: 塔顶两种颜色各占多少才算"混色"（保留常量，供旧调用/文档引用）
TOP_MIXED_MIN_RATIO = 0.20

#: 红蓝像素的行/列中位数差至少占框高/框宽的多少，才算"上下"或"左右"排列
TOP_AXIS_MIN_SEP = 0.12


def _top_orientation(hsv, bbox, thresholds, min_sep: float,
                     min_saturation=40, min_value=40, ignore_border_px=0):
    """判断塔顶是"竖放"还是"横放"。

    - 竖放：红蓝上下排列（红在上 = red，蓝在上 = blue）；
    - 横放：红蓝左右排列（没有"哪块在上"，返回 mixed）；
    - 单色/斜放/看不清：orientation 为 single/ambiguous。

    只看红蓝像素的行/列中位数，不看面积。相机默认正装（图像上方=物理上方）。
    """
    red_segments = thresholds.get("red")
    blue_segments = thresholds.get("blue")
    if not red_segments or not blue_segments:
        return "none", None
    height, width = hsv.shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox, width, height)
    border = int(ignore_border_px)
    x1, y1, x2, y2 = x1+border, y1+border, x2-border, y2-border
    if x2 - x1 < 8 or y2 - y1 < 8:
        return "none", None
    roi = hsv[y1:y2, x1:x2]
    valid = valid_pixel_mask(roi, min_saturation, min_value) > 0
    red_mask = (mask_for_segments(roi, red_segments) > 0) & valid
    blue_mask = (mask_for_segments(roi, blue_segments) > 0) & valid
    red_px = int(np.count_nonzero(red_mask))
    blue_px = int(np.count_nonzero(blue_mask))
    if red_px + blue_px < 4:
        return "none", None
    if blue_px < 0.05 * red_px:
        return "single", "red"
    if red_px < 0.05 * blue_px:
        return "single", "blue"

    red_rows = np.nonzero(red_mask)[0]
    blue_rows = np.nonzero(blue_mask)[0]
    red_cols = np.nonzero(red_mask)[1]
    blue_cols = np.nonzero(blue_mask)[1]
    d_row = abs(float(np.median(red_rows)) - float(np.median(blue_rows))) / float(y2 - y1)
    d_col = abs(float(np.median(red_cols)) - float(np.median(blue_cols))) / float(x2 - x1)
    if d_row >= min_sep and d_row >= 1.2 * d_col:
        upper = "red" if np.median(red_rows) < np.median(blue_rows) else "blue"
        return "vertical", upper
    if d_col >= min_sep and d_col >= 1.2 * d_row:
        return "horizontal", None
    return "ambiguous", None


def judge_top_color(hsv, bbox, thresholds, upper_fraction: float = 0.5,
                    mixed_min: float = TOP_MIXED_MIN_RATIO,
                    axis_min_sep: float = TOP_AXIS_MIN_SEP, **kwargs):
    """判"塔顶"（红蓝半砖）的颜色/摆放方向。

    先判方向，再判颜色：
      1. **竖放**（红蓝上下）：取上面那块的颜色 → `red` / `blue`；
      2. **横放**（红蓝左右并排）：没有"哪块在上" → `mixed`；
      3. 单色：返回该颜色（方向无法区分，reason=single）；
      4. 斜放/看不清 → `None`。

    :param upper_fraction: 保留参数（兼容旧调用），当前不参与判定
    :param mixed_min: 保留参数（兼容旧调用），当前不参与判定
    :param axis_min_sep: 行/列中位数差至少占框高/框宽的比例
    :param kwargs: 有效像素、裁边和判定门槛；hit_ratio_min 用于红蓝合计占比，
                   不要求上方颜色占多数，min_valid_pixels 不足时返回 None
    :return: `ColorVerdict`；`name` 可能是 `"red"` / `"blue"` / `"mixed"` / `None`
    """
    ratio_kwargs = {key: kwargs[key] for key in
                    ("min_saturation", "min_value", "ignore_border_px")
                    if key in kwargs}
    ratios, n_valid = color_ratios(hsv, bbox, thresholds, **ratio_kwargs)
    if n_valid < int(kwargs.get('min_valid_pixels', DEFAULT_MIN_VALID_PIXELS)):
        return ColorVerdict(None, 0.0, int(n_valid), ratios, reason='too_few_valid_pixels')
    # 先确认框内确实有足够红蓝，再判断谁在上面；少量红噪点不能代表整个塔顶。
    combined = min(1.0, ratios.get('red', 0.0) + ratios.get('blue', 0.0))
    if combined < float(kwargs.get('hit_ratio_min', 0.50)):
        return ColorVerdict(None, combined, int(n_valid), ratios, reason='below_hit_ratio_min')
    orientation, payload = _top_orientation(hsv, bbox, thresholds, axis_min_sep, **ratio_kwargs)
    if orientation == "none":
        return ColorVerdict(None, 0.0, int(n_valid), ratios, reason="no_colour")
    if orientation == "single":
        name = payload
        return ColorVerdict(name, float(ratios.get(name, 0.0)), int(n_valid), ratios,
                            reason="single_colour")
    if orientation == "horizontal":
        return ColorVerdict("mixed", float(max(ratios.get("red", 0.0),
                                               ratios.get("blue", 0.0))),
                            int(n_valid), ratios, reason="horizontal")
    if orientation == "vertical":
        name = payload
        return ColorVerdict(name, float(ratios.get(name, 0.0)), int(n_valid), ratios,
                            reason="vertical")
    return ColorVerdict(None, 0.0, int(n_valid), ratios, reason="ambiguous_orientation")


def judge_color(hsv: np.ndarray,
                bbox: BBox,
                thresholds: dict,
                min_saturation: int = 40,
                min_value: int = 40,
                hit_ratio_min: float = 0.50,
                ignore_border_px: int = 0,
                min_valid_pixels: int = DEFAULT_MIN_VALID_PIXELS) -> ColorVerdict:
    """判断框内是什么颜色。

    判定顺序：有效像素够不够 -> 有没有颜色超过 hit_ratio_min -> 取命中率最高的那个。
    命中率并列时按阈值表里的键序取第一个（稳定可复现）。
    """
    ratios, n_valid = color_ratios(hsv, bbox, thresholds,
                                   min_saturation=min_saturation,
                                   min_value=min_value,
                                   ignore_border_px=ignore_border_px)
    if n_valid < int(min_valid_pixels):
        return ColorVerdict(None, 0.0, n_valid, ratios, reason="too_few_valid_pixels")

    best_name, best_ratio = None, 0.0
    for name in thresholds:
        r = ratios[name]
        if r > best_ratio:
            best_name, best_ratio = name, r
    if best_name is None or best_ratio < float(hit_ratio_min):
        return ColorVerdict(None, best_ratio, n_valid, ratios, reason="below_hit_ratio_min")
    return ColorVerdict(best_name, best_ratio, n_valid, ratios, reason="ok")
