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
        hits = int(np.count_nonzero(mask_for_segments(roi, segments) > 0))
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


#: 塔顶两种颜色各占多少才算"混色"（上下叠放时侧面看会各占一半）
TOP_MIXED_MIN_RATIO = 0.20

#: 上下两块颜色像素的"行中位数"差至少占框高的多少，才算上下叠放
#: 太小时红蓝可能在左右方向并排（相机横滚/侧视），不能按"上方"判
TOP_VERTICAL_MIN_SEP = 0.12


def _vertical_top_color(hsv, bbox, thresholds, min_sep: float, **kwargs):
    """按"谁在上面"判塔顶颜色（固定/正装相机的兜底）。

    不比较红蓝面积，而是比较两种颜色像素的**行中位数**：
    红色像素整体更靠上 → `red`，蓝色更靠上 → `blue`。
    如果两种颜色在中位数上几乎同高（左右并排/相机横滚），返回 None，
    交给调用方按 `mixed`/`None` 处理。

    只在"上半块判不出"时调用，不会覆盖已有的清晰单色结果。
    """
    red_segments = thresholds.get("red")
    blue_segments = thresholds.get("blue")
    if not red_segments or not blue_segments:
        return None
    height, width = hsv.shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox, width, height)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    roi = hsv[y1:y2, x1:x2]
    red_mask = mask_for_segments(roi, red_segments) > 0
    blue_mask = mask_for_segments(roi, blue_segments) > 0
    red_px = int(np.count_nonzero(red_mask))
    blue_px = int(np.count_nonzero(blue_mask))
    if red_px + blue_px < 4:
        return None
    # 只有一种颜色：不在这里下结论，留给 judge_color/整框逻辑
    if blue_px < 0.05 * red_px or red_px < 0.05 * blue_px:
        return None
    red_row = float(np.median(np.nonzero(red_mask)[0]))
    blue_row = float(np.median(np.nonzero(blue_mask)[0]))
    box_h = y2 - y1
    if abs(red_row - blue_row) < float(min_sep) * box_h:
        return None                      # 左右并排/横滚：没有"上/下"
    name = "red" if red_row < blue_row else "blue"
    ratio_kwargs = {key: kwargs[key] for key in
                    ("min_saturation", "min_value", "ignore_border_px")
                    if key in kwargs}
    ratios, n_valid = color_ratios(hsv, bbox, thresholds, **ratio_kwargs)
    return ColorVerdict(name=name, ratio=float(ratios.get(name, 0.0)),
                        valid_pixels=int(n_valid), ratios=ratios,
                        reason=f"vertical red_row={red_row:.0f} blue_row={blue_row:.0f}")


def judge_top_color(hsv, bbox, thresholds, upper_fraction: float = 0.5,
                    mixed_min: float = TOP_MIXED_MIN_RATIO,
                    vertical_min_sep: float = TOP_VERTICAL_MIN_SEP, **kwargs):
    """判"塔顶"（红蓝上下叠放的半砖）的颜色。

    四条路，按顺序：
      1. **上半块判得出单色** → 就是塔顶颜色（多数情况）；
      2. 上半块判不出、但红蓝有明显上下关系 → 取**上方**那块的颜色
         （红在半砖上面就是 `red`，不看哪种颜色面积大）；
      3. 上下关系也不明显、而整框红蓝各占够多 → `mixed`（左右并排/横滚时诚实报混色）；
      4. 都不满足 → `None`（看不清就报不确定）。

    :param hsv: 整帧 HSV 图
    :param bbox: `(x1,y1,x2,y2)` 像素
    :param thresholds: 颜色阈值表（`load_thresholds` 的返回）
    :param upper_fraction: 取上面多少（0.5 = 上半）
    :param mixed_min: 红、蓝占比都 ≥ 它才算混色
    :param vertical_min_sep: 红蓝行中位数差至少占框高的比例（默认 0.12）
    :param kwargs: 透传给 `judge_color`（min_saturation / hit_ratio_min 等）
    :return: `ColorVerdict`；`name` 可能是 `"red"` / `"blue"` / `"mixed"` / `None`
    """
    upper = judge_color(hsv, upper_part_bbox(bbox, upper_fraction), thresholds, **kwargs)
    if upper.name is not None:
        return upper

    vertical = _vertical_top_color(hsv, bbox, thresholds, vertical_min_sep, **kwargs)
    if vertical is not None:
        return vertical

    whole = judge_color(hsv, bbox, thresholds, **kwargs)
    red = float(whole.ratios.get("red", 0.0)) if getattr(whole, "ratios", None) else 0.0
    blue = float(whole.ratios.get("blue", 0.0)) if getattr(whole, "ratios", None) else 0.0
    if red >= mixed_min and blue >= mixed_min:
        return whole.__class__(name="mixed", ratio=max(red, blue),
                               valid_pixels=whole.valid_pixels, ratios=whole.ratios,
                               reason=f"红蓝各占 {red:.2f}/{blue:.2f}")
    return whole


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
