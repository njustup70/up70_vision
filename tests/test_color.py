"""core.color 的测试：阈值校验、红色跨色相环绕、有效像素分母、命中率门槛。

关键的两条（README「团队约定」第 2、3 条）在这里用具体数字钉住：
    · H 是 0-179，红色要写两段并在 0/179 处环绕
    · 判色分母是"有效像素"，不是 ROI 总像素
"""

import sys
from pathlib import Path

try:
    from up70_vision.core.color import (judge_color, mask_for_segments, parse_thresholds,
                                        color_ratios, valid_pixel_mask)
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from up70_vision.core.color import (judge_color, mask_for_segments, parse_thresholds,
                                        color_ratios, valid_pixel_mask)

import numpy as np

RED = {"red": [(0, 120, 70, 10, 255, 255), (170, 120, 70, 179, 255, 255)]}
RED_BLUE = {
    "red": [(0, 120, 70, 10, 255, 255), (170, 120, 70, 179, 255, 255)],
    "blue": [(105, 120, 70, 135, 255, 255)],
}


def _hsv(h: int, w: int, hue: int, sat: int = 255, val: int = 255) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = hue
    img[:, :, 1] = sat
    img[:, :, 2] = val
    return img


# ---------------------------------------------------------------- 阈值解析
def test_parse_thresholds_ok():
    parsed = parse_thresholds({"red": [[0, 120, 70, 10, 255, 255], [170, 120, 70, 179, 255, 255]]})
    assert parsed["red"] == ((0, 120, 70, 10, 255, 255), (170, 120, 70, 179, 255, 255))


def test_parse_thresholds_rejects_bad_input():
    for bad, keyword in [
        ({"red": [[10, 120, 70, 0, 255, 255]]}, "h_lo"),          # h_lo > h_hi
        ({"green": [[0, 120, 70, 200, 255, 255]]}, "179"),         # H 超 179
        ({"red": [[0, 120, 70, 10, 255, 255]] * 3}, "1 或 2"),      # 段数 3
        ({"red": [[0, 120, 70]]}, "6 个数"),                        # 分量不足
        ({"red": [[0, 120, 70, 10, 255, 300]]}, "255"),            # V 超 255
    ]:
        try:
            parse_thresholds(bad)
        except ValueError as exc:
            assert keyword in str(exc), f"报错信息里应含 {keyword!r}，实际：{exc}"
        else:
            raise AssertionError(f"应当报错：{bad}")


# ---------------------------------------------------------------- 红色环绕
def test_red_wraps_around_hue_zero():
    seg = parse_thresholds(RED)["red"]
    near_179 = mask_for_segments(_hsv(4, 4, 178), seg)
    near_0 = mask_for_segments(_hsv(4, 4, 5), seg)
    middle = mask_for_segments(_hsv(4, 4, 90), seg)
    assert int(near_179.sum()) > 0, "H=178 应判为红（第二段）"
    assert int(near_0.sum()) > 0, "H=5 应判为红（第一段）"
    assert int(middle.sum()) == 0, "H=90 不该判为红"


def test_judge_color_uses_valid_pixel_denominator():
    """一半红一半全黑：有效像素 50，命中 50 -> 命中率必须是 1.0（不是 0.5）。"""
    hsv = _hsv(10, 10, 0)
    hsv[:, 5:, 1] = 0          # 右半：饱和度 0（灰/黑，不算有效像素）
    hsv[:, 5:, 2] = 10         # 右半：亮度 10（低于 V 门槛）
    ratios, n_valid = color_ratios(hsv, (0, 0, 10, 10), parse_thresholds(RED),
                                   min_saturation=120, min_value=70)
    assert n_valid == 50, f"有效像素应为 50，实得 {n_valid}"
    assert abs(ratios["red"] - 1.0) < 1e-9, f"命中率应为 1.0，实得 {ratios['red']}"

    verdict = judge_color(hsv, (0, 0, 10, 10), parse_thresholds(RED),
                          min_saturation=120, min_value=70, hit_ratio_min=0.9)
    assert verdict.name == "red" and abs(verdict.ratio - 1.0) < 1e-9
    assert verdict.valid_pixels == 50


def test_judge_color_too_few_valid_pixels():
    hsv = np.zeros((10, 10, 3), np.uint8)      # 全黑：S=0 V=0，没有有效像素
    verdict = judge_color(hsv, (0, 0, 10, 10), parse_thresholds(RED),
                          min_saturation=120, min_value=70)
    assert verdict.name is None
    assert verdict.reason == "too_few_valid_pixels"
    assert verdict.valid_pixels == 0


def test_judge_color_below_hit_ratio_min():
    hsv = np.zeros((10, 20, 3), np.uint8)
    hsv[:, :10, 0] = 0         # 左半红
    hsv[:, 10:, 0] = 120       # 右半蓝
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = 255
    thresholds = parse_thresholds(RED_BLUE)
    ratios, n_valid = color_ratios(hsv, (0, 0, 20, 10), thresholds,
                                   min_saturation=120, min_value=70)
    assert n_valid == 200, f"有效像素应为 200，实得 {n_valid}"
    assert abs(ratios["red"] - 0.5) < 1e-9 and abs(ratios["blue"] - 0.5) < 1e-9

    strict = judge_color(hsv, (0, 0, 20, 10), thresholds,
                         min_saturation=120, min_value=70, hit_ratio_min=0.6)
    assert strict.name is None and strict.reason == "below_hit_ratio_min"

    loose = judge_color(hsv, (0, 0, 20, 10), thresholds,
                        min_saturation=120, min_value=70, hit_ratio_min=0.4)
    assert loose.name == "red" and abs(loose.ratio - 0.5) < 1e-9


def test_valid_pixel_mask_counts():
    hsv = _hsv(4, 4, 0)
    hsv[0, 0, 1] = 10          # 一个像素饱和度不足
    mask = valid_pixel_mask(hsv, min_saturation=120, min_value=70)
    assert int(np.count_nonzero(mask)) == 15


def _run_standalone() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
        except Exception as exc:                     # noqa: BLE001
            failed += 1
            print(f"  ❌ {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n  test_color: {len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
