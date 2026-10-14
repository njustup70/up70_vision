"""管线与合成源的测试：可重复性 +「18 个色块」这个自检数字。

自检期望值来自 configs/default.yaml：
    input.synthetic.num_frames=3、blocks_per_frame=6  ->  总色块 18 = 3 x 6
"""

import sys
from pathlib import Path

try:
    from up70_vision.core.detectors.base import build_detector
    from up70_vision.core.pipeline import (ColorConfig, match_to_truth, process_frame,
                                           process_frames, summarize)
    from up70_vision.core.color import parse_thresholds
    from up70_vision.core.types import Detection
    from up70_vision.io.synthetic import SyntheticSource
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from up70_vision.core.detectors.base import build_detector
    from up70_vision.core.pipeline import (ColorConfig, match_to_truth, process_frame,
                                           process_frames, summarize)
    from up70_vision.core.color import parse_thresholds
    from up70_vision.core.types import Detection
    from up70_vision.io.synthetic import SyntheticSource

import numpy as np

THRESHOLDS_RAW = {
    "red": [[0, 120, 70, 10, 255, 255], [170, 120, 70, 179, 255, 255]],
    "yellow": [[20, 120, 70, 40, 255, 255]],
    "green": [[45, 120, 70, 75, 255, 255]],
    "cyan": [[80, 120, 70, 100, 255, 255]],
    "blue": [[105, 120, 70, 135, 255, 255]],
    "magenta": [[135, 120, 70, 165, 255, 255]],
}


def _source(**kw):
    params = dict(width=320, height=240, num_frames=3, blocks_per_frame=6, seed=2027, noise_sigma=8.0)
    params.update(kw)
    return SyntheticSource(colors=list(THRESHOLDS_RAW), **params)


# ---------------------------------------------------------------- 合成源
def test_synthetic_is_reproducible():
    a = [f.image_bgr.copy() for f in _source()]
    b = [f.image_bgr.copy() for f in _source()]
    assert len(a) == 3
    for i in range(3):
        assert np.array_equal(a[i], b[i]), f"同一 seed 的第 {i} 帧应逐像素一致"


def test_synthetic_ground_truth_counts():
    frames = list(_source())
    assert len(frames) == 3
    for f in frames:
        assert len(f.ground_truth) == 6, f"每帧应有 6 个真值块，实得 {len(f.ground_truth)}"
        colors = [name for name, _ in f.ground_truth]
        assert len(set(colors)) == 6, "同一帧里颜色不应重复"


# ---------------------------------------------------------------- 检测器
def test_blob_detector_finds_six_blocks():
    det = build_detector("blob", {"min_area_px": 400, "morph_kernel": 5, "max_detections": 32})
    assert det.name == "blob"
    frame = list(_source())[0]
    found = det.detect(frame.image_bgr)
    assert len(found) == 6, f"应检出 6 个色块，实得 {len(found)}"
    for d in found:
        assert d.class_name == "unknown" and d.class_id == -1, "占位检测器按约定填 unknown/-1"
        x1, y1, x2, y2 = d.bbox_xyxy
        assert 0 <= x1 < x2 <= 320 and 0 <= y1 < y2 <= 240, "bbox 必须在图内且半开上界"


def test_unknown_detector_name_raises():
    try:
        build_detector("yolo")
    except ValueError as exc:
        assert "blob" in str(exc), "报错里应列出可用检测器"
    else:
        raise AssertionError("未知检测器应报错")


# ---------------------------------------------------------------- 匹配计分
def test_match_to_truth_numbers():
    dets = [Detection(bbox_xyxy=(0, 0, 10, 10), color="red"),
            Detection(bbox_xyxy=(100, 100, 110, 110), color="green")]
    truth = (("red", (0, 0, 10, 10)), ("blue", (100, 100, 110, 110)))
    matched, correct = match_to_truth(dets, truth, iou_min=0.5)
    assert (matched, correct) == (2, 1), f"应匹配 2 个、颜色对 1 个，实得 {(matched, correct)}"

    matched, correct = match_to_truth(dets, truth, iou_min=1.1)
    assert (matched, correct) == (0, 0), "门槛高于 1 时不该匹配"


# ---------------------------------------------------------------- 端到端
def test_pipeline_end_to_end_18_blocks():
    thresholds = parse_thresholds(THRESHOLDS_RAW)
    detector = build_detector("blob", {"min_area_px": 400, "morph_kernel": 5, "max_detections": 32})
    cfg = ColorConfig(min_saturation=40, min_value=40, hit_ratio_min=0.5,
                      ignore_border_px=0, min_valid_pixels=25)

    results = process_frames(_source(), detector, thresholds, cfg)
    summary = summarize(results)

    assert summary["frames"] == 3
    assert summary["detections"] == 18, f"总色块应为 3x6=18，实得 {summary['detections']}"
    assert summary["uncertain"] == 0, f"不该有判不出颜色的块：{summary['color_counts']}"
    assert summary["truth_total"] == 18
    assert summary["matched"] == 18, f"应与真值全部匹配，实得 {summary['matched']}"
    assert summary["color_correct"] == 18, f"颜色应全对，实得 {summary['color_correct']}"
    assert summary["color_accuracy"] == 1.0
    assert set(summary["color_counts"]) == set(THRESHOLDS_RAW), \
        f"六种颜色都应出现一次以上：{summary['color_counts']}"


def test_color_config_validation():
    try:
        ColorConfig(hit_ratio_min=1.5).validate()
    except ValueError as exc:
        assert "hit_ratio_min" in str(exc)
    else:
        raise AssertionError("hit_ratio_min > 1 应报错")

    try:
        ColorConfig(min_saturation=300).validate()
    except ValueError as exc:
        assert "min_saturation" in str(exc)
    else:
        raise AssertionError("min_saturation > 255 应报错")


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
    print(f"\n  test_pipeline: {len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
