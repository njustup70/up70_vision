"""管线与合成源的测试：可重复性 +「18 个色块」这个自检数字。

自检期望值来自 configs/default.yaml：
    input.synthetic.num_frames=3、blocks_per_frame=6  ->  总色块 18 = 3 x 6
"""

import sys
from pathlib import Path

try:
    from vision.cli.run import build_detector
    from vision.yolo.pipeline import (ColorConfig, match_to_truth, process_frame,
                                           process_frames, summarize)
    from vision.yolo.color_check import parse_thresholds
    from vision.aruco.types import ArucoMarker, is_marker
    from vision.types import Detection
    from vision.sources.synthetic import SyntheticSource
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vision.cli.run import build_detector
    from vision.yolo.pipeline import (ColorConfig, match_to_truth, process_frame,
                                           process_frames, summarize)
    from vision.yolo.color_check import parse_thresholds
    from vision.aruco.types import ArucoMarker, is_marker
    from vision.types import Detection
    from vision.sources.synthetic import SyntheticSource

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
    # 同上：之前用 "yolo" 当反例，yolo 落地后必须换成真不存在的名字
    try:
        build_detector("no_such_detector")
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


# ---------------------------------------------------------------- ArUco（stub）
def test_stub_aruco_returns_contract_markers():
    """stub 必须按 docs/interface.md 第 2.2 节返回 ArucoMarker，且四条约定都成立。"""
    detector = build_detector("stub_aruco", {"ids": [0, 1, 2], "side_px": 90, "moving": True})
    assert detector.name == "stub_aruco"
    frame = list(_source())[0]

    markers = detector.detect(frame.image_bgr)
    assert len(markers) == 3, f"应伪造 3 个码，实得 {len(markers)}"
    assert all(isinstance(m, ArucoMarker) for m in markers), "返回的必须是 ArucoMarker"
    assert [m.marker_id for m in markers] == [0, 1, 2], "契约 2.2 第 2 条：按 marker_id 升序"
    for m in markers:
        assert m.corners_px.dtype == np.float32 and m.corners_px.shape == (4, 2)
        # 契约 2.2 第 3 条：center_px 是四角算术平均
        mean = m.corners_px.astype(np.float64).mean(axis=0)
        assert abs(m.center_px[0] - mean[0]) < 1e-3 and abs(m.center_px[1] - mean[1]) < 1e-3, \
            f"中心应是四角均值：{m.center_px} vs {tuple(mean)}"

    # 宽度 320 时三个码横排：中心 y 都在中线 120，第二个固定在 (160, 120)
    assert markers[1].center_px == (160.0, 120.0), f"中间的码应居中：{markers[1].center_px}"
    # moving=True：第一个码在两帧之间动过（stub 靠内部帧计数）
    again = detector.detect(frame.image_bgr)
    assert again[0].center_px != markers[0].center_px, "moving=True 时第一个码每帧应移动"


def test_stub_aruco_moves_only_the_first_marker():
    detector = build_detector("stub_aruco", {"ids": (0, 1, 2), "side_px": 90,
                                            "moving": True, "radius_px": 60})
    frame = list(_source())[0]
    first = detector.detect(frame.image_bgr)
    second = detector.detect(frame.image_bgr)
    assert first[1].center_px == second[1].center_px == (160.0, 120.0), "第二个码不动"
    assert first[2].center_px == second[2].center_px, "第三个码不动"
    # 布局：宽 320、gap = 90*1.6 = 144、三个码 -> 基准 x = 16 / 160 / 304，y 都在 120。
    # 第一个码每帧转 12°、半径 60：第 1 帧在 0°，第 2 帧在 12°。
    assert abs(first[0].center_px[0] - (16.0 + 60.0)) < 1e-6, f"{first[0].center_px}"
    assert abs(second[0].center_px[0] - (16.0 + 60.0 * np.cos(np.radians(12)))) < 1e-3
    assert abs(second[0].center_px[1] - (120.0 + 60.0 * np.sin(np.radians(12)))) < 1e-3


def test_pipeline_skips_color_for_markers():
    """ArUco 码没有颜色语义：不应被判色、不该算"未判定"，也不参与色块真值匹配。"""
    detector = build_detector("stub_aruco", {"ids": [0, 1, 2]})
    thresholds = parse_thresholds(THRESHOLDS_RAW)
    frame = list(_source())[0]                 # 合成帧带 6 个色块真值

    result = process_frame(frame, detector, thresholds, ColorConfig())
    assert result.ids == [0, 1, 2]
    assert len(result.detections) == 3
    assert result.color_counts() == {}, "码不该产生颜色分布"
    assert result.uncertain == 0, "码不是'判不出颜色'，不该计入未判定"
    assert (result.matched, result.color_correct) == (0, 0), "码不参与色块真值匹配"
    assert result.truth_total == 6, "真值仍来自合成源"


def test_pipeline_end_to_end_stub_aruco():
    thresholds = parse_thresholds(THRESHOLDS_RAW)
    detector = build_detector("stub_aruco", {"ids": [0, 1, 2]})
    results = process_frames(_source(), detector, thresholds, ColorConfig())
    summary = summarize(results)

    assert summary["frames"] == 3
    assert summary["detections"] == 9, f"3 帧 x 3 码 = 9，实得 {summary['detections']}"
    assert summary["ids"] == [0, 1, 2]
    assert summary["color_counts"] == {} and summary["uncertain"] == 0
    assert summary["mean_ratio"] == 0.0, "没有判色就不该有命中率"


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
