"""真 ArUco 检测器的测试：契约形状 + 任务单验收第一条的**具体数字**。

验的是什么（对应任务单验收清单）
    "同一张图片中的多个 ArUco 码全部正确识别（ID、四角、中心）"
    "画面无码时程序正常运行，不异常退出"
    所以这里不写 `assert len(dets) > 0`，而是钉住具体数字：
        · 多码图 multi_grid_6：6 个码、id 恰好 {0..5}、中心与真值**完全相等**
        · 四角误差 ≤ 0.25 px（实测最大 0.23 px），并且四角顺序（TL→TR→BR→BL）
          用"上边的倾角"单独验一遍——顺序拧了不会报错，只会让 solvePnP 全错
        · 无码图返回**空列表**（不是 None、不抛异常）
    真值来自 assets/aruco/*.json，由 tools/aruco/gen_scenes.py 生成（重跑逐字节一致）。

环境要求
    需要 OpenCV >= 4.7 的新版 aruco API。容器（cv2 4.14）里全跑；
    宿主 apt 版 cv2 4.5.4 没有这个 API，会自动**跳过**并打一句提示，不算失败。
    跑法：docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision \
              python tests/test_aruco.py
"""

import json
import math
import sys
import unittest
from pathlib import Path

try:
    from vision.config.schema import ConfigError, from_dict
    from vision.detectors.aruco import ArucoDetector
    from vision.detectors.base import build_detector
    from vision.types import ArucoMarker
    from vision.tools.aruco.gen_scenes import (load_truth, scene_rotated)
    from vision.image_io import read_image_bgr
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vision.config.schema import ConfigError, from_dict
    from vision.detectors.aruco import ArucoDetector
    from vision.detectors.base import build_detector
    from vision.types import ArucoMarker
    from vision.tools.aruco.gen_scenes import (load_truth, scene_rotated)
    from vision.image_io import read_image_bgr

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "aruco"

# 实测的最大四角误差：多码 0.14 px、旋转 0.23 px。容差留一点余量写 0.25 px，
# 既能挡住"顺序拧了"（那会跳到几百 px），也不会因为插值差几百分之一就红。
MAX_CORNER_ERR_PX = 0.25

ARUCO_READY = hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector")


def _require_aruco() -> None:
    if not ARUCO_READY:
        raise unittest.SkipTest(
            f"cv2 {cv2.__version__} 没有新版 aruco API（需 >= 4.7）。"
            "在容器里跑：docker compose -f .devcontainer/docker-compose.yml run --rm "
            "up70-vision python tests/test_aruco.py")
    if not ASSETS.is_dir():
        raise unittest.SkipTest(
            f"测试图组不存在：{ASSETS}。先生成：python -m vision.tools.aruco.gen_scenes")


def _load(name: str):
    """载入图组里的一张图 + 它的真值。"""
    image_path = ASSETS / f"{name}.png"
    return read_image_bgr(str(image_path)), load_truth(str(image_path))


def _by_id(detections, marker_id):
    hits = [d for d in detections if int(d.marker_id) == int(marker_id)]
    assert len(hits) == 1, f"id={marker_id} 应恰好检出 1 个，实得 {len(hits)} 个"
    return hits[0]


def _corner_error(det, truth_marker) -> float:
    """按下标一一对应的最大角点误差（顺序错了这里就会很大）。"""
    want = np.asarray(truth_marker["corners"], dtype=np.float64)
    have = np.asarray(det.corners_px, dtype=np.float64)
    return float(np.max(np.linalg.norm(want - have, axis=1)))


def _top_edge_angle_deg(det) -> float:
    """四角里 0→1 那条边（码的"上边"）在图像里的倾角，单位度。"""
    pts = np.asarray(det.corners_px, dtype=np.float64)
    return math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))


# ---------------------------------------------------------------- 契约形状
def test_detector_uses_the_dictionary_from_the_task():
    _require_aruco()
    detector = ArucoDetector()
    assert detector.name == "aruco"
    assert detector.dictionary_name == "DICT_4X4_50", \
        f"任务单指定 DICT_4X4_50，实得 {detector.dictionary_name}"
    assert detector.corner_refine == "subpix"


def test_detections_satisfy_the_contract_and_are_json_ready():
    """契约 2.2 的四条：ArucoMarker、float32 四角、四角均值为中心、按 id 升序且可 JSON 化。"""
    _require_aruco()
    image_bgr, truth = _load("scenes/multi_row_3")
    markers = ArucoDetector().detect(image_bgr)
    assert len(markers) == 3

    for m in markers:
        assert isinstance(m, ArucoMarker), "契约 3.2：detect() 返回 list[ArucoMarker]"
        assert type(m.marker_id) is int, f"marker_id 必须是 Python int，实得 {type(m.marker_id)}"
        assert m.corners_px.dtype == np.float32, "契约 2.2：corners_px 是 float32"
        assert m.corners_px.shape == (4, 2)
        assert isinstance(m.center_px, tuple) and len(m.center_px) == 2
        assert all(type(v) is float for v in m.center_px), "契约 2.2：center_px 是浮点元组"
        # 契约 2.2 第 3 条：center_px 是四角算术平均（不是 bbox 中心）
        mean = m.corners_px.astype(np.float64).mean(axis=0)
        assert abs(m.center_px[0] - mean[0]) < 1e-3 and abs(m.center_px[1] - mean[1]) < 1e-3

    # numpy 标量会让 json.dump 直接 TypeError；契约要求的是 Python int / float
    text = json.dumps([{"marker_id": m.marker_id, "center_px": list(m.center_px)}
                       for m in markers])
    assert json.loads(text)[0]["marker_id"] == 0
    assert json.loads(text)[0]["center_px"] == [190.0, 260.0]


def test_detect_rejects_non_bgr_input():
    _require_aruco()
    detector = ArucoDetector()
    gray = np.zeros((120, 160), dtype=np.uint8)
    try:
        detector.detect(gray)
    except ValueError as exc:
        assert "HxWx3" in str(exc), f"报错应说清形状要求，实际：{exc}"
    else:
        raise AssertionError("单通道灰度图应当被拒绝（契约要求 HxWx3 BGR）")

    try:
        detector.detect(np.zeros((120, 160, 3), dtype=np.float32))
    except ValueError as exc:
        assert "uint8" in str(exc), f"报错应提到 uint8，实际：{exc}"
    else:
        raise AssertionError("float32 图应当被拒绝（契约要求 uint8）")


# ---------------------------------------------------------------- 多码图
def test_multi_row_3_ids_and_centers_are_exact():
    _require_aruco()
    image_bgr, truth = _load("scenes/multi_row_3")
    detections = ArucoDetector().detect(image_bgr)

    assert [int(d.marker_id) for d in detections] == [0, 1, 2], "输出按 id 升序，且三个都要检出"
    for marker in truth["markers"]:
        det = _by_id(detections, marker["id"])
        assert list(det.center_px) == [float(v) for v in marker["center"]], \
            f"id={marker['id']} 中心应与真值完全相等：{det.center_px} vs {marker['center']}"
        assert _corner_error(det, marker) <= MAX_CORNER_ERR_PX


def test_multi_grid_6_finds_all_six_with_small_corner_error():
    _require_aruco()
    image_bgr, truth = _load("scenes/multi_grid_6")
    detections = ArucoDetector().detect(image_bgr)

    assert len(detections) == 6, f"3x2 网格应检出 6 个，实得 {len(detections)}"
    assert [int(d.marker_id) for d in detections] == [0, 1, 2, 3, 4, 5]
    worst = max(_corner_error(_by_id(detections, m["id"]), m) for m in truth["markers"])
    assert worst <= MAX_CORNER_ERR_PX, f"四角最大误差 {worst:.3f} px 超过 {MAX_CORNER_ERR_PX} px"
    # 六个码的中心就是生成时的网格点，逐个钉死
    assert [(round(_by_id(detections, i).center_px[0]), round(_by_id(detections, i).center_px[1]))
            for i in range(6)] == \
        [(190, 215), (380, 215), (570, 215), (190, 405), (380, 405), (570, 405)]


# ---------------------------------------------------------------- 旋转图
def test_rotated_3_corner_order_follows_the_marker():
    _require_aruco()
    image_bgr, truth = _load("scenes/rotated_3")
    detections = ArucoDetector().detect(image_bgr)
    assert [int(d.marker_id) for d in detections] == [0, 1, 2]

    # 上边倾角：0° / -30° / -60°（图像坐标系 y 向下，正角在图上看着是逆时针）
    for marker_id, want_angle in ((0, 0.0), (1, -30.0), (2, -60.0)):
        det = _by_id(detections, marker_id)
        got_angle = _top_edge_angle_deg(det)
        assert abs(got_angle - want_angle) < 1.0, \
            f"id={marker_id} 上边倾角应为 {want_angle}°，实得 {got_angle:.2f}°（四角顺序错了会差很多）"

    # 边长也要对：转过角度的四边形，四条边都该是生成时的码边长
    side_want = float(truth["params"]["side_px"])
    for det in detections:
        pts = np.asarray(det.corners_px, dtype=np.float64)
        sides = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
        assert max(abs(s - side_want) for s in sides) <= 1.0, \
            f"四条边应都约 {side_want:.0f} px：{sides}"

    worst = max(_corner_error(_by_id(detections, m["id"]), m) for m in truth["markers"])
    assert worst <= MAX_CORNER_ERR_PX, f"旋转图四角最大误差 {worst:.3f} px"


def test_rotated_single_45_is_a_diamond():
    _require_aruco()
    image_bgr, truth = _load("scenes/rotated_single_45")
    detections = ArucoDetector().detect(image_bgr)
    assert [int(d.marker_id) for d in detections] == [0]

    det = detections[0]
    pts = np.asarray(det.corners_px, dtype=np.float64)
    # 转 45° 后，"左上角"在图上变成最左边那个点，"右上角"变成最上面那个点
    assert np.argmin(pts[:, 0]) == 0, f"第 0 个角应是最左点，实得下标 {np.argmin(pts[:, 0])}"
    assert np.argmin(pts[:, 1]) == 1, f"第 1 个角应是最上点，实得下标 {np.argmin(pts[:, 1])}"
    assert _corner_error(det, truth["markers"][0]) <= 0.1, "45° 单码的四角误差实测只有 0.05 px"
    assert det.center_px == (410.0, 240.0)


# ---------------------------------------------------------------- 无码图 / 单码图
def test_empty_scenes_return_empty_list():
    _require_aruco()
    for name in ("scenes/empty_plain", "scenes/empty_blocks"):
        image_bgr, truth = _load(name)
        assert truth["markers"] == [], f"{name} 的真值里不该有码"
        detections = ArucoDetector().detect(image_bgr)
        assert detections == [], f"{name} 应返回空列表，实得 {len(detections)} 个（误报）"


def test_single_marker_center_is_image_center():
    _require_aruco()
    image_bgr, truth = _load("markers/marker_00")
    detections = ArucoDetector().detect(image_bgr)
    assert [int(d.marker_id) for d in detections] == [0]
    assert detections[0].center_px == (180.0, 180.0)
    assert image_bgr.shape[:2] == (360, 360), "单码图 360x360（码 200 px + 留白 80 px）"
    assert _corner_error(detections[0], truth["markers"][0]) <= 0.15


# ---------------------------------------------------------------- 工厂 / 配置 / 生成器
def test_build_detector_and_config_accept_aruco():
    _require_aruco()
    detector = build_detector("aruco")
    assert isinstance(detector, ArucoDetector)
    assert detector.dictionary_name == "DICT_4X4_50"

    other = build_detector("aruco", {"dictionary": "DICT_4X4_100", "max_detections": 4})
    assert other.dictionary_name == "DICT_4X4_100"
    assert other.max_detections == 4

    cfg = from_dict({"detector": {"name": "aruco",
                                  "aruco": {"dictionary": "DICT_4X4_50",
                                            "corner_refine": "contour",
                                            "min_perimeter_rate": 0.02,
                                            "max_detections": 8}}})
    assert cfg.detector.name == "aruco"
    assert cfg.detector.params["corner_refine"] == "contour"
    assert cfg.detector.params["max_detections"] == 8


def test_config_rejects_bad_aruco_params():
    _require_aruco()
    for bad, keyword in (({"corner_refine": "shiny"}, "corner_refine"),
                         ({"min_perimeter_rate": 0.0}, "min_perimeter_rate"),
                         ({"max_detections": -1}, "max_detections"),
                         ({"dictionary": "4x4_50x"}, "dictionary")):
        params = {"dictionary": "DICT_4X4_50"}
        params.update(bad)
        try:
            from_dict({"detector": {"name": "aruco", "aruco": params}})
        except ConfigError as exc:
            assert keyword in str(exc), f"报错应提到 {keyword!r}，实际：{exc}"
        else:
            raise AssertionError(f"应当报错：{bad}")


def test_unknown_dictionary_and_bad_params_raise():
    _require_aruco()
    try:
        ArucoDetector(dictionary="DICT_9X9_999")
    except ValueError as exc:
        assert "DICT_4X4_50" in str(exc), f"报错应列出可用字典，实际：{exc}"
    else:
        raise AssertionError("未知字典应当报 ValueError")

    for kwargs in ({"corner_refine": "shiny"}, {"min_perimeter_rate": 0.0}, {"max_detections": -1}):
        try:
            ArucoDetector(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应当报 ValueError：{kwargs}")


def test_max_detections_keeps_the_largest():
    _require_aruco()
    from vision.tools.aruco.gen_scenes import (make_marker_image, place_marker,
                                             predefined_dictionary)
    dictionary, _name = predefined_dictionary("DICT_4X4_50")
    canvas = np.full((420, 640, 3), 255, dtype=np.uint8)
    place_marker(canvas, make_marker_image(dictionary, 0, 160), cx=180, cy=210)   # 大码
    place_marker(canvas, make_marker_image(dictionary, 5, 80), cx=460, cy=210)    # 小码

    assert sorted(int(d.marker_id) for d in ArucoDetector().detect(canvas)) == [0, 5]
    kept = ArucoDetector(max_detections=1).detect(canvas)
    assert [int(d.marker_id) for d in kept] == [0], \
        "只留 1 个时应留面积更大的 id=0（160px 的码 > 80px 的码）"


def test_generator_truth_matches_detector_on_fresh_params():
    """不走已提交的图，现场造一张（10°/25°/80°、id 0/7/42）再对真值。"""
    _require_aruco()
    from vision.tools.aruco.gen_scenes import predefined_dictionary
    dictionary, _name = predefined_dictionary("DICT_4X4_50")
    scene = scene_rotated(dictionary, ids=(0, 7, 42), angles=(10.0, 25.0, 80.0),
                          side_px=120, cols=3, gap_px=70, canvas_size=(820, 480))
    detections = ArucoDetector().detect(scene.image_bgr)

    assert sorted(int(d.marker_id) for d in detections) == [0, 7, 42]
    for placement in scene.markers:
        det = _by_id(detections, placement.marker_id)
        want = {"corners": placement.corners, "center": placement.center}
        assert abs(det.center_px[0] - placement.center[0]) < 0.51
        assert abs(det.center_px[1] - placement.center[1]) < 0.51
        err = _corner_error(det, want)
        assert err <= 0.35, f"id={placement.marker_id} 现场造图的四角误差 {err:.3f} px"

    # 同一个场景造两遍必须逐像素一致（图组能进版本库的前提）
    again = scene_rotated(dictionary, ids=(0, 7, 42), angles=(10.0, 25.0, 80.0),
                          side_px=120, cols=3, gap_px=70, canvas_size=(820, 480))
    assert np.array_equal(scene.image_bgr, again.image_bgr), "同参数造图必须逐像素可复现"


# ---------------------------------------------------------------- 独立运行
def _run_standalone() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = skipped = 0
    for fn in tests:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"  ⏭  {fn.__name__}: {exc}")
        except Exception as exc:                     # noqa: BLE001
            failed += 1
            print(f"  ❌ {fn.__name__}: {type(exc).__name__}: {exc}")
    done = len(tests) - failed - skipped
    print(f"\n  test_aruco: {done}/{len(tests)} 通过"
          + (f"，{skipped} 项跳过" if skipped else "")
          + (f"，{failed} 项失败" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
