"""types 的测试：IoU、半开上界、裁剪、序列化。

断言全部带具体数字——判色/几何算错了不会抛异常，只会"看起来不太准"，
没有具体数字的断言发现不了（README「团队约定」）。

跑法（二选一）
    pytest tests/test_types.py
    python tests/test_types.py          # 不依赖 pytest
"""

import json
import sys
from pathlib import Path

try:
    from vision.types import is_marker
    from vision.aruco.types import ArucoMarker
    from vision.types import Detection, Frame, bbox_from_xywh, clip_bbox, iou
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vision.types import is_marker
    from vision.aruco.types import ArucoMarker
    from vision.types import Detection, Frame, bbox_from_xywh, clip_bbox, iou

import numpy as np


def test_iou_same_box_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_half_overlap():
    # 交 5*10=50；并 100+100-50=150 -> 50/150 = 0.3333
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-9


def test_iou_disjoint_and_zero_area():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


def test_bbox_half_open_upper_bound():
    box = bbox_from_xywh(10, 20, 30, 40)
    assert box == (10, 20, 40, 60)
    det = Detection(bbox_xyxy=box)
    assert det.width == 30 and det.height == 40 and det.area == 1200
    assert det.center == (25.0, 40.0)


def test_clip_bbox():
    assert clip_bbox((-5, -5, 700, 500), 640, 480) == (0, 0, 640, 480)
    assert clip_bbox((10, 10, 20, 20), 640, 480) == (10, 10, 20, 20)


def test_detection_to_dict_is_json_serializable():
    det = Detection(bbox_xyxy=(1, 2, 11, 22), color="red", color_ratio=0.75, valid_pixels=40)
    payload = json.dumps({"data": [det.to_dict()]})     # 不能抛异常
    back = json.loads(payload)["data"][0]
    assert back["bbox_xyxy"] == [1, 2, 11, 22]
    assert back["color"] == "red"
    assert back["center"] == [6.0, 12.0]
    assert back["valid_pixels"] == 40


def test_frame_rejects_non_bgr():
    ok = Frame(image_bgr=np.zeros((4, 5, 3), np.uint8))
    assert (ok.height, ok.width, ok.size) == (4, 5, (5, 4))
    try:
        Frame(image_bgr=np.zeros((4, 5), np.uint8))
    except ValueError as exc:
        assert "HxWx3" in str(exc)
    else:
        raise AssertionError("单通道图应该被拒绝")


def test_aruco_marker_normalizes_to_contract():
    """ArucoMarker 的四条约定：float32、升序由检测器保证、中心是四角均值、可 JSON 化。

    这里用一个**非对称四边形**，让"四角算术平均"与"bbox 中心"不相等（6, 4.5) vs (6, 5)），
    这样断言才真的能钉住契约第 3 条。
    """
    corners = [[0, 0], [10, 0], [12, 8], [2, 10]]          # int 列表：应被归一成 float32
    marker = ArucoMarker(marker_id=7, corners_px=corners, center_px=(6.0, 4.5))
    assert marker.corners_px.dtype == np.float32, "契约要求 corners_px 是 float32"
    assert marker.corners_px.shape == (4, 2), "契约要求 (4,2)"
    assert marker.center_px == (6.0, 4.5)
    assert marker.center == (6.0, 4.5), "center 是 center_px 的别名"
    assert marker.bbox_xyxy == (0, 0, 12, 10), "派生外接框（半开上界）"
    assert marker.bbox_xyxy != (0, 0, 13, 11), "不能多算 1px"
    # bbox 中心是 (6.0, 5.0)，与四角均值 (6.0, 4.5) **不同**——这正是不能用 bbox 代替四角的原因
    assert marker.center_px != (6.0, 5.0)
    assert is_marker(marker) and not is_marker(Detection(bbox_xyxy=(0, 0, 1, 1)))

    back = json.loads(json.dumps({"m": [marker.to_dict()]}))["m"][0]
    assert back["marker_id"] == 7
    assert back["center_px"] == [6.0, 4.5]
    assert back["corners_px"][2] == [12.0, 8.0]


def test_aruco_marker_rejects_bad_input():
    good = [[0, 0], [10, 0], [10, 10], [0, 10]]
    for kwargs, keyword in (
            ({"marker_id": "7", "corners_px": good, "center_px": (5.0, 5.0)}, "marker_id"),
            ({"marker_id": 7, "corners_px": [[0, 0], [1, 1]], "center_px": (0.0, 0.0)},
             "corners_px"),
            ({"marker_id": 7, "corners_px": good, "center_px": (5.0,)}, "center_px")):
        try:
            ArucoMarker(**kwargs)
        except ValueError as exc:
            assert keyword in str(exc), f"报错应提到 {keyword!r}，实际：{exc}"
        else:
            raise AssertionError(f"应当报错：{kwargs}")

    # 输入是 float64 数组时也要归一成 float32（不能只接受 float32 输入）
    f64 = np.asarray(good, dtype=np.float64)
    assert ArucoMarker(marker_id=1, corners_px=f64, center_px=(5.0, 5.0)).corners_px.dtype \
        == np.float32


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
    print(f"\n  test_types: {len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
