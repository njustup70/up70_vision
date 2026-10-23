"""api 门面的回归测试：一次调用拿结果、参数能盖住默认值、坏图明确报错。

被测的是 `vision/api.py`，它不复制逻辑、只把
「配置 -> 工厂 -> 管线」这条原链路包成一个函数，所以这里只验门面本身：
返回类型对不对、参数有没有真的传下去、坏输入是不是明确失败。
"""

import sys
from pathlib import Path

try:
    from vision.api import detect
    from vision.aruco.types import ArucoMarker
    from vision.types import Detection
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vision.api import detect
    from vision.aruco.types import ArucoMarker
    from vision.types import Detection

import numpy as np

# 用仓库自带的配置与阈值表，测试不依赖当前工作目录
_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _split_image_bgr() -> np.ndarray:
    """造一张左红右绿的 640x480 图（左半 BGR 纯红、右半纯绿）。

    为什么不用纯色图：纯色图在通道顺序写反时仍是纯色，判色照样"通过"；
    左右两半分色能顺带抓住 BGR/RGB 写反的错。
    """
    image_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    image_bgr[:, :320] = (0, 0, 255)
    image_bgr[:, 320:] = (0, 255, 0)
    return image_bgr


def test_detect_blob_returns_detections_with_color():
    """判色族返回 list[Detection]，且颜色字段已被管线填好。"""
    boxes = detect(_split_image_bgr(), "blob", config=str(_CONFIG), min_area_px=100)
    assert boxes, "整张图都是彩色，至少应检出一个框"
    assert all(isinstance(box, Detection) for box in boxes), \
        f"blob 应返回 Detection，实得 {[type(b).__name__ for b in boxes]}"
    assert boxes[0].bbox_xyxy == (0, 0, 640, 480), f"框应覆盖整图，实得 {boxes[0].bbox_xyxy}"
    assert boxes[0].color == "red", f"左半纯红应判成 red，实得 {boxes[0].color!r}"


def test_detect_aruco_family_returns_markers():
    """ArUco 族返回 list[ArucoMarker]；stub 默认造 3 个码，ID 升序。"""
    markers = detect(_split_image_bgr(), "stub_aruco")
    assert len(markers) == 3, f"stub_aruco 默认 ids=[0,1,2]，应出 3 个码，实得 {len(markers)}"
    assert all(isinstance(marker, ArucoMarker) for marker in markers)
    assert [int(m.marker_id) for m in markers] == [0, 1, 2], "契约要求按 marker_id 升序"


def test_detect_params_override_defaults():
    """**params 会真的传进检测器**（不是只存在签名里）。"""
    markers = detect(_split_image_bgr(), "stub_aruco", ids=[5], side_px=60)
    assert [int(m.marker_id) for m in markers] == [5], \
        f"ids=[5] 应覆盖默认的 [0,1,2]，实得 {[int(m.marker_id) for m in markers]}"
    side = float(markers[0].corners_px[1][0] - markers[0].corners_px[0][0])
    assert abs(side - 60.0) < 1e-6, f"side_px=60 应生效，实测边长 {side}"


def test_detect_rejects_non_bgr_image():
    """单通道图 / None 一律抛 ValueError，不返回空列表。"""
    for bad, name in ((np.zeros((480, 640), dtype=np.uint8), "单通道灰度图"), (None, "None")):
        try:
            detect(bad, "blob")
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name} 应当抛 ValueError")


def test_detect_unknown_detector_lists_available_names():
    """检测器名写错时报错里要列出可用值（现场改配置不用猜）。"""
    try:
        detect(_split_image_bgr(), "no_such_detector")
    except ValueError as exc:
        assert "aruco" in str(exc), f"报错应列出可用检测器，实际：{exc}"
    else:
        raise AssertionError("未知检测器名应当抛 ValueError")


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
    print(f"\n  test_api: {len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
