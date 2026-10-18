"""相机标定的测试：用「已知答案」的合成数据反推，不需要真相机。

思路
    先编一套内参 K 和畸变系数，用它把棋盘上的点投影到图上（这就是"拍照"）；
    再把投影出来的点喂给 `calibrate()`，看能不能把 K 和畸变找回来。
    这条路子不依赖相机、板子和打印精度，几秒钟就能跑完，而且**答案已知**。

断言全部带具体数字：标定算错了不会抛异常，只会"看起来不太准"。
"""

import json
import sys
from pathlib import Path

try:
    from up70_vision.core.calibration import (DIST_ORDER, MIN_VIEWS, RMS_USABLE,
                                              BoardSpec, CameraIntrinsics, board_object_points,
                                              calibrate, find_board_corners, undistort_image,
                                              undistort_points)
    from up70_vision.tools.gen_chessboard import inject_png_dpi, render_board, write_pdf
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from up70_vision.core.calibration import (DIST_ORDER, MIN_VIEWS, RMS_USABLE,
                                              BoardSpec, CameraIntrinsics, board_object_points,
                                              calibrate, find_board_corners, undistort_image,
                                              undistort_points)
    from up70_vision.tools.gen_chessboard import inject_png_dpi, render_board, write_pdf

import numpy as np

# 合成的"真相"：1280x800 的相机，焦距 800 px，光心略偏
TRUE_SIZE = (1280, 800)
TRUE_K = np.array([[800.0, 0.0, 640.0],
                   [0.0, 800.0, 400.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
TRUE_DIST = np.array([-0.28, 0.12, 0.0012, -0.0008, 0.0], dtype=np.float64)
BOARD = BoardSpec(cols=9, rows=6, square_mm=15.0)


def _synthetic_views(views: int = 20, noise_px: float = 0.2, seed: int = 2027):
    """造一批"拍到"的角点：用 TRUE_K/TRUE_DIST 把板子投影到多个位姿。

    :return: `(image_points, image_size)`，可直接喂给 `calibrate()`
    """
    import cv2

    grid = board_object_points(BOARD)
    rng = np.random.default_rng(seed)
    image_points = []
    for index in range(views):
        # 板子放在相机前方 400~900 mm，左右上下各挪一些，三个轴都转一点
        # 远近拉开（300~1100mm）+ 大角度倾斜：平面靶标下焦距与距离是耦合的，
        # 视角太集中时 fx 会被"摊平"，这是标定本身的退化，不是代码错
        tvec = np.array([rng.uniform(-80, 80), rng.uniform(-60, 60),
                         rng.uniform(300, 1100)], dtype=np.float64)
        angle = np.radians([rng.uniform(-35, 35), rng.uniform(-35, 35),
                            rng.uniform(-30, 30)])
        rvec = np.asarray(angle, dtype=np.float64)
        projected, _ = cv2.projectPoints(grid, rvec, tvec, TRUE_K, TRUE_DIST)
        jitter = rng.normal(0.0, noise_px, projected.reshape(-1, 2).shape)
        image_points.append((projected.reshape(-1, 2) + jitter).reshape(-1, 1, 2).astype(np.float32))
        _ = index
    return image_points, TRUE_SIZE


# ---------------------------------------------------------------- 板子参数
def test_board_object_points_numbers():
    grid = board_object_points(BOARD)
    assert grid.shape == (54, 3), f"9x6 应有 54 个点，实得 {grid.shape}"
    assert grid.dtype == np.float32
    # 第一个点是原点、第二个点往右一个格子、最后一行的第一个点往下一个纵向格子
    assert tuple(grid[0]) == (0.0, 0.0, 0.0)
    assert tuple(grid[1]) == (15.0, 0.0, 0.0)
    assert tuple(grid[9]) == (0.0, 15.0, 0.0), "第 10 个点应换行（9 列）"
    assert tuple(grid[-1]) == (120.0, 75.0, 0.0), f"最后一个点应是 (120,75,0)，实得 {grid[-1]}"
    assert np.allclose(grid[:, 2], 0.0), "板子是 z=0 平面"


def test_board_spec_validation():
    for bad, keyword in ((dict(cols=2, rows=6, square_mm=20.0), "内角点"),
                         (dict(cols=9, rows=6, square_mm=0.0), "边长")):
        try:
            BoardSpec(**bad)
        except ValueError as exc:
            assert keyword in str(exc), f"报错应提到 {keyword!r}，实际：{exc}"
        else:
            raise AssertionError(f"应当报错：{bad}")
    spec = BoardSpec(cols=10 - 1, rows=7 - 1, square_mm=15.0)
    assert spec.inner_corners == 54, "10x7 格 -> 9x6 内角点 = 54 个"


# ---------------------------------------------------------------- 找角点
def test_find_corners_on_rendered_board():
    image, _square_px = render_board(BOARD, dpi=300)
    corners = find_board_corners(image, BOARD)
    assert corners is not None, "自己画的板子应该能认出角点"
    assert corners.shape == (BOARD.inner_corners, 1, 2) == (54, 1, 2)
    assert corners.dtype == np.float32
    # 左上角那个角点应该落在白边之后、第一格交界的附近
    x, y = corners[0, 0]
    assert 0 < x < image.shape[1] and 0 < y < image.shape[0]


def test_find_corners_returns_none_on_blank_and_rejects_gray():
    blank = np.full((240, 320, 3), 255, dtype=np.uint8)
    assert find_board_corners(blank, BOARD) is None, "空白图应返回 None（不是 []）"
    try:
        find_board_corners(np.zeros((240, 320), dtype=np.uint8), BOARD)
    except ValueError as exc:
        assert "HxWx3" in str(exc)
    else:
        raise AssertionError("单通道灰度应被拒绝（与 detect() 同口径）")


# ---------------------------------------------------------------- 解算
def test_calibrate_recovers_known_intrinsics():
    image_points, image_size = _synthetic_views()
    result = calibrate(image_points, image_size, BOARD)

    assert result.views == 20 and result.board == BOARD
    assert abs(result.fx - 800.0) / 800.0 < 0.03, f"fx 应接近 800，实得 {result.fx:.1f}"
    assert abs(result.fy - 800.0) / 800.0 < 0.03, f"fy 应接近 800，实得 {result.fy:.1f}"
    # 主点是标定里最不稳的一个（与畸变系数强耦合）：实测不同随机种子下偏 4~6 px 是常态，
    # 所以这里按"几像素内"要求，而不是要求它是 640.0
    assert abs(result.cx - 640.0) < 6.0, f"cx 应接近 640，实得 {result.cx:.1f}"
    assert abs(result.cy - 400.0) < 6.0, f"cy 应接近 400，实得 {result.cy:.1f}"
    # 加了 0.2px 的噪声，重投影误差应该也在这一档
    assert result.rms < 0.6, f"RMS 应 <0.6 px，实得 {result.rms:.3f}"
    assert len(result.per_view_rms) == 20
    assert result.dist_coeffs.shape == (1, 5), "畸变系数个数与顺序必须与契约一致"
    assert abs(result.dist_coeffs.ravel()[0] - (-0.28)) < 0.12, \
        f"k1 应接近 -0.28，实得 {result.dist_coeffs.ravel()[0]:.3f}"


def test_calibrate_rejects_too_few_views():
    image_points, image_size = _synthetic_views(views=MIN_VIEWS - 1)
    try:
        calibrate(image_points, image_size, BOARD)
    except ValueError as exc:
        assert "至少" in str(exc) and str(MIN_VIEWS) in str(exc)
    else:
        raise AssertionError("有效图不够时应报错")


def test_calibrate_rejects_wrong_corner_count():
    image_points, image_size = _synthetic_views(views=4)
    broken = [image_points[0], image_points[1][:20], image_points[2]]
    try:
        calibrate(broken, image_size, BOARD)
    except ValueError as exc:
        assert "角点数" in str(exc) and "9x6" in str(exc), f"报错应说清板子参数，实际：{exc}"
    else:
        raise AssertionError("角点个数与板子不符时应报错")


# ---------------------------------------------------------------- 去畸变与序列化
def test_undistort_keeps_center_when_no_distortion():
    import cv2

    flat = CameraIntrinsics(image_size=TRUE_SIZE, camera_matrix=TRUE_K,
                            dist_coeffs=np.zeros(5), rms=0.0, per_view_rms=(), views=1,
                            board=BOARD)
    image = np.full((TRUE_SIZE[1], TRUE_SIZE[0], 3), 255, dtype=np.uint8)
    cv2.circle(image, (640, 400), 20, (0, 0, 255), -1)
    fixed = undistort_image(image, flat)
    ys, xs = np.nonzero((fixed == np.array([0, 0, 255])).all(axis=2))
    assert abs(xs.mean() - 640) < 0.6 and abs(ys.mean() - 400) < 0.6, \
        "无畸变时去畸变不该把中心挪走"
    # 有畸变时，远离中心的点会被挪动（方向不必精确，但必须动）
    bent = CameraIntrinsics(image_size=TRUE_SIZE, camera_matrix=TRUE_K,
                            dist_coeffs=TRUE_DIST, rms=0.0, per_view_rms=(), views=1,
                            board=BOARD)
    moved = undistort_points([(1200, 760)], bent)
    assert np.linalg.norm(moved[0] - np.array([1200.0, 760.0])) > 0.5, \
        f"有畸变时角落点必须被挪动，实得 {moved[0]}"


def test_camera_intrinsics_dict_roundtrip_and_quality():
    image_points, image_size = _synthetic_views()
    result = calibrate(image_points, image_size, BOARD)
    payload = json.dumps(result.to_dict())              # numpy 标量会导致 TypeError
    back = CameraIntrinsics.from_dict(json.loads(payload))
    assert back.image_size == result.image_size
    assert np.allclose(back.camera_matrix, result.camera_matrix)
    assert np.allclose(back.dist_coeffs, result.dist_coeffs)
    assert back.views == result.views and back.board == result.board
    assert back.to_dict()["dist_order"] == list(DIST_ORDER)

    assert result.quality in ("good", "usable", "poor")
    assert f"fx={result.fx:.1f}" in result.summary()
    assert RMS_USABLE == 1.0, "RMS 判据阈值是给别人看的文档口径，别悄悄改"


# ---------------------------------------------------------------- 打印件本身
def test_png_carries_dpi_metadata():
    """PNG 必须写 pHYs，否则打印驱动会按 200 dpi 猜（实测把 15 mm 打成 23 mm）。"""
    import struct
    import tempfile

    image, _ = render_board(BOARD, dpi=300)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "board.png")
        import cv2
        assert cv2.imwrite(path, image)
        inject_png_dpi(path, 300)
        raw = Path(path).read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    offset, found = 8, None
    while offset < len(raw):
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        kind = raw[offset + 4:offset + 8]
        if kind == b"pHYs":
            x_ppm, _y_ppm, unit = struct.unpack(">IIB", raw[offset + 8:offset + 17])
            found = (x_ppm, unit)
            break
        if kind == b"IDAT":
            break
        offset += 12 + length
    assert found is not None, "PNG 里没有 pHYs 块"
    ppm, unit = found
    assert unit == 1, "单位应为米"
    assert abs(ppm * 0.0254 - 300) < 0.5, f"应记录 300 dpi，实得 {ppm * 0.0254:.1f}"


def test_pdf_matches_physical_size():
    """PDF 的图必须按「像素/dpi*72」精确排布：打印时选实际大小就是 1:1。"""
    import re
    import tempfile

    image, _ = render_board(BOARD, dpi=300)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "board.pdf")
        page_mm, image_mm = write_pdf(path, image, 300, "a4")
        raw = Path(path).read_text(encoding="latin1")

    assert raw.startswith("%PDF-1.4") and raw.rstrip().endswith("%%EOF")
    assert "xref" in raw and "/Subtype /Image" in raw
    # A4 页面（点）：595.28 x 841.89
    assert abs(page_mm[0] - 210.0) < 0.5 and abs(page_mm[1] - 297.0) < 0.5, page_mm
    # 图在纸上的物理尺寸必须等于 像素/300*25.4
    height, width = image.shape[:2]
    assert abs(image_mm[0] - width / 300 * 25.4) < 0.05, image_mm
    assert abs(image_mm[1] - height / 300 * 25.4) < 0.05, image_mm
    # 内容流里的 cm 矩阵尺寸应与换算一致（点）
    match = re.search(r"q ([\d.]+) 0 0 ([\d.]+) ", raw)
    assert match, "内容流里没有图像变换矩阵"
    assert abs(float(match.group(1)) - width / 300 * 72) < 0.05
    assert abs(float(match.group(2)) - height / 300 * 72) < 0.05


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
    print(f"\n  test_calibration: {len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
