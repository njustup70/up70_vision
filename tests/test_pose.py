"""ArUco 位姿估计的测试：真值反推 + 四条"不报错但会错"的负面情形。

验的是什么（对应任务单第 2 条"位姿估计"）
    "在已知 ArUco 码实际边长和相机标定参数的前提下，估计码相对相机的三维位置与姿态"
    所以这里不写"能算出个数就行"，而是钉住具体误差：
        · 500 mm 正视：距离误差 < 0.5%（实测 0.27%）、倾角误差 < 0.5°（实测 0.1°）
        · 400 mm 与 800 mm 两张图：距离比 2.0 ± 2%
        · 边长翻倍 → 距离翻倍（尺度只由边长决定，这条错了所有距离都错且不报错）
        · 面内转 0/90/180/270°：距离与倾角不变，只有 yaw 跟着转
    真值不是量出来的，是"先填位姿再渲染"得到的（tools/gen_pose_scenes.py），
    所以误差里不含卷尺误差。

为什么还要写 4 条"应该报错/应该不动"的用例
    下面四种情况都不会抛异常、只会给出"看起来正常"的错误结果，是这一层最容易埋雷的地方：
        1. 交换相邻两个角点  → 重投影 RMS 从 0.03 px 涨到 240 px（靠 RMS 硬校验兜住）
        2. 循环平移角点      → **位姿层完全看不出来**（正方形对称），只有 yaw 变 90/180°
        3. 三个共线点        → 无解，必须抛错而不是返回一个凑合的数
        4. 角点已去畸变还传 D → 二次去畸变，偏心时距离偏 14.6%（实测）

环境要求
    需要 OpenCV >= 4.7 的新版 aruco API（渲染器要用 generateImageMarker）。
    低版本会自动跳过并提示，不算失败。
    跑法：python tests/test_pose.py
"""

import math
import sys
import unittest
from pathlib import Path

try:
    from up70_vision.core.calibration import CameraIntrinsics, undistort_points
    from up70_vision.core.detectors.aruco import ArucoDetector
    from up70_vision.core.pose import (DEFAULT_METHOD, marker_object_points, pose_metrics,
                                       solve_pose)
    from up70_vision.core.types import ArucoMarker, MarkerPose
    from up70_vision.tools.gen_pose_scenes import PoseSpec, render_scene
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from up70_vision.core.calibration import CameraIntrinsics, undistort_points
    from up70_vision.core.detectors.aruco import ArucoDetector
    from up70_vision.core.pose import (DEFAULT_METHOD, marker_object_points, pose_metrics,
                                       solve_pose)
    from up70_vision.core.types import ArucoMarker, MarkerPose
    from up70_vision.tools.gen_pose_scenes import PoseSpec, render_scene

import cv2
import numpy as np

# 码边长（毫米）。测试用 40 mm 与实际印刷无关：所有误差都是"相对"的，
# 换成实测边长后比例关系不变（第 4 个用例专门验证这一点）。
MARKER_MM = 40.0

# 实测误差（40 mm 码、1280x1024、码约 70–95 px 宽）：
#   距离相对误差 0.04%–0.77%，倾角绝对误差 0.04°–1.77°（15°–60° 各档实测）。
# 倾角误差比距离误差大一个量级，原因是几何上的：四个点没有冗余，PnP 总能精确穿过
# 这四个点（所以 RMS 恒在 0.02–0.14 px），角点定位的零点几像素噪声会直接变成
# 零点几度的姿态误差。**"RMS 小"不等于"姿态准"，这两件事在本层是分开的。**
# 容差按实测定：距离 1%、倾角 2°，既能挡住"模型点顺序写错"（那会到 100% 以上），
# 也不会因为渲染的像素离散化差一点点就红。
MAX_DISTANCE_ERR = 0.01
MAX_TILT_ERR_DEG = 2.0

ARUCO_READY = hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector")


def _require_aruco() -> None:
    if not ARUCO_READY:
        raise unittest.SkipTest(
            f"cv2 {cv2.__version__} 没有新版 aruco API（需 >= 4.7）；"
            "在容器里跑：python tests/test_pose.py")


def _render(spec: PoseSpec, dist_coeffs=None):
    """渲染一个场景并检出码；返回 `(marker, truth)`。

    :param spec: 场景定义（真值的来源）
    :param dist_coeffs: 非 None 时按畸变模型投四角，用来验证"去畸变重投影"这条口径
    :return: `(ArucoMarker, truth)`；truth 里带 camera_matrix
    """
    _require_aruco()
    image, truth = render_scene(spec, dist_coeffs=dist_coeffs)
    markers = ArucoDetector().detect(image)
    assert markers, f"渲染出来的图没检出码（场景 {spec.name}）"
    return markers[0], truth


def _estimate(marker, truth, marker_size_mm=MARKER_MM, dist_coeffs=None):
    """按真值图里的内参求位姿。"""
    return solve_pose(np.asarray(marker.corners_px, dtype=np.float64), marker_size_mm,
                      np.asarray(truth["camera_matrix"]), dist_coeffs,
                      marker_id=marker.marker_id)


def _distance_error(pose, truth) -> float:
    """距离的相对误差（比例，不是百分数）。"""
    return abs(pose.distance_mm - truth["distance_mm"]) / truth["distance_mm"]


def _yaw_delta(deg_a: float, deg_b: float) -> float:
    """两个角度之差，折算到 (-180, 180]，避免 179 与 -179 差出 358°。"""
    return ((deg_a - deg_b + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------- 精度基线
def test_frontal_pose_matches_truth():
    """500 mm 正视：距离误差 < 1%，倾角误差 < 1°，yaw ≈ 0。"""
    marker, truth = _render(PoseSpec("frontal", 500.0, marker_size_mm=MARKER_MM))
    pose = _estimate(marker, truth)

    assert _distance_error(pose, truth) < MAX_DISTANCE_ERR, \
        f"距离 {pose.distance_mm:.1f} mm 与真值 {truth['distance_mm']:.1f} mm 差太多"
    assert abs(pose.tilt_deg - truth["tilt_deg"]) < MAX_TILT_ERR_DEG, \
        f"倾角 {pose.tilt_deg:.2f}° 与真值 {truth['tilt_deg']:.2f}° 差太多"
    assert abs(_yaw_delta(pose.yaw_deg, truth["yaw_deg"])) < 2.0
    # 正视时重投影应当极小（实测 0.000 px）；这个数一旦到几十，说明解算走偏了
    assert pose.reproj_rms_px < 1.0, f"重投影 RMS {pose.reproj_rms_px:.3f} px 偏大"


def test_tilted_pose_matches_truth():
    """倾角 30°、450 mm：倾角误差 < 1°，距离误差 < 1%（实测 0.44%）。"""
    marker, truth = _render(PoseSpec("tilt30", 450.0, tilt_deg=30.0, marker_size_mm=MARKER_MM))
    pose = _estimate(marker, truth)
    assert abs(pose.tilt_deg - truth["tilt_deg"]) < MAX_TILT_ERR_DEG, \
        f"倾角 {pose.tilt_deg:.2f}° 与真值 {truth['tilt_deg']:.2f}° 差得超过 {MAX_TILT_ERR_DEG}°"
    assert _distance_error(pose, truth) < MAX_DISTANCE_ERR, \
        f"距离 {pose.distance_mm:.1f} mm 与真值 {truth['distance_mm']:.1f} mm 差太多"


# ---------------------------------------------------------------- 契约形状
def test_pose_fields_are_json_ready_and_use_float64():
    """MarkerPose：float64、distance = ||t||、tz > 0、to_dict 可 JSON 序列化。"""
    marker, truth = _render(PoseSpec("contract", 600.0, tilt_deg=10.0,
                                     marker_size_mm=MARKER_MM))
    pose = _estimate(marker, truth)

    assert isinstance(pose, MarkerPose)
    assert pose.rvec.dtype == np.float64 and pose.rvec.shape == (3,)
    assert pose.tvec_mm.dtype == np.float64 and pose.tvec_mm.shape == (3,)
    assert pose.tvec_mm[2] > 0, "可见的码必须在相机前方"
    # distance 是斜距 ||t||，不是 t[2]：码偏出画面中心时两者不同
    assert abs(pose.distance_mm - float(np.linalg.norm(pose.tvec_mm))) < 1e-9
    assert abs(pose.tz_mm - float(pose.tvec_mm[2])) < 1e-9

    import json
    data = json.loads(json.dumps(pose.to_dict()))
    assert data["marker_id"] == marker.marker_id
    assert isinstance(data["tvec_mm"][0], float) and len(data["tvec_mm"]) == 3
    assert isinstance(data["ambiguous"], bool)


# ---------------------------------------------------------------- 尺度与旋转不变性
def test_distance_is_proportional_to_true_distance():
    """400 mm 与 800 mm：估计距离之比应 ≈ 2.0（±2%）。"""
    near_marker, near_truth = _render(PoseSpec("near", 400.0, marker_size_mm=MARKER_MM))
    far_marker, far_truth = _render(PoseSpec("far", 800.0, marker_size_mm=MARKER_MM))
    ratio = _estimate(far_marker, far_truth).distance_mm / _estimate(near_marker, near_truth).distance_mm
    assert abs(ratio - 2.0) < 0.04, f"距离比 {ratio:.3f} 偏离 2.0 太多"


def test_marker_size_drives_the_scale():
    """同一张图，边长翻倍 → 距离翻倍：尺度只由边长决定（这正是坑 40 的位姿版）。"""
    marker, truth = _render(PoseSpec("scale", 500.0, marker_size_mm=MARKER_MM))
    one = _estimate(marker, truth, marker_size_mm=MARKER_MM)
    two = _estimate(marker, truth, marker_size_mm=MARKER_MM * 2.0)
    assert abs(two.distance_mm / one.distance_mm - 2.0) < 0.01, \
        "边长翻倍时距离没有翻倍，说明尺度不是由 marker_size_mm 决定的"


def test_spin_only_changes_yaw():
    """面内转 0/90/180/270°：距离与倾角不变（< 1% / < 1°），yaw 每次差 90°。"""
    poses = []
    for spin in (0.0, 90.0, 180.0, 270.0):
        marker, truth = _render(PoseSpec(f"spin{int(spin)}", 450.0, spin_deg=spin,
                                          marker_size_mm=MARKER_MM))
        pose = _estimate(marker, truth)
        poses.append(pose)
        assert _distance_error(pose, truth) < MAX_DISTANCE_ERR, \
            f"spin={spin}° 时距离变差了：{pose.distance_mm:.1f} vs {truth['distance_mm']:.1f}"
        assert abs(pose.tilt_deg - truth["tilt_deg"]) < MAX_TILT_ERR_DEG

    # 渲染器的 spin 是"码在平面内转 +spin 度"，而 yaw 按图像顺时针为正，
    # 所以两者的符号相反（实测 spin=90 → yaw=-90）。这里按测量值钉死方向。
    for index, pose in enumerate(poses[1:], start=1):
        delta = _yaw_delta(pose.yaw_deg, poses[0].yaw_deg)
        expected = _yaw_delta(-90.0 * index, 0.0)
        assert abs(_yaw_delta(delta, expected)) < 3.0, \
            f"第 {index} 次旋转的 yaw 差是 {delta:.1f}°，应接近 {expected:.1f}°"


# ---------------------------------------------------------------- 四条例外
def test_adjacent_corner_swap_is_rejected():
    """交换相邻两个角点：重投影 RMS 爆炸，必须抛 ValueError（实测 241 px）。"""
    marker, truth = _render(PoseSpec("swap", 450.0, tilt_deg=20.0, marker_size_mm=MARKER_MM))
    corners = np.asarray(marker.corners_px, dtype=np.float64)[[1, 0, 2, 3]]
    try:
        solve_pose(corners, MARKER_MM, np.asarray(truth["camera_matrix"]), None, marker_id=0)
    except ValueError as exc:
        assert "重投影" in str(exc) or "前方" in str(exc), f"报错应说明原因，实际：{exc}"
    else:
        raise AssertionError("角点被交换后仍返回了位姿，重投影硬校验没起作用")


def test_cyclic_corner_shift_is_invisible_to_pose():
    """循环平移角点：距离与倾角不变、RMS 仍很小，只有 yaw 差 90°/180°。

    这条用例记录的是一个**已知局限**，不是待修 bug：正方形四角循环错位等价于
    "码在平面内转了 90°/180°"，从几何上无法与真实旋转区分。所以角点顺序必须由
    检测层保证（tests/test_aruco.py 的"上边倾角"断言），位姿层拦不住。
    """
    marker, truth = _render(PoseSpec("shift", 450.0, tilt_deg=20.0, marker_size_mm=MARKER_MM))
    base = _estimate(marker, truth)
    matrix = np.asarray(truth["camera_matrix"])
    corners = np.asarray(marker.corners_px, dtype=np.float64)

    for shift, expected_delta in ((1, -90.0), (2, 180.0)):
        pose = solve_pose(np.roll(corners, shift, axis=0), MARKER_MM, matrix, None, marker_id=0)
        assert _distance_error(pose, truth) < MAX_DISTANCE_ERR, \
            f"平移 {shift} 位后距离变了，与「正方形对称」的前提不符"
        assert pose.reproj_rms_px < 1.0, "平移后 RMS 仍应很小（这正是它拦不住的原因）"
        delta = _yaw_delta(pose.yaw_deg, base.yaw_deg)
        assert abs(_yaw_delta(delta, expected_delta)) < 3.0, \
            f"平移 {shift} 位的 yaw 差是 {delta:.1f}°，应接近 {expected_delta:.0f}°"


def test_collinear_corners_are_rejected():
    """三个共线点 + 一个点：无解，必须抛 ValueError 而不是返回凑合的数。"""
    matrix = np.array([[800.0, 0, 640.0], [0, 800.0, 512.0], [0, 0, 1.0]])
    corners = np.array([[100.0, 100.0], [200.0, 100.0], [300.0, 100.0], [200.0, 300.0]])
    try:
        solve_pose(corners, MARKER_MM, matrix, None, marker_id=0)
    except ValueError:
        pass
    else:
        raise AssertionError("共线角点应当抛 ValueError")


def test_bad_marker_size_and_empty_input_are_handled():
    """边长必须为正；空列表返回空列表（不抛异常）。"""
    for bad in (0.0, -1.0, float("nan")):
        try:
            marker_object_points(bad)
        except ValueError:
            continue
        raise AssertionError(f"边长 {bad!r} 应当被拒绝")

    from up70_vision.core.pose import estimate_poses
    matrix = np.array([[800.0, 0, 640.0], [0, 800.0, 512.0], [0, 0, 1.0]])
    assert estimate_poses([], MARKER_MM, matrix, None) == []


def test_undistorted_corners_must_not_get_dist_coeffs():
    """角点已去畸变时：dist 传 None 误差 < 1.5%；再传 D 会偏 > 10%（实测 14.6%）。

    用偏心 + 带畸变的渲染来放大这个差别：画面中心的畸变接近 0，二次去畸变
    在那里看不出来，但码偏到画面边缘时误差会大到直接影响抓取。
    """
    dist = np.array([[-0.30, 0.12, 0.001, -0.001, 0.0]])
    marker, truth = _render(PoseSpec("edge", 450.0, tilt_deg=15.0,
                                      offset_mm=(180.0, -120.0), marker_size_mm=MARKER_MM),
                            dist_coeffs=dist)
    matrix = np.asarray(truth["camera_matrix"])
    intrinsics = CameraIntrinsics(image_size=tuple(truth["canvas"]), camera_matrix=matrix,
                                  dist_coeffs=dist, rms=0.0)
    ideal = undistort_points(np.asarray(marker.corners_px, dtype=np.float64), intrinsics)

    correct = solve_pose(ideal, MARKER_MM, matrix, None, marker_id=marker.marker_id)
    doubled = solve_pose(ideal, MARKER_MM, matrix, dist, marker_id=marker.marker_id)

    correct_err = _distance_error(correct, truth)
    doubled_err = _distance_error(doubled, truth)
    assert correct_err < 0.015, f"去畸变后距离误差 {correct_err * 100:.2f}% 偏大"
    assert doubled_err > 0.10, \
        f"再传一次畸变系数后误差只有 {doubled_err * 100:.2f}%，说明这个坑在本用例里没被放大"


# ---------------------------------------------------------------- 独立运行
def _run_standalone() -> int:
    """不依赖 pytest：把本文件里所有 `test_*` 函数挨个跑一遍，打印通过/跳过/失败。

    仓库的验证命令是 `python tests/test_*.py`（镜像里没装 pytest），所以每个测试文件
    都自带这个入口；与 pytest 的差别只是没有 fixture 和参数化。

    :return: 0 = 全过；1 = 有失败（跳过不算失败）
    """
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
    print(f"\n  test_pose: {done}/{len(tests)} 通过"
          + (f"，{skipped} 项跳过" if skipped else "")
          + (f"，{failed} 项失败" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
