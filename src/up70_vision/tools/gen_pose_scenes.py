"""造带真值的 ArUco 位姿测试图：给定距离/倾角/光照，渲染出图 + 真值 JSON。

用大白话说
    要判断"位姿算得准不准"，必须知道正确答案。真机拍照没有正确答案，只有卷尺量出的
    一个距离；所以这里用相机模型反过来造图：先假定码在某个位置、以某个角度对着相机、
    码有多大，用内参把它投影到画面上得到四个角，再把码的图案按这四个角贴进图里。
    这样一来，"码在多远、斜多少度"是我们自己填进去的，图就是标准答案。

放什么 / 不放什么
    放"造数据"：场景定义（`PoseSpec`）、渲染（`render_scene`）、写盘（`write_scene`）、
    扫描矩阵（`build_suite`）。
    **不做**误差统计（在 `tools/pose_sweep.py`）、不做逐图对真值核位姿（在
    `tools/pose_check.py`）、不 import `viz/`（它画的是给人看的叠加图）。

依赖谁 / 被谁用
    依赖 `core.pose.marker_object_points` 与 `core.pose.pose_metrics`（真值与求解必须同口径）、
    numpy、OpenCV。被 `tests/test_pose.py`（当夹具）与 `tools/pose_sweep.py`（批量）调用。

对外约定
    · `render_scene()` 返回 `(image_bgr, truth)`；truth 里的 `rvec`/`tvec_mm`/`corners_px`
      都是**我们自己填的真值**，不是估计值
    · 码的四角投影出画面时抛 `ValueError`：宁可报错，也不悄悄裁掉半个码
      （被裁的码检不出来，统计里会被当成"算法失败"，把结论带偏）
    · 单位：距离、边长一律毫米；画布尺寸一律像素

已知的坑
    · **背景必须是浅色**。码的黑边框一旦和深色背景连成一片，检测必然失败，
      而这与位姿算法无关（教程第八节）。所以这里固定用 240 左右的浅灰底。
    · **码要留白边**：`generateImageMarker(..., borderBits=0)` 生成的图就是码本身，
      周围没有白边；贴到浅色背景上相当于自带静区，但**贴到深色或花纹背景上会直接失效**。
    · 注入畸变时只投四个角、图内部仍按单应变换贴（不是逐像素的畸变模型）。
      对"整块码很小、几乎是一个平面"的场景误差可忽略；这是刻意的近似，
      真机畸变由真机抽测覆盖。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from ..core.calibration import scale_intrinsics
from ..core.pose import marker_object_points, pose_metrics

#: 默认画布。与相机标定文档里的那台相机同分辨率（1280×1024）。
DEFAULT_CANVAS = (1280, 1024)

#: 默认焦距（像素）。取自 `docs/相机标定.md` 里那台相机的量级（fx≈800）。
#: 换相机就换它，或用 `--calib` 直接读标定文件。
DEFAULT_FX = 800.0

#: 默认码边长（毫米）。**这是占位值**：真实边长必须用卡尺量黑框外沿后填进来，
#: 填错会让所有距离按同比例错，而且重投影误差完全看不出来（progress.md 坑 40 的位姿版）。
DEFAULT_MARKER_MM = 40.0

#: 背景灰度。浅色是硬要求：黑边框与深背景连成一片时检测必然失败。
BACKGROUND_GRAY = 240

#: 正视时码坐标系相对相机坐标系的旋转（实测值，见 core/pose.py 模块头）：
#: 码 x 轴→相机 x、码 y 轴→相机 -y（码的"上边"朝上）、码 z 轴→相机 -z（指向相机）。
_FRONTAL_ROTATION = np.diag([1.0, -1.0, -1.0])

#: `generateImageMarker` 的白边宽度（模块数）。1 格足够让检测器分出"码与背景"，
#: 又不至于让码在画面里显得太小。注意它会改变源图的总模块数（见 `render_scene`）。
_BORDER_BITS = 1


@dataclass(frozen=True)
class PoseSpec:
    """一个合成场景的完整定义（也是它真值的来源）。

    用大白话说
        "这个码在多远、斜多少度、转多少度、多亮、多糊"——把这几个数填好，
        渲染器就能造出一张标准答案已知的图。

    属性
        name (str)            场景名，同时是文件名前缀（ASCII，别用中文）
        distance_mm (float)   码中心到相机的距离（毫米），沿光轴量
        tilt_deg (float)      码面绕相机 x 轴倾斜的角度；0 = 正对相机
        spin_deg (float)      码在自身平面内转动的角度（贴纸转了多少）
        marker_id (int)       用哪个 ID 的码（DICT_4X4_50）
        marker_size_mm (float) 码边长（毫米），黑框外沿到外沿
        brightness (float)    亮度倍数；1.0 = 原样，>1 可能过曝到全白
        ambient (float)       环境光偏置（灰度级），**加在乘亮度之前**。纯乘亮度模拟不了
                              过曝：黑的地方乘 2 还是黑，只有把黑"照亮"才会丢掉对比度
        noise_sigma (float)   叠加的高斯噪声标准差（灰度级）
        blur_px (int)         线性运动模糊核长度（像素）；0 = 不模糊
        canvas_size (tuple)   画布 (宽, 高)，像素

    不变量
        距离、边长必须为正；`tilt_deg` 在 [0, 85]（90° 时码侧对相机、投影退化成一条线）。
    """

    name: str
    distance_mm: float
    tilt_deg: float = 0.0
    spin_deg: float = 0.0
    marker_id: int = 0
    marker_size_mm: float = DEFAULT_MARKER_MM
    brightness: float = 1.0
    ambient: float = 0.0
    noise_sigma: float = 0.0
    blur_px: int = 0
    canvas_size: tuple = DEFAULT_CANVAS
    # 侧向偏移（毫米）：让码不总在画面正中，用来覆盖"码偏到边缘、畸变更大"的情况
    offset_mm: tuple = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("PoseSpec.name 不能为空（它同时是文件名前缀）")
        if not np.isfinite(self.distance_mm) or self.distance_mm <= 0:
            raise ValueError(f"distance_mm 必须为正：{self.distance_mm!r}")
        if not np.isfinite(self.marker_size_mm) or self.marker_size_mm <= 0:
            raise ValueError(f"marker_size_mm 必须为正：{self.marker_size_mm!r}")
        if not 0.0 <= float(self.tilt_deg) < 89.0:
            raise ValueError(
                f"tilt_deg 必须在 [0, 89)：90° 时码侧对相机、投影退化成一条线，"
                f"渲染出来无法检测；收到 {self.tilt_deg!r}")
        if int(self.blur_px) < 0:
            raise ValueError(f"blur_px 不能为负：{self.blur_px!r}")
        if float(self.ambient) < 0:
            raise ValueError(f"ambient 不能为负（它是「把黑的地方照亮」的量）：{self.ambient!r}")


def synthetic_intrinsics(width: int, height: int, fx: float = DEFAULT_FX) -> np.ndarray:
    """造一套"够真"的内参：焦距按画布宽度等比缩放，光心放画面正中。

    为什么按宽度缩放
        1280×1024 与 640×480 想表达同一个视场角，焦距就必须跟着分辨率等比缩
        （否则小图相当于换了一颗长焦镜头，两次扫描没法比）。
        这台相机的 fx 与 fx 一致取同一个数：像素是方的，fy ≠ fx 只会引入一个
        额外的纵横比，对这个任务没有信息量。

    :param width: 画布宽（像素）
    :param height: 画布高（像素）
    :param fx: 1280 宽时的焦距（像素）
    :return: (3,3) float64 内参矩阵
    """
    return np.array([[fx * width / DEFAULT_CANVAS[0], 0.0, width / 2.0],
                     [0.0, fx * width / DEFAULT_CANVAS[0], height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_from(tilt_deg: float, spin_deg: float = 0.0) -> np.ndarray:
    """由"倾斜角 + 面内转角"造一个旋转矩阵（码坐标系 → 相机坐标系）。

    用大白话说
        先把码正对相机摆好（`_FRONTAL_ROTATION`），然后：
        · 绕相机 x 轴转 `tilt_deg` —— 码"低头/抬头"，这就是倾角；
        · 绕码自己的 z 轴（法向）转 `spin_deg` —— 贴纸在纸面上转了一圈。

    公式
        `R = Rx(tilt) · R_frontal · Rz(spin)`
        第三个列向量 `R[:,2]` 是码法向；它与相机 -z 的夹角就是 `tilt`
        （`tilt = acos(-R[2,2])`，见 `core.pose.pose_metrics`）。

    :param tilt_deg: 倾斜角（度），0 = 正对
    :param spin_deg: 面内转角（度）
    :return: (3,3) float64 旋转矩阵
    """
    tilt = np.radians(float(tilt_deg))
    spin = np.radians(float(spin_deg))
    rot_x = np.array([[1.0, 0.0, 0.0],
                      [0.0, np.cos(tilt), -np.sin(tilt)],
                      [0.0, np.sin(tilt), np.cos(tilt)]], dtype=np.float64)
    rot_z = np.array([[np.cos(spin), -np.sin(spin), 0.0],
                      [np.sin(spin), np.cos(spin), 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_x @ _FRONTAL_ROTATION @ rot_z


def spec_rvec_tvec(spec: PoseSpec) -> tuple:
    """把 `PoseSpec` 换成 OpenCV 要的 `(rvec, tvec)`。

    :param spec: 场景定义
    :return: `(rvec(3,), tvec(3,))`，tvec 单位毫米
    """
    import cv2
    rotation = rotation_from(spec.tilt_deg, spec.spin_deg)
    rvec, _ = cv2.Rodrigues(rotation)
    dx, dy = (float(v) for v in spec.offset_mm)
    tvec = np.array([dx, dy, float(spec.distance_mm)], dtype=np.float64)
    return rvec.reshape(3), tvec


def _background(width: int, height: int) -> np.ndarray:
    """浅灰渐变背景：比纯白更接近桌面，又足够亮、让码的黑边框有对比。

    加一点横向渐变而不是纯色：纯色背景配自适应阈值时，全图只有一个灰度级，
    "黑与白"的判断会退化到完全依赖 Otsu 的分割点；有渐变更像真实场景。
    """
    row = np.linspace(BACKGROUND_GRAY - 25, BACKGROUND_GRAY + 15, width, dtype=np.float32)
    plane = np.tile(row, (height, 1))
    return cv2.cvtColor(np.clip(plane, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _apply_blur(image_bgr: np.ndarray, blur_px: int) -> np.ndarray:
    """线性运动模糊：一个方向上拖影，模拟曝光期内码在动。

    为什么用线性核而不是高斯核：码在曝光期内是匀速移动，拖影是"一条线上的叠加"，
    高斯模糊会把边缘变成均匀的软过渡，两者在二值化时的失败方式不一样。
    """
    length = int(blur_px)
    if length <= 0:
        return image_bgr
    length = max(3, length)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length      # 水平方向的匀速拖影
    return cv2.filter2D(image_bgr, -1, kernel)


def render_scene(spec: PoseSpec,
                 camera_matrix=None,
                 dist_coeffs=None,
                 dictionary_name: str = "DICT_4X4_50") -> tuple:
    """渲染一个场景，返回 `(image_bgr, truth)`。

    用大白话说
        按场景定义把码"摆"到相机前方，投影出四个角，再把码的图案按这四个角贴到背景上。
        最后按需要调亮度、加噪声、加模糊——这些只影响"好不好认"，不改真值。

    :param spec: 场景定义
    :param camera_matrix: (3,3) 内参；None 表示按画布尺寸造一套（`synthetic_intrinsics`）
    :param dist_coeffs: 畸变系数或 None；非 None 时只把四个角按畸变模型投影（见模块头"已知的坑"）
    :param dictionary_name: OpenCV 预定义字典名
    :return: `(image_bgr, truth)`；truth 是可直接 `json.dump` 的 dict，含
        `scene / marker_id / marker_size_mm / distance_mm / tilt_deg / yaw_deg /
         rvec / tvec_mm / corners_px / camera_matrix / params`
    :raises ValueError: 四个角投到画布外（宁可报错，也不造一张"半个码"的图）
    """
    width, height = (int(v) for v in spec.canvas_size)
    matrix = synthetic_intrinsics(width, height) if camera_matrix is None \
        else np.asarray(camera_matrix, dtype=np.float64)

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    object_points = marker_object_points(spec.marker_size_mm)
    rvec, tvec = spec_rvec_tvec(spec)

    # ---- 步骤 1：把四个角投到图像上（这一步同时给出真值的角点）----
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, matrix, dist_coeffs)
    corners_px = projected.reshape(4, 2).astype(np.float64)
    margin = 2.0
    if (corners_px[:, 0].min() < margin or corners_px[:, 1].min() < margin
            or corners_px[:, 0].max() > width - 1 - margin
            or corners_px[:, 1].max() > height - 1 - margin):
        raise ValueError(
            f"场景 {spec.name} 的码投到了画布外（距离 {spec.distance_mm:.0f} mm、"
            f"倾角 {spec.tilt_deg:.0f}°）：四角 {np.round(corners_px, 1).tolist()}，"
            f"画布 {width}x{height}。调小距离/倾角，或换更大的画布")

    # ---- 步骤 2：造码的图案并贴到四个角围成的四边形里 ----
    # side_px 取四边形最长边的两倍：贴图分辨率比目标大，缩放时是降采样，边缘不容易锯齿。
    # 上限 800：再大对 100 px 的码没有增益，只是白花内存。
    longest = float(max(np.linalg.norm(corners_px[i] - corners_px[(i + 1) % 4])
                        for i in range(4)))
    side_px = int(min(800, max(64, round(longest * 2))))
    # `borderBits` 是**黑边框**的厚度（不是白边），生成的图从黑框到黑框铺满整幅。
    # 实测（128/240 px 各量一遍）：黑像素的包围盒 = 整张图，占比恒为 1.000。
    # 所以贴图时把整张源图对到投影出来的四角即可，不必往里收——
    # 收一次就会让贴上去的码小一圈，进而让"检测到的角点"和真值角点差十几像素。
    marker_gray = cv2.aruco.generateImageMarker(dictionary, int(spec.marker_id),
                                                side_px, borderBits=_BORDER_BITS)

    canvas = _background(width, height)
    # 源图的四角就是黑框外沿（整幅图），顺序与 corners_px 一致（TL/TR/BR/BL）
    src = np.array([[0.0, 0.0], [side_px, 0.0],
                    [side_px, side_px], [0.0, side_px]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(src, corners_px.astype(np.float32))
    warped = cv2.warpPerspective(marker_gray, transform, (width, height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=BACKGROUND_GRAY)
    # 只在有码的区域覆盖，避免把整幅背景都换成灰度图
    mask = cv2.warpPerspective(np.full_like(marker_gray, 255), transform, (width, height),
                               flags=cv2.INTER_NEAREST)
    canvas[mask > 0] = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)[mask > 0]

    # ---- 步骤 3：光照 / 噪声 / 模糊，只改"好不好认"，不改真值 ----
    if float(spec.brightness) != 1.0 or float(spec.ambient) != 0.0:
        # 先转 float 再截断：uint8 上直接乘会溢出回绕，得到"负亮度"的黑图。
        # 顺序是 (像素 + 环境光) × 增益：环境光先把黑框抬亮，增益再把白格截到 255，
        # 两步合起来才是"过曝"——只乘的话黑框仍是 0，检测反而更容易。
        canvas = np.clip((canvas.astype(np.float32) + float(spec.ambient))
                         * float(spec.brightness), 0, 255).astype(np.uint8)
    if float(spec.noise_sigma) > 0:
        rng = np.random.default_rng(abs(hash(spec.name)) % (2 ** 32))
        noise = rng.normal(0.0, float(spec.noise_sigma), canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    canvas = _apply_blur(canvas, spec.blur_px)

    distance_mm, tilt_deg, yaw_deg = pose_metrics(rvec, tvec)
    truth = {
        "scene": spec.name,
        "marker_id": int(spec.marker_id),
        "marker_size_mm": float(spec.marker_size_mm),
        "distance_mm": round(distance_mm, 4),
        "tilt_deg": round(tilt_deg, 4),
        "yaw_deg": round(yaw_deg, 4),
        "rvec": [float(v) for v in rvec],
        "tvec_mm": [round(float(v), 4) for v in tvec],
        "corners_px": [[round(float(x), 3), round(float(y), 3)] for x, y in corners_px],
        "camera_matrix": [[float(v) for v in row] for row in matrix],
        "canvas": [width, height],
        "params": {
            "tilt_deg": float(spec.tilt_deg), "spin_deg": float(spec.spin_deg),
            "brightness": float(spec.brightness), "ambient": float(spec.ambient),
            "noise_sigma": float(spec.noise_sigma),
            "blur_px": int(spec.blur_px), "offset_mm": list(spec.offset_mm),
        },
    }
    return canvas, truth


def write_scene(out_dir: str, spec: PoseSpec, image_bgr: np.ndarray, truth: dict) -> tuple:
    """把图和真值写盘：`<name>.png` + `<name>.json`。

    :return: `(png 路径, json 路径)`
    """
    from ..utils.image_io import write_image
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{spec.name}.png")
    json_path = os.path.join(out_dir, f"{spec.name}.json")
    write_image(png_path, image_bgr)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(truth, handle, ensure_ascii=False, indent=2)
    return png_path, json_path


def build_suite(base_marker_mm: float = DEFAULT_MARKER_MM,
                base_distance_mm: float = 450.0) -> list:
    """生成"单因素扫描"的整张测试矩阵（一次只动一个维度）。

    用大白话说
        想知道"到底多远开始认不出"，就固定其它条件、只把距离一档一档往上加；
        想知道"能斜多少度"，就固定距离、只加倾角。一次动一个变量，结论才归因得清。

    维度与档位
        距离   300 / 450 / 675 / 1000 / 1500 / 2000 mm（等比 1.5）
        倾角   0 / 15 / 30 / 45 / 60 / 75°（距离取 `base_distance_mm`）
        滚转   0 / 90 / 180 / 270°（验证旋转不变性）
        亮度   0.35 / 0.60 / 1.00 / 1.40（外加 2.2 的过曝档）
        噪声   0 / 8 / 16 灰度级
        模糊   0 / 5 / 9 / 13 px
        分辨率 1280×1024 与 640×480（同一视场，验证"每格像素数"这条硬下限）

    :param base_marker_mm: 码边长（毫米），**必须是实测值**
    :param base_distance_mm: 除"距离"维度外，其它维度统一用这个距离
    :return: `list[PoseSpec]`，名字都是 ASCII（`d0500_t00` 这种），可直接当文件名
    """
    specs = []

    def add(name: str, **kwargs) -> None:
        kwargs.setdefault("marker_size_mm", float(base_marker_mm))
        specs.append(PoseSpec(name=name, **kwargs))

    for dist in (300.0, 450.0, 675.0, 1000.0, 1500.0, 2000.0):
        add(f"dist_{int(dist):04d}", distance_mm=dist)
    for tilt in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0):
        add(f"tilt_{int(tilt):02d}", distance_mm=base_distance_mm, tilt_deg=tilt)
    for spin in (0.0, 90.0, 180.0, 270.0):
        add(f"spin_{int(spin):03d}", distance_mm=base_distance_mm, spin_deg=spin)
    for factor in (0.35, 0.60, 1.00, 1.40, 2.20, 3.00):
        # 2.2 / 3.0 两档额外加环境光：只有"黑框被照亮 + 白格截顶"才叫过曝。
        # 单纯乘 2.2 时黑框还是 0、白格还是 255，等于什么都没变（实测 100% 通过）；
        # 加上偏移才是真过曝：2.2 + 60 属于"发灰但仍能认"，3.0 + 110 直接丢码。
        ambient = 110.0 if factor >= 3.0 else (60.0 if factor > 2.0 else 0.0)
        add(f"bright_{int(factor * 100):03d}", distance_mm=base_distance_mm,
            brightness=factor, ambient=ambient)
    for sigma in (0.0, 8.0, 16.0):
        add(f"noise_{int(sigma):02d}", distance_mm=base_distance_mm, noise_sigma=sigma)
    for blur in (0, 5, 9, 13):
        add(f"blur_{int(blur):02d}", distance_mm=base_distance_mm, blur_px=blur)
    add("res_1280", distance_mm=base_distance_mm, canvas_size=(1280, 1024))
    add("res_0640", distance_mm=base_distance_mm, canvas_size=(640, 480))
    return specs


def build_parser():
    """命令行参数：目录、码边长、只造某一组。"""
    import argparse
    ap = argparse.ArgumentParser(
        prog="up70-vision-gen-pose-scenes",
        description="造带真值的 ArUco 位姿测试图（距离/倾角/滚转/光照/噪声/模糊）")
    ap.add_argument("--out", default=os.path.join("output", "pose_scenes"),
                    help="输出目录，默认 output/pose_scenes")
    ap.add_argument("--marker-mm", type=float, default=DEFAULT_MARKER_MM,
                    help=f"码边长（毫米，量黑框外沿），默认 {DEFAULT_MARKER_MM}")
    ap.add_argument("--distance-mm", type=float, default=450.0,
                    help="除距离维度外的基准距离，默认 450")
    ap.add_argument("--calib", default="",
                    help="内参 YAML（configs/camera_calibration.yaml）；不给就用合成内参")
    ap.add_argument("--suite", action="store_true",
                    help="造整张扫描矩阵（默认只造一张 450 mm 正视图，用于快速自检）")
    return ap


def _load_intrinsics(calib_path: str):
    """从标定 YAML 读内参；没给路径就返回 None。

    读文件这件事放在工具层：`core/` 不许读文件（契约第 5 节）。

    :param calib_path: 形如 `configs/camera_calibration.yaml`；空字符串表示用合成内参
    :return: `CameraIntrinsics` 或 None
    """
    if not calib_path:
        return None
    from ..config.loader import load_intrinsics
    return load_intrinsics(calib_path)


def main(argv=None) -> int:
    """造图并打印一行汇总；退出码 0 = 全部生成成功。

    :param argv: 参数列表；None 表示取 `sys.argv[1:]`
    :return: 0 成功 / 2 参数或环境错误（内参文件读不出来）
    副作用
        往 `--out` 目录写 PNG 与同名 JSON
    """
    args = build_parser().parse_args(argv)
    specs = (build_suite(args.marker_mm, args.distance_mm) if args.suite
             else [PoseSpec(name="single_frontal", distance_mm=args.distance_mm,
                            marker_size_mm=args.marker_mm)])
    try:
        intrinsics = _load_intrinsics(args.calib)
    except Exception as exc:                       # 标定文件写坏了要在这停住，别造一堆假图
        print(f"❌ 读内参失败：{exc}")
        return 2
    print(f"码边长 {args.marker_mm:.1f} mm   内参 "
          f"{args.calib if intrinsics is not None else '合成（fx=800@1280 等比缩放）'}   "
          f"共 {len(specs)} 个场景")

    written = 0
    for spec in specs:
        if intrinsics is None:
            image, truth = render_scene(spec, None, None)
        else:
            # 真内参按画布尺寸等比缩放：1280 的内参直接套到 640 画布上等于换了一颗长焦
            # 镜头，那样造出来的图与"同一颗相机降分辨率输出"不是一回事（progress 坑 3）。
            used = (intrinsics if tuple(intrinsics.image_size) == tuple(spec.canvas_size)
                    else scale_intrinsics(intrinsics, spec.canvas_size))
            image, truth = render_scene(spec, used.camera_matrix, used.dist_coeffs)
        write_scene(args.out, spec, image, truth)
        written += 1
    print(f"✅ 已写 {written} 个场景到 {args.out}（每张图配一个同名 .json 真值）")
    return 0
