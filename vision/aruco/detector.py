"""真 ArUco 检测器：按 docs/interface.md 第 2.2 / 3.2 节的契约返回 `list[ArucoMarker]`。

任务单对应关系
    任务单「搭档 Owner：检测核心」的产出之一就是本文件：
    `detect()`，DICT_4X4_50，**同图多码、旋转均可识别**。
    契约（而不是本文件的实现）定义返回值：`ArucoMarker{marker_id, corners_px, center_px}`，
    角点顺序 TL→TR→BR→BL（码自身坐标系起点）、`corners_px` 是 (4,2) float32、
    `center_px` 是四角**算术平均**的浮点元组；按 `marker_id` 升序；没码时返回 `[]`、不抛异常。

为什么用 `cv2.aruco.ArucoDetector` 这个类
    ArUco 自 OpenCV 4.7 起进了主包 opencv-python，同时 API 改版：老教程里的
    `cv2.aruco.DetectorParameters_create()` + `cv2.aruco.detectMarkers(...)` 写法
    在新版会直接报错（任务单坑 1、2）。本仓库钉 opencv-python>=4.10，
    因此统一走新 API；**不要**为了老教程去装 opencv-contrib-python。

角点顺序为什么可以原样透传
    OpenCV 返回的四角本来就是「码自身坐标里的左上角起、顺时针」——
    正好等于契约要求的 TL→TR→BR→BL。所以这里只归一 dtype，不做重排；
    一旦重排，`solvePnP` 阶段就会拿到拧过的四边形。
"""

import numpy as np

from .types import ArucoMarker

DEFAULT_DICTIONARY = "DICT_4X4_50"

#: 角点精化方式（configs/default.yaml 的 detector.aruco.corner_refine 只能填这些）
CORNER_REFINE_MODES = ("none", "subpix", "contour")

#: 检测到的最小码周长，占图像周长的比例（OpenCV 默认 0.03）。
#: 调小能捞到更小的码，但也更容易把画面里的方框误当成码；本仓库先按默认值走。
DEFAULT_MIN_PERIMETER_RATE = 0.03

#: 哪些 OpenCV 才算"没有新版 aruco API"
MIN_OPENCV_VERSION = (4, 7, 0)


def _opencv_version_tuple() -> tuple:
    import cv2
    parts = []
    for chunk in str(cv2.__version__).split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts[:3])


def check_opencv_aruco(need_generate: bool = False) -> None:
    """确认当前 OpenCV 有新版 aruco API；没有就抛一句带处理办法的人话。

    :param need_generate: 还要生成码图时传 True（`generateImageMarker` 与检测是两套 API，
        有的构建/版本会缺其中一个；`tools/aruco/gen_scenes.py` 造图前要这一条）
    """
    import cv2
    if need_generate and not hasattr(cv2.aruco, "generateImageMarker"):
        raise RuntimeError(
            f"当前 cv2 {cv2.__version__} 没有 generateImageMarker（生成 ArUco 码图需要）。"
            "请用 opencv-python>=4.7（容器里是 4.14）")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
        version = _opencv_version_tuple()
        raise RuntimeError(
            f"当前 cv2 {cv2.__version__} 没有新版 aruco API（ArucoDetector，需 >= "
            f"{'.'.join(str(v) for v in MIN_OPENCV_VERSION)}）。"
            "处理办法：用 opencv-python>=4.7（本仓库钉 >=4.10，容器镜像里是 4.14）；"
            "**不要**装 opencv-contrib-python —— 它与 opencv-python 共用 cv2 顶层模块，"
            "两个一起装会互相覆盖（见 docs/progress.md 坑 2）")


def predefined_dictionary(name: str):
    """按名字取预定义字典。

    :param name: "DICT_4X4_50"（任务单指定）或省略前缀的 "4x4_50"，大小写不敏感
    :return: (cv2.aruco.Dictionary, 规范名)
    :raises ValueError: 不是 OpenCV 预定义字典（报错里给常见可用值）
    """
    import cv2
    check_opencv_aruco()
    key = str(name).strip().upper()
    if key and not key.startswith("DICT_"):
        key = "DICT_" + key
    dictionary_id = getattr(cv2.aruco, key, None)
    if not isinstance(dictionary_id, int):
        common = ["DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_6X6_250", "DICT_ARUCO_ORIGINAL"]
        raise ValueError(f"未知 ArUco 字典 {name!r}；常见可用值：{common}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id), key


class ArucoDetector:
    """DICT_4X4_50 的 ArUco 检测器（同图多码 + 旋转都能出）。

    输出顺序：**按 `marker_id` 升序**（契约 2.2 第 2 条）——同一块码每帧都排在同一位，
    日志与叠加图逐帧对比起来才不跳；`marker_id` 相同时保持 OpenCV 的检出顺序（稳定排序）。
    """

    name = "aruco"

    def __init__(self,
                 dictionary: str = DEFAULT_DICTIONARY,
                 corner_refine: str = "subpix",
                 min_perimeter_rate: float = DEFAULT_MIN_PERIMETER_RATE,
                 max_detections: int = 0):
        """
        :param dictionary: 字典名，默认 DICT_4X4_50（任务单指定）
        :param corner_refine: 角点精化 none/subpix/contour；subpix 能把角点做到亚像素
        :param min_perimeter_rate: 检出码的最小周长占图像周长的比例
        :param max_detections: 单帧最多返回几个（0 = 不限）；超出时保留**面积最大**的几个
        """
        import cv2
        check_opencv_aruco()

        mode = str(corner_refine).strip().lower()
        if mode not in CORNER_REFINE_MODES:
            raise ValueError(f"detector.aruco.corner_refine 必须是 {list(CORNER_REFINE_MODES)}，"
                             f"收到 {corner_refine!r}")
        self.dictionary, self.dictionary_name = predefined_dictionary(dictionary)
        self.corner_refine = mode
        self.min_perimeter_rate = float(min_perimeter_rate)
        self.max_detections = int(max_detections)
        if self.min_perimeter_rate <= 0:
            raise ValueError(f"detector.aruco.min_perimeter_rate 必须为正：{self.min_perimeter_rate}")
        if self.max_detections < 0:
            raise ValueError(f"detector.aruco.max_detections 不能为负（0 表示不限）：{self.max_detections}")

        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = {
            "none": cv2.aruco.CORNER_REFINE_NONE,
            "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
            "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
        }[mode]
        params.minMarkerPerimeterRate = self.min_perimeter_rate
        self._detector = cv2.aruco.ArucoDetector(self.dictionary, params)

    # ---------------------------------------------------------------- 契约接口
    def detect(self, image_bgr: np.ndarray) -> list:
        """在 BGR 图上检测 ArUco 码。

        :param image_bgr: HxWx3 uint8 BGR 图（契约要求；单通道灰度会被拒绝）
        :return: `list[ArucoMarker]`，按 `marker_id` 升序；**无码时 `[]`**（不是 None）
        :raises ValueError: 入参不是 HxWx3 uint8 的 BGR 图
        """
        import cv2
        if not isinstance(image_bgr, np.ndarray):
            raise ValueError(f"detect() 需要 np.ndarray，收到 {type(image_bgr).__name__}")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(
                f"契约要求 HxWx3 的 BGR 图，收到形状 {image_bgr.shape}；"
                "单通道灰度图请先 cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)")
        if image_bgr.dtype != np.uint8:
            raise ValueError(f"契约要求 uint8 图，收到 dtype {image_bgr.dtype}")

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return []                            # 契约 2.2 第 1 条：无码返回 []

        markers = []
        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            pts = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
            # ArucoMarker.__post_init__ 会校验形状并归一 dtype，这里再算四角算术平均
            markers.append(ArucoMarker(marker_id=int(marker_id),
                                       corners_px=pts,
                                       center_px=(float(pts[:, 0].mean()),
                                                  float(pts[:, 1].mean()))))

        if self.max_detections:
            markers = self._keep_largest(markers, self.max_detections)
        markers.sort(key=lambda m: m.marker_id)  # 契约 2.2 第 2 条：按 marker_id 升序
        return markers

    # ---------------------------------------------------------------- 内部
    @staticmethod
    def _keep_largest(markers: list, limit: int) -> list:
        """按四角围出的面积从大到小取前 limit 个（面积用鞋带公式算，旋转也准）。"""

        def area_of(marker) -> float:
            pts = np.asarray(marker.corners_px, dtype=np.float64)
            x, y = pts[:, 0], pts[:, 1]
            return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

        return sorted(markers, key=area_of, reverse=True)[:int(limit)]
