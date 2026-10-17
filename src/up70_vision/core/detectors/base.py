"""检测器插槽：把「出候选框」一步做成可替换的。

放什么
    Detector 抽象基类 + 按名字构造的工厂。
不放什么
    不放判色逻辑。检测器只回答"哪里有东西"，"是什么颜色"由 core/color.py 填。
    这个切分让两阶段独立替换：换模型不用动判色，调阈值不用动检测。
依赖谁
    只依赖 core.types。上层（cli/config）把配置拆成普通参数传进来，
    因此本模块 **不 import config**，避免包级循环依赖。

返回值结构按检测器族分两种（docs/interface.md 第 3 节）
    · 判色族（`blob`）：`list[Detection]`，轴对齐 bbox，中心 = bbox 中心，带颜色字段
    · ArUco（`aruco` / `stub_aruco`）：`list[ArucoMarker]`，四角点，中心 = 四角算术平均
    两者都：**无目标返回 `[]`**（不是 None，不抛异常）；形参名固定 `image_bgr`。
"""

from abc import ABC, abstractmethod

# 工厂认识的名字（configs/default.yaml 的 detector.name 只能填这些）
DETECTOR_NAMES = ("blob", "stub_aruco", "aruco")


class Detector(ABC):
    """检测器接口。实现类必须是纯算法：给 numpy 图，返回检出物列表。"""

    name: str = "base"

    @abstractmethod
    def detect(self, image_bgr) -> list:
        """在 BGR 图上出检出物。

        :param image_bgr: HxWx3 uint8 BGR 图（**形参名是契约**：允许 `detect(image_bgr=x)`）
        :return: `list[Detection]`（判色族）或 `list[ArucoMarker]`（ArUco）；
                 **什么都不返回时给 `[]`，不要给 None**。
                 判色族的 bbox 用绝对像素坐标、右/下半开上界，class_name 填 "unknown"、
                 class_id 填 -1（判色阶段才填颜色）；ArUco 的码没有颜色。
        """
        raise NotImplementedError


def build_detector(name: str, params: dict | None = None) -> Detector:
    """工厂：按名字构造检测器。

    :param name: "blob"（判色占位检测器）、"stub_aruco"（按契约返回假码的 stub）
                 或 "aruco"（真 ArUco，DICT_4X4_50）
    :param params: 该检测器的参数字典（来自 configs/default.yaml 的 detector.<name> 段）
    :raises ValueError: 名字未知（报错里列出可用值）
    :raises RuntimeError: 选了 aruco 但当前 OpenCV 没有新版 aruco API（需 >= 4.7）
    """
    params = dict(params or {})
    key = str(name).strip().lower()
    if key == "blob":
        from .blob import BlobDetector      # 局部 import：新增检测器时不必改这里
        return BlobDetector(**params)
    if key in ("stub_aruco", "stub"):
        from .stub_aruco import StubArucoDetector
        return StubArucoDetector(**params)
    if key == "aruco":
        from .aruco import ArucoDetector
        return ArucoDetector(**params)
    raise ValueError(f"未知检测器 {name!r}；当前可用：{list(DETECTOR_NAMES)}")
