"""检测器插槽：把「出候选框」一步做成可替换的。

放什么
    Detector 抽象基类 + 按名字构造的工厂。
不放什么
    不放判色逻辑。检测器只回答"哪里有东西"，"是什么颜色"由 core/color.py 填。
    这个切分让两阶段独立替换：换模型不用动判色，调阈值不用动检测。
依赖谁
    只依赖 core.types。上层（cli/config）把配置拆成普通参数传进来，
    因此本模块 **不 import config**，避免包级循环依赖。
"""

from abc import ABC, abstractmethod

from ..types import Detection


class Detector(ABC):
    """检测器接口。实现类必须是纯算法：给 numpy 图，返回 Detection 列表。"""

    name: str = "base"

    @abstractmethod
    def detect(self, frame_bgr) -> list:
        """在 BGR 图上出候选框。

        :param frame_bgr: HxWx3 uint8 BGR 图
        :return: Detection 列表；bbox 用绝对像素坐标、右/下半开上界；
                 class_name 填 "unknown"、class_id 填 -1（判色阶段才填颜色）
        """
        raise NotImplementedError


def build_detector(name: str, params: dict | None = None) -> Detector:
    """工厂：按名字构造检测器。

    :param name: 目前支持 "blob"
    :param params: 该检测器的参数字典（来自 configs/default.yaml 的 detector.<name> 段）
    :raises ValueError: 名字未知（报错里列出可用值）
    """
    params = dict(params or {})
    key = str(name).strip().lower()
    if key == "blob":
        from .blob import BlobDetector      # 局部 import：新增检测器时不必改这里
        return BlobDetector(**params)
    raise ValueError(f"未知检测器 {name!r}；当前可用：['blob']")
