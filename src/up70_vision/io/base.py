"""输入层接口：把「图像从哪来」收进统一接口。

对外只暴露 read() -> Frame | None：
    None 表示这一路输入已经结束。上层管线不关心图像是合成的、回放的、还是实拍的。

依赖谁
    只依赖 core.types（Frame）。配置由上层拆成普通参数传进来，
    因此本模块不 import config，避免包级依赖打结。

摄像头（camera.py）
    本节目的相机是海康 MV-CU013-A0UC，走 **USB3 Vision（GenICam）**，
    不是 UVC，也没有 /dev/video*，因此 cv2.VideoCapture 打不开它。
    已实测：Aravis 0.8.20 / 0.8.33 都取不到帧（相机 SIRM_PAYLOAD_SIZE 恒为 1MB、
    不随请求更新），只有官方 MVS SDK 能出帧。详见 docs/progress.md 第五节。
    因此 camera 源实现在 io/camera.py（MVS SDK），**没有 MVS 的机器也能
    import 本包**，只在真正接相机时报错并提示怎么装（见 .devcontainer/pkgs/README.md）。
"""

from abc import ABC, abstractmethod

from ..core.types import Frame
from ..utils.logging import get_logger

log = get_logger("io.base")

SOURCE_NAMES = ("synthetic", "replay", "camera")


class FrameSource(ABC):
    """输入源接口。实现类只需保证：read() 返回 Frame 或 None。"""

    name: str = "base"

    @abstractmethod
    def read(self) -> Frame | None:
        """取下一帧；没有更多帧时返回 None。"""
        raise NotImplementedError

    def close(self) -> None:
        """释放资源（默认什么都不做）。"""
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __iter__(self):
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame


def build_source(source_type: str,
                 synthetic_params: dict | None = None,
                 replay_params: dict | None = None,
                 camera_params: dict | None = None,
                 colors: list | None = None) -> FrameSource:
    """工厂：按 input.source 构造输入源。

    :param source_type: "synthetic" / "replay" / "camera"
    :param synthetic_params: configs/default.yaml 的 input.synthetic 段
    :param replay_params: configs/default.yaml 的 input.replay 段
    :param camera_params: configs/default.yaml 的 input.camera 段
    :param colors: 合成源要用的颜色名列表（应与阈值表里的颜色一致）
    :raises ValueError: 名字未知
    :raises RuntimeError: 选了 camera 但镜像/宿主机没有 MVS SDK
    """
    key = str(source_type).strip().lower()
    if key == "synthetic":
        from .synthetic import SyntheticSource
        return SyntheticSource(colors=colors, **dict(synthetic_params or {}))
    if key == "replay":
        from .replay import ReplaySource
        return ReplaySource(**dict(replay_params or {}))
    if key == "camera":
        from .camera import MvsCameraSource
        return MvsCameraSource(**dict(camera_params or {}))
    raise ValueError(f"未知输入源 {source_type!r}；可用值：{list(SOURCE_NAMES)}")
