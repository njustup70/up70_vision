"""输入层：把「图像从哪来」收进统一接口。

放什么
    所有输入源，对外只暴露 read() -> Frame | None。上层管线不关心图像是
    合成的、回放的、还是摄像头实拍的。

不放什么
    不放检测或判色逻辑。

依赖谁
    只依赖 types（Frame）。**不 import config** —— 配置由 `cli/run.py` 拆成普通参数
    传给 build_source（见本文件的 build_source）。这一行以前写的是"依赖 types 与 config"，
    与 base.py 和 docs/interface.md 第 5 节矛盾，按契约欠账三改正。

当前状态
    本文件（FrameSource 接口 + build_source 工厂 + read_image_bgr）、synthetic.py（合成图，带 ground truth）、
    replay.py（图片 / 视频回放）、camera.py（真相机，走 MVS SDK）均已就位。

关于真相机（重要）
    本节目的相机是海康 MV-CU013-A0UC，走 **USB3 Vision（GenICam），不是 UVC**：
    宿主机上没有 /dev/video*，也没有"用户在 video 组"这回事——权限靠 udev 规则给，
    设备节点是 /dev/bus/usb/...。所以 cv2.VideoCapture 打不开它，OpenCV 的 UVC 路径
    在这颗相机上用不了；必须走厂商 SDK（MVS），实现见 camera.py。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import cv2
import numpy as np

from ..types import Frame
from ..logging import get_logger

log = get_logger(__name__)


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


def read_image_bgr(path: str) -> np.ndarray:
    """读图，返回 BGR uint8 数组（H, W, 3）。

    :raises FileNotFoundError: 路径不存在或不是可解码的图片
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"图片不存在: {path}")
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"cv2.imread 返回 None（文件损坏或不是图片）: {path}")
    log.debug("读入图片 %s 形状=%s", path, image_bgr.shape)
    return image_bgr
