"""图像读写：薄封装，把 OpenCV 两个"静默失败"接口管起来。

为什么需要这层
    cv2.imread 读不到文件返回 None、cv2.imwrite 失败返回 False，两者都不抛异常
    （见 docs/progress.md 坑 6、7）。业务代码里散落 if 判断容易漏，
    所以统一在这里判掉并抛出明确异常。

平台说明
    标准环境是 Ubuntu（Docker 镜像里也是），cv2 原生支持 UTF-8 路径，
    因此不需要为 Windows 的 ANSI 路径写 fromfile/tofile 兜底。
"""

import os

import cv2
import numpy as np

from .logging import get_logger

log = get_logger("utils.image_io")


def ensure_dir(path: str) -> str:
    """确保目录存在，返回该目录（方便链式调用）。"""
    if not path:
        raise ValueError("目录路径不能为空")
    os.makedirs(path, exist_ok=True)
    return path


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


def write_image(path: str, image_bgr: np.ndarray) -> str:
    """写图，显式检查返回值。

    :raises IOError: cv2.imwrite 返回 False（父目录不存在、无写权限、扩展名不支持等）
    """
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        raise ValueError("要写入的图像为空")
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    ok = cv2.imwrite(path, image_bgr)
    if not ok:
        raise IOError(f"cv2.imwrite 返回 False，写入失败: {path}")
    log.debug("写出图片 %s 形状=%s", path, image_bgr.shape)
    return path


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    """BGR -> 灰度。单独列出是为了让调用方不用记 cv2 的颜色码。"""
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """BGR -> HSV。**判色前必须过这一步**（把 BGR 当 HSV 用不会报错，只会全判错）。"""
    if image_bgr.ndim == 2:
        raise ValueError("判色需要三通道 BGR 图，收到单通道灰度图")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
