"""回放源：把一个图片目录或一个视频文件当成帧序列来喂。

为什么需要它
    直接接摄像头做不到"同一段素材跑两遍逐帧一致"，而对比两次调参效果恰恰需要这个
    （见 data/README.md）。把现场录下来的视频放进来，阈值怎么改都能复盘。

两种模式
    kind=auto（默认）：path 是目录 -> 图片序列；是文件 -> 视频
    frame_stride：隔几帧取一帧（视频很大时用来抽样）
    max_frames：最多取多少帧（0 = 不限）
"""

import os

import cv2

from ..core.types import Frame
from ..utils.image_io import read_image_bgr
from ..utils.logging import get_logger
from .base import FrameSource

log = get_logger("io.replay")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class ReplaySource(FrameSource):
    name = "replay"

    def __init__(self, path: str, kind: str = "auto", frame_stride: int = 1, max_frames: int = 0):
        self.path = str(path)
        self.kind = str(kind).strip().lower()
        self.frame_stride = max(1, int(frame_stride))
        self.max_frames = max(0, int(max_frames))
        self._index = 0          # 对外发布的帧序号
        self._cursor = 0         # 素材内部游标（按 stride 递增）
        self._images = []
        self._cap = None

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"回放路径不存在：{self.path}")

        if self.kind not in ("auto", "image_dir", "video"):
            raise ValueError(f"kind 必须是 auto/image_dir/video，收到 {kind!r}")

        is_dir = os.path.isdir(self.path)
        mode = self.kind
        if mode == "auto":
            mode = "image_dir" if is_dir else "video"
        if mode == "image_dir" and not is_dir:
            raise ValueError(f"kind=image_dir 但 {self.path} 不是目录")

        if mode == "image_dir":
            names = sorted(n for n in os.listdir(self.path) if n.lower().endswith(IMAGE_SUFFIXES))
            if not names:
                raise ValueError(f"目录里没有图片（支持 {IMAGE_SUFFIXES}）：{self.path}")
            self._images = [os.path.join(self.path, n) for n in names]
            self.mode = "image_dir"
            total = len(self._images)
            log.info("回放图片序列：%d 张（stride=%d, max_frames=%d）",
                     total, self.frame_stride, self.max_frames)
        else:
            self._cap = cv2.VideoCapture(self.path)
            if not self._cap.isOpened():
                raise IOError(f"cv2.VideoCapture 打不开视频：{self.path}")
            self.mode = "video"
            log.info("回放视频：%s（stride=%d, max_frames=%d）",
                     os.path.basename(self.path), self.frame_stride, self.max_frames)

    # ---------------------------------------------------------------- 内部
    def _read_image_frame(self) -> Frame | None:
        pos = self._cursor * self.frame_stride
        self._cursor += 1
        if pos >= len(self._images):
            return None
        if self.max_frames and self._index >= self.max_frames:
            return None
        path = self._images[pos]
        image_bgr = read_image_bgr(path)
        frame = Frame(image_bgr=image_bgr, index=self._index,
                      source=f"replay:{os.path.basename(path)}", stamp=0.0)
        self._index += 1
        return frame

    def _read_video_frame(self) -> Frame | None:
        if self.max_frames and self._index >= self.max_frames:
            return None
        # 先按 stride 丢弃中间帧
        for _ in range(self.frame_stride - 1):
            if not self._cap.grab():
                return None
        ok, image_bgr = self._cap.read()
        if not ok or image_bgr is None:
            return None
        frame = Frame(image_bgr=image_bgr, index=self._index,
                      source=f"replay:{os.path.basename(self.path)}",
                      stamp=float(self._cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
        self._index += 1
        return frame

    # ---------------------------------------------------------------- 接口
    def read(self) -> Frame | None:
        if self.mode == "image_dir":
            return self._read_image_frame()
        return self._read_video_frame()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
