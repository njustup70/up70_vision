"""合成色块源：不需要任何素材文件，直接造出带 ground truth 的图。

它的价值是**可重复**：同一个 seed 跑两遍逐像素一致，色块的颜色与位置都是已知的，
所以"检出几个、颜色判对几个"能写成具体数字的断言（见 tests/）。
这也是 docs/progress.md 里"自检期望值 总色块 18 = 3 帧 x 6 块"的来源。

排布方式
    把画面切成 ceil(sqrt(n)) x ceil(n/cols) 个格子，每格放一个随机大小、随机位置的
    色块——保证互不重叠、且块数与配置严格一致（seed 只改变位置与噪声，不改变总数）。
"""

from math import ceil, sqrt

import cv2
import numpy as np

from ..types import Frame, bbox_from_xywh
from . import FrameSource

# 内置颜色 → BGR。判色用的是 HSV 阈值表，这里只负责"把颜色画出来"。
COLOR_BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "cyan": (255, 255, 0),
    "blue": (255, 0, 0),
    "magenta": (255, 0, 255),
}


class SyntheticSource(FrameSource):
    name = "synthetic"

    def __init__(self,
                 width: int = 640,
                 height: int = 480,
                 num_frames: int = 3,
                 blocks_per_frame: int = 6,
                 seed: int = 2027,
                 noise_sigma: float = 8.0,
                 background_bgr=(30, 30, 30),
                 colors=None,
                 min_side_ratio: float = 0.12,
                 max_side_ratio: float = 0.22):
        self.width = int(width)
        self.height = int(height)
        self.num_frames = int(num_frames)
        self.blocks_per_frame = int(blocks_per_frame)
        self.seed = int(seed)
        self.noise_sigma = float(noise_sigma)
        self.background_bgr = tuple(int(v) for v in background_bgr)
        self.min_side_ratio = float(min_side_ratio)
        self.max_side_ratio = float(max_side_ratio)

        available = list(colors) if colors else list(COLOR_BGR)
        unknown = [c for c in available if c not in COLOR_BGR]
        if unknown:
            raise ValueError(f"合成源不认识这些颜色 {unknown}；已内置：{sorted(COLOR_BGR)}")
        if self.blocks_per_frame > len(available):
            raise ValueError(
                f"blocks_per_frame({self.blocks_per_frame}) 不能超过可用颜色数({len(available)})，"
                f"否则同一帧里颜色必然重复、判色无法逐块校验")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"画面尺寸非法：{self.width}x{self.height}")
        if self.num_frames < 0:
            raise ValueError(f"num_frames 不能为负：{self.num_frames}")
        if len(self.background_bgr) != 3:
            raise ValueError(f"background_bgr 需要 3 个分量：{self.background_bgr}")

        self.colors = available
        self._rng = np.random.default_rng(self.seed)
        self._index = 0

    # ---------------------------------------------------------------- 内部
    def _cell_grid(self) -> tuple:
        cols = ceil(sqrt(self.blocks_per_frame))
        rows = ceil(self.blocks_per_frame / cols)
        return cols, rows

    def _place_blocks(self) -> tuple:
        """在一帧里排布色块，返回 (图像, ground_truth)。

        ground_truth = ((color_name, (x1,y1,x2,y2)), ...)
        """
        image = np.full((self.height, self.width, 3), self.background_bgr, dtype=np.uint8)
        cols, rows = self._cell_grid()
        cell_w = self.width / cols
        cell_h = self.height / rows
        side_min = max(8.0, min(self.width, self.height) * self.min_side_ratio)
        side_max = max(side_min + 1.0, min(self.width, self.height) * self.max_side_ratio)

        picked = self._rng.choice(self.colors, size=self.blocks_per_frame, replace=False)
        truth = []
        for i, color in enumerate(picked):
            row, col = divmod(i, cols)
            side = float(self._rng.uniform(side_min, side_max))
            side = min(side, cell_w * 0.8, cell_h * 0.8)
            margin_x = max(0.0, (cell_w - side) / 2.0)
            margin_y = max(0.0, (cell_h - side) / 2.0)
            cx = col * cell_w + margin_x + float(self._rng.uniform(0.0, max(1.0, margin_x * 2)))
            cy = row * cell_h + margin_y + float(self._rng.uniform(0.0, max(1.0, margin_y * 2)))
            x1 = int(round(min(max(cx - side / 2.0, col * cell_w), (col + 1) * cell_w - side)))
            y1 = int(round(min(max(cy - side / 2.0, row * cell_h), (row + 1) * cell_h - side)))
            x2 = int(round(min(x1 + side, self.width)))
            y2 = int(round(min(y1 + side, self.height)))
            cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), COLOR_BGR[str(color)], thickness=-1)
            truth.append((str(color), bbox_from_xywh(x1, y1, x2 - x1, y2 - y1)))

        if self.noise_sigma > 0:
            noise = self._rng.normal(0.0, self.noise_sigma, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image, tuple(truth)

    # ---------------------------------------------------------------- 接口
    def read(self) -> Frame | None:
        if self._index >= self.num_frames:
            return None
        image_bgr, truth = self._place_blocks()
        frame = Frame(image_bgr=image_bgr,
                      index=self._index,
                      source="synthetic",
                      stamp=0.0,
                      ground_truth=truth)
        self._index += 1
        return frame

    def close(self) -> None:
        return None
