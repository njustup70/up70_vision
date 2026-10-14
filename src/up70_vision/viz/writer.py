"""落盘：把结果写成叠加图 / JSON / CSV，并管理输出目录。

放什么
    叠加图、逐帧 JSON、逐检测 CSV 的写入与目录管理。
不放什么
    不改变检测结果，只负责呈现。落盘的数据要能被 pandas / numpy 直接读成数组
    （CSV 一行一个检测；JSON 一帧一个对象）。

三个文件
    <dir>/frame_0000.png   叠加图（每帧一张；save_overlay 控制）
    <dir>/results.json     每帧一条记录（含 detections 明细）
    <dir>/detections.csv   一行一个检测，列固定（便于 pandas.read_csv 直接读）
    <dir>/summary.json     summarize() 的汇总（自检数字都在这）
"""

import csv
import json
import os

from ..utils.image_io import ensure_dir, write_image
from ..utils.logging import get_logger

log = get_logger("viz.writer")

CSV_HEADER = ["frame", "source", "color", "ratio", "valid_pixels", "score",
              "x1", "y1", "x2", "y2", "cx", "cy"]


class OutputWriter:
    def __init__(self,
                 out_dir: str,
                 enabled: bool = True,
                 save_overlay: bool = True,
                 save_json: bool = True,
                 save_csv: bool = True,
                 prefix: str = "frame"):
        self.out_dir = str(out_dir)
        self.enabled = bool(enabled)
        self.save_overlay = bool(save_overlay)
        self.save_json = bool(save_json)
        self.save_csv = bool(save_csv)
        self.prefix = str(prefix)
        self._frames = []
        self._rows = []
        self.written = {"overlays": 0, "json": None, "csv": None, "summary": None}
        if self.enabled:
            ensure_dir(self.out_dir)

    # ---------------------------------------------------------------- 写入
    def write_frame(self, frame, result, overlay_bgr=None) -> dict:
        """写一帧（叠加图立刻落盘，JSON/CSV 先在内存里攒，close() 时一次写出）。"""
        paths = {}
        if not self.enabled:
            return paths
        if self.save_overlay and overlay_bgr is not None:
            path = os.path.join(self.out_dir, f"{self.prefix}_{frame.index:04d}.png")
            paths["overlay"] = write_image(path, overlay_bgr)
            self.written["overlays"] += 1
        if self.save_json:
            self._frames.append(result.to_dict())
        if self.save_csv:
            for det in result.detections:
                cx, cy = det.center
                self._rows.append([
                    frame.index, frame.source,
                    det.color if det.color else "unknown",
                    round(float(det.color_ratio), 4), int(det.valid_pixels),
                    round(float(det.score), 4),
                    int(det.bbox_xyxy[0]), int(det.bbox_xyxy[1]),
                    int(det.bbox_xyxy[2]), int(det.bbox_xyxy[3]),
                    round(cx, 2), round(cy, 2),
                ])
        return paths

    def close(self, summary: dict | None = None) -> dict:
        """把所有缓存写到磁盘，返回写出的文件路径字典。"""
        if not self.enabled:
            return self.written
        if self.save_json and self._frames:
            path = os.path.join(self.out_dir, "results.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._frames, f, ensure_ascii=False, indent=2)
            self.written["json"] = path
        if self.save_csv:
            path = os.path.join(self.out_dir, "detections.csv")
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
                writer.writerows(self._rows)
            self.written["csv"] = path
        if summary is not None:
            path = os.path.join(self.out_dir, "summary.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            self.written["summary"] = path
        log.info("落盘完成：叠加图 %d 张 -> %s", self.written["overlays"], self.out_dir)
        return self.written
