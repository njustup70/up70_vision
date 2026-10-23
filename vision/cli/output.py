"""CLI 的输出层：落盘（PNG/JSON/CSV/summary）与 HUD 绘制（D13/D15）。

为什么在 cli/ 而不是根级或功能包：
    `OutputWriter` 只有批处理路径用；`write_image_with_truth` 是 lab 借用。
    它不是「两个功能包都要用」的共用件，所以不进根级（D15 的判据）。
"""

from __future__ import annotations

import csv
import json
import logging
import os

import cv2
import numpy as np

from ..aruco.draw import draw_markers
from ..types import Detection, Frame, is_marker
from ..yolo.draw import draw_detections

log = logging.getLogger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_text(image_bgr, text: str, org, scale: float = 0.5,
             color=(255, 255, 255), thickness: int = 1, outline: bool = False) -> None:
    """在图上写一行 ASCII 文字（可带黑描边）；中文会变成 ???（Hershey 无汉字）。"""
    if outline:
        cv2.putText(image_bgr, text, org, FONT, scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
    cv2.putText(image_bgr, text, org, FONT, scale, color, thickness, cv2.LINE_AA)

def ensure_dir(path: str) -> str:
    """确保目录存在，返回该目录（方便链式调用）。"""
    if not path:
        raise ValueError("目录路径不能为空")
    os.makedirs(path, exist_ok=True)
    return path

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

def write_image_with_truth(image_bgr: np.ndarray, image_path: str, truth: dict) -> tuple:
    """把一张图和它的真值写成 `<name>.png` + `<name>.json`（同名同目录）。

    两个生成器（`gen_aruco`、`gen_pose_scenes`）原先各写一遍这段"存图 + 存 JSON"，
    产物还略有出入（一个末尾带换行、一个不带）；合并后真值文件的格式只有一处定义。

    :param image_bgr: 要写的图（HxWx3 BGR）
    :param image_path: 图片路径；真值路径由它换后缀得到
    :param truth: 真值字典（直接 `json.dump`）
    :return: `(png 路径, json 路径)`
    """
    json_path = os.path.splitext(image_path)[0] + ".json"
    ensure_dir(os.path.dirname(image_path) or ".")
    write_image(image_path, image_bgr)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(truth, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return image_path, json_path

def draw_all(frame_bgr: np.ndarray, detections) -> np.ndarray:
    """按检出物类型分流到对应功能的绘制：码走 ArUco，检测框走 YOLO。

    分流放在 cli 里（D13）：两个功能包互不 import，谁也别画别人的东西。
    """
    if detections and is_marker(detections[0]):
        return draw_markers(frame_bgr, detections)
    return draw_detections(frame_bgr, detections)


def draw_hud(canvas: np.ndarray, lines, origin=(10, 22), scale: float = 0.6,
             color=(0, 255, 255), line_gap: int = 22) -> None:
    """左上角信息栏（帧号/检出数/汇总之类），原图就地修改。"""
    x, y = origin
    for i, text in enumerate(lines):
        put_text(canvas, text, (x, y + i * line_gap), scale, color, 2, outline=True)


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
                x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
                if is_marker(det):
                    # ArUco 码没有颜色语义：color 写 "-"、ratio/valid_pixels 写 0，ID 进 marker_id 列
                    row_color, ratio, valid, score = "-", 0.0, 0, 1.0
                    kind, marker_id = "marker", int(det.marker_id)
                else:
                    row_color = det.color if det.color else "unknown"
                    ratio = round(float(det.color_ratio), 4)
                    valid = int(det.valid_pixels)
                    score = round(float(det.score), 4)
                    kind, marker_id = "detection", ""
                self._rows.append([
                    frame.index, frame.source, row_color, ratio, valid, score,
                    x1, y1, x2, y2, round(cx, 2), round(cy, 2), kind, marker_id,
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
