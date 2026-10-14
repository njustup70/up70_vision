"""可视化：在 BGR 图上画检测框与标签。

两条硬约定
    1. 图上的文字**一律 ASCII 英文**：cv2.putText 用的 Hershey 矢量字体不含汉字字形，
       传中文会画出问号而且不报错（见 docs/progress.md 坑 11）。中文说明只进日志。
    2. 本模块不改变检测结果，只负责呈现（要落盘的数据由 viz/writer.py 负责）。
"""

import cv2
import numpy as np

# 判色结果 -> 画框颜色（BGR）。unknown 用灰色，表示"没判出颜色"。
COLOR_DRAW_BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 215, 255),
    "green": (0, 200, 0),
    "cyan": (255, 255, 0),
    "blue": (255, 128, 0),
    "magenta": (255, 0, 255),
    "unknown": (160, 160, 160),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(name: str | None):
    return COLOR_DRAW_BGR.get(str(name), COLOR_DRAW_BGR["unknown"])


def put_text(image_bgr: np.ndarray, text: str, org, scale: float = 0.5,
             color=(255, 255, 255), thickness: int = 1) -> None:
    """写字。强制 ASCII——非 ASCII 字符会被替换成 '?'，避免出现"看不出问题的问题"。"""
    safe = str(text).encode("ascii", "replace").decode("ascii")
    cv2.putText(image_bgr, safe, (int(org[0]), int(org[1])), FONT, scale, color, thickness)


def draw_detections(frame_bgr: np.ndarray,
                    detections,
                    thickness: int = 2,
                    show_center: bool = True,
                    show_label: bool = True) -> np.ndarray:
    """在图上画框 + 标签，返回新图（不改原图）。"""
    canvas = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
        color = color_for(det.color)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if show_center:
            cx, cy = det.center
            cv2.drawMarker(canvas, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        if show_label:
            name = det.color if det.color else "unknown"
            put_text(canvas, f"{name} {det.color_ratio:.2f}",
                     (x1 + 4, y1 + 16), 0.5, (255, 255, 255), 1)
    return canvas


def draw_hud(canvas: np.ndarray, lines, origin=(10, 22), scale: float = 0.6,
             color=(0, 255, 255), line_gap: int = 22) -> None:
    """左上角信息栏（帧号/检出数/汇总之类），原图就地修改。"""
    x, y = origin
    for i, text in enumerate(lines):
        put_text(canvas, text, (x, y + i * line_gap), scale, color, 2)


def make_contact_sheet(images, cols: int = 4, tile_width: int = 320):
    """把多张图拼成一张总览图（调参复盘用）。"""
    if not images:
        raise ValueError("没有图片可拼")
    tiles = []
    for img in images:
        h, w = img.shape[:2]
        scale = tile_width / float(w)
        tiles.append(cv2.resize(img, (tile_width, max(1, int(h * scale)))))
    tile_h = max(t.shape[0] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * tile_h, cols * tile_width, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * tile_h:r * tile_h + tile.shape[0], c * tile_width:(c + 1) * tile_width] = tile
    return sheet
