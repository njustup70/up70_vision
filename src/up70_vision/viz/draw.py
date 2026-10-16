"""可视化：在 BGR 图上画检测框与标签。

两条硬约定
    1. 图上的文字**一律 ASCII 英文**：cv2.putText 用的 Hershey 矢量字体不含汉字字形，
       传中文会画出问号而且不报错（见 docs/progress.md 坑 11）。中文说明只进日志。
    2. 本模块不改变检测结果，只负责呈现（要落盘的数据由 viz/writer.py 负责）。
"""

import cv2
import numpy as np

from ..core.types import is_marker

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

# ArUco 码固定用绿框：码没有颜色语义，不该借用上面那套判色调色板
MARKER_DRAW_BGR = (0, 255, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(name: str | None):
    return COLOR_DRAW_BGR.get(str(name), COLOR_DRAW_BGR["unknown"])


def put_text(image_bgr: np.ndarray, text: str, org, scale: float = 0.5,
             color=(255, 255, 255), thickness: int = 1, outline: bool = False) -> None:
    """写字。强制 ASCII——非 ASCII 字符会被替换成 '?'，避免出现"看不出问题的问题"。

    :param outline: True 时先描一层黑边再写字。**白底上画白字是什么都看不见的**
        （浅色场景 / 白底素材上标签会整个消失，且不报错），加黑描边后
        白底、黑底、彩色底都读得清，不必按背景改字色。
    """
    safe = str(text).encode("ascii", "replace").decode("ascii")
    pos = (int(org[0]), int(org[1]))
    if outline:
        cv2.putText(image_bgr, safe, pos, FONT, scale, (0, 0, 0), int(thickness) + 2)
    cv2.putText(image_bgr, safe, pos, FONT, scale, color, thickness)


def draw_detections(frame_bgr: np.ndarray,
                    detections,
                    thickness: int = 2,
                    show_center: bool = True,
                    show_label: bool = True) -> np.ndarray:
    """在图上画框/四角 + 标签 + 中心，返回新图（不改原图）。

    两种检出物（docs/interface.md 第 2 节）
        · `Detection`（判色族）：按 `bbox_xyxy` 画轴对齐矩形，标签是颜色与命中率
        · `ArucoMarker`：按 `corners_px` 画**四边形**（旋转过的码也贴得住，
          矩形会丢掉旋转信息），标签是 `id=N`，中心用契约的 `center_px`

    两处容易画错的细节
        · bbox 是**半开上界**（覆盖 x1..x2-1），而 `cv2.rectangle` 是闭区间：
          直接传 x2,y2 会多画 1 px，所以这里传 `x2-1, y2-1`
        · 标签一律带黑描边：白底上白字看不见，且不报错（复核意见二.2）
    """
    canvas = frame_bgr.copy()
    for det in detections:
        if is_marker(det):
            pts = np.asarray(det.corners_px, dtype=np.int32).reshape(-1, 1, 2)
            flat = pts.reshape(-1, 2)
            cv2.polylines(canvas, [pts], isClosed=True, color=MARKER_DRAW_BGR,
                          thickness=thickness)
            center = det.center_px
            label = f"id={int(det.marker_id)}"
            label_org = (int(flat[:, 0].min()) + 4, max(14, int(flat[:, 1].min()) - 6))
        else:
            x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
            cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), color_for(det.color), thickness)
            center = det.center
            name = det.color if det.color else "unknown"
            label = f"{name} {det.color_ratio:.2f}"
            label_org = (x1 + 4, y1 + 16)
        if show_center:
            cv2.drawMarker(canvas, (int(center[0]), int(center[1])), (0, 0, 255),
                           cv2.MARKER_CROSS, 14, 2)
        if show_label:
            put_text(canvas, label, label_org, 0.5, (255, 255, 255), 1, outline=True)
    return canvas


def draw_hud(canvas: np.ndarray, lines, origin=(10, 22), scale: float = 0.6,
             color=(0, 255, 255), line_gap: int = 22) -> None:
    """左上角信息栏（帧号/检出数/汇总之类），原图就地修改。"""
    x, y = origin
    for i, text in enumerate(lines):
        put_text(canvas, text, (x, y + i * line_gap), scale, color, 2, outline=True)


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
