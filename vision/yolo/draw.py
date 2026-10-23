"""YOLO/判色族绘制：检测框、类别或颜色标签。线宽偏粗，与 ArUco 角点区分（D13）。"""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_text(image_bgr, text: str, org, scale: float = 0.5,
             color=(255, 255, 255), thickness: int = 1, outline: bool = False) -> None:
    """在图上写一行 ASCII 文字（可带黑描边）。写中文会变成 `???`——Hershey 没有汉字字形。

    这个几行小工具在 `aruco/draw.py`、`yolo/draw.py`、`cli/output.py` 各留一份：
    D13 明确规定画图风格可以分叉，重复这点代码换取功能包之间零依赖是划算的。
    """
    if outline:
        cv2.putText(image_bgr, text, org, FONT, scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
    cv2.putText(image_bgr, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def color_for(name: str | None):
    return COLOR_DRAW_BGR.get(str(name), COLOR_DRAW_BGR["unknown"])

def detection_label(det) -> tuple[str, float]:
    """拼检测框的标签文字与数值。

    为什么不能只按颜色拼
        判色族（`blob`）的 `class_name` 恒为 "unknown"，只有 `color` 有意义；
        而 YOLO（`detector.name=yolo`）恰好相反：`color` 是 None（框里没有我们定义的
        颜色），`class_name` 才是真信息（"person"/"bus"）。原先一律按颜色拼，
        YOLO 的框就全显示成 `unknown`，还把 `color_ratio` 当成了置信度显示。

    :param det: `Detection`
    :return: `(标签文字, 数值)`——有真实类名时是 `(类名, 置信度)`，
             否则是 `(颜色或 "unknown", 颜色命中率)`
    """
    name = str(getattr(det, "class_name", "") or "")
    if name and name != "unknown":
        return name, float(getattr(det, "score", 1.0))
    return (det.color or "unknown"), float(getattr(det, "color_ratio", 0.0))

def draw_detections(frame_bgr: np.ndarray,
                    detections,
                    thickness: int = 2,
                    show_center: bool = True,
                    show_label: bool = True) -> np.ndarray:
    """在图上画框/四角 + 标签 + 中心，返回新图（不改原图）。

    两种检出物（docs/interface.md 第 2 节）
        · `Detection`（判色族）：按 `bbox_xyxy` 画轴对齐矩形，标签是颜色与命中率；
      若检测器给了真实类名（YOLO），标签改用「类名 + 置信度」（见 `detection_label`）
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
            name, metric = detection_label(det)
            label = f"{name} {metric:.2f}"
            label_org = (x1 + 4, y1 + 16)
        if show_center:
            cv2.drawMarker(canvas, (int(center[0]), int(center[1])), (0, 0, 255),
                           cv2.MARKER_CROSS, 14, 2)
        if show_label:
            put_text(canvas, label, label_org, 0.5, (255, 255, 255), 1, outline=True)
    return canvas
