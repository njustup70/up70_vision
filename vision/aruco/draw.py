"""ArUco 专用绘制：码的四边形、id 标签与位姿坐标轴（D13）。

图上文字一律 ASCII：Hershey 字体没有汉字字形，中文会画成问号且不报错。
"""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

#: ArUco 码固定绿框：码没有颜色语义，不借用判色调色板
MARKER_DRAW_BGR = (0, 255, 0)


def put_text(image_bgr, text: str, org, scale: float = 0.5,
             color=(255, 255, 255), thickness: int = 1, outline: bool = False) -> None:
    """在图上写一行 ASCII 文字（可带黑描边）。中文会变成 ???（Hershey 无汉字字形）。

    这个几行小工具在 `aruco/draw.py`、`yolo/draw.py`、`cli/output.py` 各留一份：
    D13 允许画图风格分叉，重复这点代码换取功能包之间零依赖。
    """
    if outline:
        cv2.putText(image_bgr, text, org, FONT, scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
    cv2.putText(image_bgr, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_markers(frame_bgr: np.ndarray,
                 markers,
                 thickness: int = 2,
                 show_center: bool = True,
                 show_label: bool = True) -> np.ndarray:
    """画每个码的四边形、中心与 `id=N` 标签，返回新图（不改原图）。

    用四边形而不是矩形：码旋转后 bbox 会包住一圈背景，四边形才贴得住。
    """
    canvas = frame_bgr.copy()
    for marker in markers:
        pts = np.asarray(marker.corners_px, dtype=np.int32).reshape(-1, 1, 2)
        flat = pts.reshape(-1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=MARKER_DRAW_BGR, thickness=thickness)
        center = marker.center_px
        if show_center:
            cv2.drawMarker(canvas, (int(center[0]), int(center[1])), (0, 0, 255),
                           cv2.MARKER_CROSS, 14, 2)
        if show_label:
            label = f"id={int(marker.marker_id)}"
            org = (int(flat[:, 0].min()) + 4, max(14, int(flat[:, 1].min()) - 6))
            put_text(canvas, label, org, 0.5, (255, 255, 255), 1, outline=True)
    return canvas


def draw_pose_axes(frame_bgr: np.ndarray,
                   poses,
                   camera_matrix,
                   axis_mm: float,
                   thickness: int = 2,
                   show_labels: bool = True,
                   dist_coeffs=None) -> np.ndarray:
    """在图上画每个码的三维坐标轴，返回新图（不改原图）。

    用大白话说
        每个码画一个"小十字架"：红轴指向码的右边、绿轴指向码的上边、蓝轴垂直纸面
        往外戳。看图的人一眼就能判断"码是正对镜头还是斜着、歪了多少"——
        光看四边形框看不出倾斜，看轴的透视缩短就很直观。

    为什么轴长要由调用方给
        `viz/` 不知道码有多大（那在 `pose.marker_size_mm` 里）。默认取边长的 0.6 倍
        是"够长能看清、又不至于盖住旁边的码"，但真正合适的值取决于画面里码的密度。

    :param frame_bgr: 原始 BGR 图
    :param poses: `list[MarkerPose]`；空列表返回原图的副本
    :param camera_matrix: (3,3) 内参（与解位姿时同一个）
    :param axis_mm: 轴长（毫米）；一般取 `pose.marker_size_mm * 0.6`
    :param thickness: 线宽（像素）
    :param show_labels: 是否在轴端写 x/y/z（ASCII）
    :param dist_coeffs: 投影时用的畸变系数。**必须与解位姿时的口径一致**：
        解算时把 `D` 传进了 PnP（原始畸变像素空间），这里就要传同一个 `D`；
        解算时角点已 `undistort_points(..., P=K)` 掰直、PnP 传的是 None，这里也传 None。
    :return: 画好轴的新图

    为什么口径必须成对
        轴是"码坐标系里的三个点"投到图像上。投影用的 (K, D) 与解位姿用的不一致，
        轴就会与码的四个角对不齐。实测偏差随畸变增大：中间区域几像素、画面边缘更明显。
        早期版本固定用 `dist=None` 投影，而两条真实调用路径都是带 `D` 解的，
        属于"注释写着去畸变空间、代码在原始空间"的口径错位（复核意见 P3-4）。
    """
    canvas = frame_bgr.copy()
    if not poses:
        return canvas

    origin = np.zeros((1, 3), dtype=np.float64)
    axis_points = np.array([[float(axis_mm), 0.0, 0.0],
                            [0.0, float(axis_mm), 0.0],
                            [0.0, 0.0, float(axis_mm)]], dtype=np.float64)
    # x 红、y 绿、z 蓝（BGR 顺序），与 OpenCV 的惯例一致，别改成"好看的配色"
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    labels = ("x", "y", "z")

    for pose in poses:
        base, _ = cv2.projectPoints(origin, pose.rvec, pose.tvec_mm, camera_matrix, dist_coeffs)
        tips, _ = cv2.projectPoints(axis_points, pose.rvec, pose.tvec_mm, camera_matrix,
                                    dist_coeffs)
        # reshape：projectPoints 返回 (N,1,2)，直接索引会拿到嵌套数组
        origin_px = tuple(int(round(v)) for v in np.asarray(base).reshape(2))
        for tip, color, label in zip(np.asarray(tips).reshape(3, 2), colors, labels):
            tip_px = tuple(int(round(v)) for v in tip)
            cv2.arrowedLine(canvas, origin_px, tip_px, color, thickness, tipLength=0.2)
            if show_labels:
                put_text(canvas, label, (tip_px[0] + 2, tip_px[1] - 2), 0.4, color, 1, outline=True)
        cv2.circle(canvas, origin_px, 2, (255, 255, 255), -1)
    return canvas
