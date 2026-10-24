"""深度图与点云的基本运算：不包含任何抓取/吸盘的业务判断。

用大白话说
    这个文件只回答「数」的问题：像素怎么变成三维点、点怎么变少、哪一堆点构成
    一个平面、哪些像素连成一团。它不认识「物块」「顶面」「可吸」这些概念。

准入规则（判断一段新代码该不该放这里）
    能用一句不提到「吸盘 / 抓取 / 吸附面」的话完整描述它的输入输出吗？
        能   -> 放这里（本文件）
        不能 -> 放 `core.py`（那是抓取业务）
    例子：`backproject(depth, k) -> 点云` 不用提吸盘，属于这里；
         `score_suction(...) -> 可吸性分数` 必然要提吸盘，属于 `core.py`。

为什么单独拆出来
    反投影、体素降采样、平面拟合、深度图连通域这四件事与吸盘无关，将来做
    深度-彩色对齐、手眼标定、多目标分割时都要再用。混在 `core.py` 里会让
    那个文件同时承担两种抽象级别，改一处要读整篇。
    只依赖 numpy 与 opencv，不 import ROS，可单独进 pytest。

单位
    点坐标一律**毫米**，深度一律**毫米**（Orbbec 16UC1 + enable_depth_scale=true 的
    原生输出）。若上游改成米，这里所有阈值会整体错一个量级。
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import Intrinsics

#: 三点确定一个平面；少于这个数谈不上「拟合」，是输入错误而不是结果不好。
MIN_PLANE_POINTS = 3


def backproject(depth_mm: np.ndarray, intrinsics: Intrinsics) -> tuple:
    """深度图 -> 点云。返回 (points (N,3) 毫米, pixel_uv (N,2))。

    深度是 z 深度（沿光轴），所以是 x=(u-cx)z/fx、y=(v-cy)z/fy 的直接代入。
    把 z 深度当欧氏距离会让点云边缘整体外扩——两者只在图像正中心相等。
    """
    v, u = np.nonzero(depth_mm > 0)
    z = depth_mm[v, u].astype(np.float32)
    x = (u.astype(np.float32) - intrinsics.cx) * z / intrinsics.fx
    y = (v.astype(np.float32) - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=1), np.stack([u, v], axis=1)


def voxel_downsample(points: np.ndarray, leaf_mm: float) -> np.ndarray:
    """体素降采样：按 leaf 分格，每格用落点均值代表。

    分格后先摊平成一维整数编码再排序，不用 `np.unique(axis=0)`——
    后者会把每一行当成 void 逐字节比较，14 万点要几十毫秒，直接吃掉帧预算。
    一维编码下走的是普通整数排序，快一个量级。
    """
    if leaf_mm <= 0 or len(points) == 0:
        return points

    keys = np.floor(points / leaf_mm).astype(np.int64)
    origin = keys.min(axis=0)
    span = keys.max(axis=0) - origin + 1
    # 每轴格子数远小于 2000，三维编码在 int64 里不会溢出
    code = ((keys[:, 0] - origin[0]) * span[1] + (keys[:, 1] - origin[1])) * span[2] \
        + (keys[:, 2] - origin[2])

    _, inverse, counts = np.unique(code, return_inverse=True, return_counts=True)
    n_vox = len(counts)
    sums = np.zeros((n_vox, 3), dtype=np.float64)
    for axis in range(3):
        sums[:, axis] = np.bincount(inverse, weights=points[:, axis], minlength=n_vox)
    return (sums / counts[:, None]).astype(np.float32)


def fit_plane_ransac(points: np.ndarray, thresh_mm: float, iters: int,
                     rng: np.random.Generator) -> tuple:
    """RANSAC 找最大平面，返回 (法向, 偏移 d, inlier 掩码)，平面写作 n·p + d = 0。

    三点采样得到的平面受点噪声影响很大，所以找到内点后用它做一次最小二乘精化
    （SVD 取最小奇异向量），否则法向会在多次调用之间跳动。
    """
    n_pts = len(points)
    if n_pts < MIN_PLANE_POINTS:
        raise ValueError(f"点太少（{n_pts}），无法拟合平面")

    best_count = 0
    best_inliers = None
    best_normal = None
    best_d = 0.0

    for _ in range(max(int(iters), 1)):
        idx = rng.choice(n_pts, MIN_PLANE_POINTS, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = float(-normal @ p0)
        inliers = np.abs(points @ normal + d) < thresh_mm
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers, best_normal, best_d = count, inliers, normal, d

    if best_inliers is None or best_count < MIN_PLANE_POINTS:
        raise ValueError(
            "RANSAC 找不到任何平面：检查 plane_thresh_mm 是不是按米填的（应为毫米）"
        )

    pts = points[best_inliers]
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[2].astype(np.float64)
    d = float(-normal @ centroid)
    inliers = np.abs(points @ normal + d) < thresh_mm
    return normal, d, inliers


def label_clusters(depth_mm: np.ndarray, jump_mm: float) -> np.ndarray:
    """按「深度连续」切连通域，返回标签图（0 表示背景/边界）。

    深度图是组织化点云：相邻像素在 3D 空间本来就挨着，只有跨物体边界时深度
    才跳变。所以在跳变处切断，剩下的连通域就是欧式聚类的等价结果，
    而且只用一遍 cv2.connectedComponents，比 KD 树聚类快两个量级。
    """
    valid = depth_mm > 0
    filled = np.where(valid, depth_mm, 0).astype(np.float32)

    jump_h = np.abs(np.diff(filled, axis=1)) > jump_mm
    jump_v = np.abs(np.diff(filled, axis=0)) > jump_mm

    border = np.zeros_like(valid)
    border[:, :-1] |= jump_h
    border[:, 1:] |= jump_h
    border[:-1, :] |= jump_v
    border[1:, :] |= jump_v
    border |= ~valid

    work = (valid & ~border).astype(np.uint8)
    _, labels = cv2.connectedComponents(work, connectivity=8)
    return labels
