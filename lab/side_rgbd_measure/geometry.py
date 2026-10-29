"""从框内深度提取平面候选，输出可复查的点掩码与毫米测量。

用大白话说
    框只是粗略选区；我们不猜哪个面是箱子，而是列出最多三个平面。
    平面拟合和尺寸只使用有效深度，点云重心不是完整物面或箱体中心。
    本模块不依赖检测器、ROS、支撑面和抓取接口。
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from vision.depth import Intrinsics
from vision.depth.primitives import backproject, fit_plane_ransac, voxel_downsample


@dataclass(frozen=True)
class PlaneParams:
    """拟合调试参数：体素与距离为毫米；次数、随机种子及点数均为整数。"""

    leaf_mm: float = 4.0
    threshold_mm: float = 6.0
    iterations: int = 200
    seed: int = 0
    min_points: int = 120
    max_planes: int = 3

    def validate(self):
        """拒绝不合理参数，避免计算结果无法解释。"""
        if (not np.isfinite(self.leaf_mm) or self.leaf_mm <= 0 or
                not np.isfinite(self.threshold_mm) or self.threshold_mm <= 0 or
                self.iterations < 1 or self.min_points < 3 or not 1 <= self.max_planes <= 3):
            raise ValueError('invalid plane parameters')
        if self.seed < 0:
            raise ValueError('seed must be nonnegative')


def visible_extent(points, normal):
    """投影到平面并求最小外接矩形；返回长短边毫米，不声称完整物面尺寸。"""
    unit = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(unit)
    if norm <= 0 or len(points) < 3:
        raise ValueError('invalid plane or insufficient points')
    unit /= norm
    axis = np.eye(3)[int(np.argmin(np.abs(unit)))]
    e1 = np.cross(unit, axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(unit, e1)
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0)
    xy = np.column_stack((centered @ e1, centered @ e2)).astype(np.float32)
    _, (w, h), _ = cv2.minAreaRect(xy)
    return [float(max(w, h)), float(min(w, h))]


def _nondegenerate(points):
    """排除共线或重复点，避免 SVD 给不唯一的法向。"""
    if len(points) < 3:
        return False
    singular = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False)
    return len(singular) >= 2 and singular[1] > 1e-6


def measure_planes(depth, intrinsics, box, valid_color=None, params=None):
    """对半开像素框拟合候选；返回 JSON 结果和同尺寸候选掩码列表。

    :param depth: uint16 二维深度，单位毫米，零为无效。
    :param intrinsics: 与深度网格匹配的 Intrinsics。
    :param box: 校正网格内半开 (x1,y1,x2,y2)。
    :param valid_color: 映射到原彩色图的有效像素掩码，None 表示全有效。
    :return: (测量字典, 每个候选的 uint8 掩码)；不能拟合时保留原因。
    :raises ValueError: 输入不匹配或参数无效。
    """
    params = params or PlaneParams()
    params.validate()
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError('depth must be uint16 mm image')
    h, w = depth.shape
    if (intrinsics.width, intrinsics.height) != (w, h):
        raise ValueError('depth and intrinsics size mismatch')
    x1, y1, x2, y2 = map(int, box)
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError('invalid half-open bounding box')
    if valid_color is None:
        valid_color = np.ones((h, w), dtype=bool)
    if valid_color.shape != (h, w):
        raise ValueError('valid color mask size mismatch')
    roi = np.zeros((h, w), dtype=bool)
    roi[y1:y2, x1:x2] = True
    valid = roi & (depth > 0) & valid_color.astype(bool)
    clean = np.where(valid, depth, 0).astype(np.uint16)
    points, uv = backproject(clean, intrinsics)
    result = {'status': 'no_plane', 'reason': 'insufficient valid points',
              'bbox_xyxy': [x1, y1, x2, y2],
              'valid_depth_ratio': float(np.count_nonzero(valid) / ((x2-x1)*(y2-y1))),
              'valid_points': int(len(points)), 'params': asdict(params),
              'candidates': [], 'selected_index': None}
    masks = []
    remaining = np.arange(len(points))
    rng = np.random.default_rng(params.seed)
    for _ in range(params.max_planes):
        cloud = points[remaining]
        down = voxel_downsample(cloud, params.leaf_mm)
        if len(down) < params.min_points or not _nondegenerate(down):
            break
        try:
            normal, d, inliers = fit_plane_ransac(down, params.threshold_mm,
                                                  params.iterations, rng)
        except (ValueError, np.linalg.LinAlgError):
            break
        if np.count_nonzero(inliers) < params.min_points or not _nondegenerate(down[inliers]):
            break
        local = np.abs(cloud @ normal + d) < params.threshold_mm
        selected = remaining[local]
        pts = points[selected].astype(np.float64)
        if not _nondegenerate(pts):
            break
        centroid = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
        normal = vt[-1]
        if float(normal @ (-centroid)) < 0:
            normal = -normal
        d = float(-normal @ centroid)
        residual = np.abs(pts @ normal + d)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[uv[selected, 1], uv[selected, 0]] = 255
        candidates = result['candidates']
        candidates.append({'index': len(candidates), 'normal': normal.tolist(), 'd_mm': d,
                           'plane_distance_mm': abs(d), 'centroid_mm': centroid.tolist(),
                           'centroid_z_mm': float(centroid[2]),
                           'inlier_count': len(pts), 'downsampled_inliers': int(inliers.sum()),
                           'inlier_ratio': float(len(pts) / len(points)),
                           'residual_rms_mm': float(np.sqrt(np.mean(residual ** 2))),
                           'residual_p95_mm': float(np.percentile(residual, 95)),
                           'visible_extent_mm': visible_extent(pts, normal)})
        masks.append(mask)
        remaining = remaining[~local]
    if len(masks) == 1:
        result.update(status='candidate_unverified', reason='side identity requires human review',
                      selected_index=0)
    elif len(masks) > 1:
        result.update(status='ambiguous', reason='multiple plane candidates require selection')
    elif len(points) >= params.min_points:
        result['reason'] = 'not enough nondegenerate plane inliers'
    return result, masks
