"""从框内深度提取平面候选，输出毫米测量与对应诊断绘制。

用大白话说
    框只是粗略选区；我们不猜哪个面是箱子，而是列出最多三个平面。
    平面拟合和尺寸只使用有效深度，点云重心不是完整物面或箱体中心。
    绘制函数只修改图像，不改变候选或测量；本模块不依赖检测器和 ROS。
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .types import Intrinsics
from .primitives import backproject, fit_plane_ransac, voxel_downsample


@dataclass(frozen=True)
class PlaneParams:
    """拟合与选面参数：距离为毫米，比例为 0–1；默认 strict 保留原有口径。"""

    leaf_mm: float = 4.0
    threshold_mm: float = 6.0
    iterations: int = 200
    seed: int = 0
    min_points: int = 120
    max_planes: int = 3
    selection_policy: str = 'strict'
    core_ratio: float = 0.6
    min_valid_ratio: float = 0.5
    min_support_ratio: float = 0.6
    min_core_coverage: float = 0.6
    min_support_margin: float = 0.2

    def validate(self):
        """拒绝不合理参数，避免计算结果无法解释。"""
        if (not np.isfinite(self.leaf_mm) or self.leaf_mm <= 0 or
                not np.isfinite(self.threshold_mm) or self.threshold_mm <= 0 or
                self.iterations < 1 or self.min_points < 3 or not 1 <= self.max_planes <= 3):
            raise ValueError('invalid plane parameters')
        if self.seed < 0:
            raise ValueError('seed must be nonnegative')
        if self.selection_policy not in ('strict', 'dominant'):
            raise ValueError('selection_policy must be strict or dominant')
        for name in ('core_ratio', 'min_valid_ratio', 'min_support_ratio',
                     'min_core_coverage', 'min_support_margin'):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1 or (name == 'core_ratio' and value == 0):
                raise ValueError(f'{name} must be in [0, 1], core_ratio must be positive')


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
    # 中央矩形的分母包含无效像素；否则只剩少量深度时覆盖率会虚高。
    inset = (1.0 - params.core_ratio) / 2.0
    cx1 = x1 + int(np.floor((x2-x1) * inset))
    cy1 = y1 + int(np.floor((y2-y1) * inset))
    cx2 = x2 - int(np.floor((x2-x1) * inset))
    cy2 = y2 - int(np.floor((y2-y1) * inset))
    core_area = (cx2-cx1) * (cy2-cy1)
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
                           'core_coverage': float(np.count_nonzero(mask[cy1:cy2, cx1:cx2]) / core_area),
                           'residual_rms_mm': float(np.sqrt(np.mean(residual ** 2))),
                           'residual_p95_mm': float(np.percentile(residual, 95)),
                           'visible_extent_mm': visible_extent(pts, normal)})
        masks.append(mask)
        remaining = remaining[~local]
    if params.selection_policy == 'dominant' and masks:
        ranked = sorted(result['candidates'], key=lambda c: (-c['inlier_ratio'], c['index']))
        best = ranked[0]
        runner = ranked[1]['inlier_ratio'] if len(ranked) > 1 else 0.0
        checks = ((result['valid_depth_ratio'] >= params.min_valid_ratio, 'low_valid_ratio'),
                  (best['inlier_ratio'] >= params.min_support_ratio, 'low_support_ratio'),
                  (best['core_coverage'] >= params.min_core_coverage, 'low_core_coverage'),
                  # 比例由整数计数相除；微小浮点误差不能把恰好 20% 的边界判成不通过。
                  (best['inlier_ratio'] - runner + 1e-12 >= params.min_support_margin, 'low_support_margin'),
                  (best['residual_rms_mm'] <= params.threshold_mm, 'high_residual_rms'))
        reason = next((reason for okay, reason in checks if not okay), None)
        if reason:
            result.update(status='ambiguous', reason=reason)
        else:
            result.update(status='candidate_unverified', selected_index=best['index'],
                          reason='dominant plane selected; side identity requires human review')
    elif len(masks) == 1:
        result.update(status='candidate_unverified', reason='side identity requires human review',
                      selected_index=0)
    elif len(masks) > 1:
        result.update(status='ambiguous', reason='multiple plane candidates require selection')
    elif len(points) >= params.min_points:
        result['reason'] = 'not enough nondegenerate plane inliers'
    return result, masks


def draw_review_box(image, box):
    """黑色描边加粗绿框，确保缩小显示时仍能与黄色平面区分。"""
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2-1, y2-1), (0, 0, 0), 8)
    cv2.rectangle(image, (x1, y1), (x2-1, y2-1), (0, 255, 0), 4)


def draw_detection_label(image, detection, index, plane_index=None):
    """将框号、模型类名和置信度写在黑底上，供检测图与平面图对应复查。

    用大白话说
        框号对应 choices 中的 detection；类名照模型输出，不替用户判断选面正确。
        人工框显示 manual，不伪造置信度。靠近图边时将标签移回图内。

    :param image: 原地绘制的 BGR 图；仅改变可视化像素。
    :param detection: 含 bbox_xyxy、class_name、score 的检测记录。
    :param index: 从 0 开始的检测框号。
    :param plane_index: 平面候选号；None 表示仅标检测框。
    :return: None；不修改检测记录或测量数据。
    """
    name = str(detection['class_name']).encode('ascii', errors='replace').decode('ascii')
    label = f"{index}: {name}"
    if detection['score'] is not None:
        label += f" {detection['score']:.2f}"
    if plane_index is not None:
        label += f" | plane={plane_index}"
    font, scale, thickness, padding = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 4
    h, w = image.shape[:2]
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    if tw + 2 * padding > w:
        scale *= max(1, w - 2 * padding) / max(1, tw)
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    label_height = th + baseline + 2 * padding
    x1, y1, _, _ = detection['bbox_xyxy']
    left = max(0, min(x1, w - tw - 2 * padding))
    top = max(0, min(y1 - label_height, h - label_height))
    cv2.rectangle(image, (left, top), (left + tw + 2 * padding - 1,
                                     top + label_height - 1), (0, 0, 0), -1)
    cv2.putText(image, label, (left + padding, top + padding + th),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def plane_overlay(image, mask, box):
    """低透明度黄色表示拟合点；框最后绘制，不被覆盖色淹没。"""
    overlay = image.copy()
    overlay[mask > 0] = (overlay[mask > 0].astype(np.float32) * 0.65 +
                         np.array([0, 255, 255]) * 0.35).astype(np.uint8)
    draw_review_box(overlay, box)
    return overlay
