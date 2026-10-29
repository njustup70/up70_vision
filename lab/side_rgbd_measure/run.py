"""逐对运行人工框或检测器，保存每个候选和失败原因。

用大白话说
    每个检测框都可看到，不因置信度高就猜它是箱子。一个框才能自动拟合，
    多个框或多个平面必须人工选择；候选始终是诊断结果，不发布机械臂。
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from vision.depth import Intrinsics
from .geometry import PlaneParams, measure_planes
from .input import rectified_color_and_mask


def _save_image(path, image):
    """保存图像，避免 OpenCV 静默写入失败。"""
    if not cv2.imwrite(str(path), image):
        raise OSError(f'cannot write image: {path}')


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


def _summarize(rows):
    """仅统计有候选的帧；总数与失败状态同时保留，法向波动相对平均法向。"""
    selected = [r['measurement']['candidates'][r['measurement']['selected_index']]
                for r in rows if r.get('measurement') and
                r['measurement']['selected_index'] is not None]
    stats = {'frame_count': len(rows), 'status_counts': dict(Counter(r['status'] for r in rows)),
             'selected_count': len(selected), 'position_std_mm': None,
             'max_axis_3sigma_mm': None, 'normal_angle_std_deg': None,
             'normal_angle_rms_deg': None, 'normal_angle_p95_deg': None}
    if len(selected) >= 2:
        positions = np.asarray([s['centroid_mm'] for s in selected])
        std = positions.std(axis=0)
        normals = np.asarray([s['normal'] for s in selected])
        mean = normals.mean(axis=0)
        norm = np.linalg.norm(mean)
        angles = np.degrees(np.arccos(np.clip(normals @ (mean / norm), -1, 1))) if norm > 1e-12 else None
        stats.update(position_std_mm=std.tolist(), max_axis_3sigma_mm=float(3 * std.max()),
                     normal_angle_std_deg=None if angles is None else float(angles.std()),
                     normal_angle_rms_deg=None if angles is None else float(np.sqrt(np.mean(angles**2))),
                     normal_angle_p95_deg=None if angles is None else float(np.percentile(angles, 95)))
    return stats


def _detector_boxes(detector, color):
    """检测器返回项目约定的 Detection；把所有框和置信度留在报告中。"""
    return [{'bbox_xyxy': list(d.bbox_xyxy), 'class_id': int(d.class_id),
             'class_name': d.class_name, 'score': float(d.score)} for d in detector.detect(color)]


def run_frames(summary, ci, di, frames, out, *, box=None, detector=None, params=None,
               choices=None, weights=None, requested_count=None):
    """分析所有选中图对并创建新报告目录；choices 为 pair_id 到候选序号的人工选择。

    :param detector: 已配置检测器，None 时使用 box；传入的图为校正后 BGR。
    :param choices: 每帧单独选择检测框与平面，禁止静默应用于其他帧。
    :return: 汇总结果，文件保留每帧状态与候选；目录已存在时抛异常。
    """
    if (box is None) == (detector is None):
        raise ValueError('provide exactly one manual box or detector')
    params = params or PlaneParams()
    params.validate()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'frames').mkdir()
    h, w = frames[0][2].shape
    k = Intrinsics.from_camera_info(di['k'], w, h)
    rows = []
    choices = choices or {}
    for meta, color, depth in frames:
        pid = meta['pair_id']
        image, valid_color = rectified_color_and_mask(color, ci, di)
        detections = _detector_boxes(detector, image) if detector else [
            {'bbox_xyxy': list(box), 'class_id': None, 'class_name': 'manual', 'score': None}]
        row = {'pair': meta, 'detections': detections, 'measurement': None,
               'status': 'no_detection', 'reason': 'no detection'}
        choice = choices.get(pid, {})
        if not isinstance(choice, dict):
            raise ValueError(f'invalid selection for {pid}')
        detection_index = choice.get('detection')
        if detection_index is not None and (not isinstance(detection_index, int) or
                isinstance(detection_index, bool) or not 0 <= detection_index < len(detections)):
            raise ValueError(f'invalid detection selection for {pid}')
        if len(detections) > 1 and detection_index is None:
            row.update(status='ambiguous_detection', reason='multiple boxes require manual selection')
        elif detections:
            detection_index = 0 if detection_index is None else detection_index
            row['selected_detection'] = detection_index
            bbox = detections[detection_index]['bbox_xyxy']
            result, masks = measure_planes(depth, k, bbox, valid_color, params)
            if 'plane' in choice:
                index = choice['plane']
                if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(masks):
                    raise ValueError(f'invalid candidate selection for {pid}')
                result.update(status='candidate_unverified', selected_index=index,
                              reason='user selected candidate; side and accuracy require review')
            row.update(status=result['status'], reason=result['reason'], measurement=result)
            for i, mask in enumerate(masks):
                overlay = plane_overlay(image, mask, bbox)
                draw_detection_label(overlay, detections[detection_index], detection_index,
                                     plane_index=i)
                _save_image(out / 'frames' / f'{pid}_plane{i}.png', overlay)
                _save_image(out / 'frames' / f'{pid}_plane{i}_mask.png', mask)
        display = image.copy()
        for index, det in enumerate(detections):
            draw_review_box(display, det['bbox_xyxy'])
            draw_detection_label(display, det, index)
        _save_image(out / 'frames' / f'{pid}_boxes.png', display)
        rows.append(row)
    weight_info = None
    if weights:
        path = Path(weights)
        weight_info = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'imgsz': detector.imgsz, 'conf': detector.conf, 'iou': detector.iou,
                       'classes': detector.classes, 'class_names': detector.class_names,
                       'resolved_class_names': detector._names}
    stats = _summarize(rows)
    output = {'session': summary.get('session'), 'source_bag': summary.get('bag'),
              'frame_id': di.get('frame_id'), 'depth_unit': 'mm',
              'color_grid': 'rectified_to_depth_K', 'geometry_status': (summary.get('geometry') or {}).get('status'),
              'spatial_status': (summary.get('spatial') or {}).get('status'),
              'requested_count': requested_count or len(rows), 'actual_count': len(rows),
              'mode': 'yolo' if detector else 'manual', 'model': weight_info,
              'params': vars(params), 'stats': stats, 'frames': rows,
              'warning': 'candidate_unverified; not a grasp target or absolute accuracy acceptance'}
    (out / 'results.json').write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    with (out / 'frames.csv').open('w', newline='', encoding='utf8') as f:
        columns = ['pair_id', 'color_index', 'depth_index', 'color_stamp_ns', 'depth_stamp_ns',
                   'dt_ms', 'status', 'reason', 'detections', 'measurement']
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row['pair'].get(key) for key in columns},
                             'status': row['status'], 'reason': row['reason'],
                             'detections': json.dumps(row['detections'], ensure_ascii=False),
                             'measurement': json.dumps(row['measurement'], ensure_ascii=False)})
    (out / 'report.md').write_text(
        f"# 单箱侧面离线测量\n\n来源：{output['session']}；模式：{output['mode']}；"
        f"请求/实际帧数：{output['requested_count']}/{len(rows)}。\n\n"
        f"状态计数：`{stats['status_counts']}`；选中候选：{stats['selected_count']}。\n\n"
        f"位置各轴标准差（mm）：`{stats['position_std_mm']}`；最大轴 3σ（mm）："
        f"`{stats['max_axis_3sigma_mm']}`；法向相对平均方向 RMS/P95（度）："
        f"`{stats['normal_angle_rms_deg']}` / `{stats['normal_angle_p95_deg']}`。\n\n"
        f"几何：{output['geometry_status']}；空间：{output['spatial_status']}。"
        "候选须人工复核；波动不代表绝对精度。完整数据与选面图见 results.json 和 frames/。\n",
        encoding='utf8')
    return output
