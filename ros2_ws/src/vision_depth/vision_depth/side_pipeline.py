"""RGB-D 图对处理：解码、测量、渲染及保存对应快照。

用大白话说
    结果始终带着原始图对，窗口只在对应彩色图上画框和中心。
    每个框独立选面，过期时图像仍可看但测量叠字全部撤去。
"""
import json
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from vision.depth import Intrinsics
from vision.depth.rgbd import ColorRectifier, validate_images
from vision.depth.surface import PlaneParams, measure_planes, draw_detection_label, draw_review_box

# BGR：选中面黄色，其余候选按当前帧内序号着色；编号不是跨帧跟踪 ID。
_PLANE_COLORS = ((255, 160, 0), (255, 80, 180), (180, 180, 255))
_SELECTED_COLOR = (0, 255, 255)


class SidePipeline:
    """持有单个检测器和校正映射缓存，只在一个工作线程中调用。"""

    def __init__(self, detector, params=None):
        """detector 遵循 detect(image_bgr) 接口，params 为毫米平面参数。"""
        self.detector = detector
        self.params = params or PlaneParams()
        self.params.validate()
        self.rectifier = ColorRectifier()

    def __call__(self, job):
        """解码、校正、检测及逐框测量，返回拥有原图对的独立结果字典。"""
        pair, ci, di = job['pair'], job['ci'], job['di']
        t0 = time.monotonic()
        color, depth = image_array(pair.color.payload), image_array(pair.depth.payload, depth=True)
        for frame, info in ((pair.color, ci), (pair.depth, di)):
            if frame.payload.header.frame_id != info['frame_id']:
                raise ValueError('Image frame differs from CameraInfo')
        validate_images(color, depth, ci, di)
        image, valid = self.rectifier.apply(color, ci, di)
        t1 = time.monotonic()
        detections = self.detector.detect(image)
        t2 = time.monotonic()
        h, w = depth.shape
        k = Intrinsics.from_camera_info(di['k'], w, h)
        observations = []
        for detection in detections:
            box = tuple(detection.bbox_xyxy)
            row = {'detection': {'bbox_xyxy': list(box), 'class_id': int(detection.class_id),
                                 'class_name': str(detection.class_name), 'score': float(detection.score)},
                   'status': 'INVALID_BOX', 'reason': '', 'measurement': None,
                   'centroid_px': None, 'masks': []}
            try:
                result, masks = measure_planes(depth, k, box, valid, self.params)
                status = {'candidate_unverified': 'CANDIDATE', 'ambiguous': 'AMBIGUOUS',
                          'no_plane': 'NO_PLANE'}[result['status']]
                row.update(status=status, reason=result['reason'], measurement=result, masks=masks)
                if status == 'CANDIDATE':
                    x, y, z = result['candidates'][result['selected_index']]['centroid_mm']
                    if not np.isfinite([x, y, z]).all() or z <= 0:
                        raise ValueError('invalid centroid')
                    row['centroid_px'] = [k.fx*x/z+k.cx, k.fy*y/z+k.cy]
            except (ValueError, np.linalg.LinAlgError) as exc:
                row.update(status='INVALID_BOX', reason=str(exc), centroid_px=None)
            observations.append(row)
        t3 = time.monotonic()
        return {'pair': {'color_stamp_ns': pair.color.stamp_ns, 'depth_stamp_ns': pair.depth.stamp_ns,
                         'dt_ms': (pair.depth.stamp_ns-pair.color.stamp_ns)/1e6, 'epoch': pair.epoch},
                'received': pair.received, 'color': color, 'depth': depth, 'image': image,
                'ci': ci, 'di': di, 'params': self.params, 'observations': observations,
                'timing_ms': {'decode_rectify': (t1-t0)*1000, 'inference': (t2-t1)*1000,
                              'geometry': (t3-t2)*1000, 'total': (t3-t0)*1000}}


def _text(canvas, line, y, color=(235, 235, 235)):
    """在信息区写 ASCII，避免中文被 OpenCV 渲染为问号。"""
    line = str(line).encode('ascii', errors='replace').decode('ascii')
    cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def colorize_depth(depth, max_mm=2000):
    """以固定 0–2000 mm 色标显示深度；零值涂黑，超过上限的值饱和。"""
    if depth.ndim != 2 or depth.dtype != np.uint16 or max_mm <= 0:
        raise ValueError('depth must be uint16 mm and max_mm positive')
    scaled = np.clip(depth.astype(np.float32) * (255.0 / max_mm), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    colored[depth == 0] = 0
    return colored


def render_result(result, now, stats, overlay=True, max_age_ms=500, depth_visible=True):
    """返回图像及 stale 标志；过期图上不绘制任何框、平面、中心或坐标。"""
    age = max(0., (now-result['received'])*1000)
    stale = age > max_age_ms
    image = result['image'].copy()
    depth_image = None
    if depth_visible:
        # 每个已处理图对只上色一次；GUI 多次刷新只复制缓存的颜色图。
        if 'depth_color' not in result:
            result['depth_color'] = colorize_depth(result['depth'])
        depth_image = result['depth_color'].copy()
    observations = result['observations']
    if not stale:
        for index, row in enumerate(observations):
            if overlay:
                selected = (row['measurement'] or {}).get('selected_index')
                for plane_index, mask in enumerate(row['masks']):
                    color = _SELECTED_COLOR if plane_index == selected else _PLANE_COLORS[plane_index % len(_PLANE_COLORS)]
                    image[mask > 0] = (image[mask > 0].astype(float)*0.65+
                                       np.array(color)*0.35).astype(np.uint8)
            detection = row['detection']
            draw_review_box(image, detection['bbox_xyxy'])
            draw_detection_label(image, detection, index)
            if depth_image is not None:
                draw_review_box(depth_image, detection['bbox_xyxy'])
                draw_detection_label(depth_image, detection, index)
            if row['centroid_px'] is not None:
                point = tuple(round(v) for v in row['centroid_px'])
                for color, size, thick in [((0, 0, 0), 25, 7), ((255, 255, 255), 23, 5),
                                           ((255, 0, 255), 21, 3)]:
                    cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, size, thick)
                    if depth_image is not None:
                        cv2.drawMarker(depth_image, point, color, cv2.MARKER_CROSS, size, thick)
    if depth_image is not None:
        cv2.rectangle(depth_image, (0, 0), (depth_image.shape[1]-1, 24), (0, 0, 0), -1)
        caption = 'DEPTH | BLACK=INVALID BLUE=NEAR RED=FAR | 0-2000mm SAT'
        if stale:
            caption = 'STALE | ' + caption
        cv2.putText(depth_image, caption, (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
    sizes = [96 + 24 * len((row['measurement'] or {}).get('candidates', []))
             for row in observations]
    height = max(image.shape[0], 270+(0 if stale else sum(sizes)))
    panel = np.full((height, 510, 3), 28, np.uint8)
    _text(panel, 'RGB-D SIDE | VISIBLE INLIER CENTROID', 24)
    _text(panel, 'STALE - measurements hidden' if stale else
          ('NO_DETECTION' if not observations else 'CANDIDATES - NOT VALIDATED'), 48,
          (0, 180, 255) if stale else (150, 230, 150))
    _text(panel, f"Input RGB/D FPS: {stats.get('color_fps', 0):.1f}/{stats.get('depth_fps', 0):.1f}", 74)
    _text(panel, f"Process FPS(avg): {stats.get('process_fps', 0):.1f} | ReceiveAge: {age:.0f} ms", 98)
    t = result['timing_ms']
    _text(panel, f"PairDt: {result['pair']['dt_ms']:+.3f} ms", 122)
    _text(panel, f"Inference: {t['inference']:.1f} | Geometry: {t['geometry']:.1f} ms", 146)
    _text(panel, f"SyncDrop: {stats.get('sync_dropped', 0)} ProcessDrop: {stats.get('processing_dropped', 0)}", 170)
    _text(panel, f"Stale: {stats.get('stale_count', 0)} Errors: {stats.get('errors', 0)}", 194)
    _text(panel, 'Q/Esc: quit | M: mask | D: depth | S: snapshot', 218)
    _text(panel, 'Optical mm: X right, Y down, Z forward', 242)
    if not stale:
        y = 270
        for index, row in enumerate(observations):
            _text(panel, f"{index}: {row['detection']['class_name']} {row['detection']['score']:.2f} | {row['status']}", y)
            m = row['measurement']
            if row['centroid_px'] is not None:
                c = m['candidates'][m['selected_index']]
                x, yy, z = c['centroid_mm']
                _text(panel, f'CENTROID X={x:.1f} Y={yy:.1f} Z={z:.1f}', y+24, (255, 150, 255))
                _text(panel, f"RMS={c['residual_rms_mm']:.2f} mm VALID={m['valid_depth_ratio']:.1%}", y+48)
            elif m:
                _text(panel, f"Planes={len(m['candidates'])} VALID={m['valid_depth_ratio']:.1%}", y+24)
                _text(panel, row['reason'][:60], y+48, (0, 180, 255))
            else:
                _text(panel, row['reason'][:65], y+24)
            if m:
                for plane, candidate in enumerate(m['candidates']):
                    label = 'SELECTED' if plane == m['selected_index'] else 'OTHER'
                    color = _SELECTED_COLOR if plane == m['selected_index'] else _PLANE_COLORS[plane % len(_PLANE_COLORS)]
                    _text(panel, f"P{plane} {label} support={candidate['inlier_ratio']:.1%} core={candidate['core_coverage']:.1%}", y+72+24*plane, color)
            y += sizes[index]
    depth_width = depth_image.shape[1] if depth_image is not None else 0
    canvas = np.zeros((height, image.shape[1]+depth_width+panel.shape[1], 3), np.uint8)
    canvas[:image.shape[0], :image.shape[1]] = image
    if depth_image is not None:
        canvas[:depth_image.shape[0], image.shape[1]:image.shape[1]+depth_width] = depth_image
    canvas[:, image.shape[1]+depth_width:] = panel
    return canvas, stale


def image_array(msg, depth=False):
    """按编码、step 和字节序解码；返回独立 BGR 或 uint16 毫米数组。"""
    enc = str(msg.encoding)
    if depth and enc not in ('16UC1', 'mono16'):
        raise ValueError('depth encoding must be 16UC1/mono16 in mm')
    if not depth and enc not in ('bgr8', 'rgb8'):
        raise ValueError('color encoding must be bgr8/rgb8')
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    bpp = 2 if depth else 3
    if h <= 0 or w <= 0 or step < w*bpp or len(msg.data) < h*step:
        raise ValueError('invalid image dimensions, step, or data length')
    rows = np.frombuffer(bytes(msg.data), np.uint8, count=h*step).reshape(h, step)
    raw = np.ascontiguousarray(rows[:, :w*bpp])
    if depth:
        return raw.view('>u2' if msg.is_bigendian else '<u2').reshape(h, w).astype(np.uint16)
    color = raw.reshape(h, w, 3)
    return color[:, :, ::-1].copy() if enc == 'rgb8' else color.copy()


def info_dict(msg):
    """相机几何快照不包含逐帧变化的时间戳，便于检测内参变化。"""
    roi = getattr(msg, 'roi', None)
    if (getattr(msg, 'binning_x', 0) > 1 or getattr(msg, 'binning_y', 0) > 1 or
            (roi and (roi.x_offset or roi.y_offset or roi.width not in (0, msg.width)
                      or roi.height not in (0, msg.height)))):
        raise ValueError('binned or cropped CameraInfo not supported')
    return {'width': int(msg.width), 'height': int(msg.height),
            'frame_id': str(msg.header.frame_id), 'distortion_model': str(msg.distortion_model),
            'k': [float(v) for v in msg.k], 'd': [float(v) for v in msg.d],
            'r': [float(v) for v in msg.r], 'p': [float(v) for v in msg.p]}


def save_snapshot(root, result, shown, metadata):
    """排他创建快照目录；保存图对、校正图、掩码和实际显示状态，返回新目录。"""
    path = Path(root) / (datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(path/'pair.npz', color=result['color'], depth=result['depth'])
    for name, img in [('rectified.png', result['image']), ('overlay.png', shown)]:
        if not cv2.imwrite(str(path/name), img):
            raise OSError(f'cannot save {name}')
    observations = []
    for index, row in enumerate(result['observations']):
        observations.append({k: v for k, v in row.items() if k != 'masks'})
        for plane, mask in enumerate(row['masks']):
            if not cv2.imwrite(str(path/f'box{index}_plane{plane}_mask.png'), mask):
                raise OSError('cannot save plane mask')
    data = {**metadata, 'pair': result['pair'], 'color_info': result['ci'], 'depth_info': result['di'],
            'timing_ms': result['timing_ms'], 'observations': observations,
            'plane_params': asdict(result['params']), 'depth_unit': 'mm',
            'color_grid': 'rectified_to_depth_K', 'center_definition': 'visible_inlier_centroid',
            'user_annotation_completed': False, 'user_side_review_completed': False,
            'geometry_status': 'UNVERIFIED', 'spatial_status': 'UNVERIFIED'}
    (path/'result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    return path
