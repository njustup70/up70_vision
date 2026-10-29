"""读取 RGB-D 报告和配对帧，确保图、内参、单位与配对规则一致。

用大白话说
    默认使用既有报告中的一对图；需要连续帧时重新扫描原包并复算配对。
    报告、原包与 CameraInfo 有冲突时拒绝测量，不能拿另一包参数凑数。
"""
from pathlib import Path

import cv2
import numpy as np

from lab.manual_rgbd_fusion import load_report, rectify_color
from lab.rgbd_check.bag import load_images, scan
from lab.rgbd_check.pairing import pair_frames


def rectified_color_and_mask(color, ci, di):
    """将彩色映射到深度网格；插值涉及图外像素时该位置不参与测量。"""
    rectified = rectify_color(color, ci, di)
    h, w = color.shape[:2]
    kc = np.asarray(ci['k'], dtype=np.float64).reshape(3, 3)
    kd = np.asarray(di['k'], dtype=np.float64).reshape(3, 3)
    mx, my = cv2.initUndistortRectifyMap(kc, np.asarray(ci['d'], dtype=np.float64),
                                         None, kd, (w, h), cv2.CV_32FC1)
    valid = cv2.remap(np.full((h, w), 255, np.uint8), mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0) == 255
    return rectified, valid


def load_frames(report, bag=None, count=1):
    """读取已通过完整性和同步检查的报告；按需要从 bag 等间隔读取不同帧。

    :return: (summary, color_info, depth_info, [(metadata, BGR, depth_mm), ...])。
    :raises ValueError: 报告和 bag 的模式、相机信息、时间配对或单位不符合要求。
    """
    summary, selected, color, depth, ci, di = load_report(Path(report))
    capture = summary.get('capture') or {}
    if capture.get('depth_unit') != 'mm':
        raise ValueError('report depth_unit must be mm')
    for role, info in (('color', ci), ('depth', di)):
        profile = capture.get(f'{role}_profile') or {}
        if (profile.get('frame_id') != info['frame_id'] or
                profile.get('width') != info['width'] or profile.get('height') != info['height']):
            raise ValueError(f'{role} image profile and CameraInfo do not match')
    if count < 1:
        raise ValueError('frame count must be positive')
    if bag is None:
        if count != 1:
            raise ValueError('multiple frames require --bag')
        return summary, ci, di, [(selected, color, depth)]
    bag = Path(bag)
    recorded_bag = Path(summary.get('bag', ''))
    if not summary.get('bag') or recorded_bag.resolve() != bag.resolve():
        raise ValueError('bag path does not match report source')
    topics = {role: capture[f'{role}_topic'] for role in
              ('color', 'depth', 'color_info', 'depth_info')}
    scanned = scan(bag, topics)
    if scanned.geometry_changes or scanned.infos.get('color_info') != ci or scanned.infos.get('depth_info') != di:
        raise ValueError('bag CameraInfo differs from report or changes in bag')
    for role, info in (('color', ci), ('depth', di)):
        if any(f.frame_id != info['frame_id'] or (f.width, f.height) != (info['width'], info['height'])
               for f in scanned.frames[role]):
            raise ValueError(f'{role} image and CameraInfo geometry do not match')
    pairing = pair_frames(scanned.frames['color'], scanned.frames['depth'],
                          warmup_ns=round(capture['warmup_s'] * 1e9),
                          stable_duration_s=capture['duration_s'])
    if pairing.status != 'PASS' or pairing.integrity.get('status') != 'PASS':
        raise ValueError('bag integrity or sync does not pass on recalculation')
    pairs = pairing.pairs
    if not pairs:
        raise ValueError('bag has no paired frames')
    indices = np.linspace(0, len(pairs)-1, min(count, len(pairs)), dtype=int)
    chosen = [pairs[int(i)] for i in indices]
    colors = load_images(bag, topics['color'], [p.color.index for p in chosen], 'color')
    depths = load_images(bag, topics['depth'], [p.depth.index for p in chosen], 'depth')
    frames = []
    for pair in chosen:
        meta = {'pair_id': f'c{pair.color.index}_d{pair.depth.index}',
                'color_index': pair.color.index, 'depth_index': pair.depth.index,
                'color_stamp_ns': pair.color.stamp_ns, 'depth_stamp_ns': pair.depth.stamp_ns,
                'dt_ms': pair.dt_ns / 1e6}
        c, d = colors[pair.color.index], depths[pair.depth.index]
        if c.shape != color.shape or c.dtype != np.uint8 or d.shape != depth.shape or d.dtype != np.uint16:
            raise ValueError('bag frame size or encoding changed')
        frames.append((meta, c, d))
    return summary, ci, di, frames
