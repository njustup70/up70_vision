"""读取并校验 RGB-D 离线报告及选中图对。"""

import json
from pathlib import Path

import numpy as np


def load_report(path: Path):
    """读取确定性选中帧及配套内参；拒绝不完整或非对齐采集。"""
    summary = json.loads((path / 'summary.json').read_text(encoding='utf8'))
    if (summary.get('capture') or {}).get('mode') != 'aligned':
        raise ValueError('需要 aligned 采集，不能把原始深度当成彩色网格')
    for key in ('capture_integrity', 'sync'):
        if (summary.get(key) or {}).get('status') != 'PASS':
            raise ValueError(f'{key} 未通过，不能使用此配对帧')
    selected = summary.get('selected') or {}
    pair_id = selected.get('pair_id')
    if not pair_id or '/' in pair_id or '\\' in pair_id:
        raise ValueError('报告缺少有效配对编号')
    infos = (summary.get('geometry') or {}).get('camera_info') or {}
    color_info, depth_info = infos.get('color_info'), infos.get('depth_info')
    if not color_info or not depth_info:
        raise ValueError('缺少配套彩色或对齐深度 CameraInfo')
    if color_info.get('frame_id') != depth_info.get('frame_id') or not color_info.get('frame_id'):
        raise ValueError('彩色与对齐深度的光学坐标系不一致')
    if color_info.get('distortion_model') != 'plumb_bob':
        raise ValueError('仅支持 plumb_bob 彩色畸变模型')
    if not depth_info.get('d') or any(abs(float(x)) > 1e-12 for x in depth_info['d']):
        raise ValueError('当前原型只支持零畸变的对齐深度网格')
    with np.load(path / 'selected' / f'{pair_id}.npz', allow_pickle=False) as images:
        color, depth = images['color'].copy(), images['depth'].copy()
    h, w = depth.shape
    if color.shape != (h, w, 3) or color.dtype != np.uint8 or depth.dtype != np.uint16:
        raise ValueError('要求同尺寸 BGR 彩色图和 uint16 毫米深度图')
    for info in (color_info, depth_info):
        if (info.get('width'), info.get('height')) != (w, h):
            raise ValueError('图像与 CameraInfo 尺寸不一致')
    return summary, selected, color, depth, color_info, depth_info

