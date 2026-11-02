"""ROS 图像按编码、行跨度和字节序转换为 BGR 或毫米深度。"""

import numpy as np


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

