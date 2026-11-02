"""RGB-D 像素网格检查与彩色校正，不依赖 ROS、检测器或文件读取。

用大白话说
    对齐深度和原始彩色可能使用不同畸变网格；先把彩色采样到深度网格，
    然后才能按同一像素框取深度。映射缓存只保留一组内参，变化时重建。
"""
import cv2
import numpy as np


def validate_images(color, depth, ci, di):
    """检查同尺寸 BGR/毫米 uint16 图及两路 CameraInfo 的几何约定。"""
    if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
        raise ValueError('color must be uint8 BGR')
    if depth.dtype != np.uint16 or depth.ndim != 2 or color.shape[:2] != depth.shape:
        raise ValueError('depth must be same-size uint16 mm')
    h, w = depth.shape
    for info in (ci, di):
        if (info.get('width'), info.get('height')) != (w, h):
            raise ValueError('image and CameraInfo dimensions differ')
    if not ci.get('frame_id') or ci['frame_id'] != di.get('frame_id'):
        raise ValueError('color and aligned depth must share optical frame')


class ColorRectifier:
    """缓存彩色到深度网格的采样表；valid 掩码排除采样涉及图外的像素。"""

    def __init__(self):
        """空缓存；不预设某台相机内参或图像分辨率。"""
        self._key = None
        self._maps = None
        self._valid = None

    def apply(self, color, ci, di):
        """返回校正 BGR 与只读有效掩码；深度不插值，不修改输入图。"""
        h, w = color.shape[:2]
        kc = np.asarray(ci['k'], dtype=np.float64).reshape(3, 3)
        kd = np.asarray(di['k'], dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(ci['d'], dtype=np.float64)
        dd = np.asarray(di.get('d', [0]*5), dtype=np.float64)
        if (not np.isfinite(kc).all() or not np.isfinite(kd).all() or
                not np.isfinite(distortion).all() or not np.isfinite(dd).all() or
                kc[0, 0] <= 0 or kc[1, 1] <= 0 or kd[0, 0] <= 0 or kd[1, 1] <= 0):
            raise ValueError('invalid camera calibration')
        if ci.get('distortion_model', 'plumb_bob') != 'plumb_bob' or np.any(np.abs(dd) > 1e-12):
            raise ValueError('require plumb_bob color and zero-distortion aligned depth')
        key = (h, w, tuple(kc.flat), tuple(kd.flat), tuple(distortion))
        if key != self._key:
            self._maps = cv2.initUndistortRectifyMap(kc, distortion, None, kd, (w, h), cv2.CV_32FC1)
            self._valid = cv2.remap(np.full((h, w), 255, np.uint8), *self._maps,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) == 255
            self._valid.setflags(write=False)
            self._key = key
        return cv2.remap(color, *self._maps, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT), self._valid


def rectified_color_and_mask(color, ci, di):
    """兼容离线单次调用，返回校正 BGR 与有效掩码。"""
    return ColorRectifier().apply(color, ci, di)
