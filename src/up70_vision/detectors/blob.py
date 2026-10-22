"""占位检测器：饱和度/亮度分割 + 轮廓提取（YOLO 就位前把链路跑通）。

为什么先做它
    YOLO 需要标注数据集和确定的物块本体，都还没定（见 docs/progress.md 第七节）。
    而"哪里有东西"这一步用颜色分割就能先顶上，链路先跑通、接口先冻结，
    将来把 detectors/ 换成 yolo.py 即可，判色与下游一行都不用改。

两条约定（detectors/__init__.py）
    1. 返回的 Detection.class_name 填 "unknown"、class_id 填 -1
    2. bbox 用绝对像素坐标，右/下边界是半开上界
"""

import cv2
import numpy as np

from ..color import valid_pixel_mask
from ..types import Detection, bbox_from_xywh
from .base import Detector


class BlobDetector(Detector):
    name = "blob"

    def __init__(self,
                 min_area_px: int = 400,
                 morph_kernel: int = 5,
                 max_detections: int = 32,
                 min_saturation: int = 40,
                 min_value: int = 40):
        """
        :param min_area_px: 面积下限，低于它的轮廓丢掉（滤掉噪点碎块）
        :param morph_kernel: 形态学核尺寸（奇数）；先开运算去小黑点，再闭运算补小洞
        :param max_detections: 单帧最多保留几个候选框（按面积从大到小）
        :param min_saturation: 前景判定的饱和度下限
        :param min_value: 前景判定的亮度下限
        """
        self.min_area_px = int(min_area_px)
        self.morph_kernel = int(morph_kernel)
        self.max_detections = int(max_detections)
        self.min_saturation = int(min_saturation)
        self.min_value = int(min_value)
        if self.min_area_px < 0:
            raise ValueError(f"min_area_px 不能为负：{self.min_area_px}")
        if self.morph_kernel <= 0 or self.morph_kernel % 2 == 0:
            raise ValueError(f"morph_kernel 必须是正奇数：{self.morph_kernel}")
        if self.max_detections <= 0:
            raise ValueError(f"max_detections 必须为正：{self.max_detections}")

    def detect(self, image_bgr: np.ndarray) -> list:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = valid_pixel_mask(hsv, self.min_saturation, self.min_value)

        kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)     # 去小黑点
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)    # 补小洞

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(c)
            found.append((float(area), bbox_from_xywh(x, y, w, h)))

        found.sort(key=lambda item: item[0], reverse=True)
        return [
            Detection(bbox_xyxy=bbox, class_name="unknown", class_id=-1, score=1.0)
            for _, bbox in found[:self.max_detections]
        ]
