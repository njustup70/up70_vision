"""YOLO 检测器：用 ONNX + `cv2.dnn` 跑推理，返回 `list[Detection]`。

为什么走 ONNX 而不是直接装 ultralytics
    实测（i7-14650HX，1280x1024 一帧已在内存）：
        · ultralytics + torch（CPU）             17.0 ms/帧
        · ONNX + cv2.dnn（容器里 cv2 4.14）      26.4 ms/帧
    相机最高 25 fps（40 ms 预算），两者都够跑；而 ONNX 只多一个 12.9 MB 的权重文件，
    不往镜像里塞 1.4 GB 的 torch。真要训练/微调时再单独装 torch，那是另一条路。

    所以本模块 **只依赖 cv2 与 numpy**，根级模块的分层约定（不 import io/viz/config、
    不读文件、不开窗口）依然成立；权重路径由配置层传进来，这里只在构造/首次推理时检查。

和契约的关系（docs/interface.md 3.1）
    接口没变：`detect(image_bgr) -> list[Detection]`，无目标返回 `[]`。
    唯一的扩展是 `class_name`：判色族（`blob`）填 `"unknown"`，这里填**模型自己的类名**
    （如 `"person"`、`"cup"`），`class_id` 填模型里的类别索引。

输出的坐标系
    ONNX 模型吃 640x640 的方形输入，而相机图是 1280x1024，所以要先 letterbox
    （等比缩放 + 补灰边），推理完再按同一组 scale/pad 还原回原图像素坐标。
    letterbox 与还原是本模块最容易写错的一步：**scale 用错、pad 减错，框就会整体
    偏几十像素，而且不报任何错**。`_unletterbox()` 与 `letterbox()` 互逆，测试里专门验它。
"""

from __future__ import annotations

import ast
import os
from typing import Iterable, Sequence

import cv2
import numpy as np

from ..types import Detection
from .base import Detector

#: letterbox 的填充色（YOLO 官方预处理用 114 的灰，模型见过这个值，别改成 0）
PAD_VALUE = 114

#: 默认权重路径（相对仓库根；由配置层的 `detector.yolo.weights` 覆盖）
DEFAULT_WEIGHTS = "assets/models/yolov8n.onnx"


def letterbox(image_bgr: np.ndarray, size: int,
              pad_value: int = PAD_VALUE) -> tuple[np.ndarray, float, float, float]:
    """等比缩放并把图放到 `size x size` 的正方形画布中央。

    :param image_bgr: HxWx3 的 BGR 图
    :param size: 目标边长（YOLO 的 `imgsz`）
    :param pad_value: 补边颜色
    :return: `(画布, scale, pad_x, pad_y)`；`scale` 是缩放比例，`pad_x/pad_y` 是左上角补边像素
    """
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"图像尺寸非法：{w}x{h}")
    scale = min(size / w, size / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
    if canvas.shape[0] != size or canvas.shape[1] != size:      # 取整误差兜一手
        canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)
    return canvas, scale, left, top


def unletterbox(box_xyxy: Sequence[float], scale: float, pad_x: float, pad_y: float,
                image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """把 letterbox 画布上的框还原成原图像素坐标（右/下为半开上界，且裁到图内）。

    :param box_xyxy: `(x1, y1, x2, y2)`，letterbox 画布坐标
    :param scale: `letterbox()` 返回的缩放比例
    :param pad_x: `letterbox()` 返回的左补边
    :param pad_y: `letterbox()` 返回的上补边
    :param image_size: 原图 `(宽, 高)`
    :return: 原图坐标的 `(x1, y1, x2, y2)`，整数
    """
    if scale <= 0:
        raise ValueError(f"scale 必须为正，收到 {scale}")
    width, height = image_size
    x1 = (box_xyxy[0] - pad_x) / scale
    y1 = (box_xyxy[1] - pad_y) / scale
    x2 = (box_xyxy[2] - pad_x) / scale
    y2 = (box_xyxy[3] - pad_y) / scale
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = int(round(min(max(x1, 0.0), width)))
    y1 = int(round(min(max(y1, 0.0), height)))
    x2 = int(round(min(max(x2, 0.0), width)))
    y2 = int(round(min(max(y2, 0.0), height)))
    return x1, y1, x2, y2


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """两个 `(x1, y1, x2, y2)` 框的交并比。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def nms(boxes: Sequence[Sequence[float]], scores: Sequence[float],
        iou_threshold: float = 0.45) -> list[int]:
    """按分数从高到低做贪心 NMS，返回保留下来的下标（保持分数降序）。

    :param boxes: 若干 `(x1, y1, x2, y2)`
    :param scores: 与 boxes 等长的置信度
    :param iou_threshold: 超过它的框视为重复
    :return: 保留框的下标列表
    """
    if len(boxes) != len(scores):
        raise ValueError(f"boxes({len(boxes)}) 与 scores({len(scores)}) 长度不一致")
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    kept: list[int] = []
    for idx in order:
        if all(_iou(boxes[idx], boxes[other]) <= iou_threshold for other in kept):
            kept.append(idx)
    return kept


def decode_predictions(raw: np.ndarray, conf_threshold: float, iou_threshold: float,
                       scale: float, pad_x: float, pad_y: float,
                       image_size: tuple[int, int],
                       names: dict[int, str] | None = None,
                       classes: Iterable[int] | None = None,
                       max_detections: int = 32) -> list[Detection]:
    """把 ONNX 的原始输出解码成 `Detection` 列表（纯函数，便于单测）。

    :param raw: 模型输出，接受 `(1, 4+nc, N)` 或 `(1, N, 4+nc)` 两种排布
    :param conf_threshold: 置信度下限
    :param iou_threshold: NMS 的 IoU 阈值
    :param scale: letterbox 的缩放比例
    :param pad_x: letterbox 的左补边
    :param pad_y: letterbox 的上补边
    :param image_size: 原图 `(宽, 高)`
    :param names: 类别索引 -> 类名；缺省时用 `str(索引)`
    :param classes: 只保留这些类别的索引；None/空表示全部
    :param max_detections: 单帧最多返回几个（按分数降序截断）
    :return: `Detection` 列表；无目标返回 `[]`
    """
    pred = np.asarray(raw, dtype=np.float32)
    if pred.ndim != 3 or pred.shape[0] != 1:
        raise ValueError(f"模型输出应为 (1, C, N) 或 (1, N, C)，收到 {pred.shape}")
    pred = pred[0]
    # 两种排布：通道数通常远小于候选框数（84 vs 8400），据此判断要不要转置
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                      # -> (N, 4+nc)
    if pred.shape[1] < 5:
        raise ValueError(f"模型输出至少要有 4 个框参数 + 1 个类别，收到 {pred.shape}")
    boxes_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    keep_mask = scores >= float(conf_threshold)
    allowed = None if classes is None else {int(c) for c in classes}
    if allowed:
        keep_mask &= np.isin(class_ids, list(allowed))
    if not np.any(keep_mask):
        return []

    boxes_cxcywh = boxes_cxcywh[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]

    cx, cy, bw, bh = boxes_cxcywh.T
    boxes_xyxy = np.stack([cx - bw / 2.0, cy - bh / 2.0,
                           cx + bw / 2.0, cy + bh / 2.0], axis=1)
    kept = nms(boxes_xyxy.tolist(), scores.tolist(), iou_threshold)

    results: list[Detection] = []
    for idx in kept[:max(0, int(max_detections))]:
        x1, y1, x2, y2 = unletterbox(boxes_xyxy[idx], scale, pad_x, pad_y, image_size)
        if x2 <= x1 or y2 <= y1:           # 完全裁掉的退化框没有意义
            continue
        cls = int(class_ids[idx])
        results.append(Detection(bbox_xyxy=(x1, y1, x2, y2),
                                 class_name=(names or {}).get(cls, str(cls)),
                                 class_id=cls,
                                 score=float(scores[idx])))
    return results


class YoloDetector(Detector):
    """ONNX 权重 + `cv2.dnn` 的 YOLO 检测器（默认 CPU）。"""

    name = "yolo"

    def __init__(self,
                 weights: str = DEFAULT_WEIGHTS,
                 conf: float = 0.25,
                 iou: float = 0.45,
                 imgsz: int = 640,
                 max_detections: int = 32,
                 classes: Sequence[int] | None = None,
                 class_names: Sequence[str] | None = None,
                 device: str = "cpu"):
        """
        :param weights: ONNX 权重路径（相对仓库根或绝对路径）
        :param conf: 置信度下限 (0, 1)
        :param iou: NMS 的 IoU 阈值 (0, 1]
        :param imgsz: 模型输入边长（必须是 32 的倍数）
        :param max_detections: 单帧最多返回几个
        :param classes: 只保留这些类别索引；None/空表示全部
        :param class_names: 类名表（按类别索引排列）；给了就用它，
                            否则尝试从 ONNX 元数据读，再没有就退化成索引字符串。
                            容器里没装 onnx 包时，这条是拿到可读类名的唯一办法
        :param device: 目前只支持 "cpu"（cv2.dnn 在这台机器上没有 CUDA 后端）
        """
        super().__init__()
        if not weights:
            raise ValueError("weights 不能为空")
        if not 0.0 < float(conf) < 1.0:
            raise ValueError(f"conf 必须在 (0, 1) 之间，收到 {conf}")
        if not 0.0 < float(iou) <= 1.0:
            raise ValueError(f"iou 必须在 (0, 1] 之间，收到 {iou}")
        if int(imgsz) <= 0 or int(imgsz) % 32 != 0:
            raise ValueError(f"imgsz 必须是 32 的正整数倍，收到 {imgsz}")
        if int(max_detections) <= 0:
            raise ValueError(f"max_detections 必须为正，收到 {max_detections}")
        if str(device).lower() != "cpu":
            raise ValueError(f"目前只支持 device='cpu'（cv2.dnn 无 CUDA 后端），收到 {device!r}")

        self.weights = os.fspath(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.max_detections = int(max_detections)
        self.classes = [int(c) for c in (classes or [])]
        self.class_names = [str(n) for n in (class_names or [])]
        self._net = None                   # 惰性加载：构造时不碰文件，detect 时才读
        self._names: dict[int, str] = {}

    # ---------------------------------------------------------------- 加载
    def _ensure_loaded(self) -> None:
        """首次推理前把 ONNX 载入（测试里可以先塞一个假网络绕开这一步）。"""
        if self._net is not None:
            return
        if not os.path.isfile(self.weights):
            raise FileNotFoundError(
                f"YOLO 权重不存在：{self.weights}；"
                f"先导出一次（需要 ultralytics）：python lab/export_yolo.py --weights yolov8n.pt")
        self._net = cv2.dnn.readNetFromONNX(self.weights)
        self._names = _read_class_names(self.weights)

    # ---------------------------------------------------------------- 推理
    def detect(self, image_bgr: np.ndarray) -> list:
        """在 BGR 图上出候选框。

        :param image_bgr: HxWx3 uint8 BGR 图
        :return: `list[Detection]`；无目标返回 `[]`
        """
        if image_bgr is None or getattr(image_bgr, "ndim", 0) != 3 or image_bgr.shape[2] != 3:
            shape = None if image_bgr is None else getattr(image_bgr, "shape", None)
            raise ValueError(f"image_bgr 必须是 HxWx3 的 BGR 图，收到 {shape}")
        self._ensure_loaded()

        canvas, scale, pad_x, pad_y = letterbox(image_bgr, self.imgsz)
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (self.imgsz, self.imgsz),
                                     swapRB=True, crop=False)
        self._net.setInput(blob)
        raw = self._net.forward()
        height, width = image_bgr.shape[:2]
        names = ({i: n for i, n in enumerate(self.class_names)}
                 if self.class_names else self._names)
        return decode_predictions(raw, self.conf, self.iou, scale, pad_x, pad_y,
                                  (width, height), names=names,
                                  classes=self.classes, max_detections=self.max_detections)


def _read_class_names(onnx_path: str) -> dict[int, str]:
    """读类名表。两级兜底：先看同名 `.names.json`（导出时写，零依赖），再试 ONNX 元数据。

    为什么优先 sidecar
        容器里没有 `onnx` 这个包（也不想为了读一行元数据装 30 MB），所以导出脚本会在
        权重旁边写一个 `yolov8n.names.json`。读它只需要标准库 json。
        两者都没有时返回空表，类名退化成类别索引的字符串（如 "0"），不影响检测本身。

    :param onnx_path: ONNX 权重路径
    :return: `{类别索引: 类名}`；读不到返回 `{}`
    """
    sidecar = os.path.splitext(onnx_path)[0] + ".names.json"
    try:
        import json
        with open(sidecar, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return {int(k): str(v) for k, v in data.items()}
        if isinstance(data, list):
            return {i: str(v) for i, v in enumerate(data)}
    except (OSError, ValueError):
        pass
    try:
        import onnx                          # 可选依赖：装了才读元数据
        model = onnx.load(onnx_path, load_external_data=False)
        for prop in model.metadata_props:
            if prop.key == "names":
                parsed = ast.literal_eval(prop.value)
                return {int(k): str(v) for k, v in parsed.items()}
    except Exception:                        # noqa: BLE001 —— 元数据读不到不影响推理
        return {}
    return {}
