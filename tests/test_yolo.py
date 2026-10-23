"""YOLO 检测器（ONNX + cv2.dnn）的测试：前后处理 + 契约形状。

验的是什么
    这一层最容易出错的不是"模型准不准"，而是**坐标变换**：letterbox 的 scale/pad
    用错，框会整体偏几十像素，而且不报任何错。所以测试的重点是：
        · `letterbox()` / `unletterbox()` 是否互逆（1280x1024 → 640x640 → 还原）
        · 置信度过滤、NMS 去重、max_detections 限流
        · 无目标时返回 `[]`（不是 None），权重缺失时报错信息带路径
        · 工厂能按名字造出来、配置层的参数校验能拦住越界值
    真模型（12.9 MB 的 ONNX）不进仓库，所以用**假网络**注入：把 `detector._net`
    替换成一个返回固定数组的对象，就能不依赖权重验证整条推理链。

环境要求
    只需要 `cv2` 与 `numpy`（容器里就有），**不需要 torch / ultralytics / 权重文件**。
    跑法：python tests/test_yolo.py
"""

import sys
from pathlib import Path

import numpy as np

try:
    from vision.config.schema import ConfigError, from_dict
    from vision.cli.run import build_detector
    from vision.yolo.detector import (YoloDetector, decode_predictions,
                                                 letterbox, nms, unletterbox)
    from vision.types import Detection
    from vision.yolo.draw import detection_label
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vision.config.schema import ConfigError, from_dict
    from vision.cli.run import build_detector
    from vision.yolo.detector import (YoloDetector, decode_predictions,
                                                 letterbox, nms, unletterbox)
    from vision.types import Detection
    from vision.yolo.draw import detection_label


# ---------------------------------------------------------------- 假网络
class _FakeNet:
    """最小可用的假网络：setInput 记住输入，forward 返回预先摆好的数组。"""

    def __init__(self, raw: np.ndarray):
        self._raw = raw
        self.last_input = None
        self.forward_calls = 0

    def setInput(self, blob):                     # noqa: N802 —— 对齐 cv2.dnn 的命名
        self.last_input = blob

    def forward(self):
        self.forward_calls += 1
        return self._raw


def _raw_with_boxes(candidates: int = 100, channels: int = 84) -> np.ndarray:
    """造一个 (1, channels, candidates) 的原始输出（全 0，交给调用方填）。"""
    return np.zeros((1, channels, candidates), dtype=np.float32)


def _put_box(raw: np.ndarray, index: int, cx: float, cy: float, w: float, h: float,
             class_id: int, score: float) -> None:
    """在第 index 个候选里写一条检测（前 4 个是 cx,cy,w,h，第 4+class_id 是分数）。"""
    raw[0, 0, index] = cx
    raw[0, 1, index] = cy
    raw[0, 2, index] = w
    raw[0, 3, index] = h
    raw[0, 4 + class_id, index] = score


# ---------------------------------------------------------------- letterbox
def test_letterbox_keeps_aspect_and_centers():
    image = np.zeros((1024, 1280, 3), dtype=np.uint8)      # 相机图
    canvas, scale, pad_x, pad_y = letterbox(image, 640)
    assert canvas.shape == (640, 640, 3), canvas.shape
    assert abs(scale - 0.5) < 1e-6, scale                # 640/1280
    assert pad_x == 0, pad_x
    assert abs(pad_y - 64) <= 1, pad_y                   # (640 - 512) / 2


def test_unletterbox_is_inverse_of_letterbox():
    image = np.zeros((1024, 1280, 3), dtype=np.uint8)
    _, scale, pad_x, pad_y = letterbox(image, 640)
    # 原图 (100,200)-(300,400) 在画布上应是 (50,164)-(150,264)
    box = (100, 200, 300, 400)
    canvas_box = (box[0] * scale + pad_x, box[1] * scale + pad_y,
                  box[2] * scale + pad_x, box[3] * scale + pad_y)
    back = unletterbox(canvas_box, scale, pad_x, pad_y, (1280, 1024))
    assert all(abs(a - b) <= 1 for a, b in zip(back, box)), (back, box)


def test_unletterbox_clips_to_image():
    got = unletterbox((-50, -50, 5000, 5000), 1.0, 0, 0, (640, 480))
    assert got == (0, 0, 640, 480), got


# ---------------------------------------------------------------- NMS / 解码
def test_nms_removes_overlapping_duplicates():
    boxes = [(10, 10, 110, 110), (12, 12, 112, 112), (300, 300, 400, 400)]
    kept = nms(boxes, [0.9, 0.8, 0.7], iou_threshold=0.45)
    assert kept == [0, 2], kept


def test_decode_filters_low_confidence():
    raw = _raw_with_boxes()
    _put_box(raw, 0, 100, 100, 50, 50, 0, 0.90)
    _put_box(raw, 1, 200, 200, 50, 50, 0, 0.10)
    dets = decode_predictions(raw, 0.25, 0.45, 1.0, 0, 0, (640, 640))
    assert len(dets) == 1, len(dets)
    assert abs(dets[0].score - 0.90) < 1e-5


def test_decode_keeps_highest_class_per_box():
    raw = _raw_with_boxes()
    _put_box(raw, 0, 100, 100, 40, 40, 0, 0.30)
    raw[0, 4 + 5, 0] = 0.80                      # 同一个框的另一个类别分数更高
    dets = decode_predictions(raw, 0.25, 0.45, 1.0, 0, 0, (640, 640))
    assert len(dets) == 1
    assert dets[0].class_id == 5, dets[0].class_id


def test_decode_returns_empty_list_when_all_below_threshold():
    raw = _raw_with_boxes()
    _put_box(raw, 0, 100, 100, 40, 40, 0, 0.05)
    dets = decode_predictions(raw, 0.25, 0.45, 1.0, 0, 0, (640, 640))
    assert dets == [], dets


def test_decode_respects_max_detections():
    raw = _raw_with_boxes()
    for i in range(10):
        _put_box(raw, i, 50 + i * 60, 300, 40, 40, 0, 0.5 + i * 0.01)
    dets = decode_predictions(raw, 0.25, 0.45, 1.0, 0, 0, (640, 640), max_detections=3)
    assert len(dets) == 3, len(dets)


def test_decode_filters_by_classes():
    raw = _raw_with_boxes()
    _put_box(raw, 0, 100, 100, 40, 40, 2, 0.9)
    _put_box(raw, 1, 300, 300, 40, 40, 7, 0.9)
    dets = decode_predictions(raw, 0.25, 0.45, 1.0, 0, 0, (640, 640), classes=[7])
    assert len(dets) == 1 and dets[0].class_id == 7, dets


def test_decode_uses_class_names():
    raw = _raw_with_boxes()
    _put_box(raw, 0, 100, 100, 40, 40, 3, 0.9)
    dets = decode_predictions(raw, 0.25, 0.45, 1.0, 0, 0, (640, 640), names={3: "cup"})
    assert dets[0].class_name == "cup", dets[0].class_name


# ---------------------------------------------------------------- 检测器
def test_detect_maps_boxes_back_to_original_pixels():
    """整条链：1280x1024 输入 → 假网络给出画布坐标 → 应还原成原图坐标。"""
    image = np.zeros((1024, 1280, 3), dtype=np.uint8)
    _, scale, pad_x, pad_y = letterbox(image, 640)
    raw = _raw_with_boxes()
    # 原图 (100,200)-(300,400) -> 画布 (50,164)-(150,264) -> cx,cy,w,h
    x1, y1 = 100 * scale + pad_x, 200 * scale + pad_y
    x2, y2 = 300 * scale + pad_x, 400 * scale + pad_y
    _put_box(raw, 0, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0, 0.9)

    det = YoloDetector(weights="dummy.onnx")
    det._net = _FakeNet(raw)                     # 绕过文件加载
    got = det.detect(image)
    assert len(got) == 1, got
    box = got[0].bbox_xyxy
    assert all(abs(a - b) <= 1 for a, b in zip(box, (100, 200, 300, 400))), (box, (100, 200, 300, 400))
    assert isinstance(got[0], Detection)


def test_detect_returns_empty_list_for_empty_prediction():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    det = YoloDetector(weights="dummy.onnx")
    det._net = _FakeNet(_raw_with_boxes())
    assert det.detect(image) == []


def test_missing_weights_raises_readable_error():
    det = YoloDetector(weights="assets/models/definitely_missing.onnx")
    try:
        det.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    except FileNotFoundError as exc:
        text = str(exc)
        assert "definitely_missing.onnx" in text and "export_yolo" in text, text
    else:
        raise AssertionError("权重缺失时应当抛 FileNotFoundError")


def test_detect_rejects_bad_image():
    det = YoloDetector(weights="dummy.onnx")
    det._net = _FakeNet(_raw_with_boxes())
    try:
        det.detect(np.zeros((64, 64), dtype=np.uint8))
    except ValueError as exc:
        assert "HxWx3" in str(exc), exc
    else:
        raise AssertionError("灰度图应当抛 ValueError")


# ---------------------------------------------------------------- 标签
def test_label_uses_class_name_when_detector_gives_one():
    """YOLO 的框：标签应是「类名 + 置信度」，不是「unknown + 命中率」。"""
    det = Detection(bbox_xyxy=(0, 0, 10, 10), class_name="person", class_id=0,
                    score=0.89, color=None, color_ratio=0.15)
    name, metric = detection_label(det)
    assert name == "person", name
    assert abs(metric - 0.89) < 1e-6, metric


def test_label_falls_back_to_color_for_blob_detector():
    """判色族的框：class_name 是 "unknown"，应按颜色 + 命中率显示（原行为不变）。"""
    det = Detection(bbox_xyxy=(0, 0, 10, 10), color="red", color_ratio=0.72)
    name, metric = detection_label(det)
    assert name == "red", name
    assert abs(metric - 0.72) < 1e-6, metric


def test_label_stays_unknown_when_nothing_is_known():
    det = Detection(bbox_xyxy=(0, 0, 10, 10))
    name, metric = detection_label(det)
    assert name == "unknown" and metric == 0.0, (name, metric)


# ---------------------------------------------------------------- 注册与配置
def test_factory_registers_yolo():
    det = build_detector("yolo", {"weights": "dummy.onnx"})
    assert isinstance(det, YoloDetector)
    assert det.name == "yolo"


def test_config_rejects_out_of_range_values():
    for params, field in (
        ({"weights": "w.onnx", "conf": 1.5}, "conf"),
        ({"weights": "w.onnx", "iou": 0.0}, "iou"),
        ({"weights": "w.onnx", "imgsz": 100}, "imgsz"),
        ({"weights": "", "conf": 0.25}, "weights"),
    ):
        data = {"detector": {"name": "yolo", "yolo": params}}
        try:
            from_dict(data)
        except ConfigError as exc:
            assert field in str(exc), (field, str(exc))
        else:
            raise AssertionError(f"{params} 应当被配置校验拦下")


def test_config_accepts_valid_yolo_section():
    cfg = from_dict({"detector": {"name": "yolo",
                                  "yolo": {"weights": "assets/models/yolov8n.onnx",
                                           "conf": 0.3, "iou": 0.5, "imgsz": 320,
                                           "classes": [0, 2]}}})
    assert cfg.detector.name == "yolo"
    assert cfg.detector.params["imgsz"] == 320


# ---------------------------------------------------------------- 独立运行
def _run_standalone() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
        except Exception as exc:                     # noqa: BLE001
            failed += 1
            print(f"  ❌ {fn.__name__}: {type(exc).__name__}: {exc}")
    done = len(tests) - failed
    print(f"\n  test_yolo: {done}/{len(tests)} 通过" + (f"，{failed} 项失败" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
