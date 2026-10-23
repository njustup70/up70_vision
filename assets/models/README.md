# 模型权重（不进版本库）

这个目录放**模型文件**，内容全部被 `.gitignore` 忽略（`*.onnx` / `*.pt`），
所以仓库里只有这份说明，克隆下来是空的。

## 需要什么

`detectors/yolo.py` 只认 **ONNX**（用 `cv2.dnn` 推理，容器里不需要 torch）：

```
assets/models/yolov8n.onnx        12.9 MB   模型本体
assets/models/yolov8n.names.json  类名表    导出时自动生成（容器里没有 onnx 包也能读）
```

## 怎么得到（在装了 ultralytics 的机器上，跑一次就够）

```bash
# 本机是 ~/.venvs/yolo（torch CPU 版 + ultralytics）
# 该 venv 需要能 import 本包：在里面 `pip install -e .`，或按下面这样带 PYTHONPATH
PYTHONPATH=src ~/.venvs/yolo/bin/python -m vision export-yolo --weights yolov8n.pt
```

输入 `.pt` 不存在时 ultralytics 会自动下载（约 6 MB）。导出脚本会把 ONNX 和类名
sidecar 一起写进本目录，并打印运行时该用的 `--set` 参数。

## 选别的模型

```bash
# 更准但更慢 / 更快但更粗
PYTHONPATH=src ~/.venvs/yolo/bin/python -m vision export-yolo --weights yolo11s.pt --imgsz 640
```

导完把 `configs/default.yaml` 的 `detector.yolo.weights` 指过去即可。

## 实测参考（这台机器，纯 CPU）

| 输入 | 耗时 | 备注 |
|---|---|---|
| 640x640 直接前向 | 26 ms | `cv2.dnn` 纯推理 |
| 1280x1024 一帧（含 letterbox + NMS） | 26~35 ms | 容器 cv2 4.14 |

相机最高 25 fps（40 ms/帧预算），够用。

## ⚠️ 许可

YOLOv8 的**权重与代码是 AGPL-3.0**。队内自用没问题；如果要对外发布程序（比赛提交、
开源、交付给学校），AGPL 要求一并提供源码。正式交付前请跟队长确认，或换
Apache-2.0 / MIT 的模型（YOLOX、NanoDet 等，接口一样挂在这个目录下）。

## 预训练权重认什么

COCO 的 80 类（人、车、瓶子、杯子…）。**不认识我们的色块**——实测在 `data/calib`
的棋盘图上是 0 个框，属正常。要它认物块，得标注数据 + 微调（见 `docs/progress.md` 的 M5）。
