"""YOLO 权重导出：`.pt` -> `.onnx` + 类名表。

放什么
    `export.py`：在装了 ultralytics 的独立 venv 里跑一次，产出 `assets/models/*.onnx`。
    仓库本体只依赖 `cv2.dnn` 做推理，**不需要 torch**——所以导出与推理刻意分家。

怎么跑
    python -m up70_vision export-yolo --weights yolov8n.pt
"""
