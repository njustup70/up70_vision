"""判色一族：调阈值、看实时判色效果。

放什么
    · `hsv_tuner.py`     交互调 HSV 阈值，按键写回 `configs/color_thresholds.yaml`
    · `color_measure.py` 海康 MVS 相机实时判色窗口（曝光/增益可调，存图）

不放什么
    不放判色算法（那是 `color.py`）。这两个都需要 GUI（X11/Wayland + 非 headless 的 OpenCV）。

怎么跑
    python -m up70_vision hsv-tuner --image data/clips/sample.jpg --colors red
    python -m up70_vision mvs-color
"""
