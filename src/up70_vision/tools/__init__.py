"""人用的工具脚本，不属于运行管线的一部分。

这些脚本的特点是：需要人在旁边看着、需要鼠标键盘交互（如拖滑条调阈值），
因此永远不进自动化流程，也不被管线 import。

依赖谁
    可以依赖任意层——它是包的最上层消费方。

当前为空，未来第一件事
    写 tools/hsv_tuner.py：交互调 HSV 阈值，按 s 写回 configs/color_thresholds.yaml。
    注意它需要 GUI：Ubuntu 上 cv2.imshow 要求有 X11 或 Wayland，
    服务器版装的 OpenCV 常常是 headless 版，窗口打不开。
    先用 python -c "import cv2; print(cv2.getBuildInformation())" 看 GUI 一行再决定。
"""
