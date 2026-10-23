"""人用的工具，不属于运行管线的一部分。按功能分成七个子包，找工具只看一个目录。

这些工具的特点是：给人用、或给验证用——要么需要人在旁边看着（拖滑条调阈值），
要么是"造素材 + 对答案"的离线工具。它们**不被运行管线（cli）import**。

子包（2026-10-22 从 17 个平铺文件分组而来）
    env/          环境自检 + OpenCV 冒烟（装环境装坏了时唯一还能跑的东西）
    aruco/        造 ArUco 测试图组、按真值核对检测结果
    blob/         判色：交互调 HSV 阈值、相机实时判色窗口
    yolo/         导出 ONNX 权重（推理在 detectors/yolo.py）
    calibration/  打印棋盘、采图解内参
    pose/         造带真值的位姿图、逐图核对、批量扫描
    doc/          教程 md -> html 的构建

依赖谁
    可以依赖任意层——它是包的最上层消费方。

怎么跑
    一律走统一入口：`python -m vision <子命令>`（见 `__main__.py` 的登记表，
    七个子包各至少有一个子命令，`gen_html` 对应 `gen-html`）。

注意 GUI
    blob/ 下两个（hsv_tuner / color_measure）需要 X11/Wayland，且 OpenCV 不能是 headless 版。
    先用 python -c "import cv2; print(cv2.getBuildInformation())" 看 GUI 一行再决定。
"""
