"""人用的工具脚本，不属于运行管线的一部分。

这些脚本的特点是：给人用、或给验证用——要么需要人在旁边看着（拖滑条调阈值），
要么是"造素材 + 对答案"的离线工具。它们**不被运行管线（cli/pipeline）import**，
其中纯离线的三个（`gen_aruco` / `aruco_check` / `mvs_color_measure`）可以直接命令行跑。

依赖谁
    可以依赖任意层——它是包的最上层消费方。

当前有什么
    · hsv_tuner.py         交互调 HSV 阈值，按键写回 configs/color_thresholds.yaml
    · mvs_color_measure.py 海康 MVS 相机实时判色窗口（曝光/增益可调，存图）
    · gen_aruco.py         生成 ArUco 测试图组 + 真值 JSON（单码 / 多码 / 旋转 / 无码）
    · aruco_check.py       拿图组对真值自检（ID、四角、中心），一条命令给出具体数字

注意 GUI
    hsv_tuner 与 mvs_color_measure 需要 X11/Wayland，且 OpenCV 不能是 headless 版。
    先用 python -c "import cv2; print(cv2.getBuildInformation())" 看 GUI 一行再决定。
    gen_aruco / aruco_check 是纯离线的，不需要任何显示环境。
"""
