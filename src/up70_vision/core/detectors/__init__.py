"""检测器插槽：把「出候选框」这一步做成可替换的。

放什么
    Detector 抽象基类、按 detector.name 构造的工厂，以及各检测器实现。

不放什么
    不放判色逻辑。检测器只负责"哪里有东西"，"是什么颜色"由 core/color.py 填。
    这个切分让两个阶段可以独立替换——换模型不用动判色，调阈值不用动检测。

依赖谁
    只依赖 core.types 与 core.color（占位检测器需要借用判色的阈值表生成前景掩码）。

当前有什么
    base.py（抽象基类 + build_detector 工厂）、blob.py（饱和度/亮度分割 + 轮廓提取，
    返回 `list[Detection]`）、stub_aruco.py（按契约返回**假** `ArucoMarker`）、
    aruco.py（**真 ArUco，DICT_4X4_50**，走 OpenCV 4.7+ 的 `ArucoDetector` 新 API）。
    YOLO 待接。

两种返回值（docs/interface.md 第 2、3 节）
    · 判色族 `list[Detection]`：轴对齐 bbox，class_name 填 "unknown"、class_id 填 -1
    · ArUco `list[ArucoMarker]`：四角点，按 `marker_id` 升序，无码返回 `[]`
    两者都：无目标返回 `[]`、形参名固定 `image_bgr`、不读文件不开窗口。
"""
