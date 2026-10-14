"""检测器插槽：把「出候选框」这一步做成可替换的。

放什么
    Detector 抽象基类、按 detector.name 构造的工厂，以及各检测器实现。

不放什么
    不放判色逻辑。检测器只负责"哪里有东西"，"是什么颜色"由 core/color.py 填。
    这个切分让两个阶段可以独立替换——换模型不用动判色，调阈值不用动检测。

依赖谁
    只依赖 core.types 与 core.color（占位检测器需要借用判色的阈值表生成前景掩码）。

当前为空，未来第一件事
    先写 base.py：抽象基类 + build_detector 工厂；
    再写 blob.py：饱和度/亮度分割 + 轮廓提取，在 YOLO 就位前跑通链路。
    两条约定：返回的 Detection.class_name 填 "unknown"、class_id 填 -1；
    bbox 用绝对像素坐标，且右/下边界是半开上界。
"""
