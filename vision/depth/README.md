# `depth/`：深度相机（暂不开工）

这个目录是**占位**，里面没有任何 `.py`。

按 `docs/迁移计划书` D11：深度相机是计划内的第三个功能，但现在不开工。

- 现在**不要**在这里写代码，也**不要**往根级 `types.py` 里塞 3D 类型
  （`DepthFrame` / `PointCloud` / `GraspPoint` 一个都别提前定义）；
- 真开工时按 `aruco/` `yolo/` 的形状建：`depth/types.py` + `depth/core.py`，
  3D 类型只放 `depth/types.py`，与根级 `types.py` 互不 import。

一句话：**空目录只是记一笔"这里将来有东西"，提前搭结构才是被批评的那种过度工程。**
