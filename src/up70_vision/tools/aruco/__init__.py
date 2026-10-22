"""ArUco 一族：造测试图组 + 按真值核对检测结果。

放什么
    · `gen_scenes.py` 生成测试图组与真值 JSON（单码 / 多码 / 旋转 / 无码），落在 `assets/aruco/`
    · `check.py`      拿图组对真值自检（ID、四角、中心），一条命令给出具体数字

不放什么
    不放检测算法本身——那是 `detectors/aruco.py`（本目录只负责"造题"与"判卷"）。

怎么跑
    python -m up70_vision gen-aruco
    python -m up70_vision check-aruco
"""
