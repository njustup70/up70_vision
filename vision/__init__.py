"""up70-vision：UP70 战队视觉组仓库（判色 / ArUco / 位姿 + 一个统一命令入口）。

项目目标
    2D 视觉基础任务——识别场上物体的颜色与位置，为后续配合机械臂、接入激光融合打底。
    算法主线是级联结构：先出候选框（后期换成 YOLO），再在框内做 HSV 判色。
    两段互相独立，可各自替换。

模块地图（2026-10-22 扁平化：已无 `core/`、`utils/` 中间层）
    types.py        Frame / Detection / ArucoMarker / MarkerPose 数据结构
    color.py        HSV 判色：多段阈值、红色跨色相环绕、命中率
    pipeline.py     逐帧跑「检测 -> 判色 -> 匹配计分」
    calibration.py  相机内参、去畸变、按分辨率等比缩放
    pose.py         solvePnP 位姿解算 + 重投影与前向性两条硬校验
    image_io.py     图像读写（Windows 中文路径兜底）
    logging.py      日志配置（整个进程调一次）
    detectors/      检测器一族：blob 判色 / aruco 真码 / stub_aruco 假码 / yolo
    config/         配置与阈值表的加载、校验（自成子系统，保留为子包）
    io/             输入源：camera（MVS）/ replay / synthetic
    viz/            画框、坐标轴、HUD、落盘
    cli/            命令行编排：parsers / run / live
    api.py          库调用门面（ROS 2 节点将来 import 这里）
    __main__.py     统一命令入口 `python -m vision`（含 clean）

依赖方向（硬约束，下层不知道上层存在）
    config ┐
           ├─> 根级算法模块 ─> io ─> viz ─> cli ─> __main__.py
    image_io/logging ┘                           └─> api.py
    方向是单向的：根级算法模块（types/color/pipeline/calibration/pose/detectors）
    永远不 import io / viz / cli，否则算法就无法脱离硬件单独测试。

为什么扁平而不是按"层"建子包
    扁平之前是 `core/` + `utils/`：改一行判色要在 `core/color.py` 与 `utils/logging.py`
    之间来回跳，import 写成 `from ..core.pipeline import`（数点数、错一个点就 ImportError）。
    扁平后同类文件都躺在包根，一眼能看完；只保留两个"真的是一族"的子包：
    `config/`（配置子系统）与 `detectors/`（可替换的检测器实现）。
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
