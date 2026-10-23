"""up70_vision：运行时代码包（要 import、要测、要评审、有架构约束）。

目录（按功能竖切，D2）
    types.py       唯一共用件：Frame / Detection / 几何函数
    api.py         库门面：detect(image_bgr, detector)
    aruco/         ArUco：detector / pose / calibration / draw / types
    yolo/          YOLO 与判色：detector / candidates / color_check / pipeline / stub / draw
    sources/       输入源：camera（MVS）/ replay / synthetic + read_image_bgr
    config/        YAML 加载、字段校验、--set 覆盖
    cli/           命令适配层：parsers / run / live / output
    depth/         深度相机占位（只有 README）

依赖方向（硬约束，`tests/test_architecture.py` 会检查）
    aruco/ 与 yolo/ 互不 import；功能包不 import cli、config；
    根级 types.py 不 import 任何功能包；vision/ 不 import lab/。
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
