"""位姿核对（`--dir` 逐图 / `--sweep` 批量扫描）。入口见 `lab.pose_check.cli`。

五个文件按"读一次要读多少行"切：
    check.py      判定门槛 + 真值加载 + 单图核对
    sweep.py      `--sweep`：造矩阵、逐档统计、出 8 张单指标曲线
    report.py     `Row` / raw.csv / summary.md 与统计小函数
    gen_scenes.py 造带位姿真值的合成图
    cli.py        参数表、目录遍历、按模式分发（`main`）

跑法：`python -m lab.pose_check --dir output/pose_scenes`（或 `--sweep`）
"""

from .check import (DEFAULT_MAX_DIST_ERR, DEFAULT_MAX_RMS_PX, DEFAULT_MAX_TILT_ERR_DEG,
                    check_one)
from .cli import build_parser, main

__all__ = ["main", "build_parser", "check_one",
           "DEFAULT_MAX_DIST_ERR", "DEFAULT_MAX_TILT_ERR_DEG", "DEFAULT_MAX_RMS_PX"]
