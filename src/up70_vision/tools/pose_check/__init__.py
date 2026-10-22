"""对一批图逐张核位姿：估计值与真值并排打出来，超阈值标 ⚠️；`--sweep` 跑整张精度矩阵。

用大白话说
    `aruco_check.py` 管"码找得对不对"（ID、四角、中心）；这个工具管"算出来的距离和角度
    准不准"。给它一个图片目录，它一张张跑：检测 → 解位姿 → 和同名 JSON 里的真值比，
    把每个码的"真值 / 估计 / 误差"打成一行。有真值的图能算出误差，没真值的图只报自洽性
    （重投影 RMS）——这两种图的结论不能混着看。

包的五个文件（按"读一次要读多少行"切）
    check.py    判定门槛（DEFAULT_MAX_* / TILT_JUDGE_MIN_DEG）+ 真值加载 + 单图核对
    sweep.py    `--sweep`：造矩阵、逐档统计、出 8 张单指标曲线
    report.py   `Row` / `raw.csv` / `summary.md` 与统计小函数
    cli.py      参数表、目录遍历、逐图模式驱动、按模式分发（`main`）
    __init__.py 只做导出，不写逻辑

    拆之前是一个 817 行的单文件，混了四件互不影响的事（逐图核对 / 批量扫描 /
    CSV 与汇总 / matplotlib 出图）。门槛常量**只在 `check.py` 定义一份**，
    两种模式都从那里取——历史上这处门槛写了两份，改一处忘一处就会让
    "raw.csv 里通过"与"逐图核对里通过"变成两个意思。

依赖谁 / 被谁用
    `core.detectors.aruco`（找码）+ `core.pose`（解位姿）+ `core.calibration`（内参缩放）
    + `gen_pose_scenes`（造图）+ `config.loader`（读内参 YAML）。
    入口：`python -m up70_vision pose-check`。

对外约定
    · 真值 JSON 由 `gen_pose_scenes.py` 生成（含 `rvec`/`tvec_mm`/`distance_mm`/`tilt_deg`）
    · 只有 `assets/aruco/*.json`（由 `gen_aruco` 生成）时也能跑：那种真值没有位姿字段，
      所以只核对"检出 + 重投影 RMS"，不核对距离与角度
    · 退出码：0 = 全部通过（或扫描跑完）；1 = 有图未检出 / ID 不符 / 误差超限 / 解算失败；
      2 = 参数或环境错误
"""

from .check import (DEFAULT_MAX_DIST_ERR, DEFAULT_MAX_RMS_PX, DEFAULT_MAX_TILT_ERR_DEG,
                    check_one)
from .cli import build_parser, main

__all__ = ["main", "build_parser", "check_one",
           "DEFAULT_MAX_DIST_ERR", "DEFAULT_MAX_TILT_ERR_DEG", "DEFAULT_MAX_RMS_PX"]
