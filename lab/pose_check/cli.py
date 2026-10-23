"""命令行与逐图模式的驱动：参数表、目录遍历、逐张打印、按模式分发。

用大白话说
    这是"把参数接住，再决定走哪条路"的地方：给 `--dir` 就一张张核、边核边打表；
    给 `--sweep` 就把活交给 `sweep.py` 跑整张矩阵。

放什么 / 不放什么
    放 `build_parser()`、`main()`、逐图模式的驱动 `_run_check()` 与目录遍历 `iter_images()`。
    **不放**单图核对（`check.py`）、批量统计与出图（`sweep.py`）、CSV 与报告（`report.py`）。
    没有把 `main()` 放进 `__init__.py`：包初始化时不该带参数解析与分支逻辑。

依赖谁 / 被谁用
    `check.py`（单图核对与门槛）、`sweep.py`（批量）、`config.loader`（读内参）、
    `calibration`（按分辨率缩放内参）、`detectors.aruco`（找码）。
    被 `__main__.py` 的 `pose-check` 子命令以及包内的 `__init__.py`（只做转发导出）调用。

对外约定
    `main(argv)` 返回退出码：0 = 全部通过 / 0 = 扫描跑完；1 = 有图不合格；
    2 = 参数或环境错误（没有图片、内参读不出来、两套内参都对不上）。
"""

import sys
from pathlib import Path

# `lab/` 不是包：直接 `python lab/xxx.py` 跑，脚本目录会进 sys.path，
# 但仓库根不会。这一行把仓库根补进去，才能 `import vision`（lab 可以依赖 vision，反过来不行）。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
import glob
import os

import numpy as np

from vision.config.loader import load_intrinsics
from vision.calibration import scale_intrinsics
from vision.detectors.aruco import ArucoDetector
from vision.pose import DEFAULT_METHOD
from .gen_scenes import DEFAULT_MARKER_MM
from .check import (DEFAULT_MAX_DIST_ERR, DEFAULT_MAX_RMS_PX, DEFAULT_MAX_TILT_ERR_DEG,
                    IMAGE_SUFFIXES, K_MISMATCH_RATIO, check_one, load_truth)
from .sweep import run_sweep


def iter_images(root: str):
    """目录下所有图片，按文件名排序（顺序稳定，输出可 diff）。

    :param root: 图片目录
    :return: 绝对/相对路径列表（与传入的 root 风格一致），排序去重
    """
    paths = []
    for suffix in IMAGE_SUFFIXES:
        paths.extend(glob.glob(os.path.join(root, "*" + suffix)))
    return sorted(set(paths))


def run_check(args) -> int:
    """跑整个目录并打印汇总；退出码 0 = 全部通过。

    :param args: 已解析的命令行参数
    :return: 0 全过 / 1 有不合格 / 2 参数或环境错误
    """
    images = iter_images(args.dir)
    if not images:
        print(f"❌ {args.dir} 下没有图片（看看路径或 --dir）")
        return 2

    base_intrinsics = None
    if args.intrinsics:
        try:
            base_intrinsics = load_intrinsics(args.intrinsics)
        except Exception as exc:                       # 标定文件写坏了就别往下跑
            print(f"❌ 读内参失败：{exc}")
            return 2

    detector = ArucoDetector()
    print(f"目录：{args.dir}   共 {len(images)} 张   码边长 {args.marker_mm} mm   "
          f"内参 {args.intrinsics or '取每张图真值'}")
    print("-" * 104)
    print(f"{'图名':<22}{'码数':>4}{'真值d':>9}{'估计d':>9}{'距离误差':>9}"
          f"{'真值tilt':>9}{'估计tilt':>9}{'RMS':>7}  结论")

    records, size_warned, k_warned = [], False, False
    for path in images:
        truth = load_truth(path)
        if base_intrinsics is not None:
            used = base_intrinsics
            if truth is not None and "canvas" in truth:
                want = tuple(int(v) for v in truth["canvas"])
                if want != tuple(base_intrinsics.image_size):
                    # 尺寸不同时按比例缩放内参，而不是硬套：硬套等于换了一颗长焦镜头，
                    # 实测 640×480 图上距离会解成 1571.9 mm（真值 450 mm，偏 249%）
                    used = scale_intrinsics(base_intrinsics, want)
                    if not size_warned:
                        size_warned = True
                        print(f"⚠️  内参按 {base_intrinsics.image_size} 标定，图是 {want}；"
                              "已按比例缩放（近似：畸变系数不缩放，严谨请重标定）")
            # 尺寸对得上但 fx 差得远，说明这套 K 根本不是渲染这张图用的那套：
            # 距离会整体偏（实测 fx 1321 顶 800 时 31 张全偏 65.8%）。
            # 只告警、不覆盖：--intrinsics 是显式参数，静默改成真值 K 会让
            # "我想试另一套内参"变成不可能。
            if not k_warned and truth is not None and "camera_matrix" in truth:
                truth_fx = float(np.asarray(truth["camera_matrix"], dtype=np.float64)[0, 0])
                used_fx = float(np.asarray(used.camera_matrix, dtype=np.float64)[0, 0])
                if truth_fx > 0 and abs(used_fx - truth_fx) / truth_fx > K_MISMATCH_RATIO:
                    k_warned = True
                    print(f"⚠️  内参 fx 与真值不一致：你给 {used_fx:.1f}，真值是 {truth_fx:.1f}"
                          f"（差 {abs(used_fx - truth_fx) / truth_fx * 100:.1f}%）；"
                          "继续用你给的那套，距离会整体偏")
            camera_matrix = used.camera_matrix
            dist_coeffs = used.dist_coeffs
        elif truth is not None and "camera_matrix" in truth:
            camera_matrix = np.asarray(truth["camera_matrix"], dtype=np.float64)
            dist_coeffs = None
        else:
            print(f"❌ {os.path.basename(path)} 既没有内参文件也没有真值相机参数；"
                  "用 --intrinsics 指定")
            return 2

        record = check_one(path, detector, camera_matrix, dist_coeffs, args.marker_mm,
                           args.method, args.max_dist_err, args.max_tilt_err, args.max_rms)
        records.append(record)
        truth_distance = float(truth["distance_mm"]) if truth and "distance_mm" in truth else float("nan")
        truth_tilt = float(truth["tilt_deg"]) if truth and "tilt_deg" in truth else float("nan")
        print(f"{record['image']:<22}{record['detected']:>4}{truth_distance:>9.1f}"
              f"{record['distance_mm']:>9.1f}{(record['distance_err'] or 0) * 100:>8.2f}%"
              f"{truth_tilt:>9.1f}{record['tilt_deg']:>9.1f}{record['reproj_rms_px']:>7.2f}  "
              f"{'✅' if record['ok'] else '⚠️ ' + record['reason']}")

    passed = sum(1 for r in records if r["ok"])
    print("-" * 104)
    print(f"  通过 {passed}/{len(records)}")
    print(f"\n{'✅ 位姿自检通过' if passed == len(records) else '❌ 位姿自检未通过'}")
    return 0 if passed == len(records) else 1


def build_parser() -> argparse.ArgumentParser:
    """命令行参数：逐图模式（`--dir`）与扫描模式（`--sweep`）共用一套参数。

    :return: 填好参数的 `ArgumentParser`；`--dir` 与 `--sweep` 互斥且必选其一
    """
    ap = argparse.ArgumentParser(
        prog="up70-vision-pose-check",
        description="位姿核对：逐个目录图核对（--dir）或跑距离/倾角/光照的批量扫描（--sweep）")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dir", help="图片目录（同名 .json 为真值）；逐图模式")
    mode.add_argument("--sweep", action="store_true",
                      help="扫描模式：距离/倾角/滚转/亮度/噪声/模糊逐档测检出率与误差")
    ap.add_argument("--out", default=os.path.join("output", "pose_sweep"),
                    help="扫描模式的输出目录，默认 output/pose_sweep")
    ap.add_argument("--intrinsics", default="",
                    help="内参 YAML；逐图模式不给则用真值里的 camera_matrix，"
                         "扫描模式不给则按画布尺寸造合成内参")
    ap.add_argument("--marker-mm", type=float, default=None,
                    help=f"码边长（毫米）；扫描模式默认 {DEFAULT_MARKER_MM}，"
                         "逐图模式默认 40（真值里有该字段时以真值为准）")
    ap.add_argument("--method", default=DEFAULT_METHOD, help=f"求解方法，默认 {DEFAULT_METHOD}")
    ap.add_argument("--max-dist-err", type=float, default=DEFAULT_MAX_DIST_ERR,
                    help=f"距离相对误差上限，默认 {DEFAULT_MAX_DIST_ERR}")
    ap.add_argument("--max-tilt-err", type=float, default=DEFAULT_MAX_TILT_ERR_DEG,
                    help=f"倾角绝对误差上限（度），默认 {DEFAULT_MAX_TILT_ERR_DEG}")
    ap.add_argument("--max-rms", type=float, default=DEFAULT_MAX_RMS_PX,
                    help=f"重投影 RMS 上限（像素），默认 {DEFAULT_MAX_RMS_PX}")
    return ap


def main(argv=None) -> int:
    """按模式分发：`--sweep` 走批量扫描，`--dir` 走逐图核对。

    :param argv: 参数列表；None 表示取 `sys.argv[1:]`
    :return: 逐图模式见 `run_check`；扫描模式见 `sweep.run_sweep`
    """
    args = build_parser().parse_args(argv)
    if args.sweep:
        if args.marker_mm is None:
            args.marker_mm = DEFAULT_MARKER_MM
        return run_sweep(args)
    if args.marker_mm is None:
        args.marker_mm = 40.0
    return run_check(args)
