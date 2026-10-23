"""位姿核对的落盘与统计：一条记录的结构、raw.csv、人看的 summary.md，以及几个统计小函数。

用大白话说
    扫描完一堆图以后，要把每个场景的结果存成表（CSV，给机器看），再把统计与结论写成
    一份能直接贴进 PR 的报告（Markdown，给人看）。这个文件就是那两支笔。

放什么 / 不放什么
    放 `Row`（一行测量结果）、`write_csv()`、`write_summary()`，以及它们用的统计函数
    （中位数、检出率、"哪一档开始不能用"）。**不放**检测、解位姿、出图：
    那分别属于 `check.py` 与 `sweep.py`。

依赖谁 / 被谁用
    只依赖 numpy（统计）与标准库（csv / os）。被 `sweep.py` 调用；
    `median_value()` 也被 `sweep.py` 的曲线用（一处算法只写一遍）。

对外约定
    · `write_csv()` 的列由 `CSV_COLUMNS` 固定，顺序不可随意调整（下游按列名读，不按下标）
    · 空集合统计返回 `nan` 而不是 0：报告里显示 nan 比显示 0 更诚实
"""

import csv
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Row:
    """一个场景的测量结果（也是 `raw.csv` 的一行）。

    属性
        scene (str)        场景名
        group (str)        所属维度：dist / tilt / spin / bright / noise / blur / res
        level (float)      该维度上的档位（距离 mm / 角度 / 倍数 / 像素）
        detected (int)     检出码数
        ok (bool)          是否通过（检出、ID 正确、误差与 RMS 都在门槛内）
        distance_mm / distance_err / tilt_deg / tilt_err_deg / yaw_err_deg /
        reproj_rms_px (float)  估计值与误差；未检出时是 nan
        cell_px (float)    码在图像里"每格"占多少像素（边长 / 6），判可检性的关键量
        reason (str)       未通过的原因
    """

    scene: str
    group: str
    level: float
    detected: int
    ok: bool
    distance_mm: float
    distance_err: float
    tilt_deg: float
    tilt_err_deg: float
    yaw_err_deg: float
    reproj_rms_px: float
    cell_px: float
    reason: str = ""


CSV_COLUMNS = ("scene", "group", "level", "detected", "ok", "distance_mm", "distance_err",
               "tilt_deg", "tilt_err_deg", "yaw_err_deg", "reproj_rms_px", "cell_px", "reason")


def group_and_level(scene: str) -> tuple:
    """从场景名（`dist_0450` 这类）解析出"维度 + 档位"。

    名字由 `gen_pose_scenes.build_suite` 生成，格式固定；解析失败就归到 `other`、
    档位记 0，这样新增场景名忘了改这里也不会崩，只是不进任何一张曲线。

    :param scene: 场景名，形如 `dist_0450` / `tilt_15` / `bright_220`
    :return: `(维度, 档位浮点值)`；解析不出来时 `("other", 0.0)`
    """
    try:
        group, _, number = scene.rpartition("_")
        return (group or "other"), float(number)
    except ValueError:
        return "other", 0.0


def median_value(values) -> float:
    """中位数；空列表返回 nan（报告里显示成 nan 比显示 0 更诚实）。

    为什么取中位数而不是均值：单张失败样本的误差可能到几十个点，均值会被它拉走；
    中位数反映"典型情况"。

    :param values: 可迭代的浮点数，nan 会被跳过
    :return: 中位数；一个有限值都没有时返回 nan
    """
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def write_csv(path: str, rows) -> str:
    """把每行测量结果写成 CSV（真值/估计/误差/原因都在里面）。

    :param path: 输出路径；父目录不存在时自动建
    :param rows: `list[Row]`
    :return: 实际写入的路径
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in CSV_COLUMNS})
    return path


def _usable_rate(rows) -> float:
    """一组场景的检出率（检出即算，不看误差是否超限）。"""
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.detected > 0) / float(len(rows))


def _first_fail_after(rows) -> str:
    """按档位排序后，找"还能用"的最后一档与"第一档不能用的"，拼成一句结论。"""
    ordered = sorted(rows, key=lambda row: row.level)
    last_ok = None
    for row in ordered:
        if row.detected > 0 and row.ok:
            last_ok = row
        elif last_ok is not None:
            return (f"{last_ok.level:g} 还能用，{row.level:g} 开始不可用"
                    f"（原因：{row.reason or '检出但误差超限'}）")
    return "全部档位都通过" if ordered else "无样本"


def write_summary(path: str, rows, specs_count: int, marker_mm: float,
                  intrinsics_note: str) -> str:
    """写人看的汇总：分组统计表 + 失败清单 + 三条能直接引用的结论。

    :param path: 输出路径（`summary.md`）；父目录不存在时自动建
    :param rows: `list[Row]`
    :param specs_count: 本次一共跑了多少个合成场景
    :param marker_mm: 本次用的码边长（毫米），写进报告头，避免"这份数字是哪批码的"搞混
    :param intrinsics_note: 用了哪套内参的一句话描述
    :return: 实际写入的路径
    """
    lines = ["# 位姿精度扫描报告", "",
             f"- 样本数：{specs_count} 个合成场景",
             f"- 码边长：{marker_mm} mm（**占位值时要换成实测值**）",
             f"- 内参：{intrinsics_note}",
             "", "## 分组统计", "",
             "| 维度 | 档位数 | 检出率 | 距离误差中位数 | 倾角误差中位数 | 通过数 |",
             "|---|---|---|---|---|---|"]
    groups = {}
    for row in rows:
        groups.setdefault(row.group, []).append(row)
    for group in sorted(groups):
        bucket = groups[group]
        lines.append(
            f"| {group} | {len(bucket)} | {_usable_rate(bucket) * 100:.0f}% | "
            f"{median_value([row.distance_err for row in bucket]) * 100:.2f}% | "
            f"{median_value([row.tilt_err_deg for row in bucket]):.2f}° | "
            f"{sum(1 for row in bucket if row.ok)} |")

    lines += ["", "## 失败样本", ""]
    failed = [row for row in rows if not row.ok]
    if failed:
        lines += ["| 场景 | 档位 | 检出 | 原因 |", "|---|---|---|---|"]
        for row in failed:
            lines.append(f"| {row.scene} | {row.level:g} | {row.detected} | {row.reason} |")
    else:
        lines.append("无。")

    detected_rows = [row for row in rows if row.detected > 0]
    missing = [row for row in rows if row.detected == 0]
    lines += ["", "## 结论（都由本表数字直接得出）", ""]
    if detected_rows:
        lines.append(f"- 检出样本的每格像素范围：{min(row.cell_px for row in detected_rows):.1f}"
                     f"–{max(row.cell_px for row in detected_rows):.1f} px")
    if missing:
        # 只统计"未检出"的样本，并列出场景名：它们的原因不限于尺寸（还有过曝），
        # 混进"每格像素"里当尺寸结论会被误读
        lines.append(f"- 未检出样本：{', '.join(row.scene for row in missing)}"
                     f"（每格 {min(row.cell_px for row in missing):.1f}"
                     f"–{max(row.cell_px for row in missing):.1f} px；原因不限于尺寸）")
    for group, label in (("dist", "距离"), ("tilt", "倾角"), ("bright", "亮度"),
                         ("blur", "模糊"), ("noise", "噪声")):
        if group in groups:
            lines.append(f"- {label}：{_first_fail_after(groups[group])}")
    lines += ["", "> 距离误差只在**检出**的样本上统计：没检出与算不准是两种失败，",
              "> 混在一起算会让曲线在拐点处虚高，也没法归因。",
              "",
              "> 真值倾角 < 10° 的样本只判距离、不判倾角：正对相机时绕 x/y 轴转 10° 只让投影高度"
              "缩 1.5%，",
              "> 与角点噪声同量级（实测误差可达 10°）。误差值仍记在 raw.csv 里，只是不计入通过与否。",
              "",
              "> 这些数字来自**合成图**（无噪声压缩、无镜头畸变、边缘是理想直线），",
              "> 属于识别能力的**上限**。真机要额外留余量：教程第三节给的"
              "「每格 5–10 px」是按实拍算的，",
              "> 合成图在 3 px 附近还能认出，不代表场上也能。"]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path
