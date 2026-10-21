"""对一批图逐张核位姿：每张图把估计值与真值并排打出来，超阈值标 ⚠️。

用大白话说
    `aruco_check.py` 管"码找得对不对"（ID、四角、中心）；这个工具管"算出来的距离和角度
    准不准"。给它一个图片目录，它一张张跑：检测 → 解位姿 → 和同名 JSON 里的真值比，
    把每个码的"真值 / 估计 / 误差"打成一行。有真值的图能算出误差，没真值的图只报自洽性
    （重投影 RMS）——这两种图的结论不能混着看。

放什么 / 不放什么
    放"逐图核对 + 打印 + 退出码"，以及 `--sweep` 的批量扫描 / 统计 / 出图
    （原 `tools/pose_sweep.py`，已合并进来：两边共用同一套门槛与字段名，
    分两个入口时改一处就得同步另一处）。不做渲染（`tools/gen_pose_scenes.py`）。

依赖谁 / 被谁用
    `core.detectors.aruco`（找码）+ `core.pose`（解位姿）+ `config.loader`（读内参 YAML）。
    薄入口是 `scripts/pose_check.py`；扫描模式复用本文件的 `_check_rendered()`。

对外约定
    · 真值 JSON 由 `tools/gen_pose_scenes.py` 生成（含 `rvec`/`tvec_mm`/`distance_mm`/`tilt_deg`）
    · 只有 `assets/aruco/*.json`（由 `gen_aruco` 生成）时也能跑：那种真值没有位姿字段，
      所以只核对"检出 + 重投影 RMS"，不核对距离与角度
    · 退出码：0 = 全部通过；1 = 有图未检出 / ID 不符 / 误差超限 / 解算失败

已知的坑
    · **内参必须和图的分辨率匹配**。1280×1024 的内参套到 640×480 的图上，距离会整体偏，
      程序不报错。本工具在两者不一致时会打一行警告（不中止：缩放回放是常见调试手段）。
    · 真值图里的 `marker_size_mm` 才是这批图的真实边长；命令行 `--marker-mm` 只用于
      "没有真值字段"的老图组。两者不一致时以真值为准并提示。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from ..config.loader import load_intrinsics
from ..core.calibration import scale_camera_matrix, scale_intrinsics
from ..core.detectors.aruco import ArucoDetector
from ..core.pose import DEFAULT_METHOD, estimate_pose
from ..utils.image_io import read_image_bgr
from ..viz.draw import put_text
from .gen_pose_scenes import (DEFAULT_MARKER_MM, PoseSpec, build_suite, render_scene,
                              synthetic_intrinsics)

#: 判定"通过"的默认门槛。依据见 docs/位姿估计.md 的实测表：
#: 450–600 mm、40 mm 码时距离误差 < 1%、倾角误差 < 2°，留一倍余量。
DEFAULT_MAX_DIST_ERR = 0.04
DEFAULT_MAX_TILT_ERR_DEG = 3.0
DEFAULT_MAX_RMS_PX = 2.0

#: `--intrinsics` 与真值里的 `fx` 相对差超过这个比例就告警。
#: 5% 是"肉眼还看不出偏、但距离已经偏 5%"的量级；只告警不覆盖参数（见 main 里的说明）。
K_MISMATCH_RATIO = 0.05

#: 真值倾角小于这个角度时**不判倾角**（误差照记，只是不算通过与否）。
#: 理由：码正对相机时，绕 x/y 轴转 10° 只让投影高度缩 1.5%——67 px 的码差 1 px，
#: 与角点定位噪声同量级，所以"正视时的倾角"在几何上就是弱可观的，实测误差可达 10°。
#: 拿 3° 门槛去卡它，测的是不可观测量，不是算法好坏。倾斜 15° 以上才判。
TILT_JUDGE_MIN_DEG = 10.0

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def _truth_path_for(image_path: str) -> str:
    """同名 JSON 真值：`foo.png` → `foo.json`。"""
    return os.path.splitext(image_path)[0] + ".json"


def _load_truth(image_path: str):
    """读真值 JSON；没有就返回 None（老图组只有 PNG 的情况）。"""
    path = _truth_path_for(image_path)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _yaw_delta(deg_a: float, deg_b: float) -> float:
    """角度差折算到 (-180, 180]，避免 179° 与 -179° 差出 358°。"""
    return ((deg_a - deg_b + 180.0) % 360.0) - 180.0


def check_one(image_path: str,
              detector,
              camera_matrix,
              dist_coeffs=None,
              marker_size_mm: float = 40.0,
              method: str = DEFAULT_METHOD,
              max_dist_err: float = DEFAULT_MAX_DIST_ERR,
              max_tilt_err_deg: float = DEFAULT_MAX_TILT_ERR_DEG,
              max_rms_px: float = DEFAULT_MAX_RMS_PX,
              tilt_judge_min_deg: float = TILT_JUDGE_MIN_DEG) -> dict:
    """核对一张图，返回一条可打印、可入 CSV 的记录。

    用大白话说
        打开图 → 找码 → 对每个码解位姿 → 和真值比距离与倾角。返回的 dict 里
        既有"估了多少"，也有"差多少"，外加一个 `ok` 供调用方决定退出码。

    :param image_path: 图片路径
    :param detector: 已建好的 `ArucoDetector`（复用能省掉每张图重复建字典的开销）
    :param camera_matrix: (3,3) 内参
    :param dist_coeffs: 畸变系数或 None
    :param marker_size_mm: 码边长（毫米）；真值里有 `marker_size_mm` 时以真值为准
    :param method: 位姿求解方法
    :param max_dist_err: 距离相对误差上限（比例）
    :param max_tilt_err_deg: 倾角绝对误差上限（度）
    :param max_rms_px: 重投影 RMS 上限（像素）
    :param tilt_judge_min_deg: 真值倾角小于它时不判倾角（理由见常量说明）
    :return: dict，键见 `_blank_record()`；`reason` 非空表示没通过
    """
    image_bgr = read_image_bgr(image_path)
    truth = _load_truth(image_path)
    name = os.path.basename(image_path)
    record = {"image": name, "detected": 0, "id_ok": 0, "distance_mm": float("nan"),
              "distance_err": float("nan"), "tilt_deg": float("nan"),
              "tilt_err_deg": float("nan"), "yaw_deg": float("nan"),
              "reproj_rms_px": float("nan"), "ok": False, "reason": ""}

    if truth is not None:
        marker_size_mm = float(truth.get("marker_size_mm", marker_size_mm))

    markers = detector.detect(image_bgr)
    record["detected"] = len(markers)
    if not markers:
        record["reason"] = "未检出"
        return record

    want_id = None if truth is None else int(truth["marker_id"])
    target = next((m for m in markers if want_id is None or int(m.marker_id) == want_id), None)
    if target is None:
        record["reason"] = f"ID 不符（检出 {[int(m.marker_id) for m in markers]}，期望 {want_id}）"
        return record
    record["id_ok"] = 1

    try:
        pose = estimate_pose(target, marker_size_mm, camera_matrix, dist_coeffs,
                             method=method, max_reproj_rms_px=max_rms_px)
    except ValueError as exc:
        record["reason"] = f"解算失败：{exc}"
        return record

    record["distance_mm"] = pose.distance_mm
    record["tilt_deg"] = pose.tilt_deg
    record["yaw_deg"] = pose.yaw_deg
    record["reproj_rms_px"] = pose.reproj_rms_px

    if truth is None or "distance_mm" not in truth:
        # 老图组（gen_aruco 造的）没有位姿真值：只能验自洽，不能断言准不准
        record["ok"] = pose.reproj_rms_px <= max_rms_px
        record["reason"] = "" if record["ok"] else "重投影 RMS 超限"
        return record

    truth_distance = float(truth["distance_mm"])
    record["distance_err"] = abs(pose.distance_mm - truth_distance) / truth_distance
    record["tilt_err_deg"] = abs(pose.tilt_deg - float(truth["tilt_deg"]))
    yaw_err = abs(_yaw_delta(pose.yaw_deg, float(truth["yaw_deg"])))

    problems = []
    if record["distance_err"] > max_dist_err:
        problems.append(f"距离误差 {record['distance_err'] * 100:.1f}% > {max_dist_err * 100:.0f}%")
    # 真值接近正视时不判倾角：那个自由度本身弱可观（见 TILT_JUDGE_MIN_DEG 的说明），
    # 误差仍然记进 record，报告里看得到，只是不计入"通过/不通过"
    if float(truth["tilt_deg"]) >= tilt_judge_min_deg and record["tilt_err_deg"] > max_tilt_err_deg:
        problems.append(f"倾角误差 {record['tilt_err_deg']:.1f}° > {max_tilt_err_deg:.0f}°")
    if pose.reproj_rms_px > max_rms_px:
        problems.append(f"RMS {pose.reproj_rms_px:.2f}px > {max_rms_px:.1f}px")
    # yaw 只报告不判定：码小、倾角大时 0.2 px 的角点噪声就能让面内角跳几度，
    # 把它做成一票否决会让"能用的图"被判成失败
    record["yaw_err_deg"] = yaw_err
    record["ok"] = not problems
    record["reason"] = "；".join(problems)
    return record




def _iter_images(root: str):
    """目录下所有图片，按文件名排序（顺序稳定，输出可 diff）。"""
    paths = []
    for suffix in IMAGE_SUFFIXES:
        paths.extend(glob.glob(os.path.join(root, "*" + suffix)))
    return sorted(set(paths))


#: 图幅与边距（像素）。宽一点是为了 x 轴刻度标签不重叠。
CHART_SIZE = (820, 480)
CHART_MARGIN = (90, 56, 40, 60)      # 左、上、右、下；上边距要放得下标题 + 最高刻度

#: 统计里"误差"取中位数而不是均值：单张失败样本的误差可能到几十个点，
#: 均值会被它拉走，中位数反映"典型情况"。
QUANTILE = 0.5

#: 判"这一段还能用"的检出率门槛。低于它有 1/5 的场景认不出，实际场上不可接受。
USABLE_DETECT_RATE = 0.8


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


def _group_and_level(scene: str) -> tuple:
    """从场景名（`dist_0450` 这类）解析出"维度 + 档位"。

    名字由 `build_suite` 生成，格式固定；解析失败就归到 `other`、档位记 0，
    这样新增场景名忘了改这里也不会崩，只是不进任何一张曲线。
    """
    try:
        group, _, number = scene.rpartition("_")
        return (group or "other"), float(number)
    except ValueError:
        return "other", 0.0


def _cell_px(corners_px) -> float:
    """码在图像里"每格"多少像素：四边平均长 / 6。

    为什么是 6 而不是 4：`DICT_4X4_50` 的数据区是 4 格，但码本身还有一圈黑边框，
    检测器看到的黑框是 6 格宽（4 数据 + 每边 1 格边框）。教程第三节给的硬下限
    "每格 5–10 px"说的就是这 6 格里的每一格。
    """
    pts = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    edges = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    return float(np.mean(edges)) / 6.0


def run_suite(specs,
              marker_size_mm: float = DEFAULT_MARKER_MM,
              method: str = "ippe_square",
              camera_matrix=None,
              camera_size=None,
              max_dist_err: float = 0.04,
              max_tilt_err_deg: float = 3.0,
              max_rms_px: float = 2.0) -> list:
    """逐个场景渲染、检测、解位姿并比对真值。

    :param specs: `list[PoseSpec]`，一般来自 `build_suite()`
    :param marker_size_mm: 码边长（毫米）；场景自带的边长优先
    :param method: 位姿求解方法
    :param camera_matrix: 使用哪套内参；None 表示按每个场景的画布自造合成内参
    :param camera_size: `camera_matrix` 对应的 `(宽, 高)`；与场景画布不同时自动等比缩放
    :return: `list[Row]`，顺序与 `specs` 一致
    """
    detector = ArucoDetector()
    rows = []
    for spec in specs:
        # 内参必须与画布尺寸匹配：1280 的内参套到 640 画布上等于换了一颗长焦镜头，
        # "每格像素"这条结论会全错（progress.md 坑 3 的扫描版）
        matrix = (synthetic_intrinsics(spec.canvas_size[0], spec.canvas_size[1])
                  if camera_matrix is None
                  else scale_camera_matrix(camera_matrix, camera_size or spec.canvas_size,
                                           spec.canvas_size))
        image_bgr, truth = render_scene(spec, matrix)
        cell = _cell_px(truth["corners_px"])
        group, level = _group_and_level(spec.name)

        # 用与 pose_check 同名的字段与同一套门槛（见 _check_rendered 的说明），
        # 保证"raw.csv 里通过"与"pose_check 里通过"是同一个意思
        record = _check_rendered(image_bgr, truth, detector, matrix, spec.marker_size_mm,
                                 method, max_dist_err, max_tilt_err_deg, max_rms_px)

        rows.append(Row(scene=spec.name, group=group, level=level,
                        detected=record["detected"], ok=bool(record["ok"]),
                        distance_mm=record["distance_mm"],
                        distance_err=record["distance_err"], tilt_deg=record["tilt_deg"],
                        tilt_err_deg=record["tilt_err_deg"],
                        yaw_err_deg=record.get("yaw_err_deg", float("nan")),
                        reproj_rms_px=record["reproj_rms_px"], cell_px=cell,
                        reason=record["reason"]))
    return rows


def _check_rendered(image_bgr, truth, detector, camera_matrix, marker_size_mm, method,
                    max_dist_err, max_tilt_err_deg, max_rms_px) -> dict:
    """对"已经渲染在内存里"的一张图做和 `pose_check.check_one` 相同的核对。

    为什么不用 `check_one`：那个函数从磁盘读图、自己找同名 JSON；这里的图还没落盘。
    两边的门槛与字段保持一一对应，`raw.csv` 的列也对齐，改动时两边一起改。
    """
    from ..core.pose import estimate_pose

    record = {"detected": 0, "ok": False, "distance_mm": float("nan"),
              "distance_err": float("nan"), "tilt_deg": float("nan"),
              "tilt_err_deg": float("nan"), "yaw_err_deg": float("nan"),
              "reproj_rms_px": float("nan"), "reason": ""}
    markers = detector.detect(image_bgr)
    record["detected"] = len(markers)
    want_id = int(truth["marker_id"])
    target = next((m for m in markers if int(m.marker_id) == want_id), None)
    if target is None:
        record["reason"] = "未检出" if not markers else "ID 不符"
        return record

    try:
        pose = estimate_pose(target, float(truth["marker_size_mm"]), camera_matrix, None,
                             method=method, max_reproj_rms_px=max_rms_px)
    except ValueError as exc:
        record["reason"] = f"解算失败：{exc}"
        return record

    truth_distance = float(truth["distance_mm"])
    record["distance_mm"] = pose.distance_mm
    record["tilt_deg"] = pose.tilt_deg
    record["reproj_rms_px"] = pose.reproj_rms_px
    record["distance_err"] = abs(pose.distance_mm - truth_distance) / truth_distance
    record["tilt_err_deg"] = abs(pose.tilt_deg - float(truth["tilt_deg"]))
    record["yaw_err_deg"] = abs(((pose.yaw_deg - float(truth["yaw_deg"]) + 180.0) % 360.0) - 180.0)

    problems = []
    if record["distance_err"] > max_dist_err:
        problems.append(f"距离误差 {record['distance_err'] * 100:.1f}%")
    # 与 pose_check 同一条规矩：接近正视时不判倾角（那个自由度弱可观），误差照记
    if (float(truth["tilt_deg"]) >= TILT_JUDGE_MIN_DEG
            and record["tilt_err_deg"] > max_tilt_err_deg):
        problems.append(f"倾角误差 {record['tilt_err_deg']:.1f}°")
    if pose.reproj_rms_px > max_rms_px:
        problems.append(f"RMS {pose.reproj_rms_px:.2f}px")
    record["ok"] = not problems
    record["reason"] = "；".join(problems)
    return record


def write_csv(path: str, rows) -> str:
    """把每行测量结果写成 CSV（真值/估计/误差/原因都在里面）。"""
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


def _median(values) -> float:
    """中位数；空列表返回 nan（报告里显示成 nan 比显示 0 更诚实）。"""
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


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
    """写人看的汇总：分组统计表 + 失败清单 + 三条能直接引用的结论。"""
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
            f"{_median([row.distance_err for row in bucket]) * 100:.2f}% | "
            f"{_median([row.tilt_err_deg for row in bucket]):.2f}° | "
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


def _plot(path: str, series, x_label: str, y_label: str, title: str,
          y_range=None, reference: tuple = None) -> str:
    """画一张单指标折线图（OpenCV 手绘，ASCII 标签）。

    :param path: 输出 PNG
    :param series: `[(名字, [(x, y), ...]), ...]`；点按 x 自动排序
    :param x_label: x 轴标题（ASCII，如 "distance (mm)"）
    :param y_label: y 轴标题（ASCII，如 "detect rate"）
    :param title: 图标题（ASCII）
    :param y_range: `(y_min, y_max)`；None 时按数据自动（y_min 固定 0）
    :param reference: `(y 值, 说明)`，画一条虚线做参照（如"可用门槛 0.8"）
    :return: 输出路径
    """
    width, height = CHART_SIZE
    left, top, right, bottom = CHART_MARGIN
    plot_w = width - left - right
    plot_h = height - top - bottom
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    points_all = [(x, y) for _name, pts in series for x, y in pts if np.isfinite(y)]
    if not points_all:
        put_text(canvas, "no data", (left, top + 30), 0.8, (0, 0, 255), 2)
        cv2.imwrite(path, canvas)
        return path

    xs = [x for x, _y in points_all]
    x_min, x_max = min(xs), max(xs)
    if x_max - x_min < 1e-9:                     # 只有一个档位时给个对称的假范围
        x_min, x_max = x_min - 1.0, x_max + 1.0
    if y_range is None:
        y_max = max(y for _x, y in points_all)
        y_range = (0.0, max(y_max * 1.15, 1e-6))
    y_min, y_max = y_range
    if y_max - y_min < 1e-9:
        y_max = y_min + 1.0

    def to_px(x, y):
        px = left + (x - x_min) / (x_max - x_min) * plot_w
        py = top + plot_h - (y - y_min) / (y_max - y_min) * plot_h
        return int(round(px)), int(round(py))

    # 坐标轴与网格：4 条横线够读数，再多会盖住折线
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (0, 0, 0), 1)
    for i in range(5):
        y_value = y_min + (y_max - y_min) * i / 4.0
        _x, py = to_px(x_min, y_value)
        cv2.line(canvas, (left, py), (left + plot_w, py), (220, 220, 220), 1)
        put_text(canvas, f"{y_value:.2f}", (6, py + 5), 0.45, (0, 0, 0))
        x_value = x_min + (x_max - x_min) * i / 4.0
        px, _y = to_px(x_value, y_min)
        cv2.line(canvas, (px, top + plot_h), (px, top + plot_h + 4), (0, 0, 0), 1)
        # 量级大的轴（距离/亮度×100）取整显示：101.25 这种刻度没人能一眼读出档位
        label = f"{x_value:.0f}" if abs(x_value) >= 10 else f"{x_value:g}"
        put_text(canvas, label, (px - 12, top + plot_h + 22), 0.45, (0, 0, 0))

    if reference is not None:
        ref_value, ref_label = reference
        if y_min <= ref_value <= y_max:
            _x, py = to_px(x_min, ref_value)
            for px in range(left, left + plot_w, 12):     # 手绘虚线：cv2 没有 dash 参数
                cv2.line(canvas, (px, py), (min(px + 6, left + plot_w), py), (0, 128, 255), 1)
            put_text(canvas, ref_label, (left + 6, py - 6), 0.45, (0, 128, 255))

    colors = ((0, 0, 255), (0, 140, 0), (255, 0, 0), (0, 128, 255))
    for index, (name, pts) in enumerate(series):
        clean = sorted((x, y) for x, y in pts if np.isfinite(y))
        if not clean:
            continue
        color = colors[index % len(colors)]
        poly = np.array([to_px(x, y) for x, y in clean], dtype=np.int32)
        if len(poly) >= 2:
            cv2.polylines(canvas, [poly], False, color, 2)
        for px, py in poly:
            cv2.circle(canvas, (int(px), int(py)), 3, color, -1)
        # 只有多条线时才画图例：单指标一张图，图例只会压住折线本身
        if len(series) > 1:
            put_text(canvas, name, (left + plot_w - 90, top + 16 + index * 18), 0.5,
                     color, 1, outline=True)

    put_text(canvas, title, (left, 22), 0.55, (0, 0, 0), 1)
    put_text(canvas, x_label, (left + plot_w // 2 - 40, height - 12), 0.5, (0, 0, 0))
    put_text(canvas, y_label, (6, top - 16), 0.45, (0, 0, 0))
    cv2.imwrite(path, canvas)
    return path


def _median_by_level(rows, group: str, getter) -> list:
    """按档位把某个误差量取中位数，得到折线图要的点。

    :param rows: `list[Row]`
    :param group: 维度名（dist / tilt / bright ...）
    :param getter: 取哪个字段，如 `lambda row: row.distance_err`
    :return: `[(档位, 中位数)]`，按档位升序；未检出样本的误差是 nan，自动被 `_median` 滤掉
    """
    buckets = {}
    for row in rows:
        if row.group == group:
            buckets.setdefault(row.level, []).append(getter(row))
    return sorted((level, _median(values)) for level, values in buckets.items())


def plot_all(rows, out_dir: str) -> list:
    """按"一图一指标"出全部曲线，返回生成的文件路径列表。

    编号与报告里的顺序一致；语义名用 ASCII（文件名会被贴进 PR 和聊天窗口）。
    """
    os.makedirs(out_dir, exist_ok=True)
    groups = {}
    for row in rows:
        groups.setdefault(row.group, []).append(row)

    def points(group, getter):
        return sorted((row.level, getter(row)) for row in groups.get(group, []))

    def rate_point(group):
        """把一组场景按档位折叠成"该档位的检出率"。

        同一档位只有一个场景时它只能是 0 或 1；档位有重复样本时取平均。
        """
        buckets = {}
        for row in groups.get(group, []):
            buckets.setdefault(row.level, []).append(1.0 if row.detected > 0 else 0.0)
        return sorted((level, float(np.mean(values))) for level, values in buckets.items())

    written = []
    written.append(_plot(os.path.join(out_dir, "01_detect_rate_vs_distance.png"),
                         [("detect", rate_point("dist"))], "distance (mm)", "detect rate",
                         "detect rate vs distance", y_range=(0.0, 1.05),
                         reference=(USABLE_DETECT_RATE, f"usable {USABLE_DETECT_RATE:.1f}")))
    written.append(_plot(os.path.join(out_dir, "02_dist_err_vs_distance.png"),
                         [("median err", _median_by_level(rows, "dist",
                                                          lambda row: row.distance_err))],
                         "distance (mm)", "distance error", "distance error vs distance",
                         reference=(0.04, "limit 4%")))
    written.append(_plot(os.path.join(out_dir, "03_detect_rate_vs_tilt.png"),
                         [("detect", rate_point("tilt"))], "tilt (deg)", "detect rate",
                         "detect rate vs tilt", y_range=(0.0, 1.05),
                         reference=(USABLE_DETECT_RATE, f"usable {USABLE_DETECT_RATE:.1f}")))
    written.append(_plot(os.path.join(out_dir, "04_tilt_err_vs_tilt.png"),
                         [("tilt err", points("tilt", lambda row: row.tilt_err_deg))],
                         "tilt (deg)", "tilt error (deg)", "tilt error vs tilt",
                         reference=(3.0, "limit 3 deg")))
    written.append(_plot(os.path.join(out_dir, "05_detect_rate_vs_brightness.png"),
                         [("detect", rate_point("bright"))], "brightness factor", "detect rate",
                         "detect rate vs brightness", y_range=(0.0, 1.05),
                         reference=(USABLE_DETECT_RATE, f"usable {USABLE_DETECT_RATE:.1f}")))
    written.append(_plot(os.path.join(out_dir, "06_detect_rate_vs_blur.png"),
                         [("detect", rate_point("blur"))], "motion blur (px)", "detect rate",
                         "detect rate vs motion blur", y_range=(0.0, 1.05),
                         reference=(USABLE_DETECT_RATE, f"usable {USABLE_DETECT_RATE:.1f}")))
    written.append(_plot(os.path.join(out_dir, "07_detect_rate_vs_noise.png"),
                         [("detect", rate_point("noise"))], "noise sigma", "detect rate",
                         "detect rate vs noise", y_range=(0.0, 1.05),
                         reference=(USABLE_DETECT_RATE, f"usable {USABLE_DETECT_RATE:.1f}")))
    # 每格像素 vs 检出：把"距离太远"翻译成可复现的几何量。
    # **只用 dist / res 两组**：这两组里每格像素的变化只来自"变小"，其它条件不变；
    # 把倾角 75°、过曝这些失败样本也塞进来，曲线会出现"8 px 反而认不出"的假拐点。
    written.append(_plot(os.path.join(out_dir, "08_cell_px_vs_detect_rate.png"),
                         [("detect", _binned_by_cell(rows, groups=("dist", "res")))],
                         "pixels per cell", "detect rate",
                         "detect rate vs pixels per cell (size only)", y_range=(0.0, 1.05),
                         reference=(USABLE_DETECT_RATE, f"usable {USABLE_DETECT_RATE:.1f}")))
    return written


def _binned_by_cell(rows, groups=("dist",)) -> list:
    """按"每格像素"分箱统计检出率（箱宽 2 px），把横轴从"距离"换成几何量。

    :param rows: `list[Row]`
    :param groups: 只统计这些维度的样本（默认只看距离组，理由见调用处）
    :return: `[(每格像素, 检出率)]`，按横轴升序
    """
    buckets = {}
    for row in rows:
        if not np.isfinite(row.cell_px) or row.group not in groups:
            continue
        key = round(row.cell_px / 2.0) * 2.0
        buckets.setdefault(key, []).append(1.0 if row.detected > 0 else 0.0)
    return sorted((key, float(np.mean(values))) for key, values in buckets.items())


def _run_check(args) -> int:
    """跑整个目录并打印汇总；退出码 0 = 全部通过。

    :param args: 已解析的命令行参数
    :return: 0 全过 / 1 有不合格 / 2 参数或环境错误
    """
    images = _iter_images(args.dir)
    if not images:
        print(f"❌ {args.dir} 下没有图片（看看路径或 --dir）")
        return 2

    base_intrinsics = None
    if args.intrinsics:
        from ..config.loader import load_intrinsics
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
        truth = _load_truth(path)
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


def _run_sweep(args) -> int:
    """跑完整张扫描矩阵，落盘 raw.csv / summary.md / 各指标 PNG。

    :param args: 已解析的命令行参数
    :return: 0 = 跑完（失败样本记进报告）；2 = 参数或环境错误
    副作用
        往 `--out` 目录写 CSV、Markdown 与若干 PNG
    """
    os.makedirs(args.out, exist_ok=True)

    camera_matrix, camera_size = None, None
    note = "合成（fx=800@1280 等比缩放）"
    if args.intrinsics:
        try:
            intrinsics = load_intrinsics(args.intrinsics)
        except Exception as exc:
            print(f"❌ 读内参失败：{exc}")
            return 2
        camera_matrix = intrinsics.camera_matrix
        camera_size = tuple(intrinsics.image_size)
        note = f"{args.intrinsics} {camera_size[0]}x{camera_size[1]}"

    specs = build_suite(args.marker_mm)
    print(f"码边长 {args.marker_mm} mm   内参 {note}   场景 {len(specs)} 个")
    rows = run_suite(specs, marker_size_mm=args.marker_mm, method=args.method,
                     camera_matrix=camera_matrix, camera_size=camera_size)

    csv_path = write_csv(os.path.join(args.out, "raw.csv"), rows)
    summary_path = write_summary(os.path.join(args.out, "summary.md"), rows, len(specs),
                                 args.marker_mm, note)
    images = plot_all(rows, args.out)

    passed = sum(1 for row in rows if row.ok)
    detected = sum(1 for row in rows if row.detected > 0)
    print(f"  检出 {detected}/{len(rows)}   通过 {passed}/{len(rows)}")
    print(f"  产物：{csv_path}")
    print(f"        {summary_path}")
    for path in images:
        print(f"        {path}")
    print("✅ 扫描完成（失败样本已写进 summary.md）")
    return 0

# ============================================================================
# 扫描模式（原 tools/pose_sweep.py，已合并进本文件）
#   为什么合并：两个工具共用同一套"通过"门槛与字段名，原先分两个文件、两个入口，
#   改一个门槛要同步两处；现在只有 `pose_check.py` 一个 CLI，模式由 `--sweep` 选。
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    """命令行参数：逐图模式（`--dir`）与扫描模式（`--sweep`）共用一套参数。"""
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
    :return: 逐图模式见 `_run_check`；扫描模式见 `_run_sweep`
    """
    args = build_parser().parse_args(argv)
    if args.sweep:
        if args.marker_mm is None:
            args.marker_mm = DEFAULT_MARKER_MM
        return _run_sweep(args)
    if args.marker_mm is None:
        args.marker_mm = 40.0
    return _run_check(args)


if __name__ == "__main__":
    import sys
    sys.exit(main())
