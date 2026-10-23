"""批量扫描：造一张"距离 x 倾角 x 光照 x 噪声 x 模糊 x 分辨率"的矩阵，逐格测检出率与误差。

用大白话说
    不满足于"这张图准不准"，而是问"到多远开始认不出、斜多少度开始认错、光多暗开始丢"。
    做法是让 `gen_pose_scenes` 按预设档位造一整批图，逐张检测 + 解位姿 + 对真值，
    最后落盘 CSV、Markdown 报告和 8 张单指标曲线。

放什么 / 不放什么
    放 `run_suite()`（跑整张矩阵）、`_check_rendered()`（对内存里已渲染的图核对）、
    `plot_all()` 与各张曲线的取点逻辑、`_run_sweep()`（本模式的驱动）。
    **不放**判定门槛（在 `check.py`，两种模式共用一份）、不放 CSV/MD 排版（在 `report.py`）。

依赖谁 / 被谁用
    `gen_pose_scenes`（造图）+ `detectors.aruco`（找码）+ `pose`（解位姿）
    + `calibration`（内参缩放）+ `viz.draw.put_text`（画曲线）。
    只被 `cli.py` 的 `main()` 在 `--sweep` 时调用。

对外约定
    `run_suite()` 返回 `list[Row]`，顺序与 `specs` 一致；`_run_sweep()` 返回
    0（跑完，失败样本记进报告）/ 2（内参读不出来等环境错误）。
    **扫描失败不返回 1**：这是"测量"不是"验收"，量出来哪些档位不行本身就是产出。

已知的坑
    · 内参必须与画布尺寸匹配：1280 的内参套到 640 画布上等于换了一颗长焦镜头，
      "每格像素"这条结论会全错（progress.md 坑 3 的扫描版）。`_run_sweep()` 自动等比缩放。
    · 每格像素按 6 算不是 4：`DICT_4X4_50` 的数据区是 4 格，但外面还有一圈黑边框（见 `_cell_px()`）。
"""

import os

import cv2
import numpy as np

from vision.config.loader import load_intrinsics
from vision.calibration import scale_camera_matrix
from vision.detectors.aruco import ArucoDetector
from vision.pose import estimate_pose
from vision.viz.draw import put_text
from .gen_scenes import (DEFAULT_MARKER_MM, build_suite, render_scene,
                               synthetic_intrinsics)
from .check import TILT_JUDGE_MIN_DEG
from .report import Row, group_and_level, median_value, write_csv, write_summary

#: 图幅与边距（像素）。宽一点是为了 x 轴刻度标签不重叠。
CHART_SIZE = (820, 480)
CHART_MARGIN = (90, 56, 40, 60)      # 左、上、右、下；上边距要放得下标题 + 最高刻度

#: 判"这一段还能用"的检出率门槛。低于它有 1/5 的场景认不出，实际场上不可接受。
USABLE_DETECT_RATE = 0.8


def _cell_px(corners_px) -> float:
    """码在图像里"每格"多少像素：四边平均长 / 6。

    为什么是 6 而不是 4：`DICT_4X4_50` 的数据区是 4 格，但码本身还有一圈黑边框，
    检测器看到的黑框是 6 格宽（4 数据 + 每边 1 格边框）。教程第三节给的硬下限
    "每格 5–10 px"说的就是这 6 格里的每一格。

    :param corners_px: (4,2) 四角像素坐标（任意顺序，按边长平均不受顺序影响）
    :return: 每格像素数
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
    :param max_dist_err: 距离相对误差上限（比例），与 `check.py` 的门槛同名同义
    :param max_tilt_err_deg: 倾角绝对误差上限（度）
    :param max_rms_px: 重投影 RMS 上限（像素）
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
        group, level = group_and_level(spec.name)

        # 用与逐图核对同名的字段与同一套门槛（见 _check_rendered 的说明），
        # 保证"raw.csv 里通过"与"逐图核对里通过"是同一个意思
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
    """对"已经渲染在内存里"的一张图做和 `check.check_one` 相同的核对。

    为什么不用 `check_one`：那个函数从磁盘读图、自己找同名 JSON；这里的图还没落盘。
    两边的门槛与字段保持一一对应，`raw.csv` 的列也对齐，改动时两边一起改。

    :return: 与 `check.check_one` 同构的 dict（少 `image` / `id_ok` 两个只在逐图模式用的键）
    """
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
    # 与逐图核对同一条规矩：接近正视时不判倾角（那个自由度弱可观），误差照记
    if (float(truth["tilt_deg"]) >= TILT_JUDGE_MIN_DEG
            and record["tilt_err_deg"] > max_tilt_err_deg):
        problems.append(f"倾角误差 {record['tilt_err_deg']:.1f}°")
    if pose.reproj_rms_px > max_rms_px:
        problems.append(f"RMS {pose.reproj_rms_px:.2f}px")
    record["ok"] = not problems
    record["reason"] = "；".join(problems)
    return record


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
    :return: `[(档位, 中位数)]`，按档位升序；未检出样本的误差是 nan，自动被滤掉
    """
    buckets = {}
    for row in rows:
        if row.group == group:
            buckets.setdefault(row.level, []).append(getter(row))
    return sorted((level, median_value(values)) for level, values in buckets.items())


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


def plot_all(rows, out_dir: str) -> list:
    """按"一图一指标"出全部曲线，返回生成的文件路径列表。

    :param rows: `list[Row]`
    :param out_dir: 输出目录（自动创建）
    :return: 8 个 PNG 的路径列表

    编号与报告里的顺序一致；语义名用 ASCII（文件名会被贴进 PR 和聊天窗口）。
    **一图只放一个指标**：两张分开看还有意义的曲线不许塞进同一张图。
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


def run_sweep(args) -> int:
    """跑完整张扫描矩阵，落盘 raw.csv / summary.md / 各指标 PNG。

    :param args: 已解析的命令行参数（见 `cli.build_parser()`）
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
        camera_size = tuple(int(intrinsics.image_size))
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
