"""对一批图逐张核位姿：每张图把估计值与真值并排打出来，超阈值标 ⚠️。

用大白话说
    `aruco_check.py` 管"码找得对不对"（ID、四角、中心）；这个工具管"算出来的距离和角度
    准不准"。给它一个图片目录，它一张张跑：检测 → 解位姿 → 和同名 JSON 里的真值比，
    把每个码的"真值 / 估计 / 误差"打成一行。有真值的图能算出误差，没真值的图只报自洽性
    （重投影 RMS）——这两种图的结论不能混着看。

放什么 / 不放什么
    放"逐图核对 + 打印 + 退出码"。**不做**批量扫描与画图（那是 `tools/pose_sweep.py`），
    不做渲染（`tools/gen_pose_scenes.py`）。

依赖谁 / 被谁用
    `core.detectors.aruco`（找码）+ `core.pose`（解位姿）+ `config.loader`（读内参 YAML）。
    薄入口是 `scripts/pose_check.py`；`tools/pose_sweep.py` 复用本文件的 `check_one()`。

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
import glob
import json
import os

import numpy as np

from ..core.calibration import scale_intrinsics
from ..core.detectors.aruco import ArucoDetector
from ..core.pose import DEFAULT_METHOD, estimate_pose
from ..utils.image_io import read_image_bgr

#: 判定"通过"的默认门槛。依据见 docs/位姿估计.md 的实测表：
#: 450–600 mm、40 mm 码时距离误差 < 1%、倾角误差 < 2°，留一倍余量。
DEFAULT_MAX_DIST_ERR = 0.04
DEFAULT_MAX_TILT_ERR_DEG = 3.0
DEFAULT_MAX_RMS_PX = 2.0

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
    :return: dict，字段见 `CSV_COLUMNS`；`reason` 非空表示没通过
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


CSV_COLUMNS = ("image", "detected", "id_ok", "distance_mm", "distance_err", "tilt_deg",
               "tilt_err_deg", "yaw_deg", "yaw_err_deg", "reproj_rms_px", "ok", "reason")


def _iter_images(root: str):
    """目录下所有图片，按文件名排序（顺序稳定，输出可 diff）。"""
    paths = []
    for suffix in IMAGE_SUFFIXES:
        paths.extend(glob.glob(os.path.join(root, "*" + suffix)))
    return sorted(set(paths))


def build_parser() -> argparse.ArgumentParser:
    """命令行参数：目录、内参、码边长、门槛。"""
    ap = argparse.ArgumentParser(
        prog="up70-vision-pose-check",
        description="逐张核 ArUco 位姿：估计值与真值并排打印，超阈值标 ⚠️")
    ap.add_argument("--dir", required=True, help="图片目录（同名 .json 为真值）")
    ap.add_argument("--intrinsics", default="",
                    help="内参 YAML；不给则用每张图真值里的 camera_matrix")
    ap.add_argument("--marker-mm", type=float, default=40.0,
                    help="码边长（毫米）；真值里有该字段时以真值为准。默认 40")
    ap.add_argument("--method", default=DEFAULT_METHOD, help=f"求解方法，默认 {DEFAULT_METHOD}")
    ap.add_argument("--max-dist-err", type=float, default=DEFAULT_MAX_DIST_ERR,
                    help=f"距离相对误差上限，默认 {DEFAULT_MAX_DIST_ERR}")
    ap.add_argument("--max-tilt-err", type=float, default=DEFAULT_MAX_TILT_ERR_DEG,
                    help=f"倾角绝对误差上限（度），默认 {DEFAULT_MAX_TILT_ERR_DEG}")
    ap.add_argument("--max-rms", type=float, default=DEFAULT_MAX_RMS_PX,
                    help=f"重投影 RMS 上限（像素），默认 {DEFAULT_MAX_RMS_PX}")
    return ap


def main(argv=None) -> int:
    """跑整个目录并打印汇总；退出码 0 = 全部通过。

    :param argv: 参数列表；None 表示取 `sys.argv[1:]`
    :return: 0 全过 / 1 有不合格 / 2 参数或环境错误
    """
    args = build_parser().parse_args(argv)
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

    records, size_warned = [], False
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
