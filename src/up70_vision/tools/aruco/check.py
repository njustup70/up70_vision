#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArUco 识别自检：拿 assets/aruco/ 的图组对真值，一张一张报数字。

它回答的问题只有一个
    任务单验收清单第一条：**"同一张图片中的多个 ArUco 码全部正确识别（ID、四角、中心）"**。
    所以每一张图都打印"期望几个 / 实得几个 / id 集合 / 中心最大误差 / 四角最大误差"，
    四个都对才打 ✅。误差是实测值，不是"看起来对"。

四角误差为什么能查出顺序错
    比对是**按下标一一对应**的（真值第 k 个角 vs 实测第 k 个角）。契约要求四角顺序
    是 TL→TR→BR→BL，顺序一拧，误差立刻从 <1px 跳到几百 px，不用另写一条断言。

用法（在仓库根目录）
    python -m up70_vision.tools.aruco.check                      # 默认查 assets/aruco
    python -m up70_vision.tools.aruco.check --only multi_grid_6  # 只查一张
    python -m up70_vision.tools.aruco.check --max-err-px 1.5     # 调容差（默认 1.5）

退出码：0 = 全部通过；1 = 有图对不上；2 = 环境/参数不对（如没生成图组、cv2 太老）
"""

import argparse
import os
import sys

import numpy as np

from ...detectors.aruco import ArucoDetector, check_opencv_aruco
from ...image_io import read_image_bgr
from ...logging import get_logger, setup_logging
from .gen_scenes import DEFAULT_OUT_DIR, iter_scene_images, load_truth

log = get_logger("tools.aruco.check")

DEFAULT_MAX_ERR_PX = 1.5


def _match_by_id(markers: list, truth_markers: list):
    """把实测结果与真值按 id + 最近中心配成对，返回 [(truth, det)]。

    同一张图里正常不会出现重复 id；真出现时按中心距离贪婪配，不在这里报错——
    重复 ID 本身是图的问题，由 id 集合检查那一关去说。
    """
    remaining = list(markers)
    pairs = []
    for truth in truth_markers:
        wanted = int(truth["id"])
        cands = [m for m in remaining if int(m.marker_id) == wanted]   # ArucoMarker.marker_id
        if not cands:
            pairs.append((truth, None))
            continue
        center = np.asarray(truth["center"], dtype=np.float64)
        best = min(cands, key=lambda m: float(np.linalg.norm(
            np.asarray(m.center_px, dtype=np.float64) - center)))
        remaining.remove(best)
        pairs.append((truth, best))
    return pairs


def check_one(image_path: str, max_err_px: float = DEFAULT_MAX_ERR_PX) -> dict:
    """查一张图。

    :return: dict(name, ok, expected, got, ids_ok, center_err, corner_err, note)
    """
    truth = load_truth(image_path)
    image_bgr = read_image_bgr(image_path)
    detector = ArucoDetector(dictionary=truth.get("dictionary", "DICT_4X4_50"))
    markers = detector.detect(image_bgr)                 # list[ArucoMarker]（契约 3.2）

    expected_ids = sorted(int(m["id"]) for m in truth["markers"])
    got_ids = sorted(int(m.marker_id) for m in markers)
    ids_ok = expected_ids == got_ids

    pairs = _match_by_id(markers, truth["markers"])
    center_err = 0.0
    corner_err = 0.0
    for one_truth, marker in pairs:
        if marker is None:
            continue
        center_err = max(center_err, float(np.linalg.norm(
            np.asarray(marker.center_px, dtype=np.float64)
            - np.asarray(one_truth["center"], dtype=np.float64))))
        want = np.asarray(one_truth["corners"], dtype=np.float64)
        have = np.asarray(marker.corners_px, dtype=np.float64)
        corner_err = max(corner_err, float(np.max(np.linalg.norm(want - have, axis=1))))

    ok = bool(ids_ok and center_err <= max_err_px and corner_err <= max_err_px)
    note = ""
    if not ids_ok:
        note = f"id 对不上：期望 {expected_ids}，实得 {got_ids}"
    elif not ok:
        note = f"误差超容差 {max_err_px}px"
    return {"name": os.path.basename(image_path),
            "ok": ok,
            "expected": len(truth["markers"]),
            "got": len(markers),
            "ids_ok": ids_ok,
            "center_err": center_err,
            "corner_err": corner_err,
            "note": note}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="aruco_check",
        description="拿 ArUco 测试图组对真值做自检（ID、四角、中心）")
    ap.add_argument("--dir", default=DEFAULT_OUT_DIR, help=f"图组目录（默认 {DEFAULT_OUT_DIR}）")
    ap.add_argument("--only", default=None, help="只查名字里含这个串的图，如 multi")
    ap.add_argument("--max-err-px", type=float, default=DEFAULT_MAX_ERR_PX,
                    help=f"中心/四角允许的最大误差（像素，默认 {DEFAULT_MAX_ERR_PX}）")
    ap.add_argument("--log-level", default="WARNING", help="日志等级，默认 WARNING")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if str(args.dir) == DEFAULT_OUT_DIR and not os.path.isdir(args.dir):
        # 允许在仓库任意子目录里调用：从本文件往上找仓库根（有 assets/ 的那层）
        here = os.path.dirname(os.path.abspath(__file__))
        for _ in range(6):
            here = os.path.dirname(here)
            candidate = os.path.join(here, DEFAULT_OUT_DIR)
            if os.path.isdir(candidate):
                args.dir = candidate
                break

    try:
        check_opencv_aruco()
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 2

    if not os.path.isdir(args.dir):
        print(f"❌ 图组目录不存在：{args.dir}")
        print("   先生成：python -m up70_vision.tools.aruco.gen_scenes")
        return 2

    images = iter_scene_images(args.dir)
    if args.only:
        images = [p for p in images if args.only in os.path.basename(p)]
    if not images:
        print(f"❌ {args.dir} 下没有可查的图（看看 --only 是不是写错了）")
        return 2
    missing_truth = [p for p in images if not os.path.isfile(os.path.splitext(p)[0] + ".json")]

    print(f"图组：{args.dir}   共 {len(images)} 张   容差：{args.max_err_px} px")
    print("-" * 96)
    results = []
    for path in images:
        try:
            res = check_one(path, max_err_px=args.max_err_px)
        except (ValueError, RuntimeError, FileNotFoundError, IOError) as exc:
            print(f"  {os.path.basename(path):<22} ❌ {type(exc).__name__}: {exc}")
            results.append({"name": os.path.basename(path), "ok": False, "expected": 0,
                            "got": 0, "ids_ok": False, "center_err": float("inf"),
                            "corner_err": float("inf"), "note": str(exc)})
            continue
        results.append(res)
        flag = "✅" if res["ok"] else "❌"
        ids_flag = "✅" if res["ids_ok"] else "❌"
        print(f"  {res['name']:<22} 期望 {res['expected']} 个 / 实得 {res['got']} 个  "
              f"ids {ids_flag}  中心最大误差 {res['center_err']:5.2f}px  "
              f"四角最大误差 {res['corner_err']:5.2f}px  {flag} {res['note']}")

    passed = sum(1 for r in results if r["ok"])
    worst_center = max(r["center_err"] for r in results)
    worst_corner = max(r["corner_err"] for r in results)
    print("-" * 96)
    print(f"  通过 {passed}/{len(results)}   最大中心误差 {worst_center:.2f}px   "
          f"最大四角误差 {worst_corner:.2f}px")
    if missing_truth:
        print(f"  ⚠️ {len(missing_truth)} 张图没有真值 JSON，测不出对错（重新生成图组即可）")
    print(f"\n{'✅ 自检通过' if passed == len(results) and not missing_truth else '❌ 自检未通过'}")
    print("   这条命令可以直接贴进 PR 描述（协作规范第 4 条要求具体数字）")
    return 0 if passed == len(results) and not missing_truth else 1


if __name__ == "__main__":
    sys.exit(main())
