"""CPU 与 GPU 推理的耗时 + 一致性对比。

为什么需要它
    换 `device=cuda` 之后要回答两个问题：
        1. 快了多少（推理 vs 整帧管线，别只看模型）；
        2. 结果还是不是同一个（同一张图，bbox/score 必须落在容差内）。
    两个都满足才敢把比赛默认值改成 cuda；只看"快"不看"对"，等于埋雷。

用法
    # 本机默认：仓库里的 test 图，CPU 与 CUDA 各跑一遍
    ~/.venvs/up70-2d/bin/python lab/bench_yolo.py

    # 只测 CPU（没装驱动时）
    ~/.venvs/up70-2d/bin/python lab/bench_yolo.py --devices cpu

    # 指定模型与张数
    ~/.venvs/up70-2d/bin/python lab/bench_yolo.py \
        --weights knowledge/models/yolov8n_v4.onnx --limit 20 --devices cpu,cuda

输出
    每个设备一段：首帧（含加载）、稳态中位/90 分位、整帧管线 fps；
    若同时测了两个设备，末尾给一致性对比（bbox ±1 px、score ±0.02）。

退出码
    0 = 全部设备都跑通且一致性通过；1 = 某设备失败或结果不一致。
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import time

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DEFAULT_IMAGES = "data/yolo_real_v4/images/test"
BBOX_TOL_PX = 1
SCORE_TOL = 0.02


def load_images(pattern: str, limit: int) -> list[tuple[str, np.ndarray]]:
    """按目录或通配符取图，最多 limit 张。"""
    paths = sorted(glob.glob(os.path.join(pattern, "*.jpg")) or glob.glob(pattern))
    if not paths:
        raise SystemExit(f"❌ {pattern} 下没有 jpg")
    out = []
    for path in paths[:limit]:
        image = cv2.imread(path)
        if image is None:
            raise SystemExit(f"❌ 读不出 {path}")
        out.append((os.path.basename(path), image))
    return out


def bench(device: str, images, params: dict) -> dict:
    """在一个设备上跑一遍，返回耗时与每张图的 detections。"""
    from vision.cli.run import build_detector
    from vision.types import Frame
    from vision.yolo.pipeline import process_frame

    detector = build_detector("yolo", dict(params, device=device))
    if getattr(detector, "device", device) != device:
        raise SystemExit(f"❌ 设备没生效：请求 {device}，实际 {getattr(detector, 'device', '?')}")

    t0 = time.perf_counter()
    detector.detect(images[0][1])                        # 首帧含 ONNX 加载 / CUDA 初始化
    warm_ms = (time.perf_counter() - t0) * 1000

    det_ms, full_ms, results = [], [], {}
    for name, image in images:
        start = time.perf_counter()
        dets = detector.detect(image)
        det_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        process_frame(Frame(image, 0, f"bench:{device}"), detector, {}, None)
        full_ms.append((time.perf_counter() - start) * 1000)
        results[name] = [d.to_dict() for d in dets]

    def stats(values):
        return (statistics.median(values),
                sorted(values)[int(len(values) * 0.9) - 1] if len(values) > 1 else values[0])

    det_med, det_p90 = stats(det_ms)
    full_med, _ = stats(full_ms)
    print(f"  {device:5s} 首帧 {warm_ms:7.1f} ms | 推理 中位 {det_med:6.1f} / P90 {det_p90:6.1f} ms "
          f"| 整帧 {full_med:6.1f} ms → {1000 / full_med:.0f} fps")
    return {"device": device, "results": results,
            "det_med": det_med, "full_med": full_med, "warm_ms": warm_ms}


def compare(ref: dict, other: dict) -> int:
    """逐图逐框对比；返回不一致的条数。"""
    bad = 0
    for name, want in ref["results"].items():
        got = other["results"].get(name, [])
        if len(want) != len(got):
            print(f"  ⚠️ {name}: 框数不同 {len(want)} → {len(got)}")
            bad += 1
            continue
        for a, b in zip(want, got):
            if a["class_name"] != b["class_name"] or a["color"] != b["color"]:
                print(f"  ⚠️ {name}: class/color {a['class_name']}/{a['color']} → "
                      f"{b['class_name']}/{b['color']}")
                bad += 1
            elif any(abs(x - y) > BBOX_TOL_PX for x, y in zip(a["bbox_xyxy"], b["bbox_xyxy"])):
                print(f"  ⚠️ {name}: bbox {a['bbox_xyxy']} → {b['bbox_xyxy']}")
                bad += 1
            elif abs(a["score"] - b["score"]) > SCORE_TOL:
                print(f"  ⚠️ {name}: score {a['score']} → {b['score']}")
                bad += 1
    return bad


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CPU / GPU 推理耗时与一致性对比")
    parser.add_argument("--weights", default="knowledge/models/yolov8n_mix4.onnx")
    parser.add_argument("--images", default=DEFAULT_IMAGES, help="图片目录或通配符")
    parser.add_argument("--limit", type=int, default=15, help="最多测几张（默认 15）")
    parser.add_argument("--devices", default="cpu,cuda", help="逗号分隔：cpu,cuda")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args(argv)

    os.chdir(REPO)                                        # 权重与配置按仓库根解析
    images = load_images(args.images, args.limit)
    params = {"weights": args.weights, "conf": args.conf, "iou": 0.45,
              "imgsz": args.imgsz, "max_detections": 32}
    print(f"模型 {args.weights} | {len(images)} 张 | imgz {args.imgsz} | conf {args.conf}")

    reports, failed = {}, 0
    for device in [d.strip() for d in args.devices.split(",") if d.strip()]:
        try:
            reports[device] = bench(device, images, params)
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            print(f"  {device:5s} ❌ {exc}")
            failed += 1

    if "cpu" in reports and "cuda" in reports:
        print("\n一致性（以 cpu 为基准，bbox ±%d px、score ±%.2f）：" % (BBOX_TOL_PX, SCORE_TOL))
        bad = compare(reports["cpu"], reports["cuda"])
        speedup = reports["cpu"]["det_med"] / reports["cuda"]["det_med"]
        print(f"  推理加速 {speedup:.1f}×（{reports['cpu']['det_med']:.1f} → "
              f"{reports['cuda']['det_med']:.1f} ms）")
        if bad:
            print(f"  ❌ {bad} 处不一致")
            return 1
        print("  ✅ 全部在容差内")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
