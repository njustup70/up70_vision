"""评测一个 YOLO 权重在某个数据集划分上的"指标 + 错误分类"。

只报 mAP 不够，因为"两块被框成一块""重复框""误检"在 mAP 里看不出区别。
这个脚本按 IoU 0.5 逐个匹配，额外统计四类错误的**数量**：

    TP / FP / FN     常规检测指标（类别必须一致才算 TP）
    重复框            同一真值被多个预测命中
    一框盖两块          一个预测框里装着 ≥2 个真值框的中心（"合并"）
    类别错            命中真值但 block/top 判反

用法
    PYTHONPATH=. python lab/eval_errors.py --data data/yolo_real_v2 --split test \
        --model knowledge/models/yolov8n_mix2.onnx

    # 换阈值扫描
    PYTHONPATH=. python lab/eval_errors.py --data data/yolo_real_v2 --split test \
        --model runs/detect/mix3/weights/best.pt --conf 0.25 0.45 0.65
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2

from vision.yolo.detector import YoloDetector

IOU_MATCH = 0.5


def load_gt(img_path: str, data_root: str) -> list[tuple[int, float, float, float, float]]:
    """按 ultralytics 布局找标签：images/<split>/x.jpg → labels/<split>/x.txt。"""
    rel = os.path.relpath(img_path, os.path.join(data_root, "images"))
    lab = os.path.join(data_root, "labels", os.path.splitext(rel)[0] + ".txt")
    if not os.path.exists(lab):
        return []
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    out = []
    for line in open(lab, encoding="utf-8"):
        p = line.split()
        if len(p) != 5:
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
        out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def evaluate(model: str, data_root: str, split: str, conf: float) -> dict:
    det = YoloDetector(weights=model, conf=conf)
    stat = dict(tp=0, fp=0, fn=0, dup=0, merge=0, cls=0, images=0, gts=0)
    for path in sorted(glob.glob(os.path.join(data_root, "images", split, "*.jpg"))):
        gts = load_gt(path, data_root)
        stat["images"] += 1
        stat["gts"] += len(gts)
        dets = det.detect(cv2.imread(path))
        used = [False] * len(gts)
        for d in dets:
            box = d.bbox_xyxy
            centers = sum(1 for g in gts
                          if box[0] <= (g[1] + g[3]) / 2 <= box[2]
                          and box[1] <= (g[2] + g[4]) / 2 <= box[3])
            if centers >= 2:
                stat["merge"] += 1
            hits = [(i, iou(box, g[1:])) for i, g in enumerate(gts) if iou(box, g[1:]) >= IOU_MATCH]
            if not hits:
                stat["fp"] += 1
                continue
            bi = max(hits, key=lambda t: t[1])[0]
            if gts[bi][0] != d.class_id:
                stat["cls"] += 1
            if used[bi]:
                stat["dup"] += 1
            else:
                used[bi] = True
                stat["tp"] += 1
        stat["fn"] += used.count(False)
    p = stat["tp"] / max(1, stat["tp"] + stat["fp"])
    r = stat["tp"] / max(1, stat["tp"] + stat["fn"])
    stat.update(precision=p, recall=r, f1=2 * p * r / max(1e-9, p + r))
    return stat


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="指标 + 错误分类")
    ap.add_argument("--data", default="data/yolo_real_v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", required=True)
    ap.add_argument("--conf", type=float, nargs="+", default=[0.45])
    args = ap.parse_args(argv)

    print(f"{'conf':>5} {'图':>4} {'真值':>5} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'P':>6} {'R':>6} {'F1':>6} {'重复':>4} {'合并':>4} {'类别错':>5}")
    for conf in args.conf:
        s = evaluate(args.model, args.data, args.split, conf)
        print(f"{conf:5.2f} {s['images']:4d} {s['gts']:5d} {s['tp']:4d} {s['fp']:4d} {s['fn']:4d} "
              f"{s['precision']:6.3f} {s['recall']:6.3f} {s['f1']:6.3f} "
              f"{s['dup']:4d} {s['merge']:4d} {s['cls']:5d}")
    print("\n注：合并 = 一个预测框里装着 ≥2 个真值框中心；类别必须一致才算 TP。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
