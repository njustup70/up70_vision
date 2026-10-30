"""用现有模型给一批图预标注（YOLO txt），人工只需改错，不用从零画框。

为什么值得
    手工标 100 张约 1–2 小时；模型预标注后人工核对通常 10–15 分钟。
    错的框（漏检/合并/误检）在标注工具里右键删、重拖即可。

用法
    PYTHONPATH=. python lab/prelabel_dataset.py --images data/yolo_real/g6_far \
        --labels data/yolo_real/g6_far_labels

    # 注意：--labels 一定要给**独立目录**，别和 --images 相同——
    # 写进图片目录会让 jpg 和 txt 混在一起，之后建 images/labels 结构时容易漏挪。

输出
    <labels>/<同名>.txt，每行 `cls cx cy w h`（归一化），class 0=block 1=top
    同名 `.names.json` 一致：{"0": "block", "1": "top"}
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2

from vision.yolo.detector import YoloDetector


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="模型预标注 → YOLO txt")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model", default="knowledge/models/yolov8n_mix2.onnx")
    ap.add_argument("--conf", type=float, default=0.35, help="预标注放宽一点，宁可多框")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                   glob.glob(os.path.join(args.images, "*.png")))
    if not files:
        print(f"❌ {args.images} 下没有图片")
        return 1
    if os.path.abspath(args.labels) == os.path.abspath(args.images):
        print("❌ --labels 不能和 --images 是同一个目录（txt 会和图片混在一起）", file=sys.stderr)
        return 2
    os.makedirs(args.labels, exist_ok=True)
    det = YoloDetector(weights=args.model, conf=args.conf)

    n_box, empty = 0, 0
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        dets = det.detect(img)
        stem = os.path.splitext(os.path.basename(path))[0]
        lines = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox_xyxy
            lines.append(f"{d.class_id} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                         f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        with open(os.path.join(args.labels, stem + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        n_box += len(lines)
        empty += 0 if lines else 1

    print(f"  预标注 {len(files)} 张：共 {n_box} 个框，其中 {empty} 张没框（需人工检查/补）")
    print(f"  标签写到 {args.labels}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
