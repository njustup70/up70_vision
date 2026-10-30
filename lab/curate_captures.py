"""从采集到的原始帧里自动挑出"有物块"的样本，丢掉只有环境的废帧。

为什么需要
    `lab/capture_dataset.py` 是"边摆边拍"，人一走动、镜头一晃就会存进一堆天花板、
    空地面、笔记本。手工翻 200 张很费时，而**判据其实很硬**：

      1. YOLO 至少检出 min-det 个 `block`/`top`（默认 1 个）
      2. 亮度在 [min-mean, max-mean]、过曝比例 < max-sat（剔糊/过曝帧）
      3. 检出的框不过小（< min-box-px，太小的远景帧标注价值低）

    满足就拷到目标目录，不满足就丢（可选 --delete 直接从源目录删）。

用法
    # 先看统计，不动文件
    PYTHONPATH=. python lab/curate_captures.py --dir output/grab --tag g6_far --dry-run

    # 挑出有物块的帧，归档到数据目录
    PYTHONPATH=. python lab/curate_captures.py --dir output/grab --tag g6_far \
        --out data/yolo_real/g6_far

    # 直接把废帧删掉（省磁盘）
    PYTHONPATH=. python lab/curate_captures.py --dir output/grab --tag g6_far \
        --out data/yolo_real/g6_far --delete
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import cv2

from vision.yolo.detector import YoloDetector


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="从采集帧里挑出有物块的样本")
    ap.add_argument("--dir", default="output/grab", help="采集帧所在目录")
    ap.add_argument("--tag", default="", help="只处理这个前缀（空=全部 jpg）")
    ap.add_argument("--out", default="", help="保留下来的图拷到哪（空=只统计不拷）")
    ap.add_argument("--model", default="knowledge/models/yolov8n_mix2.onnx")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--min-det", type=int, default=1, help="至少检出几个目标才算有用")
    ap.add_argument("--min-box-px", type=int, default=24, help="框最短边小于它就不算（远景糊了）")
    ap.add_argument("--min-mean", type=float, default=55.0)
    ap.add_argument("--max-mean", type=float, default=190.0)
    ap.add_argument("--max-sat", type=float, default=0.30, help="过曝像素占比上限")
    ap.add_argument("--delete", action="store_true", help="把不合格的从源目录删掉")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计")
    args = ap.parse_args(argv)

    pattern = os.path.join(args.dir, (args.tag + "_*" if args.tag else "*") + ".jpg")
    files = sorted(f for f in glob.glob(pattern) if not f.endswith("latest.jpg"))
    if not files:
        print(f"❌ {pattern} 没有图片")
        return 1

    det = YoloDetector(weights=args.model, conf=args.conf)
    if args.out and not args.dry_run:
        os.makedirs(args.out, exist_ok=True)

    keep, drop = [], {"暗": 0, "过曝": 0, "无目标": 0, "目标太小": 0}
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean, sat = float(gray.mean()), float((gray >= 250).mean())
        if mean < args.min_mean:
            drop["暗"] += 1
            continue
        if mean > args.max_mean or sat > args.max_sat:
            drop["过曝"] += 1
            continue
        dets = det.detect(img)
        if len(dets) < args.min_det:
            drop["无目标"] += 1
            continue
        side = min(min(d.width, d.height) for d in dets)
        if side < args.min_box_px:
            drop["目标太小"] += 1
            continue
        keep.append(path)
        if args.out and not args.dry_run:
            idx = len(keep) - 1
            shutil.copyfile(path, os.path.join(args.out, f"frame_{idx:04d}.jpg"))

    if args.delete and not args.dry_run:
        for path in files:
            if path not in keep:
                os.remove(path)

    total = len(files)
    print(f"  扫描 {total} 张 → 保留 {len(keep)} 张（{100*len(keep)/max(1,total):.0f}%）")
    for k, v in drop.items():
        if v:
            print(f"    丢弃·{k}: {v} 张")
    if args.out and not args.dry_run:
        print(f"  已拷到 {args.out}/（文件名 frame_0000.jpg …）")
    if args.delete and not args.dry_run:
        print("  已删除源目录里的废帧")
    return 0


if __name__ == "__main__":
    sys.exit(main())
