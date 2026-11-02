"""按"连拍串"划分 train/val/test，避免近重复帧跨集导致 mAP 虚高。

为什么不能随机切
    相机连拍时相邻帧几乎一样（实测新采集的 128 张里，30 张能找到相似度 >0.95 的"双胞胎"）。
    随机按 7:1.5:1.5 切，val/test 里大量帧在 train 有近重复 → mAP 会虚高，
    上真机才发现"怎么不灵了"。

做法
    1. 用 32×32 灰度图的余弦相似度把帧聚成"场景串"（相似度 > --sim 视为同一串）；
    2. **整串**分配到 train/val/test（串内不拆开）；
    3. 按来源分组（目录名）分层：每组各自出 val/test，避免某一组完全缺席；
    4. 输出 ultralytics 结构 + dataset.yaml，并打印"最大跨集相似度"用于自检。

用法
    # 合成一份多来源清单（例如 原有 120 张 + 新采 128 张）
    PYTHONPATH=. python lab/split_dataset.py \
        --source data/yolo_real2/images/train --labels data/yolo_real2/labels/train --name old \
        --source data/yolo_real/g6_far            --labels data/yolo_real/g6_far_labels --name new \
        --out data/yolo_real_v2 --sim 0.90 --val-ratio 0.15 --test-ratio 0.15
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np


def feat(path: str, size: int = 32) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return np.zeros(size * size, np.float32)
    g = cv2.resize(g, (size, size)).astype(np.float32)
    g -= g.mean()
    n = np.linalg.norm(g)
    return (g / (n if n > 1e-6 else 1)).ravel()


def cluster(feats: np.ndarray, sim: float) -> list[list[int]]:
    """贪心聚类：与已有簇中心相似度 > sim 就并入。"""
    centers, groups = [], []
    for i, f in enumerate(feats):
        placed = False
        for k, c in enumerate(centers):
            if float(np.dot(f, c)) / (np.linalg.norm(c) + 1e-9) > sim:
                groups[k].append(i)
                centers[k] = centers[k] + f          # 增量更新中心
                placed = True
                break
        if not placed:
            centers.append(f.copy())
            groups.append([i])
    return groups


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="按连拍串划分数据集")
    ap.add_argument("--source", action="append", required=True,
                    help="图片目录（可重复；与 --labels 一一对应）")
    ap.add_argument("--labels", action="append", required=True,
                    help="标签目录（与 --source 顺序对应）")
    ap.add_argument("--name", action="append", default=None, help="来源短名（可重复，可省）")
    ap.add_argument("--out", default="data/yolo_real_v2")
    ap.add_argument("--sim", type=float, default=0.90, help="相似度阈值，>它算同一条连拍")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args(argv)

    if len(args.source) != len(args.labels):
        print("❌ --source 与 --labels 数量必须一致", file=sys.stderr)
        return 2
    names = args.name or [f"src{i}" for i in range(len(args.source))]

    rng = np.random.default_rng(args.seed)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    total = {"train": 0, "val": 0, "test": 0}
    report = []
    for src, lab, name in zip(args.source, args.labels, names):
        files = sorted(glob.glob(os.path.join(src, "*.jpg")))
        files = [f for f in files if os.path.exists(
            os.path.join(lab, os.path.splitext(os.path.basename(f))[0] + ".txt"))]
        if not files:
            print(f"  跳过 {name}（{src} 没有配对的 jpg+txt）")
            continue
        groups = cluster(np.stack([feat(f) for f in files]), args.sim)
        order = rng.permutation(len(groups))
        groups = [groups[i] for i in order]
        n = len(files)
        n_val = max(1, int(round(n * args.val_ratio)))
        n_test = max(1, int(round(n * args.test_ratio)))
        take = {"test": 0, "val": 0, "train": 0}
        for g in groups:
            if take["test"] < n_test:
                split = "test"
            elif take["val"] < n_val:
                split = "val"
            else:
                split = "train"
            take[split] += len(g)
            for idx in g:
                f = files[idx]
                stem = f"{name}_{os.path.splitext(os.path.basename(f))[0]}"
                shutil.copyfile(f, os.path.join(args.out, "images", split, stem + ".jpg"))
                shutil.copyfile(os.path.join(lab, os.path.splitext(os.path.basename(f))[0] + ".txt"),
                                os.path.join(args.out, "labels", split, stem + ".txt"))
                total[split] += 1
        report.append((name, n, len(groups), take))

    with open(os.path.join(args.out, "dataset.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\n"
                 f"val: images/val\ntest: images/test\nnames: [block, top]\n")

    print(f"  输出 {args.out}/：{total}")
    for name, n, ng, take in report:
        print(f"    {name}: {n} 张 / {ng} 条连拍串 → train {take['train']} val {take['val']} test {take['test']}")

    # 自检：val/test 与 train 的最大相似度（越低越干净）
    def load(split):
        return {p: feat(p) for p in glob.glob(os.path.join(args.out, "images", split, "*.jpg"))}
    tr = load("train")
    for split in ("val", "test"):
        other = load(split)
        if not other or not tr:
            continue
        mx = [max(float(np.dot(f, t)) for t in tr.values()) for f in other.values()]
        print(f"    {split} vs train 最大相似度：中位 {np.median(mx):.3f}  最大 {max(mx):.3f}"
              f"  >0.95 的帧 {(np.array(mx) > 0.95).sum()} 张")
    return 0


if __name__ == "__main__":
    sys.exit(main())
