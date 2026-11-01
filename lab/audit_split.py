"""审计数据集划分：train/val/test 之间还有没有近重复帧。

为什么需要它
    `split_dataset.py` 按"连拍串"切分，但串是贪心聚出来的，**不保证传递**：
    A 像 B、B 像 C，不代表 A 像 C。实际数据里就出现过
    `g6far_frame_0120`（train）与 `g6far_frame_0121`（test）只差 1 帧、相似度 0.97
    却跨了集——这种帧会让 test 指标虚高，且换场景后完全不可信。

这个脚本查三件事
    1. **跨集最大相似度**：每张 val/test 图在 train 里最像的一张有多像；
    2. **相邻帧漏切**：同前缀、帧号相差 ≤ --frame-gap、相似度 ≥ --sim 的对，逐对列出；
    3. **来源分布**：每个采集组（文件名前缀）在三个划分里各有多少张。

用法
    PYTHONPATH=. python lab/audit_split.py --data data/yolo_real_v2
    PYTHONPATH=. python lab/audit_split.py --data data/yolo_real_v2 --frame-gap 10 --sim 0.85

退出码：0 = 没发现硬泄漏；1 = 有相邻帧跨集或最大相似度 ≥ --fail-sim（默认 0.95）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

SPLITS = ("train", "val", "test")


def feat(path: str, size: int = 32) -> np.ndarray:
    """32×32 灰度图 -> 去均值、L2 归一化的向量（与 split_dataset.py 同一口径）。"""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return np.zeros(size * size, dtype=np.float32)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray -= gray.mean()
    norm = float(np.linalg.norm(gray))
    return (gray / norm if norm > 1e-6 else gray).ravel()


def frame_number(path: str) -> int | None:
    """从 `g6far_frame_0121.jpg` 这类名字里取最后的帧号；取不到返回 None。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else None


def source_name(path: str) -> str:
    """采集组名：`g6far_frame_0121` -> `g6far`；没有 `_frame_` 就用整段 stem。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_frame_")[0] if "_frame_" in stem else stem


def load_split(data: str, split: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data, "images", split, "*.jpg")))


def audit(data: str, sim: float, frame_gap: int, fail_sim: float, size: int) -> dict:
    """返回审计结果字典（同时用于打印与 --json）。"""
    files = {s: load_split(data, s) for s in SPLITS}
    missing = [s for s, items in files.items() if not items]
    if missing:
        raise FileNotFoundError(f"缺少划分：{missing}（{data}/images/<split>）")

    feats = {s: np.stack([feat(p, size) for p in files[s]]) for s in SPLITS}
    result = {"data": os.path.abspath(data), "size": size, "frame_gap": frame_gap,
              "sim": sim, "fail_sim": fail_sim, "splits": {}, "cross": {}, "adjacent": []}

    for split in SPLITS:
        result["splits"][split] = {
            "images": len(files[split]),
            "sources": {},
        }
        counts = result["splits"][split]["sources"]
        for path in files[split]:
            name = source_name(path)
            counts[name] = counts.get(name, 0) + 1

    for split in ("val", "test"):
        matrix = feats["train"] @ feats[split].T               # (train, split)
        best = matrix.max(axis=0)
        best_idx = matrix.argmax(axis=0)
        result["cross"][split] = {
            "max": round(float(best.max()), 4),
            "median": round(float(np.median(best)), 4),
            "over_sim": int((best >= sim).sum()),
            "over_095": int((best >= 0.95).sum()),
            "worst": [],
        }
        for i in np.argsort(best)[-5:][::-1]:
            result["cross"][split]["worst"].append({
                "image": os.path.basename(files[split][int(i)]),
                "train_match": os.path.basename(files["train"][int(best_idx[i])]),
                "similarity": round(float(best[i]), 4),
            })

    for split in ("val", "test"):
        for i, path in enumerate(files[split]):
            digits = frame_number(path)
            if digits is None:
                continue
            best = (feats["train"] @ feats[split][i])
            for j, train_path in enumerate(files["train"]):
                if source_name(train_path) != source_name(path):
                    continue
                other = frame_number(train_path)
                if other is None or abs(digits - other) > frame_gap:
                    continue
                if float(best[j]) >= sim:
                    result["adjacent"].append({
                        "split": split,
                        "image": os.path.basename(path),
                        "train_match": os.path.basename(train_path),
                        "frame_delta": digits - other,
                        "similarity": round(float(best[j]), 4),
                    })

    result["adjacent"].sort(key=lambda row: row["similarity"], reverse=True)
    result["leaky"] = bool(result["adjacent"] or any(
        info["max"] >= fail_sim for info in result["cross"].values()))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="审计 train/val/test 的近重复帧泄漏")
    parser.add_argument("--data", default="data/yolo_real_v2", metavar="DIR",
                        help="数据集根目录（含 images/<split>）")
    parser.add_argument("--sim", type=float, default=0.90,
                        help="相邻帧判为泄漏的相似度下限（默认 0.90）")
    parser.add_argument("--frame-gap", type=int, default=5, metavar="N",
                        help="帧号差 ≤N 视为相邻（默认 5）")
    parser.add_argument("--fail-sim", type=float, default=0.95,
                        help="跨集最大相似度 ≥ 它直接判失败（默认 0.95）")
    parser.add_argument("--size", type=int, default=32, help="特征边长（默认 32）")
    parser.add_argument("--json", metavar="PATH", help="把完整结果写成 JSON")
    args = parser.parse_args(argv)

    try:
        result = audit(args.data, args.sim, args.frame_gap, args.fail_sim, args.size)
    except FileNotFoundError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    print(f"数据集：{result['data']}（特征 {result['size']}×{result['size']} 灰度）")
    print("\n各划分 / 来源：")
    for split in SPLITS:
        info = result["splits"][split]
        sources = "  ".join(f"{k}:{v}" for k, v in sorted(info["sources"].items()))
        print(f"  {split:5s} {info['images']:4d} 张  {sources}")

    print("\n跨集相似度（每张图找 train 里最像的一张）：")
    for split in ("val", "test"):
        info = result["cross"][split]
        print(f"  {split:4s} vs train  最大 {info['max']:.3f}  中位 {info['median']:.3f}  "
              f"≥{result['sim']:.2f} 的 {info['over_sim']} 张  ≥0.95 的 {info['over_095']} 张")
        for row in info["worst"]:
            print(f"      {row['image']}  <->  {row['train_match']}  {row['similarity']:.3f}")

    print(f"\n相邻帧跨集（同组、帧号差 ≤{args.frame_gap}、相似度 ≥{args.sim:.2f}）："
          f"{len(result['adjacent'])} 对")
    for row in result["adjacent"][:20]:
        print(f"  {row['split']:4s} {row['image']}  <->  {row['train_match']}  "
              f"Δ帧 {row['frame_delta']:+d}  相似 {row['similarity']:.3f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"\n完整结果：{args.json}")

    if result["leaky"]:
        print("\n❌ 发现泄漏：相邻帧跨集，或跨集最大相似度 ≥ "
              f"{result['fail_sim']:.2f}。这个 test 上的 mAP 不能当泛化指标。")
        return 1
    print("\n✅ 没有发现相邻帧跨集；跨集最大相似度也低于失败线。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
