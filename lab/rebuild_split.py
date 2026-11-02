"""按"采集场次"重建数据集划分（train/val/test 各自用整组，绝不组内分层）。

为什么不用 split_dataset.py 的分法
    它让每个采集组**各自**出 val/test，于是同一场次的相邻帧会同时出现在 train 和 test；
    `lab/audit_split.py` 在 v2 上实测到 8 对相邻帧跨集、跨集最大相似度 0.971。
    换场景后指标会掉，所以 v3 改成**整组留出**：一组只能进一个划分。

默认划分（可用参数改）
    train: g1front + g2top + g6far   （近景正对 + 塔顶 + 远景，789 框）
    val  : g5dist                    （距离梯度，52 框；调 conf 用）
    test : g3tilt + g4stack          （斜视 + 叠放，173 框；两组都是 v2 里最差的）

    为什么 test 选 g3/g4：v2 上它们的 R 只有 0.75 / 0.78，是最能暴露问题的两组；
    为什么 g6far 放 train：它是唯一的大规模远景组（646 框），留作 test 会让训练集只剩 368 框；
    代价：v3 的 test **不测远景泛化**。远景泛化另建一折（train=g1~g5, test=g6far）再评。

用法
    PYTHONPATH=. python lab/rebuild_split.py            # 默认划分 -> data/yolo_real_v3
    PYTHONPATH=. python lab/rebuild_split.py --overwrite
    PYTHONPATH=. python lab/rebuild_split.py --test-groups g4stack,g5dist --val-groups g3tilt

产物
    data/yolo_real_v3/{images,labels}/{train,val,test}/ + dataset.yaml + split_manifest.json
    建完自动跑一遍 audit_split；有泄漏（跨集相似度 ≥0.95 或相邻帧跨集）就返回 1。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.audit_split import audit  # noqa: E402

DEFAULT_TRAIN = "g1front,g2top,g6far"
DEFAULT_VAL = "g5dist"
DEFAULT_TEST = "g3tilt,g4stack"


def group_of(path: str) -> str:
    """`g6far_frame_0121.jpg` -> `g6far`；没有 `_frame_` 时用整段 stem。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_frame_")[0] if "_frame_" in stem else stem


def parse_groups(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def collect(src: str) -> dict[str, list[tuple[str, str]]]:
    """按组收集 `(图片, 标签)`；缺标签直接报错，避免静默丢样本。"""
    images = sorted(glob.glob(os.path.join(src, "images", "*", "*.jpg")))
    grouped: dict[str, list[tuple[str, str]]] = {}
    missing = []
    for image in images:
        stem = os.path.splitext(image)[0]
        rel = os.path.relpath(stem, os.path.join(src, "images"))
        label = os.path.join(src, "labels", rel + ".txt")
        if not os.path.isfile(label):
            missing.append(image)
            continue
        grouped.setdefault(group_of(image), []).append((image, label))
    if missing:
        raise FileNotFoundError(f"{len(missing)} 张图缺标签，例如：{missing[0]}")
    return grouped


def count_boxes(label: str) -> int:
    with open(label, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="按采集场次重建 train/val/test（整组留出）")
    ap.add_argument("--src", default="data/yolo_real_v2", help="源数据集（含 images/labels）")
    ap.add_argument("--out", default="data/yolo_real_v3", help="输出数据集目录")
    ap.add_argument("--train-groups", default=DEFAULT_TRAIN)
    ap.add_argument("--val-groups", default=DEFAULT_VAL)
    ap.add_argument("--test-groups", default=DEFAULT_TEST)
    ap.add_argument("--overwrite", action="store_true", help="输出目录非空时先删掉")
    args = ap.parse_args(argv)

    grouped = collect(args.src)
    assignment = {
        "train": parse_groups(args.train_groups),
        "val": parse_groups(args.val_groups),
        "test": parse_groups(args.test_groups),
    }
    all_groups = set(grouped)
    assigned = [g for groups in assignment.values() for g in groups]
    duplicated = sorted({g for g in assigned if assigned.count(g) > 1})
    unknown = sorted(set(assigned) - all_groups)
    unassigned = sorted(all_groups - set(assigned))
    if duplicated or unknown or unassigned:
        print(f"❌ 分组不合法：重复={duplicated} 未知={unknown} 未分配={unassigned}", file=sys.stderr)
        print(f"   源数据里实际有：{sorted(all_groups)}", file=sys.stderr)
        return 2

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            print(f"❌ {out} 已存在且非空；确认要覆盖就加 --overwrite", file=sys.stderr)
            return 2
        shutil.rmtree(out)

    manifest = {"source": os.path.abspath(args.src), "out": os.path.abspath(out),
                "groups": {}, "splits": {}, "boxes": {}}
    total_images = total_boxes = 0

    print(f"源：{args.src}")
    print(f"{'划分':6s} {'组':32s} {'图':>4s} {'框':>5s}")
    for split, groups in assignment.items():
        dst_img = out / "images" / split
        dst_lab = out / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lab.mkdir(parents=True, exist_ok=True)
        n_images = n_boxes = 0
        for group in groups:
            for image, label in grouped[group]:
                shutil.copy2(image, dst_img / os.path.basename(image))
                shutil.copy2(label, dst_lab / os.path.basename(label))
                n_images += 1
                n_boxes += count_boxes(label)
        manifest["groups"][split] = groups
        manifest["splits"][split] = n_images
        manifest["boxes"][split] = n_boxes
        total_images += n_images
        total_boxes += n_boxes
        print(f"{split:6s} {','.join(groups):32s} {n_images:4d} {n_boxes:5d}")

    with open(out / "dataset.yaml", "w", encoding="utf-8") as handle:
        handle.write(f"path: {os.path.abspath(out)}\ntrain: images/train\n"
                     f"val: images/val\ntest: images/test\nnames: [block, top]\n")

    with open(out / "split_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"\n合计 {total_images} 张 / {total_boxes} 框 -> {out}")

    result = audit(str(out), sim=0.90, frame_gap=5, fail_sim=0.95, size=32)
    print("\n自动审计：")
    for split in ("val", "test"):
        info = result["cross"][split]
        print(f"  {split:4s} vs train 最大相似度 {info['max']:.3f}  "
              f"≥0.90 的 {info['over_sim']} 张")
    print(f"  相邻帧跨集 {len(result['adjacent'])} 对")
    print(f"  清单：{out / 'split_manifest.json'}")
    if result["leaky"]:
        print("\n❌ 审计未通过：这个划分仍有泄漏，不要拿去训练。")
        return 1
    print("\n✅ 审计通过：整组留出，无相邻帧跨集。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
