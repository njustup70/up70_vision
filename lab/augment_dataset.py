"""用实拍照片扩充训练集：几何变换 + 光照扰动，标签同步变换。

为什么需要
    真图只有 120 张，`top` 召回只有 0.78。补拍要人、要时间；而**翻折/90° 旋转/亮度扰动**
    能免费翻几倍样本量，且标签可以严格同步（不是"猜"出来的伪标签）。

只做不会改变语义的增强
    · 水平翻转、90°/180°/270° 旋转、小幅缩放与平移
    · 亮度/对比度、gamma、轻微模糊与噪声
    **不改色相（Hue）**：颜色是类别语义的一部分（红/蓝由 HSV 判），调色相等于造假样本。

标签怎么保证还对
    每张图记录一个像素级仿射矩阵 M，用它变换框的四个角，再取外接矩形；变换后
    · 任一角跑出画面 → 丢弃这个样本（不裁掉目标，避免"半截框"教坏模型）
    · 框边长 < 8 px → 丢弃
    保存前断言所有坐标都在 [0,1]。

用法
    # 把 train 扩到 4 倍（原图也一起拷进去）
    PYTHONPATH=. python lab/augment_dataset.py --src data/yolo_real2 --out data/yolo_real2_aug --factor 4

    # 只扩某一档、并把扩出来的图存成预览拼图供人检查
    PYTHONPATH=. python lab/augment_dataset.py --src data/yolo_real2 --out /tmp/aug --factor 2 --preview
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import sys

import cv2
import numpy as np

MIN_SIDE_PX = 8


def load_labels(path: str, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    """读 YOLO 标签 → [(cls, x1, y1, x2, y2)]（像素）。"""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) != 5:
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
        out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def transform_boxes(boxes, M, w, h):
    """按仿射矩阵 M 变换框，返回 (新框列表, 是否全部合法)。

    用"四角变换再取外接矩形"，这样翻折/90° 旋转/缩放都只需要一个矩阵，
    不用为每种变换单独推公式（单独推最容易在 90° 旋转上把 w/h 弄反）。
    """
    out = []
    for cls, x1, y1, x2, y2 in boxes:
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], np.float64).T
        moved = (M @ pts).T
        xs, ys = moved[:, 0], moved[:, 1]
        bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max(), ys.max()
        if bx1 < -1 or by1 < -1 or bx2 > w + 1 or by2 > h + 1:
            return [], False                      # 有目标跑出画面：整张丢弃
        if (bx2 - bx1) < MIN_SIDE_PX or (by2 - by1) < MIN_SIDE_PX:
            return [], False                      # 太小了，标了也没用
        out.append((cls, float(np.clip(bx1, 0, w)), float(np.clip(by1, 0, h)),
                    float(np.clip(bx2, 0, w)), float(np.clip(by2, 0, h))))
    return out, True


def geometry_tf(kind: str, w: int, h: int) -> np.ndarray:
    """返回像素级 2x3 仿射矩阵。"""
    if kind == "flip":
        return np.array([[-1, 0, w - 1], [0, 1, 0]], dtype=np.float64)
    if kind == "rot90":
        # 逆时针 90°：新宽=h、新高=w
        return np.array([[0, 1, 0], [-1, 0, w - 1]], dtype=np.float64)
    if kind == "rot180":
        return np.array([[-1, 0, w - 1], [0, -1, h - 1]], dtype=np.float64)
    if kind == "rot270":
        return np.array([[0, -1, h - 1], [1, 0, 0]], dtype=np.float64)
    if kind == "shift":
        dx, dy = random.uniform(-0.05, 0.05) * w, random.uniform(-0.05, 0.05) * h
        return np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64)
    if kind == "zoom":
        s = random.uniform(0.9, 1.1)
        return cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
    return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)


def photometric(img: np.ndarray, kind: str) -> np.ndarray:
    """只动亮度/对比度/清晰度，不动色相。"""
    if kind == "bright":
        return cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2),
                                   beta=random.uniform(-25, 25))
    if kind == "gamma":
        g = random.uniform(0.75, 1.35)
        lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], np.uint8)
        return cv2.LUT(img, lut)
    if kind == "blur":
        return cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 1.0))
    if kind == "noise":
        n = np.random.normal(0, random.uniform(3, 8), img.shape)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return img


GEOMETRIES = ["flip", "rot90", "rot180", "rot270", "shift", "zoom"]
PHOTOS = ["bright", "gamma", "blur", "noise"]


def augment_one(img, boxes, w, h, rng: random.Random):
    """随机挑一组增强，返回 (新图, 新框, 新宽, 新高) 或 None（不可用）。"""
    random.seed(rng.random())
    geo = rng.choice(GEOMETRIES)
    M = geometry_tf(geo, w, h)
    new_w, new_h = (h, w) if geo in ("rot90", "rot270") else (w, h)
    out = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
    new_boxes, ok = transform_boxes(boxes, M, new_w, new_h)
    if not ok or not new_boxes:
        return None
    out = photometric(out, rng.choice(PHOTOS + [""]))
    return out, new_boxes, new_w, new_h


def write_labels(path: str, boxes, w: int, h: int) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for cls, x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            assert 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1
            fh.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="实拍数据集增强（几何 + 光照，标签同步）")
    ap.add_argument("--src", default="data/yolo_real2")
    ap.add_argument("--out", default="data/yolo_real2_aug")
    ap.add_argument("--splits", default="train", help="要增强的划分，逗号分隔")
    ap.add_argument("--factor", type=int, default=4, help="每个划分扩到几倍（含原图）")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--preview", action="store_true", help="生成增强样例拼图 output/aug_preview.jpg")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    total = {"orig": 0, "aug": 0, "skip": 0}
    tiles = []

    for split in splits:
        img_dir = os.path.join(args.src, "images", split)
        lab_dir = os.path.join(args.src, "labels", split)
        out_img = os.path.join(args.out, "images", split)
        out_lab = os.path.join(args.out, "labels", split)
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lab, exist_ok=True)
        for path in sorted(glob.glob(os.path.join(img_dir, "*.*"))):
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            stem = os.path.splitext(os.path.basename(path))[0]
            boxes = load_labels(os.path.join(lab_dir, stem + ".txt"), w, h)
            if not boxes:
                continue
            shutil.copyfile(path, os.path.join(out_img, stem + ".jpg"))
            shutil.copyfile(os.path.join(lab_dir, stem + ".txt"),
                            os.path.join(out_lab, stem + ".txt"))
            total["orig"] += 1
            for k in range(1, max(1, args.factor)):
                got = augment_one(img, boxes, w, h, rng)
                if got is None:
                    total["skip"] += 1
                    continue
                aimg, abox, aw, ah = got
                name = f"{stem}_a{k:02d}"
                cv2.imwrite(os.path.join(out_img, name + ".jpg"), aimg,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                write_labels(os.path.join(out_lab, name + ".txt"), abox, aw, ah)
                total["aug"] += 1
                if args.preview and len(tiles) < 20:
                    vis = aimg.copy()
                    for cls, x1, y1, x2, y2 in abox:
                        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                                      (0, 200, 0) if cls == 1 else (0, 0, 255), 2)
                    tiles.append(cv2.resize(vis, (320, 256)))

    # 其余划分原样拷贝，保证 dataset.yaml 结构完整
    for split in ("val", "test"):
        if split in splits:
            continue
        for sub in ("images", "labels"):
            s = os.path.join(args.src, sub, split)
            d = os.path.join(args.out, sub, split)
            if os.path.isdir(s):
                os.makedirs(d, exist_ok=True)
                for f in glob.glob(os.path.join(s, "*")):
                    shutil.copyfile(f, os.path.join(d, os.path.basename(f)))

    yaml_src = os.path.join(args.src, "dataset.yaml")
    if os.path.exists(yaml_src):
        text = open(yaml_src, encoding="utf-8").read()
        lines = [f"path: {os.path.abspath(args.out)}" if ln.startswith("path:") else ln
                 for ln in text.splitlines()]
        open(os.path.join(args.out, "dataset.yaml"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    if args.preview and tiles:
        while len(tiles) % 5:
            tiles.append(np.zeros((256, 320, 3), np.uint8))
        cv2.imwrite("output/aug_preview.jpg",
                    np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("  预览: output/aug_preview.jpg")

    print(f"  原图 {total['orig']} 张，新增 {total['aug']} 张，丢弃 {total['skip']} 次（框会跑出画面）")
    print(f"  输出: {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
