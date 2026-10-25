"""造合成 YOLO 数据集：红蓝方块与"半红半蓝"塔顶（规则 V2 的尺寸与配色）。

用大白话说
    物块还没做出来，但规则书已经把尺寸、颜色写死了（350 mm 塔基/中段、200 mm 塔顶、
    场地底色 RGB），所以可以先用几何渲染造一批「带自动标注」的训练图：
    立方体按真实尺寸投影到 1280×1024 画面，框由投影顶点直接算出来——**标注零成本**。

两条设计决定
    · **只标两类**：`block`（350 mm 立方体，红的蓝的都算）与 `top`（200 mm 塔顶）。
      红/蓝归属交给 HSV 判色——规则用颜色区分队伍，颜色比形状更可靠；
      塔顶本身就是红蓝各一半，颜色分不出来，只能靠尺寸/形状，这才是模型要学的。
    · **先算框、后画图**：每个候选物体先投影算 bbox，通不过检查的直接不画。
      早期版本是"先画再筛"：被筛掉的物体留在图上却没有标签，模型会把它们当背景学，
      这是合成数据最隐蔽的错（2026-10-25 修）。

塔顶长什么样（按实物照片修正）
    实物是**一块红半砖和一块蓝半砖并起来**：切割面竖直穿过立方体中线。
    所以：垂直于切割面的两个侧面（±x）整面各一色；另外两个侧面和顶面（±z、顶面）
    被中线一分为二——红一半、蓝一半。**不是**每个面都左右分色（早期版本画错成那样）。

产物（ultralytics 直接吃）
    data/yolo_synth/
    ├── images/train/*.jpg      labels/train/*.txt
    ├── images/val/*.jpg        labels/val/*.txt
    ├── dataset.yaml
    └── preview.jpg             带框预览（6 张拼一起，肉眼验标注用）

坐标约定（别混，混了框会绕着图中心转）
    · OpenCV 相机系：x 向右、y 向下、z 向前（光轴）；单位毫米。
    · 方块绕 y 轴（竖直轴）转，即"在地面上转个方向"。
    · 地面在相机下方 `--camera-height-mm` 处，方块底面贴地。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# 规则 V2 附录 1 的配色（RGB）
RED = (223, 34, 34)
BLUE = (50, 0, 255)
FIELD_COLORS = [
    (240, 210, 210),   # 地面区（红）
    (170, 210, 230),   # 地面区（蓝）
    (245, 240, 200),   # 公共区
    (190, 190, 185),   # L2 / 隔板
    (40, 100, 50),     # 建塔处（绿）
    (245, 170, 60),    # 储存区（红/橙）
]

BLOCK_MM = 350.0
TOP_MM = 200.0
CLASS_BLOCK = 0
CLASS_TOP = 1
CLASS_NAMES = ["block", "top"]

#: 立方体 6 个面（顶点下标按 sx,sy,sz 各取 -1/1 的顺序）
FACES = [
    (0, 1, 3, 2),  # x = -h
    (4, 6, 7, 5),  # x = +h
    (0, 4, 5, 1),  # y = -h（上）
    (2, 3, 7, 6),  # y = +h（下）
    (0, 2, 6, 4),  # z = -h（后）
    (1, 5, 7, 3),  # z = +h（前）
]
#: 哪些面沿 x 方向"跨越"中线（需要一分为二）。x=±h 的两个面整面同色，不在此列。
SPLIT_FACES = {2, 3, 4, 5}


def bgr(rgb) -> tuple:
    """规则书给的是 RGB，OpenCV 要 BGR。"""
    r, g, b = rgb
    return (int(b), int(g), int(r))


def rotate_y(points: np.ndarray, yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return points @ rot.T


def cube_vertices(size_mm: float, yaw_rad: float) -> np.ndarray:
    """以原点为中心、绕 y 轴转 `yaw` 的立方体 8 个顶点（相机系，毫米）。"""
    h = size_mm / 2.0
    local = np.array(
        [[sx * h, sy * h, sz * h]
         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=np.float64,
    )
    return rotate_y(local, yaw_rad)


def project(points: np.ndarray, intr: dict) -> np.ndarray:
    """相机系三维点 → 像素坐标（齐次投影）。"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = intr["fx"] * x / z + intr["cx"]
    v = intr["fy"] * y / z + intr["cy"]
    return np.stack([u, v], axis=1)


def face_normal(verts: np.ndarray, face, cube_center: np.ndarray) -> np.ndarray:
    """由顶点叉积算面法向，并翻成"朝外"（写死 ±x/±y/±z 在旋转后会出错）。"""
    a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.zeros(3)
    n = n / norm
    face_center = verts[list(face)].mean(axis=0)
    if float(n @ (face_center - cube_center)) < 0.0:
        n = -n
    return n


def shade(color_bgr, factor: float):
    return tuple(int(min(255, max(0, c * factor))) for c in color_bgr)


def split_polygon_at_x0(local_quad: np.ndarray):
    """把局部坐标里的四边形按 x=0 切成两半，返回 (x<=0 的多边形, x>=0 的多边形)。

    在**局部三维坐标**里切，不要用投影后的 2D 像素切：透视投影下"像素中点"
    不对应真实中点的投影，切出来的分界线会歪。
    """
    left, right = [], []
    n = len(local_quad)
    for i in range(n):
        p, q = local_quad[i], local_quad[(i + 1) % n]
        if p[0] <= 0:
            left.append(p)
        if p[0] >= 0:
            right.append(p)
        if p[0] * q[0] < 0:                     # 这条边跨过 x=0
            t = p[0] / (p[0] - q[0])
            mid = p + t * (q - p)
            left.append(mid)
            right.append(mid)
    return np.array(left), np.array(right)


def draw_cube(img, verts, intr, base_color_bgr, light_dir, cube_center, yaw,
              half_two_color=None):
    """按画家算法画一个立方体。

    :param half_two_color: 非 None 时按"红半砖 + 蓝半砖"画：切割面穿过局部 x=0，
                           垂直于切割面的两个侧面整面上色，其余面一分为二。
    """
    pts2d = project(verts, intr)
    order = sorted(range(len(FACES)), key=lambda f: -verts[list(FACES[f])][:, 2].mean())
    c, s = math.cos(-yaw), math.sin(-yaw)
    rot_inv = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    for fi in order:
        face = FACES[fi]
        normal = face_normal(verts, face, cube_center)
        center = verts[list(face)].mean(axis=0)
        if float(center @ normal) >= 0.0:        # 背向相机（相机在原点）
            continue
        quad2d = pts2d[list(face)]
        if not np.all(np.isfinite(quad2d)) or cv2.contourArea(quad2d.astype(np.float32)) < 1.0:
            continue

        lam = max(0.0, float(normal @ light_dir))
        factor = 0.55 + 0.45 * lam

        if half_two_color is None or fi not in SPLIT_FACES:
            color = base_color_bgr
            if half_two_color is not None:
                # x=±h 的两个面整面同色：看这半块在 x 的哪一侧
                local_x = [(verts[i] - cube_center) @ rot_inv.T for i in face]
                color = base_color_bgr if local_x[0][0] < 0 else half_two_color
            cv2.fillConvexPoly(img, quad2d.astype(np.int32), shade(color, factor))
            continue

        # 跨中线的面（顶面与 ±z 侧面）：按局部 x=0 切两半，分别上两种颜色
        local_quad = np.array([(verts[i] - cube_center) @ rot_inv.T for i in face])
        for part, color in ((split_polygon_at_x0(local_quad)[0], base_color_bgr),
                            (split_polygon_at_x0(local_quad)[1], half_two_color)):
            if len(part) < 3:
                continue
            world = rotate_y(part, yaw) + cube_center
            pix = project(world, intr).astype(np.int32)
            if cv2.contourArea(pix.astype(np.float32)) < 0.5:
                continue
            cv2.fillConvexPoly(img, pix, shade(color, factor))


def draw_contact_shadow(img, verts, intr, strength: float = 0.22) -> None:
    """底面投影成一块阴影压暗背景——否则方块看着像浮在空中。"""
    bottom = verts[[2, 3, 7, 6]]
    centroid = bottom.mean(axis=0)
    bottom = centroid + (bottom - centroid) * 0.88
    pts = project(bottom, intr).astype(np.int32)
    if not np.all(np.isfinite(pts)):
        return
    h, w = img.shape[:2]
    if pts[:, 0].max() < 0 or pts[:, 0].min() > w or pts[:, 1].max() < 0 or pts[:, 1].min() > h:
        return
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, pts, (0, 0, 0))
    cv2.addWeighted(overlay, strength, img, 1.0 - strength, 0, dst=img)


def make_background(width, height, field_color_bgr, bg_image=None, rng=None):
    """背景：场地纯色 + 光照渐变 + 地平线压暗 + 噪声；给了实拍图就随机裁一块贴上去。"""
    if bg_image is not None:
        bh, bw = bg_image.shape[:2]
        scale = max(width / bw, height / bh) * rng.uniform(1.0, 1.4)
        resized = cv2.resize(bg_image, (int(bw * scale) + 1, int(bh * scale) + 1))
        x0 = rng.integers(0, max(1, resized.shape[1] - width))
        y0 = rng.integers(0, max(1, resized.shape[0] - height))
        bg = resized[y0:y0 + height, x0:x0 + width].copy()
    else:
        bg = np.full((height, width, 3), field_color_bgr, dtype=np.uint8)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        grad = (xx / width * 0.25 + yy / height * 0.15).astype(np.float32)
        bg = np.clip(bg.astype(np.float32) * (1.0 + 0.35 * (0.5 - grad))[:, :, None], 0, 255)
    horizon = int(rng.uniform(0.45, 0.62) * height)
    bg[horizon:] = np.clip(bg[horizon:].astype(np.float32) * 0.82, 0, 255).astype(np.uint8)
    noise = rng.normal(0.0, 3.0, size=bg.shape).astype(np.float32)
    return np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def plan_slots(n_objects: int, rng, width: int) -> list:
    """把画面横向分成若干格：初筛位置，真正的重叠由 bbox IoU 再查一遍。"""
    cols = max(1, min(4, n_objects))
    rows = int(math.ceil(n_objects / cols))
    slots = []
    for r in range(rows):
        for c in range(cols):
            if len(slots) >= n_objects:
                break
            cx = width * (c + 0.5) / cols + rng.uniform(-0.06, 0.06) * width
            slots.append((cx, r))
    return slots


def bbox_of(verts, intr):
    pts = project(verts, intr)
    if not np.all(np.isfinite(pts)):
        return None
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def iou_2d(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-6, area_a + area_b - inter)


def render_one(rng, intr, args, bg_image=None):
    """渲染一张图，返回 (image, labels, n_drawn)。

    关键顺序：**先算 bbox → 过滤 → 只画通过过滤的物体**。
    反过来（先画再筛）会留下"没标签的物体"，模型会把它们当背景学。
    """
    width, height = int(intr["width"]), int(intr["height"])
    img = make_background(width, height, bgr(rng.choice(FIELD_COLORS)), bg_image, rng)
    light = np.array([-0.4, -0.6, 0.7], dtype=np.float64)
    light /= np.linalg.norm(light)

    n_obj = int(rng.integers(args.min_objects, args.max_objects + 1))
    candidates, accepted_boxes = [], []
    for slot_x, row in plan_slots(n_obj, rng, width):
        is_top = rng.random() < args.top_ratio
        size = TOP_MM if is_top else BLOCK_MM
        yaw = rng.uniform(0, math.pi) if is_top else rng.uniform(0, 2 * math.pi)
        z = rng.uniform(args.min_dist_mm, args.max_dist_mm)
        ground_y = args.camera_height_mm - size / 2.0
        x_cam = (slot_x - intr["cx"]) * z / intr["fx"]
        y_cam = ground_y + row * size * 1.15
        center = np.array([x_cam, y_cam, z], dtype=np.float64)
        verts = cube_vertices(size, yaw) + center

        bbox = bbox_of(verts, intr)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        # 要求整个投影都落在画面内（留边）：部分出画的框会系统性偏小，让模型学错位置
        if x1 < args.margin_px or y1 < args.margin_px \
                or x2 > width - args.margin_px or y2 > height - args.margin_px:
            continue
        if (x2 - x1) < args.min_side_px or (y2 - y1) < args.min_side_px:
            continue
        if any(iou_2d(bbox, other) > 0.02 for other in accepted_boxes):
            continue

        accepted_boxes.append(bbox)
        candidates.append({
            "center": center, "verts": verts, "yaw": yaw, "bbox": bbox,
            "class_id": CLASS_TOP if is_top else CLASS_BLOCK,
            "color": bgr(RED) if rng.random() < 0.5 else bgr(BLUE),
            "is_top": is_top,
        })

    candidates.sort(key=lambda c: -c["center"][2])      # 远的先画
    labels = []
    for obj in candidates:
        other = bgr(BLUE) if obj["color"] == bgr(RED) else bgr(RED)
        draw_contact_shadow(img, obj["verts"], intr)
        draw_cube(img, obj["verts"], intr, obj["color"], light, obj["center"], obj["yaw"],
                  half_two_color=other if obj["is_top"] else None)
        x1, y1, x2, y2 = obj["bbox"]
        labels.append((
            obj["class_id"],
            (x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height,
            (x2 - x1) / width, (y2 - y1) / height,
        ))

    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.0, 0.8))
    if rng.random() < 0.6:
        img = np.clip(img.astype(np.float32) * rng.uniform(0.75, 1.15), 0, 255).astype(np.uint8)
    return img, labels, len(candidates)


def draw_preview(images, labels, intr, out_path, cols=3):
    """把带框的图拼成一张预览，肉眼检查标注对不对。"""
    tiles = []
    for img, labs in zip(images, labels):
        canvas = img.copy()
        h, w = canvas.shape[:2]
        for cid, cx, cy, bw, bh in labs:
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            color = (0, 220, 0) if cid == CLASS_BLOCK else (0, 220, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
            cv2.putText(canvas, CLASS_NAMES[cid], (x1 + 4, max(20, y1 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        tiles.append(cv2.resize(canvas, (intr["width"] // 2, intr["height"] // 2)))
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * tiles[0].shape[0], cols * tiles[0].shape[1], 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * t.shape[0]:(r + 1) * t.shape[0], c * t.shape[1]:(c + 1) * t.shape[1]] = t
    cv2.imwrite(out_path, sheet)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="造合成 YOLO 数据集（350 mm 方块 / 200 mm 半红半蓝塔顶）")
    ap.add_argument("--out", default="data/yolo_synth", help="输出目录")
    ap.add_argument("--num", type=int, default=300, help="图片总数")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--fx", type=float, default=1321.67, help="默认取真机内参")
    ap.add_argument("--fy", type=float, default=1323.56)
    ap.add_argument("--cx", type=float, default=642.19)
    ap.add_argument("--cy", type=float, default=532.83)
    ap.add_argument("--camera-height-mm", type=float, default=600.0,
                    help="相机离地高度（方块底面贴地）")
    ap.add_argument("--min-dist-mm", type=float, default=900.0)
    ap.add_argument("--max-dist-mm", type=float, default=3500.0)
    ap.add_argument("--min-objects", type=int, default=1)
    ap.add_argument("--max-objects", type=int, default=3)
    ap.add_argument("--top-ratio", type=float, default=0.35, help="塔顶占比")
    ap.add_argument("--min-side-px", type=int, default=12, help="框最小边长（过滤太小的）")
    ap.add_argument("--margin-px", type=int, default=4,
                    help="框必须离画面边缘至少这么多像素（部分出画的框会系统性偏小）")
    ap.add_argument("--bg-dir", default="", help="实拍背景图目录（留空=用规则里的场地色）")
    ap.add_argument("--retries", type=int, default=6,
                    help="一张图里所有物体都被筛掉时，换一帧重试几次")
    ap.add_argument("--preview", type=int, default=6, help="预览拼图用几张")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    intr = {"fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy,
            "width": args.width, "height": args.height}

    bgs = []
    if args.bg_dir:
        bgs = [cv2.imread(str(p)) for p in sorted(Path(args.bg_dir).glob("*"))
               if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        bgs = [b for b in bgs if b is not None]
        if not bgs:
            print(f"❌ {args.bg_dir} 里没有可用图片")
            return 2

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_val = int(round(args.num * args.val_ratio))
    previews, preview_labels = [], []
    stats = {0: 0, 1: 0}
    skipped_empty = 0
    written = 0
    for i in range(args.num):
        # 重试几次：一帧里所有候选都被"必须完整在画面内/不许重叠"筛掉时换一帧再试，
        # 否则会有近 20% 的图是空的（实测 300 张里空 57 张）
        for _ in range(args.retries):
            img, labels, n_drawn = render_one(rng, intr, args,
                                              random.choice(bgs) if bgs else None)
            assert n_drawn == len(labels), "画出来的物体数与标签数不一致（回归防护）"
            if labels:
                break
        if not labels:
            skipped_empty += 1
            continue
        split = "val" if i < n_val else "train"
        name = f"{i:05d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(out / "labels" / split / f"{name}.txt", "w", encoding="utf-8") as f:
            for cid, cx, cy, bw, bh in labels:
                f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                stats[cid] = stats.get(cid, 0) + 1
        if len(previews) < args.preview:
            previews.append(img)
            preview_labels.append(labels)
        written += 1

    (out / "dataset.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names: [{', '.join(CLASS_NAMES)}]\n",
        encoding="utf-8",
    )
    (out / "meta.json").write_text(json.dumps({
        "classes": CLASS_NAMES, "num": args.num, "val": n_val, "seed": args.seed,
        "written": written, "skipped_empty": skipped_empty,
        "intrinsics": {k: intr[k] for k in ("fx", "fy", "cx", "cy", "width", "height")},
        "block_mm": BLOCK_MM, "top_mm": TOP_MM,
        "counts": {CLASS_NAMES[k]: v for k, v in stats.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_preview(previews, preview_labels, intr, str(out / "preview.jpg"))

    print(f"✅ 已写 {written} 张（val {min(n_val, written)} / train {written - min(n_val, written)}）→ {out}")
    print(f"   框数：block {stats.get(0, 0)}，top {stats.get(1, 0)}；空图跳过 {skipped_empty} 张")
    print(f"   预览：{out / 'preview.jpg'}；训练配置：{out / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
