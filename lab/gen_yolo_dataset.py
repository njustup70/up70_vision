"""造合成 YOLO 数据集：红蓝方块与半红半蓝塔顶（规则 V2 的尺寸与配色）。

用大白话说
    物块还没做出来，但规则书已经把尺寸、颜色写死了（350 mm 塔基/中段、200 mm 半红半蓝
    塔顶、场地底色 RGB），所以可以先用几何渲染造一批「带自动标注」的训练图：
    立方体按真实尺寸投影到 1280×1024 画面，框由投影顶点直接算出来——**标注零成本**。

两条设计决定
    · **只标两类**：`block`（350 mm 立方体，红的蓝的都算）与 `top`（200 mm 半红半蓝）。
      红/蓝归属交给 HSV 判色——规则用颜色区分队伍，颜色比形状更可靠；
      塔顶本来就是半红半蓝，颜色分不出来，只能靠尺寸/形状，这才是模型要学的。
    · **物体不互相遮挡**：按网格放，宁可少一两个也不要"框把别人的身体包进来"。
      真图里的遮挡等有了实物再补，合成阶段先把"干净样本"训稳。

放什么 / 不放什么
    只放"造图 + 算框 + 写 YOLO 目录结构"。训练不在这里（用 ultralytics，只有
    `~/.venvs/yolo` 装了 torch，仓库依赖里没有）。

怎么用
    python lab/gen_yolo_dataset.py --out data/yolo_synth --num 300
    python lab/gen_yolo_dataset.py --out data/yolo_synth --num 300 --bg-dir data/photos

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
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# 规则 V2 附录 1 的配色（RGB）
RED = (223, 34, 34)
BLUE = (50, 0, 255)
GOLD = (218, 165, 32)
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


def bgr(rgb) -> tuple:
    """规则书给的是 RGB，OpenCV 要 BGR。"""
    r, g, b = rgb
    return (int(b), int(g), int(r))


def cube_vertices(size_mm: float, yaw_rad: float) -> np.ndarray:
    """以原点为中心、绕 y 轴转 `yaw` 的立方体 8 个顶点（相机系，毫米）。"""
    h = size_mm / 2.0
    local = np.array(
        [[sx * h, sy * h, sz * h]
         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=np.float64,
    )
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    # 绕 y 轴（竖直轴）旋转：x' = x c + z s, z' = -x s + z c
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return local @ rot.T


#: 立方体 6 个面，每个面是 4 个顶点索引（顶点顺序按下标：sx,sy,sz 各取 -1/1）
FACES = [
    (0, 1, 3, 2),  # x = -h（左）
    (4, 6, 7, 5),  # x = +h（右）
    (0, 4, 5, 1),  # y = -h（上）
    (2, 3, 7, 6),  # y = +h（下）
    (0, 2, 6, 4),  # z = -h（后）
    (1, 5, 7, 3),  # z = +h（前）
]
#: 与 FACES 一一对应。顺序别按 ±x/±y/±z 想当然写——之前写反了，画出来是"盒子内壁"
FACE_NORMALS = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]


def project(points: np.ndarray, intr: dict) -> np.ndarray:
    """相机系三维点 → 像素坐标（齐次投影）。"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = intr["fx"] * x / z + intr["cx"]
    v = intr["fy"] * y / z + intr["cy"]
    return np.stack([u, v], axis=1)


def visible_faces(verts: np.ndarray) -> list:
    """背向相机的面不画：法向与（面心 - 相机原点）点积为负即朝向相机。"""
    out = []
    for idx, (face, normal) in enumerate(zip(FACES, FACE_NORMALS)):
        face_pts = verts[list(face)]
        center = face_pts.mean(axis=0)
        n = np.array(normal, dtype=np.float64)
        # 面的法向随 yaw 转
        if float(center @ n) < 0.0:
            out.append(idx)
    return out


def shade(color_bgr, factor: float):
    return tuple(int(min(255, max(0, c * factor))) for c in color_bgr)



def draw_contact_shadow(img, verts, intr, strength: float = 0.22) -> None:
    """把立方体底面投影成一块阴影，压暗背景——不然方块看起来像浮在空中。

    做法是最土的"接地阴影"：底面四边形（索引 2/3/7/6）直接当阴影形状，
    用 alpha 混合压暗。没有软边，但足够让模型学到"物体在下、影子在下"的关系。
    """
    bottom = verts[[2, 3, 7, 6]]
    # 收 12%：阴影边界紧贴方块底面，看起来才像"接触阴影"而不是地上一块补丁
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



def face_normal(verts, face, cube_center) -> np.ndarray:
    """由顶点直接算面法向（叉积），并翻成"朝外"。

    不能像最初那样按固定 ±x/±y/±z 写死：立方体绕 y 轴转过之后，法向也跟着转，
    写死的法向会把该画的面剔掉、把背面画出来——表现为方块"缺一面、能看见内壁"。
    """
    a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0])
    n = n / norm
    face_center = verts[list(face)].mean(axis=0)
    if float(n @ (face_center - cube_center)) < 0.0:
        n = -n
    return n


def draw_cube(img, verts, intr, class_id, base_color_bgr, light_dir,
              half_two_color=None, cube_center=None):
    """按画家算法画一个立方体，返回像素 bbox（可能超出画面，由调用方裁剪）。"""
    if cube_center is None:
        cube_center = verts.mean(axis=0)
    pts2d = project(verts, intr)
    order = sorted(range(len(FACES)), key=lambda f: -verts[list(FACES[f])][:, 2].mean())
    drawn = []
    for fi in order:
        face = FACES[fi]
        normal = face_normal(verts, face, cube_center)
        center = verts[list(face)].mean(axis=0)
        if float(center @ normal) >= 0.0:   # 背向相机（相机在原点）
            continue
        quad = pts2d[list(face)]
        if not np.all(np.isfinite(quad)):
            continue
        if cv2.contourArea(quad.astype(np.float32)) < 1.0:
            continue
        lam = max(0.0, float(normal @ light_dir))
        factor = 0.55 + 0.45 * lam
        if half_two_color is not None and fi in (0, 1, 4, 5):
            # 塔顶侧面：沿本地 x 一分为二，红半与蓝半（规则 V2：4 个侧面半红半蓝）
            a, b, c, d = face
            mid_ab = (verts[a] + verts[b]) / 2
            mid_dc = (verts[d] + verts[c]) / 2
            left = project(np.stack([verts[a], verts[d], mid_dc, mid_ab]), intr).astype(np.int32)
            right = project(np.stack([mid_ab, mid_dc, verts[c], verts[b]]), intr).astype(np.int32)
            cv2.fillConvexPoly(img, left, shade(base_color_bgr, factor))
            cv2.fillConvexPoly(img, right, shade(half_two_color, factor * 0.95))
        else:
            cv2.fillConvexPoly(img, quad.astype(np.int32), shade(base_color_bgr, factor))
        drawn.append(quad)
    xs = pts2d[:, 0]
    ys = pts2d[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def make_background(width, height, field_color_bgr, bg_image=None, rng=None):
    """背景：场地纯色 + 光照渐变 + 噪声；给了实拍图就随机裁一块贴上去。"""
    if bg_image is not None:
        bh, bw = bg_image.shape[:2]
        scale = max(width / bw, height / bh) * rng.uniform(1.0, 1.4)
        resized = cv2.resize(bg_image, (int(bw * scale) + 1, int(bh * scale) + 1))
        x0 = rng.integers(0, max(1, resized.shape[1] - width))
        y0 = rng.integers(0, max(1, resized.shape[0] - height))
        bg = resized[y0:y0 + height, x0:x0 + width].copy()
    else:
        bg = np.full((height, width, 3), field_color_bgr, dtype=np.uint8)
        # 对角光照渐变：模拟场地灯
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        grad = (xx / width * 0.25 + yy / height * 0.15).astype(np.float32)
        bg = np.clip(bg.astype(np.float32) * (1.0 + 0.35 * (0.5 - grad))[:, :, None], 0, 255)
    # 地平线：相机水平时在地平线在 cy；线下压暗一点，方块看起来才"站在地上"
    horizon = int(rng.uniform(0.45, 0.62) * height)
    floor = bg[horizon:].astype(np.float32) * 0.82
    bg[horizon:] = np.clip(floor, 0, 255).astype(np.uint8)
    noise = rng.normal(0.0, 3.0, size=bg.shape).astype(np.float32)
    bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return bg


def plan_slots(n_objects: int, rng, width: int) -> list:
    """把画面横向分成若干格，保证物体不横着叠在一起（避免框互相包住）。"""
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


def render_one(rng, intr, args, bg_image=None):
    """渲染一张图，返回 (image, labels)，labels 是 [(class_id, cx, cy, w, h)] 归一化。"""
    width, height = int(intr["width"]), int(intr["height"])
    img = make_background(width, height, bgr(rng.choice(FIELD_COLORS)), bg_image, rng)

    light = np.array([-0.4, -0.6, 0.7], dtype=np.float64)
    light /= np.linalg.norm(light)

    n_obj = int(rng.integers(args.min_objects, args.max_objects + 1))
    labels = []
    for slot_x, row in plan_slots(n_obj, rng, width):
        is_top = rng.random() < args.top_ratio
        size = TOP_MM if is_top else BLOCK_MM
        yaw = rng.uniform(0, math.pi) if is_top else rng.uniform(0, 2 * math.pi)

        # 距离：近处大、远处小；越远的往画面上方挪，做出"摆在地上"的观感
        z = rng.uniform(args.min_dist_mm, args.max_dist_mm)
        ground_y = args.camera_height_mm - size / 2.0
        # 用像素反推横向位置：先选画面上的横坐标，再换成相机系 x
        x_cam = (slot_x - intr["cx"]) * z / intr["fx"]
        y_cam = ground_y + row * size * 1.15

        center = np.array([x_cam, y_cam, z], dtype=np.float64)
        verts = cube_vertices(size, yaw) + center

        color = bgr(RED if rng.random() < 0.5 else BLUE)
        half = bgr(BLUE) if color == bgr(RED) else bgr(RED)
        draw_contact_shadow(img, verts, intr)
        bbox = draw_cube(
            img, verts, intr,
            CLASS_TOP if is_top else CLASS_BLOCK,
            color, light,
            half_two_color=half if is_top else None,
            cube_center=center,
        )

        x1, y1, x2, y2 = bbox
        # 框裁到画面内；要求主体可见、够大
        cx_, cy_ = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if not (0 <= cx_ < width and 0 <= cy_ < height):
            continue
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(width - 1.0, x2), min(height - 1.0, y2)
        w, h = x2 - x1, y2 - y1
        if w < args.min_side_px or h < args.min_side_px:
            continue
        if (w * h) / float(width * height) > 0.6:
            continue
        labels.append((
            CLASS_TOP if is_top else CLASS_BLOCK,
            (x1 + x2) / 2.0 / width,
            (y1 + y2) / 2.0 / height,
            w / width,
            h / height,
        ))

    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.0, 0.8))
    if rng.random() < 0.6:
        img = np.clip(img.astype(np.float32) * rng.uniform(0.75, 1.15), 0, 255).astype(np.uint8)
    return img, labels


def draw_preview(images, labels, intr, out_path, cols=3):
    """把带框的图拼成一张预览，肉眼检查标注对不对。"""
    tiles = []
    for img, labs in zip(images, labels):
        canvas = img.copy()
        h, w = canvas.shape[:2]
        for cid, cx, cy, bw, bh in labs:
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
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
    ap.add_argument("--bg-dir", default="", help="实拍背景图目录（留空=用规则里的场地色）")
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
    for i in range(args.num):
        for _ in range(6):  # 不合格（框出画/太小）就重画一张
            img, labels = render_one(rng, intr, args, random.choice(bgs) if bgs else None)
            if labels:
                break
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

    (out / "dataset.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names: [{', '.join(CLASS_NAMES)}]\n",
        encoding="utf-8",
    )
    (out / "meta.json").write_text(json.dumps({
        "classes": CLASS_NAMES, "num": args.num, "val": n_val, "seed": args.seed,
        "intrinsics": {k: intr[k] for k in ("fx", "fy", "cx", "cy", "width", "height")},
        "block_mm": BLOCK_MM, "top_mm": TOP_MM,
        "counts": {CLASS_NAMES[k]: v for k, v in stats.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_preview(previews, preview_labels, intr, str(out / "preview.jpg"))

    print(f"✅ 已写 {args.num} 张（train {args.num - n_val} / val {n_val}）→ {out}")
    print(f"   框数：block {stats.get(0, 0)}，top {stats.get(1, 0)}")
    print(f"   预览：{out / 'preview.jpg'}；训练配置：{out / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
