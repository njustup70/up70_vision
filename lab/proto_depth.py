"""B1-B3 原型：反投影 -> 体素降采样 -> RANSAC 去桌面 -> 聚类。

跑法（容器内，用 S4/S5 的合成数据）
    docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision \
        python lab/proto_depth.py data/syn/single_01

    # 换成真实 bag 抽出来的深度图，指定内参
    ... python lab/proto_depth.py data/real_01 --fx 409.5 --fy 409.5 --cx 424.33 --cy 241.33

为什么先在 lab/ 写
    ROS2 接入方案 10.2 的四步落地：原型在 lab/ 跑通 -> 参数稳定 -> 提炼进
    vision/depth/ -> fuse_node 改 import。参数（leaf_size、distance_threshold、
    cluster_tolerance）现在全靠实测调，直接写进正式模块会在签名未定型时
    反复返工。

三步各自在做什么
    B1 降采样：点云降到十分之一，RANSAC 的耗时直接跟着降。leaf 取 3~5mm，
       比噪声（0.4~1.7mm）大一个量级，所以不会把噪声当成结构。
    B2 去桌面：RANSAC 找最大平面。桌面是视场里最大的一块平面，天然是最大的
       inlier 集合。阈值不能只取 3σ —— A4 实测墙面/桌面本身就有 2~5mm 起伏，
       取太紧会把真实桌面点判成外点。
    B3 聚类：这里不实现真正的欧式聚类（没有 scipy，手写 KD 树太慢），
       而是利用「深度图是组织化点云」这个事实：相邻像素的 3D 距离近似等于
       横向像素间距，只有跨物体边界时才会跳变。所以在深度跳变处切断连通域，
       得到的就是欧式聚类的等价结果，而且是一遍 cv2.connectedComponents。

已知的坑
    · RANSAC 的 inlier 阈值单位是毫米，别按米填。写 0.01 会让每个点都算外点，
      表现是「找不到平面」，而不是报错。
    · 平面法向的方向没有约定（n 和 -n 都是解）。算夹角时必须取绝对值，
      否则会得到 180° 而不是 0°。
    · 深度跳变阈值太小时，噪声会在物体内部切断连通域，一个物块裂成好几块；
      太大时，物块和桌面连成一片。经验取 15~25mm，比噪声大一个量级。
    · 物块贴桌面那一圈的像素是双目的混合像素，深度值介于物体和桌面之间，
      既不算桌面也不算物体，会让包围盒偏大。后面 B4 取顶面 20% 就是为了绕开它。

退出码：0 成功；1 数据读不了或场景里找不到桌面；2 参数非法。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# RANSAC 精化后重新判 inlier 用的阈值，与主阈值一致
DEFAULT_PLANE_THRESH_MM = 6.0
DEFAULT_LEAF_MM = 4.0
DEFAULT_CLUSTER_JUMP_MM = 18.0


def load_data(data_dir: Path) -> tuple:
    """读 S4 生成的 depth.npy + meta.json；也接受只有 depth.npy 的目录。

    返回 (depth_mm uint16, intrinsics dict, meta dict|None)。
    """
    depth_path = data_dir / "depth.npy"
    if not depth_path.exists():
        raise FileNotFoundError(f"找不到 {depth_path}")
    depth = np.load(depth_path)
    if depth.dtype != np.uint16:
        raise ValueError(f"depth.npy 应为 uint16 毫米，实际是 {depth.dtype}")

    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    cam = meta["camera"] if meta else {}
    k = {
        "fx": cam.get("fx", 409.5), "fy": cam.get("fy", 409.5),
        "cx": cam.get("cx", depth.shape[1] / 2.0), "cy": cam.get("cy", depth.shape[0] / 2.0),
    }
    return depth, k, meta


def backproject(depth: np.ndarray, k: dict) -> tuple:
    """深度图 -> 点云。返回 (点云 (N,3) mm, 像素坐标 (N,2))。

    深度是 z 深度（沿光轴），所以这一步是 x=(u-cx)z/fx、y=(v-cy)z/fy 的直接代入，
    不需要再乘余弦。误把深度当欧氏距离会让点云边缘整体外扩。
    """
    v, u = np.nonzero(depth > 0)
    z = depth[v, u].astype(np.float32)
    x = (u.astype(np.float32) - k["cx"]) * z / k["fx"]
    y = (v.astype(np.float32) - k["cy"]) * z / k["fy"]
    return np.stack([x, y, z], axis=1), np.stack([u, v], axis=1)


def voxel_downsample(points: np.ndarray, leaf: float) -> np.ndarray:
    """体素降采样：按 leaf 分格，每格取落点均值。

    用 np.unique 的 return_inverse 做分格，再用 bincount 求均值——
    比 Python 循环快两个量级，也不需要 scipy。
    """
    if leaf <= 0:
        return points
    keys = np.floor(points / leaf).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    n_vox = len(counts)
    sums = np.zeros((n_vox, 3), dtype=np.float64)
    for axis in range(3):
        sums[:, axis] = np.bincount(inverse, weights=points[:, axis], minlength=n_vox)
    return (sums / counts[:, None]).astype(np.float32)


def ransac_plane(points: np.ndarray, thresh: float, iters: int,
                 rng: np.random.Generator) -> tuple:
    """RANSAC 找最大平面，返回 (法向, 偏移 d, inlier 掩码)。

    平面写作 n·p + d = 0。找到后用它自己的 inlier 做一次最小二乘精化——
    三点采样出来的平面精度受点噪声影响很大，精化后法向才稳。
    """
    best_count = 0
    best_inliers = None
    n_pts = len(points)
    if n_pts < 3:
        raise ValueError(f"点太少（{n_pts}），没法拟合平面")

    for _ in range(iters):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = float(-normal @ p0)
        inliers = np.abs(points @ normal + d) < thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers, best_normal, best_d = count, inliers, normal, d

    if best_inliers is None or best_count < 3:
        raise ValueError("RANSAC 没找到任何平面：检查 inlier 阈值是不是按米填的")

    # 最小二乘精化：用 SVD 取最小奇异向量当法向
    pts = points[best_inliers]
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[2]
    d = float(-normal @ centroid)
    inliers = np.abs(points @ normal + d) < thresh
    return normal.astype(np.float64), d, inliers


def label_clusters(depth: np.ndarray, tol_mm: float) -> tuple:
    """按「深度连续」切连通域，返回 (labels, 簇数量)。

    深度图是组织化点云：相邻像素在 3D 空间里本来就挨着，只有跨物体边界时
    深度才跳变。所以在跳变处切断，剩下的连通域就是欧式聚类的等价结果。

    比真正的欧式聚类快两个量级，也不需要 KD 树。
    """
    valid = depth > 0
    filled = np.where(valid, depth, 0).astype(np.float32)

    # 相邻像素的深度差超过 tol 视为边界；边界两侧的像素都不参与连通
    jump_h = np.abs(np.diff(filled, axis=1)) > tol_mm
    jump_v = np.abs(np.diff(filled, axis=0)) > tol_mm

    border = np.zeros_like(valid)
    border[:, :-1] |= jump_h
    border[:, 1:] |= jump_h
    border[:-1, :] |= jump_v
    border[1:, :] |= jump_v
    # 无效像素本身就是边界：depth==0 与有效值的差会大于 tol，但 0 参与 diff 时
    # 会污染判断，所以显式再排除一次
    border |= ~valid

    work = (valid & ~border).astype(np.uint8)
    count, labels = cv2.connectedComponents(work, connectivity=8)
    return labels, count - 1


def describe_cluster(points: np.ndarray) -> dict:
    """给一个簇算点数和包围盒（毫米）。"""
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return {
        "points": int(len(points)),
        "size_mm": (hi - lo).round(1).tolist(),
        "center_mm": ((hi + lo) / 2).round(1).tolist(),
    }


def fit_top_face(cluster_points: np.ndarray, plane_normal: np.ndarray,
                 plane_d: float, top_ratio: float = 0.2) -> tuple:
    """B4：只取离桌面最远的那层点拟合平面，返回 (中心, 法向, 用到的点数)。

    **不能拿整个簇拟合。** 物块是立体的，整簇点在三个面上，拟合出来的是一个
    斜着的平面，中心还落在方块中间而不是上表面。取顶部 top_ratio 才是吸附面。

    「顶部」按到桌面的距离算，不是按相机系 y —— 相机是斜着装的，
    用 y 当高度会把倾斜也带进去。
    """
    dist = np.abs(cluster_points @ plane_normal + plane_d)
    cutoff = np.percentile(dist, 100.0 * (1.0 - top_ratio))
    top_pts = cluster_points[dist >= cutoff]
    if len(top_pts) < 3:
        return None, None, 0

    centroid = top_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(top_pts - centroid, full_matrices=False)
    normal = vt[2]
    # 法向朝向没有约定，统一成与桌面法向同侧（朝相机），下游才好直接用
    if float(normal @ plane_normal) < 0:
        normal = -normal

    # 重心投影回拟合平面，保证吸附点确实在面上
    offset = float((centroid - top_pts[0]) @ normal)
    centroid = centroid - offset * normal
    return centroid.astype(np.float64), normal.astype(np.float64), int(len(top_pts))


def score_suction(depth: np.ndarray, k: dict, top_center: np.ndarray,
                  top_normal: np.ndarray, vertical: np.ndarray,
                  cluster_points: np.ndarray, plate_mm: float = 20.0) -> dict:
    """B5：按计划书 3.3 的五条判据给可吸性打分。

    五条里前四条都在「吸盘圆盘」范围内算，第五条看整簇点数。
    每条给一个 0~1 的子分，最终分数取最小值——任何一条不达标，
    吸盘就会漏气，用加权平均会把一个致命短板平均掉。
    """
    radius = plate_mm / 2.0
    z = float(top_center[2])
    if z <= 0:
        return {"score": 0.0, "detail": {}}

    # 吸附点投到图像上，圆盘半径换算成像素
    u = k["fx"] * top_center[0] / z + k["cx"]
    v = k["fy"] * top_center[1] / z + k["cy"]
    r_px = radius * k["fx"] / z

    height, width = depth.shape
    win = int(np.ceil(r_px)) + 2
    y0, x0 = int(round(v)), int(round(u))
    ys = slice(max(0, y0 - win), min(height, y0 + win + 1))
    xs = slice(max(0, x0 - win), min(width, x0 + win + 1))
    sub = depth[ys, xs]
    gy, gx = np.mgrid[ys.start:ys.stop, xs.start:xs.stop]
    in_disk = np.sqrt((gx - u) ** 2 + (gy - v) ** 2) <= r_px

    disk_total = int(in_disk.sum())
    if disk_total == 0:
        return {"score": 0.0, "detail": {}}

    disk_depth = sub[in_disk]
    # 判据 4：圆盘内空洞比例
    hole_ratio = float((disk_depth == 0).mean())

    valid = disk_depth > 0
    n_points = int(valid.sum())

    # 判据 2：圆盘内点到吸附平面的距离 < 3mm 的比例
    if n_points:
        vv, uu = np.mgrid[ys.start:ys.stop, xs.start:xs.stop]
        vv, uu = vv[in_disk][valid], uu[in_disk][valid]
        zz = disk_depth[valid].astype(np.float32)
        px = (uu - k["cx"]) * zz / k["fx"]
        py = (vv - k["cy"]) * zz / k["fy"]
        pts = np.stack([px, py, zz], axis=1)
        # 平面过吸附点、法向 top_normal：点到平面距离 = |(p - c)·n|
        offsets = np.abs((pts - top_center) @ top_normal)
        flat_ratio = float((offsets < 3.0).mean())
    else:
        flat_ratio = 0.0

    # 判据 3：法向与竖直方向夹角
    cos_angle = float(np.abs(top_normal @ vertical))
    angle = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    subs = {
        "surface_points": min(1.0, n_points / 120.0),
        "flatness": min(1.0, flat_ratio / 0.8),
        "tilt": 1.0 if angle <= 45.0 else max(0.0, 1.0 - (angle - 45.0) / 45.0),
        "holes": 1.0 if hole_ratio <= 0.2 else max(0.0, 1.0 - (hole_ratio - 0.2) / 0.8),
        "cluster_size": min(1.0, len(cluster_points) / 300.0),
    }
    score = min(subs.values())

    return {
        "score": round(float(score), 3),
        "detail": {
            "disk_points": n_points,
            "disk_points_need": 120,
            "flat_ratio": round(flat_ratio, 3),
            "flat_ratio_need": 0.8,
            "tilt_deg": round(angle, 1),
            "tilt_need_max": 45.0,
            "hole_ratio": round(hole_ratio, 3),
            "hole_ratio_need_max": 0.2,
            "cluster_points": int(len(cluster_points)),
            "cluster_points_need": 300,
        },
        "sub_scores": {key: round(val, 3) for key, val in subs.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="B1-B3 原型：反投影 + 降采样 + RANSAC 去桌面 + 聚类",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("data_dir", type=Path, help="含 depth.npy（和可选 meta.json）的目录")
    parser.add_argument("--fx", type=float, default=None, help="不给就从 meta.json 读")
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--roi", type=float, default=0.6,
                        help="中心 ROI 占原图的边长比例，1.0 表示用全图，默认 0.6")
    parser.add_argument("--leaf", type=float, default=DEFAULT_LEAF_MM, help="体素边长（毫米）")
    parser.add_argument("--plane-thresh", type=float, default=DEFAULT_PLANE_THRESH_MM,
                        help="RANSAC 桌面 inlier 阈值（毫米）")
    parser.add_argument("--cluster-tol", type=float, default=DEFAULT_CLUSTER_JUMP_MM,
                        help="深度跳变阈值（毫米）")
    parser.add_argument("--ransac-iters", type=int, default=200)
    parser.add_argument("--min-points", type=int, default=100,
                        help="簇点数低于它就算噪声，丢弃")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        depth, k, meta = load_data(args.data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"数据读不了：{exc}", file=sys.stderr)
        return 1

    # 命令行显式给的内参优先于 meta
    for key in ("fx", "fy", "cx", "cy"):
        value = getattr(args, key)
        if value is not None:
            k[key] = value

    rng = np.random.default_rng(args.seed)

    full_shape = depth.shape
    total_pixels = int(np.count_nonzero(depth))

    # ---- B1 第一步：ROI 裁剪 ----
    # 全图 848x480 = 40 万点，直接做 200 次 RANSAC 是秒级，进 ROS 节点（15Hz）
    # 根本跑不动。裁掉四周既提速，也顺带躲开视场边缘那圈噪声更大、失效更多的区域。
    # 注意内参主点要跟着平移，否则反投影会整体错位。
    if args.roi < 1.0:
        h, w = depth.shape
        roi_w, roi_h = int(w * args.roi), int(h * args.roi)
        x0, y0 = (w - roi_w) // 2, (h - roi_h) // 2
        depth = depth[y0:y0 + roi_h, x0:x0 + roi_w]
        k = dict(k, cx=k["cx"] - x0, cy=k["cy"] - y0)

    points, pixels = backproject(depth, k)
    print("=" * 62)
    print(f"数据: {args.data_dir}")
    print(f"  原图      : {full_shape[1]}x{full_shape[0]}, 有效像素 {total_pixels}")
    print(f"ROI 裁剪  : {args.roi:.2f} -> {depth.shape[1]}x{depth.shape[0]}, "
          f"剩 {len(points)} 点（原始有效点的 {len(points) / max(total_pixels, 1) * 100:.1f}%）")

    # ---- B1 第二步：体素降采样 ----
    down = voxel_downsample(points, args.leaf)
    ratio = len(down) / max(total_pixels, 1)
    print(f"B1 降采样 : leaf={args.leaf}mm -> {len(down)} 点"
          f"（相对原图有效点共 1/{1 / max(ratio, 1e-9):.1f}，"
          f"相对 ROI 内点 1/{len(points) / max(len(down), 1):.1f}）")

    # ---- B2 RANSAC 去桌面 ----
    t_start = time.perf_counter()
    try:
        normal, d, table_inliers = ransac_plane(down, args.plane_thresh,
                                                args.ransac_iters, rng)
    except ValueError as exc:
        print(f"B2 失败：{exc}", file=sys.stderr)
        return 1
    ransac_ms = (time.perf_counter() - t_start) * 1000.0

    normal = normal / np.linalg.norm(normal)
    # 法向方向没有约定，夹角必须取绝对值，否则会得到 180°
    if meta:
        vertical = np.asarray(meta["objects"][0]["top_normal_cam"], dtype=np.float64)
        vertical /= np.linalg.norm(vertical)
    else:
        vertical = np.array([0.0, 0.0, -1.0])
    cos_angle = float(np.abs(normal @ vertical))
    angle = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    plane_z = float(down[table_inliers][:, 2].mean())

    print(f"B2 去桌面 : inlier {int(table_inliers.sum())} 点（{table_inliers.mean() * 100:.1f}%），"
          f"法向 [{normal[0]:.3f} {normal[1]:.3f} {normal[2]:.3f}]，"
          f"耗时 {ransac_ms:.0f}ms")
    print(f"            与竖直方向夹角 {angle:.2f}°"
          f" {'[OK <3°]' if angle < 3.0 else '[WARN 超 3°]'}")
    print(f"            平面平均深度 {plane_z:.1f}mm")

    # ---- B3 聚类 ----
    # 把桌面点所在的像素抹掉，再按深度连续性切连通域
    table_mask = np.zeros(depth.shape, dtype=bool)
    dist = np.abs(points @ normal + d)
    on_table = dist < args.plane_thresh
    table_mask[pixels[on_table, 1], pixels[on_table, 0]] = True

    remaining = depth.copy()
    remaining[table_mask] = 0

    labels, n_clusters = label_clusters(remaining, args.cluster_tol)
    print(f"B3 聚类   : 去掉桌面后剩 {int(np.count_nonzero(remaining))} 像素，"
          f"得到 {n_clusters} 个候选簇")

    clusters = []
    for label in range(1, labels.max() + 1):
        mask = labels == label
        count = int(mask.sum())
        if count < args.min_points:
            continue
        v, u = np.nonzero(mask)
        z = depth[v, u].astype(np.float32)
        pts = np.stack([(u - k["cx"]) * z / k["fx"],
                        (v - k["cy"]) * z / k["fy"], z], axis=1)
        info = describe_cluster(pts)
        info["label"] = int(label)

        # ---- B4 顶面拟合 ----
        top_center, top_normal, n_top = fit_top_face(pts, normal, d)
        if top_center is not None:
            info["top_center_mm"] = top_center.round(1).tolist()
            info["top_normal"] = top_normal.round(3).tolist()
            info["top_points_used"] = n_top
            # ---- B5 可吸性 ----
            info["suction"] = score_suction(depth, k, top_center, top_normal,
                                            vertical, pts)
        clusters.append(info)

    clusters.sort(key=lambda item: item["points"], reverse=True)
    for rank, info in enumerate(clusters, start=1):
        print(f"  簇{rank}: {info['points']} 点, 包围盒 {info['size_mm']} mm, "
              f"中心 {info['center_mm']} mm")
        if "top_center_mm" in info:
            print(f"       B4 顶面: 中心 {info['top_center_mm']} mm, "
                  f"法向 {info['top_normal']}, 用顶部 {info['top_points_used']} 点拟合")
            detail = info["suction"]["detail"]
            print(f"       B5 可吸: score={info['suction']['score']}  "
                  f"（取五项最小值）")
            print(f"          圆盘点数 {detail['disk_points']}/{detail['disk_points_need']} | "
                  f"平整 {detail['flat_ratio']}/{detail['flat_ratio_need']} | "
                  f"倾斜 {detail['tilt_deg']}°/{detail['tilt_need_max']}° | "
                  f"空洞 {detail['hole_ratio']}/{detail['hole_ratio_need_max']} | "
                  f"簇点 {detail['cluster_points']}/{detail['cluster_points_need']}")

    print("-" * 62)
    verdicts = [
        f"B1 ROI+降采样 合计 1/{1 / max(ratio, 1e-9):.1f}（目标 ≤1/10）",
        f"B2 桌面法向夹角 {angle:.2f}°（<3° 为达标）",
        f"B3 簇数 {len(clusters)}（单目标场景应为 1）",
    ]
    for line in verdicts:
        print("  " + line)

    if meta and clusters and "top_center_mm" in clusters[0]:
        truth = np.asarray(meta["objects"][0]["top_center_cam_mm"], dtype=np.float64)
        found = np.asarray(clusters[0]["top_center_mm"], dtype=np.float64)
        error = float(np.linalg.norm(found - truth))
        print(f"  B4 与真值: 拟合 {found.round(1).tolist()} vs 真值 {truth.round(1).tolist()}"
              f" -> 位置误差 {error:.2f} mm")
        print(f"  B5 可吸性: {clusters[0]['suction']['score']}"
              f"（可吸与不可吸应给出明显不同的分数）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
