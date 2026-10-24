"""深度图 -> 抓取目标：纯几何算法，不 import ROS，不做 IO。

用大白话说
    给一张深度图和内参，找出桌面上凸起的物块，输出「吸附点 + 法向 + 可吸性分数」。
    不依赖 YOLO，也不依赖彩色图——第零段只靠几何就能把 demo 跑起来。

放什么 / 不放什么
    只放算法。读 bag、读 npy、命令行参数都在 `lab/`；收消息发消息在 ROS 包。
    这样同一份代码能被 lab 脚本、ROS 节点、单元测试三处共用，不需要复制。

五步链路（对应计划书 B1~B5）
    1. ROI 裁剪 + 体素降采样：40 万点降到十分之一，RANSAC 从秒级降到十几毫秒
    2. RANSAC 去桌面：桌面是视场里最大的平面，天然是最大的 inlier 集合
    3. 聚类：利用深度图是**组织化点云**的特性，在深度跳变处切断连通域
    4. 顶面拟合：只取离桌面最远的那层点，整簇拟合会得到一个斜平面
    5. 可吸性打分：五条判据取最小值

深度单位
    全程毫米。Orbbec 的 16UC1 配 `enable_depth_scale=true` 输出就是毫米，
    所以内参和深度不需要额外换算。若上游改成米，点云会整体缩小一千倍。

已知的坑
    · ROI 裁剪后内参主点必须同步平移，否则反投影整体错位。这是本模块里
      最容易埋进去、又最难从结果上看出来的错。
    · RANSAC 阈值单位是毫米。按米填（比如 0.006）会让每个点都算外点，
      表现是「找不到桌面」，而不是报错。
    · 平面法向方向没有约定（n 和 -n 都是解），算夹角必须取绝对值。
    · 深度跳变阈值太小时噪声会在物体内部切断连通域，一个物块裂成几块；
      太大时物块和桌面连成一片。经验 15~25mm，比噪声大一个量级。
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import GraspTarget, Intrinsics, SuctionParams

#: 判定「相邻像素属于同一物体」时，深度差的上限是参数里的 cluster_jump_mm。
#: 这里只放与算法结构绑定的常量。
_MIN_PLANE_POINTS = 3


def _validate(depth: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """校验输入尺寸与内参一致，返回 float32 的深度副本。

    尺寸不一致时直接抛错而不是警告：那意味着内参和图像不是同一次配置，
    点云会整体错位，而这种错在结果上表现为「物块位置偏了几厘米」，
    很难联想到是尺寸问题。
    """
    if depth.ndim != 2:
        raise ValueError(f"depth 必须是二维 (H, W)，收到 {depth.shape}")
    height, width = depth.shape
    if (width, height) != (int(intrinsics.width), int(intrinsics.height)):
        raise ValueError(
            f"深度图尺寸 {width}x{height} 与内参的 "
            f"{intrinsics.width}x{intrinsics.height} 不一致；"
            f"换分辨率后要同步换内参，不能只改一边"
        )
    return depth.astype(np.float32, copy=False)


def backproject(depth_mm: np.ndarray, intrinsics: Intrinsics) -> tuple:
    """深度图 -> 点云。返回 (points (N,3) 毫米, pixel_uv (N,2))。

    深度是 z 深度（沿光轴），所以是 x=(u-cx)z/fx、y=(v-cy)z/fy 的直接代入。
    把 z 深度当欧氏距离会让点云边缘整体外扩——两者只在图像正中心相等。
    """
    v, u = np.nonzero(depth_mm > 0)
    z = depth_mm[v, u].astype(np.float32)
    x = (u.astype(np.float32) - intrinsics.cx) * z / intrinsics.fx
    y = (v.astype(np.float32) - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=1), np.stack([u, v], axis=1)


def voxel_downsample(points: np.ndarray, leaf_mm: float) -> np.ndarray:
    """体素降采样：按 leaf 分格，每格用落点均值代表。

    用 np.unique 的 return_inverse 做分格，再用 bincount 求均值。
    比逐格 Python 循环快两个量级，也不需要 scipy。
    """
    if leaf_mm <= 0 or len(points) == 0:
        return points
    keys = np.floor(points / leaf_mm).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    n_vox = len(counts)
    sums = np.zeros((n_vox, 3), dtype=np.float64)
    for axis in range(3):
        sums[:, axis] = np.bincount(inverse, weights=points[:, axis], minlength=n_vox)
    return (sums / counts[:, None]).astype(np.float32)


def fit_plane_ransac(points: np.ndarray, thresh_mm: float, iters: int,
                     rng: np.random.Generator) -> tuple:
    """RANSAC 找最大平面，返回 (法向, 偏移 d, inlier 掩码)，平面写作 n·p + d = 0。

    三点采样得到的平面受点噪声影响很大，所以找到内点后用它做一次最小二乘精化
    （SVD 取最小奇异向量），否则法向会在多次调用之间跳动。
    """
    n_pts = len(points)
    if n_pts < _MIN_PLANE_POINTS:
        raise ValueError(f"点太少（{n_pts}），无法拟合平面")

    best_count = 0
    best_inliers = None
    best_normal = None
    best_d = 0.0

    for _ in range(max(int(iters), 1)):
        idx = rng.choice(n_pts, _MIN_PLANE_POINTS, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = float(-normal @ p0)
        inliers = np.abs(points @ normal + d) < thresh_mm
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers, best_normal, best_d = count, inliers, normal, d

    if best_inliers is None or best_count < _MIN_PLANE_POINTS:
        raise ValueError(
            "RANSAC 找不到任何平面：检查 plane_thresh_mm 是不是按米填的（应为毫米）"
        )

    pts = points[best_inliers]
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[2].astype(np.float64)
    d = float(-normal @ centroid)
    inliers = np.abs(points @ normal + d) < thresh_mm
    return normal, d, inliers


def label_clusters(depth_mm: np.ndarray, jump_mm: float) -> np.ndarray:
    """按「深度连续」切连通域，返回标签图（0 表示背景/边界）。

    深度图是组织化点云：相邻像素在 3D 空间本来就挨着，只有跨物体边界时深度
    才跳变。所以在跳变处切断，剩下的连通域就是欧式聚类的等价结果，
    而且只用一遍 cv2.connectedComponents，比 KD 树聚类快两个量级。
    """
    valid = depth_mm > 0
    filled = np.where(valid, depth_mm, 0).astype(np.float32)

    jump_h = np.abs(np.diff(filled, axis=1)) > jump_mm
    jump_v = np.abs(np.diff(filled, axis=0)) > jump_mm

    border = np.zeros_like(valid)
    border[:, :-1] |= jump_h
    border[:, 1:] |= jump_h
    border[:-1, :] |= jump_v
    border[1:, :] |= jump_v
    border |= ~valid

    work = (valid & ~border).astype(np.uint8)
    _, labels = cv2.connectedComponents(work, connectivity=8)
    return labels


def fit_top_face(points: np.ndarray, plane_normal: np.ndarray, plane_d: float,
                 top_ratio: float) -> tuple:
    """取离桌面最远的一层点拟合平面，返回 (吸附点, 法向, 用到的点数)。

    **不能拿整个簇拟合。** 物块是立体的，整簇点在多个面上，拟合出来是个斜平面，
    中心还会落在方块内部。取顶部一层才是真正的吸附面。

    「顶部」按到桌面的距离算，不是按相机系 y —— 相机是斜着装的，
    用 y 当高度会把安装倾角混进判断里。
    """
    dist = np.abs(points @ plane_normal + plane_d)
    cutoff = float(np.percentile(dist, 100.0 * (1.0 - top_ratio)))
    top_pts = points[dist >= cutoff]
    if len(top_pts) < _MIN_PLANE_POINTS:
        return None, None, 0

    centroid = top_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(top_pts - centroid, full_matrices=False)
    normal = vt[2]
    # 统一朝相机一侧。判据用「指向原点」（相机就在原点），不用「与桌面法向同向」——
    # 后者依赖 RANSAC 给出的符号，而那个符号本身是没有约定的。
    if float(normal @ (-centroid)) < 0:
        normal = -normal

    # 把重心投影回拟合平面，保证吸附点确实落在面上
    offset = float((centroid - top_pts[0]) @ normal)
    centroid = centroid - offset * normal
    return centroid.astype(np.float64), normal.astype(np.float64), int(len(top_pts))


def score_suction(depth_mm: np.ndarray, intrinsics: Intrinsics, top_center: np.ndarray,
                  top_normal: np.ndarray, vertical: np.ndarray,
                  cluster_points: np.ndarray, params: SuctionParams) -> tuple:
    """按计划书 3.3 的五条判据打可吸性分数，返回 (分数, 明细 dict)。

    前四条在吸盘圆盘范围内算，第五条看整簇点数。最终分数取五项最小值：
    任何一条不达标吸盘就会漏气，加权平均会把某个致命短板平均掉。
    """
    radius = params.plate_mm / 2.0
    z = float(top_center[2])
    if z <= 0:
        return 0.0, {}

    # 吸附点投到图像上，圆盘半径换算成像素
    u = intrinsics.fx * top_center[0] / z + intrinsics.cx
    v = intrinsics.fy * top_center[1] / z + intrinsics.cy
    r_px = radius * intrinsics.fx / z

    height, width = depth_mm.shape
    win = int(np.ceil(r_px)) + 2
    y0, x0 = int(round(v)), int(round(u))
    ys = slice(max(0, y0 - win), min(height, y0 + win + 1))
    xs = slice(max(0, x0 - win), min(width, x0 + win + 1))
    sub = depth_mm[ys, xs]
    gy, gx = np.mgrid[ys.start:ys.stop, xs.start:xs.stop]
    in_disk = np.sqrt((gx - u) ** 2 + (gy - v) ** 2) <= r_px

    disk_total = int(in_disk.sum())
    if disk_total == 0:
        return 0.0, {}

    disk_depth = sub[in_disk]
    hole_ratio = float((disk_depth == 0).mean())

    valid = disk_depth > 0
    n_points = int(valid.sum())

    if n_points:
        vv = gy[in_disk][valid]
        uu = gx[in_disk][valid]
        zz = disk_depth[valid].astype(np.float32)
        pts = np.stack([(uu - intrinsics.cx) * zz / intrinsics.fx,
                        (vv - intrinsics.cy) * zz / intrinsics.fy, zz], axis=1)
        # 点到吸附平面的距离：(p - c)·n
        offsets = np.abs((pts - top_center) @ top_normal)
        flat_ratio = float((offsets < params.flat_distance_mm).mean())
    else:
        flat_ratio = 0.0

    cos_angle = float(np.abs(top_normal @ vertical))
    tilt_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    subs = {
        "surface_points": min(1.0, n_points / max(params.min_surface_points, 1)),
        "flatness": min(1.0, flat_ratio / max(params.flat_ratio_min, 1e-6)),
        "tilt": 1.0 if tilt_deg <= params.tilt_max_deg
                else max(0.0, 1.0 - (tilt_deg - params.tilt_max_deg) / params.tilt_max_deg),
        "holes": 1.0 if hole_ratio <= params.hole_ratio_max
                 else max(0.0, 1.0 - (hole_ratio - params.hole_ratio_max)
                          / max(1.0 - params.hole_ratio_max, 1e-6)),
        "cluster_size": min(1.0, len(cluster_points) / max(params.min_cluster_points, 1)),
    }
    detail = {
        "disk_points": n_points,
        "flat_ratio": round(flat_ratio, 3),
        "tilt_deg": round(tilt_deg, 1),
        "hole_ratio": round(hole_ratio, 3),
        "cluster_points": int(len(cluster_points)),
        "limiting": min(subs, key=subs.get),
    }
    return float(min(subs.values())), detail


def analyze(depth_mm: np.ndarray, intrinsics: Intrinsics,
            params: SuctionParams | None = None,
            rng: np.random.Generator | None = None) -> dict:
    """跑完整链路，返回中间结果与目标列表（调试和离线评估用）。

    与 `detect_grasp_targets` 的区别：这个把平面方程、簇数、每个簇的明细都
    返回出来，方便定位「到底是分割错了还是打分了」。ROS 节点用薄封装那个就行。
    """
    params = params or SuctionParams()
    rng = rng or np.random.default_rng(0)
    depth = _validate(depth_mm, intrinsics)

    # ---- 1. ROI 裁剪 ----
    # 全图 40 万点做 200 次 RANSAC 是秒级，15Hz 节点根本来不及。
    # 裁剪同时也躲开了视场边缘噪声更大、失效更多的区域。
    height, width = depth.shape
    roi_w, roi_h = int(width * params.roi_ratio), int(height * params.roi_ratio)
    x0, y0 = (width - roi_w) // 2, (height - roi_h) // 2
    roi = depth[y0:y0 + roi_h, x0:x0 + roi_w]
    # 主点必须跟着平移，否则整张点云错位
    k_roi = Intrinsics(fx=intrinsics.fx, fy=intrinsics.fy,
                       cx=intrinsics.cx - x0, cy=intrinsics.cy - y0,
                       width=roi_w, height=roi_h)

    points, pixels = backproject(roi, k_roi)
    result = {"points_total": int(len(points)), "plane": None, "clusters": [],
              "targets": [], "reason": None}
    if len(points) < 50:
        result["reason"] = f"ROI 内有效点太少（{len(points)}）"
        return result

    # ---- 2. 降采样 + RANSAC ----
    down = voxel_downsample(points, params.leaf_mm)
    result["points_downsampled"] = int(len(down))
    try:
        normal, d, inliers = fit_plane_ransac(down, params.plane_thresh_mm,
                                              params.ransac_iters, rng)
    except ValueError as exc:
        result["reason"] = str(exc)
        return result

    # 平面法向也统一朝相机。RANSAC 给出的 n 和 -n 都是合法解，不统一的话
    # 下游算「法向与竖直夹角」时符号会随机变，虽然取了绝对值不影响角度，
    # 但输出到 ROS 的朝向会时正时反，控制组那边会被坑。
    centroid_down = down.mean(axis=0)
    if float(normal @ (-centroid_down)) < 0:
        normal = -normal
        d = -d

    result["plane"] = {
        "normal": normal.round(4).tolist(),
        "d": round(float(d), 2),
        "inlier_ratio": round(float(inliers.mean()), 3),
    }

    # ---- 3. 去掉桌面再聚类 ----
    on_table = np.abs(points @ normal + d) < params.plane_thresh_mm
    remaining = roi.copy()
    remaining[pixels[on_table, 1], pixels[on_table, 0]] = 0

    labels = label_clusters(remaining, params.cluster_jump_mm)

    targets = []
    for label in range(1, int(labels.max()) + 1):
        mask = labels == label
        count = int(mask.sum())
        if count < params.min_cluster_points:
            continue

        v_idx, u_idx = np.nonzero(mask)
        z = roi[v_idx, u_idx].astype(np.float32)
        pts = np.stack([(u_idx - k_roi.cx) * z / k_roi.fx,
                        (v_idx - k_roi.cy) * z / k_roi.fy, z], axis=1)

        top_center, top_normal, n_top = fit_top_face(pts, normal, d, params.top_ratio)
        if top_center is None:
            continue

        # 竖直方向取桌面法向本身：物块顶面就该平行于桌面
        score, detail = score_suction(roi, k_roi, top_center, top_normal,
                                      normal, pts, params)
        lo, hi = pts.min(axis=0), pts.max(axis=0)

        result["clusters"].append({
            "label": label, "points": count,
            "top_points": n_top,
            "top_center_mm": top_center.round(2).tolist(),
            "top_normal": top_normal.round(4).tolist(),
            "bbox_size_mm": (hi - lo).round(2).tolist(),
            "score": round(score, 3),
            "score_detail": detail,
        })
        targets.append(GraspTarget(
            position_mm=top_center,
            normal=top_normal,
            score=score,
            bbox_size_mm=hi - lo,
            points=count,
            target_id=len(targets),
        ))

    targets.sort(key=lambda item: item.score, reverse=True)
    for index, target in enumerate(targets):
        object.__setattr__(target, "target_id", index)

    result["targets"] = targets
    return result


def detect_grasp_targets(depth_mm: np.ndarray, intrinsics: Intrinsics,
                         params: SuctionParams | None = None,
                         rng: np.random.Generator | None = None) -> list:
    """门面函数：深度图 -> 抓取目标列表（按可吸性分数降序）。

    找不到桌面或没有合格目标时返回空列表，不抛异常——场景里本来就可能没有
    可吸的东西，节点不该因为一帧没找到就崩。真正的输入错误（尺寸不符等）
    仍然抛 ValueError，因为那属于配置问题。
    """
    result = analyze(depth_mm, intrinsics, params, rng)
    return result["targets"]
