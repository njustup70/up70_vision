"""从真实 bag 跑抓取检测，并用直尺/水平仪量到的真值核对结果（M2 入口）。

跑法（容器内）
    docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision \
        python lab/bag_detect.py bags/raw_002 \
            --truth-height-mm 150 --truth-size-mm 150 148 150 --truth-tilt-deg 42

    # 只关心检出、不带真值（检验 1/2/3 跳过，仍会跑 4/5）
    ... python lab/bag_detect.py bags/raw_002

    # 只取前 40 帧做时间中位数；把合成后的深度图存下来，便于反复调参
    ... python lab/bag_detect.py bags/raw_002 --frames 40 --save data/real/box_01.npy

五条检验（每条都配一个「独立输入」，不独立就没有检验力）
    1 盒高   顶面中心到拟合桌面平面的距离  vs  直尺量的盒高        容差 2mm
    2 倾角   由桌面法向反推的光轴倾角        vs  手机水平仪读数      容差 3°
    3 尺寸   顶面平面内的二维跨度            vs  直尺量的长宽        容差 5mm
    4 平行   顶面法向与桌面法向的夹角        vs  0°（盒子顶面是平的） 容差 3°
    5 检出   簇数、逐帧位置/分数抖动         vs  1 个目标、抖动要小

    为什么第 1 条不用 Δ=H·cos(tilt)：真机上「物块正下方那块桌面」被挡住了，
    看不见；任何构造它的办法都要用到拟合出的竖直方向，结果会变成恒等式，
    误差永远是 0。改成「点到平面距离 = 盒高」后，输入只有直尺量出来的 H，
    输出是深度图算出来的距离，两边独立。
    合成数据里能用 Δ=H·cos(tilt)，是因为渲染器把被遮挡的桌面深度也存下来了
    （见 lab/gen_synthetic_depth.py 的 raycast 返回的 t_table）。

为什么要做时间中位数
    真实相机的单帧噪声虽然只有零点几毫米，但降采样、平面拟合、聚类都是非线性步骤，
    单帧偶尔会跳一下（多一个碎簇、顶面法向偏两度）。静止场景下相机不动、物块不动，
    把 N 帧逐像素取中位数，噪声按 sqrt(N) 下降，而且顺手把随机飞点抹掉。

    这不是「作弊」——真实部署时相机拍一帧就够快，但离线验证阶段先把算法本身的
    问题和不稳定因素分开排掉，比一上来就做时序滤波更容易定位问题。

为什么不用 lab/bag_to_images.py
    那个工具把帧存成 png 给人看，16bit 深度存成图之后就丢了内参，而反投影必须要
    内参。这里直接从 bag 读原始消息，内参一起拿出来。

已知的坑
    · 时间中位数必须先剔除 0（无效深度）。不剔除的话，中位数会被 0 拉向无效值，
      表现是大片区域深度变成 0 或异常小。
    · 有效帧比例低于一半时，中位数没有意义（多数帧都是空的），脚本会警告。
    · bag 里的深度话题名不一定带前缀，找不到时会列出实际存在的话题。
    · 检验 3 有固有偏置：顶面薄片会带进正面靠近顶边的一圈点，同时盒边混合像素
      被排除，两者方向相反、各差几毫米。若只有检验 3 报 5~8mm 而 1/2/4 都过，
      先怀疑这个偏置，不要去改 fx。
    · 逐帧抖动用的是**单帧原图**，不是时间中位数；两者数字不一样是正常的。

退出码：0 全部通过（或有检验因缺真值跳过）；1 有检验不通过/读不到数据/算法报错；
        2 参数非法。
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.depth import Intrinsics, SuctionParams, analyze  # noqa: E402

DEPTH_TOPIC = "/camera/depth/image_raw"
INFO_TOPIC = "/camera/depth/camera_info"

#: 各条检验的容差。定这么严是有依据的：相机在 0.5m 处噪声约 0.4~1.7mm（A4 实测），
#: 而真值来自直尺（±1mm）与水平仪（±1~2°）。容差比误差源小就没有意义，
#: 比它大太多又抓不住问题，所以按「3 倍误差源」取。
TOL_HEIGHT_MM = 2.0
TOL_TILT_DEG = 3.0
TOL_SIZE_MM = 5.0
TOL_PARALLEL_DEG = 3.0
TOL_JITTER_MM = 3.0          # 单帧位置抖动，3σ 量级

#: 深度图在每处深度跳变边上必然丢掉的像素数（单边）。
#: 来源是三步叠加：驱动/生成器的边缘置零 1 圈、跳变检测把两侧各标 1 圈、
#: 连通域再把贴着空洞的一圈当边界剔掉，实测约 3px。
#: 顶面跨度是「最外侧点之间的距离」，所以每条边都会少这么一截；
#: 这是深度相机的物理限制，不是算法错，因此按像素换算成毫米加进容差。
EDGE_LOSS_PX = 3

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass(frozen=True)
class Truth:
    """手工真值。只有直尺/水平仪能提供的那几个数，全部来自测量，不来自算法。"""

    height_mm: float | None = None
    size_mm: tuple | None = None       # (长, 宽, 高)，直尺量的三边
    tilt_deg: float | None = None      # 光轴偏离竖直的角度，手机水平仪读数
    n_targets: int = 1                 # 场景里摆了几个可吸物块


def decode_depth(msg) -> np.ndarray:
    """sensor_msgs/Image(16UC1) -> (H, W) uint16 毫米。

    step 不一定等于 width*2：驱动可能按行对齐补 padding，直接按 width*2 铺开
    会让图像斜着错位，而且错位之后「看起来还有内容」，很难发现。
    """
    if str(msg.encoding) not in ("16UC1", "mono16"):
        raise ValueError(f"深度编码是 {msg.encoding!r}，只处理 16UC1（毫米）")

    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    raw = bytes(msg.data)
    row_bytes = width * 2

    if step == row_bytes:
        flat = np.frombuffer(raw, dtype=np.uint16)
    else:
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)[:, :row_bytes]
        flat = np.ascontiguousarray(rows).view(np.uint16)
    return flat.reshape(height, width).copy()


def load_bag(bag_dir: Path, max_frames: int, depth_topic: str,
             info_topic: str) -> tuple:
    """读 bag，返回 (深度帧列表, 内参 dict, 话题统计)。"""
    from rosbags.highlevel import AnyReader

    if not bag_dir.exists():
        raise FileNotFoundError(f"bag 不存在：{bag_dir}")

    frames = []
    info = None
    total = 0

    with AnyReader([bag_dir]) as reader:
        depth_conns = [c for c in reader.connections if c.topic == depth_topic]
        if not depth_conns:
            topics = sorted({c.topic for c in reader.connections})
            raise ValueError(f"{bag_dir} 里没有 {depth_topic}。实际话题：{topics}")

        info_conns = [c for c in reader.connections if c.topic == info_topic]
        if info_conns:
            for conn, _, raw in reader.messages(connections=info_conns):
                msg = reader.deserialize(raw, conn.msgtype)
                info = {"fx": float(msg.k[0]), "fy": float(msg.k[4]),
                        "cx": float(msg.k[2]), "cy": float(msg.k[5]),
                        "width": int(msg.width), "height": int(msg.height)}
                break

        for conn, _, raw in reader.messages(connections=depth_conns):
            total += 1
            if len(frames) >= max_frames:
                continue                     # 继续计数，让「总帧数」是准的
            frames.append(decode_depth(reader.deserialize(raw, conn.msgtype)))

    if not frames:
        raise ValueError(f"{bag_dir} 的深度话题里没有消息")
    if info is None:
        raise ValueError(
            f"bag 里没有 {info_topic}，无法反投影。"
            f"录 bag 时要把 camera_info 一起录进来"
        )
    return frames, info, total


def temporal_median(frames: list) -> tuple:
    """逐像素时间中位数，返回 (合成深度图, 有效帧比例图的中位数)。

    无效深度（0）必须先剔除：留着 0 一起取中位数，结果会被拉向 0，
    表现是大片区域深度变成 0 或异常小，看起来像「相机坏了」。
    """
    stack = np.stack(frames).astype(np.float32)
    valid = stack > 0
    stack[~valid] = np.nan

    with np.errstate(invalid="ignore"):
        median = np.nanmedian(stack, axis=0)

    coverage = float(valid.mean())
    median = np.where(np.isfinite(median), median, 0.0)
    return np.round(median).astype(np.uint16), coverage


def project(point_mm: np.ndarray, intrinsics: Intrinsics) -> tuple:
    """相机系毫米点 -> 全图像素坐标（反投影的逆运算，只给画图用）。"""
    z = float(point_mm[2])
    if z <= 0:
        return None
    u = intrinsics.fx * float(point_mm[0]) / z + intrinsics.cx
    v = intrinsics.fy * float(point_mm[1]) / z + intrinsics.cy
    return u, v


def _row(name: str, measured: str, expected: str, error: str, status: str) -> dict:
    return {"check": name, "measured": measured, "expected": expected,
            "error": error, "status": status}


def evaluate_checks(result: dict, truth: Truth, intrinsics: Intrinsics) -> list:
    """由一次 analyze 的结果与手工真值算出五条检验（第 5 条只判簇数）。"""
    rows = []
    plane = result["plane"]
    clusters = result["clusters"]

    if not clusters:
        rows.append(_row("1 盒高", "-", "-", "-", FAIL))
        rows.append(_row("2 倾角", "-", "-", "-", FAIL))
        rows.append(_row("3 尺寸", "-", "-", "-", FAIL))
        rows.append(_row("4 顶面平行", "-", "-", "-", FAIL))
        rows.append(_row("5 检出", "0 簇", f"{truth.n_targets} 簇", "-", FAIL))
        return rows

    # 取可吸性最高的那一簇作为「目标物块」；检验 1~4 都针对它
    best = max(clusters, key=lambda c: c["score"])
    normal = np.asarray(plane["normal"], dtype=np.float64)
    d = float(plane["d"])
    center = np.asarray(best["top_center_mm"], dtype=np.float64)
    top_normal = np.asarray(best["top_normal"], dtype=np.float64)

    # ---- 1 盒高：点到桌面平面的距离 ----
    if truth.height_mm is None:
        rows.append(_row("1 盒高", "-", "未提供 --truth-height-mm", "-", SKIP))
    else:
        distance = abs(float(normal @ center + d))
        error = distance - truth.height_mm
        status = PASS if abs(error) <= TOL_HEIGHT_MM else FAIL
        rows.append(_row("1 盒高", f"{distance:.1f} mm",
                         f"{truth.height_mm:.1f} mm（直尺）",
                         f"{error:+.1f} mm", status))

    # ---- 2 倾角：桌面法向与相机光轴的夹角 ----
    # 桌面是水平的，所以相机系里桌面法向与 z 轴的夹角就是光轴偏离竖直的角度。
    if truth.tilt_deg is None:
        rows.append(_row("2 倾角", "-", "未提供 --truth-tilt-deg", "-", SKIP))
    else:
        cos = float(min(1.0, abs(normal[2]) / max(float(np.linalg.norm(normal)), 1e-9)))
        tilt_fit = float(np.degrees(np.arccos(cos)))
        error = tilt_fit - truth.tilt_deg
        status = PASS if abs(error) <= TOL_TILT_DEG else FAIL
        rows.append(_row("2 倾角", f"{tilt_fit:.1f}°",
                         f"{truth.tilt_deg:.1f}°（水平仪）",
                         f"{error:+.1f}°", status))

    # ---- 3 尺寸：顶面平面内的跨度 ----
    # 只比长、宽两个方向；高度由检验 1 负责，这里比第三个维度是重复。
    # 容差 = 基础噪声 + 每条边丢掉的像素换算成毫米（见 EDGE_LOSS_PX）。
    # 这是五条里最弱的一条：量到的只是「还留有深度的那块顶面」，边缘那一圈
    # 在真实相机上同样会缺。若只有它超差而 1/2/4 都过，先怀疑边缘缺省，
    # 不要去改 fx。
    size = best.get("top_size_mm") or [0.0, 0.0]
    pixel_mm = float(center[2]) / intrinsics.fx if intrinsics.fx > 0 else 0.0
    tol_size = TOL_SIZE_MM + 2 * EDGE_LOSS_PX * pixel_mm
    if truth.size_mm is None:
        rows.append(_row("3 尺寸", f"{size[0]:.1f}×{size[1]:.1f} mm",
                         "未提供 --truth-size-mm", "-", SKIP))
    else:
        truth_sides = sorted((float(truth.size_mm[0]), float(truth.size_mm[1])),
                             reverse=True)
        err_long = size[0] - truth_sides[0]
        err_short = size[1] - truth_sides[1]
        worst = max(abs(err_long), abs(err_short))
        status = PASS if worst <= tol_size else FAIL
        rows.append(_row("3 尺寸", f"{size[0]:.1f}×{size[1]:.1f} mm",
                         f"{truth_sides[0]:.1f}×{truth_sides[1]:.1f} mm"
                         f"（直尺，容差 {tol_size:.1f}）",
                         f"{err_long:+.1f}/{err_short:+.1f} mm", status))

    # ---- 4 顶面平行：盒子顶面应当与桌面平行 ----
    cos_par = float(min(1.0, abs(top_normal @ normal)))
    angle = float(np.degrees(np.arccos(cos_par)))
    status = PASS if angle <= TOL_PARALLEL_DEG else FAIL
    rows.append(_row("4 顶面平行", f"{angle:.1f}°", "0°（容差 3°）",
                     f"{angle:+.1f}°", status))

    # ---- 5 检出：可吸候选数 ----
    # 不能直接数簇数：真实场景里桌面的边角、远处的杂物都会成为簇，
    # 空场景实测就有十几个。这些簇的可吸性分数都是 0（平整度秒否），
    # 控制组也拿不到它们，所以「目标数」只数分数大于 0 的候选。
    candidates = [c for c in clusters if c["score"] > 0.0]
    count = len(candidates)
    status = PASS if count == truth.n_targets else FAIL
    rows.append(_row("5 检出", f"{count} 个候选（共 {len(clusters)} 簇）",
                     f"{truth.n_targets} 个", "-", status))

    return rows


def jitter_stats(frames: list, intrinsics: Intrinsics, params: SuctionParams,
                 seed: int) -> dict:
    """逐帧单独跑一次检测，统计位置与分数的抖动。

    用单帧而不是时间中位数：时间中位数是「静止场景的理想输入」，
    它会把抖动抹平，测出来的稳定性会好得不像真的。这里要看的是
    相机真实吐出来的那一帧稳不稳。
    """
    positions, scores, counts = [], [], []
    for frame in frames:
        # 每帧用独立种子：共用同一个 generator 会让「第几帧」影响随机性，
        # 同一帧重复跑得到不同结果，不利于复现。
        result = analyze(frame, intrinsics, params, np.random.default_rng(seed))
        candidates = [c for c in result["clusters"] if c["score"] > 0.0]
        counts.append(len(candidates))
        if not candidates:
            continue
        best = max(candidates, key=lambda c: c["score"])
        positions.append(np.asarray(best["top_center_mm"], dtype=np.float64))
        scores.append(float(best["score"]))

    stats = {
        "frames": len(frames),
        "frames_with_target": len(positions),
        "candidate_count_hist": {str(k): int(counts.count(k)) for k in sorted(set(counts))},
        "pos_std_mm": None,
        "score_std": None,
        "pos_median_mm": None,
    }
    if positions:
        arr = np.stack(positions)
        stats["pos_median_mm"] = arr.mean(axis=0).round(2).tolist()
        stats["pos_std_mm"] = round(float(np.std(arr, axis=0).max()), 2)
        stats["score_std"] = round(float(np.std(scores)), 3)
    return stats


def save_preview(depth: np.ndarray, centers: list, intrinsics: Intrinsics,
                 path: Path) -> bool:
    """存深度预览图（近亮远暗），并把检出的吸附点画上去。

    摆位自查全靠它：0 簇时先看这张图，就能分清是「盒子不在 ROI 里」
    还是「倾角太平导致深度差切不开」。
    """
    import cv2

    valid = depth > 0
    if not valid.any():
        return False
    lo = float(depth[valid].min())
    hi = float(depth[valid].max())
    span = max(hi - lo, 1.0)
    scaled = np.where(valid, 255.0 * (hi - depth) / span, 0.0).astype(np.uint8)
    canvas = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)

    for index, center in enumerate(centers):
        uv = project(np.asarray(center, dtype=np.float64), intrinsics)
        if uv is None:
            continue
        u, v = int(round(uv[0])), int(round(uv[1]))
        cv2.circle(canvas, (u, v), 6, (0, 0, 255), 2)
        cv2.drawMarker(canvas, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
        cv2.putText(canvas, str(index), (u + 8, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), canvas))


def save_viz(depth: np.ndarray, result: dict, intrinsics: Intrinsics,
             params: SuctionParams, path: Path) -> bool:
    """存一张「看得懂」的可视化：深度伪彩 + ROI 框 + 每个簇的编号/点数/分数。

    与 `--save-png` 的分工：
        `--save-png` 是灰度和吸附点，判断摆位够不够（一眼看盒子在不在画面里）；
        这张图回答「到底检出了些什么」——每个簇多大、分数多少、落在画面哪一块。
    颜色：绿=BEST（分数最高）、黄=其他候选（分数>0）、灰=背景簇（分数=0）。
    深度映射用近暖远冷，肉眼能直接分出前景和背景。
    """
    import cv2

    valid = depth > 0
    if not valid.any():
        return False
    values = depth[valid]
    lo, hi = np.percentile(values, [5, 95])
    span = max(float(hi - lo), 1.0)
    norm = np.clip((depth.astype(np.float32) - lo) / span, 0.0, 1.0)
    # 近处暖色（红）、远处冷色（蓝）：先反相再上 TURBO
    canvas = cv2.applyColorMap((255.0 * (1.0 - norm)).astype(np.uint8),
                               cv2.COLORMAP_TURBO)
    canvas[~valid] = (0, 0, 0)

    height, width = depth.shape
    roi_w, roi_h = int(width * params.roi_ratio), int(height * params.roi_ratio)
    x0, y0 = (width - roi_w) // 2, (height - roi_h) // 2

    best = max(result["clusters"], key=lambda c: c["score"], default=None)

    # 每个簇的像素区域半透明着色。坐标标签只能说明「有这个簇」，
    # 着色才能回答「它到底是画面里的哪一块」——这是判断误检最直接的办法。
    label_map = result.get("label_map")
    if label_map is not None:
        palette = [(60, 220, 60), (40, 200, 250), (250, 160, 60),
                   (200, 90, 230), (90, 220, 220), (120, 120, 250),
                   (60, 120, 200), (200, 200, 90)]
        full = np.zeros((height, width), dtype=np.int32)
        full[y0:y0 + roi_h, x0:x0 + roi_w] = label_map
        overlay = canvas.copy()
        for index, cluster in enumerate(result["clusters"]):
            mask = full == int(cluster["label"])
            if not mask.any():
                continue
            color = (0, 255, 0) if cluster is best else palette[index % len(palette)]
            overlay[mask] = color
        canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)

    cv2.rectangle(canvas, (x0, y0), (x0 + roi_w, y0 + roi_h), (255, 255, 255), 1)
    cv2.putText(canvas, "ROI 0.6", (x0 + 4, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 把「被判为吸附面」的像素单独标成洋红，并画出该面的法向箭头。
    # 这是回答「圆点落在顶面还是正面」的唯一直接证据：
    #   洋红像素连成一片且贴在物体上沿 → 取到的是顶面；
    #   洋红像素铺满整个侧面轮廓 → 取成了正面（那说明 top_ratio 取错层）。
    slab_pixels = None
    if best is not None and label_map is not None:
        from vision.depth.primitives import backproject

        k_roi = Intrinsics(fx=intrinsics.fx, fy=intrinsics.fy,
                           cx=intrinsics.cx - x0, cy=intrinsics.cy - y0,
                           width=roi_w, height=roi_h)
        pts, pix = backproject(depth[y0:y0 + roi_h, x0:x0 + roi_w], k_roi)
        in_cluster = label_map[pix[:, 1], pix[:, 0]] == int(best["label"])
        center = np.asarray(best["top_center_mm"], dtype=np.float64)
        normal = np.asarray(best["top_normal"], dtype=np.float64)
        offs = np.abs((pts - center) @ normal)
        slab = in_cluster & (offs <= params.flat_distance_mm)
        ys = np.clip(pix[slab, 1] + y0, 0, height - 1)
        xs = np.clip(pix[slab, 0] + x0, 0, width - 1)
        canvas[ys, xs] = (255, 0, 255)
        slab_pixels = int(slab.sum())

        start = project(center, intrinsics)
        tip = project(center + normal * 60.0, intrinsics)
        if start is not None and tip is not None:
            cv2.arrowedLine(canvas, (int(start[0]), int(start[1])),
                            (int(tip[0]), int(tip[1])), (255, 0, 255), 2,
                            tipLength=0.3)
        table_n = np.asarray(result["plane"]["normal"], dtype=np.float64)
        angle = float(np.degrees(np.arccos(min(1.0, abs(float(normal @ table_n))))))
        cv2.putText(canvas,
                    f"magenta = grasp face ({slab_pixels}px)  "
                    f"angle vs table = {angle:.1f}deg",
                    (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 1)

    for index, cluster in enumerate(result["clusters"]):
        uv = project(np.asarray(cluster["top_center_mm"], dtype=np.float64),
                     intrinsics)
        if uv is None:
            continue
        u, v = int(round(uv[0])), int(round(uv[1]))
        if cluster is best:
            color, tag = (0, 255, 0), "BEST"
        elif cluster["score"] > 0.0:
            color, tag = (0, 255, 255), "CAND"
        else:
            color, tag = (200, 200, 200), "bg"
        cv2.circle(canvas, (u, v), 7, color, 2)
        text = f"{index}|{cluster['points']}p|{cluster['score']:.2f}|{tag}"
        cv2.putText(canvas, text, (u + 9, max(12, v - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), canvas))


def _display_width(text: str) -> int:
    """按终端显示宽度算字符串长度（中文/全角算 2）。

    直接用 `str.ljust` 排版会让中文列全部错位——它按字符数补空格，
    而一个汉字占两格。表格是给人看的，错位就等于没排。
    """
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def print_checks(rows: list) -> None:
    """打一张能直接贴进报告的检验表。"""
    widths = (12, 20, 34, 18)
    header = (_pad("检验", widths[0]) + _pad("实测", widths[1])
              + _pad("真值/期望", widths[2]) + _pad("误差", widths[3]) + "判定")
    print("\n" + "=" * 88)
    print("五条检验")
    print("=" * 88)
    print(header)
    print("-" * 88)
    for row in rows:
        print(_pad(row["check"], widths[0]) + _pad(row["measured"], widths[1])
              + _pad(row["expected"], widths[2]) + _pad(row["error"], widths[3])
              + row["status"])
    print("-" * 88)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从真实 bag 跑抓取检测，并用直尺/水平仪真值核对（M2 验证入口）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bag", type=Path, help="bag 目录")
    parser.add_argument("--frames", type=int, default=60,
                        help="取前多少帧做时间中位数，默认 60")
    parser.add_argument("--jitter", type=int, default=30,
                        help="逐帧跑多少帧统计抖动，默认 30；填 0 跳过")
    parser.add_argument("--last-frame", action="store_true",
                        help="不用时间中位数，只用最后一帧（看单帧表现）")
    parser.add_argument("--depth-topic", default=DEPTH_TOPIC)
    parser.add_argument("--info-topic", default=INFO_TOPIC)
    parser.add_argument("--truth-height-mm", type=float, default=None,
                        help="直尺量的盒高（毫米），检验 1 的独立输入")
    parser.add_argument("--truth-size-mm", type=float, nargs=3, default=None,
                        metavar=("SX", "SY", "H"),
                        help="直尺量的长宽高（毫米），检验 3 用前两个")
    parser.add_argument("--truth-tilt-deg", type=float, default=None,
                        help="手机水平仪量的光轴偏离竖直角度，检验 2 用")
    parser.add_argument("--truth-targets", type=int, default=1,
                        help="场景里摆了几个可吸物块，检验 5 用，默认 1")
    parser.add_argument("--save", type=Path, default=None,
                        help="把合成后的深度图存成 npy，方便反复调参")
    parser.add_argument("--save-png", type=Path, default=None,
                        help="存深度预览图（含检出点），摆位自查用")
    parser.add_argument("--save-viz", type=Path, default=None,
                        help="存深度伪彩图（含 ROI 框与每个簇的编号/点数/分数），"
                             "用来判断「到底检出了些什么」")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="把结论写成 JSON，贴报告或进 CI 用")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.frames < 1:
        print("--frames 必须 >= 1", file=sys.stderr)
        return 2
    if args.jitter < 0:
        print("--jitter 不能为负", file=sys.stderr)
        return 2
    if args.truth_targets < 1:
        print("--truth-targets 必须 >= 1", file=sys.stderr)
        return 2

    # 一次读够：时间中位数要 frames 帧，抖动要 jitter 帧，取大者
    need = max(args.frames, args.jitter)
    try:
        frames, info, total = load_bag(args.bag, need,
                                       args.depth_topic, args.info_topic)
    except (FileNotFoundError, ValueError) as exc:
        print(f"读不了：{exc}", file=sys.stderr)
        return 1

    if args.last_frame:
        depth = frames[-1].copy()
        coverage = float((depth > 0).mean())
        source = "最后一帧"
    else:
        depth, coverage = temporal_median(frames[:args.frames])
        source = f"{min(len(frames), args.frames)} 帧时间中位数"

    if coverage < 0.5:
        print(f"警告：有效深度只占 {coverage * 100:.1f}%，"
              f"中位数可能不可靠（多数帧这块区域本来就是空的）", file=sys.stderr)

    intrinsics = Intrinsics(fx=info["fx"], fy=info["fy"], cx=info["cx"],
                            cy=info["cy"], width=info["width"], height=info["height"])
    params = SuctionParams()
    rng = np.random.default_rng(args.seed)
    truth = Truth(height_mm=args.truth_height_mm,
                  size_mm=tuple(args.truth_size_mm) if args.truth_size_mm else None,
                  tilt_deg=args.truth_tilt_deg,
                  n_targets=args.truth_targets)

    print("=" * 64)
    print(f"bag: {args.bag}")
    print(f"  深度     : {info['width']}x{info['height']}, bag 内共 {total} 帧，"
          f"用了 {len(frames)} 帧")
    print(f"  内参     : fx={info['fx']:.1f} fy={info['fy']:.1f} "
          f"cx={info['cx']:.1f} cy={info['cy']:.1f}")
    print(f"  合成方式 : {source}，有效像素 {coverage * 100:.1f}%")
    print(f"  深度范围 : {depth[depth > 0].min()}~{depth.max()} mm"
          if (depth > 0).any() else "  深度范围 : 全为空")

    try:
        result = analyze(depth, intrinsics, params, rng)
    except ValueError as exc:
        print(f"\n算法报错：{exc}", file=sys.stderr)
        return 1

    if result["reason"]:
        print(f"\n没能检测：{result['reason']}")
        print("先看 --save-png 的预览：盒子是否在 ROI 内、倾角是否太平"
              "（深度差小于 cluster_jump_mm 时物块和桌面会连成一片）")
        return 1

    plane = result["plane"]
    print(f"\n桌面: inlier {plane['inlier_ratio'] * 100:.1f}%, "
          f"法向 [{plane['normal'][0]:.3f} {plane['normal'][1]:.3f} "
          f"{plane['normal'][2]:.3f}]")

    clusters = result["clusters"]
    print(f"簇数: {len(clusters)}")
    for rank, cluster in enumerate(clusters, start=1):
        center = cluster["top_center_mm"]
        normal = cluster["top_normal"]
        size = cluster.get("top_size_mm", [0.0, 0.0])
        print(f"  簇{rank}: {cluster['points']} 点, "
              f"顶面中心 [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}] mm")
        print(f"        法向 [{normal[0]:.3f} {normal[1]:.3f} {normal[2]:.3f}], "
              f"顶面跨度 {size[0]:.1f}×{size[1]:.1f} mm, "
              f"包围盒 {cluster['bbox_size_mm']} mm")
        detail = cluster["score_detail"]
        print(f"        可吸性 {cluster['score']}（限制项 {detail.get('limiting')}）, "
              f"圆盘点 {detail.get('disk_points')}, 平整 {detail.get('flat_ratio')}, "
              f"倾斜 {detail.get('tilt_deg')}°, 空洞 {detail.get('hole_ratio')}")

    rows = evaluate_checks(result, truth, intrinsics)
    print_checks(rows)

    jitter = None
    if args.jitter > 0 and len(frames) > 1:
        jitter = jitter_stats(frames[:args.jitter], intrinsics, params, args.seed)
        print("\n逐帧抖动（单帧原图，不取中位数）")
        print(f"  帧数 {jitter['frames']}，出目标 {jitter['frames_with_target']} 帧，"
              f"候选数分布 {jitter['candidate_count_hist']}")
        if jitter["pos_std_mm"] is not None:
            status = PASS if jitter["pos_std_mm"] <= TOL_JITTER_MM else "WARN"
            print(f"  位置最大轴 std {jitter['pos_std_mm']:.2f} mm"
                  f"（容差 {TOL_JITTER_MM} mm，{status}），"
                  f"分数 std {jitter['score_std']:.3f}")

    if args.save_png:
        centers = [c["top_center_mm"] for c in clusters]
        ok = save_preview(depth, centers, intrinsics, args.save_png)
        print(f"\n预览图: {'已存 ' + str(args.save_png) if ok else '写盘失败'}")

    if args.save_viz:
        ok = save_viz(depth, result, intrinsics, params, args.save_viz)
        print(f"可视化: {'已存 ' + str(args.save_viz) if ok else '写盘失败'}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save, depth)
        print(f"合成深度图已存: {args.save}（可再用 lab/proto_depth.py 调参）")

    if args.json_out:
        payload = {
            "bag": str(args.bag),
            "frames_used": min(len(frames), args.frames),
            "coverage": round(coverage, 4),
            "intrinsics": info,
            "plane": plane,
            "clusters": clusters,
            "checks": rows,
            "jitter": jitter,
            "truth": {"height_mm": truth.height_mm,
                      "size_mm": list(truth.size_mm) if truth.size_mm else None,
                      "tilt_deg": truth.tilt_deg,
                      "n_targets": truth.n_targets},
            "verdict": ("pass" if all(r["status"] != FAIL for r in rows) else "fail"),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
        print(f"JSON 已存: {args.json_out}")

    failed = [r["check"] for r in rows if r["status"] == FAIL]
    skipped = [r["check"] for r in rows if r["status"] == SKIP]
    print("\n" + "-" * 64)
    if failed:
        print(f"结论：{len(failed)} 条未通过 -> {', '.join(failed)}")
        return 1
    if skipped:
        print(f"结论：通过的检验全部达标；{len(skipped)} 条因缺少真值跳过"
              f"（{', '.join(skipped)}）")
        return 0
    print("结论：五条检验全部达标。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
