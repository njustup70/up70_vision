"""S4 合成深度图生成器：光线投射 + 实测噪声 + 无效值。

跑法（容器内）
    docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision \
        python lab/gen_synthetic_depth.py --out data/syn/single_01

    # 换参数：相机高 500mm、倾角 20°、物块 60x40x50
    ... python lab/gen_synthetic_depth.py --out data/syn/s2 \
        --cam-height 500 --tilt 20 --block 60 40 50 --seed 7

产出
    <out>/depth.npy     uint16 深度图，单位毫米（与 Orbbec 的 16UC1 一致）
    <out>/meta.json     相机位姿内参、场景、真值、噪声模型、seed、自检结果
    <out>/preview.png   归一化 8bit 预览，近亮远暗，肉眼检查用

为什么要合成数据
    B 阶段要做「俯视桌面 -> 去掉桌面 -> 找物块」，而真实的俯视场景现在拿不到
    （物块规格未定、支架未到）。合成数据的价值是真值精确已知、seed 固定可复现、
    能批量造反光与遮挡。但它替代不了真实数据：反光和弱纹理的真实失效模式
    合成不出来，那是 S2 的事。

几何约定
    世界系：桌面是 z=0 平面，z 向上，单位毫米。
    相机系：OpenCV 约定，x 右、y 下、z 沿光轴向前。
    深度值：**沿光轴的 z 深度**，不是到相机的欧氏距离。Orbbec 的 16UC1 就是
    z 深度，反投影公式 x=(u-cx)z/fx 也按 z 深度写。两者混用是反投影错位的
    常见来源。

    相机位于 (0, 0, cam_height)，光轴偏离竖直 tilt 度、朝 +x 俯视。
    物块默认放在光轴与桌面的交点，也就是图像正中，便于一眼看出对错。

噪声模型
    用 A4 实测拟合的 sigma(mm) = 3.98e-6 * z^1.74，不是计划书原定的 4e-6 * z^2。
    实测 663/1029/1605mm 处噪声 0.36/0.60/1.68mm，4z² 会高估 5~7 倍。
    合成数据噪声偏大，会让人把 distance_threshold 和聚类容差都调得过宽，
    换真实数据立刻暴露。但合成区只有三个距离点，模型本身有外推成分，
    真要严谨应在目标距离实测补点。

自检（S4 的验收标准）
    物块顶面与桌面的深度差 = 物块高 × cos(倾角)。

    推导：同一条视线上两点高度相差 H，视线与竖直夹角为 tilt，则沿视线方向的
    距离差是 H·cos(tilt)。图像中心像素的视线近似等于光轴，而 z 深度在中心处
    等于欧氏距离，所以公式对中心像素成立。

    实现上不用「中心一个像素」，而是取物块投影点附近一小块窗口，逐像素算
    「桌面深度 - 顶面深度」再取中位数。逐像素相减能消掉倾斜带来的空间梯度，
    比拿多个像素的绝对深度取中位数准得多（后者在 window 内就能差 3mm）。

    这个自检比任何单元测试都直接：坐标系搞反、旋转矩阵转置、倾角用错，
    这个数字立刻不对。

已知的坑
    · 深度是 z 深度不是欧氏距离。误当欧氏距离反投影，图像边缘会整体外扩。
    · 旋转矩阵方向容易反。约定 R 是「相机系 -> 世界系」，验证方法：
      R @ [0,0,1] 必须等于光轴在世界系的方向（tilt=0 时应为 (0,0,-1)）。
    · 射线与某轴平行时必须单独处理，否则 (lo-o)/0 得到 ±inf，
      slab 相交会误判成永远命中或永远不命中，表现为物块侧面穿透。
    · 量化到 1mm 是刻意的：真实相机就是 1mm 台阶。不加量化的合成数据
      会比真实数据还准，调参时过度乐观。

退出码：0 成功且自检通过；1 自检不通过或写盘失败；2 参数非法。
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# A4 实测拟合：sigma(mm) = NOISE_COEF * z(mm) ** NOISE_EXP
NOISE_COEF = 3.98e-6
NOISE_EXP = 1.74

# 深度量化台阶（毫米）：真实相机 16UC1 + enable_depth_scale=true 就是 1mm
DEPTH_QUANTUM_MM = 1.0

# 自检容差：合成数据没有别的误差源，超过这个数就是几何写错了
SELF_CHECK_TOL_MM = 0.5


@dataclass(frozen=True)
class Box:
    """轴对齐长方体：底面坐在 z=0 桌面上，顶面高度 height。"""

    cx: float
    cy: float
    sx: float
    sy: float
    height: float

    def bounds(self) -> tuple:
        """世界系三个轴上的 (下界, 上界)。"""
        return (
            (self.cx - self.sx / 2, self.cx + self.sx / 2),
            (self.cy - self.sy / 2, self.cy + self.sy / 2),
            (0.0, self.height),
        )

    def top_center(self) -> np.ndarray:
        """顶面中心的世界坐标，自检和真值都用它。"""
        return np.array([self.cx, self.cy, self.height], dtype=np.float64)


def camera_rotation(tilt_deg: float, yaw_deg: float) -> np.ndarray:
    """相机系 -> 世界系的旋转矩阵。

    先让光轴朝 +x 俯视 tilt 度，再整体绕世界 z 轴转 yaw 度。
    检查方向有没有搞反：R @ [0,0,1] 就是光轴在世界系的方向，
    tilt=0 时应得到 (0,0,-1)。
    """
    tilt = np.radians(tilt_deg)
    yaw = np.radians(yaw_deg)

    z_axis = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])   # 光轴
    x_axis = np.array([np.cos(tilt), 0.0, np.sin(tilt)])    # 相机 x，与光轴正交
    y_axis = np.cross(z_axis, x_axis)                       # 右手系

    base = np.stack([x_axis, y_axis, z_axis], axis=1)
    rotate_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rotate_yaw @ base


def pixel_rays(width: int, height: int, k: dict) -> np.ndarray:
    """每个像素在相机系下的射线方向 (H, W, 3)，z 分量恒为 1。

    保持 z=1 而不归一化：求交参数 t 直接就是 z 深度，省一次余弦。
    """
    u, v = np.meshgrid(np.arange(width, dtype=np.float64),
                       np.arange(height, dtype=np.float64))
    dirs = np.empty((height, width, 3), dtype=np.float64)
    dirs[..., 0] = (u - k["cx"]) / k["fx"]
    dirs[..., 1] = (v - k["cy"]) / k["fy"]
    dirs[..., 2] = 1.0
    return dirs


def _slab_intersect(origin: np.ndarray, dirs: np.ndarray, box: Box) -> np.ndarray:
    """射线与轴对齐长方体求交，返回进入参数 t，未命中为 inf。

    标准 slab 方法：三个轴各算一个参数区间，取交集。
    某轴平行时必须单独判定：否则 (lo-o)/0 得到 ±inf，在 max/min 里会变成
    「永远命中」或「永远不命中」，表现为物块侧面被穿透。
    """
    t_enter = np.full(dirs.shape[:2], -np.inf)
    t_exit = np.full(dirs.shape[:2], np.inf)
    eps = 1e-9

    for axis, (lo, hi) in enumerate(box.bounds()):
        d = dirs[..., axis]
        o = float(origin[axis])

        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - o) / d
            t2 = (hi - o) / d
        near = np.minimum(t1, t2)
        far = np.maximum(t1, t2)

        parallel = np.abs(d) < eps
        if parallel.any():
            inside = (o >= lo) and (o <= hi)
            near = np.where(parallel, -np.inf if inside else np.inf, near)
            far = np.where(parallel, np.inf if inside else -np.inf, far)

        t_enter = np.maximum(t_enter, near)
        t_exit = np.minimum(t_exit, far)

    hit = (t_enter <= t_exit) & (t_exit > 0)
    t_hit = np.where(t_enter > 0, t_enter, t_exit)
    return np.where(hit, t_hit, np.inf)


def raycast(width: int, height: int, k: dict, cam_pos: np.ndarray,
            rotation: np.ndarray, boxes: list) -> tuple:
    """渲染深度图，返回 (最终深度, 桌面深度)，单位毫米。

    桌面深度单独返回：物块挡住的那块桌面看不见，但几何上仍然存在，
    自检要靠它算「如果没有物块，这个像素该是多深」。
    """
    dirs_cam = pixel_rays(width, height, k)
    dirs_world = dirs_cam @ rotation.T

    d_z = dirs_world[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_table = (0.0 - float(cam_pos[2])) / d_z
    # 射线朝上时打不到桌面（相机水平或朝天），记为无穷远
    t_table = np.where(d_z < 0, t_table, np.inf)

    depth = t_table.copy()
    for box in boxes:
        t_box = _slab_intersect(cam_pos, dirs_world, box)
        depth = np.where(t_box < depth, t_box, depth)

    return depth, t_table


def apply_noise(depth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """按实测模型加高斯噪声，并量化到 1mm。无效像素保持无效。"""
    finite = np.isfinite(depth)
    sigma = NOISE_COEF * np.where(finite, depth, 0.0) ** NOISE_EXP

    noisy = depth.copy()
    noisy[finite] = depth[finite] + rng.normal(0.0, sigma[finite])
    # 夹到非负：噪声可能把很近的点推到负值，负值会被读成无效
    noisy[finite] = np.clip(noisy[finite], DEPTH_QUANTUM_MM, None)
    noisy[finite] = np.round(noisy[finite] / DEPTH_QUANTUM_MM) * DEPTH_QUANTUM_MM
    return noisy


def apply_invalid(depth: np.ndarray, min_range: float, max_range: float,
                  edge_drop_px: int, jump_mm: float = 20.0) -> tuple:
    """把不该有深度的地方置 0，返回 (带空洞的深度, 统计)。

    三类无效：
      1. 超量程：真实相机近端约 0.2~0.3m，远端有限，超出就是 0
      2. 深度跳变边缘：双目的混合像素，真实数据上就是物块轮廓那圈毛刺
      3. 无交点：理论上不该出现，留着兜底
    """
    valid = np.isfinite(depth)
    out_of_range = valid & ((depth > max_range) | (depth < min_range))

    edge = np.zeros_like(valid)
    if edge_drop_px > 0 and valid.any():
        filled = np.where(valid, depth, np.median(depth[valid]))
        jump = np.zeros_like(valid)
        jump[:, :-1] |= np.abs(np.diff(filled, axis=1)) > jump_mm
        jump[:-1, :] |= np.abs(np.diff(filled, axis=0)) > jump_mm
        for _ in range(edge_drop_px):
            grown = jump.copy()
            grown[:, 1:] |= jump[:, :-1]
            grown[:, :-1] |= jump[:, 1:]
            grown[1:, :] |= jump[:-1, :]
            grown[:-1, :] |= jump[1:, :]
            jump = grown
        edge = jump & valid

    out = depth.copy()
    out[~valid | out_of_range | edge] = 0.0

    stats = {
        "pixels_total": int(depth.size),
        "valid_pixels": int((out > 0).sum()),
        "invalid_out_of_range": int(out_of_range.sum()),
        "invalid_edge": int(edge.sum()),
        "invalid_no_hit": int((~valid).sum()),
    }
    return out, stats


def self_check(clean_depth: np.ndarray, k: dict, cam_pos: np.ndarray,
               rotation: np.ndarray, box: Box, tilt_deg: float) -> dict:
    """S4 验收：物块顶面与其正下方桌面点的深度差 == 物块高 × cos(倾角)。

    这里必须说清「哪两点」，很容易搞混：

    · 计划书那条公式对应的是**竖直对齐**的两点：物块顶面中心 (x,y,H) 和它正
      下方的桌面点 (x,y,0)。这两点不在同一条视线上（视线是斜的），深度差是
      (0,0,H) 投影到光轴上的长度 = H·cos(tilt)。
    · 如果改成「同一个像素的射线上」两点（射线先穿桌面再穿物块顶面），
      深度差是 H/cos(tilt)，正好是反过来的。25° 时一个是 45.3mm、一个是 55.2mm，
      差 10mm 很容易被当成 bug。

    本函数验的是前者：顶面深度取**渲染结果**，正下方桌面深度用解析式算
    （那个点被物块挡着，在图上根本看不见）。顺带把渲染深度和解析深度比一次，
    这一步能抓住坐标系写反、旋转矩阵转置、求交算错这类问题——它们会让
    渲染值和解析值对不上，而只看 Δ 是看不出来的。
    """
    p_top_cam = rotation.T @ (box.top_center() - cam_pos)
    p_bottom_cam = rotation.T @ (np.array([box.cx, box.cy, 0.0]) - cam_pos)

    if p_top_cam[2] <= 0 or p_bottom_cam[2] <= 0:
        return {"ok": False, "reason": "物块顶面中心落在相机后面，位姿参数不合理"}

    u = k["fx"] * p_top_cam[0] / p_top_cam[2] + k["cx"]
    v = k["fy"] * p_top_cam[1] / p_top_cam[2] + k["cy"]
    height, width = clean_depth.shape

    y0, x0 = int(round(v)), int(round(u))
    if not (0 <= y0 < height and 0 <= x0 < width):
        return {"ok": False, "reason": f"物块投影到图像外 (u={u:.0f}, v={v:.0f})"}

    window = 5
    ys = slice(max(0, y0 - window), min(height, y0 + window + 1))
    xs = slice(max(0, x0 - window), min(width, x0 + window + 1))
    patch = clean_depth[ys, xs]

    # 只取落在物块顶面的像素：与解析顶面深度接近的那些
    on_top = np.isfinite(patch) & (np.abs(patch - p_top_cam[2]) < 5.0)
    if not on_top.any():
        return {"ok": False, "reason": "窗口内找不到物块顶面像素（检查物块位置与位姿）"}

    top_render = float(np.median(patch[on_top]))
    render_error = abs(top_render - p_top_cam[2])

    delta = float(p_bottom_cam[2]) - top_render
    expected = box.height * np.cos(np.radians(tilt_deg))
    error = abs(delta - expected)

    ok = bool(error <= SELF_CHECK_TOL_MM and render_error <= SELF_CHECK_TOL_MM)
    return {
        "ok": ok,
        "pixel": [round(float(u), 1), round(float(v), 1)],
        "top_depth_render_mm": round(top_render, 3),
        "top_depth_analytic_mm": round(float(p_top_cam[2]), 3),
        "render_error_mm": round(render_error, 3),
        "table_below_depth_mm": round(float(p_bottom_cam[2]), 3),
        "delta_measured_mm": round(delta, 3),
        "delta_expected_mm": round(float(expected), 3),
        "error_mm": round(error, 3),
        "pixels_used": int(on_top.sum()),
    }


def save_preview(depth: np.ndarray, path: Path) -> bool:
    """存 8bit 预览图：近亮远暗，方便一眼确认渲染是否合理。"""
    import cv2

    valid = depth > 0
    if not valid.any():
        return False
    lo = float(depth[valid].min())
    hi = float(depth[valid].max())
    span = max(hi - lo, 1.0)
    scaled = np.where(valid, 255.0 * (hi - depth) / span, 0.0).astype(np.uint8)
    return bool(cv2.imwrite(str(path), scaled))


def main() -> int:
    """解析参数、渲染、自检、落盘。"""
    parser = argparse.ArgumentParser(
        description="S4 合成深度图生成器（光线投射 + 实测噪声 + 无效值）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", required=True, type=Path, help="输出目录")
    parser.add_argument("--cam-height", type=float, default=600.0,
                        help="相机离桌面高度（毫米），默认 600")
    parser.add_argument("--tilt", type=float, default=25.0,
                        help="光轴偏离竖直的角度（度），默认 25（计划书推荐）")
    parser.add_argument("--yaw", type=float, default=0.0, help="绕竖直轴旋转（度）")
    parser.add_argument("--block", nargs=3, type=float, default=[50.0, 50.0, 50.0],
                        metavar=("SX", "SY", "H"), help="物块长宽高（毫米）")
    parser.add_argument("--block-xy", nargs=2, type=float, default=None,
                        metavar=("X", "Y"), help="物块中心平面位置，默认放光轴交点")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--img-height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=409.5)
    parser.add_argument("--fy", type=float, default=409.5)
    parser.add_argument("--cx", type=float, default=424.33)
    parser.add_argument("--cy", type=float, default=241.33)
    parser.add_argument("--seed", type=int, default=0, help="随机种子，固定它才能复现")
    parser.add_argument("--min-range", type=float, default=200.0, help="近端量程（毫米）")
    parser.add_argument("--max-range", type=float, default=3000.0, help="远端量程（毫米）")
    parser.add_argument("--edge-drop-px", type=int, default=1,
                        help="深度跳变处向外扩几圈置为无效，默认 1")
    args = parser.parse_args()

    if args.width < 8 or args.img_height < 8:
        print("图像尺寸太小", file=sys.stderr)
        return 2
    if not (0.0 <= args.tilt < 90.0):
        print("--tilt 必须在 [0, 90) 内", file=sys.stderr)
        return 2

    k = {"fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy}
    cam_pos = np.array([0.0, 0.0, args.cam_height], dtype=np.float64)
    rotation = camera_rotation(args.tilt, args.yaw)

    # 物块默认放在光轴与桌面的交点，即图像正中
    if args.block_xy is None:
        block_xy = (args.cam_height * np.tan(np.radians(args.tilt)), 0.0)
    else:
        block_xy = tuple(args.block_xy)
    box = Box(cx=block_xy[0], cy=block_xy[1],
              sx=args.block[0], sy=args.block[1], height=args.block[2])

    rng = np.random.default_rng(args.seed)

    clean_depth, t_table = raycast(args.width, args.img_height, k, cam_pos,
                                   rotation, [box])
    check = self_check(clean_depth, k, cam_pos, rotation, box, args.tilt)

    noisy = apply_noise(clean_depth, rng)
    final, invalid_stats = apply_invalid(noisy, args.min_range, args.max_range,
                                        args.edge_drop_px)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_u16 = np.where(final > 0, final, 0.0).astype(np.uint16)
    depth_path = out_dir / "depth.npy"
    np.save(depth_path, depth_u16)

    # 无噪声的几何深度单独存一份：fake_camera_node 要在它上面逐帧重新加噪。
    # 只发 depth.npy 的话每帧是同一张图，时间滤波、滑动平均、跟踪这类逻辑
    # 在合成数据上全部「完美」，一到真机就崩。
    clean_path = out_dir / "clean_depth.npy"
    np.save(clean_path, np.where(np.isfinite(clean_depth), clean_depth, 0.0).astype(np.float32))

    preview_ok = save_preview(depth_u16.astype(np.float64), out_dir / "preview.png")

    meta = {
        "generator": "lab/gen_synthetic_depth.py",
        "seed": args.seed,
        "files": {
            "depth": "depth.npy",
            "clean_depth": "clean_depth.npy",
            "preview": "preview.png",
        },
        "files_note": "depth.npy 含噪声与无效值，是「一帧真实数据」；"
                      "clean_depth.npy 是无噪声几何值，供 fake_camera_node 逐帧加噪。",
        "camera": {
            "width": args.width, "height": args.img_height,
            "fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy,
            "position_mm": cam_pos.tolist(),
            "tilt_deg": args.tilt, "yaw_deg": args.yaw,
            "depth_type": "z_depth",
        },
        "table": {"z_mm": 0.0},
        "objects": [{
            "id": 0,
            "type": "box",
            "center_mm": [box.cx, box.cy, box.height / 2],
            "size_mm": [box.sx, box.sy, box.height],
            "top_center_mm": box.top_center().tolist(),
            "top_normal_world": [0.0, 0.0, 1.0],
            # 真值转相机系：B 阶段要拿它和检测结果直接比
            "top_center_cam_mm": (rotation.T @ (box.top_center() - cam_pos)).tolist(),
            "top_normal_cam": (rotation.T @ np.array([0.0, 0.0, 1.0])).tolist(),
        }],
        "noise": {
            "model": "sigma_mm = coef * z_mm ** exp",
            "coef": NOISE_COEF, "exp": NOISE_EXP,
            "quantum_mm": DEPTH_QUANTUM_MM,
            "source": "A4 实测拟合（663/1029/1605mm 处 0.36/0.60/1.68mm）",
        },
        "invalid": {
            "min_range_mm": args.min_range,
            "max_range_mm": args.max_range,
            "edge_drop_px": args.edge_drop_px,
            "stats": invalid_stats,
        },
        "self_check": check,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    valid_ratio = invalid_stats["valid_pixels"] / invalid_stats["pixels_total"]
    print("=" * 62)
    print(f"合成深度图 -> {out_dir}")
    print(f"  相机      : 高 {args.cam_height:.0f}mm, 倾角 {args.tilt:.0f}°, yaw {args.yaw:.0f}°")
    print(f"  物块      : {args.block[0]:.0f}x{args.block[1]:.0f}x{args.block[2]:.0f}mm "
          f"@ ({box.cx:.1f}, {box.cy:.1f})")
    print(f"  有效像素  : {valid_ratio * 100:.1f}% "
          f"(边缘 {invalid_stats['invalid_edge']}, 超距 {invalid_stats['invalid_out_of_range']})")
    print(f"  深度范围  : {depth_u16[depth_u16 > 0].min()}~{depth_u16.max()} mm")
    print("-" * 62)
    print(f"自检: 深度差实测 {check.get('delta_measured_mm', 'n/a')} mm, "
          f"理论 {check.get('delta_expected_mm', 'n/a')} mm "
          f"(= 物块高 × cos(倾角))")
    print(f"  渲染-解析一致性: {check.get('render_error_mm', 'n/a')} mm "
          f"(顶面渲染 {check.get('top_depth_render_mm', 'n/a')} vs "
          f"解析 {check.get('top_depth_analytic_mm', 'n/a')})")
    if check.get("ok"):
        print(f"  [OK] 误差 {check['error_mm']} mm，在 {SELF_CHECK_TOL_MM} mm 容差内")
    else:
        reason = check.get("reason") or f"误差 {check.get('error_mm')} mm 超出容差"
        print(f"  [FAIL] {reason}")
        print("  几何写错了，先别拿这批数据往下走。")
        return 1

    print(f"  预览图    : {'有' if preview_ok else '无（写盘失败）'}")
    print(f"  meta      : {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
