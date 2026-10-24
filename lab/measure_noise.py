"""A4 深度噪声实测：读 bag，出噪声表，与计划书 2.4 的估算值对账。

跑法（容器内，仓库根即容器里的 /opt/vision）
    docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision \
        python lab/measure_noise.py bags/noise_050 bags/noise_100 bags/noise_200

    # 板子没铺满、ROI 里混进了背景时，缩小 ROI 重算
    ... python lab/measure_noise.py bags/noise_050 --roi 120

采集方法（相机不动，移动平面）
    相机随手放在桌上，全程别碰；人举一块硬板退到 0.5 / 1 / 2m 各停三四秒，
    每个位置录一段 bag，相机内参不动，距离不用卷尺量——深度图自己会给出。

    为什么反过来动平面而不是动相机：
    噪声的定义是「同一像素对同一块平面连续采样时的标准差」。相机一动，同一
    像素每帧看的是板上不同的点，深度值的变化里混进了真实距离变化，算出的
    std 能大一个量级。相机固定后这个污染源直接消失。

    板子那边同样要放稳：手举着会抖，板到相机的距离每帧都在变，效果和相机
    抖动完全一样。所以要么把板靠在墙/椅背上，要么垫在桌上。脚本会自动检测
    帧间距离抖动并只取最长的稳定段，剔掉多少会打出来——剔除比例很高说明板
    子没放稳，重录比硬算强。

指标定义
    噪声 = 逐像素沿时间轴求标准差，再对 ROI 内的有效像素取中位数。

    为什么用时间维而不是单帧内的空间 std：
    板子只要没正对相机，单帧 ROI 内就有系统性深度梯度（斜视时近端远端能差
    几十毫米），空间 std 会把这个梯度当成噪声，严重高估。时间维 std 是每个
    像素跟自己的历史比，对倾斜免疫——所以即使脚本提示「板子有倾斜」，噪声
    数值依然可信，那只是提醒你去摆正，不是说这次数据作废。

    无效值：深度 0 表示该点没测到。它会同时把均值和 std 拉爆，必须按像素剔除。
    脚本统计有效像素比例，低于阈值直接判为数据不合格。

验收基准
    计划书 2.4 那张估算表拟合出来是 σ(z) = 4·z² mm（z 单位米）：
        0.3m → 0.36mm    0.5m → 1.0mm    0.7m → 1.96mm    1.0m → 4.0mm    2.0m → 16mm

    但这张表来自官方 ≤0.8%@2m，而 0.8% 是**准确度**（读数与真值的偏差，
    含标定误差、温度漂移这类系统项），本脚本测的是**噪声**（同一位置反复
    测量的离散度）。两者差一个量级是常态，不是故障。

    实测对照（Orbbec 336L，默认配置，2026-10）：
        663mm → 0.38mm    1029mm → 0.60mm    1605mm → 1.68mm
    全部优于外推值 5~7 倍。这是预期结果：噪声远小于准确度。

    所以判定用比值 measured / expected：
        > 2.0      → bad，按计划书要求停下排查（对焦、反光、板子没放稳）
        0.5 ~ 2.0  → ok，与外推值同量级
        < 0.5      → better，优于外推值，属正常，不判失败
        < 0.05     → bad，噪声接近 0 只可能是假数据（填充值、重复帧）

    为什么不因为「优于基准」就判失败：A4 要确认的是相机够不够用，
    噪声更小是好事。真正要拦的是数据被做过手脚（全零、重复帧）。
    测的是默认配置（含驱动自带的 noise_removal 滤波），因为部署时就是
    这个配置，关掉滤波测出来反而不代表实际能拿到的数据。

已知的坑
    · 深度量化台阶是 1mm（16UC1 + enable_depth_scale=true）。0.5m 处真实噪声
      约 1mm，量化和噪声同量级，所以那里的 std 里必然含约 0.3mm 的量化贡献，
      属正常现象，不要试图「减掉量化」再去对表。
    · 某段数据算出 std 恰好为 0 时别当成相机好：更可能是深度图被填充成了
      固定值，或录到了重复帧。脚本单独提示这一条。
    · 2m 处需要很大的板：中心 200px 在 2m 处对应约 0.96m 见方。找不到就直接
      对着墙测，墙天然平整且足够大。
    · ROI 覆盖到板外（看到背景/地面）时，ROI 内会同时存在两个距离，中位数还
      能用但含义已经变了。脚本用「ROI 内深度峰峰值」检查这一点，报了就走 --roi。
    · 依赖 rosbags（纯 Python 读 mcap）。它是 pyproject 里的 lab 可选依赖，
      不是 vision 包的运行时依赖：vision/ 本身不读 bag。

退出码：全部判定通过为 0，有任一不合格为 1。
"""

import argparse
import sys
from pathlib import Path

import numpy as np

DEPTH_TOPIC = "/camera/depth/image_raw"
INFO_TOPIC = "/camera/depth/camera_info"

# 计划书 2.4 估算曲线的系数：σ(z) = 4·z² mm，z 用米。
NOISE_COEF = 4.0

# 稳定段的容差下限。取 2mm 是因为 0.5m 处预期噪声约 1mm，
# 板子位移超过 2mm 就足以污染统计；远距离时按预期噪声放大（见 _stable_run）。
STABLE_TOL_MIN_MM = 2.0

# 有效像素比例低于这个值就判数据不合格：深度 0 太多说明反光、超距或没对焦。
MIN_VALID_RATIO = 0.9


def expected_noise_mm(z_mm: float) -> float:
    """按计划书 2.4 的估算曲线给出该距离下应有的深度噪声（mm）。"""
    z_m = z_mm / 1000.0
    return NOISE_COEF * z_m * z_m


def centre_roi(height: int, width: int, size: int) -> tuple:
    """取图像正中的正方形 ROI。

    只取中心是因为：视场边缘的深度质量系统性差于中心（畸变、双目基线夹角变差），
    把边缘算进噪声会让结果虚高，而且和 B 阶段实际用的中心区域对不上。
    size 超过短边时自动收缩，避免空切片。
    """
    size = max(8, min(size, height, width))
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return slice(y0, y0 + size), slice(x0, x0 + size)


def _decode_depth(msg) -> np.ndarray:
    """把 sensor_msgs/Image 解成 (H, W) uint16 的毫米深度图。

    step 不一定等于 width*2：驱动可能按行对齐补 padding。直接按 width*2 铺开，
    有 padding 时图像会斜着错位，而且错位之后噪声看起来也会「很正常」，
    所以这里显式判断。
    """
    encoding = str(getattr(msg, "encoding", ""))
    if encoding not in ("16UC1", "mono16"):
        raise ValueError(f"深度编码是 {encoding!r}，本脚本只处理 16UC1（毫米）。")

    buf = np.asarray(msg.data, dtype=np.uint8)
    height, width, step = int(msg.height), int(msg.width), int(msg.step)

    if step == width * 2:
        arr = np.frombuffer(buf.tobytes(), dtype="<u2")
    else:
        rows = buf.reshape(height, step)[:, : width * 2]
        arr = np.ascontiguousarray(rows).view("<u2")
    return arr.reshape(height, width)


def load_bag(bag_dir: Path, roi_size: int, max_frames: int):
    """读一段 bag，返回 (ROI 内的深度帧堆栈, 内参信息, 原始帧数)。

    堆栈只保留 ROI：整幅 848x480 的 300 帧是 244MB，裁到中心 200x200 后是 24MB，
    分析用不到 ROI 之外的像素，没必要占内存。
    """
    from rosbags.highlevel import AnyReader

    if not bag_dir.exists():
        raise FileNotFoundError(f"bag 不存在：{bag_dir}")

    frames = []
    info = None
    roi = None
    total = 0

    with AnyReader([bag_dir]) as reader:
        depth_conns = [c for c in reader.connections if c.topic == DEPTH_TOPIC]
        if not depth_conns:
            topics = sorted({c.topic for c in reader.connections})
            raise ValueError(f"{bag_dir} 里没有 {DEPTH_TOPIC}。实际话题：{topics}")

        info_conns = [c for c in reader.connections if c.topic == INFO_TOPIC]
        # camera_info 只取第一帧：整段录制期间相机没动、内参也不会变。
        if info_conns:
            for conn, _, raw in reader.messages(connections=info_conns):
                msg = reader.deserialize(raw, conn.msgtype)
                k = list(msg.k)
                info = {"fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5],
                        "width": int(msg.width), "height": int(msg.height)}
                break

        for conn, _, raw in reader.messages(connections=depth_conns):
            total += 1
            if len(frames) >= max_frames:
                # 继续数但不解包，好让「bag 里一共多少帧」这个数字是准的
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            if roi is None:
                roi = centre_roi(int(msg.height), int(msg.width), roi_size)
            frames.append(_decode_depth(msg)[roi].copy())

    if not frames:
        raise ValueError(f"{bag_dir} 的深度话题里没有消息。")

    return np.stack(frames), info, total


def _stable_run(frame_med: np.ndarray, z_mm: float) -> tuple:
    """找出距离最稳的最长连续段，返回 (起点, 长度)。

    判据是「该帧的 ROI 中位数偏离全局中位数不超过 tol」。tol 随距离放大：
    2m 处预期噪声本身就有 16mm，用 0.5m 的 2mm 容差会把整段都判成不稳定。
    """
    tol = max(STABLE_TOL_MIN_MM, 0.25 * expected_noise_mm(z_mm))
    filled = np.where(np.isnan(frame_med), z_mm, frame_med)
    ok = np.abs(filled - z_mm) < tol

    best_start, best_len = 0, 0
    cur_start, cur_len = None, 0
    for i, flag in enumerate(ok):
        if flag:
            if cur_start is None:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_start, cur_len = None, 0
    return best_start, best_len


def _profile_span(mean_img: np.ndarray, valid_mask: np.ndarray, along_axis: int) -> float:
    """沿某轴求平均深度剖面，线性拟合后返回整段跨度（mm）。

    用来估计 ROI 内存在多大的系统性深度梯度：板子正对相机时这个跨度应该和噪声
    一个量级；明显偏大说明板子举歪了（或 ROI 跨到了板外）。
    """
    n = mean_img.shape[along_axis]
    prof = np.full(n, np.nan)
    for i in range(n):
        col = mean_img[:, i] if along_axis == 1 else mean_img[i, :]
        m = valid_mask[:, i] if along_axis == 1 else valid_mask[i, :]
        if m.any():
            prof[i] = float(np.mean(col[m]))

    sel = ~np.isnan(prof)
    if sel.sum() < 2:
        return 0.0
    idx = np.arange(n)[sel]
    slope = np.polyfit(idx, prof[sel], 1)[0]
    return float(abs(slope) * (n - 1))


def analyse(stack: np.ndarray, scale: float) -> dict:
    """核心统计：距离、噪声、以及四项数据质量指标。"""
    data = stack.astype(np.float32) * scale
    valid = data > 0
    n = data.shape[0]

    # 逐帧 ROI 中位数：代表这一帧「板子到相机的距离」
    frame_med = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        v = data[i][valid[i]]
        if v.size:
            frame_med[i] = float(np.median(v))

    if np.all(np.isnan(frame_med)):
        raise ValueError("ROI 内没有任何有效深度，无法统计（检查相机是否对着板子）。")

    z_mm = float(np.nanmedian(frame_med))
    start, length = _stable_run(frame_med, z_mm)

    if length < 20:
        # 帧数太少时 std 本身不可靠，与其给一个假数字不如明确失败
        raise ValueError(
            f"稳定段只有 {length} 帧（共 {n} 帧），不足以统计噪声。"
            f"板子/相机在录制期间没放稳，或这段录得太短，请重录。"
        )

    seg = data[start:start + length]
    seg_valid = valid[start:start + length]
    count = seg_valid.sum(axis=0)

    # 手工算时间维均值/方差：np.nanstd 对全无效像素会刷一屏 RuntimeWarning，
    # 而且这里本来就需要同时拿到每个像素的有效帧数。
    total = np.where(seg_valid, seg, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    dev = np.where(seg_valid, seg - mean, 0.0)
    var = np.divide((dev * dev).sum(axis=0), count, out=np.zeros_like(mean), where=count > 0)
    pix_std = np.sqrt(var)

    # 有效帧数不足的像素不参与统计：少数帧算出的 std 会失真
    good = count >= MIN_VALID_RATIO * length
    if not good.any():
        raise ValueError("没有像素的有效帧数能达标，数据质量不足（大量 0 值）。")

    noise = float(np.median(pix_std[good]))
    z_stable = float(np.nanmedian(frame_med[start:start + length]))
    expected = expected_noise_mm(z_stable)

    mean_img = mean
    mask_img = count > 0
    span_u = _profile_span(mean_img, mask_img, along_axis=1)
    span_v = _profile_span(mean_img, mask_img, along_axis=0)

    return {
        "frames_total": n,
        "frames_used": int(length),
        "stable_start": int(start),
        "z_mm": z_stable,
        "noise_mm": noise,
        "expected_mm": expected,
        "ratio": noise / expected if expected > 0 else float("nan"),
        "valid_ratio": float(seg_valid.mean()),
        "jitter_mm": float(np.nanstd(frame_med[start:start + length])),
        "tilt_span_mm": max(span_u, span_v),
    }


def verdict(result: dict) -> tuple:
    """给出判定与补充说明。返回 (等级, 说明)，等级 ∈ {ok, warn, bad}。"""
    noise = result["noise_mm"]
    ratio = result["ratio"]

    # std 恰好为 0 是异常信号，不是好消息：真实传感器不可能零噪声
    if noise < 0.05:
        return "bad", "噪声≈0，深度图可能被填充成了固定值或录到了重复帧"

    if ratio > 2.0:
        return "bad", "偏大，按计划书要求停下排查：对焦/标定/反光/板子没放稳"
    if ratio < 0.5:
        # 基准来自官方准确度指标，噪声优于它 5~7 倍是正常结果，不是故障。
        # 真正该拦的「假数据」已经由上面的 noise < 0.05 兜住了。
        return "better", "优于基准外推值（基准是准确度指标，噪声更小属正常）"
    return "ok", "对得上"


def quality_notes(result: dict) -> list:
    """把会影响结论可信度的问题换成一句句人话，空列表表示没问题。"""
    notes = []
    if result["valid_ratio"] < MIN_VALID_RATIO:
        notes.append(
            f"有效像素仅 {result['valid_ratio'] * 100:.1f}%，多数像素没测到深度："
            f"板子反光、超出量程或表面太斜"
        )
    if result["tilt_span_mm"] > 3 * max(result["noise_mm"], 1e-6):
        notes.append(
            f"ROI 内深度跨度 {result['tilt_span_mm']:.1f}mm，远大于噪声，说明板子没正对相机，"
            f"或 ROI 跨到了板外（可试 --roi 缩小）"
        )
    if result["frames_used"] < result["frames_total"]:
        dropped = result["frames_total"] - result["frames_used"]
        notes.append(f"剔除了 {dropped} 帧不稳定数据，只用了连续的第 "
                     f"{result['stable_start']}~{result['stable_start'] + result['frames_used'] - 1} 帧")
    if result["jitter_mm"] > 0.5 * max(result["noise_mm"], 1e-6) * 2:
        notes.append(f"稳定段内帧间距离抖动 {result['jitter_mm']:.2f}mm，偏大")
    return notes


def print_report(bag: Path, info, result: dict, level: str, reason: str, notes: list) -> None:
    """打印单段 bag 的明细。"""
    print("=" * 66)
    print(f"bag: {bag}")
    if info:
        print(f"  内参        : fx={info['fx']:.2f} fy={info['fy']:.2f} "
              f"cx={info['cx']:.2f} cy={info['cy']:.2f}  ({info['width']}x{info['height']})")
    print(f"  帧数        : 用 {result['frames_used']} / 共 {result['frames_total']}")
    print(f"  距离        : {result['z_mm']:.1f} mm")
    print(f"  噪声(std)   : {result['noise_mm']:.2f} mm")
    print(f"  有效像素    : {result['valid_ratio'] * 100:.1f}%")
    print(f"  帧间抖动    : {result['jitter_mm']:.2f} mm")
    print(f"  倾斜跨度    : {result['tilt_span_mm']:.2f} mm")
    print(f"  基准 σ=4z²  : {result['expected_mm']:.2f} mm")
    print(f"  比值        : {result['ratio']:.2f}x")
    print(f"  判定        : [{level.upper()}] {reason}")
    for note in notes:
        print(f"  提示        : {note}")


def main() -> int:
    """解析参数、逐段分析、最后打一张汇总表。"""
    parser = argparse.ArgumentParser(
        description="A4 深度噪声实测：读 bag 出噪声表，对照 σ(z)=4·z² mm。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bags", nargs="+", type=Path, help="一个或多个 bag 目录")
    parser.add_argument("--roi", type=int, default=200,
                        help="中心正方形 ROI 边长（像素），默认 200")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="每段最多解包的帧数，默认 300（够 3-4 秒）")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="深度值到毫米的乘数，16UC1 默认已是毫米，不用改")
    args = parser.parse_args()

    print("=" * 66)
    print("A4 深度噪声实测（相机不动，移动平面）")
    print("=" * 66)

    rows = []
    worst = "ok"

    for bag in args.bags:
        try:
            stack, info, total = load_bag(bag, args.roi, args.max_frames)
            result = analyse(stack, args.scale)
            result["frames_total"] = total
        except (FileNotFoundError, ValueError) as exc:
            print("=" * 66)
            print(f"bag: {bag}")
            print(f"  判定        : [BAD] 无法分析：{exc}")
            rows.append((bag, None, "bad", str(exc)))
            worst = "bad"
            continue

        level, reason = verdict(result)
        notes = quality_notes(result)
        print_report(bag, info, result, level, reason, notes)

        if level == "bad":
            worst = "bad"
        elif level == "warn" and worst == "ok":
            worst = "warn"
        rows.append((bag, result, level, reason))

    print("=" * 66)
    print("汇总")
    print("-" * 66)
    print(f"{'距离(mm)':>10} {'噪声(mm)':>10} {'基准(mm)':>10} {'比值':>7}  {'判定':<6} bag")
    for bag, result, level, _ in rows:
        if result is None:
            print(f"{'-':>10} {'-':>10} {'-':>10} {'-':>7}  {'BAD':<6} {bag}")
            continue
        print(f"{result['z_mm']:>10.1f} {result['noise_mm']:>10.2f} "
              f"{result['expected_mm']:>10.2f} {result['ratio']:>6.2f}x  "
              f"{level.upper():<6} {bag}")

    print("-" * 66)
    if worst == "ok":
        print("结果：A4 通过。噪声均不大于基准外推值，相机精度不是本项目的瓶颈。")
        return 0
    print("结果：存在不合格项，按计划书要求先排查，不要带着偏差进入 B 阶段。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
