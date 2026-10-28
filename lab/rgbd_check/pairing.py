"""确定性一对一配对：把同一段时间里的彩色帧与深度帧配成对，并给出可复算的统计。

用大白话说
    彩色和深度各发各的，配对的唯一依据是 header 里的拍摄时间。规则写死成：
    以彩色为基准逐帧处理，在「还没用过、且晚于上次匹配深度」的候选里，
    选时间差最小的那一帧；差一样时选更早拍的；每帧最多用一次。
    同样的 bag、同样的容差，跑多少遍都是同一份配对索引——这是后面所有
    偏移结论能复查的前提。

为什么不求「最大匹配数」
    最大匹配会把某个彩色帧硬塞给一个远处的深度帧，只为了让配对率好看。
    本阶段要查的是「时间上对不对得上」，不是「能不能凑满」；凑出来的配对
    会把真正的问题掩盖掉。

已知的坑
    · 时间戳全用整数纳秒（`sec*1e9+nanosec`）。转成浮点秒再算差，几十纳秒
      的差异会被吃掉，10 ms 容差下的边界判定就不可复现了。
    · 零时间戳不能参与配对，但**要计入分母**：它是「收到过、但不可用」的证据。
    · 重复时间戳只让首帧（bag 顺序里第一个）参与配对；倒退、重复、零值
      都算完整性问题，哪怕配对率看起来很好也要报 FAIL。
    · 预热段（默认前 3 秒）只用于分段排除，不做「先配完再删」，否则
      「上次匹配深度」这个指针会跨过预热边界，配对结果跟分段方式有关。
"""

import bisect
from dataclasses import dataclass, field

from vision.depth.timing import capture_coverage

from .bag import FrameMeta
from .status import FAIL, PASS, UNVERIFIED

#: 默认配对容差（纳秒）。10 ms 是计划书定的工程门槛，不是厂家精度承诺。
#: 30 fps 的帧周期是 33.3 ms，容差不能接近它，否则相邻两拍会被配成一对。
DEFAULT_TOL_NS = 10_000_000

#: 默认预热（纳秒）：bag 开头这段驱动在丢帧、曝光在收敛，统计一律排除。
DEFAULT_WARMUP_NS = 3_000_000_000

#: 双流同频配置下的配对率门槛。两侧都要满足。
MIN_PAIR_RATE = 0.99


@dataclass(frozen=True)
class Pair:
    """一对配上的帧。`dt_ns` 定义为 深度 - 彩色，正数表示深度拍得晚。"""

    color: FrameMeta
    depth: FrameMeta
    dt_ns: int

    @property
    def abs_dt_ns(self) -> int:
        return abs(self.dt_ns)


@dataclass
class PairingResult:
    """配对结果 + 统计 + 状态。`to_dict()` 直接进 summary.json。"""

    pairs: list = field(default_factory=list)
    unmatched: list = field(default_factory=list)
    anomalies: list = field(default_factory=list)
    nearest: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    rates: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    frequency: dict = field(default_factory=dict)
    integrity: dict = field(default_factory=dict)
    tail_dropped: dict = field(default_factory=dict)
    stable_start_bag_ns: int = 0
    warmup_dropped: dict = field(default_factory=dict)
    status: str = UNVERIFIED
    reasons: list = field(default_factory=list)
    tol_ns: int = DEFAULT_TOL_NS
    warmup_ns: int = DEFAULT_WARMUP_NS

    def to_dict(self) -> dict:
        """给 summary.json 的精简视图（不含逐帧列表，那些另有 CSV）。"""
        return {
            "status": self.status,
            "reasons": self.reasons,
            "rules": {
                "tol_ns": self.tol_ns,
                "warmup_ns": self.warmup_ns,
                "min_pair_rate": MIN_PAIR_RATE,
                "signed_dt": "depth_stamp_ns - color_stamp_ns",
                "match": "以彩色为基准，容差内取时间差最小；等距取较早，再按原始序号；每帧最多一次",
                "quantile": "线性插值（numpy.percentile 默认）",
            },
            "counts": self.counts,
            "rates": self.rates,
            "stats": self.stats,
            "frequency": self.frequency,
            "integrity": self.integrity,
            "tail_dropped": self.tail_dropped,
            "warmup_dropped": self.warmup_dropped,
            "anomaly_count": len(self.anomalies),
            "stable_start_bag_ns": self.stable_start_bag_ns,
        }


def _percentile(values, q: float):
    """线性插值分位数；空集合返回 None（报告里写 null，不写 0 冒充测量）。"""
    if not values:
        return None
    import numpy as np                                     # 只在统计时用到

    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _interval_stats(bag_times_ns: list, stable_s) -> dict:
    """帧间隔与频率。频率按「稳定段帧数 / 声明稳定时长」算。

    为什么不用 1/平均间隔：驱动在启动时会一次吐好几帧、之后长时间停发，
    平均间隔会被那几帧稀释，看起来「帧率正常」，其实已经停了。
    """
    times = sorted(bag_times_ns)
    gaps = [b - a for a, b in zip(times, times[1:])]
    count = len(times)
    fps = None
    if stable_s and stable_s > 0:
        fps = round(count / stable_s, 3)
    return {
        "frames": count,
        "fps": fps,
        "gap_p50_ms": None if _percentile(gaps, 50) is None
        else round(_percentile(gaps, 50) / 1e6, 2),
        "gap_p95_ms": None if _percentile(gaps, 95) is None
        else round(_percentile(gaps, 95) / 1e6, 2),
        "gap_max_ms": None if not gaps else round(max(gaps) / 1e6, 2),
        "span_s": None if not times else round((times[-1] - times[0]) / 1e9, 3),
    }


def _scan_anomalies(frames, role: str) -> tuple:
    """按 bag 原始顺序查零值 / 重复 / 倒退时间戳。

    :return: `(异常列表, 允许参与配对的序号集合)`；重复只放行首帧。
    """
    anomalies = []
    allowed = set()
    seen = {}
    prev = None
    for frame in frames:
        if frame.stamp_ns == 0:
            anomalies.append({"role": role, "index": frame.index, "stamp_ns": 0,
                              "kind": "zero_stamp", "reason": "header 时间戳为 0"})
            continue
        if frame.stamp_ns in seen:
            anomalies.append({"role": role, "index": frame.index,
                              "stamp_ns": frame.stamp_ns, "kind": "duplicate_stamp",
                              "reason": f"与序号 {seen[frame.stamp_ns]} 时间戳相同"})
            continue
        seen[frame.stamp_ns] = frame.index
        if prev is not None and frame.stamp_ns < prev:
            anomalies.append({"role": role, "index": frame.index,
                              "stamp_ns": frame.stamp_ns, "kind": "backwards_stamp",
                              "reason": f"小于前一帧的 {prev}"})
            continue
        prev = frame.stamp_ns
        allowed.add(frame.index)
    return anomalies, allowed


def frequency_stats(frames, cutoff: int, duration_s) -> dict:
    """接收间隔、header 频率和固定 10 秒趋势，三者不互相冒充。"""
    stats = _interval_stats([f.bag_time_ns for f in frames], duration_s)
    stamps = [f.stamp_ns for f in frames]
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    valid = bool(gaps) and all(g > 0 for g in gaps) and all(t > 0 for t in stamps)
    stats["header_fps"] = round((len(stamps) - 1) * 1e9 / (stamps[-1] - stamps[0]), 3) \
        if valid else None
    stats["header_gap_p95_ms"] = round(_percentile(gaps, 95) / 1e6, 3) if valid else None
    stats["header_gap_max_ms"] = round(max(gaps) / 1e6, 3) if valid else None
    stats["fps_definition"] = "稳定窗口帧数 / 声明时长；header_fps 为 (帧数-1) / header 跨度"
    stats["windows_10s"] = []
    if duration_s:
        duration_ns = round(duration_s * 1e9)
        for left in range(0, duration_ns, 10_000_000_000):
            right = min(left + 10_000_000_000, duration_ns)
            window = [f for f in frames if cutoff + left <= f.bag_time_ns < cutoff + right]
            item = _interval_stats([f.bag_time_ns for f in window], (right - left) / 1e9)
            h = [f.stamp_ns for f in window]
            item["header_fps"] = round((len(h) - 1) * 1e9 / (h[-1] - h[0]), 3) \
                if len(h) > 1 and all(b > a > 0 for a, b in zip(h, h[1:])) else None
            item.update(start_s=left / 1e9, end_s=right / 1e9)
            stats["windows_10s"].append(item)
    return stats


def pair_frames(colors, depths, tol_ns: int = DEFAULT_TOL_NS,
                warmup_ns: int = DEFAULT_WARMUP_NS,
                stable_duration_s=None) -> PairingResult:
    """主函数：配对 + 统计 + 状态。

    :param colors: 彩色 `FrameMeta` 列表（bag 原始顺序）
    :param depths: 深度 `FrameMeta` 列表（bag 原始顺序）
    :param tol_ns: 配对容差（纳秒），默认 10 ms
    :param warmup_ns: 预热时长（纳秒），默认 3 s
    :param stable_duration_s: 声明的稳定段时长（秒）；None 时用 bag 跨度估
    :return: `PairingResult`
    """
    result = PairingResult(tol_ns=tol_ns, warmup_ns=warmup_ns)
    result.integrity = capture_coverage(
        {"color": [f.bag_time_ns for f in colors],
         "depth": [f.bag_time_ns for f in depths]}, warmup_ns, stable_duration_s)
    if not colors or not depths:
        result.reasons.append(f"缺流：彩色 {len(colors)} 帧、深度 {len(depths)} 帧")
        result.status = UNVERIFIED
        return result

    all_frames = list(colors) + list(depths)
    start = min(f.bag_time_ns for f in all_frames)
    cutoff = start + warmup_ns
    result.stable_start_bag_ns = cutoff

    # 完整性问题在分段之前查：预热里的坏时间戳同样是证据，不能因为「反正要删」就跳过。
    color_anomalies, color_allowed = _scan_anomalies(colors, "color")
    depth_anomalies, depth_allowed = _scan_anomalies(depths, "depth")
    result.anomalies = color_anomalies + depth_anomalies

    end = result.integrity["stable_end_bag_ns"]
    stable_colors = [f for f in colors if f.bag_time_ns >= cutoff and
                     (end is None or f.bag_time_ns < end)]
    stable_depths = [f for f in depths if f.bag_time_ns >= cutoff and
                     (end is None or f.bag_time_ns < end)]
    result.warmup_dropped = {
        "color": sum(f.bag_time_ns < cutoff for f in colors),
        "depth": sum(f.bag_time_ns < cutoff for f in depths),
    }
    result.tail_dropped = {role: sum(end is not None and f.bag_time_ns >= end for f in frames)
                           for role, frames in (("color", colors), ("depth", depths))}
    result.counts = {"color": len(stable_colors), "depth": len(stable_depths)}

    legal_colors = sorted((f for f in stable_colors if f.index in color_allowed),
                          key=lambda f: (f.stamp_ns, f.index))
    legal_depths = sorted((f for f in stable_depths if f.index in depth_allowed),
                          key=lambda f: (f.stamp_ns, f.index))

    # 诊断用：不设容差的最近帧时间差，允许复用，不参与正式配对与配对率。
    if legal_depths:
        stamps = [d.stamp_ns for d in legal_depths]
        for color in legal_colors:
            pos = _nearest_index(stamps, color.stamp_ns)
            nearest = legal_depths[pos]
            result.nearest.append({
                "color_index": color.index,
                "depth_index": nearest.index,
                "dt_ms": round((nearest.stamp_ns - color.stamp_ns) / 1e6, 3),
            })

    used = set()
    start_ptr = 0
    for color in legal_colors:
        while start_ptr < len(legal_depths) and legal_depths[start_ptr].index in used:
            start_ptr += 1
        best = None
        j = start_ptr
        while j < len(legal_depths) and legal_depths[j].stamp_ns <= color.stamp_ns + tol_ns:
            depth = legal_depths[j]
            if depth.index not in used and depth.stamp_ns >= color.stamp_ns - tol_ns:
                dt = depth.stamp_ns - color.stamp_ns
                key = (abs(dt), depth.stamp_ns, depth.index)
                if best is None or key < best[0]:
                    best = (key, depth, dt, j)
            j += 1
        if best is None:
            result.unmatched.append({
                "role": "color", "index": color.index, "stamp_ns": color.stamp_ns,
                "reason": f"±{tol_ns / 1e6:.1f} ms 内没有未被占用的深度帧",
            })
            continue
        _, depth, dt, position = best
        used.add(depth.index)
        start_ptr = max(start_ptr, position + 1)
        result.pairs.append(Pair(color=color, depth=depth, dt_ns=dt))

    matched_depths = {p.depth.index for p in result.pairs}
    for depth in legal_depths:
        if depth.index in matched_depths:
            continue
        result.unmatched.append({
            "role": "depth", "index": depth.index, "stamp_ns": depth.stamp_ns,
            "reason": "没有彩色帧在容差内选中它（或已被更近的彩色帧占用）",
        })

    # 异常帧也要计入分母：漏掉它们会把配对率算高。
    result.rates = {
        "color": round(len(result.pairs) / len(stable_colors), 4) if stable_colors else 0.0,
        "depth": round(len(result.pairs) / len(stable_depths), 4) if stable_depths else 0.0,
    }

    stable_s = stable_duration_s
    if stable_s is None and stable_colors and stable_depths:
        last = max(f.bag_time_ns for f in stable_colors + stable_depths)
        first = min(f.bag_time_ns for f in stable_colors + stable_depths)
        stable_s = (last - first) / 1e9

    abs_dt = [p.abs_dt_ns for p in result.pairs]
    signed = [p.dt_ns for p in result.pairs]
    result.stats = {
        "pairs": len(result.pairs),
        "abs_dt_p50_ms": None if _percentile(abs_dt, 50) is None
        else round(_percentile(abs_dt, 50) / 1e6, 3),
        "abs_dt_p95_ms": None if _percentile(abs_dt, 95) is None
        else round(_percentile(abs_dt, 95) / 1e6, 3),
        "abs_dt_max_ms": None if not abs_dt else round(max(abs_dt) / 1e6, 3),
        "signed_dt_p50_ms": None if _percentile(signed, 50) is None
        else round(_percentile(signed, 50) / 1e6, 3),
        "signed_dt_min_ms": None if not signed else round(min(signed) / 1e6, 3),
        "signed_dt_max_ms": None if not signed else round(max(signed) / 1e6, 3),
    }
    result.frequency = {
        "color": frequency_stats(stable_colors, cutoff, stable_s),
        "depth": frequency_stats(stable_depths, cutoff, stable_s),
    }

    # 状态：时间戳异常一票否决；配对率两侧都要过线；没配上任何一对照样不算通过。
    if not result.pairs:
        result.status = UNVERIFIED
        result.reasons.append("稳定段内没有配上任何一帧")
        return result

    if result.anomalies:
        result.status = FAIL
        kinds = sorted({a["kind"] for a in result.anomalies})
        result.reasons.append(f"时间戳异常 {len(result.anomalies)} 处（{', '.join(kinds)}），"
                              "完整性检查不通过")
    low = [role for role, rate in result.rates.items() if rate < MIN_PAIR_RATE - 1e-12]
    if low:
        result.status = FAIL
        for role in low:
            result.reasons.append(f"{role} 配对率 {result.rates[role]:.2%} < {MIN_PAIR_RATE:.0%}")
    if result.status != FAIL:
        result.status = PASS
        result.reasons.append(
            f"配对率 彩色 {result.rates['color']:.2%} / 深度 {result.rates['depth']:.2%}，"
            f"容差 ±{tol_ns / 1e6:.1f} ms；P95 仅作描述，不作为独立通过证据")
    return result


def _nearest_index(sorted_stamps: list, target: int) -> int:
    """在有序时间戳里找离 target 最近的位置；等距取较早那个（与配对规则一致）。"""
    pos = bisect.bisect_left(sorted_stamps, target)
    if pos == 0:
        return 0
    if pos >= len(sorted_stamps):
        return len(sorted_stamps) - 1
    before, after = sorted_stamps[pos - 1], sorted_stamps[pos]
    return pos - 1 if target - before <= after - target else pos
