"""RGB-D 时间窗口与覆盖检查；纯整数计算，不读 bag，也不依赖 ROS。

录包外壳与离线工具共用这一规则，避免同一份数据在线说完整、离线说不足。
bag 接收时间只决定窗口，图像 header 时间仍由配对工具使用。
"""

import math


def capture_coverage(times: dict, warmup_ns: int, duration_s=None) -> dict:
    """检查两路图像是否覆盖 [最早图像+预热, 起点+声明时长]。

    times 为 role -> bag 整数纳秒序列。终点以后的帧证明录制覆盖完整，
    但不参与半开稳定窗口内的配对；不添加隐含的时间容差。
    """
    if warmup_ns < 0 or (duration_s is not None and
                         (not math.isfinite(duration_s) or duration_s <= 0)):
        raise ValueError("预热须非负，声明稳定时长须为有限正数")
    streams = {role: list(times.get(role, [])) for role in ("color", "depth")}
    all_times = [t for stream in streams.values() for t in stream]
    start = min(all_times) + warmup_ns if all_times else None
    end = start + round(duration_s * 1e9) if start is not None and duration_s else None
    result = {"status": "UNVERIFIED", "reasons": [], "declared_duration_s": duration_s,
              "stable_start_bag_ns": start, "stable_end_bag_ns": end, "streams": {}}
    for role, stream in streams.items():
        first, last = (min(stream), max(stream)) if stream else (None, None)
        missing_start = max(0, first - start) if first is not None else None
        missing_end = max(0, end - last) if last is not None and end is not None else None
        covered = max(0, min(last, end) - max(first, start)) \
            if first is not None and end is not None else None
        result["streams"][role] = {
            "first_bag_ns": first, "last_bag_ns": last,
            "available_after_warmup_s": max(0, last - start) / 1e9 if last is not None else None,
            "covered_duration_s": covered / 1e9 if covered is not None else None,
            "missing_start_s": missing_start / 1e9 if missing_start is not None else None,
            "missing_end_s": missing_end / 1e9 if missing_end is not None else None,
        }
        if not stream:
            result["reasons"].append(f"{role} 缺少图像")
        elif end is not None and (missing_start or missing_end):
            result["reasons"].append(
                f"{role} 覆盖不足：起点缺 {missing_start / 1e9:.6f} s，"
                f"末尾缺 {missing_end / 1e9:.6f} s")
    if end is None:
        result["reasons"].append("缺少声明时长或图像，无法验收稳定段覆盖")
    else:
        result["status"] = "FAIL" if result["reasons"] else "PASS"
    return result
