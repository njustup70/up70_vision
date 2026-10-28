"""命令行：`inspect` / `analyze` / `annotate` 三个子命令与退出码。

用大白话说
    `inspect` 先看 bag 里有什么；`analyze` 算配对、挑中间那一对、出图和报告；
    `annotate` 开窗口点边界。三个都是纯离线，不需要 source ROS，也不需要相机。

为什么入口固定在 `python -m lab.rgbd_check`
    仓库规范不新建 `scripts/`；`lab/` 里现有工具都是包 + `__main__.py` 的形式
    （见 `lab/pose_check`），新工具跟着走，别自成一套。

退出码（计划书 7.2）
    0 = 正常完成，已执行的检查没有失败（仍可能有 UNVERIFIED）
    1 = 分析完成，但有验收失败（FAIL）
    2 = 参数、输入、依赖或运行错误，没完成分析

已知的坑
    · 没有标注时空间结论必须是 UNVERIFIED，不能因为「配对看起来不错」就写 PASS。
    · 输出目录存在一律拒绝：报告里混进上一次的图，比没有报告更糟。
"""

import argparse
import json
import sys
import math
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from .bag import BagError, inspect_bag, load_images, scan
from .pairing import DEFAULT_TOL_NS, DEFAULT_WARMUP_NS, pair_frames
from .geometry import assess_geometry
from .grid import CELL_MM_DEFAULT, run_grid_check
from .report import (read_meta, refresh, save_selected, write_overlays, write_pairs,
                     write_report, write_unmatched, write_annotations, load_annotations,
                     write_spatial)
from .select import OFFSET_LIMIT_PX, REQUIRED_CONDITIONS, pick_mid_pair
from .status import FAIL, UNVERIFIED, worst

#: 旧 bag（raw_001..003）没有 capture.topics，只能按驱动默认话题猜；猜错了 scan 会报出实际话题。
DEFAULT_TOPICS = {
    "color": "/camera/color/image_raw",
    "depth": "/camera/depth/image_raw",
    "color_info": "/camera/color/camera_info",
    "depth_info": "/camera/depth/camera_info",
}


def build_parser() -> argparse.ArgumentParser:
    """参数表。子命令各自独立，避免 analyze 的参数混进 inspect。"""
    parser = argparse.ArgumentParser(
        prog="python -m lab.rgbd_check",
        description="离线检查同步 RGB-D bag：话题、配对、静态像素对齐")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="列出 bag 里的话题、编码、尺寸与内参")
    p_inspect.add_argument("bag", type=Path, help="bag 目录")

    p_summary = sub.add_parser("summarize", help="汇总三个场景的新报告，以 session + pair_id 区分图对")
    p_summary.add_argument("reports", nargs="+", type=Path, help="analyze 输出目录，每场景只提供一份")
    p_summary.add_argument("--out", required=True, type=Path, help="新的汇总目录")

    p_compare = sub.add_parser("compare-replay", help="比较原包与 ROS 回放后重录包的 header/内容/配对")
    p_compare.add_argument("original", type=Path)
    p_compare.add_argument("replay", type=Path)
    p_compare.add_argument("--out", required=True, type=Path)

    p_analyze = sub.add_parser("analyze", help="配对、选帧、出叠加图与报告")
    p_analyze.add_argument("bag", type=Path, help="bag 目录")
    p_analyze.add_argument("--out", type=Path, required=True, help="输出目录（必须不存在）")
    p_analyze.add_argument("--meta", type=Path, default=None,
                           help="场景 meta.yaml；默认读 bag 目录下的 meta.yaml")
    p_analyze.add_argument("--tol-ms", type=float, default=DEFAULT_TOL_NS / 1e6,
                           help="配对容差（毫秒），默认 10；不接受接近帧周期的值")
    p_analyze.add_argument("--warmup-s", type=float, default=None,
                           help="排除的预热时长（秒），默认读 meta.yaml，否则 3")
    p_analyze.add_argument("--annotations", type=Path, default=None,
                           help="已有标注 JSON；复制进输出目录并复算空间偏移")
    p_analyze.add_argument("--condition-id", default=None,
                          help="场景编号；默认取 meta.yaml 的 scene.condition_id")
    for role, topic in DEFAULT_TOPICS.items():
        p_analyze.add_argument(f"--{role.replace('_', '-')}-topic", default=None,
                               help=f"{role} 话题；默认取 meta.yaml，其次 {topic}")

    p_grid = sub.add_parser("grid-check", help="棋盘核查彩色与对齐深度是否同一像素网格")
    p_grid.add_argument("reports", nargs="+", type=Path, help="analyze 输出目录（需含 selected/ 与 summary.json）")
    p_grid.add_argument("--cell-mm", type=float, default=CELL_MM_DEFAULT,
                        help=f"棋盘方格边长（毫米），必须是尺量值；默认 {CELL_MM_DEFAULT}")
    p_grid.add_argument("--corners", default="6x9",
                        help="内角点排列，格式 列x行；默认 6x9（A4 竖版 7x10 格）")

    p_annotate = sub.add_parser("annotate", help="开本地窗口点选边界（每边 4 次点击）")
    p_annotate.add_argument("out", type=Path, help="analyze 的输出目录")
    p_annotate.add_argument("--unmeasurable-reason", default="",
                            help="按 u 记「不可测」时的默认原因。窗口在后台起时没有终端可输入，"
                                 "原因只能靠这个参数带进来，否则只会记成一句通用文案")
    p_annotate.add_argument("--clear-edges", action="store_true",
                            help="清掉已标的全部边再开窗口（重标整段时用）")
    p_annotate.add_argument("--drop-edge", action="append", default=[], metavar="EDGE_ID",
                            help="只删指定边（可重复，例如 --drop-edge edge2），其余保留")
    return parser


def _topics_from(meta: dict, args) -> dict:
    """话题优先级：命令行 > meta.yaml > 驱动默认值。"""
    configured = ((meta.get("capture") or {}).get("topics") or {})
    topics = {}
    for role, default in DEFAULT_TOPICS.items():
        given = getattr(args, f"{role}_topic", None)
        topics[role] = (given or configured.get(role) or default)
    return topics


def _flatten_sync(result) -> dict:
    """把配对结果摊平成报告要用的字段（键名固定，report.py 按名字取）。"""
    data = result.to_dict()
    return {
        "status": data["status"],
        "reasons": data["reasons"],
        "color_rate": result.rates.get("color"),
        "depth_rate": result.rates.get("depth"),
        "pairs": result.stats.get("pairs"),
        "color_frames": result.counts.get("color"),
        "depth_frames": result.counts.get("depth"),
        "abs_dt_p50_ms": result.stats.get("abs_dt_p50_ms"),
        "abs_dt_p95_ms": result.stats.get("abs_dt_p95_ms"),
        "abs_dt_max_ms": result.stats.get("abs_dt_max_ms"),
        "signed_dt_p50_ms": result.stats.get("signed_dt_p50_ms"),
        "signed_dt_min_ms": result.stats.get("signed_dt_min_ms"),
        "signed_dt_max_ms": result.stats.get("signed_dt_max_ms"),
        "warmup_dropped": data["warmup_dropped"],
        "anomaly_count": data["anomaly_count"],
        "frequency": data["frequency"],
        "rules": data["rules"],
    }


def run_inspect(args) -> int:
    """打印话题表与内参摘要；不写文件。"""
    try:
        report = inspect_bag(args.bag)
    except BagError as exc:
        print(f"❌ {exc}")
        return 2

    if not report["images"]:
        print(f"❌ {args.bag} 里没有图像话题；实际有：{report['bag_topics']}")
        return 2

    print(f"bag：{args.bag}")
    print(f"{'话题':<44}{'编码':<10}{'尺寸':<12}{'帧数':>6}  {'帧率':>6}  frame_id")
    for entry in report["images"]:
        size = f"{entry.get('width')}x{entry.get('height')}"
        fps = entry.get("fps_estimate")
        print(f"{entry['topic']:<44}{entry.get('encoding', '?'):<10}{size:<12}"
              f"{entry['messages']:>6}  {(f'{fps:.2f}' if fps else '—'):>6}  "
              f"{entry.get('frame_id', '')}")

    if report["infos"]:
        print("\n内参（CameraInfo 首帧）：")
        for info in report["infos"]:
            k = info.get("k") or [0] * 9
            print(f"  {info['topic']:<44} {info.get('width')}x{info.get('height')} "
                  f"fx={k[0]:.2f} fy={k[4]:.2f} cx={k[2]:.2f} cy={k[5]:.2f} "
                  f"D={len(info.get('d') or [])} 项 {info.get('distortion_model')}")
    else:
        print("\n⚠ 没有 CameraInfo：空间检查缺内参证据")
    return 0


def run_analyze(args) -> int:
    """配对 + 选帧 + 出图 + 写报告。"""
    out: Path = args.out
    if out.exists():
        print(f"❌ 输出目录已存在，拒绝覆盖：{out}")
        return 2
    if args.tol_ms <= 0:
        print("❌ --tol-ms 必须大于 0")
        return 2
    if not math.isfinite(args.tol_ms) or (args.warmup_s is not None and
                                       (not math.isfinite(args.warmup_s) or args.warmup_s < 0)):
        print("❌ 容差须为有限正数，预热须为有限非负数")
        return 2

    meta_path = args.meta or (args.bag / "meta.yaml")
    meta = read_meta(meta_path)
    topics = _topics_from(meta, args)
    capture = meta.get("capture") or {}
    if args.warmup_s is None:
        args.warmup_s = capture.get("warmup_s", DEFAULT_WARMUP_NS / 1e9)
    if not isinstance(args.warmup_s, (int, float)) or not math.isfinite(args.warmup_s) or args.warmup_s < 0:
        print("❌ 预热须为有限非负数")
        return 2
    duration_s = capture.get("duration_s")
    if duration_s is not None and (not isinstance(duration_s, (int, float)) or
                                   not math.isfinite(duration_s) or duration_s <= 0):
        print("❌ meta 中 duration_s 须为有限正数")
        return 2
    condition = (args.condition_id or (meta.get("scene") or {}).get("condition_id")
                 or "unassigned")

    try:
        scan_result = scan(args.bag, topics)
    except BagError as exc:
        print(f"❌ {exc}")
        return 2

    result = pair_frames(scan_result.frames["color"], scan_result.frames["depth"],
                         tol_ns=int(args.tol_ms * 1e6),
                         warmup_ns=int(args.warmup_s * 1e9),
                         stable_duration_s=duration_s)
    selection = pick_mid_pair(result, duration_s or 10.0)

    out.mkdir(parents=True)
    write_pairs(out, result)
    write_unmatched(out, result)

    selected_entry = None
    if selection is not None:
        try:
            frames = load_images(args.bag, topics["color"], [selection.pair.color.index],
                                 "color")
            depths = load_images(args.bag, topics["depth"], [selection.pair.depth.index],
                                 "depth")
        except BagError as exc:
            print(f"❌ 选中帧解码失败：{exc}")
            return 2
        color = frames[selection.pair.color.index]
        depth = depths[selection.pair.depth.index]
        pid = f"c{selection.pair.color.index}_d{selection.pair.depth.index}"
        label = (f"{condition} | {pid} | dt={selection.pair.dt_ns / 1e6:+.2f} ms | "
                 f"color {color.shape[1]}x{color.shape[0]} | depth {depth.shape[1]}x{depth.shape[0]}")
        write_overlays(out, pid, color, depth, label)
        selected_entry = save_selected(out, selection, color, depth, condition,
                                       session=meta.get("session") or args.bag.name)

    summary = {
        "schema_version": 1,
        "analysis_version": 2,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "session": meta.get("session") or args.bag.name,
        "bag": str(args.bag),
        "meta": str(meta_path),
        "capture": {
            "mode": capture.get("mode"),
            "color_topic": topics["color"],
            "depth_topic": topics["depth"],
            "color_info_topic": topics["color_info"],
            "depth_info_topic": topics["depth_info"],
            "color_profile": (capture.get("actual_profile") or {}).get("color"),
            "depth_profile": (capture.get("actual_profile") or {}).get("depth"),
            "time_domain": capture.get("time_domain"),
            "warmup_s": args.warmup_s,
            "duration_s": duration_s,
            "depth_unit": capture.get("depth_unit"),
            "driver_version": (meta.get("environment") or {}).get("driver_version"),
        },
        "camera_info": scan_result.infos,
        "sync": _flatten_sync(result),
        "capture_integrity": result.integrity,
        "geometry": assess_geometry(scan_result, meta),
        "tail_dropped": result.tail_dropped,
        "selected": selected_entry,
        "reasons": list(result.reasons),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.annotations:
        if not args.annotations.is_file():
            print(f"❌ 标注文件不存在：{args.annotations}")
            return 2
        try:
            data = json.loads(args.annotations.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"❌ 标注不是合法 JSON：{exc}")
            return 2
        write_annotations(out, data)
    elif not (out / "annotations.json").is_file():
        write_annotations(out, load_annotations(out))

    refresh(out, capture.get("mode"))
    write_report(out, json.loads((out / "summary.json").read_text(encoding="utf-8")))

    sync = summary["sync"]
    print(f"配对：{sync['pairs']} 对；配对率 彩色 {sync['color_rate']} / 深度 {sync['depth_rate']}"
          f"（容差 ±{args.tol_ms:g} ms，预热 {args.warmup_s:g} s）")
    print(f"时间同步：{sync['status']}")
    print(f"采集完整性：{result.integrity['status']}")
    for reason in result.integrity["reasons"]:
        print(f"  · {reason}")
    for reason in sync["reasons"]:
        print(f"  · {reason}")
    spatial = json.loads((out / "summary.json").read_text(encoding="utf-8")).get("spatial") or {}
    print(f"空间对齐：{spatial.get('status')}（门槛 {OFFSET_LIMIT_PX:g} px；"
          f"{'有' if selected_entry else '没有'}选中图对）")
    print(f"报告：{out / 'report.html'}")
    final = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    return 1 if final["overall_status"] == FAIL else 0


def run_summarize(args) -> int:
    """只汇总声明的场景；不按配对编号覆盖，也不自动挑选数值最好的重复场景。"""
    if args.out.exists():
        print(f"❌ 输出目录已存在，拒绝覆盖：{args.out}")
        return 2
    sources, rows, seen = [], [], set()
    try:
        for folder in args.reports:
            source = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            if source.get("analysis_version") != 2:
                raise ValueError(f"{folder} 是旧报告，请先用新版 analyze 复算")
            condition = (source.get("selected") or {}).get("condition_id")
            if condition not in REQUIRED_CONDITIONS:
                raise ValueError(f"{folder} 不是三场景之一")
            if condition in seen:
                raise ValueError(f"场景 {condition} 有重复报告；须明确保留哪一次采集")
            seen.add(condition)
            rows.extend(r for r in source["spatial"]["rows"] if r["condition_id"] == condition)
            sources.append((folder, source))
    except (OSError, ValueError, KeyError) as exc:
        print(f"❌ 无法汇总：{exc}")
        return 2
    for condition in REQUIRED_CONDITIONS:
        if condition not in seen:
            # 与逐包报告相同的两条占位，不允许漏采条件从表里消失。
            from .report import SPATIAL_COLUMNS
            for i in range(2):
                item = dict.fromkeys(SPATIAL_COLUMNS)
                item.update(condition_id=condition, edge_id=f"edge{i + 1}",
                            status=UNVERIFIED, measurement_status=UNVERIFIED,
                            limit_px=OFFSET_LIMIT_PX, note="本条件未提供报告")
                rows.append(item)
    summary = {"schema_version": 1, "analysis_version": 2, "session": "three_scenes",
               "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "capture": {"mode": "aligned"},
               "sources": [{"session": s["session"], "report": str(p.resolve() / "report.html")}
                           for p, s in sources], "reasons": []}
    for key in ("capture_integrity", "sync", "geometry"):
        summary[key] = {"status": worst(s[key]["status"] for _, s in sources),
                        "sessions": {s["session"]: s[key] for _, s in sources}}
    args.out.mkdir(parents=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows.sort(key=lambda r: (REQUIRED_CONDITIONS.index(r["condition_id"]), r["edge_id"]))
    write_spatial(args.out, rows)
    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    write_report(args.out, summary)
    print(f"三场景：{summary['spatial']['conditions']}；软件检查总体 {summary['overall_status']}")
    print(f"报告：{args.out / 'report.html'}")
    return 1 if summary["overall_status"] == FAIL else 0


def run_annotate(args) -> int:
    """开窗口点选，然后重算空间结果。"""
    from .annotate import run as run_window

    out: Path = args.out
    if not (out / "summary.json").is_file():
        print(f"❌ {out} 不是 analyze 的输出目录（缺 summary.json）")
        return 2
    code = run_window(out, unmeasurable_reason=args.unmeasurable_reason,
                      clear_edges=args.clear_edges, drop_edges=args.drop_edge)
    if code != 0:
        return code

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    rows = refresh(out, (summary.get("capture") or {}).get("mode"))
    for row in rows:
        value = row["abs_offset_px"]
        print(f"  {row['condition_id']:<12} {row['edge_id']:<6} "
              f"{'—' if value is None else f'{value:.2f} px':>10}  {row['status']}")
    final = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    spatial_status = (final.get("spatial") or {}).get("status", UNVERIFIED)
    print(f"空间对齐：{spatial_status}（门槛 {OFFSET_LIMIT_PX:g} px）")
    return 1 if final["overall_status"] == FAIL else 0


def main(argv=None) -> int:
    """解析参数并分发；异常一律收敛成退出码 2，不把堆栈甩给用户。"""
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        return run_inspect(args)
    if args.command == "analyze":
        return run_analyze(args)
    if args.command == "annotate":
        return run_annotate(args)
    if args.command == "summarize":
        return run_summarize(args)
    if args.command == "grid-check":
        try:
            cols, rows = (int(v) for v in str(args.corners).lower().split("x"))
        except ValueError:
            print(f"❌ --corners 需要写成 列x行，例如 6x9；收到 {args.corners}")
            return 2
        return run_grid_check(args.reports, args.cell_mm, (cols, rows))
    if args.command == "compare-replay":
        from .replay import compare_replay
        topics = ((read_meta(args.original / "meta.yaml").get("capture") or {}).get("topics")
                  or DEFAULT_TOPICS)
        try:
            result = compare_replay(args.original, args.replay, topics, args.out)
        except BagError as exc:
            print(f"❌ {exc}")
            return 2
        print(f"ROS 回放一致性：{result['status']}；报告：{args.out / 'report.html'}")
        return 1 if result["status"] == FAIL else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
