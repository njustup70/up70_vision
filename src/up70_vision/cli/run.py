"""批处理入口 `run()`：读配置 -> 建源 -> 逐帧（检测 -> 判色 -> 画图 -> 落盘）-> 汇总。

用大白话说
    一趟把"手里这批帧"跑完再交报表：拿一帧，找目标、判颜色、画框、存图，
    全部跑完后把总数、颜色分布、命中率和真值比对结果打出来，顺便按自检规则判个对错。
    适合回归验收、离线素材、CI——不适合"盯着看"，要实时看用 `cli/live.py`。

放什么 / 不放什么
    放**编排**：按什么顺序建资源、什么时候收尾、退出码怎么定。
    不放具体实现——检测在 `detectors/`、读帧在 `io/`、画图落盘在 `viz/`、
    打印与自检判定在本文件末尾的「终端输出」一节。本文件读起来应该像一份流程说明。

依赖谁 / 被谁用
    config（配置/阈值表）、detectors 与 pipeline（工厂 + 管线）、io（输入源）、
    viz（画图/落盘）、logging（日志）。被 `__main__.py` 的 `run` 子命令调用；
    将来 ROS 2 节点要批处理也走这里，不抄第二份流程。

对外约定
    `run(argv)` 返回退出码：0 自检通过 / 1 一帧都没处理或自检未通过 / 2 配置或环境错误。
    终端输出与拆分前逐字节一致——它只搬了代码位置，没改一个字符。
"""

import sys

from ..config.loader import (load_config, load_thresholds,
                             resolve_thresholds_path)
from ..config.schema import ConfigError
from ..detectors.base import build_detector
from ..pipeline import ColorConfig, process_frames, summarize
from ..types import is_marker
from ..io.base import build_source
from ..logging import setup_logging
from ..viz.draw import draw_detections, draw_hud
from ..viz.writer import OutputWriter
from .parsers import build_parser


def run(argv=None) -> int:
    """判色批处理入口（原有功能，未改动行为）。

    用大白话说
        按配置决定画面从哪来（合成图 / 回放 / 相机），一帧一帧地检测并判色，
        边跑边把结果落盘、往终端打一行；全部跑完后交一份汇总报表。

    流程：读配置 -> 建输入源 -> 建检测器 -> 逐帧（检测 -> 判色 -> 画图 -> 落盘）-> 汇总自检。
    与 `cli.live.run_aruco_live()` 的分工：这里跑完全部帧再汇总，适合合成图/回放与回归验收；
    实时显示用 `run_aruco_live()`。

    :param argv: 命令行参数列表；None 表示取 `sys.argv[1:]`
    :return: 退出码。0 = 自检通过；1 = 自检未通过 / 一帧都没处理；2 = 配置或环境错误

    副作用
        读配置文件与阈值表；按 `output.*` 写叠加图 / JSON / CSV；
        选 `input.source=camera` 时占用相机（结束时 `close()` 释放）。
    """
    args = build_parser().parse_args(argv)

    try:
        cfg, config_path = load_config(args.config, args.overrides)
    except ConfigError as exc:
        print(f"❌ 配置错误：{exc}", file=sys.stderr)
        return 2
    if args.out:
        cfg.output.dir = args.out
    if args.no_output:
        cfg.output.enabled = False

    setup_logging(cfg.log.level)
    thresholds_path = resolve_thresholds_path(cfg, config_path)
    try:
        thresholds = load_thresholds(thresholds_path)
    except ConfigError as exc:
        print(f"❌ 阈值表错误：{exc}", file=sys.stderr)
        return 2

    if args.dump_config:
        print_config(cfg, config_path, thresholds_path)
        return 0

    print_config(cfg, config_path, thresholds_path)
    print(f"颜色       : {list(thresholds)}")
    if cfg.input.source == "camera":
        print("提示       : 相机是独占资源（MVS 客户端 / 实时窗口同时只能开一个）；"
              "实时流不会自己结束，用 --limit N 或 Ctrl+C 停。")

    # ---- 建输入源 / 检测器 / 输出 ----
    try:
        source = build_source(cfg.input.source,
                              synthetic_params=dict(vars(cfg.input.synthetic)),
                              replay_params=dict(vars(cfg.input.replay)),
                              camera_params=dict(vars(cfg.input.camera)),
                              colors=list(thresholds))
    except (ValueError, FileNotFoundError, IOError, RuntimeError) as exc:
        print(f"❌ 无法建立输入源：{exc}", file=sys.stderr)
        return 2

    detector = build_detector(cfg.detector.name, cfg.detector.params)
    color_cfg = ColorConfig(min_saturation=cfg.color.min_saturation,
                            min_value=cfg.color.min_value,
                            hit_ratio_min=cfg.color.hit_ratio_min,
                            ignore_border_px=cfg.color.ignore_border_px,
                            min_valid_pixels=cfg.color.min_valid_pixels)
    writer = OutputWriter(cfg.output.dir,
                          enabled=cfg.output.enabled,
                          save_overlay=cfg.output.save_overlay,
                          save_json=cfg.output.save_json,
                          save_csv=cfg.output.save_csv,
                          prefix="frame")

    # ---- 逐帧处理 ----

    def on_result(frame, result):
        """每帧回调：画框 + HUD，落盘，然后打一行终端摘要。"""
        canvas = draw_detections(frame.image_bgr, result.detections)
        hud = [
            f"{frame.source} #{frame.index}",
            f"det={len(result.detections)} uncertain={result.uncertain}",
            f"matched={result.matched}/{result.truth_total} correct={result.color_correct}",
        ]
        if result.ids:
            hud.insert(1, f"ids={result.ids}")
        draw_hud(canvas, hud)
        writer.write_frame(frame, result, canvas)
        # 终端输出（任务单要求：实时绘制 + 终端输出，且要带 ID 与中心点坐标）
        print(f"[帧{frame.index:>3}] {frame.source} 目标={len(result.detections)} "
              f"ids={result.ids} 中心={format_centers(result.detections)} "
              f"颜色={result.color_counts()} 未判定={result.uncertain}")

    frames = source
    if args.limit:
        def limited():
            """只放行前 limit 帧。

            **限帧只在这一处做**。以前 on_result 里还额外 `raise StopIteration` 想断流，
            结果把 process_frames 打断、results 被外层 `except StopIteration` 清空，
            于是 `--limit N` 一律报"一帧都没处理"并返回 1（复核意见一.1，见 progress.md 坑 30）。
            """
            for i, frame in enumerate(source):
                if i >= args.limit:
                    return
                yield frame
        frames = limited()

    results: list = []
    summary: dict | None = None
    try:
        results = process_frames(frames, detector, thresholds, color_cfg, on_result=on_result)
        if results:
            summary = summarize(results)
    finally:
        # 输入源与输出都要收尾。以前 writer.close(summary) 只在成功路径调用，
        # 中途抛异常时 results.json / summary.json 会整个丢（复核意见二.5）。
        source.close()
        written = writer.close(summary)

    if not results:
        print("❌ 一帧都没处理（输入源为空？）", file=sys.stderr)
        return 1

    # ---- 汇总与自检（打印在 report.py，这里只取"过没过"）----
    return 0 if print_batch_summary(cfg, summary, written) else 1


# ================================================================ 终端输出
# 以下三个函数原先在 cli/report.py（2026-10-22 合并进来：只有本文件用，独立成文件让"找一个功能"多跳一次）。
# 它们只负责排版与判定，不改任何检测结果。

def format_centers(detections, limit: int = 8) -> str:
    """把每个检出物的 "id@(x,y)" 拼成一行（超出 limit 个就截断）。

    用大白话说
        一帧里可能有十几个框，全打出来会刷屏，所以只列前 8 个，剩下的折成 `…+N`。
        ArUco 码前面写它的 ID，色块没有 ID，写个 `-` 占位。

    飞书任务单任务二第 4 条要求"在终端输出识别到的 ArUco 码信息，**包括 ID 和中心点坐标**"。
    判色族的框没有码 ID，写成 `-@(x,y)`；一帧十几个框时截断，免得刷屏。

    :param detections: 本帧的检出物列表（`Detection` 与 `ArucoMarker` 混排也可）
    :param limit: 最多列几个；超出部分只报个数
    :return: 形如 `[12@(320,240) -@(88,96) …+3]` 的单行字符串
    """
    parts = []
    for det in list(detections)[:limit]:
        cx, cy = det.center
        tag = str(int(det.marker_id)) if is_marker(det) else "-"
        parts.append(f"{tag}@({int(round(cx))},{int(round(cy))})")
    if len(detections) > limit:
        parts.append(f"…+{len(detections) - limit}")
    return "[" + " ".join(parts) + "]"


def print_config(cfg, config_path, thresholds_path) -> None:
    """把本次真正生效的配置打一遍（两个入口共用）。

    目的不是"好看"，而是让日志/PR 里能直接看到**实际生效**的值：
    配置文件路径、阈值表路径、输入源与它的参数、判色阈值、检测器、输出、日志等级。
    用 `--set` 覆盖后，这里打印的就是覆盖后的结果，不用再回头翻命令行。

    :param cfg: 校验过的 `AppConfig`
    :param config_path: 配置文件绝对路径（便于确认读的是哪一份）
    :param thresholds_path: 解析后的阈值表绝对路径
    **本函数在 `cli/run.py` 与 `cli/live.py` 里各有一份完全相同的副本**（2026-10-22 合并 report.py 时的决定：不为 30 行打印代码再开一个共享文件）。
    改一处必须同步另一处：`grep -rn "def print_config" src/` 能找全两处。
    """
    print(f"配置文件   : {config_path}")
    print(f"阈值表     : {thresholds_path}")
    print(f"输入源     : {cfg.input.source}")
    if cfg.input.source == "synthetic":
        s = cfg.input.synthetic
        print(f"  合成     : {s.width}x{s.height} {s.num_frames}帧 x {s.blocks_per_frame}块 "
              f"seed={s.seed} 噪声={s.noise_sigma}")
    elif cfg.input.source == "camera":
        c = cfg.input.camera
        print(f"  相机     : 第{c.index}台 曝光{c.exposure_us:.0f}us 增益{c.gain_db:.1f}dB "
              f"超时{c.timeout_ms}ms 最多{c.max_frames or '不限'}帧")
    else:
        r = cfg.input.replay
        print(f"  回放     : {r.path} kind={r.kind} stride={r.frame_stride} max={r.max_frames}")
    print(f"判色       : S>={cfg.color.min_saturation} V>={cfg.color.min_value} "
          f"命中率>={cfg.color.hit_ratio_min} 最少有效像素={cfg.color.min_valid_pixels}")
    print(f"检测器     : {cfg.detector.name} {cfg.detector.params}")
    print(f"输出       : enabled={cfg.output.enabled} dir={cfg.output.dir} "
          f"overlay={cfg.output.save_overlay} json={cfg.output.save_json} csv={cfg.output.save_csv}")
    print(f"日志等级   : {cfg.log.level}")


def print_batch_summary(cfg, summary, written) -> bool:
    """打批处理的汇总与自检，返回"自检是否通过"。

    用大白话说
        全部帧跑完了，先把数字列出来（几帧、检出多少、颜色分布、命中率、和真值比对了多少），
        再按"这一批输入是什么 + 用了哪个检测器"挑一条自检规则来核对，
        最后给一句 ✅/❌。这一句就是 `run()` 的退出码依据。

    :param cfg: 已校验的 `AppConfig`（自检规则要按 input.source 与 detector.name 分支）
    :param summary: `pipeline.summarize()` 的返回值
    :param written: `OutputWriter.close()` 的返回值（落盘路径字典）
    :return: True = 自检通过（调用方返回 0）；False = 未通过（调用方返回 1）

    自检规则表（改之前先看这里）
        ┌──────────────────────────────┬──────────────────────────────────────┐
        │ 条件                          │ 要求                                  │
        ├──────────────────────────────┼──────────────────────────────────────┤
        │ synthetic + blob              │ 检出数 == 帧数 x blocks_per_frame；    │
        │                               │ 匹配上的框颜色全对                     │
        │ synthetic + aruco             │ 合成图里没有码，检出数必须为 0（不误报）│
        │ synthetic + 其它检测器         │ 跳过色块数检查（色块数对它不适用）      │
        │ 任意源 + stub_aruco            │ 检出数 == 帧数 x len(ids)，且 id 集合一致│
        │ 非合成源 + aruco               │ 无真值，只报检出数量，不做对错判定      │
        └──────────────────────────────┴──────────────────────────────────────┘
    """
    print("\n" + "=" * 62)
    print(f"  帧数        : {summary['frames']}")
    print(f"  检出总数    : {summary['detections']}（未判出颜色 {summary['uncertain']} 个）")
    if summary["ids"]:
        print(f"  出现过的 id : {summary['ids']}")
    if summary["color_counts"]:
        print(f"  颜色分布    : {summary['color_counts']}")
        print(f"  平均命中率  : {summary['mean_ratio']}")
    if summary["truth_total"]:
        print(f"  与真值匹配  : {summary['matched']}/{summary['truth_total']}"
              f"  颜色正确 {summary['color_correct']}（正确率 {summary['color_accuracy']}）")
    print("=" * 62)

    ok = True
    # 合成图自检只对 blob 有意义：那 18 个色块本来就是给 blob 找的。
    # 换成 stub_aruco（伪造码）时不该拿色块数去要求它。
    if cfg.input.source == "synthetic" and cfg.detector.name == "blob":
        expected = cfg.input.synthetic.num_frames * cfg.input.synthetic.blocks_per_frame
        got = summary["detections"]
        print(f"  自检(合成图): 期望 {expected} 个色块，实得 {got} 个 -> "
              f"{'✅ 通过' if got == expected else '❌ 不一致'}")
        ok = ok and got == expected
        if summary["matched"]:
            all_right = summary["color_correct"] == summary["matched"]
            print(f"  自检(判色)  : 匹配 {summary['matched']} 个里判对 "
                  f"{summary['color_correct']} 个 -> {'✅ 通过' if all_right else '❌ 有判错'}")
            ok = ok and all_right
    elif cfg.input.source == "synthetic" and cfg.detector.name == "aruco":
        # 合成图里只有色块、一个 ArUco 码都没有：这一条顺手验了验收清单第 3 条
        # （画面无码时程序正常运行、不误报）
        got = summary["detections"]
        print(f"  自检(无码)  : 合成图里没有 ArUco 码，期望 0 个，实得 {got} 个 -> "
              f"{'✅ 通过' if got == 0 else '❌ 误报'}")
        ok = ok and got == 0
    elif cfg.input.source == "synthetic" and cfg.detector.name != "blob":
        print(f"  自检(合成图): 跳过节数检查（检测器是 {cfg.detector.name}，色块数不适用）")
    if cfg.detector.name == "stub_aruco":
        want_ids = list(cfg.detector.params.get("ids", [0, 1, 2]))
        expected = summary["frames"] * len(want_ids)
        got = summary["detections"]
        ids_ok = set(summary["ids"]) == set(int(i) for i in want_ids) or not want_ids
        print(f"  自检(stub)  : 期望 {expected} 个码、id 集合 {sorted(int(i) for i in want_ids)}，"
              f"实得 {got} 个、id 集合 {summary['ids']} -> "
              f"{'✅ 通过' if (got == expected and ids_ok) else '❌ 不一致'}")
        ok = ok and got == expected and ids_ok
    if cfg.detector.name == "aruco" and cfg.input.source != "synthetic":
        # 回放/相机源没有 ground truth，CLI 这边只能报"检出了什么"；
        # 要按真值逐张核 ID / 四角 / 中心，那是 tools/aruco/check.py 的活。
        print(f"  自检(aruco) : 本次共 {summary['detections']} 个码、id 集合 {summary['ids']}"
              f"（本输入源无真值，不做对错判定）")
        print("              按真值逐张核 ID/四角/中心：python -m up70_vision check-aruco")
    if cfg.output.enabled:
        print(f"  落盘        : {written}")
    print(f"\n{'✅ 运行成功' if ok else '❌ 自检未通过'}")
    return ok

