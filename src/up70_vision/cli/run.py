"""批处理入口 `run()`：读配置 -> 建源 -> 逐帧（检测 -> 判色 -> 画图 -> 落盘）-> 汇总。

用大白话说
    一趟把"手里这批帧"跑完再交报表：拿一帧，找目标、判颜色、画框、存图，
    全部跑完后把总数、颜色分布、命中率和真值比对结果打出来，顺便按自检规则判个对错。
    适合回归验收、离线素材、CI——不适合"盯着看"，要实时看用 `cli/live.py`。

放什么 / 不放什么
    放**编排**：按什么顺序建资源、什么时候收尾、退出码怎么定。
    不放具体实现——检测在 `core/`、读帧在 `io/`、画图落盘在 `viz/`、
    打印与自检判定在 `cli/report.py`。本文件读起来应该像一份流程说明。

依赖谁 / 被谁用
    config（配置/阈值表）、core（工厂 + 管线）、io（输入源）、viz（画图/落盘）、
    utils（日志）。被 `__main__.py` 的 `run` 子命令调用；
    将来 ROS 2 节点要批处理也走这里，不抄第二份流程。

对外约定
    `run(argv)` 返回退出码：0 自检通过 / 1 一帧都没处理或自检未通过 / 2 配置或环境错误。
    终端输出与拆分前逐字节一致（判据见 `cli/report.py` 的模块头）。
"""

import sys

from ..config.loader import (load_config, load_thresholds,
                             resolve_thresholds_path)
from ..config.schema import ConfigError
from ..core.detectors.base import build_detector
from ..core.pipeline import ColorConfig, process_frames, summarize
from ..io.base import build_source
from ..utils.logging import setup_logging
from ..viz.draw import draw_detections, draw_hud
from ..viz.writer import OutputWriter
from .parsers import build_parser
from .report import format_centers, print_batch_summary, print_config


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
