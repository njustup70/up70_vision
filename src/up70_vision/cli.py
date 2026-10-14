"""命令行入口：运行逻辑的唯一实现处。

scripts/ 里的文件只做"薄入口"——它们 import 本模块并转交 argv，不写业务逻辑，
这样将来接 ROS 2 节点时，节点也只是"另一个调用方"，不必抄一份流程。

一次运行的流程
    读配置 -> 建输入源 -> 建检测器 -> 逐帧（检测 -> 判色 -> 画图 -> 落盘）-> 汇总
"""

import argparse
import os
import sys

from .config.loader import (load_config, load_thresholds, resolve_thresholds_path)
from .config.schema import ConfigError
from .core.detectors.base import build_detector
from .core.pipeline import ColorConfig, process_frames, summarize
from .io.base import build_source
from .utils.logging import get_logger, setup_logging
from .viz.draw import draw_detections, draw_hud
from .viz.writer import OutputWriter

log = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="up70-vision",
        description="UP70 视觉：颜色识别与场上目标感知（当前阶段：合成图/回放跑通判色链路）")
    ap.add_argument("--config", default=None, help="配置文件路径（默认 configs/default.yaml）")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="key=value", help="点分覆盖配置，可多次，如 --set color.hit_ratio_min=0.6")
    ap.add_argument("--out", default=None, help="覆盖 output.dir")
    ap.add_argument("--no-output", action="store_true", help="不落盘，只在终端看汇总")
    ap.add_argument("--dump-config", action="store_true", help="打印生效配置后退出")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少帧（0=不限）")
    return ap


def _print_config(cfg, config_path, thresholds_path) -> None:
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


def run(argv=None) -> int:
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
        _print_config(cfg, config_path, thresholds_path)
        return 0

    _print_config(cfg, config_path, thresholds_path)
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
    frame_counter = {"n": 0}
    overlays = []

    def on_result(frame, result):
        canvas = draw_detections(frame.image_bgr, result.detections)
        draw_hud(canvas, [
            f"{frame.source} #{frame.index}",
            f"det={len(result.detections)} uncertain={result.uncertain}",
            f"matched={result.matched}/{result.truth_total} correct={result.color_correct}",
        ])
        overlays.append(canvas)
        writer.write_frame(frame, result, canvas)
        frame_counter["n"] += 1
        if args.limit and frame_counter["n"] >= args.limit:
            raise StopIteration

    frames = source
    try:
        if args.limit:
            def limited():
                for i, frame in enumerate(source):
                    if i >= args.limit:
                        return
                    yield frame
            frames = limited()
        results = process_frames(frames, detector, thresholds, color_cfg, on_result=on_result)
    except StopIteration:
        results = []
    finally:
        source.close()

    if not results:
        print("❌ 一帧都没处理（输入源为空？）", file=sys.stderr)
        writer.close()
        return 1

    summary = summarize(results)
    written = writer.close(summary)

    # ---- 汇总与自检 ----
    print("\n" + "=" * 62)
    print(f"  帧数        : {summary['frames']}")
    print(f"  候选框总数  : {summary['detections']}（未判出颜色 {summary['uncertain']} 个）")
    print(f"  颜色分布    : {summary['color_counts']}")
    print(f"  平均命中率  : {summary['mean_ratio']}")
    if summary["truth_total"]:
        print(f"  与真值匹配  : {summary['matched']}/{summary['truth_total']}"
              f"  颜色正确 {summary['color_correct']}（正确率 {summary['color_accuracy']}）")
    print("=" * 62)

    ok = True
    if cfg.input.source == "synthetic":
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
    if cfg.output.enabled:
        print(f"  落盘        : {written}")
    print(f"\n{'✅ 运行成功' if ok else '❌ 自检未通过'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
