"""命令行入口：运行逻辑的唯一实现处。

用大白话说这个文件在干什么
    相机（或图片目录、或程序自己画的合成图）会一帧一帧地把画面送进来。
    我们每拿到一帧，就找一次 ArUco 码，在画面上把码的四个角框出来、
    把 ID 和中心坐标打到终端。看窗口的人按 q 就停。
    两个入口只是"节奏"不同：
        · run()            像交作业：全部跑完再给一份汇总报表（回归测试、离线素材用）
        · run_aruco_live() 像看直播：每来一帧就刷新一次画面，随时可以停（演示用）

scripts/ 里的文件只做"薄入口"——它们 import 本模块并转交 argv，不写业务逻辑，
这样将来接 ROS 2 节点时，节点也只是"另一个调用方"，不必抄一份流程。

两个入口
    run()            判色批处理：读配置 -> 建输入源 -> 建检测器 -> 逐帧（检测 -> 判色 -> 画图 -> 落盘）-> 汇总
    run_aruco_live() 实时 ArUco：逐帧 检测 -> 绘制 -> 窗口/落盘，终端节流输出（任务二）

共同的骨架（两个入口都按这个顺序，顺序本身有原因）
    1. 解析参数      argparse；除了 --help 基本不会失败
    2. 读配置 + 校验  load_config / load_thresholds；失败一律友好报错并返回 2
    3. 建检测器      先建检测器：相机是独占资源，检测器有问题时不该先把相机打开
    4. 建输入源      build_source；相机 / 回放 / 合成三种，失败时释放已建资源
    5. 逐帧循环      read -> detect -> 绘制 -> 输出；异常 / Ctrl+C 都跳到 finally
    6. 收尾          source.close() / destroyAllWindows() / writer.close()
    7. 汇总 + 退出码 0 成功、1 没处理任何帧或落盘失败、2 配置或环境错误

退出码约定（两个入口一致，脚本/CI 靠它判断）
    0  正常跑完
    1  跑了但结果不可用：一帧都没处理、自检不一致、落盘失败
    2  根本没跑起来：配置错、阈值表错、检测器/输入源/窗口建不起来
    其他  Python 未捕获异常（正常路径不该出现）

为什么 print 都写在这里而不是 scripts/
    scripts/ 是给命令行用的薄壳；将来 ROS 2 节点 import 的是本模块的函数，
    不应该被迫复用"给终端看"的输出格式，也不该让两处各写一份流程。
"""

import argparse
import os
import sys
import time
from collections import deque

import cv2

from .config.loader import (load_config, load_intrinsics, load_thresholds,
                            resolve_calibration_path, resolve_thresholds_path)
from .config.schema import ConfigError
from .core.detectors.base import build_detector
from .core.pipeline import ColorConfig, process_frames, summarize
from .core.pose import estimate_pose
from .core.types import is_marker
from .io.base import build_source
from .utils.image_io import write_image
from .utils.logging import get_logger, setup_logging
from .viz.draw import draw_detections, draw_hud, draw_pose_axes
from .viz.writer import OutputWriter

log = get_logger("cli")


def _format_centers(detections, limit: int = 8) -> str:
    """把每个检出物的 "id@(x,y)" 拼成一行（超出 limit 个就截断）。

    飞书任务单任务二第 4 条要求"在终端输出识别到的 ArUco 码信息，**包括 ID 和中心点坐标**"。
    判色族的框没有码 ID，写成 `-@(x,y)`；一帧十几个框时截断，免得刷屏。
    """
    parts = []
    for det in list(detections)[:limit]:
        cx, cy = det.center
        tag = str(int(det.marker_id)) if is_marker(det) else "-"
        parts.append(f"{tag}@({int(round(cx))},{int(round(cy))})")
    if len(detections) > limit:
        parts.append(f"…+{len(detections) - limit}")
    return "[" + " ".join(parts) + "]"


def _add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """给解析器挂上两个入口共用的三个参数。

    用大白话说
        命令行里敲的 `--config xxx`、`--set a.b=c`、`--limit 180`，
        在这里被"翻译"成程序里能直接用的三个值：配置文件路径、覆盖项列表、最多跑几帧。
        这个函数只做登记，不读文件、不跑帧。

    抽出来是为了让 `run()` 与 `run_aruco_live()` 的 `--config/--set/--limit`
    语义永远一致——两处各写一份，迟早有一处忘了同步。

    约定
        · 三个参数的 dest 分别是 `config` / `overrides` / `limit`，调用方直接读属性
        · `--set` 用 `action="append"` 收集成列表，**顺序即覆盖顺序**（后写的生效）
        · `--limit` 用 `type=int`：非数字在 argparse 阶段就报错，不会带进业务层

    例子
        python scripts/run_aruco_live.py \\
            --config configs/default.yaml \\
            --set detector.name=stub_aruco \\
            --set detector.stub_aruco.mode=none \\
            --limit 180

    :param ap: 待填充的 `ArgumentParser`（调用方已经设好 prog / description）
    :return: 同一个 `ap`，方便链式书写
    """
    # --config：配置文件路径。None 表示走 config.loader 的默认查找顺序
    #   （当前目录下的 configs/default.yaml，其次环境变量 UP70_CONFIG）。
    #   传相对路径时，阈值表会相对**配置文件所在目录**解析（见 resolve_thresholds_path）。
    ap.add_argument("--config", default=None, help="配置文件路径（默认 configs/default.yaml）")
    # --set：点分覆盖，只改已经存在的字段（不会凭空加键），可重复出现。
    #   · 值按字段现有类型自动转换：int 字段收 "180"、bool 字段收 "true"/"false"
    #   · 键写错时立刻报 ConfigError，并列出该层可用的键——不会静默忽略
    #   例：--set detector.name=stub_aruco --set detector.stub_aruco.mode=none
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="key=value", help="点分覆盖配置，可多次，如 --set color.hit_ratio_min=0.6")
    # --limit：最多处理多少帧，0 = 不限。实时入口靠它让相机流自己停下来
    #   （相机是无限流，没有 --limit 时会一直跑到 Ctrl+C）。
    #   计数用"已处理帧数"而不是 frame.index：两者在回放跳帧（frame_stride>1）时不同。
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少帧（0=不限）")
    return ap


def build_parser() -> argparse.ArgumentParser:
    """判色批处理入口 `run()` 的参数。

    用大白话说
        这里列的是"跑一遍判色、交一份结果"时能调的开关：
        用哪份配置、临时把结果存到哪、要不要存、要不要只看看配置就退出。

    在共用参数之上追加批处理专用项：输出目录覆盖、关闭落盘、打印配置。

    参数一览
        --config       配置文件路径（见 `_add_common_args`）
        --set          点分覆盖配置，可多次
        --limit        最多处理多少帧，0 = 不限
        --out          临时覆盖 output.dir（不改配置文件）
        --no-output    完全不落盘，只在终端看汇总
        --dump-config  打印生效配置后退出（不建输入源、不跑帧）

    例子
        python -m up70_vision.cli --set color.hit_ratio_min=0.6 --limit 3 --no-output

    :return: 填好所有参数的 `ArgumentParser`（解析结果由 `run()` 消费）
    """
    ap = argparse.ArgumentParser(
        prog="up70-vision",
        description="UP70 视觉：颜色识别与场上目标感知（当前阶段：合成图/回放跑通判色链路）")
    _add_common_args(ap)
    # --out：临时换输出目录（不改配置文件），方便多次实验分目录保存。
    #   注意它只改内存里的 cfg.output.dir，不会写回 YAML。
    ap.add_argument("--out", default=None, help="覆盖 output.dir")
    # --no-output：完全不落盘，只在终端看汇总（调参时少写一堆文件）。
    #   自检与终端输出照常，受影响的是 OutputWriter。
    ap.add_argument("--no-output", action="store_true", help="不落盘，只在终端看汇总")
    # --dump-config：只打印生效配置然后退出，用来确认 --set 到底改到了什么。
    #   它在"建输入源/建检测器"之前就返回，所以不需要相机、也不需要写权限。
    ap.add_argument("--dump-config", action="store_true", help="打印生效配置后退出")
    return ap


def build_live_parser() -> argparse.ArgumentParser:
    """实时 ArUco 入口 `run_aruco_live()` 的参数。

    用大白话说
        除了前面那三样（配置、覆盖、跑几帧），这里多两个开关：
        · 要不要开窗口看画面（默认开；没显示器的容器里要关掉）
        · 不开窗口时，隔几帧存一张图下来看

    比 `build_parser()` 少了批处理专用的 `--out/--no-output/--dump-config`，多了
    显示开关。`--headless` 是显式开关而不是"imshow 失败就降级"：静默换行为会让
    "看不到窗口"变成"以为在 headless 跑"（容器里没有 X server 时最容易踩）。

    参数一览（前三个与 `build_parser()` 完全一致）
        --config / --set / --limit   见 `_add_common_args`
        --headless                   不建窗口，改成按帧落盘
        --save-every N               headless 的落盘间隔（帧）

    例子
        # 原生机器，开窗口看实时画面（Ctrl+C / q / 关窗口退出）
        python scripts/run_aruco_live.py --set detector.name=stub_aruco

        # 容器里没有 X server：不建窗口，每 10 帧落一张
        python scripts/run_aruco_live.py --set detector.name=stub_aruco \\
            --headless --save-every 10 --limit 60

    为什么没有 --out / --no-output
        实时入口的落盘是"过程采样"（`output/aruco/`），不是批处理结果目录；
        给它加输出目录开关只会让两套输出语义混在一起。要改落盘节奏用 --save-every。

    :return: 填好所有参数的 `ArgumentParser`（解析结果由 `run_aruco_live()` 消费）
    """
    ap = argparse.ArgumentParser(
        prog="up70-vision-live",
        description="UP70 视觉 · 实时 ArUco：读帧 -> 检测 -> 绘制 -> 窗口/落盘（任务二）")
    _add_common_args(ap)
    # --headless：不建窗口，改成每隔若干帧把叠加图写到磁盘。
    #   容器里没有 X server，这是唯一可复现的验收路径（窗口模式在原生机器上验）。
    #   store_true：不写 = 窗口模式；写了 = headless。
    ap.add_argument("--headless", action="store_true",
                    help="不建窗口，每 --save-every 帧把叠加图写到 output/aruco/（容器里没有 X server 时用）")
    # --save-every：headless 的落盘间隔（帧）。默认 30 帧 ≈ 30fps 下每秒一张；
    #   写盘本身有 IO 开销，所以不逐帧写。下限由 `max(1, ...)` 兜住（0/负数都按 1 处理）。
    ap.add_argument("--save-every", dest="save_every", type=int, default=30,
                    help="headless 时每 N 帧落一张图（默认 30；至少 1）")
    return ap


def _print_config(cfg, config_path, thresholds_path) -> None:
    """把本次真正生效的配置打一遍（两个入口共用）。

    目的不是"好看"，而是让日志/PR 里能直接看到**实际生效**的值：
    配置文件路径、阈值表路径、输入源与它的参数、判色阈值、检测器、输出、日志等级。
    用 `--set` 覆盖后，这里打印的就是覆盖后的结果，不用再回头翻命令行。

    :param cfg: 校验过的 `AppConfig`
    :param config_path: 配置文件绝对路径（便于确认读的是哪一份）
    :param thresholds_path: 解析后的阈值表绝对路径
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


def run(argv=None) -> int:
    """判色批处理入口（原有功能，未改动行为）。

    流程：读配置 -> 建输入源 -> 建检测器 -> 逐帧（检测 -> 判色 -> 画图 -> 落盘）-> 汇总自检。
    与 `run_aruco_live()` 的分工：这里跑完全部帧再汇总，适合合成图/回放与回归验收；
    实时显示用 `run_aruco_live()`。

    :param argv: 命令行参数列表；None 表示取 `sys.argv[1:]`
    :return: 退出码。0 = 自检通过；1 = 自检未通过 / 一帧都没处理；2 = 配置或环境错误
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
    overlays = []

    def on_result(frame, result):
        canvas = draw_detections(frame.image_bgr, result.detections)
        hud = [
            f"{frame.source} #{frame.index}",
            f"det={len(result.detections)} uncertain={result.uncertain}",
            f"matched={result.matched}/{result.truth_total} correct={result.color_correct}",
        ]
        if result.ids:
            hud.insert(1, f"ids={result.ids}")
        draw_hud(canvas, hud)
        overlays.append(canvas)
        writer.write_frame(frame, result, canvas)
        # 终端输出（任务单要求：实时绘制 + 终端输出，且要带 ID 与中心点坐标）
        print(f"[帧{frame.index:>3}] {frame.source} 目标={len(result.detections)} "
              f"ids={result.ids} 中心={_format_centers(result.detections)} "
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

    # ---- 汇总与自检 ----
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
        # 要按真值逐张核 ID / 四角 / 中心，那是 tools/aruco_check.py 的活。
        print(f"  自检(aruco) : 本次共 {summary['detections']} 个码、id 集合 {summary['ids']}"
              f"（本输入源无真值，不做对错判定）")
        print("              按真值逐张核 ID/四角/中心：python scripts/check_aruco.py")
    if cfg.output.enabled:
        print(f"  落盘        : {written}")
    print(f"\n{'✅ 运行成功' if ok else '❌ 自检未通过'}")
    return 0 if ok else 1


# ================================================================ 实时循环（任务二）
# 这一段的常量都是"策略选择"，改它们会直接改变演示观感，所以逐个写明为什么是这个值。
# 每个常量先写一句"人话"，再写技术原因。

# 人话：弹出来的那个窗口叫什么名字。
# 技术：它同时是 cv2.imshow / cv2.waitKey / cv2.getWindowProperty 的窗口句柄，
# 三处必须传同一个字符串，否则 waitKey 收不到这个窗口的按键、关闭检测也查不到它。
# 必须是 ASCII：OpenCV 的 highgui 在某些后端上不接受中文（见 progress.md 坑 33）。
LIVE_WINDOW = "up70-vision: aruco live"

# 人话：终端最多每半秒说一句话，别刷屏。
# 技术：30 fps 下逐帧打印就是每秒 30 行，滚得比看得快；0.5 s 一行 ≈ 每秒两次，
# 既能看出"还在跑"，又不至于把关键信息冲掉。
LIVE_PRINT_INTERVAL_S = 0.5

# 人话：算"每秒几帧"时，看最近 30 次翻页花了多久，不要被某一次卡顿带偏。
# 技术：取最近 N 帧的耗时均值——
#   · 太小（如 1）→ fps 随单帧抖动乱跳，看不出真实水平
#   · 太大（如 300）→ 相机预热/掉帧后要好几秒才反映出来
# 30 帧 ≈ 30fps 下 1 秒，既能平滑又不迟钝。
LIVE_FPS_WINDOW = 30

# 人话：不开窗口时，截图存到这个文件夹。
# 技术：路径相对当前工作目录（仓库根），由 utils.image_io.write_image 自动建目录。
# **不跟随 cfg.output.dir**：实时入口的输出是"过程采样"（每 N 帧一张），
# 与批处理的结果目录（含 results.json / csv，会被同名文件覆盖）分开，避免互相踩。
LIVE_OUT_SUBDIR = os.path.join("output", "aruco")

# 人话：码上画的那三根轴，长度取码边长的多少倍。
# 技术：0.6 倍是实测出来的折中——比 0.5 更容易看出倾斜（透视缩短更明显），
# 又不会长到压住紧邻的另一个码（多码图里码间距约 1.5 倍边长）。
POSE_AXIS_RATIO = 0.6

# 人话：位姿解算失败时，终端最多报几次，剩下的只记次数。
# 技术：相机 30 fps 下若某个码持续算不出来，逐帧打印就是每秒 30 行刷屏，
# 把"帧率掉了"这类真正要看的信息冲掉；3 次足够让人知道"有问题"。
POSE_WARN_LIMIT = 3

# 人话：窗口左上角最多列几个码的位姿数字。
# 技术：HUD 每行 22 px、从 y=22 起排；码多时全列会把画面糊住，且会盖到画面中央的码。
HUD_POSE_LIMIT = 3


def _format_live_line(elapsed: float, frames: int, fps: float, markers, poses=None) -> str:
    """把本帧状态拼成一行固定格式的终端摘要（便于 diff、贴进 PR）。

    用大白话说
        把"跑了多久、第几帧、现在几帧每秒、看到几个码、每个码在哪"串成一句话。
        开了位姿之后，每个码后面再多三个数："多远、斜多少、转多少"。
        码多的时候这句话就长一点，没码的时候只有前半句。

    格式
        `t=<秒>s | frames=<已处理帧数> | fps=<滑动平均> | markers=<个数>
         [| id=N center=(x, y)[ d=<毫米>mm tilt=<度> yaw=<度>]]*`
        每个 marker 追加一段；开了位姿时该段尾部再多三个字段，**字段顺序固定**。

    输出示例（字段之间是 " | "，与本函数实际输出一致）
        无码：`t=3.20s | frames=96 | fps=29.8 | markers=0`
        两码：`t=3.20s | frames=96 | fps=29.8 | markers=2 | id=3 center=(320.0, 240.0) | id=7 center=(501.0, 180.5)`
        开位姿：`... | markers=1 | id=7 center=(501.0, 180.5) d=512mm tilt=12.4 yaw=-30.1`

    为什么是"固定格式 + 一行一条"
        · 一行一条便于 `grep markers=` / `awk` 统计，也便于直接贴进 PR；
        · 固定字段顺序才可以用 diff 对比两次运行的差异；
        · 不用 `\\r` 原地刷新：重定向到文件后会糊成一团，日志也没法回看。

    :param elapsed: 从循环开始到本帧的秒数（time.monotonic 差值，非墙钟时间）
    :param frames: 已处理的帧数（含本帧）
    :param fps: 最近 `LIVE_FPS_WINDOW` 帧的滑动平均帧率
    :param markers: 本帧的 `list[ArucoMarker]`（无码时是 `[]`）
    :param poses: 本帧的 `list[MarkerPose]`；None 或空列表表示没开位姿，
        **此时输出与加位姿之前逐字节一致**（别让新功能改了别人的日志格式）
    :return: 单行字符串，调用方直接 print（不带换行符）
    """
    # 前半段：时间 / 帧数 / 帧率 / 本帧码数。这些都是标量，顺序固定。
    parts = [f"t={elapsed:.2f}s", f"frames={frames}", f"fps={fps:.1f}",
             f"markers={len(markers)}"]
    # 位姿按 id 建索引：markers 可能因解算失败而缺位姿，不能按下标硬对
    pose_by_id = {int(pose.marker_id): pose for pose in (poses or [])}
    # 后半段：每个码一段。markers 已由检测器按 marker_id 升序，输出顺序才稳定可 diff。
    for marker in markers:
        cx, cy = marker.center_px
        # center 用契约里的四角算术平均（ArucoMarker.center_px），不是 bbox 中心：
        # 码倾斜时两者不是同一个位置，用错会让"中心点"看起来偏。
        segment = f"id={int(marker.marker_id)} center=({cx:.1f}, {cy:.1f})"
        pose = pose_by_id.get(int(marker.marker_id))
        if pose is not None:
            # 距离取整到毫米、角度一位小数：终端一行要能扫读，多给的位数没人看
            segment += (f" d={pose.distance_mm:.0f}mm"
                        f" tilt={pose.tilt_deg:.1f} yaw={pose.yaw_deg:.1f}")
        parts.append(segment)
    # 用 " | " 连接：人在终端里一眼能分清字段的分组边界
    return " | ".join(parts)


def run_aruco_live(argv=None) -> int:
    """实时循环：读帧 -> 检测 -> 绘制 -> 显示/落盘 -> 等键。

    用大白话说
        打开相机（或回放/合成图），一帧一帧地拿：每拿一帧就找一次码，
        在画面上把码框出来，显示到窗口（或存成图片），顺手在终端报一句
        "现在有几个码、分别在哪"。码移动、转动、多出几个，下一帧就按新结果重画。

    举个具体例子（30 fps 相机，画面里先是 id=3、id=7 两个码）
        第 1 帧：检测到 2 个码 -> 在画面上画四边形、中心点、id 标签 ->
                显示到窗口 -> 终端打一行：
                    t=0.03s frames=1 fps=30.2 markers=2 | id=3 center=(320.0, 240.0) | id=7 center=(501.0, 180.5)
        第 2~14 帧：同样处理，但终端不重复打（还没到 0.5 秒）
        第 90 帧：id=7 被移出画面 -> 结果变成 1 个码 -> 终端立刻补一行 markers=1
        用户按 q：循环结束 -> 关窗口、还相机 -> 打印本次跑了多少帧

    与 `run()` 的区别
        `run()` 是批处理：全部读完才汇总、只落盘、不开窗口。本函数逐帧更新画面，
        每帧的检测结果**都是新算的，不跨帧保留**——把上一帧的框接着画会让"码移走了
        框还留着"看起来像通过（任务单第 5 条最常见的假通过）。

    单帧处理顺序（循环体就是这张图，顺序不能换）
        ┌─ read() 取一帧 ───────────────────────────────────────────┐
        │  1. 量帧间隔 dt，更新 fps 滑动平均                        │
        │  2. detect(image_bgr) -> list[ArucoMarker]（每帧重算）    │
        │  3. draw_detections 画四边形/中心/id 标签                 │
        │  4. （pose.enabled 时）逐码 estimate_pose -> 画坐标轴      │
        │  5. draw_hud 画左上角 HUD（来源#帧号 / markers / fps）    │
        │  6. 输出：headless 落盘 **或** 窗口 imshow + waitKey(1)   │
        │  7. frames += 1；本帧有码则 with_markers += 1            │
        │  8. 终端节流：到 0.5s 或 id 集合变了就打一行              │
        │  9. --limit 到了就 break                                  │
        └───────────────────────────────────────────────────────────┘

    启动顺序（每一步失败都返回 2，且不会留下占用的资源）
        解析参数 -> 读配置+校验 -> 读阈值表 -> 打印生效配置 -> 检查 DISPLAY
        -> 建检测器（先）-> 读内参（仅开启位姿时）-> 建输入源（后）-> 建窗口（仅窗口模式）

        检测器先于输入源：相机是**独占资源**，检测器名字/参数写错时不该先把相机打开；
        窗口建在输入源之后：没有输入源就没有画面，先建窗口只会闪一下。

    退出路径（全部走 finally 清理）
        ┌────────────────────┬──────────────────────────────────────────┐
        │ 触发               │ 说明                                     │
        ├────────────────────┼──────────────────────────────────────────┤
        │ 窗口按 q / ESC     │ 仅窗口模式；waitKey(1) 读到键            │
        │ 点窗口右上角关闭   │ WND_PROP_VISIBLE < 1；不查会不停重建窗口 │
        │ read() 返回 None   │ 回放放完 / 相机 max_frames 到            │
        │ --limit N 帧跑满   │ 相机会无限出帧，靠它自己停               │
        │ Ctrl+C             │ KeyboardInterrupt，打印后正常收尾         │
        └────────────────────┴──────────────────────────────────────────┘
        `source.close()` 与 `destroyAllWindows()` 一律放 finally：异常路径上
        没释放设备（尤其 MVS 相机），下次打开必失败。

    终端节奏（任务单第 4 条）
        每 `LIVE_PRINT_INTERVAL_S` 秒打一行；结果集变化（码增减 / ID 变化）时立即再打一行。
        无码时也打 `markers=0`——否则"还活着但没检测到"与"卡死了"在日志上分不清。
        不用 `\\r` 原地刷新：重定向到文件后会糊成一团。

    :param argv: 命令行参数列表；None 表示取 `sys.argv[1:]`（与 `run()` 一致）
    :return: 退出码。0 = 正常跑完；1 = 一帧都没处理或落盘失败；
        2 = 配置 / 阈值表 / 检测器 / 输入源 / 窗口建不起来
    :raises SystemExit: 只在 `--help` / 参数非法时由 argparse 抛出（与 `run()` 一致）
    """
    # ---- 1. 解析参数（--help / 参数非法时 argparse 自己退出）----
    args = build_live_parser().parse_args(argv)

    # ---- 2. 读配置并校验；--set 在这里生效 ----
    # 人话：把 YAML 里写的参数和命令行 --set 覆盖的项合成一份最终参数。
    #       名字写错、值越界都在这时拦下，不会带着错误配置继续往下跑。
    # 失败统一返回 2：这时还没有任何资源被占用，直接退出即可。
    try:
        cfg, config_path = load_config(args.config, args.overrides)
    except ConfigError as exc:
        print(f"❌ 配置错误：{exc}", file=sys.stderr)
        return 2
    # ---- 3. 读阈值表 ----
    # 人话：找 ArUco 码其实用不到颜色阈值；但"合成图"输入源要按颜色表造色块，
    #       所以照样读一遍——文件缺了/写坏了，现在就说，别等跑起来才炸。
    # 技术：实时 ArUco 本身不判色，但输入源（尤其合成源）要按颜色造图。
    thresholds_path = resolve_thresholds_path(cfg, config_path)
    try:
        thresholds = load_thresholds(thresholds_path)
    except ConfigError as exc:
        print(f"❌ 阈值表错误：{exc}", file=sys.stderr)
        return 2

    # ---- 4. 打印本次生效的配置（含 --set 覆盖后的值）----
    _print_config(cfg, config_path, thresholds_path)
    print(f"运行模式   : {'headless（落盘，不建窗口）' if args.headless else '窗口（q / ESC 退出）'}")

    # ---- 5. 窗口模式的预检：没有 DISPLAY 就提前报错 ----
    # 人话：容器里没有显示器时，开窗口会直接崩，还给一堆 C++ 报错。
    #       与其让用户猜，不如先说"没有显示器，加 --headless"。
    # 技术：OpenCV 的 Qt 后端在缺 X server 时会直接 abort 整个进程
    # （"Could not load the Qt platform plugin xcb"），Python 端 catch 不到。
    if not args.headless and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print("❌ 没有 DISPLAY：容器里没有 X server，窗口模式起不来。"
              "用 --headless（每 N 帧落盘），或到原生机器上开窗口。", file=sys.stderr)
        return 2

    # ---- 6. 建检测器（先）----
    # 人话：先把"找码的工具"准备好，再去开相机。工具都建不起来时，
    #       没必要把相机占上（相机同一时刻只允许一个程序用）。
    # 技术：相机是独占资源，检测器名字/参数写错时不该先把相机打开；
    # 这里也顺带兜住"任务一未合入时选了 detector.name=aruco"的情况，
    # 报错是"检测器不可用 + 可用列表"，而不是 traceback。
    try:
        detector = build_detector(cfg.detector.name, cfg.detector.params)
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f"❌ 检测器不可用：{exc}", file=sys.stderr)
        return 2

    # ---- 6.5 位姿：只在 pose.enabled 时建（默认关闭，老用户零感知）----
    # 人话：要用位姿就得先拿到"这台相机的内参"，没有就算不出毫米。
    #       内参文件不进版本库，新 clone 的仓库一定没有，所以这里分两种处理：
    #         required=true  -> 直接报错退出（现场/CI 用，避免"以为在算位姿其实没算"）
    #         required=false -> 打警告、本次只跑检测（默认，别让缺文件挡住演示）
    intrinsics = None
    if cfg.pose.enabled:
        calib_path = resolve_calibration_path(cfg, config_path)
        try:
            intrinsics = load_intrinsics(calib_path)
        except ConfigError as exc:
            if cfg.calibration.required:
                print(f"❌ 位姿已启用但内参不可用：{exc}", file=sys.stderr)
                return 2
            print(f"⚠️  位姿已启用但内参不可用，本次只跑检测：{exc}", file=sys.stderr)
        else:
            print(f"内参       : {calib_path}  {intrinsics.summary()}")
            print(f"位姿       : 码边长 {cfg.pose.marker_size_mm} mm  "
                  f"method={cfg.pose.method} refine={cfg.pose.refine}")

    # ---- 7. 建输入源（后）----
    # 人话：按配置决定画面从哪来——真相机、图片/视频回放，还是程序自己画的合成图。
    # 技术：配置对象是 dataclass，这里拆成普通参数传进工厂
    # （io 层不 import config，避免循环依赖）。
    try:
        source = build_source(cfg.input.source,
                              synthetic_params=dict(vars(cfg.input.synthetic)),
                              replay_params=dict(vars(cfg.input.replay)),
                              camera_params=dict(vars(cfg.input.camera)),
                              colors=list(thresholds))
    # OSError 覆盖 FileNotFoundError / IOError（回放目录不存在、视频打不开等）
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"❌ 无法建立输入源：{exc}", file=sys.stderr)
        return 2

    # ---- 8. 输出通道：headless 落盘 / 窗口 ----
    # 人话：要么开个窗口实时看，要么（容器里）每 N 帧存一张图，事后翻图看。
    # save_every 兜底 >=1：0 或负数会让取模结果恒为 0（等于逐帧写盘）。
    save_every = max(1, int(args.save_every))
    if args.headless:
        print(f"落盘       : {LIVE_OUT_SUBDIR}/ 每 {save_every} 帧一张")
    else:
        # WINDOW_NORMAL：允许用户拖拽缩放窗口；默认的 WINDOW_AUTOSIZE 在大分辨率相机下
        # 会超出屏幕，看不到全图。
        try:
            cv2.namedWindow(LIVE_WINDOW, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            # 窗口建不起来时输入源已经打开了（可能是相机），必须先释放再返回
            source.close()
            print(f"❌ 无法创建窗口（容器里没有 X server？加 --headless）：{exc}",
                  file=sys.stderr)
            return 2

    frames = 0            # 已处理帧数（含本帧）；--limit 判定与汇总都用它，不用 frame.index
    with_markers = 0      # 至少检出一个码的帧数；用来区分"有码帧 / 无码帧"
    saved = 0             # headless 实际落盘的 PNG 张数（汇总时报告）
    pose_failures = 0     # 位姿解算失败的次数；只进日志与汇总，不影响退出码
    size_checked = False  # 内参尺寸与帧尺寸是否已比对过（只提醒一次，别刷屏）
    dts = deque(maxlen=LIVE_FPS_WINDOW)   # 最近 N 帧的帧间隔（秒），fps 的滑动平均窗口
    t_start = time.monotonic()            # 循环起点，终端摘要的 t= 与最终平均帧率都以它为基准
    t_prev = t_start                      # 上一帧时刻：每帧更新，用于算 dt
    last_print = 0.0                      # 上一次打终端摘要的时刻（节流用；0 表示还没打过）
    last_ids = None                       # 上一帧的 id 列表；与当前不同就立即打一行（结果集变化）
    save_error = None                     # 落盘失败的异常；循环结束后单独报，避免被 finally 吞掉

    try:
        try:
            for frame in source:                       # FrameSource.__iter__ 内部调 read()，None 即结束
                # ---- 帧率：用相邻两帧的 monotonic 间隔做滑动平均 ----
                # 人话：记一下"距离上一帧过了多久"，取最近 30 次的平均，算出每秒几帧。
                now = time.monotonic()                 # 不用 time.time()：它会被系统时钟调整影响
                if now > t_prev:                       # 防御 dt<=0（同一时钟刻度内连读两帧）
                    dts.append(now - t_prev)
                t_prev = now
                fps = (len(dts) / sum(dts)) if dts else 0.0    # 均值 → 平滑单帧抖动

                # ---- 检测：每帧重新算，绝不把上一帧的结果留着 ----
                # 人话：把这一帧交给"找码工具"，它返回"这一帧里有哪几个码、在哪"。
                #       上一帧的结果到这里就作废——否则码移走了、框还留在原地。
                markers = detector.detect(frame.image_bgr)
                canvas = draw_detections(frame.image_bgr, markers)   # 返回副本，不改原帧
                ids = [int(m.marker_id) for m in markers]           # 升序（检测器契约保证）

                # ---- 位姿：逐码解算，单个码失败不影响这一帧的其它码 ----
                # 人话：知道码在画面哪、再知道码有多大和相机内参，就能算出它离得多远、
                #       斜了多少。算失败（比如码太糊、重投影误差超限）就跳过这个码，
                #       继续处理下一个——实时场景不能因为一个坏样本整帧退出。
                poses = []
                if intrinsics is not None:
                    # 分辨率不匹配只提醒一次：套错内参时距离会整体偏，但程序不报错
                    if not size_checked:
                        size_checked = True
                        if tuple(frame.image_bgr.shape[1::-1]) != tuple(intrinsics.image_size):
                            print(f"⚠️  内参是按 {intrinsics.image_size[0]}x{intrinsics.image_size[1]} "
                                  f"标定的，当前帧是 {frame.image_bgr.shape[1]}x{frame.image_bgr.shape[0]}；"
                                  "距离会整体偏，换分辨率要重标定", file=sys.stderr)
                    for marker in markers:
                        try:
                            poses.append(estimate_pose(
                                marker, cfg.pose.marker_size_mm, intrinsics.camera_matrix,
                                # 角点没在这里去畸变（图像也没去），所以畸变系数原样传进去：
                                # 这与"先 undistort_points 再传 None"是两种口径，别混用
                                intrinsics.dist_coeffs,
                                method=cfg.pose.method, refine=cfg.pose.refine,
                                max_reproj_rms_px=cfg.pose.max_reproj_rms_px,
                                # 两个歧义门槛必须从配置传进来，否则 --set 就是个摆设
                                ambiguity_ratio=cfg.pose.ambiguity_ratio,
                                ambiguity_deg=cfg.pose.ambiguity_deg))
                        except ValueError as exc:
                            pose_failures += 1
                            if pose_failures <= POSE_WARN_LIMIT:   # 只报前几次，避免刷屏
                                print(f"⚠️  码 {marker.marker_id} 跳过位姿：{exc}", file=sys.stderr)
                    if poses:
                        # 画轴要跟解位姿用同一套 (K, D)：位姿是在带畸变的像素空间里解的
                        # （上面把 D 传进了 PnP），这里再用 dist=None 投就会在画面边缘错开
                        canvas = draw_pose_axes(canvas, poses, intrinsics.camera_matrix,
                                                axis_mm=cfg.pose.marker_size_mm * POSE_AXIS_RATIO,
                                                dist_coeffs=intrinsics.dist_coeffs)

                # HUD：来源#帧号 / 检出数与 ID / 帧率，再逐码追加"距离/倾角/面内角"。
                # 只写 ASCII（Hershey 字体无中文）。位姿行必须画在画面上：
                # 窗口里看三根轴只能判方向、读不出数字，而终端行 0.5 s 才一条，
                # 全屏看画面时终端往往被挡在后面。
                hud_lines = [f"{frame.source} #{frame.index}",
                             f"markers={len(markers)} ids={ids}",
                             f"fps={fps:.1f}"]
                for pose in poses[:HUD_POSE_LIMIT]:
                    hud_lines.append(f"id {int(pose.marker_id)} d {pose.distance_mm:.0f}mm "
                                     f"tilt {pose.tilt_deg:.1f} yaw {pose.yaw_deg:.1f}")
                draw_hud(canvas, hud_lines)

                # ---- 输出：headless 落盘 / 窗口显示，二选一，不自动降级 ----
                # 人话：要么把画好的图显示出来给人看，要么存成文件以后看。
                if args.headless:
                    if frame.index % save_every == 0:   # 每 N 帧一张，避免写盘拖慢循环
                        # 文件名用 6 位帧号：ls 排序即播放顺序，事后拼视频/抽查都方便
                        write_image(os.path.join(LIVE_OUT_SUBDIR, f"frame_{frame.index:06d}.png"),
                                    canvas)
                        saved += 1
                else:
                    cv2.imshow(LIVE_WINDOW, canvas)
                    key = cv2.waitKey(1) & 0xFF     # 必须调用：刷新窗口 + 帧率控制，不再额外 sleep
                    if key in (ord("q"), 27):       # 27 = ESC
                        print("退出：收到 q / ESC")
                        break
                    # 用户点窗口右上角关闭后，窗口已销毁；不查这个属性 imshow 会不停重建它
                    if cv2.getWindowProperty(LIVE_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        print("退出：窗口已关闭")
                        break

                # ---- 计数：frames 是"处理了几帧"（不是 frame.index）----
                # 人话：这一帧算处理完了，记一笔；顺便记下"这帧有码"，最后好统计。
                # 技术：回放可以跳帧（frame_stride>1），相机索引也可能不从 0 开始，
                # 所以 --limit 与汇总一律用 frames，保证"跑满 N 帧"语义准确。
                frames += 1
                if markers:                             # 记录"有码帧"，汇总时与无码帧对比
                    with_markers += 1

                # ---- 终端节流：到点或结果集变了才打，避免 30fps 刷屏 ----
                # 人话：平时每半秒报一句"还活着、看到几个码"；码的数量或编号一变，
                #       不等半秒，马上补一句——那是演示时最该被看见的瞬间。
                # 两个触发条件：
                #   1) ids 变化（码出现/消失/换 ID）→ 立刻打，这是最该被看见的事件；
                #   2) 距上次 >= 0.5s → 打一行"心跳"，无码时也有输出。
                # flush=True：重定向到文件时行缓冲不会攒着，tail -f 能实时看到。
                tick = time.monotonic()
                if ids != last_ids or (tick - last_print) >= LIVE_PRINT_INTERVAL_S:
                    print(_format_live_line(tick - t_start, frames, fps, markers, poses),
                          flush=True)
                    last_print = tick
                    last_ids = ids

                if args.limit and frames >= args.limit:  # 相机是无限流，靠它自己停
                    print(f"退出：--limit {args.limit} 帧已跑满")
                    break
        except KeyboardInterrupt:
            # Ctrl+C 不是错误：打印一行后走 finally，退出码按"已处理帧数"决定
            print("\n退出：收到 Ctrl+C")
        except OSError as exc:
            # 落盘失败（无权限 / 磁盘满 / 目录不可写）。先记下，等 finally 清理完
            # 再统一报错并返回 1——直接 raise 会跳过下面的汇总输出。
            save_error = exc
    finally:
        # 无论正常结束、break、Ctrl+C 还是异常，这两步都必须执行：
        #   · 相机不 close → 句柄被独占，下次打开失败（MVS 尤其明显）
        #   · 窗口不销毁 → 进程残留窗口，或下次 namedWindow 拿到脏状态
        source.close()
        if not args.headless:
            cv2.destroyAllWindows()

    # ---- 收尾汇总：给终端/PR 一份可核对的数字 ----
    # 平均帧率用"总帧数 / 总耗时"，与循环里的滑动平均 fps 不是一回事：
    # 前者反映整段运行，后者只反映最近 30 帧（用于 HUD 实时显示）。
    elapsed = time.monotonic() - t_start
    avg_fps = frames / elapsed if elapsed > 0 else 0.0
    print("=" * 62)
    print(f"  帧数        : {frames}（有码 {with_markers} / 无码 {frames - with_markers}）")
    print(f"  平均帧率    : {avg_fps:.1f} fps")
    if intrinsics is not None:
        # 失败次数单独报：它是"有多少帧拿不到毫米"的唯一线索，不算失败退出
        print(f"  位姿        : 边长 {cfg.pose.marker_size_mm} mm，解算失败 {pose_failures} 次")
    if args.headless:
        print(f"  落盘        : {LIVE_OUT_SUBDIR}（{saved} 张）")
    print("=" * 62)

    # 退出码优先级：落盘失败 > 空输入 > 成功。
    # 落盘失败优先，因为"看起来跑完了但图没存下来"比"没取到帧"更容易被忽略。
    if save_error is not None:
        print(f"❌ 落盘失败：{save_error}", file=sys.stderr)
        return 1
    if not frames:
        print("❌ 一帧都没处理（输入源为空？）", file=sys.stderr)
        return 1
    print("✅ 运行成功")
    return 0


if __name__ == "__main__":
    sys.exit(run())
