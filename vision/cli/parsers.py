"""参数解析：`run()` 与 `run_aruco_live()` 各一套 argparse，三个参数由共用函数登记。

用大白话说
    这里只做一件事：把命令行里敲的字翻译成程序能直接用的值——用哪份配置、
    临时改哪几项（`--set a.b=c`）、最多跑几帧。不读文件、不建资源、不跑帧。

放什么 / 不放什么
    放 `build_parser()` / `build_live_parser()` 与它们共用的 `_add_common_args()`。
    **不放**任何执行逻辑，也不放默认值的来源（默认值在 `configs/default.yaml`）。

依赖谁 / 被谁用
    只依赖 argparse（纯参数表）；被 `cli/run.py` 与 `cli/live.py` 各调一次。
    这两个解析器只描述"命令行长什么样"，与具体执行实现解耦。

对外约定
    · 三个共用参数的 dest 固定为 `config` / `overrides` / `limit`
    · 参数非法时由 argparse 自己 `SystemExit(2)`，调用方不必再接一层
    · `--help` 时打印本文件的参数表后 `SystemExit(0)`
"""

import argparse


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
        python -m vision live \\
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
        python -m vision run --set color.hit_ratio_min=0.6 --limit 3 --no-output

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
        python -m vision live --set detector.name=stub_aruco

        # 容器里没有 X server：不建窗口，每 10 帧落一张
        python -m vision live --set detector.name=stub_aruco \\
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
