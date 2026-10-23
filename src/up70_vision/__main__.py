"""单一命令入口：`python -m up70_vision <子命令> [参数...]`。

用大白话说
    以前想跑一件事，得先猜它在哪一层：`scripts/` 还是 `src/up70_vision/tools/`？
    而且有六个动作存在两种叫法，名字还倒过来（`scripts/check_aruco.py` 背后叫
    `tools/aruco/check.py`）。这个文件把 19 个入口收成一道门，门上有指示牌：
    不带参数敲 `python -m up70_vision`，就是那张十五行的指示牌（14 个功能 + clean）。

放什么 / 不放什么
    放「子命令名 -> 目标模块与函数」的登记表，以及 `clean`（清产物）这一件事。
    **不放任何业务逻辑**：每个子命令都只是把 argv 原样转交给原有实现，
    因此本文件引入的那天，任何一条命令的行为与改造前逐字节一致。

依赖谁 / 被谁用
    目标模块在 `_dispatch()` 里才 import，所以 `python -m up70_vision --help`
    连 cv2 都不会加载。上层是人和 `demo.sh`（旧的 `scripts/*.py` 薄壳已于 2026-10-22 删除）。

对外约定
    · `python -m up70_vision`                 打印指示牌后退出，退出码 0
    · `python -m up70_vision <子命令> -h`      转交给目标自己的 argparse（参数与改造前一致）
    · 退出码沿用目标函数：0 成功 / 1 结果不可用 / 2 没跑起来
    · 新增子命令只需在 `COMMANDS` 里加一行；`.github/workflows/tests.yml` 的
      「scripts/ 未登记文件」检查会拦住"又长出一个没人知道的新入口"

已知的坑
    · 参数转发**不能用 `nargs=REMAINDER`**：子解析器里的 REMAINDER 碰到以 `-`
      开头的串（`--set`、`-h`）不生效，实测 `run --help` 会报
      "unrecognized arguments"。这里改用"子解析器不注册参数 + `parse_known_args()`"
      （见 `build_parser()` 与 `main()`）。
    · `tools/mvs_color_measure.main()` 原来不吃 argv（直接读 `sys.argv`），
      会让 `mvs-color` 这三个字被当成非法参数——已补 `argv=None` 形参
      （见该文件的 `main()`）。
    · `doctor` 与 `demo.sh` 的目标脚本是自包含的（不 import 本包），
      所以它们在登记表里是 `NO_ARGV`：多给参数说明用错了，直接报错而不是静默忽略。
"""

import argparse
import importlib
import os
import shutil
import sys


def ensure_utf8_stdout() -> bool:
    """把 `sys.stdout` 的编码切成 UTF-8；已经是 UTF-8 时什么都不做。

    用大白话说
        Linux 容器的终端是 UTF-8，`print("⚠️")` 一切正常。Windows 控制台默认是 GBK
        （`cp936`），同一个 `print` 会抛 `UnicodeEncodeError`，退出码变成 1，
        于是一个"跑完了、结果也没问题"的命令看起来像崩溃。

    为什么不是把 ✅/⚠️ 换成 ASCII
        仓库的终端输出、PR 描述、文档里都在抄这些符号；只在个别命令里换成 `[OK]`
        会让同一份输出出现两套风格。真正的问题在"终端编码"，不在符号本身。

    为什么吞掉异常
        `reconfigure` 在 stdout 被重定向成不可重配的对象时（某些 IDE 的捕获器）
        会抛 `AttributeError` / `OSError`。这时唯一能做的是不折腾，让调用方按原编码
        继续跑——报错退出反而把"能出结果"变成"直接崩"。

    :return: 是否真的改动过。容器里（UTF-8）返回 False，Windows 控制台返回 True。

    位置说明
        原先是 `utils/console.py`（2026-10-22 扁平化时内联到这里）：它唯一的调用方是
        `main()`，为一个 10 行的函数保留整个 `utils/` 子包不值。副作用是没有走
        `python -m up70_vision` 而是直接 `python -m up70_vision.tools.xxx` 时，
        Windows 终端不会自动切 UTF-8——统一入口是文档里唯一推荐的跑法。
    """
    stream = sys.stdout
    if stream is None:
        return False
    if str(getattr(stream, "encoding", "")).lower().replace("-", "") == "utf8":
        return False
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        return False
    return True


#: 子命令登记表：名字 -> (模块名, 函数名, 一行说明)。
#: 顺序即 `--help` 里的显示顺序，按"最常用 -> 最少用"排；`_dispatch()` 只认这张表。
COMMANDS = {
    "run": ("up70_vision.cli", "run",
            "判色/检测批处理：读完所有帧再汇总"),
    "live": ("up70_vision.cli", "run_aruco_live",
             "实时 ArUco：逐帧检测 + 窗口/落盘"),
    "doctor": ("up70_vision.tools.env.check_env", "main",
               "环境自检：Python / opencv / 图像读写往返"),
    "demo": ("up70_vision.tools.env.demo", "main",
             "OpenCV 冒烟：造图 -> 阈值 -> 找轮廓 -> 落盘"),
    "check-aruco": ("up70_vision.tools.aruco.check", "main",
                    "按真值核对图组：ID / 四角 / 中心"),
    "pose-check": ("up70_vision.tools.pose", "main",
                   "位姿精度核对：--dir 逐图 / --sweep 批量扫描"),
    "calibrate": ("up70_vision.tools.calibration.calibrate", "main",
                  "相机标定：采棋盘图 -> 内参 + 畸变"),
    "gen-chessboard": ("up70_vision.tools.calibration.gen_board", "main",
                       "生成棋盘 PNG + PDF，打印时选「实际大小」"),
    "gen-aruco": ("up70_vision.tools.aruco.gen_scenes", "main",
                  "生成 ArUco 测试图组 + 真值 JSON"),
    "gen-scenes": ("up70_vision.tools.pose.gen_scenes", "main",
                   "生成带真值的位姿测试图"),
    "export-yolo": ("up70_vision.tools.yolo.export", "main",
                    "把 YOLO .pt 导出成 ONNX + 类名表"),
    "hsv-tuner": ("up70_vision.tools.blob.hsv_tuner", "main",
                  "拖滑条调 HSV 阈值，回车写回 color_thresholds.yaml"),
    "mvs-color": ("up70_vision.tools.blob.color_measure", "main",
                  "海康 MVS 相机实时判色窗口"),
    "gen-html": ("up70_vision.tools.doc.gen_html", "main",
                 "教程 md -> 单文件 html（需要 pandoc；构建产物不进库）"),
}

#: 不吃命令行的子命令：目标是自包含脚本（没有 argparse），多给参数说明用错了。
NO_ARGV = frozenset({"doctor", "demo"})

#: `clean` 的一行说明。挂在 `COMMANDS` 之外，因为它调的是本文件里的函数，不是别的模块。
CLEAN_HELP = "清理运行产物：output/ 下的内容（留 .gitkeep）与所有 __pycache__/"


def _clean(root: str = ".") -> int:
    """删掉运行产物与字节码缓存，让目录回到"刚 clone 完"的样子。

    用大白话说
        `output/` 里是跑出来的叠加图、JSON、CSV（226 个文件 25 MB），
        各处还有 `.pyc`。这些都不进版本库，但会塞满本地目录，
        人和 AI 翻目录时先撞上的就是这堆噪声。这个函数把它们删掉。

    :param root: 清理的起点目录，默认当前目录（应当在仓库根执行）
    :return: 0 正常清理（包括"本来就没有"）；2 起点目录不存在

    副作用
        不可恢复地删除文件：`output/` 下的全部内容（`.gitkeep` 除外）与所有
        `__pycache__/` 目录。**`.git` 目录被跳过**——版本库不在清理范围内。
        `data/` 不动：那是本地素材（图片/视频），删了要重新找素材。
    """
    base = os.path.abspath(root)
    if not os.path.isdir(base):
        print(f"❌ 目录不存在：{base}", file=sys.stderr)
        return 2

    files_removed = 0
    dirs_removed = 0

    # ---- 1. output/ 下的内容：目录本身保留（靠 .gitkeep），里面的东西全删 ----
    out_dir = os.path.join(base, "output")
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if name == ".gitkeep":        # 占位文件：git 靠它保住空目录
                continue
            path = os.path.join(out_dir, name)
            if os.path.isdir(path) and not os.path.islink(path):
                files_removed += sum(len(f) for _d, _s, f in os.walk(path))
                shutil.rmtree(path)
                dirs_removed += 1
            else:
                os.remove(path)
                files_removed += 1

    # ---- 2. 所有 __pycache__/：边走边删，删完不再往里走 ----
    for current, subdirs, _files in os.walk(base):
        if ".git" in subdirs:
            subdirs.remove(".git")        # 版本库里的对象不是缓存，别碰
        if "__pycache__" in subdirs:
            target = os.path.join(current, "__pycache__")
            files_removed += len(os.listdir(target))
            shutil.rmtree(target)
            dirs_removed += 1
            subdirs.remove("__pycache__")

    print(f"✅ 已清理：{files_removed} 个文件、{dirs_removed} 个目录（output/ 与 __pycache__/）")
    if not files_removed and not dirs_removed:
        print("   （本来就没有产物，什么都没删）")
    return 0


def _dispatch(name: str, argv) -> int:
    """把 argv 转交给子命令背后的原实现。

    :param name: `COMMANDS` 里的键（已由 argparse 保证合法）
    :param argv: 剩下的参数列表，原样交给目标函数
    :return: 目标的退出码（0 成功 / 1 结果不可用 / 2 没跑起来）；目标返回 None 时按 0

    为什么用 importlib 而不是文件顶部 import
        `--help` 与 `clean` 不需要 cv2、numpy。顶部 import 会让"看个指示牌"
        也付出 0.5 秒的加载代价，并且在没有装依赖的机器上连帮助都看不到。
    """
    module_name, attr, _desc = COMMANDS[name]
    module = importlib.import_module(module_name)
    func = getattr(module, attr)

    # 自包含脚本（doctor）没有 argparse：给了参数就是敲错了，别静默吞掉
    if name in NO_ARGV:
        if argv:
            print(f"❌ {name} 不接受参数（收到 {list(argv)}）；"
                  f"直接敲 `python -m up70_vision {name}`", file=sys.stderr)
            return 2
        return int(func() or 0)

    return int(func(argv) or 0)


def build_parser() -> argparse.ArgumentParser:
    """顶层参数解析器：一张子命令指示牌，每个子命令把剩余参数原样放行。

    :return: 填好 14 个功能子命令 + `clean` 的 `ArgumentParser`

    关键写法
        · 功能子命令的子解析器**一个参数都不注册**，且 `add_help=False`；
          参数靠 `parse_known_args()` 原样收集（见 `main()`）。
        · 不用 `nargs=argparse.REMAINDER` 收参数：子解析器里的 REMAINDER
          遇到以 `-` 开头的串（`--set`、`-h`）不生效，会变成
          "unrecognized arguments"，实测 `run --help` 直接报错。
    """
    parser = argparse.ArgumentParser(
        prog="python -m up70_vision",
        description="UP70 视觉：一个入口，14 个功能子命令 + clean（不带参数时打印这张表）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例：python -m up70_vision run --set input.source=synthetic --limit 3\n"
               "    python -m up70_vision pose-check --dir output/pose_scenes\n"
               "    python -m up70_vision clean")
    subparsers = parser.add_subparsers(dest="command", metavar="<子命令>")

    for name, (_module, _attr, desc) in COMMANDS.items():
        # 空壳子解析器：只负责在 --help 里露个名字和一行说明，参数交给目标自己
        subparsers.add_parser(name, help=desc, description=desc, add_help=False)

    clean = subparsers.add_parser("clean", help=CLEAN_HELP, description=CLEAN_HELP)
    clean.add_argument("--root", default=".", help="要清理的目录，默认当前目录（应在仓库根执行）")
    return parser


def main(argv=None) -> int:
    """入口：解析子命令并分发。

    :param argv: 命令行参数列表；None 表示取 `sys.argv[1:]`
    :return: 0 = 正常（含"打印指示牌"与"清理完成"）；其余见 `_dispatch`

    副作用
        `clean` 会删文件（不可恢复），其余子命令的副作用由各自实现决定。
    """
    # Windows 的 GBK 终端会把 ✅/⚠️ 打成 UnicodeEncodeError（进程退出码变 1，
    # 看起来像崩溃）。Linux 容器里本来就是 UTF-8，这个调用什么都不做。
    ensure_utf8_stdout()

    parser = build_parser()
    # parse_known_args：顶层只认子命令名，剩下的一切（--set / --limit / -h ...）
    # 原样落进 rest，顺序不变，交给目标自己的 argparse 去解释
    args, rest = parser.parse_known_args(argv)

    if not args.command:
        # 不带参数 = 问"有哪些命令"。打指示牌后返回 0，不当作错误。
        parser.print_help()
        return 0

    if args.command == "clean":
        return _clean(args.root)

    return _dispatch(args.command, list(rest))


if __name__ == "__main__":
    sys.exit(main())
