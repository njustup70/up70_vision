"""实时入口 `run_aruco_live()`：逐帧 检测 -> 绘制 -> 窗口/落盘 -> 终端节流输出。

用大白话说
    打开相机（或回放/合成图），一帧一帧地拿：每拿一帧就找一次码，在画面上把码框出来，
    显示到窗口（或存成图片），顺手在终端报一句"现在有几个码、分别在哪"。
    开着位姿时还会在码上画三根轴、在 HUD 里写出距离和角度。

放什么 / 不放什么
    放实时循环本身，以及它专用的常量（窗口名、打印间隔、帧率窗口、落盘目录等）。
    **不放**批处理逻辑（那是 `cli/run.py`）；打印排版在本文件末尾的「终端输出」一节。
    检测、位姿、画图、收帧分别来自根级算法模块 / viz / io，本文件只做"每帧按顺序调一遍"。

依赖谁 / 被谁用
    上面的那一整串（config / 根级算法模块 / io / viz）。被 `__main__.py` 的 `live` 子命令调用。

对外约定
    `run_aruco_live(argv)` 返回退出码：0 正常跑完 / 1 一帧都没处理或落盘失败 /
    2 配置、检测器、输入源、窗口任一个建不起来。
    终端摘要一行一条、字段顺序固定（拼行函数是本文件末尾的 `format_live_line()`），
    **没开位姿时输出与加位姿之前逐字节一致**。
"""

import os
import sys
import time
from collections import deque

import cv2

from ..config.loader import (load_config, load_intrinsics, load_thresholds,
                             resolve_calibration_path, resolve_thresholds_path)
from ..config.schema import ConfigError
from .run import build_detector
from ..aruco.pose import estimate_pose
from ..sources import build_source
from .output import write_image
from ..logging import setup_logging
from .output import draw_all, draw_hud
from ..aruco.draw import draw_pose_axes
from .parsers import build_live_parser

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
# 技术：路径相对当前工作目录（仓库根），由 image_io.write_image 自动建目录。
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
#: HUD 防抖窗口：最近多少帧一起取中位数。27 fps 下 9 帧约 0.33 s，肉眼读得出来又跟得上手
POSE_MEDIAN_WINDOW = 9


def median_pose_line(history) -> str | None:
    """把最近若干帧的 (d, tilt, yaw) 压成一行中位数，用于 HUD 稳定读数。

    为什么需要：实时 HUD 是逐帧刷新的，角点噪声会让 `d` 在小数位上跳、码接近正视时
    还会在两解之间翻，肉眼根本读不出"到底多少"。取中位数能同时解决两类抖动。

    :param history: [(distance_mm, tilt_deg, yaw_deg), ...]，已剔除 ambiguous 帧
    :return: 形如 `med d 511 tilt 6.4 (n=9)` 的字符串；样本不足 3 条时返回 None
    """
    if len(history) < 3:
        return None
    ds = sorted(x[0] for x in history)
    ts = sorted(x[1] for x in history)
    ys = sorted(x[2] for x in history)
    mid = len(history) // 2
    return (f"med d {ds[mid]:.0f}mm tilt {ts[mid]:.1f} yaw {ys[mid]:.1f} "
            f"(n={len(history)})")


def pose_status_line(enabled: bool, n_poses: int, failures: int) -> str | None:
    """HUD 里"位姿到底算没算"的状态行（没有解时也要有一行，别让人对着画面瞎猜）。

    :param enabled: 配置里是否开了位姿
    :param n_poses: 这一帧成功解出的位姿数
    :param failures: 累计解算失败次数
    :return: 要画的字符串；不需要画时返回 None
    """
    if not enabled:
        return None
    if n_poses:
        return None                      # 有解时每个码单独一行，由调用方追加
    return f"pose: no solution (fail {failures})"


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

    与 `cli.run.run()` 的区别
        `run()` 是批处理：全部读完才汇总、只落盘、不开窗口。本函数逐帧更新画面，
        每帧的检测结果**都是新算的，不跨帧保留**——把上一帧的框接着画会让"码移走了
        框还留着"看起来像通过（任务单第 5 条最常见的假通过）。

    单帧处理顺序（循环体就是这张图，顺序不能换）
        ┌─ read() 取一帧 ───────────────────────────────────────────┐
        │  1. 量帧间隔 dt，更新 fps 滑动平均                        │
        │  2. detect(image_bgr) -> list[ArucoMarker]（每帧重算）    │
        │  3. draw_all 画四边形/中心/id 标签（按检出物类型分流）    │
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
    print_config(cfg, config_path, thresholds_path)
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

    # 本入口专用于 ArUco（要画四边形/中心/id、可选算位姿）。
    # 判色族与 YOLO 返回的是 Detection，没有 marker_id，硬跑下去只会在第一帧
    # 抛一句 `'Detection' object has no attribute 'marker_id'`，所以在这里直接说清楚。
    if str(cfg.detector.name).lower() not in ("aruco", "stub_aruco", "stub"):
        print(f"❌ 本入口只跑 ArUco 检测器（当前 detector.name={cfg.detector.name!r}）；"
              f"要用判色/YOLO 检测器请走批处理入口：python -m vision run "
              f"--set detector.name={cfg.detector.name}", file=sys.stderr)
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
    pose_history = deque(maxlen=POSE_MEDIAN_WINDOW)   # HUD 防抖窗口（非歧义帧）
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
                canvas = draw_all(frame.image_bgr, markers)   # 返回副本，不改原帧
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
                status = pose_status_line(cfg.pose.enabled, len(poses), pose_failures)
                if status:
                    hud_lines.append(status)
                # 防抖：只把"非歧义"的帧放进窗口，再取中位数——HUD 上单独一行
                if poses and not poses[0].ambiguous:
                    pose_history.append((poses[0].distance_mm, poses[0].tilt_deg,
                                         poses[0].yaw_deg))
                if pose_history:
                    med = median_pose_line(pose_history)
                    if med:
                        hud_lines.append(med)
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
                    print(format_live_line(tick - t_start, frames, fps, markers, poses),
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

    # ---- 收尾汇总：给终端/PR 一份可核对的数字（排版在 report.py）----
    elapsed = time.monotonic() - t_start
    avg_fps = frames / elapsed if elapsed > 0 else 0.0
    print_live_summary(
        frames=frames, with_markers=with_markers, avg_fps=avg_fps,
        # 失败次数单独报：它是"有多少帧拿不到毫米"的唯一线索，不算失败退出
        pose_note=(None if intrinsics is None
                   else f"边长 {cfg.pose.marker_size_mm} mm，解算失败 {pose_failures} 次"),
        saved_dir=(LIVE_OUT_SUBDIR if args.headless else None),
        saved_count=saved)

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


# ================================================================ 终端输出
# 以下三个函数原先在 cli/report.py（2026-10-22 合并进来：只有本文件用，独立成文件让"找一个功能"多跳一次）。
# 它们只负责排版，不改任何检测结果。

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


def format_live_line(elapsed: float, frames: int, fps: float, markers, poses=None) -> str:
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
    :param fps: 最近 `cli/live.py` 里 `LIVE_FPS_WINDOW` 帧的滑动平均帧率
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


def print_live_summary(frames: int, with_markers: int, avg_fps: float,
                       pose_note: str | None = None,
                       saved_dir: str | None = None,
                       saved_count: int = 0) -> None:
    """打实时入口的收尾汇总（给终端/PR 一份可核对的数字）。

    用大白话说
        退出循环后告诉人："总共处理了几帧、其中几帧有码、平均几帧每秒"。
        开了位姿就再说一句"解算失败几次"，headless 模式再说一句"图存在哪、存了几张"。

    :param frames: 已处理帧数（`--limit` 与汇总一律用它，不用 frame.index）
    :param with_markers: 至少检出一个码的帧数（用来区分"有码帧 / 无码帧"）
    :param avg_fps: 整段运行的平均帧率 = 帧数 / 总耗时；与 HUD 里的滑动平均不是一回事
    :param pose_note: 形如 `边长 82.5 mm，解算失败 0 次`；None 表示没开位姿，这一行不打印
    :param saved_dir: headless 的落盘目录；None 表示窗口模式，这一行不打印
    :param saved_count: headless 实际落盘的 PNG 张数

    平均帧率用"总帧数 / 总耗时"，与循环里的滑动平均 fps 不是一回事：
    前者反映整段运行，后者只反映最近 30 帧（用于 HUD 实时显示）。
    """
    print("=" * 62)
    print(f"  帧数        : {frames}（有码 {with_markers} / 无码 {frames - with_markers}）")
    print(f"  平均帧率    : {avg_fps:.1f} fps")
    if pose_note is not None:
        # 失败次数单独报：它是"有多少帧拿不到毫米"的唯一线索，不算失败退出
        print(f"  位姿        : {pose_note}")
    if saved_dir is not None:
        print(f"  落盘        : {saved_dir}（{saved_count} 张）")
    print("=" * 62)

