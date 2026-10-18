"""相机标定工具：拍一批棋盘图（或读一个图目录）-> 解内参 + 畸变 -> 写 YAML。

用大白话说
    标定分两步，这个工具一步做完：
      第一步「采图」：把棋盘板在镜头前摆十几个不同姿势各拍一张（太正、太斜、靠近、退远、转到
        画面四角都拍一点）。程序自动判断这张图能不能用——找不到板子就不收。
      第二步「解算」：把这些图交给 `core/calibration.py`，得到焦距、光心、畸变系数，
        再写进 `configs/camera_calibration.yaml` 给阶段二的位姿估计用。

放什么 / 不放什么
    放"采图 + 解算 + 落盘 + 打印报告"。**纯算法在 `core/calibration.py`**：
    找角点、解 K、去畸变都在那边，这里只做 IO 与命令行。
    不碰实时检测/绘制——那是 `cli.py` 的事。

依赖谁 / 被谁用
    依赖 `core.calibration`（算法）、`io.base`（抓帧）、`config`（取相机参数）、`utils`（读写图与日志）。
    薄入口：`scripts/calibrate_camera.py`。

两种用法
    ① 已有图：`python scripts/calibrate_camera.py --images data/calib`
       （图可以从手机拍、也可以用 `--capture` 先采）
    ② 现场采图：`python scripts/calibrate_camera.py --capture 15`
       （走 `--set input.source=camera` 的相机；每收一张打印一次进度）

退出码
    0 成功（且 RMS 没超过 `--max-rms`）
    1 没跑起来或结果不可用：有效图不够、一张都没找到板子、RMS 超标、写盘失败
    2 参数非法（argparse 给的）

已知的坑
    · `--square-mm` 必须填**量出来的**格子边长，不是打印标称值（坑 40）。
    · 板子只占画面一小块时 `findChessboardCorners` 会失败。让它至少占画面 1/3（坑 41）。
    · 采图时相机是独占资源，程序退出会 close；别同时开另一个取帧程序。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from ..config.loader import load_config, load_thresholds, resolve_thresholds_path
from ..core.calibration import (MIN_VIEWS, RMS_USABLE, BoardSpec, CameraIntrinsics,
                                calibrate, find_board_corners, undistort_image)
from ..io.base import build_source
from ..utils.image_io import ensure_dir, read_image_bgr, write_image
from ..utils.logging import get_logger, setup_logging

log = get_logger("tools.calibrate_camera")

#: 标定结果默认落这里。阶段二的位姿估计会读它（改路径要同步那边）。
DEFAULT_OUT = "configs/camera_calibration.yaml"

#: 现场采图默认存这里（已被 .gitignore 排除，不占版本库）
DEFAULT_CAPTURE_DIR = "data/calib"

#: 采图时认这些后缀
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")

#: 从图目录读图时的上限，防止误指到超大目录
MAX_IMAGES = 200


def parse_board(text: str, square_mm: float) -> BoardSpec:
    """把 `"9x6"` 这样的字符串解析成板子参数。

    :param text: 形如 `9x6`（**内角点**个数）；也接受 `9,6`、`9 6`
    :param square_mm: 格子边长（毫米）
    :return: `BoardSpec`
    :raises ValueError: 格式不对，或行列数非法
    """
    cleaned = str(text).lower().replace("×", "x").replace(",", " ").replace("x", " ")
    parts = [p for p in cleaned.split() if p]
    if len(parts) != 2:
        raise ValueError(f"--board 需要「横向x纵向」两个数（如 9x6），收到 {text!r}")
    return BoardSpec(cols=int(parts[0]), rows=int(parts[1]), square_mm=float(square_mm))


def collect_from_dir(directory: str, board: BoardSpec) -> tuple[list, list, list]:
    """从目录里读图并逐张找角点。

    用大白话说
        把目录里所有图片挨个过一遍，找到完整棋盘的收进结果，找不到的记下文件名——
        后面打印报告时要能说清"哪几张没用上、为什么"。

    :param directory: 图目录
    :param board: 板子参数
    :return: `(image_points, used_names, skipped_names)`；
             image_points 是 `find_board_corners` 的返回值列表（只含成功的）
    :raises OSError: 目录不存在
    """
    if not os.path.isdir(directory):
        raise OSError(f"图目录不存在：{directory}")
    paths = sorted(p for p in glob.glob(os.path.join(directory, "*"))
                   if p.lower().endswith(IMAGE_SUFFIXES))[:MAX_IMAGES]
    if not paths:
        raise OSError(f"{directory} 下没有图片（认这些后缀：{', '.join(IMAGE_SUFFIXES)}）")

    image_points, used, skipped = [], [], []
    for path in paths:
        name = os.path.basename(path)
        try:
            image = read_image_bgr(path)
        except Exception as exc:                        # noqa: BLE001
            log.warning("读不了 %s：%s", name, exc)
            skipped.append(f"{name}（读不了）")
            continue
        corners = find_board_corners(image, board)
        if corners is None:
            skipped.append(name)
            continue
        image_points.append(corners)
        used.append(name)
        print(f"  ✅ {name:<28} 找到 {board.inner_corners} 个内角点")
    return image_points, used, skipped


def capture_from_source(cfg, board: BoardSpec, count: int, out_dir: str,
                        colors: list | None = None,
                        max_frames: int = 0) -> tuple[list, list]:
    """从配置里的输入源抓帧，找到棋盘就存下来，凑够 count 张为止。

    用大白话说
        对着一块棋盘慢慢换姿势，程序每看到一次完整的板子就"咔嚓"存一张，并告诉你已经收了几张。
        存图是为了以后能重标（用 `--images` 复用这批图），而不是只留一堆数字。

    :param cfg: `AppConfig`（用 `input.source` 与 `input.camera`）
    :param board: 板子参数
    :param count: 要收多少张
    :param out_dir: 图存哪
    :param colors: 阈值表里的颜色名（只有合成源用得上；相机源忽略它）
    :param max_frames: 最多看多少帧（0 = 不限）；相机是无限流，靠它兜底
    :return: `(image_points, saved_paths)`
    :raises OSError: 输入源构造失败（相机被占用 / SDK 缺失）
    """
    ensure_dir(out_dir)
    source = build_source(cfg.input.source,
                          synthetic_params=dict(vars(cfg.input.synthetic)),
                          replay_params=dict(vars(cfg.input.replay)),
                          camera_params=dict(vars(cfg.input.camera)),
                          colors=colors)
    image_points, saved = [], []
    seen = 0
    try:
        while len(saved) < count:
            frame = source.read()
            if frame is None:
                print(f"  输入源结束（共看了 {seen} 帧，收到 {len(saved)} 张）")
                break
            seen += 1
            if max_frames and seen > max_frames:
                print(f"  已看满 {max_frames} 帧，停止采图")
                break
            corners = find_board_corners(frame.image_bgr, board)
            if corners is None:
                if seen % 30 == 0:
                    print(f"  … 已看 {seen} 帧，还没找到完整的板子（收到 {len(saved)} 张）")
                continue
            index = len(saved) + 1
            path = os.path.join(out_dir, f"calib_{index:02d}.png")
            write_image(path, frame.image_bgr)
            image_points.append(corners)
            saved.append(path)
            print(f"  ✅ 第 {index}/{count} 张 -> {path}")
    finally:
        # 相机是独占资源：异常路径也必须释放，否则下次打不开（坑 31）
        source.close()
    return image_points, saved


def write_calibration(path: str, intrinsics: CameraIntrinsics) -> str:
    """把标定结果写成 YAML（带注释头，说明谁在读它）。

    :param path: 输出路径
    :param intrinsics: 标定结果
    :return: 实际写入的路径
    :raises OSError: 目录建不出来或写失败
    """
    import datetime
    import yaml

    directory = os.path.dirname(os.path.abspath(path))
    ensure_dir(directory)
    header = (
        "# 相机内参与畸变系数（由 scripts/calibrate_camera.py 生成，别手改）\n"
        "#\n"
        "# 读这份文件的地方：阶段二的位姿估计（solvePnP）与去畸变。\n"
        "#   camera_matrix : 3x3 内参矩阵 [[fx,0,cx],[0,fy,cy],[0,0,1]]\n"
        "#   dist_coeffs   : 5 个畸变系数，顺序 k1,k2,p1,p2,k3（OpenCV 约定）\n"
        "#   image_size    : [width, height]，去畸变时必须与它一致\n"
        "#   rms_px        : 重投影 RMS（像素），越小越好；>1 说明这批图拍得不够好\n"
        f"# 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        # default_flow_style=None：短的列表（如 image_size）写成 [1280, 800]，长列表保持块状
        yaml.safe_dump(intrinsics.to_dict(), handle, allow_unicode=True,
                       sort_keys=False, default_flow_style=None)
    return path


def print_report(intrinsics: CameraIntrinsics, used: list, skipped: list,
                 max_rms: float) -> bool:
    """打印标定报告，并判断结果是否达标。

    :return: 是否达标（RMS <= max_rms）
    """
    print()
    print("=" * 72)
    print(f"  标定结果   {intrinsics.summary()}")
    print("-" * 72)
    print(f"  内参矩阵 K : fx={intrinsics.fx:9.2f}  fy={intrinsics.fy:9.2f}")
    print(f"               cx={intrinsics.cx:9.2f}  cy={intrinsics.cy:9.2f}  "
          f"(画面 {intrinsics.image_size[0]}x{intrinsics.image_size[1]})")
    names = ", ".join(f"{k}={v:+.4f}" for k, v in
                      zip(("k1", "k2", "p1", "p2", "k3"), intrinsics.dist_coeffs.ravel()))
    print(f"  畸变系数   : {names}")
    print(f"  重投影 RMS : {intrinsics.rms:.3f} px  （<0.5 好，<1.0 可用，>1.0 别直接用）")
    print(f"  有效 / 尝试的图：{intrinsics.views} / {intrinsics.views + len(skipped)}")
    if skipped:
        print(f"  没用上的图  : {', '.join(skipped[:8])}"
              f"{' …' if len(skipped) > 8 else ''}")
    if intrinsics.per_view_rms:
        worst = max(range(len(intrinsics.per_view_rms)), key=lambda i: intrinsics.per_view_rms[i])
        print(f"  单张最差   : 第 {worst + 1} 张，{intrinsics.per_view_rms[worst]:.3f} px"
              f"（明显偏大就重拍这张）")
    ok = intrinsics.rms <= float(max_rms)
    if not ok:
        print(f"  ⚠️  RMS 超过 {max_rms:g} px：常见原因是板子没贴平、有反光、"
              f"或只拍了正中几个姿势。补拍「靠近画面四角 + 大角度倾斜」的图再标一次。")
    if intrinsics.views < MIN_VIEWS:
        ok = False
    print("=" * 72)
    return ok


def preview_undistort(intrinsics: CameraIntrinsics, sample_path: str) -> None:
    """开窗看「原图 vs 矫正后」，按任意键关闭（需要 GUI，容器里要 X + GUI 库）。

    :param intrinsics: 标定结果
    :param sample_path: 拿哪张图做对比
    """
    import cv2

    image = read_image_bgr(sample_path)
    fixed = undistort_image(image, intrinsics)
    # 尺寸可能因 K 与畸变略有不同，统一到原图尺寸便于左右并排
    fixed = cv2.resize(fixed, (image.shape[1], image.shape[0]))
    canvas = np.hstack([image, fixed])
    win = "calibrate_camera: left=raw right=undistorted (any key to close)"
    cv2.imshow(win, canvas)
    print(f"  预览已打开：左边原图、右边矫正后；窗口上按任意键关闭")
    cv2.waitKey(0)
    cv2.destroyWindow(win)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="calibrate_camera",
        description="相机标定：算内参矩阵与畸变系数（棋盘板）",
        epilog="例：python scripts/calibrate_camera.py --capture 15  "
               "或  python scripts/calibrate_camera.py --images data/calib")
    ap.add_argument("--images", default=None,
                    help="从图目录标定（与 --capture 二选一）")
    ap.add_argument("--capture", type=int, default=0, metavar="N",
                    help="现场从输入源采 N 张再标定（0 = 不采）")
    ap.add_argument("--capture-dir", default=DEFAULT_CAPTURE_DIR,
                    help=f"采到的图存哪（默认 {DEFAULT_CAPTURE_DIR}）")
    ap.add_argument("--board", default="9x6",
                    help="内角点个数，横向x纵向（默认 9x6；不是格子数）")
    ap.add_argument("--square-mm", type=float, default=15.0,
                    help="格子边长毫米（默认 15；**填尺子量到的值**）")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"标定结果写哪（默认 {DEFAULT_OUT}）")
    ap.add_argument("--from", dest="from_file", default=None,
                    help="不重新标，只读回一份已有标定并打印")
    ap.add_argument("--max-rms", type=float, default=RMS_USABLE,
                    help=f"RMS 超过它就判为不可用并返回 1（默认 {RMS_USABLE}）")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="采图时最多看多少帧（0 = 不限；相机是无限流，靠它兜底）")
    ap.add_argument("--preview", action="store_true",
                    help="标定完开窗对比原图/矫正图（需要 GUI）")
    ap.add_argument("--config", default=None, help="配置文件路径（默认 configs/default.yaml）")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="key=value",
                    help="临时覆盖配置，可多次，如 --set input.source=camera")
    ap.add_argument("--log-level", default="INFO", help="日志等级，默认 INFO")
    return ap


def main(argv=None) -> int:
    """命令行入口。

    :return: 0 成功；1 结果不可用 / 没跑起来；2 参数非法（argparse）
    """
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    # 只读回已有标定：打印完就退出，不碰相机
    if args.from_file:
        import yaml
        with open(args.from_file, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        intrinsics = CameraIntrinsics.from_dict(data)
        print_report(intrinsics, [], [], args.max_rms)
        return 0 if intrinsics.rms <= args.max_rms else 1

    try:
        board = parse_board(args.board, args.square_mm)
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    # ---- 第一步：拿图（目录读 / 现场采） ----
    used, skipped, image_points = [], [], []
    if args.images:
        print(f"从目录读图：{args.images}   板子 {board.cols}x{board.rows}  格子 {board.square_mm:g} mm")
        try:
            image_points, used, skipped = collect_from_dir(args.images, board)
        except OSError as exc:
            print(f"❌ {exc}", file=sys.stderr)
            return 1
    elif args.capture > 0:
        # load_config 内部已经应用了 --set 覆盖，别在外面再调一次 apply_overrides
        cfg, loaded_path = load_config(args.config, args.overrides)
        # 颜色名只有合成源用得上（相机源不看它），但 build_source 的参数要一起给
        thresholds = load_thresholds(resolve_thresholds_path(cfg, loaded_path))
        print(f"现场采图：源={cfg.input.source}  目标 {args.capture} 张  -> {args.capture_dir}")
        print("         （对着棋盘换姿势：正对、倾斜、靠近、退远、转到画面四角）")
        try:
            image_points, saved = capture_from_source(
                cfg, board, args.capture, args.capture_dir,
                colors=list(thresholds), max_frames=args.max_frames)
        except (OSError, RuntimeError) as exc:
            print(f"❌ 采图失败：{exc}", file=sys.stderr)
            return 1
        used = [os.path.basename(p) for p in saved]
    else:
        print("❌ 要指定 --images DIR 或 --capture N（二选一）", file=sys.stderr)
        return 2

    sample = None
    if args.images:
        paths = sorted(p for p in glob.glob(os.path.join(args.images, "*"))
                       if p.lower().endswith(IMAGE_SUFFIXES))
        sample = paths[0] if paths else None
    elif used:
        sample = os.path.join(args.capture_dir, used[0])

    if not image_points:
        print(f"❌ 一张都没找到棋盘：检查板子参数 --board {board.cols}x{board.rows}、"
              f"光照、以及板子是否占了画面 1/3 以上", file=sys.stderr)
        return 1
    if len(image_points) < MIN_VIEWS:
        print(f"❌ 有效图只有 {len(image_points)} 张，至少要 {MIN_VIEWS} 张", file=sys.stderr)
        return 1

    # ---- 第二步：解算 ----
    sample_image = read_image_bgr(sample)
    image_size = (sample_image.shape[1], sample_image.shape[0])
    try:
        intrinsics = calibrate(image_points, image_size, board)
    except ValueError as exc:
        print(f"❌ 标定失败：{exc}", file=sys.stderr)
        return 1

    ok = print_report(intrinsics, used, skipped, args.max_rms)
    try:
        path = write_calibration(args.out, intrinsics)
        print(f"  已写入     : {path}")
        print(f"  下一步     : python scripts/calibrate_camera.py --from {path}   # 复核")
    except OSError as exc:
        print(f"❌ 写盘失败：{exc}", file=sys.stderr)
        return 1

    if args.preview and sample:
        try:
            preview_undistort(intrinsics, sample)
        except Exception as exc:                        # noqa: BLE001
            print(f"⚠️  预览窗口打不开（{exc}）；不影响标定结果")

    return 0 if ok else 1
