"""相机标定工具：拍一批棋盘图（或读一个图目录）-> 解内参 + 畸变 -> 写 YAML。

用大白话说
    标定分两步，这个工具一步做完：
      第一步「采图」：把棋盘板在镜头前摆十几个不同姿势各拍一张（太正、太斜、靠近、退远、转到
        画面四角都拍一点）。程序自动判断这张图能不能用——找不到板子就不收。
      第二步「解算」：把这些图交给 `aruco/calibration.py`，得到焦距、光心、畸变系数，
        再写进 `configs/camera_calibration.yaml` 给阶段二的位姿估计用。

放什么 / 不放什么
    放"采图 + 解算 + 落盘 + 打印报告"。**纯算法在 `aruco/calibration.py`**：
    找角点、解 K、去畸变都在那边，这里只做 IO 与命令行。
    不碰实时检测/绘制——那是 `cli/run.py` 的事。

依赖谁 / 被谁用
    依赖 `calibration`（算法）、`io.base`（抓帧）、`config`（取相机参数）、`image_io` / `logging`（读写图与日志）。
    入口：`python lab/calibrate_camera.py`。

两种用法
    ① 已有图：`python lab/calibrate_camera.py --images data/calib`
       （图可以从手机拍、也可以用 `--capture` 先采）
    ② 现场采图：`python lab/calibrate_camera.py --capture 15`
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

from vision.config.loader import load_config, load_thresholds, resolve_thresholds_path
from vision.aruco.calibration import (MIN_VIEWS, RMS_USABLE, BoardSpec, CameraIntrinsics,
                                calibrate, find_board_corners, undistort_image)
from vision.io.base import build_source
from vision.image_io import ensure_dir, read_image_bgr, write_image
from vision.logging import get_logger, setup_logging

log = get_logger("tools.calibration.calibrate")

#: 标定结果默认落这里。阶段二的位姿估计会读它（改路径要同步那边）。
DEFAULT_OUT = "configs/camera_calibration.yaml"

#: 现场采图默认存这里（已被 .gitignore 排除，不占版本库）
DEFAULT_CAPTURE_DIR = "data/calib"

#: 采图时认这些后缀
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")

#: 从图目录读图时的上限，防止误指到超大目录
MAX_IMAGES = 200

#: 「太像就不收」的阈值：中心距离占画面宽度比例、格距相对差、倾角差（度）。
#: 不设这道闸，快速连拍会收进七八张同一姿势的图——那会让焦距与主点跑飞（progress 坑 43）
DEDUP_CENTER_RATIO = 0.05
DEDUP_SCALE_RATIO = 0.06
DEDUP_ANGLE_DEG = 6.0

#: 清晰度下限（Laplacian 方差）。低于它就拒收：实测糊图会让 RMS 从 0.5 涨到 1.6 px。
#: 参考：1280x1024 下清晰的棋盘一般在 200~400；手抖/失焦常在 100 以下。
SHARP_MIN_DEFAULT = 120.0

#: 「站稳才拍」需要连续多少帧姿态几乎不变（手抖/运动模糊会让角点发漂）
STABLE_FRAMES = 3
STABLE_CENTER_RATIO = 0.01
STABLE_SCALE_RATIO = 0.02


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


def _pose_signature(corners: np.ndarray, image_size: tuple[int, int],
                    board: BoardSpec | None = None) -> tuple:
    """把一张图里棋盘的姿态压成 4 个数，用来判断「稳不稳」和「像不像」。

    用大白话说
        中心在哪、格子多大、转了多少度——这三件事足够区分"同一姿势"和"换了个姿势"。
        全部按画面宽度归一化，换分辨率也不受影响。

    :param corners: `find_board_corners()` 的返回值 `(N,1,2)`
    :param image_size: `(width, height)`
    :param board: 板子参数（用来把角点摆回 rows×cols 的网格）
    :return: `(cx_norm, cy_norm, spacing_norm, angle_deg)`；角度取 0~180°
    """
    spec = board or BoardSpec()
    width, height = (int(v) for v in image_size)
    grid = np.asarray(corners, dtype=np.float32).reshape(int(spec.rows), int(spec.cols), 2)
    center = grid.reshape(-1, 2).mean(axis=0)
    # 横向相邻角点间距当尺度：对透视不敏感，也不受格子总数影响
    spacing = float(np.mean(np.linalg.norm(np.diff(grid, axis=1), axis=2)))
    top_row = grid[0]
    angle = float(np.degrees(np.arctan2(*(top_row[-1] - top_row[0])[::-1])))
    return (float(center[0]) / width, float(center[1]) / height,
            spacing / width, angle % 180.0)


def _is_too_similar(a: tuple, b: tuple, center_ratio: float, scale_ratio: float,
                    angle_deg: float) -> bool:
    """两个姿态签名是否"像到可以当作同一张"。"""
    d_center = float(np.hypot(a[0] - b[0], a[1] - b[1]))
    d_scale = abs(a[2] - b[2]) / max(a[2], 1e-6)
    d_angle = min(abs(a[3] - b[3]), 180.0 - abs(a[3] - b[3]))
    return d_center < center_ratio and d_scale < scale_ratio and d_angle < angle_deg


def _next_index(out_dir: str) -> int:
    """下一个可用的文件名编号，保证补拍不覆盖已有的图。

    :param out_dir: 采图目录
    :return: 最小可用序号（目录里已有 calib_01/02/03 时返回 4）
    """
    import re

    used = []
    for path in glob.glob(os.path.join(out_dir, "calib_*.png")):
        match = re.search(r"calib_(\d+)\.png$", os.path.basename(path))
        if match:
            used.append(int(match.group(1)))
    return (max(used) + 1) if used else 1


def _existing_signatures(out_dir: str, board: BoardSpec) -> list:
    """把目录里已有图的棋盘姿态读出来，供「补拍时避开已有姿势」用。

    :param out_dir: 采图目录
    :param board: 板子参数
    :return: 若干 `_pose_signature()` 的返回值；读不了/认不出的图直接跳过
    """
    signatures = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.png"))):
        try:
            image = read_image_bgr(path)
        except Exception:                                # noqa: BLE001
            continue
        corners = find_board_corners(image, board)
        if corners is not None:
            signatures.append(_pose_signature(corners, (image.shape[1], image.shape[0]), board))
    return signatures


def capture_from_source(cfg, board: BoardSpec, count: int, out_dir: str,
                        colors: list | None = None,
                        max_frames: int = 0,
                        preview: bool = False,
                        fresh: bool = False,
                        min_sharp: float = SHARP_MIN_DEFAULT) -> tuple[list, list]:
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
    :param preview: 是否开实时窗口（显示当前画面 + 是否认出棋盘 + 已收几张；按 q 中止）
    :param fresh: True 表示当作全新一批（不读已有图的姿势、编号从 1 开始并覆盖）
    :param min_sharp: 清晰度下限；低于它的帧不收（0 表示不检查）
    :return: `(image_points, saved_paths)`；preview 下按 q 会提前结束，已收的仍然有效
    :raises OSError: 输入源构造失败（相机被占用 / SDK 缺失）
    """
    import cv2

    ensure_dir(out_dir)
    source = build_source(cfg.input.source,
                          synthetic_params=dict(vars(cfg.input.synthetic)),
                          replay_params=dict(vars(cfg.input.replay)),
                          camera_params=dict(vars(cfg.input.camera)),
                          colors=colors)
    # 补拍场景：目录里可能已经有图。先把它们的姿态读进来当"已占用"，
    # 这样同一姿势不会再收第二遍；编号也从下一个可用值开始，避免覆盖
    signatures = _existing_signatures(out_dir, board) if not fresh else []
    if signatures and not fresh:
        print(f"  目录里已有 {len(signatures)} 张可用图，会避开这些姿势", flush=True)
    next_index = 1 if fresh else _next_index(out_dir)
    image_points, saved = [], []
    seen = 0
    last_sig, stable = None, 0
    window = "calibrate_camera: capture (q = stop)"
    if preview:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
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
            sharp = cv2.Laplacian(cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2GRAY),
                                  cv2.CV_64F).var()
            ready, hint = False, ""
            if corners is not None and min_sharp > 0 and sharp < min_sharp:
                # 闸 3：糊的不收 —— 角点会抖，重投影误差直接从 0.5 涨到 1.5+
                hint = f"too blurry (sharp {sharp:.0f} < {min_sharp:.0f}) - focus / steady"
                corners = None                  # 当作没认出，走"跳过"分支
            if corners is not None:
                sig = _pose_signature(corners, (frame.image_bgr.shape[1], frame.image_bgr.shape[0]),
                                  board)
                # 闸 1：稳不稳 —— 连续 STABLE_FRAMES 帧姿态几乎不变才算"站稳了"
                if last_sig is not None and _is_too_similar(sig, last_sig, STABLE_CENTER_RATIO,
                                                            STABLE_SCALE_RATIO, 2.0):
                    stable += 1
                else:
                    stable = 1
                last_sig = sig
                # 闸 2：像不像 —— 与已收的某张太像就不收
                duplicate = next((i + 1 for i, old in enumerate(signatures)
                                  if _is_too_similar(sig, old, DEDUP_CENTER_RATIO,
                                                     DEDUP_SCALE_RATIO, DEDUP_ANGLE_DEG)), None)
                if stable < STABLE_FRAMES:
                    hint = f"hold still ({stable}/{STABLE_FRAMES})"
                elif duplicate:
                    hint = f"too similar to #{duplicate} - change pose"
                else:
                    ready = True
            if preview:
                # 实时窗口：让摄影师知道"现在这一帧能不能用"，否则只能干等终端
                show = frame.image_bgr.copy()
                if corners is not None:
                    cv2.drawChessboardCorners(show, (int(board.cols), int(board.rows)), corners, True)
                status = ("saved %d/%d" % (len(saved), count)) if ready else hint
                hud = f"{status}   sharp {sharp:4.0f}   q to stop"
                color = (0, 220, 0) if ready else (0, 0, 255)
                cv2.putText(show, hud, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 5)
                cv2.putText(show, hud, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow(window, show)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    print("  按 q 中止采图", flush=True)
                    break
            if corners is None:
                if seen % 30 == 0:
                    print(f"  … 已看 {seen} 帧，还没找到完整的板子（收到 {len(saved)} 张）",
                          flush=True)
                continue
            if not ready:
                continue                        # 没站稳、或和已收的太像 —— 等它换个姿势
            index = next_index + len(saved)
            path = os.path.join(out_dir, f"calib_{index:02d}.png")
            write_image(path, frame.image_bgr)
            image_points.append(corners)
            saved.append(path)
            signatures.append(sig)
            print(f"  ✅ 本批 {len(saved)}/{count} 张 -> {path}   sharp {sharp:.0f}", flush=True)
    finally:
        # 相机是独占资源：异常路径也必须释放，否则下次打不开（坑 31）
        source.close()
        if preview:
            cv2.destroyWindow(window)
    return image_points, saved


def refine_without_outliers(image_points: list, image_size: tuple[int, int],
                            board: BoardSpec, first: CameraIntrinsics,
                            factor: float = 2.0) -> tuple[CameraIntrinsics, list]:
    """按「单张 RMS 明显偏大」剔除离群图，再用剩下的图重标一次。

    用大白话说
        一批图里常有一两张拍糊的、反光的、或者板子没压平的。它们会让整批的误差指标变差，
        也会把焦距/畸变拽偏。做法很朴素：先标一次，看每张图的误差，把"比中位数大一倍以上"
        的挑出去，剩下的再标一次。剔掉的图会打印出来，方便你重拍那几张。

    :param image_points: 每张图的角点（与 `first.per_view_rms` 一一对应）
    :param image_size: `(width, height)`
    :param board: 板子参数
    :param first: 第一次标定的结果
    :param factor: 判定阈值 = factor × 中位数（默认 2.0，另设 0.5 px 的下限）
    :return: `(新的标定结果, 被剔除的下标列表)`；没东西可剔就原样返回
    """
    import statistics

    errors = list(first.per_view_rms)
    if len(errors) <= MIN_VIEWS:
        return first, []
    median = statistics.median(errors)
    limit = max(float(factor) * median, 0.5)
    keep = [i for i, value in enumerate(errors) if value <= limit]
    dropped = [i for i in range(len(errors)) if i not in keep]
    if not dropped or len(keep) < MIN_VIEWS:
        return first, []
    second = calibrate([image_points[i] for i in keep], image_size, board,
                       flags=first.flags)
    if second.rms >= first.rms * 0.98:          # 剔完没变好，就退回原结果
        return first, []
    return second, dropped


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
        "# 相机内参与畸变系数（由 python lab/calibrate_camera.py 生成，别手改）\n"
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
        epilog="例：python lab/calibrate_camera.py --capture 15  "
               "或  python lab/calibrate_camera.py --images data/calib")
    ap.add_argument("--images", default=None,
                    help="从图目录标定（与 --capture 二选一）")
    ap.add_argument("--capture", type=int, default=0, metavar="N",
                    help="现场从输入源采 N 张再标定（0 = 不采）")
    ap.add_argument("--capture-dir", default=DEFAULT_CAPTURE_DIR,
                    help=f"采到的图存哪（默认 {DEFAULT_CAPTURE_DIR}）")
    ap.add_argument("--board", default="9x6",
                    help="内角点个数，横向x纵向（默认 9x6；不是格子数）")
    ap.add_argument("--square-mm", type=float, default=None,
                    help="格子边长毫米（**必填**；填尺子量到的实际值，不是打印标称值）")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"标定结果写哪（默认 {DEFAULT_OUT}）")
    ap.add_argument("--from", dest="from_file", default=None,
                    help="不重新标，只读回一份已有标定并打印")
    ap.add_argument("--max-rms", type=float, default=RMS_USABLE,
                    help=f"RMS 超过它就判为不可用并返回 1（默认 {RMS_USABLE}）")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="采图时最多看多少帧（0 = 不限；相机是无限流，靠它兜底）")
    ap.add_argument("--min-sharp", dest="min_sharp", type=float, default=SHARP_MIN_DEFAULT,
                    help=f"清晰度下限（Laplacian 方差），低于它就不收（默认 {SHARP_MIN_DEFAULT:g}；0=不检查）")
    ap.add_argument("--fresh", action="store_true",
                    help="当作全新一批：编号从 1 开始（会覆盖同目录已有图）、不避开已有姿势")
    ap.add_argument("--free-k3", action="store_true",
                    help="允许 6 次径向项 k3 自由（默认固定为 0：普通镜头用不到，放开容易过拟合）")
    ap.add_argument("--keep-outliers", action="store_true",
                    help="不剔除误差明显偏大的图（默认会把比中位数大一倍以上的挑掉后重标一次）")
    ap.add_argument("--preview", action="store_true",
                    help="开实时窗口：采图时显示当前帧与已收张数（q 中止），标定完再对比原图/矫正图")
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

    # 只读回已有标定：打印完就退出，不碰相机（这条路径不需要格子边长）
    if args.from_file:
        import yaml
        with open(args.from_file, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        intrinsics = CameraIntrinsics.from_dict(data)
        print_report(intrinsics, [], [], args.max_rms)
        return 0 if intrinsics.rms <= args.max_rms else 1

    if args.square_mm is None:
        # 宁可报错，也不要拿"标称值"去标：我们实测过 15 mm 的板子打出来是 23 mm，
        # 填错不会让内参错，但会让位姿距离整体偏 53%（坑 40）
        print("❌ 必须给出实测格子边长：--square-mm <毫米>（用尺子量，不是打印标称值）",
              file=sys.stderr)
        return 2
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
                colors=list(thresholds), max_frames=args.max_frames,
                preview=args.preview, fresh=args.fresh, min_sharp=args.min_sharp)
        except (OSError, RuntimeError) as exc:
            print(f"❌ 采图失败：{exc}", file=sys.stderr)
            return 1
        used = [os.path.basename(p) for p in saved]
        # 补拍语义：采完对**整个目录**标定，把以前采的图也算进来（否则只用了这次新收的几张）
        if saved:
            print(f"  正在对目录里全部图片重新标定：{args.capture_dir}", flush=True)
            image_points, used, skipped = collect_from_dir(args.capture_dir, board)
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
    # 普通镜头用不到 6 次项：放开 k3 会让它和 k1/k2 互相抵消（实测 rms 几乎不变、系数却失真），
    # 所以默认固定 k3 = 0；确实需要（广角/鱼眼）再加 --free-k3
    import cv2
    flags = 0 if args.free_k3 else cv2.CALIB_FIX_K3
    try:
        intrinsics = calibrate(image_points, image_size, board, flags=flags)
    except ValueError as exc:
        print(f"❌ 标定失败：{exc}", file=sys.stderr)
        return 1

    if not args.keep_outliers:
        refined, dropped = refine_without_outliers(image_points, image_size, board, intrinsics)
        if dropped:
            names = ", ".join(used[i] if i < len(used) else f"第{i+1}张" for i in dropped)
            print(f"  剔除离群   : {names}  "
                  f"（RMS {intrinsics.rms:.2f} -> {refined.rms:.2f} px）")
            intrinsics = refined
    ok = print_report(intrinsics, used, skipped, args.max_rms)
    try:
        path = write_calibration(args.out, intrinsics)
        print(f"  已写入     : {path}")
        print(f"  下一步     : python lab/calibrate_camera.py --from {path}   # 复核")
    except OSError as exc:
        print(f"❌ 写盘失败：{exc}", file=sys.stderr)
        return 1

    if args.preview and sample:
        try:
            preview_undistort(intrinsics, sample)
        except Exception as exc:                        # noqa: BLE001
            print(f"⚠️  预览窗口打不开（{exc}）；不影响标定结果")

    return 0 if ok else 1
