"""造一张可打印的棋盘标定板（黑白格），供 `calibrate_camera.py` 使用。

用大白话说
    标定要用"标准件"：一块格子边长已知的黑白棋盘。手边没有就去买/去借也行，但最常见的情况
    是"现在就要、手里没有"。这个工具负责生成一张能直接打印的图：**格子在打印后必须是正方形**，
    所以它按「毫米 + DPI」算像素，而不是按图里随便多少像素。

放什么 / 不放什么
    只放"把一块板画出来"。找角点、解内参都在 `core/calibration.py` 与 `calibrate_camera.py`。

依赖谁 / 被谁用
    依赖 `core.calibration`（板子参数）与 `utils.image_io`（写图）。薄入口见 `scripts/gen_chessboard.py`。

对外约定
    · **`--cols/--rows` 是内角点个数**，不是格子数（10x7 格 -> 9x6 内角点）。这也是标定工具要填的数。
    · 输出图四周留一圈白边（静区），打印后别裁掉：没有白边，最外圈格子会与纸边/桌面连成一片，
      `findChessboardCorners` 会找不到板子。
    · 退出码：0 成功；1 写图失败；2 参数非法（argparse 给的）。

已知的坑
    · **打印机缩放**：几乎每台打印机都会把"标称尺寸"缩放 1~2%。打印完必须用尺子量一下格子边长，
      把量到的值填给 `--square-mm`（坑 40）。注意它不影响内参，影响的是"距离算出来差多少"：
      我们这台打印机把标称 15 mm 打成了 23 mm（+53%）。
    · 纸会弯。贴在硬纸板上再拍：板子一旦不平，标出来的畸变系数会把"纸的弯"也学进去（坑 42）。
"""

from __future__ import annotations

import argparse
import os
import sys

from ..core.calibration import BoardSpec
from ..utils.image_io import ensure_dir, write_image
from ..utils.logging import get_logger, setup_logging

log = get_logger("tools.gen_chessboard")

#: 默认打印 DPI。300 是喷墨/激光的常用分辨率，200 也能用但边缘更容易糊。
DEFAULT_DPI = 300

#: A4 的可打印区域（毫米）。超出就提示"要么改小格子、要么换 A3"，而不是让人打出来被裁掉。
A4_PRINTABLE_MM = (190.0, 270.0)


def render_board(board: BoardSpec, dpi: int = DEFAULT_DPI, margin_squares: int = 1):
    """画出棋盘图（纯函数，不写文件）。

    用大白话说
        一块 N×M 内角点的板子，实际是 (N+1)×(M+1) 个格子交错涂黑。像素边长按
        `边长(mm) / 25.4 * DPI` 算，四周再各留 `margin_squares` 个格子的白边当静区。

    :param board: 板子参数（内角点个数 + 格子边长，毫米）
    :param dpi: 打印分辨率，决定像素尺寸
    :param margin_squares: 四周白边宽度（以格子为单位，默认 1）
    :return: `(image_bgr, square_px)`；image_bgr 是 HxWx3 uint8 的白底黑格图
    :raises ValueError: dpi 或白边非法
    """
    import numpy as np

    if int(dpi) <= 0:
        raise ValueError(f"dpi 必须为正，收到 {dpi}")
    if int(margin_squares) < 0:
        raise ValueError(f"margin_squares 不能为负，收到 {margin_squares}")

    # 一个格子占多少像素；round 保证全图格子等宽（否则角点检测会以为是透视）
    square_px = max(8, int(round(float(board.square_mm) / 25.4 * float(dpi))))
    squares_x = int(board.cols) + 1
    squares_y = int(board.rows) + 1
    margin_px = int(margin_squares) * square_px

    width = squares_x * square_px + 2 * margin_px
    height = squares_y * square_px + 2 * margin_px
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    # (i + j) 为偶数的格子涂黑：这个相位决定了左上角是黑格，OpenCV 两种相位都能认
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                y0 = margin_px + row * square_px
                x0 = margin_px + col * square_px
                image[y0:y0 + square_px, x0:x0 + square_px] = 0
    return image, square_px


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gen_chessboard",
        description="生成可打印的棋盘标定板（内角点个数不要填成格子数）",
        epilog="例：python scripts/gen_chessboard.py --cols 9 --rows 6 --square-mm 15")
    ap.add_argument("--cols", type=int, default=9, help="横向**内角点**个数（默认 9）")
    ap.add_argument("--rows", type=int, default=6, help="纵向**内角点**个数（默认 6）")
    ap.add_argument("--square-mm", type=float, default=15.0,
                    help="格子边长毫米（默认 15；打印后要用尺子量，把量到的值填给标定工具）")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"打印 DPI（默认 {DEFAULT_DPI}）")
    ap.add_argument("--margin", type=int, default=1, help="四周白边宽度（以格子为单位，默认 1）")
    ap.add_argument("--out", default="assets/calib/chessboard_9x6_15mm.png",
                    help="输出路径（默认 assets/calib/chessboard_9x6_15mm.png）")
    ap.add_argument("--log-level", default="INFO", help="日志等级，默认 INFO")
    return ap


def main(argv=None) -> int:
    """命令行入口：画板子 -> 写文件 -> 打印打印与测量提示。

    :return: 0 成功；1 写图失败
    """
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    board = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square_mm)
    image, square_px = render_board(board, dpi=args.dpi, margin_squares=args.margin)

    width_mm = image.shape[1] / args.dpi * 25.4
    height_mm = image.shape[0] / args.dpi * 25.4
    print(f"棋盘    : {board.cols}x{board.rows} 内角点（= {board.cols + 1}x{board.rows + 1} 格）"
          f"  格子 {board.square_mm:g} mm")
    print(f"图像    : {image.shape[1]}x{image.shape[0]} px  @ {args.dpi} dpi"
          f"（一个格子 {square_px} px）")
    print(f"打印尺寸: {width_mm:.1f} x {height_mm:.1f} mm")
    if width_mm > A4_PRINTABLE_MM[0] or height_mm > A4_PRINTABLE_MM[1]:
        print(f"⚠️  超过 A4 可打印区 {A4_PRINTABLE_MM[0]:g}x{A4_PRINTABLE_MM[1]:g} mm："
              f"把 --square-mm 改小（比如 15 -> 12），或换 A3/胶装到硬纸板上分两张拼")

    ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    write_image(args.out, image)
    print(f"已写入  : {args.out}")
    print()
    print("打印后必做（否则标定结果会整体偏 1~2%）：")
    print("  1. 用尺子量一个格子的实际边长，把「标定工具的 --square-mm」改成量到的值；")
    print(f"     标称 {board.square_mm:g} mm，量到多少就填多少（本机实测：标称 15 mm 量出 23 mm）。")
    print("  2. 别裁掉白边（静区）；把纸贴在硬纸板上，别留卷边。")
    print("  3. 用本工具默认参数时，标定工具也用它默认的 9x6。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
