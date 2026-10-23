"""造一张可打印的棋盘标定板（黑白格），供 `calibrate_camera.py` 使用。

用大白话说
    标定要用"标准件"：一块格子边长已知的黑白棋盘。手边没有就去买/去借也行，但最常见的情况
    是"现在就要、手里没有"。这个工具负责生成一张能直接打印的图：**格子在打印后必须是正方形**，
    所以它按「毫米 + DPI」算像素，而不是按图里随便多少像素。

放什么 / 不放什么
    只放"把一块板画出来"。找角点、解内参都在 `aruco/calibration.py` 与 `calibrate_camera.py`。

依赖谁 / 被谁用
    依赖 `calibration`（板子参数）与 `image_io`（写图）。入口：`python lab/gen_chessboard.py`。

对外约定
    · **`--cols/--rows` 是内角点个数**，不是格子数（10x7 格 -> 9x6 内角点）。这也是标定工具要填的数。
    · 输出图四周留一圈白边（静区），打印后别裁掉：没有白边，最外圈格子会与纸边/桌面连成一片，
      `findChessboardCorners` 会找不到板子。
    · 退出码：0 成功；1 写图失败；2 参数非法（argparse 给的）。

已知的坑
    · **打印机缩放**：几乎每台打印机都会把"标称尺寸"缩放 1~2%。打印完必须用尺子量一下格子边长，
      把量到的值填给 `--square-mm`（坑 40）。注意它不影响内参，影响的是"距离算出来差多少"：
      我们这台打印机把标称 15 mm 打成了 23 mm（+53%）。
    · 我们踩过这个坑的根因是 **PNG 没写 DPI**，打印驱动按 200 dpi 猜（300/200 = 1.5 倍）。
      所以本工具除了 PNG 还会**写一份 PDF**：PDF 的页面尺寸是精确的物理尺寸（点 = 1/72 英寸），
      打印时选「实际大小」必然 1:1，不依赖驱动猜 DPI。
    · 纸会弯。贴在硬纸板上再拍：板子一旦不平，标出来的畸变系数会把"纸的弯"也学进去（坑 42）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# `lab/` 不是包：直接 `python lab/xxx.py` 跑，脚本目录会进 sys.path，
# 但仓库根不会。这一行把仓库根补进去，才能 `import vision`（lab 可以依赖 vision，反过来不行）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import os
import sys

from vision.aruco.calibration import BoardSpec
from vision.cli.output import ensure_dir, write_image
from vision.logging import get_logger, setup_logging

log = get_logger("tools.calibration.gen_board")

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


def inject_png_dpi(path: str, dpi: int) -> None:
    """给已有的 PNG 插入 `pHYs` 块，把 DPI 写进去（驱动就不会瞎猜）。

    用大白话说
        PNG 本身不强制记录"每个像素多大"，很多打印机驱动读不到就按 200 dpi 猜——我们标称
        15 mm 的格子因此被打成 23 mm（差 1.5 倍）。`pHYs` 这个块就是专门记这个的：
        每米多少像素 + 单位是"米"。插在 IHDR 之后、IDAT 之前即可。

    :param path: PNG 路径（原地改写）
    :param dpi: 每英寸像素数（300 是常见打印分辨率）
    :raises OSError: 不是合法 PNG
    """
    import struct
    import zlib

    with open(path, "rb") as handle:
        raw = handle.read()
    signature = b"\x89PNG\r\n\x1a\n"
    if raw[:8] != signature:
        raise OSError(f"不是合法 PNG：{path}")

    # IHDR 是第一个块：结构 = 长度(4) + 类型(4) + 数据(长度) + CRC(4)
    ihdr_length = struct.unpack(">I", raw[8:12])[0]
    insert_at = 8 + 12 + ihdr_length

    ppm = int(round(float(dpi) / 0.0254))           # 每英寸 dpi -> 每米像素
    payload = struct.pack(">IIB", ppm, ppm, 1)      # x ppm, y ppm, 单位=米
    chunk = (struct.pack(">I", len(payload)) + b"pHYs" + payload
             + struct.pack(">I", zlib.crc32(b"pHYs" + payload) & 0xFFFFFFFF))

    with open(path, "wb") as handle:
        handle.write(raw[:insert_at] + chunk + raw[insert_at:])


def write_pdf(path: str, image_bgr, dpi: int, paper: str = "a4") -> tuple:
    """把棋盘写成一份**物理尺寸精确**的 PDF，打印时选「实际大小」就是 1:1。

    用大白话说
        PDF 的坐标单位是"点"（1 pt = 1/72 英寸），页面框写多大，打印机就得按多大印——
        不需要驱动去猜 DPI。把图按 `像素数 / dpi * 72` 换算成点、居中放在页面里，
        打出来格子就是设计值。

    :param path: 输出 PDF 路径
    :param image_bgr: 棋盘图（HxWx3）
    :param dpi: 设计分辨率（决定图的物理尺寸）
    :param paper: `"a4"` 用 A4 页面居中；`"auto"` 让页面正好等于图的大小
    :return: `(页面尺寸(mm), 图尺寸(mm))`
    :raises OSError: 写文件失败
    """
    import zlib

    import numpy as np

    height, width = image_bgr.shape[:2]
    img_pt = (width / float(dpi) * 72.0, height / float(dpi) * 72.0)     # 图在纸上的大小（点）
    if paper == "auto":
        page_pt = (img_pt[0] + 36.0, img_pt[1] + 36.0)                  # 四周留 12.7 mm 白边
    else:
        page_pt = (595.28, 841.89)                                      # A4 210x297 mm
    origin = ((page_pt[0] - img_pt[0]) / 2.0, (page_pt[1] - img_pt[1]) / 2.0)

    rgb = np.ascontiguousarray(image_bgr[:, :, ::-1]).tobytes()          # BGR -> RGB
    compressed = zlib.compress(rgb, 6)
    content = (f"q {img_pt[0]:.3f} 0 0 {img_pt[1]:.3f} {origin[0]:.3f} {origin[1]:.3f} "
               f"cm /Im0 Do Q").encode()

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_pt[0]:.3f} {page_pt[1]:.3f}] "
         f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>").encode(),
        (f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
         f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
         f"/Length {len(compressed)} >>\nstream\n").encode() + compressed + b"\nendstream",
        (f"<< /Length {len(content)} >>\nstream\n").encode() + content + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF\n").encode()

    with open(path, "wb") as handle:
        handle.write(bytes(out))
    return ((page_pt[0] / 72.0 * 25.4, page_pt[1] / 72.0 * 25.4),
            (img_pt[0] / 72.0 * 25.4, img_pt[1] / 72.0 * 25.4))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gen_chessboard",
        description="生成可打印的棋盘标定板（内角点个数不要填成格子数）",
        epilog="例：python lab/gen_chessboard.py --cols 9 --rows 6 --square-mm 15")
    ap.add_argument("--cols", type=int, default=9, help="横向**内角点**个数（默认 9）")
    ap.add_argument("--rows", type=int, default=6, help="纵向**内角点**个数（默认 6）")
    ap.add_argument("--square-mm", type=float, default=15.0,
                    help="格子边长毫米（默认 15；打印后要用尺子量，把量到的值填给标定工具）")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"打印 DPI（默认 {DEFAULT_DPI}）")
    ap.add_argument("--margin", type=int, default=1, help="四周白边宽度（以格子为单位，默认 1）")
    ap.add_argument("--out", default="knowledge/calibration/chessboard_9x6_15mm.png",
                    help="输出路径（默认 knowledge/calibration/chessboard_9x6_15mm.png）")
    ap.add_argument("--no-pdf", action="store_true",
                    help="只写 PNG，不生成 PDF（默认会同时写一份 1:1 的 PDF）")
    ap.add_argument("--paper", choices=["a4", "auto"], default="a4",
                    help="PDF 页面：a4=固定 A4 居中（默认）；auto=页面正好等于图的大小")
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
    inject_png_dpi(args.out, args.dpi)          # 不写 DPI，打印驱动会按 200dpi 猜（实测放大 1.5 倍）
    print(f"已写入  : {args.out}（已写入 {args.dpi} dpi 元数据）")

    if not args.no_pdf:
        # 内容比 A4 还宽/高时，按 A4 出 PDF 只会被裁或被驱动缩小（1:1 就没了），
        # 所以自动换成"页面正好等于图"的尺寸，并明确要求更大的纸
        paper = args.paper
        too_big = width_mm > A4_PRINTABLE_MM[0] or height_mm > A4_PRINTABLE_MM[1]
        if paper == "a4" and too_big:
            paper = "auto"
            print(f"  ⚠️  内容 {width_mm:.1f}x{height_mm:.1f} mm 超过 A4 可打印区，"
                  f"PDF 已改用「页面=图」的尺寸：必须用 A3 或更大纸张，否则打不出 1:1")
        pdf_path = os.path.splitext(args.out)[0] + ".pdf"
        try:
            page_mm, img_mm = write_pdf(pdf_path, image, args.dpi, paper)
            print(f"PDF     : {pdf_path}")
            print(f"          页面 {page_mm[0]:.1f} x {page_mm[1]:.1f} mm，"
                  f"棋盘内容 {img_mm[0]:.1f} x {img_mm[1]:.1f} mm（按 {args.dpi} dpi 换算）")
        except OSError as exc:                  # 写不出 PDF 不该让整件事失败
            print(f"⚠️  PDF 写失败：{exc}（PNG 仍可用）")

    print()
    print("怎么打印才 1:1：")
    print("  1. **优先用刚生成的 PDF**：纸张选 A4，缩放选「实际大小 / 100%」，别选「适合页面」；")
    print("     PDF 的页面尺寸是精确的物理尺寸，不依赖驱动去猜 DPI。")
    print("  2. 用 PNG 也行：它现在带 300 dpi 元数据，但个别驱动仍会改缩放，打完要确认。")
    print(f"  3. 打完核对：格子应当是 {board.square_mm:g} mm（设计值）。若量出来偏了，就把量到的值填给")
    print("     标定工具的 --square-mm —— 它是必填的，填错会让位姿距离整体偏。")
    print("  4. 别裁掉白边（静区）；把纸贴在硬纸板上，别留卷边。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
