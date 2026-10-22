#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HSV 阈值调节器：拖滑条看掩膜，回车把阈值写回 configs/color_thresholds.yaml。

为什么需要它
    判色的唯一参数来源是阈值表（configs/color_thresholds.yaml），而场地一换光照
    阈值就得重调。让人肉眼盯着掩膜拖滑条，比"改数值 → 跑一遍 → 看输出图"快一个数量级。

用法
    python -m up70_vision.tools.hsv_tuner --image data/one_red_block.jpg
    python -m up70_vision.tools.hsv_tuner --source synthetic
    python -m up70_vision.tools.hsv_tuner --source replay --path data/clips/site.mp4

窗口键位（先点一下窗口再按）
    1..9      切换正在调的颜色（顺序取自阈值表的键序）
    , .       切换正在调的**段**（红色有两段：偏橙的 0-10 与偏紫的 170-179）
    r         从文件重新载入当前颜色（放弃改动）
    回车/s    把当前颜色的当前段写回阈值表（自动备份 .bak）
    p         打印当前颜色的 YAML 片段
    n         打印命中率统计（整图有效像素/命中像素）
    q/ESC     退出

注意
    cv2.imshow 需要 X11/Wayland，且 OpenCV 不能是 headless 版（见 tools/__init__.py）。
    图上的文字一律 ASCII——Hershey 字体没有汉字字形。
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np
import yaml

from ..config.loader import (find_default_config, read_yaml, resolve_path,
                             split_thresholds_doc)
from ..core.color import mask_for_segments, parse_thresholds, valid_pixel_mask
from ..io.base import build_source
from ..utils.image_io import read_image_bgr
from ..utils.logging import get_logger, setup_logging

log = get_logger("tools.hsv_tuner")

WIN_PARAM = "params"
WIN_MAIN = "result"
TRACKBARS = ["H_min", "H_max", "S_min", "S_max", "V_min", "V_max", "min_area"]
SEGMENT_RE = re.compile(r"^(\s*-\s*)\[[^\]]*\](\s*(?:#.*)?)$")


# ---------------------------------------------------------------- 写回
def write_back_segments(path: str, color: str, segments) -> str:
    """把某颜色的所有段写回 YAML，**保留注释与格式**。

    只替换该颜色名下那几行 `- [..]` 里的数字，其余字节不动（用 yaml.safe_dump 会
    把所有注释吃掉，那个损失比方便更大）。写之前先存一份 .bak。

    :return: 备份文件路径
    :raises ValueError: 文件里找不到该颜色，或段数对不上
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines_before = list(lines)              # 备份用：改之前的原始行

    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^{re.escape(color)}\s*:", line):
            start = i
            break
    if start is None:
        raise ValueError(f"阈值表里没有颜色 {color!r}：{path}")

    seg_idx = 0
    for j in range(start + 1, len(lines)):
        m = SEGMENT_RE.match(lines[j].rstrip("\n"))
        if not m:
            break
        if seg_idx < len(segments):
            numbers = ", ".join(str(int(v)) for v in segments[seg_idx])
            ending = "\n" if lines[j].endswith("\n") else ""
            lines[j] = f"{m.group(1)}[{numbers}]{m.group(2)}{ending}"
        seg_idx += 1
    if seg_idx != len(segments):
        raise ValueError(
            f"颜色 {color!r} 在文件里有 {seg_idx} 段，但内存里有 {len(segments)} 段——"
            f"请手动对齐后重试（本工具不改段数，只改数值）")

    backup = path + ".bak"
    with open(backup, "w", encoding="utf-8") as f:
        f.writelines(lines_before)          # .bak = 本次改动之前的版本
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return backup


def yaml_snippet(color: str, segments) -> str:
    body = "".join(f"  - [{', '.join(str(int(v)) for v in seg)}]\n" for seg in segments)
    return f"{color}:\n{body}"


# ---------------------------------------------------------------- 取图
class FrameFeeder:
    """把 --image / --source 两种输入统一成 next_frame()。"""

    def __init__(self, args):
        self.source = None
        self.image = None
        if args.image:
            self.image = read_image_bgr(args.image)
            self.desc = f"image:{os.path.basename(args.image)}"
        else:
            cfg_dir = os.path.dirname(os.path.abspath(args.config or find_default_config()))
            path_arg = args.path or ""
            if path_arg and not os.path.isabs(path_arg):
                path_arg = os.path.abspath(path_arg)
            self.source = build_source(
                args.source,
                synthetic_params={"width": 640, "height": 480, "num_frames": 10 ** 6,
                                  "blocks_per_frame": 6, "seed": args.seed},
                replay_params={"path": path_arg or os.path.join(cfg_dir, "..", "data", "clips"),
                               "kind": "auto", "frame_stride": args.frame_stride, "max_frames": 0},
                colors=args.colors)
            self.desc = f"{args.source}"
        self._last = None

    def next_frame(self):
        if self.image is not None:
            self._last = self.image
        else:
            frame = self.source.read()
            if frame is None:                     # 回放结束就停在最后一帧
                return self._last
            self._last = frame.image_bgr
        return self._last


# ---------------------------------------------------------------- 界面
def nothing(_):
    pass


def build_ui(segments):
    cv2.namedWindow(WIN_PARAM, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PARAM, 460, 330)
    cv2.imshow(WIN_PARAM, np.zeros((60, 300), np.uint8))
    cv2.waitKey(50)          # 先让窗口实体化再挂滑条（否则报 NULL window handler）
    h_lo, s_lo, v_lo, h_hi, s_hi, v_hi = segments
    limits = [(0, 179), (0, 179), (0, 255), (0, 255), (0, 255), (0, 255), (0, 50000)]
    values = [h_lo, h_hi, s_lo, s_max_default(s_hi), v_lo, v_hi, 400]
    for name, (lo, hi), val in zip(TRACKBARS, limits, values):
        cv2.createTrackbar(name, WIN_PARAM, int(min(max(val, lo), hi)), hi, nothing)


def s_max_default(v):
    return 255 if v > 255 else v


def read_ui():
    """从滑条读出当前段：返回 (h_lo,s_lo,v_lo,h_hi,s_hi,v_hi) 与面积下限。

    注意滑条的排列：H_min,H_max,S_min,S_max,V_min,V_max —— 与阈值六元组的分量顺序
    (h_lo,s_lo,v_lo,h_hi,s_hi,v_hi) 不同，这里做一次映射，避免抄错。
    """
    h_min, h_max, s_min, s_max, v_min, v_max, min_area = (
        cv2.getTrackbarPos(n, WIN_PARAM) for n in TRACKBARS)
    return (h_min, s_min, v_min, h_max, s_max, v_max), min_area


def push_ui(segment, min_area=None):
    h_lo, s_lo, v_lo, h_hi, s_hi, v_hi = segment
    for name, val in zip(TRACKBARS, [h_lo, h_hi, s_lo, s_hi, v_lo, v_hi]):
        cv2.setTrackbarPos(name, WIN_PARAM, int(val))
    if min_area is not None:
        cv2.setTrackbarPos("min_area", WIN_PARAM, int(min_area))


def stats_for(hsv, segment, name):
    """整图统计：当前段的命中像素、有效像素、命中率。"""
    seg = parse_thresholds({name: [[*segment]]})[name]
    mask = mask_for_segments(hsv, seg)
    valid = valid_pixel_mask(hsv, 0, 0) > 0            # 整图都算分母：这里只做参考
    n_valid = int(np.count_nonzero(valid))
    hits = int(np.count_nonzero(mask > 0))
    return mask, hits, n_valid, (hits / n_valid if n_valid else 0.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hsv_tuner", description="HSV 阈值调节器")
    ap.add_argument("--image", default=None, help="要调的单张图（推荐：一张包含目标颜色的实拍图）")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "replay"],
                    help="不给 --image 时用哪种输入源")
    ap.add_argument("--path", default=None, help="--source replay 时的视频/图片目录")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--config", default=None, help="配置文件（用来定位阈值表所在目录）")
    ap.add_argument("--thresholds", default=None, help="直接指定阈值表路径")
    ap.add_argument("--colors", nargs="*", default=None, help="只调这些颜色（默认全部）")
    args = ap.parse_args(argv)

    setup_logging("INFO")

    config_path = args.config or find_default_config()
    if args.thresholds:
        thresholds_path = os.path.abspath(args.thresholds)
    else:
        cfg_data = read_yaml(config_path)
        rel = ((cfg_data.get("color") or {}).get("thresholds_file") or "color_thresholds.yaml")
        thresholds_path = resolve_path(os.path.dirname(os.path.abspath(config_path)), rel)

    # 阈值表文件头还有一段保留键 `color:`（判色门槛），它不是颜色，先摘掉再解析
    raw = read_yaml(thresholds_path)
    colors, _settings = split_thresholds_doc(raw, thresholds_path)
    parsed = parse_thresholds(colors)
    names = list(parsed) if not args.colors else [c for c in args.colors if c in parsed]
    if not names:
        print(f"❌ 没有可调的颜色（阈值表：{list(parsed)}）", file=sys.stderr)
        return 2
    # 内存里保存可变的段列表
    table = {n: [list(seg) for seg in parsed[n]] for n in parsed}

    feeder = FrameFeeder(args)
    frame = feeder.next_frame()
    if frame is None:
        print("❌ 没有取到任何图", file=sys.stderr)
        return 2

    color_idx, seg_idx = 0, 0
    build_ui(tuple(table[names[color_idx]][seg_idx]))
    print(f"阈值表：{thresholds_path}")
    print(f"输入  ：{feeder.desc}")
    print(f"颜色  ：{names}（按数字键切换；红色有 2 段，用 , . 切换段）")
    print("先点一下窗口再按键；回车/s 写回文件，q 退出")

    def current_segment():
        return tuple(table[names[color_idx]][seg_idx])

    while True:
        if frame is None:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        segment, min_area = read_ui()
        name = names[color_idx]
        mask, hits, n_valid, ratio = stats_for(hsv, segment, name)

        canvas = frame.copy()
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = 0
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            kept += 1
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        head = [
            f"color={name} seg{seg_idx + 1}/{len(table[name])}",
            f"seg=[{', '.join(str(v) for v in segment)}]",
            f"hits={hits} valid={n_valid} ratio={ratio:.3f}",
            f"boxes>={min_area}: {kept}",
            "1-9 color  , . seg  Enter write  q quit",
        ]
        for i, text in enumerate(head):
            cv2.putText(canvas, text.encode("ascii", "replace").decode(),
                        (10, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow(WIN_MAIN, canvas)
        cv2.imshow(WIN_PARAM, mask)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == 13 or key == ord("s"):          # 回车 / s 写回
            table[name][seg_idx] = list(segment)
            try:
                backup = write_back_segments(thresholds_path, name, table[name])
                print(f"✅ 已写回 {name}：{table[name]}（备份 {os.path.basename(backup)}）")
            except ValueError as exc:
                print(f"❌ 写回失败：{exc}", file=sys.stderr)
        elif key == ord("p"):
            print(yaml_snippet(name, table[name]))
        elif key == ord("n"):
            print(f"{name} seg{seg_idx + 1}: 命中 {hits} / 有效 {n_valid} = {ratio:.3f}")
        elif key == ord(",") or key == ord("."):
            step = -1 if key == ord(",") else 1
            seg_idx = (seg_idx + step) % len(table[name])
            push_ui(table[name][seg_idx])
            print(f"当前段 -> {name} seg{seg_idx + 1}")
        elif key == ord("r"):
            fresh_colors, _fresh_settings = split_thresholds_doc(
                read_yaml(thresholds_path), thresholds_path)
            fresh = parse_thresholds(fresh_colors)
            table[name] = [list(seg) for seg in fresh[name]]
            push_ui(table[name][seg_idx])
            print(f"已重新载入 {name}：{table[name]}")
        elif ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(names):
                color_idx, seg_idx = idx, 0
                push_ui(table[names[color_idx]][seg_idx])
                print(f"当前颜色 -> {names[color_idx]}（{len(table[names[color_idx]])} 段）")
        elif args.source == "replay":
            frame = feeder.next_frame()

        if args.image is not None:
            frame = feeder.next_frame()              # 单图模式：每轮重取同一张

    cv2.destroyAllWindows()
    if feeder.source is not None:
        feeder.source.close()
    print("退出（未按回车时的改动不会写回文件）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
