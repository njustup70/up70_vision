
import sys
from pathlib import Path

# `lab/` 不是包：直接 `python lab/xxx.py` 跑，脚本目录会进 sys.path，
# 但仓库根不会。这一行把仓库根补进去，才能 `import vision`（lab 可以依赖 vision，反过来不行）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArUco 测试图组生成器：造出「多码图 / 旋转图 / 无码图」，并随图写一份真值 JSON。

为什么需要它
    任务单的验收标准是"同一张图片中的多个 ArUco 码全部正确识别（ID、四角、中心）"。
    要能断言这句话，就得先有一批**知道标准答案**的图：每个码 ID 多少、四角在哪、
    中心在哪。本工具把答案算准（不是事后标的），和 PNG 同名写一份 .json，
    测试与自检脚本照着它比对，不需要人工标点。

真值是怎么算准的
    贴图用的是仿射变换 M（cv2.getRotationMatrix2D + 平移），**真值用同一个 M 算**：
    四角 = M · [源图四角, 1]，中心 = M · 源图中心。所以真值与画出来的像素是同一套
    几何，误差只来自插值（实测 ≤ 1 px，见 tests/test_aruco.py 的容差）。

放哪
    生成物默认写 tests/assets/aruco/：markers/（单码图）+ scenes/（多码、旋转、无码）。
    都是小体积 PNG（单张十几 KB），随仓库分发，规则见 assets/README.md。

用法
    python lab/gen_aruco.py                     # 生成整套到 tests/assets/aruco
    python lab/gen_aruco.py --list-scenes        # 看有哪些场景
    python lab/gen_aruco.py --scene multi rotated
    python lab/gen_aruco.py --ids 0 1 2 3 --side-px 120
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from vision.aruco.detector import check_opencv_aruco, predefined_dictionary
from vision.cli.output import ensure_dir, write_image_with_truth
from vision.logging import get_logger, setup_logging

log = get_logger("tools.aruco.gen_scenes")

# 任务单指定的字典：4x4、50 个 ID。换字典会让已有真值全部作废，所以写死成默认值。
DEFAULT_DICTIONARY = "DICT_4X4_50"
DEFAULT_OUT_DIR = "tests/assets/aruco"

# 一张图里一个码的边长（像素）。小于 ~30 px 时检测率会明显下降——这不是理论值，
# 是本工具造图时"能不能稳定检出"的下限参考，调参别低于它。
MIN_SIDE_PX = 30

# 场景名 -> 中文说明（--list-scenes 与 --help 用）
SCENE_HELP = {
    "markers": "单码图：每个 ID 一张，码居中、四周留白（最干净的识别验证）",
    "multi": "多码图：一张图里横排 3 个 + 3x2 排 6 个不同的 ID",
    "rotated": "旋转图：3 个码分别转 0° / 30° / 60°，外加一个 45° 的单码图",
    "empty": "无码图：纯背景一张 + 带色块一张（验证无码时返回空列表、不误报）",
}
ALL_SCENES = ("markers", "multi", "rotated", "empty")


# ---------------------------------------------------------------- 造码 / 贴码
def make_marker_image(dictionary, marker_id: int, side_px: int = 200,
                      border_bits: int = 1) -> np.ndarray:
    """生成一张单码灰度图（side_px × side_px，黑色边框是码本体的一部分）。

    OpenCV 返回的码图**整张就是码**：外框黑边的最外沿就是检测器报的四角，
    所以贴图时按整张图的边界算真值即可。
    """
    if side_px < MIN_SIDE_PX:
        raise ValueError(f"码边长太小（{side_px} < {MIN_SIDE_PX} px），检测会不稳定")
    if border_bits < 1:
        raise ValueError(f"border_bits 至少为 1：{border_bits}")
    check_opencv_aruco(need_generate=True)
    return cv2.aruco.generateImageMarker(dictionary, int(marker_id), int(side_px),
                                         borderBits=int(border_bits))


@dataclass
class Placement:
    """一个码贴上去之后的真值。"""

    marker_id: int
    center: list            # [x, y]，图像像素坐标
    corners: list           # [[x, y]] × 4，顺序 TL → TR → BR → BL
    side_px: int
    angle_deg: float

    def to_dict(self) -> dict:
        return {"id": int(self.marker_id),
                "center": [round(float(v), 3) for v in self.center],
                "corners": [[round(float(x), 3), round(float(y), 3)] for x, y in self.corners],
                "side_px": int(self.side_px),
                "angle_deg": round(float(self.angle_deg), 3)}


def place_marker(canvas_bgr: np.ndarray, marker_gray: np.ndarray,
                 cx: float, cy: float, angle_deg: float = 0.0) -> Placement:
    """把码图贴到画布上「中心在 (cx, cy)、绕中心转 angle_deg 度」的位置。

    实现要点（也是真值能算准的原因）
        · 旋转在**放得下整个码**的方画布 big×big 上做，不能原地转——原尺寸转会把四角切掉
        · 真值四角 = M · [源图外边界四角, 1] + 贴图偏移，用的就是同一个 M
        · 外边界取 ±0.5 而不是 0 / side-1：检测器报的是黑框**最外沿**（像素边界），
          不是最外侧黑像素的中心。差这半个像素，四角误差就从 ~0.1px 变成 ~0.5px
        · 白底一起贴上去：码四周必须有安静区，否则相邻码会互相干扰识别

    :return: Placement（该码的真值）
    :raises ValueError: 贴图位置越界（图太小或码太大）
    """
    side = int(marker_gray.shape[0])
    src_center = ((side - 1) / 2.0, (side - 1) / 2.0)
    rad = math.radians(float(angle_deg))
    big = int(math.ceil(side * (abs(math.cos(rad)) + abs(math.sin(rad))))) + 2

    matrix = cv2.getRotationMatrix2D(src_center, float(angle_deg), 1.0)
    # 把旋转中心从"源图中心"挪到"big×big 画布中心"
    matrix[0, 2] += big / 2.0 - src_center[0]
    matrix[1, 2] += big / 2.0 - src_center[1]
    rotated = cv2.warpAffine(marker_gray, matrix, (big, big),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    dx = int(round(float(cx) - big / 2.0))
    dy = int(round(float(cy) - big / 2.0))
    height, width = canvas_bgr.shape[:2]
    if dx < 0 or dy < 0 or dx + big > width or dy + big > height:
        raise ValueError(
            f"贴图越界：中心 ({cx:.0f},{cy:.0f})、旋转 {angle_deg}°、码边长 {side}px 时需要"
            f" {big}x{big} 的方框，落到 [{dx},{dy}]~[{dx + big},{dy + big}]，"
            f"画布只有 {width}x{height}；把码改小或把画布放大")

    canvas_bgr[dy:dy + big, dx:dx + big] = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

    # 码图整张就是码：外边界在像素坐标系里是 [-0.5, side-0.5]（像素边界，不是像素中心）
    edge = 0.5
    local = np.array([[-edge, -edge], [side - edge, -edge],
                      [side - edge, side - edge], [-edge, side - edge]], dtype=np.float64)
    homog = np.hstack([local, np.ones((4, 1), dtype=np.float64)])
    corners = (matrix @ homog.T).T + np.array([dx, dy], dtype=np.float64)
    center = matrix @ np.array([src_center[0], src_center[1], 1.0]) + np.array([dx, dy])
    return Placement(marker_id=-1,                     # 由调用方填真实 id
                     center=[float(center[0]), float(center[1])],
                     corners=[[float(x), float(y)] for x, y in corners],
                     side_px=side, angle_deg=float(angle_deg))


# ---------------------------------------------------------------- 场景
@dataclass
class Scene:
    """一张测试图 = 图像 + 它的真值。"""

    name: str
    image_bgr: np.ndarray
    markers: list = field(default_factory=list)        # list[Placement]
    params: dict = field(default_factory=dict)

    def truth(self, dictionary_name: str, image_rel: str) -> dict:
        height, width = self.image_bgr.shape[:2]
        return {
            "scene": self.name,
            "image": image_rel,
            "dictionary": dictionary_name,
            "width": int(width),
            "height": int(height),
            "generator": "lab/gen_aruco",
            "params": dict(self.params),
            "markers": [p.to_dict() for p in self.markers],
        }


def _white_canvas(width: int, height: int) -> np.ndarray:
    return np.full((int(height), int(width), 3), 255, dtype=np.uint8)


def scene_single(dictionary, marker_id: int, side_px: int = 240,
                 margin_px: int = 40) -> Scene:
    """单码图：码居中，四周留白 margin_px。"""
    margin_px = max(int(margin_px), int(math.ceil(side_px * 0.4)))   # 留白至少 0.4 个码宽
    width = height = side_px + 2 * margin_px
    canvas = _white_canvas(width, height)
    placement = place_marker(canvas, make_marker_image(dictionary, marker_id, side_px),
                             cx=width / 2.0, cy=height / 2.0, angle_deg=0.0)
    placement.marker_id = int(marker_id)
    return Scene(name=f"marker_{int(marker_id):02d}", image_bgr=canvas, markers=[placement],
                 params={"side_px": int(side_px), "margin_px": int(margin_px), "angle_deg": 0.0})


def _grid_positions(count: int, cols: int, width: int, height: int, gap_px: int, side_px: int):
    """算 count 个码在画布上的中心点（严格等距，肉眼与断言都好核对）。"""
    rows = int(math.ceil(count / float(cols)))
    step_x = side_px + gap_px
    step_y = side_px + gap_px
    total_w = step_x * (cols - 1)
    total_h = step_y * (rows - 1)
    x0 = width / 2.0 - total_w / 2.0
    y0 = height / 2.0 - total_h / 2.0
    return [(x0 + (i % cols) * step_x, y0 + (i // cols) * step_y) for i in range(count)]


def scene_multi(dictionary, ids=(0, 1, 2), side_px: int = 130, cols: int = 3,
                gap_px: int = 60, canvas_size=(760, 520), name: str | None = None) -> Scene:
    """多码图：ids 按 cols 列网格排布，全部不旋转。"""
    ids = [int(i) for i in ids]
    width, height = canvas_size
    canvas = _white_canvas(width, height)
    centers = _grid_positions(len(ids), cols, width, height, gap_px, side_px)
    markers = []
    for marker_id, (cx, cy) in zip(ids, centers):
        placement = place_marker(canvas, make_marker_image(dictionary, marker_id, side_px),
                                 cx=cx, cy=cy, angle_deg=0.0)
        placement.marker_id = marker_id
        markers.append(placement)
    return Scene(name=name or f"multi_{len(ids)}", image_bgr=canvas, markers=markers,
                 params={"ids": ids, "side_px": int(side_px), "cols": int(cols),
                         "gap_px": int(gap_px), "angle_deg": 0.0})


def scene_rotated(dictionary, ids=(0, 1, 2), angles=(0.0, 30.0, 60.0), side_px: int = 140,
                  cols: int = 3, gap_px: int = 70, canvas_size=(820, 480),
                  name: str | None = None) -> Scene:
    """旋转图：每个码绕自身中心转一个角度（默认 0° / 30° / 60°）。

    这一组是验收清单"旋转时持续更新"的图片版：真值四角是转过角度的四边形，
    不是轴对齐矩形——四角顺序（TL→TR→BR→BL）对不上时误差会立刻暴露。
    """
    ids = [int(i) for i in ids]
    angles = [float(a) for a in angles]
    if len(angles) != len(ids):
        raise ValueError(f"ids 与 angles 数量必须一致：{len(ids)} vs {len(angles)}")
    width, height = canvas_size
    canvas = _white_canvas(width, height)
    centers = _grid_positions(len(ids), int(cols), width, height, gap_px, side_px)
    markers = []
    for marker_id, angle, (cx, cy) in zip(ids, angles, centers):
        placement = place_marker(canvas, make_marker_image(dictionary, marker_id, side_px),
                                 cx=cx, cy=cy, angle_deg=angle)
        placement.marker_id = marker_id
        markers.append(placement)
    return Scene(name=name or f"rotated_{len(ids)}", image_bgr=canvas, markers=markers,
                 params={"ids": ids, "angles": angles, "side_px": int(side_px),
                         "gap_px": int(gap_px)})


def scene_empty_plain(canvas_size=(640, 480), seed: int = 2027) -> Scene:
    """无码图（纯背景）：白底 + 一点灰度渐变，一个有棱角的东西都没有。"""
    width, height = canvas_size
    gradient = np.tile(np.linspace(200, 250, width, dtype=np.uint8), (height, 1))
    canvas = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    return Scene(name="empty_plain", image_bgr=canvas, markers=[],
                 params={"seed": int(seed)})


def scene_empty_blocks(canvas_size=(640, 480), count: int = 5, seed: int = 2027) -> Scene:
    """无码图（有干扰）：画几块有棱有角的实心色块，专门查误报。

    色块是矩形的、边界又直又长，是最容易被误当成码的东西；这一张必须返回空列表。
    """
    width, height = canvas_size
    canvas = _white_canvas(width, height)
    rng = np.random.default_rng(int(seed))
    palette = [(0, 0, 255), (0, 215, 255), (0, 200, 0), (255, 255, 0), (255, 128, 0), (255, 0, 255)]
    for i in range(int(count)):
        w = int(rng.integers(70, 150))
        h = int(rng.integers(70, 150))
        x = int(rng.integers(10, max(11, width - w - 10)))
        y = int(rng.integers(10, max(11, height - h - 10)))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), palette[i % len(palette)], thickness=-1)
    return Scene(name="empty_blocks", image_bgr=canvas, markers=[],
                 params={"count": int(count), "seed": int(seed)})


def build_scenes(scene: str, dictionary, ids=(0, 1, 2), side_px: int = 130) -> list:
    """按场景名造 Scene 列表。scene="all" 时造全部。"""
    key = str(scene).strip().lower()
    names = ALL_SCENES if key == "all" else (key,)
    for one in names:
        if one not in SCENE_HELP:
            raise ValueError(f"未知场景 {one!r}；可用：{list(SCENE_HELP)} 或 all")

    scenes = []
    for one in names:
        if one == "markers":
            scenes += [scene_single(dictionary, i, side_px=max(int(side_px), 200)) for i in ids]
        elif one == "multi":
            row_ids = [int(i) for i in ids][:3]
            scenes.append(scene_multi(dictionary, ids=row_ids, side_px=int(side_px),
                                      cols=len(row_ids), canvas_size=(760, 520),
                                      name=f"multi_row_{len(row_ids)}"))
            grid_ids = list(range(6))                  # 固定 6 个 ID：3x2 网格是"多码"最有代表性的一张
            scenes.append(scene_multi(dictionary, ids=grid_ids, side_px=int(side_px),
                                      cols=3, canvas_size=(760, 620), name="multi_grid_6"))
        elif one == "rotated":
            turn_ids = [int(i) for i in ids][:3]
            scenes.append(scene_rotated(dictionary, ids=turn_ids, angles=(0.0, 30.0, 60.0),
                                        side_px=int(side_px), name=f"rotated_{len(turn_ids)}"))
            scenes.append(scene_rotated(dictionary, ids=[turn_ids[0]], angles=(45.0,),
                                        side_px=int(side_px), cols=1, gap_px=0, name="rotated_single_45"))
        elif one == "empty":
            scenes.append(scene_empty_plain())
            scenes.append(scene_empty_blocks())
    return scenes


# ---------------------------------------------------------------- 落盘
def write_scene(out_dir: str, scene: Scene, dictionary_name: str,
                subdir: str = "scenes") -> list:
    """把一张场景图写成 PNG + 同名真值 JSON。

    :return: 写出的文件路径列表 [png, json]
    """
    target = os.path.join(out_dir, subdir)
    ensure_dir(target)
    png_rel = os.path.join(subdir, f"{scene.name}.png")
    png_path = os.path.join(out_dir, png_rel)
    truth = scene.truth(dictionary_name, png_rel.replace(os.sep, "/"))
    png_path, json_path = write_image_with_truth(scene.image_bgr, png_path, truth)
    log.info("写出 %s：%d 个码 -> %s", scene.name, len(scene.markers), png_rel)
    return [png_path, json_path]


def generate(out_dir: str = DEFAULT_OUT_DIR, scene: str = "all", dictionary_name: str = DEFAULT_DICTIONARY,
             ids=(0, 1, 2), side_px: int = 130) -> list:
    """生成图组，返回写出的文件列表。"""
    dictionary, name = predefined_dictionary(dictionary_name)
    written = []
    for one in build_scenes(scene, dictionary, ids=ids, side_px=side_px):
        subdir = "markers" if one.name.startswith("marker_") else "scenes"
        written += write_scene(out_dir, one, name, subdir=subdir)
    return written


# ---------------------------------------------------------------- 真值读取（自检脚本与测试共用）
def truth_path_for(image_path: str) -> str:
    """给一张场景图，返回它的真值 JSON 路径（同名 .json）。"""
    return os.path.splitext(image_path)[0] + ".json"


def load_truth(image_path: str) -> dict:
    """读出一张场景图的真值。

    :raises FileNotFoundError: 同名 .json 不存在（图组没生成过，或漏了真值）
    """
    path = truth_path_for(image_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"找不到真值文件 {path}；图组请用 python lab/gen_aruco.py 重新生成")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_scene_images(root: str):
    """遍历图组目录下的所有场景图（markers/ 与 scenes/），按路径排序，保证可复现。"""
    found = []
    for sub in ("markers", "scenes"):
        directory = os.path.join(root, sub)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith(".png"):
                found.append(os.path.join(directory, name))
    return found


# ---------------------------------------------------------------- 命令行
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gen_aruco",
        description="生成 ArUco 测试图组（多码图 / 旋转图 / 无码图）+ 每张图的真值 JSON")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR,
                    help=f"输出目录（默认 {DEFAULT_OUT_DIR}，相对仓库根）")
    ap.add_argument("--scene", nargs="+", default=["all"],
                    help=f"要生成哪些场景：{' / '.join(ALL_SCENES)} / all（可多选）")
    ap.add_argument("--dictionary", default=DEFAULT_DICTIONARY,
                    help=f"ArUco 字典（默认 {DEFAULT_DICTIONARY}，任务单指定）")
    ap.add_argument("--ids", nargs="+", type=int, default=[0, 1, 2],
                    help="用哪些码 ID（默认 0 1 2）")
    ap.add_argument("--side-px", type=int, default=130, help="码边长像素（默认 130）")
    ap.add_argument("--list-scenes", action="store_true", help="列出场景后退出")
    ap.add_argument("--log-level", default="INFO", help="日志等级，默认 INFO")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if args.list_scenes:
        for name in ALL_SCENES:
            print(f"  {name:<8} {SCENE_HELP[name]}")
        print("\n  用 --scene all 一次生成全部")
        return 0

    try:
        written = generate(out_dir=args.out, scene=" ".join(args.scene),
                           dictionary_name=args.dictionary, ids=args.ids, side_px=args.side_px)
    except (ValueError, RuntimeError, IOError) as exc:
        print(f"❌ 生成失败：{exc}")
        return 2

    pngs = [p for p in written if p.endswith(".png")]
    print(f"\n✅ 生成 {len(pngs)} 张图 + {len(written) - len(pngs)} 份真值 JSON -> {args.out}")
    for path in pngs:
        print(f"   {os.path.relpath(path, args.out)}")
    print("\n  看图：ls {0}/scenes  比对：python lab/check_aruco.py --dir {0}"
          .format(args.out))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
