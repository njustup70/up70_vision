"""本地 OpenCV 点选窗口：每对图点 4 次，得到两条边的原图坐标。

用大白话说
    左边是彩色、右边是对齐深度伪彩。先沿彩色里一条清楚的箱体边点起点、终点，
    再到右边沿同一条实物边点它的起点、终点。程序立刻画出两条线和中点测量位置，
    人看着没问题按 `s` 存下；点歪了按 `r` 重选；这条边压根看不出来按 `u` 记不可测。
    一对图标两条方向不同的边（例如上边 + 右边），共 8 次点击。

为什么不做网页、不让人手填 JSON
    手填坐标等于让人肉眼看像素编号，误差比要测的偏移还大；网页又要另养一套前端。
    本机窗口是最短路径：点击直接拿原图像素，且不用引入新依赖。

为什么必须做坐标换算
    窗口里的图是按显示比例缩放、还可能局部放大的。直接把窗口坐标当测量坐标，
    在缩放 ≠ 1 时会得到系统性偏差，而且看起来「挺合理」——这是本文件最危险的一处。
    规则固定：原图坐标 = 显示坐标 / 显示比例 + 裁剪偏移，左右面板各自先扣面板原点；
    顶部还有一条 34 像素的说明栏，也要扣掉。

已知的坑
    · 边缘检测不参与：这是人工确认的对应参考，不是自动真值（计划书 D2）。
    · 点选记录里不存法向——法向由测量时按端点算，避免人手填一个和线段不垂直的方向。
    · 窗口尺寸超出屏幕时 OpenCV 会截断，别用全屏尺寸做默认。
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .measure import edge_offset_px
from .report import false_color, load_annotations, load_selected, write_annotations

#: 每个面板的显示尺寸（像素）。两块并排约 1500 宽，普通笔记本也放得下。
PANEL_W = 760
PANEL_H = 560

#: 顶部说明栏高度。鼠标坐标是整个窗口的，扣面板原点时必须连它一起扣。
BAR_H = 34

#: 允许的放大档位；1 表示「整幅按面板缩放」，不是原始像素。
ZOOM_LEVELS = (1.0, 2.0, 4.0)

#: 每对图要标的边数（计划书 D2：方向不同的两条，例如上边与右边）。
EDGES_PER_PAIR = 2

_WINDOW = "rgbd_check annotate | s=保存 r=重选 u=不可测 z=放大 Esc=退出"


def view_region(center, zoom: float, image_size) -> tuple:
    """算出当前视图对应原图的哪一块。

    :param center: 视区中心的原图坐标 (x, y)
    :param zoom: 放大倍数，1 起步
    :param image_size: 原图 (w, h)
    :return: `(x0, y0, w, h)`，已夹在图像范围内
    """
    img_w, img_h = image_size
    zoom = max(1.0, float(zoom))
    view_w = min(float(img_w), PANEL_W / zoom)
    view_h = min(float(img_h), PANEL_H / zoom)
    x0 = min(max(0.0, center[0] - view_w / 2.0), img_w - view_w)
    y0 = min(max(0.0, center[1] - view_h / 2.0), img_h - view_h)
    return x0, y0, view_w, view_h


def display_to_image(panel_x: float, panel_y: float, panel_origin, region) -> tuple:
    """显示坐标 -> 原图坐标。先扣面板原点，再按比例还原并加裁剪偏移。

    :param panel_x: 窗口内的 x
    :param panel_y: 窗口内的 y
    :param panel_origin: 该面板左上角的窗口坐标 `(ox, oy)`
    :param region: `view_region()` 的返回值
    :return: 原图浮点坐标 `(x, y)`（不取整：缩放后亚像素是真实信息）
    """
    x0, y0, view_w, view_h = region
    return (x0 + (panel_x - panel_origin[0]) * view_w / PANEL_W,
            y0 + (panel_y - panel_origin[1]) * view_h / PANEL_H)


def in_image(point, image_size) -> bool:
    """点是否在原图范围内（含边界）。"""
    x, y = point
    return 0 <= x <= image_size[0] and 0 <= y <= image_size[1]


def validate_segment(seg, image_size) -> str | None:
    """检查一条线段能否用；不能用时返回原因。端点重合必须拒绝，不能凑方向。"""
    if not seg or len(seg) != 2:
        return "线段不完整"
    (x1, y1), (x2, y2) = seg
    if abs(x2 - x1) < 1e-6 and abs(y2 - y1) < 1e-6:
        return "两个端点重合，无法确定边的方向"
    if not in_image((x1, y1), image_size) or not in_image((x2, y2), image_size):
        return "端点超出原图范围"
    return None


@dataclass
class PairState:
    """一对图的标注状态；`add_edge()` 是唯一往 annotations 里写边的入口。"""

    pair_id: str
    condition_id: str
    image_size: tuple
    edges: list = field(default_factory=list)

    def add_edge(self, edge_id: str, color_seg, depth_seg) -> str | None:
        """校验并加入一条边；返回 `None` 表示成功，否则是拒绝原因。"""
        for seg, name in ((color_seg, "彩色参考线"), (depth_seg, "深度对应线")):
            problem = validate_segment(seg, self.image_size)
            if problem:
                return f"{name}：{problem}"
        if len(self.edges) >= EDGES_PER_PAIR:
            return f"每对图只标 {EDGES_PER_PAIR} 条边，多的不记"
        measured = edge_offset_px(color_seg, depth_seg)
        self.edges.append({
            "edge_id": edge_id,
            "color_segment": [[round(v, 3) for v in pt] for pt in color_seg],
            "depth_segment": [[round(v, 3) for v in pt] for pt in depth_seg],
            "confirmed": True,
            "reason": measured.reason or "",
            "measurement": measured.to_dict(),
            "manual_review": None,
        })
        return None

    def add_unmeasurable(self, edge_id: str, reason: str) -> None:
        """记一条「不可测」的边：保留空位置与原因，不能算进合格。"""
        self.edges.append({
            "edge_id": edge_id, "color_segment": None, "depth_segment": None,
            "confirmed": False, "reason": reason or "人工判定不可测",
            "measurement": None, "manual_review": None,
        })

    def to_frame(self) -> dict:
        return {"pair_id": self.pair_id, "condition_id": self.condition_id,
                "image_size": list(self.image_size), "edges": self.edges}


class _Window:
    """窗口内的一次标注会话：状态机 + 画图 + 键位。"""

    def __init__(self, entry: dict, color: np.ndarray, depth: np.ndarray):
        self.entry = entry
        self.color = color
        self.depth = depth
        self.depth_vis = false_color(depth)
        self.state = PairState(pair_id=entry["pair_id"], condition_id=entry["condition_id"],
                               image_size=(color.shape[1], color.shape[0]))
        self.zoom = 1.0
        self.center = (color.shape[1] / 2.0, color.shape[0] / 2.0)
        self.draft = []
        self.message = "左图点彩色边起点（共 4 次点击）；s 保存 / r 重选 / u 不可测"
        self.cursor = (0, 0)

    # ---------- 坐标 ----------

    def region_color(self):
        return view_region(self.center, self.zoom,
                           (self.color.shape[1], self.color.shape[0]))

    def region_depth(self):
        return view_region(self.center, self.zoom,
                           (self.depth.shape[1], self.depth.shape[0]))

    def panel_of(self, x: int) -> str:
        return "color" if x < PANEL_W else "depth"

    def to_image(self, x: int, y: int):
        """窗口坐标 -> (面板, 原图坐标)。"""
        panel = self.panel_of(x)
        if panel == "color":
            origin, region, size = (0, BAR_H), self.region_color(), (
                self.color.shape[1], self.color.shape[0])
        else:
            origin, region, size = (PANEL_W, BAR_H), self.region_depth(), (
                self.depth.shape[1], self.depth.shape[0])
        point = display_to_image(x, y, origin, region)
        return panel, point, size

    # ---------- 画图 ----------

    def _render_panel(self, img, region, lines, points) -> np.ndarray:
        x0, y0, w, h = region
        crop = img[int(y0):int(y0 + h), int(x0):int(x0 + w)]
        panel = cv2.resize(crop, (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST)
        scale_x, scale_y = PANEL_W / w, PANEL_H / h

        def to_panel(pt):
            return (int((pt[0] - x0) * scale_x), int((pt[1] - y0) * scale_y))

        for seg, color in lines:
            cv2.line(panel, to_panel(seg[0]), to_panel(seg[1]), color, 1, cv2.LINE_AA)
        for pt, color in points:
            cv2.circle(panel, to_panel(pt), 3, color, -1, cv2.LINE_AA)
        return panel

    def render(self) -> np.ndarray:
        color_lines, color_points, depth_lines, depth_points = [], [], [], []
        for edge in self.state.edges:
            if edge.get("color_segment"):
                color_lines.append((edge["color_segment"], (0, 200, 0)))
            if edge.get("depth_segment"):
                depth_lines.append((edge["depth_segment"], (0, 200, 0)))
            measured = edge.get("measurement") or {}
            if measured.get("midpoint"):
                color_points.append((measured["midpoint"], (0, 0, 255)))
            if measured.get("hit"):
                depth_points.append((measured["hit"], (0, 0, 255)))

        # 正在点的这条：已点的点画黄圈，方便看出点了几次。
        for index, point in enumerate(self.draft):
            (color_points if index < 2 else depth_points).append((point, (0, 255, 255)))

        left = self._render_panel(self.color, self.region_color(), color_lines, color_points)
        right = self._render_panel(self.depth_vis, self.region_depth(), depth_lines,
                                   depth_points)
        canvas = np.hstack([left, right])
        cv2.line(canvas, (PANEL_W, 0), (PANEL_W, PANEL_H), (90, 90, 90), 1)
        bar = np.zeros((BAR_H, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, self.message, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, canvas])

    # ---------- 交互 ----------

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor = (x, y)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.state.edges) >= EDGES_PER_PAIR:
            self.message = "两条边已标完；Esc 保存并返回"
            return
        panel, point, size = self.to_image(x, y)
        if not in_image(point, size):
            self.message = "点在图像外，忽略"
            return
        want = "color" if len(self.draft) < 2 else "depth"
        if panel != want:
            self.message = f"这时该在{'彩色' if want == 'color' else '深度'}面板点"
            return
        self.draft.append(point)
        self._advance()

    def _advance(self) -> None:
        steps = ("彩色起点", "彩色终点", "深度起点", "深度终点")
        if len(self.draft) < 4:
            self.message = f"已点 {len(self.draft)}/4，下一步：{steps[len(self.draft)]}"
            return
        edge_id = f"edge{len(self.state.edges) + 1}"
        problem = self.state.add_edge(edge_id, (self.draft[0], self.draft[1]),
                                      (self.draft[2], self.draft[3]))
        self.draft = []
        if problem:
            self.message = f"{problem}；重选这条边"
            return
        measured = self.state.edges[-1]["measurement"]
        offset = measured.get("offset_px")
        if offset is None:
            self.message = f"这条边量不出来（{measured.get('reason')}）；s 存下、r 重选"
        else:
            self.message = f"偏移 {offset:+.2f} px；s 存下、r 重选"
        if len(self.state.edges) >= EDGES_PER_PAIR:
            self.message += "；两条边够了，Esc 保存"

    def on_key(self, key: int) -> str:
        """处理按键，返回 `save` / `quit` / `none`。"""
        if key == -1 or key == 255:
            return "none"
        char = chr(key & 0xFF).lower()
        if char == "s":
            self.draft = []
            self.message = "已保存本窗口标注；Esc 退出"
            return "save"
        if char == "r":
            self.draft = []
            self.message = "重选当前边：左图点彩色起点"
            return "none"
        if char == "u":
            reason = input("这条边不可测的原因（回车用默认）：").strip()
            self.state.add_unmeasurable(f"edge{len(self.state.edges) + 1}", reason)
            self.draft = []
            self.message = f"已记为不可测：{reason or '人工判定不可测'}"
            return "save"
        if char == "z":
            self.zoom = ZOOM_LEVELS[(ZOOM_LEVELS.index(self.zoom) + 1) % len(ZOOM_LEVELS)]
            self.center = self.to_image(*self.cursor)[1]
            self.message = f"放大 {self.zoom:g}x（再按 z 循环、以鼠标位置为中心）"
            return "none"
        if key == 27:
            return "quit"
        return "none"


def _persist(out: Path, annotations: dict, done: dict) -> None:
    """把已完成的 frame 写回 annotations.json（增量保存：中途崩溃不丢已标内容）。"""
    annotations["frames"] = [done[key] for key in sorted(done)]
    write_annotations(out, annotations)


def run(out: Path) -> int:
    """跑完整个标注会话；返回退出码（0 有保存 / 2 没有可标注的图对）。"""
    entries = load_selected(out)
    if not entries:
        print("❌ 这个目录里没有 analyze 选出的图对（先跑 analyze）")
        return 2

    annotations = load_annotations(out)
    done = {frame["pair_id"]: frame for frame in annotations.get("frames", [])}

    for entry in entries:
        data = np.load(entry["_npz"])
        window = _Window(entry, data["color"], data["depth"])
        previous = done.get(entry["pair_id"])
        if previous and previous.get("edges"):
            window.state.edges = previous["edges"]
            window.message = "已载入上次的标注；可直接 Esc 退出或重标"

        cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(_WINDOW, window.on_mouse)
        while True:
            cv2.imshow(_WINDOW, window.render())
            if window.on_key(cv2.waitKey(20)) in ("save", "quit"):
                break
        cv2.destroyWindow(_WINDOW)

        done[entry["pair_id"]] = window.state.to_frame()
        _persist(out, annotations, done)
        _save_annotated_image(out, entry, window)
        _ask_manual_review(out, entry, window.state, annotations, done)

    _persist(out, annotations, done)
    cv2.destroyAllWindows()
    return 0


def _save_annotated_image(out: Path, entry: dict, window: "_Window") -> None:
    """存一张带线段与测量位置的图，报告里作为证据。"""
    folder = out / "overlays"
    folder.mkdir(exist_ok=True)
    cv2.imwrite(str(folder / f"{entry['pair_id']}_annotated.png"), window.render())


def _ask_manual_review(out: Path, entry: dict, state: "PairState", annotations: dict,
                       done: dict) -> None:
    """终端里问一句人工复查结论：数值只量中点，两端张开只能靠人看。"""
    measured = [e for e in state.edges if e.get("measurement")]
    if not measured:
        print(f"  {entry['condition_id']}：没有可测量的边，跳过人工复查")
        return
    if any(e.get("manual_review") for e in measured):
        return
    line = "、".join(f"{e['edge_id']} {e['measurement']['offset_px']:+.2f} px"
                    for e in measured)
    print(f"  {entry['condition_id']}：{line}")
    answer = input("    人工复查（看 overlays 里的 *_annotated.png）：两端无明显张开、"
                   "无整体错位？[y/N/回车=跳过] ").strip().lower()
    review = {"y": "yes", "n": "no"}.get(answer)
    if not review:
        print("    保持未复查（报告里记 UNVERIFIED）")
        return
    for edge in measured:
        edge["manual_review"] = review
    done[entry["pair_id"]] = state.to_frame()
    _persist(out, annotations, done)
