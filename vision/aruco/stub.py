"""按契约返回假 ArUco 码的 stub（任务单第一节第 4 条：让两边互不阻塞）。

用大白话说
    它不真的看图。它按公式在画面里"摆"出几个假的 ArUco 码，
    让"取帧 -> 找码 -> 画框 -> 终端输出 -> 存图"整条链路先跑通。
    等真检测器就位，换掉它就行，上层（绘制/终端/落盘）一行都不用改。

为什么需要它
    "输入源 + 可视化 + 编排"不该等真 ArUco 实现写完才开始。stub 按
    vision/docs/interface.md 第 2.2 / 3.2 节的契约返回假的 `ArucoMarker`，
    这样相机、画框、入口、落盘都能先跑通；真实现（`aruco.py`）就位后直接换掉。

契约（原文见 vision/docs/interface.md 第 2.2 与 3.2 节）
    · 返回 `list[ArucoMarker]`，**无码时 `[]`**（不是 None）
    · 按 `marker_id` **升序**
    · `center_px` = 四角**算术平均**（不是 bbox 中心）
    · `corners_px` 顺时针、起点是码自身坐标系的左上角，dtype float32
    · `detect(image_bgr)` 形参名固定

⚠️ **调用方每帧必须恰好调用一次 `detect`**
    本 stub 用**自身的帧计数器**决定"现在是第几帧、码转到哪儿"（签名里没有帧号）。
    隔帧调用、或一帧调两次，都会让运动错位——症状是"画面里码的位置不对"，**不报错**。

行为
    · 默认造 3 个码，id 固定 0/1/2，横排居中
    · moving=True（默认）时第一个码绕自己的基准位置转圈 + 自转，
      角点是真的转过角度的四边形，用来验证"码移动 / 旋转时持续更新"

    第 0 帧的版面（640×480、side_px=90、gap=1.6×90=144）：

        (0,0) ┌────────────────────────────────────────────┐
              │                                            │
              │        ┌──┐      ┌──┐      ┌──┐            │
              │        │ 0│      │ 1│      │ 2│            │  ← 中心 y = 240
              │        └──┘      └──┘      └──┘            │
              │       x=176     x=320     x=464            │  ← 中心 x（整排居中）
              └────────────────────────────────────────────┘
        moving=True 时 id=0 的码中心沿 x=176 附近半径 60 的圆移动，同时自转；
        id=1 / id=2 保持不动（用于"多码同时出现，其中一个在动"的场景）。

无码模式（任务单第 6 条「无码时正常运行不异常退出」）
    · `mode="none"`：`detect()` 恒返回 `[]`
    · `empty_after=N`：前 N 帧正常出码，从第 N 帧起返回 `[]`（演"码移出画面"）；0 = 不启用
    两者都不改类名、不改 `detect` 签名、不改返回类型——不动契约。
    **走无码分支时内部帧计数器照样自增**，否则切回正常模式时码的运动位置会错位。

    YAML 里这样写（或在命令行 `--set` 覆盖）：

        detector:
          name: stub_aruco
          stub_aruco:
            ids: [0, 1, 2]
            mode: none          # 全程无码
            # empty_after: 90   # 或者：跑满 90 帧后开始无码
"""

import math

import numpy as np

from .types import ArucoMarker


class StubArucoDetector:
    """按契约伪造 ArUco 码的检测器（无模型、无相机也能跑通链路）。

    用大白话说
        它像一个"假演员"：导演（上层代码）让它演"画面里有 3 个码"，它就按约定
        交出 3 个码的位置；让它演"码被拿走了"，它就从某一帧起交空列表。
        真正的检测器换成真的看图，但交出来的东西格式一模一样。

    属性（构造时定型，之后只读；`frame_count` 除外，每调一次 `detect` 加 1）
        name (str)              检测器名，工厂 `build_detector` 用它对齐配置里的 `detector.name`
        ids (list[int])         要伪造的码 ID，列表顺序 = 画面里从左到右的顺序
        side_px (int)           码边长（像素）；码是正方形，四角由它算出来
        moving (bool)           第一个码是否动（绕基准位置转圈 + 自转）
        radius_px (int)         转圈半径（像素）；moving=False 时无意义
        deg_per_frame (float)   每帧转过的角度（度）；决定圈速与自转速度
        mode (str)              "normal" 正常出码 / "none" 恒返回 []（造无码输入）
        empty_after (int)       >0 时前 N 帧正常、第 N 帧起返回 []；0 = 不启用
        frame_count (int)       已调用 `detect` 的次数，既是"当前帧号"也是运动相位来源
    """

    name = "stub_aruco"

    def __init__(self,
                 ids=(0, 1, 2),
                 side_px: int = 90,
                 moving: bool = True,
                 radius_px: int = 60,
                 deg_per_frame: float = 12.0,
                 mode: str = "normal",
                 empty_after: int | None = 0):
        """构造 stub 检测器；参数全部做类型归一与合法性检查（配置写错时早报错）。

        :param ids: 要伪造的码 ID 列表（顺序即从左到右）
        :param side_px: 码边长（像素）
        :param moving: 第一个码是否动起来（转圈 + 自转）
        :param radius_px: 圆周运动半径
        :param deg_per_frame: 每帧转过的角度（决定运动速度）
        :param mode: "normal" 正常出码 / "none" 恒返回 []（造无码输入用）
        :param empty_after: 跑到第 N 帧后开始返回 []；0（默认）= 不启用
        :raises ValueError: ids 为空、边长非正、半径/角度为负、mode 非法、empty_after 为负

        例子
            StubArucoDetector()                          # 默认：3 个码，id 0 动、1/2 静
            StubArucoDetector(ids=(5,), moving=False)    # 只伪造 id=5，完全静止
            StubArucoDetector(mode="none")               # 恒无码（第 6 条的最小输入）
            StubArucoDetector(empty_after=90)            # 前 90 帧有码，第 90 帧起无码
        """
        # 一律存成明确类型：配置来自 YAML，可能是 int / float / 字符串
        self.ids = [int(i) for i in ids]                  # 伪造的码 ID（顺序 = 从左到右）
        self.side_px = int(side_px)                       # 码边长（像素）
        self.moving = bool(moving)                        # 第一个码是否转圈 + 自转
        self.radius_px = int(radius_px)                   # 转圈半径（像素）
        self.deg_per_frame = float(deg_per_frame)         # 每帧转过的角度（度）
        # mode / empty_after：任务单第 6 条"无码时正常运行"的定量输入源。
        # 归一化（去空格、转小写）后再校验，避免 "NONE" 这种写法被误判成非法。
        self.mode = str(mode).strip().lower()
        if self.mode not in ("normal", "none"):
            raise ValueError(f"mode 只能是 'normal' / 'none'，收到 {mode!r}")
        # None / "" 都当作 0（不启用）；YAML 里写 `empty_after:` 留空时就是 None
        self.empty_after = 0 if empty_after in (None, "") else int(empty_after)
        if self.empty_after < 0:
            raise ValueError(f"empty_after 不能为负：{self.empty_after}")
        self.frame_count = 0                              # 已调用的次数；决定码转到哪一帧
        if not self.ids:
            raise ValueError("ids 不能为空（stub 至少要伪造一个码）")
        if self.side_px <= 0:
            raise ValueError(f"side_px 必须为正：{self.side_px}")
        if self.radius_px < 0:
            raise ValueError(f"radius_px 不能为负：{self.radius_px}")

    def detect(self, image_bgr: np.ndarray) -> list:
        """按当前帧号伪造一组 `ArucoMarker`（**每次调用都算"过了一帧"**）。

        用大白话说
            第 1 次调用就交"第 1 帧的样子"，第 2 次调用交"第 2 帧的样子"，
            码的位置会按设定挪一点、转一点。所以调用方每帧必须正好调一次，
            调多或调少都会让"码走到哪了"对不上——而且不会报错，只是位置看起来不对。

        图像内容不参与计算，只取它的宽高来摆放码；因此没有相机/没有素材也能跑。
        每帧的结果都重新构造，不缓存、不沿用上一帧。

        :param image_bgr: HxWx3 uint8 BGR 图（形参名是契约；本 stub 只用它的 shape）
        :return: `list[ArucoMarker]`；`normal` 模式下按 `marker_id` 升序，
            `none` / 过了 `empty_after` 时返回 `[]`（不是 None）
        """
        # ---- 步骤 1：帧号先记下再自增 ----
        # index = "本次调用对应第几帧"（从 0 开始）；frame_count 随后 +1。
        # 无码分支提前 return 时也必须已经自增，否则：
        #   · empty_after 会一直等不到第 N 帧（永远漏数）
        #   · 切回正常模式时运动相位倒退，码的位置会突然跳一下
        index = self.frame_count
        self.frame_count += 1

        # ---- 步骤 2：无码模式直接返回空列表 ----
        # 两种触发条件都归一成 []，上层只判"空列表"这一种情况：
        #   mode == "none"           → 恒无码
        #   empty_after > 0 且到了帧 → 从第 N 帧起无码
        # 注意返回的是 [] 而不是 None：契约要求"无码 = 空列表"。
        if self.mode == "none" or (self.empty_after > 0 and index >= self.empty_after):
            return []

        # ---- 步骤 3：算本帧的版面（只看图像尺寸，不看内容）----
        height, width = image_bgr.shape[:2]        # 只取尺寸，不看像素内容
        # 相邻码中心距 = 1.6 倍边长：留出 0.6 条边的空隙，码之间不会视觉上贴在一起，
        # 也保证旋转 / 移动时有足够余量、不会互相压住
        gap = self.side_px * 1.6
        total = gap * (len(self.ids) - 1)          # 首尾码中心之间的总跨度（只有 1 个码时为 0）
        x_start = width / 2.0 - total / 2.0        # 最左码中心 x：让整排水平居中
        y_mid = height / 2.0                       # 所有码中心 y：画面垂直中线
        # 当前相位（度）：第 index 帧转过 index * deg_per_frame 度；不动的场景恒为 0
        angle = (index * self.deg_per_frame) if self.moving else 0.0

        # ---- 步骤 4：逐个码算中心与自转角，再交给 _marker 造四角点 ----
        markers = []
        for position, marker_id in enumerate(self.ids):
            # position 是"第几个码"（0 起，决定横向位置）；marker_id 才是伪造的 ID
            cx = x_start + position * gap          # 本码中心 x
            cy = y_mid                             # 本码中心 y
            spin = 0.0                             # 本码自转角（度），只有第一个码非零
            if position == 0 and self.moving:
                # 第一个码：中心沿半径 radius_px 的圆走一圈，同时自转同样角度，
                # 这样"移动"与"旋转"能在同一帧里被验证到（任务单第 5 条）。
                # 极坐标 → 直角坐标：cx = cx0 + r·cosθ，cy = cy0 + r·sinθ
                rad = math.radians(angle)
                cx += self.radius_px * math.cos(rad)
                cy += self.radius_px * math.sin(rad)
                spin = angle                      # 第一个码同时自转，角点不再是轴对齐
            markers.append(self._marker(marker_id, cx, cy, spin))

        # ---- 步骤 5：按 marker_id 升序返回（契约 2.2 第 2 条）----
        # ids 配置的顺序是"从左到右"，与 ID 大小无关（比如 ids=[5, 1, 3] 也合法），
        # 所以这里必须再排一次——终端的 id 列表才稳定，可 diff。
        markers.sort(key=lambda m: m.marker_id)
        return markers

    def _marker(self, marker_id: int, cx: float, cy: float, spin_deg: float) -> ArucoMarker:
        """按 中心 + 边长 + 自转角 造一个码的四角点与中心。

        用大白话说
            拿一张正方形纸片，先按指定角度转一下，再把它挪到指定位置，
            最后记下四个角现在的坐标。码的"身份"（ID）本身不参与几何计算。

        角点顺序 **TL → TR → BR → BL**（码自身坐标系）：先在原点构造轴对齐正方形，
        再用旋转矩阵转 `spin_deg`，最后平移到 `(cx, cy)`。顺序与 OpenCV 原生顺序一致，
        所以上层绘制 / 后续 solvePnP 不需要再做重排。

        几何过程（以边长 2h、中心在原点的正方形为例）：

            (-h,-h) TL ─────── TR (h,-h)          ① 以中心为原点摆好轴对齐正方形
                    │    ↑    │                    ② 绕原点旋转 spin_deg
                    │  中心   │                    ③ 平移到 (cx, cy)
            (-h, h) BL ─────── BR (h, h)

        旋转方向
            用的是数学上标准的逆时针旋转矩阵，但图像坐标 y 轴向下，
            所以在屏幕上看是**顺时针**转。四个角点的相对顺序不变，仍是 TL→TR→BR→BL。

        :param marker_id: 码的 ID（DICT_4X4_50 取值 0~49）
        :param cx: 码中心 x（像素）
        :param cy: 码中心 y（像素）
        :param spin_deg: 自转角（度）；0 = 不转（角点轴对齐）
        :return: `ArucoMarker`，`corners_px` 为 (4,2) float32，`center_px` = 四角算术平均
        """
        half = self.side_px / 2.0
        # 以中心为原点的四个角，顺序 TL / TR / BR / BL（y 轴向下，与图像坐标一致）
        base = np.array([[-half, -half], [half, -half], [half, half], [-half, half]],
                        dtype=np.float64)
        # 角度转弧度（math 的三角函数吃弧度，配置里的 deg_per_frame 是度）
        rad = math.radians(spin_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        # 二维旋转矩阵 R = [[cos, -sin], [sin, cos]]。
        # corners 的每一行是一个列向量 [x, y]，所以用 行向量 @ R.T 实现 R·v。
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
        corners = (base @ rot.T) + np.array([cx, cy], dtype=np.float64)   # 先转后平移
        # 契约 2.2 第 3 条：center_px 是四角算术平均（旋转后仍然等于原中心）
        # —— 这里正好可以当作自检：旋转不改变四个角的均值
        center = corners.mean(axis=0)
        # ArucoMarker 内部会把 corners 归一成 float32；这里显式转一次，契约一眼可见
        return ArucoMarker(marker_id=int(marker_id),
                           corners_px=corners.astype(np.float32),
                           center_px=(float(center[0]), float(center[1])))
