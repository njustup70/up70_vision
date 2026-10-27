"""一条边的近似像素偏移：中点沿法向打到深度线段上，量两点距离。

用大白话说
    人在彩色图上沿箱子的一条边点两个点（参考线），再在深度伪彩图上点同一条
    实物边的两个点（对应线）。程序取参考线中点，沿它的法向往外打一条射线，
    与对应线的交点就是「同一条边在深度图里的位置」，两者距离就是这条边的偏移。

为什么只量中点
    冒烟检查要回答的是「有没有明显整体错位」，不是「亚像素精度是多少」。
    中点一个数就够判 5 px 门槛；两端是否张开另由人在放大图上看。
    多取几个点、算 P50/P95，只会让「沿同一条边密集取点」看起来像独立实验。

已知的坑
    · 平行（含几乎平行）时没有唯一交点，必须报 UNVERIFIED，不能给一个巨大的数。
    · 交点落在线段外同样不算——那说明这边的两条线根本不是同一条边，
      延长线段去凑交点是自欺。
    · 有符号偏移是沿参考线法向算的。不同朝向的边符号含义不同，不能直接相加抵消，
      所以报告里同时给绝对值与方向。
"""

from dataclasses import dataclass

from .status import PASS, UNVERIFIED

#: 叉积小于这个值就当作平行。像素坐标量级下 1e-9 足够小，不算「差不多平行」。
_PARALLEL_EPS = 1e-9


@dataclass(frozen=True)
class Measurement:
    """一条边的测量结果。`offset_px` 为 None 表示没测出来，原因在 `reason`。"""

    offset_px: float | None
    status: str
    reason: str
    midpoint: tuple
    normal: tuple
    hit: tuple | None

    @property
    def abs_px(self) -> float | None:
        return None if self.offset_px is None else abs(self.offset_px)

    def to_dict(self) -> dict:
        return {
            "offset_px": None if self.offset_px is None else round(self.offset_px, 3),
            "abs_offset_px": None if self.abs_px is None else round(self.abs_px, 3),
            "status": self.status,
            "reason": self.reason,
            "midpoint": [round(v, 2) for v in self.midpoint],
            "normal": [round(v, 6) for v in self.normal],
            "hit": None if self.hit is None else [round(v, 2) for v in self.hit],
        }


def _segment_ok(seg) -> bool:
    """两个端点、长度不为零。重合端点无法定义方向。"""
    if not seg or len(seg) != 2:
        return False
    (x1, y1), (x2, y2) = seg
    return abs(x2 - x1) > 1e-6 or abs(y2 - y1) > 1e-6


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def edge_offset_px(color_seg, depth_seg) -> Measurement:
    """算一条边的近似偏移。

    :param color_seg: 彩色参考线 `((x1, y1), (x2, y2))`，原图浮点坐标
    :param depth_seg: 深度对应线 `((x1, y1), (x2, y2))`，同上
    :return: `Measurement`；算不出来时 `offset_px` 为 None、`status` 为 UNVERIFIED
    """
    origin = (0.0, 0.0)
    if not _segment_ok(color_seg):
        return Measurement(None, UNVERIFIED, "彩色参考线端点重合或缺失", origin, (0.0, 0.0), None)
    if not _segment_ok(depth_seg):
        return Measurement(None, UNVERIFIED, "深度对应线端点重合或缺失", origin, (0.0, 0.0), None)

    (cx1, cy1), (cx2, cy2) = color_seg
    (dx1, dy1), (dx2, dy2) = depth_seg

    mx, my = (cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0
    # 参考线法向：把方向转 90°，再归一化。方向取「左法向」，符号只用于报告方向。
    vx, vy = cx2 - cx1, cy2 - cy1
    length = (vx * vx + vy * vy) ** 0.5
    nx, ny = -vy / length, vx / length

    ux, uy = dx2 - dx1, dy2 - dy1
    denom = _cross(nx, ny, ux, uy)
    if abs(denom) < _PARALLEL_EPS:
        return Measurement(None, UNVERIFIED, "彩色参考线与深度对应线平行，无唯一交点",
                           (mx, my), (nx, ny), None)

    # 解 m + t*n = d1 + u*v
    wx, wy = dx1 - mx, dy1 - my
    t = _cross(wx, wy, ux, uy) / denom
    u = _cross(wx, wy, nx, ny) / denom
    if not (-1e-9 <= u <= 1.0 + 1e-9):
        return Measurement(None, UNVERIFIED,
                           f"交点落在线段外（线段参数 u={u:.3f}），两条线可能不是同一条边",
                           (mx, my), (nx, ny), None)

    hit = (mx + t * nx, my + t * ny)
    return Measurement(offset_px=float(t), status=PASS, reason="",
                       midpoint=(mx, my), normal=(nx, ny), hit=hit)
