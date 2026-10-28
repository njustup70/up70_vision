"""从配对结果里挑出「每段一张」的代表帧，并定义本轮要采的三个场景。

用大白话说
    一段 10 秒的 bag 有几百对图，冒烟检查只要中间那一对——挑法必须写死，
    否则「挑到好看的那一帧」就成了人为挑选证据。规则：以稳定段起点的彩色
    header 时间加半个稳定段为目标，取彩色时间离它最近的那一对；等距取更早的。

为什么要固定场景清单
    三个场景（中心近、中心远、右侧中）是计划书 D1 定的最小集合。清单写在代码里，
    报告才能把「没采的」标成 UNVERIFIED，而不是干脆不提——漏采和合格在
    报告里长得一样，是最容易糊弄自己的一种错。
"""

from dataclasses import dataclass

from .pairing import PairingResult

#: 本轮必采的三个场景。顺序固定，报告逐行对照。
REQUIRED_CONDITIONS = ("center_near", "center_far", "right_mid")

#: 每个场景要标的边界数（上边 + 右边之类，方向不同、不平行）。
EDGES_PER_PAIR = 2

#: 冒烟门槛：六条边的近似偏移都不超过这个像素数，才允许记 PASS。
#: 5 px 是待现场复核的排错门槛，不是厂家精度承诺。
OFFSET_LIMIT_PX = 5.0


@dataclass(frozen=True)
class Selection:
    """选中的那一对，以及为什么选它（目标时刻与时间差都要能复查）。"""

    pair: object            # PairingResult.pairs 里的元素
    target_stamp_ns: int
    delta_ms: float

    def to_dict(self) -> dict:
        return {
            "pair_id": pair_id(self.pair.color.index, self.pair.depth.index),
            "color_index": self.pair.color.index,
            "depth_index": self.pair.depth.index,
            "color_stamp_ns": self.pair.color.stamp_ns,
            "depth_stamp_ns": self.pair.depth.stamp_ns,
            "dt_ms": round(self.pair.dt_ns / 1e6, 3),
            "target_stamp_ns": self.target_stamp_ns,
            "delta_ms": self.delta_ms,
        }


def pair_id(color_index: int, depth_index: int) -> str:
    """配对编号：`c<彩色序号>_d<深度序号>`。标注文件靠它认图，不能重复。"""
    return f"c{color_index}_d{depth_index}"


def pick_mid_pair(result: PairingResult, stable_duration_s: float) -> Selection | None:
    """取稳定段中点附近的一对。

    :param result: 配对结果（已排除预热）
    :param stable_duration_s: 声明的稳定段时长（秒），通常 10
    :return: 选中的 `Selection`；一对都没配上时返回 None
    """
    if not result.pairs:
        return None
    first_stamp = min(p.color.stamp_ns for p in result.pairs)
    target = first_stamp + int(stable_duration_s * 0.5 * 1e9)

    best = None
    for pair in result.pairs:
        delta = abs(pair.color.stamp_ns - target)
        key = (delta, pair.color.stamp_ns, pair.color.index)
        if best is None or key < best[0]:
            best = (key, pair, delta)
    _, pair, delta = best
    return Selection(pair=pair, target_stamp_ns=target, delta_ms=round(delta / 1e6, 3))
