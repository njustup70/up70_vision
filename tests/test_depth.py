"""深度模块回归：用人工合成的深度图验证 B 阶段链路，不依赖任何外部数据。

跑法（容器内，或任何装了 numpy + opencv 的环境）
    python tests/test_depth.py

为什么自己造图而不是读 data/syn
    回归测试不能依赖「某次生成的数据还在」。这里直接在内存里造一个桌面 + 方块的
    深度图，几行代码就能覆盖：单目标、多目标、空场景、输入校验。

造图用的是「正对相机」的简化场景（没有安装倾角），因为这一层要验的是
算法链路的正确性，安装角度的影响属于标定范畴。带倾角的真实几何由
lab/gen_synthetic_depth.py 负责，那个有独立的自检。

退出码：全部通过为 0，有失败为 1。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from vision.depth import (  # noqa: E402
    GraspTarget,
    Intrinsics,
    SuctionParams,
    analyze,
    detect_grasp_targets,
)

WIDTH, HEIGHT = 848, 480
FX = FY = 409.5
CX, CY = 424.33, 241.33

_failures = []


def _ok(message: str) -> None:
    print(f"[通过] {message}")


def _fail(message: str) -> None:
    print(f"[失败] {message}")
    _failures.append(message)


def _expect_error(fn, message: str) -> None:
    """验证某段代码确实抛 ValueError。不抛比抛错更危险，必须显式测。"""
    try:
        fn()
    except ValueError:
        _ok(message)
    except Exception as exc:                       # noqa: BLE001
        _fail(f"{message}：抛的不是 ValueError 而是 {type(exc).__name__}: {exc}")
    else:
        _fail(f"{message}：没报错")


def make_scene(blocks, table_z=600.0, gap_z=None):
    """造一张深度图：整张桌面 + 若干凸起方块。

    :param blocks: [(center_u, center_v, size_mm, height_mm), ...] 像素质心与物理尺寸
    :param table_z: 桌面深度（毫米）
    画面全部有效，不造空洞——空洞单独在可吸性测试里验。
    """
    depth = np.full((HEIGHT, WIDTH), table_z, dtype=np.float32)
    for center_u, center_v, size_mm, height_mm in blocks:
        half = int(round(size_mm * FX / table_z / 2))
        u0, u1 = int(center_u) - half, int(center_u) + half
        v0, v1 = int(center_v) - half, int(center_v) + half
        depth[v0:v1, u0:u1] = table_z - height_mm
    return np.round(depth).astype(np.uint16)


def check_types() -> None:
    """类型层校验：坏输入必须当场拒绝，而不是流到算法里变成怪结果。"""
    _expect_error(lambda: Intrinsics(fx=0.0, fy=1.0, cx=0.0, cy=0.0, width=10, height=10),
                  "焦距为 0 时 Intrinsics 拒绝构造")
    _expect_error(lambda: Intrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=0, height=10),
                  "图像宽为 0 时 Intrinsics 拒绝构造")

    k = Intrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=10, height=10)
    _expect_error(lambda: GraspTarget(position_mm=[0, 0, 0], normal=[0, 0, 0],
                                      score=1.0, bbox_size_mm=[1, 1, 1], points=1),
                  "零向量法向被拒绝")
    _expect_error(lambda: GraspTarget(position_mm=[0, 0, 0], normal=[0, 0, 1],
                                      score=1.5, bbox_size_mm=[1, 1, 1], points=1),
                  "score 越界被拒绝")
    _expect_error(lambda: GraspTarget(position_mm=[0, 0], normal=[0, 0, 1],
                                      score=1.0, bbox_size_mm=[1, 1, 1], points=1),
                  "位置形状不对被拒绝")

    target = GraspTarget(position_mm=[0, 0, 500], normal=[0, 0, -2],
                         score=0.5, bbox_size_mm=[1, 1, 1], points=1)
    if abs(np.linalg.norm(target.normal) - 1.0) < 1e-9:
        _ok(f"法向自动归一化（{target.normal.tolist()}）")
    else:
        _fail("法向没有归一化")
    if k.fx == 1.0:
        _ok("Intrinsics 正常构造")


def check_input_validation() -> None:
    """尺寸与内参不一致时必须抛错——那意味着内参和图像不是同一次配置。"""
    k = Intrinsics(fx=FX, fy=FY, cx=CX, cy=CY, width=WIDTH, height=HEIGHT)
    depth = make_scene([(CX, CY, 50.0, 50.0)])
    _expect_error(lambda: detect_grasp_targets(depth[:200, :200], k),
                  "尺寸与内参不符时拒绝计算")
    _expect_error(lambda: detect_grasp_targets(depth[0], k),
                  "一维输入被拒绝")


def check_single_target() -> None:
    """单目标：应该只找到一个簇，吸附点落在顶面中心，分数高。"""
    depth = make_scene([(CX, CY, 50.0, 50.0)], table_z=600.0)
    k = Intrinsics(fx=FX, fy=FY, cx=CX, cy=CY, width=WIDTH, height=HEIGHT)

    result = analyze(depth, k)
    if result["reason"]:
        _fail(f"单目标分析失败：{result['reason']}")
        return
    if len(result["clusters"]) == 1:
        _ok("单目标场景聚类数 = 1")
    else:
        _fail(f"单目标场景聚类数 = {len(result['clusters'])}，应为 1")
        return

    target = result["targets"][0]
    expected = np.array([0.0, 0.0, 550.0])       # 物块正对相机中心，顶面比桌面高 50mm
    error = float(np.linalg.norm(target.position_mm - expected))
    if error < 5.0:
        _ok(f"吸附点与理论位置误差 {error:.2f}mm（< 5mm）")
    else:
        _fail(f"吸附点误差 {error:.2f}mm 超过 5mm，位置 {target.position_mm.tolist()}")

    if target.score > 0.9:
        _ok(f"可吸性分数 {target.score}（> 0.9）")
    else:
        _fail(f"可吸性分数只有 {target.score}，限制项 "
              f"{result['clusters'][0]['score_detail'].get('limiting')}")

    if float(target.normal @ np.array([0.0, 0.0, -1.0])) > 0.99:
        _ok(f"法向朝相机侧（{target.normal.round(3).tolist()}）")
    else:
        _fail(f"法向方向不对：{target.normal.round(3).tolist()}")


def check_multi_target() -> None:
    """多目标：两个分开的方块应该各成一个簇，按分数降序返回。"""
    depth = make_scene([(300.0, 240.0, 50.0, 50.0), (560.0, 240.0, 45.0, 40.0)])
    k = Intrinsics(fx=FX, fy=FY, cx=CX, cy=CY, width=WIDTH, height=HEIGHT)

    targets = detect_grasp_targets(depth, k)
    if len(targets) == 2:
        _ok("多目标场景找到 2 个目标")
    else:
        _fail(f"多目标场景找到 {len(targets)} 个目标，应为 2")
        return
    if targets[0].score >= targets[1].score:
        _ok(f"按可吸性降序（{targets[0].score} >= {targets[1].score}）")
    else:
        _fail("结果没有按可吸性降序排列")


def check_empty_scene() -> None:
    """纯桌面：没有凸起物就不该输出任何目标。"""
    depth = make_scene([])
    k = Intrinsics(fx=FX, fy=FY, cx=CX, cy=CY, width=WIDTH, height=HEIGHT)

    targets = detect_grasp_targets(depth, k)
    if not targets:
        _ok("空场景返回空列表（不崩溃、不硬凑目标）")
    else:
        _fail(f"空场景却返回了 {len(targets)} 个目标")


def check_small_block_not_suckable() -> None:
    """小物块：比吸盘还小，分数应该明显低于正常物块。

    这里把簇点数下限放宽到 30：12mm 物块在 600mm 处只有约 8 像素宽，
    切掉一圈边界像素后剩三十来点，用默认的 300 会连簇都不存在，
    测不到「点数不够所以吸不住」这个判据本身。
    """
    params = SuctionParams(min_cluster_points=30)
    k = Intrinsics(fx=FX, fy=FY, cx=CX, cy=CY, width=WIDTH, height=HEIGHT)

    normal = detect_grasp_targets(make_scene([(CX, CY, 50.0, 50.0)]), k, params)
    small = detect_grasp_targets(make_scene([(CX, CY, 12.0, 12.0)]), k, params)

    if not normal or not small:
        _fail(f"对比测试拿不到目标（正常 {len(normal)}，小物块 {len(small)}）")
        return
    if normal[0].score - small[0].score > 0.3:
        _ok(f"小物块分数明显更低（{normal[0].score} vs {small[0].score}）")
    else:
        _fail(f"小物块分数没拉开差距（{normal[0].score} vs {small[0].score}）")


def check_holes_lower_score() -> None:
    """圆盘内挖掉一半像素（模拟反光丢深度），可吸性应该下降。"""
    depth = make_scene([(CX, CY, 50.0, 50.0)])
    holed = depth.copy()
    center_v, center_u = int(CY), int(CX)
    holed[center_v - 10:center_v + 10, center_u - 10:center_u + 10] = 0

    k = Intrinsics(fx=FX, fy=FY, cx=CX, cy=CY, width=WIDTH, height=HEIGHT)
    plain = detect_grasp_targets(depth, k)
    broken = detect_grasp_targets(holed, k)

    if not plain or not broken:
        _fail(f"空洞测试拿不到目标（完整 {len(plain)}，空洞 {len(broken)}）")
        return
    if broken[0].score < plain[0].score:
        _ok(f"空洞降低可吸性（{plain[0].score} -> {broken[0].score}）")
    else:
        _fail(f"空洞没有降低分数（{plain[0].score} vs {broken[0].score}）")


def main() -> int:
    print("=" * 60)
    print("vision.depth 回归测试")
    print("=" * 60)

    check_types()
    check_input_validation()
    check_single_target()
    check_multi_target()
    check_empty_scene()
    check_small_block_not_suckable()
    check_holes_lower_score()

    print("-" * 60)
    if _failures:
        print(f"结果：{len(_failures)} 项未通过")
        for item in _failures:
            print(f"  - {item}")
        return 1
    print("结果：全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
