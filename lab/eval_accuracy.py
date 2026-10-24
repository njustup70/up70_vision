"""M3 误差评估：批量跑合成场景，和真值配对后统计位置误差。

跑法（容器内）
    docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision \
        python lab/eval_accuracy.py data/syn/single_01 data/syn/multi_01

    # 只看统计不看每目标明细
    ... python lab/eval_accuracy.py data/syn/* --quiet

为什么不能直接「第 i 个检测对第 i 个真值」
    检测结果按可吸性分数排序，真值按摆放顺序，两者顺序没有任何关系。
    多目标场景里直接按下标配对，误差会算成几十上百毫米（实际是几毫米），
    而这种错配看起来就像「算法崩了」。必须先按空间距离做匹配再算误差。

匹配策略
    贪心最近邻：把所有「检测-真值」对按距离升序排，依次取用，超过阈值就停。
    简单、确定（同样输入永远同样结果，便于复现），场景里目标数很少时足够。
    更严谨的做法是匈牙利算法求全局最优，但这里不需要——目标之间通常隔着
    几个物块宽度，贪心不会配错。

指标口径
    位置误差：检测的吸附点与真值的顶面中心之间的三维欧氏距离（毫米）
    漏检    ：真值里存在、但没有检测匹配上的目标数
    误检    ：检测里存在、但没有真值匹配上的目标数

    M3 的验收线是**位置误差中位数 < 10mm**。漏检和误检不在验收线里，
    但它们决定了这个中位数是否可信——如果漏检一半，剩下那一半很准也没意义。

退出码：0 全部场景都有结果；1 有场景读不了或全都没检出。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.depth import Intrinsics, SuctionParams, detect_grasp_targets  # noqa: E402

#: 匹配阈值（毫米）。超过这个距离就认为「不是同一个物体」，
#: 只用来防止把漏检和误检硬凑成一对，不参与误差统计。
MATCH_MAX_MM = 30.0

#: M3 验收线
TARGET_MEDIAN_MM = 10.0


def load_scene(data_dir: Path) -> tuple:
    """读一个场景的深度图、内参与真值。

    真值取 meta.json 里每个物块的 top_center_cam_mm —— 那是相机系坐标，
    与检测输出同一坐标系，不需要再做变换。
    """
    depth_path = data_dir / "depth.npy"
    meta_path = data_dir / "meta.json"
    if not depth_path.exists():
        raise FileNotFoundError(f"{data_dir} 里没有 depth.npy")
    if not meta_path.exists():
        raise FileNotFoundError(f"{data_dir} 里没有 meta.json，无法取真值")

    depth = np.load(depth_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cam = meta["camera"]
    intrinsics = Intrinsics(fx=cam["fx"], fy=cam["fy"], cx=cam["cx"], cy=cam["cy"],
                            width=cam["width"], height=cam["height"])

    truths = [np.asarray(obj["top_center_cam_mm"], dtype=np.float64)
              for obj in meta.get("objects", [])]
    return depth, intrinsics, truths, meta


def match_targets(detections: list, truths: list, max_mm: float) -> tuple:
    """贪心最近邻配对，返回 (匹配列表, 漏检下标, 误检下标)。

    匹配列表每项是 (检测下标, 真值下标, 距离)。
    """
    if not detections or not truths:
        return [], list(range(len(truths))), list(range(len(detections)))

    pairs = []
    for i, det in enumerate(detections):
        for j, truth in enumerate(truths):
            pairs.append((float(np.linalg.norm(det.position_mm - truth)), i, j))
    pairs.sort()

    used_det, used_truth = set(), set()
    matches = []
    for distance, i, j in pairs:
        if distance > max_mm:
            break
        if i in used_det or j in used_truth:
            continue
        used_det.add(i)
        used_truth.add(j)
        matches.append((i, j, distance))

    missed = [j for j in range(len(truths)) if j not in used_truth]
    spurious = [i for i in range(len(detections)) if i not in used_det]
    return matches, missed, spurious


def evaluate_scene(data_dir: Path, params: SuctionParams, seed: int,
                   max_mm: float) -> dict:
    """跑一个场景并统计。"""
    depth, intrinsics, truths, meta = load_scene(data_dir)
    rng = np.random.default_rng(seed)
    detections = detect_grasp_targets(depth, intrinsics, params, rng)

    matches, missed, spurious = match_targets(detections, truths, max_mm)
    errors = [distance for _, _, distance in matches]

    return {
        "scene": data_dir.name,
        "truth_count": len(truths),
        "detected": len(detections),
        "matched": len(matches),
        "missed": len(missed),
        "spurious": len(spurious),
        "errors": errors,
        "scores": [det.score for det in detections],
        "tilt_deg": meta["camera"].get("tilt_deg"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="M3 误差评估：批量跑合成场景，统计位置误差",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("scenes", nargs="+", type=Path, help="场景目录（含 depth.npy + meta.json）")
    parser.add_argument("--match-mm", type=float, default=MATCH_MAX_MM,
                        help="配对距离上限（毫米），超过就算两个不同物体")
    parser.add_argument("--seed", type=int, default=0, help="RANSAC 随机种子")
    parser.add_argument("--quiet", action="store_true", help="只打汇总，不打每场景明细")
    args = parser.parse_args()

    params = SuctionParams()
    results = []
    failed = 0

    print("=" * 68)
    print("M3 合成数据误差评估")
    print("=" * 68)

    for scene in args.scenes:
        try:
            result = evaluate_scene(scene, params, args.seed, args.match_mm)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"[跳过] {scene}：{exc}")
            failed += 1
            continue

        results.append(result)
        if not args.quiet:
            print(f"\n{result['scene']}  (倾角 {result['tilt_deg']}°)")
            print(f"  真值 {result['truth_count']} | 检出 {result['detected']} | "
                  f"匹配 {result['matched']} | 漏检 {result['missed']} | "
                  f"误检 {result['spurious']}")
            if result["errors"]:
                formatted = ", ".join(f"{value:.2f}" for value in result["errors"])
                print(f"  位置误差(mm): {formatted}")
            else:
                print("  位置误差(mm): 无匹配目标")

    if not results:
        print("\n没有任何场景评估成功。")
        return 1

    all_errors = [value for result in results for value in result["errors"]]
    total_truth = sum(result["truth_count"] for result in results)
    total_matched = sum(result["matched"] for result in results)
    total_spurious = sum(result["spurious"] for result in results)

    print("\n" + "-" * 68)
    print("汇总")
    print("-" * 68)
    print(f"  场景数        : {len(results)}")
    print(f"  真值总数      : {total_truth}")
    print(f"  匹配成功      : {total_matched}"
          f"（漏检 {total_truth - total_matched}，"
          f"漏检率 {(1 - total_matched / max(total_truth, 1)) * 100:.1f}%）")
    print(f"  误检          : {total_spurious}")

    if not all_errors:
        print("  位置误差      : 无数据")
        return 1

    errors = np.asarray(all_errors)
    median = float(np.median(errors))
    p95 = float(np.percentile(errors, 95))
    print(f"  位置误差中位数: {median:.2f} mm（M3 目标 < {TARGET_MEDIAN_MM} mm）")
    print(f"  位置误差 P95  : {p95:.2f} mm")
    print(f"  最大误差      : {float(errors.max()):.2f} mm")

    ok = median < TARGET_MEDIAN_MM and total_matched > 0
    print("-" * 68)
    if ok:
        print("结果：M3 位置精度达标。")
        return 0
    print("结果：未达标，先看漏检与误检——中位数再准，样本量不够也不能用。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
