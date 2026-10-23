"""逐图核对：一张图 -> 找码 -> 解位姿 -> 和同名真值 JSON 比，判定通过与否。

用大白话说
    `aruco_check.py` 管"码找得对不对"（ID、四角、中心）；这里管"算出来的距离和角度准不准"。
    给它一张图，它找到码、解出位姿，再和同名的 JSON 真值比距离与倾角，
    返回一条"真值 / 估计 / 误差 / 通过没通过"的记录。

放什么 / 不放什么
    放**判定门槛**（`DEFAULT_MAX_*` 三个、`TILT_JUDGE_MIN_DEG`）与单图核对 `check_one()`。
    门槛只在本文件定义一份，扫描模式（`sweep.py`）也从这里取——历史上这两处各写一份，
    改一个门槛要同步两个文件，是这条链最容易出错的地方。
    **不放**批量统计、CSV、出图（在 `report.py` 与 `sweep.py`），也不放参数解析（在 `cli.py`）。

依赖谁 / 被谁用
    `detectors.aruco`（找码）+ `pose`（解位姿）+ `image_io`（读图）。
    被 `cli.py`（逐图模式）与 `sweep.py`（复用同一套门槛）调用。

对外约定
    `check_one()` 返回 dict，键见函数文档；`ok=False` 时 `reason` 必非空。
    真值 JSON 由 `gen_pose_scenes.py` 生成；只有 `assets/aruco/*.json`（由 `gen_aruco` 生成）
    时也能跑：那种真值没有位姿字段，所以只核对"检出 + 重投影 RMS"。

已知的坑
    · **内参必须和图的分辨率匹配**。1280×1024 的内参套到 640×480 的图上，距离会整体偏，
      程序不报错。逐图模式在两者不一致时会按比例缩放并打警告（不中止：缩放回放是常见调试手段）。
    · 真值图里的 `marker_size_mm` 才是这批图的真实边长；命令行 `--marker-mm` 只用于
      "没有真值字段"的老图组。两者不一致时以真值为准。
"""

import json
import os

from vision.pose import DEFAULT_METHOD, estimate_pose
from vision.image_io import read_image_bgr

#: 判定"通过"的默认门槛。依据见 docs/位姿估计.md 的实测表：
#: 450–600 mm、40 mm 码时距离误差 < 1%、倾角误差 < 2°，留一倍余量。
DEFAULT_MAX_DIST_ERR = 0.04
DEFAULT_MAX_TILT_ERR_DEG = 3.0
DEFAULT_MAX_RMS_PX = 2.0

#: `--intrinsics` 与真值里的 `fx` 相对差超过这个比例就告警。
#: 5% 是"肉眼还看不出偏、但距离已经偏 5%"的量级；只告警不覆盖参数（见 cli 里的说明）。
K_MISMATCH_RATIO = 0.05

#: 真值倾角小于这个角度时**不判倾角**（误差照记，只是不算通过与否）。
#: 理由：码正对相机时，绕 x/y 轴转 10° 只让投影高度缩 1.5%——67 px 的码差 1 px，
#: 与角点定位噪声同量级，所以"正视时的倾角"在几何上就是弱可观的，实测误差可达 10°。
#: 拿 3° 门槛去卡它，测的是不可观测量，不是算法好坏。倾斜 15° 以上才判。
TILT_JUDGE_MIN_DEG = 10.0

#: 认哪些后缀是图片。逐图模式遍历目录时用；大小写敏感（Linux 上 .PNG 不会被扫到）。
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def truth_path_for(image_path: str) -> str:
    """同名 JSON 真值：`foo.png` → `foo.json`。"""
    return os.path.splitext(image_path)[0] + ".json"


def load_truth(image_path: str):
    """读真值 JSON；没有就返回 None（老图组只有 PNG 的情况）。

    :param image_path: 图片路径（真值取同名 .json）
    :return: dict 或 None
    :raises json.JSONDecodeError: 真值文件存在但不是合法 JSON（写坏了就该立刻停）
    """
    path = truth_path_for(image_path)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def yaw_delta(deg_a: float, deg_b: float) -> float:
    """角度差折算到 (-180, 180]，避免 179° 与 -179° 差出 358°。"""
    return ((deg_a - deg_b + 180.0) % 360.0) - 180.0


def check_one(image_path: str,
              detector,
              camera_matrix,
              dist_coeffs=None,
              marker_size_mm: float = 40.0,
              method: str = DEFAULT_METHOD,
              max_dist_err: float = DEFAULT_MAX_DIST_ERR,
              max_tilt_err_deg: float = DEFAULT_MAX_TILT_ERR_DEG,
              max_rms_px: float = DEFAULT_MAX_RMS_PX,
              tilt_judge_min_deg: float = TILT_JUDGE_MIN_DEG) -> dict:
    """核对一张图，返回一条可打印、可入 CSV 的记录。

    用大白话说
        打开图 → 找码 → 对每个码解位姿 → 和真值比距离与倾角。返回的 dict 里
        既有"估了多少"，也有"差多少"，外加一个 `ok` 供调用方决定退出码。

    :param image_path: 图片路径
    :param detector: 已建好的 `ArucoDetector`（复用能省掉每张图重复建字典的开销）
    :param camera_matrix: (3,3) 内参
    :param dist_coeffs: 畸变系数或 None
    :param marker_size_mm: 码边长（毫米）；真值里有 `marker_size_mm` 时以真值为准
    :param method: 位姿求解方法
    :param max_dist_err: 距离相对误差上限（比例）
    :param max_tilt_err_deg: 倾角绝对误差上限（度）
    :param max_rms_px: 重投影 RMS 上限（像素）
    :param tilt_judge_min_deg: 真值倾角小于它时不判倾角（理由见常量说明）
    :return: dict，键为 image / detected / id_ok / distance_mm / distance_err /
        tilt_deg / tilt_err_deg / yaw_err_deg / reproj_rms_px / ok / reason；
        `reason` 非空表示没通过
    :raises FileNotFoundError: 图片不存在（`read_image_bgr` 抛）
    """
    image_bgr = read_image_bgr(image_path)
    truth = load_truth(image_path)
    name = os.path.basename(image_path)
    record = {"image": name, "detected": 0, "id_ok": 0, "distance_mm": float("nan"),
              "distance_err": float("nan"), "tilt_deg": float("nan"),
              "tilt_err_deg": float("nan"), "yaw_deg": float("nan"),
              "reproj_rms_px": float("nan"), "ok": False, "reason": ""}

    if truth is not None:
        marker_size_mm = float(truth.get("marker_size_mm", marker_size_mm))

    markers = detector.detect(image_bgr)
    record["detected"] = len(markers)
    if not markers:
        record["reason"] = "未检出"
        return record

    want_id = None if truth is None else int(truth["marker_id"])
    target = next((m for m in markers if want_id is None or int(m.marker_id) == want_id), None)
    if target is None:
        record["reason"] = f"ID 不符（检出 {[int(m.marker_id) for m in markers]}，期望 {want_id}）"
        return record
    record["id_ok"] = 1

    try:
        pose = estimate_pose(target, marker_size_mm, camera_matrix, dist_coeffs,
                             method=method, max_reproj_rms_px=max_rms_px)
    except ValueError as exc:
        record["reason"] = f"解算失败：{exc}"
        return record

    record["distance_mm"] = pose.distance_mm
    record["tilt_deg"] = pose.tilt_deg
    record["yaw_deg"] = pose.yaw_deg
    record["reproj_rms_px"] = pose.reproj_rms_px

    if truth is None or "distance_mm" not in truth:
        # 老图组（gen_aruco 造的）没有位姿真值：只能验自洽，不能断言准不准
        record["ok"] = pose.reproj_rms_px <= max_rms_px
        record["reason"] = "" if record["ok"] else "重投影 RMS 超限"
        return record

    truth_distance = float(truth["distance_mm"])
    record["distance_err"] = abs(pose.distance_mm - truth_distance) / truth_distance
    record["tilt_err_deg"] = abs(pose.tilt_deg - float(truth["tilt_deg"]))
    yaw_err = abs(yaw_delta(pose.yaw_deg, float(truth["yaw_deg"])))

    problems = []
    if record["distance_err"] > max_dist_err:
        problems.append(f"距离误差 {record['distance_err'] * 100:.1f}% > {max_dist_err * 100:.0f}%")
    # 真值接近正视时不判倾角：那个自由度本身弱可观（见 TILT_JUDGE_MIN_DEG 的说明），
    # 误差仍然记进 record，报告里看得到，只是不计入"通过/不通过"
    if float(truth["tilt_deg"]) >= tilt_judge_min_deg and record["tilt_err_deg"] > max_tilt_err_deg:
        problems.append(f"倾角误差 {record['tilt_err_deg']:.1f}° > {max_tilt_err_deg:.0f}°")
    if pose.reproj_rms_px > max_rms_px:
        problems.append(f"RMS {pose.reproj_rms_px:.2f}px > {max_rms_px:.1f}px")
    # yaw 只报告不判定：码小、倾角大时 0.2 px 的角点噪声就能让面内角跳几度，
    # 把它做成一票否决会让"能用的图"被判成失败
    record["yaw_err_deg"] = yaw_err
    record["ok"] = not problems
    record["reason"] = "；".join(problems)
    return record
