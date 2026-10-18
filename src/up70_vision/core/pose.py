"""ArUco 位姿估计：从四个角点解出码相对相机的三维位置与姿态。

用大白话说
    相机看到纸上的一个方块码，除了知道"这是 7 号码、它落在第 320 列"，还能算出
    "它在我正前方 500 毫米、斜了 30 度、在画面里转了 12 度"。做法分三步：
        1. 码的四个角在纸上的真实位置是已知的（码边长量过），把它们写成四个三维点；
        2. 这四个点在图里落在哪四个像素，检测器已经给了（`ArucoMarker.corners_px`）；
        3. 把「四个三维点 ↔ 四个像素点」和相机内参一起解一个几何方程（PnP），
           得到的解就是码中心相对相机的位置（毫米）与朝向（旋转）。
    整个文件只做这件事，不读文件、不开窗口、不画图。

放什么 / 不放什么
    放纯算法：给 numpy 角点 + 边长 + 内参 → 返回 `MarkerPose`。
    **不读文件、不保存文件、不开窗口、不 import io/viz/config**（契约第 5 节）。
    读内参 YAML 在 `config/loader.py`，画坐标轴在 `viz/draw.py`，打印在 `cli.py`，
    批量核误差在 `tools/pose_check.py`。

依赖谁 / 被谁用
    只依赖 numpy 与 OpenCV（`cv2.solvePnP*`）。上层是 `viz/`（画坐标轴）、
    `cli.py`（实时终端行）、`tools/pose_check.py`（批量）、`tools/pose_sweep.py`（精度扫描）。

对外约定
    · 空输入：`estimate_poses([])` 返回 `[]`，不抛异常（与检测器的"无码返回 []"对齐）
    · 角点形状必须 (4,2)、边长必须 > 0、内参必须 (3,3)，否则抛 `ValueError`
    · 解出的重投影 RMS 超过 `max_reproj_rms_px` 抛 `ValueError`（理由见"已知的坑"第一条）
    · 解出的 `tvec_mm[2] <= 0` 抛 `ValueError`（可见的码不可能在相机后方）
    · 单位：边长进、距离出，一律**毫米**；不做 mm/m 的隐式换算

已知的坑（都会"不报错，只让数字错"）
    · **模型点顺序写错 → 垃圾解**：`SOLVEPNP_IPPE_SQUARE` 要求四个模型点按固定顺序给，
      顺序错了实测 RMS 993 px（真值 160.8 px 的下 0.02 px），程序不报错。
      所以 `marker_object_points()` 是全仓唯一实现处，不要在别处再拼一份四个点。
    · **二次去畸变**：角点若已用 `undistort_points()` 掰直，这里必须传 `dist_coeffs=None`；
      再传畸变系数等于去畸变两次（实测 k1=-0.30 时距离偏 0.62%）。
    · **平面位姿天然两解**：两解的倾角相同、距离只差 0.1–0.5%，旋转夹角恰为 `2×tilt`。
      `ambiguous` 只是"两解难以区分"的诊断位，不代表结果不能用——位置比姿态稳。
    · **180° 角点错位**：四个角点整体循环平移两位（等价于码转 180°），重投影误差可以是
      0.0000 px，距离和倾角都"看着对"，只有 `tvec_mm[2]` 的符号能抓到（实测为负）。
"""

from __future__ import annotations

import numpy as np

from .types import ArucoMarker, MarkerPose

#: 可选的求解方法（configs/default.yaml 的 pose.method 只能填这些）
#:   ippe_square：专为「正方形、共面、四个点」设计，闭式解、不要初值；能给出两个解
#:   iterative  ：OpenCV 的通用迭代解，需要初值；只在排查问题时用
METHOD_NAMES = ("ippe_square", "iterative")

#: 默认求解方法。IPPE_SQUARE 是 ArUco 这类正方形码的专用解法，也正好是弃用的
#: `estimatePoseSingleMarkers` 内部用的那个（任务单坑 3）。
DEFAULT_METHOD = "ippe_square"

#: 重投影 RMS 上限（像素）。超过它基本只有三种可能：模型点顺序错、角点被交换、
#: 内参/分辨率不匹配。实测正常解算在 0.02–0.9 px，异常情况会到几十到几千 px，
#: 所以 2.0 这个门槛离噪声很远、离故障很近。
DEFAULT_MAX_REPROJ_RMS_PX = 2.0

#: 判定"两解难以区分"的相对误差比：`rms_次优 / rms_最优 < 3` 才算接近。
#: 用比例而不用绝对像素，是因为绝对差随距离变小（实测同一倾角下 500 mm 处 0.68 px、
#: 1000 mm 处 0.34 px），固定阈值会把远处的码误标成 ambiguous。
AMBIGUITY_RATIO = 3.0

#: 判定"两解难以区分"的旋转夹角（度）。两解夹角等于 `2×倾角`，所以正视角附近
#: 它们本来就重合（夹角 0），此时不该报 ambiguous。
AMBIGUITY_DEG = 10.0

#: 数值比较的零值保护：RMS 可以是 0.0（理想合成数据），除法前统一垫一层
_EPS = 1e-6


def marker_object_points(marker_size_mm: float) -> np.ndarray:
    """码的四个角在**码自身坐标系**里的坐标（毫米）。

    用大白话说
        把码当成一张贴在纸上的正方形，中心是原点，边长是 `marker_size_mm`。
        四个角就这么四个坐标，顺序必须和图像里检测到的四个角一一对应。

    点的顺序（IPPE_SQUARE 的硬性要求，写错不报错、只给垃圾解）
        ```
        (-s/2, +s/2) TL ─────── TR (+s/2, +s/2)      s = marker_size_mm
              │       ↑       │                      第 1 个点对应图里左上角
              │     中心      │                      顺序 TL -> TR -> BR -> BL
        (-s/2, -s/2) BL ─────── BR (+s/2, -s/2)
        ```
        注意 y 轴的正方向指向码的**上边**（不是图像里的向下）。这不是笔误：
        `cv2.SOLVEPNP_IPPE_SQUARE` 内部按这个约定解，换一套坐标会得到"翻转"的位姿。
        按这个约定解出来的码坐标系是：x 轴指向码的右边、y 轴指向上边、
        z 轴垂直纸面向外（**指向相机**）——正视时实测 `R = diag(1, -1, -1)`。

    :param marker_size_mm: 码的边长（毫米），量**黑框外沿到外沿**，不含白边
    :return: (4, 3) float64，四行依次是 TL、TR、BR、BL
    :raises ValueError: 边长不是正数（负边长会让解出的距离反向，且不报错）
    """
    side = float(marker_size_mm)
    if not np.isfinite(side) or side <= 0:
        raise ValueError(f"marker_size_mm 必须是正数（毫米），收到 {marker_size_mm!r}")
    half = side / 2.0
    return np.array([[-half, +half, 0.0],
                     [+half, +half, 0.0],
                     [+half, -half, 0.0],
                     [-half, -half, 0.0]], dtype=np.float64)


def _check_camera_matrix(camera_matrix) -> np.ndarray:
    """把内参规整成 (3,3) float64；形状不对就报错（不会自动补齐）。"""
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix 必须是 3x3，收到 {matrix.shape}")
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError(f"camera_matrix 的 fx/fy 必须为正且有限，收到 {matrix.tolist()}")
    return matrix


def _check_corners(corners_px) -> np.ndarray:
    """把角点规整成 (4,2) float64；形状不对就报错。"""
    corners = np.asarray(corners_px, dtype=np.float64)
    if corners.size != 8:
        raise ValueError(f"corners_px 必须是 4 个点（8 个数），收到形状 {corners.shape}")
    corners = corners.reshape(4, 2)
    if not np.isfinite(corners).all():
        raise ValueError(f"corners_px 含 NaN / Inf：{corners.tolist()}")
    return corners


def _reproject_rms(object_points: np.ndarray, corners: np.ndarray, rvec: np.ndarray,
                   tvec: np.ndarray, camera_matrix: np.ndarray, dist_coeffs) -> float:
    """把四个模型点按当前位姿投回图像，与实测角点求 RMS（像素）。

    为什么自己算而不用 `solvePnP` 返回的误差
        `solvePnPGeneric` 也会给一组误差，但含义（是否已平方、是否按点平均）在不同
        OpenCV 版本上要现查文档；自己投一遍只有四行，且与 `MarkerPose.reproj_rms_px`
        的定义天然一致——这个数是要写进报告和断言的，口径必须是我们自己说了算。
    """
    projected, _ = _project_points(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    # 必须先 reshape：cv2.projectPoints 返回 (N,1,2)，直接与 (N,2) 相减会被广播成
    # (N,N,2)——不报错，只是误差从 0.001 px 变成 90 px，看起来像"算法坏了"。
    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    diff = corners.reshape(-1, 2) - projected
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _project_points(object_points, rvec, tvec, camera_matrix, dist_coeffs):
    """`cv2.projectPoints` 的薄封装：只在本模块内用，集中处理 dist 为空的情况。

    单独包一层是为了让"dist_coeffs=None 也要能投"这件事只有一个实现处：
    OpenCV 接受 None，但传 `np.zeros((1,5))` 与 None 在内部走的不是同一条分支。
    """
    import cv2
    return cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)


def _rotation_angle_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    """两个旋转之间的夹角（度）。用旋转矩阵的迹算，避免直接比 rvec 的分量。

    公式：`cos θ = (trace(Ra^T · Rb) − 1) / 2`。
    直接比 rvec 三个分量是错的——旋转向量不是欧氏空间里的向量，
    `(0,0,π)` 与 `(0,0,−π)` 表示同一个旋转，分量差却是 2π。
    """
    import cv2
    matrix_a, _ = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64).reshape(3, 1))
    matrix_b, _ = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64).reshape(3, 1))
    cosine = (np.trace(matrix_a.T @ matrix_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def pose_metrics(rvec, tvec_mm) -> tuple[float, float, float]:
    """由旋转向量与平移向量算出三个可读量：距离、倾角、面内转角。

    用大白话说
        `rvec`/`tvec_mm` 是给机器用的，人和报告要的是"多远、斜多少、转多少"。
        三个量都在这里算，**全仓只此一处**——渲染器造真值、求解器出结果、
        报告算误差，用的必须是同一套口径，否则误差表里会混进"两套公式之差"。

    公式
        distance = ||t||                     （斜距，不是 t[2]）
        tilt     = acos(-R[2,2])             （码面法向指向相机，正视时为 0）
        yaw      = atan2(R[1,0], R[0,0])     （码 x 轴在图像里的方向角，顺时针为正）

    :param rvec: (3,) 或 (3,1) 旋转向量
    :param tvec_mm: (3,) 或 (3,1) 平移向量，毫米
    :return: `(distance_mm, tilt_deg, yaw_deg)`
    """
    import cv2
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    tvec = np.asarray(tvec_mm, dtype=np.float64).reshape(3)
    distance = float(np.linalg.norm(tvec))
    # 负号的理由见 solve_pose 里的同一段注释：正视时 R[2,2] = -1（实测）
    tilt = float(np.degrees(np.arccos(np.clip(-rotation[2, 2], -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return distance, tilt, yaw


def solve_pose(corners_px: np.ndarray,
               marker_size_mm: float,
               camera_matrix,
               dist_coeffs=None,
               marker_id: int = -1,
               method: str = DEFAULT_METHOD,
               refine: bool = True,
               max_reproj_rms_px: float = DEFAULT_MAX_REPROJ_RMS_PX) -> MarkerPose:
    """从四个角点解出一个码的位姿（本模块的唯一求解入口）。

    用大白话说
        已知"码四个角在纸上的坐标"和"它们在画面里的像素位置"，再知道相机内参，
        就能反推出码离相机多远、朝哪边。这里顺便挑掉明显不对的解：
        投回图像差得太多、或者解到相机后方的，一律当失败抛出来。

    :param corners_px: (4,2)，图像像素坐标，顺序 TL→TR→BR→BL；**若已去过畸变，
        这里必须是掰直后的坐标，且 `dist_coeffs` 要传 None**
    :param marker_size_mm: 码边长（毫米），黑框外沿到外沿
    :param camera_matrix: (3,3) 内参矩阵；用了 `undistort_points(..., P=K)` 时传原 K
    :param dist_coeffs: 畸变系数（1,5）或 None。**角点已掰直时必须传 None**
    :param marker_id: 写进返回值的编号；不知道就留 -1（不影响解算）
    :param method: `ippe_square`（默认）或 `iterative`
    :param refine: True 时用 `solvePnPRefineLM` 再精化一次，只有 RMS 更小才采用
    :param max_reproj_rms_px: 重投影 RMS 上限；超过就抛 ValueError
    :return: `MarkerPose`
    :raises ValueError: 入参形状/取值非法、解算失败、RMS 超限、解到相机后方

    单帧的处理顺序（顺序不能换）
        ┌─ 1. 校验入参（形状、正负、有限）─────────────────────────┐
        │  2. 生成四个模型点（顺序 TL→TR→BR→BL）                    │
        │  3. solvePnPGeneric 求全部解（IPPE_SQUARE 恒给 2 个）      │
        │  4. 每个解各自算重投影 RMS，挑 RMS 最小且 z>0 的           │
        │  5. 可选 LM 精化，只接受"更小"的那个                       │
        │  6. 硬校验：RMS 上限、tvec[2]>0                            │
        │  7. 算派生量（距离/倾角/yaw）与 ambiguous                  │
        └───────────────────────────────────────────────────────────┘
    """
    import cv2

    corners = _check_corners(corners_px)
    matrix = _check_camera_matrix(camera_matrix)
    object_points = marker_object_points(marker_size_mm)

    method = str(method).strip().lower()
    if method not in METHOD_NAMES:
        raise ValueError(f"pose.method 必须是 {list(METHOD_NAMES)} 之一，收到 {method!r}")
    limit = float(max_reproj_rms_px)
    if limit <= 0:
        raise ValueError(f"max_reproj_rms_px 必须为正，收到 {max_reproj_rms_px!r}")

    # ---- 步骤 1：求解（IPPE_SQUARE 走 generic，能一次拿到两个候选解）----
    if method == "ippe_square":
        count, rvecs, tvecs, _errs = cv2.solvePnPGeneric(
            object_points, corners, matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        candidates = [(np.asarray(rvecs[i], dtype=np.float64).reshape(3),
                       np.asarray(tvecs[i], dtype=np.float64).reshape(3)) for i in range(count)]
    else:
        ok, rvec, tvec = cv2.solvePnP(object_points, corners, matrix, dist_coeffs,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        candidates = [(np.asarray(rvec, dtype=np.float64).reshape(3),
                       np.asarray(tvec, dtype=np.float64).reshape(3))] if ok else []

    # ---- 步骤 2：逐个候选算 RMS；z<=0 的解直接扔（码不可能在相机后方）----
    scored = []
    for rvec, tvec in candidates:
        if tvec[2] <= 0:                       # 翻到相机后面的解，一定是错的
            continue
        rms = _reproject_rms(object_points, corners, rvec, tvec, matrix, dist_coeffs)
        scored.append((rms, rvec, tvec))
    if not scored:
        raise ValueError(
            f"位姿解算失败（id={marker_id}）：没有落在相机前方的解，"
            "多半是角点顺序与模型点顺序对不上")

    scored.sort(key=lambda item: item[0])
    rms, rvec, tvec = scored[0]

    # ---- 步骤 3：可选精化。只接受"RMS 更小"，否则保留原解 ----
    # 精化是数值优化，可能把解带偏；用 RMS 做裁判才不会越优化越差。
    if refine:
        refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
            object_points, corners, matrix, dist_coeffs,
            rvec.reshape(3, 1).copy(), tvec.reshape(3, 1).copy())
        refined_rvec = np.asarray(refined_rvec, dtype=np.float64).reshape(3)
        refined_tvec = np.asarray(refined_tvec, dtype=np.float64).reshape(3)
        if refined_tvec[2] > 0:
            refined_rms = _reproject_rms(object_points, corners, refined_rvec, refined_tvec,
                                         matrix, dist_coeffs)
            if refined_rms < rms:
                rms, rvec, tvec = refined_rms, refined_rvec, refined_tvec

    # ---- 步骤 4：硬校验。这两条是"看起来正常其实全错"的唯一出口 ----
    if rms > limit:
        raise ValueError(
            f"重投影误差过大（id={marker_id}）：RMS={rms:.3f} px 超过上限 {limit:.3f} px。"
            "常见原因：模型点顺序与角点不匹配、角点被交换、内参与图像分辨率不匹配")
    if tvec[2] <= 0:
        raise ValueError(f"位姿解算失败（id={marker_id}）：tvec[2]={tvec[2]:.3f} mm 不为正")

    # ---- 步骤 5：派生量。三个量统一由 pose_metrics 算 ----
    # 渲染器造真值、报告算误差都调它，误差表里才不会混进"两套公式之差"。
    _distance, tilt, yaw = pose_metrics(rvec, tvec)

    # 二义性：两个解都"足够像"时才标记。比例判据 + 夹角判据同时成立才算，
    # 因为正视时两解本来重合（夹角≈0），不该被标成 ambiguous。
    ambiguous = False
    if len(scored) > 1:
        second_rms, second_rvec, _second_tvec = scored[1]
        angle = _rotation_angle_deg(rvec, second_rvec)
        ambiguous = (second_rms / max(rms, _EPS) < AMBIGUITY_RATIO) and (angle > AMBIGUITY_DEG)

    return MarkerPose(marker_id=int(marker_id), rvec=rvec, tvec_mm=tvec,
                      distance_mm=float(np.linalg.norm(tvec)),
                      tilt_deg=tilt, yaw_deg=yaw,
                      reproj_rms_px=rms, ambiguous=ambiguous)


def estimate_pose(marker: ArucoMarker,
                  marker_size_mm: float,
                  camera_matrix,
                  dist_coeffs=None,
                  method: str = DEFAULT_METHOD,
                  refine: bool = True,
                  max_reproj_rms_px: float = DEFAULT_MAX_REPROJ_RMS_PX) -> MarkerPose:
    """对一个已检测到的码求位姿（`solve_pose` 的 `ArucoMarker` 版本）。

    用大白话说
        检测器给的是"码在图像里长什么样"，这里把它换算成"码在空间里的位置"。
        它只是把 `ArucoMarker` 的三个字段拆开传给 `solve_pose`，自己不做几何计算。

    :param marker: `ArucoMarker`（检测器产出，四角已是亚像素 float32）
    :param marker_size_mm: 码边长（毫米）
    :param camera_matrix: (3,3) 内参
    :param dist_coeffs: 畸变系数或 None；角点若已掰直必须传 None
    :param method: 见 `solve_pose`
    :param refine: 是否做 LM 精化
    :param max_reproj_rms_px: 重投影 RMS 上限
    :return: `MarkerPose`，`marker_id` 取自入参
    :raises ValueError: 见 `solve_pose`
    """
    if not isinstance(marker, ArucoMarker):
        raise ValueError(f"estimate_pose 需要 ArucoMarker，收到 {type(marker).__name__}")
    return solve_pose(marker.corners_px, marker_size_mm, camera_matrix, dist_coeffs,
                      marker_id=marker.marker_id, method=method, refine=refine,
                      max_reproj_rms_px=max_reproj_rms_px)


def estimate_poses(markers: list,
                   marker_size_mm: float,
                   camera_matrix,
                   dist_coeffs=None,
                   method: str = DEFAULT_METHOD,
                   refine: bool = True,
                   max_reproj_rms_px: float = DEFAULT_MAX_REPROJ_RMS_PX) -> list:
    """批量求位姿，输出顺序与输入一致。

    用大白话说
        一帧里有几个码就解几次，返回一条一条的位姿。没码就返回空列表。

    注意它**不排序**：排序是检测器的契约（按 `marker_id` 升序），这里再排一次只会
    让"谁调用谁负责"变模糊，也容易在两处写出不同的排序键。

    :param markers: `list[ArucoMarker]`；空列表返回空列表
    :return: `list[MarkerPose]`，顺序与入参一一对应
    :raises ValueError: 任何一个码解算失败都会抛——批量场景要"跳过坏样本"时，
        调用方自己逐个调 `estimate_pose` 并 try/except，别指望这里静默丢掉
        （静默跳过会让"10 个码只出了 8 个"看起来像正常）
    """
    return [estimate_pose(marker, marker_size_mm, camera_matrix, dist_coeffs,
                          method=method, refine=refine,
                          max_reproj_rms_px=max_reproj_rms_px)
            for marker in markers]
