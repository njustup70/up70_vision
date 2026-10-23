"""ArUco 自己的数据类型：`ArucoMarker` 与 `MarkerPose`（D2 功能竖切）。

根级 `types.py` 只放跨功能共用件（`Frame` / `Detection` / 几何函数）；
这两个类只有 ArUco 用（检测、位姿、绘制），所以住在 `aruco/` 里。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArucoMarker:
    """一个 ArUco 码（契约见 docs/interface.md 第 2.2 节）。

    为什么不复用 `Detection`
        `Detection` 的核心是轴对齐的 `bbox_xyxy`，它的"中心"是 bbox 中心；
        而 ArUco 的本质是四角点构成的任意四边形，中心是**四点算术平均**。
        码一倾斜，两者就不是同一个位置——所以这是两种结构，不要混用。
        码也没有颜色，故这里没有 color/color_ratio 字段。

    四条必须一致的约定（stub 与真实现最容易分歧的四处）
        1. 无码时检测器返回 `[]`，不返回 `None`、不返回 `[None]`
        2. 按 `marker_id` **升序**排列（不排序时终端输出会抖动）
        3. `center_px` 是四角**算术平均**，不是 bbox 中心
        4. `corners_px` 顺时针、起点是「码自身坐标系」的左上角（直接沿用 OpenCV 原生顺序）

    `corners_px` 存 float32 而不是 int：它是阶段二 `solvePnP` 的输入，
    这里取整等于提前丢掉亚像素精度且不可逆。取整只发生在绘制那一刻。
    """

    marker_id: int
    corners_px: np.ndarray                       # (4, 2) float32，像素坐标
    center_px: tuple                             # (float, float)，像素坐标

    def __post_init__(self):
        if not isinstance(self.marker_id, int) or isinstance(self.marker_id, bool):
            raise ValueError(f"marker_id 必须是 int，收到 {type(self.marker_id).__name__}")
        corners = np.asarray(self.corners_px, dtype=np.float32)   # 归一成契约要求的 float32
        if corners.shape != (4, 2):
            raise ValueError(f"corners_px 必须是 (4, 2)，收到 {corners.shape}")
        center = tuple(float(v) for v in self.center_px)
        if len(center) != 2:
            raise ValueError(f"center_px 必须是 (x, y)，收到 {self.center_px!r}")
        # frozen dataclass 里改字段得走 object.__setattr__
        object.__setattr__(self, "corners_px", corners)
        object.__setattr__(self, "center_px", center)

    @property
    def bbox_xyxy(self) -> BBox:
        """四角的外接框（半开上界）。判色 ROI、IoU、落盘都用它。

        名字与 `Detection.bbox_xyxy` 一致，是为了让消费方能统一写 `det.bbox_xyxy`；
        注意它是**派生**出来的，旋转过的码里它与四角并不等价。
        """
        pts = np.asarray(self.corners_px, dtype=np.float64)
        x1, y1 = np.floor(pts.min(axis=0)).astype(int)
        x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
        return (int(x1), int(y1), int(x2), int(y2))

    @property
    def center(self) -> tuple:
        """`center_px` 的别名——让统一消费方写 `det.center`（含义按各自定义）。"""
        return self.center_px

    def to_dict(self) -> dict:
        """转成可 JSON 序列化的 dict（下游/落盘用）。"""
        cx, cy = self.center_px
        return {
            "marker_id": int(self.marker_id),
            "corners_px": [[round(float(x), 2), round(float(y), 2)] for x, y in self.corners_px],
            "center_px": [round(float(cx), 2), round(float(cy), 2)],
            "bbox_xyxy": [int(v) for v in self.bbox_xyxy],
        }


@dataclass(frozen=True)
class MarkerPose:
    """一个 ArUco 码相对相机的三维位姿（契约见 docs/interface.md 2.3 节）。

    用大白话说
        相机拍到一个码，除了"这是 7 号码、它在画面第 320 列"，还能说清三件事：
        它在我前方多远（`distance_mm`）、斜了多少度（`tilt_deg`）、在画面里转了多少度
        （`yaw_deg`）。三个数背后是同一个东西：码中心在相机坐标系里的坐标 `tvec_mm`，
        加上一个 3×3 旋转 `rvec`。另外两个量都是从它们算出来的，方便人读和画图。

    属性
        marker_id (int)        字典 DICT_4X4_50 的编号，0–49
        rvec (ndarray)         (3,) float64，Rodrigues 旋转向量；要看角度就转成矩阵
        tvec_mm (ndarray)      (3,) float64，码中心在相机坐标系下的坐标，**单位毫米**；
                               `[2]` 恒 > 0（可见的码不可能在相机后方）
        distance_mm (float)    `||tvec_mm||`，不是 `tvec_mm[2]`——码偏出画面中心时两者不同
        tilt_deg (float)       码面法向与光轴的夹角，0 = 正对相机，90 = 侧对，范围 0–180
        yaw_deg (float)        码 x 轴在图像平面内的方向角，(−180, 180]，图像 y 向下故顺时针为正
        reproj_rms_px (float)  把四角投回图像的 RMS 误差（像素）。它不是"精度"，是
                               "这次解算自洽不自洽"——四角拍得糊、或模型点顺序写错，它就会大
        ambiguous (bool)       平面位姿的两个解难以区分时置 True。**只作诊断**，不代表结果不能用

    不变量（`__post_init__` 里校验）
        `rvec` / `tvec_mm` 一律 float64、形状 (3,)；`tvec_mm[2] > 0`；两个向量长度有限。

    为什么不复用 `ArucoMarker`
        `ArucoMarker` 是"码在图像里的位置"（像素），本类是"码在空间里的位置"（毫米）。
        两者之间隔着内参与一次 PnP 求解，混成一个类会让"要不要内参"变模糊。

    用法
        >>> pose = MarkerPose(marker_id=7, rvec=np.zeros(3), tvec_mm=np.array([0., 0., 500.]),
        ...                   distance_mm=500.0, tilt_deg=0.0, yaw_deg=0.0, reproj_rms_px=0.2)
        >>> pose.to_dict()["distance_mm"]
        500.0
    """

    marker_id: int
    rvec: np.ndarray
    tvec_mm: np.ndarray
    distance_mm: float
    tilt_deg: float
    yaw_deg: float
    reproj_rms_px: float
    ambiguous: bool = False

    def __post_init__(self):
        if not isinstance(self.marker_id, int) or isinstance(self.marker_id, bool):
            raise ValueError(f"marker_id 必须是 int，收到 {type(self.marker_id).__name__}")
        # float64 不是随手选的：这些量要参与毫米级的距离比较，float32 在 1 m 处
        # 的相对精度只到 1e-7，够用；真正的原因是 PnP 内部就是 double，别在中途丢精度。
        rvec = np.asarray(self.rvec, dtype=np.float64).reshape(-1)
        tvec = np.asarray(self.tvec_mm, dtype=np.float64).reshape(-1)
        if rvec.shape != (3,):
            raise ValueError(f"rvec 必须是 3 个分量，收到 {rvec.shape}")
        if tvec.shape != (3,):
            raise ValueError(f"tvec_mm 必须是 3 个分量，收到 {tvec.shape}")
        # 可见的码一定在相机前方。这条不是"防御性编程"：180° 的角点错位会让重投影
        # 误差为 0 而解出 z 为负的解，只有这个符号能把它暴露出来。
        if not np.isfinite(tvec).all() or tvec[2] <= 0.0:
            raise ValueError(
                f"tvec_mm[2] 必须为正（码在相机前方），收到 t={tvec.tolist()}；"
                "常见原因是角点顺序与模型的点顺序对不上")
        object.__setattr__(self, "rvec", rvec)
        object.__setattr__(self, "tvec_mm", tvec)
        object.__setattr__(self, "distance_mm", float(np.linalg.norm(tvec)))
        object.__setattr__(self, "tilt_deg", float(self.tilt_deg))
        object.__setattr__(self, "yaw_deg", float(self.yaw_deg))
        object.__setattr__(self, "reproj_rms_px", float(self.reproj_rms_px))
        object.__setattr__(self, "ambiguous", bool(self.ambiguous))

    @property
    def tz_mm(self) -> float:
        """光轴方向上的距离（毫米）。抓取深度用它；`distance_mm` 是斜距。"""
        return float(self.tvec_mm[2])

    def to_dict(self) -> dict:
        """转成可 JSON 序列化的 dict（下游/落盘用）。

        角度保留 2 位、距离保留 2 位、RMS 保留 3 位：再多的位数在毫米级任务里没有意义，
        却会让 CSV 和 diff 变得难读。
        """
        return {
            "marker_id": int(self.marker_id),
            "rvec": [round(float(v), 6) for v in self.rvec],
            "tvec_mm": [round(float(v), 2) for v in self.tvec_mm],
            "distance_mm": round(self.distance_mm, 2),
            "tilt_deg": round(self.tilt_deg, 2),
            "yaw_deg": round(self.yaw_deg, 2),
            "reproj_rms_px": round(self.reproj_rms_px, 3),
            "ambiguous": bool(self.ambiguous),
        }


def is_marker(det) -> bool:
    """这个检测结果是 ArUco 码吗？（消费方用它分流判色/绘制/落盘）"""
    return isinstance(det, ArucoMarker)
