"""相机标定：解出「像素坐标 ↔ 相机坐标」的几何关系，也就是内参矩阵与畸变系数。

用大白话说
    镜头不是理想的小孔。画面边角会被拉弯，像素位置和真实方向之间差着一个固定的变换。
    标定就是拿一块黑白格子板，在镜头前摆十几个不同姿势各拍一张：程序知道每个内角点在
    板子上的真实坐标（比如第 3 个角点在 40 mm 处），也知道它在图里落在哪个像素上，
    于是反推出「焦距是多少像素、光心偏在哪、画面被拉弯多少」——
    前两个装在 3×3 的内参矩阵 K 里，最后那个装在 5 个畸变系数里。

放什么 / 不放什么
    放纯算法：给 numpy 图 → 找角点；给角点 → 解 K 与畸变；给 K 与畸变 → 去畸变。
    **不读文件、不保存文件、不开窗口、不 import io/viz/config**（契约第 5 节）。
    读图、抓帧、写 YAML、开预览窗口都是 `tools/calibration/calibrate.py` 那层的事。

依赖谁 / 被谁用
    只依赖 numpy 与 OpenCV。上层是 `tools/calibration/calibrate.py`（命令行标定工具，
    入口 `python -m up70_vision calibrate`）；阶段二的位姿估计（`solvePnP`）直接消费本模块的
    `CameraIntrinsics`，所以字段名与单位是给别人用的，别随手改。

对外约定
    · `find_board_corners()` 找不到棋盘返回 `None`——不是抛异常，也不是空数组。
      调用方靠它数「这张图能不能用」，把"没拍到"与"拍照出错"分开。
    · `calibrate()` 有效图少于 `MIN_VIEWS` 张抛 `ValueError`（解不出来，早点炸比返回垃圾 K 好）。
    · 返回的 `CameraIntrinsics` 是冻结 dataclass：拿到手就不许改，防止下游"顺手修一下"。
    · 尺寸一律 `(width, height)`、像素一律 `(x, y)`；板的单位随 `BoardSpec.square_mm`。
    · 畸变系数顺序固定为 OpenCV 的 `[k1, k2, p1, p2, k3]`（见 `DIST_ORDER`），不要自己重排。

已知的坑
    · 打印出来的棋盘会被打印机缩放，`square_mm` 必须**用尺子量**后填。注意它**不影响内参**
      （尺度会被位姿吸收），影响的是位姿估计出来的距离：我们这台打印机把标称 15 mm 打成了
      23 mm（+53%），按 15 填会让距离少算 35%（坑 40）。
    · 板子只占画面一小块、或反光/过曝时 `findChessboardCorners` 返回 False。别急着换算法，
      先看是不是板子太小、光照太硬（坑 41）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

#: 畸变系数的固定顺序（OpenCV 的约定）。写错顺序不会报错，只会让去畸变"看起来歪一点"。
DIST_ORDER = ("k1", "k2", "p1", "p2", "k3")

#: 至少要几张有效图才允许标定。2 张也能算出数，但参数几乎不可信；3 张是能报错的下限。
MIN_VIEWS = 3

#: 重投影 RMS 的常用判据（像素）。超过 1.0 说明图拍得不够好或板子参数填错了，结果别直接用。
RMS_GOOD = 0.5
RMS_USABLE = 1.0

# 亚像素精化的终止条件：最多 30 次迭代、或角点位置变化小于 0.001 px
_SUBPIX_CRITERIA = (3, 30, 0.001)          # (TERM_CRITERIA_EPS|MAX_ITER, 次数, 阈值)


@dataclass(frozen=True)
class BoardSpec:
    """标定板参数：**内角点**个数 + 一个格子的边长。

    用大白话说
        别把 `cols` 当成"格子数"。一块 10×7 格的黑白棋盘板，**内角点**是 9×6 个（四条边上
        的角点不算，因为它们不一定出现在图里）。填错这个数，角点顺序检查会通过，但解出来的
        焦距是错的——**不报错**，所以这是最容易踩的一个参数。

    属性
        cols      : int   横向内角点个数（默认 9）
        rows      : int   纵向内角点个数（默认 6）
        square_mm : float 一个格子的边长，单位毫米（默认 15.0，**打印后要用尺子量**）
    """

    cols: int = 9
    rows: int = 6
    square_mm: float = 15.0

    def __post_init__(self) -> None:
        if int(self.cols) < 3 or int(self.rows) < 3:
            raise ValueError(
                f"内角点至少 3x3 才标得出来，收到 {self.cols}x{self.rows}；"
                "注意填的是**内角点**个数，不是格子数（10x7 格 -> 9x6 内角点）")
        if float(self.square_mm) <= 0:
            raise ValueError(f"格子边长必须为正（毫米），收到 {self.square_mm}")

    @property
    def inner_corners(self) -> int:
        """内角点总数 = cols × rows（每张图成功找到时角点个数应当正好是它）。"""
        return int(self.cols) * int(self.rows)

    def to_dict(self) -> dict:
        """转成能直接写 YAML / JSON 的纯 Python 字典。"""
        return {"cols": int(self.cols), "rows": int(self.rows),
                "square_mm": float(self.square_mm)}

    @classmethod
    def from_dict(cls, data: dict) -> "BoardSpec":
        """从字典还原（`--from` 读回标定结果时用）。"""
        return cls(cols=int(data["cols"]), rows=int(data["rows"]),
                   square_mm=float(data["square_mm"]))


def board_object_points(board: BoardSpec | None = None) -> np.ndarray:
    """生成板子上每个内角点的**世界坐标**（板子自己当 z=0 平面）。

    用大白话说
        板子摆在哪、转多少度都不知道，但我们知道角点之间的相对位置：从左到右每隔一个格子
        边长一个点，从上到下每隔一个格子边长一行。于是把板子钉在 z=0 平面、左下角当原点，
        每个角点的坐标就都能写死。

    :param board: 板子参数；省略时用默认 9x6、20 mm
    :return: `(N, 3)` float32，N = cols*rows；第 i 行第 j 列的点是 `(j*sq, i*sq, 0)`

    例子：9x6、20 mm 的板子 —— 第一个点是 (0,0,0)，第二个点是 (20,0,0)，
    最后一行的第一个点是 (0,100,0)、最后一个是 (160,100,0)。
    """
    spec = board or BoardSpec()
    cols, rows, sq = int(spec.cols), int(spec.rows), float(spec.square_mm)
    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq
    return grid


def find_board_corners(image_bgr: np.ndarray, board: BoardSpec | None = None,
                       refine: bool = True) -> np.ndarray | None:
    """在一张 BGR 图里找棋盘的内角点。

    用大白话说
        先转灰度，让 OpenCV 去找"这个 9x6 的格子阵在哪"，找到后再把每个角点位置精化到
        亚像素（不然标定误差里一半来自"角点只精确到整像素"）。

    :param image_bgr: HxWx3 uint8 的 BGR 图（单通道灰度会被拒绝——与契约里 `detect()` 同口径）
    :param board: 板子参数；省略时用默认 9x6、20 mm
    :param refine: 是否做亚像素精化（默认 True；关掉只能复现"不精化时差多少"）
    :return: 找到时 `(N, 1, 2)` float32（N = cols*rows，顺序是 OpenCV 的"左右扫描"），
             找不到返回 `None`。**注意不是 []**：调用方要在意"这张图没法用"。
    :raises ValueError: 入参不是 HxWx3 的 BGR 或不是 uint8
    """
    import cv2

    if not isinstance(image_bgr, np.ndarray):
        raise ValueError(f"需要 np.ndarray，收到 {type(image_bgr).__name__}")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"需要 HxWx3 的 BGR 图，收到形状 {image_bgr.shape}")
    if image_bgr.dtype != np.uint8:
        raise ValueError(f"需要 uint8 图，收到 dtype {image_bgr.dtype}")

    spec = board or BoardSpec()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ADAPTIVE_THRESH：光照不匀时按局部阈值二值化；NORMALIZE_IMAGE：先做亮度和对比度归一，
    # 这两面旗子一起用是 OpenCV 文档推荐的组合，能显著提高"图上板子偏暗/偏亮"时的成功率
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (int(spec.cols), int(spec.rows)),
                                               flags=flags)
    if not found:
        return None

    if refine:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                _SUBPIX_CRITERIA[1], _SUBPIX_CRITERIA[2])
        # 搜索窗 11x11：棋盘格在 1280x1024 里通常有 60~120 px 宽，11 够用且快
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return np.asarray(corners, dtype=np.float32)


@dataclass(frozen=True)
class CameraIntrinsics:
    """一次标定的结果：内参矩阵 + 畸变系数 + 误差，够下游去畸变与位姿估计直接吃。

    用大白话说
        内参矩阵 K 是 3×3：
            fx, fy  焦距，"多少像素"（不是毫米）——它把"真实世界里的一个方向角"换算成"图上偏几个像素"
            cx, cy  光心，光轴穿过画面的那个像素位置（理想是画面正中，实际常偏几十像素）
        畸变系数 5 个：
            k1, k2, k3  径向畸变（画面边缘向外鼓 = 桶形，向内收 = 枕形）
            p1, p2      切向畸变（镜头与感光面没完全平行，往一个方向斜）

    属性
        image_size   : (int, int)    标定时用的图像尺寸 (width, height)；去畸变必须用同一尺寸
        camera_matrix: (3,3) float64 [[fx,0,cx],[0,fy,cy],[0,0,1]]
        dist_coeffs  : (1,5) float64 [k1,k2,p1,p2,k3]，顺序见 `DIST_ORDER`
        rms          : float         所有图的重投影 RMS（像素），越小越好：<0.5 好、<1.0 可用
        per_view_rms : tuple[float,…] 每张图的 RMS；单独偏大那张通常拍糊了或板子弯了
        views        : int           参与标定的有效图张数
        board        : BoardSpec     用了哪块板（重标定时要一致）
        flags        : int           传给 cv2.calibrateCamera 的 flags，复现时要用同一个

    不变量（`__post_init__` 里校验）
        camera_matrix 是 (3,3)、dist_coeffs 是 (1,5)；尺寸与板子参数都合法。
    """

    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms: float
    per_view_rms: tuple[float, ...] = field(default_factory=tuple)
    views: int = 0
    board: BoardSpec = field(default_factory=BoardSpec)
    flags: int = 0

    def __post_init__(self) -> None:
        width, height = (int(v) for v in self.image_size)
        if width <= 0 or height <= 0:
            raise ValueError(f"image_size 必须是正的 (width, height)，收到 {self.image_size}")
        matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        dist = np.asarray(self.dist_coeffs, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"camera_matrix 必须是 3x3，收到 {matrix.shape}")
        if dist.size != len(DIST_ORDER):
            raise ValueError(
                f"dist_coeffs 必须是 {len(DIST_ORDER)} 个（{','.join(DIST_ORDER)}），"
                f"收到 {dist.size} 个")
        # numpy 标量会让 json.dump 直接 TypeError，这里统一转成 Python 标量
        object.__setattr__(self, "image_size", (width, height))
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "dist_coeffs", dist.reshape(1, -1))
        object.__setattr__(self, "rms", float(self.rms))
        object.__setattr__(self, "per_view_rms", tuple(float(v) for v in self.per_view_rms))
        object.__setattr__(self, "views", int(self.views))
        object.__setattr__(self, "flags", int(self.flags))

    # ---------------------------------------------------------------- 只读便捷属性
    @property
    def fx(self) -> float:
        """x 方向焦距（像素）。"""
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        """y 方向焦距（像素）。"""
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        """光心 x（像素）。"""
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        """光心 y（像素）。"""
        return float(self.camera_matrix[1, 2])

    @property
    def quality(self) -> str:
        """一句话质量判据：`good` / `usable` / `poor`（阈值见 `RMS_GOOD` / `RMS_USABLE`）。"""
        if self.rms <= RMS_GOOD:
            return "good"
        return "usable" if self.rms <= RMS_USABLE else "poor"

    def summary(self) -> str:
        """给人看的一行结论（带具体数字，方便贴进 PR / progress）。"""
        return (f"{self.image_size[0]}x{self.image_size[1]}  views={self.views}  "
                f"rms={self.rms:.3f}px({self.quality})  fx={self.fx:.1f} fy={self.fy:.1f} "
                f"cx={self.cx:.1f} cy={self.cy:.1f}  dist="
                f"[{', '.join(f'{v:.4f}' for v in self.dist_coeffs.ravel())}]")

    def to_dict(self) -> dict:
        """转成能写 YAML / JSON、且能被 `from_dict` 原样读回的字典。"""
        return {
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.ravel().tolist(),
            "dist_order": list(DIST_ORDER),
            "rms_px": self.rms,
            "per_view_rms_px": list(self.per_view_rms),
            "views": self.views,
            "board": self.board.to_dict(),
            "flags": self.flags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraIntrinsics":
        """从字典还原（`--from configs/camera_calibration.yaml` 时用）。"""
        matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        dist = np.asarray(data["dist_coeffs"], dtype=np.float64)
        return cls(
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
            camera_matrix=matrix, dist_coeffs=dist,
            rms=float(data.get("rms_px", 0.0)),
            per_view_rms=tuple(float(v) for v in data.get("per_view_rms_px", [])),
            views=int(data.get("views", 0)),
            board=BoardSpec.from_dict(data["board"]) if "board" in data else BoardSpec(),
            flags=int(data.get("flags", 0)),
        )


def calibrate(image_points: Sequence[np.ndarray],
              image_size: tuple[int, int],
              board: BoardSpec | None = None,
              flags: int = 0) -> CameraIntrinsics:
    """用若干张图里找到的角点，解出内参矩阵与畸变系数。

    用大白话说
        板子上的点我们知道真实坐标，图上的点我们也量出来了，剩下就是"找一个变换，让真实坐标
        投影回图里时和量到的位置尽量重合"。`cv2.calibrateCamera` 干的就是这件事；
        它返回的重投影误差 RMS 就是"平均每个角点差多少像素"，是我们判断标定好坏的主要依据。

    步骤
        1. 校验：有效图少于 `MIN_VIEWS` 张直接报错（不返回垃圾结果）
        2. 对每张图配一份世界坐标（同一块板，所以是同一份）
        3. `cv2.calibrateCamera` 解 K、畸变、每张图的外参
        4. 用解出来的参数把每个点**重新投影**回图像，逐张算 RMS（`per_view_rms`）

    :param image_points: 序列，每项是 `find_board_corners()` 的返回值 `(N,1,2)`
    :param image_size: `(width, height)`，必须是**拍这些图时**的实际尺寸
    :param board: 板子参数；省略时用默认 9x6、20 mm
    :param flags: 透传给 `cv2.calibrateCamera`（如 `cv2.CALIB_FIX_K3`）
    :return: `CameraIntrinsics`
    :raises ValueError: 有效图少于 3 张、图像尺寸非法、或某张图角点个数与板子不符
    """
    import cv2

    spec = board or BoardSpec()
    views = [np.asarray(p, dtype=np.float32) for p in image_points if p is not None]
    if len(views) < MIN_VIEWS:
        raise ValueError(
            f"至少需要 {MIN_VIEWS} 张有效图才能标定，当前只有 {len(views)} 张；"
            f"每张图都应该正好找到 {spec.inner_corners} 个内角点（{spec.cols}x{spec.rows}）")
    width, height = (int(v) for v in image_size)
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size 必须是正的 (width, height)，收到 {image_size}")

    for index, points in enumerate(views):
        if points.size != spec.inner_corners * 2:
            raise ValueError(
                f"第 {index} 张图的角点数是 {points.size // 2}，与板子的 {spec.inner_corners} "
                f"（{spec.cols}x{spec.rows}）不符——板子参数填错了，或者换了板子")

    grid = board_object_points(spec)
    object_points = [grid.copy() for _ in views]

    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, views, (width, height), None, None, flags=flags)

    # 逐张算重投影 RMS：用解出来的外参把世界点投回图像，与量到的角点求欧氏距离
    per_view = []
    for points, rvec, tvec in zip(views, rvecs, tvecs):
        projected, _ = cv2.projectPoints(grid, rvec, tvec, matrix, dist)
        error = np.linalg.norm(points.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
        # 单张的 RMS：哪一张明显偏大，通常是那张拍糊了或板子没贴平
        per_view.append(float(np.sqrt(np.mean(error ** 2))))

    return CameraIntrinsics(image_size=(width, height), camera_matrix=matrix,
                            dist_coeffs=dist, rms=float(rms), per_view_rms=tuple(per_view),
                            views=len(views), board=spec, flags=int(flags))


def scale_camera_matrix(camera_matrix, from_size, to_size) -> np.ndarray:
    """把内参按图像尺寸等比缩放（保持视场角不变）。

    用大白话说
        同一颗镜头可以输出 1280×1024，也可以降采样成 640×480。画面变小了，但"看到多宽"
        没变，所以焦距（以像素计）必须跟着变小：fx 按宽度比例缩、fy 按高度比例缩，
        光心 cx/cy 也按同样比例移动。

    为什么必须缩
        1280 的内参直接套到 640 的图上，等效于换了一颗长焦镜头：物体看起来大一倍，
        算出来的距离会整体偏小一半，而**程序不会报错**（progress.md 坑 3 的缩放版）。
        实测：23 mm 码、450 mm，套错内参时 640×480 图上的距离解成 1571.9 mm（偏 249%）。

    :param camera_matrix: (3,3) 原内参
    :param from_size: 原内参对应的 `(宽, 高)`
    :param to_size: 目标 `(宽, 高)`
    :return: (3,3) float64 新内参；`from_size == to_size` 时返回副本（不改入参）
    :raises ValueError: 尺寸里有非正数

    近似
        畸变系数**不随分辨率线性缩放**（它描述的是归一化坐标上的形变）。所以这里只缩
        内参；换分辨率后要严谨还是得重标定，本函数只够"回放调试时别算出离谱距离"。
    """
    matrix = np.array(camera_matrix, dtype=np.float64, copy=True)
    if matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix 必须是 3x3，收到 {matrix.shape}")
    from_w, from_h = (float(v) for v in from_size)
    to_w, to_h = (float(v) for v in to_size)
    if min(from_w, from_h, to_w, to_h) <= 0:
        raise ValueError(f"图像尺寸必须为正：{from_size} -> {to_size}")
    if (from_w, from_h) == (to_w, to_h):
        return matrix
    matrix[0, :] *= to_w / from_w          # fx 与 cx 按宽度缩放
    matrix[1, :] *= to_h / from_h          # fy 与 cy 按高度缩放
    return matrix


def scale_intrinsics(intrinsics: CameraIntrinsics, image_size) -> CameraIntrinsics:
    """把一整份标定结果缩放到另一个图像尺寸（内参缩放 + 尺寸字段替换）。

    :param intrinsics: 原标定结果
    :param image_size: 目标 `(宽, 高)`
    :return: 新的 `CameraIntrinsics`；畸变系数原样带过去（见 `scale_camera_matrix` 的近似说明）
    """
    return CameraIntrinsics(image_size=tuple(int(v) for v in image_size),
                            camera_matrix=scale_camera_matrix(intrinsics.camera_matrix,
                                                              intrinsics.image_size, image_size),
                            dist_coeffs=intrinsics.dist_coeffs, rms=intrinsics.rms,
                            per_view_rms=intrinsics.per_view_rms, views=intrinsics.views,
                            board=intrinsics.board, flags=intrinsics.flags)


def undistort_image(image_bgr: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """把"被镜头拉弯"的画面掰直（去畸变）。

    用大白话说
        标定解出来的畸变系数，反过来用在每一帧上：把每个像素挪到它"本该在"的位置。
        结果是画面四角不再鼓/凹，直线看起来是直的——阶段二位姿估计需要直的直线。

    :param image_bgr: HxWx3 uint8 的 BGR 图
    :param intrinsics: 标定结果；`image_size` 最好与原图一致（不一致 OpenCV 会按原图重算，但语义会变）
    :return: 新的 BGR 图（**不改入参**）；返回图可能与原图尺寸不同，取决于 K 与畸变
    :raises ValueError: 入参不是 HxWx3 的 BGR 图
    """
    import cv2

    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"需要 HxWx3 的 BGR 图，收到 {getattr(image_bgr, 'shape', type(image_bgr))}")
    return cv2.undistort(image_bgr, intrinsics.camera_matrix, intrinsics.dist_coeffs)


def undistort_points(points_px: Iterable, intrinsics: CameraIntrinsics) -> np.ndarray:
    """把几个像素点坐标按同一套畸变公式掰直（画框、投影要用）。

    :param points_px: 形如 `[(x, y), ...]` 或 `(N,2)` 数组
    :param intrinsics: 标定结果
    :return: `(N, 2)` float32，掰直后的像素坐标
    """
    import cv2

    array = np.asarray(list(points_px), dtype=np.float64).reshape(-1, 1, 2)
    # 传 P 表示"用这套内参把归一化坐标再投回像素"，返回的是单个数组（不是 (result, _) 元组）
    fixed = cv2.undistortPoints(array, intrinsics.camera_matrix,
                                intrinsics.dist_coeffs, P=intrinsics.camera_matrix)
    return fixed.reshape(-1, 2).astype(np.float32)
