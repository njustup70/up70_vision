"""共同像素网格核查：彩色畸变网格 vs 对齐深度零畸变网格，差几个像素。

为什么不能「反投影再投回去」
---------------------------
对齐深度本来就落在彩色像素网格上：像素 (u,v) 的深度值 Z，是沿彩色 K 的射线量到
的距离。所以「用彩色 K 反投影出 3D，再用彩色 K 投回彩色」是恒等变换，残差恒为 0，
什么也证明不了。写在这里是为了防止后来人再走一遍这条路。

真正要回答的问题
----------------
彩色图的像素网格带畸变（`CameraInfo.d` 非零，且采集时 `enable_color_undistortion`
为 false），而对齐深度是 SW D2C 用目标 K + 零畸变投出来的。两套网格的差异可以直接
从 (K, d) 算出来：把每个「彩色像素」反畸变到理想网格，位移量就是这个像素在两套网
格里的错位量。

棋盘格在其中的角色
------------------
只用于**独立验证 d 是否可信**：已知方格边长的平面靶，PnP 应把重投影残差压到亚像素；
若把 d 置零后残差明显变大且呈径向分布，说明彩色确实是带畸变的。注意反过来的局限：
靶子放不到画面边缘时（深度在那里没有值），d 只能被局部验证，边缘的位移量就只是
「按发布值算出来的上限」，不是量出来的。

判据
----
沿用本阶段的 5 px 门槛：位移场上限 ≤5 px 才算两套网格可视为一致。结论必须带上
「依赖驱动发布的 d」这一条，因为棋盘覆盖不到边缘。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#: 棋盘方格边长（毫米）。打印会缩放，这里必须是尺量值，不是文件里的标称值。
CELL_MM_DEFAULT = 23.0

#: 内角点排列（列 x 行）。OpenCV 的 (cols, rows) 必须按这个顺序传，转置虽然也能检出，
#: 但对象点的行列就反了，PnP 会给出错的姿态。
CORNERS_DEFAULT = (6, 9)

#: 本阶段统一的空间门槛。
SHIFT_LIMIT_PX = 5.0


@dataclass(frozen=True)
class BoardFit:
    """一次棋盘 PnP 的结果。"""

    rms_px: float
    max_px: float
    error_px: np.ndarray
    corners_px: np.ndarray
    tvec: np.ndarray


def board_object_points(cell_mm: float = CELL_MM_DEFAULT,
                        corners: tuple[int, int] = CORNERS_DEFAULT) -> np.ndarray:
    """靶面对象点：z=0 平面上的等距网格，单位毫米。"""
    cols, rows = corners
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(cell_mm)
    return obj


def find_board(gray: np.ndarray,
               corners: tuple[int, int] = CORNERS_DEFAULT) -> np.ndarray | None:
    """检出内角点；找不到返回 None（调用方据此判 UNVERIFIED，不硬凑）。"""
    ok, found = cv2.findChessboardCornersSB(
        gray, corners, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    return found.reshape(-1, 2) if ok else None


def fit_board(corners_px: np.ndarray, k: np.ndarray, d: np.ndarray,
              obj: np.ndarray) -> BoardFit | None:
    """用给定畸变系数做 PnP，返回重投影残差。"""
    ok, rvec, tvec = cv2.solvePnP(obj, corners_px.reshape(-1, 1, 2), k, d,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, d)
    err = np.linalg.norm(proj.reshape(-1, 2) - corners_px, axis=1)
    return BoardFit(rms_px=float(np.sqrt((err ** 2).mean())),
                    max_px=float(err.max()),
                    error_px=err,
                    corners_px=corners_px,
                    tvec=tvec.reshape(3))


def displacement_field(k: np.ndarray, d: np.ndarray, width: int, height: int,
                       step: int = 40):
    """两套网格的位移场。

    :return: (采样像素, 理想网格像素, 位移模长 px)
    """
    us = np.arange(step // 2, width, step, dtype=np.float64)
    vs = np.arange(step // 2, height, step, dtype=np.float64)
    grid_u, grid_v = np.meshgrid(us, vs)
    pix = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1).astype(np.float32)
    ideal = cv2.undistortPoints(pix.reshape(-1, 1, 2), k, d, P=k).reshape(-1, 2)
    return pix, ideal, np.linalg.norm(ideal - pix, axis=1)


def shift_summary(k: np.ndarray, d: np.ndarray, width: int, height: int) -> dict:
    """把位移场压成几个可比较的数：最大值与半径 0.5 处的代表值。"""
    pix, _, shift = displacement_field(k, d, width, height)
    cx, cy = float(k[0, 2]), float(k[1, 2])
    radius = np.linalg.norm((pix - np.array([cx, cy])) / float(k[0, 0]), axis=1)
    band = (radius > 0.45) & (radius < 0.55)
    return {
        "max_px": round(float(shift.max()), 2),
        "max_at_px": [int(round(x)) for x in pix[int(np.argmax(shift))]],
        "at_r0_5_px": round(float(np.median(shift[band])), 2) if band.any() else None,
    }


def check_report(out: Path, cell_mm: float = CELL_MM_DEFAULT,
                 corners: tuple[int, int] = CORNERS_DEFAULT) -> dict:
    """对一份 analyze 输出做网格核查；结果写进 `grid_check.json`。

    只用报告目录里已有的东西（summary.json 的内参、selected/ 的选帧），
    因此不需要回头找 bag，复算时也不依赖 ROS。
    """
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    info = (summary.get("camera_info") or {}).get("color_info") or {}
    if not info:
        raise ValueError(f"{out} 的 summary.json 里没有彩色内参")

    k = np.asarray(info["k"], dtype=np.float64).reshape(3, 3)
    d = np.asarray(info["d"], dtype=np.float64).reshape(-1)
    width, height = int(info["width"]), int(info["height"])

    npz_path = sorted((out / "selected").glob("*.npz"))
    if not npz_path:
        raise ValueError(f"{out} 没有选帧（先跑 analyze）")
    image = np.load(npz_path[0])["color"]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

    obj = board_object_points(cell_mm, corners)
    found = find_board(gray, corners)
    result = {
        "cell_mm": cell_mm,
        "corners": list(corners),
        "board_center_px": None if found is None else
        [round(float(v), 1) for v in found.mean(axis=0)],
        "pnp": None,
        "grid_shift": shift_summary(k, d, width, height),
    }

    if found is None:
        result["status"] = "UNVERIFIED"
        result["reason"] = "选帧里没检出棋盘内角点；不能据此判断网格，重拍或核对角点排列"
    else:
        fit = fit_board(found, k, d, obj)
        fit_zero = fit_board(found, k, np.zeros_like(d), obj)
        if fit is None:
            result["status"] = "UNVERIFIED"
            result["reason"] = "PnP 不收敛"
        else:
            result["pnp"] = {
                "rms_px": round(fit.rms_px, 3),
                "max_px": round(fit.max_px, 3),
                "rms_zero_distortion_px": None if fit_zero is None else round(fit_zero.rms_px, 3),
                "distance_mm": round(float(fit.tvec[2]), 1),
            }
            limit = result["grid_shift"]["max_px"]
            if limit <= SHIFT_LIMIT_PX:
                result["status"] = "PASS"
                result["reason"] = (
                    f"位移场上限 {limit} px ≤ {SHIFT_LIMIT_PX} px；但上限按驱动发布的 d 算出，"
                    "棋盘未覆盖到画面边缘，边缘未被独立验证")
            else:
                result["status"] = "FAIL"
                result["reason"] = (
                    f"位移场上限 {limit} px > {SHIFT_LIMIT_PX} px；彩色畸变与零畸变对齐口径不一致")

    (out / "grid_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_grid_check(dirs, cell_mm: float = CELL_MM_DEFAULT,
                   corners: tuple[int, int] = CORNERS_DEFAULT) -> int:
    """CLI 入口；返回 0=全部可判 / 2=有目录拿不到结论。"""
    failed = 0
    for folder in dirs:
        try:
            result = check_report(Path(folder), cell_mm, corners)
        except (OSError, ValueError, KeyError) as exc:
            print(f"❌ {folder}：{exc}")
            failed += 1
            continue
        pnp = result.get("pnp") or {}
        shift = result["grid_shift"]
        print(f"{folder}：{result['status']}")
        print(f"  · 板心像素 {result['board_center_px']}，距离 {pnp.get('distance_mm')} mm")
        print(f"  · PnP 残差 RMS {pnp.get('rms_px')} px（d 置零 {pnp.get('rms_zero_distortion_px')} px）")
        print(f"  · 网格位移 上限 {shift['max_px']} px @ {shift['max_at_px']}，r=0.5 处 {shift['at_r0_5_px']} px")
        print(f"  · {result['reason']}")
    return 0 if failed == 0 else 2
