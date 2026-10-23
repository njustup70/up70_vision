#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
海康 MV-CU013-A0UC 实时颜色/轮廓测量（MVS SDK 版）

它是什么
    tools/ 下的人工交互工具（不进运行管线）：实时窗口 + 6 个 HSV 滑条 + 面积下限，
    在画面上直接看"框有没有贴住目标、偏左偏右多少像素"，用来现场调阈值与曝光。
    · 与 tools/blob/hsv_tuner.py 的分工：hsv_tuner 的目标是**把阈值写回配置文件**；
      本工具的目标是**在真实场景里看判色/定位效果**（只读，不改配置）。
    · 取帧走厂商 MVS SDK：这颗相机是 USB3 Vision（不是 UVC），OpenCV 的 UVC 路径
      与 Aravis 都用不了，详见 knowledge/docs/progress.md 第五节。

依赖
    MVS SDK（/opt/MVS；镜像里已装，见 .devcontainer/pkgs/README.md）+ opencv-python + numpy。
    没装 MVS 会直接报错并提示怎么装。

用法（仓库根目录；容器内或已装 MVS 的宿主机）
    python lab/mvs_color_measure.py                   # 实时窗口
    python lab/mvs_color_measure.py --exposure 80000 --gain 10
    python lab/mvs_color_measure.py --no-window --frames 5 --out t.png
    python vision/tools/blob/color_measure.py               # 直接跑也行（脚本自带 MVS 路径兜底）

窗口键位（先用鼠标点一下窗口再按）：
    q / ESC  退出          s  保存当前画面+掩膜
    1 2 3 4 5  红/绿/蓝/暗色/亮色 预设
    空格  暂停/继续
    [ ]   曝光 ×0.8 / ×1.25      （这颗相机很吃曝光，常调）
    - =   增益 -2dB / +2dB
    f     打印当前曝光/增益/帧率

注意：相机是独占的 —— MVS 客户端、arv 工具或别的程序开着时，本脚本打不开。
"""

import argparse
import os
import sys
import time
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof

import cv2
import numpy as np

# ---- MVS 环境（MvImport 靠 MVCAM_COMMON_RUNENV 找 .so；不依赖 shell 配置）----
os.environ.setdefault("MVCAM_SDK_PATH", "/opt/MVS")
os.environ.setdefault("MVCAM_COMMON_RUNENV", "/opt/MVS/lib")
MVS_IMPORT = "/opt/MVS/Samples/64/Python/MvImport"
if not os.path.isdir(MVS_IMPORT):
    print(f"❌ 找不到 MvImport: {MVS_IMPORT}（MVS 装了吗？）")
    sys.exit(2)
sys.path.insert(0, MVS_IMPORT)
from MvCameraControl_class import *   # noqa: E402

# 窗口标题必须纯英文（中文标题会让 createTrackbar 报 NULL window handler）
WIN_PARAM = "params"
WIN_MAIN = "result"

PRESETS = {
    "1": ("红色", (170, 10, 100, 255, 70, 255)),
    "2": ("绿色", (45, 85, 80, 255, 60, 255)),
    "3": ("蓝色", (95, 135, 100, 255, 60, 255)),
    "4": ("暗色/黑色", (0, 179, 0, 255, 0, 80)),
    "5": ("亮色/白色", (0, 179, 0, 60, 180, 255)),
}
TRACKBARS = [("H_min", 179, 170), ("H_max", 179, 10),
             ("S_min", 255, 100), ("S_max", 255, 255),
             ("V_min", 255, 70), ("V_max", 255, 255),
             ("min_area", 50000, 500)]

# 滑条名与窗口标题必须纯 ASCII：中文会让 createTrackbar 报 NULL window handler（复核意见二.4）


# ============================================================ MVS 相机
class MvsCamera:
    def __init__(self, index=0, exposure=None, gain=None):
        self.ok = False
        self.exposure = float(exposure) if exposure else None
        self.gain = float(gain) if gain else None
        self.exp_min = self.exp_max = None
        self.gain_min = self.gain_max = None
        self.w = self.h = None

        MvCamera.MV_CC_Initialize()
        self.dev_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, self.dev_list)
        if ret != 0 or self.dev_list.nDeviceNum == 0:
            print("❌ 没枚举到相机（USB3 口？udev 权限？）")
            return
        print(f"找到 {self.dev_list.nDeviceNum} 个设备")
        for i in range(self.dev_list.nDeviceNum):
            info = cast(self.dev_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if info.nTLayerType == MV_USB_DEVICE:
                name = bytes(info.SpecialInfo.stUsb3VInfo.chModelName).split(b"\x00")[0].decode("gbk", "ignore")
                print(f"  [{i}] USB3 {name}")

        self.cam = MvCamera()
        dev_info = cast(self.dev_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
        if self.cam.MV_CC_CreateHandle(dev_info) != 0:
            print("❌ 创建句柄失败")
            return
        if self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
            print("❌ 打开设备失败 —— 相机被别的程序占用了？"
                  "（MVS 客户端 / 另一个 probe 进程。同一时刻只能有一个程序用相机）")
            self.cam.MV_CC_DestroyHandle()
            return

        # 关触发 + 关自动曝光/增益（手动设置才生效）
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        self.cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)

        self.w = self._get_int("Width")
        self.h = self._get_int("Height")
        self.exp_min, self.exp_max = self._get_float_range("ExposureTime")
        self.gain_min, self.gain_max = self._get_float_range("Gain")
        print(f"  分辨率 {self.w}x{self.h} | 曝光范围 {self.exp_min:.0f}~{self.exp_max:.0f}us"
              f" | 增益范围 {self.gain_min:.1f}~{self.gain_max:.1f}dB")

        # 默认给一个能看清的曝光（实测 20ms 太暗、80ms 合适）
        if self.exposure is None:
            self.exposure = min(max(40000.0, self.exp_min or 100), self.exp_max or 100000)
        if self.gain is None:
            self.gain = min(max(5.0, self.gain_min or 0), self.gain_max or 20)
        self.set_exposure(self.exposure)
        self.set_gain(self.gain)

        if self.cam.MV_CC_StartGrabbing() != 0:
            print("❌ 开始取流失败")
            return
        self.nbytes = int(self.w) * int(self.h) * 3
        self.buf = (c_ubyte * self.nbytes)()
        self.info = MV_FRAME_OUT_INFO_EX()
        self.ok = True
        print(f"✅ 相机就绪：曝光 {self.exposure:.0f}us，增益 {self.gain:.1f}dB")

    def _get_int(self, key):
        v = MVCC_INTVALUE()
        memset(byref(v), 0, sizeof(v))
        return int(v.nCurValue) if self.cam.MV_CC_GetIntValue(key, v) == 0 else None

    def _get_float_range(self, key):
        v = MVCC_FLOATVALUE()
        memset(byref(v), 0, sizeof(v))
        if self.cam.MV_CC_GetFloatValue(key, v) == 0:
            return float(v.fMin), float(v.fMax)
        return None, None

    def set_exposure(self, us):
        us = float(np.clip(us, self.exp_min or 5, self.exp_max or 1e6))
        if self.cam.MV_CC_SetFloatValue("ExposureTime", us) == 0:
            self.exposure = us
        return self.exposure

    def set_gain(self, db):
        db = float(np.clip(db, self.gain_min or 0, self.gain_max or 32))
        if self.cam.MV_CC_SetFloatValue("Gain", db) == 0:
            self.gain = db
        return self.gain

    def grab(self, timeout_ms=1000):
        """查询式取帧：没数据立即返回 None，调用方循环重试"""
        memset(byref(self.info), 0, sizeof(self.info))
        ret = self.cam.MV_CC_GetImageForBGR(self.buf, self.nbytes, self.info, timeout_ms)
        if ret != 0:
            return None
        return np.frombuffer(self.buf, dtype=np.uint8).reshape(
            self.info.nHeight, self.info.nWidth, 3).copy()

    def close(self):
        if not getattr(self, "cam", None):
            return
        try:
            if self.ok:
                self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
        finally:
            MvCamera.MV_CC_Finalize()


# ============================================================ 界面与算法
def nothing(_):
    pass


def build_ui():
    cv2.namedWindow(WIN_PARAM, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PARAM, 420, 300)
    cv2.imshow(WIN_PARAM, np.zeros((60, 300), np.uint8))
    cv2.waitKey(50)                      # 先让窗口实体化，再挂滑块
    for name, hi, init in TRACKBARS:
        cv2.createTrackbar(name, WIN_PARAM, init, hi, nothing)


def get_params():
    v = [cv2.getTrackbarPos(name, WIN_PARAM) for name, _, _ in TRACKBARS]
    return (v[0], v[1], v[2], v[3], v[4], v[5]), v[6]


def apply_preset(key):
    if key not in PRESETS:
        return None
    name, (hmin, hmax, smin, smax, vmin, vmax) = PRESETS[key]
    for tb, val in zip(["H_min", "H_max", "S_min", "S_max", "V_min", "V_max"],
                       [hmin, hmax, smin, smax, vmin, vmax]):
        cv2.setTrackbarPos(tb, WIN_PARAM, val)
    return name


def make_mask(hsv, params):
    hmin, hmax, smin, smax, vmin, vmax = params
    if hmin <= hmax:
        mask = cv2.inRange(hsv, np.array([hmin, smin, vmin], np.uint8),
                           np.array([hmax, smax, vmax], np.uint8))
    else:                                # 红色跨 0 环绕
        m1 = cv2.inRange(hsv, np.array([hmin, smin, vmin], np.uint8),
                         np.array([179, smax, vmax], np.uint8))
        m2 = cv2.inRange(hsv, np.array([0, smin, vmin], np.uint8),
                         np.array([hmax, smax, vmax], np.uint8))
        mask = cv2.bitwise_or(m1, m2)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def analyse(mask, frame, min_area):
    vis = frame.copy()
    h, w = frame.shape[:2]
    cx0, cy0 = w // 2, h // 2
    cv2.drawMarker(vis, (cx0, cy0), (255, 255, 255), cv2.MARKER_CROSS, 30, 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        cx, cy = x + bw // 2, y + bh // 2
        results.append({"area": area, "bbox": (x, y, bw, bh), "center": (cx, cy)})
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(vis, f"({cx},{cy})", (x + 4, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if results:
        t = max(results, key=lambda r: r["area"])
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        cv2.drawContours(vis, [cv2.boxPoints(rect).astype(int)], 0, (0, 200, 255), 2)
        cx, cy = t["center"]
        dx, dy = cx - cx0, cy - cy0
        cv2.line(vis, (cx0, cy0), (cx, cy), (255, 0, 255), 1)
        cv2.putText(vis, f"offset dx={dx} dy={dy}px angle={rect[2]:.1f}deg",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(vis, f"targets={len(results)}", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return vis, results


def main(argv=None):
    """实时判色窗口（需要 MVS SDK 与相机，缺一不可）。

    :param argv: 参数列表；None 表示取 `sys.argv[1:]`。
        收形参是为了让 `python lab/mvs_color_measure.py --index 1` 能用：
        统一入口会把"mvs-color"之外的参数原样传进来，而直接读 `sys.argv`
        会把这几个字当成非法参数。
    :return: 0 正常；1 相机打不开或一帧都没取到
    """
    ap = argparse.ArgumentParser(description="海康相机实时颜色/轮廓测量（MVS）")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--exposure", type=float, default=None, help="曝光 us")
    ap.add_argument("--gain", type=float, default=None, help="增益 dB")
    ap.add_argument("--min-area", type=float, default=500)
    ap.add_argument("--no-window", action="store_true", help="无窗口自检")
    ap.add_argument("--frames", type=int, default=3, help="无窗口模式抓几帧")
    ap.add_argument("--out", default="mvs_color.png")
    args = ap.parse_args(argv)

    cam = MvsCamera(args.index, args.exposure, args.gain)
    if not cam.ok:
        return 1

    if args.no_window:
        got = 0
        vis = None
        for i in range(args.frames):
            frame = None
            for _ in range(300):
                frame = cam.grab()
                if frame is not None:
                    break
                time.sleep(0.01)
            if frame is None:
                print("❌ 取帧超时")
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = make_mask(hsv, PRESETS["1"][1])          # 红预设
            vis, res = analyse(mask, frame, args.min_area)
            got += 1
            print(f"  [{i+1}] {frame.shape} 均值={frame.mean():.1f} 目标数={len(res)}")
        if vis is not None:
            cv2.imwrite(args.out, vis)
            print(f"✅ 无窗口自检完成，结果图: {args.out}")
        cam.close()
        return 0 if got else 1

    build_ui()
    apply_preset("1")
    print("提示：先点一下窗口，再按 1/2/3/4/5 切预设；[ ] 调曝光；- = 调增益；q 退出")

    paused = False
    fps = 0.0
    last_print = 0.0
    idx = 0
    while True:
        t0 = time.time()
        frame = None if paused else cam.grab(timeout_ms=1000)
        if frame is None:
            if not paused:
                time.sleep(0.005)
            else:
                time.sleep(0.03)
            continue
        idx += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        params, min_area = get_params()
        mask = make_mask(hsv, params)
        vis, results = analyse(mask, frame, min_area)

        dt = time.time() - t0
        fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0)
        cv2.putText(vis, f"{fps:.1f} FPS", (vis.shape[1] - 130, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis, f"exp {cam.exposure:.0f}us  gain {cam.gain:.1f}dB",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        now = time.time()
        if now - last_print > 0.2:
            last_print = now
            if results:
                r = max(results, key=lambda x: x["area"])
                print(f"[{idx}] 目标数={len(results)} 最大: 中心={r['center']} "
                      f"面积={r['area']:.0f}px² 外接框={r['bbox']}")
            else:
                print(f"[{idx}] 没找到目标（调滑块 / 按 1-5 换预设 / [ ] 调曝光）")

        if paused:
            cv2.putText(vis, "PAUSED", (vis.shape[1] // 2 - 60, vis.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow(WIN_MAIN, vis)
        cv2.imshow(WIN_PARAM, mask)
        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
            print("暂停" if paused else "继续")
        elif key == ord("s"):
            fn = f"mvs_saved_{int(time.time())}.png"
            cv2.imwrite(fn, vis)
            cv2.imwrite(fn.replace(".png", "_mask.png"), mask)
            print(f"已保存 {fn}")
        elif key == ord("["):
            print(f"曝光 → {cam.set_exposure(cam.exposure * 0.8):.0f}us")
        elif key == ord("]"):
            print(f"曝光 → {cam.set_exposure(cam.exposure * 1.25):.0f}us")
        elif key == ord("-"):
            print(f"增益 → {cam.set_gain(cam.gain - 2):.1f}dB")
        elif key == ord("="):
            print(f"增益 → {cam.set_gain(cam.gain + 2):.1f}dB")
        elif key == ord("f"):
            print(f"曝光 {cam.exposure:.0f}us | 增益 {cam.gain:.1f}dB | {fps:.1f} fps")
        elif chr(key) in PRESETS:
            print(f"已载入预设：{apply_preset(chr(key))}")

    cam.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
