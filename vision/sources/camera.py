"""海康 MV-CU013-A0UC 相机源（厂商 MVS SDK）。

为什么不走 OpenCV，也不走 Aravis
    · 这颗相机是 **USB3 Vision（GenICam）**，不是 UVC：宿主机上没有 /dev/video*，
      cv2.VideoCapture 根本打不开它。
    · Aravis 0.8.20 与自建 0.8.33 实测都取不到帧（相机 U3V 的 SIRM_PAYLOAD_SIZE
      恒为 1 MB、不随请求更新），官方 arv-camera-test-0.8 同样失败。
    · 官方 MVS SDK 正常：实测 28.1 fps @ 20 ms 曝光、7.3 fps @ 80 ms。
    完整证据见 knowledge/docs/progress.md 第五节坑 17、18。

依赖与降级
    MvImport 是 MVS 自带的 ctypes 封装，**import 期就要读环境变量** MVCAM_COMMON_RUNENV
    来定位 .so，所以本模块把 import 放在 __init__ 里、不放在模块顶层：
    没装 MVS 的机器照样能 import vision，只在真正接相机时报一句人话。

接口
    FrameSource：read() -> Frame | None。None = 这路输入结束了
    （取帧超时、相机被拔、或已达到 input.camera.max_frames）。
"""

import importlib
import os
import sys
import time
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof

import numpy as np

from ..types import Frame
from ..logging import get_logger
from . import FrameSource

log = get_logger("io.camera")

DEFAULT_SDK_PATH = "/opt/MVS"


class MvsCameraSource(FrameSource):
    name = "camera"

    def __init__(self,
                 index: int = 0,
                 exposure_us: float = 80000.0,
                 gain_db: float = 10.0,
                 timeout_ms: int = 1000,
                 max_frames: int = 0):
        """
        :param index: 多台相机时选第几台
        :param exposure_us: 曝光时间（微秒）。**<= 0 表示不改动相机当前值**。
            这颗相机很吃曝光：照度不足时优先加光/开大光圈，一味拉长曝光会掉帧率、
            在移动目标上留拖影。
        :param gain_db: 增益（dB）。负值表示不改动。
        :param timeout_ms: 单次取帧的等待上限；超时后 read() 返回 None 结束这一路
        :param max_frames: 最多取多少帧，0 = 不限（实时场景靠 Ctrl+C 或 --limit 停）
        """
        self.index = int(index)
        self.exposure_us = float(exposure_us)
        self.gain_db = float(gain_db)
        self.timeout_ms = int(timeout_ms)
        self.max_frames = int(max_frames)

        self._index = 0
        self._sdk = False            # MV_CC_Initialize 已调用
        self._opened = False         # 句柄已创建（close() 的守卫；语义见 _open()）
        self._device_open = False    # MV_CC_OpenDevice 已成功
        self._grabbing = False
        self._frame_count_warned = False
        self.model = "unknown"
        self.width = self.height = 0

        self.mv = _load_mvs()                     # 失败时抛 RuntimeError（人话）
        try:
            self._open()
        except Exception:
            # 半初始化失败也要释放：相机是**独占**资源，句柄没关掉下次必打不开
            # （复核意见二.1 / vision/docs/interface.md 欠账二）
            self.close()
            raise

    # ------------------------------------------------------------ 打开
    def _open(self) -> None:
        mv = self.mv
        mv.MvCamera.MV_CC_Initialize()
        self._sdk = True

        dev_list = mv.MV_CC_DEVICE_INFO_LIST()
        ret = mv.MvCamera.MV_CC_EnumDevices(mv.MV_USB_DEVICE | mv.MV_GIGE_DEVICE, dev_list)
        if ret != 0:
            raise RuntimeError(f"MVS 枚举设备失败：{self._err(ret)}")
        if dev_list.nDeviceNum == 0:
            raise RuntimeError(
                "MVS 没枚举到任何相机。检查：USB3 线插好？宿主机 udev 规则配了？"
                "（容器里 /dev/bus/usb 透传了吗）")
        if self.index >= dev_list.nDeviceNum:
            raise RuntimeError(f"只找到 {dev_list.nDeviceNum} 台相机，要的是第 {self.index} 台")

        dev_info = cast(dev_list.pDeviceInfo[self.index],
                        POINTER(mv.MV_CC_DEVICE_INFO)).contents
        self.model = _decode(dev_info.SpecialInfo.stUsb3VInfo.chModelName) \
            if dev_info.nTLayerType == mv.MV_USB_DEVICE else "unknown"

        self.cam = mv.MvCamera()
        ret = self.cam.MV_CC_CreateHandle(dev_info)
        if ret != 0:
            raise RuntimeError(f"创建相机句柄失败：{self._err(ret)}")
        # 句柄一旦创建就置位：此后任何一步失败，close() 都能把它收掉。
        # 若把 _opened 留到"全部成功"再置位，StartGrabbing 失败时设备仍被独占占用，
        # 而 close() 会因为 _opened=False 直接返回 → 句柄泄漏、下次必打不开（复核意见二.1）
        self._opened = True

        try:
            ret = self.cam.MV_CC_OpenDevice(mv.MV_ACCESS_Exclusive, 0)
            if ret != 0:
                raise RuntimeError(
                    f"打开相机失败：{self._err(ret)}。相机是**独占**资源——"
                    f"MVS 客户端 / 探针 / 实时窗口同时只能开一个，先关掉占用者")
            self._device_open = True

            # 关触发 + 关自动曝光/增益（否则手动设的值会被自动算法改回去）
            self.cam.MV_CC_SetEnumValue("TriggerMode", mv.MV_TRIGGER_MODE_OFF)
            self.cam.MV_CC_SetEnumValue("ExposureAuto", mv.MV_EXPOSURE_AUTO_MODE_OFF)
            self.cam.MV_CC_SetEnumValue("GainAuto", mv.MV_GAIN_MODE_OFF)
            if self.exposure_us > 0:
                self.cam.MV_CC_SetFloatValue("ExposureTime", self.exposure_us)
            if self.gain_db >= 0:
                self.cam.MV_CC_SetFloatValue("Gain", self.gain_db)

            self.width = self._get_int("Width") or 0
            self.height = self._get_int("Height") or 0
            self.exposure_us = self._get_float("ExposureTime")
            self.gain_db = self._get_float("Gain")

            ret = self.cam.MV_CC_StartGrabbing()
            if ret != 0:
                raise RuntimeError(f"开始取流失败：{self._err(ret)}")
            self._grabbing = True
        except Exception:
            self.close()                 # 半初始化失败也要把设备释放掉
            raise

        self.nbytes = int(self.width) * int(self.height) * 3
        self.buf = (c_ubyte * self.nbytes)()
        self.info = mv.MV_FRAME_OUT_INFO_EX()
        log.info("相机就绪：%s %dx%d 曝光%.0fus 增益%.1fdB",
                 self.model, self.width, self.height, self.exposure_us, self.gain_db)

    # ------------------------------------------------------------ 取帧
    def read(self) -> Frame | None:
        if self.max_frames and self._index >= self.max_frames:
            return None
        mv = self.mv
        deadline = time.monotonic() + max(0.05, self.timeout_ms / 1000.0)

        while True:
            memset(byref(self.info), 0, sizeof(self.info))
            ret = self.cam.MV_CC_GetImageForBGR(self.buf, self.nbytes, self.info, self.timeout_ms)
            if ret == 0:
                break
            if time.monotonic() >= deadline:
                if not self._frame_count_warned:
                    self._frame_count_warned = True
                    log.warning("取帧超时（%d ms，%s）——相机被拔了？还是被别的程序抢占了？",
                                self.timeout_ms, self._err(ret))
                return None
            time.sleep(0.005)          # MVS 是查询式接口：没数据立刻返回错误码，得自己重试

        image = np.frombuffer(self.buf, dtype=np.uint8).reshape(
            self.info.nHeight, self.info.nWidth, 3).copy()
        frame = Frame(image_bgr=image,
                      index=self._index,
                      source=f"camera:mvs:{self.model}",
                      stamp=time.time())
        self._index += 1
        return frame

    def close(self) -> None:
        """释放相机与 SDK。**幂等**：半初始化失败、重复调用都安全。"""
        try:
            if self._grabbing:
                self.cam.MV_CC_StopGrabbing()
                self._grabbing = False
            if self._device_open:
                self.cam.MV_CC_CloseDevice()       # 只在设备确实打开过时关
                self._device_open = False
            if self._opened:
                self.cam.MV_CC_DestroyHandle()     # 句柄创建过就销毁
                self._opened = False
        except Exception as exc:                      # noqa: BLE001
            log.warning("关闭相机时出错（忽略）：%s", exc)
        finally:
            if self._sdk:
                self.mv.MvCamera.MV_CC_Finalize()
                self._sdk = False
            self._opened = False

    # ------------------------------------------------------------ 小工具
    def _get_int(self, key):
        v = self.mv.MVCC_INTVALUE()
        memset(byref(v), 0, sizeof(v))
        return int(v.nCurValue) if self.cam.MV_CC_GetIntValue(key, v) == 0 else None

    def _get_float(self, key, fallback: float = 0.0) -> float:
        v = self.mv.MVCC_FLOATVALUE()
        memset(byref(v), 0, sizeof(v))
        return float(v.fCurValue) if self.cam.MV_CC_GetFloatValue(key, v) == 0 else fallback

    def _err(self, code: int) -> str:
        value = code & 0xFFFFFFFF
        try:
            errc = importlib.import_module("MvErrorDefine_const")
            for name, const in vars(errc).items():
                if isinstance(const, int) and name.startswith("MV_E_") and const == value:
                    return f"{name}(0x{value:08X})"
        except Exception:                             # noqa: BLE001
            pass
        return f"0x{value:08X}"


# ---------------------------------------------------------------- 模块级
def _decode(char_array) -> str:
    raw = memoryview(char_array).tobytes()
    cut = raw.find(b"\x00")
    if cut >= 0:
        raw = raw[:cut]
    for enc in ("gbk", "utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _load_mvs():
    """定位并 import MVS 的 Python 封装；失败时抛带处理办法的 RuntimeError。

    import 路径优先级：环境变量 UP70_MVS_IMPORT > $MVCAM_SDK_PATH/Samples/64/Python/MvImport
    """
    sdk_path = os.environ.get("MVCAM_SDK_PATH") or DEFAULT_SDK_PATH
    os.environ.setdefault("MVCAM_SDK_PATH", sdk_path)
    os.environ.setdefault("MVCAM_COMMON_RUNENV", os.path.join(sdk_path, "lib"))

    import_dir = os.environ.get("UP70_MVS_IMPORT") or os.path.join(
        sdk_path, "Samples", "64", "Python", "MvImport")
    if not os.path.isdir(import_dir):
        raise RuntimeError(
            f"找不到 MVS 的 Python 封装：{import_dir}\n"
            f"  说明：input.source=camera 需要厂商 MVS SDK（这颗相机走 USB3 Vision，"
            f"不是 UVC，OpenCV / Aravis 都取不到帧）。\n"
            f"  处理：把 MVS 的 deb 放到 .devcontainer/pkgs/ 后重新 build 镜像"
            f"（见 .devcontainer/pkgs/README.md）；宿主机跑则直接装 MVS 到 {sdk_path}。")
    if import_dir not in sys.path:
        sys.path.insert(0, import_dir)
    try:
        return importlib.import_module("MvCameraControl_class")
    except Exception as exc:                          # noqa: BLE001
        raise RuntimeError(
            f"import MvCameraControl_class 失败（{import_dir}）：{exc}\n"
            f"  常见原因：MVCAM_COMMON_RUNENV 指向的目录下没有 libMvCameraControl.so。") from exc
