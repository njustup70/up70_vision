"""OpenCV 冒烟测试：确认这台机器上的 OpenCV 真的能跑。

跑法
    ./demo.sh                                # 推荐：构建 + 自检 + 冒烟一条命令跑完
    python -m vision demo                   # 只跑冒烟测试

做什么
    只用 OpenCV 与 numpy 走一遍视觉最常用的六个动作：造图、转颜色空间、
    阈值分割、找轮廓、按掩码叠加、写图落盘。

    全程**不 import 本包的其他模块**——它是环境自检，不是业务实现。
    这样 src 里的实现再长，也不会和它形成第二份「检测 -> 判色」逻辑。

应该看到什么
    [自检] cv2 4.14.0 | numpy 2.2.6
    [自检] 检出轮廓 1 个 | 叠加图已写出 output/demo/opencv_smoke.png
    [自检] OpenCV 冒烟测试 -> PASS

退出码：PASS 为 0，FAIL 为 1。
"""

import sys
from pathlib import Path

# `lab/` 不是包：直接 `python lab/xxx.py` 跑，脚本目录会进 sys.path，
# 但仓库根不会。这一行把仓库根补进去，才能 `import vision`（lab 可以依赖 vision，反过来不行）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:  # Ubuntu 上最常见：缺 libGL.so.1
    print(f"[自检] 导入依赖失败：{exc}")
    print("[自检] Ubuntu 上 opencv-python 需要系统库，请先执行：")
    print("         sudo apt-get install -y libgl1 libglib2.0-0t64")
    raise SystemExit(1) from exc

# 仓库根由本文件位置反推：lab/demo.py 往上两层（lab -> 根）。
# 用 __file__ 而不是 cwd：在容器里从任意目录调用都能把图写到仓库的 output/demo/。
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_PNG = _REPO_ROOT / "output" / "demo" / "opencv_smoke.png"


def build_sample_bgr() -> np.ndarray:
    """造一张左红右绿的 BGR 图，形状 (480, 640, 3)。

    为什么不用纯色图：纯色图在通道顺序写错时（BGR 当 RGB）会得到另一个纯色，
    逐像素比较依然可能"通过"；左右两半不同色能暴露通道交换。
    """
    image_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    image_bgr[:, :320] = (0, 0, 255)  # 左半：BGR 的纯红
    image_bgr[:, 320:] = (0, 255, 0)  # 右半：BGR 的纯绿
    return image_bgr


def main() -> int:
    """跑一遍最小视觉链路，返回退出码。"""
    image_bgr = build_sample_bgr()

    # BGR -> HSV。OpenCV 的 H 范围是 0-179，不是 0-360；
    # 从教程抄 0-360 的写法时必须先除以 2，否则阈值整体偏移。
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 只取"红"这一段。红色的色相紧贴 0 与 179 的环绕点，
    # 真实阈值表里通常要写两段（如 [0,10] 与 [170,179]）再取并集。
    mask_red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))

    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按掩码叠回原图：红色区域保留、其余变黑，用来肉眼确认分割范围对不对
    overlay_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_red)

    _SMOKE_PNG.parent.mkdir(parents=True, exist_ok=True)
    written = cv2.imwrite(str(_SMOKE_PNG), overlay_bgr)

    print(f"[自检] cv2 {cv2.__version__} | numpy {np.__version__}")
    print(
        f"[自检] 检出轮廓 {len(contours)} 个 | "
        f"叠加图{'已写出' if written else '写出失败'} {_SMOKE_PNG}"
    )

    if not written:
        # imwrite 返回 False 而不抛异常，这是 OpenCV 少见的"静默失败"接口，必须显式查
        print("[自检] OpenCV 冒烟测试 -> FAIL（检查 output/ 目录是否可写）")
        return 1

    print("[自检] OpenCV 冒烟测试 -> PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
