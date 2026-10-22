"""环境自检：装完依赖先跑这个，确认环境没坑。

跑法
    python -m up70_vision doctor

检查项与通过标准
    1. Python >= 3.10（低于它仓库代码里的类型写法会报错）
    2. opencv 相关的发行包恰好 1 个（装了两个会互相覆盖 cv2）
    3. cv2 / numpy / PyYAML 都能 import 且版本在 pyproject 声明的范围内
    4. 图像读写往返一致：写一张图再读回，逐像素相同。故意用中文文件名，
       顺带验证文件系统与 locale 对非 ASCII 路径没问题
    5. torch 与 CUDA（可选，没装就只提示不报错）

本脚本自包含：不 import up70_vision，因此 src 还是空骨架时也能跑。

退出码：全部通过为 0，有任一失败为 1。
"""

import importlib
import importlib.metadata as md
import sys
import tempfile
from pathlib import Path

MIN_PYTHON = (3, 10)
# 期望的版本区间，与 pyproject.toml 的依赖声明保持一致
EXPECTED = {
    "numpy": (1, 26, 3, 0),
    "cv2": (4, 10, 5, 0),
    "yaml": (6, 0, 99, 0),
}

_failures = []


def _ok(msg: str) -> None:
    print(f"[通过] {msg}")


def _fail(msg: str) -> None:
    print(f"[失败] {msg}")
    _failures.append(msg)


def _warn(msg: str) -> None:
    print(f"[提示] {msg}")


def _version_tuple(text: str) -> tuple:
    """把 "4.12.0" 这类字符串转成可比较的整数元组，非数字段一律当 0。"""
    parts = []
    for chunk in str(text).split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts[:4] + [0] * (4 - len(parts)))


def check_python() -> None:
    """检查 Python 版本。"""
    if sys.version_info[:2] >= MIN_PYTHON:
        _ok(f"Python {sys.version.split()[0]}")
    else:
        _fail(
            f"Python 版本过低：{sys.version.split()[0]}，需要 >= "
            f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}。"
            f"未来 ROS 2 Humble 容器里的 Python 是 3.10，本地请对齐。"
        )


def check_opencv_packages() -> None:
    """检查 opencv 发行包数量。

    这是真实踩过的坑：opencv-python 与 opencv-python-headless（或 conda 装的
    opencv 包）同时存在时，两者的顶层模块都是 cv2，谁后装谁覆盖。
    症状是 cv2.imshow 属性检查存在、调用时才报错。
    """
    names = set()
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if "opencv" in name:
            names.add(name)

    if len(names) == 1:
        _ok(f"opencv 发行包数量 = 1（{min(names)}）")
        return
    if not names:
        _fail("没找到任何 opencv 发行包，请执行 pip install -e . -i https://mirrors.aliyun.com/pypi/simple/")
        return

    _fail(
        f"opencv 发行包数量 = {len(names)}，存在互相覆盖风险：{sorted(names)}。"
        f"只保留 opencv-python，把其余的卸掉，例如："
        f"pip uninstall -y opencv-python-headless opencv-contrib-python"
    )


def check_imports() -> None:
    """检查核心依赖能 import 且版本合规。"""
    for module_name, (lo_a, lo_b, hi_a, hi_b) in EXPECTED.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            _fail(f"import {module_name} 失败：{exc}")
            continue

        version_text = getattr(module, "__version__", "0")
        version = _version_tuple(version_text)
        lower = (lo_a, lo_b, 0, 0)
        upper = (hi_a, hi_b, 0, 0)

        if lower <= version < upper:
            _ok(f"{module_name} {version_text}")
        else:
            _fail(
                f"{module_name} 版本 {version_text} 不在期望区间 "
                f"[{lo_a}.{lo_b}, {hi_a}.{hi_b})"
            )


def check_image_roundtrip() -> None:
    """图像写读往返一致性测试。

    Ubuntu 上 cv2 原生支持 UTF-8 路径，直接 cv2.imread / cv2.imwrite 即可，
    不需要为 Windows 的 ANSI 文件接口做 np.fromfile 兜底。
    这里故意用中文文件名：能顺带抓住 locale 配错（如 LANG=C）导致的
    非 ASCII 路径读写失败——这类问题不报错，只会静默返回 None。
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        _fail(f"导入 numpy / cv2 失败：{exc}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "中文目录" / "红色块.png"

        # 变量 original：np.ndarray，测试图案
        #   例：shape=(480, 640, 3)，左半红右半绿
        # 为什么不用纯色：纯色图在通道顺序写错时（BGR 当 RGB）会得到另一个纯色，
        #   逐像素比较依然可能"通过"；左右两半不同色能暴露通道交换。
        original = np.zeros((480, 640, 3), dtype=np.uint8)
        original[:, :320] = (0, 0, 255)  # 左半：BGR 的纯红
        original[:, 320:] = (0, 255, 0)  # 右半：BGR 的纯绿

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # imwrite 返回 False 而不抛异常，是 OpenCV 少见的"静默失败"接口，必须显式查
            written = cv2.imwrite(str(target), original)
        except Exception as exc:  # noqa: BLE001 - 这里要把任何写入异常都报出来
            _fail(f"写入失败：{exc}")
            return

        if not written or not target.exists() or target.stat().st_size == 0:
            _fail(f"写入后文件不存在或为空（imwrite 返回 {written}）：{target}")
            return

        loaded = cv2.imread(str(target), cv2.IMREAD_COLOR)
        if loaded is None:
            # 读不到时 OpenCV 不抛异常，只返回 None，必须显式判空
            _fail(f"读回失败（返回 None）：{target}")
            return

        if loaded.shape != original.shape:
            _fail(f"读回尺寸不一致：写入 {original.shape}，读回 {loaded.shape}")
            return

        if not np.array_equal(original, loaded):
            diff = int(np.count_nonzero(original != loaded))
            _fail(f"读回像素不一致，共有 {diff} 个像素不同（PNG 是无损格式，应当为 0）")
            return

        _ok(f"图像读写往返一致（{original.shape[0]}x{original.shape[1]}，差异像素 0）")


def check_torch_cuda() -> None:
    """检查 torch 与 CUDA。没装 torch 时只提示，不算失败。"""
    try:
        import torch
    except ImportError:
        _warn("未安装 torch（本轮不需要；接 YOLO 训练时再装）")
        return

    cuda_ok = bool(torch.cuda.is_available())
    if cuda_ok:
        _ok(f"torch {torch.__version__}，CUDA 可用，设备={torch.cuda.get_device_name(0)}")
    else:
        _warn(
            f"torch {torch.__version__} 已装但 CUDA 不可用（会退回 CPU，训练会非常慢）。"
            f"RTX 5060 是 Blackwell sm_120 架构，必须装 cu128 及以上的 wheel，"
            f"cu121 会静默装成 CPU 版。"
        )


def main() -> int:
    """跑完全部检查，返回退出码。"""
    print("=" * 60)
    print("up70-vision 环境自检")
    print("=" * 60)

    check_python()
    check_imports()
    check_opencv_packages()
    check_image_roundtrip()
    check_torch_cuda()

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
