"""控制台编码：让 Windows 的 GBK 终端也能打印 ✅/⚠️ 这类符号。

用大白话说
    Linux 容器的终端是 UTF-8，`print("⚠️")` 一切正常。Windows 的控制台默认是 GBK
    （`cp936`），同一个 `print` 会抛 `UnicodeEncodeError`，进程退出码变成 1，
    于是一个"跑完了、结果也没问题"的工具看起来像崩溃了。

为什么不是把符号换成 ASCII
    仓库既有工具（`tools/aruco_check.py`、`cli.py`）都在用 ✅/❌/⚠️，`progress.md`
    与 PR 描述里也直接抄这些输出。只在新的三个工具里换成 `[OK]`，会让同一份终端输出
    出现两套风格。真正的问题在"终端编码"，不在符号本身。

放什么 / 不放什么
    只放"把标准输出切成 UTF-8"这一件事。不碰 `logging`（它有自己的 handler），
    不碰文件读写（`utils/image_io.py` 管）。
"""

import sys


def ensure_utf8_stdout() -> bool:
    """把 `sys.stdout` 的编码改成 UTF-8；已经是 UTF-8 时什么都不做。

    :return: 是否真的改动过。容器里（UTF-8）返回 False，Windows 控制台返回 True。

    为什么吞掉异常
        `reconfigure` 在 stdout 被重定向成不可重配的对象时（例如某些 IDE 的捕获器）
        会抛 `AttributeError` 或 `OSError`。这时唯一能做的就是不折腾，
        让调用方按原编码继续跑——报错退出反而把"能出结果"变成"直接崩"。
    """
    stream = sys.stdout
    if stream is None:
        return False
    if str(getattr(stream, "encoding", "")).lower().replace("-", "") == "utf8":
        return False
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        return False
    return True
