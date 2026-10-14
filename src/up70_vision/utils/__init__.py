"""通用工具层：最底层，不依赖任何同层模块。

放什么
    与业务无关的通用能力：图像读写、日志配置等。

不放什么
    不放与业务相关的工具——那些应该进 core / io / viz 对应的一层。

依赖谁
    不依赖任何同层模块。它是整个包里被所有层 import 的地方。

当前为空，未来第一件事
    先写 utils/logging.py（日志配置，整个进程调一次），
    再写 utils/image_io.py（图像读写；Ubuntu 上 cv2 原生支持 UTF-8 路径，
    直接 cv2.imread / cv2.imwrite 即可，不需要为 Windows 的 ANSI 路径做 fromfile 兜底）。
"""
