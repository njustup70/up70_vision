"""输入层：把「图像从哪来」收进统一接口。

放什么
    所有输入源，对外只暴露 read() -> Frame | None。上层管线不关心图像是
    合成的、回放的、还是摄像头实拍的。

不放什么
    不放检测或判色逻辑。

依赖谁
    只依赖 core.types（Frame）。**不 import config** —— 配置由 cli.py 拆成普通参数
    传给 build_source（见 base.py）。这一行以前写的是"依赖 core.types 与 config"，
    与 base.py 和 docs/interface.md 第 5 节矛盾，按契约欠账三改正。

当前状态
    base.py（FrameSource 接口 + build_source 工厂）、synthetic.py（合成图，带 ground truth）、
    replay.py（图片 / 视频回放）、camera.py（真相机，走 MVS SDK）均已就位。

关于真相机（重要）
    本节目的相机是海康 MV-CU013-A0UC，走 **USB3 Vision（GenICam），不是 UVC**：
    宿主机上没有 /dev/video*，也没有"用户在 video 组"这回事——权限靠 udev 规则给，
    设备节点是 /dev/bus/usb/...。所以 cv2.VideoCapture 打不开它，OpenCV 的 UVC 路径
    在这颗相机上用不了；必须走厂商 SDK（MVS），实现见 camera.py。
"""
