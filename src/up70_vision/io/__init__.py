"""输入层：把「图像从哪来」收进统一接口。

放什么
    所有输入源，对外只暴露 read() -> Frame | None。上层管线不关心图像是
    合成的、回放的、还是摄像头实拍的。

不放什么
    不放检测或判色逻辑。

依赖谁
    依赖 core.types 与 config。

当前为空，未来第一件事
    先写 io/base.py：FrameSource 抽象接口 + build_source 工厂；
    再按需写 synthetic.py（合成图，带 ground truth，用于自检）与 replay.py（图片/视频回放）。
    摄像头（UVC）等相机到位后再写，且需先确认当前用户在 video 组里，
    否则 cv2.VideoCapture(0) 打不开。
"""
