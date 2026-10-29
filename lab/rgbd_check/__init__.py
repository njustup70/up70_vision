
"""同步 RGB-D 离线检查：话题、配对、静态像素对齐（入口见 `lab.rgbd_check.cli`）。

五个文件按「读一次要读多少」切：
    bag.py       读 bag、解图像、拿内参
    pairing.py   确定性一对一配对与时间统计
    select.py    每段取中间一对、三场景清单与门槛
    measure.py   一条边的中点法向偏移
    annotate.py  本地 OpenCV 四次点选
    report.py    CSV / summary / 叠加图 / report.html
    cli.py       子命令与退出码

跑法：
    python -m lab.rgbd_check inspect bags/raw_004
    python -m lab.rgbd_check analyze bags/raw_004 --out output/rgbd/raw_004
    python -m lab.rgbd_check annotate output/rgbd/raw_004

不依赖 ROS 运行时；需要 `[lab]` 可选依赖里的 rosbags 才能读 bag。
"""

from .cli import build_parser, main

__all__ = ["main", "build_parser"]
