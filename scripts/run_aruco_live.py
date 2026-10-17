#!/usr/bin/env python3
"""薄入口：只把 argv 转交给 up70_vision.cli.run_aruco_live()。

注意
    真正的实时循环唯一实现在 src/up70_vision/cli.py。这里刻意不写任何业务代码——
    将来接 ROS 2 节点时，节点也走同一个 run_aruco_live()，不会出现第二份"检测 -> 绘制"流程。

用法（在仓库根目录）
    # 容器里（没有 X server）：不建窗口，每 30 帧落一张图
    python scripts/run_aruco_live.py --set detector.name=stub_aruco --headless --limit 60

    # 无码验证：stub 的 none 模式，全程 markers=0
    python scripts/run_aruco_live.py --set detector.name=stub_aruco \
        --set detector.stub_aruco.mode=none --headless --limit 180

    # 真识别 + 相机（需要任务一先合入 main 提供 detector.name=aruco；原生机器有 X server）
    python scripts/run_aruco_live.py --set detector.name=aruco --set input.source=camera
"""

import sys
from pathlib import Path

try:
    from up70_vision.cli import run_aruco_live
except ModuleNotFoundError:
    # 未 pip install -e . 时（比如直接在宿主机跑），把 src/ 挂上，行为与安装后一致
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from up70_vision.cli import run_aruco_live

if __name__ == "__main__":
    sys.exit(run_aruco_live())
