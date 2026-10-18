#!/usr/bin/env python3
"""薄入口：只把 argv 转交给 up70_vision.tools.pose_check.main()。

逐张核位姿：对每个码打印"真值 / 估计 / 误差"，超阈值标 ⚠️。
真值来自同名 JSON（`tools/gen_pose_scenes.py` 造的，含距离与倾角）。

用法（在仓库根目录）
    python scripts/pose_check.py --dir output/pose_scenes
    python scripts/pose_check.py --dir data/pose/real \\
        --intrinsics configs/camera_calibration.yaml --marker-mm 23
"""

import sys
from pathlib import Path

try:
    from up70_vision.tools.pose_check import main
except ModuleNotFoundError:
    # 未 pip install -e . 时（比如直接在宿主机跑），把 src/ 挂上，行为与安装后一致
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from up70_vision.tools.pose_check import main

if __name__ == "__main__":
    sys.exit(main())
