#!/usr/bin/env python3
"""薄入口：只把 argv 转交给 up70_vision.tools.aruco_check.main()。

它验的是任务单验收清单第一条——"同一张图片中的多个 ArUco 码全部正确识别
（ID、四角、中心）"。测试图组与真值在 assets/aruco/（生成器见 gen_aruco）。

用法（在仓库根目录）
    python scripts/check_aruco.py
    python scripts/check_aruco.py --only rotated
    python scripts/check_aruco.py --max-err-px 1.5
"""

import sys
from pathlib import Path

try:
    from up70_vision.tools.aruco_check import main
except ModuleNotFoundError:
    # 未 pip install -e . 时（比如直接在宿主机跑），把 src/ 挂上，行为与安装后一致
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from up70_vision.tools.aruco_check import main

if __name__ == "__main__":
    sys.exit(main())
