#!/usr/bin/env python3
"""薄入口：只把 argv 转交给 up70_vision.tools.gen_aruco.main()。

生成 ArUco 测试图组（单码 / 多码 / 旋转 / 无码）与每张图的真值 JSON，
输出到 assets/aruco/。产物随仓库分发，重跑逐像素可复现。

用法（在仓库根目录）
    python scripts/gen_aruco.py
    python scripts/gen_aruco.py --scene multi rotated
    python scripts/gen_aruco.py --list-scenes
"""

import sys
from pathlib import Path

try:
    from up70_vision.tools.gen_aruco import main
except ModuleNotFoundError:
    # 未 pip install -e . 时（比如直接在宿主机跑），把 src/ 挂上，行为与安装后一致
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from up70_vision.tools.gen_aruco import main

if __name__ == "__main__":
    sys.exit(main())
