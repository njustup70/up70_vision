#!/usr/bin/env python3
"""薄入口：只把 argv 转交给 up70_vision.cli.run()。

注意
    真正的运行逻辑唯一实现在 src/up70_vision/cli.py。这里刻意不写任何业务代码——
    将来接 ROS 2 节点时，节点也走同一个 cli.run()，不会出现第二份"检测 -> 判色"流程。

用法（在仓库根目录）
    python scripts/run.py
    python scripts/run.py --set color.hit_ratio_min=0.6
    python scripts/run.py --config configs/default.yaml --out output/demo
"""

import sys
from pathlib import Path

try:
    from up70_vision.cli import run
except ModuleNotFoundError:
    # 未 pip install -e . 时（比如直接在宿主机跑），把 src/ 挂上，行为与安装后一致
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from up70_vision.cli import run

if __name__ == "__main__":
    sys.exit(run())
