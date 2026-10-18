#!/usr/bin/env python3
"""薄入口：只把 argv 转交给 up70_vision.tools.pose_sweep.main()。

任务单第 3 条：不同距离 / 旋转角度 / 光照条件下的识别效果测试。
跑完在输出目录留下 raw.csv、summary.md 与每指标一张 PNG。

用法（在仓库根目录）
    python scripts/pose_sweep.py
    python scripts/pose_sweep.py --marker-mm 23 --out output/pose_sweep/20261018
    python scripts/pose_sweep.py --intrinsics configs/camera_calibration.yaml
"""

import sys
from pathlib import Path

try:
    from up70_vision.tools.pose_sweep import main
except ModuleNotFoundError:
    # 未 pip install -e . 时（比如直接在宿主机跑），把 src/ 挂上，行为与安装后一致
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from up70_vision.tools.pose_sweep import main

if __name__ == "__main__":
    # Windows 控制台是 GBK，打印 ✅/⚠️ 会 UnicodeEncodeError 并把退出码变成 1
    # （复核意见 P1-2）。容器里本来就是 UTF-8，这行是空操作。
    from up70_vision.utils.console import ensure_utf8_stdout

    ensure_utf8_stdout()
    sys.exit(main())
