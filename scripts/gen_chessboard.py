#!/usr/bin/env python3
"""薄入口：只把 argv 转交给 `up70_vision.tools.gen_chessboard.main()`。

真正的实现在 src/up70_vision/tools/gen_chessboard.py，这里刻意不写任何业务代码。
"""
import sys
from pathlib import Path

try:
    from up70_vision.tools.gen_chessboard import main
except ModuleNotFoundError:
    # 未 pip install -e . 时（比如直接在宿主机跑），把 src/ 挂上，行为与安装后一致
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from up70_vision.tools.gen_chessboard import main

if __name__ == "__main__":
    sys.exit(main())
