#!/usr/bin/env python3
"""薄入口：只把 argv 转交给 up70_vision.tools.gen_pose_scenes.main()。

造"答案已知"的 ArUco 位姿测试图：给定距离/倾角/滚转/光照/噪声/模糊，
把码按相机模型投影后贴进图里，同时写一份同名 JSON 真值。
任务单第 3 条的精度扫描（`scripts/pose_sweep.py`）与位姿单测都靠它出数据。

用法（在仓库根目录）
    python scripts/gen_pose_scenes.py                        # 只造一张 450 mm 正视图
    python scripts/gen_pose_scenes.py --suite                # 造整张扫描矩阵
    python scripts/gen_pose_scenes.py --marker-mm 23 --suite  # 边长按实测值填
"""

import sys
from pathlib import Path

try:
    from up70_vision.tools.gen_pose_scenes import main
except ModuleNotFoundError:
    # 未 pip install -e . 时（比如直接在宿主机跑），把 src/ 挂上，行为与安装后一致
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from up70_vision.tools.gen_pose_scenes import main

if __name__ == "__main__":
    # Windows 控制台是 GBK，打印 ✅/⚠️ 会 UnicodeEncodeError 并把退出码变成 1
    # （复核意见 P1-2）。容器里本来就是 UTF-8，这行是空操作。
    from up70_vision.utils.console import ensure_utf8_stdout

    ensure_utf8_stdout()
    sys.exit(main())
