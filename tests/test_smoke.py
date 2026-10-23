"""端到端冒烟：跑真实入口，断言"产物真的落出来了"。

为什么需要它（这次重构的教训）
    107 项单测都是"调函数、验返回值"，覆盖不到入口层的**副作用**：
      · `viz` 拆散后 `live` 还在调旧函数名 → 单测全绿，一跑就 NameError
      · `cli/output.py` 搬类时漏了模块级 `CSV_HEADER` → 单测全绿，一落盘就 NameError
    这类问题只有"真的跑一遍入口 + 看一眼产物"才能发现，所以单独放在这里。

覆盖什么
    · `run`（blob + synthetic）：叠加图 / JSON / CSV / summary 四种产物
    · `run`（aruco + 回放）：走 ArUco 绘制分支（四边形），不炸
    · `live --headless`（aruco + 回放，每帧落盘）：走分流绘制 + OutputWriter
    · 入口清单：恰好 6 条子命令（迁移计划书 D3）

跑法：`python -m pytest tests/test_smoke.py -q`
"""

import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
try:
    import vision  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))

from vision.cli import run_aruco_live, run          # noqa: E402

OUT = ROOT / "output" / "_smoke"


def _fresh_dir(name: str) -> pathlib.Path:
    d = OUT / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def test_run_writes_all_artifacts_for_blob():
    """判色批处理：1 帧 6 个色块，四类产物都要落出来。"""
    out = _fresh_dir("blob")
    code = run(["--out", str(out), "--limit", "1"])
    assert code == 0, f"退出码 {code}"

    for name in ("frame_0000.png", "results.json", "detections.csv", "summary.json"):
        assert (out / name).is_file(), f"缺少产物 {name}"

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["detections"] == 6, summary
    assert summary["color_correct"] == 6, summary

    # CSV 表头是这次重构漏掉过的常量，顺手钉住列数与首列
    lines = (out / "detections.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",")[0] == "frame", lines[0]
    assert len(lines) == 1 + 6, f"CSV 应 6 行数据，实际 {len(lines) - 1}"


def test_run_writes_overlay_for_aruco_replay():
    """ArUco 批处理：回放多码图，走四边形绘制分支，产物齐全且检出 6 个码。"""
    out = _fresh_dir("aruco")
    # 单独准备一个只含多码图的目录：scenes/ 里按文件名排序第一张是"无码"图，
    # 用 --limit 1 会正好取到它，测不出绘制分支。
    scenes = _fresh_dir("aruco_scenes")
    shutil.copy2(ROOT / "tests" / "assets" / "aruco" / "scenes" / "multi_grid_6.png", scenes)
    code = run(["--out", str(out), "--limit", "1",
                "--set", "detector.name=aruco",
                "--set", "input.source=replay",
                "--set", f"input.replay.path={scenes}"])
    assert code == 0, f"退出码 {code}"
    assert (out / "frame_0000.png").is_file(), "ArUco 叠加图没落盘"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["detections"] == 6, summary


def test_live_headless_draws_and_writes():
    """实时入口（headless，每帧落盘）：覆盖 draw_all 分流与 OutputWriter 在 live 里的用法。

    live 的输出目录是固定的 `output/aruco/`（它刻意不跟随 `output.dir`：实时落盘是
    "过程采样"，不该混进批处理的结果目录），所以这里比对"跑之前 / 跑之后"的文件数。
    """
    scenes = _fresh_dir("live_scenes")
    shutil.copy2(ROOT / "tests" / "assets" / "aruco" / "scenes" / "multi_row_3.png", scenes)
    live_out = ROOT / "output" / "aruco"
    shutil.rmtree(live_out, ignore_errors=True)      # 每次从空目录开始：文件名是 frame_000000，
                                                     # 不清的话上一轮的图会让"文件数没变"误判

    code = run_aruco_live(["--headless", "--limit", "2", "--save-every", "1",
                           "--set", "detector.name=aruco",
                           "--set", "input.source=replay",
                           "--set", f"input.replay.path={scenes}"])
    assert code == 0, f"退出码 {code}"
    written = list(live_out.rglob("frame_*.png"))
    assert written, f"live headless 没写出叠加图（{live_out}）"


def test_main_entry_lists_exactly_six_subcommands():
    """入口只允许 6 条（D3）：run/live/doctor/demo/hsv-tuner/clean。"""
    from vision.__main__ import COMMANDS, LAB_SCRIPTS

    names = set(COMMANDS) | set(LAB_SCRIPTS) | {"clean"}
    assert names == {"run", "live", "doctor", "demo", "hsv-tuner", "clean"}, names
