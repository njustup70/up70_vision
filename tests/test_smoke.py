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


def test_yolo_batch_runs_and_writes():
    """YOLO 批处理冒烟：验权重路径 / 解码 / 落盘这条链能通。

    识别精度不在这里验（那要 COCO 图与真值）；这里只要入口跑通、产物落齐。
    权重不进版本库（`knowledge/models/*.onnx`），没有就跳过——CI 与新人机器上通常没有。
    """
    weights = ROOT / "knowledge" / "models" / "yolov8n.onnx"
    if not weights.is_file():
        import pytest
        pytest.skip("没有 YOLO 权重：先 python lab/export_yolo.py --weights yolov8n.pt")

    out = _fresh_dir("yolo")
    scenes = _fresh_dir("yolo_scenes")
    shutil.copy2(ROOT / "tests" / "assets" / "aruco" / "scenes" / "multi_row_3.png", scenes)
    code = run(["--out", str(out), "--limit", "1",
                "--set", "detector.name=yolo",
                "--set", "input.source=replay",
                "--set", f"input.replay.path={scenes}"])
    assert code == 0, f"退出码 {code}"
    for name in ("frame_0000.png", "results.json", "detections.csv", "summary.json"):
        assert (out / name).is_file(), f"缺少产物 {name}"


def test_yolo_weights_path_points_into_knowledge_models():
    """默认权重路径必须指向 knowledge/models（S4 搬家后曾漏改，一跑就 FileNotFoundError）。

    直接读 YAML 的 `detector.yolo` 段：`AppConfig.detector.params` 只装**当前选中**检测器的
    参数（默认 blob），拿不到 yolo 默认值。
    """
    import yaml

    data = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    weights = str(data["detector"]["yolo"]["weights"])
    assert "knowledge/models" in weights, weights


def test_live_rejects_non_aruco_detector():
    """实时入口只跑 ArUco：别的检测器要以退出码 2 + 人话提示拒绝，不能抛 traceback。"""
    code = run_aruco_live(["--headless", "--limit", "1",
                           "--set", "detector.name=blob",
                           "--set", "input.source=synthetic"])
    assert code == 2, f"应当拒绝，实际退出码 {code}"


def test_live_hud_reports_pose_status():
    """实时窗口 HUD 必须能表达"位姿算没算"（2026-10-25 加）。

    没有解时不能什么都不显示——画面里只有框没有数字，人分不清是"这个码解不出来"
    还是"位姿没开"。
    """
    from vision.cli.live import pose_status_line

    assert pose_status_line(False, 0, 0) is None          # 没开位姿：不画
    assert pose_status_line(True, 1, 0) is None           # 有解：每码单独一行，不用状态行
    assert pose_status_line(True, 0, 3) == "pose: no solution (fail 3)"


def test_live_hud_median_needs_three_samples():
    """防抖行：样本不足 3 条不画；够了就取中位数，且不受离群帧拖动。"""
    from vision.cli.live import median_pose_line

    assert median_pose_line([]) is None
    assert median_pose_line([(511, 6.4, -0.3), (512, 6.5, -0.2)]) is None

    # 9 条里混进一条离群（d=900, tilt=40）——中位数必须还在 511/6.4 附近
    history = [(511, 6.4, -0.3)] * 8 + [(900, 40.0, 170.0)]
    line = median_pose_line(history)
    assert line is not None and "d 511" in line and "tilt 6.4" in line, line
