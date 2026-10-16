"""config 层的测试：默认值来自 YAML、点分覆盖、非法值报错带字段路径。

config/__init__.py 的约定：默认值一律写在 configs/*.yaml 里，代码只定义结构与取值范围；
非法取值要抛 ConfigError 且消息里带字段路径与可用值。
"""

import os
import sys
import tempfile
import textwrap
from pathlib import Path

try:
    from up70_vision.config.loader import (apply_overrides, load_config, load_thresholds,
                                           resolve_thresholds_path, resolve_path)
    from up70_vision.config.schema import AppConfig, ConfigError, from_dict
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from up70_vision.config.loader import (apply_overrides, load_config, load_thresholds,
                                           resolve_thresholds_path, resolve_path)
    from up70_vision.config.schema import AppConfig, ConfigError, from_dict


def test_from_dict_defaults_and_validation():
    cfg = from_dict({})
    assert isinstance(cfg, AppConfig)
    assert cfg.input.source == "synthetic"
    assert (cfg.input.synthetic.num_frames, cfg.input.synthetic.blocks_per_frame) == (3, 6)
    assert cfg.color.hit_ratio_min == 0.5
    assert cfg.detector.name == "blob"
    assert cfg.detector.params["min_area_px"] == 400
    assert cfg.log.level == "INFO"
    # 相机段的默认值（曝光/增益是相机的关键参数，改了会直接影响判色）
    assert cfg.input.camera.index == 0
    assert cfg.input.camera.exposure_us == 80000.0
    assert cfg.input.camera.gain_db == 10.0
    assert cfg.input.camera.timeout_ms == 1000
    assert cfg.input.camera.max_frames == 0


def test_from_dict_reads_nested_sections():
    cfg = from_dict({
        "input": {"source": "replay",
                  "replay": {"path": "data/clips", "kind": "video", "frame_stride": 3, "max_frames": 10},
                  "synthetic": {"width": 320, "height": 240, "num_frames": 1, "blocks_per_frame": 2}},
        "color": {"min_saturation": 90, "hit_ratio_min": 0.75, "thresholds_file": "t.yaml"},
        "detector": {"name": "blob", "blob": {"min_area_px": 1000, "morph_kernel": 7, "max_detections": 8}},
        "output": {"enabled": False, "dir": "output/x", "save_overlay": False,
                   "save_json": True, "save_csv": False},
        "log": {"level": "debug"},
    })
    assert cfg.input.source == "replay"
    assert cfg.input.replay.frame_stride == 3 and cfg.input.replay.kind == "video"
    assert cfg.input.synthetic.blocks_per_frame == 2
    assert cfg.color.min_saturation == 90 and cfg.color.hit_ratio_min == 0.75
    assert cfg.color_thresholds_file == "t.yaml"
    assert cfg.detector.params["morph_kernel"] == 7
    assert cfg.output.enabled is False and cfg.output.save_csv is False


def test_camera_source_config():
    """相机段能读进来，且非法值都报 ConfigError（相机源是硬件相关代码，先靠配置校验兜住）。"""
    cfg = from_dict({"input": {"source": "camera",
                               "camera": {"index": 1, "exposure_us": 20000.0,
                                          "gain_db": 0.0, "timeout_ms": 500,
                                          "max_frames": 5}}})
    assert cfg.input.source == "camera"
    assert cfg.input.camera.index == 1
    assert cfg.input.camera.exposure_us == 20000.0
    assert cfg.input.camera.gain_db == 0.0          # 0 dB 是合法设置，不是"未设置"
    assert cfg.input.camera.max_frames == 5

    for bad, keyword in [
        ({"input": {"camera": {"index": -1}}}, "index"),
        ({"input": {"camera": {"exposure_us": -1}}}, "exposure_us"),
        ({"input": {"camera": {"gain_db": -1}}}, "gain_db"),
        ({"input": {"camera": {"timeout_ms": 0}}}, "timeout_ms"),
        ({"input": {"camera": {"max_frames": -1}}}, "max_frames"),
    ]:
        try:
            from_dict(bad)
        except ConfigError as exc:
            assert f"input.camera.{keyword}" in str(exc), \
                f"报错应含字段路径 input.camera.{keyword}，实际：{exc}"
        else:
            raise AssertionError(f"应当报错：{bad}")


def test_invalid_values_error_with_field_path():
    for bad, keyword in [
        ({"input": {"source": "webcam"}}, "input.source"),
        ({"input": {"synthetic": {"blocks_per_frame": 7}}}, "blocks_per_frame"),
        ({"input": {"synthetic": {"num_frames": -1}}}, "num_frames"),
        ({"input": {"replay": {"frame_stride": 0}}}, "frame_stride"),
        ({"color": {"hit_ratio_min": 1.5}}, "hit_ratio_min"),
        ({"color": {"min_value": 300}}, "min_value"),
        ({"detector": {"name": "blob", "blob": {"morph_kernel": 4}}}, "morph_kernel"),
        ({"log": {"level": "VERBOSE"}}, "log.level"),
    ]:
        try:
            from_dict(bad)
        except ConfigError as exc:
            assert keyword in str(exc), f"报错应含字段路径 {keyword!r}，实际：{exc}"
        else:
            raise AssertionError(f"应当报错：{bad}")


def test_apply_overrides_type_coercion():
    data = {"color": {"hit_ratio_min": 0.5, "min_saturation": 40},
            "output": {"enabled": True},
            "input": {"synthetic": {"background_bgr": [30, 30, 30]}}}
    apply_overrides(data, ["color.hit_ratio_min=0.6", "color.min_saturation=90",
                           "output.enabled=false", "input.synthetic.background_bgr=1,2,3"])
    assert data["color"]["hit_ratio_min"] == 0.6 and isinstance(data["color"]["hit_ratio_min"], float)
    assert data["color"]["min_saturation"] == 90 and isinstance(data["color"]["min_saturation"], int)
    assert data["output"]["enabled"] is False
    assert data["input"]["synthetic"]["background_bgr"] == [1, 2, 3]


def test_apply_overrides_rejects_typos():
    data = {"color": {"hit_ratio_min": 0.5}}
    for bad, keyword in [("colour.hit_ratio_min=0.6", "colour"),
                         ("color.hit_rations_min=0.6", "hit_ratio_min"),
                         ("color.hit_ratio_min", "key=value"),
                         ("color.hit_ratio_min=abc", "小数")]:
        try:
            apply_overrides(dict(data), [bad])
        except ConfigError as exc:
            assert keyword in str(exc), f"应提到 {keyword!r}，实际：{exc}"
        else:
            raise AssertionError(f"应当报错：{bad}")


def test_resolve_path_and_thresholds_relative_to_config():
    with tempfile.TemporaryDirectory() as td:
        cfg_dir = os.path.join(td, "configs")
        os.makedirs(cfg_dir)
        cfg_path = os.path.join(cfg_dir, "default.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent("""
                input: {source: synthetic}
                color: {thresholds_file: color_thresholds.yaml, hit_ratio_min: 0.5}
            """))
        with open(os.path.join(cfg_dir, "color_thresholds.yaml"), "w", encoding="utf-8") as f:
            f.write(textwrap.dedent("""
                red:
                  - [0, 120, 70, 10, 255, 255]
                  - [170, 120, 70, 179, 255, 255]
                green:
                  - [45, 120, 70, 75, 255, 255]
            """))
        cfg, loaded_path = load_config(cfg_path, ["color.hit_ratio_min=0.7"])
        assert cfg.color.hit_ratio_min == 0.7
        assert os.path.abspath(loaded_path) == os.path.abspath(cfg_path)

        thresholds_path = resolve_thresholds_path(cfg, loaded_path)
        assert thresholds_path == os.path.join(cfg_dir, "color_thresholds.yaml"), \
            f"阈值表应相对配置文件目录解析，实得 {thresholds_path}"
        thresholds = load_thresholds(thresholds_path)
        assert sorted(thresholds) == ["green", "red"]
        assert len(thresholds["red"]) == 2

        assert resolve_path("/abs", "x.yaml") == "/abs/x.yaml"
        assert resolve_path("/base", "sub/x.yaml") == "/base/sub/x.yaml"


def test_stub_aruco_detector_config():
    """detector.name=stub_aruco 的默认值与显式参数（契约另见 docs/interface.md 3.2）。"""
    default = from_dict({"detector": {"name": "stub_aruco"}})
    assert default.detector.name == "stub_aruco"
    assert default.detector.params == {"ids": [0, 1, 2], "side_px": 90, "moving": True}, \
        f"默认参数应来自 DETECTOR_DEFAULT_PARAMS：{default.detector.params}"

    cfg = from_dict({"detector": {"name": "stub_aruco",
                                  "stub_aruco": {"ids": [3, 4], "side_px": 120, "moving": False}}})
    assert cfg.detector.params["ids"] == [3, 4]
    assert cfg.detector.params["side_px"] == 120
    assert cfg.detector.params["moving"] is False


def test_detector_name_and_stub_params_are_validated():
    try:
        from_dict({"detector": {"name": "yolo"}})
    except ConfigError as exc:
        assert "stub_aruco" in str(exc), f"报错应列出可用检测器，实际：{exc}"
    else:
        raise AssertionError("未知检测器名应当报错")

    for bad, keyword in (({"ids": []}, "ids"),
                         ({"ids": ["0"]}, "ids"),
                         ({"side_px": 0}, "side_px")):
        params = {"ids": [0], "side_px": 90}
        params.update(bad)
        try:
            from_dict({"detector": {"name": "stub_aruco", "stub_aruco": params}})
        except ConfigError as exc:
            assert keyword in str(exc), f"报错应提到 {keyword!r}，实际：{exc}"
        else:
            raise AssertionError(f"应当报错：{bad}")


def _run_standalone() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
        except Exception as exc:                     # noqa: BLE001
            failed += 1
            print(f"  ❌ {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n  test_config: {len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
