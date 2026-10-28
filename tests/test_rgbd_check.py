"""同步 RGB-D 检查工具的回归：配对规则、坐标换算、偏移测量、状态与退出码。

验的是什么（对应计划书第 8 节）
    配对：完全同步、容差边界、等距、不复用、零/重复/倒退时间戳、缺流、不同帧数、
          整数纳秒精度、预热分段与分母
    解码：BGR/RGB、16 位、行 padding、长度不对
    点选与空间：显示/裁剪/面板原点换算、端点重合、越界、平行、标注错配
    报告：三场景六行、未采项保留、5 px 门槛、raw/sync 记 SKIP、状态与退出码
    可复现：同一输入两次跑出同一份配对索引

为什么大部分用例不碰真 bag
    bag 读写依赖 rosbags（可选依赖），而配对、换算、测量都是纯函数。把纯逻辑与
    读包分开测，没装 rosbags 的机器也能跑；只有端到端那条用合成小 bag。

跑法：python tests/test_rgbd_check.py（容器内；pytest 同样可用）
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.rgbd_check.annotate import (display_to_image, in_image, validate_segment,
                                     view_region, PairState, PANEL_H, PANEL_W, BAR_H)
from lab.rgbd_check.bag import FrameMeta, decode_image
from lab.rgbd_check.measure import edge_offset_px
from lab.rgbd_check.pairing import DEFAULT_TOL_NS, pair_frames
from lab.rgbd_check.report import spatial_rows
from lab.rgbd_check.select import OFFSET_LIMIT_PX, pick_mid_pair
from lab.rgbd_check.status import FAIL, PASS, SKIP, UNVERIFIED, worst


def frame(role: str, index: int, stamp_ns: int, bag_time_ns: int = None,
          size=(8, 6)) -> FrameMeta:
    """造一帧元信息；bag 时间默认等于 header 时间，方便直接对照。"""
    width, height = size
    bpp = 2 if role == "depth" else 3
    return FrameMeta(role=role, index=index, stamp_ns=stamp_ns,
                     bag_time_ns=stamp_ns if bag_time_ns is None else bag_time_ns,
                     width=width, height=height,
                     encoding="16UC1" if role == "depth" else "bgr8",
                     step=width * bpp, frame_id="cam")


MS = 1_000_000


# ---------------- 配对 ----------------

def test_perfect_sync_pairs_all():
    stamps = [1_000 * MS, 1_033 * MS, 1_066 * MS]
    colors = [frame("color", i, s) for i, s in enumerate(stamps)]
    depths = [frame("depth", i, s + MS) for i, s in enumerate(stamps)]
    result = pair_frames(colors, depths, warmup_ns=0)
    assert len(result.pairs) == 3
    assert result.status == PASS
    assert all(pair.dt_ns == MS for pair in result.pairs)
    assert result.stats["abs_dt_max_ms"] == 1.0


def test_tolerance_boundary_is_inclusive():
    colors = [frame("color", 0, 1_000 * MS)]
    inside = [frame("depth", 0, 1_000 * MS + DEFAULT_TOL_NS)]
    outside = [frame("depth", 0, 1_000 * MS + DEFAULT_TOL_NS + 1)]
    assert len(pair_frames(colors, inside, warmup_ns=0).pairs) == 1
    assert len(pair_frames(colors, outside, warmup_ns=0).pairs) == 0


def test_tie_prefers_earlier_stamp_then_index():
    colors = [frame("color", 0, 1_000 * MS)]
    # bag 顺序按时间递增（乱序会先被判成 backwards_stamp，那是另一条用例）
    depths = [frame("depth", 0, 995 * MS), frame("depth", 1, 1_005 * MS)]
    result = pair_frames(colors, depths, warmup_ns=0)
    assert result.pairs[0].depth.index == 0          # 995 更早，差一样大


def test_each_frame_used_once():
    colors = [frame("color", 0, 1_000 * MS), frame("color", 1, 1_004 * MS)]
    depths = [frame("depth", 0, 1_002 * MS)]
    result = pair_frames(colors, depths, warmup_ns=0)
    assert len(result.pairs) == 1
    assert len(result.unmatched) == 1
    assert result.rates["color"] < 0.99


def test_zero_stamp_is_anomaly_and_fails():
    colors = [frame("color", 0, 0), frame("color", 1, 1_000 * MS)]
    depths = [frame("depth", 0, 1_001 * MS)]
    result = pair_frames(colors, depths, warmup_ns=0)
    assert result.status == FAIL
    assert [a["kind"] for a in result.anomalies] == ["zero_stamp"]
    assert result.counts["color"] == 2                # 异常帧仍计入分母
    assert result.rates["color"] == 0.5


def test_duplicate_and_backwards_stamps():
    colors = [frame("color", 0, 1_000 * MS), frame("color", 1, 1_000 * MS),
              frame("color", 2, 900 * MS)]
    depths = [frame("depth", 0, 1_001 * MS)]
    result = pair_frames(colors, depths, warmup_ns=0)
    kinds = sorted(a["kind"] for a in result.anomalies)
    assert kinds == ["backwards_stamp", "duplicate_stamp"]
    assert result.status == FAIL


def test_missing_stream_is_unverified():
    result = pair_frames([frame("color", 0, 1_000 * MS)], [], warmup_ns=0)
    assert result.status == UNVERIFIED
    assert not result.pairs


def test_warmup_excluded_from_both_sides():
    # 0 s 与 1 s 两帧在预热里；3 s 之后才进稳定段
    colors = [frame("color", 0, 0), frame("color", 1, 1_000 * MS),
              frame("color", 2, 3_500 * MS), frame("color", 3, 3_533 * MS)]
    depths = [frame("depth", 0, 1 * MS), frame("depth", 1, 1_001 * MS),
              frame("depth", 2, 3_501 * MS), frame("depth", 3, 3_534 * MS)]
    result = pair_frames(colors, depths, warmup_ns=3_000 * MS)
    assert result.warmup_dropped == {"color": 2, "depth": 2}
    assert result.counts == {"color": 2, "depth": 2}
    assert len(result.pairs) == 2
    assert result.stable_start_bag_ns == 3_000 * MS


def test_nanosecond_precision_is_kept():
    colors = [frame("color", 0, 1_000_000_001)]
    depths = [frame("depth", 0, 1_000_000_002)]
    result = pair_frames(colors, depths, warmup_ns=0)
    assert result.pairs[0].dt_ns == 1
    assert result.stats["pairs"] == 1


def test_pairing_is_reproducible():
    colors = [frame("color", i, 1_000 * MS + i * 33 * MS) for i in range(5)]
    depths = [frame("depth", i, 1_000 * MS + i * 33 * MS + 3 * MS) for i in range(5)]
    first = pair_frames(colors, depths, warmup_ns=0)
    second = pair_frames(colors, depths, warmup_ns=0)
    assert [(p.color.index, p.depth.index) for p in first.pairs] == \
           [(p.color.index, p.depth.index) for p in second.pairs]


def test_pick_mid_pair_uses_stable_midpoint():
    stamps = [1_000 * MS + i * 33 * MS for i in range(11)]
    colors = [frame("color", i, s) for i, s in enumerate(stamps)]
    depths = [frame("depth", i, s + MS) for i, s in enumerate(stamps)]
    result = pair_frames(colors, depths, warmup_ns=0)
    selection = pick_mid_pair(result, 0.33)          # 半个稳定段 ≈ 165 ms
    assert selection is not None
    assert abs(selection.pair.color.stamp_ns - (stamps[0] + 165 * MS)) <= 17 * MS


# ---------------- 解码 ----------------

def test_decode_bgr_with_row_padding():
    # 3x2 bgr8，step 补到 16 字节（每行 9 字节 + 7 字节 padding）
    rows = []
    for y in range(2):
        row = bytes([10 + y, 20 + y, 30 + y] * 3)
        rows.append(row + b"\x00" * 7)
    img = decode_image(b"".join(rows), 3, 2, "bgr8", 16)
    assert img.shape == (2, 3, 3)
    assert tuple(img[1, 0]) == (11, 21, 31)


def test_decode_rgb_is_flipped_to_bgr():
    img = decode_image(bytes([1, 2, 3] * 4), 2, 2, "rgb8", 6)
    assert tuple(img[0, 0]) == (3, 2, 1)


def test_decode_depth16_with_padding():
    # 3x2 16UC1，step 补到 8 字节
    row = np.array([1, 2, 3], dtype="<u2").tobytes() + b"\x00\x00"
    img = decode_image(row * 2, 3, 2, "16UC1", 8)
    assert img.dtype == np.uint16
    assert img.tolist() == [[1, 2, 3], [1, 2, 3]]


def test_decode_rejects_short_buffer():
    try:
        decode_image(b"\x00" * 10, 4, 4, "16UC1", 8)
    except Exception as exc:                            # BagError
        assert "长度" in str(exc)
    else:
        raise AssertionError("长度不够时必须报错，不能 reshape 出一张错位图")


# ---------------- 点选坐标换算 ----------------

def test_display_to_image_with_zoom_and_panel_offset():
    # 2 倍放大、视区从 (100, 50) 开始、右面板原点 x=760、顶部说明栏 34
    region = (100.0, 50.0, PANEL_W / 2.0, PANEL_H / 2.0)
    x, y = display_to_image(PANEL_W + 200, BAR_H + 100, (PANEL_W, BAR_H), region)
    assert abs(x - (100 + 200 * 0.5)) < 1e-9
    assert abs(y - (50 + 100 * 0.5)) < 1e-9


def test_view_region_is_clamped_inside_image():
    x0, y0, w, h = view_region((0, 0), 1.0, (848, 480))
    assert (x0, y0) == (0.0, 0.0)
    assert w <= 848 and h <= 480
    x0, y0, w, h = view_region((10_000, 10_000), 2.0, (848, 480))
    assert x0 + w <= 848 and y0 + h <= 480


def test_in_image_bounds():
    assert in_image((0, 0), (10, 10))
    assert not in_image((10.1, 5), (10, 10))


def test_validate_segment_rejects_coincident_and_outside():
    assert validate_segment(((1, 1), (2, 2)), (10, 10)) is None
    assert "重合" in validate_segment(((1, 1), (1, 1)), (10, 10))
    assert "超出" in validate_segment(((1, 1), (11, 2)), (10, 10))


def test_pair_state_limits_edges_and_requires_confirm():
    state = PairState(pair_id="c0_d0", condition_id="center_near", image_size=(100, 100))
    assert state.add_edge("edge1", ((10, 10), (60, 10)), ((11, 12), (61, 12))) is None
    assert state.add_edge("edge2", ((10, 10), (10, 60)), ((13, 10), (13, 60))) is None
    assert "只标" in state.add_edge("edge3", ((10, 10), (60, 10)), ((11, 12), (61, 12)))
    state.add_unmeasurable("edge3", "空洞太多")
    frame_dict = state.to_frame()
    assert len(frame_dict["edges"]) == 2  # 不可测同样不能突破每图两条边上限


# ---------------- 偏移测量 ----------------

def test_offset_horizontal_edge():
    measured = edge_offset_px(((10, 10), (110, 10)), ((12, 13), (112, 13)))
    assert measured.offset_px is not None
    assert abs(abs(measured.offset_px) - 3.0) < 1e-6


def test_offset_skew_edge_uses_normal_distance():
    # 45° 边，法向距离应等于垂直间距，不是水平差
    measured = edge_offset_px(((0, 0), (100, 100)), ((0, 5), (100, 105)))
    assert abs(abs(measured.offset_px) - 5 / np.sqrt(2)) < 1e-6


def test_parallel_and_out_of_segment_are_unverified():
    # 水平参考线的法向是竖直的；对应线也是竖直 → 沿法向打过去没有唯一交点
    parallel = edge_offset_px(((10, 10), (120, 10)), ((60, 10), (60, 90)))
    assert parallel.offset_px is None and "平行" in parallel.reason
    outside = edge_offset_px(((10, 10), (30, 10)), ((200, 13), (260, 13)))
    assert outside.offset_px is None and "线段外" in outside.reason


def test_coincident_endpoints_unverified():
    measured = edge_offset_px(((10, 10), (10, 10)), ((10, 12), (50, 12)))
    assert measured.offset_px is None


# ---------------- 报告与状态 ----------------

def _make_out(root: Path, offset_y: float, manual_review=None, size=(200, 200)) -> Path:
    """造一个最小的 analyze 输出目录：一张选中图对 + 一份标注。"""
    out = root / "out"
    (out / "selected").mkdir(parents=True)
    pid = "c0_d0"
    (out / "selected" / f"{pid}.json").write_text(json.dumps({
        "pair_id": pid, "condition_id": "center_near", "image_size": list(size),
    }), encoding="utf-8")
    (out / "annotations.json").write_text(json.dumps({
        "schema_version": 1,
        "frames": [{
            "pair_id": pid, "condition_id": "center_near", "image_size": list(size),
            "edges": [
                {"edge_id": "edge1", "confirmed": True,
                 "color_segment": [[10, 10], [110, 10]],
                 "depth_segment": [[12, 10 + offset_y], [112, 10 + offset_y]],
                 "manual_review": manual_review},
                {"edge_id": "edge2", "confirmed": False, "color_segment": None,
                 "depth_segment": None, "reason": "空洞", "manual_review": None},
            ],
        }],
    }), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"session": "t", "spatial": {},
                                                "geometry": {"status": PASS}}),
                                      encoding="utf-8")
    return out


def test_spatial_rows_keep_missing_conditions():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_out(Path(tmp), offset_y=3.0)
        rows = spatial_rows(out, "aligned")
        conditions = [row["condition_id"] for row in rows]
        assert conditions.count("center_near") == 2       # 两条边两行
        assert "center_far" in conditions and "right_mid" in conditions
        missing = [r for r in rows if r["condition_id"] == "center_far"]
        assert missing[0]["status"] == UNVERIFIED


def test_spatial_threshold_boundary():
    with tempfile.TemporaryDirectory() as tmp:
        ok = spatial_rows(_make_out(Path(tmp), offset_y=OFFSET_LIMIT_PX,
                                   manual_review="yes"), "aligned")
        assert ok[0]["status"] == PASS
        with tempfile.TemporaryDirectory() as tmp2:
            bad = spatial_rows(_make_out(Path(tmp2), offset_y=OFFSET_LIMIT_PX + 0.5),
                               "aligned")
            assert bad[0]["status"] == FAIL


def test_manual_review_no_downgrades_pass():
    with tempfile.TemporaryDirectory() as tmp:
        rows = spatial_rows(_make_out(Path(tmp), offset_y=1.0, manual_review="no"),
                            "aligned")
        assert rows[0]["status"] == FAIL
        assert "人工复查" in rows[0]["note"]


def test_annotation_size_mismatch_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_out(Path(tmp), offset_y=1.0)
        ann = json.loads((out / "annotations.json").read_text(encoding="utf-8"))
        ann["frames"][0]["image_size"] = [640, 480]
        (out / "annotations.json").write_text(json.dumps(ann), encoding="utf-8")
        rows = spatial_rows(out, "aligned")
        assert rows[0]["status"] == UNVERIFIED
        assert "尺寸" in rows[0]["note"]


def test_raw_mode_marks_missing_spatial_as_skip():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_out(Path(tmp), offset_y=1.0)
        rows = [row for row in spatial_rows(out, "raw") if "未采" not in row["note"]]
        # center_near 有图有标注，照常判；center_far/right_mid 没图 → raw 下降级为 SKIP
        assert any(row["status"] == SKIP for row in spatial_rows(out, "raw")), rows


def test_worst_ordering():
    assert worst([PASS, UNVERIFIED, FAIL]) == FAIL
    assert worst([PASS, SKIP]) == SKIP
    assert worst([]) == UNVERIFIED


# ---------------- 端到端（需要 rosbags） ----------------

def _write_bag(path: Path, color_stamps, depth_stamps, size=(8, 6)) -> None:
    """用 rosbags 写一个极小 mcap：彩色像素值 = 帧序号，深度 = 帧序号*10+100。"""
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.typesys import Stores, get_typestore

    store = get_typestore(Stores.ROS2_JAZZY)
    Image = store.types["sensor_msgs/msg/Image"]
    CameraInfo = store.types["sensor_msgs/msg/CameraInfo"]
    Header = store.types["std_msgs/msg/Header"]
    Time = store.types["builtin_interfaces/msg/Time"]
    Roi = store.types["sensor_msgs/msg/RegionOfInterest"]
    width, height = size

    def header(stamp_ns, frame_id):
        return Header(stamp=Time(sec=stamp_ns // 10**9, nanosec=stamp_ns % 10**9),
                      frame_id=frame_id)

    # Writer 自己创建目录，而且目录已存在会直接拒绝——别先 mkdir。
    with Writer(path, version=8, storage_plugin=StoragePlugin.MCAP) as writer:
        c_conn = writer.add_connection("/camera/color/image_raw",
                                       Image.__msgtype__, typestore=store)
        d_conn = writer.add_connection("/camera/depth/image_raw",
                                       Image.__msgtype__, typestore=store)
        ci_conn = writer.add_connection("/camera/color/camera_info",
                                        CameraInfo.__msgtype__, typestore=store)
        di_conn = writer.add_connection("/camera/depth/camera_info",
                                        CameraInfo.__msgtype__, typestore=store)

        for index, stamp in enumerate(color_stamps):
            data = np.full((height, width, 3), index, dtype=np.uint8).tobytes()
            msg = Image(header=header(stamp, "color_frame"), height=height, width=width,
                        encoding="bgr8", is_bigendian=0, step=width * 3,
                        data=np.frombuffer(data, dtype=np.uint8))
            writer.write(c_conn, stamp, store.serialize_cdr(msg, Image.__msgtype__))
        for index, stamp in enumerate(depth_stamps):
            data = np.full((height, width), 100 + index * 10, dtype="<u2").tobytes()
            msg = Image(header=header(stamp, "depth_frame"), height=height, width=width,
                        encoding="16UC1", is_bigendian=0, step=width * 2,
                        data=np.frombuffer(data, dtype=np.uint8))
            writer.write(d_conn, stamp, store.serialize_cdr(msg, Image.__msgtype__))
        for conn, frame_id in ((ci_conn, "color_frame"), (di_conn, "depth_frame")):
            # d/k/r/p 与 Image.data 一样，rosbags 按 ndarray 序列化；传 list 会在写盘时报
            # `'list' object has no attribute 'view'`，所以这里显式给 np.ndarray。
            info = CameraInfo(header=header(color_stamps[0], frame_id), height=height,
                              width=width, distortion_model="plumb_bob",
                              d=np.zeros(5, dtype=np.float64),
                              k=np.array([100.0, 0.0, width / 2, 0.0, 100.0, height / 2,
                                          0.0, 0.0, 1.0], dtype=np.float64),
                              r=np.eye(3, dtype=np.float64).reshape(9),
                              p=np.array([100.0, 0.0, width / 2, 0.0, 0.0, 100.0,
                                          height / 2, 0.0, 0.0, 0.0, 1.0, 0.0],
                                         dtype=np.float64),
                              binning_x=0, binning_y=0,
                              roi=Roi(x_offset=0, y_offset=0, height=0, width=0,
                                      do_rectify=False))
            writer.write(conn, color_stamps[0],
                         store.serialize_cdr(info, CameraInfo.__msgtype__))


def _require_rosbags():
    try:
        import rosbags  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("没装 rosbags（[lab] 可选依赖）；合成 bag 用例跳过")


def test_end_to_end_analyze_writes_report():
    _require_rosbags()
    from lab.rgbd_check.cli import main

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stamps = [1_000 * MS + i * 33 * MS for i in range(5)]
        _write_bag(root / "bag", stamps, [s + MS for s in stamps])
        out = root / "out"
        code = main(["analyze", str(root / "bag"), "--out", str(out), "--warmup-s", "0"])
        assert code == 0, code
        for name in ("pairs.csv", "unmatched.csv", "summary.json", "report.html",
                     "spatial_edges.csv", "annotations.json"):
            assert (out / name).is_file(), name
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert summary["sync"]["pairs"] == 5
        assert summary["sync"]["status"] == PASS
        assert summary["spatial"]["status"] == UNVERIFIED      # 没标注不许判合格
        assert list((out / "overlays").glob("*_overlay.png"))


def test_end_to_end_low_pair_rate_exits_one():
    _require_rosbags()
    from lab.rgbd_check.cli import main

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stamps = [1_000 * MS, 1_033 * MS, 1_066 * MS]
        _write_bag(root / "bag", stamps, [1_000 * MS + 200 * MS, 1_033 * MS, 1_066 * MS])
        code = main(["analyze", str(root / "bag"), "--out", str(root / "out"),
                     "--warmup-s", "0"])
        assert code == 1
        summary = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
        assert summary["sync"]["status"] == FAIL


def test_existing_output_dir_is_refused():
    _require_rosbags()
    from lab.rgbd_check.cli import main

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stamps = [1_000 * MS]
        _write_bag(root / "bag", stamps, stamps)
        (root / "out").mkdir()
        assert main(["analyze", str(root / "bag"), "--out", str(root / "out")]) == 2


def _run_standalone() -> int:
    """不依赖 pytest：挨个跑本文件的 `test_*`，打印通过/跳过/失败。

    :return: 0 = 全过；1 = 有失败（跳过不算失败）
    """
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = skipped = 0
    for fn in tests:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"  ⏭  {fn.__name__}: {exc}")
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"  ❌ {fn.__name__}: {type(exc).__name__}: {exc}")
    done = len(tests) - failed - skipped
    print(f"\n  test_rgbd_check: {done}/{len(tests)} 通过"
          + (f"，{skipped} 项跳过" if skipped else "")
          + (f"，{failed} 项失败" if failed else ""))
    return 1 if failed else 0


def test_replay_comparison_keeps_headers_content_and_pairing():
    _require_rosbags()
    from lab.rgbd_check.cli import main
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stamps = [1_000*MS+i*33*MS for i in range(5)]
        _write_bag(root/'original', stamps, [s+MS for s in stamps])
        _write_bag(root/'same', stamps, [s+MS for s in stamps])
        assert main(['compare-replay', str(root/'original'), str(root/'same'),
                     '--out', str(root/'pass')]) == 0
        _write_bag(root/'changed', stamps, [s+2*MS for s in stamps])
        assert main(['compare-replay', str(root/'original'), str(root/'changed'),
                     '--out', str(root/'fail')]) == 1
        result = json.loads((root/'fail'/'summary.json').read_text())
        assert result['streams']['depth']['status'] == 'FAIL'
        assert result['pairing']['status'] == 'FAIL'


def test_full_pair_rate_but_short_capture_exits_one():
    _require_rosbags()
    from lab.rgbd_check.cli import main
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stamps = [1_000*MS+i*33*MS for i in range(5)]
        _write_bag(root/'bag', stamps, [s+MS for s in stamps])
        (root/'bag'/'meta.yaml').write_text('capture:\n  warmup_s: 0\n  duration_s: 10\n')
        assert main(['analyze',str(root/'bag'),'--out',str(root/'out')]) == 1
        summary=json.loads((root/'out'/'summary.json').read_text())
        assert summary['sync']['status'] == PASS
        assert summary['capture_integrity']['status'] == FAIL
        assert summary['capture']['warmup_s'] == 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
