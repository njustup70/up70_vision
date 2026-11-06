"""交接包、纯状态缓存及只读入口的无硬件回归。"""
from dataclasses import FrozenInstanceError
import hashlib
import importlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'integration/driverlogic'))
from MainLogic.adapters.rgbd import RgbdState
from lab.export_driverlogic import export_bundle


def message(stamp=12_345_678_901, rows=None, frame='camera_color_optical_frame'):
    """生成与真实消息字段一致的样例。"""
    return NS(header=NS(stamp=NS(sec=stamp//10**9, nanosec=stamp % 10**9), frame_id=frame),
              candidates=[candidate()] if rows is None else rows)


def candidate(x=.12, z=.5):
    """构造正深度、法向朝相机的红色 block。"""
    result = NS(hypothesis=NS(class_id='block', score=.75),
                pose=NS(pose=NS(position=NS(x=x, y=-.02, z=z),
                                orientation=NS(x=1., y=0., z=0., w=0.))))
    return NS(class_id=0, color_id=1, detection=NS(results=[result]))


def test_snapshot_preserves_units_header_and_is_immutable():
    """输出保持米、采集时间和光学系，不与可变 ROS 输入共享对象。"""
    clock = [10.]
    state = RgbdState(clock=lambda: clock[0])
    assert state.snapshot().status == 'waiting'
    msg = message()
    state.update(msg)
    first = state.snapshot()
    assert first.stamp_ns == 12_345_678_901
    assert first.frame_id == 'camera_color_optical_frame'
    assert first.candidates[0].position_m == (.12, -.02, .5)
    assert first.candidates[0].normal == (0., 0., -1.)
    assert (first.candidates[0].class_id, first.candidates[0].color_id,
            first.candidates[0].class_name, first.candidates[0].score) == (0, 1, 'block', .75)
    with pytest.raises(FrozenInstanceError):
        first.candidates[0].color_id = 2
    msg.candidates[0].detection.results[0].pose.pose.position.x = 999.
    assert state.snapshot().candidates[0].position_m[0] == .12
    clock[0] = 10.5
    assert state.snapshot().status == 'active'
    clock[0] = 10.500001
    assert state.snapshot().status == 'stale'
    assert not state.snapshot().candidates
    assert first.candidates  # 历史快照不可变，每次决策需重新读。


def test_empty_same_stamp_restart_and_close():
    """同时间戳撤销也生效；接收重新开始可恢复，关闭后不能恢复旧目标。"""
    state = RgbdState()
    state.update(message())
    state.update(message(rows=[]))
    assert state.snapshot().status == 'empty'
    assert not state.snapshot().candidates
    state.update(message(stamp=1, rows=[candidate(.3), candidate(.7)]))
    assert [v.position_m[0] for v in state.snapshot().candidates] == [.3, .7]
    state.close()
    state.close()
    state.update(message())
    assert state.snapshot().status == 'closed'
    assert not state.snapshot().candidates


@pytest.mark.parametrize('field,value', [('z', 0.), ('z', float('nan'))])
def test_invalid_candidate_cannot_retain_previous_target(field, value):
    """非法深度使本帧无效，不能沿用上一帧。"""
    state = RgbdState()
    state.update(message())
    row = candidate()
    setattr(row.detection.results[0].pose.pose.position, field, value)
    assert state.update(message(rows=[row])) == 1
    assert state.invalid_candidates == 1
    assert not state.snapshot().candidates
    assert state.snapshot().status == 'invalid'


def test_bad_rows_do_not_suppress_good_rows_and_header_failure_clears():
    """缺失 results 和退化四元数只跳过对应项，损坏 header 撤销整帧。"""
    state = RgbdState()
    bad = candidate()
    bad.detection.results[0].pose.pose.orientation.x = 0.
    state.update(message(rows=[NS(detection=NS(results=[])), bad, candidate()]))
    assert len(state.snapshot().candidates) == 1
    assert state.invalid_candidates == 2
    state.update(message(frame=''))
    assert state.snapshot().status == 'invalid'
    assert not state.snapshot().candidates


@pytest.mark.parametrize('age', [0., -1., float('inf'), float('nan')])
def test_bad_timeout_is_rejected(age):
    """无效超时不能悄悄禁用断流保护。"""
    with pytest.raises(ValueError, match='max_age_s'):
        RgbdState(age)


def test_concurrent_readers_never_observe_mixed_frames():
    """ROS 与业务线程并行时，时间戳和位置来自同一次整帧更新。"""
    state = RgbdState(max_age_s=60)
    barrier = threading.Barrier(2)
    errors = []

    def writer():
        """连续写入可按 stamp 核对的帧。"""
        barrier.wait()
        for value in range(1, 501):
            state.update(message(stamp=value, rows=[candidate(float(value))]))
            if value % 20 == 0:
                time.sleep(0)

    worker = threading.Thread(target=writer)
    worker.start()
    barrier.wait()
    for _ in range(1000):
        snapshot = state.snapshot()
        if snapshot.candidates and snapshot.candidates[0].position_m[0] != snapshot.stamp_ns:
            errors.append(snapshot)
        time.sleep(0)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert not errors


def test_export_complete_hashed_and_models_separate(tmp_path):
    """交付包含运行依赖和校验和，模型与源码分开，无缓存和离线工具。"""
    model_dir = tmp_path/'models'
    model_dir.mkdir()
    (model_dir/'yolov8n_mix4.onnx').write_bytes(b'test asset, not a runnable model')
    (model_dir/'yolov8n_mix4.names.json').write_text('{"0":"block","1":"top"}')
    out = export_bundle(tmp_path/'handoff', model_dir)
    manifest = json.loads((out/'MANIFEST.json').read_text())
    assert manifest['driverlogic_branch'] == 'driverLogic_2027'
    for row in manifest['files']:
        path = out/row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        assert not any(p in path.relative_to(out).parts for p in ('__pycache__', 'lab', 'bags', 'output'))
    for relative in ('runtime/vision/config/loader.py', 'runtime/vision/aruco/pose.py',
                     'runtime/ros2_ws/src/vision_depth/vision_depth/launch_transport.py',
                     'runtime/ros2_ws/src/vision_interfaces/CMakeLists.txt',
                     'runtime/ros2_ws/src/vision_depth/resource/vision_depth',
                     'runtime/COLCON_IGNORE', 'mainlogic/MainLogic/MAIN/rgbd_readonly.py'):
        assert (out/relative).is_file(), relative
    assert len(list((out/'model_assets').iterdir())) == 2
    assert not list((out/'runtime').rglob('*.onnx'))
    before = (out/'MANIFEST.json').read_bytes()
    with pytest.raises(FileExistsError):
        export_bundle(out)
    assert (out/'MANIFEST.json').read_bytes() == before


def test_export_missing_or_mismatched_assets_leaves_no_output(tmp_path):
    """缺少或类别不配套的模型必须在输出前失败。"""
    model_dir = tmp_path/'models'
    model_dir.mkdir()
    with pytest.raises(ValueError, match='model asset'):
        export_bundle(tmp_path/'missing', model_dir)
    (model_dir/'yolov8n_mix4.onnx').write_bytes(b'test')
    (model_dir/'yolov8n_mix4.names.json').write_text('["other"]')
    with pytest.raises(ValueError, match='mix4 names'):
        export_bundle(tmp_path/'wrong', model_dir)
    assert not (tmp_path/'missing').exists()
    assert not (tmp_path/'wrong').exists()


def test_readonly_module_import_has_no_hardware_or_ros_initialization():
    """只读入口可直接导入且不加载串口/底盘模块。"""
    before = set(sys.modules)
    module = importlib.import_module('MainLogic.MAIN.rgbd_readonly')
    assert callable(module.async_main)
    added = set(sys.modules)-before
    assert not any('serial_node' in name or name.endswith('.Move') for name in added)


def test_export_rejects_recursion_into_source():
    """输出位于源树内部时拒绝，避免边复制边递归增长。"""
    with pytest.raises(ValueError, match='copied source tree'):
        export_bundle(ROOT/'vision/this-directory-must-not-be-created')
    assert not (ROOT/'vision/this-directory-must-not-be-created').exists()


def test_export_failure_cleans_only_its_new_directory(tmp_path, monkeypatch):
    """复制中途失败不留下半包，也不触碰既有目录。"""
    import lab.export_driverlogic as exporter
    preserved = tmp_path/'preserved'
    preserved.mkdir()
    (preserved/'note').write_text('keep')

    def fail_copy(source, target):
        """模拟中途的磁盘写入错误。"""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('partial')
        raise OSError('test disk failure')

    monkeypatch.setattr(exporter, '_copy', fail_copy)
    with pytest.raises(OSError, match='disk failure'):
        exporter.export_bundle(tmp_path/'failed')
    assert not (tmp_path/'failed').exists()
    assert (preserved/'note').read_text() == 'keep'
