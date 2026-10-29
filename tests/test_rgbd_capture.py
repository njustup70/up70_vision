"""ROS 外壳回归：本次 recorder 就绪与实际落盘覆盖，不需要相机。"""
import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
pytest.importorskip('rosbag2_py')
module_path = Path(__file__).resolve().parents[1]/'ros2_ws/src/vision_depth/vision_depth/rgbd_capture.py'
spec = importlib.util.spec_from_file_location('capture_test_module', module_path)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def test_waits_for_own_recorder_all_four_subscriptions(monkeypatch):
    topics = {role: '/'+role for role in ('color','depth','color_info','depth_info')}
    attempts = []
    fake = SimpleNamespace(
        get_subscriptions_info_by_topic=lambda topic: [SimpleNamespace(node_name='other_recorder')]
        + ([SimpleNamespace(node_name='my_recorder')] if attempts else []),
        get_logger=lambda: SimpleNamespace(error=lambda message: None))
    monkeypatch.setattr(capture.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(capture.rclpy, 'spin_once', lambda *a, **k: attempts.append('spin'))
    proc = SimpleNamespace(poll=lambda: None)
    assert capture.RgbdCapture._wait_recorder_ready(fake, proc, 'my_recorder', topics, 1)
    assert attempts == ['spin']


def test_exited_recorder_is_not_ready(monkeypatch):
    monkeypatch.setattr(capture.rclpy, 'ok', lambda: True)
    fake = SimpleNamespace(get_logger=lambda: SimpleNamespace(error=lambda message: None))
    assert not capture.RgbdCapture._wait_recorder_ready(fake, SimpleNamespace(poll=lambda: 1),
                                                       'gone', {'color':'/c'}, 1)


def test_native_bag_coverage_checks_actual_written_times(tmp_path):
    import rosbag2_py
    path = tmp_path/'bag'
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    topics = {'color': '/c', 'depth': '/d'}
    for name in topics.values():
        writer.create_topic(rosbag2_py.TopicMetadata(id=0, name=name, type='sensor_msgs/msg/Image',
                                                    serialization_format='cdr'))
    for name, times in [('/c',[0,10**9,6*10**9]),('/d',[0,10**9,4*10**9])]:
        for t in times:
            writer.write(name, b'\x00\x01\x00\x00', t)
    del writer
    result = capture.RgbdCapture._recorded_coverage(None, path, topics,
        {'bag_storage':'mcap','warmup_s':1.,'duration_s':5.})
    assert result['status'] == 'FAIL'
    assert result['streams']['color']['missing_end_s'] == 0
    assert result['streams']['depth']['missing_end_s'] == 2
