"""共享图像解码和报告读取的边界及兼容入口。"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ros2_ws/src/vision_depth'))
from vision_depth.image_decode import image_array
from vision_depth.side_pipeline import image_array as old_image_array
from lab.rgbd_check.report_reader import load_report
from lab.manual_rgbd_fusion import load_report as old_load_report


@pytest.mark.parametrize('encoding,big', [('16UC1', False), ('16UC1', True), ('mono16', False), ('mono16', True)])
def test_depth_encoding_padding_endian(encoding, big):
    """深度编码和行填充不改变毫米数值。"""
    raw = b''.join(np.array(row, dtype='>u2' if big else '<u2').tobytes() + b'xx' for row in [[1, 256], [1024, 65535]])
    msg = SimpleNamespace(encoding=encoding, height=2, width=2, step=6, data=raw, is_bigendian=big)
    for decode in (image_array, old_image_array):
        assert decode(msg, True).tolist() == [[1, 256], [1024, 65535]]


@pytest.mark.parametrize('encoding,expected', [('rgb8', [3, 2, 1]), ('bgr8', [1, 2, 3])])
def test_color_encoding_padding(encoding, expected):
    """彩色编码输出 BGR 并跳过行填充。"""
    msg = SimpleNamespace(encoding=encoding, height=2, width=1, step=5, data=b'\x01\x02\x03xx'*2, is_bigendian=0)
    assert image_array(msg).tolist() == [[expected], [expected]]


@pytest.mark.parametrize('change', [{'encoding': 'bad'}, {'step': 1}, {'data': b'\x00'}, {'height': 0}])
def test_bad_image(change):
    """非法编码、行跨度、尺寸和数据不足均拒绝。"""
    fields = dict(encoding='16UC1', height=1, width=2, step=4, data=b'\0'*4, is_bigendian=0)
    fields.update(change)
    with pytest.raises(ValueError):
        image_array(SimpleNamespace(**fields), True)


def test_report_compatibility(tmp_path):
    """原路径和公共入口返回同一报告，失败条件一致。"""
    info = {'frame_id': 'camera', 'distortion_model': 'plumb_bob', 'd': [0.0]*5, 'width': 2, 'height': 1}
    summary = {'capture': {'mode': 'aligned'}, 'capture_integrity': {'status': 'PASS'}, 'sync': {'status': 'PASS'},
               'selected': {'pair_id': 'pair'}, 'geometry': {'camera_info': {'color_info': info, 'depth_info': info}}}
    (tmp_path/'selected').mkdir()
    np.savez(tmp_path/'selected/pair.npz', color=np.zeros((1, 2, 3), np.uint8), depth=np.ones((1, 2), np.uint16))
    path = tmp_path/'summary.json'
    path.write_text(json.dumps(summary))
    assert old_load_report is load_report
    result = load_report(tmp_path)
    assert result[0] == summary and result[2].shape == (1, 2, 3) and result[3].tolist() == [[1, 1]]
    for section, key, value in [('capture', 'mode', 'raw'), ('sync', 'status', 'FAIL'), ('selected', 'pair_id', '../bad')]:
        original = summary[section][key]
        summary[section][key] = value
        path.write_text(json.dumps(summary))
        with pytest.raises(ValueError):
            load_report(tmp_path)
        summary[section][key] = original


def test_ros_compatibility_and_qos():
    """ROS 旧入口保持错误处理和每个相机队列的 QoS。"""
    pytest.importorskip('rclpy')
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
    from vision_depth import depth_viewer, fake_camera_node, fuse_node
    from vision_depth.sensor_qos import sensor_qos
    msg = SimpleNamespace(encoding='16UC1', height=1, width=1, step=4,
                          data=b'\x01\x02xx', is_bigendian=True)
    assert fuse_node.image_to_depth_array(msg).tolist() == [[258]]
    assert depth_viewer._to_depth_mm(msg).tolist() == [[258]]
    msg.encoding = 'bad'
    assert depth_viewer._to_depth_mm(msg) is None
    assert depth_viewer._to_bgr(msg) is None
    with pytest.raises(ValueError):
        fuse_node.image_to_depth_array(msg)
    msg.encoding, msg.data = '16UC1', b'\0'
    with pytest.raises(ValueError):
        depth_viewer._to_depth_mm(msg)
    for factory in (sensor_qos, fuse_node.sensor_qos, fake_camera_node.sensor_qos):
        for depth in (1, 2, 5):
            qos = factory(depth)
            assert (qos.depth, qos.reliability, qos.durability, qos.history) == (
                depth, ReliabilityPolicy.BEST_EFFORT, DurabilityPolicy.VOLATILE, HistoryPolicy.KEEP_LAST)
