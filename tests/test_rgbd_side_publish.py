"""侧面候选 ROS 发布：消息语义和失效清除。"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ros2_ws/src/vision_depth'))
from vision_depth.side_publish import CandidatePublisher


class Publisher:
    """缓存消息，模拟 ROS 发布端。"""
    def __init__(self):
        """初始化保存列表。"""
        self.messages = []

    def publish(self, value):
        """保留消息用于断言。"""
        self.messages.append(value)


class Node:
    """提供发布器工厂。"""
    def __init__(self):
        """记录按话题创建的发布器。"""
        self.pubs = {}

    def create_publisher(self, message_type, topic, qos):
        """记录 QoS 和发布器。"""
        pub = Publisher()
        pub.qos = qos
        self.pubs[topic] = pub
        return pub


def observation(xyz, name='block', status='CANDIDATE'):
    """创建一个选面结果。"""
    return {'status': status, 'detection': {'class_name': name, 'score': 0.75},
            'measurement': {'selected_index': 0, 'candidates': [{'centroid_mm': xyz}]}}


def result(rows, received=10.0):
    """构造包含原始深度时间和 frame 的已完成图对。"""
    return {'received': received, 'pair': {'depth_stamp_ns': 12_345_678_901},
            'di': {'frame_id': 'camera_color_optical_frame'}, 'observations': rows}


def test_multibox_fields_and_markers():
    """多框分别发布，单位、header、类别与标记来自同一结果。"""
    node = Node()
    publisher = CandidatePublisher(node, 500)
    publisher.publish(result([observation([100, -20, 500]), observation([-40, 50, 700], 'top')]), 10.1)
    array = publisher.detections.messages[-1]
    assert (array.header.stamp.sec, array.header.stamp.nanosec, array.header.frame_id) == (12, 345678901, 'camera_color_optical_frame')
    assert len(array.detections) == 2
    for det, point, name in zip(array.detections, [(0.1, -0.02, 0.5), (-0.04, 0.05, 0.7)], ['block', 'top']):
        pose = det.results[0].pose.pose
        assert (pose.position.x, pose.position.y, pose.position.z) == point
        assert det.header == array.header and det.id == ''
        assert det.results[0].hypothesis.class_id == name
        assert det.results[0].hypothesis.score == 0.75
        assert pose.orientation.w == det.bbox.center.orientation.w == 1.0
        assert det.bbox.size.x == det.bbox.size.y == det.bbox.size.z == 0
    markers = publisher.markers.messages[-1].markers
    assert len(markers) == 5 and markers[0].action == markers[0].DELETEALL
    assert markers[1].pose.position.z == .5 and markers[1].lifetime.nanosec == 500_000_000
    assert 'CANDIDATE top' in markers[4].text


def test_empty_invalid_expiry_and_recovery():
    """无候选、非法坐标及过期不会保留旧标记，恢复时可重新发布。"""
    publisher = CandidatePublisher(Node(), 500)
    publisher.publish(result([observation([1, 2, 300])]), 10.0)
    publisher.expire(10.5)
    assert len(publisher.detections.messages) == 1
    publisher.expire(10.501)
    assert not publisher.detections.messages[-1].detections
    assert publisher.detections.messages[-1].header.stamp.sec == 12
    assert publisher.markers.messages[-1].markers[0].action == 3
    publisher.expire(11.0)
    assert len(publisher.detections.messages) == 2
    publisher.publish(result([observation([math.nan, 2, 3]), observation([1, 2, -1]),
                              observation([1, 2, 3], status='AMBIGUOUS')], 11), 11)
    assert not publisher.detections.messages[-1].detections
    publisher.publish(result([observation([1, 2, 3])], 12), 12.6)
    assert len(publisher.detections.messages) == 3
    publisher.publish(result([observation([1, 2, 300])], 13), 13)
    publisher.clear()
    publisher.clear()
    assert publisher.counts == {'published': 3, 'candidates': 2, 'cleared': 2}


def test_step_only_publishes_new_results_and_clears_errors_and_epoch(monkeypatch):
    """循环不重复发布，处理错误和 epoch 改变均撤销结果。"""
    from vision_depth import rgbd_side_viewer as viewer
    class Worker:
        """模拟工作线程返回结果。"""
        def __init__(self):
            """初始化待返回项。"""
            self.ready = None
        def take(self):
            """取走结果一次。"""
            value, self.ready = self.ready, None
            return value
        def invalidate(self):
            """丢弃旧任务。"""
            self.ready = None
    class Sync:
        """模拟空输入。"""
        epoch = 1
        def poll(self, now):
            """没有新图对。"""
            return []
    fake = SimpleNamespace(sync=Sync(), epoch_seen=1, worker=Worker(), last_result=None,
                           stream_ok={'color': False, 'depth': False}, infos={},
                           publisher=CandidatePublisher(Node(), 500), problems={}, stale_seen=False,
                           ages=[], inference_ms=[], geometry_ms=[], counts=__import__('collections').Counter())
    value = result([observation([1, 2, 300])])
    value['timing_ms'] = {'inference': 1, 'geometry': 1}
    fake.worker.ready = {'job': {'pair': SimpleNamespace(epoch=1)}, 'value': value, 'error': None}
    monkeypatch.setattr(viewer.time, 'monotonic', lambda: 10.1)
    viewer.RgbdSideViewer.step(fake)
    viewer.RgbdSideViewer.step(fake)
    assert fake.publisher.counts['published'] == 1
    fake.worker.ready = {'job': {'pair': SimpleNamespace(epoch=1)}, 'value': None, 'error': 'failure'}
    viewer.RgbdSideViewer.step(fake)
    assert fake.publisher.counts['cleared'] == 1
    fake.publisher.publish(value, 10.1)
    fake.sync.epoch = 2
    viewer.RgbdSideViewer.step(fake)
    assert fake.publisher.counts['cleared'] == 2
