"""侧面候选 ROS 发布：消息语义和失效清除。"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def observation(xyz, name='block', status='CANDIDATE', normal=(0, 0, -1), color=None, class_id=None):
    """创建一个选面结果。"""
    return {'status': status, 'detection': {'class_name': name, 'score': 0.75,
                                           'class_id': (1 if name == 'top' else 0) if class_id is None else class_id,
                                           'color': color},
            'measurement': {'selected_index': 0, 'candidates': [{'centroid_mm': xyz, 'normal': normal}]}}


def rotated_z(q):
    """按 ROS 四元数约定恢复局部 +Z 的光学系方向。"""
    return (2*(q.x*q.z + q.w*q.y), 2*(q.y*q.z - q.w*q.x),
            1 - 2*(q.x*q.x + q.y*q.y))


@pytest.mark.parametrize('normal', [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                                    (0, 0, 1), (0, 0, -1), (1, 2, -3)])
def test_quaternion_all_axis_directions(normal):
    """几何转换对正反三个轴和斜向均能还原局部 +Z。"""
    from vision_depth.geometry import quaternion_from_z_axis
    x, y, z, w = quaternion_from_z_axis(normal)
    expected = np.array(normal) / np.linalg.norm(normal)
    assert rotated_z(SimpleNamespace(x=x, y=y, z=z, w=w)) == pytest.approx(expected, abs=1e-6)


def result(rows, received=10.0):
    """构造包含原始深度时间和 frame 的已完成图对。"""
    return {'received': received, 'pair': {'depth_stamp_ns': 12_345_678_901},
            'di': {'frame_id': 'camera_color_optical_frame'}, 'observations': rows}


def test_multibox_fields_and_markers():
    """多框分别发布，单位、header、类别与标记来自同一结果。"""
    node = Node()
    publisher = CandidatePublisher(node, 500)
    publisher.publish(result([observation([100, -20, 500], color='blue'),
                              observation([-40, 50, 700], 'top', color='mixed')]), 10.1)
    array = publisher.detections.messages[-1]
    assert (array.header.stamp.sec, array.header.stamp.nanosec, array.header.frame_id) == (12, 345678901, 'camera_color_optical_frame')
    assert len(array.candidates) == 2
    assert [(c.class_id, c.color_id) for c in array.candidates] == [(0, 2), (1, 3)]
    for item, point, name in zip(array.candidates, [(0.1, -0.02, 0.5), (-0.04, 0.05, 0.7)], ['block', 'top']):
        det = item.detection
        pose = det.results[0].pose.pose
        assert (pose.position.x, pose.position.y, pose.position.z) == point
        assert det.header == array.header and det.id == ''
        assert det.results[0].hypothesis.class_id == name
        assert det.results[0].hypothesis.score == 0.75
        assert rotated_z(pose.orientation) == pytest.approx((0, 0, -1), abs=1e-6)
        assert det.bbox.center.orientation.w == 1.0
        assert det.bbox.size.x == det.bbox.size.y == det.bbox.size.z == 0
    markers = publisher.markers.messages[-1].markers
    assert len(markers) == 7 and markers[0].action == markers[0].DELETEALL
    assert markers[1].pose.position.z == .5 and markers[1].lifetime.nanosec == 500_000_000
    assert 'CANDIDATE|top|class=1|color=3|p=' in markers[5].text
    assert 'n=(0.000,0.000,-1.000)' in markers[5].text
    assert not any(char.isspace() for char in markers[5].text)
    for index, item in enumerate(array.candidates):
        det = item.detection
        arrow = markers[3+3*index]
        start, end = arrow.points
        pos = det.results[0].pose.pose.position
        assert (start.x, start.y, start.z) == pytest.approx((pos.x, pos.y, pos.z))
        assert tuple((b-a)/.1 for a, b in zip((start.x, start.y, start.z),
                                             (end.x, end.y, end.z))) == pytest.approx(
                                                 rotated_z(det.results[0].pose.pose.orientation), abs=1e-6)
    assert len({m.id for m in markers[1:]}) == 6


def test_rviz_label_compacts_whitespace_in_class_name():
    """类别名包含空白时，标签仍使用紧凑分隔符且话题类别不变。"""
    publisher = CandidatePublisher(Node(), 500)
    publisher.publish(result([observation([1, 2, 300], name='box  top')]), 10)
    assert publisher.detections.messages[-1].candidates[0].detection.results[0].hypothesis.class_id == 'box  top'
    label = publisher.markers.messages[-1].markers[2].text
    assert '|box_top|' in label and not any(char.isspace() for char in label)


@pytest.mark.parametrize('color,color_id', [('red', 1), ('blue', 2), ('green', 0),
                                         ('yellow', 0), ('cyan', 0), ('magenta', 0),
                                         ('mixed', 3), (None, 0), ('unmapped', 0)])
def test_color_numbers_and_model_class_survive_serialization(color, color_id):
    """固定颜色编号与模型原始索引经过真实 ROS 序列化仍保持对应。"""
    from rclpy.serialization import serialize_message, deserialize_message
    from vision_interfaces.msg import RgbdCandidateArray
    publisher = CandidatePublisher(Node(), 500)
    publisher.publish(result([observation([1, 2, 300], color=color, class_id=42)]), 10)
    message = publisher.detections.messages[-1]
    copy = deserialize_message(serialize_message(message), RgbdCandidateArray)
    assert copy == message
    assert copy.candidates[0].class_id == 42
    assert copy.candidates[0].color_id == color_id
    assert copy.candidates[0].detection.results[0].hypothesis.class_id == 'block'


def test_extended_candidates_ros_subscription_and_expiry():
    """真实 ROS 订阅能收到同一候选的类别、颜色、位置，以及过期空数组。"""
    import time
    import rclpy
    from vision_interfaces.msg import RgbdCandidateArray
    context = rclpy.context.Context()
    rclpy.init(context=context)
    node = rclpy.create_node('test_candidate_attributes', context=context)
    executor = rclpy.executors.SingleThreadedExecutor(context=context)
    executor.add_node(node)
    received = []
    subscription = node.create_subscription(RgbdCandidateArray, '/rgbd_side/candidates', received.append, 1)
    publisher = CandidatePublisher(node, 500)
    try:
        deadline = time.monotonic()+5
        while publisher.detections.get_subscription_count() < 1 and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.02)
        assert publisher.detections.get_subscription_count() >= 1
        publisher.publish(result([observation([100, 200, 500], color='red')]), 10)
        while not received and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.02)
        assert len(received) == 1
        item = received[0].candidates[0]
        assert (item.class_id, item.color_id) == (0, 1)
        assert item.detection.results[0].pose.pose.position.z == .5
        publisher.expire(10.501)
        while len(received) < 2 and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.02)
        assert len(received) == 2 and not received[-1].candidates
        assert received[-1].header == received[0].header
    finally:
        executor.shutdown()
        node.destroy_subscription(subscription)
        node.destroy_node()
        context.shutdown()


@pytest.mark.parametrize('normal', [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                                    (0, 0, 1), (0, 0, -1), (1, 2, -3)])
def test_normal_quaternion_direction(normal):
    """正反轴向与斜向法向按相机方向翻转，四元数正确恢复方向。"""
    publisher = CandidatePublisher(Node(), 500)
    point = (100, 200, 300)
    publisher.publish(result([observation(point, normal=normal)]), 10)
    actual = rotated_z(publisher.detections.messages[-1].candidates[0].detection.results[0].pose.pose.orientation)
    length = math.hypot(*normal)
    expected = [n / length for n in normal]
    if sum(n*p for n, p in zip(expected, point)) > 0:
        expected = [-n for n in expected]
    assert actual == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize('normal', [None, (0, 0, 0), (0, 0, 1e-9),
                                    (math.nan, 0, 1), (math.inf, 0, 1), (1, 2)])
def test_invalid_normal_drops_only_its_candidate(normal):
    """非法法向不发布，其余候选和对应箭头不受影响。"""
    publisher = CandidatePublisher(Node(), 500)
    publisher.publish(result([observation([1, 2, 300], normal=normal),
                              observation([1, 2, 400], name='top')]), 10)
    assert len(publisher.detections.messages[-1].candidates) == 1
    assert publisher.detections.messages[-1].candidates[0].detection.results[0].hypothesis.class_id == 'top'
    assert len(publisher.markers.messages[-1].markers) == 4
    assert publisher.counts['invalid_normal'] == 1


def test_missing_normal_replaces_previous_candidate_with_empty_result():
    """法向字段缺失时撤销前一帧的候选和箭头，不伪造单位姿态。"""
    publisher = CandidatePublisher(Node(), 500)
    publisher.publish(result([observation([1, 2, 300])]), 10)
    missing = observation([1, 2, 300])
    del missing['measurement']['candidates'][0]['normal']
    publisher.publish(result([missing], 10.1), 10.1)
    assert not publisher.detections.messages[-1].candidates
    assert [m.action for m in publisher.markers.messages[-1].markers] == [3]
    assert not publisher.active and publisher.counts['invalid_normal'] == 1


def test_empty_invalid_expiry_and_recovery():
    """无候选、非法坐标及过期不会保留旧标记，恢复时可重新发布。"""
    publisher = CandidatePublisher(Node(), 500)
    publisher.publish(result([observation([1, 2, 300])]), 10.0)
    publisher.expire(10.5)
    assert len(publisher.detections.messages) == 1
    publisher.expire(10.501)
    assert not publisher.detections.messages[-1].candidates
    assert publisher.detections.messages[-1].header.stamp.sec == 12
    assert publisher.markers.messages[-1].markers[0].action == 3
    publisher.expire(11.0)
    assert len(publisher.detections.messages) == 2
    publisher.publish(result([observation([math.nan, 2, 3]), observation([1, 2, -1]),
                              observation([1, 2, 3], status='AMBIGUOUS')], 11), 11)
    assert not publisher.detections.messages[-1].candidates
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
