"""将可见平面候选转成 ROS 消息，并在失效时撤销。

用大白话说
    每个处理好的图对只发一次坐标；图断了或结果过期时清空显示。
    坐标是相机光学系中可见平面点的重心，不是物体姿态或抓取指令。
"""
from collections import Counter
from copy import deepcopy
import math

from builtin_interfaces.msg import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray


class CandidatePublisher:
    """发布候选及其标记，并按接收后年龄管理有效期。

    :param node: 拥有两个 ROS 发布器的节点
    :param max_age_ms: 图对接收后的有效时间，毫秒
    """

    def __init__(self, node, max_age_ms):
        """建立不保留历史的可靠话题，计数供运行摘要使用。"""
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.detections = node.create_publisher(Detection3DArray, '/rgbd_side/candidates', qos)
        self.markers = node.create_publisher(MarkerArray, '/rgbd_side/markers', qos)
        self.max_age = max_age_ms / 1000.0
        self.header = None
        self.received = None
        self.active = False
        self.counts = Counter()

    def publish(self, result, now):
        """发布一对新结果；若已过期，只撤销上次有效结果。"""
        if now - result['received'] > self.max_age:
            self.clear()
            return
        stamp = result['pair']['depth_stamp_ns']
        header = Header()
        header.stamp.sec, header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        header.frame_id = result['di']['frame_id']
        array = Detection3DArray()
        array.header = header
        marker_array = MarkerArray()
        marker_array.markers.append(self._delete_all(header))
        for row in result['observations']:
            if row['status'] != 'CANDIDATE':
                continue
            measurement = row['measurement']
            if not measurement or measurement['selected_index'] is None:
                continue
            xyz = measurement['candidates'][measurement['selected_index']]['centroid_mm']
            if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz) or xyz[2] <= 0:
                continue
            point = [float(v) / 1000.0 for v in xyz]
            detection = Detection3D()
            detection.header = header
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = row['detection']['class_name']
            hypothesis.hypothesis.score = float(row['detection']['score'])
            hypothesis.pose.pose.position.x, hypothesis.pose.pose.position.y, hypothesis.pose.pose.position.z = point
            hypothesis.pose.pose.orientation.w = 1.0
            detection.results.append(hypothesis)
            detection.bbox.center.position.x, detection.bbox.center.position.y, detection.bbox.center.position.z = point
            detection.bbox.center.orientation.w = 1.0
            array.detections.append(detection)
            index = len(array.detections) - 1
            marker_array.markers.extend(self._candidate_markers(header, index, point, row['detection']['class_name']))
        self.detections.publish(array)
        self.markers.publish(marker_array)
        self.header = deepcopy(header)
        self.received = result['received']
        self.active = bool(array.detections)
        self.counts['published'] += 1
        self.counts['candidates'] += len(array.detections)

    def expire(self, now):
        """没有新帧时按单调时钟检查过期；仅撤销一次。"""
        if self.active and now - self.received > self.max_age:
            self.clear()

    def clear(self):
        """以最后采集时间发空数组并清除标记；没有活动结果时不重复发。"""
        if not self.active:
            return
        array = Detection3DArray()
        array.header = self.header
        markers = MarkerArray()
        markers.markers.append(self._delete_all(self.header))
        self.detections.publish(array)
        self.markers.publish(markers)
        self.active = False
        self.counts['cleared'] += 1

    @staticmethod
    def _delete_all(header):
        """构造此专用标记话题上的全清消息。"""
        marker = Marker()
        marker.header = header
        marker.action = Marker.DELETEALL
        return marker

    def _candidate_markers(self, header, index, point, class_name):
        """创建球与文字；生命周期提供进程意外退出后的回收。"""
        lifetime_ns = max(1, round(self.max_age * 1_000_000_000))
        result = []
        for offset, kind in enumerate((Marker.SPHERE, Marker.TEXT_VIEW_FACING)):
            marker = Marker()
            marker.header = header
            marker.ns = 'rgbd_side_candidate'
            marker.id = 2 * index + offset
            marker.type = kind
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = point
            marker.pose.orientation.w = 1.0
            marker.lifetime = Duration(sec=lifetime_ns // 1_000_000_000,
                                       nanosec=lifetime_ns % 1_000_000_000)
            marker.color.a = 1.0
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.8, 0.0
            if kind == Marker.SPHERE:
                marker.scale.x = marker.scale.y = marker.scale.z = 0.025
            else:
                marker.pose.position.y -= 0.035
                marker.scale.z = 0.035
                marker.text = f'CANDIDATE {class_name} {point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f} m'
            result.append(marker)
        return result
