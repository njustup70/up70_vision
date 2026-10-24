"""fuse_node：深度相机抓取检测的 ROS 外壳。

用大白话说
    相机节点把深度图发到 ROS 上，这个节点收下来、交给算法、再把「吸附点 + 法向 +
    可吸性分数」发出去。它自己不算任何东西——算法住在 `vision/depth/core.py`。

放什么 / 不放什么
    只放三类代码：收消息、调 `vision.depth`、发消息。
    **算法一行都不在这里。** ROS2 接入方案第七节写了判据：本文件超过 200 行
    就说明算法漏进来了。

两个话题，别混
    `/grasp_targets`（Detection3DArray）给控制组，装吸附点、朝向、分数、包围盒。
    `/grasp_markers`（MarkerArray）只喂 RViz，画成箭头方便肉眼确认朝向对不对。
    真值另有独立话题 `/ground_truth_targets`，由 fake_camera_node 发——
    真值和检测结果必须分开发，混在一起就成了「用真值验证真值」，误差永远为 0。

箭头怎么画才看得懂
    用 Marker 的 points 模式（起点、终点）而不是 pose 模式。ARROW 的 pose 模式
    默认朝局部 +x，要把 +x 转到法向还得额外构造一次旋转，容易把方向搞反；
    points 模式下起终点直接就是几何位置，歪没歪一眼能看出来。

已知的坑
    · 深度图用 `best_effort` 的 QoS：相机是 sensor 数据，默认的 `reliable` QoS
      在回调处理慢于发布频率时会堆积，表现为「频率越来越低」。
    · `camera_info` 单独缓存最新的一个，不做时间同步：内参在一次上电内不变，
      取最新的就够，做同步反而会引入丢帧。
    · 算法抛 ValueError（尺寸不符、找不到桌面）时只警告一次并跳过这一帧，
      不能让节点崩——场景里本来就可能连续几帧没有可用平面。
    · 输出的位置是**米**。算法内部全程毫米，这里统一除 1000。混用会让
      RViz 里箭头飞到几公里外，或者缩成一个点。
"""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from vision.depth import Intrinsics, SuctionParams, detect_grasp_targets

from .geometry import quaternion_from_z_axis

#: 箭头长度（毫米）。只是给人看的，不参与计算。
ARROW_LENGTH_MM = 50.0


def sensor_qos(depth: int = 5) -> QoSProfile:
    """相机数据专用 QoS：best_effort + 只留最近几帧。

    用 reliable 的话，一旦回调处理慢于发布频率，队列会堆满并反过来拖慢发布端，
    表现是「频率越来越低」，很容易被误判成相机或驱动的问题。
    """
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def image_to_depth_array(msg: Image) -> np.ndarray:
    """sensor_msgs/Image(16UC1) -> (H, W) uint16 毫米数组。

    step 不一定等于 width*2：驱动可能按行对齐补 padding。直接按 width*2 铺开，
    有 padding 时图像会斜着错位，而且错位后「看起来还有内容」，极难发现。
    """
    if msg.encoding not in ("16UC1", "mono16"):
        raise ValueError(f"深度编码是 {msg.encoding!r}，只处理 16UC1（毫米）")

    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    raw = bytes(msg.data)
    row_bytes = width * 2

    if step == row_bytes:
        flat = np.frombuffer(raw, dtype=np.uint16)
    else:
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)[:, :row_bytes]
        flat = np.ascontiguousarray(rows).view(np.uint16)
    return flat.reshape(height, width)


class FuseNode(Node):
    """收深度图 → 调算法 → 发抓取目标与可视化箭头。"""

    def __init__(self):
        super().__init__("fuse_node")

        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("info_topic", "/camera/depth/camera_info")
        self.declare_parameter("output_topic", "/grasp_targets")
        self.declare_parameter("marker_topic", "/grasp_markers")
        self.declare_parameter("output_frame", "camera_link")
        self.declare_parameter("detect_every", 1)
        self.declare_parameter("min_score", 0.0)
        self.declare_parameter("use_2d_dets", False)

        depth_topic = self.get_parameter("depth_topic").value
        info_topic = self.get_parameter("info_topic").value
        out_topic = self.get_parameter("output_topic").value
        marker_topic = self.get_parameter("marker_topic").value
        self._out_frame = self.get_parameter("output_frame").value
        self._detect_every = max(int(self.get_parameter("detect_every").value), 1)
        self._min_score = float(self.get_parameter("min_score").value)
        # use_2d_dets 目前没有意义：第零段走纯几何，不依赖 YOLO。
        # 留着参数是为了接口不随阶段变化，控制组那边不用改。
        if bool(self.get_parameter("use_2d_dets").value):
            self.get_logger().warn("use_2d_dets=true 目前未实现，仍按纯几何检测")

        self._intrinsics = None       # 最新的内参（Intrinsics）
        self._params = SuctionParams()
        self._rng = np.random.default_rng(0)
        self._frames = 0
        self._detections = 0
        self._warned = False

        self.create_subscription(CameraInfo, info_topic, self._on_info, sensor_qos(1))
        self.create_subscription(Image, depth_topic, self._on_depth, sensor_qos(5))
        self._pub = self.create_publisher(Detection3DArray, out_topic, 10)
        self._marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)

        self.get_logger().info(
            f"fuse_node 起来了：收 {depth_topic} + {info_topic}，"
            f"发 {out_topic} + {marker_topic}，每 {self._detect_every} 帧检测一次"
        )

    def _on_info(self, msg: CameraInfo) -> None:
        """缓存内参。只认第一帧的形状，之后不再重建。"""
        if self._intrinsics is None:
            try:
                self._intrinsics = Intrinsics.from_camera_info(msg.k, msg.width, msg.height)
            except ValueError as exc:
                self.get_logger().error(f"内参不合法，检测无法进行：{exc}")
                return
            self.get_logger().info(
                f"拿到内参：{msg.width}x{msg.height} "
                f"fx={self._intrinsics.fx:.1f} fy={self._intrinsics.fy:.1f} "
                f"cx={self._intrinsics.cx:.1f} cy={self._intrinsics.cy:.1f}"
            )

    def _on_depth(self, msg: Image) -> None:
        """收到一帧深度：按节流调算法，然后把结果和箭头一起发出去。"""
        self._frames += 1
        if self._frames % self._detect_every:
            return
        if self._intrinsics is None:
            if self._frames == 1:
                self.get_logger().info("还没收到 camera_info，先跳过检测")
            return

        try:
            depth = image_to_depth_array(msg)
        except ValueError as exc:
            if not self._warned:
                self.get_logger().error(str(exc))
                self._warned = True
            return

        t_start = time.perf_counter()
        try:
            targets = detect_grasp_targets(depth, self._intrinsics,
                                           self._params, self._rng)
        except ValueError as exc:
            # 尺寸不符、找不到桌面等：跳过这一帧，别让节点崩
            if not self._warned:
                self.get_logger().warn(f"这一帧跳过：{exc}")
                self._warned = True
            return
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        targets = [t for t in targets if t.score >= self._min_score]
        header = msg.header
        self._pub.publish(self._make_detections(targets, header))
        self._marker_pub.publish(self._make_markers(targets, header))

        if targets:
            self._detections += 1
        if self._frames % (self._detect_every * 30) == 0:
            best = targets[0].score if targets else 0.0
            self.get_logger().info(
                f"帧 {self._frames}: {len(targets)} 个目标, 最高分 {best:.2f}, "
                f"耗时 {elapsed_ms:.0f}ms"
            )

    def _make_detections(self, targets: list, header) -> Detection3DArray:
        """转成 vision_msgs/Detection3DArray。

        bbox 的中心用吸附点近似：算法返回的是「吸附点 + 包围盒尺寸」，
        没有单独给包围盒中心。包围盒在这里只用于筛选和显示，
        控制组真正要的是 pose，所以不做额外换算。
        """
        array = Detection3DArray()
        array.header = header
        array.header.frame_id = self._out_frame

        for target in targets:
            detection = Detection3D()
            detection.header = array.header
            detection.id = f"target_{target.target_id}"

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = "suction"
            hypothesis.hypothesis.score = float(target.score)
            # 算法给毫米，ROS 约定米
            position = target.position_mm / 1000.0
            hypothesis.pose.pose.position.x = float(position[0])
            hypothesis.pose.pose.position.y = float(position[1])
            hypothesis.pose.pose.position.z = float(position[2])
            qx, qy, qz, qw = quaternion_from_z_axis(target.normal)
            hypothesis.pose.pose.orientation.x = qx
            hypothesis.pose.pose.orientation.y = qy
            hypothesis.pose.pose.orientation.z = qz
            hypothesis.pose.pose.orientation.w = qw
            detection.results.append(hypothesis)

            detection.bbox.center.position.x = float(position[0])
            detection.bbox.center.position.y = float(position[1])
            detection.bbox.center.position.z = float(position[2])
            detection.bbox.center.orientation.w = 1.0
            size = target.bbox_size_mm / 1000.0
            detection.bbox.size.x = float(size[0])
            detection.bbox.size.y = float(size[1])
            detection.bbox.size.z = float(size[2])

            array.detections.append(detection)
        return array

    def _make_markers(self, targets: list, header) -> MarkerArray:
        """画箭头：从吸附点沿法向伸出 ARROW_LENGTH_MM，绿色可吸、红色不可吸。

        每帧先清空再画，否则目标消失后旧箭头会留在 RViz 里，
        看起来像「仍然检测到」，很容易误导。
        """
        array = MarkerArray()

        clear = Marker()
        clear.header = header
        clear.header.frame_id = self._out_frame
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        for target in targets:
            start = target.position_mm / 1000.0
            end = (target.position_mm + target.normal * ARROW_LENGTH_MM) / 1000.0

            marker = Marker()
            marker.header = header
            marker.header.frame_id = self._out_frame
            marker.ns = "grasp_normal"
            marker.id = int(target.target_id)
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.008            # 杆径
            marker.scale.y = 0.016            # 箭头直径
            marker.scale.z = 0.02             # 箭头长度
            if target.score >= 0.5:
                marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
            else:
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            marker.color.a = 1.0
            marker.points = []
            marker.points.append(Point(x=float(start[0]), y=float(start[1]),
                                       z=float(start[2])))
            marker.points.append(Point(x=float(end[0]), y=float(end[1]),
                                       z=float(end[2])))
            array.markers.append(marker)
        return array


def main(args=None):
    rclpy.init(args=args)
    node = FuseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
