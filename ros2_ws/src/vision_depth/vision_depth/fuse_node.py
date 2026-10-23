"""fuse_node：深度相机抓取检测的 ROS 外壳。

用大白话说
    相机节点把深度图发到 ROS 上，这个节点收下来、交给算法、再把「吸附点 + 法向」
    发出去给控制组。它自己不算任何东西——算法住在 `vision/depth/core.py`。

放什么 / 不放什么
    只放三类代码：收消息、调 `vision.depth.core`、发消息。
    **算法一行都不在这里**。ROS2 接入方案第七节写了判据：本文件超过 200 行
    就说明算法漏进来了。

当前状态（落地顺序 S1 第 3 步）
    **故意先发空数组。** 算法一行没写的时候，先验证话题链路、频率、消息格式全对，
    把「接口问题」和「算法问题」彻底分开。这个顺序能省掉后面最难查的一类故障。
    等 S2（算法进了 `vision/depth/`）之后，只改一个 import 就接上真算法（S3）。

依赖谁 / 被谁用
    上游：`orbbec_camera`（`/camera/depth/image_raw` + `camera_info`）
    下游：控制组（`/grasp_targets`，`vision_msgs/Detection3DArray`）
    不 import `vision.depth`——现在还没有那个模块（S0 阶段原型在 `lab/`），
    硬 import 会让节点起不来。

已知的坑
    · 深度图用 `best_effort` 的 QoS：相机是 sensor 数据，默认的 `reliable` QoS
      在回调处理慢于发布频率时会堆积，表现为「频率越来越低」。
    · `camera_info` 单独缓存最新的一个，不与深度图做时间同步：内参在一次上电内
      不变，取最新的就够，做同步反而会引入丢帧。
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection3DArray


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


class FuseNode(Node):
    """收深度图 → 调算法 → 发抓取目标。

    第零段走纯几何模式（`use_2d_dets=false`）：不需要 YOLO 的 2D 框，
    不依赖搭档进度。
    """

    def __init__(self):
        super().__init__("fuse_node")

        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("info_topic", "/camera/depth/camera_info")
        self.declare_parameter("output_topic", "/grasp_targets")
        self.declare_parameter("output_frame", "camera_link")
        self.declare_parameter("use_2d_dets", False)

        depth_topic = self.get_parameter("depth_topic").value
        info_topic = self.get_parameter("info_topic").value
        out_topic = self.get_parameter("output_topic").value
        self._out_frame = self.get_parameter("output_frame").value

        self._info = None            # 最新的 camera_info
        self._frames = 0             # 收到的深度帧数
        self._has_info = False

        self.create_subscription(CameraInfo, info_topic, self._on_info, sensor_qos(1))
        self.create_subscription(Image, depth_topic, self._on_depth, sensor_qos(5))
        self._pub = self.create_publisher(Detection3DArray, out_topic, 10)

        self.get_logger().info(
            f"fuse_node 起来了：收 {depth_topic} + {info_topic}，发 {out_topic}（当前发空数组）"
        )

    def _on_info(self, msg: CameraInfo) -> None:
        if not self._has_info:
            self._has_info = True
            self.get_logger().info(
                f"拿到内参：{msg.width}x{msg.height} "
                f"fx={msg.k[0]:.1f} fy={msg.k[4]:.1f} cx={msg.k[2]:.1f} cy={msg.k[5]:.1f}"
            )
        self._info = msg

    def _on_depth(self, msg: Image) -> None:
        """收到一帧深度图。

        当前阶段只统计并转发空数组；接上算法后这里改成
        `core.detect_grasp_targets(depth_mm, intrinsics, params)`。
        """
        self._frames += 1
        if self._frames == 1:
            self.get_logger().info(
                f"收到第一帧深度：{msg.width}x{msg.height} "
                f"encoding={msg.encoding} step={msg.step}"
            )
        if self._frames % 100 == 0:
            self.get_logger().info(f"已收 {self._frames} 帧深度图")

        out = Detection3DArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._out_frame
        self._pub.publish(out)


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
