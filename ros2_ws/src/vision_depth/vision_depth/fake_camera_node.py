"""fake_camera_node：把 S4 生成的合成深度图当作真相机发到 ROS 上。

用大白话说
    B 阶段要「收深度图 -> 找物块 -> 发抓取点」。真机上这一段依赖相机在场，
    但相机不在手边、俯视桌面场景也还没搭起来时，就没法调试。这个节点把
    `lab/gen_synthetic_depth.py` 造的那一帧数据循环发出来，话题、消息格式、
    内参与真相机完全一致，所以下游代码一行都不用改。

放什么 / 不放什么
    只放三类代码：读文件、发消息、定时。**不做任何几何计算**——
    真值已经在 meta.json 里算好了，这里只负责转发。

真值必须走独立话题
    真值发到 `/ground_truth_targets`，检测结果发到 `/grasp_targets`，
    两个话题分开。混成一个的话，拿真值去验证真值，误差永远是 0，
    这个坑比任何算法 bug 都隐蔽。

为什么每帧要重新加噪声
    如果只把 depth.npy 循环发出去，每一帧都是同一张图，帧间完全一致。
    那样时间滤波、滑动平均、跟踪这些逻辑在合成数据上会表现得「完美」，
    一到真机就崩。所以默认从 clean_depth.npy 出发，每帧重新采一次噪声。
    有效的空洞位置保持一致（反光位在真实场景里也是固定的）。

已知的坑
    · 深度用 `16UC1` + `step = width*2`，与 Orbbec 一致。写成 `mono16` 或
      step 填错，下游 reshape 出来会斜，而且图看起来「还有内容」，很难发现。
    · 内参从 meta.json 读，不是从真实相机读。换数据目录就要换个 meta，
      所以 data_dir 必须指到具体某一次生成的结果，不能指到父目录。
    · 深度图 QoS 用 best_effort：下游如果按 reliable 订阅会收不到数据，
      这是 ROS 里 sensor 话题的常见不兼容，报错信息只有一句「不匹配」。
    · 物块尺寸在 meta 里是毫米，ROS 消息里是米。这里统一除以 1000，
      混用会导致 bbox 大一千倍或小一千倍，而位置看起来还是对的。
"""

import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from .geometry import quaternion_from_z_axis

# 与 lab/gen_synthetic_depth.py 保持一致的噪声模型
NOISE_COEF = 3.98e-6
NOISE_EXP = 1.74


def sensor_qos(depth: int = 5) -> QoSProfile:
    """相机数据专用 QoS：best_effort + 只留最近几帧。

    与真相机节点保持一致。用 reliable 时，回调慢于发布频率会堆积，
    表现为「帧率越来越低」，很容易被误判成驱动问题。
    """
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class FakeCameraNode(Node):
    """按固定频率回放一组合成深度数据。"""

    def __init__(self):
        super().__init__("fake_camera_node")

        self.declare_parameter("data_dir", "")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("info_topic", "/camera/depth/camera_info")
        self.declare_parameter("truth_topic", "/ground_truth_targets")
        self.declare_parameter("frame_id", "camera_depth_optical_frame")
        self.declare_parameter("fps", 15.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("add_noise", True)
        self.declare_parameter("noise_seed", 0)

        data_dir = Path(self.get_parameter("data_dir").value)
        if not str(data_dir) or not data_dir.is_dir():
            raise ValueError(
                f"data_dir 必须指到某一次 S4 生成的目录（含 meta.json 和 depth.npy），"
                f"收到 {data_dir!r}。先跑 lab/gen_synthetic_depth.py 生成一份。"
            )

        self._frame_id = self.get_parameter("frame_id").value
        self._loop = bool(self.get_parameter("loop").value)
        self._add_noise = bool(self.get_parameter("add_noise").value)
        self._rng = np.random.default_rng(int(self.get_parameter("noise_seed").value))

        meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
        self._meta = meta
        self._depth = np.load(data_dir / "depth.npy")          # uint16，含噪声与空洞
        self._valid = self._depth > 0

        clean_path = data_dir / "clean_depth.npy"
        self._clean = np.load(clean_path) if clean_path.exists() else None
        if self._clean is None and self._add_noise:
            self.get_logger().warn(
                "没有 clean_depth.npy，逐帧加噪用不了，会直接回放 depth.npy（帧间完全一致）。"
                "重新跑一次 gen_synthetic_depth.py 就会有。"
            )

        # 噪声标准差只跟几何深度有关，预先算好，别在 15Hz 的回调里重算 40 万次幂。
        # 用 float32：这张图每帧要过一遍，float64 白占一倍内存带宽。
        if self._clean is not None:
            self._sigma = np.where(
                self._valid, NOISE_COEF * self._clean ** NOISE_EXP, 0.0
            ).astype(np.float32)
        else:
            self._sigma = None

        height, width = self._depth.shape
        self._info = self._build_info(meta["camera"], height, width)
        self._truth = self._build_truth(meta)

        depth_topic = self.get_parameter("depth_topic").value
        info_topic = self.get_parameter("info_topic").value
        truth_topic = self.get_parameter("truth_topic").value

        self._depth_pub = self.create_publisher(Image, depth_topic, sensor_qos(2))
        self._info_pub = self.create_publisher(CameraInfo, info_topic, sensor_qos(2))
        # 真值不是相机数据，用可靠传输：丢掉一帧真值会让误差统计少一个样本
        self._truth_pub = self.create_publisher(Detection3DArray, truth_topic, 10)

        fps = max(float(self.get_parameter("fps").value), 0.1)
        self._timer = self.create_timer(1.0 / fps, self._tick)
        self._frames = 0

        self.get_logger().info(
            f"fake_camera_node 启动：{data_dir}，"
            f"{width}x{height}@{fps:.1f}Hz，"
            f"物块 {len(self._truth.detections)} 个，"
            f"逐帧加噪={'开' if self._clean is not None and self._add_noise else '关'}"
        )
        self.get_logger().info(
            f"  深度 -> {depth_topic}   内参 -> {info_topic}   真值 -> {truth_topic}"
        )

    def _build_info(self, cam: dict, height: int, width: int) -> CameraInfo:
        """按 meta 里的内参构造 CameraInfo。畸变全部填 0：合成数据没有畸变。"""
        info = CameraInfo()
        info.height = int(height)
        info.width = int(width)
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _build_truth(self, meta: dict) -> Detection3DArray:
        """把 meta.json 里的物块转成真值消息。只算一次，每帧复用（只换时间戳）。"""
        array = Detection3DArray()
        for obj in meta["objects"]:
            detection = Detection3D()
            detection.id = f"gt_{obj['id']}"

            top = np.asarray(obj["top_center_cam_mm"], dtype=np.float64)
            normal = np.asarray(obj["top_normal_cam"], dtype=np.float64)
            size = np.asarray(obj["size_mm"], dtype=np.float64)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(obj["type"])
            hyp.hypothesis.score = 1.0
            # meta 是毫米，ROS 约定米
            hyp.pose.pose.position.x = float(top[0] / 1000.0)
            hyp.pose.pose.position.y = float(top[1] / 1000.0)
            hyp.pose.pose.position.z = float(top[2] / 1000.0)
            qx, qy, qz, qw = quaternion_from_z_axis(normal)
            hyp.pose.pose.orientation.x = qx
            hyp.pose.pose.orientation.y = qy
            hyp.pose.pose.orientation.z = qz
            hyp.pose.pose.orientation.w = qw
            detection.results.append(hyp)

            # 包围盒中心 = 顶面中心沿法向往里退半个高
            center = top - 0.5 * size[2] * normal
            detection.bbox.center.position.x = float(center[0] / 1000.0)
            detection.bbox.center.position.y = float(center[1] / 1000.0)
            detection.bbox.center.position.z = float(center[2] / 1000.0)
            detection.bbox.center.orientation.w = 1.0
            detection.bbox.size.x = float(size[0] / 1000.0)
            detection.bbox.size.y = float(size[1] / 1000.0)
            detection.bbox.size.z = float(size[2] / 1000.0)

            array.detections.append(detection)
        return array

    def _make_depth(self, stamp) -> Image:
        """生成一帧深度图消息。

        噪声用 standard_normal 乘 sigma，而不是 rng.normal(0, sigma)：
        后者在 scale 是数组时要逐元素走一遍分发，40 万像素每次慢一个量级，
        实测帧率从 15Hz 掉到 4.6Hz。standard_normal 走的是同一条快路径。
        """
        if self._clean is not None and self._add_noise:
            pixels = self._clean.astype(np.float32, copy=True)
            noise = self._rng.standard_normal(pixels.shape, dtype=np.float32)
            pixels += noise * self._sigma
            # 空洞位置沿用 depth.npy：真实场景里反光位固定，不该每帧乱变
            pixels[~self._valid] = 0.0
            np.clip(pixels, 1.0, None, out=pixels)
            data = np.round(pixels).astype(np.uint16)
        else:
            data = self._depth

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.height = int(data.shape[0])
        msg.width = int(data.shape[1])
        msg.encoding = "16UC1"
        msg.is_bigendian = 0
        msg.step = int(data.shape[1]) * 2
        msg.data = data.tobytes()
        return msg

    def _tick(self) -> None:
        """定时器回调：发一帧深度 + 内参 + 真值。"""
        stamp = self.get_clock().now().to_msg()

        depth_msg = self._make_depth(stamp)
        self._depth_pub.publish(depth_msg)

        self._info.header.stamp = stamp
        self._info.header.frame_id = self._frame_id
        self._info_pub.publish(self._info)

        # 真值数组构造一次就够，每帧只更新时间戳
        self._truth.header.stamp = stamp
        self._truth.header.frame_id = self._frame_id
        for detection in self._truth.detections:
            detection.header = self._truth.header
        self._truth_pub.publish(self._truth)

        self._frames += 1
        if self._frames == 1:
            self.get_logger().info("已发出第一帧（深度 + 内参 + 真值）")
        if not self._loop and self._frames >= 1:
            self.get_logger().info("loop=false，只发一帧，退出")
            self._timer.cancel()
            raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = FakeCameraNode()
    except (ValueError, FileNotFoundError) as exc:
        print(f"[fake_camera_node] 启动失败：{exc}")
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
