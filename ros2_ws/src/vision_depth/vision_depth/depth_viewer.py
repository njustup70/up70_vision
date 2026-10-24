"""depth_viewer：OpenCV 窗口实时看深度图和彩色图。

用大白话说
    起一个窗口，左边彩色图、右边深度图（伪彩色），中间打印中心点距离、
    有效像素比例和帧率。看真实深度图长什么样、找失效模式（反光、飞点、
    边缘缺失）就靠它——计划书 S2 任务。

为什么不用 cv_bridge
    手动解析 `sensor_msgs/Image` 就够了：16UC1 直接 `np.frombuffer`，
    bgr8/rgb8 reshape 成 HxWx3。少一个依赖，少一类「版本不匹配」的报错。

放哪 / 为什么在这
    它需要 rclpy，按 ROS2 接入方案 7.1 的判据放在 ROS 包里，不放 `lab/`。

已知的坑
    · 深度图里 0 是**无效值**，不是「距离 0」。伪彩色时单独涂黑，
      否则会被当成最近距离，看起来像一团深蓝的假目标。
    · 渲染在主线程、回调只缓存最新帧（`spin_once` 轮询），不在回调里
      `imshow`：cv2 的窗口不是线程安全的，多线程调用会随机崩。
    · 量程默认 1000mm。安装高度 0.5–0.7m，超过量程的区域会全红，
      用 `--ros-args -p max_depth_mm:=1500` 调。
"""

import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_depth.fuse_node import sensor_qos


def _to_depth_mm(msg: Image) -> np.ndarray | None:
    """16UC1 深度图 -> (H, W) 毫米数组。"""
    if msg.encoding not in ("16UC1", "mono16"):
        return None
    return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)


def _to_bgr(msg: Image) -> np.ndarray | None:
    """彩色图 -> (H, W, 3) BGR 数组。"""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding == "bgr8":
        return arr.reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        return arr.reshape(msg.height, msg.width, 3)[:, :, ::-1]
    return None


def _colorize(depth_mm: np.ndarray, max_mm: int) -> np.ndarray:
    """深度图 -> 伪彩色 BGR。无效值（0）涂黑。"""
    valid = depth_mm > 0
    clipped = np.clip(depth_mm.astype(np.float32), 0, max_mm) / float(max_mm)
    color = cv2.applyColorMap((clipped * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color[~valid] = 0
    return color


class DepthViewer(Node):
    """只缓存最新一帧，渲染交给主线程。"""

    def __init__(self):
        super().__init__("depth_viewer")

        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("max_depth_mm", 1000)

        self._depth = None
        self._color = None
        self._frames = 0
        self._t0 = time.time()

        self.create_subscription(
            Image, self.get_parameter("depth_topic").value, self._on_depth, sensor_qos(1)
        )
        self.create_subscription(
            Image, self.get_parameter("color_topic").value, self._on_color, sensor_qos(1)
        )

    def _on_depth(self, msg: Image) -> None:
        self._depth = msg
        self._frames += 1

    def _on_color(self, msg: Image) -> None:
        self._color = msg

    @property
    def max_mm(self) -> int:
        return int(self.get_parameter("max_depth_mm").value)

    def fps(self) -> float:
        dt = time.time() - self._t0
        return self._frames / dt if dt > 0.1 else 0.0


def main(args=None):
    rclpy.init(args=args)
    node = DepthViewer()
    node.get_logger().info(
        f"depth_viewer 起来了，量程 {node.max_mm}mm。按 q 退出。"
    )

    cv2.namedWindow("color", cv2.WINDOW_NORMAL)
    cv2.namedWindow("depth", cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)

            depth_msg = node._depth
            if depth_msg is not None:
                depth_mm = _to_depth_mm(depth_msg)
                if depth_mm is not None:
                    if node._frames % 30 == 0:
                        h, w = depth_mm.shape
                        center = depth_mm[h // 2, w // 2]
                        valid_ratio = float((depth_mm > 0).mean())
                        node.get_logger().info(
                            f"中心 {center}mm  有效像素 {valid_ratio:.1%}  "
                            f"帧率 {node.fps():.1f}Hz"
                        )
                    cv2.imshow("depth", _colorize(depth_mm, node.max_mm))

            if node._color is not None:
                bgr = _to_bgr(node._color)
                if bgr is not None:
                    cv2.imshow("color", bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
