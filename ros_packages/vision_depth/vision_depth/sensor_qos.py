"""相机图像和内参的传感器 QoS。"""

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy


def sensor_qos(depth: int = 5) -> QoSProfile:
    """best effort、volatile、keep last；队列深度由调用方指定。"""
    return QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                      durability=QoSDurabilityPolicy.VOLATILE,
                      history=QoSHistoryPolicy.KEEP_LAST, depth=depth)
