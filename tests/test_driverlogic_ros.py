"""通过真实 ROS 序列化和节点验证适配器，无相机、无串口。"""
import asyncio
import importlib
from pathlib import Path
import sys
import time
from types import ModuleType

import pytest

pytest.importorskip('rclpy')
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from vision_interfaces.msg import RgbdCandidate, RgbdCandidateArray
from vision_msgs.msg import ObjectHypothesisWithPose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'integration/driverlogic'))
from MainLogic.adapters.rgbd import RgbdAdapter


def array(empty=False):
    """构造真实自定义消息，使用非默认的类别与混色编号。"""
    msg = RgbdCandidateArray()
    msg.header.frame_id = 'camera_color_optical_frame'
    msg.header.stamp.sec, msg.header.stamp.nanosec = 12, 345678901
    if not empty:
        item = RgbdCandidate()
        item.class_id, item.color_id = 1, 3
        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id, result.hypothesis.score = 'top', .8
        result.pose.pose.position.x, result.pose.pose.position.z = .12, .6
        result.pose.pose.orientation.x = 1.
        result.pose.pose.orientation.w = 0.
        item.detection.results = [result]
        msg.candidates = [item]
    return msg


def test_real_topic_receive_empty_expire_and_close():
    """真实消息接收保持单位与属性，撤销、过期、重启和关闭均生效。"""
    context = Context()
    rclpy.init(context=context)
    node = Node('handoff_test', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    adapter = RgbdAdapter(node, topic='/handoff_test/candidates')
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.VOLATILE)
    publisher = node.create_publisher(RgbdCandidateArray, '/handoff_test/candidates', qos)

    def deliver(msg, expected):
        """等待发现及序列化传递，最长五秒；不用固定 sleep 猜发现时间。"""
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            publisher.publish(msg)
            executor.spin_once(timeout_sec=.05)
            if adapter.snapshot().status == expected:
                return adapter.snapshot()
        pytest.fail(f'no {expected} result; got {adapter.snapshot()}')

    try:
        snapshot = deliver(array(), 'active')
        assert snapshot.stamp_ns == 12_345_678_901
        assert snapshot.frame_id == 'camera_color_optical_frame'
        item = snapshot.candidates[0]
        assert (item.class_id, item.color_id, item.class_name) == (1, 3, 'top')
        assert item.position_m == pytest.approx((.12, 0., .6))
        assert item.normal == pytest.approx((0., 0., -1.))
        assert not deliver(array(empty=True), 'empty').candidates
        deliver(array(), 'active')
        # 消息完全停止后，snapshot 按接收端单调时钟撤销旧目标。
        deadline = time.monotonic()+.7
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.02)
        assert adapter.snapshot().status == 'stale'
        assert not adapter.snapshot().candidates
        assert deliver(array(), 'active').candidates
        adapter.close()
        adapter.close()
        assert adapter.snapshot().status == 'closed'
        assert context.ok()
    finally:
        adapter.close()
        executor.shutdown()
        node.destroy_node()
        context.shutdown()


def test_readonly_entry_uses_existing_node_and_cleans_subscription(monkeypatch, capsys):
    """实际只读协程复用现有节点；取消后撤销订阅，不启动其他进程。"""
    context = Context()
    rclpy.init(context=context)
    node = Node('readonly_entry_test', context=context)
    link = ModuleType('MainLogic.link')
    bridge = ModuleType('MainLogic.link.ros_bridge_node')
    bridge.RosBridgeNodeInstance = node
    monkeypatch.setitem(sys.modules, 'MainLogic.link', link)
    monkeypatch.setitem(sys.modules, 'MainLogic.link.ros_bridge_node', bridge)
    module = importlib.import_module('MainLogic.MAIN.rgbd_readonly')
    adapters = []

    def factory(existing_node):
        """保留实际适配器，用于验证 coroutine 退出清理。"""
        assert existing_node is node
        adapter = RgbdAdapter(existing_node)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(module, 'RgbdAdapter', factory)

    async def exercise():
        """只执行只读入口，不导入默认运动入口。"""
        task = asyncio.create_task(module.async_main())
        await asyncio.sleep(.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(exercise())
        assert len(adapters) == 1
        assert adapters[0].snapshot().status == 'closed'
        assert 'RGBD status=waiting' in capsys.readouterr().out
    finally:
        node.destroy_node()
        context.shutdown()
