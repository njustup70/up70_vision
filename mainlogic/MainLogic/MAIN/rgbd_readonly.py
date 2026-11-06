"""MainLogic 的 RGB-D 只读入口，不初始化串口、不注册速度或机构动作。

通过 python -m MainLogic.Main --main-module rgbd_readonly 加载。
复用 Main.py 已初始化的 ROS 节点和 executor；不要用 R2n_Main 的底盘测试入口验收视觉。
"""
import asyncio

from MainLogic.adapters.rgbd import RgbdAdapter


async def async_main():
    """每秒打印最新快照，退出时清理订阅；不启动相机或第二个 ROS 节点。"""
    from MainLogic.link.ros_bridge_node import RosBridgeNodeInstance

    adapter = RgbdAdapter(RosBridgeNodeInstance)
    try:
        while True:
            snapshot = adapter.snapshot()
            print(f'RGBD status={snapshot.status} frame={snapshot.frame_id!r} '
                  f'stamp_ns={snapshot.stamp_ns} age_s={snapshot.age_s} '
                  f'candidates={snapshot.candidates}', flush=True)
            await asyncio.sleep(1.0)
    finally:
        adapter.close()
