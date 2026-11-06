"""将 RGB-D 候选话题转换为主逻辑可读取的不可变快照。

用大白话说
    ROS 回调整帧替换数据，业务协程调用 snapshot() 读取；锁保护两个线程。
    空数组立即撤销，断流超时后读取不到旧目标。这里不做 TF、不发送动作。
    接口口径见交付分支根 README 的数据与使用约定。
"""
from dataclasses import dataclass
import math
import threading
import time


@dataclass(frozen=True)
class RgbdCandidate:
    """单个可见面候选；位置单位米，法向为相机光学系单位向量。"""

    class_id: int
    color_id: int
    class_name: str
    score: float
    position_m: tuple[float, float, float]
    normal: tuple[float, float, float]


@dataclass(frozen=True)
class RgbdSnapshot:
    """只读整帧快照；stamp_ns 为采集时间，age_s 仅为本地接收后年龄。

    status 为 waiting / active / empty / stale / invalid / closed。
    只有 active 有候选；其余状态的 candidates 均为空元组。
    """

    stamp_ns: int
    frame_id: str
    candidates: tuple[RgbdCandidate, ...]
    age_s: float | None
    status: str


def _candidate(item):
    """解析消息的一项；字段缺失、非有限坐标或退化四元数时抛 ValueError。

    :param item: RgbdCandidate ROS 消息；使用 results[0] 中的位姿
    :return: 不含可变 ROS 对象的候选；不会修改原消息
    """
    if not item.detection.results:
        raise ValueError('candidate.results is empty')
    result = item.detection.results[0]
    p, q = result.pose.pose.position, result.pose.pose.orientation
    position = tuple(float(v) for v in (p.x, p.y, p.z))
    quaternion = tuple(float(v) for v in (q.x, q.y, q.z, q.w))
    score = float(result.hypothesis.score)
    if not all(math.isfinite(v) for v in (*position, *quaternion, score)) or position[2] <= 0:
        raise ValueError('candidate contains invalid coordinates or score')
    length = math.hypot(*quaternion)
    if not math.isfinite(length) or length <= 1e-9:
        raise ValueError('candidate quaternion is degenerate')
    x, y, z, w = (v/length for v in quaternion)
    # 四元数旋转矩阵的第三列，即局部 +Z 在光学系的方向。
    normal = (2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y))
    return RgbdCandidate(int(item.class_id), int(item.color_id),
                         str(result.hypothesis.class_id), score, position, normal)


class RgbdState:
    """与 ROS 无关的线程安全缓存，便于无相机测试。

    max_age_s 默认 0.5，沿用视觉接收后有效期作为交接初值，不代表总链路延迟。
    clock 默认为单调时钟；测试可注入可控时钟。
    """

    def __init__(self, max_age_s=0.5, clock=time.monotonic):
        """建立空缓存；有效期必须为有限正数，否则抛 ValueError。"""
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError(f'max_age_s={max_age_s!r}; expected a finite positive number')
        self._max_age = max_age_s
        self._clock = clock
        self._lock = threading.Lock()
        self._stamp = 0
        self._frame = ''
        self._candidates = ()
        self._received = None
        self._status = 'waiting'
        self._invalid_candidates = 0

    def update(self, msg):
        """整帧替换缓存，空或损坏输入不能保留上一帧目标。

        :param msg: 含 header 和 candidates 的 ROS 消息
        :return: 本帧被跳过的坏候选数；closed 后忽略更新
        """
        with self._lock:
            if self._status == 'closed':
                return 0
            self._received = self._clock()
            self._candidates = ()
            self._stamp, self._frame = 0, ''
            try:
                stamp = msg.header.stamp
                self._stamp = int(stamp.sec)*1_000_000_000+int(stamp.nanosec)
                self._frame = str(msg.header.frame_id)
                if self._stamp <= 0 or not self._frame.strip():
                    raise ValueError('missing acquisition stamp or frame')
                rows = msg.candidates
            except (AttributeError, TypeError, ValueError, OverflowError):
                # 损坏的消息头无法解释坐标系，立即撤销整帧。
                self._status = 'invalid'
                return 0
            candidates = []
            invalid = 0
            for item in rows:
                try:
                    candidates.append(_candidate(item))
                except (AttributeError, TypeError, ValueError, OverflowError):
                    # 坏框不影响同一帧的其他框；通过计数和适配器日志暴露。
                    invalid += 1
            self._invalid_candidates += invalid
            self._candidates = tuple(candidates)
            self._status = 'active' if candidates else ('invalid' if invalid else 'empty')
            return invalid

    def snapshot(self):
        """返回不可变快照；超过有效期时返回空候选，不把采集时间当接收时间。

        :return: RgbdSnapshot；调用方可持有快照，但每次决策前须重新读取
        """
        with self._lock:
            age = None if self._received is None else max(0., self._clock()-self._received)
            status = self._status
            candidates = self._candidates
            if status == 'active' and age > self._max_age:
                status, candidates = 'stale', ()
            return RgbdSnapshot(self._stamp, self._frame, candidates, age, status)

    @property
    def invalid_candidates(self):
        """返回从启动至今跳过的坏候选数，供诊断读取。"""
        with self._lock:
            return self._invalid_candidates

    def close(self):
        """撤销全部候选并拒绝后续更新；重复调用无副作用。"""
        with self._lock:
            self._status, self._candidates = 'closed', ()


class RgbdAdapter:
    """挂到已由主逻辑初始化并 spin 的 ROS 节点上，不创建第二个 executor。

    用法：adapter = RgbdAdapter(RosBridgeNodeInstance)，随后 adapter.snapshot()。
    主节点销毁前调用 close()。导入本模块不要求启动 ROS，构造适配器时才导入消息。
    """

    def __init__(self, node, topic='/rgbd_side/candidates', max_age_s=0.5):
        """以可靠、易失、深度 1 建订阅；topic 和有效期可按集成需求传入。"""
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from vision_interfaces.msg import RgbdCandidateArray

        self._node = node
        self._state = RgbdState(max_age_s)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self._subscription = node.create_subscription(RgbdCandidateArray, topic, self._receive, qos)

    def _receive(self, msg):
        """ROS 线程只更新快照；坏候选有计数并给出警告，不执行模型或动作。"""
        invalid = self._state.update(msg)
        if invalid:
            self._node.get_logger().warning(f'RGBD: ignored {invalid} invalid candidates')

    def snapshot(self):
        """供业务线程读取当前快照；空数组、过期、关闭时候选均为空。"""
        return self._state.snapshot()

    def close(self):
        """撤销状态并移除订阅；不关闭主逻辑的 ROS 上下文。"""
        self._state.close()
        if self._subscription is not None:
            self._node.destroy_subscription(self._subscription)
            self._subscription = None
