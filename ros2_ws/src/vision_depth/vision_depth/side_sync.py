"""实时图对调度：整数时间配对与单槽后台任务，避免旧帧积压。

用大白话说
    各缓存十帧，等后续深度到齐或等待五十毫秒，再选容差内最近的帧。
    配过的帧立即移除；时间倒退清空旧缓存，防止回放循环把两轮图像混在一起。
    配对后的任务只保留一个待处理槽，新图替换旧图；不导入 ROS。
"""
from collections import Counter, deque
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class TimedFrame:
    """图像载荷及其 header 纳秒时间、本地收到的单调秒时间。"""
    stamp_ns: int
    received: float
    payload: object


@dataclass(frozen=True)
class ImagePair:
    """一次一对一配对；epoch 在时间或几何重置后增加，用来废弃旧任务。"""
    color: TimedFrame
    depth: TimedFrame
    epoch: int

    @property
    def received(self):
        """年龄从这一对中较早收到的图像开始计时。"""
        return min(self.color.received, self.depth.received)


class RgbdSynchronizer:
    """按彩色顺序配对，有界队列与错误计数供窗口和诊断使用。"""

    def __init__(self, tolerance_ns=10_000_000, wait_s=0.05, capacity=10):
        """配置正容差纳秒、等待秒和最大缓存帧数。"""
        if tolerance_ns <= 0 or wait_s <= 0 or capacity < 1:
            raise ValueError('invalid synchronizer parameters')
        self.tolerance_ns, self.wait_s, self.capacity = tolerance_ns, wait_s, capacity
        self.queues = {'color': deque(), 'depth': deque()}
        self.last = {'color': None, 'depth': None}
        self.seen = {'color': deque(maxlen=capacity*4), 'depth': deque(maxlen=capacity*4)}
        self.epoch = 0
        self.counts = Counter()

    def reset(self):
        """清空旧图与时间基准，计数保留；调用者同时废弃旧结果。"""
        for role in self.queues:
            self.queues[role].clear()
            self.seen[role].clear()
            self.last[role] = None
        self.epoch += 1

    def add(self, role, stamp_ns, received, payload):
        """只缓存消息；返回是否接受，零值/重复拒绝，时间倒退触发新 epoch。"""
        if role not in self.queues:
            raise ValueError('unknown stream')
        if stamp_ns <= 0:
            self.counts['zero_stamp'] += 1
            return False
        if stamp_ns in self.seen[role]:
            self.counts['duplicate'] += 1
            return False
        if self.last[role] is not None and stamp_ns < self.last[role]:
            self.counts['rollback'] += 1
            self.reset()
        self.seen[role].append(stamp_ns)
        self.last[role] = stamp_ns
        q = self.queues[role]
        if len(q) == self.capacity:
            q.popleft()
            self.counts['overflow'] += 1
        q.append(TimedFrame(int(stamp_ns), received, payload))
        self.counts[f'{role}_received'] += 1
        return True

    def poll(self, now):
        """取所有已能确定的配对；未到等待期限的彩色仍保留，深度不复用。"""
        colors, depths = self.queues['color'], self.queues['depth']
        pairs = []
        while colors:
            c = colors[0]
            watermark_ready = depths and depths[-1].stamp_ns >= c.stamp_ns + self.tolerance_ns
            if not watermark_ready and now-c.received < self.wait_s:
                break
            colors.popleft()
            while depths and depths[0].stamp_ns < c.stamp_ns-self.tolerance_ns:
                depths.popleft()
                self.counts['unmatched_depth'] += 1
            candidates = [(abs(d.stamp_ns-c.stamp_ns), d.stamp_ns, i)
                          for i, d in enumerate(depths)
                          if abs(d.stamp_ns-c.stamp_ns) <= self.tolerance_ns]
            if not candidates:
                self.counts['unmatched_color'] += 1
                continue
            _, _, index = min(candidates)
            # 深度序号保持单调：不能把较早深度帧留给下一张彩色倒序配对。
            for _ in range(index):
                depths.popleft()
                self.counts['unmatched_depth'] += 1
            pairs.append(ImagePair(c, depths.popleft(), self.epoch))
            self.counts['paired'] += 1
        while depths and now-depths[0].received >= self.wait_s and not colors:
            depths.popleft()
            self.counts['unmatched_depth'] += 1
        return pairs


class LatestWorker:
    """仅保留一个 pending 和一个完成结果；invalidate 使执行中的旧任务也失效。"""

    def __init__(self, process):
        """process 在后台接收 job，GUI 不在此线程运行。"""
        self.process = process
        self.condition = threading.Condition()
        self.pending = None
        self.result = None
        self.generation = 0
        self.stopped = False
        self.counts = Counter()
        self.durations = deque(maxlen=2000)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, job):
        """替换尚未执行的任务；正在执行的任务不强制中断。"""
        with self.condition:
            if self.pending is not None:
                self.counts['processing_dropped'] += 1
            self.pending = (job, self.generation)
            self.condition.notify()

    def invalidate(self):
        """清空待处理与已完成结果；旧 epoch 的执行结果将丢弃。"""
        with self.condition:
            if self.pending is not None:
                self.counts['reset_pending_dropped'] += 1
            self.pending = self.result = None
            self.generation += 1

    def take(self):
        """原子取走最近一个结果，返回 None 表示尚无新结果。"""
        with self.condition:
            result, self.result = self.result, None
            return result

    def stats(self):
        """返回计数与有界处理耗时历史的副本。"""
        with self.condition:
            return dict(self.counts), list(self.durations)

    def close(self):
        """停止接收任务并最多等待两秒；退出 GUI 不无限等待推理。"""
        with self.condition:
            self.stopped = True
            if self.pending is not None:
                self.counts['shutdown_pending_dropped'] += 1
            self.pending = None
            self.condition.notify()
        self.thread.join(timeout=2.0)

    def _loop(self):
        """串行执行任务，仅发布仍属于当前 generation 的结果。"""
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopped or self.pending is not None)
                if self.stopped:
                    return
                job, generation = self.pending
                self.pending = None
            started = time.monotonic()
            try:
                value, error = self.process(job), None
            except Exception as exc:  # 后台异常必须交给主线程显示，不能无声退出。
                value, error = None, f'{type(exc).__name__}: {exc}'
            finished = time.monotonic()
            with self.condition:
                self.counts['processed'] += 1
                self.counts['errors'] += int(error is not None)
                self.durations.append((finished-started)*1000)
                if generation == self.generation and not self.stopped:
                    self.result = {'value': value, 'error': error, 'job': job,
                                   'started': started, 'finished': finished}
                else:
                    self.counts['invalidated_result'] += 1
