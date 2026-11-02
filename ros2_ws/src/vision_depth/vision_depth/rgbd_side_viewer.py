"""实时侧面候选查看器：ROS 缓存、后台测量、主线程窗口与按键。

用大白话说
    窗口上的图、框和中心来自同一对 RGB-D；超过接收后年龄就撤掉坐标。
    本节点不发布抓取结果，保存快照也不等于用户已经标注或复核。
"""
import hashlib
import json
import os
import time
import uuid
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from vision.depth.surface import PlaneParams
from vision.yolo import YoloDetector
from .side_pipeline import SidePipeline, info_dict, render_result, save_snapshot
from .side_sync import LatestWorker, RgbdSynchronizer


def checked_detector(weights, imgsz, conf, iou, classes, names, max_detections):
    """预热并核对支持的 ONNX 输出，缺模型/类名/不匹配维度均明确失败。"""
    path = Path(weights).expanduser()
    if not path.is_file():
        raise ValueError(f'model not found: {path}')
    detector = YoloDetector(str(path), imgsz=imgsz, conf=conf, iou=iou,
                            classes=classes, class_names=names, max_detections=max_detections)
    detector.detect(np.zeros((imgsz, imgsz, 3), np.uint8))
    class_map = dict(enumerate(names)) if names else detector._names
    if not class_map or sorted(class_map) != list(range(len(class_map))):
        raise ValueError('provide complete class_names or compatible model class metadata')
    shape = tuple(detector._net.forward().shape)
    channels = 4+len(class_map)
    # 当前解码器只支持 8/16/32 三尺度的密集 YOLO 输出。
    # 仅检查六列会把 end-to-end NMS 的 (1,300,6) 错当成两类原始输出。
    candidates = sum((imgsz//stride)**2 for stride in (8, 16, 32))
    if shape not in ((1, channels, candidates), (1, candidates, channels)):
        raise ValueError(f'unsupported ONNX output {shape}; require dense 4+nc, N={candidates}')
    if classes and any(c not in class_map for c in classes):
        raise ValueError('requested class id absent from model')
    metadata = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'imgsz': imgsz, 'conf': conf, 'iou': iou, 'classes': classes,
                'class_names': class_map, 'output_shape': shape,
                'output_layout': 'dense_yolo_4_plus_nc_strides_8_16_32', 'backend': 'opencv_cpu'}
    return detector, metadata


def _percentiles(values):
    """只汇总有界历史，无数据时返回 null。"""
    return {'samples': len(values), 'p50': float(np.percentile(values, 50)) if values else None,
            'p95': float(np.percentile(values, 95)) if values else None}


class RgbdSideViewer(Node):
    """仅窗口测量：消息缓存与 GUI 在主线程，单工作线程执行检测和几何。"""

    def __init__(self):
        """读取参数，验证模型和 aligned 输入声明，再建立订阅。"""
        super().__init__('rgbd_side_viewer')
        defaults = {'color_topic': '/camera/color/image_raw', 'depth_topic': '/camera/depth/image_raw',
                    'color_info_topic': '/camera/color/camera_info', 'depth_info_topic': '/camera/depth/camera_info',
                    'weights': 'knowledge/models/yolov8n_mix4.onnx', 'imgsz': 640, 'conf': 0.45,
                    'iou': 0.45, 'max_detections': 32, 'target_classes': [-1], 'class_names': [''],
                    'aligned_depth': True, 'depth_unit': 'mm', 'sync_tolerance_ms': 10.0,
                    'sync_wait_ms': 50.0, 'queue_capacity': 10, 'max_age_ms': 500.0,
                    'leaf_mm': 4.0, 'threshold_mm': 6.0, 'iterations': 200, 'seed': 0,
                    'min_points': 120, 'selection_policy': 'dominant',
                    'core_ratio': 0.6, 'min_valid_ratio': 0.5, 'min_support_ratio': 0.6,
                    'min_core_coverage': 0.6, 'min_support_margin': 0.2,
                    'output_dir': 'output/realtime_side', 'headless': False,
                    'max_runtime_s': 0.0}
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.config = {name: self.get_parameter(name).value for name in defaults}
        c = self.config
        if not c['aligned_depth'] or c['depth_unit'] != 'mm':
            raise ValueError('requires explicitly declared aligned depth in mm')
        if (not all(np.isfinite(c[n]) and c[n] > 0 for n in
                    ('sync_tolerance_ms', 'sync_wait_ms', 'max_age_ms')) or
                not np.isfinite(c['max_runtime_s']) or c['max_runtime_s'] < 0):
            raise ValueError('invalid timing parameters')
        params = PlaneParams(c['leaf_mm'], c['threshold_mm'], c['iterations'], c['seed'], c['min_points'],
                             selection_policy=c['selection_policy'], core_ratio=c['core_ratio'],
                             min_valid_ratio=c['min_valid_ratio'], min_support_ratio=c['min_support_ratio'],
                             min_core_coverage=c['min_core_coverage'], min_support_margin=c['min_support_margin'])
        params.validate()
        classes = list(c['target_classes'])
        if classes == [-1]:
            classes = []
        elif any(v < 0 for v in classes):
            raise ValueError('target_classes uses [-1] for all or nonnegative ids')
        names = list(c['class_names'])
        if names == ['']:
            names = []
        detector, self.model = checked_detector(c['weights'], c['imgsz'], c['conf'], c['iou'],
                                                 classes, names, c['max_detections'])
        self.sync = RgbdSynchronizer(round(c['sync_tolerance_ms']*1e6), c['sync_wait_ms']/1000,
                                    c['queue_capacity'])
        self.infos = {}
        self.stream_ok = {'color': False, 'depth': False}
        self.problems = {}
        self.arrivals = {'color': deque(maxlen=120), 'depth': deque(maxlen=120)}
        self.worker = LatestWorker(SidePipeline(detector, params))
        self.last_result = None
        self.epoch_seen = self.sync.epoch
        self.stale_seen = False
        self.counts = Counter()
        self.ages = deque(maxlen=2000)
        self.inference_ms = deque(maxlen=2000)
        self.geometry_ms = deque(maxlen=2000)
        self.started = time.monotonic()
        self.run_dir = Path(c['output_dir']).expanduser()/('run_'+time.strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.subscriptions_ = []
        for role in ('color', 'depth'):
            self.subscriptions_.append(self.create_subscription(
                Image, c[f'{role}_topic'], self._image_callback(role), qos_profile_sensor_data))
            self.subscriptions_.append(self.create_subscription(
                CameraInfo, c[f'{role}_info_topic'], self._info_callback(role), qos_profile_sensor_data))
        self.get_logger().info(f'ready: {self.run_dir}; candidate centers are not validated object centers')

    def _image_callback(self, role):
        """回调只检查几何外壳、统计和缓存消息，不解图或运行模型。"""
        def callback(msg):
            """接收单路图像，缺少配套内参时等待，几何冲突时撤销旧结果。"""
            now = time.monotonic()
            self.arrivals[role].append(now)
            info = self.infos.get(role)
            okay = bool(info and msg.header.frame_id == info['frame_id'] and
                        (msg.width, msg.height) == (info['width'], info['height']))
            self.stream_ok[role] = okay
            if not okay:
                self.problems[role] = f'{role}: WAIT_INFO or geometry mismatch'
                self.sync.reset()
                return
            self.problems.pop(role, None)
            stamp = int(msg.header.stamp.sec)*1_000_000_000+int(msg.header.stamp.nanosec)
            self.sync.add(role, stamp, now, msg)
        return callback

    def _info_callback(self, role):
        """内参变化使缓存与执行中任务失效，等待两路新图。"""
        def callback(msg):
            """缓存几何参数；相同内参的逐帧消息不会触发重置。"""
            try:
                info = info_dict(msg)
            except ValueError as exc:
                self.problems[f'{role}_info'] = str(exc)
                self.infos.pop(role, None)
                self.stream_ok = {'color': False, 'depth': False}
                self.sync.reset()
                return
            self.problems.pop(f'{role}_info', None)
            if self.infos.get(role) != info:
                self.infos[role] = info
                self.stream_ok = {'color': False, 'depth': False}
                self.sync.reset()
                self.counts['geometry_resets'] += 1
        return callback

    def stats(self, now):
        """窗口使用接收频率和运行平均处理频率，不将接收后年龄称为曝光延迟。"""
        counts, _ = self.worker.stats()
        result = dict(counts)
        for role, arrival in self.arrivals.items():
            recent = [v for v in arrival if now-v <= 2.0]
            result[f'{role}_fps'] = ((len(recent)-1)/(recent[-1]-recent[0])
                                      if len(recent) > 1 and now-recent[-1] < .5 else 0.)
        result['process_fps'] = counts.get('processed', 0)/max(now-self.started, 1e-6)
        result['sync_dropped'] = sum(v for k, v in self.sync.counts.items()
                                     if k in ('overflow', 'unmatched_color', 'unmatched_depth', 'duplicate', 'zero_stamp'))
        result['stale_count'] = self.counts['stale']
        return result

    def step(self):
        """提交最近图对并取走最新结果，返回当前可显示结果与错误文本。"""
        now = time.monotonic()
        if self.sync.epoch != self.epoch_seen:
            self.worker.invalidate()
            self.last_result = None
            self.epoch_seen = self.sync.epoch
        pairs = self.sync.poll(now)
        if all(self.stream_ok.values()) and len(self.infos) == 2:
            for pair in pairs:
                self.worker.submit({'pair': pair, 'ci': self.infos['color'], 'di': self.infos['depth']})
        ready = self.worker.take()
        if ready and ready['job']['pair'].epoch == self.sync.epoch:
            self.last_result = ready['value']
            self.stale_seen = False
            if ready['error']:
                self.problems['pipeline'] = ready['error']
            else:
                self.problems.pop('pipeline', None)
                value = ready['value']
                self.ages.append((now-value['received'])*1000)
                self.inference_ms.append(value['timing_ms']['inference'])
                self.geometry_ms.append(value['timing_ms']['geometry'])
                for row in value['observations']:
                    self.counts[row['status']] += 1
        return self.last_result, '; '.join(self.problems.values()) or 'WAIT_SYNC / CAMERA_INFO'

    def close(self):
        """终止后台并写性能摘要；历史只保留最近 2000 个样本，非精度验收报告。"""
        self.worker.close()
        now = time.monotonic()
        counts, durations = self.worker.stats()
        summary = {'runtime_s': now-self.started, 'model': self.model, 'config': self.config,
                   'worker_stopped': not self.worker.thread.is_alive(),
                   'sync_counts': dict(self.sync.counts), 'worker_counts': counts,
                   'status_counts': dict(self.counts), 'stats': self.stats(now),
                   'history_limit': 2000, 'compute_ms': _percentiles(durations),
                   'receive_age_first_display_ms': _percentiles(list(self.ages)),
                   'inference_ms': _percentiles(list(self.inference_ms)),
                   'geometry_ms': _percentiles(list(self.geometry_ms)),
                   'geometry_status': 'UNVERIFIED', 'spatial_status': 'UNVERIFIED',
                   'user_annotation_completed': False}
        (self.run_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def run_window(node_factory, args=None):
    """OpenCV GUI 仅在主线程运行，Q/M/S 控制显示与快照，headless 用于回放验证。"""
    rclpy.init(args=args)
    node = None
    overlay = True
    depth_visible = True
    try:
        node = node_factory()
        headless = node.config['headless']
        if not headless:
            if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
                raise ValueError('GUI display unavailable; use headless:=true for replay tests')
            cv2.namedWindow('RGBD Side Viewer', cv2.WINDOW_NORMAL)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.005)
            # 四路 30 Hz 消息不能只在每次 GUI 刷新时接收一条；先有界排空
            # 就绪回调，避免 waitKey 的等待使 RGB/深度在队列中错开。
            deadline = time.monotonic()+.004
            for _ in range(31):
                rclpy.spin_once(node, timeout_sec=0.)
                if time.monotonic() >= deadline:
                    break
            result, problem = node.step()
            now = time.monotonic()
            if result is None:
                shown = np.zeros((480, 1000, 3), np.uint8)
                cv2.putText(shown, problem.encode('ascii', 'replace').decode()[:110], (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 200, 255), 1)
                stale = False
            else:
                shown, stale = render_result(result, now, node.stats(now), overlay,
                                             node.config['max_age_ms'], depth_visible)
                if stale and not node.stale_seen:
                    node.counts['stale'] += 1
                    node.stale_seen = True
            if not headless:
                cv2.imshow('RGBD Side Viewer', shown)
                key = cv2.waitKey(1) & 255
                if cv2.getWindowProperty('RGBD Side Viewer', cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord('q'), 27):
                    break
                if key == ord('m'):
                    overlay = not overlay
                if key == ord('d'):
                    depth_visible = not depth_visible
                if key == ord('s') and result is not None:
                    try:
                        path = save_snapshot(node.run_dir/'snapshots', result, shown,
                                             {'model': node.model, 'config': node.config,
                                              'stale': stale, 'depth_visible': depth_visible,
                                              'stats': node.stats(now)})
                        node.get_logger().info(f'snapshot: {path}')
                    except (OSError, cv2.error) as exc:
                        node.get_logger().error(f'snapshot failed: {exc}')
            if node.config['max_runtime_s'] and now-node.started >= node.config['max_runtime_s']:
                break
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'rgbd_side_viewer error: {type(exc).__name__}: {exc}')
        return 2
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main(args=None):
    """启动同模块的主线程窗口；后台线程只执行测量。"""
    return run_window(RgbdSideViewer, args)


if __name__ == '__main__':
    raise SystemExit(main())
