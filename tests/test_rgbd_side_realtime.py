"""实时侧面链路：配对、单槽线程、投影、过期撤回与快照一致性。"""
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ros2_ws/src/vision_depth'))

from vision.depth.surface import PlaneParams
from vision.types import Detection
from vision_depth.side_sync import ImagePair, LatestWorker, RgbdSynchronizer, TimedFrame
from vision_depth.side_pipeline import SidePipeline, colorize_depth, image_array, render_result, save_snapshot


@pytest.mark.parametrize('delta,matched', [(10_000_000, True), (-10_000_000, True), (10_000_001, False)])
def test_pairing_integer_boundary(delta, matched):
    """±10 ms 包含边界，超一纳秒不配对。"""
    sync = RgbdSynchronizer()
    sync.add('color', 100_000_000, 0., 'c')
    sync.add('depth', 100_000_000+delta, 0., 'd')
    assert bool(sync.poll(.06)) == matched
    assert not sync.poll(.2)


def test_pairing_waits_for_nearest_and_earlier_tie():
    """等待后续帧，等距取较早帧，且不会复用深度。"""
    sync = RgbdSynchronizer()
    sync.add('color', 100_000_000, 0., 'c')
    sync.add('depth', 98_000_000, .001, 'earlier')
    assert not sync.poll(.01)
    sync.add('depth', 101_000_000, .02, 'later')
    assert sync.poll(.06)[0].depth.payload == 'later'
    tie = RgbdSynchronizer()
    tie.add('color', 100_000_000, 0., 'c')
    tie.add('depth', 98_000_000, 0., 'earlier')
    tie.add('depth', 102_000_000, 0., 'later')
    assert tie.poll(.06)[0].depth.payload == 'earlier'
    assert not tie.poll(.1)


def test_pairing_bounds_zero_duplicate_rollback_and_dropout():
    """队列始终有界，异常计数保留，时间倒退清除旧 epoch。"""
    sync = RgbdSynchronizer(capacity=3)
    assert not sync.add('color', 0, 0., None)
    for index in range(5):
        sync.add('color', 100+index, 0., index)
    assert len(sync.queues['color']) == 3 and sync.counts['overflow'] == 2
    assert not sync.add('color', 104, 0., None)
    old = sync.epoch
    sync.add('color', 50, .1, 'new')
    assert sync.epoch == old+1 and len(sync.queues['color']) == 1
    assert not sync.poll(.2)
    assert sync.counts['unmatched_color'] == 1
    assert sync.counts['zero_stamp'] == sync.counts['duplicate'] == sync.counts['rollback'] == 1


def test_worker_bounded_pending_and_invalidation():
    """慢任务不阻塞提交，待处理任务被更新任务替换，重置后旧结果不回流。"""
    entered, release, processed_latest = threading.Event(), threading.Event(), threading.Event()
    def process(job):
        """第一项等待，用事件验证另两项替换关系。"""
        if job == 1:
            entered.set()
            assert release.wait(2)
        if job == 3:
            processed_latest.set()
        return job
    worker = LatestWorker(process)
    try:
        worker.submit(1)
        assert entered.wait(1)
        worker.submit(2)
        worker.submit(3)
        assert worker.stats()[0]['processing_dropped'] == 1
        release.set()
        assert processed_latest.wait(1)
        with worker.condition:
            worker.condition.wait_for(lambda: worker.result and worker.result['value'] == 3, timeout=.1)
        value = worker.take()
        assert value['value'] == 3
        worker.invalidate()
        assert worker.take() is None
    finally:
        release.set()
        worker.close()


def image_message(array, encoding, stamp=100, frame='optical', padding=0, big=False):
    """构造带行尾填充的消息外壳，不需要 ROS 安装。"""
    arr = array.astype('>u2' if big else '<u2') if array.dtype == np.uint16 else array
    rows = arr.view(np.uint8).reshape(array.shape[0], -1)
    data = np.pad(rows, ((0, 0), (0, padding)))
    return NS(width=array.shape[1], height=array.shape[0], step=data.shape[1], encoding=encoding,
              is_bigendian=big, data=data.tobytes(),
              header=NS(frame_id=frame, stamp=NS(sec=0, nanosec=stamp)))


def test_message_decode_padding_endian_rgb_and_bad_data():
    """大端深度和 RGB 的行尾填充不改变像素，短数据明确失败。"""
    depth = np.array([[500, 600], [700, 800]], np.uint16)
    assert np.array_equal(image_array(image_message(depth, '16UC1', padding=4, big=True), True), depth)
    rgb = np.array([[[10, 20, 30]]], np.uint8)
    assert image_array(image_message(rgb, 'rgb8', padding=2))[0, 0].tolist() == [30, 20, 10]
    bad = image_message(depth, '16UC1')
    bad.data = b''
    with pytest.raises(ValueError):
        image_array(bad, True)


class BoxesDetector:
    """返回配置框，并保留送入的校正图。"""
    def __init__(self, boxes):
        """boxes 为校正网格中的半开框。"""
        self.boxes = boxes
        self.seen = None

    def detect(self, image):
        """不触发真实推理，检查编排和几何。"""
        self.seen = image.copy()
        return [Detection(bbox_xyxy=b, class_name='block', class_id=0, score=.9) for b in self.boxes]


def make_job(depth=None, distortion=None):
    """生成光学网格相符的 160×120 RGB-D 图对。"""
    depth = np.full((120, 160), 600, np.uint16) if depth is None else depth
    color = np.full((120, 160, 3), 50, np.uint8)
    info = {'width': 160, 'height': 120, 'frame_id': 'optical', 'distortion_model': 'plumb_bob',
            'k': [200., 0., 80., 0., 200., 60., 0., 0., 1.], 'd': [0.]*5}
    ci = dict(info, d=distortion or [0.]*5)
    pair = ImagePair(TimedFrame(100, 1., image_message(color, 'bgr8')),
                     TimedFrame(101, 1.001, image_message(depth, '16UC1')), 0)
    return {'pair': pair, 'ci': ci, 'di': info}


def test_pipeline_multibox_projection_and_one_failure():
    """多个框独立测量，错误框不污染其他框，中心正确投影回同一张图。"""
    detector = BoxesDetector([(20, 20, 75, 100), (85, 20, 140, 100), (-1, 0, 1, 2)])
    result = SidePipeline(detector)(make_job())
    a, b, bad = result['observations']
    assert a['status'] == b['status'] == 'CANDIDATE'
    assert bad['status'] == 'INVALID_BOX' and bad['centroid_px'] is None
    assert a['centroid_px'] == pytest.approx([47., 59.5], abs=1e-5)
    assert result['pair']['color_stamp_ns'] == 100
    assert detector.seen.shape == (120, 160, 3)


def test_pipeline_reuses_yolo_color_for_same_rectified_boxes():
    """同帧蓝色方块、横放塔顶分别保留模型类别和 blue/mixed 判色。"""
    from vision.config.loader import load_thresholds, read_thresholds_settings
    from vision.yolo.pipeline import ColorConfig
    class Detector:
        """返回两个不同类别的框。"""
        def detect(self, image):
            """推理一次，颜色留给共用 HSV 管线。"""
            return [Detection((10, 10, 70, 110), 'block', 0, .9),
                    Detection((90, 10, 150, 110), 'top', 1, .8)]
    job = make_job()
    image = np.zeros((120, 160, 3), np.uint8)
    image[10:110, 10:70] = (255, 0, 0)
    image[10:110, 90:120] = (0, 0, 255)
    image[10:110, 120:150] = (255, 0, 0)
    job['pair'].color.payload.data = image.tobytes()
    path = str(ROOT/'configs/color_thresholds.yaml')
    pipeline = SidePipeline(Detector(), thresholds=load_thresholds(path),
                            color_cfg=ColorConfig(**read_thresholds_settings(path)))
    rows = pipeline(job)['observations']
    assert [r['status'] for r in rows] == ['CANDIDATE', 'CANDIDATE']
    assert [(r['detection']['class_id'], r['detection']['color']) for r in rows] == [(0, 'blue'), (1, 'mixed')]
    assert all(r['detection']['valid_pixels'] >= 25 for r in rows)
    # 下一帧全灰时撤掉颜色；不能把前一帧的颜色带到同一位置的新候选上。
    job['pair'].color.payload.data = np.full_like(image, 100).tobytes()
    assert all(r['detection']['color'] is None for r in pipeline(job)['observations'])


def test_ambiguous_no_detection_and_calibration_failure():
    """多平面不输出中心；无框不沿用任何候选；不匹配内参使整对无效。"""
    depth = np.full((120, 160), 500, np.uint16)
    depth[:, 80:] = 800
    row = SidePipeline(BoxesDetector([(0, 0, 160, 120)]))(make_job(depth))['observations'][0]
    assert row['status'] == 'AMBIGUOUS' and row['centroid_px'] is None
    assert not SidePipeline(BoxesDetector([]))(make_job())['observations']
    job = make_job()
    job['di'] = dict(job['di'], frame_id='different')
    with pytest.raises(ValueError):
        SidePipeline(BoxesDetector([]))(job)


def test_selected_nonzero_candidate_is_used_for_projection_render_and_snapshot(monkeypatch, tmp_path):
    """选择非零候选时，投影、显示和快照都对应同一平面。"""
    from vision_depth import side_pipeline
    candidates = [
        {'centroid_mm': [0., 0., 800.], 'inlier_ratio': .2, 'core_coverage': .1, 'residual_rms_mm': 1.},
        {'centroid_mm': [60., 0., 600.], 'inlier_ratio': .8, 'core_coverage': .9, 'residual_rms_mm': 1.}]
    measurement = {'status': 'candidate_unverified', 'reason': 'selected', 'selected_index': 1,
                   'candidates': candidates, 'valid_depth_ratio': 1.}
    masks = [np.zeros((120, 160), np.uint8), np.ones((120, 160), np.uint8)*255]
    monkeypatch.setattr(side_pipeline, 'measure_planes', lambda *args: (measurement, masks))
    result = SidePipeline(BoxesDetector([(0, 0, 160, 120)]))(make_job())
    row = result['observations'][0]
    assert row['centroid_px'] == pytest.approx([100., 60.])
    lines = []
    monkeypatch.setattr(side_pipeline, '_text', lambda canvas, line, y, color=None: lines.append(line))
    shown, stale = render_result(result, 1.2, {})
    assert not stale and np.any(np.all(shown[:120, :160] == [255, 0, 255], axis=2))
    assert shown[60, 100].tolist() == [255, 0, 255]
    assert any('Z=600.0' in line for line in lines)
    assert any('P1 SELECTED' in line for line in lines)
    saved = save_snapshot(tmp_path, result, shown, {})
    data = json.loads((saved/'result.json').read_text())
    assert data['observations'][0]['measurement']['selected_index'] == 1
    assert data['observations'][0]['measurement']['candidates'][1]['centroid_mm'][2] == 600.


def test_stale_hides_all_geometry_and_snapshot_keeps_pair(tmp_path):
    """过期撤去框、中心和坐标；快照仍可追溯原始图对且人工复核为 false。"""
    result = SidePipeline(BoxesDetector([(20, 20, 140, 100)]))(make_job())
    shown, stale = render_result(result, 1.2, {})
    old, stale_old = render_result(result, 2., {})
    assert not stale and stale_old
    assert np.any(np.all(shown[:120, :160] == [255, 0, 255], axis=2))
    assert np.array_equal(old[:120, :160], result['image'])
    folder = save_snapshot(tmp_path, result, old, {'stale': True})
    data = json.loads((folder/'result.json').read_text())
    assert data['pair']['color_stamp_ns'] == 100 and data['stale'] is True
    assert data['user_annotation_completed'] is False
    with np.load(folder/'pair.npz') as pair:
        assert np.array_equal(pair['depth'], result['depth'])


def test_depth_visualization_keeps_invalid_black_and_can_be_hidden():
    """深度零值始终黑色；关闭深度面板不改变融合图及其时间对齐。"""
    depth = np.full((120, 160), 600, np.uint16)
    depth[40:50, 80:90] = 0
    colored = colorize_depth(depth)
    assert np.all(colored[40:50, 80:90] == 0)
    assert np.any(colored[50, 80] != 0)
    result = SidePipeline(BoxesDetector([]))(make_job(depth))
    both, _ = render_result(result, 1.2, {}, depth_visible=True)
    cached = result['depth_color']
    render_result(result, 1.3, {}, depth_visible=True)
    assert result['depth_color'] is cached
    fused, _ = render_result(result, 1.2, {}, depth_visible=False)
    assert both.shape[1] == fused.shape[1] + depth.shape[1]
    assert np.array_equal(both[:120, :160], fused[:120, :160])
    assert np.all(both[40:50, 160+80:160+90] == 0)


@pytest.mark.parametrize('model_name', ['yolov8n_mix2', 'yolov8n_mix4'])
def test_real_model_output_if_ros_and_weights_available(model_name):
    """可用 ROS 和本地模型时验证真实 ONNX 输出；否则明确跳过。"""
    if not (ROOT/f'knowledge/models/{model_name}.onnx').is_file():
        pytest.skip('local partner model not present')
    pytest.importorskip('rclpy')
    from vision_depth.rgbd_side_viewer import checked_detector
    detector, metadata = checked_detector(str(ROOT/f'knowledge/models/{model_name}.onnx'),
                                           640, .45, .45, [], [], 32)
    assert list(metadata['output_shape']) == [1, 6, 8400]
    assert detector.detect(np.zeros((120, 160, 3), np.uint8)) == []


def test_worker_discards_inflight_result_after_epoch_reset():
    """重置发生在慢任务执行期间时，完成后也不能恢复旧坐标。"""
    entered, release = threading.Event(), threading.Event()
    def process(job):
        """用事件保持任务处于运行状态。"""
        entered.set()
        release.wait(1)
        return job
    worker = LatestWorker(process)
    try:
        worker.submit('old')
        assert entered.wait(1)
        worker.invalidate()
        release.set()
        deadline = time.monotonic()+1
        while worker.stats()[0].get('processed', 0) < 1 and time.monotonic() < deadline:
            time.sleep(.005)
        assert worker.stats()[0]['processed'] == 1
        assert worker.take() is None
    finally:
        release.set()
        worker.close()


def test_snapshot_rejects_collision(tmp_path, monkeypatch):
    """文件名碰撞时拒绝覆盖已有快照。"""
    from vision_depth import side_pipeline
    monkeypatch.setattr(side_pipeline.uuid, 'uuid4', lambda: NS(hex='a'*32))
    monkeypatch.setattr(side_pipeline, 'datetime', NS(now=lambda: NS(strftime=lambda fmt: 'fixed_')))
    result = SidePipeline(BoxesDetector([]))(make_job())
    shown, _ = render_result(result, 1.1, {})
    save_snapshot(tmp_path, result, shown, {})
    with pytest.raises(FileExistsError):
        save_snapshot(tmp_path, result, shown, {})


def test_camera_info_change_clears_cached_pairs():
    """内参变化清空已缓存图对，两路新图确认前不继续沿用旧几何。"""
    pytest.importorskip('rclpy')
    from collections import Counter
    from vision_depth.rgbd_side_viewer import RgbdSideViewer
    fake = NS(infos={}, stream_ok={'color': True, 'depth': True}, problems={},
              sync=RgbdSynchronizer(), counts=Counter())
    message = NS(width=160, height=120, header=NS(frame_id='optical'), distortion_model='plumb_bob',
                 k=[200., 0, 80., 0, 200., 60., 0, 0, 1.], d=[0.]*5,
                 r=np.eye(3).ravel().tolist(), p=[0.]*12, binning_x=0, binning_y=0,
                 roi=NS(x_offset=0, y_offset=0, width=0, height=0))
    callback = RgbdSideViewer._info_callback(fake, 'color')
    callback(message)
    epoch = fake.sync.epoch
    callback(message)
    assert fake.sync.epoch == epoch
    fake.sync.add('color', 100, 0., None)
    message.k[0] = 210.
    callback(message)
    assert fake.sync.epoch == epoch+1 and not fake.sync.queues['color']
    assert fake.stream_ok == {'color': False, 'depth': False}
    message.binning_x = 2
    callback(message)
    assert 'color' not in fake.infos and 'color_info' in fake.problems


def test_nms_export_not_misread_as_two_class_dense_output(tmp_path, monkeypatch):
    """六列 NMS 输出不等于两类 YOLO 原始输出，启动必须拒绝。"""
    pytest.importorskip('rclpy')
    from vision_depth import rgbd_side_viewer as viewer
    weights = tmp_path/'fake.onnx'
    weights.write_bytes(b'test-only stub')
    stub = NS(detect=lambda image: [], _names={0: 'block', 1: 'top'},
              _net=NS(forward=lambda: np.zeros((1, 300, 6), np.float32)))
    monkeypatch.setattr(viewer, 'YoloDetector', lambda *args, **kwargs: stub)
    with pytest.raises(ValueError, match='require dense'):
        viewer.checked_detector(str(weights), 640, .45, .45, [], [], 32)


def test_camera_launch_does_not_override_viewer_config(monkeypatch):
    """相机子 launch 的 config_file 不得覆盖查看器的 YAML 参数路径。"""
    pytest.importorskip('rclpy')
    from launch import LaunchService
    from launch.actions import SetLaunchConfiguration
    from launch.substitutions import LaunchConfiguration
    path = ROOT/'ros2_ws/src/vision_depth/launch/rgbd_side_viewer.launch.py'
    spec = importlib.util.spec_from_file_location('side_viewer_launch_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = []
    monkeypatch.setattr(module, 'get_package_share_directory', lambda _: '/tmp/side-share')
    monkeypatch.setattr(module, 'PythonLaunchDescriptionSource', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'IncludeLaunchDescription',
                        lambda *args, **kwargs: SetLaunchConfiguration('config_file', '/tmp/camera.yaml'))
    monkeypatch.setattr(module, '_viewer',
                        lambda context: seen.append(LaunchConfiguration('config_file').perform(context)) or [])
    service = LaunchService()
    service.include_launch_description(module.generate_launch_description())
    assert service.run() == 0
    assert seen == ['/tmp/side-share/config/rgbd_side_viewer.yaml']
