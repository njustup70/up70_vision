"""侧面拟合、检测编排及人工真值对照的离线测试。"""
import json

import cv2
import numpy as np
import pytest

from lab.side_rgbd_measure.evaluate import displacement, evaluate
from lab.side_rgbd_measure.geometry import PlaneParams, measure_planes, visible_extent
from lab.side_rgbd_measure.input import rectified_color_and_mask
from lab.side_rgbd_measure.run import draw_detection_label, run_frames
from vision.depth import Intrinsics
from vision.types import Detection


def make_intrinsics(w=160, h=120):
    """构造与合成图相符的内参。"""
    return Intrinsics(fx=200., fy=200., cx=w/2, cy=h/2, width=w, height=h)


def test_plane_measurement_and_mask():
    """固定深度的平面应得到已知光轴距离，深度洞及网格无效区不可入选。"""
    depth = np.full((120, 160), 600, np.uint16)
    depth[30:40, 30:40] = 0
    mask = np.ones(depth.shape, bool)
    mask[:, :20] = False
    result, masks = measure_planes(depth, make_intrinsics(), (10, 10, 150, 110), mask)
    assert result['status'] == 'candidate_unverified'
    assert len(masks) == 1
    candidate = result['candidates'][0]
    assert candidate['plane_distance_mm'] == pytest.approx(600, abs=1e-5)
    assert candidate['centroid_z_mm'] == pytest.approx(600)
    assert candidate['normal'] == pytest.approx([0, 0, -1], abs=1e-6)
    assert candidate['residual_rms_mm'] < 1e-5
    assert not masks[0][:, :20].any() and not masks[0][30:40, 30:40].any()


def test_two_planes_and_no_points():
    """两个足够大的平面应报告歧义；全零深度没有候选。"""
    depth = np.full((120, 160), 500, np.uint16)
    depth[:, 80:] = 800
    result, masks = measure_planes(depth, make_intrinsics(), (0, 0, 160, 120))
    assert result['status'] == 'ambiguous' and result['selected_index'] is None
    assert len(masks) == 2
    assert sorted(c['plane_distance_mm'] for c in result['candidates']) == pytest.approx([500, 800], abs=0.01)
    empty, masks = measure_planes(np.zeros_like(depth), make_intrinsics(), (0, 0, 160, 120))
    assert empty['status'] == 'no_plane' and not masks


def test_dominant_selects_main_plane_and_strict_preserves_ambiguity():
    """清晰主体加少量背景可选主面，旧策略仍报告两个候选歧义。"""
    depth = np.full((120, 160), 600, np.uint16)
    depth[:, :20] = 800
    strict, _ = measure_planes(depth, make_intrinsics(), (0, 0, 160, 120))
    result, masks = measure_planes(depth, make_intrinsics(), (0, 0, 160, 120),
                                   params=PlaneParams(selection_policy='dominant'))
    assert strict['status'] == 'ambiguous' and len(masks) == 2
    assert result['status'] == 'candidate_unverified'
    chosen = result['candidates'][result['selected_index']]
    assert chosen['centroid_z_mm'] == pytest.approx(600, abs=1)
    assert chosen['core_coverage'] >= .6 and chosen['inlier_ratio'] >= .6


def test_dominant_rejects_similar_planes_sparse_and_edge_background():
    """相近候选、缺深度或仅边缘主面不能误选为主体。"""
    params = PlaneParams(selection_policy='dominant')
    depth = np.full((120, 160), 500, np.uint16)
    depth[:, 80:] = 800
    result, _ = measure_planes(depth, make_intrinsics(), (0, 0, 160, 120), params=params)
    assert result['status'] == 'ambiguous' and result['selected_index'] is None
    depth[:] = 0
    depth[40:80, 50:110] = 600
    result, _ = measure_planes(depth, make_intrinsics(), (0, 0, 160, 120), params=params)
    assert result['status'] == 'ambiguous' and result['reason'] == 'low_valid_ratio'
    depth[:] = 0
    result, _ = measure_planes(depth, make_intrinsics(), (0, 0, 160, 120), params=params)
    assert result['status'] == 'no_plane' and result['selected_index'] is None
    depth[:] = 800
    depth[24:96, 32:128] = 500
    result, _ = measure_planes(depth, make_intrinsics(), (0, 0, 160, 120), params=params)
    assert result['status'] == 'ambiguous' and result['selected_index'] is None
    assert result['reason'] == 'low_core_coverage'


def test_dominant_can_select_nonzero_candidate(monkeypatch):
    """当降采样使小平面先出现时，按原始内点占比选择后一个主体面。"""
    import vision.depth.surface as surface
    depth = np.full((120, 160), 600, np.uint16)
    depth[:, :20] = 800
    def fit_in_order(points, threshold, iterations, rng):
        """固定先拟合远处窄条，以覆盖真实候选顺序不等于面积顺序的情况。"""
        z = 800. if np.any(np.isclose(points[:, 2], 800.)) else 600.
        return np.array([0., 0., -1.]), z, np.isclose(points[:, 2], z)
    monkeypatch.setattr(surface, 'fit_plane_ransac', fit_in_order)
    result, _ = measure_planes(depth, make_intrinsics(), (0, 0, 160, 120),
                               params=PlaneParams(selection_policy='dominant'))
    assert result['status'] == 'candidate_unverified'
    assert result['selected_index'] == 1
    assert result['candidates'][1]['centroid_z_mm'] == pytest.approx(600)


def test_dominant_includes_exact_support_margin_boundary():
    """主体占 60%、次候选占 40% 时应包含 20% 差值的浮点边界。"""
    depth = np.full((120, 160), 800, np.uint16)
    depth[:, :96] = 600
    result, _ = measure_planes(depth, make_intrinsics(), (0, 0, 160, 120),
                               params=PlaneParams(selection_policy='dominant'))
    assert result['status'] == 'candidate_unverified'


def test_core_coverage_denominator_includes_invalid_pixels():
    """中央大孔洞不能因除以有效点数而得到虚高覆盖率。"""
    depth = np.full((120, 160), 600, np.uint16)
    depth[24:96, 32:90] = 0
    result, _ = measure_planes(depth, make_intrinsics(), (0, 0, 160, 120),
                               params=PlaneParams(selection_policy='dominant'))
    assert result['valid_depth_ratio'] > .5
    assert result['candidates'][0]['inlier_ratio'] == pytest.approx(1.)
    assert result['candidates'][0]['core_coverage'] < .6
    assert result['status'] == 'ambiguous' and result['reason'] == 'low_core_coverage'


def test_selection_parameters_are_validated():
    """无效策略或越界比例应在处理前拒绝。"""
    for params in (PlaneParams(selection_policy='other'), PlaneParams(core_ratio=0),
                   PlaneParams(min_core_coverage=1.1), PlaneParams(min_valid_ratio=float('nan'))):
        with pytest.raises(ValueError):
            params.validate()


def test_line_degenerate_and_size_mismatch():
    """共线点无法决定平面，图与内参尺寸不匹配时拒绝。"""
    depth = np.zeros((120, 160), np.uint16)
    depth[60, :] = 600
    result, _ = measure_planes(depth, make_intrinsics(), (0, 0, 160, 120),
                               params=PlaneParams(leaf_mm=1, min_points=3))
    assert result['status'] == 'no_plane'
    with pytest.raises(ValueError):
        measure_planes(depth, make_intrinsics(120, 120), (0, 0, 160, 120))


def test_visible_extent_rotated_3d_rectangle():
    """矩形在平面内及三维中旋转后长短边仍为 120 和 40 mm。"""
    xy = np.array([[-60, -20], [-60, 20], [60, 20], [60, -20]], dtype=float)
    z = np.array([0., 0., 1.])
    r, _ = cv2.Rodrigues(np.array([0.5, -0.4, 0.9]))
    pts = np.column_stack([xy, np.zeros(4)]) @ r.T
    assert visible_extent(pts, r @ z) == pytest.approx([120, 40], abs=0.0001)


def test_color_rectification_mask_handles_distortion():
    """非零畸变的图外采样区被掩码排除，映射与 OpenCV 相符。"""
    image = np.full((80, 100, 3), 180, np.uint8)
    k = [70, 0, 50, 0, 70, 40, 0, 0, 1]
    ci = {'k': k, 'd': [0.5, 0.2, 0., 0., 0.]}
    rectified, valid = rectified_color_and_mask(image, ci, {'k': k})
    assert rectified.shape == image.shape and valid.shape == image.shape[:2]
    assert valid[40, 50] and not valid[0, 0]


class FakeDetector:
    """只用于检查检测器输入和空框、多框路径，不需要真实模型。"""
    def __init__(self, boxes):
        self.boxes = boxes
        self.seen = []

    def detect(self, image):
        """记录收到的校正图并返回指定框。"""
        self.seen.append(image.copy())
        return [Detection(bbox_xyxy=b, class_name='box', class_id=0, score=0.9) for b in self.boxes]


def test_detector_empty_multiple_and_choice(tmp_path):
    """无框、多框都不猜目标；明确选择后仍标记未验证。"""
    h, w = 120, 160
    k = [200, 0, 80, 0, 200, 60, 0, 0, 1]
    ci, di = {'k': k, 'd': [0]*5}, {'k': k, 'frame_id': 'camera'}
    frames = [({'pair_id': 'c1_d1'}, np.zeros((h, w, 3), np.uint8), np.full((h, w), 600, np.uint16))]
    summary = {'session': 'test', 'geometry': {'status': 'UNVERIFIED'}, 'spatial': {'status': 'UNVERIFIED'}}
    empty = run_frames(summary, ci, di, frames, tmp_path/'empty', detector=FakeDetector([]))
    assert empty['frames'][0]['status'] == 'no_detection'
    boxes = [(0, 0, 80, 120), (80, 0, 160, 120)]
    many = run_frames(summary, ci, di, frames, tmp_path/'many', detector=FakeDetector(boxes))
    assert many['frames'][0]['status'] == 'ambiguous_detection'
    chosen = run_frames(summary, ci, di, frames, tmp_path/'chosen', detector=FakeDetector(boxes),
                        choices={'c1_d1': {'detection': 0, 'plane': 0}})
    assert chosen['frames'][0]['status'] == 'candidate_unverified'
    assert (tmp_path/'chosen'/'frames'/'c1_d1_plane0_mask.png').exists()
    assert json.loads((tmp_path/'chosen'/'results.json').read_text())['actual_count'] == 1
    with pytest.raises(FileExistsError):
        run_frames(summary, ci, di, frames, tmp_path/'chosen', box=boxes[0])


def test_manual_plane_choice_overrides_dominant(tmp_path):
    """人工逐帧指定平面应覆盖 dominant 自动选择，仍保持未验证。"""
    k = [200, 0, 80, 0, 200, 60, 0, 0, 1]
    ci, di = {'k': k, 'd': [0]*5}, {'k': k, 'frame_id': 'camera'}
    depth = np.full((120, 160), 600, np.uint16)
    depth[:, :20] = 800
    frames = [({'pair_id': 'c1_d1'}, np.zeros((120, 160, 3), np.uint8), depth)]
    summary = {'session': 'test', 'geometry': {'status': 'UNVERIFIED'}, 'spatial': {'status': 'UNVERIFIED'}}
    params = PlaneParams(selection_policy='dominant')
    result = run_frames(summary, ci, di, frames, tmp_path/'override', box=(0, 0, 160, 120),
                        params=params, choices={'c1_d1': {'plane': 1}})
    row = result['frames'][0]
    assert row['status'] == 'candidate_unverified'
    assert row['measurement']['selected_index'] == 1
    assert 'user selected' in row['reason']


def test_evaluation_counts_unlabeled_and_gates():
    """标注外的结果不进入识别指标，无完整边界不计算尺寸误差。"""
    cand = {'visible_extent_mm': [100, 50], 'centroid_mm': [0, 0, 600]}
    row = {'pair': {'pair_id': 'p1'}, 'detections': [{'bbox_xyxy': [0, 0, 10, 10]}],
           'measurement': {'selected_index': 0, 'candidates': [cand]}}
    result = {'frames': [row], 'stats': {}, 'frame_id': 'optical'}
    annotation = {'frames': {'p1': {'boxes': [[0, 0, 10, 10]], 'side_review': 'correct',
                                         'complete_boundary': False},
                             'missing': {'boxes': []}},
                  'side_size_mm': [100, 50], 'size_source': 'caliper'}
    metrics = evaluate(result, annotation)
    assert (metrics['tp'], metrics['fp'], metrics['fn']) == (1, 0, 0)
    assert metrics['missing_result_frames'] == ['missing'] and metrics['size_errors'] == []
    assert displacement(result, result, {})['status'] == 'UNVERIFIED'
    ref = {'same_face': True, 'complete_boundary': True, 'orientation_unchanged': True,
           'camera_fixed': True, 'source': 'ruler', 'distance_mm': 0,
           'reviewed_pair_ids': ['p1'], 'baseline_reviewed_pair_ids': ['p1']}
    assert displacement(result, result, ref)['error_mm'] == 0


def test_noisy_plane_with_holes_and_fixed_seed():
    """带噪声和平面孔洞可复现，距离仍接近 600 mm。"""
    rng = np.random.default_rng(7)
    depth = np.clip(600 + rng.normal(0, 1.5, (120, 160)), 1, 65535).astype(np.uint16)
    depth[20:35, 40:60] = 0
    a, _ = measure_planes(depth, make_intrinsics(), (0, 0, 160, 120))
    b, _ = measure_planes(depth, make_intrinsics(), (0, 0, 160, 120))
    assert a == b
    assert a['candidates'][0]['plane_distance_mm'] == pytest.approx(600, abs=2)
    assert a['candidates'][0]['residual_rms_mm'] < 2


def test_reference_comparisons_require_review():
    """独立真值只对确认正确的选面比较，未知帧不会凭空生成精度。"""
    candidate = {'plane_distance_mm': 602., 'normal': [0., 0., -1.],
                 'visible_extent_mm': [100., 50.]}
    row = {'pair': {'pair_id': 'p1'}, 'detections': [],
           'measurement': {'selected_index': 0, 'candidates': [candidate]}}
    result = {'frames': [row], 'stats': {}, 'frame_id': 'optical'}
    reference = {'frames': {'p1': {'side_review': 'correct', 'complete_boundary': True}},
                 'side_size_mm': [100., 50.], 'size_source': 'ruler',
                 'plane_distance_reference': {'value_mm': 600., 'source': 'reference instrument',
                                              'reference': 'camera_optical_center_to_plane'},
                 'normal_reference': {'normal': [0, 0, -1], 'frame_id': 'optical', 'source': 'fixture'}}
    outcome = evaluate(result, reference)
    assert outcome['absolute_distance']['errors'][0]['error_mm'] == 2
    assert outcome['absolute_orientation']['errors'][0]['error_deg'] == pytest.approx(0)
    assert outcome['size_errors'][0]['error_mm'] == [0, 0]
    reference['frames']['p1']['side_review'] = 'unknown'
    assert evaluate(result, reference)['absolute_distance']['status'] == 'UNVERIFIED'


def test_input_rejects_image_and_camera_info_frame_mismatch(monkeypatch):
    """采集报告中图像 frame 不匹配时拒绝读取，即使 RGB-D 配对已通过。"""
    from lab.side_rgbd_measure import input as input_module
    color = np.zeros((120, 160, 3), np.uint8)
    depth = np.ones((120, 160), np.uint16)
    info = {'width': 160, 'height': 120, 'frame_id': 'camera'}
    profile = {'width': 160, 'height': 120, 'frame_id': 'wrong'}
    summary = {'capture': {'depth_unit': 'mm', 'color_profile': profile,
                           'depth_profile': dict(profile, frame_id='camera')}}
    monkeypatch.setattr(input_module, 'load_report',
                        lambda path: (summary, {}, color, depth, info, info))
    with pytest.raises(ValueError, match='profile and CameraInfo'):
        input_module.load_frames('ignored')


def test_reference_rejects_invalid_or_unselected_truth():
    """缺少选面不得标为正确，畸形参考值不能生成误差。"""
    row = {'pair': {'pair_id': 'p1'}, 'detections': [], 'measurement': None}
    result = {'frames': [row], 'stats': {}, 'frame_id': 'optical'}
    with pytest.raises(ValueError, match='missing selected plane'):
        evaluate(result, {'frames': {'p1': {'side_review': 'correct'}}})
    with pytest.raises(ValueError, match='side_size_mm'):
        evaluate(result, {'frames': {}, 'side_size_mm': 123, 'size_source': 'ruler'})
    with pytest.raises(ValueError, match='truth boxes'):
        evaluate(result, {'frames': {'p1': {'boxes': [[0, 0, 0, 10]]}}})


def test_detection_label_stays_inside_narrow_image():
    """框号标签要落在图内：贴右边缘、图很窄时也不能画到图外或抛错。"""
    image = np.zeros((40, 60, 3), np.uint8)
    draw_detection_label(image, {'bbox_xyxy': [58, 38, 59, 39], 'class_name': 'top',
                                 'score': 0.6043}, 3, plane_index=1)
    assert image.shape == (40, 60, 3)
    assert image.any(), '没有画出任何标签像素'
    narrow = np.zeros((40, 24, 3), np.uint8)
    draw_detection_label(narrow, {'bbox_xyxy': [22, 20, 23, 39], 'class_name': 'block',
                                  'score': None}, 0)
    assert narrow.shape == (40, 24, 3)
    assert narrow.any(), '窄图里也应画出标签'
