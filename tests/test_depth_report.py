"""验证报告的候选状态、统计与 ROS 光学坐标契约。"""
from pathlib import Path
import importlib.util

import numpy as np
import pytest

from lab.depth_report.analyze import classify, read_truth, summarize
from lab.bag_detect import Truth, evaluate_checks
from vision.depth import Intrinsics


def candidate(score=0.9, normal=(0, 0, -1)):
    return {'score': score, 'top_center_mm': [0, 0, 500], 'top_normal': normal}


def test_candidate_statuses():
    assert classify({'clusters': [], 'reason': None}) == ('missing', None)
    assert classify({'clusters': [candidate(), candidate()], 'reason': None}) == ('ambiguous', None)
    assert classify({'clusters': [candidate(normal=(0, 0, 1))], 'reason': None}) == ('invalid_normal', None)
    assert classify({'clusters': [candidate()], 'reason': None})[0] == 'ok'
    assert classify({'clusters': [], 'reason': 'no plane'})[0] == 'failed'


def test_summary_uses_median_and_counts_missing():
    rows = [{'status': 'ok', 'x_mm': x, 'y_mm': 0, 'z_mm': 500} for x in (0, 1, 100)]
    rows.append({'status': 'missing'})
    result = summarize(rows)
    assert result['position_median_mm'] == [1, 0, 500]
    assert result['counts'] == {'ok': 3, 'missing': 1}
    assert result['valid_frames'] == 3


def test_missing_truth_is_skipped():
    result = {'plane': {'normal': [0, 0, -1], 'd': 600},
              'clusters': [{'score': 1, 'top_center_mm': [0, 0, 550],
                            'top_normal': [0, 0, -1], 'top_size_mm': [50, 50]}]}
    rows = evaluate_checks(result, Truth(), Intrinsics(400, 400, 424, 240, 848, 480))
    assert [r['status'] for r in rows[:3]] == ['SKIP'] * 3


def test_empty_scene_and_unknown_ground_truth(tmp_path):
    assert read_truth(tmp_path)[0].n_targets is None
    (tmp_path / 'meta.yaml').write_text('scene:\n  objects: []\n', encoding='utf-8')
    truth, _ = read_truth(tmp_path)
    assert truth.n_targets == 0


def test_output_frame_and_quaternion():
    path = Path(__file__).resolve().parents[1] / 'ros2_ws/src/vision_depth/vision_depth/geometry.py'
    spec = importlib.util.spec_from_file_location('depth_geometry', path)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    assert geometry.optical_output_frame('depth_optical', 'depth_optical', '') == 'depth_optical'
    with pytest.raises(ValueError):
        geometry.optical_output_frame('depth_optical', 'depth_optical', 'camera_link')
    with pytest.raises(ValueError):
        geometry.optical_output_frame('depth_optical', 'other', '')
    with pytest.raises(ValueError):
        geometry.optical_output_frame('camera_link', 'camera_link', '')
    x, y, z, w = geometry.quaternion_from_z_axis([0, 0, -1])
    rotated_z = np.array([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)])
    np.testing.assert_allclose(rotated_z, [0, 0, -1], atol=1e-8)
