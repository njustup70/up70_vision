"""人工框融合离线原型：网格校正、拒绝歧义与候选关联。"""
from unittest.mock import patch

import numpy as np
import pytest

from lab.manual_rgbd_fusion import rectify_color, run_fusion, select_cluster


def test_rectification_identity_for_zero_distortion():
    """零畸变和相同 K 应保留原图像素。"""
    img = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    k = [40, 0, 15, 0, 40, 10, 0, 0, 1]
    actual = rectify_color(img, {'k': k, 'd': [0] * 5}, {'k': k})
    np.testing.assert_array_equal(actual, img)


def test_cluster_association_refuses_ambiguous_and_empty():
    """多簇不能按分数猜目标，空框也不应产生候选。"""
    labels = np.zeros((20, 30), dtype=np.int32)
    labels[2:12, 2:12] = 1
    labels[2:12, 15:25] = 2
    assert select_cluster(labels, (0, 0, 30, 20))[0] == 'ambiguous'
    assert select_cluster(labels, (0, 15, 30, 20))[0] == 'no_cluster'
    assert select_cluster(labels, (0, 0, 13, 15))[:2] == ('candidate', 1)


def test_fusion_detects_background_joined_to_box():
    """只框中大簇小部分时拒绝输出关联候选。"""
    depth = np.full((20, 30), 600, np.uint16)
    labels = np.ones((20, 30), np.int32)
    fake = {'plane': {'normal': [0, 0, -1], 'd': 600}, 'label_map': labels,
            'clusters': [{'label': 1, 'points': 600}], 'reason': None}
    info = {'k': [100, 0, 15, 0, 100, 10, 0, 0, 1]}
    with patch('lab.manual_rgbd_fusion.analyze', return_value=fake):
        result, mask = run_fusion(depth, info, (0, 0, 10, 10))
    assert result['status'] == 'ambiguous'
    assert result['candidate'] is None
    assert not mask.any()


def test_fusion_candidate_is_always_unverified():
    """匹配成功也只能产生待人工验证的候选。"""
    depth = np.full((20, 30), 600, np.uint16)
    labels = np.zeros((20, 30), np.int32)
    labels[2:12, 2:12] = 1
    fake = {'plane': {'normal': [0, 0, -1], 'd': 600}, 'label_map': labels,
            'clusters': [{'label': 1, 'points': 100}], 'reason': None}
    info = {'k': [100, 0, 15, 0, 100, 10, 0, 0, 1]}
    with patch('lab.manual_rgbd_fusion.analyze', return_value=fake):
        result, mask = run_fusion(depth, info, (0, 0, 14, 14))
    assert result['status'] == 'candidate_unverified'
    assert np.count_nonzero(mask) == 100


def test_invalid_box_rejected():
    """越界框拒绝处理，不做隐式裁切。"""
    with pytest.raises(ValueError):
        run_fusion(np.zeros((20, 30), np.uint16), {}, (-1, 0, 8, 8))
