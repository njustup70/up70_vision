"""把人工标注和独立测量与离线结果对照，不把缺失真值写成通过。

用大白话说
    标注文件列出每帧的箱子框和选面复查，尺寸/位移真值另外记录来源。
    只对标注过的帧算识别指标；缺数据时写 UNVERIFIED，不偷偷补假数据。
"""
import argparse
import json
from pathlib import Path

import numpy as np


def _iou(a, b):
    """半开像素框交并比；退化框返回零。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    area = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return area / (aa + bb - area) if aa + bb - area else 0.0


def _counts(detections, truth):
    """按 IoU 从大到小一对一匹配，阈值固定 0.5。"""
    pairs = sorted(((float(_iou(d['bbox_xyxy'], b)), i, j)
                    for i, d in enumerate(detections) for j, b in enumerate(truth)), reverse=True)
    used_d, used_t = set(), set()
    for iou, di, ti in pairs:
        if iou < 0.5:
            break
        if di not in used_d and ti not in used_t:
            used_d.add(di)
            used_t.add(ti)
    return len(used_d), len(detections)-len(used_d), len(truth)-len(used_t)


def evaluate(result, annotation):
    """根据帧标注计算框指标和人工确认的尺寸误差；不为未标注帧补造真值。"""
    labeled = annotation.get('frames', {})
    rows = {r['pair']['pair_id']: r for r in result['frames']}
    tp = fp = fn = 0
    reviews = {'correct': 0, 'incorrect': 0, 'unknown': 0}
    size_errors, missing, verified_candidates = [], [], []
    truth_size = annotation.get('side_size_mm')
    if truth_size is not None:
        if (not annotation.get('size_source') or not isinstance(truth_size, (list, tuple)) or
                len(truth_size) != 2 or not all(isinstance(x, (int, float)) and
                not isinstance(x, bool) and np.isfinite(x) and x > 0 for x in truth_size)):
            raise ValueError('side_size_mm requires two positive numbers and size_source')
        truth_size = sorted(map(float, truth_size), reverse=True)
    for pid, truth in labeled.items():
        if pid not in rows:
            missing.append(pid)
            continue
        row = rows[pid]
        if 'boxes' in truth:
            if (not isinstance(truth['boxes'], list) or any(
                    not isinstance(b, list) or len(b) != 4 or
                    any(not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v)
                        for v in b) or b[0] >= b[2] or b[1] >= b[3] for b in truth['boxes'])):
                raise ValueError('truth boxes must be nonempty half-open xyxy boxes')
            a, b, c = _counts(row['detections'], truth['boxes'])
            tp, fp, fn = tp+a, fp+b, fn+c
        review = truth.get('side_review', 'unknown')
        if review not in reviews:
            raise ValueError('side_review must be correct, incorrect, or unknown')
        reviews[review] += 1
        m = row.get('measurement') or {}
        index = m.get('selected_index')
        if review == 'correct' and index is None:
            raise ValueError(f'{pid}: cannot mark a missing selected plane correct')
        if review == 'correct' and index is not None:
            verified_candidates.append((pid, m['candidates'][index]))
        if truth_size and truth.get('complete_boundary') is True and review == 'correct' and index is not None:
            sizes = m['candidates'][index]['visible_extent_mm']
            size_errors.append({'pair_id': pid, 'error_mm': [sizes[i]-truth_size[i] for i in range(2)]})
    annotated_boxes = sum(1 for pid, v in labeled.items() if pid in rows and 'boxes' in v)
    distance_results, angle_results = [], []
    distance_ref = annotation.get('plane_distance_reference')
    if distance_ref:
        value = distance_ref.get('value_mm')
        if (distance_ref.get('reference') != 'camera_optical_center_to_plane' or
                not distance_ref.get('source') or not isinstance(value, (int, float)) or
                isinstance(value, bool) or not np.isfinite(value) or value < 0):
            raise ValueError('plane distance requires optical-center reference, value_mm and source')
        distance_results = [{'pair_id': pid, 'error_mm': c['plane_distance_mm']-value}
                            for pid, c in verified_candidates]
    normal_ref = annotation.get('normal_reference')
    if normal_ref:
        ref = np.asarray(normal_ref.get('normal'), dtype=float)
        if (ref.shape != (3,) or not np.isfinite(ref).all() or np.linalg.norm(ref) < 1e-12 or
                not normal_ref.get('source') or normal_ref.get('frame_id') != result.get('frame_id')):
            raise ValueError('normal reference requires finite normal, source and matching frame')
        ref /= np.linalg.norm(ref)
        angle_results = [{'pair_id': pid, 'error_deg': float(np.degrees(np.arccos(
            np.clip(np.abs(np.dot(c['normal'], ref)), 0, 1))))} for pid, c in verified_candidates]
    return {'annotated_box_frames': annotated_boxes, 'missing_result_frames': missing,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'side_review_counts': reviews, 'size_reference_mm': truth_size,
            'size_source': annotation.get('size_source'), 'size_errors': size_errors,
            'absolute_distance': {'status': 'MEASURED' if distance_results else 'UNVERIFIED',
                                  'reference': distance_ref, 'errors': distance_results},
            'absolute_orientation': {'status': 'MEASURED' if angle_results else 'UNVERIFIED',
                                     'reference': normal_ref, 'errors': angle_results},
            'repeatability': result['stats'],
            'note': '逐帧人工复查及参考条件决定适用性；重复性不是绝对精度。'}


def displacement(result, baseline, reference):
    """比较两段重心中位数的位移模长，必须有人确认同面、完整可见和姿态一致。"""
    if (reference.get('same_face') is not True or reference.get('complete_boundary') is not True or
            reference.get('orientation_unchanged') is not True or not reference.get('source')):
        return {'status': 'UNVERIFIED', 'reason': '缺少独立位移参考或同面/边界/姿态确认'}
    distance = reference.get('distance_mm')
    if not isinstance(distance, (int, float)) or isinstance(distance, bool) or not np.isfinite(distance) or distance < 0:
        raise ValueError('displacement distance_mm must be finite and nonnegative')
    def positions(data, reviewed):
        """只取逐帧确认过正确侧面的重心；没有复核不生成位移。"""
        return [r['measurement']['candidates'][r['measurement']['selected_index']]['centroid_mm']
                for r in data['frames'] if r['pair']['pair_id'] in reviewed and r.get('measurement') and
                r['measurement']['selected_index'] is not None]
    a = positions(baseline, reference.get('baseline_reviewed_pair_ids', []))
    b = positions(result, reference.get('reviewed_pair_ids', []))
    if not a or not b or baseline.get('frame_id') != result.get('frame_id'):
        return {'status': 'UNVERIFIED', 'reason': '缺少候选或光学坐标系不一致'}
    # 标注来自操作者，报告仍保留明确的相机固定前提。
    if reference.get('camera_fixed') is not True:
        return {'status': 'UNVERIFIED', 'reason': '相机固定条件未确认'}
    value = float(np.linalg.norm(np.median(b, axis=0) - np.median(a, axis=0)))
    return {'status': 'MEASURED', 'reference_mm': distance, 'measured_mm': value,
            'error_mm': value-distance, 'source': reference['source'],
            'note': '位移比较不等于绝对位置精度'}


def main(argv=None):
    """读取结果与人工标注，独立创建评估 JSON 和 Markdown。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path, help='results.json')
    parser.add_argument('--annotations', type=Path, required=True, help='人工标注 JSON')
    parser.add_argument('--compare', type=Path, help='同一组帧的另一模式 results.json')
    parser.add_argument('--compare-annotations', type=Path, help='另一模式独立的逐帧侧面复查标注')
    parser.add_argument('--baseline', type=Path, help='位移基准段 results.json')
    parser.add_argument('--out', type=Path, required=True, help='不存在的新输出目录')
    args = parser.parse_args(argv)
    try:
        result = json.loads(args.result.read_text(encoding='utf8'))
        annotation = json.loads(args.annotations.read_text(encoding='utf8'))
        metrics = {'primary': evaluate(result, annotation)}
        if args.compare:
            compare = json.loads(args.compare.read_text(encoding='utf8'))
            if not result.get('source_bag') or result.get('source_bag') != compare.get('source_bag'):
                raise ValueError('comparison must use the same bag')
            first_ids = {r['pair']['pair_id'] for r in result['frames']}
            second_ids = {r['pair']['pair_id'] for r in compare['frames']}
            if first_ids != second_ids:
                raise ValueError('comparison must use the same paired frames')
            if not args.compare_annotations:
                raise ValueError('comparison requires independent --compare-annotations')
            compared_annotation = json.loads(args.compare_annotations.read_text(encoding='utf8'))
            main_boxes = {p: a['boxes'] for p, a in annotation.get('frames', {}).items() if 'boxes' in a}
            other_boxes = {p: a['boxes'] for p, a in compared_annotation.get('frames', {}).items() if 'boxes' in a}
            if main_boxes != other_boxes:
                raise ValueError('comparison annotations must use the same frame box truth')
            metrics['compare'] = evaluate(compare, compared_annotation)
        if args.baseline:
            baseline = json.loads(args.baseline.read_text(encoding='utf8'))
            metrics['displacement'] = displacement(result, baseline, annotation.get('displacement', {}))
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / 'evaluation.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
        (args.out / 'evaluation.md').write_text(
            '# 单箱侧面评估\n\n仅统计提供人工标注的帧；缺失的真值为 UNVERIFIED。\n\n'
            + '```json\n' + json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)
            + '\n```\n', encoding='utf8')
        print(f'evaluation: {args.out}')
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f'error: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
