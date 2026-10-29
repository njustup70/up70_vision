"""读取真实深度 bag，逐帧检测并保存可复算的数据。

用法：python -m lab.depth_report.analyze bags/raw_003 --out output/depth/raw_003
依赖 rosbags；绘图由独立模块完成。位置是深度光学系毫米，时间为图像 header。
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from lab.bag_detect import Truth, decode_depth, evaluate_checks, temporal_median
from vision.depth import Intrinsics, SuctionParams, analyze

DEPTH = '/camera/depth/image_raw'
INFO = '/camera/depth/camera_info'


def classify(result, threshold=0.5):
    """无合格候选、一个候选、多候选分别返回 missing、ok、ambiguous。"""
    if result.get('reason'):
        return 'failed', None
    candidates = [c for c in result['clusters'] if c['score'] >= threshold]
    if not candidates:
        return 'missing', None
    if len(candidates) != 1:
        return 'ambiguous', None
    chosen = candidates[0]
    if np.dot(chosen['top_normal'], chosen['top_center_mm']) >= 0:
        return 'invalid_normal', None
    return 'ok', chosen


def summarize(rows):
    """只汇总唯一合格候选；其他帧保留并分别计数。"""
    counts = dict(Counter(row['status'] for row in rows))
    valid = [row for row in rows if row['status'] == 'ok']
    result = {'counts': counts, 'frames': len(rows), 'valid_frames': len(valid),
              'position_median_mm': None, 'position_std_mm': None,
              'max_axis_std_mm': None, 'three_sigma_mm': None}
    if valid:
        positions = np.asarray([[r['x_mm'], r['y_mm'], r['z_mm']] for r in valid])
        result.update(position_median_mm=np.median(positions, axis=0).tolist(),
                      position_std_mm=np.std(positions, axis=0).tolist(),
                      max_axis_std_mm=float(np.std(positions, axis=0).max()),
                      three_sigma_mm=(3 * np.std(positions, axis=0)).tolist())
    return result


def read_bag(path):
    """读取原始图像与完整内参；保留 header 时间和 frame_id。"""
    from rosbags.highlevel import AnyReader
    frames, stamps, frame_ids = [], [], []
    with AnyReader([path]) as reader:
        info_connections = [c for c in reader.connections if c.topic == INFO]
        depth_connections = [c for c in reader.connections if c.topic == DEPTH]
        if not info_connections or not depth_connections:
            raise ValueError('bag 缺深度图或 camera_info')
        info = None
        for conn, _, raw in reader.messages(connections=info_connections):
            msg = reader.deserialize(raw, conn.msgtype)
            info = {'frame_id': str(msg.header.frame_id), 'width': int(msg.width),
                    'height': int(msg.height), 'distortion_model': str(msg.distortion_model),
                    'k': list(map(float, msg.k)), 'd': list(map(float, msg.d)),
                    'r': list(map(float, msg.r)), 'p': list(map(float, msg.p))}
            break
        for conn, bag_stamp, raw in reader.messages(connections=depth_connections):
            msg = reader.deserialize(raw, conn.msgtype)
            frames.append(decode_depth(msg))
            stamps.append(int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec))
            frame_ids.append(str(msg.header.frame_id))
    if not frames or info is None:
        raise ValueError('bag 没有深度帧或内参')
    if any(f != info['frame_id'] for f in frame_ids) or not info['frame_id']:
        raise ValueError('深度图与 camera_info 的 frame_id 为空或不一致')
    return frames, stamps, info


def read_truth(path):
    """读取手工参考，缺字段标记为未知，不推断或套用其他 bag 的数值。"""
    meta_path = path / 'meta.yaml'
    meta = yaml.safe_load(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    meta = meta or {}
    scene = meta.get('scene') or {}
    objects = scene.get('objects') or []
    obj = objects[0] if objects else {}
    size = obj.get('size_mm')
    return Truth(height_mm=obj.get('height_mm'), size_mm=tuple(size) if size else None,
                 tilt_deg=(meta.get('setup') or {}).get('tilt_deg'),
                 n_targets=len(objects) if 'objects' in scene else None), meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames, stamps, info = read_bag(args.bag)
    k = info['k']
    intr = Intrinsics(k[0], k[4], k[2], k[5], info['width'], info['height'])
    params = SuctionParams()
    if len(frames) < 60:
        raise ValueError(f'需要至少 60 帧，实际 {len(frames)}')
    median_depth, coverage = temporal_median(frames[:60])
    median_result = analyze(median_depth, intr, params, np.random.default_rng(args.seed))
    if median_result['reason']:
        raise ValueError(f"时间中位数图检测失败: {median_result['reason']}")
    median_status, chosen = classify(median_result)
    best = max(median_result['clusters'], key=lambda c: c['score'], default=None)
    truth, metadata = read_truth(args.bag)
    checks = evaluate_checks(median_result, truth, intr)
    if truth.n_targets == 0:
        checks[3] = {'check': '4 顶面平行', 'measured': '-', 'expected': '空场景无物块',
                     'error': '-', 'status': 'SKIP'}
    elif truth.n_targets is None:
        checks[3]['expected'] = '目标及顶面与桌面平行关系未确认'
        checks[3]['status'] = 'SKIP'
    rows = []
    for index, (frame, stamp) in enumerate(zip(frames, stamps)):
        start = time.perf_counter()
        try:
            result = analyze(frame, intr, params, np.random.default_rng(args.seed))
            status, selected = classify(result)
            reason = result.get('reason') or ('' if status == 'ok' else status)
            clusters = result['clusters']
        except (ValueError, RuntimeError) as exc:
            status, selected, clusters, reason = 'failed', None, [], str(exc)
        row = {'frame': index, 'timestamp_ns': stamp, 'status': status, 'reason': reason,
               'positive_candidates': sum(c['score'] > 0 for c in clusters),
               'qualified_candidates': sum(c['score'] >= 0.5 for c in clusters),
               'candidate_scores': json.dumps([{'label': c['label'], 'score': c['score']} for c in clusters]),
               'best_score': max((c['score'] for c in clusters), default=None),
               'cluster_count': len(clusters), 'elapsed_ms': (time.perf_counter() - start) * 1000,
               'x_mm': None, 'y_mm': None, 'z_mm': None, 'nx': None, 'ny': None, 'nz': None,
               'score': None, 'label': None, 'jump_mm': None, 'suspected_switch': False}
        if selected:
            for key, value in zip(('x_mm', 'y_mm', 'z_mm'), selected['top_center_mm']):
                row[key] = value
            for key, value in zip(('nx', 'ny', 'nz'), selected['top_normal']):
                row[key] = value
            row['score'], row['label'] = selected['score'], selected['label']
            if rows and rows[-1]['status'] == 'ok':
                before = np.array([rows[-1][key] for key in ('x_mm', 'y_mm', 'z_mm')])
                row['jump_mm'] = float(np.linalg.norm(np.array(selected['top_center_mm']) - before))
                row['suspected_switch'] = row['jump_mm'] > 30.0
        rows.append(row)
        if (index + 1) % 30 == 0:
            print(f'processed {index + 1}/{len(frames)}', flush=True)
    summary = summarize(rows)
    summary['suspected_switch_frames'] = [r['frame'] for r in rows if r['suspected_switch']]
    summary['abnormal_frames'] = [r['frame'] for r in rows if r['status'] != 'ok' or r['suspected_switch']]
    summary['jump_threshold_mm'] = 30.0
    if truth.n_targets == 0:
        summary['empty_scene_candidate_frames'] = sum(r['qualified_candidates'] > 0 for r in rows)
        summary['empty_scene_candidate_fraction'] = summary['empty_scene_candidate_frames'] / len(rows)
    summary['static_scene_verification'] = 'unverified: inspect first/middle/last images; static cannot be established from a single bag automatically'
    summary['absolute_position_accuracy'] = 'unverified: no independent optical XYZ ground truth'
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    payload = {'bag': str(args.bag), 'revision': revision, 'working_tree_dirty': dirty,
               'camera_info': info, 'depth_unit': 'mm (recorded configuration; verify)',
               'image_geometry': 'rectification/distortion relationship not independently verified',
               'params': asdict(params), 'seed': args.seed, 'score_threshold': 0.5,
               'median_depth_frames': 60, 'median_depth_coverage': coverage,
               'median_depth_status': median_status, 'median_depth_target': chosen,
               'median_depth_best_diagnostic': best, 'median_depth_plane': median_result['plane'],
               'median_depth_clusters': median_result['clusters'], 'checks': checks,
               'truth': {'height_mm': truth.height_mm, 'size_mm': truth.size_mm,
                         'tilt_deg': truth.tilt_deg, 'n_targets': truth.n_targets,
                         'center_optical_mm': None, 'metadata': metadata},
               'single_frame_summary': summary}
    with (args.out / 'frames.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.out / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding='utf-8')
    np.savez_compressed(args.out / 'visual_data.npz', median_depth=median_depth,
                        label_map=median_result['label_map'],
                        sample_first=frames[0], sample_middle=frames[len(frames)//2],
                        sample_last=frames[-1])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
