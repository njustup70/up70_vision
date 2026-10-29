"""离线人工框 + 对齐深度原型，不发布可执行抓取目标。

输入是 rgbd_check analyze 的报告目录。先用该包的 CameraInfo 将彩色图映射到
对齐深度网格，再在映射后的图中框选。完整深度图仍用于支持平面拟合，框只用
于关联已有深度簇。结果是调试候选，空间对齐和顶面可见性仍需人工独立验证。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from vision.depth import Intrinsics, SuctionParams, analyze


def load_report(path: Path):
    """读取确定性选中帧及配套内参；拒绝不完整或非对齐采集。"""
    summary = json.loads((path / 'summary.json').read_text(encoding='utf8'))
    if (summary.get('capture') or {}).get('mode') != 'aligned':
        raise ValueError('需要 aligned 采集，不能把原始深度当成彩色网格')
    for key in ('capture_integrity', 'sync'):
        if (summary.get(key) or {}).get('status') != 'PASS':
            raise ValueError(f'{key} 未通过，不能使用此配对帧')
    selected = summary.get('selected') or {}
    pair_id = selected.get('pair_id')
    if not pair_id or '/' in pair_id or '\\' in pair_id:
        raise ValueError('报告缺少有效配对编号')
    infos = (summary.get('geometry') or {}).get('camera_info') or {}
    color_info, depth_info = infos.get('color_info'), infos.get('depth_info')
    if not color_info or not depth_info:
        raise ValueError('缺少配套彩色或对齐深度 CameraInfo')
    if color_info.get('frame_id') != depth_info.get('frame_id') or not color_info.get('frame_id'):
        raise ValueError('彩色与对齐深度的光学坐标系不一致')
    if color_info.get('distortion_model') != 'plumb_bob':
        raise ValueError('仅支持 plumb_bob 彩色畸变模型')
    if not depth_info.get('d') or any(abs(float(x)) > 1e-12 for x in depth_info['d']):
        raise ValueError('当前原型只支持零畸变的对齐深度网格')
    with np.load(path / 'selected' / f'{pair_id}.npz', allow_pickle=False) as images:
        color, depth = images['color'].copy(), images['depth'].copy()
    h, w = depth.shape
    if color.shape != (h, w, 3) or color.dtype != np.uint8 or depth.dtype != np.uint16:
        raise ValueError('要求同尺寸 BGR 彩色图和 uint16 毫米深度图')
    for info in (color_info, depth_info):
        if (info.get('width'), info.get('height')) != (w, h):
            raise ValueError('图像与 CameraInfo 尺寸不一致')
    return summary, selected, color, depth, color_info, depth_info


def rectify_color(color: np.ndarray, color_info: dict, depth_info: dict) -> np.ndarray:
    """将原彩色图采样到目标深度 K 的零畸变网格，不插值深度。"""
    kc = np.asarray(color_info['k'], dtype=np.float64).reshape(3, 3)
    kd = np.asarray(depth_info['k'], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(color_info['d'], dtype=np.float64)
    if not np.isfinite(kc).all() or not np.isfinite(kd).all() or not np.isfinite(distortion).all():
        raise ValueError('相机参数包含非有限值')
    h, w = color.shape[:2]
    mx, my = cv2.initUndistortRectifyMap(kc, distortion, None, kd, (w, h), cv2.CV_32FC1)
    return cv2.remap(color, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def select_cluster(labels: np.ndarray, box: tuple, min_pixels: int = 50):
    """比较框与簇的像素重叠；多个有效簇时拒绝猜测。"""
    x1, y1, x2, y2 = box
    inside = labels[y1:y2, x1:x2]
    ids, counts = np.unique(inside[inside > 0], return_counts=True)
    candidates = [(int(label), int(count)) for label, count in zip(ids, counts)
                  if count >= min_pixels]
    if not candidates:
        return 'no_cluster', None, candidates
    if len(candidates) > 1:
        return 'ambiguous', None, candidates
    return 'candidate', candidates[0][0], candidates


def run_fusion(depth: np.ndarray, info: dict, box: tuple) -> tuple:
    """完整场景估计支撑平面，再用框关联簇；输出仅供离线检查。"""
    h, w = depth.shape
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError('框须是图像内非空的半开像素坐标 x1,y1,x2,y2')
    # 离线原型允许画面任意位置的人工框；平面仍由完整场景估计。
    params = SuctionParams(roi_ratio=1.0)
    rw, rh = int(w * params.roi_ratio), int(h * params.roi_ratio)
    x0, y0 = (w - rw) // 2, (h - rh) // 2
    result = {'status': 'unverified', 'reason': '', 'bbox_xyxy': list(box),
              'roi_xyxy': [x0, y0, x0 + rw, y0 + rh],
              'valid_depth_ratio': round(float(np.mean(depth[y1:y2, x1:x2] > 0)), 4),
              'candidate': None, 'overlaps': [], 'plane': None}
    mask = np.zeros((h, w), dtype=np.uint8)
    if x1 < x0 or y1 < y0 or x2 > x0 + rw or y2 > y0 + rh:
        result.update(status='outside_roi', reason='框超出深度算法的中心 ROI，不能忽略被裁掉的区域')
        return result, mask
    intrinsics = Intrinsics.from_camera_info(info['k'], w, h)
    data = analyze(depth, intrinsics, params)
    result['plane'] = data.get('plane')
    if data.get('reason') or 'label_map' not in data:
        result.update(status='no_plane', reason=data.get('reason') or '没有深度簇')
        return result, mask
    labels = data['label_map']
    local_box = (x1-x0, y1-y0, x2-x0, y2-y0)
    # 只关联经过现有深度算法筛选的候选，避免把小噪声连通域当目标。
    valid_ids = {c['label'] for c in data['clusters']}
    eligible = np.where(np.isin(labels, list(valid_ids)), labels, 0)
    status, label, overlaps = select_cluster(eligible, local_box)
    result['overlaps'] = [{'label': i, 'pixels_in_box': n} for i, n in overlaps]
    if label is None:
        result.update(status=status, reason='框中无有效候选簇' if status == 'no_cluster' else '框中有多个候选簇，无法确定目标')
        return result, mask
    chosen = next(c for c in data['clusters'] if c['label'] == label)
    # 即使簇有效，若大部分落在框外，也不确定它属于所选物体。
    count_inside = overlaps[0][1]
    fraction = count_inside / max(int(chosen['points']), 1)
    if fraction < 0.5:
        result.update(status='ambiguous', reason='候选簇多数像素在框外，可能连接背景或其他物体')
        return result, mask
    mask[y0:y0+rh, x0:x0+rw] = (labels == label).astype(np.uint8) * 255
    result.update(status='candidate_unverified', candidate=chosen,
                  reason='候选仅用于离线诊断；支撑面、顶面及物理对齐未独立验证')
    return result, mask


def main(argv=None) -> int:
    """从报告选框并保存 JSON、网格图和簇叠加图；拒绝覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path, help='rgbd_check analyze 输出目录')
    parser.add_argument('--out', type=Path, required=True, help='新建输出目录')
    parser.add_argument('--bbox', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='校正后网格的半开框；不传则在窗口用鼠标选择')
    args = parser.parse_args(argv)
    try:
        summary, selected, color, depth, ci, di = load_report(args.report)
        rectified = rectify_color(color, ci, di)
        if args.bbox is None:
            x, y, w, h = cv2.selectROI('Select box on rectified color', rectified, False, False)
            cv2.destroyAllWindows()
            if w == 0 or h == 0:
                print('未选框，未生成报告')
                return 2
            box = (int(x), int(y), int(x+w), int(y+h))
        else:
            box = tuple(args.bbox)
        result, mask = run_fusion(depth, di, box)
        result.update(session=summary.get('session'), pair_id=selected['pair_id'],
                      color_stamp_ns=selected.get('color_stamp_ns'),
                      depth_stamp_ns=selected.get('depth_stamp_ns'),
                      dt_ms=selected.get('dt_ms'), frame_id=di['frame_id'],
                      color_grid='rectified_to_depth_K', depth_unit='mm',
                      geometry_status=(summary.get('geometry') or {}).get('status'),
                      warning='空间对齐、顶面语义与抓取精度未验收；不可用于控制机械臂')
        args.out.mkdir(parents=True, exist_ok=False)
        overlay = rectified.copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2-1, y2-1), (0, 255, 0), 2)
        for filename, img in [('rectified.png', rectified), ('cluster.png', overlay), ('cluster_mask.png', mask)]:
            if not cv2.imwrite(str(args.out / filename), img):
                raise OSError(f'无法保存 {filename}')
        (args.out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(f"{result['status']}: {result['reason']}\n报告：{args.out}")
        return 0
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f'输入或运行错误：{exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
