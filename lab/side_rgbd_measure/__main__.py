"""侧面离线测量命令：读报告、选框或加载 YOLO、保存可复查结果。"""
import argparse
import json
import os
from pathlib import Path

import cv2

from lab.rgbd_check.bag import BagError
from vision.yolo import YoloDetector
from .geometry import PlaneParams
from .input import load_frames, rectified_color_and_mask
from .run import run_frames


def main(argv=None):
    """执行测量；输入错误和输出目录存在时退出 2。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path, help='rgbd_check analyze 报告目录')
    parser.add_argument('--out', type=Path, required=True, help='不存在的新输出目录')
    parser.add_argument('--bag', type=Path, help='与报告一致的原 bag；多帧时必需')
    parser.add_argument('--frames', type=int, default=1, help='稳定段均匀采样数量，默认 1')
    parser.add_argument('--bbox', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='校正网格的人工半开框；未指定权重和框时打开窗口')
    parser.add_argument('--weights', type=Path, help='搭档的 ONNX 权重')
    parser.add_argument('--imgsz', type=int, default=640, help='模型输入边长')
    parser.add_argument('--class-id', type=int, action='append', help='保留指定类别，可重复')
    parser.add_argument('--class-names', nargs='+', help='按类别索引排序的全部类名')
    parser.add_argument('--conf', type=float, default=0.25, help='模型置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='模型 NMS IoU 阈值')
    parser.add_argument('--choices', type=Path, help='按 pair_id 指定 detection / plane 的 JSON')
    parser.add_argument('--leaf-mm', type=float, default=4.0)
    parser.add_argument('--threshold-mm', type=float, default=6.0)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--min-points', type=int, default=120)
    args = parser.parse_args(argv)
    try:
        if args.bbox and args.weights:
            raise ValueError('--bbox and --weights are mutually exclusive')
        if args.out.exists():
            raise ValueError('output directory already exists')
        summary, ci, di, frames = load_frames(args.report, args.bag, args.frames)
        params = PlaneParams(args.leaf_mm, args.threshold_mm, args.iterations, args.seed, args.min_points)
        params.validate()
        choices = json.loads(args.choices.read_text(encoding='utf8')) if args.choices else {}
        if not isinstance(choices, dict):
            raise ValueError('choices must be an object indexed by pair_id')
        detector, box = None, args.bbox
        if args.weights:
            if not args.weights.is_file():
                raise ValueError('model weights not found')
            detector = YoloDetector(weights=str(args.weights), conf=args.conf, iou=args.iou,
                                    imgsz=args.imgsz, classes=args.class_id,
                                    class_names=args.class_names)
        elif box is None:
            if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
                raise ValueError('GUI display unavailable; provide --bbox or --weights')
            image, _ = rectified_color_and_mask(frames[0][1], ci, di)
            x, y, w, h = cv2.selectROI('Select side on rectified color', image, False, False)
            cv2.destroyAllWindows()
            if not w or not h:
                raise ValueError('no box selected')
            box = (int(x), int(y), int(x+w), int(y+h))
        result = run_frames(summary, ci, di, frames, args.out, box=box, detector=detector,
                            params=params, choices=choices, weights=args.weights,
                            requested_count=args.frames)
        print(f"processed {result['actual_count']}/{result['requested_count']} frames: {args.out}")
        return 0
    except (ValueError, KeyError, OSError, BagError, cv2.error, json.JSONDecodeError) as exc:
        print(f'error: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
