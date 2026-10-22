"""把 YOLO 权重导出成 ONNX（顺带写一份类名 sidecar）。

为什么需要它
    仓库里跑推理只依赖 OpenCV（`core/detectors/yolo.py` 用 `cv2.dnn`），而导出
    ONNX 需要 `ultralytics` + `torch`。所以导出**只在开发机上做一次**，产物是：
        assets/models/yolov8n.onnx          12.9 MB 的模型
        assets/models/yolov8n.names.json    类名表（零依赖可读，容器里不需要 onnx 包）
    两者都不进版本库（`.gitignore` 已忽略 *.onnx / *.pt）。

怎么跑
    # 在装了 ultralytics 的环境里（本机是 ~/.venvs/yolo；该 venv 里也要 pip install -e 本仓库）
    ~/.venvs/yolo/bin/python -m up70_vision export-yolo --weights yolov8n.pt
    # 自定义：换模型、换输入尺寸
    ~/.venvs/yolo/bin/python -m up70_vision export-yolo --weights yolo11s.pt --imgsz 480 --out assets/models

导出后怎么用
    # 注意：判色族/YOLO 走**批处理入口**；live 是 ArUco 专用（要画四边形和位姿轴）
    python -m up70_vision run --set detector.name=yolo \
        --set input.source=replay --set input.replay.path=assets/aruco/scenes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 仓库根：src/up70_vision/tools/export_yolo.py 往上三层（tools -> up70_vision -> src -> 根）。
# 用 __file__ 而不是 cwd：默认输出目录固定在仓库的 assets/models/，从哪调用都一样。
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "assets" / "models"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 YOLO .pt 导出为 ONNX，并写类名 sidecar（不训练、不改权重）")
    parser.add_argument("--weights", required=True, metavar="PATH",
                        help="输入权重（如 yolov8n.pt；不存在时 ultralytics 会自动下载）")
    parser.add_argument("--imgsz", type=int, default=640, metavar="N",
                        help="导出输入边长（32 的倍数，默认 640；运行时 imgsz 必须与它一致）")
    parser.add_argument("--opset", type=int, default=12, metavar="N",
                        help="ONNX opset（默认 12；容器里 cv2 4.14 实测可用）")
    parser.add_argument("--out", default=str(DEFAULT_OUT), metavar="DIR",
                        help=f"输出目录（默认 {DEFAULT_OUT}）")
    parser.add_argument("--name", default=None, metavar="STEM",
                        help="输出文件名（不含扩展名；默认沿用输入的 stem）")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        print("❌ 导出需要 ultralytics（和 torch）。请用装了它的解释器跑，例如：")
        print("     ~/.venvs/yolo/bin/python -m up70_vision export-yolo --weights yolov8n.pt")
        return 2

    out_dir = Path(os.path.expanduser(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    # 记一下传入的 .pt 是不是本来就在：不在时 ultralytics 会自动下载到**当前目录**，
    # 导出完得把它挪进权重目录，否则仓库根会多出一个 6.5 MB 的野文件（还容易误提交）。
    given = Path(args.weights)
    downloaded = not given.exists()
    model = YOLO(args.weights)
    names = {int(k): str(v) for k, v in (model.names or {}).items()}

    exported = Path(model.export(format="onnx", imgsz=int(args.imgsz), opset=int(args.opset)))
    stem = args.name or Path(args.weights).stem
    target_onnx = out_dir / f"{stem}.onnx"
    if exported.resolve() != target_onnx.resolve():
        target_onnx.write_bytes(exported.read_bytes())
        exported.unlink(missing_ok=True)

    sidecar = out_dir / f"{stem}.names.json"
    sidecar.write_text(json.dumps({str(k): v for k, v in sorted(names.items())},
                                  ensure_ascii=False, indent=2), encoding="utf-8")

    if downloaded:
        cached = out_dir / f"{stem}.pt"
        if given.exists() and given.resolve() != cached.resolve():
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(given.read_bytes())
            given.unlink(missing_ok=True)
            print(f"✅ 权重    : {cached}（自动下载的 .pt 已挪进来，仓库根不再留野文件）")

    print(f"✅ 模型    : {target_onnx}（{target_onnx.stat().st_size / 1e6:.1f} MB）")
    print(f"✅ 类名表  : {sidecar}（{len(names)} 类）")
    print(f"   运行时用 : --set detector.name=yolo --set detector.yolo.weights={target_onnx.relative_to(ROOT)}")
    print("⚠️  权重不进版本库（.gitignore 已忽略 *.onnx）；YOLOv8 权重是 AGPL-3.0，对外发布前确认许可")
    return 0


if __name__ == "__main__":
    sys.exit(main())
