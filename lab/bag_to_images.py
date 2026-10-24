"""把 bag 里的图像话题抽成 png，供标定和离线分析用。

跑法（容器内）
    docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision \
        python lab/bag_to_images.py bags/calib_ir --topic /camera/ir/image_raw \
        --out data/calib_ir

为什么需要它
    lab/calibrate_camera.py 的采图走 `vision.sources.build_source`，也就是海康相机源，
    接不了 ROS 话题。标定算法（vision/aruco/calibration.py）写得挺全，没必要为了接
    ROS 再写一份，所以中间加这一层：ROS 里录 bag → 抽成 png → 走 `--images` 复用现成标定。

    这个需求不止 A3 有：S2 要看真实失效模式、B 阶段要对照点云复查，都要把 bag 里的
    帧导出来看。所以单独做个工具，而不是塞进标定脚本。

支持编码
    mono8 / 8UC1     红外图。A3 标深度内参走这条（关激光后 IR 图能看到棋盘格）
    mono16 / 16UC1   深度图，按 16bit 无损存
    bgr8 / rgb8      彩色图，统一存成 BGR（OpenCV 约定）

已知的坑
    · step 不一定等于 width*bpp。驱动可能按行对齐补 padding，直接按 width*bpp 铺开
      会让图像斜着错位，而且错位后「看起来还挺像一张图」，很难发现。
    · 16UC1 存出来是 16bit png，用普通看图软件打开会全白或全黑——那是没拉窗，
      不是数据坏了。标定只用 mono8，深度图抽出来是给人复查用的。
    · 标定图别抽太密：相邻帧几乎是同一姿势，calibrate_camera 的去重闸会拒掉，
      白占目录。默认 `--every 10`（30fps 下约 0.33 秒一张）够用。
    · 文件名固定成 `calib_XX.png`，因为 calibrate_camera 按 `*.png` glob 后排序，
      顺序影响「最像的那张」提示里的编号，不影响标定结果。

退出码：0 成功；1 bag 里没有该话题或一帧都没抽出来；2 参数非法。
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# encoding -> (dtype, 通道数)
_ENCODINGS = {
    "mono8": (np.uint8, 1),
    "8UC1": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    "16UC1": (np.uint16, 1),
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
}


def decode_image(msg) -> np.ndarray:
    """把 sensor_msgs/Image 解成 numpy 数组，形状 (H,W) 或 (H,W,3)。

    统一处理 step padding，并把 rgb8 转成 BGR——下游 cv2.imwrite 按 BGR 理解三通道。
    """
    encoding = str(getattr(msg, "encoding", ""))
    if encoding not in _ENCODINGS:
        raise ValueError(
            f"不支持的编码 {encoding!r}，本工具只认 {sorted(_ENCODINGS)}。"
            f"遇到别的编码先确认它是什么，别猜着解。"
        )

    dtype, channels = _ENCODINGS[encoding]
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    row_bytes = width * np.dtype(dtype).itemsize * channels

    raw = bytes(msg.data)
    if step == row_bytes:
        flat = np.frombuffer(raw, dtype=dtype)
    else:
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)[:, :row_bytes]
        flat = np.ascontiguousarray(rows).view(dtype)

    image = flat.reshape(height, width) if channels == 1 else flat.reshape(height, width, channels)
    if encoding == "rgb8":
        image = image[:, :, ::-1].copy()
    return image


def main() -> int:
    """解析参数、按间隔抽帧、落盘并打印摘要。"""
    parser = argparse.ArgumentParser(
        description="把 bag 里的图像话题抽成 png。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bag", type=Path, help="bag 目录")
    parser.add_argument("--topic", default="/camera/ir/image_raw",
                        help="图像话题，默认红外（A3 标定用）")
    parser.add_argument("--out", default="data/bag_images", help="输出目录")
    parser.add_argument("--every", type=int, default=10,
                        help="每隔多少帧存一张，默认 10")
    parser.add_argument("--max", type=int, default=40, help="最多存多少张，默认 40")
    parser.add_argument("--prefix", default="calib", help="文件名前缀，默认 calib")
    args = parser.parse_args()

    if args.every < 1:
        print("--every 必须 >= 1", file=sys.stderr)
        return 2
    if not args.bag.exists():
        print(f"bag 不存在：{args.bag}", file=sys.stderr)
        return 1

    from rosbags.highlevel import AnyReader

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen = 0
    saved = 0
    shape_text = ""

    with AnyReader([args.bag]) as reader:
        conns = [c for c in reader.connections if c.topic == args.topic]
        if not conns:
            topics = sorted({c.topic for c in reader.connections})
            print(f"{args.bag} 里没有 {args.topic}。实际话题：{topics}", file=sys.stderr)
            return 1

        for index, (conn, _, raw) in enumerate(reader.messages(connections=conns)):
            seen += 1
            if index % args.every:
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            image = decode_image(msg)
            path = out_dir / f"{args.prefix}_{saved + 1:02d}.png"
            # imwrite 失败时只返回 False 不抛异常，必须显式查
            if not cv2.imwrite(str(path), image):
                print(f"写图失败：{path}", file=sys.stderr)
                return 1
            if not shape_text:
                shape_text = f"{image.shape} dtype={image.dtype} encoding={msg.encoding}"
            saved += 1
            if saved >= args.max:
                break

    print(f"话题      : {args.topic}")
    print(f"看过帧数  : {seen}")
    print(f"落盘张数  : {saved} -> {out_dir}/")
    if shape_text:
        print(f"图像规格  : {shape_text}")
    if saved == 0:
        print("一张都没抽出来：检查 --every 是否大于总帧数，或话题里是否真有数据。",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
