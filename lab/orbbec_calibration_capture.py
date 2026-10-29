"""Orbbec 彩色标定采图：订阅 ROS 图像，窗口按空格保存一个静止姿态。

用大白话说
    相机由现有 RGB-D launch 启动。这里仅预览原始彩色图，复用仓库的棋盘检测，
    每按一次空格保存一张无叠字原图。每次运行建新目录，原有图片和标定不覆盖。
    采完再用 lab/calibrate_camera.py --images 解算，不在这个窗口自动更新内参。

职责与边界
    ROS 读取、GUI、文件保存属于 lab；不修改相机、D2C 或运行时接口。
    ROS 延迟导入，因此离线测试与 --help 不依赖 ROS。窗口文字使用 ASCII。
    格宽由用户提供，默认 23 mm 是近似值；会在 session.json 明确记录来源。

用法（先 source ROS 与仓库 DDS 配置）
    python -m lab.orbbec_calibration_capture --square-mm 23
    Space 保存；Q/Esc 结束。图像在会话的 images/，标定输出建议用 result.yaml。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from vision.aruco.calibration import BoardSpec, find_board_corners

# 沿用当前已验证档位；标定途中换档会让同一批图的 K 失去意义。
IMAGE_SIZE = (848, 480)
# 3 次检测且持续半秒，防止刚移动完或只检测到一帧时误拍。
STABLE_SECONDS = 0.5
STABLE_CHECKS = 3
# 平均角点位移小于 2 px 视为暂时静止，只是采图提示，不是标定验收指标。
STABLE_MOTION_PX = 2.0


class CaptureSession:
    """只追加一轮采图，记录原图、原始时间戳和配套 CameraInfo。

    用大白话说
        root 每次必须是新目录；图片和旁边的 JSON 都用排他创建，遇到已有文件报错。
        board 表示本轮角点布局和格宽，count 是已成功保存张数；不写旧标定路径。
    """

    def __init__(self, root: Path, board: BoardSpec, size_source: str):
        """新建会话；root 已存在时报 FileExistsError，不恢复或清空旧目录。"""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.images = self.root / "images"
        self.images.mkdir()
        self.board = board
        self.count = 0
        self.saved_stamps = set()
        record = {
            "created_at": datetime.now().astimezone().isoformat(),
            "board": board.to_dict(), "square_size_source": size_source,
            "image_size": list(IMAGE_SIZE), "color_topic": "/camera/color/image_raw",
            "calibration_output": str(self.root / "result.yaml"),
            "note": "Raw color only; no calibration, rectification or D2C changes applied",
        }
        with (self.root / "session.json").open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)

    @classmethod
    def resume(cls, root: Path, board: BoardSpec):
        """显式续拍完整会话；只验证旧数据，遇到缺配对文件或参数变化则拒绝。"""
        root = Path(root).resolve()
        record = json.loads((root / "session.json").read_text(encoding="utf-8"))
        if record.get("board") != board.to_dict() or record.get("image_size") != list(IMAGE_SIZE):
            raise ValueError("Resume board or image size differs from original session")
        obj = cls.__new__(cls)
        obj.root, obj.images, obj.board = root, root / "images", board
        obj.count, obj.saved_stamps = 0, set()
        images = sorted(obj.images.glob("calib_*.png"))
        if len(list(obj.images.glob("calib_*.json"))) != len(images):
            raise ValueError("Incomplete image/metadata pairs; old files are preserved")
        for index, path in enumerate(images, 1):
            if path.name != f"calib_{index:03d}.png":
                raise ValueError("Image sequence has gaps; old files are preserved")
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            stamp = metadata["stamp_ns"]
            image = cv2.imread(str(path))
            if image is None or image.shape != (480, 848, 3) or stamp <= 0 or stamp in obj.saved_stamps:
                raise ValueError("Invalid saved image or timestamp; old files are preserved")
            obj.saved_stamps.add(stamp)
            obj.count += 1
        return obj

    def save(self, frame, corners, stamp_ns: int, camera_info: dict) -> Path:
        """保存原图与来源；拒绝错误尺寸、缺角点、重复/零时间戳及已有文件。

        :param frame: 原始 BGR uint8 图像，无 HUD。
        :param corners: 检测到的完整棋盘角点。
        :param stamp_ns: 图像原始 header 纳秒，绝不使用接收时刻替代。
        :param camera_info: 当前配置的相机内参与 frame_id 快照。
        :return: 新图片路径；IO 失败保留现场，抛异常而不覆盖。
        """
        if frame.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3) or frame.dtype != np.uint8:
            raise ValueError("Expected raw 848x480 BGR uint8")
        if corners is None or corners.shape != (self.board.inner_corners, 1, 2):
            raise ValueError("A full chessboard is required")
        if stamp_ns <= 0 or stamp_ns in self.saved_stamps:
            raise ValueError("Invalid or already saved image timestamp")
        if not camera_info:
            raise ValueError("CameraInfo is required")
        path = self.images / f"calib_{self.count + 1:03d}.png"
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise OSError("PNG encoding failed")
        # xb 与 x 都是排他创建，检查和写入之间即使有另一进程也不能覆盖。
        with path.open("xb") as handle:
            handle.write(encoded.tobytes())
        with path.with_suffix(".json").open("x", encoding="utf-8") as handle:
            json.dump({"stamp_ns": stamp_ns, "camera_info": camera_info,
                       "corners_px": corners.reshape(-1, 2).tolist()}, handle, indent=2)
        self.saved_stamps.add(stamp_ns)
        self.count += 1
        return path


def parser():
    """采图参数；输出目录可指定，但必须不存在，默认时间戳加随机后缀。"""
    ap = argparse.ArgumentParser(description=__doc__)
    paths = ap.add_mutually_exclusive_group()
    paths.add_argument("--out", type=Path, help="全新会话目录，已存在则拒绝")
    paths.add_argument("--resume", type=Path, help="显式续拍此会话，保留旧图并继续编号")
    ap.add_argument("--square-mm", type=float, default=23.0, help="格宽毫米，默认近似 23")
    ap.add_argument("--size-source", choices=("approximate", "measured"),
                    default="approximate", help="默认 approximate，不能把口述近似写成尺测")
    return ap


def main(argv=None):
    """订阅原始图像并开窗；只响应空格存图，Q/Esc 退出并保留全部已存数据。"""
    args = parser().parse_args(argv)
    board = BoardSpec(cols=6, rows=9, square_mm=args.square_mm)
    root = args.out or Path("data/orbbec_color_calibration") / (
        datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    session = (CaptureSession.resume(args.resume, board) if args.resume else
               CaptureSession(root.resolve(), board, args.size_source))
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    node = rclpy.create_node("orbbec_calibration_capture")
    latest, info = {}, {}

    def on_image(msg):
        """只接受当前档位原始 RGB/BGR；按 step 去 padding，再统一为 BGR。"""
        if msg.encoding not in ("rgb8", "bgr8") or (msg.width, msg.height) != IMAGE_SIZE:
            return
        rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        frame = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        latest.update(frame=frame[:, :, ::-1].copy() if msg.encoding == "rgb8" else frame.copy(),
                      stamp=int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec),
                      frame_id=str(msg.header.frame_id), received=time.monotonic())

    def on_info(msg):
        """原样保存内参与畸变，不将彩色 K 复制到任何深度消息。"""
        info.update(width=int(msg.width), height=int(msg.height), frame_id=str(msg.header.frame_id),
                    k=list(map(float, msg.k)), d=list(map(float, msg.d)),
                    r=list(map(float, msg.r)), p=list(map(float, msg.p)),
                    distortion_model=str(msg.distortion_model))

    node.create_subscription(Image, "/camera/color/image_raw", on_image, qos_profile_sensor_data)
    node.create_subscription(CameraInfo, "/camera/color/camera_info", on_info, qos_profile_sensor_data)
    title = "Orbbec calibration - SPACE capture / Q finish"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1018, 576)
    last_stamp, last_corners, stable_since, stable_count = None, None, 0.0, 0
    notice, notice_until = "", 0.0
    print(f"SESSION {session.root}\nSPACE capture | Q / Esc finish", flush=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if not latest:
                blank = np.zeros((480, 848, 3), np.uint8)
                cv2.putText(blank, "Waiting for raw color...", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 255), 2)
                cv2.imshow(title, blank)
                if cv2.waitKey(20) & 255 in (27, ord("q")):
                    break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            frame, stamp = latest["frame"], latest["stamp"]
            corners = find_board_corners(frame, board)
            now = time.monotonic()
            if stamp != last_stamp:
                if corners is not None and last_corners is not None:
                    movement = float(np.linalg.norm(corners - last_corners, axis=2).mean())
                    stable_count = stable_count + 1 if movement < STABLE_MOTION_PX else 1
                else:
                    stable_count = 1 if corners is not None else 0
                if stable_count <= 1:
                    stable_since = now
                last_stamp, last_corners = stamp, corners
            fresh = now - latest["received"] < 0.5
            info_ok = bool(info) and (info["width"], info["height"]) == IMAGE_SIZE and (
                info["frame_id"] == latest["frame_id"])
            ready = corners is not None and stable_count >= STABLE_CHECKS and (
                now - stable_since >= STABLE_SECONDS) and fresh and info_ok
            if corners is None:
                hint = "NO BOARD - show all 6x9 inner corners"
            elif not info_ok:
                hint = "WAIT - compatible CameraInfo required"
            elif not ready:
                hint = "HOLD STILL - waiting for stable corners"
            else:
                hint = "READY - SPACE to save"
            if not fresh:
                hint = "STREAM STALE - cannot save"
            shown = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(shown, (6, 9), corners, True)
            cv2.rectangle(shown, (0, 0), (848, 65), (20, 20, 20), -1)
            cv2.putText(shown, f"Saved {session.count} | {hint}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 240, 0) if ready else (0, 200, 255), 1)
            cv2.putText(shown, notice if now < notice_until else "SPACE save | Q/Esc finish | raw images only",
                        (10, 51), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            cv2.imshow(title, shown)
            key = cv2.waitKey(1) & 255
            if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                print(f"EXIT requested by {'Q/Esc' if key in (27, ord('q')) else 'window close'}", flush=True)
                break
            if key == 32:
                if ready:
                    try:
                        path = session.save(frame, corners, stamp, dict(info))
                        print(f"SAVED {session.count}: {path.name}", flush=True)
                        notice = f"SAVED {session.count} - change position AND tilt"
                        stable_count, stable_since, last_corners = 0, now, None
                    except (OSError, ValueError) as exc:
                        notice = f"NOT SAVED: {exc}"
                        print(notice, flush=True)
                else:
                    notice = f"NOT SAVED - {hint}"
                    print(notice, flush=True)
                notice_until = now + 2.0
    except KeyboardInterrupt:
        # 重开窗口时用 SIGINT 结束旧进程，保留已存图片并正常清理 GUI。
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        # ROS 的 SIGINT 处理器可能已经关闭 context，不能重复 shutdown。
        rclpy.try_shutdown()
        print(f"FINISHED {session.count} images in {session.images}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
