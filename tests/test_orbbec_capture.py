"""Orbbec 采图保护：不覆盖历史数据，拒绝不完整棋盘和重复帧，不需要 ROS。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pytest

from lab.calibrate_camera import write_calibration
from lab.orbbec_calibration_capture import CaptureSession
from vision.aruco.calibration import BoardSpec, CameraIntrinsics


def test_existing_session_is_preserved(tmp_path):
    """已存在的会话拒绝启动，旧内容保持原字节。"""
    root = tmp_path / "old"
    root.mkdir()
    original = root / "session.json"
    original.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        CaptureSession(root, BoardSpec(6, 9, 23), "approximate")
    assert original.read_bytes() == b"original"


def test_save_preserves_raw_pixels_and_rejects_repeat_or_missing_board(tmp_path):
    """PNG 保持原始像素，缺角点和重复时间戳不能增加已存张数。"""
    session = CaptureSession(tmp_path / "new", BoardSpec(6, 9, 23), "approximate")
    frame = np.full((480, 848, 3), (12, 34, 56), dtype=np.uint8)
    corners = np.zeros((54, 1, 2), np.float32)
    with pytest.raises(ValueError):
        session.save(frame, None, 100, {"frame_id": "color"})
    path = session.save(frame, corners, 100, {"frame_id": "color"})
    assert np.array_equal(cv2.imread(str(path)), frame)
    with pytest.raises(ValueError):
        session.save(frame, corners, 100, {"frame_id": "color"})
    assert session.count == 1
    assert len(list(session.images.glob("*.png"))) == 1


def test_existing_image_is_not_overwritten(tmp_path):
    """即使目标文件在会话创建后才出现，排他写入仍保护原文件。"""
    session = CaptureSession(tmp_path / "new", BoardSpec(6, 9, 23), "approximate")
    path = session.images / "calib_001.png"
    path.write_bytes(b"old image")
    with pytest.raises(FileExistsError):
        session.save(np.zeros((480, 848, 3), np.uint8),
                     np.zeros((54, 1, 2), np.float32), 100, {"frame_id": "color"})
    assert path.read_bytes() == b"old image"


def test_resume_appends_and_preserves_old_images(tmp_path):
    """续拍从下一编号开始，历史图片保持原字节；格宽变化时拒绝续拍。"""
    root = tmp_path / "session"
    board = BoardSpec(6, 9, 23)
    session = CaptureSession(root, board, "approximate")
    frame = np.zeros((480, 848, 3), np.uint8)
    corners = np.zeros((54, 1, 2), np.float32)
    old_path = session.save(frame, corners, 100, {"frame_id": "color"})
    original = old_path.read_bytes()
    continued = CaptureSession.resume(root, board)
    assert continued.count == 1
    new_path = continued.save(frame, corners, 101, {"frame_id": "color"})
    assert new_path.name == "calib_002.png"
    assert old_path.read_bytes() == original
    with pytest.raises(ValueError):
        CaptureSession.resume(root, BoardSpec(6, 9, 24))


def test_resume_rejects_incomplete_pair(tmp_path):
    """缺图片的元数据会阻止续拍，不能跳过坏记录后冒充完整数据。"""
    root = tmp_path / "session"
    board = BoardSpec(6, 9, 23)
    CaptureSession(root, board, "approximate")
    (root / "images" / "calib_001.json").write_text('{}')
    with pytest.raises(ValueError):
        CaptureSession.resume(root, board)


def test_no_overwrite_calibration_preserves_original(tmp_path):
    """标定输出采用排他写入时，已有 YAML 不能被覆盖。"""
    result = CameraIntrinsics((848, 480), np.eye(3), np.zeros(5), 0.2)
    path = tmp_path / "result.yaml"
    write_calibration(str(path), result, overwrite=False)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_calibration(str(path), result, overwrite=False)
    assert path.read_bytes() == original
