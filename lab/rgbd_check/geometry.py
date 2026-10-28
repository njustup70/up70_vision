"""几何前置证据检查：相同 K/frame_id 不能代替像素网格与深度 Z 语义。"""

from .status import FAIL, PASS, SKIP, UNVERIFIED


def assess_geometry(scan, meta: dict) -> dict:
    capture = meta.get("capture") or {}
    evidence = capture.get("geometry_validation") or {}
    result = {"status": UNVERIFIED, "reasons": [], "evidence": evidence,
              "camera_info": scan.infos}
    if capture.get("mode") in ("raw", "sync"):
        result.update(status=SKIP, reasons=["未启用 D2C，空间对齐不适用"])
        return result
    if scan.geometry_changes:
        result.update(status=FAIL, reasons=scan.geometry_changes)
        return result
    color, depth = scan.infos.get("color_info"), scan.infos.get("depth_info")
    if not color or not depth:
        result["reasons"].append("缺配套 CameraInfo")
    else:
        for role, info in (("color", color), ("depth", depth)):
            frames = scan.frames.get(role, [])
            if not frames or any((f.width, f.height, f.frame_id) !=
                                 (info["width"], info["height"], info["frame_id"]) for f in frames):
                result["reasons"].append(f"{role} 图像与 CameraInfo 的尺寸或 frame 不一致")
        if (color["width"], color["height"], color["frame_id"]) != \
                (depth["width"], depth["height"], depth["frame_id"]):
            result["reasons"].append("D2C 两路尺寸或光学系不同")
    required = ("target_pixel_grid", "depth_z_definition", "backprojection", "optical_frame", "sources")
    if evidence.get("status") != PASS or evidence.get("same_pixel_grid") is not True or \
            any(not evidence.get(key) for key in required):
        result["reasons"].append("尚无完整的同像素网格、深度 Z 定义和反投影依据")
        if color and depth and color.get("d") != depth.get("d"):
            result["reasons"].append("彩色 D 非同于对齐深度 D：须核查 raw/去畸变网格，不能直接复制 K 或改 frame_id")
    if not result["reasons"]:
        result["status"] = PASS
    return result
