"""读 bag：话题发现、图像解码、内参提取。

用大白话说
    `inspect` 与 `analyze` 都从这里拿数据。两遍读：第一遍只收「时间戳与几何信息」，
    第二遍只解选中那几帧的像素。这样 120 秒、两三 GB 的 bag 也不会把内存撑爆
    （一帧 848x480 深度就是 800 KB，300 帧就是 240 MB，全留内存没必要）。

放什么 / 不放什么
    放 `scan()`（元信息一遍）、`load_images()`（按序号取像素）、`decode_image()`（编码处理），
    以及 `BagError` 这个统一异常。**不放**配对（`pairing.py`）、统计（`report.py`）、
    窗口（`annotate.py`）。

依赖谁 / 被谁用
    只依赖 numpy 与 rosbags（函数内 import：没有 rosbags 时 `inspect` 报错，但
    `python -m lab.rgbd_check --help` 仍然能用，测试也能在没有 rosbags 的机器上跑纯逻辑）。

已知的坑
    · `step` 不一定等于 `width * bpp`：驱动可能按行对齐补 padding，直接按 width 铺开
      会让图像斜着错位，而且错位后「看起来还有内容」，很难发现。
    · 对齐后深度可能复用原话题，也可能另发话题；所以话题名一律从参数/配置来，
      代码里不写死。
    · rosbags 解 Image 会把整个 data 复制一遍，第一遍扫描只留 header 字段，
      其余立刻丢掉，否则大 bag 会吃掉几个 GB。
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: 只处理这两类消息；别的一律忽略（点云在采集阶段就关掉了）。
IMAGE_TYPE = "sensor_msgs/msg/Image"
INFO_TYPE = "sensor_msgs/msg/CameraInfo"

#: 角色 -> 该角色接受的消息类型。角色名与 session.yaml 的 capture.topics 对齐。
ROLE_TYPES = {
    "color": IMAGE_TYPE,
    "depth": IMAGE_TYPE,
    "color_info": INFO_TYPE,
    "depth_info": INFO_TYPE,
}

#: 彩色编码 -> 是否 BGR 顺序。RGB 要翻通道，不能直接当 BGR 存。
_COLOR_ENCODINGS = {"bgr8": False, "rgb8": True}
#: 8 位单通道，本阶段只作诊断（例如对齐质量），不参与测量。
_MONO8 = ("mono8", "8UC1")
#: 16 位深度，单位毫米。
_DEPTH16 = ("16UC1", "mono16")


class BagError(RuntimeError):
    """bag 读不了 / 话题不对 / 数据长度不对。CLI 把它映射成退出码 2。"""


@dataclass(frozen=True)
class FrameMeta:
    """一帧的元信息（不含像素），配对与统计只用这些字段。"""

    role: str
    index: int          # 该话题在 bag 里的原始序号，从 0 起
    stamp_ns: int       # header 整数纳秒；0 表示驱动没填（异常）
    bag_time_ns: int    # rosbag2 记录的接收时间，只用于分段
    width: int
    height: int
    encoding: str
    step: int
    frame_id: str


@dataclass
class BagScan:
    """一次扫描的结果：话题表、每路帧元信息、两侧内参。"""

    path: Path
    topics: dict
    frames: dict = field(default_factory=dict)      # role -> [FrameMeta]
    infos: dict = field(default_factory=dict)       # role -> CameraInfo 摘要
    geometry_changes: list = field(default_factory=list)
    topic_table: list = field(default_factory=list)  # 所有图像/内参话题，给 inspect 用


def _reader():
    """延迟 import rosbags：没装时给出能看懂的提示，而不是 ModuleNotFoundError 堆栈。"""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:                     # pragma: no cover - 环境问题
        raise BagError(
            "读 bag 需要 rosbags（pyproject 的 [lab] 可选依赖）。"
            "容器内：docker compose -f .devcontainer/docker-compose.yml run --rm "
            "up70-vision python -m lab.rgbd_check inspect <bag>") from exc
    return AnyReader


def decode_image(data, width: int, height: int, encoding: str, step: int) -> np.ndarray:
    """把一帧的原始字节解成 ndarray。

    :param data: 消息里的 data（bytes / 一维 uint8）
    :param width: 宽
    :param height: 高
    :param encoding: ROS 编码字符串
    :param step: 每行字节数（可能大于 width*bpp）
    :return: 深度 (H, W) uint16 毫米；彩色 (H, W, 3) uint8 BGR；mono8 (H, W) uint8
    :raises BagError: 编码不支持或长度不够（长度不对必须报错，不能 reshape 出一张错位图）
    """
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    need = step * height
    if buf.size < need:
        raise BagError(f"图像数据长度 {buf.size} 小于 step*height={need}"
                       f"（{width}x{height}, {encoding}, step={step}），不能解码")

    if encoding in _DEPTH16:
        row_bytes = width * 2
        if step == row_bytes:
            flat = buf[:need].view("<u2")
        else:
            rows = buf[:need].reshape(height, step)[:, :row_bytes]
            flat = np.ascontiguousarray(rows).view("<u2")
        return flat.reshape(height, width).copy()

    if encoding in _COLOR_ENCODINGS or encoding in _MONO8:
        channels = 1 if encoding in _MONO8 else 3
        row_bytes = width * channels
        if step == row_bytes:
            img = buf[:need].reshape(height, width, channels) if channels > 1 \
                else buf[:need].reshape(height, width)
        else:
            rows = buf[:need].reshape(height, step)[:, :row_bytes]
            img = rows.reshape(height, width, channels) if channels > 1 \
                else rows.reshape(height, width)
        img = np.ascontiguousarray(img)
        if _COLOR_ENCODINGS.get(encoding):
            img = img[:, :, ::-1].copy()               # RGB -> BGR，OpenCV 约定
        return img

    raise BagError(f"不支持的编码 {encoding!r}；本工具只处理 16UC1/mono16/bgr8/rgb8/mono8")


def _info_summary(msg) -> dict:
    """CameraInfo -> 可落盘的字典。K/D/R/P 原样保留，不做任何"修正"。"""
    return {
        "frame_id": str(msg.header.frame_id),
        "width": int(msg.width),
        "height": int(msg.height),
        "distortion_model": str(msg.distortion_model),
        "k": [float(v) for v in msg.k],
        "d": [float(v) for v in msg.d],
        "r": [float(v) for v in msg.r],
        "p": [float(v) for v in msg.p],
        "binning_x": int(msg.binning_x),
        "binning_y": int(msg.binning_y),
    }


def _topic_table(reader) -> list:
    """列出 bag 里所有图像/内参与 TF 话题，给 `inspect` 与人工选话题用。"""
    table = []
    for conn in reader.connections:
        if conn.msgtype in (IMAGE_TYPE, INFO_TYPE) or conn.topic in ("/tf", "/tf_static"):
            table.append({
                "topic": conn.topic,
                "type": conn.msgtype,
                "messages": int(conn.msgcount),
            })
    return sorted(table, key=lambda row: row["topic"])


def _resolve_topics(scan_topics: dict, table: list) -> dict:
    """话题缺失时给一条能照着改的报错，而不是让 KeyError 冒出去。"""
    available = [row for row in table if row["type"] in (IMAGE_TYPE, INFO_TYPE)]
    missing = [f"{role}={topic or '<未配置>'}" for role, topic in scan_topics.items()
               if not topic]
    if missing:
        raise BagError(f"这些角色没配话题：{missing}；bag 里实际有的：{available}")
    return scan_topics


def scan(bag_dir: Path, topics: dict) -> BagScan:
    """第一遍：话题核对 + 每帧元信息 + 两侧内参（不保留像素）。

    :param bag_dir: bag 目录（含 metadata.yaml）
    :param topics: 角色 -> 话题名，四个都要给全
    :return: `BagScan`
    :raises BagError: 目录不存在、话题缺失、类型不对、bag 打不开
    """
    bag_dir = Path(bag_dir)
    if not bag_dir.exists():
        raise BagError(f"bag 不存在：{bag_dir}")
    AnyReader = _reader()

    try:
        with AnyReader([bag_dir]) as reader:
            table = _topic_table(reader)
            topics = _resolve_topics(topics, table)
            by_topic = {conn.topic: conn for conn in reader.connections}
            for role, topic in topics.items():
                conn = by_topic.get(topic)
                if conn is None:
                    raise BagError(f"{role} 话题 {topic} 不在 bag 里；实际有：{table}")
                if conn.msgtype != ROLE_TYPES[role]:
                    raise BagError(f"{role} 话题 {topic} 类型是 {conn.msgtype}，"
                                   f"期望 {ROLE_TYPES[role]}")

            result = BagScan(path=bag_dir, topics=dict(topics), topic_table=table)
            for role, topic in topics.items():
                result.frames[role] = []

            for role, topic in topics.items():
                conn = by_topic[topic]
                index = 0
                for _, bag_time, raw in reader.messages(connections=[conn]):
                    msg = reader.deserialize(raw, conn.msgtype)
                    if role in ("color", "depth"):
                        result.frames[role].append(FrameMeta(
                            role=role,
                            index=index,
                            stamp_ns=(int(msg.header.stamp.sec) * 1_000_000_000
                                      + int(msg.header.stamp.nanosec)),
                            bag_time_ns=int(bag_time),
                            width=int(msg.width),
                            height=int(msg.height),
                            encoding=str(msg.encoding),
                            step=int(msg.step),
                            frame_id=str(msg.header.frame_id),
                        ))
                        first, current = result.frames[role][0], result.frames[role][-1]
                        if (first.width, first.height, first.encoding, first.frame_id) != \
                                (current.width, current.height, current.encoding, current.frame_id):
                            note = f"{role} 图像几何/编码在序号 {index} 变化"
                            if not result.geometry_changes:
                                result.geometry_changes.append(note)
                    elif role not in result.infos:
                        # 内参在整段里应当不变；只留第一帧，后面变化了会在 analyze 里比对。
                        result.infos[role] = _info_summary(msg)
                    elif result.infos[role] != _info_summary(msg):
                        note = f"{role} CameraInfo 在序号 {index} 变化"
                        if not any(item.startswith(f"{role} CameraInfo") for item in result.geometry_changes):
                            result.geometry_changes.append(note)
                    index += 1
            return result
    except BagError:
        raise
    except Exception as exc:                        # noqa: BLE001 - rosbags 的异常类型不稳定
        raise BagError(f"读 {bag_dir} 失败：{type(exc).__name__}: {exc}") from exc


def load_images(bag_dir: Path, topic: str, indices, role: str) -> dict:
    """第二遍：只解指定序号的像素。

    :param bag_dir: bag 目录
    :param topic: 话题名
    :param indices: 需要解码的序号集合（`FrameMeta.index`）
    :param role: 角色名，只用于报错信息
    :return: `{index: ndarray}`
    :raises BagError: 话题不在 bag 里或解码失败
    """
    AnyReader = _reader()
    wanted = set(int(i) for i in indices)
    out = {}
    if not wanted:
        return out

    with AnyReader([Path(bag_dir)]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise BagError(f"{role} 话题 {topic} 不在 bag 里")
        index = 0
        for conn, _, raw in reader.messages(connections=conns):
            if index in wanted:
                msg = reader.deserialize(raw, conn.msgtype)
                out[index] = decode_image(msg.data, int(msg.width), int(msg.height),
                                          str(msg.encoding), int(msg.step))
            index += 1
            if len(out) == len(wanted):
                break
    missing = sorted(wanted - set(out))
    if missing:
        raise BagError(f"{role} 话题 {topic} 少了序号 {missing}（bag 被截断？）")
    return out


def inspect_bag(bag_dir: Path) -> dict:
    """`inspect` 用：每个图像话题只看首尾两帧，够回答「是什么、多少帧、什么编码、多大」。

    不整段解码是有原因的：120 秒、30 fps 的彩色是 3600 帧 × 3 MB，全解一遍要几分钟，
    而 inspect 只想让人确认「话题选对了没有」。要看数据本身用 analyze。

    :param bag_dir: bag 目录
    :return: `{"images": [...], "infos": [...], "bag_topics": [...]}`
    :raises BagError: bag 打不开
    """
    bag_dir = Path(bag_dir)
    if not bag_dir.exists():
        raise BagError(f"bag 不存在：{bag_dir}")
    AnyReader = _reader()

    try:
        with AnyReader([bag_dir]) as reader:
            topics = _topic_table(reader)
            images, infos = [], []
            for conn in reader.connections:
                if conn.msgtype == INFO_TYPE:
                    for _, _, raw in reader.messages(connections=[conn]):
                        infos.append({"topic": conn.topic, **_info_summary(
                            reader.deserialize(raw, conn.msgtype))})
                        break
                    continue
                if conn.msgtype != IMAGE_TYPE:
                    continue
                first = last = None
                count = 0
                for _, bag_time, raw in reader.messages(connections=[conn]):
                    if count == 0:
                        first = (bag_time, reader.deserialize(raw, conn.msgtype))
                    last = (bag_time, raw)
                    count += 1
                entry = {"topic": conn.topic, "messages": count,
                         "type": conn.msgtype}
                if first is not None:
                    _, msg = first
                    entry.update({
                        "encoding": str(msg.encoding),
                        "width": int(msg.width),
                        "height": int(msg.height),
                        "step": int(msg.step),
                        "frame_id": str(msg.header.frame_id),
                        "first_stamp_ns": int(msg.header.stamp.sec) * 1_000_000_000
                        + int(msg.header.stamp.nanosec),
                    })
                if first is not None and last is not None and count > 1:
                    last_msg = reader.deserialize(last[1], conn.msgtype)
                    entry["last_stamp_ns"] = int(last_msg.header.stamp.sec) * 1_000_000_000 \
                        + int(last_msg.header.stamp.nanosec)
                    span = (last[0] - first[0]) / 1e9
                    entry["bag_span_s"] = round(span, 3)
                    entry["fps_estimate"] = round((count - 1) / span, 2) if span > 0 else None
                images.append(entry)
            return {"images": sorted(images, key=lambda i: i["topic"]),
                    "infos": sorted(infos, key=lambda i: i["topic"]),
                    "bag_topics": topics}
    except BagError:
        raise
    except Exception as exc:                        # noqa: BLE001 - rosbags 异常类型不稳定
        raise BagError(f"读 {bag_dir} 失败：{type(exc).__name__}: {exc}") from exc


def main() -> int:
    """直接 `python lab/rgbd_check/bag.py` 时给一句提示，避免误用。"""
    print("这是 lab.rgbd_check 的读包模块，不是入口；用 python -m lab.rgbd_check inspect BAG",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
