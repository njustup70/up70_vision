"""rgbd_capture：按一段场景配置录一份同步 RGB-D bag，并把证据一起落盘。

用大白话说
    相机由 rgbd_capture.launch.py 起好以后，这个节点负责到底：
    看清四路话题在不在、数据是不是真的在流、把当时的参数与 QoS 存下来、
    调 `ros2 bag record` 录「预热 + 稳定段」，到点发 SIGINT 让 bag 正常收尾，
    最后写 meta.yaml。这样一段 bag 自己就能说明「录的时候是什么配置」。

为什么不用 `ros2 bag record` 手工敲
    手工录最容易漏三件事，而这三件事决定了数据能不能用：
      1. 话题名：对齐后深度可能换话题，甚至复用原话题；录错一个就白录。
      2. QoS：相机是 best_effort；用默认 reliable 录会出现「话题有频率但数据不完整」。
      3. 参数快照：过两周不会记得当时同步开没开、时间域是 global 还是 device。
    所以录包这件事固定成一个入口，参数和证据一起产出，而不是靠记性。

预热为什么算在录制里
    驱动启动后前几秒会丢帧、曝光也在收敛。计划书要求「预热 3 秒 + 稳定段」，
    两段都录进 bag，由离线工具按 bag 时间排除前 3 秒——这样「排除了多少」是
    可复查的，而不是靠人掐表。

已知的坑
    · `ros2 bag record -o DIR` 要求 DIR 不存在。所以快照先写进同级临时目录，
      录完再搬进 bag 目录；中途失败也照样搬，失败证据不能丢。
    · SIGINT 之后要等 bag 自己写完 metadata.yaml，立刻 kill 会得到一个打不开的 bag。
    · 对齐深度可能复用 /camera/depth/image_raw，也可能另发话题。自动发现只在
      「每种角色唯一候选」时才敢填，多个候选一律报错等人指定，猜错方向比报错更贵。
    · 停止输入后 `ros2 bag record` 不会自己退出，所以正常收尾全靠这里发信号；
      节点被 Ctrl-C 时也要先收尾再退出。

退出码：0 = 录制完成且 meta 完整；1 = 录制跑完但 bag 不完整/中途异常；
        2 = 参数、输入或环境错误（没开始录或提前中止）。
"""

import json
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

#: 四种角色的期望消息类型。话题名可配，类型不可配——类型不对说明配错了话题。
EXPECTED_TYPES = {
    "color": "sensor_msgs/msg/Image",
    "depth": "sensor_msgs/msg/Image",
    "color_info": "sensor_msgs/msg/CameraInfo",
    "depth_info": "sensor_msgs/msg/CameraInfo",
}

#: 每次采集固定记录的话题（存在才记）。TF 一起录，后面查外参来源时用得上。
OPTIONAL_TOPICS = ("/tf", "/tf_static")

#: `ros2 topic info -v` 里 QoS 字段的两种写法都要认：
#: 老版本是 `History: KEEP_LAST` + `Depth: 5`，新版本合成 `History (Depth): KEEP_LAST (5)`。
_QOS_FIELD = re.compile(r"^\s*(Reliability|Durability|History(?: \(Depth\))?|Depth):\s*(\S+)")


def _now_iso() -> str:
    """本地时区的 ISO 时间戳，给人看；统计一律用整数纳秒，不用这个。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_topic_info(text: str) -> dict:
    """从 `ros2 topic info -v` 的文本里抠出 QoS 与类型。

    :param text: 命令原始输出
    :return: `{"type": str, "qos": {reliability, durability, history, depth}}`，
             解析不到的字段不出现（宁可缺，也不猜）
    """
    info = {"type": "", "qos": {}}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Topic type:"):
            info["type"] = stripped.split(":", 1)[1].strip()
            continue
        match = _QOS_FIELD.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip().lower()
        if key.startswith("History"):
            # 形如 `History (Depth): KEEP_LAST (5)`，括号里那个 5 也要收下；
            # 老版本是下一行 `Depth: 5`，两种都认。
            info["qos"]["history"] = value
            depth_match = re.search(r"\((\d+)\)", line)
            if depth_match:
                info["qos"]["depth"] = int(depth_match.group(1))
        elif key == "Depth":
            info["qos"]["depth"] = int(value)
        elif key == "Reliability":
            info["qos"]["reliability"] = value
        elif key == "Durability":
            info["qos"]["durability"] = value
    return info


def qos_overrides(per_topic: dict) -> dict:
    """把解析出来的 QoS 转成 `ros2 bag record --qos-profile-overrides-path` 的格式。

    只写能确定的键：缺了就用 rosbag2 的默认值，总比写一个错的好。

    :param per_topic: `{topic: {"qos": {...}}}`
    :return: 可直接 `yaml.safe_dump` 的字典
    """
    overrides = {}
    for topic, info in per_topic.items():
        qos = info.get("qos") or {}
        entry = {}
        if "reliability" in qos:
            entry["reliability"] = qos["reliability"]
        if "durability" in qos:
            entry["durability"] = qos["durability"]
        if "history" in qos:
            entry["history"] = qos["history"]
        if "depth" in qos:
            entry["depth"] = int(qos["depth"])
        if entry:
            overrides[topic] = entry
    return overrides


def _run(cmd: list, timeout: float = 20.0) -> tuple:
    """跑一条只读的 ROS 命令行，返回 `(成功, 输出+错误)`。失败不抛异常，交给调用方记录。"""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    text = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, text.strip()


def driver_version(pkg: str = "orbbec_camera") -> str | None:
    """读驱动包的版本号；读不到返回 None（meta 里写 null，不写猜测值）。"""
    ok, text = _run(["ros2", "pkg", "xml", pkg])
    if not ok:
        return None
    try:
        return ElementTree.fromstring(text).findtext("version")
    except ElementTree.ParseError:
        return None


class RgbdCapture(Node):
    """一次采集 = 一次运行。发起、等待、录、收尾、写 meta，都在 `run()` 里。"""

    def __init__(self):
        super().__init__("rgbd_capture")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("duration_s", 0.0)
        self.declare_parameter("warmup_s", 3.0)
        self.declare_parameter("session_config", "")
        self.declare_parameter("camera_node", "camera")
        self.declare_parameter("ready_timeout_s", 20.0)
        self.declare_parameter("bag_storage", "mcap")
        # 四路话题留空表示自动发现；只有唯一候选时才自动填。
        for role in EXPECTED_TYPES:
            self.declare_parameter(f"topic_{role}", "")

        self._probe = {}           # role -> {"stamp", "width", "height", "encoding", "frame_id", "step"}
        self._info = {}            # role -> CameraInfo 字典
        self._subs = []            # 探测用订阅，探测完就销毁，别一直吃带宽
        self._exited = False       # 收到 Ctrl-C 的标记

    # ---------- 参数与配置 ----------

    def _params(self) -> dict:
        """把 ROS 参数取成普通字典，顺手做类型归一。"""
        return {
            "output_dir": str(self.get_parameter("output_dir").value),
            "duration_s": float(self.get_parameter("duration_s").value),
            "warmup_s": float(self.get_parameter("warmup_s").value),
            "session_config": str(self.get_parameter("session_config").value),
            "camera_node": str(self.get_parameter("camera_node").value),
            "ready_timeout_s": float(self.get_parameter("ready_timeout_s").value),
            "bag_storage": str(self.get_parameter("bag_storage").value),
            "topics": {role: str(self.get_parameter(f"topic_{role}").value).strip()
                       for role in EXPECTED_TYPES},
        }

    def _load_session(self, path: str) -> dict:
        """读 session.yaml。配置写坏了就别往下跑——照着坏配置录出来的 bag 没用。"""
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"{path} 的内容不是 YAML 映射")
        return cfg

    # ---------- 话题发现与就绪 ----------

    def _graph(self) -> dict:
        """当前图里的 `topic -> [types]`，顺便过滤掉隐藏话题。"""
        return {topic: types for topic, types in self.get_topic_names_and_types()
                if not topic.startswith("/_")}

    def _discover(self, topics: dict) -> dict:
        """给没配的四路话题自动找候选。

        规则故意保守：只有「角色唯一候选」才敢填。对齐后深度可能出现两个候选
        （原始 + 对齐），这时候报错让人指定，不猜——猜错方向，后面所有结论都反了。

        :param topics: 参数里给的话题映射，空串表示没配
        :return: 补齐后的映射
        :raises ValueError: 某角色没有候选或有多个候选
        """
        graph = self._graph()
        images = sorted(t for t, types in graph.items() if EXPECTED_TYPES["color"] in types)
        infos = sorted(t for t, types in graph.items() if EXPECTED_TYPES["color_info"] in types)

        def pick(candidates: list, role: str) -> str:
            if not candidates:
                raise ValueError(f"{role} 没有可用话题；当前图像话题：{images}；"
                                 f"内参话题：{infos}")
            if len(candidates) > 1:
                raise ValueError(f"{role} 有多个候选，不敢猜：{candidates}。"
                                 "请在 session.yaml 的 capture.topics 里显式指定")
            return candidates[0]

        resolved = dict(topics)
        if not resolved["color"]:
            resolved["color"] = pick([t for t in images if "/color/" in t or "color" in t
                                      and "depth" not in t], "color")
        if not resolved["depth"]:
            resolved["depth"] = pick([t for t in images if t != resolved["color"]], "depth")
        if not resolved["color_info"]:
            resolved["color_info"] = pick([t for t in infos if t != resolved["depth_info"]
                                           and "color" in t], "color_info")
        if not resolved["depth_info"]:
            resolved["depth_info"] = pick([t for t in infos if t != resolved["color_info"]],
                                          "depth_info")
        return resolved

    def _check_topics(self, topics: dict) -> list:
        """核对四路话题的类型与发布者数量，返回问题列表（空 = 全过）。"""
        problems = []
        graph = self._graph()
        for role, topic in topics.items():
            want = EXPECTED_TYPES[role]
            types = graph.get(topic)
            if not types:
                problems.append(f"{role}: 话题 {topic} 不在图里")
                continue
            if want not in types:
                problems.append(f"{role}: {topic} 类型是 {types}，期望 {want}")
                continue
            count = self.count_publishers(topic)
            if count < 1:
                problems.append(f"{role}: {topic} 没有发布者（有话题、不吐数据，先看驱动日志）")
        return problems

    def _attach_probes(self, topics: dict) -> None:
        """给四路话题挂一次性订阅：拿到第一帧就记几何信息，同时证明数据真的在流。

        用 best_effort 订阅：它对 reliable 与 best_effort 两种发布端都兼容
        （反过来不行）。探测完立刻销毁，避免 120 秒录制时多一份解包开销。
        """

        def on_color(msg: Image) -> None:
            self._probe["color"] = self._geometry(msg)

        def on_depth(msg: Image) -> None:
            self._probe["depth"] = self._geometry(msg)

        def on_info(role: str):
            def callback(msg: CameraInfo) -> None:
                k = list(msg.k)
                self._info[role] = {
                    "frame_id": msg.header.frame_id,
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "distortion_model": msg.distortion_model,
                    "k": k,
                    "d": list(msg.d),
                    "r": list(msg.r),
                    "p": list(msg.p),
                    "binning_x": int(msg.binning_x),
                    "binning_y": int(msg.binning_y),
                }
            return callback

        self._subs = [
            self.create_subscription(Image, topics["color"], on_color, qos_profile_sensor_data),
            self.create_subscription(Image, topics["depth"], on_depth, qos_profile_sensor_data),
            self.create_subscription(CameraInfo, topics["color_info"], on_info("color"),
                                     qos_profile_sensor_data),
            self.create_subscription(CameraInfo, topics["depth_info"], on_info("depth"),
                                     qos_profile_sensor_data),
        ]

    def _geometry(self, msg: Image) -> dict:
        """一帧图像的几何信息；时间戳保存成整数纳秒，不用浮点秒。"""
        return {
            "stamp_ns": int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec),
            "width": int(msg.width),
            "height": int(msg.height),
            "encoding": str(msg.encoding),
            "step": int(msg.step),
            "frame_id": msg.header.frame_id,
        }

    def _burst(self, deadline_s: float) -> bool:
        """在限定时间内 spin，直到四路都拿到至少一帧。"""
        end = time.monotonic() + deadline_s
        while time.monotonic() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if len(self._probe) >= 2 and len(self._info) >= 2:
                return True
        return False

    # ---------- 快照 ----------

    def _snapshot(self, stage: Path, args: dict, topics: dict, session: dict) -> dict:
        """把「录之前的世界」存成证据：参数、QoS、内参与档位。

        :return: meta 里 `capture` 段要用的字段
        """
        stage.mkdir(parents=True, exist_ok=False)
        info_dir = stage / "topic_info"
        info_dir.mkdir()

        per_topic = {}
        recorded = list(topics.values())
        for topic in recorded + [t for t in OPTIONAL_TOPICS if self.count_publishers(t) > 0]:
            ok, text = _run(["ros2", "topic", "info", "-v", topic])
            (info_dir / (topic.strip("/").replace("/", "_") + ".txt")).write_text(
                text + "\n", encoding="utf-8")
            if ok:
                per_topic[topic] = parse_topic_info(text)

        ok, text = _run(["ros2", "param", "dump", f"/{self.get_parameter('camera_node').value}"])
        (stage / "camera_params.yaml").write_text(text + "\n", encoding="utf-8")
        if not ok:
            self.get_logger().warning("ros2 param dump 失败，参数快照以日志与命令行记录为准")

        (stage / "camera_info_color.yaml").write_text(
            yaml.safe_dump(self._info.get("color"), allow_unicode=True), encoding="utf-8")
        (stage / "camera_info_depth.yaml").write_text(
            yaml.safe_dump(self._info.get("depth"), allow_unicode=True), encoding="utf-8")
        (stage / "stream_probe.json").write_text(
            json.dumps({**self._probe, "topics": topics, "args": args}, ensure_ascii=False,
                       indent=2), encoding="utf-8")
        (stage / "session.yaml").write_text(
            yaml.safe_dump(session, allow_unicode=True, sort_keys=False), encoding="utf-8")

        # 发布端 QoS 决定录包 QoS；写不出覆盖文件就退回 rosbag2 默认值，但要留痕。
        overrides = qos_overrides(per_topic)
        qos_path = stage / "bag_qos_overrides.yaml"
        qos_path.write_text(yaml.safe_dump(overrides, allow_unicode=True), encoding="utf-8")
        if len(overrides) != len(per_topic):
            self.get_logger().warning("部分话题的 QoS 没解析出来，这些话题用 rosbag2 默认值录制")

        return {
            "snapshots": "snapshots",
            "geometry_evidence": "snapshots/stream_probe.json",
            "qos_overrides": overrides,
            "recorded_topics": recorded + [t for t in OPTIONAL_TOPICS
                                           if self.count_publishers(t) > 0],
        }

    # ---------- 收尾与 meta ----------

    def _bag_summary(self, bag_dir: Path, duration_s: float) -> dict:
        """读 rosbag2 自己写的 metadata.yaml，拿每个话题的消息数算实际帧率。

        比额外订阅一路来数帧划算：录制期间不再多一份解包开销，
        而且数字来自 bag 本身，和离线工具看到的是同一份。
        """
        meta_path = bag_dir / "metadata.yaml"
        summary = {"bag_metadata": str(meta_path.relative_to(bag_dir.parent)),
                   "message_count": {}, "actual_fps": {}, "bag_duration_s": None}
        if not meta_path.is_file():
            return summary
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            summary["metadata_error"] = str(exc)
            return summary

        info = meta.get("rosbag2_bagfile_information") or {}
        summary["bag_duration_s"] = (info.get("duration") or {}).get("nanoseconds", 0) / 1e9
        for entry in info.get("topics_with_message_count") or []:
            meta_topic = (entry.get("topic_metadata") or {})
            name = meta_topic.get("name")
            count = int(entry.get("message_count") or 0)
            if not name:
                continue
            summary["message_count"][name] = count
            # 实际频率按稳定段算：预热那段驱动在丢帧，算进去会低估。
            span = summary["bag_duration_s"] or duration_s
            stable = max(span - self._warmup_s, 1e-6)
            summary["actual_fps"][name] = round(count / stable, 2)
        return summary

    def _write_meta(self, output_dir: Path, session: dict, args: dict, topics: dict,
                    extra: dict) -> None:
        """写 meta.yaml：采集器只补实际信息，不改写用户实测的 scene 字段。"""
        meta = {
            "schema_version": 1,
            "session": session.get("session"),
            "scene": session.get("scene"),
            "capture": {
                "mode": (session.get("capture") or {}).get("mode"),
                "camera_node": args["camera_node"],
                "warmup_s": args["warmup_s"],
                "duration_s": args["duration_s"],
                "topics": topics,
                "depth_unit": "mm",
                "time_domain": extra.pop("time_domain", None),
                "requested_profile": extra.pop("requested_profile", None),
                "actual_profile": extra.pop("actual_profile", None),
                **extra,
            },
            "environment": {
                "ros_distro": os.environ.get("ROS_DISTRO"),
                "driver": "orbbec_camera",
                "driver_version": driver_version(),
                "host": os.uname().nodename,
            },
        }
        (output_dir / "meta.yaml").write_text(
            yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # ---------- 主流程 ----------

    def run(self) -> int:
        """一次采集的完整流程；返回值就是进程退出码。"""
        args = self._params()
        out = Path(args["output_dir"]).expanduser()
        if not args["output_dir"]:
            self.get_logger().error("必须给 output_dir")
            return 2
        if not out.is_absolute():
            self.get_logger().error(f"output_dir 要用绝对路径：{out}")
            return 2
        if out.exists():
            self.get_logger().error(f"{out} 已存在；本工具不覆盖、不复用已有 bag，换个编号")
            return 2
        if args["duration_s"] <= 0:
            self.get_logger().error("duration_s 必须大于 0（这是稳定段时长，不含预热）")
            return 2
        if args["warmup_s"] < 0:
            self.get_logger().error("warmup_s 不能是负数")
            return 2
        if not Path(args["session_config"]).is_file():
            self.get_logger().error(f"session_config 不存在：{args['session_config']}")
            return 2

        try:
            session = self._load_session(args["session_config"])
        except (OSError, ValueError, yaml.YAMLError) as exc:
            self.get_logger().error(f"读 session_config 失败：{exc}")
            return 2

        self._warmup_s = args["warmup_s"]
        topics = dict(args["topics"])
        # 参数没给就取 session.yaml 里的；两边都没有再自动发现。
        cfg_topics = ((session.get("capture") or {}).get("topics") or {})
        for role in EXPECTED_TYPES:
            if not topics[role]:
                topics[role] = str(cfg_topics.get(role) or "").strip()

        stage = out.parent / f".{out.name}_staging"
        if stage.exists():
            self.get_logger().error(f"临时目录已存在（上次没收尾？）：{stage}")
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            if not all(topics.values()):
                topics = self._discover(topics)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return 2

        self.get_logger().info(f"等待四路话题：{topics}")
        self._attach_probes(topics)
        deadline = args["ready_timeout_s"]
        if not self._burst(deadline):
            problems = self._check_topics(topics)
            self.get_logger().error(
                f"{deadline:.0f} 秒内没凑齐图像与内参；问题：{problems or '收到话题但没数据'}")
            self._destroy_probes()
            self._write_failure(session, args, topics, stage, out, problems)
            return 2

        problems = self._check_topics(topics)
        if problems:
            self.get_logger().error(f"话题核对失败：{problems}")
            self._destroy_probes()
            self._write_failure(session, args, topics, stage, out, problems)
            return 2

        try:
            snapshot = self._snapshot(stage, args, topics, session)
        except OSError as exc:
            self.get_logger().error(f"写快照失败：{exc}")
            self._destroy_probes()
            return 2
        self._destroy_probes()

        total_s = args["warmup_s"] + args["duration_s"]
        cmd = ["ros2", "bag", "record", "-o", str(out), "--storage", args["bag_storage"]]
        qos_path = stage / "bag_qos_overrides.yaml"
        if qos_path.is_file() and snapshot["qos_overrides"]:
            cmd += ["--qos-profile-overrides-path", str(qos_path)]
        cmd += snapshot["recorded_topics"]

        self.get_logger().info(f"开始录制 {total_s:.1f} 秒（预热 {args['warmup_s']:.1f} + "
                               f"稳定段 {args['duration_s']:.1f}）：{' '.join(cmd)}")
        started = _now_iso()
        try:
            proc = subprocess.Popen(cmd)
        except OSError as exc:
            self.get_logger().error(f"起不了 ros2 bag record：{exc}")
            self._write_failure(session, args, topics, stage, out, [str(exc)])
            return 2

        interrupted = not self._wait(total_s, proc)
        end = _now_iso()
        ok = self._stop_recorder(proc)

        # 快照搬进 bag 目录：bag 目录是 rosbag2 建的，只能在录完之后搬。
        self._move_snapshots(stage, out)

        summary = self._bag_summary(out, args["duration_s"])
        requested = (session.get("capture") or {}).get("requested_profile")
        actual = {}
        for role in ("color", "depth"):
            probe = self._probe.get(role) or {}
            actual[role] = {
                "width": probe.get("width"),
                "height": probe.get("height"),
                "encoding": probe.get("encoding"),
                "frame_id": probe.get("frame_id"),
                "fps": summary["actual_fps"].get(topics[role]),
                "message_count": summary["message_count"].get(topics[role]),
            }
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        extra = {
            "complete": bool(ok and not interrupted and (out / "metadata.yaml").is_file()),
            "reason": None if (ok and not interrupted) else
                      ("节点被中断，录制提前结束" if interrupted else "ros2 bag record 非正常退出"),
            "started_at": started,
            "finished_at": end,
            "time_domain": (session.get("capture") or {}).get("time_domain"),
            "requested_profile": requested,
            "actual_profile": actual,
            "message_count": summary["message_count"],
            "actual_fps": summary["actual_fps"],
            "bag_duration_s": summary["bag_duration_s"],
            "bag_size_bytes": size,
            "snapshots": snapshot["snapshots"],
            "geometry_evidence": snapshot["geometry_evidence"],
        }
        if (out / "metadata.yaml").is_file():
            self._write_meta(out, session, args, topics, extra)
        else:
            self.get_logger().error("bag 没有生成 metadata.yaml，说明录制没正常收尾")
            return 1

        if not extra["complete"]:
            self.get_logger().error(f"录制不完整：{extra['reason']}；bag 与快照保留在 {out}")
            return 1
        self.get_logger().info(
            f"录制完成：{out}（{size / 1e6:.1f} MB，稳定段 {args['duration_s']:.1f} s）")
        return 0

    def _wait(self, total_s: float, proc: subprocess.Popen) -> bool:
        """等录制到点；节点被 Ctrl-C 时提前返回 False。"""
        end = time.monotonic() + total_s
        while time.monotonic() < end:
            if proc.poll() is not None:
                self.get_logger().error("ros2 bag record 自己退出了，先看它的输出")
                return False
            if not rclpy.ok():
                self.get_logger().warning("收到中断，先让 bag 正常收尾")
                return False
            rclpy.spin_once(self, timeout_sec=0.2)
        return True

    def _stop_recorder(self, proc: subprocess.Popen, timeout_s: float = 30.0) -> bool:
        """发 SIGINT 让 rosbag2 写完 metadata；超时才升级到 SIGTERM/KILL。"""
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.get_logger().error(f"SIGINT 后 {timeout_s:.0f} 秒还没退出，强制结束；"
                                    "这份 bag 可能不完整")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            return False
        return proc.returncode == 0

    def _destroy_probes(self) -> None:
        """销毁探测订阅，避免录制期间多一份解包开销。"""
        for sub in self._subs:
            self.destroy_subscription(sub)
        self._subs = []

    def _move_snapshots(self, stage: Path, out: Path) -> None:
        """把快照搬进 bag 目录；失败也要尽量留证。"""
        if not stage.exists():
            return
        try:
            shutil.move(str(stage), str(out / "snapshots"))
        except OSError as exc:
            self.get_logger().error(f"快照搬移失败，仍在 {stage}：{exc}")

    def _write_failure(self, session: dict, args: dict, topics: dict, stage: Path,
                       out: Path, problems: list) -> None:
        """没开始录就失败时，把原因写进 out 目录的 meta，避免只剩一条终端日志。"""
        try:
            out.mkdir(parents=True, exist_ok=True)
            extra = {
                "complete": False,
                "reason": "未开始录制：" + ("；".join(problems) if problems else "未知原因"),
                "topics": topics,
            }
            self._write_meta(out, session, args, topics, extra)
            if stage.exists():
                shutil.move(str(stage), str(out / "snapshots"))
        except OSError as exc:
            self.get_logger().error(f"写失败记录也失败了：{exc}")


def main(argv=None) -> int:
    """节点入口：跑完一次就退出，退出码 0/1/2 见模块头。"""
    rclpy.init(args=argv)
    node = RgbdCapture()
    try:
        return node.run()
    except KeyboardInterrupt:
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
