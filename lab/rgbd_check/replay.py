"""比较直接读取的原包与 ROS 回放后重录的包；不依赖 ROS 运行时。"""

import hashlib
import html
import json
from pathlib import Path

from .bag import BagError, FrameMeta, _reader, ROLE_TYPES
from .pairing import pair_frames


def fingerprint(bag: Path, topics: dict) -> tuple:
    """按话题保留 header、完整序列化内容散列和配对所需元信息。

    bag 接收时间重录后必然改变，因此不作为内容一致性的要求。
    """
    topics = {role: topics[role] for role in ROLE_TYPES}
    records, frames = {role: [] for role in topics}, {"color": [], "depth": []}
    with _reader()([bag]) as reader:
        for role, topic in topics.items():
            conns = [c for c in reader.connections if c.topic == topic]
            if not conns or any(c.msgtype != ROLE_TYPES[role] for c in conns):
                raise BagError(f"{bag} 缺少正确类型的 {role}: {topic}")
            for conn, time_ns, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                stamp_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
                records[role].append((stamp_ns, hashlib.sha256(raw).hexdigest()))
                if role in frames:
                    frames[role].append(FrameMeta(role, len(frames[role]), stamp_ns, int(time_ns),
                        int(msg.width), int(msg.height), str(msg.encoding), int(msg.step),
                        str(msg.header.frame_id)))
    pairs = pair_frames(frames['color'], frames['depth'], warmup_ns=0)
    signature = [(p.color.stamp_ns, p.depth.stamp_ns) for p in pairs.pairs]
    return records, signature


def compare_replay(original: Path, replay: Path, topics: dict, out: Path) -> dict:
    if out.exists():
        raise BagError(f"输出目录已存在，拒绝覆盖：{out}")
    try:
        before, original_pairs = fingerprint(original, topics)
        after, replay_pairs = fingerprint(replay, topics)
    except Exception as exc:
        raise BagError(f"回放一致性读取失败：{exc}") from exc
    streams = {}
    for role in topics:
        a, b = before[role], after[role]
        mismatch = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        streams[role] = {"original_count": len(a), "replay_count": len(b),
                         "status": "PASS" if a and a == b else "FAIL",
                         "first_mismatch_indices": mismatch[:5],
                         "original_sequence_sha256": hashlib.sha256(json.dumps(a).encode()).hexdigest(),
                         "replay_sequence_sha256": hashlib.sha256(json.dumps(b).encode()).hexdigest()}
    equal_pairs = original_pairs == replay_pairs and bool(original_pairs)
    result = {"status": "PASS" if equal_pairs and all(s['status'] == 'PASS' for s in streams.values())
              else "FAIL", "original": str(original), "replay": str(replay), "streams": streams,
              "pairing": {"status": "PASS" if equal_pairs else "FAIL",
                          "original_pairs": len(original_pairs), "replay_pairs": len(replay_pairs),
                          "warmup_s": 0, "tolerance_ms": 10},
              "note": "比较整包四路的 header 与完整 CDR 内容散列及 header 配对；接收时间允许变化。"
                      "仅证明回放一致性，不证明原包采集完整或空间精度合格。"}
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>ROS 回放一致性</title>'
        '<h1>ROS 回放一致性：'+result['status']+'</h1><pre>'+html.escape(
        json.dumps(result, ensure_ascii=False, indent=2))+'</pre>', encoding='utf-8')
    return result
