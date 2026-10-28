"""落盘：pairs/unmatched/summary/spatial_edges、叠加图、annotations 与 report.html。

用大白话说
    这里只干「把结果写成文件和图」这一件事。CSV 给机器读（也方便 git diff），
    report.html 给人看（结论、配置、统计、失败样例），叠加图给人核。

为什么不用 matplotlib
    容器镜像 `up70-vision:dev` 里没有 matplotlib（实测），而报告不该因为少一个
    绘图库就出不来。叠加图全部用 cv2 画，HTML 手写——代价是几行字符串拼接，
    换来的是「在哪台机器上都能出报告」。

已知的坑
    · 图片路径一律写相对 OUT 的相对路径：整个 output/rgbd/<session> 拷走以后
      report.html 还能打开。
    · 时间戳写整数纳秒，缺值写空。写 0 会被误读成「测到了 0」，那是两回事。
    · 空间结论只允许两种来源：人工标注算出来的，或者明确写「没验证」。
      没有标注就绝不允许 report 里出现一个「看起来还行」的偏移数字。
"""

import csv
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from .measure import edge_offset_px, edge_diagnostics, nonparallel_segments
from .select import EDGES_PER_PAIR, OFFSET_LIMIT_PX, REQUIRED_CONDITIONS, pair_id
from .status import FAIL, PASS, SKIP, UNVERIFIED, worst

PAIR_COLUMNS = ("pair_id", "color_index", "color_stamp_ns", "depth_index", "depth_stamp_ns",
                "dt_ms", "status", "color_image", "depth_image", "overlay_image")
UNMATCHED_COLUMNS = ("role", "index", "stamp_ns", "kind", "reason")
SPATIAL_COLUMNS = ("session", "condition_id", "pair_id", "edge_id", "offset_px", "abs_offset_px",
                   "limit_px", "manual_review", "angle_deg", "along_offsets_px",
                   "measurement_status", "status", "note")


def _write_csv(path: Path, columns, rows) -> None:
    """固定列顺序写 CSV；空集合也写表头，下游按列名读不会因为文件不存在而炸。"""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key))
                             for key in columns})


def write_pairs(out: Path, result) -> list:
    """pairs.csv：两侧原始序号、整数时间戳、有符号时间差、配对编号与图路径。"""
    rows = []
    for pair in result.pairs:
        pid = pair_id(pair.color.index, pair.depth.index)
        rows.append({
            "pair_id": pid,
            "color_index": pair.color.index,
            "color_stamp_ns": pair.color.stamp_ns,
            "depth_index": pair.depth.index,
            "depth_stamp_ns": pair.depth.stamp_ns,
            "dt_ms": round(pair.dt_ns / 1e6, 3),
            "status": "matched",
            "color_image": f"overlays/{pid}_color.png",
            "depth_image": f"overlays/{pid}_depth_falsecolor.png",
            "overlay_image": f"overlays/{pid}_overlay.png",
        })
    _write_csv(out / "pairs.csv", PAIR_COLUMNS, rows)
    return rows


def write_unmatched(out: Path, result) -> None:
    """unmatched.csv：未配对帧与异常帧，分母完整保留。"""
    rows = [{"role": item["role"], "index": item["index"], "stamp_ns": item["stamp_ns"],
             "kind": item.get("kind", "unmatched"), "reason": item["reason"]}
            for item in result.unmatched] + \
           [{"role": item["role"], "index": item["index"], "stamp_ns": item["stamp_ns"],
             "kind": item["kind"], "reason": item["reason"]}
            for item in result.anomalies]
    _write_csv(out / "unmatched.csv", UNMATCHED_COLUMNS, rows)


def false_color(depth: np.ndarray) -> np.ndarray:
    """16 位深度 -> 伪彩 BGR。无效像素（0）涂黑，避免被当成「很近」的一片。"""
    valid = depth > 0
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    near, far = np.percentile(depth[valid], 1), np.percentile(depth[valid], 99)
    span = max(far - near, 1.0)
    scaled = np.clip((depth.astype(np.float32) - near) / span * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[~valid] = (0, 0, 0)
    return colored


def _fit_height(img: np.ndarray, height: int) -> np.ndarray:
    """按高度等比缩放；报告里左右并排要对齐，不能直接硬拉成同一尺寸（会改形状）。"""
    h, w = img.shape[:2]
    if h == height:
        return img
    scale = height / h
    return cv2.resize(img, (max(1, int(round(w * scale))), height), interpolation=cv2.INTER_AREA)


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    """在图顶部留一条黑边写说明：条件、配对编号、尺寸、时间差都在这里。"""
    bar = np.zeros((28, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([bar, img])


def write_overlays(out: Path, pid: str, color: np.ndarray, depth: np.ndarray, label: str) -> None:
    """一组图：原始彩色、深度伪彩、有效深度掩码、左右并排的对照图。"""
    folder = out / "overlays"
    folder.mkdir(exist_ok=True)

    colored = false_color(depth)
    mask = np.where(depth > 0, 255, 0).astype(np.uint8)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    height = 480
    left = _caption(_fit_height(color, height), f"color {color.shape[1]}x{color.shape[0]}")
    right = _caption(_fit_height(colored, height),
                     f"depth {depth.shape[1]}x{depth.shape[0]}")
    overlay = np.hstack([left, right])
    cv2.putText(overlay, label, (6, overlay.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(folder / f"{pid}_color.png"), color)
    cv2.imwrite(str(folder / f"{pid}_depth_falsecolor.png"), colored)
    cv2.imwrite(str(folder / f"{pid}_valid_mask.png"), mask_bgr)
    cv2.imwrite(str(folder / f"{pid}_overlay.png"), overlay)


def save_selected(out: Path, selection, color: np.ndarray, depth: np.ndarray,
                  condition_id: str, notes: str = "", session: str = "") -> dict:
    """把选中的那一对存成 npz：`annotate` 用它点选，不需要再打开 bag。"""
    folder = out / "selected"
    folder.mkdir(exist_ok=True)
    pid = pair_id(selection.pair.color.index, selection.pair.depth.index)
    np.savez_compressed(folder / f"{pid}.npz", color=color, depth=depth)
    entry = {
        "session": session,
        "pair_id": pid,
        "condition_id": condition_id,
        "image_size": [int(color.shape[1]), int(color.shape[0])],
        "depth_size": [int(depth.shape[1]), int(depth.shape[0])],
        "notes": notes,
        **selection.to_dict(),
    }
    (folder / f"{pid}.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry


def load_selected(out: Path) -> list:
    """读出 analyze 选好的图对；没有就返回空列表（annotate 据此报错）。"""
    folder = out / "selected"
    if not folder.is_dir():
        return []
    entries = []
    summary_path = out / "summary.json"
    session = json.loads(summary_path.read_text(encoding="utf-8")).get("session", "") \
        if summary_path.is_file() else ""
    for path in sorted(folder.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        entry.setdefault("session", session)
        entry["_npz"] = str(path.with_suffix(".npz"))
        entries.append(entry)
    return entries


def load_annotations(out: Path) -> dict:
    """读 annotations.json；没有时返回空骨架（不伪造 frames）。"""
    path = out / "annotations.json"
    if not path.is_file():
        return {"schema_version": 1, "frames": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    summary_path = out / "summary.json"
    session = json.loads(summary_path.read_text(encoding="utf-8")).get("session", "") \
        if summary_path.is_file() else ""
    for frame in data.get("frames", []):
        frame.setdefault("session", session)
    return data


def write_annotations(out: Path, data: dict) -> None:
    """标注落盘。写完立刻回读一次，防止写出半截 JSON 还继续算下去。"""
    data = {"schema_version": 1, **data}
    path = out / "annotations.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    json.loads(path.read_text(encoding="utf-8"))


def _check_annotation(frame: dict, selected: dict | None) -> str | None:
    """核对标注与选中图能否对上；对不上返回原因，不"尽力匹配"。"""
    if selected is None:
        return f"配对编号 {frame.get('pair_id')} 不在本次选中的图对里"
    if frame.get("session", "") != selected.get("session", "") or \
            frame.get("condition_id") != selected.get("condition_id"):
        return "标注的 session 或场景与选中图对不一致"
    if list(frame.get("image_size") or []) != list(selected.get("image_size") or []):
        return (f"标注尺寸 {frame.get('image_size')} 与选中图 {selected.get('image_size')} 不一致")
    if not frame.get("edges"):
        return "没有边界标注"
    return None


def spatial_rows(out: Path, mode: str | None) -> list:
    """把标注算成六行（三场景 × 两条边），没采/没标的场景也要留行。

    :param out: 报告目录
    :param mode: 采集模式；raw/sync 下空间检查记 SKIP（这两档本来就没开 D2C）
    :return: `spatial_edges.csv` 的行
    """
    selected = load_selected(out)
    annotations = {(f.get("session", ""), f.get("pair_id")): f
                   for f in load_annotations(out)["frames"]}
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    geometry = summary.get("geometry") or {}
    rows = []
    for condition in REQUIRED_CONDITIONS:
        matched = [entry for entry in selected if entry.get("condition_id") == condition]
        entry = matched[0] if len(matched) == 1 else {}
        frame = annotations.get((entry.get("session", ""), entry.get("pair_id")), {})
        problem = "本条件没有唯一选中图对（未采、未配对或多候选）" if not entry else \
            _check_annotation(frame, entry)
        edges = frame.get("edges") or []
        structure_problem = ""
        if len(edges) > EDGES_PER_PAIR or len({e.get("edge_id") for e in edges}) != len(edges):
            structure_problem = "边界数量或编号不符合每图两条边要求"
        if len(edges) == EDGES_PER_PAIR and all(e.get("confirmed") for e in edges):
            if any(not nonparallel_segments(edges[0].get(key), edges[1].get(key))
                   for key in ("color_segment", "depth_segment")):
                structure_problem = "两条边缺失、退化或平行，不能验证二维错位"
        for index in range(EDGES_PER_PAIR):
            edge = edges[index] if index < len(edges) else {}
            row = {"session": entry.get("session") or summary.get("session", ""),
                   "condition_id": condition, "pair_id": entry.get("pair_id", ""),
                   "edge_id": edge.get("edge_id") or f"edge{index + 1}",
                   "offset_px": None, "abs_offset_px": None, "limit_px": OFFSET_LIMIT_PX,
                   "manual_review": edge.get("manual_review"), "angle_deg": None,
                   "along_offsets_px": [None, None, None], "measurement_status": UNVERIFIED,
                   "status": UNVERIFIED, "note": problem or edge.get("reason") or ""}
            rows.append(row)
            if problem:
                continue
            if not edge.get("confirmed"):
                row["note"] = edge.get("reason") or "未标注或未确认"
                continue
            # 导入标注也要检查原图边界，不能只依赖窗口的点击校验。
            segments = [edge.get(key) for key in ("color_segment", "depth_segment")]
            width, height = entry["image_size"]
            try:
                outside = any(not (0 <= x < width and 0 <= y < height)
                              for segment in segments for x, y in segment)
                measured = edge_offset_px(*segments)
            except (TypeError, ValueError, OverflowError):
                row["note"] = "线段缺失或坐标格式错误"
                continue
            if outside or measured.offset_px is None:
                row["note"] = "端点超出原图范围" if outside else measured.reason
                continue
            row.update(offset_px=round(measured.offset_px, 3), abs_offset_px=round(measured.abs_px, 3),
                       **edge_diagnostics(*segments))
            row["measurement_status"] = PASS if measured.abs_px <= OFFSET_LIMIT_PX else FAIL
            review = edge.get("manual_review")
            row["status"] = row["measurement_status"]
            if review == "no":
                row.update(status=FAIL, note="人工复查发现明显张开或整体错位")
            elif review != "yes" and row["status"] == PASS:
                row.update(status=UNVERIFIED, note="数值合格，尚未人工复查")
            if structure_problem and row["status"] != FAIL:
                row.update(status=UNVERIFIED, note=structure_problem)
            if geometry.get("status") != PASS:
                row.update(status=UNVERIFIED,
                           note=(row["note"] + "；几何语义尚未核清").strip("；"))
    if mode in ("raw", "sync"):
        for row in rows:
            row.update(status=SKIP, note=f"{mode} 模式未开 D2C，空间检查不适用")
    return rows


def write_spatial(out: Path, rows: list) -> None:
    """写 spatial_edges.csv，并把汇总塞回 summary.json。"""
    _write_csv(out / "spatial_edges.csv", SPATIAL_COLUMNS, rows)
    summary_path = out / "summary.json"
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["spatial"] = {
        "status": worst(row["status"] for row in rows),
        "limit_px": OFFSET_LIMIT_PX,
        "edges_per_pair": EDGES_PER_PAIR,
        "rows": [{key: row[key] for key in SPATIAL_COLUMNS} for row in rows],
        "conditions": {condition: worst(row["status"] for row in rows
                                          if row["condition_id"] == condition)
                       for condition in REQUIRED_CONDITIONS},
        "note": "只报每条边的近似中点偏移；全工作区与毫米精度未验收",
    }
    update_overall(summary)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def update_overall(summary: dict) -> None:
    """软件检查总体状态；现场运动、回放等仍需各自的证据，不能由这里代验收。"""
    mode = (summary.get("capture") or {}).get("mode")
    items = [(summary.get(key) or {}).get("status", UNVERIFIED)
             for key in ("capture_integrity", "sync")]
    if mode not in ("raw", "sync"):
        items += [(summary.get(key) or {}).get("status", UNVERIFIED)
                  for key in ("geometry", "spatial")]
    summary["overall_status"] = worst(items)


def _fmt(value, digits: int = 2) -> str:
    """表格里的小数格式化；None 显示成 em dash，不显示 0。"""
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def write_report(out: Path, summary: dict) -> None:
    """手写 report.html：结论在最上面，失败样例与「没验证」逐项列出。"""
    sync = summary.get("sync") or {}
    spatial = summary.get("spatial") or {}
    capture = summary.get("capture") or {}
    integrity = summary.get("capture_integrity") or {}
    geometry = summary.get("geometry") or {}
    pair = summary.get("selected") or {}

    def row(label, value):
        return f"<tr><th>{html.escape(str(label))}</th><td>{_fmt(value)}</td></tr>"

    def status_span(status):
        color = {"PASS": "#1a7f37", "FAIL": "#b42318",
                 "UNVERIFIED": "#9a6700", "SKIP": "#57606a"}.get(status, "#57606a")
        return f'<span class="status" style="color:{color}">{html.escape(str(status))}</span>'

    sync_rows = "".join(
        row(key, sync.get(key)) for key in
        ("status", "color_rate", "depth_rate", "pairs", "color_frames", "depth_frames",
         "abs_dt_p50_ms", "abs_dt_p95_ms", "abs_dt_max_ms", "signed_dt_p50_ms")
        if key in sync
    )
    reasons = "".join(f"<li>{html.escape(str(r))}</li>" for r in summary.get("reasons") or [])
    integrity_rows = "".join(row(role, value) for role, value in (integrity.get("streams") or {}).items())
    geometry_rows = "".join(row(key, value) for key, value in geometry.items())
    frequency_rows = ""
    trend_rows = ""
    for role, value in (sync.get("frequency") or {}).items():
        frequency_rows += (f"<tr><td>{html.escape(role)}</td><td>{_fmt(value.get('fps'))}</td>"
                           f"<td>{_fmt(value.get('header_fps'))}</td>"
                           f"<td>{_fmt(value.get('gap_p95_ms'))}</td>"
                           f"<td>{_fmt(value.get('gap_max_ms'))}</td></tr>")
        for window in value.get("windows_10s") or []:
            trend_rows += (f"<tr><td>{html.escape(role)}</td><td>{window['start_s']}–{window['end_s']}</td>"
                           f"<td>{window['frames']}</td><td>{_fmt(window.get('fps'))}</td>"
                           f"<td>{_fmt(window.get('header_fps'))}</td>"
                           f"<td>{_fmt(window.get('gap_max_ms'))}</td></tr>")

    spatial_rows_html = ""
    for item in spatial.get("rows") or []:
        spatial_rows_html += (
            "<tr>"
            f"<td>{html.escape(str(item.get('session') or ''))}</td>"
            f"<td>{html.escape(str(item.get('condition_id') or ''))}</td>"
            f"<td>{html.escape(str(item.get('pair_id') or ''))}</td>"
            f"<td>{html.escape(str(item.get('edge_id') or ''))}</td>"
            f"<td>{_fmt(item.get('offset_px'))}</td>"
            f"<td>{_fmt(item.get('abs_offset_px'))}</td>"
            f"<td>{_fmt(item.get('limit_px'), 1)}</td>"
            f"<td>{html.escape(str(item.get('manual_review') or '—'))}</td>"
            f"<td>{_fmt(item.get('angle_deg'))}</td>"
            f"<td>{_fmt(item.get('along_offsets_px'))}</td>"
            f"<td>{status_span(item.get('status'))}</td>"
            f"<td>{html.escape(str(item.get('note') or ''))}</td>"
            "</tr>")

    images = ""
    pid = pair.get("pair_id")
    if pid:
        images = (
            f'<figure><img src="overlays/{html.escape(pid)}_overlay.png">'
            f"<figcaption>选中配对 {html.escape(pid)}（左彩色 / 右深度伪彩）</figcaption></figure>"
            f'<figure><img src="overlays/{html.escape(pid)}_valid_mask.png">'
            "<figcaption>有效深度掩码（白 = 非零）</figcaption></figure>"
        )

    output_links = "".join(f'<li><a href="{name}">{name}</a></li>' for name in
                           ("pairs.csv", "unmatched.csv", "summary.json", "spatial_edges.csv", "annotations.json")
                           if (out / name).is_file())
    source_links = "".join(
        f'<li><a href="{html.escape(os.path.relpath(item["report"], out))}">'
        f'{html.escape(item["session"])}</a></li>' for item in summary.get("sources", []))

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>RGB-D 同步与对齐冒烟报告</title>
<style>
 body{{font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC",sans-serif;
      margin:2rem auto;max-width:1040px;padding:0 1rem;color:#1f2328;line-height:1.6}}
 h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #d0d7de}}
 table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.92rem}}
 th,td{{border:1px solid #d0d7de;padding:.35rem .5rem;text-align:left;vertical-align:top}}
 th{{background:#f6f8fa;white-space:nowrap}} .status{{font-weight:600}}
 figure{{margin:1rem 0}} img{{max-width:100%;border:1px solid #d0d7de}}
 code{{background:#f6f8fa;padding:.1rem .3rem;border-radius:4px}}
 .warn{{background:#fff8c5;border-left:4px solid #d4a72c;padding:.6rem .8rem}}
</style></head><body>
<h1>RGB-D 同步与静态像素对齐冒烟报告</h1>
<p>会话 <code>{html.escape(str(summary.get('session') or '—'))}</code>；
条件 <code>{html.escape(str(pair.get('condition_id') or '—'))}</code>；
生成时间 {html.escape(str(summary.get('generated_at') or '—'))}</p>
<p class="warn">结论边界：只回答「同一段 bag 内彩色与深度的时间配对、以及三个场景的静态像素偏移」。
不等于全工作区验收，也不含三维毫米精度。没测的项目一律标 UNVERIFIED / SKIP，不用报告代替验收。</p>

<h2>结论</h2>
<table>
 <tr><th>软件检查总体（不代替现场阶段验收）</th><td>{status_span(summary.get('overall_status', UNVERIFIED))}</td></tr>
 <tr><th>采集完整性</th><td>{status_span(integrity.get('status', UNVERIFIED))}</td></tr>
 <tr><th>时间同步</th><td>{status_span(sync.get('status'))}</td></tr>
 <tr><th>几何语义</th><td>{status_span(geometry.get('status', UNVERIFIED))}</td></tr>
 <tr><th>空间对齐</th><td>{status_span(spatial.get('status'))}</td></tr>
</table>
<ul>{reasons}</ul>

<h2>实际时间覆盖</h2>
<p>配对率只描述已收到的窗口内图像，不能证明录满声明时长。终点余量不参与配对统计。</p>
<table>{row('声明稳定时长 s', integrity.get('declared_duration_s'))}{integrity_rows}</table>
<p>{html.escape('；'.join(integrity.get('reasons') or []))}</p>

<h2>采集与配置</h2>
<table>
 {row('采集模式', capture.get('mode'))}
 {row('彩色话题', capture.get('color_topic'))}
 {row('深度话题', capture.get('depth_topic'))}
 {row('彩色档位', capture.get('color_profile'))}
 {row('深度档位', capture.get('depth_profile'))}
 {row('时间域', capture.get('time_domain'))}
 {row('冷启动预热', (str(capture.get('warmup_s')) + ' s') if capture.get('warmup_s') is not None else None)}
 {row('稳定段', (str(capture.get('duration_s')) + ' s') if capture.get('duration_s') is not None else None)}
</table>

<h2>时间同步</h2>
<table>{sync_rows}</table>
<p>规则：按 header 整数纳秒配对，容差固定 ±{_fmt((sync.get('rules') or {}).get('tol_ns', 0) / 1e6, 1)} ms；
每帧最多用一次；有符号时间差 = 深度 − 彩色。P95 只作描述，不作为独立通过证据。</p>

<h2>静态像素对齐（三场景 / 六条边）</h2>
<table>
 <tr><th>session</th><th>条件</th><th>配对</th><th>边界</th><th>有符号偏移 px</th><th>绝对偏移 px</th>
 <th>门槛 px</th><th>人工复查</th><th>夹角 °</th><th>沿线 5/50/95% 偏移 px</th><th>状态</th><th>备注</th></tr>
 {spatial_rows_html}
</table>
<p>门槛 {_fmt(spatial.get('limit_px'), 1)} px 是待现场复核的排错门槛；数值只代表每条边的中点，
两端是否张开由人工复查给出。</p>

<h2>几何语义前置检查</h2><table>{geometry_rows}</table>
<h2>频率与接收间隔</h2>
<p>窗口 fps = 窗口帧数 / 声明时长；header fps = (帧数−1) / header 跨度。
两者分开看，接收间隔反映落盘接收节奏，header 统计不能证明曝光完全同步。</p>
<table><tr><th>流</th><th>窗口 fps</th><th>header fps</th><th>接收间隔 P95 ms</th><th>最大接收间隔 ms</th></tr>{frequency_rows}</table>
<table><tr><th>流</th><th>窗口 s</th><th>帧数</th><th>窗口 fps</th><th>header fps</th><th>最大接收间隔 ms</th></tr>{trend_rows}</table>

<h2>选中图对</h2>
{images or '<p>本段没有选中图对。</p>'}

<h2>输出文件</h2>
<ul>
 {output_links}
</ul>
<h2>来源报告</h2><ul>{source_links}</ul>
</body></html>
"""
    (out / "report.html").write_text(doc, encoding="utf-8")


def refresh(out: Path, mode: str | None) -> list:
    """标注变了以后重算空间结果并重写报告；返回六行结果。"""
    rows = spatial_rows(out, mode)
    write_spatial(out, rows)
    summary_path = out / "summary.json"
    if summary_path.is_file():
        write_report(out, json.loads(summary_path.read_text(encoding="utf-8")))
    return rows


def read_meta(path: Path) -> dict:
    """读 bag 的 meta.yaml；没有就返回空字典（报告里写未知，不编造）。"""
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
