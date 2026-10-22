"""终端输出与自检判定：生效配置、逐帧一行、批处理汇总、实时收尾。

用大白话说
    程序跑起来之后要跟人说话：用了哪份配置、这一帧检出几个、最后总共跑成什么样、
    自检算不算通过。这个文件就是"说话的那张嘴"，它不改任何结果，只负责排版和判定。

放什么 / 不放什么
    放各类输出与一条判定：
        · `print_config()`        生效配置（含 --set 覆盖后的值）
        · `format_centers()`      一帧的"ID@(x,y)"摘要（任务单要求终端带 ID 与中心点）
        · `format_live_line()`    实时入口每 0.5 s 一行的固定格式摘要
        · `print_batch_summary()` 批处理的汇总 + 自检，返回"是否通过"给退出码用
        · `print_live_summary()`  实时入口的收尾三行
    **不放**检测、判色、落盘逻辑——它只读 `cfg` 与 `summary`，不碰传感器也不写字。

依赖谁 / 被谁用
    只依赖 `core.types.is_marker()`（拼中心点时要区分"码"与"色块"）。
    被 `cli/run.py` 与 `cli/live.py` 调用。

对外约定
    `print_batch_summary()` 返回 `bool`：True = 自检通过（退出码 0），False = 未通过（退出码 1）。
    **打印出来的每一行都与拆分前逐字节一致**——这次拆文件只动代码位置，不动输出，
    所以"改前改后 diff 终端输出"是这项重构唯一可靠的验收方式。
"""

from ..core.types import is_marker


def format_centers(detections, limit: int = 8) -> str:
    """把每个检出物的 "id@(x,y)" 拼成一行（超出 limit 个就截断）。

    用大白话说
        一帧里可能有十几个框，全打出来会刷屏，所以只列前 8 个，剩下的折成 `…+N`。
        ArUco 码前面写它的 ID，色块没有 ID，写个 `-` 占位。

    飞书任务单任务二第 4 条要求"在终端输出识别到的 ArUco 码信息，**包括 ID 和中心点坐标**"。
    判色族的框没有码 ID，写成 `-@(x,y)`；一帧十几个框时截断，免得刷屏。

    :param detections: 本帧的检出物列表（`Detection` 与 `ArucoMarker` 混排也可）
    :param limit: 最多列几个；超出部分只报个数
    :return: 形如 `[12@(320,240) -@(88,96) …+3]` 的单行字符串
    """
    parts = []
    for det in list(detections)[:limit]:
        cx, cy = det.center
        tag = str(int(det.marker_id)) if is_marker(det) else "-"
        parts.append(f"{tag}@({int(round(cx))},{int(round(cy))})")
    if len(detections) > limit:
        parts.append(f"…+{len(detections) - limit}")
    return "[" + " ".join(parts) + "]"


def format_live_line(elapsed: float, frames: int, fps: float, markers, poses=None) -> str:
    """把本帧状态拼成一行固定格式的终端摘要（便于 diff、贴进 PR）。

    用大白话说
        把"跑了多久、第几帧、现在几帧每秒、看到几个码、每个码在哪"串成一句话。
        开了位姿之后，每个码后面再多三个数："多远、斜多少、转多少"。
        码多的时候这句话就长一点，没码的时候只有前半句。

    格式
        `t=<秒>s | frames=<已处理帧数> | fps=<滑动平均> | markers=<个数>
         [| id=N center=(x, y)[ d=<毫米>mm tilt=<度> yaw=<度>]]*`
        每个 marker 追加一段；开了位姿时该段尾部再多三个字段，**字段顺序固定**。

    输出示例（字段之间是 " | "，与本函数实际输出一致）
        无码：`t=3.20s | frames=96 | fps=29.8 | markers=0`
        两码：`t=3.20s | frames=96 | fps=29.8 | markers=2 | id=3 center=(320.0, 240.0) | id=7 center=(501.0, 180.5)`
        开位姿：`... | markers=1 | id=7 center=(501.0, 180.5) d=512mm tilt=12.4 yaw=-30.1`

    为什么是"固定格式 + 一行一条"
        · 一行一条便于 `grep markers=` / `awk` 统计，也便于直接贴进 PR；
        · 固定字段顺序才可以用 diff 对比两次运行的差异；
        · 不用 `\\r` 原地刷新：重定向到文件后会糊成一团，日志也没法回看。

    :param elapsed: 从循环开始到本帧的秒数（time.monotonic 差值，非墙钟时间）
    :param frames: 已处理的帧数（含本帧）
    :param fps: 最近 `cli/live.py` 里 `LIVE_FPS_WINDOW` 帧的滑动平均帧率
    :param markers: 本帧的 `list[ArucoMarker]`（无码时是 `[]`）
    :param poses: 本帧的 `list[MarkerPose]`；None 或空列表表示没开位姿，
        **此时输出与加位姿之前逐字节一致**（别让新功能改了别人的日志格式）
    :return: 单行字符串，调用方直接 print（不带换行符）
    """
    # 前半段：时间 / 帧数 / 帧率 / 本帧码数。这些都是标量，顺序固定。
    parts = [f"t={elapsed:.2f}s", f"frames={frames}", f"fps={fps:.1f}",
             f"markers={len(markers)}"]
    # 位姿按 id 建索引：markers 可能因解算失败而缺位姿，不能按下标硬对
    pose_by_id = {int(pose.marker_id): pose for pose in (poses or [])}
    # 后半段：每个码一段。markers 已由检测器按 marker_id 升序，输出顺序才稳定可 diff。
    for marker in markers:
        cx, cy = marker.center_px
        # center 用契约里的四角算术平均（ArucoMarker.center_px），不是 bbox 中心：
        # 码倾斜时两者不是同一个位置，用错会让"中心点"看起来偏。
        segment = f"id={int(marker.marker_id)} center=({cx:.1f}, {cy:.1f})"
        pose = pose_by_id.get(int(marker.marker_id))
        if pose is not None:
            # 距离取整到毫米、角度一位小数：终端一行要能扫读，多给的位数没人看
            segment += (f" d={pose.distance_mm:.0f}mm"
                        f" tilt={pose.tilt_deg:.1f} yaw={pose.yaw_deg:.1f}")
        parts.append(segment)
    # 用 " | " 连接：人在终端里一眼能分清字段的分组边界
    return " | ".join(parts)


def print_config(cfg, config_path, thresholds_path) -> None:
    """把本次真正生效的配置打一遍（两个入口共用）。

    目的不是"好看"，而是让日志/PR 里能直接看到**实际生效**的值：
    配置文件路径、阈值表路径、输入源与它的参数、判色阈值、检测器、输出、日志等级。
    用 `--set` 覆盖后，这里打印的就是覆盖后的结果，不用再回头翻命令行。

    :param cfg: 校验过的 `AppConfig`
    :param config_path: 配置文件绝对路径（便于确认读的是哪一份）
    :param thresholds_path: 解析后的阈值表绝对路径
    """
    print(f"配置文件   : {config_path}")
    print(f"阈值表     : {thresholds_path}")
    print(f"输入源     : {cfg.input.source}")
    if cfg.input.source == "synthetic":
        s = cfg.input.synthetic
        print(f"  合成     : {s.width}x{s.height} {s.num_frames}帧 x {s.blocks_per_frame}块 "
              f"seed={s.seed} 噪声={s.noise_sigma}")
    elif cfg.input.source == "camera":
        c = cfg.input.camera
        print(f"  相机     : 第{c.index}台 曝光{c.exposure_us:.0f}us 增益{c.gain_db:.1f}dB "
              f"超时{c.timeout_ms}ms 最多{c.max_frames or '不限'}帧")
    else:
        r = cfg.input.replay
        print(f"  回放     : {r.path} kind={r.kind} stride={r.frame_stride} max={r.max_frames}")
    print(f"判色       : S>={cfg.color.min_saturation} V>={cfg.color.min_value} "
          f"命中率>={cfg.color.hit_ratio_min} 最少有效像素={cfg.color.min_valid_pixels}")
    print(f"检测器     : {cfg.detector.name} {cfg.detector.params}")
    print(f"输出       : enabled={cfg.output.enabled} dir={cfg.output.dir} "
          f"overlay={cfg.output.save_overlay} json={cfg.output.save_json} csv={cfg.output.save_csv}")
    print(f"日志等级   : {cfg.log.level}")


def print_batch_summary(cfg, summary, written) -> bool:
    """打批处理的汇总与自检，返回"自检是否通过"。

    用大白话说
        全部帧跑完了，先把数字列出来（几帧、检出多少、颜色分布、命中率、和真值比对了多少），
        再按"这一批输入是什么 + 用了哪个检测器"挑一条自检规则来核对，
        最后给一句 ✅/❌。这一句就是 `run()` 的退出码依据。

    :param cfg: 已校验的 `AppConfig`（自检规则要按 input.source 与 detector.name 分支）
    :param summary: `core.pipeline.summarize()` 的返回值
    :param written: `OutputWriter.close()` 的返回值（落盘路径字典）
    :return: True = 自检通过（调用方返回 0）；False = 未通过（调用方返回 1）

    自检规则表（改之前先看这里）
        ┌──────────────────────────────┬──────────────────────────────────────┐
        │ 条件                          │ 要求                                  │
        ├──────────────────────────────┼──────────────────────────────────────┤
        │ synthetic + blob              │ 检出数 == 帧数 x blocks_per_frame；    │
        │                               │ 匹配上的框颜色全对                     │
        │ synthetic + aruco             │ 合成图里没有码，检出数必须为 0（不误报）│
        │ synthetic + 其它检测器         │ 跳过色块数检查（色块数对它不适用）      │
        │ 任意源 + stub_aruco            │ 检出数 == 帧数 x len(ids)，且 id 集合一致│
        │ 非合成源 + aruco               │ 无真值，只报检出数量，不做对错判定      │
        └──────────────────────────────┴──────────────────────────────────────┘
    """
    print("\n" + "=" * 62)
    print(f"  帧数        : {summary['frames']}")
    print(f"  检出总数    : {summary['detections']}（未判出颜色 {summary['uncertain']} 个）")
    if summary["ids"]:
        print(f"  出现过的 id : {summary['ids']}")
    if summary["color_counts"]:
        print(f"  颜色分布    : {summary['color_counts']}")
        print(f"  平均命中率  : {summary['mean_ratio']}")
    if summary["truth_total"]:
        print(f"  与真值匹配  : {summary['matched']}/{summary['truth_total']}"
              f"  颜色正确 {summary['color_correct']}（正确率 {summary['color_accuracy']}）")
    print("=" * 62)

    ok = True
    # 合成图自检只对 blob 有意义：那 18 个色块本来就是给 blob 找的。
    # 换成 stub_aruco（伪造码）时不该拿色块数去要求它。
    if cfg.input.source == "synthetic" and cfg.detector.name == "blob":
        expected = cfg.input.synthetic.num_frames * cfg.input.synthetic.blocks_per_frame
        got = summary["detections"]
        print(f"  自检(合成图): 期望 {expected} 个色块，实得 {got} 个 -> "
              f"{'✅ 通过' if got == expected else '❌ 不一致'}")
        ok = ok and got == expected
        if summary["matched"]:
            all_right = summary["color_correct"] == summary["matched"]
            print(f"  自检(判色)  : 匹配 {summary['matched']} 个里判对 "
                  f"{summary['color_correct']} 个 -> {'✅ 通过' if all_right else '❌ 有判错'}")
            ok = ok and all_right
    elif cfg.input.source == "synthetic" and cfg.detector.name == "aruco":
        # 合成图里只有色块、一个 ArUco 码都没有：这一条顺手验了验收清单第 3 条
        # （画面无码时程序正常运行、不误报）
        got = summary["detections"]
        print(f"  自检(无码)  : 合成图里没有 ArUco 码，期望 0 个，实得 {got} 个 -> "
              f"{'✅ 通过' if got == 0 else '❌ 误报'}")
        ok = ok and got == 0
    elif cfg.input.source == "synthetic" and cfg.detector.name != "blob":
        print(f"  自检(合成图): 跳过节数检查（检测器是 {cfg.detector.name}，色块数不适用）")
    if cfg.detector.name == "stub_aruco":
        want_ids = list(cfg.detector.params.get("ids", [0, 1, 2]))
        expected = summary["frames"] * len(want_ids)
        got = summary["detections"]
        ids_ok = set(summary["ids"]) == set(int(i) for i in want_ids) or not want_ids
        print(f"  自检(stub)  : 期望 {expected} 个码、id 集合 {sorted(int(i) for i in want_ids)}，"
              f"实得 {got} 个、id 集合 {summary['ids']} -> "
              f"{'✅ 通过' if (got == expected and ids_ok) else '❌ 不一致'}")
        ok = ok and got == expected and ids_ok
    if cfg.detector.name == "aruco" and cfg.input.source != "synthetic":
        # 回放/相机源没有 ground truth，CLI 这边只能报"检出了什么"；
        # 要按真值逐张核 ID / 四角 / 中心，那是 tools/aruco_check.py 的活。
        print(f"  自检(aruco) : 本次共 {summary['detections']} 个码、id 集合 {summary['ids']}"
              f"（本输入源无真值，不做对错判定）")
        print("              按真值逐张核 ID/四角/中心：python -m up70_vision check-aruco")
    if cfg.output.enabled:
        print(f"  落盘        : {written}")
    print(f"\n{'✅ 运行成功' if ok else '❌ 自检未通过'}")
    return ok


def print_live_summary(frames: int, with_markers: int, avg_fps: float,
                       pose_note: str | None = None,
                       saved_dir: str | None = None,
                       saved_count: int = 0) -> None:
    """打实时入口的收尾汇总（给终端/PR 一份可核对的数字）。

    用大白话说
        退出循环后告诉人："总共处理了几帧、其中几帧有码、平均几帧每秒"。
        开了位姿就再说一句"解算失败几次"，headless 模式再说一句"图存在哪、存了几张"。

    :param frames: 已处理帧数（`--limit` 与汇总一律用它，不用 frame.index）
    :param with_markers: 至少检出一个码的帧数（用来区分"有码帧 / 无码帧"）
    :param avg_fps: 整段运行的平均帧率 = 帧数 / 总耗时；与 HUD 里的滑动平均不是一回事
    :param pose_note: 形如 `边长 82.5 mm，解算失败 0 次`；None 表示没开位姿，这一行不打印
    :param saved_dir: headless 的落盘目录；None 表示窗口模式，这一行不打印
    :param saved_count: headless 实际落盘的 PNG 张数

    平均帧率用"总帧数 / 总耗时"，与循环里的滑动平均 fps 不是一回事：
    前者反映整段运行，后者只反映最近 30 帧（用于 HUD 实时显示）。
    """
    print("=" * 62)
    print(f"  帧数        : {frames}（有码 {with_markers} / 无码 {frames - with_markers}）")
    print(f"  平均帧率    : {avg_fps:.1f} fps")
    if pose_note is not None:
        # 失败次数单独报：它是"有多少帧拿不到毫米"的唯一线索，不算失败退出
        print(f"  位姿        : {pose_note}")
    if saved_dir is not None:
        print(f"  落盘        : {saved_dir}（{saved_count} 张）")
    print("=" * 62)
