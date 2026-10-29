"""rgbd_capture.launch.py：RGB-D 同步采集入口，一个 mode 决定同步与 D2C。

用大白话说
    这是「彩色 + 对齐深度」那套启动方式。三档模式对应计划书步骤 B 的三段对照：
        raw      同步关、D2C 关 —— 原始双流基线，用来知道不干预时是什么样
        sync     同步开、D2C 关 —— 只看时间戳与频率有没有被同步器改动
        aligned  同步开、D2C 开 —— 融合输入候选模式，深度已对齐到彩色坐标
    纯深度调试继续用 camera_only.launch.py，两者互不影响。

参数从哪来
    唯一来源是 config/rgbd_capture.yaml（装到 share 下），优先级：
        显式启动参数  >  配置文件  >  这里声明的默认值
    命令行上把某个参数再给一遍就覆盖配置文件；但由 mode 决定的
    enable_frame_sync / depth_registration 不允许改反——给了相反的值直接报错，
    而不是悄悄按 mode 覆盖掉。理由是「我以为开了同步、其实被关掉了」这种错，
    在报告里看不出来，只能靠启动时就炸掉来避免。

为什么要透传 align_mode / time_domain
    camera_only.launch.py 的透传表里没有这三个键，所以从外层怎么传都到不了驱动，
    而「改 YAML 没生效」「同步到底开没开」正是计划书步骤 A 要消除的问题。
    本文件把同步、对齐、时间域、后处理全部显式透传，并打印最终生效值。

已知的坑
    · 驱动参数名随版本变（本机 orbbec_camera 2.9.3 已核对），换版本先
      `ros2 launch orbbec_camera gemini_330_series.launch.py --show-args`。
    · 点云必须关：1280x800 的点云 16.4 MB/帧，30fps 近 500 MB/s，会打满 DDS 缓冲，
      表现是「话题有频率但数据不完整」。
    · 关彩色时同步必须一起关，否则同步器一直等彩色帧，深度整条停发。
      本文件在 mode 里已经把这两个开关绑在一起，不用手工记。
    · enable_depth_scale 关了深度单位会变，后面所有距离错一个量级且不报错。
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

#: 模式 -> (帧同步, D2C)。顺序与计划书步骤 B 的三段对照一致，不要改名。
MODES = {
    "raw": (False, False),
    "sync": (True, False),
    "aligned": (True, True),
}

#: 空串表示「用户没显式给」，此时取配置文件的值。不能用 false 当默认值，
#: 那样就分不清「用户说了关」和「用户没说」——而这两者对冲突检查的意义完全不同。
UNSET = ""

#: 会原样透传给驱动的键。名字必须与 orbbec_camera 的 launch 参数一致。
PASSTHROUGH_ARGS = (
    "camera_name", "serial_number",
    "enable_color", "color_width", "color_height", "color_fps",
    "depth_width", "depth_height", "depth_fps", "enable_depth_scale",
    "enable_frame_sync", "depth_registration",
    "align_mode", "align_target_stream", "time_domain",
    "enable_point_cloud", "enable_colored_point_cloud",
    "enable_noise_removal_filter", "enable_spatial_filter", "enable_temporal_filter",
    "enable_decimation_filter", "enable_threshold_filter",
    "enable_hardware_noise_removal_filter", "enable_disp_outliers_filter",
)

#: 布尔键：命令行取值与 YAML 取值都要归一化成 true/false。
BOOL_ARGS = (
    "enable_color", "enable_depth_scale", "enable_frame_sync", "depth_registration",
    "enable_point_cloud", "enable_colored_point_cloud",
    "enable_noise_removal_filter", "enable_spatial_filter", "enable_temporal_filter",
    "enable_decimation_filter", "enable_threshold_filter",
    "enable_hardware_noise_removal_filter", "enable_disp_outliers_filter",
)

#: 整数键：档位，0 = 用相机默认档。
INT_ARGS = (
    "color_width", "color_height", "color_fps",
    "depth_width", "depth_height", "depth_fps",
)

ARG_DESCRIPTIONS = {
    "mode": "采集模式 raw / sync / aligned；默认取配置文件里的 aligned",
    "config_file": "配置文件路径；默认是 share/vision_depth/config/rgbd_capture.yaml",
    "camera_name": "话题前缀：/camera/...；多机时改成 /camera1 之类",
    "serial_number": "设备序列号；留空表示由驱动选第一台",
    "enable_color": "彩色流；本阶段 RGB-D 必须开",
    "color_width": "彩色宽，0 = 相机默认档",
    "color_height": "彩色高，0 = 相机默认档",
    "color_fps": "彩色帧率，0 = 相机默认档",
    "depth_width": "深度宽，0 = 相机默认档",
    "depth_height": "深度高，0 = 相机默认档",
    "depth_fps": "深度帧率，0 = 相机默认档",
    "enable_depth_scale": "深度值按 scale 换算成毫米；默认开，别关",
    "enable_frame_sync": "帧同步；由 mode 决定，显式给相反值会报错",
    "depth_registration": "D2C 深度对齐；由 mode 决定，显式给相反值会报错",
    "align_mode": "对齐实现 SW / HW；默认 SW",
    "align_target_stream": "对齐目标，本阶段只允许 COLOR",
    "time_domain": "时间域 global / device / system；默认 global",
    "enable_point_cloud": "点云；必须关，见文件头说明",
    "enable_colored_point_cloud": "彩色点云；必须关",
    "enable_noise_removal_filter": "驱动降噪滤波，默认开；开了就要在报告里注明深度被处理过",
    "enable_spatial_filter": "空间滤波，默认关",
    "enable_temporal_filter": "时间滤波，默认关",
    "enable_decimation_filter": "降采样滤波，默认关",
    "enable_threshold_filter": "阈值滤波，默认关",
    "enable_hardware_noise_removal_filter": "硬件降噪，默认关",
    "enable_disp_outliers_filter": "视差离群点滤波，默认关",
}


def _as_bool(name: str, text) -> bool:
    """把命令行/YAML 的布尔写法统一成 bool；认不出来就报错，不瞎猜。

    :param name: 参数名，用于报错信息
    :param text: 原始值（可能是 "true" / "False" / 1 / True）
    :return: 解析后的布尔值
    :raises RuntimeError: 取值不是布尔写法
    """
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    raise RuntimeError(f"参数 {name}={text!r} 不是布尔值（写 true 或 false）")


def _defaults(cfg: dict) -> dict:
    """把配置文件的嵌套结构摊平成驱动 launch 认的扁平键。

    :param cfg: YAML 读出来的字典
    :return: 扁平键值表，值已归一化成 bool / int / str
    """
    color = cfg.get("color") or {}
    depth = cfg.get("depth") or {}
    d2c = cfg.get("d2c") or {}
    cloud = cfg.get("point_cloud") or {}
    post = cfg.get("postprocess") or {}
    return {
        "camera_name": str(cfg.get("camera_name") or "camera"),
        "serial_number": str(cfg.get("serial_number") or ""),
        # 这两个键最终由 mode 决定，这里只是把配置里「记录的值」读出来，
        # 好让 mode 与配置不一致时打一行提示，而不是默默改掉。
        "enable_frame_sync": bool((cfg.get("frame_sync") or {}).get("enable", True)),
        "depth_registration": bool(d2c.get("registration", True)),
        "enable_color": bool(color.get("enable", True)),
        "color_width": int(color.get("width", 0)),
        "color_height": int(color.get("height", 0)),
        "color_fps": int(color.get("fps", 0)),
        "depth_width": int(depth.get("width", 0)),
        "depth_height": int(depth.get("height", 0)),
        "depth_fps": int(depth.get("fps", 0)),
        "enable_depth_scale": bool(depth.get("enable_depth_scale", True)),
        "align_mode": str(d2c.get("align_mode", "SW")),
        "align_target_stream": str(d2c.get("align_target_stream", "COLOR")).upper(),
        "time_domain": str(cfg.get("time_domain", "global")),
        "enable_point_cloud": bool(cloud.get("enable", False)),
        "enable_colored_point_cloud": bool(cloud.get("enable_colored", False)),
        "enable_noise_removal_filter": bool(post.get("noise_removal", True)),
        "enable_spatial_filter": bool(post.get("spatial", False)),
        "enable_temporal_filter": bool(post.get("temporal", False)),
        "enable_decimation_filter": bool(post.get("decimation", False)),
        "enable_threshold_filter": bool(post.get("threshold", False)),
        "enable_hardware_noise_removal_filter": bool(post.get("hardware_noise_removal", False)),
        "enable_disp_outliers_filter": bool(post.get("disp_outliers", False)),
    }


def _fmt(name: str, value) -> str:
    """统一成 launch 能吃的字符串：布尔写 true/false，整数不带小数点。"""
    if name in BOOL_ARGS:
        return "true" if _as_bool(name, value) else "false"
    if name in INT_ARGS:
        try:
            return str(int(value))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"参数 {name}={value!r} 不是整数") from exc
    return str(value)


def _resolve(context):
    """读配置、套模式、查冲突，返回真正要执行的 action 列表。

    放在 OpaqueFunction 里是因为这些判断要在 launch 展开时做：配置读不出来、
    模式写错、显式参数与模式打架，都应该在启动前就报错并退出，
    而不是等相机起来了、录了半段 bag 才发现同步其实没开。

    :param context: launch 上下文，用来取各 LaunchConfiguration 的最终值
    :return: action 列表（LogInfo + IncludeLaunchDescription）
    """
    cfg_path = LaunchConfiguration("config_file").perform(context).strip()
    if not cfg_path:
        cfg_path = os.path.join(get_package_share_directory("vision_depth"),
                                "config", "rgbd_capture.yaml")
    if not os.path.isfile(cfg_path):
        raise RuntimeError(f"找不到 RGB-D 配置文件：{cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise RuntimeError(f"{cfg_path} 的内容不是 YAML 映射（key: value）")

    effective = _defaults(cfg)

    mode = LaunchConfiguration("mode").perform(context).strip().lower()
    if not mode:
        mode = str(cfg.get("mode") or "aligned").strip().lower()
    if mode not in MODES:
        raise RuntimeError(f"mode={mode!r} 不认识；只能是 {' / '.join(MODES)}")
    want_sync, want_d2c = MODES[mode]

    # 命令行显式值：与 mode 相反的直接拒绝，避免「报告写着 aligned、实际没对齐」。
    explicit = {}
    for name in PASSTHROUGH_ARGS:
        value = LaunchConfiguration(name).perform(context).strip()
        if value != UNSET:
            explicit[name] = value

    if "enable_frame_sync" in explicit:
        if _as_bool("enable_frame_sync", explicit["enable_frame_sync"]) != want_sync:
            raise RuntimeError(
                f"mode={mode} 要求帧同步 {'开' if want_sync else '关'}，"
                f"但显式给了 enable_frame_sync:={explicit['enable_frame_sync']}。"
                "要改同步请直接换 mode，不要让两个开关各说各话。")
    if "depth_registration" in explicit:
        if _as_bool("depth_registration", explicit["depth_registration"]) != want_d2c:
            raise RuntimeError(
                f"mode={mode} 要求 D2C {'开' if want_d2c else '关'}，"
                f"但显式给了 depth_registration:={explicit['depth_registration']}。"
                "要改对齐请直接换 mode。")
    if "align_target_stream" in explicit:
        if explicit["align_target_stream"].upper() != "COLOR":
            raise RuntimeError(
                "本阶段只做「深度对齐到彩色」，align_target_stream 只能是 COLOR；"
                f"收到 {explicit['align_target_stream']}。")
    for name in ("enable_point_cloud", "enable_colored_point_cloud"):
        if name in explicit and _as_bool(name, explicit[name]):
            raise RuntimeError(
                f"{name}:=true 会让点云占满 DDS 缓冲（30fps 近 500 MB/s），"
                "采集入口固定关闭点云；调试点云请另起 camera_only 或官方 launch。")

    notes = []
    recorded_sync = _as_bool("enable_frame_sync", effective["enable_frame_sync"])
    if recorded_sync != want_sync:
        notes.append(f"配置里 frame_sync={recorded_sync}，按 mode={mode} 实际用 "
                     f"{want_sync}")
    effective["enable_frame_sync"] = want_sync
    effective["depth_registration"] = want_d2c

    # 显式值最后覆盖，保证「命令行 > 配置文件」。
    effective.update(explicit)

    resolved = {name: _fmt(name, effective[name]) for name in PASSTHROUGH_ARGS}

    lines = [
        f"RGB-D 采集模式：{mode}（帧同步 {'开' if want_sync else '关'}，"
        f"D2C {'开' if want_d2c else '关'}）",
        f"配置文件：{cfg_path}",
        f"彩色档位 {resolved['color_width']}x{resolved['color_height']}"
        f"@{resolved['color_fps']}；深度档位 {resolved['depth_width']}x"
        f"{resolved['depth_height']}@{resolved['depth_fps']}（0 = 相机默认档）",
        f"align_mode={resolved['align_mode']} "
        f"align_target={resolved['align_target_stream']} "
        f"time_domain={resolved['time_domain']} "
        f"depth_scale={resolved['enable_depth_scale']}",
        f"后处理：noise_removal={resolved['enable_noise_removal_filter']} "
        f"spatial={resolved['enable_spatial_filter']} "
        f"temporal={resolved['enable_temporal_filter']} "
        f"decimation={resolved['enable_decimation_filter']} "
        f"threshold={resolved['enable_threshold_filter']}",
    ]
    lines.extend(notes)

    driver_launch = os.path.join(
        get_package_share_directory("orbbec_camera"), "launch", "gemini_330_series.launch.py")
    dds_profile = os.path.join(get_package_share_directory("vision_depth"),
                               "config", "fastdds_shm_only.xml")
    lines.append(f"DDS 传输：仅 SHM（{dds_profile}）；采集端由 rgbd_capture 同步设置")

    return [
        LogInfo(msg=line) for line in lines
    ] + [
        # 大图走 UDP 会分片，接收端一忙就在 transport 层丢帧（实测 1280x720 时
        # 13 秒丢了 113 条，配对率掉到 87%）。两端都在本机，只走 SHM 最稳。
        SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", dds_profile),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(driver_launch),
            launch_arguments=resolved.items(),
        )
    ]


def generate_launch_description():
    """声明全部参数（默认空串 = 没显式给），把实际展开交给 `_resolve`。"""
    declared = [
        DeclareLaunchArgument("mode", default_value=UNSET,
                              description=ARG_DESCRIPTIONS["mode"]),
        DeclareLaunchArgument("config_file", default_value=UNSET,
                              description=ARG_DESCRIPTIONS["config_file"]),
    ]
    for name in PASSTHROUGH_ARGS:
        declared.append(DeclareLaunchArgument(name, default_value=UNSET,
                                              description=ARG_DESCRIPTIONS[name]))
    return LaunchDescription(declared + [OpaqueFunction(function=_resolve)])
