"""配置加载：读 YAML、应用命令行点分覆盖、把阈值表与内参读进来。

点分覆盖的语义
    --set color.hit_ratio_min=0.6
    · 按现有值的类型转换（int/float/bool/str），因此不用记 --set 的类型
    · 键必须已存在于 YAML 里；写错键会报错并列出同级可用键（避免"改了没生效"）
    · 相对路径以**本配置文件所在目录**为基准（与 configs/default.yaml 的注释一致）

判色参数为什么从阈值表里读
    `color_thresholds.yaml` 的文件头有一段保留键 `color:`，装的是判色门槛
    （min_saturation / min_value / hit_ratio_min / ignore_border_px / min_valid_pixels）。
    理由见该文件的注释：这些数与颜色范围是同一件事的两半，调颜色只该开一个文件。
    `load_config()` 会把它**并进** `default.yaml` 的 `color:` 段（default.yaml 里写了
    同名键就以它为准），所以 `--set color.hit_ratio_min=0.6` 与 `cfg.color.*` 的
    语义完全没变，只是默认值的来源换了一个文件。

为什么读文件的事都在这里而不在 core/
    `core/` 是纯算法层，不读文件（契约第 5 节）。内参 YAML 的解析属于"加载"，
    和阈值表同性质，所以放这一层：`load_thresholds()` 给判色，`load_intrinsics()` 给位姿。
    两处的相对路径都相对配置文件目录解析——换工作目录不会读错文件。
"""

import os

import yaml

from ..core.color import parse_thresholds
from .schema import AppConfig, ConfigError, from_dict

DEFAULT_CONFIG_CANDIDATES = (
    "configs/default.yaml",
    "configs/default.yml",
)

DEFAULT_THRESHOLDS = "color_thresholds.yaml"

#: 阈值表里的**保留键**：它的值是判色门槛（一个 mapping），不是颜色名。
#: 解析颜色范围前必须先摘掉，否则会被当成"一个名叫 color 的颜色"而报错。
THRESHOLDS_SETTINGS_KEY = "color"


def resolve_path(base_dir: str, path: str) -> str:
    """相对路径按 base_dir 解析；绝对路径原样返回。"""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def find_default_config() -> str:
    """没显式给 --config 时按约定找默认配置（也可用环境变量 UP70_CONFIG 指定）。"""
    env = os.environ.get("UP70_CONFIG", "").strip()
    if env:
        if not os.path.isfile(env):
            raise ConfigError(f"环境变量 UP70_CONFIG 指向的文件不存在：{env}")
        return env
    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    raise ConfigError(
        f"找不到默认配置（在 {os.getcwd()} 下找过 {list(DEFAULT_CONFIG_CANDIDATES)}）；"
        f"请在仓库根目录运行，或用 --config 指定，或设 UP70_CONFIG")


def _coerce(current, text: str):
    """按现有值的类型转换命令行文本。"""
    if isinstance(current, bool):
        low = text.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"布尔字段只接受 true/false/1/0，收到 {text!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(text)
        except ValueError as exc:
            raise ConfigError(f"该字段是整数，收到 {text!r}") from exc
    if isinstance(current, float):
        try:
            return float(text)
        except ValueError as exc:
            raise ConfigError(f"该字段是小数，收到 {text!r}") from exc
    if isinstance(current, (list, tuple)):
        parts = [p.strip() for p in text.split(",") if p.strip() != ""]
        try:
            return [float(p) if "." in p else int(p) for p in parts]
        except ValueError as exc:
            raise ConfigError(f"该字段是列表，用逗号分隔数字，收到 {text!r}") from exc
    return text


def apply_overrides(data: dict, overrides) -> dict:
    """把 ["a.b=c", ...] 应用到配置 dict（就地修改并返回）。"""
    for item in overrides or []:
        if "=" not in str(item):
            raise ConfigError(f"--set 需要 key=value 形式，收到 {item!r}")
        key, _, value = str(item).partition("=")
        path = [p for p in key.strip().split(".") if p]
        if not path:
            raise ConfigError(f"--set 的键不能为空：{item!r}")
        node = data
        for part in path[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ConfigError(
                    f"--set {key} 路径不存在（在 {part!r} 处断了）；"
                    f"当前可用：{sorted(node.keys())}")
            node = node[part]
        leaf = path[-1]
        if leaf not in node:
            raise ConfigError(
                f"--set {key} 的字段不存在；{'.'.join(path[:-1]) or '<root>'} 下可用："
                f"{sorted(node.keys())}")
        node[leaf] = _coerce(node[leaf], value)
    return data


def read_yaml(path: str) -> dict:
    if not os.path.isfile(path):
        raise ConfigError(f"配置文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件根必须是映射（key: value）：{path}")
    return data


def load_config(config_path: str | None = None, overrides=None) -> tuple:
    """读配置并校验（判色门槛从阈值表文件头补齐后再校验）。

    :return: (AppConfig, 配置文件绝对路径)
    """
    path = config_path or find_default_config()
    path = os.path.abspath(path)
    data = read_yaml(path)
    merge_color_settings(data, path)
    data = apply_overrides(data, overrides)
    return from_dict(data), path


def resolve_thresholds_path(cfg: AppConfig, config_path: str) -> str:
    """阈值表路径：相对路径以配置文件所在目录为基准。"""
    base_dir = os.path.dirname(os.path.abspath(config_path))
    return resolve_path(base_dir, cfg.color_thresholds_file or DEFAULT_THRESHOLDS)


def split_thresholds_doc(data: dict, source: str = "阈值表") -> tuple:
    """把阈值表拆成 (颜色表, 判色门槛)：摘掉保留键 `color:`。

    用大白话说
        阈值表一个文件装了两类东西：文件头的 `color:` 是判色门槛（5 个数），
        下面每个颜色名是 HSV 范围。解析颜色范围前必须先把 `color:` 摘掉，
        否则它会被当成"一个名叫 color 的颜色"，报的错会让人以为阈值表写坏了。

    :param data: `read_yaml()` 读出来的整份 dict
    :param source: 出错时写进消息里的来源（一般为文件路径）
    :return: `(颜色表 dict, 门槛 dict)`；没有 `color:` 段时门槛是 `{}`
    :raises ConfigError: `color:` 段不是 mapping（比如误写成了一个列表）
    """
    settings = data.get(THRESHOLDS_SETTINGS_KEY) or {}
    if not isinstance(settings, dict):
        raise ConfigError(
            f"{source}: 保留键 {THRESHOLDS_SETTINGS_KEY}: 必须是「键: 值」形式（判色门槛），"
            f"收到 {type(settings).__name__}")
    colors = {name: seg for name, seg in data.items() if name != THRESHOLDS_SETTINGS_KEY}
    return colors, dict(settings)


def read_thresholds_settings(path: str) -> dict:
    """只读阈值表文件头的判色门槛；文件不存在时返回 `{}`。

    为什么文件不存在不报错
        报"阈值表不存在"是 `load_thresholds()` 的职责，那里有更完整的消息。
        这里只是**补默认值**：读不到就让 `default.yaml` / schema 的默认值兜住，
        调用方随后一定会走到 `load_thresholds()`，错误由那里统一报。

    :param path: 阈值表路径
    :return: 门槛 dict（可能为空）
    """
    if not os.path.isfile(path):
        return {}
    _colors, settings = split_thresholds_doc(read_yaml(path), path)
    return settings


def merge_color_settings(data: dict, config_path: str) -> dict:
    """把阈值表文件头的 `color:` 段并进配置 dict 的 `color:` 段（就地修改）。

    规则
        · 阈值表里的门槛先铺底，`default.yaml` 里写了同名键就以 `default.yaml` 为准
          （兼容"临时在 default.yaml 里改一个数"的旧写法）
        · 阈值表路径先由 `color.thresholds_file` 解析（相对配置文件目录）
        · 合并发生在 `apply_overrides()` 之前，因此 `--set color.hit_ratio_min=0.6`
          仍然能覆盖（`--set` 只认已存在的键）

    :param data: `read_yaml(配置文件)` 的结果（就地修改）
    :param config_path: 配置文件绝对路径（用于解析阈值表的相对路径）
    :return: 同一个 `data`（方便链式书写）
    """
    color = data.get("color")
    if not isinstance(color, dict):
        color = {}
        data["color"] = color
    rel = color.get("thresholds_file") or DEFAULT_THRESHOLDS
    thresholds_path = resolve_path(os.path.dirname(os.path.abspath(config_path)), str(rel))
    for key, value in read_thresholds_settings(thresholds_path).items():
        color.setdefault(key, value)
    return data


def load_thresholds(path: str) -> dict:
    """读阈值表 YAML 并校验成内部结构（丢掉文件头的 `color:` 门槛段）。

    core.color.parse_thresholds 抛的是 ValueError，这里统一成 ConfigError，
    让 CLI 只需处理一种异常。

    :param path: 阈值表路径
    :return: `{颜色名: (段, ...)}`；**不含** `color` 这个保留键
    """
    data = read_yaml(path)
    colors, _settings = split_thresholds_doc(data, path)
    try:
        return parse_thresholds(colors)
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def resolve_calibration_path(cfg: AppConfig, config_path: str) -> str:
    """内参文件路径：相对路径以配置文件所在目录为基准（与阈值表同一规矩）。

    :param cfg: 已校验的配置
    :param config_path: `configs/default.yaml` 的路径
    :return: 内参 YAML 的绝对路径（文件是否存在由 `load_intrinsics` 判）
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))
    return resolve_path(base_dir, cfg.calibration.file)


def load_intrinsics(path: str):
    """读相机标定结果 YAML，返回 `CameraIntrinsics`。

    用大白话说
        位姿估计要知道"这颗镜头把像素换算成角度"的那几个数（fx/fy/cx/cy 与畸变系数）。
        那些数由 `python -m up70_vision calibrate` 标定完写成一个 YAML，这里只是把它读回来。

    :param path: 标定文件路径
    :return: `core.calibration.CameraIntrinsics`
    :raises ConfigError: 文件不存在、不是合法 YAML、缺字段或字段取值非法。

        文件不存在是**最常见**的一种：它被 `.gitignore` 排除（每台相机一份），
        新 clone 的仓库一定没有，所以报错里直接给出生成命令，别让人去猜。
    """
    from ..core.calibration import CameraIntrinsics

    if not os.path.isfile(path):
        raise ConfigError(
            f"内参文件不存在：{path}\n"
            "  位姿估计需要这台相机的内参（每台一份，不进版本库）。生成方式：\n"
            "    python -m up70_vision calibrate --images data/calib \\\n"
            "        --square-mm <实量格子毫米> --out configs/camera_calibration.yaml\n"
            "  或向做过标定的队友要一份放到该路径。")
    data = read_yaml(path)
    try:
        return CameraIntrinsics.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        # from_dict 抛的是 KeyError/ValueError，混着中文英文；统一包成 ConfigError，
        # 让 CLI 只需要处理一种异常（与 load_thresholds 一致）。
        raise ConfigError(f"{path} 不是有效的标定文件：{exc}") from exc
