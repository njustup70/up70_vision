"""配置加载：读 YAML、应用命令行点分覆盖、把阈值表读进来。

点分覆盖的语义
    --set color.hit_ratio_min=0.6
    · 按现有值的类型转换（int/float/bool/str），因此不用记 --set 的类型
    · 键必须已存在于 YAML 里；写错键会报错并列出同级可用键（避免"改了没生效"）
    · 相对路径以**本配置文件所在目录**为基准（与 configs/default.yaml 的注释一致）
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
    """读配置并校验。

    :return: (AppConfig, 配置文件绝对路径)
    """
    path = config_path or find_default_config()
    path = os.path.abspath(path)
    data = apply_overrides(read_yaml(path), overrides)
    return from_dict(data), path


def resolve_thresholds_path(cfg: AppConfig, config_path: str) -> str:
    """阈值表路径：相对路径以配置文件所在目录为基准。"""
    base_dir = os.path.dirname(os.path.abspath(config_path))
    return resolve_path(base_dir, cfg.color_thresholds_file or DEFAULT_THRESHOLDS)


def load_thresholds(path: str) -> dict:
    """读阈值表 YAML 并校验成内部结构。

    core.color.parse_thresholds 抛的是 ValueError，这里统一成 ConfigError，
    让 CLI 只需处理一种异常。
    """
    data = read_yaml(path)
    try:
        return parse_thresholds(data)
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
