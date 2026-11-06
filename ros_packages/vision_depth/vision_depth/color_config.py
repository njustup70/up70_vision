"""读取 RGB-D 判色 YAML，校验 HSV 范围与有效像素门槛。"""
from pathlib import Path

import yaml
from vision.yolo.color_check import parse_thresholds
from vision.yolo.pipeline import ColorConfig


def load_color_config(path):
    """返回 HSV 阈值与门槛；缺文件、非映射或非法门槛直接报错。

    :param path: 含颜色范围及可选 color 门槛段的 YAML 文件路径
    :return: (thresholds, ColorConfig)，读取后校验，不修改原文件
    """
    with Path(path).open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f'{path}: expected a YAML mapping')
    settings = data.get('color') or {}
    if not isinstance(settings, dict):
        raise ValueError(f'{path}: color must be a mapping')
    config = ColorConfig(**settings)
    config.validate()
    thresholds = parse_thresholds({key: value for key, value in data.items() if key != 'color'})
    return thresholds, config
