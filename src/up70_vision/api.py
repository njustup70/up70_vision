"""库调用门面：一行拿到一次检测结果，不用先过一整份配置。

用大白话说
    想在 ROS 2 节点、临时脚本或 notebook 里"就用一次检测"，本来要自己走四步：
    读配置 -> 校验 -> 建检测器 -> 建 Frame 再跑管线。这里压成一个函数：
    给一张 BGR 图，回来一串检出物（框或码）。参数上只写你要改的那几个。

放什么 / 不放什么
    只放 `detect()` 这一个门面函数。**不复制任何逻辑**：内部仍然走
    `config.loader` -> `core.detectors.base.build_detector` -> `core.pipeline`
    这条原链路，所以门面与 CLI 的行为永远一致（改算法只改一处）。
    不放 IO / 落盘 / 画图：相机、窗口、文件是 `io/` 与 `viz/` 的事，
    调用方自己决定什么时候用；门面只处理"已经拿到的那一帧"。

依赖谁 / 被谁用
    `config`（读配置与阈值表）+ `core`（检测器工厂与管线）。
    **不 import io / viz / cli**——不替调用方管相机、窗口和输出目录。
    上层：ROS 2 节点、临时脚本、notebook；回归测试见 `tests/test_api.py`。

对外约定
    `detect(image_bgr, detector="blob", config=None, **params) -> list`
        · 判色族（`blob` / `yolo`）返回 `list[Detection]`，颜色字段已填好
          （`color` 为 `None` 表示没判出来，与 CLI 一致）
        · ArUco（`aruco` / `stub_aruco`）返回 `list[ArucoMarker]`
        · 无目标返回 `[]`，不是 None，也不抛异常
        · 图不是 HxWx3 的 BGR 时抛 `ValueError`（与 `core.types.Frame` 同一条校验）
    `params` 的键与 `configs/default.yaml` 的 `detector.<name>` 段同名
    （如 `min_area_px=800` / `conf=0.4` / `dictionary="DICT_4X4_50"`），
    写了的覆盖配置值，没写的取配置值。

已知的坑
    · 不给 `config` 时**不读任何 YAML**，检测器参数取 `DETECTOR_DEFAULT_PARAMS`
      的兜底值；此时若当前目录下找不到 `configs/color_thresholds.yaml`，
      判色族会"只出框、不判色"（`color` 全是 `None`，不报错）。
      要判色就显式传 `config="configs/default.yaml"`。
    · `detector="aruco"` 需要 OpenCV >= 4.7（新版 aruco API），低了会抛
      `RuntimeError`——这不是配置问题，是环境问题。
"""

import os

from .config.loader import (DEFAULT_THRESHOLDS, load_config, load_thresholds,
                           resolve_thresholds_path)
from .config.schema import DETECTOR_DEFAULT_PARAMS, ConfigError, from_dict
from .core.detectors.base import build_detector
from .core.pipeline import ColorConfig, process_frame
from .core.types import Frame


def _resolve_thresholds(cfg, config_path):
    """找阈值表并读成内部结构；找不到就返回空表（只检测、不判色）。

    :param cfg: 已校验的 `AppConfig`
    :param config_path: `configs/default.yaml` 的绝对路径；None 表示没读配置文件
    :return: `{颜色名: (段, ...)}`；读不到时返回 `{}`

    为什么读不到不报错
        门面函数的常见用法是"接 ROS 节点、只要框"。这时阈值表缺一份
        不该让整条链路起不来——CLI 那边已经用退出码 2 明确报过错，
        这里是库调用，返回空表 + `color=None` 比抛异常更好用。
    """
    try:
        if config_path is None:
            # 没给配置：按仓库约定从当前目录找（与 CLI 的默认查找顺序一致）
            path = os.path.join("configs", DEFAULT_THRESHOLDS)
            if not os.path.isfile(path):
                return {}
        else:
            path = resolve_thresholds_path(cfg, config_path)
        return load_thresholds(path)
    except ConfigError:
        # 文件写坏了也按"没有阈值表"处理：判色族的 color 会是 None，
        # 调用方从返回值上立刻能看出来，不会静默拿到错颜色
        return {}


def detect(image_bgr, detector: str = "blob", config=None, **params) -> list:
    """对一帧 BGR 图跑一次检测（判色族顺带判色），返回检出物列表。

    用大白话说
        给一张图，告诉它用哪个检测器（`blob` 找色块 / `aruco` 找码 / `yolo` 找目标），
        它把图交给检测器，再对判色族顺手判出颜色，最后把结果列表还给你。
        没有目标时返回空列表，不会返回 None。

    :param image_bgr: HxWx3 的 uint8 BGR 图（**不是 RGB**，通道顺序写反不会报错，
        只会让判色结果全错）
    :param detector: 检测器名，`blob` / `stub_aruco` / `aruco` / `yolo`
    :param config: 配置文件路径（如 `configs/default.yaml`）；None 表示只用兜底参数
    :param params: 该检测器的参数覆盖，键名与 `configs/default.yaml` 的
        `detector.<name>` 段一致
    :return: `list[Detection]`（判色族）或 `list[ArucoMarker]`（ArUco）；无目标为 `[]`
    :raises ValueError: 图不是 HxWx3 的 BGR，或检测器名未知（报错里列出可用值）
    :raises RuntimeError: 选了 `aruco` 但 OpenCV 版本不支持新版 aruco API
    :raises ConfigError: 给了 `config` 但它读不出来 / 校验不过
    :raises FileNotFoundError: 检测器参数里的权重文件不存在（`yolo`）

    副作用
        读配置文件与阈值表（只读）；不开窗口、不写磁盘、不占用相机。

    例子
        >>> from up70_vision.api import detect
        >>> markers = detect(img_bgr, "aruco")              # 只找码
        >>> boxes = detect(img_bgr, "blob", min_area_px=800)  # 调最小面积
        >>> boxes = detect(img_bgr, config="configs/default.yaml")  # 用仓库配置判色
    """
    # ---- 1. 配置：给了文件就读文件，没给就用 schema 里的兜底默认值 ----
    if config is not None:
        cfg, config_path = load_config(config)
    else:
        cfg, config_path = from_dict({}), None

    # ---- 2. 检测器参数：同名检测器沿用配置值，换了检测器就取该名字的兜底值 ----
    name = str(detector).strip().lower()
    base_params = dict(cfg.detector.params if name == cfg.detector.name
                       else DETECTOR_DEFAULT_PARAMS.get(name, {}))
    base_params.update(params)          # 调用方写了的覆盖，没写的保留
    model = build_detector(name, base_params)

    # ---- 3. 判色参数与阈值表 ----
    thresholds = _resolve_thresholds(cfg, config_path)
    color_cfg = ColorConfig(min_saturation=cfg.color.min_saturation,
                            min_value=cfg.color.min_value,
                            hit_ratio_min=cfg.color.hit_ratio_min,
                            ignore_border_px=cfg.color.ignore_border_px,
                            min_valid_pixels=cfg.color.min_valid_pixels)

    # ---- 4. 走原链路：Frame -> process_frame（检测 + 判色）----
    # 造 Frame 而不是直接调 detector.detect()：判色族的颜色字段要靠管线填，
    # 绕过管线会得到一个"有框没颜色"的结果，与 CLI 不一致
    frame = Frame(image_bgr=image_bgr, index=0, source="api")
    return list(process_frame(frame, model, thresholds, color_cfg).detections)
