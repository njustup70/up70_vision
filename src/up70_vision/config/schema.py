"""配置结构定义与范围校验。

设计取舍
    用 dataclass + 显式校验函数，**不引 pydantic**：校验规则写在明面上、报错信息自己拼，
    队友读代码就知道为什么某个值非法，不需要先学一个框架。

不放什么
    不放业务默认值——默认值一律写在 configs/*.yaml 里，本模块只定义结构与取值范围。

字段路径报错
    ConfigError 的消息里带上字段路径（如 color.hit_ratio_min）与可用值/合法区间，
    现场改配置时不用猜。
"""

from dataclasses import dataclass, field

from ..core.pipeline import ColorConfig
from ..utils.logging import parse_level

SOURCES = ("synthetic", "replay", "camera")
REPLAY_KINDS = ("auto", "image_dir", "video")


class ConfigError(ValueError):
    """配置非法。消息里应包含字段路径与合法取值。"""


@dataclass
class SyntheticConfig:
    width: int = 640
    height: int = 480
    num_frames: int = 3
    blocks_per_frame: int = 6
    seed: int = 2027
    noise_sigma: float = 8.0
    background_bgr: list = field(default_factory=lambda: [30, 30, 30])

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ConfigError(f"input.synthetic.width/height 必须为正：{self.width}x{self.height}")
        if self.num_frames < 0:
            raise ConfigError(f"input.synthetic.num_frames 不能为负：{self.num_frames}")
        if not 1 <= self.blocks_per_frame <= 6:
            raise ConfigError(
                f"input.synthetic.blocks_per_frame 必须在 1~6（内置颜色数上限）：{self.blocks_per_frame}")
        if self.noise_sigma < 0:
            raise ConfigError(f"input.synthetic.noise_sigma 不能为负：{self.noise_sigma}")
        if len(self.background_bgr) != 3 or any(not 0 <= int(v) <= 255 for v in self.background_bgr):
            raise ConfigError(
                f"input.synthetic.background_bgr 需要 3 个 0~255 的分量：{self.background_bgr}")


@dataclass
class ReplayConfig:
    path: str = "data/clips"
    kind: str = "auto"
    frame_stride: int = 1
    max_frames: int = 0

    def validate(self) -> None:
        if not str(self.path).strip():
            raise ConfigError("input.replay.path 不能为空")
        if self.kind not in REPLAY_KINDS:
            raise ConfigError(f"input.replay.kind 必须是 {list(REPLAY_KINDS)}，收到 {self.kind!r}")
        if self.frame_stride < 1:
            raise ConfigError(f"input.replay.frame_stride 至少为 1：{self.frame_stride}")
        if self.max_frames < 0:
            raise ConfigError(f"input.replay.max_frames 不能为负（0 表示不限）：{self.max_frames}")


@dataclass
class CameraConfig:
    """真相机参数（海康 MV-CU013-A0UC，走 MVS SDK）。

    曝光/增益在这里给默认值，因为相机与场地强耦合：本机实测 80 ms + 10 dB 才够
    看清桌面；现场必须重调，而一味拉长曝光会掉帧率并在移动目标上留拖影。
    """

    index: int = 0
    exposure_us: float = 80000.0
    gain_db: float = 10.0
    timeout_ms: int = 1000
    max_frames: int = 0

    def validate(self) -> None:
        if self.index < 0:
            raise ConfigError(f"input.camera.index 不能为负：{self.index}")
        if self.exposure_us < 0:
            raise ConfigError(
                f"input.camera.exposure_us 不能为负（0 表示不改动相机当前值）：{self.exposure_us}")
        if self.gain_db < 0:
            raise ConfigError(f"input.camera.gain_db 不能为负：{self.gain_db}")
        if self.timeout_ms <= 0:
            raise ConfigError(f"input.camera.timeout_ms 必须为正：{self.timeout_ms}")
        if self.max_frames < 0:
            raise ConfigError(f"input.camera.max_frames 不能为负（0 表示不限）：{self.max_frames}")


@dataclass
class InputConfig:
    source: str = "synthetic"
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)

    def validate(self) -> None:
        if self.source not in SOURCES:
            raise ConfigError(f"input.source 必须是 {list(SOURCES)}，收到 {self.source!r}")
        self.synthetic.validate()
        self.replay.validate()
        self.camera.validate()


# 每个检测器的默认参数（configs/default.yaml 里也写了一份，这里只是兜底）
DETECTOR_DEFAULT_PARAMS = {
    "blob": {"min_area_px": 400, "morph_kernel": 5, "max_detections": 32},
    "stub_aruco": {"ids": [0, 1, 2], "side_px": 90, "moving": True},
}


@dataclass
class DetectorConfig:
    name: str = "blob"
    params: dict = field(default_factory=lambda: dict(DETECTOR_DEFAULT_PARAMS["blob"]))

    def validate(self) -> None:
        if self.name not in DETECTOR_DEFAULT_PARAMS:
            raise ConfigError(
                f"detector.name 必须是 {sorted(DETECTOR_DEFAULT_PARAMS)} 之一，收到 {self.name!r}")
        p = self.params
        if self.name == "blob":
            if int(p.get("min_area_px", 400)) < 0:
                raise ConfigError(f"detector.blob.min_area_px 不能为负：{p.get('min_area_px')}")
            k = int(p.get("morph_kernel", 5))
            if k <= 0 or k % 2 == 0:
                raise ConfigError(f"detector.blob.morph_kernel 必须是正奇数：{k}")
            if int(p.get("max_detections", 32)) <= 0:
                raise ConfigError(f"detector.blob.max_detections 必须为正：{p.get('max_detections')}")
        elif self.name == "stub_aruco":
            ids = p.get("ids", [0, 1, 2])
            if not isinstance(ids, (list, tuple)) or len(ids) == 0:
                raise ConfigError(f"detector.stub_aruco.ids 必须是非空列表，收到 {ids!r}")
            for one in ids:
                if not isinstance(one, int) or isinstance(one, bool):
                    raise ConfigError(f"detector.stub_aruco.ids 只能是整数，收到 {one!r}")
            if int(p.get("side_px", 90)) <= 0:
                raise ConfigError(f"detector.stub_aruco.side_px 必须为正：{p.get('side_px')}")


@dataclass
class OutputConfig:
    enabled: bool = True
    dir: str = "output/demo"
    save_overlay: bool = True
    save_json: bool = True
    save_csv: bool = True

    def validate(self) -> None:
        if self.enabled and not str(self.dir).strip():
            raise ConfigError("output.dir 不能为空（output.enabled=true 时必须给目录）")


@dataclass
class LogConfig:
    level: str = "INFO"

    def validate(self) -> None:
        try:
            parse_level(self.level)
        except ValueError as exc:
            raise ConfigError(f"log.level 非法：{exc}") from exc


@dataclass
class AppConfig:
    """整份配置。字段与 configs/default.yaml 的层级一一对应。"""

    input: InputConfig = field(default_factory=InputConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    color_thresholds_file: str = "color_thresholds.yaml"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def validate(self) -> "AppConfig":
        self.input.validate()
        try:                       # core 层的 ColorConfig 抛 ValueError，这里统一成 ConfigError
            self.color.validate()
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if not str(self.color_thresholds_file).strip():
            raise ConfigError("color.thresholds_file 不能为空（阈值表是判色的唯一参数来源）")
        self.detector.validate()
        self.output.validate()
        self.log.validate()
        return self


def from_dict(data: dict) -> AppConfig:
    """把 YAML 读出来的 dict 转成 AppConfig，并逐字段校验。

    未知键会被忽略（便于配置文件先行、实现后补），但**已知键的取值一定校验**。
    """
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置根必须是 dict，收到 {type(data).__name__}")

    inp = data.get("input") or {}
    syn = inp.get("synthetic") or {}
    rep = inp.get("replay") or {}
    cam = inp.get("camera") or {}
    col = data.get("color") or {}
    det = data.get("detector") or {}
    out = data.get("output") or {}
    log_cfg = data.get("log") or {}

    det_name = str(det.get("name", "blob"))
    det_defaults = dict(DETECTOR_DEFAULT_PARAMS.get(det_name, DETECTOR_DEFAULT_PARAMS["blob"]))
    det_params = det.get(det_name) or det.get("params") or det_defaults

    cfg = AppConfig(
        input=InputConfig(
            source=str(inp.get("source", "synthetic")),
            synthetic=SyntheticConfig(
                width=int(syn.get("width", 640)),
                height=int(syn.get("height", 480)),
                num_frames=int(syn.get("num_frames", 3)),
                blocks_per_frame=int(syn.get("blocks_per_frame", 6)),
                seed=int(syn.get("seed", 2027)),
                noise_sigma=float(syn.get("noise_sigma", 8.0)),
                background_bgr=list(syn.get("background_bgr", [30, 30, 30])),
            ),
            replay=ReplayConfig(
                path=str(rep.get("path", "data/clips")),
                kind=str(rep.get("kind", "auto")),
                frame_stride=int(rep.get("frame_stride", 1)),
                max_frames=int(rep.get("max_frames", 0)),
            ),
            camera=CameraConfig(
                index=int(cam.get("index", 0)),
                exposure_us=float(cam.get("exposure_us", 80000.0)),
                gain_db=float(cam.get("gain_db", 10.0)),
                timeout_ms=int(cam.get("timeout_ms", 1000)),
                max_frames=int(cam.get("max_frames", 0)),
            ),
        ),
        color=ColorConfig(
            min_saturation=int(col.get("min_saturation", 40)),
            min_value=int(col.get("min_value", 40)),
            hit_ratio_min=float(col.get("hit_ratio_min", 0.50)),
            ignore_border_px=int(col.get("ignore_border_px", 0)),
            min_valid_pixels=int(col.get("min_valid_pixels", 25)),
        ),
        color_thresholds_file=str(col.get("thresholds_file", "color_thresholds.yaml")),
        detector=DetectorConfig(name=det_name, params=dict(det_params)),
        output=OutputConfig(
            enabled=bool(out.get("enabled", True)),
            dir=str(out.get("dir", "output/demo")),
            save_overlay=bool(out.get("save_overlay", True)),
            save_json=bool(out.get("save_json", True)),
            save_csv=bool(out.get("save_csv", True)),
        ),
        log=LogConfig(level=str(log_cfg.get("level", "INFO"))),
    )
    return cfg.validate()
