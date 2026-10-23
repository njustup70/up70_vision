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

import numpy as np

from ..pipeline import ColorConfig
from ..pose import METHOD_NAMES as POSE_METHODS
from ..logging import parse_level

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
    "aruco": {"dictionary": "DICT_4X4_50", "corner_refine": "subpix",
              "min_perimeter_rate": 0.03, "max_detections": 0},
    "yolo": {"weights": "assets/models/yolov8n.onnx", "conf": 0.25, "iou": 0.45,
             "imgsz": 640, "max_detections": 32, "classes": [], "class_names": [],
             "device": "cpu"},
}


@dataclass
class DetectorConfig:
    """检测器选择与它的参数。

    用大白话说
        配置里"用哪个检测器、它的参数怎么调"这一段。比如用假码（stub_aruco），
        就写要伪造几个码、边长多少、要不要动；用真识别的名字由任务一提供。

    属性
        name (str)    检测器名，必须是 `DETECTOR_DEFAULT_PARAMS` 的键之一
                      （当前 `blob` / `stub_aruco` / `aruco`）
        params (dict) 该检测器的参数，来自 `configs/default.yaml` 的 `detector.<name>` 段；
                      整段缺省时用 `DETECTOR_DEFAULT_PARAMS[name]` 兜底

    参数速查（只列会被 `validate()` 检查的项；其余交给检测器自己报错）
        ┌────────────┬──────────────┬─────────┬──────────────────────────────┐
        │ 检测器      │ 键            │ 类型    │ 合法值 / 默认                │
        ├────────────┼──────────────┼─────────┼──────────────────────────────┤
        │ blob       │ min_area_px   │ int     │ >= 0，默认 400               │
        │            │ morph_kernel  │ int     │ 正奇数，默认 5               │
        │            │ max_detections│ int     │ >= 1，默认 32                │
        ├────────────┼──────────────┼─────────┼──────────────────────────────┤
        │ stub_aruco │ ids           │ list[int]│ 非空，默认 [0, 1, 2]         │
        │            │ side_px       │ int     │ >= 1，默认 90                │
        │            │ moving        │ bool    │ 默认 true                    │
        │            │ mode          │ str     │ "normal" / "none"，默认 normal│
        │            │ empty_after   │ int     │ >= 0，默认 0（不启用）       │
        └────────────┴──────────────┴─────────┴──────────────────────────────┘

        yolo（ONNX 推理，实现见 `detectors/yolo.py`）
            weights         str    ONNX 权重路径，默认 assets/models/yolov8n.onnx
            conf            float  (0, 1)，默认 0.25
            iou             float  (0, 1]，默认 0.45
            imgsz           int    32 的正整数倍，默认 640
            max_detections  int    >= 1，默认 32
            classes         list[int]  只保留这些类别索引，默认 []（全部）
            class_names     list[str]  类名表，默认 []（此时尝试读 ONNX 元数据）
            device          str    目前只支持 cpu

        aruco（真 ArUco，实现见 `detectors/aruco.py`；这张表放不下，单列）
            dictionary          str    `DICT_` 开头，默认 DICT_4X4_50
            corner_refine       str    none / subpix / contour，默认 subpix
            min_perimeter_rate  float  (0, 1]，默认 0.03（占图像周长的比例）
            max_detections      int    >= 0，默认 0（不限）
    """

    name: str = "blob"
    params: dict = field(default_factory=lambda: dict(DETECTOR_DEFAULT_PARAMS["blob"]))

    def validate(self) -> None:
        """校验检测器名与它自己的参数；越界的值在这里就报 `ConfigError`。

        用大白话说
            这里是配置的"安检口"：名字写错、参数写成负数/奇数不合法，就在门口拦下，
            直接告诉你"哪个字段、错在哪、合法值是什么"，不用等程序跑起来才看到英文崩溃。
            比如 mode 写成 "NONEE"，这里会说"只能是 normal / none"，而不是运行到一半才奇怪。

        校验放在配置层而不是等检测器构造时再报，是为了让 `--dump-config` / 启动阶段
        就能给出带字段路径的中文提示（构造期才报往往只剩一句英文异常）。

        校验清单
            1. `name` 在工厂认识的名单里（不在 → 报错并列出全部可用值）
            2. 按 `name` 分支校验各自的参数（见类文档的"参数速查"表）
            3. 未列入的键不在这里拦：检测器构造时会报"未知参数"，
               保证新增参数时不用同时改两处（但也绝不静默忽略）
        """
        # ---- 名字：必须在工厂认识的名单里，报错时列出全部可用值 ----
        if self.name not in DETECTOR_DEFAULT_PARAMS:
            raise ConfigError(
                f"detector.name 必须是 {sorted(DETECTOR_DEFAULT_PARAMS)} 之一，收到 {self.name!r}")
        p = self.params
        # ---- 各检测器的参数：只校验已知项，未知项留给检测器自己报（不静默忽略）----
        if self.name == "blob":
            # 最小框面积：负值会让所有候选框都被过滤掉，表现为"什么都检测不到"
            if int(p.get("min_area_px", 400)) < 0:
                raise ConfigError(f"detector.blob.min_area_px 不能为负：{p.get('min_area_px')}")
            # 形态学核边长：必须正奇数，偶数/非正会让 cv2.morphologyEx 直接抛错
            k = int(p.get("morph_kernel", 5))
            if k <= 0 or k % 2 == 0:
                raise ConfigError(f"detector.blob.morph_kernel 必须是正奇数：{k}")
            # 单帧最多保留多少个框：0 或负数等于永远不输出任何框
            if int(p.get("max_detections", 32)) <= 0:
                raise ConfigError(f"detector.blob.max_detections 必须为正：{p.get('max_detections')}")
        elif self.name == "yolo":
            # conf / iou 越界不是"效果差一点"：一个让框永远出不来，一个让框全被 NMS 吃掉
            conf = float(p.get("conf", 0.25))
            if not 0.0 < conf < 1.0:
                raise ConfigError(f"detector.yolo.conf 必须在 (0, 1) 之间：{p.get('conf')}")
            iou = float(p.get("iou", 0.45))
            if not 0.0 < iou <= 1.0:
                raise ConfigError(f"detector.yolo.iou 必须在 (0, 1] 之间：{p.get('iou')}")
            # imgsz：ONNX 是固定输入尺寸，写错会在 forward 时才抛一句英文 cv2 异常
            imgsz = int(p.get("imgsz", 640))
            if imgsz <= 0 or imgsz % 32 != 0:
                raise ConfigError(f"detector.yolo.imgsz 必须是 32 的正整数倍：{p.get('imgsz')}")
            if int(p.get("max_detections", 32)) <= 0:
                raise ConfigError(f"detector.yolo.max_detections 必须为正：{p.get('max_detections')}")
            weights = p.get("weights", "")
            if not isinstance(weights, str) or not weights.strip():
                raise ConfigError(f"detector.yolo.weights 必须是非空路径字符串：{weights!r}")
            for key in ("classes", "class_names"):
                value = p.get(key, [])
                if not isinstance(value, (list, tuple)):
                    raise ConfigError(f"detector.yolo.{key} 必须是列表：{value!r}")
        elif self.name == "stub_aruco":
            # ids：伪造哪些码。空列表会让 stub 每帧返回 []，与"无码"混在一起没法区分，
            # 所以这里要求非空；元素必须是整数（True/False 是 int 子类，要排掉）
            ids = p.get("ids", [0, 1, 2])
            if not isinstance(ids, (list, tuple)) or len(ids) == 0:
                raise ConfigError(f"detector.stub_aruco.ids 必须是非空列表，收到 {ids!r}")
            for one in ids:
                if not isinstance(one, int) or isinstance(one, bool):
                    raise ConfigError(f"detector.stub_aruco.ids 只能是整数，收到 {one!r}")
            # side_px：码边长，非正会让四角重合、画不出码
            if int(p.get("side_px", 90)) <= 0:
                raise ConfigError(f"detector.stub_aruco.side_px 必须为正：{p.get('side_px')}")
            # mode / empty_after：任务单第 6 条"无码时正常运行"的定量输入源。
            #   不写进 DETECTOR_DEFAULT_PARAMS：默认就是 normal / 0，
            #   写进去只会让默认参数字典变长、还得多改一处断言。
            #   mode=none  → 每帧都返回 []（恒无码）
            #   empty_after>0 → 前 N 帧正常出码，第 N 帧起返回 []（演"码移出画面"）
            mode = str(p.get("mode", "normal")).strip().lower()
            if mode not in ("normal", "none"):
                raise ConfigError(
                    f"detector.stub_aruco.mode 只能是 'normal' / 'none'，收到 {p.get('mode')!r}")
            if "empty_after" in p and p["empty_after"] is not None:
                try:
                    empty_after = int(p["empty_after"])
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"detector.stub_aruco.empty_after 必须是整数，收到 {p['empty_after']!r}") from exc
                if empty_after < 0:
                    raise ConfigError(
                        f"detector.stub_aruco.empty_after 不能为负：{empty_after}")
        elif self.name == "aruco":
            dictionary = str(p.get("dictionary", "DICT_4X4_50")).strip()
            if not dictionary.upper().startswith("DICT_"):
                raise ConfigError(
                    f"detector.aruco.dictionary 需要 OpenCV 预定义字典名（形如 DICT_4X4_50）："
                    f"{p.get('dictionary')!r}")
            refine = str(p.get("corner_refine", "subpix")).strip().lower()
            if refine not in ("none", "subpix", "contour"):
                raise ConfigError(
                    f"detector.aruco.corner_refine 必须是 none/subpix/contour，收到 {refine!r}")
            rate = float(p.get("min_perimeter_rate", 0.03))
            if not 0.0 < rate <= 1.0:
                raise ConfigError(
                    f"detector.aruco.min_perimeter_rate 必须在 (0, 1]（占图像周长的比例）：{rate}")
            if int(p.get("max_detections", 0)) < 0:
                raise ConfigError(
                    f"detector.aruco.max_detections 不能为负（0 表示不限）：{p.get('max_detections')}")


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
class CalibrationConfig:
    """相机内参文件的位置（`configs/camera_calibration.yaml` 这类文件）。

    用大白话说
        位姿估计离不开"这颗镜头把像素换算成角度"的那套数（内参 + 畸变）。
        那套数由 `python -m vision calibrate` 标定后写成 YAML，**每台相机一份**，
        所以它在 `.gitignore` 里、不进版本库；这里只记"去哪找它"。

    属性
        file (str)     内参文件名；**相对路径以配置文件所在目录为基准**
                       （与 `color.thresholds_file` 同一个规矩）
        required (bool) 找不到或读不出来时是否直接报错。
                       True → 启动期就失败，报错里给标定命令；
                       False → 打一条警告，位姿功能关闭，其它功能照常跑
    """

    file: str = "camera_calibration.yaml"
    required: bool = False

    def validate(self) -> None:
        if not str(self.file).strip():
            raise ConfigError(
                "calibration.file 不能为空（位姿估计需要内参；标注定文件相对 configs/ 的路径）")


@dataclass
class PoseConfig:
    """ArUco 位姿估计的参数（`pose.py` 消费）。

    用大白话说
        这一组参数决定"要不要算位姿、码有多大、算出来的结果怎么判为不可信"。
        有一半是物理量（码边长），另一半是"什么情况下宁可报错也不给数"的门槛。

    参数速查
        ┌───────────────────┬─────────┬────────────────────────────┬──────────────┐
        │ 键                 │ 类型    │ 合法值 / 含义               │ 默认          │
        ├───────────────────┼─────────┼────────────────────────────┼──────────────┤
        │ enabled            │ bool    │ 是否输出位姿                │ false         │
        │ marker_size_mm     │ float   │ > 0，码黑框外沿边长（毫米）  │ 23.0          │
        │ method             │ str     │ ippe_square / iterative     │ ippe_square   │
        │ refine             │ bool    │ 是否做 LM 精化               │ true          │
        │ ambiguity_ratio    │ float   │ > 1，两解 RMS 比小于它算难分 │ 3.0           │
        │ ambiguity_deg      │ float   │ 0–180，两解夹角超过它算难分  │ 10.0          │
        │ max_reproj_rms_px  │ float   │ > 0，超过直接抛错            │ 2.0           │
        └───────────────────┴─────────┴────────────────────────────┴──────────────┘

    两条容易踩的
        · `marker_size_mm` **必须用卡尺量黑框外沿**，不含白边。填错不会让重投影误差变大，
          只会让所有距离按同比例错（progress.md 坑 40 的位姿版）。默认值 23.0 是**本队
          那批码的实量值**（progress.md 坑 40：打印标称 15 mm 实际 23 mm），不是通用值——
          换了码就必须重量，别把它当默认安全值。
        · `enabled` 默认 false：不开位姿时，实时入口的输出与改造前**逐字节一致**，
          这样"加了功能"不会变成"改了别人的日志格式"。
    """

    enabled: bool = False
    # 默认值取本队实量的 23 mm，而不是"看起来合理"的 40：40 会让距离整体偏 +74%
    # 且重投影误差完全正常（坑 40 的位姿版）。换码必须重量。
    marker_size_mm: float = 23.0
    method: str = "ippe_square"
    refine: bool = True
    ambiguity_ratio: float = 3.0
    ambiguity_deg: float = 10.0
    max_reproj_rms_px: float = 2.0

    def validate(self) -> None:
        if not np.isfinite(self.marker_size_mm) or float(self.marker_size_mm) <= 0:
            raise ConfigError(
                f"pose.marker_size_mm 必须是正数（毫米，量黑框外沿）：{self.marker_size_mm!r}")
        if self.method not in POSE_METHODS:
            raise ConfigError(
                f"pose.method 必须是 {list(POSE_METHODS)} 之一，收到 {self.method!r}")
        if not np.isfinite(self.ambiguity_ratio) or float(self.ambiguity_ratio) <= 1.0:
            raise ConfigError(
                f"pose.ambiguity_ratio 必须大于 1（它是「次优/最优」的倍数）：{self.ambiguity_ratio!r}")
        if not 0.0 <= float(self.ambiguity_deg) <= 180.0:
            raise ConfigError(f"pose.ambiguity_deg 必须在 0–180：{self.ambiguity_deg!r}")
        if not np.isfinite(self.max_reproj_rms_px) or float(self.max_reproj_rms_px) <= 0:
            raise ConfigError(
                f"pose.max_reproj_rms_px 必须为正（像素）：{self.max_reproj_rms_px!r}")


@dataclass
class AppConfig:
    """整份配置。字段与 configs/default.yaml 的层级一一对应。"""

    input: InputConfig = field(default_factory=InputConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    color_thresholds_file: str = "color_thresholds.yaml"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    log: LogConfig = field(default_factory=LogConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)

    def validate(self) -> "AppConfig":
        self.input.validate()
        try:                       # pipeline 的 ColorConfig 抛 ValueError，这里统一成 ConfigError
            self.color.validate()
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if not str(self.color_thresholds_file).strip():
            raise ConfigError("color.thresholds_file 不能为空（阈值表是判色的唯一参数来源）")
        self.detector.validate()
        self.output.validate()
        self.log.validate()
        self.calibration.validate()
        self.pose.validate()
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
    cal = data.get("calibration") or {}
    pose = data.get("pose") or {}

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
        calibration=CalibrationConfig(
            file=str(cal.get("file", "camera_calibration.yaml")),
            required=bool(cal.get("required", False)),
        ),
        pose=PoseConfig(
            enabled=bool(pose.get("enabled", False)),
            marker_size_mm=float(pose.get("marker_size_mm", 23.0)),
            method=str(pose.get("method", "ippe_square")),
            refine=bool(pose.get("refine", True)),
            ambiguity_ratio=float(pose.get("ambiguity_ratio", 3.0)),
            ambiguity_deg=float(pose.get("ambiguity_deg", 10.0)),
            max_reproj_rms_px=float(pose.get("max_reproj_rms_px", 2.0)),
        ),
    )
    return cfg.validate()
