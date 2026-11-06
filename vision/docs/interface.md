# 接口契约

> **收录标准**：改它会不会牵连对方的写法。会，写进来；不会（画成什么样、日志打几行、命令行长什么样），不写。
> **改这份文件的流程**：PR 描述里写一句「**动了契约，动的是哪一条**」，等对方同意再合（《视觉组协作规范》第 2 条）。
> **别拿 ✅ 当「main 上已有」**：标 ✅ 的实现目前在**尚未合并**的分支 `feat/判色管线与相机接入` 上；那个分支合并前，`main` 的 `src/` 下只有各层的 `__init__.py`。核对日期 2026-10-16。

---

## 1. 四条不会报错、只会让结果不对的约定

| # | 约定 |
|---|---|
| 1 | 图像一律 **BGR**，变量名带颜色空间后缀（`image_bgr` / `frame_bgr`） |
| 2 | 框用绝对像素坐标、**半开上界**：`(x1,y1,x2,y2)` 覆盖 `x1..x2-1`，宽 = `x2-x1` |
| 3 | HSV 的 H 是 **0–179**（不是 0–360）；红色两段阈值取并集 |
| 4 | 画到图上的文字**只能是 ASCII**（`viz.draw.put_text` 会把汉字替换成 `?`，不报错） |

---

## 2. 数据结构：`types.py`

### 2.1 `Frame` ✅

所有输入源的统一产出，也是 `core` 与 `io` 之间唯一的传递结构。

```python
BBox = tuple[int, int, int, int]        # 半开上界

@dataclass(frozen=True)
class Frame:
    image_bgr: np.ndarray     # (H, W, 3) uint8，BGR
    index: int = 0            # 帧序号，从 0 开始递增
    source: str = ""          # "synthetic" / "replay:<文件名>" / "camera:mvs:<型号>"
    stamp: float = 0.0        # 采集时刻（秒）；合成与回放填 0.0
    ground_truth: tuple = ()  # 仅合成源提供：((color_name, bbox_xyxy), ...)
```

- `__post_init__` 只校验 `ndim == 3 且 shape[2] == 3`；**不校验 dtype、不校验通道顺序**。
- 属性：`height`、`width`、`size`（返回 `(宽, 高)`）、`has_ground_truth()`。

### 2.2 `ArucoMarker` ✅（任务二新增）

```python
@dataclass(frozen=True)
class ArucoMarker:
    marker_id: int          # 字典 DICT_4X4_50 的编号，取值 0–49
    corners_px: np.ndarray  # (4, 2)，dtype **float32**，单位像素
    center_px: tuple        # (float, float)，单位像素
```

**为什么不复用判色族的 `Detection`**：`Detection` 的核心是 `bbox_xyxy`（轴对齐矩形），它的中心是 bbox 中心；ArUco 的本质是四角点构成的任意四边形，中心是**四点算术平均**。码一倾斜，两者就不是同一个位置。

**四条必须一致的约定**（stub 与真实现最容易分歧的四处）

1. **无码返回 `[]`**，不返回 `None`、不返回 `[None]`。`detectMarkers` 无码时 `ids` 是 `None`，必须在检测器内部归一化，不要往上透传。
2. **按 `marker_id` 升序排列**。不排序时同一场景的终端输出会抖动，看起来像 bug，也没法 diff。
3. **`center_px` 是四角点算术平均**，不是 bbox 中心。
4. **`corners_px` 顺时针，起点是「码自身坐标系」的左上角**，直接沿用 OpenCV 原生顺序。

`corners_px` 存 `float32` 不存 `int`：它是阶段二 `solvePnP` 的输入，这里取整等于提前丢掉亚像素精度且不可逆。取整只发生在绘制那一刻。

### 2.3 `MarkerPose` ✅（阶段二新增）

```python
@dataclass(frozen=True)
class MarkerPose:
    marker_id: int
    rvec: np.ndarray        # (3,) float64，Rodrigues 旋转向量
    tvec_mm: np.ndarray     # (3,) float64，码中心在相机系下的坐标；[2] 恒 > 0
    distance_mm: float      # ||tvec_mm||（斜距，不是 tvec[2]）
    tilt_deg: float         # acos(-R[2,2])，0 = 正对相机
    yaw_deg: float          # 码 x 轴在图像平面内的方向角，(−180, 180]，顺时针为正
    reproj_rms_px: float    # 四角重投影 RMS
    ambiguous: bool = False # 两解难分（只做诊断）
```

**四条必须一致的约定**

1. **坐标口径**：码坐标系原点在码中心、`z=0` 为码面；模型点顺序是
   `(-L/2,+L/2) (+L/2,+L/2) (+L/2,-L/2) (-L/2,-L/2)`，依次对应图像的 TL/TR/BR/BL。
   这是 `SOLVEPNP_IPPE_SQUARE` 的硬性要求，写错不报错、只给垃圾解（实测 RMS 993 px）。
2. **单位毫米**：`marker_size_mm` 进、`tvec_mm` / `distance_mm` 出，不做 mm/m 隐式换算。
3. **硬校验**：`reproj_rms_px > max_reproj_rms_px` 或 `tvec_mm[2] <= 0` 一律抛
   `ValueError`——这两种情况都是"输出看起来正常、其实全错"。
4. **`ambiguous` 只做诊断**：平面位姿天然两解（夹角恰为 `2×tilt`、距离差 0.1–0.5%），
   标 True 不代表结果不可用。

> `corners_px` 存 float32（亚像素，见 2.2），`rvec`/`tvec_mm` 存 float64：前者是检测产物，
> 后者要参与毫米级比较，中途丢精度没有意义。

### 2.4 任务二用不到的（登记，别乱动）

| 名字 | 说明 |
|---|---|
| `Detection` ✅ | 判色族的产出。中心是 **bbox 中心**。字段见 `types.py`；`yolo` 也复用它（`class_name` 填模型类名） |
| `iou` / `clip_bbox` / `bbox_from_xywh` ✅ | 共用纯函数，见 `types.py`（`bbox_from_xywh` 建 stub 场景时会用到） |

---

## 3. 检测器接口：`aruco/` 与 `yolo/`

### 3.1 `Detector` ✅

```python
class Detector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, image_bgr) -> list: ...
```

- 形参名固定 **`image_bgr`**：关键字调用（`detect(image_bgr=x)`）是允许的，改名会让对方的调用直接 `TypeError`。
- 工厂 `build_detector(name, params)`：未知名字抛 `ValueError`，报错里列出可用值。
- `detect` **只收图**：不收帧号、不收 `Frame`、不读文件、不开窗口。

**已登记的 `detector.name` 取值**（工厂与 `DETECTOR_NAMES` 必须一致）：

| 取值 | 产出 | `class_name` / `class_id` |
|---|---|---|
| `blob` | `list[Detection]` | `"unknown"` / `-1`（颜色由 `yolo/color_check.py` 后续填） |
| `aruco` / `stub_aruco` | `list[ArucoMarker]` | —（码没有颜色） |
| `yolo` ✅（阶段二新增） | `list[Detection]` | **模型自己的类名与索引**（如 `"person"` / `0`） |

`yolo` 是唯一会填**真实类名**的检测器：它不经过判色，`color` 留空。
其余字段与 `blob` 同构（`bbox_xyxy` 绝对像素、半开上界，无目标返回 `[]`），
所以下游 `viz` / `cli` 不需要知道用的是哪个检测器。

### 3.2 `ArucoDetector` ✅（任务二）

同名同签名的两个实现：`aruco/detector.py`（真，搭档交付）、`aruco/stub.py`（假，脚本化场景）。

```python
def detect(self, image_bgr) -> list[ArucoMarker]
```

**stub 的使用条件（只对 stub 生效）**

> **调用方每帧必须恰好调用一次 `detect`。**

理由：stub 用**自身的帧计数器**决定当前返回哪个场景（签名里没有帧号）。隔帧调用或一帧调两次都会让场景错位。**违反它的症状是「画面显示的场景不对」，不是报错** —— 记住这条，否则很难往计数漂移上想。真实现无状态，不受此约束。

---

### 3.3 位姿接口（`aruco/pose.py`，**不是** `Detector`）

位姿不实现 `Detector`：它不吃图、吃的是检测结果，产出的是 `MarkerPose`。

```python
def estimate_pose(marker, marker_size_mm, camera_matrix, dist_coeffs=None,
                  method="ippe_square", refine=True,
                  max_reproj_rms_px=2.0,
                  ambiguity_ratio=3.0, ambiguity_deg=10.0) -> MarkerPose
def estimate_poses(markers, marker_size_mm, camera_matrix, ...) -> list[MarkerPose]
```

- **`dist_coeffs` 必须和角点的状态配对**：角点已 `undistort_points()` → 传 `None`；
  角点未去畸变（实时入口就是这么做的）→ 传标定畸变系数。混用等于去畸变两次，
  码偏到画面边缘时距离实测偏 14.6%。
- 空输入返回 `[]`；输出顺序与入参一致，这里**不排序**（排序是检测器契约的一部分）。
- 失败一律抛 `ValueError`（RMS 超限、解到相机后方、入参形状不对），不返回"凑合的解"。
  要"跳过坏样本"由调用方 try/except 决定——批量里静默丢样本会让"10 个只出 8 个"看着正常。

---

## 4. 输入源接口：`sources/`

### 4.1 `FrameSource` ✅

```python
class FrameSource(ABC):
    name: str = "base"

    @abstractmethod
    def read(self) -> Frame | None: ...

    def close(self) -> None: ...     # 默认空实现，幂等
```

- `read()` 返回 `None` = **这路输入不会再有帧了**，调用方一律当结束。
- `read()` 抛异常 = 配置或环境错误（路径不存在、SDK 缺失），**不要吞掉**。
- 支持 `with` 与 `for`。
- 工厂 `build_source(source_type, synthetic_params, replay_params, camera_params, colors)`：参数是普通 dict，由 `cli/run.py` 从配置拆出来 —— **`sources/` 不 import `config`**。

### 4.2 三个源的 `None` 语义不同（最常踩的一条）

| 源 | `read()` 的 `None` 表示 | 构造期抛的错 |
|---|---|---|
| `synthetic` | 读完 `num_frames` 帧（正常结束） | 颜色名不认识、参数非法 |
| `replay` | 素材读完（正常结束） | 路径不存在、视频打不开、目录无图 |
| `camera` | 取帧超时、相机被拔、或到 `max_frames` | 枚举不到设备、打不开（被占用）、SDK 缺失 |

**图片源读到末尾返回 `None` 是「结束」，相机源返回 `None` 是「故障」**，循环对两者的处理不一样（前者退出，后者重试或报错）。见第 6 节第 1 条。

---

## 5. 其余约定（用到再查）

- **依赖方向**：`config`/`image_io`/`logging` → 根级算法模块 → `io` → `viz` → `cli` → `__main__.py`。根级算法模块不 import `io`/`viz`/`config`，不读文件、不开窗口；逻辑写在 `cli/`（参数解析 / 批处理 / 实时循环 / 终端输出四个文件）。
- **`viz/` 的规矩**：画**检测结果**（框、标签、中心、角点、ID）→ 返回新图（照 `draw_detections` 的写法）；画 **HUD**（帧号、汇总行）→ 可以原地改。任务二加角点绘制走前者。
- **日志**：取 logger 一律 `get_logger("<模块路径>")`（如 `get_logger("viz.draw")`）。自己用 `logging.getLogger` 拼名字会让这条链路断掉，症状是"这个模块的日志不出现"。
- **读写图片**：一律走 `sources/__init__.py`（读）与 `cli/output.py`（写）（`read_image_bgr` / `write_image`），**不要直接调 `cv2.imread` / `cv2.imwrite`** —— 这两个失败时返回 `None` / `False` 而不抛异常。
- **配置**：`configs/default.yaml` 的字段名是契约（改名会让对方的 `--set` 静默失效）。改 `input.source` 的取值集合要同步 `sources/__init__.py` 的 `SOURCE_NAMES`。
- **平台**：`sources/camera.py` 只在原生跑（容器需装 MVS 的 deb + USB 透传），PR 里附原生日志与截图；其余全部在容器里验收。

---

## 6. 欠账（要改先问人，不是待办）

**一、`read()` 的 `None` 分不出「正常结束」与「故障」。** ⬜

`camera` 源把「取帧超时 / 相机被拔 / 达到 `max_frames`」统一表示成 `None`，且**一次取帧超时就结束整路**。偶发丢帧（自动曝光切换、USB 带宽抖动）会让程序在正常使用中随机退出。这是任务二 S5 的正面验收对象，改它**属于动契约**，要两人同意。

**二、`sources/camera.py` 半初始化时泄漏设备句柄。** ⬜

`MV_CC_OpenDevice` 成功后、`MV_CC_StartGrabbing` 失败时构造函数抛异常，此后没人调 `close()`，相机被占住，下次打开必失败。

**三、`io/__init__.py` 的文档与第 5 节矛盾。** ⬜

那行写着「依赖 `types` 与 **config**」，而 `sources/__init__.py` 与第 5 节都明确「`sources/` 不 import `config`」。以 `base.py` 为准，`__init__.py` 那句是笔误，提给对方改。

---

## 附：变更记录

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-10-16 | 建文件 | 契约以已落地的实现为基线，不以计划书里的设想为基线 |
| 2026-10-16 | 复核：补 `utils` 与 `viz` 的规矩，标注 `input.camera` 尚未进 main | 逐文件对照源码 |
| 2026-10-16 | 精简：9 节压到 6 节；判色族内容收进 2.3 一张表；配置字段、依赖方向、运行平台各压成一两句 | 当前只有 ArUco 部分是待建的，其余是已实现的事实登记；将来要用的等用得到时再展开 |
| 2026-10-16 | 2.2 / 3.2 由 ⬜ 改 ✅：ArUco 侧已按本文落地（`aruco.py` 返回 `list[ArucoMarker]`、`aruco/stub.py` 同名同签名） | 实现方在分支 `feat/ArUco检测核心`；**契约条目本身没有改动** |
| 2026-10-18 | 新增 2.3 `MarkerPose` 与 3.3 位姿接口；原 2.3 顺延为 2.4 | 任务单第 2 条落地，`aruco/pose.py` + `tests/test_pose.py` |
| 2026-10-18 | 3.3 的签名补 `ambiguity_ratio` / `ambiguity_deg`；2.3 的 `yaw_deg` 区间笔误 `(180, 180]` 改为 `(−180, 180]` | 复核意见 P2-1（字段原本是死的）与 P3-4（笔误） |

## 深度抓取输出：光学坐标系（2026-10-27）

`/grasp_targets` 与 `/grasp_markers` 沿用深度图 `header.frame_id` 和采集时间；
深度图与 CameraInfo 的 frame 必须一致。`output_frame` 默认为空（沿用输入），
非空时必须与输入一致，否则跳过该帧；节点不做 TF 变换。位置单位米，
局部 Z 轴对准朝相机的顶面法向，绕该轴的转角是构造值，并非物块真实 yaw。
这是 ROS 接口契约变更，合并前需按协作规范获得对方同意。

## RGB-D 可见平面候选输出（2026-11-02）

实时侧面查看器发布 `/rgbd_side/candidates`（`vision_interfaces/msg/RgbdCandidateArray`，2026-11-04 扩展）和
`/rgbd_side/markers`（`visualization_msgs/MarkerArray`），QoS 均为可靠、易失、深度 1。
每个新完成的有效图对发布一次，空数组表示该帧没有有效候选；过期、几何重置或处理错误
在存在活动候选时额外发布空数组。断流检查采用本地单调时钟，从图对较早的接收时间计算，
默认 500 ms；撤销沿用上次采集 header。订阅者还必须自行检查时间戳和断流，不能无限使用旧消息。

数组字段为 `header` 与 `candidates[]`。每项 `RgbdCandidate` 包含以下字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `detection` | `vision_msgs/msg/Detection3D` | 原有完整候选，位置、法向、类名与置信度语义沿用下文 |
| `class_id` | `int32` | YOLO 原始类别索引；当前 mix4：`0=block`（350 mm 方块）、`1=top`（200 mm 塔顶），未知为 -1；不是根据深度尺寸重新裁决的类别 |
| `color_id` | `uint8` | HSV 颜色编号；未知或未映射为 0，不影响该项几何候选的发布 |

颜色编号固定在 `RgbdCandidate.msg` 的 `COLOR_*` 常量中，不随阈值文件键顺序改变：

| 编号 | 颜色 |
|---|---|
| 0 | 未知 `unknown` |
| 1 | 红 `red` |
| 2 | 蓝 `blue` |
| 3 | 红蓝横放混色 `mixed` |

颜色表中其他颜色即使被判出，也作为未知（0）发布。

判色复用 `vision.yolo.pipeline.process_frame` 和 `configs/color_thresholds.yaml` 的颜色范围及门槛，
在 YOLO 使用的同一张校正彩色图、同一框中完成，不从 RViz 的标记色或深度伪彩图取色。
`block` 使用普通 HSV 判色；`top` 按既有口径：红蓝上下时取上方颜色，左右时为 mixed，
单色时返回该色，斜放不明确时未知。`top` 也执行有效像素数门槛，`hit_ratio_min` 用于红蓝合计命中率，
方向计算使用相同有效像素掩码及裁边。颜色仍依赖光照和检测框，未完成真机准确率验收。
查看器参数 `color_thresholds_file` 默认 `configs/color_thresholds.yaml`，相对仓库根目录；
快照的 detection 保留 `color`、`color_ratio`、`valid_pixels`，运行摘要保留阈值及其 SHA-256。
现有 `timing_ms.inference` / 摘要 `inference_ms` 计时包含检测与 HSV 判色，窗口显示为 `Detect+color`。

**订阅迁移：** 此次将原 `vision_msgs/msg/Detection3DArray` 改为自定义类型；同名话题的旧类型
不能继续订阅。收发两侧都需取得并构建 `vision_interfaces`，重新加载工作空间并重启节点。
构建查看器用 `python -m colcon build --packages-up-to vision_depth --symlink-install`，
只构建接口的下游用 `--packages-select vision_interfaces`。原 `msg.detections[i]` 改为
`msg.candidates[i].detection`；类别编号和颜色编号分别取 `msg.candidates[i].class_id`、
`msg.candidates[i].color_id`。其他使用 Detection3DArray 的 `/grasp_targets` 和
`/ground_truth_targets` 不变。此契约变更合入前按团队规范评审。

数组与各 detection 的 header 使用配对深度图的采集时间与光学 frame。`detection.results[0]` 的类别和
置信度来自 YOLO，位置为选中平面可见内点重心（米）。`id` 为空，没有跨帧身份；
`detection.results[0].pose.pose.orientation` 将局部 +Z 旋转到单位法向，法向朝相机；绕法向的转角是
构造值，不是物块真实姿态或 yaw。协方差未估计，bbox 中心复用重心、朝向为单位占位且尺寸为零，
不能解释成零误差、真实包围盒或尺寸。法向缺失、非有限或模长不大于 1e-9 的候选整项不发布。
接收端按 ROS 四元数 `(x,y,z,w)` 计算法向：
`(2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y))`。这是原四元数占位字段的语义变更，
现有订阅方必须同步更新；合入前按团队规范评审。
标记有有限生命周期。它们只是未验证候选，不是物体几何中心或抓取指令；空间对齐及绝对精度
尚未验收。此新话题契约进入团队集成前按协作规范评审。

### 主逻辑只读适配器（2026-11-06）

迁入 `driverLogic_2027` 的模块位于 `integration/driverlogic/MainLogic/adapters/rgbd.py`，
使用 `RgbdAdapter(node, topic='/rgbd_side/candidates', max_age_s=0.5)` 挂到已经初始化且由
主框架 spin 的节点。构造时创建可靠、易失、深度 1 的订阅；不初始化 ROS 上下文，不执行动作。
`snapshot()` 返回不可变 `RgbdSnapshot(stamp_ns, frame_id, candidates, age_s, status)`；
每项候选含 `class_id/color_id/class_name/score/position_m/normal`，位置米，法向光学系单位向量。
锁保护 ROS 回调与业务线程；ROS 消息不会以可变引用暴露给业务。

空数组立即替换为无候选。自上次本地接收起超过 `max_age_s`，再次读取返回 `stale` 和空候选；
`age_s` 不等于采集后总年龄，`stamp_ns` 与 frame 仍保留原采集语义。其他状态为
`waiting/active/empty/invalid/closed`，只有 active 有候选；坏候选被跳过并计数，损坏 header
撤销整帧，`close()` 撤销缓存并移除订阅。调用方每次决策前重新读取，不无限持有旧快照。
该新增 Python 接入界面不改变上述 ROS 消息和话题契约。部署和只读入口见
[交接手册](../../integration/driverlogic/README.md)。

### ROS 图像解码与离线报告读取（2026-11-02）

ROS 包内 `vision_depth.image_decode.image_array` 读取 `16UC1`/`mono16` 毫米深度和 `bgr8`/`rgb8` 彩色，处理行填充与深度字节序，拒绝非法尺寸、行跨度和数据不足。纯深度节点解码失败仍记录并跳帧；查看器对不支持的编码仍返回空值，对损坏数据抛错；融合链仍按原错误处理。旧 `fuse_node.image_to_depth_array`、`side_pipeline.image_array` 和查看器转换函数保留。相机 QoS 共享 `vision_depth.sensor_qos.sensor_qos`，各订阅与发布队列深度不变。

离线报告公共读取入口为 `lab.rgbd_check.report_reader.load_report`，原 `lab.manual_rgbd_fusion.load_report` 兼容；字段、校验及输出不变。没有变更公开话题或消息字段。
