# ROS2 接入方案：深度相机与 YOLO

## 文档说明

- 目的：定下深度相机这条线怎么进 ROS，以及 YOLO 不在 ROS 里时两边怎么对接。
- 前置：`src结构扁平化计划-up70_vision.md` 第十一节已定 `depth/` 子包与 import 规则，本文只解决 ROS 这一层。
- 现状（2026-10-23）：相机在手、临时固定可取流，第零段（数据方案）已开工，S1 的触发条件已满足。本文包结构以瘦身分支的 `vision/` 为基准；扁平化未落地时，import 名为 `up70_vision.depth`。
- 路径一律相对仓库根。

---

## 一、先解决一个矛盾

你说的两句要求，放在一起是冲突的：

1. **深度相机要进 ROS** —— 因为坐标要发给控制组。
2. **YOLO 那一侧不进 ROS** —— 因为搭档用 Python 加 ultralytics，不写 ROS 节点。

但融合节点需要 2D 框才能把点云和物体对应起来。**如果 YOLO 的结果不出现在 ROS 里，融合节点拿不到它。**

这个矛盾必须先解开，否则架构定不下来。

---

## 二、「不进 ROS」有三种含义，接法完全不同

| 含义 | 实际情况 | 接法 |
|---|---|---|
| **A. 训练不用 ROS** | 显然，训练和 ROS 无关 | 无影响 |
| **B. 推理进程不是 ROS 节点** | 最可能是这个 | 需要一个出口，见第四节 |
| **C. 推理在另一台机器上** | 有可能，控制机和 PC 分离 | 走网络，ROS2 的 DDS 天然支持 |

**关键判断：B 和 C 都要求「结果进入 ROS」，区别只在出口是什么形式。** 不存在「结果不进 ROS 还能被 ROS 节点用到」这种情况。

所以真正的问题不是「要不要进」，而是「用多少代价进」。

---

## 三、推荐架构

```
┌──────────────────────────────────────────────────────────┐
│ 硬件                                                      │
│   336L ──USB3.0──> /dev/bus/usb（libusb，不是 V4L2）      │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ ROS2 图（比赛运行时，Humble）                              │
│                                                           │
│  orbbec_camera（现成包，不自己写驱动）                      │
│    ├─> /camera/depth/image_raw      2.0 MB/帧             │
│    ├─> /camera/depth/camera_info    内参                   │
│    └─> /camera/color/image_raw      3.0 MB/帧（可选）      │
│                        ↓                                  │
│  vision_depth 包（自己写，薄适配层）                        │
│    fuse_node                                          │
│      ├─ 订阅深度图 + 2D 框                                │
│      ├─ 内部调用 vision.depth 的纯算法               │
│      └─ 输出吸附点与法向                                  │
│                        ↓                                  │
│    /grasp_targets  (vision_msgs/Detection3DArray)         │
│                        ↓                                  │
│  控制组节点（不在本线范围）                                │
└──────────────────────────────────────────────────────────┘
                        ↑
┌──────────────────────────────────────────────────────────┐
│ 非 ROS 侧（搭档的 PC 或同一台机器上的独立进程）             │
│                                                           │
│  YOLO 推理进程（ultralytics，原样不动）                     │
│    └─> /detections_2d  (Detection2DArray，约 600 B/帧)     │
│         ↑ 唯一接触点，约 20 行 rclpy 代码                  │
└──────────────────────────────────────────────────────────┘
```

三条设计原则：

1. **算法在库里，ROS 只做消息转换。** `vision.depth` 是纯 Python，能跑 pytest，不 import rclpy。
2. **相机节点用现成的 `orbbec_camera`。** 符合既有约定「深度相机驱动优先用 ROS 封装库，不自己写」。
3. **融合节点自己写，但只做三件事**：收消息、调库、发消息。

---

## 四、YOLO 侧：约 20 行代码，不算「进 ROS」

这是解开矛盾的关键。发一个 ROS 话题**不需要写 ROS 节点架构**，不需要 launch、不需要参数服务器、不需要生命周期管理。改的是「结果往哪送」这一句。

原来的写法：

```python
results = model(frame)
for box in results[0].boxes:
    print(box.xywh)          # 原来输出到终端或画图
```

改成：

```python
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D

rclpy.init()
node = Node('yolo_publisher')
pub = node.create_publisher(Detection2DArray, '/detections_2d', 10)

results = model(frame)                     # 推理部分一行不改
msg = Detection2DArray()
msg.header.stamp = node.get_clock().now().to_msg()
msg.header.frame_id = 'camera_color'
for box in results[0].boxes:
    det = Detection2D()
    det.bbox.center.position.x = float(box.xywh[0][0])
    det.bbox.center.position.y = float(box.xywh[0][1])
    det.bbox.size_x = float(box.xywh[0][2])
    det.bbox.size_y = float(box.xywh[0][3])
    det.results.append(ObjectHypothesis(score=float(box.conf[0]), ...))
    msg.detections.append(det)
pub.publish(msg)
```

**这 20 行不改变搭档的工作方式。** 他仍然是一个 Python 脚本、一个循环、一次 `model()` 调用。加进去的只是「把结果打包发出去」。

如果搭档连 rclpy 都不想碰，还有一条退路：**融合节点内部直接 import ultralytics 做推理**。这样 YOLO 完全不在 ROS 里，只是一个库调用。但代价很实际：

- YOLO 推理（GPU，20–50ms）和点云处理（CPU，150ms）串行在同一个回调里，总周期 200ms，只有 5Hz。
- GIL 争抢，两边都变慢。
- 融合节点的依赖从 3 个变成 5 个，部署失败面变大。

**所以退路能走，但不推荐。** 让 YOLO 发一个话题，是成本最低、耦合最小的方案。

---

## 五、带宽：不要订阅 `/points`

这是最容易踩的一个坑，而且会直接打爆 ROS 图。

| 话题 | 单帧大小 | 30fps 带宽 | 结论 |
|---|---|---|---|
| `/camera/depth/points`（PointCloud2，XYZRGB 16 B/点） | 1280×800×16 = **16.4 MB** | **492 MB/s** | 不可用 |
| `/camera/depth/image_raw`（16UC1） | 1280×800×2 = **2.0 MB** | 61 MB/s | 可用，跨机器要降分辨率 |
| `/camera/depth/image_raw`（降到 640×400） | 0.5 MB | 15 MB/s | 推荐默认 |
| `/detections_2d` | 约 600 B | 9 KB/s | 忽略不计 |

**结论：融合节点订阅深度图，自己做反投影。**

理由不只是带宽：

- 反投影是 30–60ms 的 numpy 向量化操作，本来就在计划里。
- 自己反投影能**先降采样深度图再算**，比先转点云再降采样省一半时间。
- Python 节点不支持 `use_intra_process_comms` 的零拷贝（那是 C++ 特性），所以 16MB 的点云每帧都要完整序列化一次，反序列化又要 20–40ms，光这一项就吃掉四分之一的时间预算。

**跨机器时（含义 C）必须降到 640×400**：1280×800 的 61 MB/s 在千兆网上是 500 Mbps，已经接近极限，加上 DDS 开销会丢帧。

---

## 六、仓库层面：新建 `vision_depth`，不动 `up70_vision`

### 6.1 结论

`up70_vision` 保持「纯 Python 库 + CLI」的定位，不塞 ROS 节点。深度相机的 ROS 部分**独立成一个 ROS2 包**。

### 6.2 四条理由

**第一，依赖装不上。** `up70_vision` 的依赖走 `pyproject.toml` + pip，而 `rclpy` 只能 `apt install ros-humble-rclpy`，pip 装不了。混在一个包里，`pip install -e .` 会直接失败，队友克隆仓库跑不起来。

**第二，这个仓库正在被批评「庞大」。** 2026-10-22 的复盘已经确认管理者批评约 60% 成立。再往里塞 ROS 节点、launch、配置文件，是把「加装饰」当整改。

**第三，生命周期不同。** 库是「调用一次、返回结果、退出」；ROS 节点是「常驻、回调、被 launch 拉起、被看门狗重启」。`viz/` 开 cv2 窗口、`cli/` 解析命令行参数，这两件事在 ROS 节点里都是错的。

**第四，职责边界清楚。** `up70_vision` 是 2D 视觉工具箱（判色、ArUco、位姿、标定），深度相机是 3D 抓取。两者在代码上几乎不重叠，硬塞在一起只会让「找一个功能跳几次」这个指标变差。

### 6.3 重叠几乎为零，这个结论可以量化

| 深度相机这条线需要的 | `up70_vision` 有吗 |
|---|---|
| 反投影 | 没有 |
| 点云降采样 / 平面分割 / 聚类 | 没有 |
| 可吸性判分 | 没有 |
| 深度相机内参 | 不需要（用出厂内参） |
| 2D 相机标定（`calibration.py`） | 有，但那是棋盘格 2D 标定，用途不同 |
| 可视化（`viz/`） | 有，但 ROS 用 RViz |
| ArUco 检测（`detectors/aruco.py`） | **有，且有用**（手眼标定用） |

**唯一的真实重叠是 ArUco 检测**，而它只用在离线的手眼标定脚本里，不进 ROS 运行时。所以依赖方向是「离线工具依赖库」，不是「ROS 节点依赖库」，边界很干净。

### 6.4 如果一定要把 depth 放进 vision/ 目录下

先分清一个概念：**「功能包」在 ROS 语境里指 ament 包，它必须有 `package.xml`。**

在 `vision/depth/` 下多放几个文件，那只是 pip 包的一个子模块，`ros2 run` 找不到它，`ros2 launch` 也拉不起来。想让它成为真正的功能包，只有两条路。

**路线一：给 `vision` 包加 `package.xml`，让它同时是 pip 包和 ament 包。**

技术上可行。ament_python 包的 `setup.py` 本来就是 pip 的 `setup.py`，只要满足三个条件：`install_requires` 不写 rclpy、`package.xml` 里写 `<depend>rclpy</depend>`、代码里延迟 import rclpy。

但有四个代价，而且都是长期代价：

| 代价 | 具体表现 |
|---|---|
| 依赖错配 | pip 装得上但 colcon 装不上，或反过来。报错信息会指向无关的地方，很难查 |
| IDE 全红 | 在没 source ROS 的环境里打开仓库，`import rclpy` 标红，类型跳转全部失效 |
| 测试分叉 | 全部既有测试要能在「有 ROS」和「没 ROS」两种环境下都通过 |
| 边界失守 | 一旦允许 `vision` 里出现 rclpy，`core.py` 里迟早也会出现，然后算法就不可测了 |

第四条是根本问题。**「core 不 import ROS」这条约定能成立，前提是包本身就没有 ROS 依赖。** 一旦包里有，就只能靠人守，而人守不住。

**路线二：算法留 `vision/depth/`，ROS 壳放 `vision_depth/`。**（推荐）

```
up70_vision/
├── pyproject.toml                 pip 包定义
├── vision/                        pip 包：纯算法，可 pytest
│   ├── types.py  image_io.py  log.py  api.py  cli.py
│   ├── config/  sources/  viz/
│   ├── aruco/  yolo/
│   └── depth/                     纯算法，不 import rclpy
│       ├── types.py               DepthFrame / PointCloud / GraspTarget
│       └── core.py                反投影 / 滤波 / 分割 / 聚类 / 可吸性
│
├── ros2_ws/                       colcon 工作空间（只有 ROS 相关）
│   └── src/vision_depth/          ament_python 包
│       ├── package.xml            depend: rclpy sensor_msgs vision_msgs tf2_ros
│       ├── setup.py               entry_points: fuse_node / calib_node
│       ├── vision_depth/
│       │   ├── fuse_node.py       只 import vision.depth.core
│       │   └── calib_node.py
│       ├── launch/
│       └── config/
│
├── lab/                           一次性工具，可 import vision 与 vision_depth
├── knowledge/
└── configs/
```

`vision_depth/fuse_node.py` 只有三类代码：收消息、调 `vision.depth.core`、发消息。算法一行都不在这里。

**为什么 `ros2_ws/` 这一层不能省：** colcon 会在工作空间根生成 `build/`、`install/`、`log/` 三个目录。直接在仓库根跑 colcon，这三个目录会跟 `vision/`、`lab/` 混在一起，`.gitignore` 要写得更小心，而且新人 `ls` 一下会以为仓库多了三个源码目录。加一层 `ros2_ws/`，产物全部关在里面。

**构建顺序有依赖，不能颠倒：**

```bash
# 1. 先装算法库（Dockerfile 里固化）
pip install -e .

# 2. 再编 ROS 包（node 要 import vision.depth）
cd ros2_ws
colcon build --packages-select vision_depth
source install/setup.bash

# 3. 运行
ros2 run vision_depth fuse_node
```

第 1 步不能省。`colcon build` 不解析 pip 依赖，它只保证 `package.xml` 里声明的东西存在。`vision` 得先装好，否则 `fuse_node` 启动时才会报 `ModuleNotFoundError`，而那时你已经在查相机了。

**两条路的实际差别：**

| | 路线一（塞进 vision/） | 路线二（独立 ament 包） |
|---|---|---|
| 仓库顶层目录数 | 不变 | +1（`ros2_ws/`） |
| 构建命令 | 一套 | 两套 |
| `ros2 run` 能否用 | 能（但需手写 launch 的 ExecuteProcess） | 正常 |
| IDE 体验 | 非 ROS 环境全红 | 正常 |
| 算法可测性 | 靠人守约定 | 物理隔离 |
| 出问题时的排查成本 | 高（依赖错配难查） | 低 |

多一个顶层目录，换掉的是「依赖错配」和「边界失守」两类问题。这两类问题的共同点是**发现得晚、修起来贵**。

---

## 七、包结构

```
up70_vision/                          仓库根
├── pyproject.toml                    pip 包定义
├── vision/                           pip 包：纯算法，可 pytest
│   ├── types.py  image_io.py  log.py  api.py  cli.py
│   ├── config/  sources/  viz/  aruco/  yolo/
│   └── depth/
│       ├── types.py                  DepthFrame / PointCloud / GraspTarget
│       └── core.py                   反投影 / 滤波 / 分割 / 聚类 / 可吸性
│
├── ros2_ws/                          colcon 工作空间（build/install/log 都关在这里）
│   └── src/vision_depth/             ament_python 包
│       ├── package.xml               depend: rclpy / sensor_msgs / vision_msgs / tf2_ros
│       ├── setup.py                  entry_points: fuse_node / calib_node
│       ├── vision_depth/
│       │   ├── __init__.py
│       │   ├── fuse_node.py          收消息 → 调 vision.depth.core → 发消息
│       │   ├── calib_node.py         发布 base_link → camera_link 静态变换
│       │   └── watchdog.py           话题断流检测（后置）
│       ├── launch/
│       │   ├── camera_only.launch.py 只起相机，调试用
│       │   ├── fuse.launch.py        相机 + 融合
│       │   └── tf_static.launch.py   标定结果
│       └── config/
│           ├── camera_params.yaml    分辨率、帧率、深度范围
│           └── handeye.yaml          标定矩阵
│
├── lab/                              一次性工具，零约束，可 import vision
│   ├── calibrate_handeye.py          复用 vision.aruco.detector
│   ├── noise_sweep.py                噪声实测
│   └── eval_accuracy.py              误差统计
│
├── knowledge/  configs/
└── tests/
```

三条边界：

- `vision_depth/` 里**不放算法**，只放消息转换。`fuse_node.py` 超过 200 行就说明算法漏进来了。
- `lab/` 里的脚本**不 import ROS 包**。它们是离线跑的，依赖 ROS 运行时会让「跑一次统计」变成「先 source 环境」。
- `ros2_ws/build`、`install`、`log` 进 `.gitignore`。这三个目录加起来几百 MB，提交上去仓库直接废。

### 7.1 文件归属：唯一判据是「要不要 rclpy」

深度相机这条线的文件，按一个判据分流，不用想别的：

> **这个文件需要 `import rclpy` 或 `from sensor_msgs...` 吗？**
> 需要 → `ros2_ws/src/vision_depth/`
> 不需要 → `vision/depth/` 或 `lab/`

按这条判据分完，结果如下：

| 文件 | 放哪 | 为什么 |
|---|---|---|
| `depth/types.py` | `vision/` | 只有 dataclass，零依赖 |
| `depth/core.py` | `vision/` | numpy 加 open3d，能 pytest |
| `depth/params.py` | `vision/` | 参数 dataclass，供 core 与节点共用 |
| `fuse_node.py` | `ros2_ws/.../vision_depth/` | 收消息、调 core、发消息 |
| `calib_node.py` | 同上 | 发静态 TF |
| `watchdog.py` | 同上 | 订阅话题、看频率 |
| `package.xml` / `setup.py` | `ros2_ws/.../vision_depth/` | ament 包定义，必须在包根 |
| `launch/*.launch.py` | 同上 | 只有 ROS 用 |
| `config/*.yaml` | 同上 | 只有 ROS 用 |
| `calibrate_handeye.py` | `lab/` | 离线跑，不需要 rclpy |
| `noise_sweep.py` | `lab/` | 同上 |
| `eval_accuracy.py` | `lab/` | 同上 |
| `test_depth.py` | `tests/` | pytest，只测 `vision/depth/` |

**注意 `params.py` 的位置。** 参数 dataclass 放 `vision/` 而不是 ROS 包，因为它同时被 `core` 和 `fuse_node` 用。如果放 ROS 包，`core` 就得反向 import ROS 包，依赖方向立刻颠倒。

**节点从 YAML 读参数、装进 `GraspParams`、再传给 `core`**，这样 `core` 完全不知道参数从哪来——测试时直接构造 `GraspParams` 就行，不需要造 YAML 文件。

### 7.2 `vision/depth/` 里绝对不能出现的东西

```python
import rclpy                          # 不行
from sensor_msgs.msg import Image     # 不行
from vision_msgs.msg import ...       # 不行
import rospkg                         # 不行
```

只要出现一条，`pip install -e .` 在有 ROS 的机器上装得上、在没 ROS 的机器上装不上，而 `tests/` 在 CI 里就跑不了。

这条靠 `tests/test_architecture.py` 断言，不靠人记：

```python
def test_depth_has_no_ros_import():
    for path in (SRC / "vision" / "depth").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import rclpy" not in text, f"{path} 里出现了 rclpy"
        assert "sensor_msgs" not in text, f"{path} 里出现了 ROS 消息"
```

---

## 八、接口定义

### 8.1 输入（融合节点订阅）

| 话题 | 类型 | 频率 | 说明 |
|---|---|---|---|
| `/camera/depth/image_raw` | `sensor_msgs/Image` | 30Hz | 16UC1，单位毫米，0 表示无效 |
| `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | 30Hz | 内参 |
| `/detections_2d` | `vision_msgs/Detection2DArray` | 10–15Hz | 搭档发 |

**两路频率不一致的处理**：深度帧取时间最近的检测结果，超过 100ms 容差就丢弃该帧，不拿旧框硬套新点云。这条已写进 `docs/interface.md`。

**`/detections_2d` 是可选输入。** 第零段与第一段走纯几何模式（`use_2d_dets:=false`，默认 false）：没有 2D 框时 `fuse_node` 直接跑 RANSAC + 聚类 + 顶面拟合，不依赖搭档进度。YOLO 就绪后再开启融合——这是计划书「B 阶段不引入 YOLO」在节点层的落点。

### 8.2 输出（融合节点发布）

| 话题 | 类型 | 频率 | 说明 |
|---|---|---|---|
| `/grasp_targets` | `vision_msgs/Detection3DArray` | 10Hz | 吸附点 + 法向 + 可吸性分数 |
| `/tf` | `tf2_msgs/TFMessage` | — | `base_link → camera_link` 静态变换 |

字段对应：

| `Detection3D` 字段 | 装什么 |
|---|---|
| `bbox.center.position` | 吸附点，基座系，米 |
| `bbox.center.orientation` | 法向作为四元数（z 轴对齐法向） |
| `results[0].hypothesis.score` | 可吸性分数 0–1 |
| `id` | 目标编号 |

**预吸附位（沿法向偏移的接近点）不放进接口**，由控制组自己算。这样换吸盘尺寸时接口不用改。

---

## 九、与 `up70_vision` 的唯一交集

只有一处，而且是离线工具：

```
lab/calibrate_handeye.py
    import vision.aruco.detector    # 复用 ArUco 检测
```

**放在 `lab/` 而不是 ROS 包里。** 手眼标定一年做几次，不是运行时功能，放进 ROS 包会让「哪些代码在跑」变模糊。`lab/` 可以 import `vision`，这条依赖方向是允许的。

要让它能 import，容器里需要把 `vision` 装成 editable 包：

```bash
pip install -e /workspace/up70_vision
```

这一条要写进 Dockerfile。**不要在 ROS 包的 `package.xml` 里声明它**，因为那不是 ROS 依赖，是 pip 依赖。

如果嫌麻烦，`calibrate_handeye.py` 直接用 `cv2.aruco` 写也就 30 行，但要自己处理 `vision.aruco` 已经踩过的坑（字典版本 API 变更、`ids` 无码时是 `None`、角点顺序）。**建议复用，不要重踩。**

---

## 十、推荐结论与落地顺序

### 10.1 推荐：路线二

**算法留 `vision/depth/`（纯 Python，可测），ROS 壳放 `ros2_ws/src/vision_depth/`（薄）。**

不给 `vision` 加 `package.xml`。理由见 6.4 第四条：一旦包里有 rclpy，「core 不 import ROS」就只能靠人守，人守不住。

### 10.2 但执行分四步，每步都有触发条件

不要一次把所有目录建齐。**过早固化结构，会搬两次。**

| 步骤 | 做什么 | 触发条件 | 落在哪 |
|---|---|---|---|
| **S0** | 写原型：反投影、滤波、分割、聚类、可吸性 | 现在就能开始 | `lab/` |
| **S1** | 建 ROS 包，起相机，发空数组 | **相机在手（已满足）** | `ros2_ws/src/vision_depth/` |
| **S2** | 算法从 `lab/` 提炼成正式模块 | **算法在相机系跑通、参数稳定** | `vision/depth/` |
| **S3** | `fuse_node` 从「发空数组」改成「调 `vision.depth.core`」 | **仓库整改完成** | 改 import，10 分钟 |

**为什么 S0 放 `lab/` 而不是直接建 `vision/depth/`：**

1. **仓库正在整改。** 瘦身 42 处变更还没提交，架构迁移方案（9.5h）也还没执行。现在往 `vision/` 里加目录，会跟正在搬的目录撞车。
2. **参数还没定。** `leaf_size`、`distance_threshold`、`cluster_tolerance` 都要靠实测试出来，这阶段代码一天改十次，不需要正式结构。
3. **`lab/` 的定位就是「零约束、用一次就扔」。** 原型代码本来就该在这里，不是妥协，是它的正确位置。

**为什么 S1 要等相机装好，不能提前建空包：**

建空包本身 20 分钟，但它唯一的验收是「起相机、有话题」。相机没装时建出来的包无法验证，只是占位。**占位的包三个月后没人知道它是干嘛的。**

**为什么 S2 要等算法稳定，不能边写边搬：**

算法在 `lab/` 里可以随意重构函数签名，一旦进了 `vision/depth/` 就有 import 约束和测试要求。**在签名还没定型时搬进去，等于给自己上枷锁。**

### 10.3 S1 内部顺序（相机在手，随时可做）

| 步骤 | 动作 | 验收 |
|---|---|---|
| 1 | 建 `vision_depth` 空包，`colcon build` 通过 | `ros2 pkg list` 里有它 |
| 2 | 装 `orbbec_camera`，起相机，确认三路话题 | `ros2 topic hz` 深度 ≥ 25Hz |
| 3 | 写 `fuse_node` 骨架：只订阅深度图，发空数组 | `ros2 topic echo /grasp_targets` 有输出 |
| 4 | 用假 2D 框对接，验证坐标换算 | 假框 + 真深度 → 数值正确 |
| 5 | 手眼标定，发布静态 TF | 标定板误差 < 15mm |

第 3 步特意做成「先发空数组」。这样在算法一行没写的时候，就能验证话题链路、频率、消息格式全对，把「接口问题」和「算法问题」彻底分开。这个顺序能省掉后面最难查的一类故障。

### 10.4 现在能做的四件事（相机在手，全部可立即开工）

1. **在 `lab/` 写反投影和聚类原型**，用数据方案 S1 录的真实 bag 或合成深度图跑通，把参数试出来。
2. **建 `ros2_ws/src/vision_depth/` 空包，跑一次 `colcon build`。** 这一步只验证构建链，不依赖相机姿态。提前知道 CMake/ament 有没有问题，比后面才发现好。
3. **和控制组对齐接口字段**（吸附点、法向、可吸性分数、id），顺便定退让位。这张表不需要任何硬件。
4. **按数据方案第九节录第一段 bag**，`meta.yaml` 与 bag 同存，先摸一遍真实失效模式。

第 2 条是唯一「提前做也不浪费」的准备工作——构建链问题是纯环境问题，和相机无关，越早暴露越好。

---

## 十一、三个坑

**不要在容器里订阅 `/points` 试带宽。** 492 MB/s 会把 DDS 缓冲区打满，表现是「话题有频率但数据不完整」，然后你会怀疑相机、怀疑驱动。从深度图开始，需要点云自己反投影。

**`orbbec_camera` 的 launch 参数名随版本变。** 330 系列的参数在 Humble 上的包和官方文档不一定一致，先 `ros2 param list` 看实际有什么，不要照抄教程。

**Python 节点之间没有零拷贝。** 别指望 `use_intra_process_comms` 省掉序列化，那是 C++ 特性。所有优化都要从「少传数据」入手，而不是「少拷贝」。

---

## 十二、`fuse_node` 怎么调 `depth` 的代码

### 12.1 import 能成立，靠三个条件

```python
# ros2_ws/src/vision_depth/vision_depth/fuse_node.py
from vision.depth import core              # 算法，pip 包
from vision.depth.types import GraspTarget # 类型，pip 包
```

这行 `import` 能不能跑通，取决于三件事，缺一件就 `ModuleNotFoundError`：

| 条件 | 谁保证 | 缺了会怎样 |
|---|---|---|
| `vision` 已被 pip 装到当前环境 | Dockerfile 里 `pip install -e /workspace/up70_vision` | `colcon build` 不报错，`ros2 run` 启动时才崩 |
| `vision` 包内没有任何 `import rclpy` | `tests/test_architecture.py` 断言 | `pip install -e .` 直接失败 |
| `vision_depth` 的 `package.xml` **不**声明 `vision` | 人工检查 | `rosdep` 找不到这个包，构建失败 |

第三条容易搞错：`package.xml` 只管 ROS 包，管不了 pip 包。**pip 依赖写在 Dockerfile，ROS 依赖写在 `package.xml`，两者不混。**

`source install/setup.bash` 会把 ament 的路径插到 `PYTHONPATH` 前面，但不会覆盖 pip 的 site-packages，所以两者能共存。

### 12.2 算法接口长什么样

`vision/depth/core.py` 对外只暴露一个纯函数，签名固定，不碰 ROS：

```python
def detect_grasp_targets(
    depth_mm: np.ndarray,          # (H, W) uint16，单位毫米，0 表示无效
    intrinsics: CameraIntrinsics,  # fx / fy / cx / cy，从 camera_info 来
    params: GraspParams,           # 降采样、平面阈值、聚类容差、吸盘直径
    bboxes: list[Bbox2D] | None = None,  # 可选 2D 框；None = 纯几何模式
) -> list[GraspTarget]:            # 无目标返回 []，不是 None
```

`bboxes` 是纯 dataclass 列表，**不是 ROS 消息**。节点收到 `Detection2DArray` 后先转成 `list[Bbox2D]` 再传入——算法层永远不接触 ROS 类型，这是 7.2 边界的落点：

```python
@dataclass(frozen=True)
class Bbox2D:
    cx: float    # 框中心 x（像素）
    cy: float    # 框中心 y（像素）
    w: float     # 框宽
    h: float     # 框高
    score: float # 检测置信度
```

`GraspTarget` 五个字段，够用且不冗余：

```python
@dataclass(frozen=True)
class GraspTarget:
    center_m: tuple[float, float, float]   # 吸附点，相机系，米
    normal: tuple[float, float, float]     # 单位向量，指向相机一侧
    score: float                            # 可吸性 0–1
    n_points: int                           # 支撑这个判断的点数，调试用
    id: int                                 # 目标编号，对应消息里的 id
```

**为什么要 `n_points`：** 出问题时第一个要问的是「这个目标有几个点支撑」。没有这个字段，你只能回去重跑一遍；有了它，日志里就能看出「3 个点的目标被当成了物块」。

### 12.3 节点骨架

```python
class FuseNode(Node):
    def __init__(self):
        super().__init__('fuse_node')

        self.declare_parameter('max_det_age_ms', 100)
        self.max_det_age_ns = self.get_parameter('max_det_age_ms').value * 1_000_000
        self.declare_parameter('use_2d_dets', False)   # 第零段/第一段默认纯几何

        # 参数从 YAML 装进 GraspParams（见 7.1），core 不知道参数从哪来
        self.params = load_grasp_params(self)

        self.latest_bboxes = None      # 最近一帧 2D 框（已转成纯 dataclass）
        self.latest_dets_ns = 0        # 它的时间戳，纳秒
        self.intrinsics = None         # 收到 camera_info 之后才处理深度

        self.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, '/camera/depth/image_raw', self.on_depth, 1)
        self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.on_info, 1)
        self.create_subscription(Detection2DArray, '/detections_2d', self.on_dets, 10)

        self.pub = self.create_publisher(Detection3DArray, '/grasp_targets', 10)

    def on_info(self, msg):
        # 内参只变一次，收到就存下来，后续每帧复用
        self.intrinsics = CameraIntrinsics(
            fx=msg.k[0], fy=msg.k[4], cx=msg.k[2], cy=msg.k[5])

    def on_dets(self, msg):
        # ROS 消息在此处转成纯 dataclass，core 永远不见 Detection2DArray
        self.latest_bboxes = [
            Bbox2D(cx=d.bbox.center.position.x, cy=d.bbox.center.position.y,
                   w=d.bbox.size_x, h=d.bbox.size_y,
                   score=d.results[0].hypothesis.score if d.results else 0.0)
            for d in msg.detections]
        self.latest_dets_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec

    def on_depth(self, msg):
        # 深度回调驱动整条链路，而不是检测回调
        if self.intrinsics is None:
            return                      # 还没内参，跳过

        bboxes = None                   # 默认纯几何，不依赖 2D 框
        if self.get_parameter('use_2d_dets').value:
            age = self.get_clock().now().nanoseconds - self.latest_dets_ns
            if self.latest_bboxes is None or age > self.max_det_age_ns:
                return                  # 检测太旧，不拿旧框硬套新点云
            bboxes = self.latest_bboxes

        depth_mm = image_to_depth_mm(msg)
        targets = core.detect_grasp_targets(
            depth_mm, self.intrinsics, self.params, bboxes)

        # M5 的核心交付：输出基座系。header 用 base_link，点先过 TF
        self.pub.publish(self.to_detection3d_array(
            targets, 'base_link', self.tf_buffer))
```

**为什么用深度回调驱动，而不是检测回调：** 深度 30Hz、检测 10–15Hz。用检测回调驱动会丢掉大部分深度帧，而且检测结果到达时手上的深度帧可能已经过期。用深度回调驱动，每帧都拿「最近的检测结果」，超时就跳过，逻辑简单且不会用错数据。

**发布前变换到基座系（M5 的核心交付）。** 上面的骨架里 `to_detection3d_array` 用 `tf_buffer` 把每个 target 的 `center_m` 与法向从 `camera_link` 变到 `base_link`，消息 header 的 `frame_id` 写 `base_link`。TF 树还没有标定结果时（第零段），先用假定外参占位——变换代码先写好，支架到位后只换 YAML，不动节点。

### 12.4 三个数据转换的坑

**坑一：`msg.step` 不等于 `width × 2`。**

驱动会做行对齐 padding，直接 `reshape(h, w)` 会把 padding 当成像素，整张图错位，而且**不报错**。正确写法：

```python
dtype = np.dtype(np.uint16)
buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
depth = buf[:, :msg.width * dtype.itemsize].copy().view(dtype).reshape(msg.height, msg.width)
```

那个 `.copy()` 不能省：`np.frombuffer` 返回的是只读数组，后面做滤波要原地修改就会报错。

**坑二：法向转四元数，正上方和正下方会崩。**

用「旋转轴 = z × n」的写法，当 n 平行于 z 时叉积为零向量，归一化得到 NaN，四元数变成 `(0,0,0,nan)`，RViz 里那个箭头直接消失，而日志里没有任何错误。

必须特判：`n ≈ (0,0,1)` 返回单位四元数，`n ≈ (0,0,-1)` 返回绕 x 轴 180°。

**坑三：深度单位是毫米，输出必须是米。**

`depth_mm` 一路传到 `core` 里都是 uint16 毫米，`core` 内部在反投影时除以 1000 转成米。**这个换算只在一个地方做**，不要在节点里也转一次，否则会出现「坐标小了 1000 倍」这种一眼看不出、但物块跑到几毫米远处的现象。

### 12.5 一个可以省掉的复杂度

**不要用 `message_filters.ApproximateTimeSynchronizer`。**

它看起来更"正确"，但实际会带来两个问题：深度 30Hz 和检测 10Hz 不同源，近似同步会持续丢帧，实际输出频率掉到 5Hz 以下；而且同步失败时没有任何日志，你只会看到"话题没输出"。

手动维护「最近一帧检测 + 超时判断」只有 6 行代码，行为完全可预测，出问题能直接打日志。

---

## 十三、一句话

**算法留在库里（纯 Python，可测），ROS 只做消息适配（薄），YOLO 只加一个 20 行的发布出口（不改工作方式）。** 三者边界清楚，各自都能独立调试。
