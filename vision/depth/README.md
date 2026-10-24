# `depth/`：深度相机 -> 抓取目标

纯几何抓取检测：给一张深度图和内参，找出桌面上凸起的物块，输出「吸附点 + 法向 + 可吸性分数」。**不依赖 YOLO，也不依赖彩色图。**

## 文件

| 文件 | 放什么 |
|---|---|
| `types.py` | `Intrinsics` / `SuctionParams` / `GraspTarget`。3D 类型只在这里，与根级 `types.py` 互不 import |
| `primitives.py` | 与吸盘无关的基本运算：反投影、体素降采样、RANSAC 平面、深度图连通域 |
| `core.py` | 抓取业务：顶面拟合、可吸性打分、编排、门面 `detect_grasp_targets()` |
| `__init__.py` | 对外只暴露 `detect_grasp_targets` / `analyze` 和三个类型 |

## 文件归属判据

只有一条：**描述这段代码时，需不需要提到「吸盘 / 抓取 / 吸附面」？**

| 不需要 → `primitives.py` | 需要 → `core.py` |
|---|---|
| `backproject`：像素 + 深度 → 三维点 | `fit_top_face`：一团点 → **物块顶面的吸附点** |
| `voxel_downsample`：一堆点 → 更少的点 | `score_suction`：**吸盘能不能吸住** |
| `fit_plane_ransac`：一堆点 → 平面方程 | `analyze` / `detect_grasp_targets`：→ 抓取目标 |
| `label_clusters`：深度图 → 每个像素属于第几团 | |

这条判据的意义是防止 `core.py` 继续膨胀。反投影、降采样、平面拟合、连通域这四件事
将来做深度-彩色对齐、手眼标定、多目标分割时都要再用，它们不该被绑在"抓取"里。
`fit_top_face` / `score_suction` 虽然也在算几何，但它们的存在理由就是吸盘，所以留在 core。

读文件、解析 bag、真值对比、命令行参数一律在 `lab/`；ROS 消息转换一律在
`ros2_ws/src/vision_depth/`。

## 对外接口

```python
from vision.depth import Intrinsics, detect_grasp_targets

k = Intrinsics(fx=409.5, fy=409.5, cx=424.33, cy=241.33, width=848, height=480)
targets = detect_grasp_targets(depth_mm, k)      # depth_mm: uint16，单位毫米
for t in targets:                                # 按可吸性分数降序
    print(t.to_dict())
```

调试用 `analyze()`，它会额外返回平面方程、每个簇的点和限制项，能分清「分割错了」还是「打分了」。

## 不 import ROS

`rclpy` 只能 apt 装、pip 装不了，混进来会让 `pip install -e .` 直接失败。ROS 外壳在 `ros2_ws/src/vision_depth/`，那边只做收消息、调这里、发消息。

`tests/test_architecture.py` 有断言守着这条，不靠人记。

## 参数从哪来

`SuctionParams` 的默认值不是拍的：

- `plane_thresh_mm = 6.0`：A4 实测墙面/桌面本身有 2~5mm 起伏，按 3σ（0.5m 处约 1mm）取会把真实桌面点判成外点
- `leaf_mm = 4.0`：比实测噪声（0.4~1.7mm）大一个量级，不会把噪声当结构
- 可吸性五条：计划书 3.3 的初值，仍需实测调整

## 现在的位置

算法先在 `lab/proto_depth.py` 里跑通并调好参数，再提炼到这里（ROS2 接入方案 10.2 的四步落地）。`lab/` 那个脚本保留作为离线调试工具，读 npy/bag 看中间结果；ROS 侧的 `fuse_node` 调的是这里。

## 已知的坑

- **深度是 z 深度，不是欧氏距离。** 反投影是 `x=(u-cx)z/fx` 的直接代入，误当欧氏距离会让点云边缘整体外扩。
- **ROI 裁剪后主点必须跟着平移**，否则反投影整体错位，表现为「物块位置偏了几厘米」，很难联想到是内参没对齐。
- **RANSAC 阈值单位是毫米。** 按米填（0.006）会让每个点都算外点，报的是「找不到桌面」。
- **平面法向方向无约定**，`n` 和 `-n` 都合法。模块内部统一成「朝相机」，算夹角仍取绝对值。
- **顶面拟合不能拿整簇。** 物块是立体的，整簇拟合出的是斜平面，中心还落在方块内部。只取离桌面最远的 20%。
- **聚类不是真正的欧式聚类。** 利用深度图是组织化点云这一特性，在深度跳变处切断连通域，结果等价且快两个量级（容器里没有 scipy，手写 KD 树太慢）。
