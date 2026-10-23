# YOLO 接入：现在能干什么、怎么用、为什么不能直接训练

> 写给队友看的版本。目标是：**不装 torch、不改代码，也能跑起 YOLO 检测**；
> 以及把"为什么 0 个框""为什么全是 unknown"这类高频疑问一次说清。

## 一、一句话结论

仓库里已经有一个可用的 YOLO 检测器（`detector.name=yolo`），它用 **ONNX 权重 + OpenCV** 推理，
容器里不需要 torch、不需要 GPU，1280×1024 一帧约 **35 ms**。
但它用的是**通用预训练模型，只认 COCO 的 80 类**（人、车、瓶子、杯子……），
**不认识我们的色块**——所以"接好了"不等于"能用了"。

## 二、三条路，为什么选了最省的那条

| 路线 | 需要装什么 | 速度（这台机器，1280×1024 一帧） | 镜像增量 | 用途 |
|---|---|---|---|---|
| A. `ultralytics` + torch（CPU） | torch 1.4 GB | 17 ms | +1.4 GB | 训练 + 推理 |
| B. `ultralytics` + torch（GPU） | 驱动 + 工具链 | 5~10 ms | +7~8 GB | 训练 |
| **C. ONNX + `cv2.dnn`（已采用）** | **什么都不用装** | **26~35 ms** | **+0**（只有 12.9 MB 权重） | 只推理 |

相机最高 25 fps（40 ms 一帧的预算），路线 C 够用；而且它是**纯推理**，不背 torch 进镜像。
将来真要训练，再单独在开发机上装 A/B，导出的 ONNX 仍然喂给同一段代码——**换模型不换代码**。

## 三、怎么用（三步，照抄就行）

### 第 1 步：拿到 ONNX 权重（一台机器做一次即可）

权重不进版本库，所以要本地生成一次。需要 `ultralytics`（本机在 `~/.venvs/yolo`）：

```bash
cd vision-main
~/.venvs/yolo/bin/python lab/export_yolo.py --weights yolov8n.pt
```

产出两个文件（都已被 `.gitignore` 忽略）：

```
assets/models/yolov8n.onnx         12.9 MB   模型
assets/models/yolov8n.names.json   类名表   容器里没有 onnx 包也能读出 "person" 这种名字
```

### 第 2 步：跑

```bash
# 用任意有 cv2 的解释器（容器里就是 python；宿主机用 ~/.venvs/vision/bin/python）
python -m vision run --set detector.name=yolo

# 想跑一张具体的图：把它们放进一个目录当"回放"喂进去
python -m vision run --set detector.name=yolo \
  --set input.source=replay --set input.replay.path=/path/to/images
```

> ⚠️ `python -m vision live` 是 **ArUco 专用**入口（要画四边形和位姿轴）。
> 用 YOLO 请走 `python -m vision run`；走错了程序会直接告诉你该用哪条命令，不会抛 traceback。

### 第 3 步：看结果

```
output/demo/frame_0000.png      带框的叠加图（框上写着 类名 + 置信度）
output/demo/results.json        每帧的检出明细（bbox / class_name / class_id / score）
output/demo/detections.csv      表格版
output/demo/summary.json        汇总统计
```

## 四、配置项（`configs/default.yaml` 的 `detector.yolo` 段）

| 键 | 人话 | 默认 |
|---|---|---|
| `weights` | ONNX 权重路径（相对仓库根） | `assets/models/yolov8n.onnx` |
| `conf` | 置信度门槛。调低 → 框变多、误检也变多 | 0.25 |
| `iou` | 去重门槛。两个框重合度超过它就只留分数高的 | 0.45 |
| `imgsz` | 模型输入边长，必须和导出时一致 | 640 |
| `max_detections` | 一帧最多留几个框 | 32 |
| `classes` | 只保留哪些类别（如 `[0]` 只看人）；空 = 全部 80 类 | `[]` |
| `class_names` | 类名表；空 = 读 sidecar / ONNX 元数据 | `[]` |
| `device` | 目前只支持 `cpu` | `cpu` |

命令行也能临时改（注意 `classes` 这类列表建议直接改 yaml）：

```bash
python -m vision run --set detector.name=yolo --set detector.yolo.conf=0.5
```

## 五、四个高频疑问

### 1. 为什么我们自己的图上「0 个框」？

**不是接入失败。** COCO 的 80 类里没有"色块"，模型没见过就不输出。
验证链路是否通，用一张 COCO 自己的图（人/车）：应能检出 5~6 个框。

### 2. 为什么带框图上写着 `unknown`？

**那是颜色，不是类名。**
- `class_name`（YOLO 给）：`person` / `bus`；
- `color`（判色阶段给）：person 身上没有我们定义的 6 种颜色 → `None`。

画图以前只按颜色拼标签 → 全显示 `unknown`，还把"颜色命中率"当成了置信度。
现在已修正：**有类名就写「类名 + 置信度」**，判色族（`blob`）行为不变。

### 3. 会不会拖慢实时？

不会。容器里实测：纯前向 26 ms，加前后处理 35 ms（1280×1024）。
相机 40 ms 曝光下还有余量；如果同时算位姿，注意总时间别超过一帧预算。

### 4. 报错 `YOLO 权重不存在` 怎么办？

按第 1 步导出一次即可。报错信息里也印了这条命令。

## 六、要不要训练？现在**不要**

训练需要三样东西，我们一个都没有：

| 前置条件 | 现状 |
|---|---|
| 物块定型（尺寸/材质/颜色、几类） | ❌ 没有 |
| 标注数据（每类几百张、多光照/角度/背景） | ❌ 0 张 |
| GPU 训练环境（驱动 + 工具链） | ❌ 没装 |

**什么时候才值得训练**：需要"同色不同物"分类、形态/正反面识别、或者 HSV 判色在现场确实
误检漏检严重。只要"按颜色分类"够用，`blob`（HSV）永远比 YOLO 划算——更快、可解释、不用数据。

真要训练时的路线（供以后参考）：

```
采图 → 标框（CVAT/labelImg）→ 划分 train/val → 微调 yolo11n（GPU）
     → 导出 ONNX → 替换 assets/models/ 里的权重 → 代码一行不改
```

## 七、许可提醒（重要）

YOLOv8 的代码与权重都是 **AGPL-3.0**。队内自用没问题；但如果程序要**对外发布**
（比赛系统提交、开源、交付），AGPL 要求一并提供源码。正式交付前请跟队长确认，
或改用 Apache-2.0 / MIT 的模型（YOLOX、NanoDet 等）——接口一样挂在这个位置。

## 八、文件在哪

| 东西 | 位置 |
|---|---|
| 检测器实现 | `vision/yolo/detector.py` |
| 工厂与注册 | `vision/aruco/ 与 yolo/base.py`（`DETECTOR_NAMES`） |
| 配置与校验 | `configs/default.yaml` 的 `detector.yolo`、`vision/config/schema.py` |
| 导出脚本 | `python lab/export_yolo.py`（实现 `vision/tools/yolo/export.py`） |
| 权重与说明 | `assets/models/`（权重被 gitignore，只有 README 入库） |
| 测试 | `tests/test_yolo.py`（20 项，不需要权重也能跑） |

## 九、实测记录（这台机器，仅供参考）

| 输入 | 耗时 |
|---|---|
| 640×640 纯前向（`cv2.dnn`，容器 cv2 4.14） | 26 ms |
| 1280×1024 一帧（含 letterbox + NMS） | 35 ms |
| 同上，宿主 venv（cv2 4.11） | 40 ms |
| `ultralytics` + torch CPU（同尺寸，对比用） | 17 ms |

`bus.jpg` 的结果：`person ×4 + bus ×1`（置信度 0.89/0.88/0.88/0.84/0.44）。
