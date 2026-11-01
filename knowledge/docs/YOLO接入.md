# YOLO 接入：现在能干什么、怎么跑、怎么优化

> 写给队友看的版本。目标是：**不装 torch、不改代码，就能跑 2 类检测**；
> 以及把"为什么 0 个框""为什么全是 unknown""窗口为什么卡"这类高频疑问一次说清。
>
> 训练数据、指标与复现命令在 [`YOLO训练与评测记录.md`](YOLO训练与评测记录.md)，本文只讲**怎么用**。

## 一、一句话结论

仓库里已经有一个可用的 YOLO 检测器（`detector.name=yolo`）：**ONNX 权重 + OpenCV `cv2.dnn`**，
容器里不需要 torch、不需要 GPU。当前部署权重仍是 **2 类真图模型 `yolov8n_mix4`**（同分布指标好，
但 v2 test 有相邻帧泄漏）；无泄漏的 **v3 基线 `yolov8n_v3`** 已训好，诚实指标是
P 0.813 / R 0.824 / mAP@0.5 0.850，换部署前要先在 v2 回归上确认不退化：

| 类别 | 是什么 |
|---|---|
| `block` | 350 mm 单色方块 |
| `top` | 200 mm 半红半蓝塔顶（红蓝上下叠放） |

| 模型 | test | P | R | mAP@0.5 |
|---|---|---|---|---|
| mix4（同分布） | v2 test 36 图/168 框 | 0.971 | 0.939 | 0.970 |
| v3 基线（无泄漏） | v3 test 52 图/173 框 | 0.813 | 0.824 | 0.850 |
| **v3_aug2（90° 旋转）** | **v3 test 52 图/173 框** | **0.912** | **0.806** | **0.885** |

mix4 训练时见过 v3 test 的 g3/g4，**不能用 mix4 在 v3 test 上的高分做泛化结论**；
当前推荐 `knowledge/models/yolov8n_v3_aug2.onnx`（类别错 18→6，代价是 FP 变多）。
颜色不由模型负责（HSV 判），距离不由 2D 负责（交给深度相机）。

## 二、三条路，为什么还是 ONNX + `cv2.dnn`

| 路线 | 需要装什么 | 速度（本机，1280×1024 一帧） | 镜像增量 | 用途 |
|---|---|---|---|---|
| A. `ultralytics` + torch（CPU） | torch 1.4 GB | 17~20 ms | +1.4 GB | 训练 + 推理 |
| B. `ultralytics` + torch（GPU） | 驱动 + 工具链 | 5~10 ms | +7~8 GB | 训练 |
| **C. ONNX + `cv2.dnn`（已采用）** | **什么都不用装** | **18~40 ms（取决于输入 480/640）** | **+0**（只有 12.7 MB 权重） | 只推理 |

相机最高 25 fps；路线 C 在 480 输入下约 18 ms，余量足够。训练仍单独在开发机的
`~/.venvs/yolo` 里做，导出的 ONNX 喂给同一段代码——**换模型不换代码**。

## 三、怎么跑（三步，照抄就行）

### 第 1 步：拿到 ONNX 权重（一台机器做一次）

权重不进版本库。两种来源：

```bash
# A. 队友直接给你两个文件（最快）：放到 knowledge/models/ 下，两个必须同名
#    yolov8n_mix4.onnx  +  yolov8n_mix4.names.json

# B. 自己从训练权重导出（需要 ultralytics，本机在 ~/.venvs/yolo）
cd "/home/sxy/桌面/vision opencv/up70_vision-main"
~/.venvs/yolo/bin/python lab/export_yolo.py \
  --weights runs/detect/mix4/weights/best.pt --imgsz 640 --name yolov8n_mix4
```

### 第 2 步：跑实时窗口（推荐）

```bash
cd "/home/sxy/桌面/vision opencv/up70_vision-main"
PYTHONPATH=. ~/.venvs/vision/bin/python -m vision live \
  --set detector.name=yolo \
  --set detector.yolo.weights=knowledge/models/yolov8n_mix4.onnx \
  --set input.source=camera --set input.camera.exposure_us=20000 \
  --max-fps 5
```

窗口里：框 + `block/red 0.91` 这样的标签（**类别/颜色/置信度**），左上角 HUD 有检出数和 fps。
按 `q` / ESC 退出。

> `--max-fps` 是必调项：640 输入单帧约 40 ms，20 fps 追着跑会把 CPU 吃满，窗口表现为卡死。
> 只想看识别对不对用 3~5，想顺一点用 6~10。`--max-fps 0` 才是不限帧。

### 第 3 步：批处理 / 离线复核

```bash
# 回放一个图片目录，落盘叠加图 + JSON + CSV
python -m vision run --set detector.name=yolo \
  --set detector.yolo.weights=knowledge/models/yolov8n_mix4.onnx \
  --set input.source=replay --set input.replay.path=data/yolo_real_v2/images/test
```

产物：`output/demo/frame_*.png`、`results.json`（每帧 bbox / class_name / color / score）、
`detections.csv`、`summary.json`。

## 四、配置项（`configs/default.yaml` 的 `detector.yolo` 段）

| 键 | 人话 | 默认 |
|---|---|---|
| `weights` | ONNX 权重路径（相对仓库根） | `knowledge/models/yolov8n.onnx`（COCO 导出，需换成 mix4） |
| `conf` | 置信度门槛。调低 → 框变多、误检也变多 | 0.45 |
| `iou` | NMS 去重门槛 | 0.45（实测 0.30~0.55 对结果几乎无影响） |
| `imgsz` | 模型输入边长，**必须与导出时一致** | 640 |
| `max_detections` | 一帧最多留几个框 | 32 |
| `classes` | 只保留哪些类别（如 `[0]` 只看 block）；空 = 全部 | `[]` |
| `class_names` | 类名表；空 = 读 sidecar / ONNX 元数据 | `[]` |
| `device` | 目前只支持 `cpu` | `cpu` |

命令行走 `--set detector.yolo.conf=0.5`；**换 480 权重时 `weights` 和 `imgsz` 必须一起改**，
否则 letterbox 与模型输入对不上，框会整体偏。

## 五、性能：480 是本机最优解（2026-11-01 实测）

同一份 `best.pt`，只改导出/推理输入尺寸：

| 输入 | ultralytics val mAP@0.5 | mAP@0.5:0.95 | `cv2.dnn` 单帧（本机） | 结论 |
|---|---|---|---|---|
| 640 | 0.970 | 0.846 | **39 ms** | 精度基准，离线复核用 |
| **480** | **0.969** | 0.843 | **18 ms** | **实时推荐**（精度几乎不变，快 2.2 倍） |
| 416 | 0.961 | 0.825 | ~13 ms | 精度开始掉，不推荐 |

导出与运行：

```bash
~/.venvs/yolo/bin/python lab/export_yolo.py \
  --weights runs/detect/mix4/weights/best.pt --imgsz 480 --name yolov8n_mix4_480

python -m vision live --set detector.name=yolo \
  --set detector.yolo.weights=knowledge/models/yolov8n_mix4_480.onnx \
  --set detector.yolo.imgsz=480 --set input.source=camera --max-fps 8
```

> 480 的代价：test 里 `g3tilt` 组少检 1 个（6→5 TP），整体 FP 从 6 降到 4。
> 36 张 test 的噪声约 ±3%，换场地/换物块后要重新比一次 640 vs 480。
>
> 可选：如果必须保 640 精度又要低延迟，可装 `onnxruntime`（本机实测 640 约 24 ms，
> 比 `cv2.dnn` 快约 1.6 倍），但容器要多一个依赖，目前**不推荐**为它改镜像。

## 六、类别与颜色怎么分工

- **YOLO 只负责 `block` / `top`**——不学红/蓝。理由：`top` 是红蓝各半，同一类本来就有两种颜色；
  把颜色塞进类别会让数据翻倍，还会因为光照/白平衡漂移而失效。
- **颜色由 HSV 判**（`configs/color_thresholds.yaml`）：
  - `block` 判整框；
  - `top` **先判上半块**；上半块判不出时，按红蓝像素的**行中位数**判“谁在上面”
    （红砖在上 = `red`，不按哪种颜色面积大）；红蓝左右并排/相机横滚时没有上下关系，
    记为 `mixed` 或 `?`。
- 画框颜色：`top` 固定品红（避免"同一类有时红有时蓝"看起来像 bug），`block` 按判出的红/蓝；
  标签里仍然写 `top/red`，两个信息不冲突。

## 七、四个高频疑问

### 1. 为什么我们自己的图上「0 个框」？

先确认用的是 **mix4 权重**，不是 COCO 的 `yolov8n.onnx`。COCO 的 80 类里没有色块，
模型没见过就不输出——那不是接入失败。`check` 方法：用 COCO 的 `bus.jpg` 应能检出
`person/bus`；换成我们自己的图应检出 `block/top`。

### 2. 为什么带框图上写着 `unknown`？

`unknown` 是**颜色**不是类别：框里没有命中红/蓝阈值（过曝、侧视、只看到白边）时，
`color = None`，标签显示 `block/?`。它表示"类别可信、颜色不确定"，
下游拿到 `?` 应当停手，不要猜颜色。

### 3. 为什么窗口会卡死/点一下就退？

CPU 被 YOLO 吃满，Qt 事件循环抢不到时间片。按顺序试：

1. `--max-fps 4`（最有效）；
2. 换 480 权重（第五节，耗时减半）；
3. 关掉其它占 CPU 的程序；
4. 仍然卡就 `--headless` 落盘，事后看叠加图。

### 4. 报错 `YOLO 权重不存在` 怎么办？

按第 1 步导出/拷贝。报错信息里也印了导出命令。

## 八、已知问题（都是"数据问题"，不是代码问题）

2026-11-01 在 test 上逐张复核（`lab/eval_errors.py`）：TP 161 / FP 6 / FN 7、类别错 1、合并 2。

| 现象 | 数量 | 说明 / 下一步 |
|---|---|---|
| FN 在画面边缘 | 4~5 | 物块被画幅截断，模型不敢框；补"边缘截断"样本 |
| 远景同色叠放 | 2 | 一个框装下两块；补远景/遮挡样本，或试 960 输入（更慢） |
| 环境误检 | 6 | 门框、地面反光、边缘碎片；补空场负样本 / 提高 conf |
| 类别错 | 1 | 极端视角 top↔block；接深度后用绝对尺寸区分，或加二阶段分类 |

test 只有 36 张（g6far 占 123/168 框），以上数字的置信区间很宽；
**扩到 ≥80 张、按采集组分层后再对外报指标**。

## 九、要不要继续训练/优化

优先级从高到低（详见训练记录第 5 节）：

1. **补难样本**：边缘截断、远景叠放、空场负样本——比调参收益大；
2. **扩 test**：现在 36 张太少，指标波动大；
3. **部署 480**：实时链路首选，精度几乎不变；
4. **换更大模型**（yolov8s / yolo11s）：mAP 可能 +1~2 点，CPU 耗时约翻倍，比赛前再权衡；
5. **二阶段分类器**（block/top 混淆时）：等深度接入、能用绝对尺寸后再做。

训练/评测命令见 [`YOLO训练与评测记录.md`](YOLO训练与评测记录.md) 第 3 节，
本文不重复。

## 十、许可提醒（重要）

YOLOv8 的代码与权重都是 **AGPL-3.0**。队内自用没问题；如果程序要**对外发布**
（比赛系统提交、开源、交付），AGPL 要求一并提供源码。正式交付前请跟队长确认，
或改用 Apache-2.0 / MIT 的模型（YOLOX、NanoDet 等）——接口一样挂在这个位置。

## 十一、文件在哪

| 东西 | 位置 |
|---|---|
| 检测器实现 | `vision/yolo/detector.py` |
| 工厂与注册 | `vision/cli/run.py` 的 `build_detector`（按名字直接建） |
| 配置与校验 | `configs/default.yaml` 的 `detector.yolo`、`vision/config/schema.py` |
| 导出脚本 | `lab/export_yolo.py` |
| 训练/评测记录 | `knowledge/docs/YOLO训练与评测记录.md` |
| 权重与说明 | `knowledge/models/`（权重被 gitignore，只有 README 入库） |
| 测试 | `tests/test_yolo.py`（23 项，不需要权重也能跑） |
