# YOLO 训练与评测记录（v1，2026-10-28）

> 目的：让**别人能复现、能公平比较**。凡是"换个人换个时间会变"的量都记在这里；
> 实测过程与坑放在 [`位姿估计真机测试计划.md`](位姿估计真机测试计划.md) 与 `progress.md`，本文只留结论与参数。
>
> 一句话结论：**2 类检测器（`block` / `top`）在 17 张真图测试集上 P 0.954 / R 0.778 / mAP@0.5 0.874**；
> 颜色不由模型负责（HSV 判），距离不由 2D 负责（交给深度相机）。

---

## 1. 数据集与实验条件

### 1.1 数据集版本与规模

| 项 | 值 |
|---|---|
| 版本 | `yolo-real-v1`（2026-10-28 采集与标注）+ `yolo-synth-v1`（规则 V2 渲染） |
| 真图总数 | **120 张**（1280×1024，JPEG） |
| 真图划分 | train **86** / val **17** / test **17** |
| 真图实例数 | `block` **203** 个、`top` **165** 个（合计 368） |
| 合成总数 | 300 张（train 240 / val 60），`block` 267 个、`top` 183 个 |
| 混合训练集 | 合成 300 + 真图 86 = **386 张**（val = 合成 60 + 真图 17） |

> 类别定义：`block` = 350 mm 单色方块（红或蓝），`top` = 200 mm 半红半蓝塔顶。
> **颜色不进类别**——红/蓝由 HSV 判（`configs/color_thresholds.yaml`）。

### 1.2 数据来源（拍摄条件）

| 项 | 值 |
|---|---|
| 相机 | 海康 MV-CU013-A0UC（USB3 Vision，MVS SDK），1280×1024 |
| 场景 | 办公室走廊 + 地面黑白地贴；白墙、玻璃门、门框、反光地面均在画面内 |
| 距离 | 约 0.4–2.5 m（含近景大目标与远景小目标） |
| 光照 | 白天自然光 + 室内灯；白墙常过曝，**曝光按场景取 6–28 ms** |
| 角度 | 俯视、平视、斜视 30–60°，含红蓝块叠放/并排/遮挡 |
| 采集批次 | 5 组（g1 正对 / g2 塔顶 / g3 斜视 / g4 叠放 / g5 距离梯度） |
| 真实机器人环境 | **无**（实验室地面摆放，非比赛场地） |

### 1.3 数据划分方法（防泄漏）

- **按采集批次分层抽样**：每组各自取 15% → test、15% → val、70% → train；
- 目的：避免同一姿势的连拍帧同时进训练集和测试集（那会让 mAP 虚高）；
- 随机种子固定为 `20261028`，划分结果落在 `data/yolo_real2/`；
- 标注工具与预填：`output/label_tool.py`（3 类预填 → 合并成 2 类），人工逐张核过。

### 1.4 模型与训练

| 项 | 值 |
|---|---|
| 模型 | **YOLOv8n**（ultralytics 8.4.x） |
| 预训练权重 | `knowledge/models/yolov8n.pt`（COCO） |
| 输入尺寸 | 640×640（**ONNX 静态导出只支持 640**，320/480 会崩） |
| 训练轮数 | 40 epoch，batch 8，CPU，workers 2，optimizer auto，lr0 0.01 |
| 数据增强 | ultralytics 默认：mosaic 1.0、translate 0.1、scale 0.5、fliplr 0.5、HSV h/s/v = 0.015/0.7/0.4 |
| 离线增强 | `lab/augment_dataset.py`（翻转/90–270° 旋转/缩放平移 + 亮度扰动，**不改色相**）；本轮训练**未启用**，仅验证过标签正确性 |

### 1.5 评价设置

| 项 | 值 |
|---|---|
| 测试集 | `data/yolo_real2` 的 `test` 划分，**17 张 / 54 个框** |
| 置信度阈值（报告中主值） | **0.45**（与部署默认 `detector.yolo.conf` 一致） |
| NMS IoU | 0.45（部署）；训练时 ultralytics 内部 val 用 0.7 |
| 匹配 IoU | 0.5（mAP@0.5）、0.5:0.95（mAP@0.5:0.95） |
| 类别匹配 | 必须类别一致才算 TP（不跨类匹配） |

### 1.6 运行环境

| 项 | 值 |
|---|---|
| 训练 | 本机 CPU（Intel i7 级别，无 GPU），`device=cpu` |
| 推理 | `cv2.dnn` + ONNX（FP32），1280×1024 输入缩放至 640 |
| 推理耗时 | **≈43 ms/帧**（CPU，单线程 dnn；约 23 fps） |
| 依赖 | opencv 4.11（宿主 venv）/ 4.14（容器）均可，**不需要 torch** 推理 |

---

## 2. 识别效果指标

### 2.1 总体（真图 test，conf 0.45）

| 指标 | 值 |
|---|---|
| Precision | **0.954** |
| Recall | **0.778** |
| F1（conf 0.45，IoU 0.5） | **0.871**（按 TP/FP/FN 复算） |
| mAP@0.5 | **0.874** |
| mAP@0.5:0.95 | **0.671** |

> mAP 是**跨置信度阈值**的综合指标；P/R/F1 与 TP/FP/FN 是**固定工作点（conf 0.45）**的结果，两者不要混着比。

### 2.2 每类（真图 test）

| 类别 | 实例数 | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| `block` | 27 | 0.907 | 0.815 | 0.884 | 0.739 |
| `top` | 27 | 1.000 | 0.741 | 0.863 | 0.604 |

**`top` 是短板**：精确率满分（没误检），但召回只有 0.74 —— 半红半蓝的塔顶在远景/贴边时容易漏。

### 2.3 TP / FP / FN（真图 test，IoU 0.5）

| 置信度 | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| 0.25 | 45 | 4 | 9 | 0.918 | 0.833 | 0.874 |
| 0.35 | 44 | 3 | 10 | 0.936 | 0.815 | 0.871 |
| **0.45（部署）** | **44** | **3** | **10** | **0.936** | **0.815** | **0.871** |
| 0.55 | 44 | 2 | 10 | 0.957 | 0.815 | 0.880 |
| 0.65 | 43 | 0 | 11 | 1.000 | 0.796 | 0.887 |

> 注：本表在 **17 张 test** 上按"类别一致 + IoU≥0.5"逐个匹配统计（一目标只配一个预测）；
> Ultralytics 的 P/R（0.954 / 0.778）口径略有差异（跨阈值积分与匹配细节），因此数值不完全相同。

### 2.4 对照实验（同划分、同评价口径）

| 训练数据 | mAP@0.5 | 说明 |
|---|---|---|
| 只用合成 300 | 0.730 | 基线；`top` 尚可（0.734），`block_red` 弱（0.517） |
| 合成 + 自动标注真图（3 类） | 0.659 | **错误标签把模型带坏**，已废弃 |
| **合成 + 人工标注真图（2 类）** | **0.874** | 本版；`block` 0.884 / `top` 0.863 |

### 2.5 已知问题（按影响排序）

| 问题 | 证据 | 改进方向 |
|---|---|---|
| `top` 召回偏低（0.74） | 远景/贴边塔顶漏检 | 补拍小目标与遮挡场景 40–60 张 |
| 环境误检（门框、深色板） | 120 张全量：FP 11 / 331 TP（conf 0.45） | 加 **20–30 张空场景负样本**重训 |
| 重复框 | 少数帧同一目标出 2 个重叠框 | 把 `detector.yolo.iou` 从 0.45 降到 0.3 |
| 大目标框偏小 | 合成 800 mm 测试中，模型框小于真值框 | 训练集补"目标占画面 >50%"的样本 |

---

## 3. 复现命令

```bash
cd "/home/sxy/桌面/vision opencv/up70_vision-main"

# 1) 训练（数据：合成 300 + 真图 86；约 25–40 分钟 CPU）
YOLO_CONFIG_DIR=runs/.cfg ~/.venvs/yolo/bin/yolo detect train \
  model=knowledge/models/yolov8n.pt data=data/yolo_mix2/dataset.yaml \
  epochs=40 imgsz=640 batch=8 device=cpu name=mix2 exist_ok=True

# 2) 在真图 test 上评（表 2.1 / 2.2 的来源）
YOLO_CONFIG_DIR=runs/.cfg ~/.venvs/yolo/bin/yolo detect val \
  model=runs/detect/mix2/weights/best.pt data=data/yolo_real2/dataset.yaml \
  split=test imgsz=640 device=cpu plots=True name=mix2_val

# 3) 导出 ONNX（运行时用，无需 torch）
PYTHONPATH=. ~/.venvs/yolo/bin/python lab/export_yolo.py \
  --weights runs/detect/mix2/weights/best.pt --out knowledge/models --name yolov8n_mix2

# 4) 部署推理（2 类 + HSV 判色）
PYTHONPATH=. ~/.venvs/vision/bin/python -m vision run \
  --set detector.name=yolo --set detector.yolo.weights=knowledge/models/yolov8n_mix2.onnx
```

产物位置：权重 `runs/detect/mix2/weights/`，曲线与混淆矩阵 `runs/detect/mix2_val/`，
模型 `knowledge/models/yolov8n_mix2.onnx`（+ 同名 `.names.json`，两个文件必须一起拷）。

---

## 4. 记录模板的使用说明

以后换数据/换模型重训，**只改数字、不改结构**，并保留上一版一节（便于横向比较）：

- 数据集变了 → 更新 1.1 / 1.2 / 1.3 与 2.4；
- 训练参数变了 → 更新 1.4；
- 评价口径变了（conf/IoU）→ **必须在 2.1 下方注明**，否则数字不可比。
