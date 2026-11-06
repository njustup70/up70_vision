# YOLO 2D 与主逻辑联调说明

> 基线：2026-11-04，分支 `feat/v4新场次`（`acdd2fc`）。正式字段以[接口契约](../../vision/docs/interface.md#2d-检测输出请求-响应接口2026-11-04)与 `Detection.to_dict()` 为准，本文只回答「怎么装、怎么跑、怎么验、什么没验」。
> 分工：本文只管 **2D 请求-响应**；RGB-D 与 ROS 话题看[视觉与整机联调说明](整机联调说明.md)。

## 1. 视觉交付什么

因为主逻辑在另一台电脑、由另一人维护，2D 侧只承诺一件事：

> 给一帧 BGR 图，返回这一帧里每个目标的 `bbox`、`class_name`、`color`、`score`。

| 接口 | 形态 | 状态 | 什么时候用 |
|---|---|---|---|
| `POST /detect` | HTTP，JSON 进 JSON 出 | ✅ 已实现（`lab/serve_2d.py`），本机端到端 44 ms | 主逻辑在另一台机器（部署形态） |
| `vision.api.detect()` | Python 进程内 | ✅ 已实现并实测 | 单帧脚本、双方对拍；逐帧请按 3.1 复用实例 |

四条最容易被理解错的语义（详见契约 C、F 节）：

1. **`color` 为 `null` 不是「没有颜色」**，是「判不出来」；`mixed` 表示红蓝左右横放。
2. **`score` 是 YOLO 置信度**，不是正确率、不是抓取成功率。
3. **`class_name` 是模型输出**，判色不改变类别；单色 `top` 可能被模型判成 `block`。
4. **没有距离、没有绝对尺寸、没有跨帧 ID**——这三样由深度和主逻辑负责。

## 2. 环境与准备

2D 侧只需要 Python + OpenCV + NumPy + PyYAML，**不需要 torch，也不需要 ROS**。

### 2.1 环境（已验证）

| 位置 | Python | OpenCV | NumPy | PyYAML | 模型 |
|---|---|---|---|---|---|
| 统一镜像 `up70-vision:humble`（#29） | 3.10.12 | 4.11.0 | 1.26.4 | 6.0.x | ⚠️ 取决于构建机（见 2.5） |
| 开发 venv `~/.venvs/up70-2d` | 3.10.12 | 4.11.0 | 1.26.4 | 6.0.2 | ✅ 本地有 |
| 系统 `python3` | 3.10.12 | **4.5.4** | 1.21.5 | 5.4.1 | ⚠️ 不达标，别用 |

镜像由 Dockerfile 显式钉住 NumPy / OpenCV，本机 venv 由 `requirements.lock` 钉住，两边版本一致：

```
numpy==1.26.4
opencv-python==4.11.0.86
PyYAML==6.0.2
```

**别装 `opencv-contrib-python` 或 `-headless`**：三个包顶层都叫 `cv2`，同时装会互相覆盖且不报错
（`python -m vision doctor` 会检查"opencv 发行包数量 = 1"）。

### 2.2 一键装环境

```bash
python3 -m venv ~/.venvs/up70-2d
~/.venvs/up70-2d/bin/pip install -r requirements.lock
~/.venvs/up70-2d/bin/pip install --no-deps -e .        # 在仓库根执行
python3 -m vision doctor                                # 期望「全部通过」
```

**若这台机器 source 过 ROS 2**（本机就是），运行前先 `env -u PYTHONPATH` 或 `unset PYTHONPATH`：
ROS 的 `PYTHONPATH` 排在 venv 之前，`pytest` 会被 ROS 的 `launch` 包带崩（报 `No module named 'lark'`），
这不是本项目的问题，但表现完全一样。

### 2.3 模型

`.onnx` 与同名 `.names.json` **必须成对**，且都不随 Git 分发。模型清单、用途与 SHA-256 的
唯一来源是[模型权重说明](../models/README.md)——本机 2D 默认用 `yolov8n_mix4.onnx`（640 输入，
`0=block`、`1=top`），2D 当前最好的是 `yolov8n_v4.onnx`（v3 test mAP50 0.905）。

换模型 = 改 `detector.yolo.weights` + 更新 `models/README.md` 的 hash + 双方重跑 3.2 对拍；
响应里的 `model@sha256` 就是要对的东西，对不上先查版本，别先怀疑算法。

### 2.4 环境准备方式（与深度侧一致）

**不提供离线安装包**。深度侧的做法是"给文档、环境自己按文档建"——[整机联调说明](整机联调说明.md)
第 3 节原话：「新环境需按 [深度 README](../../vision/depth/README.md#环境与启动) 自行准备，
而不是假设此路径随仓库分发」。2D 侧照同一条：按 2.2 建 venv，版本由 `requirements.lock` 钉住，
不再额外打 wheel 包或做 U 盘交付。

这也意味着：**比赛机的 Python 环境由部署方负责**，2D 侧只保证"按 2.2 装完就能跑"。
`pyproject.toml` 只打包 `vision*`，`configs/` 与模型不在 wheel 里——所以在仓库根运行、
或显式传 `--config`，是唯一支持的方式（见 2.5 第二条）。

### 2.5 四个坑与现在的做法

| 坑 | 原来的症状 | 现在 |
|---|---|---|
| 镜像带不带模型 | `.dockerignore` 排除 `*.onnx` / `*.names.json`，镜像里没有模型时一跑就 `FileNotFoundError` | 已加 `!` 例外，**只放行默认模型 + 480 降级版**（整个目录会白带 5 个历史权重、98 MB）。但 #29 之后镜像是全队共用的 `up70-vision:humble`，`COPY knowledge/` 复制的是**构建机上有的文件**：有就带上、没有也照常构建成功，所以**不能假设镜像一定有模型**。开发走 compose 挂载宿主目录；部署时 `-v <models>:/opt/vision/knowledge/models:ro` 挂一份最稳 |
| 环境里缺 `configs/` | 只 `pip install` 或换目录运行，找不到 `color_thresholds.yaml` → `color` 全是 `null`，**不报错** | `vision.api.detect` 现在会打 **WARNING**；调用方显式传 `config`，或在仓库根运行 |
| 版本漂移 | 本机同时存在 cv2 4.5.4 / 4.11 / 4.14 / 5.0，结果随解码差异漂移 | 新增 `requirements.lock`；Dockerfile 用 `-c requirements.lock` 约束。（未合并的 `feat/Humble容器支持` 分支也是同款钉法：NumPy 1.26.4 + OpenCV 4.11.0.86） |
| 本机 shell 被 ROS 污染 | `pytest` 导入 ROS 的 `launch` 报 `No module named 'lark'` | 运行前 `env -u PYTHONPATH`；2D 侧本身不依赖 ROS |

无桌面的机器不能开窗口；`libGL.so.1` 缺失会让 `import cv2` 直接失败（镜像里已装 `libgl1`，
原生装环境时要自己补）。

## 3. 启动与观察

### 3.1 进程内（当前唯一可跑的方式）

必须**构造一次、逐帧复用**；`vision.api.detect()` 每次都会重建检测器并重读 ONNX（实测 103 ms/帧 vs 复用 38 ms）。

```python
import numpy as np
from vision.config.loader import load_config, load_thresholds, resolve_thresholds_path
from vision.cli.run import build_detector
from vision.yolo.pipeline import ColorConfig, process_frame
from vision.types import Frame

# 一次。第二个参数把检测器切成 yolo——default.yaml 里默认是 blob，
# 直接取 cfg.detector.params 会拿到 blob 的参数去构造 yolo，报 TypeError。
cfg, cfg_path = load_config("configs/default.yaml", ["detector.name=yolo"])
th = load_thresholds(resolve_thresholds_path(cfg, cfg_path))  # 判色必需，一次
det = build_detector("yolo", cfg.detector.params)             # 一次；内部惰性加载 ONNX
cc = ColorConfig(min_saturation=cfg.color.min_saturation, min_value=cfg.color.min_value,
                 hit_ratio_min=cfg.color.hit_ratio_min,
                 ignore_border_px=cfg.color.ignore_border_px,
                 min_valid_pixels=cfg.color.min_valid_pixels)

det.detect(np.zeros((1024, 1280, 3), np.uint8))  # 预热：约 130 ms，把冷启动挡在比赛开始前

def infer(image_bgr, frame_id):                  # 主逻辑每帧调这个
    r = process_frame(Frame(image_bgr, index=frame_id, source="main"), det, th, cc)
    return [d.to_dict() for d in r.detections]   # 与契约 C 节的 detections[] 完全一致
```

不预热的话，第一帧要额外付 ~130 ms 的模型加载；模型路径写错也会拖到第一帧才暴露。

### 3.2 对拍观察（不用写代码）

对一批图片跑完整链路，产物就是双方比对的依据：

```bash
python -m vision run \
  --set detector.name=yolo --set input.source=replay \
  --set input.replay.path=<图片目录> --out output/dock_check
```

产物：`output/dock_check/results.json`（每帧一条，含 `detections[]`）、`detections.csv`、`summary.json`。
`input.replay.kind=auto` 同时接受图片目录与视频；加 `--limit N` 只跑前 N 帧。
在仓库根目录执行（或先 `pip install -e .`，否则 `python -m vision` 找不到包）。

### 3.3 HTTP 服务

```bash
# 起服务：默认只绑本机 127.0.0.1:8899。要给别的机器用加 --host 0.0.0.0（无鉴权，见 3.4）
~/.venvs/up70-2d/bin/python lab/serve_2d.py --port 8899

# 健康检查：模型 sha256、生效配置、预热耗时、队列深度、已服务/已丢弃计数
curl -s localhost:8899/healthz

# 发一帧（encoding 三选一：jpg / png / raw-bgr）
python3 - <<'EOF'
import base64, json, urllib.request
img = open("data/yolo_real_v4/images/test/g3tilt_frame_0000.jpg", "rb").read()
body = json.dumps({"api": 1, "frame_id": 42,
                   "image": {"encoding": "jpg", "data": base64.b64encode(img).decode()}}).encode()
req = urllib.request.Request("http://127.0.0.1:8899/detect", data=body,
                             headers={"Content-Type": "application/json"})
print(json.loads(urllib.request.urlopen(req).read()))
EOF
```

本机实测（1280×1024，loopback）：服务端推理中位数 **32.5 ms**，HTTP 端到端 **44.2 ms**
（含 JPEG 解码、base64 与 JSON），启动预热约 120 ms。

行为约定（与契约 3.4 / D、E 节一致）：

- **队列深度 1**：新帧到达时丢掉还没开始处理的旧帧，`/healthz` 的 `dropped` 计数可查；
- 等超过 `--timeout`（默认 200 ms）回 **504**，该帧标记取消，worker 取到直接跳过；
- 错误码：400（图/参数不合法）、413（超过 `--max-frame-bytes`，默认 8 MB）、500（推理异常）、504（超时）；
- 服务无鉴权，不加 `--host 0.0.0.0` 时只有本机能连。

### 3.4 部署拓扑与限制

- 服务只绑 `127.0.0.1` 或局域网，**无鉴权、单客户端**；单 worker、队列深度 1（新帧覆盖旧帧，宁可丢帧不要排队）。
- **比赛机与上位机不是同一台**：本机 1280×1024 稳态 38~39 ms/帧（含 letterbox + NMS，CPU），首帧 130 ms；480 输入约减半。这个数**在比赛机上不成立**，必须在目标机重跑。
- 25 fps 的预算是 40 ms，2D 侧**没有余量**；建议主逻辑按 10~15 Hz 调用。
- 单帧超时建议 200 ms，超时丢帧，**不重试**。

## 4. 验收边界与待对接

### 4.1 已知限制（写清楚，不当作待调参问题）

| 限制 | 依据 |
|---|---|
| 斜视召回偏低 | v3 的 g3tilt 组 R≈0.681；v4 整体 R 0.825 |
| 叠放目标被一个框合并 | v4 记录 15 处 |
| 图像边缘截断的块易漏 | v2 的 7 个 FN 里 4 个在边缘 |
| 单色 `top` 可能被判成 `block` | 2D 无绝对尺寸，像素高度等价（[3.5 节](YOLO2D数据泄漏与尺寸歧义.md)） |
| 颜色依赖光照 | 换场地必须重调 HSV 阈值与曝光（3~40 ms） |
| 逐帧抖动，无跟踪/平滑 | 现有「防抖」只针对 ArUco 位姿的 HUD 显示 |
| 许可 | YOLOv8 权重与代码为 AGPL-3.0，对外交付前需确认 |

精度口径只用去重后的：v4 test（g3tilt + g4stack，52 图 / 173 框）**P 0.937 / R 0.825 / mAP@0.5 0.905**。
v2 同分布 mix4 的 0.970 **不能当泛化指标**（test 有相邻帧跨 train）。

### 4.2 尚未验证

跨机结果一致性、比赛机延迟——两项还没有证据，不要按已验证对待。HTTP 服务端已在本机实测，但没在对方的机器上跑过。

### 4.3 主逻辑侧须明确（8 条）

1. **谁抓图**、送图的分辨率与编码（`png` / `jpg q≥95` / `raw-bgr`）；
2. **调用频率**（建议 10~15 Hz）；
3. **丢帧策略**（队列深度 1、新帧覆盖、不重试）；
4. **类别终判权**：单色 `top` 由谁按深度尺寸裁决；
5. 是否需要**跨帧稳定**（我们不做投票，需要就由主逻辑自己做）；
6. **时间基准**：用 `frame_id` 关联，不要用时间戳（两机时钟不同步）；
7. 是否需要**排序或去重**（我们只保证按 `score` 降序，不做跨类去重）；
8. **消费方式**：进程内 import 还是 HTTP。

## 5. 变更与责任

- **契约变更**：按《视觉组协作规范》第 2 条，PR 描述写「动了契约，动的是哪一条」，等对方同意再合；字段只增不改，删改一律升 `api` 版本号。
- **模型换版**：`.onnx` 与 `.names.json` 必须同换，更新本文第 2 节哈希与响应里的 `model`，然后双方重跑 3.2 对拍。
- **责任划分**：2D 给 bbox / class / color / score；深度给绝对尺寸；**最终类别由主逻辑按尺寸裁决**。
- **出问题先看三样**：`model` 哈希是否一致 → OpenCV 版本是否一致 → 3.2 对拍的差异是否在容差内（bbox ±1 px、`score` ±0.02，`class_name` / `color` 必须完全一致）。三样都一致才谈算法。
