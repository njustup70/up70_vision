# 项目进度

这份文档让每个成员随时知道：项目走到哪一步了、已经有什么、还缺什么、踩过哪些坑。

**维护约定**：阶段状态变化时更新本文；新踩的坑往第五节加一条（写清现象 / 原因 / 处理）；改动跟着当次提交一起进去。

> 最后更新：2026-10-14

---

## 一、当前阶段

**M1 环境标准化刚完成，M2 核心算法实现尚未开始。**

仓库现在是一个「能跑通 OpenCV、但没有任何视觉算法」的分层骨架：新人 clone 下来三条命令能验证环境，但 `src/up70_vision/` 里只有 8 个空 `__init__.py`。

---

## 二、里程碑总览

| 里程碑 | 状态 | 内容 | 阻塞条件 |
|---|---|---|---|
| M0 仓库初始化 | ✅ 完成 | 目录规范、工程元文件、配置层参数面 | — |
| M1 环境标准化 | ✅ 完成 | Docker 化：镜像 + compose + devcontainer | — |
| M2 核心算法实现 | ⬜ 未开始 | `core/` 判色与管线、`io/` 输入源、`viz/` 落盘、`config/` 加载、`cli/` 编排 | 无，可立即开始 |
| M3 回归测试 | ⬜ 未开始 | 重建 `tests/`，断言带具体数字 | 建议与 M2 同步建，别等算法写完再补 |
| M4 ROS 2 对接 | ⬜ 阻塞 | 话题适配器 + 发布节点 + 自定义消息 | 机械臂组 / 电控组定话题名与消息字段 |
| M5 模型接入 | ⬜ 阻塞 | YOLO 检测器 + 训练流水线 | 没有标注数据集、没有确定物块本体 |
| M6 真相机与场地标定 | ⬜ 阻塞 | 相机接入 + HSV 阈值按场地重调 | 相机型号未定、物块与场地未定 |

---

## 三、已完成

### M0 仓库初始化

- 分层骨架 `src/up70_vision/`：`config` / `core` / `core.detectors` / `io` / `tools` / `utils` / `viz` 共 8 个包。依赖方向单向（`utils`/`config` → `core` → `io` → `viz` → `cli` → `scripts`），每个 `__init__.py` 写明了该层放什么、不放什么、未来第一件事做什么
- 工程元文件：`pyproject.toml`（PEP 621 + src layout）、`requirements.txt`、`.gitignore`、`.gitattributes`、`.editorconfig`、`.github/PULL_REQUEST_TEMPLATE.md`
- 配置层目标参数面：`configs/default.yaml`、`configs/color_thresholds.yaml`。**当前无代码读取它们**，属"实现就位后的目标形态"

### M1 环境标准化

- `.devcontainer/Dockerfile`：底座 `ubuntu:22.04` + 系统 Python 3.10 + numpy / OpenCV / PyYAML，`pip install -e .` 可编辑安装
- `.devcontainer/docker-compose.yml`（CPU）与 `docker-compose-gpu.yml`（GPU 透传，用 `extends` 复用 CPU 定义，避免两份配置漂移）
- `.devcontainer/devcontainer.json`：VS Code Reopen in Container 一键进环境
- `.dockerignore`：排除 `.git`、缓存、运行产物、维护者个人笔记
- `scripts/demo.py`：OpenCV 冒烟测试（造图 → BGR2HSV → inRange → findContours → 叠加 → 写图）。**故意不 import `up70_vision`**，避免将来形成第二份「检测 → 判色」实现
- `scripts/check_env.py`：环境自检（Python 版本、opencv 发行包计数、依赖版本区间、图像读写往返、torch 可选提示），零 `src` 依赖

### 本地验证记录（2026-10-14）

- `check_env.py` → 全部通过；`demo.py` → PASS，退出码 0
- 8 个骨架子包全部可导入；`pyproject.toml` 经 setuptools PEP 621 读取正常
- `docker build --check` → no warnings found；`docker compose config`（CPU + GPU）→ 配置符合预期
- **尚未真正 build 过镜像**，只做了静态校验

---

## 四、进行中 / 下一步

**下一步：M2 核心算法实现**，建议按依赖顺序自底向上建：

1. `utils/logging.py` → `utils/image_io.py`（Ubuntu 上直接 `cv2.imread` / `cv2.imwrite`，不需要 Windows 兜底）
2. `core/types.py`（`Frame` / `Detection` 数据结构 + IoU，纯函数最容易验证）
3. `core/color.py`（HSV 判色，多段阈值，红色跨色相环绕）
4. `core/pipeline.py`（逐帧「检测 → 判色 → 匹配计分」）
5. `core/detectors/base.py` → `blob.py`
6. `io/base.py` → `synthetic.py` → `replay.py`
7. `viz/draw.py` → `writer.py`
8. `config/schema.py` → `loader.py`
9. `cli.py`（运行逻辑唯一实现处，`scripts/` 保持薄入口）

**同时建议**：M2 一动工就把 `tests/` 建起来（M3），不要等算法"写完"再补——判色错了不抛异常，只会让结果"看起来不太准"，没有断言具体数字就发现不了。

---

## 五、已知的坑

按「现象 / 原因 / 处理」记录，新踩到的往这里加。

### 环境类

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 1 | `ImportError: libGL.so.1: cannot open shared object file` | opencv-python 的非 headless wheel 需要 libGL，Ubuntu 默认不装 | `sudo apt-get install -y libgl1 libglib2.0-0t64`；镜像里已装好 |
| 2 | `cv2.imshow` 属性检查存在，调用时才报错 | 环境里同时有 `opencv-python` 与 `opencv-python-headless`（或 conda 的 opencv），两者顶层模块都是 `cv2`，谁后装谁覆盖 | 只保留一个：`pip uninstall -y opencv-python-headless`；`check_env.py` 有专项检查 |
| 3 | `CondaToSNonInteractiveError: Terms of Service have not been accepted` | 新版 Miniconda 的 `conda create` 要求先接受 Anaconda 服务条款 | 现已全面 Docker 化，本机不再需要 conda |
| 4 | `ModuleNotFoundError: No module named 'tomllib'` | `tomllib` 是 Python 3.11+ 才有 | 用 `setuptools.config.pyprojecttoml.read_configuration`，或第三方 `tomli` |

### 代码类

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 5 | `TypeError: can only concatenate list (not "tuple") to list` | `list + tuple` 在任何 Python 版本都非法（`check_env.py` 曾踩） | 用 `list + [0] * n`，或 `tuple(...) + (0,) * n` |
| 6 | `cv2.imwrite` 返回 `False` 但不抛异常 | OpenCV 少数"静默失败"接口之一 | 必须显式检查返回值，别假设成功 |
| 7 | `cv2.imread` 读不到文件返回 `None` 不抛异常 | 同上 | 显式判空再往下走 |
| 8 | 判色结果"看起来不太准"但不报错 | BGR 当 RGB 用不会抛异常，只会让红蓝全互换 | 图像变量一律带颜色空间后缀（`frame_bgr`）；判色前 `cvtColor(BGR2HSV)` |
| 9 | 红色判不出来 | HSV 的 H 范围是 0-179 不是 0-360；且色相在 0/179 处环绕，红色必须写两段阈值取并集 | 阈值表里红色写 `[0,10]` 与 `[170,179]` 两段 |
| 10 | 框内判色命中率偏低 | 判色分母用了 ROI 总像素，框里混入的地面背景稀释了命中率 | 分母用「有效像素」（饱和度/亮度达标的像素） |
| 11 | `cv2.putText` 画出问号且不报错 | Hershey 矢量字体不含汉字字形 | 图上一律用 ASCII 英文颜色名，中文说明只进日志 |

### 工具与工程类

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 12 | `docker build` 报找不到 Dockerfile | compose 的 `dockerfile` 路径是**相对 `context`** 解析的，不是相对 compose 文件所在目录 | Dockerfile 移目录后必须同步改这行 |
| 13 | `.dockerignore` 形同虚设 | Docker 规定它必须在构建上下文根部，放子目录会失效 | 保持在仓库根 |
| 14 | `fish: Unsupported use of '='` | fish 不支持 `VAR=value cmd` 语法 | 用 `set VAR value`；`for` 循环里也别用 `$(...)` 赋值 |
| 15 | `git add -A` 把个人笔记提交了 | 只想本地排除却写进了 `.gitignore`——它会向全队广播规则，且队友写同名文件会被静默忽略 | 个人文件用 `.git/info/exclude`，团队规则才用 `.gitignore` |
| 16 | 删了模块但 `__pycache__` 里还有它的 `.pyc` | Python 不会自动清理已删源文件的字节码缓存 | `find src -name __pycache__ -type d -exec rm -rf {} +` |
| 17 | `*.egg-info/SOURCES.txt` 列着已删文件 | `pip install -e .` 生成的元数据不会随源码变更自动同步 | 删掉 `src/up70_vision.egg-info` 后重装；该目录已被 gitignore |

---

## 六、环境状态

| 项 | 值 |
|---|---|
| 团队标准环境 | Docker 镜像：`ubuntu:22.04` + 系统 Python 3.10 |
| 镜像内容 | numpy / opencv-python / PyYAML，`pip install -e .` |
| 底座选择理由 | 战队标准发行版是 22.04；ROS 2 Humble 官方只支持 22.04 且自带 Python 3.10 |
| 本机开发环境 | Ubuntu 24.04.5 LTS；conda 环境 `up70`（Python 3.10.22 / numpy 2.2.6 / opencv-python 4.14.0.94 / PyYAML 6.0.3） |
| Docker | `/usr/bin/docker`，Compose v5.5.1，免 sudo |

---

## 七、明确不做（及原因）

| 任务 | 阻塞原因 |
|---|---|
| ROS 2 节点与话题 | 话题名与消息字段取决于机械臂组、电控组需求，先写必然返工 |
| YOLO 接入与训练流水线 | 没有标注数据集，没有确定物块本体（尺寸/材质/颜色会影响输入分辨率与增广策略） |
| 工业相机接入 | 型号未定，各家 SDK 差异过大 |
| 场上光照标定 | 等物块与场地确定后才能调 HSV 阈值 |
