# up70-vision

UP70 战队视觉任务仓库。当前阶段的目标是把 2D 视觉的基础链路跑通：**识别场上物体的颜色与位置**，为后续配合机械臂、接入激光融合打底。

算法主线是级联结构——先出候选框（后期换成 YOLO），再在框内做 HSV 判色。检测与判色两段互相独立，可各自替换。

---

## 开发环境约定

**本仓库统一用 Docker 提供环境**，不在本机装 conda / venv / 系统 Python 依赖。战队标准发行版是 Ubuntu 22.04，镜像底座因此定为 `ubuntu:22.04`——它自带 Python 3.10，也是未来 ROS 2 Humble 唯一支持的发行版。

| 项 | 约定 |
|---|---|
| 环境提供方式 | Docker，镜像定义在 `.devcontainer/Dockerfile` |
| 底座 | `ubuntu:22.04` + 系统 Python 3.10 |
| Python 依赖 | 唯一真源是 `pyproject.toml`，由镜像构建时安装 |
| 换行 | LF，由 `.gitattributes` 与 `.editorconfig` 强制，不要手工改 |

OpenCV 需要的系统库（`libgl1` / `libglib2.0-0`）已在镜像里装好，本机不用单独装。

## 快速开始

```bash
# 1. 构建镜像（只需一次）
docker compose -f .devcontainer/docker-compose.yml build

# 2. 跑环境自检
docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision

# 3. 跑 OpenCV 冒烟测试
docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision python scripts/demo.py
```

第 2 步末行应是 `结果：全部通过`；第 3 步应看到：

```
[自检] cv2 4.14.0 | numpy 2.2.6
[自检] 检出轮廓 1 个 | 叠加图已写出 output/demo/opencv_smoke.png
[自检] OpenCV 冒烟测试 -> PASS
```

退出码为 0，说明镜像里的依赖链完整、OpenCV 真的能跑。

> `scripts/demo.py` 故意写得又小、又不 import `up70_vision`——它是环境冒烟测试，不是业务实现。
> 这样将来 `src/` 里的实现就位后，不会和它形成第二份「检测 → 判色」逻辑。

源码通过 bind mount 挂进容器：**改宿主机上的代码在容器里立即生效，不用重建镜像**。只有改了 `pyproject.toml`（增删依赖）才需要重新 `build`。

VS Code 用户可以直接 Reopen in Container，配置就在 `.devcontainer/devcontainer.json`。

### GPU 机器

`.devcontainer/docker-compose-gpu.yml` 在 CPU 版基础上透传 NVIDIA GPU。当前阶段没有 CUDA 依赖，它是预留；等接 YOLO 训练、镜像里装上 cu128 的 torch 时才用得到。宿主机需先装 NVIDIA 驱动与 `nvidia-container-toolkit`。

```bash
docker compose -f .devcontainer/docker-compose-gpu.yml run --rm up70-vision python scripts/demo.py
```

### 真相机（海康 MV-CU013-A0UC）

这颗相机走 **USB3 Vision（GenICam），不是 UVC**：宿主机上不会出现 `/dev/video*`，`cv2.VideoCapture` 打不开它，也没有"把用户加进 `video` 组"这回事。接它的正确姿势是：

1. **宿主机配 udev 规则**给出 USB 访问权限（容器里改不了这个）：

   ```bash
   # /etc/udev/rules.d/99-hikrobot.rules
   SUBSYSTEM=="usb", ATTRS{idVendor}=="2bdf", MODE="0666", SYMLINK+="hik_camera"
   ```

   写完 `sudo udevadm control --reload-rules && sudo udevadm trigger`，然后**重新插拔相机**。

2. **把厂商 SDK 装进镜像**：MVS 是 deb 包（写不进 `pyproject.toml`），下载后放进 `.devcontainer/pkgs/` 再 `build` 镜像，Dockerfile 会自动装上（见该目录的 README）。

3. **给容器透传 USB**：`.devcontainer/docker-compose.yml` 里已配好 `devices: ["/dev/bus/usb:/dev/bus/usb"]`——传的是 USB 总线，不是 `/dev/video0`。

4. 跑起来：

   ```bash
   docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision \
     python scripts/run.py --set input.source=camera --limit 10
   ```

**相机是独占资源**：同一时刻只有一个程序能用它（MVS 客户端、实时调参窗口、本仓库的 `camera` 源互斥），打不开时先关掉别的占用者。

### pip 源

镜像构建默认用阿里云源（国内快）。太慢或报 `Could not find a version that satisfies the requirement` 时，用 build-arg 换源，不用改 Dockerfile：

```bash
docker compose -f .devcontainer/docker-compose.yml build --build-arg PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple
```

### 为什么是 Python 3.10

未来视觉得跑在 ROS 2 Humble 的容器里，Humble 官方只支持 Ubuntu 22.04，其自带 Python 3.10。开发环境跟部署环境对齐，能避免"本地跑得好好的、一进容器就报语法错"。因此仓库代码不使用 3.11+ 的专属语法，`requires-python` 下限保持在 3.10。

---

## 当前状态：核心算法与回归测试已就位

`src/up70_vision/` 已实现：`core/`（判色与管线、`Detection`/`ArucoMarker` 两种检出物）、`io/`（合成图 / 回放 / 真相机 MVS 源）、`viz/`（画图与落盘）、`config/`（加载与校验）、`cli.py`（编排）、`tools/hsv_tuner.py`（交互调阈值）、`core/detectors/stub_aruco.py`（按契约返回假码的 stub），并配了 **36 项**回归测试（`tests/`）。接口契约与"两种返回值"见 `docs/interface.md`；进度、验证数字与踩过的坑见 `docs/progress.md`。

**还没做**：ROS 2 对接（M4）、YOLO 接入（M5）、场地标定与阈值重调（M6）——都等外部条件确定，原因见 `progress.md` 第七节。

删掉了什么、哪些契约不能破，记录在维护者本地的设计笔记里（`docs/removed_and_design_notes.md`，**属个人文档，不进版本库**）。团队协作以下方「团队约定」与各子包 `__init__.py` 的职责说明为准。

依赖方向是硬约束，下层不知道上层存在：

```
utils ┐
      ├─> core ─> io ─> viz ─> cli ─> scripts/
config ┘
```

`core/` 不 import ROS、不读文件、不开窗口——判色对不对，要能在没有相机、没有机器人的情况下几秒钟验出来。

---

## 目录导览

| 路径 | 放什么 |
|---|---|
| `src/up70_vision/core/` | 纯算法层：判色、管线、检测器（blob 占位检测器；YOLO 待接） |
| `src/up70_vision/io/` | 输入源：合成图、回放、真相机（海康 MVS） |
| `src/up70_vision/viz/` | 可视化与落盘：画框、写 JSON/CSV |
| `src/up70_vision/config/` | 配置加载与校验 |
| `src/up70_vision/utils/` | 通用工具：图像读写、日志 |
| `src/up70_vision/tools/` | 人用的工具脚本：HSV 交互调参（`hsv_tuner.py`） |
| `configs/` | 全部可调参数。阈值一律外置到 YAML，不硬编码在代码里 |
| `scripts/` | 可直接运行的入口：`run.py`（主入口）、`demo.py`（OpenCV 冒烟）、`check_env.py`（环境自检） |
| `tests/` | 回归测试：断言带具体数字，可用 `python tests/test_xxx.py` 独立运行（镜像里没有 pytest） |
| `.devcontainer/` | 容器化配置全在这：`Dockerfile`、`docker-compose.yml`、`docker-compose-gpu.yml`、`devcontainer.json`、`pkgs/`（本地投放的安装包，如 MVS 的 deb） |
| `docs/` | `progress.md`：项目进度与已知的坑；`代码注释规范.md`：注释写到什么程度；`提交信息规范.md`：git 提交信息怎么写（维护者的设计笔记在本地，不进版本库） |
| `data/` | 数据集与回放素材（内容不进版本库） |
| `assets/` | 资源落点：模型权重、字体、示例图（内容不进版本库） |
| `output/` | 运行产物（不进版本库） |
| `.dockerignore` | 构建上下文排除项，避免把 `.git`、缓存、本地笔记打进镜像 |

---

## 团队约定

- **改 `src/` 前先跑 `python scripts/demo.py` 与 `python scripts/check_env.py`**，两个都通过再提交；有算法改动时再加跑全部回归测试（`tests/test_*.py`，共 **36** 项，用 `python` 直接跑，不依赖 pytest）：

  ```bash
  for t in tests/test_*.py; do python "$t"; done
  ```

- 命名：模块/函数/变量小写下划线，类大驼峰，常量全大写；图像变量必须带颜色空间后缀（如 `frame_bgr`）。
- 注释按 `docs/代码注释规范.md`：每个模块/类/函数（含私有）都要 docstring，类属性与配置字段要说明含义和取值；注释写「为什么、边界、症状」，不复述代码。
- 提交：中文祈使句，一句话说清改了什么；`main` 分支始终可运行。
- 图像一律按 OpenCV 的 BGR 顺序处理；检测框用绝对像素坐标，右/下边界为开区间（半开上界）。
- 新增算法功能要配「断言带具体数字」的验证，不要写 `assert x > 0`——判色错了不会抛异常，只会让结果"看起来不太准"。
- 判色的分母是「有效像素」（饱和度/亮度达标的像素），不是 ROI 总像素——否则框里混入的地面背景会稀释命中率。
- HSV 的 H 范围是 0-179，不是 0-360；红色必须写两段阈值（如 `[0,10]` 与 `[170,179]`）再取并集，因为色相在 0/179 处环绕。
- `core/` 不 import ROS、不读文件、不开窗口；`scripts/` 保持薄入口，业务逻辑唯一实现在 `src/up70_vision/cli.py`。
- 图上一律用 ASCII 英文颜色名：`cv2.putText` 用的 Hershey 字体不含汉字字形，传中文会画出问号而且不报错。
