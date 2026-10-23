# 踩过的坑（存档）

> 本文件是 `docs/progress.md` 第五节在 2026-10-22 的归档（仓库瘦身 P1-3）。
> **坑的编号没变**：代码注释、配置注释与提交信息里写的「progress.md 坑 N」，指的就是本篇的坑 N。
> 新增的坑请直接加到本篇末尾，按「现象 / 原因 / 处理」三栏写（协作规范第 5 条）。

---

按「现象 / 原因 / 处理」记录，新踩到的往这里加。

### 环境类

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 1 | `ImportError: libGL.so.1: cannot open shared object file` | opencv-python 的非 headless wheel 需要 libGL，Ubuntu 默认不装 | `sudo apt-get install -y libgl1 libglib2.0-0t64`；镜像里已装好 |
| 2 | `cv2.imshow` 属性检查存在，调用时才报错 | 环境里同时有 `opencv-python` 与 `opencv-python-headless`（或 conda 的 opencv），两者顶层模块都是 `cv2`，谁后装谁覆盖 | 只保留一个：`pip uninstall -y opencv-python-headless`；`check_env.py` 有专项检查 |
| 3 | `CondaToSNonInteractiveError: Terms of Service have not been accepted` | 新版 Miniconda 的 `conda create` 要求先接受 Anaconda 服务条款 | 现已全面 Docker 化，本机不再需要 conda |
| 4 | `ModuleNotFoundError: No module named 'tomllib'` | `tomllib` 是 Python 3.11+ 才有 | 用 `setuptools.config.pyprojecttoml.read_configuration`，或第三方 `tomli` |
| 17 | 海康 USB3 相机插上后 `ls /dev/video*` 没有任何新设备 | 它是 **USB3 Vision（GenICam）**，不是 UVC：数据走 libusb 的 `/dev/bus/usb/...`，根本不产生 v4l2 节点 | 别用 `cv2.VideoCapture` / 别往 `video` 组上靠；用厂商 SDK（MVS）或 Aravis 这类 USB3 Vision 库 |
| 18 | 相机能被 Aravis 识别、参数可读写，但**一帧都取不到**（`timeout_pop_buffer` 立即返回 None） | 该相机 U3V 的 `SIRM_PAYLOAD_SIZE` 恒为 1 MB 且不随 `SIRM_REQ_PAYLOAD_SIZE` 更新（实测把 ROI 调小也一样）；Aravis 按 1 MB 组帧，与真实 1.25 MB 帧长不符，帧被丢弃。Aravis 0.8.20 与自建 0.8.33、官方 `arv-camera-test-0.8` 均失败 | **改用官方 MVS SDK**（实测 28 fps 正常出帧）；Aravis 路线已排除 |
| 19 | 装完 MVS 后新终端里 Python 才能 import 成功 | `MvImport` 通过环境变量 `MVCAM_COMMON_RUNENV` 定位 `.so`，变量由 MVS 装到 `~/.bashrc` | 新开终端；或在脚本里 `os.environ.setdefault("MVCAM_COMMON_RUNENV", "/opt/MVS/lib")` |
| 20 | 别的程序打不开相机（`LIBUSB_ERROR_BUSY`） | 相机是**独占**资源：MVS 客户端 / 探针 / 实时窗口同时只能开一个 | 先关掉占用者再开 |
| 21 | 用 `get_trigger_source()` 看到 `Line0` 就以为"卡在触发模式" | `TriggerSource` 是保留值，`TriggerMode=Off` 时它照样有值 | 判断是否触发模式要看 **`TriggerMode`**，不是 `TriggerSource` |
| 22 | 想按"依赖唯一真源是 pyproject.toml"把 MVS 写进依赖 | MVS 是 **deb 安装包**，不是 pip 包，`dependencies` 表达不了 | 环境类依赖走 `.devcontainer/Dockerfile`；README 该补一句例外说明 |
| 23 | 宿主直接跑 `check_env.py` 报 4 项版本不符（numpy/cv2/yaml） | 宿主用的是 apt 版（numpy 1.21.5 / cv2 4.5.4 / yaml 5.4.1），而 pyproject 钉的是 pip 版区间 | 这属**预期**：标准环境是容器。要在宿主验证就 `docker compose run --rm up70-vision`；本项目代码已实测在两种版本下都正常 |
| 24 | 容器里跑相机源，MVS 报"没找到相机"，但宿主 `lsusb` 明明看得到 | 容器默认看不到宿主 USB 总线 | compose 已配 `devices: ["/dev/bus/usb:/dev/bus/usb"]`；权限（udev 规则）必须在宿主机给 |
| 25 | 在受限开发沙箱 / 精简容器里 MVS 枚举不到相机，可 `lsusb` 又列得出设备 | 沙箱的 `/dev` 是精简 devfs（**没有 `/dev/bus/usb`**），而 `lsusb` 走 sysfs 照样列设备——两者结论会打架 | 判断"相机到底可用吗"必须在**能看到 `/dev/bus/usb` 的环境**里跑（容器加 `--device`，或宿主机终端），别信 `lsusb` 单方面 |
| 26 | 镜像里相机打不开：枚举不到设备 / 权限错 | `libMvCameraControl.so` 的直接依赖只有 glibc 与 libstdc++，**libusb 是运行时 dlopen 的**，而 `ubuntu:22.04` 基础镜像没装它 | Dockerfile 里显式 `apt-get install libusb-1.0-0`（与 MVS 同一层） |
| 27 | MVS 装不进依赖声明 | 它是 **deb 包**，不是 pip 包，`pyproject.toml` 的 `dependencies` 表达不了 | 走镜像层：deb 放 `.devcontainer/pkgs/`（已被 `.gitignore` 排除，184 MB 不进版本库），Dockerfile `COPY` + `apt-get install`；`check_env.py` 不检查它 |
| 28 | 手举 ArUco 码**晃动或快速移动时识别不出来**，静置却稳定 | **运动模糊**：80 ms 曝光的整个期间，码在画面里移动的距离都会被拖成一条影。手以 20 cm/s 挥动、1 m 距离时拖影约 20 px，而 4x4 码的单格也只有 15~20 px，格子边界直接糊掉（2026-10-17 实测）。帧率低（11.6 fps）是同一原因的另一种表现，**不是检测算法慢** | 降曝光优先于提增益：40 ms 下拖影减半、帧率上限 25 fps，实测静置仍稳定识别；增益只换噪声。**首选加光**（补光灯/台灯），既不丢帧率也不加噪声。这是物理限制，任何视觉系统快速运动都会丢，验收要求是"停下来能立刻恢复"，不是"每帧必检出" |
| 28 | 相机程序退出时打印 `*******XOpenDisplay Fail*******` | MVS 的库在收尾时会尝试连 X 显示，容器 / 无显示环境里必然失败 | 无害，可忽略：退出码仍为 0，取帧与落盘都不受影响 |
| 34 | `AttributeError: module 'cv2.aruco' has no attribute 'DetectorParameters_create'`（或 `detectMarkers` 用法对不上） | ArUco 自 OpenCV **4.7** 起进主包并改版 API：老教程的 `DetectorParameters_create()` + `cv2.aruco.detectMarkers(...)` 换成 `DetectorParameters()` + `cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)` | 只认新 API（`pyproject` 钉 `opencv-python>=4.10`）；**不要**为了迁就老教程装 `opencv-contrib-python`（会与 `opencv-python` 争 `cv2` 顶层模块）。`aruco.py` 在旧版本上抛一句带处理办法的 `RuntimeError`，`tests/test_aruco.py` 自动跳过 |

### 代码类

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 5 | `TypeError: can only concatenate list (not "tuple") to list` | `list + tuple` 在任何 Python 版本都非法（`check_env.py` 曾踩） | 用 `list + [0] * n`，或 `tuple(...) + (0,) * n` |
| 6 | `cv2.imwrite` 返回 `False` 但不抛异常 | OpenCV 少数"静默失败"接口之一 | 必须显式检查返回值，别假设成功；`utils/image_io.py` 已封装 |
| 7 | `cv2.imread` 读不到文件返回 `None` 不抛异常 | 同上 | 显式判空再往下走；`utils/image_io.py` 已封装 |
| 8 | 判色结果"看起来不太准"但不报错 | BGR 当 RGB 用不会抛异常，只会让红蓝全互换 | 图像变量一律带颜色空间后缀（`frame_bgr`）；判色前 `cvtColor(BGR2HSV)` |
| 9 | 红色判不出来 | HSV 的 H 范围是 0-179 不是 0-360；且色相在 0/179 处环绕，红色必须写两段阈值取并集 | 阈值表里红色写 `[0,10]` 与 `[170,179]` 两段 |
| 10 | 框内判色命中率偏低 | 判色分母用了 ROI 总像素，框里混入的地面背景稀释了命中率 | 分母用「有效像素」（饱和度/亮度达标的像素）；`core/color.py` 已按此实现 |
| 11 | `cv2.putText` 画出问号且不报错 | Hershey 矢量字体不含汉字字形 | 图上一律用 ASCII 英文颜色名（`viz/draw.py` 还会把非 ASCII 替换掉）；中文说明只进日志 |
| 30 | `--limit N` 一旦真正生效就丢结果：打印「❌ 一帧都没处理」并返回退出码 1，可帧确实处理了、图也落了盘 | `cli.py` 的 `on_result` 里额外 `raise StopIteration` 想断流，而 `process_frames` 是普通函数，异常直接冒泡到外层 `except StopIteration: results = []`，把算好的结果清空；限帧其实已由 `limited()` 生成器做过一遍 | 限帧只留 `limited()` 一处，删掉 `on_result` 里的 raise 与外层 except（复核意见一.1）。**触发条件是 `limit ≤ 可用帧数`**，所以相机那条无限流命令 `--limit 10` 必中 |
| 31 | 相机在某次失败后"打不开"了（`LIBUSB_ERROR_BUSY` / 枚举不到），要重启程序甚至重插 | `io/camera.py` 里 `_opened` 原本只在**全部成功**后才置位；`MV_CC_OpenDevice` 成功后若 `MV_CC_StartGrabbing` 失败就抛异常，此时设备已被**独占**打开，而 `close()` 开头 `if not self._opened: return` 直接返回 → 句柄泄漏 | `_opened` 语义改为「句柄已创建」（CreateHandle 成功即置位），构造函数里 `try: self._open() except: self.close(); raise`；`close()` 做成幂等，且 `CloseDevice` 只在设备确实打开过时调用（复核意见二.1 / 契约欠账二） |
| 32 | 叠加图上"什么都没画"：ID/颜色标签一个也看不见，左上角 HUD 也几乎读不出 | `put_text` 默认写**白字**、HUD 用亮黄字，落在**白底**素材上等于没画——不报错、就是看不见（与坑 11"画问号"同一类"看不出问题的问题"） | `put_text` 加 `outline=True`：先用黑色厚一档描边再写字，白底/黑底/彩色底都读得清（复核意见二.2）。教训：**画完要真的打开图看**，只看 `count/ids` 会以为画对了 |
| 35 | ArUco 四角误差稳定卡在 ~0.55 px，怎么调都下不去 | 测试图组的真值把码边界当成了「最外侧黑像素的**中心**」（`0` 与 `side-1`）；而检测器报的是黑框**最外沿**，正好差半个像素 | 真值改用像素边界 `[-0.5, side-0.5]`：误差立刻降到 0.14~0.23 px。教训：**真值必须和贴图用同一个仿射矩阵算**，别事后目测标点 |
| 36 | 用 `--limit 300` 测实时循环，实际只跑了 **3 帧**就结束 | `--limit` 只限制**循环消费**多少帧，供给端是 `input.source`：默认 `synthetic` 只配了 `num_frames: 3`，源自己 3 帧就结束了，`--limit` 根本轮不到 | 跑长序列要同时放大源帧数：`--set input.synthetic.num_frames=300`；相机源天然无限，不受影响。**验收第 6 条（无码不异常退出）必须跑满上百帧才有说服力**——3 帧只能证明"没崩"，证明不了"持续运行" |

### 工具与工程类

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 12 | `docker build` 报找不到 Dockerfile | compose 的 `dockerfile` 路径是**相对 `context`** 解析的，不是相对 compose 文件所在目录 | Dockerfile 移目录后必须同步改这行 |
| 13 | `.dockerignore` 形同虚设 | Docker 规定它必须在构建上下文根部，放子目录会失效 | 保持在仓库根 |
| 14 | `fish: Unsupported use of '='` | fish 不支持 `VAR=value cmd` 语法 | 用 `set VAR value`；`for` 循环里也别用 `$(...)` 赋值 |
| 15 | `git add -A` 把个人笔记提交了 | 只想本地排除却写进了 `.gitignore`——它会向全队广播规则，且队友写同名文件会被静默忽略 | 个人文件用 `.git/info/exclude`，团队规则才用 `.gitignore` |
| 16 | 删了模块但 `__pycache__` 里还有它的 `.pyc` | Python 不会自动清理已删源文件的字节码缓存 | `find src -name __pycache__ -type d -exec rm -rf {} +` |
| 17 | `*.egg-info/SOURCES.txt` 列着已删文件 | `pip install -e .` 生成的元数据不会随源码变更自动同步 | 删掉 `vision.egg-info` 后重装；该目录已被 gitignore |
| 33 | `cv2.createTrackbar` 报 `NULL window handler`（或滑条看不见） | 滑条名 / 窗口标题传了中文：OpenCV 的 highgui 在某些后端上不接受非 ASCII 名字 | 滑条名与窗口标题一律 ASCII（`tools/mvs_color_measure.py` 原来有个「面积下限」，已改 `min_area`，复核意见二.4） |
| 37 | `python -m vision.tools.aruco_check` 报 `ModuleNotFoundError: vision` | 在宿主 venv 里跑但没 `pip install -e .` | `PYTHONPATH=src python -m vision check-aruco`，或在容器里跑（镜像已 `pip install -e .`）。2026-10-22 注：旧薄壳 `scripts/check_aruco.py` 已随 `scripts/` 整体删除 |
| 38 | `docker compose build` 重建镜像很慢（几分钟到十几分钟，几乎全卡在 `pip` 下载） | Dockerfile 默认 `PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/`，**2026-10-17 实测只有 304 KB/s**：`numpy`（16.8 MB）用了 56 s，`opencv-python`（**77.4 MB**）按这个速度要 4 分钟以上；`apt` 那层走 Ubuntu 官方源，国内同样慢 | **换清华源重建**：`docker compose -f .devcontainer/docker-compose.yml build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`（README 第 91 行给的是中科大源，同样可用）。注意两种改动的代价完全不同：改这个 build-arg 只让 pip 层失效重下、前面 apt/MVS 层仍走缓存；**改 Dockerfile 里的 apt 层会让它后面所有层重跑**——所以新增系统库请追加到 Dockerfile 最后一层 |
| 39 | 容器里 `cv2.imshow` 报 `Could not load the Qt platform plugin "xcb" ... even though it was found`，而 `--headless` 落盘那条路完全正常 | `opencv-python` 的 GUI 走 Qt，`libqxcb.so` 运行时 dlopen `libSM.so.6` 与 `libICE.so.6`，`ubuntu:22.04` 基础镜像里没有；**报错完全不提缺哪个库**，极易误判成"OpenCV 没编 GUI 支持"而去重装 opencv | 镜像补 `apt-get install -y --no-install-recommends libsm6 libice6`（已加进 Dockerfile 最后一层）；宿主侧还需 `xhost +local:root`、`-e DISPLAY`、挂载 `/tmp/.X11-unix`。排查命令：`ldd $CV2PATH/qt/plugins/platforms/libqxcb.so \| grep "not found"` |
| 40 | 标定出来的**距离**整体偏（我们这台偏 53%），但内参与畸变看着完全正常 | 打印机把标称 15 mm 的格子打成了 **23 mm**；`--square-mm` 填的是标称值。内参不受板子尺度影响（尺度信息被位姿吸收），所以从 RMS 上完全看不出来，只有量过才知道 | **打印完用尺子量一个格子**；工具现在把 `--square-mm` 改成**必填**，不给就直接报错（示例：标称 15 mm、实量 23 mm 就填 23） |
| 41 | 标定 RMS 3~4 px、`fx` 偏 10% 以上，但每张图都「成功找到 54 个角点」 | 板子离太远，135 mm 的棋盘在图上只有 ~120 px，54 个角点挤在一起，亚像素精化失效，检出的点已不是格子阵（实测：角点间距 13/16/5/27 px 乱跳） | 让板子占画面 **1/3 以上**：`tvec.z` 从 1000 mm 改到 220~620 mm 后，同一套流程 RMS 从 3.8 px 降到 0.39 px |
| 42 | 畸变系数 `k2/k3` 跑来跑去、每次标定差别很大 | 纸卷边 / 板子没贴平 / 强反光：镜头看到的是「弯的板」，程序只能把这份弯曲也学进畸变系数 | 棋盘贴硬纸板、用哑光纸、光源偏离正面；`k3` 不稳时可以 `cv2.CALIB_FIX_K3` 固定它 |
| 43 | 标定结果离谱：`fx` 与 `fy` 差 7%、主点偏 200 px、`k2=0.98` `k3=-1.38`，但每张图都「成功找到 54 个角点」 | 快速连拍收进了 7 张**几乎同一姿势**的图（中心差 ≤12 px、格距差 0.1 px）：重复观测把焦距与畸变的耦合解当成真解 | 采图时加「太像就不收」的闸（`DEDUP_*` 阈值），重复姿势会被提示 `too similar to #N - change pose`；已收的图可用 `--images` 重算 |
| 44 | 标定 RMS 1.9 px、单张误差 5.2 px，但参数看着正常 | 手持拍摄时**运动模糊** + 没对实焦（实测锐度 Laplacian 只有 87~152，清晰图通常 300+），角点抖动直接进重投影误差 | 捕图加「站稳才拍」（连续 3 帧姿态稳定才收）、窗口显示 `sharp` 值；采图时 **sharp 低于 120 直接拒收**、标定后自动**剔离群重标**；板子贴硬纸板、摆好停半秒 |
| 45 | 位姿解得的距离是 0.00 mm、重投影 RMS 993 px，但检测四项（ID/角点/中心）全对 | `SOLVEPNP_IPPE_SQUARE` 对四个模型点的**顺序有硬要求**；写成「y 向下」那套（`-L/2,-L/2` 开头）等于把四角配错了 | 模型点顺序固定为 `(-L/2,+L/2)(+L/2,+L/2)(+L/2,-L/2)(-L/2,-L/2)` ↔ 图像 TL/TR/BR/BL；`core/pose.marker_object_points()` 是全仓唯一实现处，并在解算后用 RMS 上限硬拦 |
| 46 | 倾角全部算成 160~180°，距离却是对的 | `tilt = acos(R[2,2])` 少了负号：正视时实测 `R = diag(1,-1,-1)`、`R[2,2] = -1`，取负号后正视才是 0° | `tilt = acos(-R[2,2])`；`pose_metrics()` 统一出口，渲染真值与求解结果用同一个函数 |
| 47 | 角点已用 `undistort_points()` 掰直，距离仍偏（画面中心 0.6%、画面边缘 14.6%） | PnP 又传了一次 `dist_coeffs`，等于去畸变两次；畸变越强、码越靠边偏得越多 | 两种口径成对用：已掰直 → `dist_coeffs=None`；未掰直（实时入口） → 传标定畸变系数 |
| 48 | 完全正对的码解算失败，报 RMS 38.6 px，而真值重投影本该 0.01 px | 输入「正对 + 严格居中」时 IPPE_SQUARE 的两个候选解一起锁到 180° 分支（正方形对称，退化了） | `solve_pose` 在最优解过不了 RMS 门槛时用 `SOLVEPNP_ITERATIVE` 兜一次，谁的重投影小用谁 |
| 49 | 亮度调到 2.2 倍，五档「过曝」全部通过，等于没测过曝 | 纯乘增益：黑框乘 2.2 仍是 0、白格仍是 255，对比度一点没丢；真过曝是「环境光把黑框照亮 + 增益把白格截顶」 | 生成器加 `ambient`，按 `(像素 + 环境光) × 增益` 成像；3.0 倍 + 环境光 110 才真的丢码 |
| 50 | 正视时倾角误差 4~16°，但重投影 RMS 只有 0.15 px，看不出来 | 四个点没有冗余，PnP 总能精确穿过它们；而码正对相机时绕 x/y 轴转 10° 只让投影高度缩 1.5%（67 px 码差 1 px），该自由度本身弱可观 | 别拿 RMS 当姿态精度；真值倾角 < 10° 的样本只判距离（`pose_check.TILT_JUDGE_MIN_DEG`），姿态精度在 15° 以上才有意义 |
| 51 | 640×480 的图上距离解成 1571.9 mm（真值 450 mm，偏 249%），检测一切正常 | 1280×1024 的内参直接套到 640×480 图上，等效换了一颗长焦镜头 | `core/calibration.scale_camera_matrix()` 按宽高比缩放 fx/cx、fy/cy；`pose_check` 与 `pose_sweep` 都自动调用（畸变系数不缩放，严谨仍要重标定） |
| 52 | 严格正视 + 严格居中的码明明在画面里，解算却抛「`tvec[2]` 必须为正」 | IPPE_SQUARE 的闭式解在这种完美对称输入上退化成 NaN（实测 23 mm 码、300 mm、合成内参 `cx=640`：两个候选的 rvec/tvec/RMS 全是 nan；同一张图换 float32 角点反而不退化）。NaN 的比较恒为 False，`tvec[2] <= 0` 与 `rms > limit` 都拦不住，最后漏到 `MarkerPose` 报一句指向「角点顺序」的错 | 候选解先过 `isfinite`；兜底判据改取**最小** RMS（原来取的是排序前的 `scored[0]`）；报错分「数值退化」与「顺序错」两类；`tests/test_pose.py` 加回归用例 |
| 53 | `--set pose.ambiguity_deg=20` 改了、校验也过了，结果一点不变 | 三个字段只在 `schema.py` 里存在：`core/pose.py` 用的是模块常量，`max_tilt_deg` 连日志提示都没实现 | `ambiguity_ratio/deg` 接进 `solve_pose` 并由 `cli` 传配置值；`max_tilt_deg` 直接删除（倾角超 60° 检测本身就丢，没有可执行动作） |
| 54 | 同一场景名两次生成的噪声图不一样，噪声档误差数字每次小幅抖动 | 种子用的是 `abs(hash(name))`，而 `str.__hash__` 每进程随机（实测三次 40350 / 30148 / 54421） | 改用 `zlib.crc32(name.encode())`；两次独立生成的 `noise_16.png` md5 一致（`893A59…5BE`） |
| 55 | Windows 下工具跑完却抛 `UnicodeEncodeError`，退出码 1 | 控制台是 GBK，而工具打印 `✅/⚠️`；容器里是 UTF-8，所以从没暴露 | 新入口调 `utils/console.ensure_utf8_stdout()`；README「开发环境约定」补了说明 |
| 56 | 坐标轴与码的四个角差几像素，画面边缘更明显 | 位姿是带着 `D` 解出来的（原始畸变像素空间），画轴却用 `dist=None` 投影；docstring 还写着「位姿在去畸变空间解出来」 | `draw_pose_axes` 增加 `dist_coeffs` 参数，`cli` 传同一个 `D`：已掰直配 None、未掰直配 D，两边必须成对 |
| 57 | YOLO 在容器里跑出 `class_name="0"`，不是 `"person"` | 类名只在 ONNX 的 `metadata_props` 里，读它要装 `onnx` 包，而容器里没有（也不想为一行元数据加 30 MB） | 导出脚本顺手写一个 `<stem>.names.json` sidecar；`_read_class_names()` 先读它（标准库 json），读不到再退回 `onnx` 元数据，最后退化成索引字符串 |
| 58 | 预训练 YOLOv8 在 `data/calib` 的棋盘图上 0 个框，看起来像接入失败 | **不是 bug**：COCO 80 类里没有"色块"，模型没见过就不输出 | 用 `bus.jpg` 跑出 6 框证明链路通；真实物块要标注数据 + 微调（M5 训练仍阻塞）。另：YOLOv8 权重是 AGPL-3.0，对外发布前需确认许可 |


> 注：上表编号 17 与「环境类」新增的 17~23 重复，下次整理时统一重排。

---

### 2026-10-16 复核整改记录（PR #3 复核意见 → 本分支）

复核对象是 `a06df7f`（当时推送的 PR head）。逐条落地情况：

| 复核条目 | 结论 |
|---|---|
| 一.1 `--limit` 丢结果返回 1（必须改） | ✅ 已修，见坑 30；实测 `--limit 3` 退出码 0 |
| 一.2 描述与推送的代码对不上（必须改） | ✅ 描述按最终状态重写；并明确 PR head 以推送后的实际提交为准 |
| 二.1 `io/camera.py` 半初始化句柄泄漏 | ✅ 已修，见坑 31 |
| 二.2 `viz/draw.py` 白底白字 | ✅ 已修，见坑 32；用纯白图 + stub 复现并看图确认 |
| 二.3 `viz/draw.py` 画框比半开上界多 1px | ✅ 改为 `(x2-1, y2-1)` |
| 二.4 `tools/mvs_color_measure.py` 中文滑条名 | ✅ 改为 ASCII `min_area`，见坑 33 |
| 二.5 `writer.close(summary)` 只在成功路径 | ✅ 移到 `finally`，中途出错也能落盘 |
| 二.6 `raise StopIteration` 作控制流 | ✅ 随一.1 一并清掉 |
| 三.1 测试项数各处不一致（27/28） | ✅ 统一为 **36**（复核整改后由 28 增至 36），README / 本文 / 双环境表同步 |
| 三.2 `progress.md` 仍写 `camera` 抛 `NotImplementedError` | ✅ 已改（相机走 `io/camera.py`，SDK 缺失时构造期报 `RuntimeError`） |
| 三.3 第四节仍把 M6 已完成的事列为待办 | ✅ 已把第 1 步标为已清 |
| 三.4 Dockerfile 注释里残留 `io/camera_mvs.py` | ✅ 改为 `io/camera.py` |
| 三.5 `io/__init__.py` 写"依赖 config"，与契约第 5 节矛盾 | ✅ 已改（契约欠账三） |
| 三.6 `configs/README.md` 仍写"src 还是空骨架" | ✅ 已改 |
| 三.7 `configs/README.md` 的 `hsv_tuner` 示例参数不对 | ✅ 改为真实的 `--colors red` / `--source synthetic` |
| 三.8 `configs/default.yaml` 注释说"改了不会有任何效果" | ✅ 已改 |
| 四（相机相关没能验证的部分） | ⏳ 仍需在**接了相机 + 装了 MVS + 透传 `/dev/bus/usb`** 的环境里实跑；本机没有相机，只能读代码 |
| 契约（`docs/interface.md` 新版） | ✅ 已按搭档那版落地：新增 `ArucoMarker` 与 `stub_aruco`，判色族 `Detection` 保留；三条欠账照登记（其中欠账二已修，欠账一按契约"要改先问人"**没动**） |
