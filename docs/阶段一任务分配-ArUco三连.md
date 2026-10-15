# 阶段一任务分配：ArUco 生成、识别与实时流（2026-10-15）

> 来源说明：本文档由郗洋与 AI 助手讨论后整理，经本人确认。目标排序：**跑通流程与协作制度 > 代码本身**。本阶段交付三样东西：一个能演示的程序、一套完整走过一遍的协作制度、第一篇踩坑记录。
> 分工基于仓库 up70_vision 现有分层结构（M0 初始化与 M1 环境标准化已完成），不另起炉灶。

## 一、共同完成（第一天，约 3 小时，两人一起）

1. **环境验证**：`docker compose -f .devcontainer/docker-compose.yml build` → `check_env.py` → `demo.py` 全部通过（这一步就是踩坑笔记的开始）
2. **学习 ArUco 原理**：编码方式、字典与 ID 的含义，约 30 分钟材料即可，不求深
3. **共同定义接口契约**，写入 `docs/interface.md`：

   ```python
   detect(image_bgr) -> list[{"id": int,
                              "corners": ndarray,  # shape (4,2), float32, 像素坐标
                              "center": (int, int)}]
   # 无码返回空列表；corners 的角点顺序、dtype 等细节当天一起敲定
   ```

4. **搭 stub**：`core/detectors/stub_aruco.py`，按契约返回假检测结果（固定 id 与角点），让两边互不阻塞
5. 在 `progress.md` 登记 Owner 分配（见下）

## 二、分头实现（第 2–4 天，互不依赖）

### 搭档 Owner：检测核心（对应任务一，2D 方向预演）

| 产出 | 位置 |
|---|---|
| 检测实现：`detect()`，DICT_4X4_50，同图多码、旋转均可识别 | `src/up70_vision/core/detectors/aruco.py` |
| 生成工具：多个不同 ID 的 ArUco 码图 | `src/up70_vision/tools/gen_aruco.py` |
| 测试图组：多码图、旋转图、无码图各若干 | `assets/aruco/` |

- 自测标准：测试图组里每个码的 ID、四角、中心全部正确
- 你在这一步练到：OpenCV 基础、函数封装、契约实现

### 郗洋 Owner：输入源与可视化（对应任务二）

| 产出 | 位置 |
|---|---|
| 摄像头输入源（**先对 stub 开发，不等检测核心**） | `src/up70_vision/io/camera.py` |
| 实时绘制边框/ID/中心点 + 终端输出 | `src/up70_vision/viz/` |
| 编排与入口：可选图片/相机输入（即任务三的整合） | `src/up70_vision/cli.py` + 薄入口 `scripts/` |

- 鲁棒性要求：码移动、旋转、多码时持续更新；无码时正常运行不退出
- 这一步练到：输入源抽象、鲁棒性设计

> **附加题（内参标定 + 位姿估计）本周不做，顺延至阶段二启动时。** 接口契约里 `detect()` 返回的四角点已足够支撑后续 solvePnP，先做不亏；相关技术坑见第四节第 3 条，届时再动手。

## 三、合跑与验收（周末，半天）

流程：互发 PR 交叉评审 → squash 合并 → stub 换真实现 → 整体验收 → 打 tag `run-YYYYMMDD`。

验收清单（对照任务单）：

- [ ] 同一张图片中的多个 ArUco 码全部正确识别（ID、四角、中心）
- [ ] 实时流中码移动 / 旋转 / 多码出现时持续更新
- [ ] 画面无码时程序正常运行，不异常退出
- [ ] 本地图片 / 摄像头输入可切换（任务三）
- [ ] 识别代码封装成函数或类，结构清晰，无单文件堆砌（任务三）
- [ ] `demo.py` 与 `check_env.py` 通过

## 四、已知技术坑（第一条踩坑笔记，记入 progress.md 第五节）

1. **不需要 `opencv-contrib-python`**：ArUco 自 OpenCV 4.7 起已移入主包 `opencv-python`，且仓库 pyproject 明令禁止安装 contrib（多个 cv2 包会互相覆盖）
2. **API 版本**：OpenCV 4.7+ 的 aruco API 改版（`cv2.aruco.ArucoDetector` 类），网上老教程的写法会报错——查资料先核对版本
3. **`estimatePoseSingleMarkers` 新版已弃用**：位姿估计直接手写 `cv2.solvePnP`（码的四角点作为 3D 点、边长已知）。附加题顺延后此条留作阶段二参考
4. **相机进容器**：`docker-compose.yml` 需加 `devices: ["/dev/video0"]` 和 `group_add: [video]`，且宿主机用户在 video 组——缺任何一样 `cv2.VideoCapture(0)` 就是打不开，且报错信息不会提示原因。README 已提示过，D1 环境验证时顺手确认
5. **容器内实时预览需 X11 转发**：`cv2.imshow` 要 `-e DISPLAY=$DISPLAY` + 挂载 `/tmp/.X11-unix` + `xhost +local:`，缺了直接报错；降级方案是用仓库现成的 `save_overlay` 约定把叠加图写盘后查看

## 五、制度维护（本阶段的隐藏交付物）

- 本阶段的每一条协作规则——接口契约、feature 分支、PR 交叉评审、squash、tag、踩坑笔记——都是第一轮实战
- **不合用的条款，合跑日小会一起修订**协作规范；修过的制度才算我们的制度
- 阶段结束时在 progress.md 记录三件事：完成了什么、卡在哪里、制度要改哪条
- 这份记录直接决定阶段二（Yolo 复现与深度相机）的分工方式
