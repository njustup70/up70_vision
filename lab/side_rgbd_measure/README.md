# 单箱可见侧面：离线 RGB-D 测量

此工具将人工框或 YOLO 框与对齐深度组合，列出平面候选，供人工复查和独立测量对照。候选是 `candidate_unverified`，不是物体中心、抓取目标或绝对精度通过。已有的顶面抓取与 ROS 接口不受影响。

纯平面计算、像素校正和绘制已提取至 `vision.depth`，本离线入口及结果字段保留。需要在线逐框显示中心时使用 [实时侧面查看器](../../vision/depth/README.md#实时侧面查看器)；实时多框各自测量，离线工具仍保留多框需明确选择的诊断行为。

## 输入和环境

在仓库根目录使用带项目依赖及 `[lab]` 可选依赖的 Python。输入为 `python -m lab.rgbd_check analyze` 产生的报告；要求采集模式 aligned、深度单位 mm、采集完整性及同步 PASS、两路 CameraInfo/尺寸/frame 相符。彩色 plumb_bob、对齐深度零畸变。几何与空间状态即使为 UNVERIFIED 也如实保留。

默认测量报告中已保存的一对图；指定 `--bag` 后重新检查原包及一对一配对，从稳定区间等间隔选取不同帧，不够时如实报告实际数量。原包路径必须与报告来源一致；相机内参变化或不一致时拒绝处理。大批量读取会把选中图像放在内存中；848×480 的 60 对约需 120 MB 图像数据，另加算法内存。

```bash
# 先生成报告；输出目录必须不存在。已有报告可直接使用。
python -m lab.rgbd_check analyze bags/<bag> --out output/rgbd/<report>

# 一帧：窗口中在校正后的图选侧面；无显示时传 --bbox x1 y1 x2 y2。
OPENBLAS_NUM_THREADS=1 python -m lab.side_rgbd_measure output/rgbd/<report> \
  --out output/side/<single>

# 静止场景：同一人工框应用于 60 对；右、下边界不包含在框里。
OPENBLAS_NUM_THREADS=1 python -m lab.side_rgbd_measure output/rgbd/<report> \
  --bag bags/<bag> --frames 60 --bbox 100 80 300 260 \
  --out output/side/<manual>
```

框内的有效点不要求属于顶面。默认体素 4 mm、平面内点阈值 6 mm、200 次 RANSAC、种子 0、至少 120 个降采样点；最多 3 个候选。这是调试初值而非准确度门槛。结果保存每次参数；`--leaf-mm`、`--threshold-mm`、`--iterations`、`--seed`、`--min-points` 可调整。评估前冻结参数。

离线默认 `--selection-policy strict`，可显式传 `--selection-policy dominant` 选择明显占优的平面；阈值和定义见[实时查看器说明](../../vision/depth/README.md#实时侧面查看器)，结果 `params` 会记录实际初值。两种策略均不自动选择多个检测框；显式 `--choices` 逐帧指定的平面覆盖自动选择，但仍为未验证候选。

## 模型与候选选择

权重不入 Git。取得搭档的 ONNX 权重后，确认模型输入尺寸、BGR 输入预处理和输出兼容现有 YOLO 解码，再提供目标类别编号。若对方只有 `.pt`，可在已安装对应 ultralytics 的环境使用 `python lab/export_yolo.py --weights <file.pt>` 导出；某些模型结构或导出格式与当前解码器不兼容，必须用真实样本单独核查，不能把文件扩展名当兼容证明。

```bash
OPENBLAS_NUM_THREADS=1 python -m lab.side_rgbd_measure output/rgbd/<report> \
  --bag bags/<bag> --frames 60 --weights /path/to/model.onnx \
  --imgsz 640 --class-id 0 --class-names box \
  --out output/side/<yolo>
```

`--class-names` 按模型类别索引给出**完整类名表**；不传则读取权重旁的 `.names.json` 或模型元数据，仍没有时显示类别编号。`--imgsz`、类别、阈值必须与实际模型相符。输出保存模型 SHA-256 和推理参数。

### 搭档模型 `yolov8n_mix2`（2026-10-29）

本地将 `yolov8n_mix2.onnx` 和 `yolov8n_mix2.names.json` 一起放入 `knowledge/models/`，保留同名配对；两者均被 Git 忽略，新 clone 需自行取得。类名表为 `0=block、1=top`，不凭名称推断对应单箱侧面或顶面。现有检测器接收 BGR 图，内部等比补边、转 RGB 并除以 255；已在项目容器 OpenCV 4.14 中验证 640 输入和 `(1,6,8400)` 输出兼容，无需重新导出或安装 torch。

以下命令从仓库根运行，使用已保存的 `raw_036/c241_d240` 一对图，不重新读 bag；容器按当前用户创建报告。输出目录必须不存在，重复运行时换新名字。

```bash
docker compose -f .devcontainer/docker-compose.yml run --rm \
  --user "$(id -u):$(id -g)" \
  -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 up70-vision \
  python -B -m lab.side_rgbd_measure output/rgbd/raw_036 \
  --weights knowledge/models/yolov8n_mix2.onnx \
  --imgsz 640 --conf 0.45 \
  --out output/side/yolov8n_mix2_raw036_01
```

不传 `--class-id`，保留两个类别；不传 `--class-names`，自动读取同名 JSON。首轮结果位于上述输出目录的 `results.json`、`report.md` 和 `frames/`。模型 SHA-256 为 `9430baaae11efcad98f2434267c21567ecb4a190347c50f4201d1df2d6a00684`。

**本次实际结果：** 校正图上检出一个 `top`，置信度约 0.6043，框为 `[339,68,486,359)`；该框跨越上下两个箱子，唯一平面候选也覆盖两块正面。此前用户确认的上方单箱人工框为 `[340,67,484,215)`，其复核不能沿用到本次 YOLO 输出。本轮仅证明模型加载、解码及侧面工具落盘链路通过，状态仍为 `candidate_unverified`，几何与空间均为 `UNVERIFIED`；没有单箱尺寸或绝对精度通过结论。

本轮未进行 YOLO 60 帧检查。先核实搭档的类别定义及训练标注，确认实际检测框对应目标单箱，再在具备 `[lab]` 依赖（含 rosbags）的环境增加 `--bag bags/raw_036 --frames 60`，并改用新输出目录。多框、多平面仍按下述规则逐帧选择，不自动裁成上方箱子，也不将高置信度当作选面正确。

检测框图与平面叠加图均显示从 0 开始的框号、模型类名和置信度，例如 `0: top 0.60`；人工框显示 `0: manual`，不显示置信度。平面图另加 `plane=N` 对应候选号，标签不表示已复核正确。如果一帧有多个检测框，`frames/<pair>_boxes.png` 会保留所有框；有多个平面时分别保存 `<pair>_planeN.png` 和掩码。默认 `strict` 不自动选择多个平面；`dominant` 仅在条件满足时自动选择。查看图后可写入每帧选择，使用新输出目录重跑：

```json
{
  "c241_d240": {"detection": 0, "plane": 1},
  "c246_d245": {"plane": 0}
}
```

使用 `--choices /path/to/choices.json`；框歧义需要先指定 `detection`，重跑查看候选平面后再指定 `plane`。选择逐帧生效，不能假定不同帧的候选号表示同一面。即使只剩一个候选，也须复查是否误选背景。

## 结果与人工评估

输出 `results.json`、`frames.csv`、`report.md` 和 `frames/`。框、法向、平面方程、内点、残差、可见重心和可见区域最小外接矩形均随帧保存；法向朝相机。`plane_distance_mm` 是光心到平面垂距，`centroid_z_mm` 是光轴深度，二者不是同一量。单帧没有可解释的重复性，统计值为 null。

只在操作者逐帧复核后准备标注 JSON。示例中的数值与帧号是格式示例，不是已测结果：

```json
{
  "size_source": "尺测，记录量具、端点和不确定度",
  "side_size_mm": [200.0, 200.0],
  "frames": {
    "c241_d240": {
      "boxes": [[100, 80, 300, 260]],
      "side_review": "correct",
      "complete_boundary": true
    },
    "c246_d245": {
      "boxes": [],
      "side_review": "unknown",
      "complete_boundary": false
    }
  }
}
```

`boxes` 是校正网格中人工标注的完整目标框；空列表表示人工确认这一帧无目标，缺少该字段表示不评估框。`side_review` 填 `correct`、`incorrect` 或 `unknown`，针对该运行的选面复查；尺寸误差同时要求 `correct`、`complete_boundary: true`、独立尺测及其来源。未标注的帧不参与框准确率；匹配阈值 IoU 0.5。没有分母时 Precision/Recall 为 null。

```bash
python -m lab.side_rgbd_measure.evaluate output/side/<manual>/results.json \
  --annotations /path/to/manual_annotations.json --out output/side/<eval>

# 两次运行必须基于同一 bag；人工框真值一致、选面复查分别记录。
python -m lab.side_rgbd_measure.evaluate output/side/<manual>/results.json \
  --annotations /path/to/manual_annotations.json \
  --compare output/side/<yolo>/results.json \
  --compare-annotations /path/to/yolo_annotations.json --out output/side/<comparison>
```

可选的独立平面垂距真值写入 `plane_distance_reference`：`{"value_mm": 500, "reference": "camera_optical_center_to_plane", "source": "..."}`。镜头前沿距离不是光心到平面的垂距，不能直接填入。独立法向真值写 `normal_reference`：`{"normal": [0, 0, -1], "frame_id": "camera_color_optical_frame", "source": "..."}`，frame 必须与结果一致。均只对 `side_review: correct` 的帧比较；没有参考则 UNVERIFIED。

位移比较用 `--baseline <基准段/results.json>`，并在标注中写 `displacement`：`distance_mm`、`source`，以及由操作者确认的 `same_face`、`complete_boundary`、`orientation_unchanged`、`camera_fixed` 均为 true；`reviewed_pair_ids` 与 `baseline_reviewed_pair_ids` 分别列出两段中已确认选面正确的帧编号。两段光学 frame 必须相同。比较两组已复核重心的中位数位移模长；这不是绝对 XYZ 误差。要比较的标注、参考与来源由操作者提供，不从相机深度自动生成真值。

## 最小实测安排

固定相机，以一个侧面平整的箱子，在基准、沿同一直线约 50 mm 和约 100 mm 三个位置各独立摆放录制两次。每段预热 3 s、稳定录 10 s，按已有 aligned 配置采集；每段选 60 对，每段五张人工框及选面复查。记录尺测宽高、实际位移、量具、端点和不确定度。不同位置或姿态重新标注；尺寸测量只在四条物理边界完整可见且无显著深度缺失时比较。

空间对齐尚未验收；结果只作此场景和输入条件下的诊断。测量精度、选择正确率和实际抓取效果需要独立真值和现场复核。
