# bags/：真实采集的 rosbag

## 规则（计划书硬性要求）

**bag 与 `meta.yaml` 必须同存。** 没有 meta 的 bag 等于废数据——过两周你不会记得
相机架了多高、倾角多少、物块摆在哪，而这三项直接决定算法参数能不能复用。

```
bags/
├── README.md          ← 本文件，入库
└── raw_001/           ← 一次采集，bag 本体不入库（见根 .gitignore）
    ├── raw_001_0.mcap
    └── meta.yaml      ← 入库
```

## 命名

`raw_XXX` 真实采集，`syn_XXX` 合成数据（`lab/gen_synthetic_depth.py` 产出）。
编号按采集顺序递增，不复用、不覆盖。

## meta.yaml 必须记的三项

1. **安装高度**（相机到桌面，mm）
2. **倾角**（相机光轴与桌面法向的夹角，度）
3. **物块位置**（桌面坐标系下的位置，或示意图路径）

此外建议记：相机序列号与固件版本、分辨率/帧率、USB 协商速率、光照条件、
物块材质与尺寸、场景类型（理想/反光/弱纹理/边缘/遮挡）。

## 场景覆盖

计划书 S1–S3 要求覆盖五种摆法：**理想、反光、弱纹理、边缘、遮挡**。
其中反光和弱纹理是必拍项——它们是这条线的主要失效来源。

## RGB-D 同步采集（2026-10 起，`raw_004` 起）

彩色 + 对齐深度这套数据有专门的入口，不要再用 `ros2 bag record` 手敲：

```bash
# 终端 1：起相机（三选一：raw / sync / aligned）
ros2 launch vision_depth rgbd_capture.launch.py mode:=aligned

# 终端 2：录「预热 3 s + 稳定段 10 s」，并自动写 meta 与参数快照
ros2 run vision_depth rgbd_capture --ros-args \
  -p output_dir:=$PWD/bags/raw_004 \
  -p duration_s:=10.0 \
  -p warmup_s:=3.0 \
  -p session_config:=$PWD/configs/session_raw_004.yaml
```

- 场景配置模板：`lab/rgbd_check/templates/session.yaml`；bag 内 meta 结构见
  `lab/rgbd_check/templates/meta.yaml`。模板放在 `lab/` 而不是 `bags/`，是因为
  `.gitignore` 只放行 `bags/*/meta.yaml`，`bags/templates/` 会被整目录忽略。
- 采集完成后，bag 目录里除 `*.mcap` 与 `meta.yaml` 外还有 `snapshots/`：驱动实际参数、
  发布端 QoS、两侧 CameraInfo 和首帧几何探测。**`snapshots/` 不入库**，但报告要引用它。
- 离线检查与出报告：

```bash
python -m lab.rgbd_check inspect bags/raw_004
python -m lab.rgbd_check analyze bags/raw_004 --out output/rgbd/raw_004
python -m lab.rgbd_check annotate output/rgbd/raw_004     # 本机窗口，每边 4 次点击
```

- 三个场景（本轮必采）：`center_near`、`center_far`、`right_mid`，每个场景 10 s 稳定段
  + 3 s 预热。遮挡样例另录一段（两个箱子前后错开），不占这三个名额。
- 同一段 bag 的彩色与深度必须来自同一次录制；两次录制不能当同次对照。

## 体积参考

深度 848×480@30fps 约 27 MB/s，单深度 35 秒约 930 MB。
**对齐双流（彩色 1280×800 + 深度）按 30–40 MB/s 估，120 秒约 4 GB。**
录制前先确认磁盘，尤其步骤 E 的连续录制段；`rgbd_capture` 结束时会打印实际体积与时长。
纯深度的最小可用集仍是 `ros2 bag record -o bags/raw_00X /camera/depth/image_raw
/camera/depth/camera_info`。
