# bags/：真实采集的 rosbag

> 2026-10-28 更正：下方历史同步 PASS 不包含实际录制时长完整性。
> 新版复算发现 raw_009–019（不含作废 raw_016）均未录满声明稳定段；配对仍通过、完整性 FAIL。
> raw_014 的实际 ROS 回放重录一致性已经通过。空间验收仍未通过，且需先核清彩色/D2C 的畸变网格。
> 参见 [验收修正与补采说明](../vision/depth/docs/RGB-D验收修正与补采说明-20261028.md)，原始数据及旧报告保留。

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
- **DDS 传输默认走仅 SHM**（`config/fastdds_shm_only.xml`）：launch 与 `rgbd_capture`
  都会自动设置，不用手动 export。实测默认 UDP 时 1280×720 双流 13 秒在 transport
  层丢 113 条、配对率只有 87%；改仅 SHM 后 288/288 全配上。要退回默认传输设
  `RGBD_CAPTURE_DDS_PROFILE=none`，但要知道代价是丢帧。
- **档位冻结在 848×480@30**（彩色与深度同频，D2C 到彩色）。1280×720@30 能跑，
  但录制约 113 MB/s 且必丢帧。
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

## 本轮 RGB-D 包清单（2026-10-27）

`raw_004`–`raw_008` 是排查用的中间包（丢帧对照、SHM 验证），`raw_009` 起是本轮正式样本：

| 包 | 用途 | 模式 | 稳定段 | 体积 | 同步结论 |
|---|---|---|---|---|---|
| `raw_009` | 环境清单 + 短包完整性 | aligned | 10 s | 750 MB | PASS |
| `raw_010` | 三模式对照：同步关、D2C 关 | raw | 30 s | 1.99 GB | PASS |
| `raw_011` | 三模式对照：同步开、D2C 关 | sync | 30 s | 1.99 GB | PASS |
| `raw_012` | 三模式对照：同步开、D2C 开 | aligned | 30 s | 1.99 GB | PASS |
| `raw_013` | 场景 `center_near` | aligned | 10 s | 604 MB | PASS（稳态仅 20.6 fps，拍摄中相机被碰倒） |
| `raw_014` | 场景 `center_near`（重录） | aligned | 10 s | 735 MB | PASS |
| `raw_015` | 场景 `center_far` | aligned | 10 s | 736 MB | PASS |
| `raw_016` | **作废**：要求「右侧」却拍成中心（x=52%） | aligned | 10 s | 743 MB | PASS（数据作废，勿引用） |
| `raw_017` | 场景 `right_mid` | aligned | 10 s | 732 MB | PASS |
| `raw_018` | 运动样例（箱子移出/移入画面） | aligned | 45 s | 2.88 GB | PASS |
| `raw_019` | 连续录制 | aligned | 120 s | 7.43 GB | PASS |

### 2026-10-28 重采（`raw_021` 起）

上面那批（`raw_009`–`raw_019`）复算后**覆盖不足声明时长**，且箱子四周是深度无效区、轮廓边量不出来。
这一轮换了摆法并重录：

| 包 | 用途 | 模式 | 稳定段 | 体积 | 完整性 |
|---|---|---|---|---|---|
| `raw_021` | 完整性冒烟（改录包器后第一段） | aligned | 10 s | 818 MB | PASS |
| `raw_022` | **诊断样本**：箱子未垫高，只有顶边可量（深度 407 mm） | aligned | 10 s | 820 MB | PASS |
| `raw_023` | 场景 `center_near`（叠放垫高 + 哑光背景板） | aligned | 10 s | 823 MB | PASS |
| `raw_024` | 场景 `center_far`（同上，推远到 700 mm） | aligned | 10 s | 822 MB | PASS |
| `raw_025` | 场景 `right_mid`（同上，右移，中心 61.9%） | aligned | 10 s | 823 MB | PASS |
| `raw_026` | 连续录制（补齐第四轮） | aligned | 120 s | 7.1 GB | PASS |
| `raw_027` | 三模式对照：同步关、D2C 关 | raw | 30 s | 2.0 GB | PASS |
| `raw_028` | 三模式对照：同步开、D2C 关 | sync | 30 s | 2.0 GB | PASS |
| `raw_029` | 三模式对照：同步开、D2C 开 | aligned | 30 s | 2.0 GB | PASS |

这一轮的摆法要点（下次重采要照做）：

1. **箱子叠放垫高**：不垫高时相机俯角让左右轮廓背后是地面（约 500 mm，与箱子只差 100 mm 且断续），只有顶边可量。
2. **哑光背景板**（约 1.17 m）：给轮廓外侧提供有效深度和 ≥400 mm 深度台阶；要覆盖箱体四周 ≥150 mm。
3. **只标顶边 + 右侧竖边**：左侧是红外投影阴影，黑带 22–53 px，深度在那一侧没有有效值。
4. 相机必须接 **USB 3.0**：2.0 下没有 848×480@30 这个档位（驱动会报 `No matched video stream profile`）。

逐包数字（帧数、配对率、时间差、帧率、黑带宽度）见
[`vision/depth/docs/progress.md`](../vision/depth/docs/progress.md) 的「2026-10-28 重采一轮」。

引用这些包时注意三件事：

1. **距离是标称值，不是实测值**。`raw_023/024/025` 已补记 `scene.nominal_distance_mm`
   （400 / 700 / 550 mm）与 `distance_source: nominal_spoken`，但 `measured_distance_mm` 仍是
   `null`：那是口述档位，不是卷尺读数。原摆放已于 2026-10-28 撤除，无法回补实测。标称值
   只能用来分组，不能当已知量参与误差计算。
2. **空间对齐全部是 `UNVERIFIED`**。三场景六条边已有标注和 `manual_review=yes` 记录，但右边紧邻深度无效黑带，不能确认深度有效区边缘就是实物边界；共同网格也未核清。「配对率 100%」只证明时间戳配对，不证明像素对齐达标。详见 [重采核查](../vision/depth/docs/RGB-D重采核查-20261028.md)。
3. **`raw_016` 不要用**。它名义上是 `right_mid`，实测箱子中心在画面中央，条件不符。

遮挡样例（两个箱子前后错开）与 `ros2 bag play` 回放一致性核对都还没做。

## 体积参考

深度 848×480@30fps 约 27 MB/s，单深度 35 秒约 930 MB。

对齐双流实测（2026-10-27，mcap）：

| 档位 | 速率 | 13 s（预热 3 + 稳定 10） | 120 s 稳定段 |
|---|---|---|---|
| 848×480@30 | ≈58–62 MB/s | ≈750 MB（`raw_009` 实测 750 MB） | 7.43 GB（`raw_019` 实测，120 s） |
| 1280×720@30 | ≈113 MB/s | ≈1.4 GB（`raw_004` 实测 1.4 GB） | ≈13.6 GB |

按 848 档估：三模式各 30 s 约 5.5 GB，整轮采集（3×30 s + 三场景 + 遮挡 + 运动 + 120 s）
约 15 GB；实测 `raw_009`–`raw_019` 共 11 段、约 20 GB（含 `raw_016` 作废段与
`raw_014` 重录段）。录制前先确认磁盘；`rgbd_capture` 结束时会打印实际体积与时长。
纯深度的最小可用集仍是 `ros2 bag record -o bags/raw_00X /camera/depth/image_raw
/camera/depth/camera_info`。
