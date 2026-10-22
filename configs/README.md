# configs 目录说明

配置是参数的唯一真源。代码里不硬编码任何阈值、尺寸或路径。

> **当前状态**：读取这些配置的代码**已就位**（`config/loader.py` + `config/schema.py`）。
> 改这里的值会直接生效；`python -m up70_vision run --set a.b=c` 可临时覆盖单次运行。
> 字段路径写错或取值非法，启动时报 `ConfigError`，消息里带字段路径与合法取值。

---

## default.yaml

全局默认配置。层级与 `--set` 的点分路径一一对应。

| 路径 | 默认 | 含义 |
|---|---|---|
| `input.source` | `synthetic` | 输入源：`synthetic` / `replay` / `camera` |
| `input.synthetic.width` / `height` | 640 / 480 | 合成图尺寸 |
| `input.synthetic.num_frames` | 3 | 合成帧数 |
| `input.synthetic.blocks_per_frame` | 6 | 每帧色块数，默认与 6 种颜色一一对应 |
| `input.synthetic.seed` | 2027 | 随机种子，固定才能复现 |
| `input.synthetic.noise_sigma` | 8.0 | 色块上的高斯噪声标准差 |
| `input.replay.path` | `data/clips` | 回放素材路径 |
| `input.replay.frame_stride` | 1 | 抽帧步长，5 表示每 5 帧取 1 帧 |
| `color.thresholds_file` | `color_thresholds.yaml` | 阈值表，相对本目录 |
| `detector.name` | `blob` | 检测器：`blob`（占位颜色连通域） |
| `detector.blob.min_area_px` | 400 | 最小框面积 |
| `detector.blob.morph_kernel` | 5 | 形态学核边长，用奇数 |
| `detector.blob.max_detections` | 32 | 单帧最多保留多少个框 |
| `output.enabled` | true | 是否落盘 |
| `output.dir` | `output/demo` | 输出目录 |
| `log.level` | INFO | 日志级别 |

### 哪个参数最值得先动

- **`color_thresholds.yaml` 文件头 `color:` 段的饱和度 / 亮度下限**：真实场地光照一变，
  最需要重调的就是这两个。调高的表现是"色块边缘被吃掉"；调低的表现是"背景里的灰色也开始被判成颜色"。
- **同段的命中率门槛**：从 0.5 提到 0.8 会更严格，混色区域会返回 `unknown`。
  下游机械臂拿到 `unknown` 会停手，拿到错颜色会抓错东西——所以宁可严一点。
- **`input.synthetic.seed`**：换种子后若自检不再是 18/18，说明阈值边缘太窄。

---

## color_thresholds.yaml

HSV 阈值表，**判色的唯一参数来源**：文件头 `color:` 段是判色门槛，下面每项是一个颜色的 HSV 范围。
加颜色只需在这里加一项，代码自动识别。

格式：

```yaml
color:                       # 保留键：判色门槛（不是颜色名）
  min_saturation: 40
  min_value: 40
  hit_ratio_min: 0.50
  ignore_border_px: 3
  min_valid_pixels: 25

颜色名:
  - [h_lo, s_lo, v_lo, h_hi, s_hi, v_hi]   # 第 1 段
  - [h_lo, s_lo, v_lo, h_hi, s_hi, v_hi]   # 第 2 段（可选）
```

**H 的范围是 0-179，不是 0-360。** 从教程抄 0-360 的写法时先除以 2。
写超范围的值会在启动时被拦下，报错信息里会提示这一点。

### 两个 S 门槛不是一回事（最容易改错的地方）

| 键 | 在哪 | 含义 |
|---|---|---|
| `min_saturation` | 文件头 `color:` 段 | "什么算**有效像素**"——命中率的**分母** |
| `s_lo` | 每个颜色段里的第 2 个数 | "什么算**这个颜色**"——命中率的**分子** |

改错一个不会报错，只会让命中率悄悄偏低。日常调参只开这一个文件；
`default.yaml` 里写同名键仍会生效（并盖住这里的值），那是给临时实验用的。

### 怎么调

不要对着代码猜数字。用调参工具（`tools/hsv_tuner.py`，已就位；需要 GUI）：

```bash
# 只调红色：--colors 后面跟阈值表里的颜色名（可以给多个）
python -m up70_vision hsv-tuner --image data/clips/sample.jpg --colors red
# 没有实拍图也可以对着合成图调
python -m up70_vision hsv-tuner --source synthetic
```

拖滑条时看中间那幅叠加图：目标颜色被红色半透明覆盖、背景干净，就是对了。
按 `s` 写回本文件（只改目标那一行，注释全部保留），按 `q` 退出不保存。

### 为什么要锁定曝光

HSV 阈值是写死的数字，而自动曝光会让画面亮度随环境光漂移。
同一块红色物块可能上午 V=200、下午 V=150，阈值下限 70 不变，
看起来"还能用"，直到某天光照再暗一点就整片丢失。

所以接真相机时，**先锁曝光和白平衡，再调阈值**。顺序反了等于白调。
