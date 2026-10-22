# assets/aruco —— ArUco 测试图组（图 + 真值）

这批图是**检测核心的自检素材**：验收标准是"同一张图片中的多个 ArUco 码全部正确识别
（ID、四角、中心）"，所以每张图都配一份同名 `.json`，写清这张图里**应该**检出哪些码、
它们的中心和四角在哪个像素上。测试与自检脚本拿它当标准答案，不用人工标点。

```
assets/aruco/
├── markers/          单码图：一个 ID 一张，码居中、四周留白 80 px
│   ├── marker_00.png     + marker_00.json
│   ├── marker_01.png     + marker_01.json
│   └── marker_02.png     + marker_02.json
└── scenes/           场景图
    ├── multi_row_3.png       横排 3 个码（ID 0/1/2，都不旋转）
    ├── multi_grid_6.png      3x2 排 6 个码（ID 0..5）
    ├── rotated_3.png         3 个码分别转 0° / 30° / 60°
    ├── rotated_single_45.png 单个码转 45°（四角变成菱形，专门查角点顺序）
    ├── empty_plain.png       无码图：纯渐变背景
    └── empty_blocks.png      无码图：5 个实心色块（专门查误报）
```

## 怎么跑

```bash
# 生成 / 重新生成整套（输出逐字节可复现，同一参数重跑结果完全一样）
python -m up70_vision.tools.aruco.gen_scenes
python -m up70_vision.tools.aruco.gen_scenes --list-scenes        # 看有哪些场景
python -m up70_vision.tools.aruco.gen_scenes --ids 0 1 2 3 --side-px 120

# 拿图组对真值自检：ID、四角、中心逐张报数字
python -m up70_vision.tools.aruco.check
python -m up70_vision.tools.aruco.check --only rotated --max-err-px 1.5

# 走完整管线（检测 + 判色 + 落盘），把图组当回放素材喂进去
python scripts/run.py --set detector.name=aruco \
  --set input.source=replay --set input.replay.path=assets/aruco/scenes

# 单元测试（容器里跑，宿主 apt 版 cv2 4.5.4 没有新版 aruco API，会自动跳过）
python tests/test_aruco.py
```

**容器里跑**（标准环境，cv2 4.14）：

```bash
docker compose -f .devcontainer/docker-compose.yml run --rm up70-vision \
  python -m up70_vision.tools.aruco.check
```

实测（2026-10-16，cv2 4.14.0）：9/9 张通过、**中心最大误差 0.00 px、四角最大误差 0.23 px**。

## 真值 JSON 长什么样

```json
{
  "scene": "multi_row_3",
  "image": "scenes/multi_row_3.png",
  "dictionary": "DICT_4X4_50",
  "width": 760,
  "height": 520,
  "generator": "up70_vision.tools.aruco.gen_scenes",
  "params": {"ids": [0, 1, 2], "side_px": 130, "cols": 3, "gap_px": 60, "angle_deg": 0.0},
  "markers": [
    {"id": 0, "center": [190.0, 260.0],
     "corners": [[125.0, 195.0], [255.0, 195.0], [255.0, 325.0], [125.0, 325.0]],
     "side_px": 130, "angle_deg": 0.0}
  ]
}
```

- `corners` 顺序与契约一致：**TL → TR → BR → BL**（码自身的左上角起、顺时针）。
  比对时是按下标一一对应的，所以**顺序错了误差会立刻从 <1 px 跳到几百 px**。
- 四角取的是黑框**最外沿**（像素边界 ±0.5），不是最外侧黑像素的中心。
  差这半个像素，四角误差会从 ~0.1 px 变成 ~0.5 px。
- 真值不是人工标的：`gen_aruco.py` 贴图时用的仿射矩阵**就是**算真值用的那个矩阵，
  所以误差只来自插值（实测 ≤ 0.23 px）。

## 加新场景

改 `src/up70_vision/tools/aruco/gen_scenes.py` 里的场景构造函数，重跑生成器；
新图与真值 JSON 会一起写出来，`aruco_check` 与 `tests/test_aruco.py` 自动带上它——
两者都是遍历目录的，不需要再登记一份清单。

单张图别超过 10 MB（`assets/README.md` 的约定）。现在整套 176 KB。
