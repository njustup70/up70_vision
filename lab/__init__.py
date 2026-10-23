"""一次性工具（`lab/`）：零架构约束，可以互相 import，可以写长脚本。

和 `vision/` 的关系（迁移计划书 D1）
    `lab/` 可以 `import vision`（复用算法）；
    **`vision/` 绝对不能 import `lab/`** —— 否则一次性工具会被运行时代码依赖，
    再也没法删。`tests/test_architecture.py` 会检查这一条。

为什么这个文件存在（计划里本来说 `lab/` 不是包）
    测试的"真值生成器"在 `lab/gen_aruco.py`、`lab/pose_check/gen_scenes.py`，
    测试要 `from lab.gen_aruco import ...`；没有 `__init__.py` 就 import 不了。
    它**不作为分发包**：`pyproject.toml` 的 include 只有 `vision*`。

跑法
    脚本式：`python lab/check_env.py`（每个文件自带 sys.path 引导）
    模块式：`python -m lab.pose_check --dir output/pose_scenes`
"""
