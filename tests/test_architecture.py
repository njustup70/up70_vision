"""架构约束测试：把「谁能 import 谁」变成会失败的断言（迁移计划书 D9）。

为什么写成测试
    写在文档里的架构约束，三个月后没人记得、半年后没人敢改。
    写成测试才有人执行——改错了 CI 立刻红，而不是等评审时被指出来。

四条规则（对应 D1 / D2 / D4 / D9）
    1. `aruco/` 与 `yolo/` 互不 import（两个功能互不相干，改一个不该碰另一个）
    2. 功能包不 import `cli`、`lab`（上层适配层与一次性工具，方向反了）
    3. 功能包不 import `config`（配置由 cli 拆成普通参数传进来）
    4. 根级 `types.py` 不 import 任何功能包（它是叶子）
    5. `vision/` 不 import `lab/`（lab 只能 import vision，反过来会让工具再也删不掉）

跑法：`python -m pytest tests/test_architecture.py -q`
"""

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
VISION = ROOT / "vision"

try:
    import vision  # noqa: F401  只是确认包在 import 路径上
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))


def _module_name(path: pathlib.Path) -> str:
    """文件路径 -> 模块名（`vision/aruco/detector.py` -> `vision.aruco.detector`）。"""
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports_of(path: pathlib.Path) -> list:
    """列出该文件 import 的所有模块名（相对 import 已按所在包解析成绝对名）。"""
    module = _module_name(path)
    package = module.rsplit(".", 1)[0] if "." in module else ""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # 相对 import：按当前包往上数
                base = package.split(".")
                keep = len(base) - (node.level - 1)
                prefix = ".".join(base[:keep])
                target = f"{prefix}.{node.module}" if node.module else prefix
            else:
                target = node.module or ""
            found.append(target)
    return found


def _vision_files():
    return [p for p in VISION.rglob("*.py")]


def test_aruco_and_yolo_do_not_import_each_other():
    for path in _vision_files():
        name = _module_name(path)
        imports = _imports_of(path)
        if name.startswith("vision.aruco"):
            bad = [m for m in imports if m.startswith("vision.yolo")]
            assert not bad, f"{name} 不该 import yolo：{bad}"
        elif name.startswith("vision.yolo"):
            bad = [m for m in imports if m.startswith("vision.aruco")]
            assert not bad, f"{name} 不该 import aruco：{bad}"


def test_feature_packages_do_not_import_cli_lab_or_config():
    for path in _vision_files():
        name = _module_name(path)
        if not (name.startswith("vision.aruco") or name.startswith("vision.yolo")):
            continue
        imports = _imports_of(path)
        for forbidden in ("vision.cli", "lab", "vision.config"):
            bad = [m for m in imports if m == forbidden or m.startswith(forbidden + ".")]
            assert not bad, f"{name} 不该 import {forbidden}：{bad}"


def test_root_types_is_a_leaf():
    imports = _imports_of(VISION / "types.py")
    bad = [m for m in imports
           if m.startswith(("vision.aruco", "vision.yolo", "vision.cli", "vision.sources", "vision.config"))]
    assert not bad, f"根级 types.py 必须保持叶子，不该 import：{bad}"


def test_vision_never_imports_lab():
    for path in _vision_files():
        imports = _imports_of(path)
        bad = [m for m in imports if m == "lab" or m.startswith("lab.")]
        assert not bad, f"{_module_name(path)} 不该 import lab（只能反过来）：{bad}"
