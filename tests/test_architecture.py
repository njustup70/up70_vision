"""架构约束测试：把「谁能 import 谁」变成会失败的断言（迁移计划书 D9）。

为什么写成测试
    写在文档里的架构约束，三个月后没人记得、半年后没人敢改。
    写成测试才有人执行——改错了 CI 立刻红，而不是等评审时被指出来。

六条规则（对应 D1 / D2 / D4 / D9 + ROS2 接入方案 6.4）
    1. 功能包之间互不 import（`aruco/` `yolo/` `depth/` 各管一摊，
       改一个不该碰另一个）
    2. 功能包不 import `cli`、`lab`（上层适配层与一次性工具，方向反了）
    3. 功能包不 import `config`（配置由 cli 拆成普通参数传进来）
    4. 根级 `types.py` 不 import 任何功能包（它是叶子）
    5. `vision/` 不 import `lab/`（lab 只能 import vision，反过来会让工具再也删不掉）
    6. **`vision/` 里任何文件都不 import ROS**（rclpy / sensor_msgs / vision_msgs …）
       这条最不能靠人守：`rclpy` 只能 apt 装，pip 装不了，一旦混进来
       `pip install -e .` 会直接失败，队友克隆仓库后跑不起来，而且报错信息
       指向的是「模块找不到」，很难联想到是架构越界。

跑法
    python tests/test_architecture.py      # 直接跑
    python -m pytest tests/test_architecture.py -q
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

#: 功能包：彼此之间、以及与架构层之间都有严格的依赖方向
FEATURE_PACKAGES = ("vision.aruco", "vision.yolo", "vision.depth")

#: ROS 侧模块的前缀。视功能包一旦 import 它们就再也装不进纯 pip 环境。
ROS_ROOTS = ("rclpy", "sensor_msgs", "vision_msgs", "geometry_msgs",
             "tf2_ros", "std_msgs", "ament_index_python", "launch")

#: 架构层：功能包不许反向依赖
ARCH_LAYERS = ("vision.cli", "vision.config", "vision.sources", "lab")


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


def _feature_of(module_name: str):
    """判断模块属于哪个功能包，不属于则返回 None。"""
    for package in FEATURE_PACKAGES:
        if module_name == package or module_name.startswith(package + "."):
            return package
    return None


def _matches(module_name: str, prefix: str) -> bool:
    return module_name == prefix or module_name.startswith(prefix + ".")


def test_feature_packages_are_mutually_isolated():
    """功能包之间互不 import。三个包各管一摊，串起来就变成一坨。"""
    for path in _vision_files():
        name = _module_name(path)
        owner = _feature_of(name)
        if owner is None:
            continue
        for target in _imports_of(path):
            target_owner = _feature_of(target)
            assert target_owner is None or target_owner == owner, (
                f"{name}（属于 {owner}）不该 import {target}（属于 {target_owner}）"
            )


def test_feature_packages_do_not_import_architecture_layers():
    """功能包不 import cli / config / sources / lab：依赖方向只能是上层指向下层。"""
    for path in _vision_files():
        name = _module_name(path)
        if _feature_of(name) is None:
            continue
        for target in _imports_of(path):
            for forbidden in ARCH_LAYERS:
                assert not _matches(target, forbidden), (
                    f"{name} 不该 import {target}（{forbidden} 属于上层或工具，方向反了）"
                )


def test_root_types_is_a_leaf():
    """根级 types.py 只放跨功能共用件，不许反向 import 功能包。"""
    imports = _imports_of(VISION / "types.py")
    bad = [m for m in imports
           if any(_matches(m, p) for p in FEATURE_PACKAGES + ("vision.cli", "vision.sources"))]
    assert not bad, f"根级 types.py 必须保持叶子，不该 import：{bad}"


def test_vision_never_imports_lab():
    """vision 不能 import lab：lab 是一次性工具，反过来依赖会让它再也删不掉。"""
    for path in _vision_files():
        for target in _imports_of(path):
            assert not _matches(target, "lab"), (
                f"{_module_name(path)} 不该 import {target}（只能反过来）"
            )


def test_vision_never_imports_ros():
    """vision/ 里任何文件都不许 import ROS。

    这是路线二的硬约束。`rclpy` 只能 `apt install ros-*-rclpy`，pip 装不了，
    混进 `vision/` 会让 `pip install -e .` 直接失败——队友克隆仓库跑不起来，
    而且报错只说「模块找不到」，看不出是架构越界。
    """
    for path in _vision_files():
        for target in _imports_of(path):
            root = target.split(".")[0]
            assert root not in ROS_ROOTS, (
                f"{_module_name(path)} 不该 import ROS 模块 {target}："
                f"ROS 外壳放 ros2_ws/src/vision_depth/，算法保持纯 Python"
            )


def main() -> int:
    """把上面几条逐个跑一遍，方便在没有 pytest 的镜像里直接执行。"""
    checks = [
        ("功能包互不 import", test_feature_packages_are_mutually_isolated),
        ("功能包不碰架构层", test_feature_packages_do_not_import_architecture_layers),
        ("根级 types 是叶子", test_root_types_is_a_leaf),
        ("vision 不 import lab", test_vision_never_imports_lab),
        ("vision 不 import ROS", test_vision_never_imports_ros),
    ]
    failures = []
    for label, check in checks:
        try:
            check()
            print(f"[通过] {label}")
        except AssertionError as exc:
            print(f"[失败] {label}：{exc}")
            failures.append(label)

    print("-" * 50)
    if failures:
        print(f"结果：{len(failures)} 项未通过")
        return 1
    print("结果：全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
