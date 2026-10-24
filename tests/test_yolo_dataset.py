"""合成 YOLO 数据集生成器的回归测试。

为什么值得测
    生成器里最容易错的是**标注坐标系**：归一化 cx/cy 落在 [0,1] 之外、w/h 为负、
    或者图像与标签数量对不上——这三种错都会让训练"能跑但学不到东西"，而且不报错。
    另外它要能反复跑出同样结果（固定 seed），否则数据无法复现。

不测什么
    渲染好不好看（那是 preview 的活）；训练效果（要 ultralytics，CI 里没有）。
"""

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
try:
    import vision  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))

from lab.gen_yolo_dataset import CLASS_NAMES, main as gen_main  # noqa: E402

OUT = ROOT / "output" / "_yolo_ds_test"


def _gen(num: int, seed: int) -> pathlib.Path:
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    code = gen_main(["--out", str(OUT), "--num", str(num),
                     "--val-ratio", "0.25", "--seed", str(seed)])
    assert code == 0, f"退出码 {code}"
    return OUT


def test_dataset_structure_and_labels_are_valid():
    out = _gen(8, 2027)

    for split, expect in (("train", 6), ("val", 2)):
        imgs = sorted((out / "images" / split).glob("*.jpg"))
        lbls = sorted((out / "labels" / split).glob("*.txt"))
        assert len(imgs) == expect, f"{split} 图数 {len(imgs)} != {expect}"
        assert len(imgs) == len(lbls), f"{split} 图与标签数量不一致"

    total = 0
    for path in (out / "labels").rglob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            assert len(parts) == 5, f"标签列数不对：{line!r}"
            cid = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
            assert 0 <= cid < len(CLASS_NAMES), f"类别越界：{cid}"
            assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0, f"中心越界：{line!r}"
            assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0, f"宽高越界：{line!r}"
            total += 1
    assert total > 0, "一个框都没生成"


def test_meta_and_yaml_are_consistent():
    out = _gen(8, 2027)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert [*meta["classes"]] == CLASS_NAMES
    assert meta["block_mm"] == 350.0 and meta["top_mm"] == 200.0

    yaml_text = (out / "dataset.yaml").read_text(encoding="utf-8")
    for key in ("path:", "train: images/train", "val: images/val", "block", "top"):
        assert key in yaml_text, f"dataset.yaml 缺少 {key!r}"

    labels = sum(len(p.read_text(encoding="utf-8").splitlines())
                 for p in (out / "labels").rglob("*.txt"))
    assert sum(meta["counts"].values()) == labels, "meta 里的框数与标签行数不一致"


def test_same_seed_gives_same_labels():
    """固定 seed 必须可复现：标签逐字节一致（渲染细节允许有随机，但标注不能飘）。"""
    a = _gen(4, 7)
    first = {p.name: p.read_text(encoding="utf-8") for p in (a / "labels").rglob("*.txt")}
    b = _gen(4, 7)
    second = {p.name: p.read_text(encoding="utf-8") for p in (b / "labels").rglob("*.txt")}
    assert first == second


def test_preview_is_written():
    out = _gen(4, 3)
    assert (out / "preview.jpg").is_file(), "没有预览图，没法肉眼验标注"
