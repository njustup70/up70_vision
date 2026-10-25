"""合成 YOLO 数据集生成器的回归测试。

为什么值得测
    生成器里最容易错的是**标注**：归一化 cx/cy 出界、w/h 为负、图和标签数量对不上，
    这三种错都会让训练"能跑但学不到东西"，而且不报错。
    还有两类只有肉眼看才发现、但可以用断言钉住的错：
      · **画了却没标**：先画后筛会留下无标签物体，模型把它们当背景学（2026-10-25 修过）；
      · **塔顶画错**：实物是红半砖+蓝半砖，框内必须同时出现红、蓝两种主色。

不测什么
    渲染好不好看（那是 preview 的活）；训练效果（要 ultralytics，CI 里没有）。
"""

import json
import pathlib
import sys
import tempfile

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
try:
    import vision  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))

from lab.gen_yolo_dataset import CLASS_NAMES, main as gen_main, render_one  # noqa: E402


def _gen(num: int, seed: int, out: pathlib.Path) -> pathlib.Path:
    code = gen_main(["--out", str(out), "--num", str(num),
                     "--val-ratio", "0.25", "--seed", str(seed)])
    assert code == 0, f"退出码 {code}"
    return out


def test_dataset_structure_and_labels_are_valid():
    with tempfile.TemporaryDirectory() as tmp:
        out = _gen(8, 2027, pathlib.Path(tmp) / "ds")
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        counts = {}
        for split, expect_max in (("train", 6), ("val", 2)):
            imgs = sorted((out / "images" / split).glob("*.jpg"))
            lbls = sorted((out / "labels" / split).glob("*.txt"))
            assert 0 < len(imgs) <= expect_max, f"{split} 图数 {len(imgs)} 不在 (0, {expect_max}]"
            assert len(imgs) == len(lbls), f"{split} 图与标签数量不一致"
            counts[split] = len(imgs)
        assert counts["train"] + counts["val"] == meta["written"] <= 8

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
    with tempfile.TemporaryDirectory() as tmp:
        out = _gen(8, 2027, pathlib.Path(tmp) / "ds")
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert [*meta["classes"]] == CLASS_NAMES
        assert meta["block_mm"] == 350.0 and meta["top_mm"] == 200.0

        yaml_text = (out / "dataset.yaml").read_text(encoding="utf-8")
        for key in ("path:", "train: images/train", "val: images/val", "block", "top"):
            assert key in yaml_text, f"dataset.yaml 缺少 {key!r}"

        labels = sum(len(p.read_text(encoding="utf-8").splitlines())
                     for p in (out / "labels").rglob("*.txt"))
        assert sum(meta["counts"].values()) == labels, "meta 里的框数与标签行数不一致"
        assert meta["written"] + meta["skipped_empty"] == meta["num"]


def test_same_seed_gives_same_labels():
    """固定 seed 必须可复现：标签逐字节一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        a = _gen(4, 7, pathlib.Path(tmp) / "a")
        first = {p.name: p.read_text(encoding="utf-8") for p in (a / "labels").rglob("*.txt")}
        b = _gen(4, 7, pathlib.Path(tmp) / "b")
        second = {p.name: p.read_text(encoding="utf-8") for p in (b / "labels").rglob("*.txt")}
        assert first == second


def test_every_drawn_object_has_a_label():
    """画出来的物体数 == 标签数（先算框后画图的回归防护）。"""

    class Args:  # 只给 render_one 用到的字段
        min_objects, max_objects = 1, 3
        top_ratio = 0.35
        min_dist_mm, max_dist_mm = 900.0, 3500.0
        camera_height_mm = 600.0
        min_side_px, margin_px = 12, 4

    intr = {"fx": 1321.67, "fy": 1323.56, "cx": 642.19, "cy": 532.83,
            "width": 1280, "height": 1024}
    for seed in range(6):
        rng = np.random.default_rng(seed)
        _, labels, n_drawn = render_one(rng, intr, Args())
        assert n_drawn == len(labels), f"seed={seed}: 画了 {n_drawn} 个但只有 {len(labels)} 个标签"


def test_top_blocks_contain_both_colours():
    """塔顶是"红半砖+蓝半砖"：每个 top 框内必须同时有红色和蓝色像素。

    这条能抓住"整面单色"的老 bug（早期版本顶面只涂一种颜色）。
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = _gen(12, 2027, pathlib.Path(tmp) / "ds")
        checked = 0
        for lbl_path in (out / "labels").rglob("*.txt"):
            img_path = out / "images" / lbl_path.parent.name / f"{lbl_path.stem}.jpg"
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            red = cv2.inRange(hsv, (0, 120, 60), (10, 255, 255)) | \
                cv2.inRange(hsv, (170, 120, 60), (180, 255, 255))
            blue = cv2.inRange(hsv, (100, 120, 60), (130, 255, 255))
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                cid, cx, cy, bw, bh = line.split()
                if int(cid) != 1:            # 只看 top
                    continue
                x1 = int((float(cx) - float(bw) / 2) * w)
                y1 = int((float(cy) - float(bh) / 2) * h)
                x2 = int((float(cx) + float(bw) / 2) * w)
                y2 = int((float(cy) + float(bh) / 2) * h)
                n_red = int(red[y1:y2, x1:x2].sum() // 255)
                n_blue = int(blue[y1:y2, x1:x2].sum() // 255)
                # 允许小框（<200 px）因为抗锯齿而少像素，但两种颜色都得出现
                assert n_red > 20 and n_blue > 20, \
                    f"{img_path.name}: 红 {n_red} 蓝 {n_blue}，塔顶应该是红蓝各一半"
                checked += 1
        assert checked > 0, "这批数据里没有塔顶，测试没验到东西"


def main() -> int:
    """不依赖 pytest 的独立运行器（镜像里没装 pytest，见视觉组协作规范）。"""
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  ❌ {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
