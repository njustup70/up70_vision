"""半自动标注小工具：预填自动框，人只负责"拖新框 / 删错框 / 改类别"。

为什么要有它：自动标注在"同色相邻"（塔顶压在塔基上）和远景小目标上必错，
纯算法修不干净；而整份数据只有 120 张、每张 1~4 个目标，人工修一遍是最省总时间的路。

操作（窗口里）：
    左键拖拽        新建一个框（用当前类别）
    右键点击        删除鼠标下最上层的框
    数字 1 / 2 / 3  切换当前类别（1=block_red 2=block_blue 3=top）
    u               撤销最后新建的框
    n / 空格        下一张（自动保存）
    p               上一张
    s               保存当前
    q / ESC         退出（已标过的都存过盘）

输出：一条 YOLO 标签一行 `cls cx cy w h`（归一化），写到 --out 目录，
文件名与图片同名（.txt）。已存在的标签会被预填，所以可以接着上次标。

用法（在仓库根执行）：
    PYTHONPATH=. python lab/label_tool.py \
        --images data/yolo_real/g6_far \
        --prefill data/yolo_real_labels/g6_far \
        --out data/yolo_real_labels/g6_far

类别：`1` = block（350 方块）、`2` = top（200 半红半蓝塔顶）。
标签存成 YOLO 格式 `cls cx cy w h`（0=block 1=top），**不含颜色**（颜色运行时由 HSV 判）。
"""
import argparse
import glob
import os

import cv2

#: 与训练一致的两类：0=block（350 方块）、1=top（200 半红半蓝塔顶）。
#: 键用 1/2 只是方便按键盘，存盘时减 1 还原成 YOLO 的 0/1。
NAMES = {1: "block", 2: "top"}
COLORS = {1: (0, 165, 255), 2: (0, 200, 0)}
#: 窗口名必须是纯 ASCII：OpenCV 的 Qt 后端用非 ASCII 名字建窗后，
#: setMouseCallback 会找不到窗口句柄，报 "NULL window handler"。
WIN = "up70 label tool  |  LMB drag = new box, RMB = delete, 1/2/3 = class, n = next, q = quit"


def load_labels(path: str) -> list[tuple[int, float, float, float, float]]:
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            p = line.split()
            if len(p) == 5:
                out.append((int(p[0]), *map(float, p[1:])))
    return out


def save_labels(path: str, boxes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for cls, cx, cy, w, h in boxes:
            fh.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/yolo_real")
    ap.add_argument("--prefill", default="output/auto_labels")
    ap.add_argument("--out", default="output/labels_manual")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, "**", "*.jpg"), recursive=True))
    if not files:
        print(f"❌ {args.images} 下没有 jpg")
        return 1

    # 鼠标回调要用的共享状态：Qt 后端必须先 imshow 过再 setMouseCallback，
    # 否则报 "NULL window handler"。所以这里先建窗口、显示一帧，再挂回调。
    state = {"boxes": [], "cls": 1, "drag": None, "shape": (1, 1, 3)}

    def on_mouse(event, x, y, flags, param):
        boxes = state["boxes"]
        h, w = state["shape"][:2]
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["drag"][2:4] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
            x0, y0 = state["drag"][0], state["drag"][1]
            state["drag"] = None
            if abs(x - x0) > 6 and abs(y - y0) > 6:
                x1, x2 = sorted((x0, x)); y1, y2 = sorted((y0, y))
                boxes.append((state["cls"], (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                              (x2 - x1) / w, (y2 - y1) / h))
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i in range(len(boxes) - 1, -1, -1):
                _, cx, cy, bw, bh = boxes[i]
                if abs(cx * w - x) < bw * w / 2 and abs(cy * h - y) < bh * h / 2:
                    boxes.pop(i)
                    break

    first = cv2.imread(files[0])
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1100, 880)
    cv2.imshow(WIN, first)
    cv2.waitKey(1)
    cv2.setMouseCallback(WIN, on_mouse)

    idx = 0
    while True:
        path = files[idx]
        img = cv2.imread(path)
        h, w = img.shape[:2]
        state["shape"] = img.shape
        stem = os.path.relpath(path, args.images).replace(os.sep, "__").rsplit(".", 1)[0]
        dst = os.path.join(args.out, stem + ".txt")
        pre = os.path.join(args.prefill, stem + ".txt")
        if os.path.exists(dst):
            state["boxes"] = load_labels(dst)
        elif os.path.exists(pre):
            state["boxes"] = load_labels(pre)
        else:
            state["boxes"] = []

        while True:
            vis = img.copy()
            for cls, cx, cy, bw, bh in state["boxes"]:
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS[cls + 1], 2)
                cv2.putText(vis, NAMES[cls + 1], (x1 + 2, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[cls + 1], 2, cv2.LINE_AA)
            if state["drag"]:
                x0, y0, x1, y1 = state["drag"]
                cv2.rectangle(vis, (x0, y0), (x1, y1), COLORS[state["cls"] + 1], 1)
            cv2.putText(vis, f"[{idx + 1}/{len(files)}] class: {NAMES[state['cls'] + 1]}  boxes: {len(state['boxes'])}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, os.path.basename(path), (8, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
            cv2.imshow(WIN, vis)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                save_labels(dst, state["boxes"])
                cv2.destroyAllWindows()
                print(f"退出，已保存 {dst}")
                return 0
            if key in (ord("1"), ord("2")):
                state["cls"] = key - ord("1")
            elif key == ord("u") and state["boxes"]:
                state["boxes"].pop()
            elif key == ord("s"):
                save_labels(dst, state["boxes"])
            elif key in (ord("n"), ord(" ")):
                save_labels(dst, state["boxes"])
                idx = min(idx + 1, len(files) - 1)
                break
            elif key == ord("p"):
                save_labels(dst, state["boxes"])
                idx = max(idx - 1, 0)
                break


if __name__ == "__main__":
    raise SystemExit(main())
