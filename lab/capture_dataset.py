"""边看边拍：开窗口给人摆位，同时每 N 秒存一张原图，用于补训练数据。

为什么不用 `python -m vision live`
    `live` 的两个模式都不适合"人一边摆、一边挑画面"：
      · 窗口模式不落盘；
      · `--headless --save-every` 不建窗口，人看不到自己在拍什么，且每 30 帧才存一张。

典型用途（2026-10 补数据时用过）
    · 远景小目标（塔在画面里只有 40–80 px）
    · 遮挡（前塔挡后塔）
    · **空场景负样本**（门框/墙/凳子/地贴，画面里不能有物块）——治环境误检最有效

用法
    PYTHONPATH=. python lab/capture_dataset.py --tag g6_far --exposure-us 20000 --interval 1.0
    PYTHONPATH=. python lab/capture_dataset.py --tag neg --exposure-us 10000 --out /tmp/neg

窗口左上角显示 `20ms saved=N`；按 q / ESC 退出。图片命名 `<out>/<tag>_000123.jpg`，
并始终覆盖一张 `latest.jpg` 便于远端（或脚本）看最新画面。

注意
    · 相机是独占资源：跑这个之前先关掉调参工具/MVS 客户端；
    · 存的是**原始帧**（没画框），可以直接当训练数据。
"""
import argparse
import os
import sys
import time

import cv2

from vision.sources import build_source

DEFAULT_OUT = "output/grab"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure-us", type=int, default=8000)
    ap.add_argument("--gain-db", type=float, default=10.0)
    ap.add_argument("--tag", default="shot", help="存图前缀，建议用采集组名（g6_far/neg）")
    ap.add_argument("--out", default=DEFAULT_OUT, help="落盘目录")
    ap.add_argument("--interval", type=float, default=1.0, help="每隔多少秒存一张")
    ap.add_argument("--frames", type=int, default=0, help="存满 N 张自动退出（0=按 q 才退出）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    src = build_source("camera", camera_params={"exposure_us": args.exposure_us,
                                               "gain_db": args.gain_db})
    win = "up70 capture   q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 768)

    t0 = time.monotonic()
    last, n_saved, n_frames = 0.0, 0, 0
    try:
        while True:
            frame = src.read()
            if frame is None:
                print("⚠️ 读帧失败，退出")
                break
            n_frames += 1
            img = frame.image_bgr
            now = time.monotonic()
            if now - last >= args.interval:
                path = os.path.join(args.out, f"{args.tag}_{n_saved:06d}.jpg")
                while os.path.exists(path):
                    n_saved += 1
                    path = os.path.join(args.out, f"{args.tag}_{n_saved:06d}.jpg")
                cv2.imwrite(path, img)
                cv2.imwrite(os.path.join(args.out, "latest.jpg"), img)
                n_saved += 1
                last = now
            disp = img.copy()
            cv2.putText(disp, f"{args.exposure_us // 1000}ms saved={n_saved}", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(win, disp)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if args.frames and n_saved >= args.frames:
                break
    finally:
        src.close()
        cv2.destroyAllWindows()
        fps = n_frames / max(1e-6, time.monotonic() - t0)
        print(f"结束：{n_frames} 帧 / 存 {n_saved} 张 / {fps:.1f} fps → {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
