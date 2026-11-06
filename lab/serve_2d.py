"""2D 检测 HTTP 服务：把「一张 BGR 图 → 一串检测结果」暴露成 POST /detect。

为什么需要它
    主逻辑在另一台电脑、由另一人维护。契约已经冻结在
    `vision/docs/interface.md` 的「2D 检测输出：请求-响应接口」一节，但仓库里
    一直没有服务端实现——对方 curl 过来只会拿到 Connection refused。本文件补上这一端。

放什么 / 不放什么
    只做传输与调度：解析 JSON、解码图像、排队、把 `process_frame` 的结果序列化。
    **算法一行都不复制**：检测器构造、判色、字段都走 `vision` 原有链路，
    所以 HTTP 的结果与进程内调用必然一致（同一份 `Detection.to_dict()`）。

并发模型（与契约 3.4 节一致）
    单 worker + 队列深度 1。新帧到达时把**还没开始处理**的那帧丢掉：实时检测宁可
    丢帧也不要排队延迟。调用方等超过 `--timeout` 就收到 504，该帧被标记 cancelled，
    worker 取到时直接跳过，不白烧算力。

安全边界
    默认只绑 127.0.0.1。要暴露给别的机器必须显式 `--host 0.0.0.0`，而且本服务
    **没有鉴权**——只在可信局域网里用。

用法
    ~/.venvs/up70-2d/bin/python lab/serve_2d.py --port 8899
    curl -s localhost:8899/healthz | python3 -m json.tool
    curl 发图示例见 knowledge/docs/YOLO2D与主逻辑联调说明.md 3.3 节。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_VERSION = 1
SUPPORTED_ENCODINGS = ("png", "jpg", "jpeg", "raw-bgr")


class Job:
    """一帧待处理任务：结果放 result，错误放 error，done 用来唤醒 HTTP 线程。"""

    __slots__ = ("frame_id", "image", "result", "error", "infer_ms", "done", "cancelled")

    def __init__(self, frame_id: int, image: np.ndarray):
        self.frame_id = frame_id
        self.image = image
        self.result: list | None = None
        self.error: str | None = None
        self.infer_ms: float = 0.0
        self.done = threading.Event()
        self.cancelled = threading.Event()


class Engine:
    """常驻检测器 + 单 worker。启动时预热，避免第一帧吃模型加载时间。"""

    def __init__(self, config_path: str, params_override: dict, timeout_s: float):
        from vision.cli.run import build_detector
        from vision.config.loader import load_config, load_thresholds, resolve_thresholds_path
        from vision.yolo.pipeline import ColorConfig

        cfg, cfg_path = load_config(config_path, ["detector.name=yolo"])
        params = dict(cfg.detector.params)
        params.update(params_override)
        self.thresholds = load_thresholds(resolve_thresholds_path(cfg, cfg_path))
        if not self.thresholds:
            raise RuntimeError("没读到颜色阈值表，判色会全是 null；检查 configs/color_thresholds.yaml")
        self.detector = build_detector("yolo", params)
        self.color_cfg = ColorConfig(min_saturation=cfg.color.min_saturation,
                                     min_value=cfg.color.min_value,
                                     hit_ratio_min=cfg.color.hit_ratio_min,
                                     ignore_border_px=cfg.color.ignore_border_px,
                                     min_valid_pixels=cfg.color.min_valid_pixels)
        self.params = params
        self.config_text = (f"conf={params.get('conf')};iou={params.get('iou')};"
                            f"imgsz={params.get('imgsz')};classes={params.get('classes') or 'all'}")
        weights = str(params.get("weights", ""))
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"权重不存在：{weights}（相对仓库根解析）")
        with open(weights, "rb") as handle:
            self.model_sha256 = hashlib.sha256(handle.read()).hexdigest()
        self.model_name = os.path.basename(weights)

        self.timeout_s = float(timeout_s)
        self.queue: queue.Queue = queue.Queue(maxsize=1)
        self.dropped = 0
        self.served = 0
        self._lock = threading.Lock()

        started = time.perf_counter()
        self.detector.detect(np.zeros((64, 64, 3), np.uint8))    # 预热：把 ONNX 加载挡在建连之前
        self.warmup_ms = (time.perf_counter() - started) * 1000.0
        self.started_at = time.time()
        self._worker = threading.Thread(target=self._loop, name="infer", daemon=True)
        self._worker.start()

    # ---------------------------------------------------------------- 调度
    def submit(self, job: Job) -> None:
        """入队；满了就丢掉**旧**的那帧（实时场景丢新帧更糟）。"""
        try:
            self.queue.put_nowait(job)
        except queue.Full:
            try:
                old = self.queue.get_nowait()
                old.cancelled.set()
                old.done.set()              # 让等它的 HTTP 线程立刻醒，回 504
                with self._lock:
                    self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(job)
            except queue.Full:
                job.cancelled.set()
                job.done.set()

    def _loop(self) -> None:
        from vision.types import Frame
        from vision.yolo.pipeline import process_frame

        while True:
            job = self.queue.get()
            if job is None:
                return
            if job.cancelled.is_set():
                continue
            try:
                t0 = time.perf_counter()
                result = process_frame(Frame(job.image, index=job.frame_id, source="http"),
                                       self.detector, self.thresholds, self.color_cfg)
                job.result = [d.to_dict() for d in result.detections]
                job.infer_ms = (time.perf_counter() - t0) * 1000.0
                with self._lock:
                    self.served += 1
            except Exception as exc:                             # noqa: BLE001 —— 错误要回给调用方
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.done.set()

    def health(self) -> dict:
        with self._lock:
            served, dropped = self.served, self.dropped
        return {"ok": True, "api": API_VERSION, "model": self.model_name,
                "model_sha256": self.model_sha256,
                "config": self.config_text, "warmup_ms": round(self.warmup_ms, 1),
                "queue_depth": self.queue.qsize(), "served": served, "dropped": dropped,
                "uptime_s": round(time.time() - self.started_at, 1),
                "python": sys.version.split()[0], "cv2": cv2.__version__,
                "numpy": np.__version__}


def decode_image(spec: dict) -> np.ndarray:
    """把请求里的 image 段解成 HxWx3 uint8 BGR；形状不对抛 ValueError。"""
    encoding = str(spec.get("encoding", "")).lower()
    if encoding not in SUPPORTED_ENCODINGS:
        raise ValueError(f"encoding 只支持 {SUPPORTED_ENCODINGS}，收到 {encoding!r}")
    raw = base64.b64decode(spec.get("data", ""), validate=True)
    if not raw:
        raise ValueError("data 为空")
    if encoding == "raw-bgr":
        width, height = int(spec.get("width", 0)), int(spec.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError("raw-bgr 必须给 width/height")
        need = width * height * 3
        if len(raw) != need:
            raise ValueError(f"raw-bgr 长度 {len(raw)} 与 {width}x{height}x3={need} 不符")
        image = np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()
    else:
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("图像解码失败（编码标错或数据损坏）")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"只接受 HxWx3 的 BGR 图，收到 {image.shape}")
    return image


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None                                    # 由 make_server 注入
    max_frame_bytes: int = 8 * 1024 * 1024

    server_version = "up70-vision-2d/1"
    protocol_version = "HTTP/1.1"

    # ---------------------------------------------------------------- 路由
    def do_GET(self) -> None:                                # noqa: N802 —— BaseHTTPRequestHandler 的约定
        if self.path.split("?")[0] in ("/healthz", "/health"):
            self._json(200, self.engine.health())
        elif self.path == "/":
            self._json(200, {"api": API_VERSION, "endpoints": ["GET /healthz", "POST /detect"]})
        else:
            self._json(404, {"error": "unknown path"})

    def do_POST(self) -> None:                               # noqa: N802
        if self.path.split("?")[0] != "/detect":
            self._json(404, {"error": "unknown path"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._json(400, {"error": "Content-Length 缺失或为 0"})
            return
        if length > self.max_frame_bytes:
            self._json(413, {"error": f"请求体 {length} 超过上限 {self.max_frame_bytes}"})
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"error": "请求体不是合法 JSON"})
            return

        frame_id = payload.get("frame_id")
        if not isinstance(frame_id, int):
            self._json(400, {"error": "frame_id 必须是整数（它会原样回显，是双方唯一的关联键）"})
            return
        if int(payload.get("api", API_VERSION)) != API_VERSION:
            self._json(400, {"error": f"api 版本不支持：{payload.get('api')}"})
            return
        try:
            image = decode_image(payload.get("image") or {})
        except (ValueError, TypeError) as exc:
            self._json(400, {"error": str(exc)})
            return

        job = Job(frame_id, image)
        self.engine.submit(job)
        if not job.done.wait(self.engine.timeout_s):
            job.cancelled.set()
            self._json(504, {"api": API_VERSION, "frame_id": frame_id,
                             "error": f"推理超时（{int(self.engine.timeout_s * 1000)} ms），该帧已丢弃，不要重试",
                             "detections": []})
            return
        if job.error:
            self._json(500, {"api": API_VERSION, "frame_id": frame_id,
                             "error": job.error, "detections": []})
            return
        height, width = image.shape[:2]
        self._json(200, {"api": API_VERSION, "frame_id": frame_id, "image_wh": [int(width), int(height)],
                         "model": f"{self.engine.model_name}@sha256:{self.engine.model_sha256}",
                         "config": self.engine.config_text,
                         "infer_ms": round(job.infer_ms, 2),
                         "detections": job.result or [], "error": None})

    # ---------------------------------------------------------------- 工具
    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:          # 默认每条日志都刷屏，压到 stderr 一行
        sys.stderr.write("[http] " + (fmt % args) + "\n")


def make_server(host: str, port: int, engine: Engine, max_frame_bytes: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,),
                   {"engine": engine, "max_frame_bytes": max_frame_bytes})
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="2D 检测 HTTP 服务（契约见 vision/docs/interface.md）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址；给别的机器用要 0.0.0.0（无鉴权）")
    parser.add_argument("--port", type=int, default=8899, help="端口（默认 8899）")
    parser.add_argument("--config", default=os.path.join(REPO, "configs", "default.yaml"))
    parser.add_argument("--weights", default="", help="覆盖权重路径（相对仓库根）")
    parser.add_argument("--conf", type=float, default=None, help="覆盖置信度下限（默认取配置 0.45）")
    parser.add_argument("--imgsz", type=int, default=None, help="覆盖输入边长（默认取配置 640）")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "gpu"],
                        help="覆盖推理设备（默认取配置；cuda 需要 onnxruntime-gpu + NVIDIA 驱动）")
    parser.add_argument("--timeout", type=float, default=0.2, help="单帧等待上限，秒（默认 0.2）")
    parser.add_argument("--max-frame-bytes", type=int, default=8 * 1024 * 1024, help="请求体上限（默认 8 MB）")
    args = parser.parse_args(argv)

    os.chdir(REPO)                                           # 权重与阈值表都按相对仓库根解析
    overrides: dict = {}
    if args.weights:
        overrides["weights"] = args.weights
    if args.conf is not None:
        overrides["conf"] = args.conf
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.device:
        overrides["device"] = args.device

    try:
        engine = Engine(args.config, overrides, args.timeout)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"❌ 启动失败：{exc}", file=sys.stderr)
        return 2

    server = make_server(args.host, args.port, engine, args.max_frame_bytes)
    print(f"✅ 2D 检测服务：http://{args.host}:{args.port}")
    print(f"   模型 {engine.model_name} sha256 {engine.model_sha256[:16]}… | 预热 {engine.warmup_ms:.0f} ms")
    print(f"   配置 {engine.config_text}")
    if args.host not in ("127.0.0.1", "localhost"):
        print("⚠️  已绑定非本机地址，且本服务没有鉴权：确认只在可信局域网使用", file=sys.stderr)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
