"""Measured faster GenEval server: eight workers, four CPU threads each.

The reward model, weights, crop order and scoring arithmetic are unchanged.
Colour crops are loaded in-process instead of forking four workers per request.
Run from the upstream reward-server checkout; do not enable gunicorn preload.
"""
import os
import socket

NUM_DEVICES = int(os.environ.get("R3_REWARD_GPUS", "8"))
USED_DEVICES = set()
bind = os.environ.get("R3_REWARD_BIND", "127.0.0.1:18085")
workers = NUM_DEVICES
worker_class = "sync"
timeout = 120

PER_GPU_PORT_BASE = 0           # off in harness v1 (per-rank routing measured: no gain once the server is fast)
WORKER_THREADS = "4"             # 8 workers x 4 = 32 OpenMP threads on the measured host
COLOR_LOADER_WORKERS = 0         # build colour crops in-process (no 4-process fork per colour check)


def pre_fork(server, worker):
    worker.device_id = next(i for i in range(NUM_DEVICES) if i not in USED_DEVICES)
    USED_DEVICES.add(worker.device_id)


def post_fork(server, worker):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.device_id)
    if WORKER_THREADS:
        os.environ["OMP_NUM_THREADS"] = WORKER_THREADS
        os.environ["MKL_NUM_THREADS"] = WORKER_THREADS
    if COLOR_LOADER_WORKERS is not None:
        import torch.utils.data as tud
        base = tud.DataLoader

        class ColorCropLoader(base):
            def __init__(self, *args, **kwargs):
                kwargs["num_workers"] = COLOR_LOADER_WORKERS
                super().__init__(*args, **kwargs)

        tud.DataLoader = ColorCropLoader   # gen_eval resolves torch.utils.data.DataLoader at call time


def post_worker_init(worker):
    if PER_GPU_PORT_BASE:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", PER_GPU_PORT_BASE + worker.device_id))
        s.listen(128)
        s.setblocking(0)
        worker.sockets.append(s)
        worker.wait_fds = worker.sockets + [worker.PIPE[0]]


def child_exit(server, worker):
    USED_DEVICES.discard(worker.device_id)
