"""gunicorn config for the Flow-GRPO GenEval reward server (github.com/yifan123/reward-server), as measured.

Four synchronous workers, one on each of the four training GPUs, on 127.0.0.1:18085. Run from the reward-server
checkout:  gunicorn -c /path/to/Faster-R3/scripts/reward/gunicorn_geneval.py "app_geneval:create_app()"
Set R3_REWARD_GPUS (default 4) and R3_REWARD_BIND to change the worker count or the address.
"""
import os

NUM_DEVICES = int(os.environ.get("R3_REWARD_GPUS", "4"))
USED_DEVICES = set()
bind = os.environ.get("R3_REWARD_BIND", "127.0.0.1:18085")
workers = NUM_DEVICES
worker_class = "sync"
timeout = 120


def pre_fork(server, worker):
    worker.device_id = next(i for i in range(NUM_DEVICES) if i not in USED_DEVICES)
    USED_DEVICES.add(worker.device_id)


def post_fork(server, worker):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.device_id)


def child_exit(server, worker):
    USED_DEVICES.discard(worker.device_id)
