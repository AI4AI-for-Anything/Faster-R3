"""Shared 80 GB GPU adaptation used for both baseline and candidate.

Enable FSDP2 ZeRO-3, copy the frozen reference directly to this rank's GPU,
and optionally stagger model loading. Tensor values and RNG order are unchanged.
Call apply(train.online_rl, log_dir) before main() under torchrun.
"""
import copy, gc, json, os, time
from datetime import timedelta

import psutil
import torch
import torch.distributed as dist

_applied = False


def apply(entry, log_dir, load_concurrency=None, zero3=True):
    global _applied
    if _applied:
        return
    _applied = True
    from train import fsdp2_utils

    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    K = int(load_concurrency if load_concurrency is not None else os.environ.get("R3_LOAD_CONCURRENCY", "8"))
    if zero3:
        fsdp2_utils.FSDP2Config.get_reshard_after_forward = lambda self: True

    os.makedirs(log_dir, exist_ok=True)
    loadlog = open(os.path.join(log_dir, f"load_rank{rank}.jsonl"), "a", buffering=1)

    def memlog(tag):
        cg = -1.0
        try:
            st = dict(l.split() for l in open("/sys/fs/cgroup/memory/memory.stat"))
            cg = (int(st.get("total_rss", 0)) + int(st.get("total_shmem", 0))) / 2**30
        except Exception:
            pass
        gpu = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_initialized() else 0.0
        loadlog.write(json.dumps(dict(t=round(time.time(), 2), tag=tag, K=K,
            rss_gib=round(psutil.Process().memory_info().rss / 2**30, 2),
            cgroup_rss_shmem_gib=round(cg, 2), gpu_alloc_gib=round(gpu, 2))) + "\n")

    WAIT = timedelta(hours=3)
    state = {"released": False, "entered": False}

    def store():
        return dist.distributed_c10d._get_default_store()

    Qwen = entry.Qwen2ForCausalLM
    orig_init = Qwen.__init__

    def init_in_turn(self, *a, **k):
        if not state["entered"]:
            state["entered"] = True
            memlog("wait_turn")
            if rank - K >= 0:
                store().wait([f"r3_load_done_{rank - K}"], WAIT)
            memlog("turn_start")
        orig_init(self, *a, **k)
    Qwen.__init__ = init_in_turn

    orig_deepcopy = copy.deepcopy

    def deepcopy_to_gpu(obj, memo=None):
        if not isinstance(obj, torch.nn.Module):
            return orig_deepcopy(obj, memo)
        memlog("ref_copy_start")
        dev = torch.device("cuda", torch.cuda.current_device())
        memo = {} if memo is None else memo
        with torch.no_grad():
            for p in obj.parameters():
                if id(p) not in memo:
                    memo[id(p)] = torch.nn.Parameter(p.detach().to(dev, copy=True), requires_grad=p.requires_grad)
            for b in obj.buffers():
                if id(b) not in memo:
                    memo[id(b)] = b.detach().to(dev, copy=True)
        out = orig_deepcopy(obj, memo)
        del memo
        memlog("ref_copy_done")
        return out
    entry.deepcopy = deepcopy_to_gpu

    orig_apply = entry.apply_fsdp2_with_activation_checkpointing

    def apply_then_release(model, cfg, *a, **k):
        out = orig_apply(model, cfg, *a, **k)
        if k.get("use_ac", True) and not state["released"]:
            gc.collect()
            memlog("sharded_release")
            state["released"] = True
            store().set(f"r3_load_done_{rank}", "1")
            store().wait([f"r3_load_done_{i}" for i in range(world)], WAIT)
            memlog("all_loaded")
        return out
    entry.apply_fsdp2_with_activation_checkpointing = apply_then_release
    memlog("adapt_applied")
