"""Capture like ``with torch.cuda.graph(graph, stream=stream)`` without its unconditional host preparation.

``torch.cuda.graph`` enters by synchronizing the whole device and releasing every cached allocator block, so
each rollout capture made the caching allocator return and then re-request large segments (cudaFree/cudaMalloc
churn while the GPU idles). The capture itself is unchanged here: same graph object, capture stream, private
memory pool, generator registration and "global" capture mode, so the recorded kernels and their arguments are
identical. Releasing cached blocks is only a memory measure, and the allocator cannot release blocks while a
capture is underway, so the original synchronize + release is still performed whenever free device memory is
below the reserve needed by a capture's private pool.
"""
import torch

FREE_RESERVE_BYTES = 4 << 30
stats = dict(lean=0, full=0, min_free_gib=None, max_pool_gib=0.0)


def capture(graph, stream, fn):
    free, _ = torch.cuda.mem_get_info(stream.device)
    gib = free / 2**30
    stats['min_free_gib'] = gib if stats['min_free_gib'] is None else min(stats['min_free_gib'], gib)
    if free < FREE_RESERVE_BYTES:
        stats['full'] += 1
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    else:
        stats['lean'] += 1
    reserved = torch.cuda.memory_reserved(stream.device)
    with torch.cuda.stream(stream):
        graph.capture_begin(capture_error_mode="global")
        try:
            output = fn()
        finally:
            graph.capture_end()
    grown = (torch.cuda.memory_reserved(stream.device) - reserved) / 2**30
    stats['max_pool_gib'] = max(stats['max_pool_gib'], grown)
    return output


def report():
    line = (f"lean={stats['lean']} full={stats['full']} "
            f"min_free_gib={-1.0 if stats['min_free_gib'] is None else stats['min_free_gib']:.2f} "
            f"max_capture_growth_gib={stats['max_pool_gib']:.2f}")
    stats.update(lean=0, full=0, min_free_gib=None, max_pool_gib=0.0)
    return line
