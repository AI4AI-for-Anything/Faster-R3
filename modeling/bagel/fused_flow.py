"""Forward-only fused kernels for the readonly static-KV MoT image-flow backbone (tier N0: bitwise identical).

Every kernel reproduces the floating-point operations of the eager PyTorch code it replaces, in the same order:

* fp32 row reductions follow ``at::native`` Reduce.cuh exactly (per-lane accumulators over the same strided
  elements, the 4-way accumulator combine, the shared-memory halving stage and the warp ``shfl_down`` tree);
  the block width is derived from the slice's row count with the same ``setReduceConfig`` formulas, and any
  configuration not reproduced here makes the caller fall back to the eager path;
* elementwise chains keep every eager intermediate rounding (bf16 stores, separate fp32 products and sums);
  kernels are launched with ``enable_fp_fusion=False`` so no multiply-add is contracted into an FMA;
* ``exp`` / ``rsqrt`` use libdevice's accurate ``expf`` / ``rsqrtf`` (the device functions behind ATen's
  ``::exp`` / ``::rsqrt``; their FTZ variants differ only for subnormal arguments/results, which cannot reach
  the stored values here: rsqrt sees var + eps >= eps, and an expf result below FLT_MIN vanishes in 1 + e);
  division is an inline non-FTZ IEEE ``div.rn.f32``, exactly ATen's float ``/``.

Only data movement differs: concatenations, fp32 staging copies, permutation gathers and scatters into the merged
K/V buffer are folded into the kernels' loads and stores.
"""
import os

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

ENABLED = os.environ.get('R3_FUSE_FLOW', '1') != '0'
stats = dict(fused_forwards=0, eager_fallbacks=0, reasons={})


def _note_fallback(reason):
    stats['eager_fallbacks'] += 1
    stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1


# ----------------------------------------------------------------------------------------------------------
# ATen inner-reduction launch configuration (Reduce.cuh setReduceConfig, CUDA, MAX_NUM_THREADS=512, vt0=4)
# ----------------------------------------------------------------------------------------------------------
def _last_pow2(n):
    n = int(n)
    for s in (1, 2, 4, 8, 16, 32):
        n |= n >> s
    return max(1, n - (n >> 1))


def inner_reduce_config(row_len, num_rows):
    """(vectorized, block_width, split_across_warps) for a contiguous fp32 mean over the last dim."""
    mnt = 512
    vec = row_len > 128            # contiguous fp32, one reduce dim, vt0 >= input_vec_size
    dim0 = row_len // 4 if vec else row_len
    dim1 = num_rows
    dim0_pow2 = _last_pow2(dim0) if dim0 < mnt else mnt
    dim1_pow2 = _last_pow2(dim1) if dim1 < mnt else mnt
    bw = min(dim0_pow2, 32)
    bh = min(dim1_pow2, mnt // bw)
    bw = min(dim0_pow2, mnt // bh)
    values_per_thread = -(-row_len // bw)
    split = values_per_thread >= min(bh * 16, 256)
    return vec, bw, split


def mean_factor(num_rows, row_len):
    """MeanOps factor: static_cast<float>(num_outputs) / numel, evaluated in float."""
    return float(np.float32(num_rows) / np.float32(np.int64(num_rows) * row_len))


def hidden_bw(num_rows, row_len):
    vec, bw, split = inner_reduce_config(row_len, num_rows)
    if not vec or split or bw not in (32, 64, 128) or (row_len // 4) % bw:
        return None
    return bw


def head_ok(num_rows, row_len=128):
    vec, bw, split = inner_reduce_config(row_len, num_rows)
    return (not vec) and (not split) and bw == 32 and row_len == 128


# ----------------------------------------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------------------------------------
@triton.jit
def _ld4(ptr, BW: tl.constexpr):
    """Contiguous [BW*4] bf16 load split into the 4 per-lane accumulator streams (element 4*lane + i)."""
    blk = tl.reshape(tl.load(ptr + tl.arange(0, BW * 4)), [BW, 2, 2])
    lo, hi = tl.split(blk)          # i in {0, 2} / {1, 3}
    v0, v2 = tl.split(lo)
    v1, v3 = tl.split(hi)
    return v0, v1, v2, v3


@triton.jit
def _st4(ptr, v0, v1, v2, v3, BW: tl.constexpr):
    tl.store(ptr + tl.arange(0, BW * 4), tl.reshape(tl.join(tl.join(v0, v2), tl.join(v1, v3)), [BW * 4]))


@triton.jit
def _rms_hidden_kernel(x_ptr, a_ptr, w_ptr, h_ptr, y_ptr, oidx_ptr, eps, factor,
                          H: tl.constexpr, BW: tl.constexpr, HAS_ADD: tl.constexpr, WRITE_H: tl.constexpr,
                          HAS_OIDX: tl.constexpr):
    """One row of H: optional h = bf16(x + a); y = bf16(w * bf16(h * rsqrt(mean(h^2) + eps))) (eager Qwen2RMSNorm).
    Lane l accumulates elements 4*(it*BW + l) + i in accumulator i, it = 0..NIT-1 in order (ATen vectorized
    inner reduction); then ((a0+a1)+a2)+a3, the shared-memory halving to 32 lanes and the shfl_down tree.
    With HAS_OIDX the normalized row r is stored at output row oidx[r] (fused unpermute)."""
    row = tl.program_id(0).to(tl.int64)
    NIT: tl.constexpr = H // 4 // BW
    xr = x_ptr + row * H
    ar = a_ptr + row * H
    hr = h_ptr + row * H
    if HAS_OIDX:
        yr = y_ptr + tl.load(oidx_ptr + row) * H
    else:
        yr = y_ptr + row * H
    a0 = tl.zeros([BW], tl.float32)
    a1 = tl.zeros([BW], tl.float32)
    a2 = tl.zeros([BW], tl.float32)
    a3 = tl.zeros([BW], tl.float32)
    for it in tl.range(0, NIT):
        v0, v1, v2, v3 = _ld4(xr + it * BW * 4, BW)
        if HAS_ADD:
            b0, b1, b2, b3 = _ld4(ar + it * BW * 4, BW)
            v0 = (v0.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)
            v1 = (v1.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)
            v2 = (v2.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)
            v3 = (v3.to(tl.float32) + b3.to(tl.float32)).to(tl.bfloat16)
            if WRITE_H:
                _st4(hr + it * BW * 4, v0, v1, v2, v3, BW)
        f0 = v0.to(tl.float32)
        f1 = v1.to(tl.float32)
        f2 = v2.to(tl.float32)
        f3 = v3.to(tl.float32)
        a0 = a0 + f0 * f0
        a1 = a1 + f1 * f1
        a2 = a2 + f2 * f2
        a3 = a3 + f3 * f3
    s = ((a0 + a1) + a2) + a3
    if BW == 128:
        s = tl.sum(tl.reshape(s, [2, 64]), axis=0)
    if BW >= 64:
        s = tl.sum(tl.reshape(s, [2, 32]), axis=0)
    s = tl.sum(tl.reshape(s, [16, 2]), axis=1)
    s = tl.sum(tl.reshape(s, [8, 2]), axis=1)
    s = tl.sum(tl.reshape(s, [4, 2]), axis=1)
    s = tl.sum(tl.reshape(s, [2, 2]), axis=1)
    s = tl.sum(s, axis=0)
    r = libdevice.rsqrt(s * factor + eps)
    for it in tl.range(0, NIT):
        v0, v1, v2, v3 = _ld4(xr + it * BW * 4, BW)
        w0, w1, w2, w3 = _ld4(w_ptr + it * BW * 4, BW)
        if HAS_ADD:
            b0, b1, b2, b3 = _ld4(ar + it * BW * 4, BW)
            v0 = (v0.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)
            v1 = (v1.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)
            v2 = (v2.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)
            v3 = (v3.to(tl.float32) + b3.to(tl.float32)).to(tl.bfloat16)
        n0 = (v0.to(tl.float32) * r).to(tl.bfloat16)
        n1 = (v1.to(tl.float32) * r).to(tl.bfloat16)
        n2 = (v2.to(tl.float32) * r).to(tl.bfloat16)
        n3 = (v3.to(tl.float32) * r).to(tl.bfloat16)
        _st4(yr + it * BW * 4, (w0.to(tl.float32) * n0.to(tl.float32)).to(tl.bfloat16),
             (w1.to(tl.float32) * n1.to(tl.float32)).to(tl.bfloat16),
             (w2.to(tl.float32) * n2.to(tl.float32)).to(tl.bfloat16),
             (w3.to(tl.float32) * n3.to(tl.float32)).to(tl.bfloat16), BW)


@triton.jit
def _head_norm_rope_kernel(xt_ptr, xg_ptr, wt_ptr, wg_ptr, cos_ptr, sin_ptr, inv_ptr, dest_ptr, out_ptr,
                           vt_ptr, vg_ptr, vout_ptr, T, eps,
                           NH: tl.constexpr, NHP: tl.constexpr, HAS_DEST: tl.constexpr, HAS_V: tl.constexpr):
    """Original-order row i <- sorted row j = inverse[i]: per-head RMSNorm (fp32, expert weight by j < T),
    then RoPE with the row's bf16 cos/sin in fp32, stored as bf16 at row i (or merged row dest[i])."""
    i = tl.program_id(0).to(tl.int64)
    j = tl.load(inv_ptr + i)
    hh = tl.arange(0, NHP)[:, None]
    ll = tl.arange(0, 32)[None, :]
    hm = hh < NH
    off = hh * 128 + ll
    if j < T:
        xr = xt_ptr + j * (NH * 128)
        wr = wt_ptr
    else:
        xr = xg_ptr + (j - T) * (NH * 128)
        wr = wg_ptr
    x0 = tl.load(xr + off, mask=hm, other=0.0).to(tl.float32)
    x1 = tl.load(xr + off + 32, mask=hm, other=0.0).to(tl.float32)
    x2 = tl.load(xr + off + 64, mask=hm, other=0.0).to(tl.float32)
    x3 = tl.load(xr + off + 96, mask=hm, other=0.0).to(tl.float32)
    a = ((x0 * x0 + x1 * x1) + x2 * x2) + x3 * x3
    a = tl.sum(tl.reshape(a, [NHP, 16, 2]), axis=2)
    a = tl.sum(tl.reshape(a, [NHP, 8, 2]), axis=2)
    a = tl.sum(tl.reshape(a, [NHP, 4, 2]), axis=2)
    a = tl.sum(tl.reshape(a, [NHP, 2, 2]), axis=2)
    a = tl.sum(a, axis=1)
    r = libdevice.rsqrt(a * 0.0078125 + eps)[:, None]
    z0 = tl.load(wr + ll).to(tl.float32) * (x0 * r)
    z1 = tl.load(wr + ll + 32).to(tl.float32) * (x1 * r)
    z2 = tl.load(wr + ll + 64).to(tl.float32) * (x2 * r)
    z3 = tl.load(wr + ll + 96).to(tl.float32) * (x3 * r)
    cr = cos_ptr + i * 128 + ll
    sr = sin_ptr + i * 128 + ll
    o0 = z0 * tl.load(cr).to(tl.float32) + (-z2) * tl.load(sr).to(tl.float32)
    o1 = z1 * tl.load(cr + 32).to(tl.float32) + (-z3) * tl.load(sr + 32).to(tl.float32)
    o2 = z2 * tl.load(cr + 64).to(tl.float32) + z0 * tl.load(sr + 64).to(tl.float32)
    o3 = z3 * tl.load(cr + 96).to(tl.float32) + z1 * tl.load(sr + 96).to(tl.float32)
    if HAS_DEST:
        orow = tl.load(dest_ptr + i)
    else:
        orow = i
    orr = out_ptr + orow * (NH * 128) + off
    tl.store(orr, o0.to(tl.bfloat16), mask=hm)
    tl.store(orr + 32, o1.to(tl.bfloat16), mask=hm)
    tl.store(orr + 64, o2.to(tl.bfloat16), mask=hm)
    tl.store(orr + 96, o3.to(tl.bfloat16), mask=hm)
    if HAS_V:
        if j < T:
            vr = vt_ptr + j * (NH * 128)
        else:
            vr = vg_ptr + (j - T) * (NH * 128)
        vo = vout_ptr + orow * (NH * 128) + off
        for c in tl.static_range(4):
            tl.store(vo + c * 32, tl.load(vr + off + c * 32, mask=hm), mask=hm)


@triton.jit
def _past_copy_kernel(kc_ptr, vc_ptr, b_ptr, p_ptr, d_ptr, mk_ptr, mv_ptr, stride_b, stride_p, ROW: tl.constexpr):
    """merged[dest[r]] = cache[batch[r], pos[r]] for K and V (rows of ROW contiguous bf16)."""
    r = tl.program_id(0).to(tl.int64)
    b = tl.load(b_ptr + r)
    p = tl.load(p_ptr + r)
    d = tl.load(d_ptr + r)
    offs = tl.arange(0, ROW)
    src = b * stride_b + p * stride_p + offs
    tl.store(mk_ptr + d * ROW + offs, tl.load(kc_ptr + src))
    tl.store(mv_ptr + d * ROW + offs, tl.load(vc_ptr + src))


@triton.jit
def _silu_mul_kernel(g_ptr, u_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """bf16(float(bf16(g / (1 + exp(-g)))) * float(u)) as ATen silu followed by a bf16 mul."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
    d = 1.0 + libdevice.exp(-g)
    # IEEE div.rn.f32 without flush-to-zero (libdevice's div_rn is .ftz; ATen's float '/' keeps subnormals)
    s = tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=r,r,r", [g, d], dtype=tl.float32, is_pure=True,
                                  pack=1)
    s = s.to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + offs, (s * u).to(tl.bfloat16), mask=m)


# ----------------------------------------------------------------------------------------------------------
# launch wrappers
# ----------------------------------------------------------------------------------------------------------
_NOFMA = dict(enable_fp_fusion=False)


def rms_rows(x, w, eps, add=None, h_out=None, y_out=None, num_warps=1, write_h=True, out_index=None,
             cfg_rows=None):
    """Eager ``Qwen2RMSNorm`` on bf16 rows (optionally after ``h = x + add``); returns (h or None, y).
    write_h=False skips storing x + add; out_index scatters normalized rows to y_out[out_index].
    cfg_rows: row count of the eager norm call being reproduced (rows of several stacked CFG branches are
    normalized independently, each with the reduction order eager uses for one branch)."""
    rows, H = x.shape
    eager_rows = rows if cfg_rows is None else cfg_rows
    bw = hidden_bw(eager_rows, H)
    assert bw is not None
    y = torch.empty_like(x) if y_out is None else y_out
    write_h = add is not None and write_h
    if write_h and h_out is None:
        h_out = torch.empty_like(x)
    if rows:
        _rms_hidden_kernel[(rows,)](x, add if add is not None else x, w, h_out if write_h else x, y,
                                       out_index if out_index is not None else x, eps, mean_factor(eager_rows, H),
                                       H=H, BW=bw, HAS_ADD=add is not None, WRITE_H=write_h,
                                       HAS_OIDX=out_index is not None, num_warps=num_warps, **_NOFMA)
    return (h_out if write_h else None), y


def head_norm_rope(xt, xg, wt, wg, cos, sin, inverse, eps, nh, out, dest=None, vt=None, vg=None, vout=None):
    n = inverse.shape[0]
    T = xt.shape[0]
    _head_norm_rope_kernel[(n,)](xt, xg, wt, wg, cos, sin, inverse, dest if dest is not None else inverse, out,
                                 vt if vt is not None else xt, vg if vg is not None else xg,
                                 vout if vout is not None else out, T, eps, NH=nh, NHP=triton.next_power_of_2(nh),
                                 HAS_DEST=dest is not None, HAS_V=vt is not None, num_warps=4 if nh > 8 else 1,
                                 **_NOFMA)


def past_copy(k_cache, v_cache, batch, pos, dest, merged_k, merged_v):
    P = batch.shape[0]
    if P == 0:
        return
    row = k_cache.shape[-2] * k_cache.shape[-1]
    _past_copy_kernel[(P,)](k_cache, v_cache, batch, pos, dest, merged_k, merged_v,
                            k_cache.stride(0), k_cache.stride(1), ROW=row, num_warps=4)


def silu_mul(g, u):
    out = torch.empty_like(g)
    n = g.numel()
    _silu_mul_kernel[(triton.cdiv(n, 2048),)](g, u, out, n, BLOCK=2048, num_warps=8, **_NOFMA)
    return out
