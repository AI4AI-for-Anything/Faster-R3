"""Forward-only fused kernels for the BAGEL text ("und") inference path that reproduce the eager PyTorch
floating-point operations exactly (bitwise), so they can replace the eager op chains without changing results.

What is reproduced (BF16 activations, BF16 weights, as in the rollout under BF16 autocast):
  * Qwen2RMSNorm:  x.float(); x*x; mean(-1) ; + eps ; rsqrt ; x * r ; .to(bf16) ; weight * y   (each step rounded
    exactly like the separate eager kernels; FMA contraction disabled).  The only non-elementwise step is the mean:
    PyTorch's CUDA reduction (ATen/native/cuda/Reduce.cuh, MeanOps: sum then * factor) adds the row elements in an
    order fixed by its launch configuration, which depends on the row length *and* the number of rows.  The kernels
    below emulate that exact order (per-thread vector/stride accumulators, accumulator combine, shared-memory halving,
    warp shuffle-down tree) for the configurations in which one thread block row reduces one output row; any other
    configuration (cross-warp or cross-CTA split) falls back to eager.
  * residual add (bf16 + bf16 -> bf16) fused in front of the norm.
  * q_norm / k_norm (RMSNorm over head_dim 128, again with the row-count dependent reduction order) + RoPE
    (q*cos + rotate_half(q)*sin with bf16 rounding after each eager op).
  * SiLU(gate) * up: x / (1 + expf(-x)) (inline non-FTZ IEEE div.rn.f32, libdevice expf; its FTZ variant differs
    only for results below FLT_MIN, which vanish in 1 + e) rounded to bf16, then * up rounded to bf16.

Every kernel is validated bitwise against the eager ops (tools/fuse_decode/test_fused_ops.py). The dispatch
helpers return None when a configuration is not covered; callers then run the original eager code.
"""
import os

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False

ENABLED = os.environ.get('R3_FUSED_DECODE', '1') != '0'
# Run independent small-M GEMMs (q | k, v and gate | up) concurrently on a side stream: identical kernels and inputs,
# only their overlap changes. Applies to rows <= CONCURRENT_MAX_ROWS (decode).
CONCURRENT = os.environ.get('R3_FUSED_DECODE_STREAMS', '1') != '0'
CONCURRENT_MAX_ROWS = 16
_side_streams = {}


def side_stream_for(current):
    """A side stream distinct from `current` (pool streams are handed out round-robin, so a capture stream can
    coincide with a previously created side stream; then the next candidate is used)."""
    cands = _side_streams.get(current.device)
    if cands is None:
        cands = [torch.cuda.Stream(device=current.device) for _ in range(2)]
        _side_streams[current.device] = cands
    for s in cands:
        if s.cuda_stream != current.cuda_stream:
            return s
    return None


def concurrent_side(x):
    """Side stream for concurrent small-M GEMMs, or None (run them sequentially)."""
    if not (ENABLED and CONCURRENT and x.is_cuda and x.dim() == 2 and x.shape[0] <= CONCURRENT_MAX_ROWS):
        return None
    return side_stream_for(torch.cuda.current_stream(x.device))
stats = dict(rms=0, add_rms=0, qk=0, silu=0, fallback=0)


def _last_pow2(n):
    return 1 << (int(n).bit_length() - 1)


def torch_mean_block_width(num_rows, row_len):
    """(vectorized, block_width) of PyTorch's CUDA mean over the last dim of a contiguous, 16-byte aligned fp32
    [num_rows, row_len] tensor (setReduceConfig), or None when that configuration splits a row across warps/CTAs."""
    vectorized = row_len > 128 and row_len % 4 == 0
    dim0 = row_len // 4 if vectorized else row_len
    max_threads = 512
    d0 = max_threads if dim0 >= max_threads else _last_pow2(dim0)
    d1 = max_threads if num_rows >= max_threads else _last_pow2(num_rows)
    bw = min(d0, 32)
    bh = min(d1, max_threads // bw)
    bw = min(d0, max_threads // bh)
    values_per_thread = -(-row_len // bw)
    if values_per_thread >= min(bh * 16, 256):
        return None
    if bw < 32:
        return None
    return vectorized, bw


def torch_mean_factor(num_rows, row_len):
    # static_cast<float>(num_output_elements) / numel  (float / int64 -> float division)
    return float(np.float32(num_rows) / np.float32(num_rows * row_len))


if _HAVE_TRITON:

    @triton.jit
    def _tree_lane0(t, BW: tl.constexpr):
        # block_x_reduce: shared-memory halving down to one warp, then shuffle-down tree (lane 0's value).
        if BW >= 512:
            t = tl.sum(tl.reshape(t, (2, 256)), axis=0)
        if BW >= 256:
            t = tl.sum(tl.reshape(t, (2, 128)), axis=0)
        if BW >= 128:
            t = tl.sum(tl.reshape(t, (2, 64)), axis=0)
        if BW >= 64:
            t = tl.sum(tl.reshape(t, (2, 32)), axis=0)
        t = tl.sum(tl.reshape(t, (16, 2)), axis=1)
        t = tl.sum(tl.reshape(t, (8, 2)), axis=1)
        t = tl.sum(tl.reshape(t, (4, 2)), axis=1)
        t = tl.sum(tl.reshape(t, (2, 2)), axis=1)
        t = tl.sum(tl.reshape(t, (1, 2)), axis=1)
        return tl.sum(t, axis=0)

    @triton.jit
    def _tree_lane0_rows(t, R: tl.constexpr):
        # the 32-lane shuffle-down tree applied to each row of a [R, 32] tile
        t = tl.sum(tl.reshape(t, (R, 16, 2)), axis=2)
        t = tl.sum(tl.reshape(t, (R, 8, 2)), axis=2)
        t = tl.sum(tl.reshape(t, (R, 4, 2)), axis=2)
        t = tl.sum(tl.reshape(t, (R, 2, 2)), axis=2)
        t = tl.sum(tl.reshape(t, (R, 1, 2)), axis=2)
        return tl.reshape(t, (R,))

    @triton.jit
    def _elem(x_row, r_row, offs, mask, HAS_RES: tl.constexpr):
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_RES:
            r = tl.load(r_row + offs, mask=mask, other=0.0).to(tl.float32)
            x = (x + r).to(tl.bfloat16).to(tl.float32)
        return x

    @triton.jit
    def _rmsnorm_kernel(x_ptr, r_ptr, w_ptr, y_ptr, ro_ptr, m_ptr, eps, factor,
                        H: tl.constexpr, BW: tl.constexpr, HAS_RES: tl.constexpr, BLOCK: tl.constexpr,
                        DBG: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        x_row = x_ptr + row * H
        r_row = r_ptr + row * H
        # vectorized thread reduce (input_vec_size 4): lane x accumulates vectors x, x+BW, ... into 4 accumulators
        V: tl.constexpr = H // 4
        K: tl.constexpr = (V + BW - 1) // BW
        lane = tl.arange(0, BW)
        a0 = tl.zeros([BW], tl.float32)
        a1 = tl.zeros([BW], tl.float32)
        a2 = tl.zeros([BW], tl.float32)
        a3 = tl.zeros([BW], tl.float32)
        for k in tl.static_range(K):
            j = lane + BW * k
            m = j < V
            e = _elem(x_row, r_row, 4 * j + 0, m, HAS_RES)
            a0 = a0 + e * e
            e = _elem(x_row, r_row, 4 * j + 1, m, HAS_RES)
            a1 = a1 + e * e
            e = _elem(x_row, r_row, 4 * j + 2, m, HAS_RES)
            a2 = a2 + e * e
            e = _elem(x_row, r_row, 4 * j + 3, m, HAS_RES)
            a3 = a3 + e * e
        t = ((a0 + a1) + a2) + a3
        total = _tree_lane0(t, BW)
        mean = total * factor
        if DBG:
            tl.store(m_ptr + row, mean)
        rs = libdevice.rsqrt(mean + eps)
        offs = tl.arange(0, BLOCK)
        msk = offs < H
        x = _elem(x_row, r_row, offs, msk, HAS_RES)
        if HAS_RES:
            tl.store(ro_ptr + row * H + offs, x.to(tl.bfloat16), mask=msk)
        y = (x * rs).to(tl.bfloat16).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=msk, other=0.0).to(tl.float32)
        tl.store(y_ptr + row * H + offs, (w * y).to(tl.bfloat16), mask=msk)

    @triton.jit
    def _head_sumsq(base, rows, RP: tl.constexpr, NR: tl.constexpr, MODE32: tl.constexpr):
        # PyTorch mean over rows of 128 (non-vectorized thread reduce, vt0 = 4).
        #   block width 32 : lane x: ((s[x] + s[x+32]) + s[x+64]) + s[x+96], then the 32-lane tree
        #   block width 64/128: lane x (<32) after halving: (s[x] + s[x+64]) + (s[x+32] + s[x+96]), then the tree
        lane = tl.arange(0, 32)
        rm = (rows < NR)[:, None]
        p = base + rows[:, None] * 128 + lane[None, :]
        c0 = tl.load(p, mask=rm, other=0.0).to(tl.float32)
        c1 = tl.load(p + 32, mask=rm, other=0.0).to(tl.float32)
        c2 = tl.load(p + 64, mask=rm, other=0.0).to(tl.float32)
        c3 = tl.load(p + 96, mask=rm, other=0.0).to(tl.float32)
        c0 = c0 * c0
        c1 = c1 * c1
        c2 = c2 * c2
        c3 = c3 * c3
        if MODE32:
            t = ((c0 + c1) + c2) + c3
        else:
            t = (c0 + c2) + (c1 + c3)
        return _tree_lane0_rows(t, RP)

    @triton.jit
    def _head_norm_rope(base, out_base, w_ptr, cos_row, sin_row, rs, rows, RP: tl.constexpr, NR: tl.constexpr):
        d = tl.arange(0, 128)
        partner = (d + 64) % 128
        rm = (rows < NR)[:, None]
        x = tl.load(base + rows[:, None] * 128 + d[None, :], mask=rm, other=0.0).to(tl.float32)
        xp = tl.load(base + rows[:, None] * 128 + partner[None, :], mask=rm, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + d).to(tl.float32)
        wp = tl.load(w_ptr + partner).to(tl.float32)
        r = rs[:, None]
        n = (w[None, :] * (x * r).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        npart = (wp[None, :] * (xp * r).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        rot = tl.where((d < 64)[None, :], -npart, npart)
        c = tl.load(cos_row + d).to(tl.float32)[None, :]
        s = tl.load(sin_row + d).to(tl.float32)[None, :]
        t1 = (n * c).to(tl.bfloat16).to(tl.float32)
        t2 = (rot * s).to(tl.bfloat16).to(tl.float32)
        tl.store(out_base + rows[:, None] * 128 + d[None, :], (t1 + t2).to(tl.bfloat16), mask=rm)

    @triton.jit
    def _qk_norm_rope_kernel(q_ptr, k_ptr, wq_ptr, wk_ptr, cos_ptr, sin_ptr, qo_ptr, ko_ptr, mq_ptr, mk_ptr,
                             cos_stride, eps, fq, fk,
                             NQH: tl.constexpr, NKH: tl.constexpr, RQ: tl.constexpr, RK: tl.constexpr,
                             Q32: tl.constexpr, K32: tl.constexpr, DBG: tl.constexpr):
        tok = tl.program_id(0).to(tl.int64)
        cos_row = cos_ptr + tok * cos_stride
        sin_row = sin_ptr + tok * cos_stride
        qb = q_ptr + tok * (NQH * 128)
        kb = k_ptr + tok * (NKH * 128)
        rq = tl.arange(0, RQ)
        rk = tl.arange(0, RK)
        sq = _head_sumsq(qb, rq, RQ, NQH, Q32)
        sk = _head_sumsq(kb, rk, RK, NKH, K32)
        if DBG:
            tl.store(mq_ptr + tok * NQH + rq, sq * fq, mask=rq < NQH)
            tl.store(mk_ptr + tok * NKH + rk, sk * fk, mask=rk < NKH)
        rsq = libdevice.rsqrt(sq * fq + eps)
        rsk = libdevice.rsqrt(sk * fk + eps)
        _head_norm_rope(qb, qo_ptr + tok * (NQH * 128), wq_ptr, cos_row, sin_row, rsq, rq, RQ, NQH)
        _head_norm_rope(kb, ko_ptr + tok * (NKH * 128), wk_ptr, cos_row, sin_row, rsk, rk, RK, NKH)

    @triton.jit
    def _silu_mul_kernel(g_ptr, u_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
        d = 1.0 + libdevice.exp(-g)
        # IEEE non-FTZ division (ATen's float '/'); libdevice div_rn would take the FTZ path under Triton's reflect
        s = tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=f,f,f", [g, d], dtype=tl.float32, is_pure=True,
                                      pack=1)
        s = s.to(tl.bfloat16).to(tl.float32)
        tl.store(o_ptr + offs, (s * u).to(tl.bfloat16), mask=m)


def _ok(*ts):
    return _HAVE_TRITON and ENABLED and all(t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous() for t in ts)


def rms_norm(x, weight, eps, residual=None, debug_mean=None):
    """weight * RMSNorm(x) exactly as Qwen2RMSNorm (bf16 in/out). With residual: x := bf16(x + residual) first and
    returns (normed, x). Returns None when not covered (caller runs eager)."""
    if x.dim() != 2 or not _ok(x, weight) or (residual is not None and (not _ok(residual) or residual.shape != x.shape)):
        stats['fallback'] += 1
        return None
    n, h = x.shape
    cfg = torch_mean_block_width(n, h)
    if cfg is None or not cfg[0] or weight.numel() != h or n == 0:
        stats['fallback'] += 1
        return None
    bw = cfg[1]
    y = torch.empty_like(x)
    ro = torch.empty_like(x) if residual is not None else y
    _rmsnorm_kernel[(n,)](x, residual if residual is not None else x, weight, y, ro,
                          debug_mean if debug_mean is not None else y, float(eps),
                          torch_mean_factor(n, h), H=h, BW=bw, HAS_RES=residual is not None,
                          BLOCK=triton.next_power_of_2(h), DBG=debug_mean is not None, num_warps=4,
                          enable_fp_fusion=False)
    if residual is not None:
        stats['add_rms'] += 1
        return y, ro
    stats['rms'] += 1
    return y


def qk_norm_rope(q, k, wq, wk, cos, sin, eps, debug_means=None):
    """q_norm(q), k_norm(k) then apply_rotary_pos_emb, exactly; q [n, Hq, 128], k [n, Hk, 128], cos/sin [n, 128]."""
    if (q.dim() != 3 or k.dim() != 3 or q.shape[-1] != 128 or k.shape[-1] != 128 or q.shape[0] != k.shape[0]
            or not _ok(q, k, wq, wk) or cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16
            or cos.dim() != 2 or cos.shape != (q.shape[0], 128) or sin.shape != cos.shape or cos.stride() != sin.stride()
            or cos.stride(1) != 1 or q.shape[0] == 0):
        stats['fallback'] += 1
        return None
    n, hq, _ = q.shape
    hk = k.shape[1]
    cq = torch_mean_block_width(n * hq, 128)
    ck = torch_mean_block_width(n * hk, 128)
    if cq is None or ck is None or cq[0] or ck[0]:
        stats['fallback'] += 1
        return None
    qo = torch.empty_like(q)
    ko = torch.empty_like(k)
    mq, mk = debug_means if debug_means is not None else (qo, ko)
    _qk_norm_rope_kernel[(n,)](q, k, wq, wk, cos, sin, qo, ko, mq, mk, cos.stride(0), float(eps),
                               torch_mean_factor(n * hq, 128), torch_mean_factor(n * hk, 128),
                               NQH=hq, NKH=hk, RQ=triton.next_power_of_2(hq), RK=triton.next_power_of_2(hk),
                               Q32=cq[1] == 32, K32=ck[1] == 32, DBG=debug_means is not None, num_warps=4,
                               enable_fp_fusion=False)
    stats['qk'] += 1
    return qo, ko


def silu_mul(gate, up):
    """act_fn(gate) * up for SiLU, exactly (bf16)."""
    if not _ok(gate, up) or gate.shape != up.shape:
        stats['fallback'] += 1
        return None
    out = torch.empty_like(gate)
    n = gate.numel()
    BLOCK = 2048
    _silu_mul_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n, BLOCK=BLOCK, num_warps=8, enable_fp_fusion=False)
    stats['silu'] += 1
    return out
