"""fuse_train: fused Triton kernels for the MoT training forward/backward (reference forward, policy forward, AC
recompute, backward).  Each kernel reproduces the floating-point operations of the eager code it replaces bit for bit:

  * same rounding points (every eager bf16 tensor is rounded to bf16 at the same place, fp32 math elsewhere),
  * same reduction order as ATen's reduce kernel for these shapes (see _row_sum_* below),
  * no FMA contraction (kernels are compiled with enable_fp_fusion=False) except where the ATen functor itself is
    contracted by nvcc, which is then written as an explicit fma,
  * same libdevice math (expf, rsqrtf) and IEEE division (div_rn).

Whenever a precondition of that argument does not hold (dtype, width, row count, strides) the eager code runs instead.
Exactness is checked by tools/fuse_train/test_fused_ops.py (bitwise forward and all gradients against eager).
"""
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_ENABLED = os.environ.get("R3_FUSED_TRAIN_OPS", "1") != "0"
_KW = dict(enable_fp_fusion=False)

if os.environ.get("TRITON_INTERPRET") == "1":   # CPU interpreter (logic tests only): numpy math
    @triton.jit
    def _rsqrt(x):
        return tl.math.rsqrt(x)

    @triton.jit
    def _exp(x):
        return tl.math.exp(x)

    @triton.jit
    def _div_rn(a, b):
        return a / b

    @triton.jit
    def _fma(a, b, c):
        return a * b + c
else:                                            # compiled: CUDA libdevice = what nvcc links into ATen
    @triton.jit
    def _rsqrt(x):
        return libdevice.rsqrt(x)                # rsqrtf (FTZ variant; argument >= eps, never denormal)

    @triton.jit
    def _exp(x):
        return libdevice.exp(x)                  # expf (accurate; tl.exp would be ex2.approx). FTZ variant: only
                                                 # flushes |result| < 2^-126, which is only ever added to 1 here

    # Triton links libdevice with __CUDA_FTZ=1 (nvidia.set_nvvm_reflect_ftz), so libdevice div_rn / fma flush
    # denormals; ATen (nvcc, no -ftz) does not.  Plain Triton fp32 ops are IEEE.  Use the non-FTZ PTX directly.
    @triton.jit
    def _div_rn(a, b):
        return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32,
                                         is_pure=True, pack=1)

    @triton.jit
    def _fma(a, b, c):
        return tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=r,r,r,r", [a, b, c], dtype=tl.float32,
                                         is_pure=True, pack=1)


# ----------------------------------------------------------------------------------------------------------------
# ATen reduce order (Reduce.cuh, torch 2.9, sum/mean over the contiguous last dim of a [R, N] fp32 tensor, R >= 16):
#   block = 32 lanes x 16 rows, one warp per row, no block_y / global reduce (values_per_thread < 256).
#   N = 3584 (> 128): vectorize_input with input_vec_size 4: lane t accumulates value_list[i] over the vectors
#       t, t+32, t+64, ... (element (32k + t)*4 + i), sequentially in k; then ((v0 + v1) + v2) + v3.
#   N = 128 (not > 128): thread_reduce_impl, vt0 = 4: lane t holds ((x[t] + x[t+32]) + x[t+64]) + x[t+96].
#   Then block_x_reduce: warp shfl_down with offsets 1, 2, 4, 8, 16 -> lane 0 = adjacent pairwise tree.
# ----------------------------------------------------------------------------------------------------------------
@triton.jit
def _combine4(acc, BLOCK_R: tl.constexpr):
    # acc [R, 32, 4] -> ((a0 + a1) + a2) + a3, [R, 32]
    acc = tl.reshape(acc, (BLOCK_R, 32, 2, 2))
    e, o = tl.split(acc)          # e: i in {0, 2}; o: i in {1, 3}
    a0, a2 = tl.split(e)
    a1, a3 = tl.split(o)
    return ((a0 + a1) + a2) + a3


@triton.jit
def _lane_tree(v, BLOCK_R: tl.constexpr):
    # v [R, 32] -> [R]: shfl_down(1,2,4,8,16) result in lane 0
    v = tl.reshape(v, (BLOCK_R, 16, 2))
    a, b = tl.split(v)
    v = a + b
    v = tl.reshape(v, (BLOCK_R, 8, 2))
    a, b = tl.split(v)
    v = a + b
    v = tl.reshape(v, (BLOCK_R, 4, 2))
    a, b = tl.split(v)
    v = a + b
    v = tl.reshape(v, (BLOCK_R, 2, 2))
    a, b = tl.split(v)
    v = a + b
    v = tl.reshape(v, (BLOCK_R, 1, 2))
    a, b = tl.split(v)
    v = a + b
    return tl.reshape(v, (BLOCK_R,))


@triton.jit
def _offsets(N: tl.constexpr):
    # [32, 4] element offsets of chunk k = 0 (vectorized: (t, i) -> 4t + i; narrow N=128: (t, i) -> t + 32 i)
    t = tl.arange(0, 32)
    i = tl.arange(0, 4)
    if N == 128:
        return t[:, None] + 32 * i[None, :]
    else:
        return 4 * t[:, None] + i[None, :]


@triton.jit
def _rms_fwd_kernel(X, W, Y, RSTD, n_rows, factor, eps, N: tl.constexpr, BLOCK_R: tl.constexpr):
    K: tl.constexpr = N // 128
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_rows
    off = _offsets(N)
    base = rows.to(tl.int64)[:, None, None] * N + off[None, :, :]
    m3 = rmask[:, None, None]
    acc = tl.zeros((BLOCK_R, 32, 4), tl.float32)
    for k in tl.static_range(K):
        xv = tl.load(X + base + k * 128, mask=m3, other=0.0).to(tl.float32)
        acc = acc + xv * xv                                   # pow(2) (= x*x), then the reduce add
    total = _lane_tree(_combine4(acc, BLOCK_R), BLOCK_R)
    var = total * factor                                       # MeanOps::project
    r = _rsqrt(var + eps)                             # torch.rsqrt(variance + eps)
    for k in tl.static_range(K):
        xv = tl.load(X + base + k * 128, mask=m3, other=0.0).to(tl.float32)
        wv = tl.load(W + off + k * 128).to(tl.float32)
        h = (xv * r[:, None, None]).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + base + k * 128, (wv[None, :, :] * h).to(tl.bfloat16), mask=m3)
    tl.store(RSTD + rows, r, mask=rmask)


@triton.jit
def _rms_bwd_kernel(GY, X, W, RSTD, GX, PROD, n_rows, inv_n, N: tl.constexpr, BLOCK_R: tl.constexpr,
                    PLUS_ZERO: tl.constexpr):
    K: tl.constexpr = N // 128
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_rows
    off = _offsets(N)
    base = rows.to(tl.int64)[:, None, None] * N + off[None, :, :]
    m3 = rmask[:, None, None]
    r = tl.load(RSTD + rows, mask=rmask, other=1.0)
    r3 = r[:, None, None]
    acc = tl.zeros((BLOCK_R, 32, 4), tl.float32)
    for k in tl.static_range(K):
        xv = tl.load(X + base + k * 128, mask=m3, other=0.0).to(tl.float32)
        gy = tl.load(GY + base + k * 128, mask=m3, other=0.0).to(tl.float32)
        wv = tl.load(W + off + k * 128).to(tl.float32)[None, :, :]
        hb = (xv * r3).to(tl.bfloat16).to(tl.float32)
        tl.store(PROD + base + k * 128, (gy * hb).to(tl.bfloat16), mask=m3)   # grad_w before sum_to
        gh = (gy * wv).to(tl.bfloat16).to(tl.float32)                         # grad of h.to(bf16)
        acc = acc + gh * xv                                                   # (grad * xf).sum(-1)
    s = _lane_tree(_combine4(acc, BLOCK_R), BLOCK_R)
    gv = (s * -0.5) * ((r * r) * r)                                            # rsqrt backward
    gsq = (gv * inv_n)[:, None, None]                                          # mean backward
    for k in tl.static_range(K):
        xv = tl.load(X + base + k * 128, mask=m3, other=0.0).to(tl.float32)
        gy = tl.load(GY + base + k * 128, mask=m3, other=0.0).to(tl.float32)
        wv = tl.load(W + off + k * 128).to(tl.float32)[None, :, :]
        gh = (gy * wv).to(tl.bfloat16).to(tl.float32)
        g1 = gh * r3                                                           # mul backward (x side)
        g2 = gsq * (xv * 2.0)                                                  # pow backward
        gx = (g1 + g2).to(tl.bfloat16)
        if PLUS_ZERO:   # what the scatter of a gather backward (0 + g) does: -0 -> +0
            gx = (gx.to(tl.float32) + 0.0).to(tl.bfloat16)
        tl.store(GX + base + k * 128, gx, mask=m3)


_BLOCK_R = {3584: 4, 128: 16}


def _rms_ok(x, weight):
    n = x.shape[-1]
    return (_ENABLED and x.is_cuda and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
            and n in _BLOCK_R and weight.shape == (n,) and x.numel() // n >= 16 and x.numel() < 2 ** 31
            and weight.is_contiguous())


def _mean_factor(rows, n):
    # ATen mean_kernel_impl: factor = float(num_outputs) / numel (int64 -> float), all in fp32
    return float(torch.tensor(rows, dtype=torch.float32) / torch.tensor(rows * n, dtype=torch.float32))


def _rms_fwd(x2, weight, eps):
    rows, n = x2.shape
    y = torch.empty_like(x2)
    rstd = torch.empty(rows, dtype=torch.float32, device=x2.device)
    br = _BLOCK_R[n]
    _rms_fwd_kernel[(triton.cdiv(rows, br),)](x2, weight, y, rstd, rows, _mean_factor(rows, n), eps, N=n,
                                              BLOCK_R=br, num_warps=4, **_KW)
    return y, rstd


class _FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, plus_zero):
        shape = x.shape
        x2 = x.contiguous().view(-1, shape[-1])
        y, rstd = _rms_fwd(x2, weight, eps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = shape
        ctx.plus_zero = plus_zero
        return y.view(shape)

    @staticmethod
    def backward(ctx, gy):
        x2, weight, rstd = ctx.saved_tensors
        rows, n = x2.shape
        gy2 = gy.contiguous().view(rows, n)
        gx = torch.empty_like(x2)
        prod = torch.empty_like(x2)
        br = _BLOCK_R[n]
        inv_n = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(float(n), dtype=torch.float32))
        _rms_bwd_kernel[(triton.cdiv(rows, br),)](gy2, x2, weight, rstd, gx, prod, rows, inv_n, N=n, BLOCK_R=br,
                                                  PLUS_ZERO=ctx.plus_zero, num_warps=4, **_KW)
        gw = None
        if ctx.needs_input_grad[1]:
            # autograd: sum_to(grad * h_bf16, weight.shape) = sum over the leading dims (keepdim) of the
            # ORIGINAL-shaped product, then view_copy
            shape = ctx.shape
            prod_s = prod.view(shape)
            gw = prod_s.sum(tuple(range(len(shape) - 1)), keepdim=True).view(n)
        return gx.view(ctx.shape), gw, None, None


def rms_norm(module, x, plus_zero=False):
    """Qwen2RMSNorm.forward(x) (bitwise), fused when possible.  plus_zero: the input gradient is returned as
    (grad + 0), i.e. exactly what eager produces when x was gathered (x = t[idx]) and the gather's backward
    scatters it into zeros; used by the split-residual layout, which drops that gather."""
    w = module.weight
    if not _rms_ok(x, w):
        assert not plus_zero, 'split layout requires the fused norm'
        return module(x)
    if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
        return _FusedRMSNorm.apply(x, w, module.variance_epsilon, bool(plus_zero))
    shape = x.shape
    y, _ = _rms_fwd(x.contiguous().view(-1, shape[-1]), w, module.variance_epsilon)
    return y.view(shape)


# ----------------------------------------------------------------------------------------------------------------
# SwiGLU: act_fn(gate) * up with act_fn = SiLU (ATen silu_kernel / silu_backward_kernel, bf16 in/out, fp32 opmath)
# ----------------------------------------------------------------------------------------------------------------
@triton.jit
def _silu_f32(g):
    return _div_rn(g, 1.0 + _exp(-g))      # x / (1 + exp(-x))


@triton.jit
def _swiglu_fwd_kernel(G, U, Y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=m, other=0.0).to(tl.float32)
    s = _silu_f32(g).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + offs, (s * u).to(tl.bfloat16), mask=m)


@triton.jit
def _swiglu_bwd_kernel(GY, G, U, GG, GU, n, BLOCK: tl.constexpr, CONTRACT: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    dy = tl.load(GY + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=m, other=0.0).to(tl.float32)
    s = _silu_f32(g).to(tl.bfloat16).to(tl.float32)
    tl.store(GU + offs, (dy * s).to(tl.bfloat16), mask=m)                 # grad of up = grad * silu(gate)
    ds = (dy * u).to(tl.bfloat16).to(tl.float32)                          # grad of silu(gate) = grad * up
    one = tl.zeros_like(g) + 1.0
    sg = _div_rn(one, 1.0 + _exp(-g))                                     # sigmoid
    if CONTRACT:
        t = _fma(g, 1.0 - sg, one)                                        # nvcc-contracted 1 + x * (1 - s)
    else:
        t = 1.0 + g * (1.0 - sg)
    tl.store(GG + offs, ((ds * sg) * t).to(tl.bfloat16), mask=m)


_SWIGLU_BLOCK = 2048
_SWIGLU_CONTRACT = int(os.environ.get("R3_FUSED_SWIGLU_CONTRACT", "1"))


class _FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        y = torch.empty_like(gate)
        n = gate.numel()
        _swiglu_fwd_kernel[(triton.cdiv(n, _SWIGLU_BLOCK),)](gate, up, y, n, BLOCK=_SWIGLU_BLOCK, num_warps=8, **_KW)
        ctx.save_for_backward(gate, up)
        return y

    @staticmethod
    def backward(ctx, gy):
        gate, up = ctx.saved_tensors
        gy = gy.contiguous()
        gg = torch.empty_like(gate)
        gu = torch.empty_like(up)
        n = gate.numel()
        _swiglu_bwd_kernel[(triton.cdiv(n, _SWIGLU_BLOCK),)](gy, gate, up, gg, gu, n, BLOCK=_SWIGLU_BLOCK,
                                                            CONTRACT=_SWIGLU_CONTRACT, num_warps=8, **_KW)
        return gg, gu


def _swiglu_ok(gate, up):
    return (_ENABLED and gate.is_cuda and gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16
            and gate.shape == up.shape and gate.is_contiguous() and up.is_contiguous() and gate.numel() < 2 ** 62)


def swiglu(act_fn, gate, up):
    """act_fn(gate) * up for act_fn = nn.SiLU (bitwise), fused when possible."""
    if not (isinstance(act_fn, torch.nn.SiLU) and _swiglu_ok(gate, up)):
        return act_fn(gate) * up
    if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
        return _FusedSwiGLU.apply(gate, up)
    y = torch.empty_like(gate)
    n = gate.numel()
    _swiglu_fwd_kernel[(triton.cdiv(n, _SWIGLU_BLOCK),)](gate, up, y, n, BLOCK=_SWIGLU_BLOCK, num_warps=8, **_KW)
    return y


# ----------------------------------------------------------------------------------------------------------------
# RoPE: q * cos + rotate_half(q) * sin (bf16 tensors, every product and the sum rounded to bf16)
# ----------------------------------------------------------------------------------------------------------------
@triton.jit
def _rope_load(X, COS, SIN, n_vec, H: tl.constexpr, BLOCK_V: tl.constexpr):
    # X [rows, H, 128] contiguous viewed as n_vec = rows*H vectors; COS/SIN [rows, 128] contiguous
    v = tl.program_id(0).to(tl.int64) * BLOCK_V + tl.arange(0, BLOCK_V)
    vm = (v < n_vec)[:, None]
    d = tl.arange(0, 64)[None, :]
    xb = v[:, None] * 128 + d
    cb = (v // H)[:, None] * 128 + d
    x1 = tl.load(X + xb, mask=vm, other=0.0).to(tl.float32)
    x2 = tl.load(X + xb + 64, mask=vm, other=0.0).to(tl.float32)
    c1 = tl.load(COS + cb, mask=vm, other=0.0).to(tl.float32)
    c2 = tl.load(COS + cb + 64, mask=vm, other=0.0).to(tl.float32)
    s1 = tl.load(SIN + cb, mask=vm, other=0.0).to(tl.float32)
    s2 = tl.load(SIN + cb + 64, mask=vm, other=0.0).to(tl.float32)
    return xb, vm, x1, x2, c1, c2, s1, s2


@triton.jit
def _rope_fwd_kernel(Q, COS, SIN, OUT, n_vec, H: tl.constexpr, BLOCK_V: tl.constexpr):
    xb, vm, x1, x2, c1, c2, s1, s2 = _rope_load(Q, COS, SIN, n_vec, H, BLOCK_V)
    # rotate_half(q) = cat(-x2, x1): first half x1*cos1 + (-x2)*sin1, second half x2*cos2 + x1*sin2
    a1 = (x1 * c1).to(tl.bfloat16).to(tl.float32)
    b1 = (-x2 * s1).to(tl.bfloat16).to(tl.float32)
    a2 = (x2 * c2).to(tl.bfloat16).to(tl.float32)
    b2 = (x1 * s2).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + xb, (a1 + b1).to(tl.bfloat16), mask=vm)
    tl.store(OUT + xb + 64, (a2 + b2).to(tl.bfloat16), mask=vm)


@triton.jit
def _rope_bwd_kernel(G, COS, SIN, GQ, n_vec, H: tl.constexpr, BLOCK_V: tl.constexpr):
    xb, vm, g1, g2, c1, c2, s1, s2 = _rope_load(G, COS, SIN, n_vec, H, BLOCK_V)
    # grad of q*cos: g*cos ; grad of rotate_half(q)*sin: r = g*sin (bf16), then cat/neg backward:
    #   x1 <- r[64:],  x2 <- -r[:64]; autograd sums the three bf16 contributions (one of them the +0 padding of
    #   a slice backward), i.e. bf16(a + b) with -0 -> +0.
    ga1 = (g1 * c1).to(tl.bfloat16).to(tl.float32)
    ga2 = (g2 * c2).to(tl.bfloat16).to(tl.float32)
    r1 = (g1 * s1).to(tl.bfloat16).to(tl.float32)
    r2 = (g2 * s2).to(tl.bfloat16).to(tl.float32)
    o1 = (ga1 + r2).to(tl.bfloat16).to(tl.float32) + 0.0
    o2 = (ga2 - r1).to(tl.bfloat16).to(tl.float32) + 0.0
    tl.store(GQ + xb, o1.to(tl.bfloat16), mask=vm)
    tl.store(GQ + xb + 64, o2.to(tl.bfloat16), mask=vm)


_ROPE_BLOCK_V = 32


def _rope_launch(kernel, a, cos, sin, out):
    rows, h, _ = a.shape
    n_vec = rows * h
    kernel[(triton.cdiv(n_vec, _ROPE_BLOCK_V),)](a, cos, sin, out, n_vec, H=h, BLOCK_V=_ROPE_BLOCK_V, num_warps=4,
                                                 **_KW)
    return out


class _FusedRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, cos, sin):
        out = _rope_launch(_rope_fwd_kernel, q, cos, sin, torch.empty_like(q))
        ctx.save_for_backward(cos, sin)
        return out

    @staticmethod
    def backward(ctx, g):
        cos, sin = ctx.saved_tensors
        g = g.contiguous()
        return _rope_launch(_rope_bwd_kernel, g, cos, sin, torch.empty_like(g)), None, None


def _rope_ok(x, cos, sin):
    return (_ENABLED and x.is_cuda and x.dtype == torch.bfloat16 and cos.dtype == torch.bfloat16
            and sin.dtype == torch.bfloat16 and x.dim() == 3 and x.shape[-1] == 128 and x.is_contiguous()
            and cos.shape == (x.shape[0], 128) and sin.shape == cos.shape and cos.is_contiguous()
            and sin.is_contiguous() and not cos.requires_grad and not sin.requires_grad)


def _rope_one(x, cos, sin):
    if torch.is_grad_enabled() and x.requires_grad:
        return _FusedRope.apply(x, cos, sin)
    return _rope_launch(_rope_fwd_kernel, x, cos, sin, torch.empty_like(x))


def apply_rotary(q, k, cos, sin, eager):
    """modeling_qwen2.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1) for packed [L, H, 128] q/k (bitwise)."""
    if not (_rope_ok(q, cos, sin) and _rope_ok(k, cos, sin)):
        return eager(q, k, cos, sin, unsqueeze_dim=1)
    return _rope_one(q, cos, sin), _rope_one(k, cos, sin)
