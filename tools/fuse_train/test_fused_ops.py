"""Bitwise check (+ timing) of modeling/bagel/fused_train_ops.py against the eager code it replaces.

Run on one GPU from the repo root:  python tools/fuse_train/test_fused_ops.py --out RESULT.json [--time]
Forward outputs and every input/weight gradient must be bit-identical (int16 view compare) to eager autograd under
the bench's deterministic settings, for realistic shapes (row counts of the MoT expert splits, q/k head norms, the
MLP width) and real layer weights from the checkpoint when available.
"""
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modeling.qwen2.modeling_qwen2 import Qwen2RMSNorm, apply_rotary_pos_emb, Qwen2RotaryEmbedding  # noqa: E402
from modeling.bagel import fused_train_ops as F  # noqa: E402

CKPT = Path(os.environ.get('MODEL_PATH', './checkpoints/BAGEL-7B-MoT')) / 'ema.safetensors'


def bits_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype in (torch.bfloat16, torch.float16):
        return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
    if a.dtype == torch.float32:
        return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
    return bool(torch.equal(a, b))


def mismatch(a, b):
    if a.shape != b.shape:
        return 'shape'
    d = (a.float() - b.float()).abs()
    n = int((a.view(torch.int16) != b.view(torch.int16)).sum()) if a.dtype == torch.bfloat16 else int((a != b).sum())
    return dict(n_diff=n, numel=a.numel(), max_abs=float(d.max()))


def real_weight(name, shape, gen):
    try:
        from safetensors import safe_open
        with safe_open(str(CKPT), 'pt') as f:
            return f.get_tensor(name).to('cuda', torch.bfloat16)
    except Exception:
        return (1 + 0.1 * torch.randn(shape, generator=gen, device='cuda')).to(torch.bfloat16)


def act(shape, gen, scale=1.0, outliers=True):
    x = torch.randn(shape, generator=gen, device='cuda') * scale
    if outliers:
        m = torch.rand(shape, generator=gen, device='cuda') < 1e-3
        x = torch.where(m, x * 30, x)
    return x.to(torch.bfloat16)


def zero_rows(t, gen, frac=0.1):
    rows = t.shape[0]
    m = torch.rand(rows, generator=gen, device='cuda') < frac
    t = t.clone()
    t[m] = 0
    return t


def check_rms(results, gen, wname, shape):
    n = shape[-1]
    w = real_weight(wname, (n,), gen)
    eager = Qwen2RMSNorm(n, eps=1e-6).cuda()
    eager.weight.data = w.clone()
    fused = Qwen2RMSNorm(n, eps=1e-6).cuda()
    fused.weight.data = w.clone()
    x = act(shape, gen, scale=float(torch.rand((), generator=gen, device='cuda')) * 4 + 0.05)
    if shape[0] > 4:
        x[1] = 0                                     # an all-zero row (padding/dead rows exist in the packs)
    gy = zero_rows(act(shape, gen, scale=1e-2), gen)
    if shape[0] > 8:
        gy[2] = (gy[2].float() * 2.0 ** -120).to(torch.bfloat16)      # tiny / subnormal upstream gradients
    xe = x.clone().requires_grad_(True)
    xf = x.clone().requires_grad_(True)
    ye = eager(xe)
    yf = F.rms_norm(fused, xf)
    ye.backward(gy)
    yf.backward(gy)
    with torch.inference_mode():
        yi = F.rms_norm(fused, x)
        ye_i = eager(x)
    row = dict(op='rms_norm', weight=wname, shape=list(shape), fused_path=F._rms_ok(x, w),
               y=bits_equal(ye, yf), y_inference=bits_equal(ye_i, yi), gx=bits_equal(xe.grad, xf.grad),
               gw=bits_equal(eager.weight.grad, fused.weight.grad))
    for k, (a, b) in dict(y=(ye, yf), gx=(xe.grad, xf.grad), gw=(eager.weight.grad, fused.weight.grad)).items():
        if not row[k]:
            row[k + '_mismatch'] = mismatch(a, b)
    results.append(row)


def check_swiglu(results, gen, shape, contract):
    F._SWIGLU_CONTRACT = contract
    act_fn = torch.nn.SiLU()
    g = act(shape, gen, scale=3.0)
    u = act(shape, gen, scale=1.0)
    g.view(-1)[:4] = torch.tensor([-100.0, 100.0, 0.0, -0.0], dtype=torch.bfloat16)
    # sigmoid / silu results near and below 2^-126 (FTZ-sensitive), and bf16 subnormal inputs
    k = min(g.numel() // 4, 4096)
    g.view(-1)[4:4 + k] = torch.linspace(-90.0, -85.0, k, device='cuda').to(torch.bfloat16)
    sub = torch.tensor([2.0 ** -130, -2.0 ** -129, 2.0 ** -133, -2.0 ** -127], device='cuda').to(torch.bfloat16)
    g.view(-1)[4 + k:8 + k] = sub
    u.view(-1)[8 + k:12 + k] = sub
    gy = zero_rows(act(shape, gen, scale=1e-2), gen)
    ge, ue = g.clone().requires_grad_(True), u.clone().requires_grad_(True)
    gf, uf = g.clone().requires_grad_(True), u.clone().requires_grad_(True)
    ye = act_fn(ge) * ue
    yf = F.swiglu(act_fn, gf, uf)
    ye.backward(gy)
    yf.backward(gy)
    with torch.inference_mode():
        yi = F.swiglu(act_fn, g, u)
    row = dict(op='swiglu', shape=list(shape), contract=contract, y=bits_equal(ye, yf),
               y_inference=bits_equal(ye.detach(), yi), g_gate=bits_equal(ge.grad, gf.grad),
               g_up=bits_equal(ue.grad, uf.grad))
    for k, (a, b) in dict(y=(ye, yf), g_gate=(ge.grad, gf.grad), g_up=(ue.grad, uf.grad)).items():
        if not row[k]:
            row[k + '_mismatch'] = mismatch(a, b)
    results.append(row)


def rotary(L, gen):
    from modeling.qwen2.configuration_qwen2 import Qwen2Config
    cfg = Qwen2Config(hidden_size=3584, num_attention_heads=28, num_key_value_heads=4, rope_theta=1000000.0,
                      max_position_embeddings=32768)
    rot = Qwen2RotaryEmbedding(config=cfg).cuda()
    pos = torch.randint(0, 3000, (L,), generator=gen, device='cuda')
    x = torch.zeros(L, 3584, device='cuda', dtype=torch.bfloat16)
    cos, sin = rot(x, pos.unsqueeze(0))
    return cos.squeeze(0), sin.squeeze(0)


def check_rope(results, gen, L):
    cos, sin = rotary(L, gen)
    q = act((L, 28, 128), gen, 2.0)
    k = act((L, 4, 128), gen, 2.0)
    gq = zero_rows(act((L, 28, 128), gen, 1e-2), gen)
    gk = zero_rows(act((L, 4, 128), gen, 1e-2), gen)
    qe, ke = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
    qf, kf = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
    oqe, oke = apply_rotary_pos_emb(qe, ke, cos, sin, unsqueeze_dim=1)
    oqf, okf = F.apply_rotary(qf, kf, cos, sin, apply_rotary_pos_emb)
    torch.autograd.backward([oqe, oke], [gq, gk])
    torch.autograd.backward([oqf, okf], [gq, gk])
    row = dict(op='rope', L=L, fused_path=F._rope_ok(q, cos, sin), q=bits_equal(oqe, oqf), k=bits_equal(oke, okf),
               gq=bits_equal(qe.grad, qf.grad), gk=bits_equal(ke.grad, kf.grad))
    for kk, (a, b) in dict(q=(oqe, oqf), k=(oke, okf), gq=(qe.grad, qf.grad), gk=(ke.grad, kf.grad)).items():
        if not row[kk]:
            row[kk + '_mismatch'] = mismatch(a, b)
    results.append(row)


def timeit(fn, reps=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


def timings(gen):
    out = {}
    # RMSNorm over the hidden size (40k packed rows: edit step), and the q-norm (28 heads)
    for name, shape in (('rms3584_40k', (40000, 3584)), ('rms128_q_40k', (40000, 28, 128)),
                        ('rms128_k_40k', (40000, 4, 128))):
        n = shape[-1]
        m = Qwen2RMSNorm(n).cuda()
        m.weight.data = m.weight.data.to(torch.bfloat16)
        x = act(shape, gen).requires_grad_(True)
        gy = act(shape, gen, 1e-2)
        def ef():
            with torch.no_grad():
                m(x)
        def ff():
            with torch.no_grad():
                F.rms_norm(m, x)
        def eb():
            y = m(x); y.backward(gy)
        def fb():
            y = F.rms_norm(m, x); y.backward(gy)
        out[name] = dict(eager_fwd_ms=timeit(ef), fused_fwd_ms=timeit(ff), eager_fwdbwd_ms=timeit(eb),
                         fused_fwdbwd_ms=timeit(fb))
    act_fn = torch.nn.SiLU()
    g = act((20000, 18944), gen).requires_grad_(True)
    u = act((20000, 18944), gen).requires_grad_(True)
    gy = act((20000, 18944), gen, 1e-2)
    def ef():
        with torch.no_grad():
            act_fn(g) * u
    def ff():
        with torch.no_grad():
            F.swiglu(act_fn, g, u)
    def eb():
        (act_fn(g) * u).backward(gy)
    def fb():
        F.swiglu(act_fn, g, u).backward(gy)
    out['swiglu_20k'] = dict(eager_fwd_ms=timeit(ef), fused_fwd_ms=timeit(ff), eager_fwdbwd_ms=timeit(eb),
                             fused_fwdbwd_ms=timeit(fb))
    cos, sin = rotary(40000, gen)
    q = act((40000, 28, 128), gen).requires_grad_(True)
    k = act((40000, 4, 128), gen).requires_grad_(True)
    gq, gk = act((40000, 28, 128), gen, 1e-2), act((40000, 4, 128), gen, 1e-2)
    def ef():
        with torch.no_grad():
            apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    def ff():
        with torch.no_grad():
            F.apply_rotary(q, k, cos, sin, apply_rotary_pos_emb)
    def eb():
        a, b = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1); torch.autograd.backward([a, b], [gq, gk])
    def fb():
        a, b = F.apply_rotary(q, k, cos, sin, apply_rotary_pos_emb); torch.autograd.backward([a, b], [gq, gk])
    out['rope_40k'] = dict(eager_fwd_ms=timeit(ef), fused_fwd_ms=timeit(ff), eager_fwdbwd_ms=timeit(eb),
                           fused_fwdbwd_ms=timeit(fb))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--time', action='store_true')
    args = ap.parse_args()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    gen = torch.Generator(device='cuda')
    gen.manual_seed(1234)
    results = []
    L = 'language_model.model.layers.13.'
    for shape in ((16, 3584), (17, 3584), (37, 3584), (1029, 3584), (4097, 3584), (31013, 3584)):
        for wn in (L + 'input_layernorm.weight', L + 'post_attention_layernorm_moe_gen.weight'):
            check_rms(results, gen, wn, shape)
    check_rms(results, gen, 'language_model.model.norm.weight', (2000, 3584))
    for shape in ((1, 28, 128), (5, 28, 128), (4097, 28, 128), (31013, 28, 128), (4, 4, 128), (7, 4, 128),
                  (31013, 4, 128)):
        for wn in (L + 'self_attn.q_norm.weight', L + 'self_attn.k_norm_moe_gen.weight'):
            check_rms(results, gen, wn, shape)
    for shape in ((1, 18944), (37, 18944), (4096, 18944)):
        for contract in (1, 0):
            check_swiglu(results, gen, shape, contract)
    for L_ in (1, 77, 4096, 31013):
        check_rope(results, gen, L_)
    report = dict(results=results, all_rms=all(r['y'] and r['gx'] and r['gw'] and r['y_inference']
                                               for r in results if r['op'] == 'rms_norm'),
                  swiglu_contract_ok={c: all(r['y'] and r['g_gate'] and r['g_up'] and r['y_inference']
                                             for r in results if r['op'] == 'swiglu' and r['contract'] == c)
                                      for c in (0, 1)},
                  all_rope=all(r['q'] and r['k'] and r['gq'] and r['gk'] for r in results if r['op'] == 'rope'),
                  torch=torch.__version__, device=torch.cuda.get_device_name())
    if args.time:
        report['timing'] = timings(gen)
    Path(args.out).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=1))
    for r in results:
        if not all(v for k, v in r.items() if isinstance(v, bool) and k != 'fused_path'):
            print('MISMATCH', json.dumps(r))


if __name__ == '__main__':
    main()
