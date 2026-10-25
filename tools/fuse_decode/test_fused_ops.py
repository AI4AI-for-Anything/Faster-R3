"""Bitwise operator tests for modeling/bagel/fused_ops.py against the eager PyTorch ops they replace.

Inputs: real activations captured from the text path (acts_decode.pt: decode n = 4, 3, 2, 1, all layers;
acts_prefill.pt: text prefill, layers 0/7/14/21/27) plus random stress shapes, and an exhaustive SiLU sweep
over every finite BF16 value. Each check compares raw bits (int16 views) and, for the norms, the fp32 mean that
the eager reduction produces (x.float().pow(2).mean(-1)) so that a different summation order cannot hide behind
the BF16 output rounding. Writes a JSON report.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fd_common as C  # noqa: E402
from modeling.bagel import fused_ops  # noqa: E402
from modeling.qwen2.modeling_qwen2 import Qwen2RMSNorm, apply_rotary_pos_emb  # noqa: E402

EPS = 1e-6


def bits_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, -1
    if a.dtype == torch.bfloat16:
        d = (a.view(torch.int16) != b.view(torch.int16))
    elif a.dtype == torch.float32:
        d = (a.view(torch.int32) != b.view(torch.int32))
    else:
        d = a != b
    n = int(d.sum())
    return n == 0, n


def load_norm_weights():
    from safetensors import safe_open
    w = {}
    with safe_open(os.path.join(C.MODEL, 'ema.safetensors'), framework='pt', device='cuda') as f:
        for k in f.keys():
            if not k.startswith('language_model.model.'):
                continue
            kk = k[len('language_model.model.'):]
            if kk.endswith(('input_layernorm.weight', 'post_attention_layernorm.weight', 'self_attn.q_norm.weight',
                            'self_attn.k_norm.weight')) or kk == 'norm.weight':
                w[kk] = f.get_tensor(k).to(torch.bfloat16)
    return w


def eager_norm(x, w):
    m = Qwen2RMSNorm(w.numel(), eps=EPS).to(device='cuda')
    m.weight.data = w
    return m(x)


class Report:
    def __init__(self):
        self.rows, self.fail = [], 0

    def add(self, kind, label, ok, nbad, **kw):
        self.rows.append(dict(kind=kind, label=label, ok=bool(ok), mismatches=nbad, **kw))
        if not ok:
            self.fail += 1
            print('MISMATCH', kind, label, nbad, kw, flush=True)


def check_rms(rep, label, x, w, residual=None):
    n, h = x.shape
    dbg = torch.empty(n, dtype=torch.float32, device='cuda')
    r = fused_ops.rms_norm(x, w, EPS, residual=residual, debug_mean=dbg)
    if r is None:
        rep.add('rms' if residual is None else 'add_rms', label, False, -2, note='not covered')
        return
    xin = x if residual is None else (residual + x)
    ref_mean = xin.to(torch.float32).pow(2).mean(-1)
    ref = eager_norm(xin, w)
    if residual is None:
        ok, nb = bits_equal(r, ref)
        okr, nbr = True, 0
    else:
        ok, nb = bits_equal(r[0], ref)
        okr, nbr = bits_equal(r[1], xin)
    okm, nbm = bits_equal(dbg, ref_mean)
    rep.add('rms' if residual is None else 'add_rms', label, ok and okm and okr, nb + nbm + nbr,
            rows=n, mean_mismatch_rows=nbm, out_mismatch=nb, res_mismatch=nbr,
            bw=fused_ops.torch_mean_block_width(n, h))


def check_qk(rep, label, q, k, wq, wk, cos, sin):
    n = q.shape[0]
    mq = torch.empty(n * q.shape[1], dtype=torch.float32, device='cuda')
    mk = torch.empty(n * k.shape[1], dtype=torch.float32, device='cuda')
    r = fused_ops.qk_norm_rope(q, k, wq, wk, cos, sin, EPS, debug_means=(mq, mk))
    if r is None:
        rep.add('qk', label, False, -2, note='not covered')
        return
    qn = eager_norm(q, wq)
    kn = eager_norm(k, wk)
    qr, kr = apply_rotary_pos_emb(qn, kn, cos, sin, unsqueeze_dim=1)
    ok1, n1 = bits_equal(r[0], qr)
    ok2, n2 = bits_equal(r[1], kr)
    ok3, n3 = bits_equal(mq, q.float().pow(2).mean(-1).reshape(-1))
    ok4, n4 = bits_equal(mk, k.float().pow(2).mean(-1).reshape(-1))
    rep.add('qk', label, ok1 and ok2 and ok3 and ok4, n1 + n2 + n3 + n4, tokens=n, q_out=n1, k_out=n2,
            q_mean=n3, k_mean=n4)


def check_silu(rep, label, g, u):
    r = fused_ops.silu_mul(g, u)
    if r is None:
        rep.add('silu', label, False, -2, note='not covered')
        return
    ref = torch.nn.SiLU()(g) * u
    ok, nb = bits_equal(r, ref)
    rep.add('silu', label, ok, nb, numel=g.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acts', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    rep = Report()
    W = load_norm_weights()
    with torch.no_grad():
        # ---------- real activations ----------
        for fname in ('acts_decode.pt', 'acts_prefill.pt'):
            path = os.path.join(args.acts, fname)
            if not os.path.exists(path):
                rep.add('setup', fname, False, -3, note='missing')
                continue
            A = torch.load(path)
            tag = fname.split('.')[0]
            cos_l, sin_l = A.get('rope.cos', []), A.get('rope.sin', [])
            for i in range(28):
                key = f'L{i}.input_layernorm.in'
                if key not in A:
                    continue
                for j, x in enumerate(A[key]):
                    x = x.cuda()
                    check_rms(rep, f'{tag}.L{i}.in.f{j}', x, W[f'layers.{i}.input_layernorm.weight'])
                    xp = A[f'L{i}.post_attention_layernorm.in'][j].cuda()
                    check_rms(rep, f'{tag}.L{i}.post.f{j}', xp, W[f'layers.{i}.post_attention_layernorm.weight'])
                    # residual add + norm: residual = layer input, x = (post-attn input - residual) in bf16
                    check_rms(rep, f'{tag}.L{i}.addpost.f{j}', (xp - x), W[f'layers.{i}.post_attention_layernorm.weight'],
                              residual=x)
                    q = A[f'L{i}.q_norm.in'][j].cuda()
                    k = A[f'L{i}.k_norm.in'][j].cuda()
                    cos = cos_l[j * 28].cuda()
                    sin = sin_l[j * 28].cuda()
                    check_qk(rep, f'{tag}.L{i}.qk.f{j}', q, k, W[f'layers.{i}.self_attn.q_norm.weight'],
                             W[f'layers.{i}.self_attn.k_norm.weight'], cos, sin)
                    g = A[f'L{i}.gate_proj.out'][j].cuda()
                    u = A[f'L{i}.up_proj.out'][j].cuda()
                    check_silu(rep, f'{tag}.L{i}.silu.f{j}', g, u)
            for j, x in enumerate(A.get('final_norm.in', [])):
                check_rms(rep, f'{tag}.final.f{j}', x.cuda(), W['norm.weight'])
            del A
        # ---------- random stress: hidden 3584 at many row counts and scales ----------
        w = W['layers.5.input_layernorm.weight']
        for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 31, 32, 63, 64, 100, 255, 256, 511, 512, 513, 600, 1000,
                  2048, 4096, 4682, 5200, 8192):
            for scale in (0.02, 1.0, 40.0):
                x = (torch.randn(n, 3584, device='cuda') * scale)
                x[:, torch.randint(0, 3584, (4,))] *= 300.0  # massive-activation channels
                x = x.to(torch.bfloat16)
                check_rms(rep, f'rand.rms.n{n}.s{scale}', x, w)
                r = (torch.randn(n, 3584, device='cuda') * scale).to(torch.bfloat16)
                check_rms(rep, f'rand.addrms.n{n}.s{scale}', x, w, residual=r)
        # ---------- random stress: q/k norm + rope ----------
        wq, wk = W['layers.3.self_attn.q_norm.weight'], W['layers.3.self_attn.k_norm.weight']
        for n in (1, 2, 3, 4, 5, 7, 8, 16, 50, 64, 100, 300, 1300, 5200):
            q = (torch.randn(n, 28, 128, device='cuda') * 3).to(torch.bfloat16)
            k = (torch.randn(n, 4, 128, device='cuda') * 3).to(torch.bfloat16)
            pos = torch.randint(0, 3000, (n,), device='cuda').float()
            inv = 1.0 / (1e6 ** (torch.arange(0, 128, 2, device='cuda').float() / 128))
            fr = pos[:, None] * inv[None, :]
            emb = torch.cat([fr, fr], -1)
            check_qk(rep, f'rand.qk.n{n}', q, k, wq, wk, emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16))
        # ---------- exhaustive SiLU over all finite bf16 values ----------
        allbits = torch.arange(-32768, 32768, dtype=torch.int32, device='cuda').to(torch.int16).view(torch.bfloat16)
        finite = allbits[torch.isfinite(allbits.float())]
        check_silu(rep, 'exhaustive.silu.up1', finite.contiguous(), torch.ones_like(finite))
        for t in range(3):
            check_silu(rep, f'exhaustive.silu.uprand{t}', finite.contiguous(),
                       (torch.randn(finite.numel(), device='cuda') * 4).to(torch.bfloat16))
        for n in (1, 4, 1000):
            g = (torch.randn(n, 18944, device='cuda') * 2).to(torch.bfloat16)
            u = (torch.randn(n, 18944, device='cuda') * 2).to(torch.bfloat16)
            check_silu(rep, f'rand.silu.n{n}', g, u)
    summ = {}
    for r in rep.rows:
        s = summ.setdefault(r['kind'], dict(checks=0, failed=0))
        s['checks'] += 1
        s['failed'] += 0 if r['ok'] else 1
    out = dict(all_identical=rep.fail == 0, summary=summ, triton=__import__('triton').__version__,
               torch=torch.__version__, device=torch.cuda.get_device_name(), rows=rep.rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print('TEST_FUSED_OPS', json.dumps(dict(all_identical=out['all_identical'], summary=summ)), flush=True)


if __name__ == '__main__':
    main()
