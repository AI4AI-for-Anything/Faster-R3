"""Bitwise tests of modeling/bagel/fused_flow.py against the eager code it replaces.

  python fusetest.py --tree <tree with fused_flow> --out <dir> --task units|model
units : kernels vs eager ops on random data with the workload's shapes (no model weights)
model : real BAGEL weights; eager vs fused backbone forwards on captured gen/edit branch inputs, then whole
        gen_image / edit_image (static graphs, SDE step, VAE decode) fused vs unfused with the same seeds;
        plus event timing of both.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--tree', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--task', required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
import flowlib  # noqa: E402
flowlib.setup(args.tree)
import torch  # noqa: E402

RES = dict(task=args.task, tree=args.tree, checks=[])


def check(name, a, b, extra=None):
    a = a.detach().contiguous(); b = b.detach().contiguous()
    same = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.uint8) if a.dtype != torch.bool else a,
                                                                     b.view(torch.uint8) if b.dtype != torch.bool else b)
    row = dict(name=name, bitwise=bool(same), shape=list(a.shape), dtype=str(a.dtype))
    if not same and a.shape == b.shape:
        d = (a.float() - b.float()).abs()
        row.update(n_diff=int((a != b).sum()), maxabs=float(d.max()), frac=float((a != b).float().mean()))
    if extra:
        row.update(extra)
    RES['checks'].append(row)
    print(('OK  ' if same else 'DIFF') + f' {name} {row}', flush=True)
    return same


def eager_rms(x, w, eps):
    from modeling.qwen2.modeling_qwen2 import Qwen2RMSNorm
    m = Qwen2RMSNorm(x.shape[-1], eps=eps).to(device=x.device, dtype=torch.bfloat16)
    with torch.no_grad():
        m.weight.copy_(w)
    return m(x)


def units():
    from modeling.bagel import fused_flow as ff
    from modeling.qwen2.modeling_qwen2 import Qwen2RMSNorm, apply_rotary_pos_emb, Qwen2RotaryEmbedding
    from modeling.bagel.qwen2_navit import Qwen2Config
    g = torch.Generator(device='cuda'); g.manual_seed(0)
    H = 3584
    eps = 1e-6
    # --- hidden RMSNorm (+ add) for every row count the configs cover
    for rows in (1, 2, 3, 4, 7, 8, 12, 16, 31, 32, 100, 511, 512, 1024, 4096, 4104, 8192, 16384):
        bw = ff.hidden_bw(rows, H)
        if bw is None:
            RES['checks'].append(dict(name=f'rms rows={rows}', unsupported=True)); continue
        for scale in (1.0, 30.0):
            x = (torch.randn(rows, H, device='cuda', generator=g) * scale).to(torch.bfloat16)
            x[:, :7] *= 50
            a = (torch.randn(rows, H, device='cuda', generator=g) * scale).to(torch.bfloat16)
            w = (torch.rand(H, device='cuda', generator=g) * 2).to(torch.bfloat16)
            ref = eager_rms(x, w, eps)
            _, y = ff.rms_rows(x, w, eps)
            check(f'rms rows={rows} bw={bw} scale={scale}', y, ref)
            href = x + a
            ref2 = eager_rms(href, w, eps)
            h, y2 = ff.rms_rows(x, w, eps, add=a)
            check(f'add+rms h rows={rows}', h, href)
            check(f'add+rms y rows={rows} bw={bw}', y2, ref2)
            # scatter to original rows + stacked CFG branches (each normalized with this row count's config)
            perm = torch.randperm(rows, device='cuda', generator=g)
            ys = torch.empty_like(x)
            ff.rms_rows(x, w, eps, add=a, y_out=ys, write_h=False, out_index=perm)
            rs = torch.empty_like(x); rs[perm] = ref2
            check(f'add+rms out_index rows={rows}', ys, rs)
            x3 = torch.cat([x, a, x]); a3 = torch.cat([a, x, x])
            _, y3 = ff.rms_rows(x3, w, eps, add=a3, cfg_rows=rows)
            r3 = torch.cat([eager_rms(x + a, w, eps), eager_rms(a + x, w, eps), eager_rms(x + x, w, eps)])
            check(f'add+rms stacked x3 cfg_rows={rows}', y3, r3)
    # --- head norm + RoPE (+ K/V scatter into merged buffer)
    cfg = Qwen2Config.from_pretrained(flowlib.MODEL + '/llm_config.json')
    rot = Qwen2RotaryEmbedding(config=cfg).to('cuda')
    for B in (1, 4, 8):
        T, G = 2 * B, 1024 * B
        n = T + G
        text_idx = torch.tensor([b * 1026 + o for b in range(B) for o in (0, 1025)], device='cuda')
        vae_idx = torch.tensor([b * 1026 + 1 + i for b in range(B) for i in range(1024)], device='cuda')
        order = torch.cat((text_idx, vae_idx))
        _, inverse = order.sort()
        pos = torch.tensor([300 + 17 * b for b in range(B) for _ in range(1026)], device='cuda')
        dummy = torch.zeros(1, n, 8, device='cuda', dtype=torch.bfloat16)
        cos, sin = rot(dummy, pos.unsqueeze(0))
        cos, sin = cos.squeeze(0), sin.squeeze(0)
        for nh in (28, 4):
            xs = (torch.randn(n, nh * 128, device='cuda', generator=g) * 3).to(torch.bfloat16)
            xt, xg = xs[:T].clone(), xs[T:].clone()
            wt = (torch.rand(128, device='cuda', generator=g) * 2).to(torch.bfloat16)
            wg = (torch.rand(128, device='cuda', generator=g) * 2).to(torch.bfloat16)
            nt = Qwen2RMSNorm(128, eps=eps).to(device='cuda', dtype=torch.bfloat16)
            ng = Qwen2RMSNorm(128, eps=eps).to(device='cuda', dtype=torch.bfloat16)
            with torch.no_grad():
                nt.weight.copy_(wt); ng.weight.copy_(wg)
            st = torch.cat((xt, xg)).view(-1, nh, 128).to(torch.float32)
            st = torch.cat((nt(st[:T]).to(torch.float32), ng(st[T:]).to(torch.float32)))[inverse]
            ref, _ = apply_rotary_pos_emb(st, st, cos, sin, unsqueeze_dim=1)
            ref = ref.to(torch.bfloat16)
            out = torch.empty((n, nh, 128), dtype=torch.bfloat16, device='cuda')
            ff.head_norm_rope(xt, xg, wt, wg, cos, sin, inverse, eps, nh, out)
            check(f'head_norm_rope B={B} nh={nh}', out, ref, dict(head_ok=ff.head_ok(T * nh)))
            if nh == 4:
                vs = torch.randn(n, nh * 128, device='cuda', generator=g).to(torch.bfloat16)
                vt, vg = vs[:T].clone(), vs[T:].clone()
                tot = n + 500
                dest = torch.randperm(tot, device='cuda', generator=g)[:n]
                mk = torch.zeros((tot, nh, 128), dtype=torch.bfloat16, device='cuda')
                mv = torch.zeros_like(mk)
                ff.head_norm_rope(xt, xg, wt, wg, cos, sin, inverse, eps, nh, mk, dest=dest, vt=vt, vg=vg, vout=mv)
                rk = torch.zeros_like(mk); rv = torch.zeros_like(mv)
                rk[dest] = ref
                rv[dest] = torch.cat((vt, vg)).view(-1, nh, 128)[inverse]
                check(f'k scatter B={B}', mk, rk)
                check(f'v scatter B={B}', mv, rv)
    # --- silu * up: exhaustive over bf16 gate values
    allb = torch.arange(-32768, 32768, device='cuda', dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    fin = allb[torch.isfinite(allb.float())]
    gate = fin.repeat(4)
    up = torch.randn(gate.numel(), device='cuda', generator=g).to(torch.bfloat16)
    ref = torch.nn.functional.silu(gate) * up
    check('silu_mul exhaustive gate', ff.silu_mul(gate, up), ref)
    gate2 = (torch.randn(4096, 18944, device='cuda', generator=g) * 4).to(torch.bfloat16)
    up2 = torch.randn(4096, 18944, device='cuda', generator=g).to(torch.bfloat16)
    check('silu_mul 4096x18944', ff.silu_mul(gate2, up2), torch.nn.functional.silu(gate2) * up2)
    # --- past copy
    kc = torch.randn(4, 900, 4, 128, device='cuda', generator=g).to(torch.bfloat16)
    vc = torch.randn(4, 900, 4, 128, device='cuda', generator=g).to(torch.bfloat16)
    pb = torch.tensor([b for b in range(4) for _ in range(300 + b)], device='cuda')
    pp = torch.tensor([p for b in range(4) for p in range(300 + b)], device='cuda')
    pd = torch.randperm(pb.numel() + 50, device='cuda', generator=g)[:pb.numel()]
    mk = torch.zeros((pb.numel() + 50, 4, 128), dtype=torch.bfloat16, device='cuda'); mv = torch.zeros_like(mk)
    ff.past_copy(kc, vc, pb, pp, pd, mk, mv)
    rk = torch.zeros_like(mk); rv = torch.zeros_like(mv)
    rk[pd] = kc[pb, pp]; rv[pd] = vc[pb, pp]
    check('past_copy k', mk, rk); check('past_copy v', mv, rv)
    # --- are GEMM rows independent of the batch (M) dimension? (would allow batching CFG branches)
    gem = []
    for (nout, nin, bias) in ((3584, 3584, True), (512, 3584, True), (3584, 3584, False), (18944, 3584, False),
                              (3584, 18944, False)):
        W = (torch.randn(nout, nin, device='cuda', generator=g) * 0.02).to(torch.bfloat16)
        b = (torch.randn(nout, device='cuda', generator=g) * 0.02).to(torch.bfloat16) if bias else None
        for base, mults in ((8, (2, 3)), (4096, (2, 3)), (4104, (2, 3))):
            X = torch.randn(base * 3, nin, device='cuda', generator=g).to(torch.bfloat16)
            with torch.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                ref = torch.nn.functional.linear(X[:base], W, b)
                for m in mults:
                    big = torch.nn.functional.linear(X[:base * m], W, b)
                    same = torch.equal(big[:base], ref)
                    gem.append(dict(nout=nout, nin=nin, bias=bias, base=base, mult=m, rows_identical=bool(same)))
    RES['gemm_row_independence'] = gem
    print('gemm rows identical:', [(r['nout'], r['nin'], r['base'], r['mult'], r['rows_identical']) for r in gem], flush=True)
    # --- flash-attn varlen: is each sequence's output independent of the other sequences in the call?
    from flash_attn import flash_attn_varlen_func
    fa = []
    for prefixes in ((350, 352, 349, 351), (2340, 2338, 2341, 2339)):
        qs, ks, vs, cq, ck = [], [], [], [0], [0]
        for p in prefixes:
            qs.append(torch.randn(1026, 28, 128, device='cuda', generator=g).to(torch.bfloat16))
            ks.append(torch.randn(p + 1026, 4, 128, device='cuda', generator=g).to(torch.bfloat16))
            vs.append(torch.randn(p + 1026, 4, 128, device='cuda', generator=g).to(torch.bfloat16))
        def call(sel):
            q = torch.cat([qs[i] for i in sel]); k = torch.cat([ks[i] for i in sel]); v = torch.cat([vs[i] for i in sel])
            cq = torch.tensor([0] + [1026 * (j + 1) for j in range(len(sel))], device='cuda', dtype=torch.int32)
            lk = [ks[i].shape[0] for i in sel]
            ck = torch.tensor([0] + list(torch.tensor(lk).cumsum(0).tolist()), device='cuda', dtype=torch.int32)
            return flash_attn_varlen_func(q, k, v, cq, ck, 1026, max(lk), causal=False)
        alone = call([0, 1, 2, 3])
        both = call([0, 1, 2, 3, 0, 1, 2, 3])   # same branch twice (as a CFG batch with longer max_k elsewhere)
        mixed = call([3, 2, 1, 0])
        fa.append(dict(prefixes=prefixes, dup_identical=bool(torch.equal(both[:4 * 1026], alone)) and
                       bool(torch.equal(both[4 * 1026:], alone)),
                       reorder_identical=bool(torch.equal(mixed.view(4, 1026, 28, 128).flip(0).reshape(-1, 28, 128), alone))))
    RES['flash_varlen_independence'] = fa
    print('flash varlen independence:', fa, flush=True)
    # --- timing of the fused kernels vs eager equivalents at the gen shapes
    x = (torch.randn(4096, H, device='cuda', generator=g)).to(torch.bfloat16)
    w = torch.ones(H, device='cuda', dtype=torch.bfloat16)
    def t(fn, it=20):
        fn(); torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(it):
            fn()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) / it * 1e3
    RES['timing_us'] = dict(
        rms_fused=t(lambda: ff.rms_rows(x, w, eps)), rms_eager=t(lambda: eager_rms(x, w, eps)),
        addrms_fused=t(lambda: ff.rms_rows(x, w, eps, add=x)), addrms_eager=t(lambda: eager_rms(x + x, w, eps)),
        rms_w2=t(lambda: ff.rms_rows(x, w, eps, num_warps=2)),
        addrms_w2=t(lambda: ff.rms_rows(x, w, eps, add=x, num_warps=2)),
        silu_fused=t(lambda: ff.silu_mul(gate2, up2)),
        silu_eager=t(lambda: torch.nn.functional.silu(gate2) * up2))
    B = 4; T, G = 8, 4096; n = T + G
    text_idx = torch.tensor([b * 1026 + o for b in range(B) for o in (0, 1025)], device='cuda')
    vae_idx = torch.tensor([b * 1026 + 1 + i for b in range(B) for i in range(1024)], device='cuda')
    _, inverse = torch.cat((text_idx, vae_idx)).sort()
    cos = torch.randn(n, 128, device='cuda').to(torch.bfloat16); sin = torch.randn(n, 128, device='cuda').to(torch.bfloat16)
    qt = torch.randn(T, 3584, device='cuda').to(torch.bfloat16); qg = torch.randn(G, 3584, device='cuda').to(torch.bfloat16)
    kt = torch.randn(T, 512, device='cuda').to(torch.bfloat16); kg = torch.randn(G, 512, device='cuda').to(torch.bfloat16)
    w128 = torch.ones(128, device='cuda', dtype=torch.bfloat16)
    qo = torch.empty(n, 28, 128, device='cuda', dtype=torch.bfloat16)
    mk = torch.empty(n + 1400, 4, 128, device='cuda', dtype=torch.bfloat16); mv = torch.empty_like(mk)
    dest = torch.arange(n, device='cuda') + 1400
    # GEMM batching potential: 2 or 3 branches as serial M=4096 calls, concurrent streams, or one batched call
    gt = {}
    streams = [torch.cuda.Stream() for _ in range(3)]
    for nm, (nout, nin) in dict(q=(3584, 3584), gate=(18944, 3584), down=(3584, 18944)).items():
        W = (torch.randn(nout, nin, device='cuda', generator=g) * 0.02).to(torch.bfloat16)
        X = torch.randn(3 * 4096, nin, device='cuda', generator=g).to(torch.bfloat16)
        for nb in (2, 3):
            xs = [X[i * 4096:(i + 1) * 4096] for i in range(nb)]
            def serial():
                for x in xs:
                    torch.nn.functional.linear(x, W)
            def concurrent():
                cur = torch.cuda.current_stream()
                for st, x in zip(streams, xs):
                    st.wait_stream(cur)
                    with torch.cuda.stream(st):
                        torch.nn.functional.linear(x, W)
                for st in streams[:nb]:
                    cur.wait_stream(st)
            gt[f'{nm}_x{nb}_serial'] = t(serial)
            gt[f'{nm}_x{nb}_streams'] = t(concurrent)
            gt[f'{nm}_x{nb}_batched'] = t(lambda: torch.nn.functional.linear(X[:nb * 4096], W))
            gt[f'{nm}_text8_x{nb}'] = t(lambda: [torch.nn.functional.linear(X[:8], W) for _ in range(nb)])
    RES['gemm_batching_us'] = gt
    print(json.dumps(gt), flush=True)
    RES['timing_us'].update(
        qnorm_rope_fused=t(lambda: ff.head_norm_rope(qt, qg, w128, w128, cos, sin, inverse, 1e-6, 28, qo)),
        knorm_rope_v_fused=t(lambda: ff.head_norm_rope(kt, kg, w128, w128, cos, sin, inverse, 1e-6, 4, mk, dest=dest,
                                                       vt=kt, vg=kg, vout=mv)))
    print(json.dumps(RES['timing_us']), flush=True)


def model():
    from modeling.bagel import fused_flow as ff
    from modeling.bagel import static_flow_graph as sfg
    from modeling.bagel.qwen2_navit import _mot_inference_layout
    env = flowlib.load(args.tree)
    lm = env['model'].language_model
    captured = []
    orig_call = sfg.StaticFlowGraph.__call__

    def capture_call(self, **kw):
        if kw.get('mode') == 'gen' and len(captured) < 6:
            kw2 = dict(kw); kw2['packed_query_sequence'] = kw['packed_query_sequence'].clone()
            captured.append(kw2)
        return orig_call(self, **kw)
    sfg.StaticFlowGraph.__call__ = capture_call
    orig_par = sfg.StaticFlowGraph.replay_independent
    sfg.StaticFlowGraph.replay_independent = lambda self, branches: None
    ff.ENABLED = False
    img, _, _ = flowlib.run_gen(env, seed=1)
    gen_kw = list(captured); captured.clear()
    flowlib.run_edit(env, img, seed=2)
    edit_kw = list(captured)
    sfg.StaticFlowGraph.__call__ = orig_call
    sfg.StaticFlowGraph.replay_independent = orig_par
    timing = {}
    for tag, kws in (('gen', gen_kw), ('edit', edit_kw)):
        for bi, kw in enumerate(kws):
            kw = dict(kw)
            kw['validated_mot_layout'] = _mot_inference_layout(kw['packed_query_sequence'], kw['packed_text_indexes'],
                                                               kw['packed_vae_token_indexes'])
            outs = {}
            for fused in (False, True):
                ff.ENABLED = fused
                before = ff.stats['fused_forwards']
                with torch.autocast('cuda', enabled=True, dtype=torch.bfloat16), torch.no_grad():
                    o = lm.forward_inference(**kw).packed_query_sequence.clone()
                    torch.cuda.synchronize()
                    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for _ in range(3):
                        lm.forward_inference(**kw)
                    e1.record(); torch.cuda.synchronize()
                outs[fused] = o
                timing[f'{tag}{bi}_{"fused" if fused else "eager"}_ms'] = e0.elapsed_time(e1) / 3
                if fused:
                    assert ff.stats['fused_forwards'] > before, ff.stats
            check(f'forward {tag} call{bi} q={kw["query_lens"].tolist()} kv={kw["key_values_lens"].tolist()}',
                  outs[True], outs[False])
    RES['forward_timing_ms'] = timing
    print(json.dumps(timing), flush=True)
    # whole image calls (static graphs + parallel replays + SDE + decode), unfused vs fused, same seeds
    T2 = {}
    res = {}
    for fused in (False, True):
        ff.ENABLED = fused
        for rep in range(2):
            torch.cuda.synchronize(); t0 = time.time()
            torch.manual_seed(1234)
            imgs, lat, lp = flowlib.run_gen(env, seed=11)
            torch.cuda.synchronize(); t1 = time.time()
            torch.manual_seed(4321)
            eimgs, eout = flowlib.run_edit(env, imgs, seed=12)
            torch.cuda.synchronize(); t2 = time.time()
            T2.setdefault(f'gen_{fused}', []).append(t1 - t0); T2.setdefault(f'edit_{fused}', []).append(t2 - t1)
        res[fused] = (imgs, lat, lp, eimgs, eout)
    for i in range(4):
        check(f'gen image {i}', torch.from_numpy(res[True][0][i]), torch.from_numpy(res[False][0][i]))
        check(f'gen latents {i}', res[True][1][i], res[False][1][i])
        check(f'gen logprob {i}', res[True][2][i], res[False][2][i])
        check(f'edit image {i}', torch.from_numpy(res[True][3][i]), torch.from_numpy(res[False][3][i]))
        a = res[True][4][i][0]; b = res[False][4][i][0]
        check(f'edit latents {i}', a.latent, b.latent)
        check(f'edit logprob {i}', a.log_prob, b.log_prob)
    RES['call_seconds'] = T2
    RES['ff_stats'] = dict(ff.stats)
    RES['max_alloc_gib'] = torch.cuda.max_memory_allocated() / 2**30
    print(json.dumps(T2), ff.stats, flush=True)


def digest(B=4):
    """Hashes of every rollout image output for fixed seeds (cross-tree bitwise comparison) + event timing."""
    import hashlib
    env = flowlib.load(args.tree)
    def h(t):
        if isinstance(t, torch.Tensor):
            t = t.detach().contiguous().cpu()
            return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
        import numpy as np
        return hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()[:16]
    rows = []
    for rep, (gi, ei) in enumerate(((5, 1), (0, 0), (5, 1))):
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
        torch.manual_seed(1234 + rep)
        e0.record()
        imgs, lat, lp = flowlib.run_gen(env, seed=11 + rep, sde_idx=gi, B=B)
        e1.record()
        torch.manual_seed(4321 + rep)
        eimgs, eout = flowlib.run_edit(env, imgs, seed=12 + rep, sde_idx=ei, B=B)
        e2.record(); torch.cuda.synchronize()
        rows.append(dict(rep=rep, sde=(gi, ei), gen_ms=e0.elapsed_time(e1), edit_ms=e1.elapsed_time(e2),
                         gen_img=[h(x) for x in imgs], gen_lat=[h(x) for x in lat], gen_lp=[h(x) for x in lp],
                         edit_img=[h(x) for x in eimgs], edit_lat=[h(o[0].latent) for o in eout],
                         edit_lp=[h(o[0].log_prob) for o in eout]))
        print(json.dumps(rows[-1]), flush=True)
    RES['digest'] = rows
    RES['max_alloc_gib'] = torch.cuda.max_memory_allocated() / 2**30
    try:
        from modeling.bagel import fused_flow as ff
        RES['ff_stats'] = dict(ff.stats)
    except ImportError:
        pass


def cfgbatch():
    """Whole gen/edit image calls with the CFG-batched fused forward vs per-branch fused graphs (same seeds)."""
    from modeling.bagel import fused_flow as ff
    env = flowlib.load(args.tree)
    res, T2 = {}, {}
    for on in (False, True, False, True):
        ff.CFG_BATCH = on
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        torch.manual_seed(1234)
        e[0].record()
        imgs, lat, lp = flowlib.run_gen(env, seed=11)
        e[1].record()
        torch.manual_seed(4321)
        eimgs, eout = flowlib.run_edit(env, imgs, seed=12)
        e[2].record(); torch.cuda.synchronize()
        T2.setdefault(f'gen_{on}', []).append(e[0].elapsed_time(e[1]))
        T2.setdefault(f'edit_{on}', []).append(e[1].elapsed_time(e[2]))
        T2.setdefault(f'peak_gib_{on}', []).append(torch.cuda.max_memory_allocated() / 2**30)
        res[on] = (imgs, lat, lp, eimgs, eout)
    for i in range(4):
        check(f'cfgb gen image {i}', torch.from_numpy(res[True][0][i]), torch.from_numpy(res[False][0][i]))
        check(f'cfgb gen latents {i}', res[True][1][i], res[False][1][i])
        check(f'cfgb gen logprob {i}', res[True][2][i], res[False][2][i])
        check(f'cfgb edit image {i}', torch.from_numpy(res[True][3][i]), torch.from_numpy(res[False][3][i]))
        check(f'cfgb edit latents {i}', res[True][4][i][0].latent, res[False][4][i][0].latent)
        check(f'cfgb edit logprob {i}', res[True][4][i][0].log_prob, res[False][4][i][0].log_prob)
    RES['call_ms'] = T2
    RES['ff_stats'] = dict(ff.stats)
    print(json.dumps(T2), ff.stats, flush=True)


{'units': units, 'model': model, 'digest': digest, 'digest8': lambda: digest(8), 'digest16': lambda: digest(16),
 'cfgbatch': cfgbatch}[args.task]()
RES['all_bitwise'] = all(c.get('bitwise', True) for c in RES['checks'])
with open(os.path.join(args.out, f'fusetest_{args.task}.json'), 'w') as f:
    json.dump(RES, f, indent=1)
print('[fusetest] all_bitwise =', RES['all_bitwise'], flush=True)
