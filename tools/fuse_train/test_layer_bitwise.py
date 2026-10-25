"""Model-level bitwise check of the fuse_train MoT training path (this tree) against a reference tree (stack_b), on
one GPU: a Qwen2Model with NL real decoder layers (checkpoint weights of layers first..first+NL-1) and the real final
MoT norms, the training activation-checkpoint wrapper on every decoder layer, bf16 autocast, and a real packed layout
(captured pack from a run) or a synthetic edit-step-like one.

Compared bit for bit: the output, the input gradient and every parameter gradient (policy path, model.train()), and
the output of the reference path (eval + ref_forward=True under inference_mode).  Optionally times both trees and
records their peak memory.

  python tools/fuse_train/test_layer_bitwise.py --ref-tree <stack_b> --out R.json [--capture cap.pt] [--layers 2]
"""
import argparse, importlib.util, json, os, sys
from functools import partial
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (CheckpointImpl, apply_activation_checkpointing,
                                                                         checkpoint_wrapper)

TREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TREE))
MODEL = Path(os.environ.get('MODEL_PATH', './checkpoints/BAGEL-7B-MoT'))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bits(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


def synthetic_pack(gen, n_samples=4, copies=3):
    """Edit-step-like pack: per sample and copy: prompt text, VAE image (full), ViT image (full), think text (cond
    copy only), noisy latent (noise), each sample padded to a multiple of 128."""
    text_idx, vit_idx, vae_idx, pos, split_lens, attn_modes, sample_lens = [], [], [], [], [], [], []
    cur = 0
    for s in range(n_samples):
        for c in range(copies):
            start, rope = cur, 0

            def text(n):
                nonlocal cur, rope
                text_idx.extend(range(cur, cur + n)); pos.extend(range(rope, rope + n))
                split_lens.append(n); attn_modes.append('causal'); cur += n; rope += n

            def image(n, kind, mode):
                nonlocal cur, rope
                text_idx.append(cur); (vit_idx if kind == 'vit' else vae_idx).extend(range(cur + 1, cur + 1 + n))
                text_idx.append(cur + 1 + n); pos.extend([rope] * (n + 2)); split_lens.append(n + 2)
                attn_modes.append(mode); cur += n + 2; rope += (0 if mode == 'noise' else 1)
            text(37)
            image(1024, 'vae', 'full')
            image(1296, 'vit', 'full')
            if c == 0:
                text(int(torch.randint(150, 380, (1,), generator=gen)))
            image(1024, 'vae', 'noise')
            pad = (-(cur - start)) % 128
            if pad:
                text(pad)
            sample_lens.append(cur - start)
    t = lambda v: torch.tensor(v, dtype=torch.long)
    return dict(packed_text_indexes=t(text_idx), packed_vit_token_indexes=t(vit_idx), packed_vae_token_indexes=t(vae_idx),
                packed_position_ids=t(pos), split_lens=split_lens, attn_modes=attn_modes, sample_lens=sample_lens,
                sequence_length=cur, packed_und_live_segments=None, packed_gen_live_segments=None)


def build(navit, cfg, first, nl, dev):
    from safetensors import safe_open
    cfg = navit.Qwen2Config.from_pretrained(str(MODEL / 'llm_config.json'))
    cfg.layer_module = 'Qwen2MoTDecoderLayer'; cfg.qk_norm = True; cfg.freeze_und = False
    cfg.num_hidden_layers = nl
    torch.manual_seed(0)
    model = navit.Qwen2Model(cfg)
    sd = {}
    with safe_open(str(MODEL / 'ema.safetensors'), 'pt') as f:
        for i in range(nl):
            pre = f'language_model.model.layers.{first + i}.'
            for k in f.keys():
                if k.startswith(pre):
                    sd[f'layers.{i}.' + k[len(pre):]] = f.get_tensor(k)
        for k in ('norm.weight', 'norm_moe_gen.weight'):
            sd[k] = f.get_tensor('language_model.model.' + k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(m.startswith(('embed_tokens', 'rotary_emb')) for m in missing), (missing, unexpected)
    model = model.to(dev)
    for prm in model.parameters():          # FSDP2 mixed precision: bf16 compute params, buffers (inv_freq) fp32
        prm.data = prm.data.to(torch.bfloat16)
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=partial(checkpoint_wrapper,
                                   checkpoint_impl=CheckpointImpl.NO_REENTRANT),
                                   check_fn=lambda m: isinstance(m, navit.Qwen2MoTDecoderLayer))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-tree', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--capture', default=None)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--first-layer', type=int, default=13)
    ap.add_argument('--time', action='store_true')
    args = ap.parse_args()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    dev = torch.device('cuda')
    new = load_module('navit_new', TREE / 'modeling/bagel/qwen2_navit.py')
    ref = load_module('navit_ref', Path(args.ref_tree) / 'modeling/bagel/qwen2_navit.py')
    from data.data_utils import create_sparse_mask
    from torch.nn.attention.flex_attention import create_block_mask

    gen = torch.Generator(device='cpu'); gen.manual_seed(7)
    x0 = None
    if args.capture:
        cap = torch.load(args.capture, map_location='cpu', weights_only=False)
        pi = cap['packed_input']
        src = 'capture:' + args.capture
        x0 = cap['layers'].get(args.first_layer)
    else:
        pi = synthetic_pack(gen)
        src = 'synthetic'
    und = pi['packed_text_indexes']
    if pi.get('packed_vit_token_indexes') is not None and len(pi['packed_vit_token_indexes']):
        und = torch.cat([und, pi['packed_vit_token_indexes']])
    und, gen_idx = und.to(dev), pi['packed_vae_token_indexes'].to(dev)
    L = int(pi['sequence_length'])
    pos = pi['packed_position_ids'].to(dev)
    if x0 is None:
        g = torch.Generator(device='cuda'); g.manual_seed(11)
        x0 = (torch.randn(L, 3584, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    x0 = x0.to(dev)
    smask = create_sparse_mask(pi['sample_lens'], pi['split_lens'], pi['attn_modes'], dev)
    bmask = create_block_mask(smask, B=1, H=28, Q_LEN=L, KV_LEN=L, device=dev, BLOCK_SIZE=128, _compile=True)
    keep = {k: pi[k] for k in ('packed_und_live_segments', 'packed_gen_live_segments') if pi.get(k) is not None}
    g = torch.Generator(device='cuda'); g.manual_seed(12)
    gy = (torch.randn(L, 3584, generator=g, device=dev) * 1e-3).to(torch.bfloat16)
    dead = torch.zeros(L, dtype=torch.bool, device=dev)
    dead[torch.randperm(L, generator=torch.Generator().manual_seed(3))[:L // 10].to(dev)] = True
    gy[dead] = 0                       # rows no loss reads (dead CFG copies, unused tokens) get exact zero grads

    models = dict(ref=build(ref, None, args.first_layer, args.layers, dev),
                  new=build(new, None, args.first_layer, args.layers, dev))

    def run(tag, grad=True):
        m = models[tag]
        m.train(grad)
        for p in m.parameters():
            p.grad = None
        x = x0.clone().requires_grad_(grad)
        kw = dict(packed_sequence=x, sample_lens=pi['sample_lens'], attention_mask=bmask, packed_position_ids=pos,
                  packed_und_token_indexes=und, packed_gen_token_indexes=gen_idx, ref_forward=not grad, **keep)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = m(**kw)
        if not grad:
            return out.detach(), None, {}
        out.backward(gy)
        return out.detach(), x.grad, {n: p.grad for n, p in m.named_parameters() if p.grad is not None}

    report = dict(source=src, L=L, n_und=int(und.numel()), n_gen=int(gen_idx.numel()), layers=args.layers,
                  first_layer=args.first_layer, live_segments=sorted(keep), x0='capture' if args.capture and
                  cap['layers'].get(args.first_layer) is not None else 'random')
    o_r, gx_r, gr = run('ref')
    o_n, gx_n, gn = run('new')
    report['policy_out'] = bits(o_r, o_n)
    report['policy_grad_input'] = bits(gx_r, gx_n)
    names = sorted(set(gr) | set(gn))
    report['param_grads_equal'] = {k: bits(gr.get(k), gn.get(k)) for k in names}
    report['n_param_grads'] = len(names)
    report['all_param_grads_equal'] = all(report['param_grads_equal'].values())
    with torch.inference_mode():
        o_ri, _, _ = run('ref', grad=False)
        o_ni, _, _ = run('new', grad=False)
    report['reference_out'] = bits(o_ri, o_ni)
    if not report['policy_out']:
        report['policy_out_maxabs'] = float((o_r.float() - o_n.float()).abs().max())
    if not report['policy_grad_input']:
        report['grad_input_maxabs'] = float((gx_r.float() - gx_n.float()).abs().max())
        report['grad_input_ndiff'] = int((gx_r.view(torch.int16) != gx_n.view(torch.int16)).sum())
    bad = [k for k, v in report['param_grads_equal'].items() if not v]
    report['param_grads_mismatch'] = {k: dict(ndiff=int((gr[k].view(torch.int16) != gn[k].view(torch.int16)).sum())
                                              if k in gr and k in gn else None,
                                              maxabs=float((gr[k].float() - gn[k].float()).abs().max())
                                              if k in gr and k in gn else None) for k in bad}
    if args.time:
        def t(fn, reps=3):
            fn(); torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(reps):
                fn()
            e1.record(); torch.cuda.synchronize()
            return e0.elapsed_time(e1) / reps

        def inf(tag):
            with torch.inference_mode():
                run(tag, grad=False)
        for tag in ('ref', 'new'):
            report[f'ms_{tag}'] = dict(train_fwd_bwd=t(lambda: run(tag)), inference_fwd=t(lambda: inf(tag)))
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            run(tag); torch.cuda.synchronize()
            report[f'peak_extra_gib_{tag}'] = (torch.cuda.max_memory_allocated() - base) / 2**30
    Path(args.out).write_text(json.dumps(report, indent=1) + '\n')
    short = {k: v for k, v in report.items() if k != 'param_grads_equal'}
    print(json.dumps(short, indent=1))


if __name__ == '__main__':
    main()
