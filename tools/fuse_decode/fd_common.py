"""Shared single-GPU setup for the fuse_decode micro-benchmarks and operator tests.

Loads only the BAGEL-7B-MoT language model (both experts, BF16) on one GPU and builds a light stand-in for
Bagel that borrows the unchanged text methods (prepare_prompts, forward_cache_update_text, prepare_start_tokens,
generate_text_batch, _decode_static_pipelined, _sample_next_token). Prompts follow the rollout: system prompt,
then the GenEval prompt, batch 4 (group_size 16 / policy_group_size 4), under BF16 autocast.
"""
import json
import os
import sys
import types

import torch

TREE = os.environ.get('FD_TREE') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if TREE not in sys.path:
    sys.path.insert(0, TREE)
MODEL = os.environ.get('MODEL_PATH', './checkpoints/BAGEL-7B-MoT')


def load_lm(device='cuda'):
    from safetensors import safe_open
    from modeling.bagel.qwen2_navit import Qwen2Config, Qwen2ForCausalLM
    from modeling.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    cfg = Qwen2Config.from_pretrained(os.path.join(MODEL, 'llm_config.json'))
    cfg.layer_module = 'Qwen2MoTDecoderLayer'
    cfg.qk_norm = True
    cfg.tie_word_embeddings = False
    cfg.freeze_und = False
    with torch.device('meta'):
        lm = Qwen2ForCausalLM(cfg)
    sd = {}
    with safe_open(os.path.join(MODEL, 'ema.safetensors'), framework='pt', device=str(device)) as f:
        for k in f.keys():
            if k.startswith('language_model.'):
                sd[k[len('language_model.'):]] = f.get_tensor(k).to(torch.bfloat16)
    missing, unexpected = lm.load_state_dict(sd, strict=False, assign=True)
    lm.model.rotary_emb = Qwen2RotaryEmbedding(config=cfg, device=device)
    missing = [m for m in missing if not m.endswith('rotary_emb.inv_freq')]
    if missing or unexpected:
        raise RuntimeError(f'missing={missing[:8]} unexpected={unexpected[:8]}')
    lm.eval()
    return lm, cfg


def load_tokenizer():
    from modeling.qwen2 import Qwen2Tokenizer
    from data.data_utils import add_special_tokens
    tok = Qwen2Tokenizer.from_pretrained(MODEL)
    tok, new_token_ids, n_new = add_special_tokens(tok)
    return tok, new_token_ids


class FakeBagel:
    def __init__(self, lm, cfg):
        self.language_model = lm
        self.use_moe = True
        self.config = types.SimpleNamespace(llm_config=cfg)
        self.device = torch.device('cuda')


def bind_bagel_methods(cls=FakeBagel):
    from modeling.bagel import bagel as bagel_mod
    for name in ('prepare_prompts', 'forward_cache_update_text', 'prepare_start_tokens', 'generate_text_batch',
                 '_decode_static_pipelined', '_sample_next_token'):
        if hasattr(bagel_mod.Bagel, name):
            setattr(cls, name, getattr(bagel_mod.Bagel, name))
    return cls


def prompts(n=4, index=0):
    path = os.path.join(TREE, 'data/rl_train/train_geneval.jsonl')
    with open(path) as f:
        rows = [json.loads(line) for _, line in zip(range(index + 1), f)]
    return [rows[index]['prompt']] * n


def update_text(model, tok, new_token_ids, texts, ctx):
    """RolloutController.update_context_text, unchanged logic."""
    from modeling.bagel.qwen2_navit import StaticKVCache
    pkv, kv_lens, ropes = ctx['past_key_values'], ctx['kv_lens'], ctx['ropes']
    gen_in, kv_lens, ropes = model.prepare_prompts(curr_kvlens=kv_lens, curr_rope=ropes, prompts=texts,
                                                   tokenizer=tok, new_token_ids=new_token_ids)
    sel = torch.arange(0, len(kv_lens), device='cuda')
    pkv = model.forward_cache_update_text(pkv, selected_cache_indices=sel, **gen_in)
    ctx.update(kv_lens=kv_lens, ropes=ropes, past_key_values=pkv)
    if isinstance(pkv, StaticKVCache):
        pkv.seq_lens[:] = torch.tensor(kv_lens, dtype=torch.int32, device='cuda')
        pkv.invalidate_host_meta()
    return ctx


def new_ctx(cfg, batch, max_len):
    from modeling.bagel.qwen2_navit import StaticKVCache
    pkv = StaticKVCache(num_layers=cfg.num_hidden_layers, batch_size=batch, max_len=max_len,
                        num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.hidden_size // cfg.num_attention_heads,
                        dtype=torch.bfloat16, device='cuda')
    return dict(kv_lens=[0] * batch, ropes=[0] * batch, past_key_values=pkv)


def build_think_context(model, tok, ids, cfg, kind='gen', batch=4, prompt_index=0, max_new=196, filler_tokens=0,
                        mixed=False):
    """gen: GEN_THINK system prompt + prompt. edit: VLM_THINK system prompt + filler text standing in for the
    ViT block (same KV length class) + reflection prompt."""
    from train.rollout_controller import GEN_THINK_SYSTEM_PROMPT, VLM_THINK_SYSTEM_PROMPT
    p = prompts(batch, prompt_index)
    if mixed:  # different prompts per row: non-uniform query lengths in the prefill (varlen path)
        p = [prompts(1, prompt_index + i)[0] for i in range(batch)]
    if kind == 'gen':
        texts = [[GEN_THINK_SYSTEM_PROMPT] * batch, p]
    else:
        tmpl = ('The description of the target image is: {prompt}\nHow to further edit the provided image to make it '
                'consistent with the target description? Please provide concrete editing instructions in a single '
                'sentence. If no editing operation is needed, answer: no further edit needed.')
        filler = ' '.join(['The image shows'] + ['objects on a table near a window,'] * (filler_tokens // 8))
        texts = [[VLM_THINK_SYSTEM_PROMPT] * batch, [filler] * batch, [tmpl.format(prompt=x) for x in p]]
    est = sum(max(len(tok.encode(x)) for x in t) + 2 for t in texts) + max_new + 2 + (1026 if kind == 'gen' else 0)
    max_len = ((est + 127) // 128) * 128
    ctx = new_ctx(cfg, batch, max_len)
    for t in texts:
        ctx = update_text(model, tok, ids, t, ctx)
    return ctx


def think(model, ids, ctx, max_new, seed, batch=4):
    g = torch.Generator(device='cuda')
    g.manual_seed(seed)
    gen_in = model.prepare_start_tokens(ctx['kv_lens'], ctx['ropes'], ids)
    return model.generate_text_batch(past_key_values=ctx['past_key_values'], max_length=max_new, do_sample=True,
                                     temperature=0.9, topk=50, top_p=0.9, end_token_id=ids['eos_token_id'],
                                     generator=g, **gen_in)
