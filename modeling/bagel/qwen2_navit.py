# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/LICENSE.
#
# This modified file is released under the same license.


from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Union
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.functional import scaled_dot_product_attention
from transformers.utils import ModelOutput

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # type: ignore
from modeling.qwen2.modeling_qwen2 import (
    Qwen2Attention, 
    Qwen2MLP, 
    Qwen2PreTrainedModel, 
    Qwen2RMSNorm, 
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)

from modeling.qwen2.configuration_qwen2 import Qwen2Config as _Qwen2Config
from modeling.bagel.fused_train_ops import rms_norm as _rms_norm, swiglu as _swiglu, apply_rotary as _apply_rotary
import os as _os
_SPLIT_RESIDUAL = _os.environ.get("R3_FUSE_TRAIN_SPLIT", "1") != "0"
from . import fused_ops


torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
# flex_attention = torch.compile(flex_attention) # , dynamic=True, mode='max-autotune'
flex_attention = torch.compile(flex_attention)


def _gather_live(x, segments):
    """Rows of x inside the (start, end) segments, concatenated (contiguous)."""
    return torch.cat([x[start:end] for start, end in segments])


def _scatter_live(live, segments, rows):
    """Inverse of _gather_live: a rows-long matrix holding `live` in the segments and zeros elsewhere."""
    pieces, pos, off = [], 0, 0
    for start, end in segments:
        if start > pos:
            pieces.append(live.new_zeros((start - pos, live.shape[1])))
        pieces.append(live[off:off + end - start])
        off += end - start
        pos = end
    if pos < rows:
        pieces.append(live.new_zeros((rows - pos, live.shape[1])))
    return torch.cat(pieces)


class _LiveRowsLinear(torch.autograd.Function):
    """F.linear on the live rows (given as (start, end) segments) of a packed expert-token matrix;
    the other (dead) rows of the output are zero.  Dead rows belong to samples no loss can read, so
    their gradient is exactly zero in the reference graph.  Backward computes dx on the live rows
    only, but dW/db exactly as autograd's addmm/mm backward does on the FULL row layout (dead grad
    rows are zero, so the dead input rows contribute exact zeros): parameter-gradient reductions
    keep the reference shape."""

    @staticmethod
    def forward(ctx, x, weight, bias, segments):
        live = F.linear(_gather_live(x, segments), weight, bias)
        ctx.save_for_backward(x, weight)
        ctx.segments = segments
        ctx.has_bias = bias is not None
        return _scatter_live(live, segments, x.shape[0])

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        grad_x = grad_w = grad_b = None
        if ctx.needs_input_grad[0]:                   # mm_mat1_backward: grad.mm(mat2.t()), live rows
            grad_x = _scatter_live(_gather_live(grad, ctx.segments).mm(weight), ctx.segments, grad.shape[0])
        if ctx.needs_input_grad[1]:
            grad_w = grad.t().mm(x)                   # mm_mat2_backward (column-major weight.t())
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_b = grad.sum((0,), keepdim=True).view(-1)  # autograd sum_to(grad, bias.shape)
        return grad_x, grad_w, grad_b, None


def _live_linear(linear, x, segments):
    if segments is None:
        return linear(x)
    if torch.is_grad_enabled():
        return _LiveRowsLinear.apply(x, linear.weight, linear.bias, segments)
    return _scatter_live(linear(_gather_live(x, segments)), segments, x.shape[0])


def _live_mlp(mlp, x, segments):
    # Qwen2MLP.forward with the SiLU(gate) * up product fused (bitwise; fused_train_ops.swiglu)
    if segments is None:
        return mlp.down_proj(_swiglu(mlp.act_fn, mlp.gate_proj(x), mlp.up_proj(x)))
    gate = _live_linear(mlp.gate_proj, x, segments)
    return _live_linear(mlp.down_proj, _swiglu(mlp.act_fn, gate, _live_linear(mlp.up_proj, x, segments)), segments)


class Qwen2Config(_Qwen2Config):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2Model`]. It is used to instantiate a
    Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # Initializing a Qwen2 style configuration
    >>> configuration = Qwen2Config()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        is_causal=True,
        _attn_implementation="flash_attention_2",
        qk_norm=True,
        layer_module="Qwen2DecoderLayer",
        freeze_und=False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            is_causal=is_causal,
            _attn_implementation=_attn_implementation,
            **kwargs,
        )
        self.qk_norm = qk_norm
        self.layer_module = layer_module
        self.freeze_und = freeze_und


class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0


class NaiveCacheMultiSeq:
    def __init__(self, num_layers: int, num_seqs: int, init_cache: Union[NaiveCache, None] = None):
        self.num_layers = num_layers
        self.num_seqs = num_seqs
        if init_cache is None:
            self.caches = [NaiveCache(num_layers) for _ in range(num_seqs)]
        else:
            self.caches = [deepcopy(init_cache) for _ in range(num_seqs)]

    def get_kv_cache(self, layer_idx: int, selected_seq: Union[List[int], None] = None) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        key_caches = []
        value_caches = []
        if selected_seq is None:
            selected_seq = range(self.num_seqs)
        for seq_idx in selected_seq:
            if 0 <= seq_idx < self.num_seqs:
                cache_instance = self.caches[seq_idx]
                key_caches.append(cache_instance.key_cache[layer_idx])
                value_caches.append(cache_instance.value_cache[layer_idx])
            else:
                raise IndexError(f"Sequence index {seq_idx} out of range for {self.num_seqs} sequences.")
        return key_caches, value_caches
    
    def get_seq_lens(self, selected_seq: List[int]) -> List[int]:
        seq_lens = []
        for seq_idx in selected_seq:
            if 0 <= seq_idx < self.num_seqs:
                seq_lens.append(self.caches[seq_idx].seq_lens)
            else:
                raise IndexError(f"Sequence index {seq_idx} out of range for {self.num_seqs} sequences.")
        return seq_lens


class StaticKVCache:
    """Pre-allocated KV cache for efficient batched inference."""

    def __init__(
        self,
        num_layers,
        batch_size,
        max_len,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        shape = (num_layers, batch_size, max_len, num_kv_heads, head_dim)
        self.key_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.value_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self._host_meta = None

    @property
    def num_seqs(self):
        return self.batch_size

    def invalidate_host_meta(self):
        self._host_meta = None

    def advance_seq_lens(self, amount, indices=None):
        self._host_meta = None
        if indices is None:
            self.seq_lens += amount
        else:
            self.seq_lens[indices] += amount


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: Optional[Union[NaiveCache, NaiveCacheMultiSeq, StaticKVCache]] = None


def _normalize_static_cache_indices(selected_cache_indices, device):
    if selected_cache_indices is None:
        return None
    if isinstance(selected_cache_indices, torch.Tensor):
        return selected_cache_indices.to(device=device, dtype=torch.int32)
    return torch.tensor(selected_cache_indices, dtype=torch.int32, device=device)


def _static_kv_host_meta(past_key_values, query_lens, selected_cache_indices):
    """Host copies of the per-forward lengths every layer needs, read with one device->host copy per forward instead
    of several .item() syncs per layer. Cached on the cache object for the (query_lens, selected_cache_indices) objects
    of the current forward (strong references are held, so identity is unambiguous); every seq_lens writer resets it."""
    cached = past_key_values._host_meta
    if cached is not None and cached[0] is query_lens and cached[1] is selected_cache_indices:
        return cached[2]
    cache_batch_idx = _normalize_static_cache_indices(selected_cache_indices, past_key_values.seq_lens.device)
    effective = past_key_values.seq_lens[cache_batch_idx.long()] if cache_batch_idx is not None else past_key_values.seq_lens
    num = len(query_lens)
    host = torch.cat([effective.to(torch.int64), query_lens.to(device=effective.device, dtype=torch.int64)]).tolist()
    eff, q = host[:num], host[num:]
    total = [e + x for e, x in zip(eff, q)]
    meta = dict(q=q, max_q=max(q), uniform_q=all(x == q[0] for x in q), max_total=max(total), sum_total=sum(total),
                sum_eff=sum(eff), max_eff=max(eff))
    past_key_values._host_meta = (query_lens, selected_cache_indices, meta)
    return meta


def _static_kv_packing_meta(past_key_values, query_lens, selected_cache_indices, device):
    """Reuse integer packing indexes while the existing host metadata is valid.

    No K/V values are cached here: each layer and flow step still reads its
    current tensors. Length updates invalidate this plan together with _host_meta.
    """
    meta = _static_kv_host_meta(past_key_values, query_lens, selected_cache_indices)
    if meta['max_total'] > past_key_values.max_len:
        raise RuntimeError(
            f"StaticKVCache overflow: need {meta['max_total']} tokens, "
            f"but max_len={past_key_values.max_len}"
        )
    if 'packing' in meta:
        return meta, meta['packing']

    num_seqs = len(query_lens)
    cache_batch_idx = _normalize_static_cache_indices(selected_cache_indices, device)
    cache_indices = cache_batch_idx.long() if cache_batch_idx is not None else torch.arange(num_seqs, device=device)
    effective_seqlens = (
        past_key_values.seq_lens[cache_indices]
        if cache_batch_idx is not None
        else past_key_values.seq_lens
    )
    total_kv_lens = effective_seqlens + query_lens
    cu_kv = torch.nn.functional.pad(torch.cumsum(total_kv_lens, dim=0), (1, 0))
    packing = dict(
        cache_batch_idx=cache_batch_idx, cache_indices=cache_indices,
        effective_seqlens=effective_seqlens, total_kv_lens=total_kv_lens,
        cu_seqlens_q=torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0)).to(torch.int32),
        cu_seqlens_k=cu_kv.to(torch.int32),
    )
    if meta['sum_eff'] > 0:
        max_past = meta['max_eff']
        past_pos = torch.arange(max_past, device=device).unsqueeze(0).expand(num_seqs, -1)
        past_mask = past_pos < effective_seqlens.unsqueeze(1)
        batch_idx = cache_indices.unsqueeze(1).expand_as(past_pos)
        past_dest = cu_kv[:-1].unsqueeze(1) + past_pos
        packing.update(past_batch=batch_idx[past_mask], past_pos=past_pos[past_mask],
                       past_dest=past_dest[past_mask])

    max_new = meta['max_q']
    new_pos = torch.arange(max_new, device=device).unsqueeze(0).expand(num_seqs, -1)
    new_mask = new_pos < query_lens.unsqueeze(1)
    new_dest = (cu_kv[:-1] + effective_seqlens).unsqueeze(1) + new_pos
    packing['new_dest'] = new_dest[new_mask]
    meta['packing'] = packing
    return meta, packing


def _merge_static_kv_prefix_and_query(
    packed_key_states,
    packed_value_states,
    query_lens,
    past_key_values,
    layer_idx,
    selected_cache_indices=None,
):
    meta, packing = _static_kv_packing_meta(
        past_key_values, query_lens, selected_cache_indices, packed_key_states.device,
    )
    k_cache = past_key_values.key_cache[layer_idx]
    v_cache = past_key_values.value_cache[layer_idx]
    total_kv = meta['sum_total']
    merged_k = packed_key_states.new_empty(total_kv, packed_key_states.shape[-2], packed_key_states.shape[-1])
    merged_v = packed_value_states.new_empty(total_kv, packed_value_states.shape[-2], packed_value_states.shape[-1])

    if meta['sum_eff'] > 0:
        past_k = k_cache[packing['past_batch'], packing['past_pos']]
        past_v = v_cache[packing['past_batch'], packing['past_pos']]
        merged_k[packing['past_dest']] = past_k
        merged_v[packing['past_dest']] = past_v

    merged_k[packing['new_dest']] = packed_key_states
    merged_v[packing['new_dest']] = packed_value_states
    return (merged_k, merged_v, packing['total_kv_lens'], packing['effective_seqlens'],
            packing['cache_indices'], packing['cache_batch_idx'])


def _static_kv_cache_attention(
    packed_query_states,
    packed_key_states,
    packed_value_states,
    query_lens,
    past_key_values,
    layer_idx,
    num_heads,
    num_kv_heads,
    head_dim,
    hidden_size,
    selected_cache_indices=None,
    is_causal=True,
):
    num_seqs = len(query_lens)
    device = packed_query_states.device
    # The cache indices, gathered cache lengths and cache write positions are the same for every layer of one
    # forward (no layer writes seq_lens), so they are computed once per forward; Qwen2Model.forward_inference opens
    # and closes that scope (_forward_prep), so nothing is reused across forwards or graph captures.
    prep = getattr(past_key_values, '_forward_prep', None)
    if prep is not None and prep.get('key') is selected_cache_indices and prep.get('device') == device:
        cache_batch_idx, cache_indices, effective_seqlens = prep['base']
    else:
        cache_batch_idx = _normalize_static_cache_indices(selected_cache_indices, device)
        cache_indices = cache_batch_idx.long() if cache_batch_idx is not None else torch.arange(num_seqs, device=device)
        effective_seqlens = (
            past_key_values.seq_lens[cache_indices]
            if cache_batch_idx is not None
            else past_key_values.seq_lens
        )
        if prep is not None:
            prep.clear()
            prep.update(key=selected_cache_indices, device=device,
                        base=(cache_batch_idx, cache_indices, effective_seqlens))
    meta = _static_kv_host_meta(past_key_values, query_lens, selected_cache_indices)
    if meta['max_total'] > past_key_values.max_len:
        raise RuntimeError(
            f"StaticKVCache overflow: need {meta['max_total']} tokens, "
            f"but max_len={past_key_values.max_len}"
        )

    k_cache = past_key_values.key_cache[layer_idx]
    v_cache = past_key_values.value_cache[layer_idx]

    if num_seqs == 1 or meta['uniform_q']:
        seqlen_q = meta['q'][0]
        q = packed_query_states.view(num_seqs, seqlen_q, num_heads, head_dim)
        k_new = packed_key_states.view(num_seqs, seqlen_q, num_kv_heads, head_dim)
        v_new = packed_value_states.view(num_seqs, seqlen_q, num_kv_heads, head_dim)
        packed_attn_output = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k_new,
            v=v_new,
            cache_seqlens=effective_seqlens,
            cache_batch_idx=cache_batch_idx,
            causal=is_causal,
        ).reshape(-1, hidden_size)
        return packed_attn_output

    merged_k, merged_v, total_kv_lens, effective_seqlens, cache_indices, _ = _merge_static_kv_prefix_and_query(
        packed_key_states,
        packed_value_states,
        query_lens,
        past_key_values,
        layer_idx,
        selected_cache_indices=selected_cache_indices,
    )
    write = prep.get('write') if prep is not None and 'base' in prep else None
    if write is not None and write[0] is query_lens:
        _, cu_seqlens_q, cu_seqlens_k, write_rows, write_cols = write
    else:
        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0)).to(torch.int32)
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(total_kv_lens, dim=0), (1, 0)).to(torch.int32)
        max_new = meta['max_q']
        write_pos = torch.arange(max_new, device=device).unsqueeze(0).expand(num_seqs, -1)
        write_mask = write_pos < query_lens.unsqueeze(1)
        write_batch = cache_indices.unsqueeze(1).expand_as(write_pos)
        write_cache_pos = effective_seqlens.unsqueeze(1) + write_pos
        write_rows, write_cols = write_batch[write_mask], write_cache_pos[write_mask]
        if prep is not None and 'base' in prep:
            prep['write'] = (query_lens, cu_seqlens_q, cu_seqlens_k, write_rows, write_cols)
    packed_attn_output = flash_attn_varlen_func(
        q=packed_query_states,
        k=merged_k,
        v=merged_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=meta['max_q'],
        max_seqlen_k=meta['max_total'],
        causal=is_causal,
    ).reshape(-1, hidden_size)

    k_cache[write_rows, write_cols] = packed_key_states
    v_cache[write_rows, write_cols] = packed_value_states
    return packed_attn_output


def _static_kv_cache_readonly_attention(
    packed_query_states,
    packed_key_states,
    packed_value_states,
    query_lens,
    past_key_values,
    layer_idx,
    num_heads,
    num_kv_heads,
    head_dim,
    hidden_size,
    selected_cache_indices=None,
    is_causal=False,
):
    meta = _static_kv_host_meta(past_key_values, query_lens, selected_cache_indices)
    merged_k, merged_v, total_kv_lens, _, _, _ = _merge_static_kv_prefix_and_query(
        packed_key_states,
        packed_value_states,
        query_lens,
        past_key_values,
        layer_idx,
        selected_cache_indices=selected_cache_indices,
    )
    cu_seqlens_q = meta['packing']['cu_seqlens_q']
    cu_seqlens_k = meta['packing']['cu_seqlens_k']
    return flash_attn_varlen_func(
        q=packed_query_states,
        k=merged_k,
        v=merged_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=meta['max_q'],
        max_seqlen_k=meta['max_total'],
        causal=is_causal,
    ).reshape(-1, hidden_size)


def pad_sequence(tensor, pad_size):
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)


class PackedAttention(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, *args, **kwargs):
        if self.training or kwargs.get('ref_forward', False):
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask: List[torch.Tensor],
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ):
        packed_query_states = self.q_proj(packed_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        if isinstance(attention_mask, List):
            packed_key_states = packed_key_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states = packed_key_states.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0), 
                        key_states.to(torch.bfloat16).unsqueeze(0), 
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states = pad_sequence(packed_query_states.permute(1, 0, 2), pad_size)
            packed_key_states = pad_sequence(packed_key_states.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states.unsqueeze(0), 
                packed_key_states.unsqueeze(0), 
                packed_value_states.unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)

        return packed_attn_output

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[Union[NaiveCache, NaiveCacheMultiSeq, StaticKVCache]] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        selected_cache_indices: Optional[List[int]] = None,
    ):
        packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if isinstance(past_key_values, StaticKVCache):
            if update_past_key_values:
                packed_attn_output = _static_kv_cache_attention(
                    packed_query_states,
                    packed_key_states,
                    packed_value_states,
                    query_lens,
                    past_key_values,
                    self.layer_idx,
                    self.num_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    self.hidden_size,
                    selected_cache_indices=selected_cache_indices,
                    is_causal=is_causal,
                )
            else:
                packed_attn_output = _static_kv_cache_readonly_attention(
                    packed_query_states,
                    packed_key_states,
                    packed_value_states,
                    query_lens,
                    past_key_values,
                    self.layer_idx,
                    self.num_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    self.hidden_size,
                    selected_cache_indices=selected_cache_indices,
                    is_causal=is_causal,
                )
            return self.o_proj(packed_attn_output), past_key_values

        if past_key_values is not None and isinstance(past_key_values, NaiveCache) and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_value_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        elif past_key_values is not None and isinstance(past_key_values, NaiveCacheMultiSeq) and past_key_values.caches[0].key_cache[self.layer_idx] is not None:
            past_key_states, past_value_states = past_key_values.get_kv_cache(self.layer_idx, selected_cache_indices)
            past_key_states = torch.cat(past_key_states, dim=0)
            past_value_states = torch.cat(past_value_states, dim=0)

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_value_states = past_value_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)

        if update_past_key_values:
            if isinstance(past_key_values, NaiveCache):
                past_key_values.key_cache[self.layer_idx] = merged_key_states
                past_key_values.value_cache[self.layer_idx] = merged_value_states
            elif isinstance(past_key_values, NaiveCacheMultiSeq):
                cumsum_lens = [0] + torch.cumsum(key_values_lens, dim=0).tolist()
                for i, select_idx in enumerate(selected_cache_indices):
                    past_key_values.caches[select_idx].key_cache[self.layer_idx] = merged_key_states[cumsum_lens[i]:cumsum_lens[i + 1]]
                    past_key_values.caches[select_idx].value_cache[self.layer_idx] = merged_value_states[cumsum_lens[i]:cumsum_lens[i + 1]]

        return packed_attn_output, past_key_values


def _mot_inference_layout(sequence, text_indexes, vae_indexes):
    """Return an expert permutation only for a complete, disjoint partition."""
    indexes = (text_indexes, vae_indexes)
    if any(index.ndim != 1 or index.dtype != torch.long or
           index.device != sequence.device for index in indexes):
        return None
    if sum(index.numel() for index in indexes) != sequence.shape[0]:
        return None
    order = torch.cat(indexes)
    sorted_indexes, inverse = order.sort()
    if not torch.equal(sorted_indexes, torch.arange(
        sequence.shape[0], device=sequence.device, dtype=order.dtype,
    )):
        return None
    return text_indexes.numel(), order, inverse


class PackedAttentionMoT(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, *args, **kwargs):
        if self.training or kwargs.get('ref_forward', False):
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
        ref_forward: Optional[bool] = False,
        packed_und_live_segments: Optional[tuple] = None,
        packed_gen_live_segments: Optional[tuple] = None,
    ):
        und_keep, gen_keep = packed_und_live_segments, packed_gen_live_segments
        packed_query_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_heads * self.head_dim))
        packed_key_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_key_value_heads * self.head_dim))
        packed_value_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_key_value_heads * self.head_dim))

        packed_sequence_und = packed_sequence[packed_und_token_indexes]
        packed_sequence_gen = packed_sequence[packed_gen_token_indexes]

        packed_query_states[packed_und_token_indexes] = _live_linear(self.q_proj, packed_sequence_und, und_keep)
        packed_query_states[packed_gen_token_indexes] = _live_linear(self.q_proj_moe_gen, packed_sequence_gen, gen_keep)

        packed_key_states[packed_und_token_indexes] = _live_linear(self.k_proj, packed_sequence_und, und_keep)
        packed_key_states[packed_gen_token_indexes] = _live_linear(self.k_proj_moe_gen, packed_sequence_gen, gen_keep)

        packed_value_states[packed_und_token_indexes] = _live_linear(self.v_proj, packed_sequence_und, und_keep)
        packed_value_states[packed_gen_token_indexes] = _live_linear(self.v_proj_moe_gen, packed_sequence_gen, gen_keep)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
        if self.config.freeze_und:
            packed_value_states[packed_und_token_indexes] = packed_value_states[packed_und_token_indexes].detach()

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        packed_query_states_[packed_und_token_indexes] = _rms_norm(self.q_norm, packed_query_states[packed_und_token_indexes])
        if self.config.freeze_und:
            packed_query_states_[packed_und_token_indexes] = packed_query_states_[packed_und_token_indexes].detach()
        packed_query_states_[packed_gen_token_indexes] = _rms_norm(self.q_norm_moe_gen, packed_query_states[packed_gen_token_indexes])

        packed_key_states_[packed_und_token_indexes] = _rms_norm(self.k_norm, packed_key_states[packed_und_token_indexes])
        if self.config.freeze_und:
            packed_key_states_[packed_und_token_indexes] = packed_key_states_[packed_und_token_indexes].detach()
        packed_key_states_[packed_gen_token_indexes] = _rms_norm(self.k_norm_moe_gen, packed_key_states[packed_gen_token_indexes])

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states_, packed_key_states_ = _apply_rotary(
            packed_query_states_, packed_key_states_, packed_cos, packed_sin, apply_rotary_pos_emb
        )

        if isinstance(attention_mask, List):
            packed_key_states_ = packed_key_states_[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states_ = packed_key_states_.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0), 
                        key_states.to(torch.bfloat16).unsqueeze(0), 
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states_ = pad_sequence(packed_query_states_.permute(1, 0, 2), pad_size)
            packed_key_states_ = pad_sequence(packed_key_states_.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states_.unsqueeze(0), # 1, num_head, L, head_dim
                packed_key_states_.unsqueeze(0), 
                packed_value_states.unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)
        packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
        packed_attn_output_[packed_und_token_indexes] = _live_linear(
            self.o_proj, packed_attn_output[packed_und_token_indexes], und_keep)
        packed_attn_output_[packed_gen_token_indexes] = _live_linear(
            self.o_proj_moe_gen, packed_attn_output[packed_gen_token_indexes], gen_keep)

        return packed_attn_output_

    def forward_train_split(self, norm_und, norm_gen, sample_lens, attention_mask, rope_und, rope_gen,
                            packed_und_token_indexes, packed_gen_token_indexes, seq_len,
                            und_keep=None, gen_keep=None):
        """forward_train on the split (per-expert) residual layout: takes the und / gen normed rows (packed
        expert-list order) and returns the und / gen o_proj outputs.  Every value-producing op is the one of
        forward_train on identically shaped inputs; only the gathers / zero-filled scatter buffers around the
        per-expert ops are gone.  The q/k/v projection Linear nodes are created first and in forward_train's order
        (q und, q gen, k und, k gen, v und, v gen), so autograd accumulates their input gradients in the same
        order."""
        nh, nkv, hd = self.num_heads, self.num_key_value_heads, self.head_dim
        q_u = _live_linear(self.q_proj, norm_und, und_keep)
        q_g = _live_linear(self.q_proj_moe_gen, norm_gen, gen_keep)
        k_u = _live_linear(self.k_proj, norm_und, und_keep)
        k_g = _live_linear(self.k_proj_moe_gen, norm_gen, gen_keep)
        v_u = _live_linear(self.v_proj, norm_und, und_keep)
        v_g = _live_linear(self.v_proj_moe_gen, norm_gen, gen_keep)
        packed_value_states = norm_und.new_zeros((seq_len, nkv * hd))
        packed_value_states[packed_und_token_indexes] = v_u
        packed_value_states[packed_gen_token_indexes] = v_g
        packed_value_states = packed_value_states.view(-1, nkv, hd)
        # q/k norm + RoPE per expert (row-wise ops: identical values to norming/rotating the scattered rows)
        q_u, k_u = _apply_rotary(_rms_norm(self.q_norm, q_u.view(-1, nh, hd), plus_zero=True),
                                 _rms_norm(self.k_norm, k_u.view(-1, nkv, hd), plus_zero=True),
                                 rope_und[0], rope_und[1], apply_rotary_pos_emb)
        q_g, k_g = _apply_rotary(_rms_norm(self.q_norm_moe_gen, q_g.view(-1, nh, hd), plus_zero=True),
                                 _rms_norm(self.k_norm_moe_gen, k_g.view(-1, nkv, hd), plus_zero=True),
                                 rope_gen[0], rope_gen[1], apply_rotary_pos_emb)
        packed_query_states_ = norm_und.new_zeros((seq_len, nh, hd))
        packed_query_states_[packed_und_token_indexes] = q_u
        packed_query_states_[packed_gen_token_indexes] = q_g
        packed_key_states_ = norm_und.new_zeros((seq_len, nkv, hd))
        packed_key_states_[packed_und_token_indexes] = k_u
        packed_key_states_[packed_gen_token_indexes] = k_g

        pad_size = sum(sample_lens) - seq_len
        packed_query_states_ = pad_sequence(packed_query_states_.permute(1, 0, 2), pad_size)
        packed_key_states_ = pad_sequence(packed_key_states_.permute(1, 0, 2), pad_size)
        packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
        packed_attn_output = flex_attention(
            packed_query_states_.unsqueeze(0),
            packed_key_states_.unsqueeze(0),
            packed_value_states.unsqueeze(0),
            enable_gqa=True,
            block_mask=attention_mask,
        )
        end_index = packed_attn_output.shape[2] - pad_size
        packed_attn_output = packed_attn_output[0, :, :end_index, :]
        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, nh * hd)
        o_u = _live_linear(self.o_proj, packed_attn_output[packed_und_token_indexes], und_keep)
        o_g = _live_linear(self.o_proj_moe_gen, packed_attn_output[packed_gen_token_indexes], gen_keep)
        return o_u, o_g

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[Union[NaiveCache, NaiveCacheMultiSeq]] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        selected_cache_indices: Optional[List[int]] = None,
        mot_inference_layout=None,
    ):
        roped = False
        side = fused_ops.concurrent_side(packed_query_sequence) if mode == 'und' else None
        if side is not None:
            # k and v projections on a side stream, concurrent with q (same GEMM calls; outputs consumed on the
            # current stream only after it waits for the side stream; the side stream first waits for the current
            # one, so a block it reuses later is never still being read).
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
                packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
            cur.wait_stream(side)
        elif mode == 'und':
            packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
            packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        if mode == 'und':
            fused = None
            if (fused_ops.ENABLED and isinstance(self.q_norm, Qwen2RMSNorm) and isinstance(self.k_norm, Qwen2RMSNorm)
                    and self.q_norm.variance_epsilon == self.k_norm.variance_epsilon):
                # q_norm, k_norm and RoPE in one kernel with the eager rounding and reduction order (bitwise).
                fused = fused_ops.qk_norm_rope(
                    packed_query_states, packed_key_states, self.q_norm.weight, self.k_norm.weight,
                    packed_query_position_embeddings[0], packed_query_position_embeddings[1],
                    self.q_norm.variance_epsilon,
                )
            if fused is not None:
                packed_query_states, packed_key_states = fused
                roped = True
            else:
                packed_query_states = self.q_norm(packed_query_states)
                packed_key_states = self.k_norm(packed_key_states)
        elif mode == 'gen' and mot_inference_layout is not None:
            text_count, _, inverse = mot_inference_layout
            packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
            text_sequence = packed_query_sequence[:text_count]
            vae_sequence = packed_query_sequence[text_count:]
            # Keep the original projection order and BF16 store boundaries.
            packed_query_states = torch.cat((
                self.q_proj(text_sequence).to(torch.bfloat16),
                self.q_proj_moe_gen(vae_sequence).to(torch.bfloat16),
            )).view(-1, self.num_heads, self.head_dim)
            packed_key_states = torch.cat((
                self.k_proj(text_sequence).to(torch.bfloat16),
                self.k_proj_moe_gen(vae_sequence).to(torch.bfloat16),
            )).view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = torch.cat((
                self.v_proj(text_sequence).to(torch.bfloat16),
                self.v_proj_moe_gen(vae_sequence).to(torch.bfloat16),
            )).view(-1, self.num_key_value_heads, self.head_dim)

            packed_query_states = packed_query_states.to(torch.float32)
            packed_query_states = torch.cat((
                self.q_norm(packed_query_states[:text_count]).to(torch.float32),
                self.q_norm_moe_gen(packed_query_states[text_count:]).to(torch.float32),
            ))[inverse]
            packed_key_states = packed_key_states.to(torch.float32)
            packed_key_states = torch.cat((
                self.k_norm(packed_key_states[:text_count]).to(torch.float32),
                self.k_norm_moe_gen(packed_key_states[text_count:]).to(torch.float32),
            ))[inverse]
            # RoPE, attention and cache packing retain the original token order.
            packed_value_states = packed_value_states[inverse]
        elif mode == 'gen':
            packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
            packed_query_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_heads * self.head_dim))
            packed_key_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))
            packed_value_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))

            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

            packed_query_states[packed_text_indexes] = self.q_proj(packed_text_query_sequence)
            packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(packed_vae_query_sequence)

            packed_key_states[packed_text_indexes] = self.k_proj(packed_text_query_sequence)
            packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(packed_vae_query_sequence)

            packed_value_states[packed_text_indexes] = self.v_proj(packed_text_query_sequence)
            packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(packed_vae_query_sequence)

            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

            packed_query_states = packed_query_states.to(torch.float32)
            packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
            packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(packed_query_states[packed_vae_token_indexes])

            packed_key_states = packed_key_states.to(torch.float32)
            packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
            packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_vae_token_indexes])

        if not roped:
            packed_cos, packed_sin = packed_query_position_embeddings
            packed_query_states, packed_key_states = apply_rotary_pos_emb(
                packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
            )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)
        if isinstance(past_key_values, StaticKVCache):
            if update_past_key_values:
                packed_attn_output = _static_kv_cache_attention(
                    packed_query_states,
                    packed_key_states,
                    packed_value_states,
                    query_lens,
                    past_key_values,
                    self.layer_idx,
                    self.num_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    self.hidden_size,
                    selected_cache_indices=selected_cache_indices,
                    is_causal=is_causal,
                )
            else:
                packed_attn_output = _static_kv_cache_readonly_attention(
                    packed_query_states,
                    packed_key_states,
                    packed_value_states,
                    query_lens,
                    past_key_values,
                    self.layer_idx,
                    self.num_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    self.hidden_size,
                    selected_cache_indices=selected_cache_indices,
                    is_causal=is_causal,
                )
            if mode == 'und':
                packed_attn_output = self.o_proj(packed_attn_output)
            elif mode == 'gen' and mot_inference_layout is not None:
                packed_attn_output = torch.cat((
                    self.o_proj(packed_attn_output[packed_text_indexes]).to(packed_attn_output.dtype),
                    self.o_proj_moe_gen(packed_attn_output[packed_vae_token_indexes]).to(packed_attn_output.dtype),
                ))
            elif mode == 'gen':
                packed_attn_output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
                packed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(packed_attn_output[packed_vae_token_indexes])
            return packed_attn_output, past_key_values

        if past_key_values is not None and isinstance(past_key_values, NaiveCache) and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_value_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        elif past_key_values is not None and isinstance(past_key_values, NaiveCacheMultiSeq) and past_key_values.caches[0].key_cache[self.layer_idx] is not None:
            past_key_states, past_value_states = past_key_values.get_kv_cache(self.layer_idx, selected_cache_indices)
            past_key_states = torch.cat(past_key_states, dim=0)
            past_value_states = torch.cat(past_value_states, dim=0)
            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_value_states = past_value_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        if mode == 'und':
            packed_attn_output = self.o_proj(packed_attn_output)
        elif mode == 'gen' and mot_inference_layout is not None:
            packed_attn_output = torch.cat((
                self.o_proj(packed_attn_output[packed_text_indexes]).to(packed_attn_output.dtype),
                self.o_proj_moe_gen(packed_attn_output[packed_vae_token_indexes]).to(packed_attn_output.dtype),
            ))
        elif mode == 'gen':
            packed_attn_output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
            packed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(packed_attn_output[packed_vae_token_indexes])

        if update_past_key_values:
            if isinstance(past_key_values, NaiveCache):
                past_key_values.key_cache[self.layer_idx] = merged_key_states
                past_key_values.value_cache[self.layer_idx] = merged_value_states
            elif isinstance(past_key_values, NaiveCacheMultiSeq):
                key_values_lens = [0] + torch.cumsum(key_values_lens, dim=0).tolist()
                for i, select_idx in enumerate(selected_cache_indices):
                    past_key_values.caches[select_idx].key_cache[self.layer_idx] = merged_key_states[key_values_lens[i]:key_values_lens[i+1]]
                    past_key_values.caches[select_idx].value_cache[self.layer_idx] = merged_value_states[key_values_lens[i]:key_values_lens[i+1]]

        return packed_attn_output, past_key_values


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PackedAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training or kwargs.get('ref_forward', False):
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)
        packed_sequence = self.mlp(packed_sequence)
        packed_sequence = residual + packed_sequence

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[Union[NaiveCache, NaiveCacheMultiSeq, StaticKVCache]] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        selected_cache_indices: Optional[List[int]] = None,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            selected_cache_indices=selected_cache_indices,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        packed_query_sequence = self.mlp(packed_query_sequence)
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


class Qwen2MoTDecoderLayer(nn.Module):
    def __init__(
        self, 
        config, 
        layer_idx: Optional[int] = None, 
        attn_module: Optional[Qwen2Attention] = PackedAttentionMoT,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.freeze_und = config.freeze_und

        self.self_attn = attn_module(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training or kwargs.get('ref_forward', False):
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
        ref_forward: Optional[bool] = False,
        packed_und_live_segments: Optional[tuple] = None,
        packed_gen_live_segments: Optional[tuple] = None,
        split_sequence: Optional[tuple] = None,
        split_rope: Optional[tuple] = None,
    ):
        if split_sequence is not None:
            # split residual layout: (und rows, gen rows) in, (und rows, gen rows) out
            hu, hg = split_sequence
            seq_len = packed_und_token_indexes.shape[0] + packed_gen_token_indexes.shape[0]
            o_u, o_g = self.self_attn.forward_train_split(
                _rms_norm(self.input_layernorm, hu, plus_zero=True),
                _rms_norm(self.input_layernorm_moe_gen, hg, plus_zero=True),
                sample_lens, attention_mask, split_rope[0], split_rope[1],
                packed_und_token_indexes, packed_gen_token_indexes, seq_len,
                packed_und_live_segments, packed_gen_live_segments)
            hu = hu + o_u
            hg = hg + o_g
            m_u = _live_mlp(self.mlp, _rms_norm(self.post_attention_layernorm, hu, plus_zero=True),
                            packed_und_live_segments)
            m_g = _live_mlp(self.mlp_moe_gen, _rms_norm(self.post_attention_layernorm_moe_gen, hg, plus_zero=True),
                            packed_gen_live_segments)
            return hu + m_u, hg + m_g
        keep_inputs = {}
        if packed_und_live_segments is not None:
            keep_inputs.update(packed_und_live_segments=packed_und_live_segments)
        if packed_gen_live_segments is not None:
            keep_inputs.update(packed_gen_live_segments=packed_gen_live_segments)

        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = _rms_norm(self.input_layernorm, packed_sequence[packed_und_token_indexes])
        packed_sequence_[packed_gen_token_indexes] = _rms_norm(self.input_layernorm_moe_gen, packed_sequence[packed_gen_token_indexes])

        # Self Attention
        packed_sequence_ = self.self_attn(
            packed_sequence=packed_sequence_,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
            ref_forward=ref_forward,
            **keep_inputs,
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
        packed_sequence = residual + packed_sequence_

        # Fully Connected
        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = _live_mlp(
            self.mlp, _rms_norm(self.post_attention_layernorm, packed_sequence[packed_und_token_indexes]), packed_und_live_segments
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
    
        packed_sequence_[packed_gen_token_indexes] = _live_mlp(
            self.mlp_moe_gen, _rms_norm(self.post_attention_layernorm_moe_gen, packed_sequence[packed_gen_token_indexes]),
            packed_gen_live_segments,
        )
        packed_sequence = residual + packed_sequence_

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[Union[NaiveCache, NaiveCacheMultiSeq]] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        selected_cache_indices: Optional[List[int]] = None,
        mot_inference_layout=None,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.input_layernorm(packed_query_sequence)
        elif mode == "gen" and mot_inference_layout is not None:
            text_count = mot_inference_layout[0]
            packed_query_sequence = torch.cat((
                self.input_layernorm(packed_query_sequence[:text_count]).to(residual.dtype),
                self.input_layernorm_moe_gen(packed_query_sequence[text_count:]).to(residual.dtype),
            ))
        elif mode == "gen":
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            packed_query_sequence_[packed_text_indexes] = self.input_layernorm(packed_query_sequence[packed_text_indexes])
            packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_moe_gen(packed_query_sequence[packed_vae_token_indexes])
            packed_query_sequence = packed_query_sequence_

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
            selected_cache_indices=selected_cache_indices,
            mot_inference_layout=mot_inference_layout,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen" and mot_inference_layout is not None:
            text_sequence = self.post_attention_layernorm(
                packed_query_sequence[:text_count],
            ).to(torch.bfloat16)
            vae_sequence = self.post_attention_layernorm_moe_gen(
                packed_query_sequence[text_count:],
            ).to(torch.bfloat16)
            packed_query_sequence = torch.cat((
                self.mlp(text_sequence).to(torch.bfloat16),
                self.mlp_moe_gen(vae_sequence).to(torch.bfloat16),
            ))
        elif mode == "gen":
            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
            packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
            packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(packed_vae_query_sequence).to(torch.bfloat16)

            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_text_query_sequence)
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(packed_vae_query_sequence)
            packed_query_sequence = packed_query_sequence_

        packed_query_sequence = residual + packed_query_sequence
        return packed_query_sequence, past_key_values


    def forward_inference_und_fused(self, x, delta, query_lens, packed_query_position_embeddings,
                                    packed_query_indexes, past_key_values, key_values_lens, packed_key_value_indexes,
                                    update_past_key_values, is_causal, selected_cache_indices):
        """mode='und' inference layer with fused kernels; same operations and roundings as forward_inference.

        x is the residual stream entering this layer when delta is None; otherwise the stream is x + delta (the
        previous layer's pending residual add, fused here into the input RMSNorm). Returns the post-attention
        stream and this layer's MLP output, whose residual add is left to the next layer / the final norm."""
        if delta is None:
            h = fused_ops.rms_norm(x, self.input_layernorm.weight, self.input_layernorm.variance_epsilon)
            if h is None:
                h = self.input_layernorm(x)
        else:
            r = fused_ops.rms_norm(delta, self.input_layernorm.weight, self.input_layernorm.variance_epsilon,
                                   residual=x)
            if r is None:
                x = x + delta
                h = self.input_layernorm(x)
            else:
                h, x = r
        attn, past_key_values = self.self_attn(
            packed_query_sequence=h,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode="und",
            selected_cache_indices=selected_cache_indices,
        )
        r = fused_ops.rms_norm(attn, self.post_attention_layernorm.weight,
                               self.post_attention_layernorm.variance_epsilon, residual=x)
        if r is None:
            x = x + attn
            h = self.post_attention_layernorm(x)
        else:
            h, x = r
        side = fused_ops.concurrent_side(h)
        if side is not None:
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                up = self.mlp.up_proj(h)
            gate = self.mlp.gate_proj(h)
            cur.wait_stream(side)
        else:
            gate = self.mlp.gate_proj(h)
            up = self.mlp.up_proj(h)
        act = fused_ops.silu_mul(gate, up) if isinstance(self.mlp.act_fn, nn.SiLU) else None
        if act is None:
            act = self.mlp.act_fn(gate) * up
        return x, self.mlp.down_proj(act), past_key_values


class Qwen2MoEDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PackedAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training or kwargs.get('ref_forward', False):
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)

        packed_sequence_new = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_und = self.mlp(packed_sequence[packed_und_token_indexes])
        packed_sequence_gen = self.mlp_moe_gen(packed_sequence[packed_gen_token_indexes])
        packed_sequence_new[packed_und_token_indexes] = packed_sequence_und
        packed_sequence_new[packed_gen_token_indexes] = packed_sequence_gen

        packed_sequence = residual + packed_sequence_new

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[Union[NaiveCache, NaiveCacheMultiSeq, StaticKVCache]] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        selected_cache_indices: Optional[List[int]] = None,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            selected_cache_indices=selected_cache_indices,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        if mode == "und":
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_query_sequence[packed_text_indexes])
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(packed_query_sequence[packed_vae_token_indexes])
            packed_query_sequence = packed_query_sequence_
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


Decoder_layer_dict = {
    "Qwen2DecoderLayer": Qwen2DecoderLayer,
    "Qwen2MoEDecoderLayer": Qwen2MoEDecoderLayer,
    "Qwen2MoTDecoderLayer": partial(Qwen2MoTDecoderLayer, attn_module=PackedAttentionMoT),
}


class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_moe = 'Mo' in config.layer_module

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        layer_module = Decoder_layer_dict[config.layer_module]
        self.layers = nn.ModuleList(
            [layer_module(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_moe:
            self.norm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, *args, **kwargs):
        if self.training or kwargs.get('ref_forward', False):
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
        ref_forward: Optional[bool] = False,
        packed_und_live_segments: Optional[tuple] = None,
        packed_gen_live_segments: Optional[tuple] = None,
    ) -> torch.Tensor:

        if self.config.freeze_und:
            packed_sequence[packed_und_token_indexes] = packed_sequence[packed_und_token_indexes].detach()

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            assert packed_und_token_indexes is not None
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
                ref_forward=ref_forward,
            )
            if packed_und_live_segments is not None:
                extra_inputs.update(packed_und_live_segments=packed_und_live_segments)
            if packed_gen_live_segments is not None:
                extra_inputs.update(packed_gen_live_segments=packed_gen_live_segments)

        if self.use_moe and self._split_ok(packed_sequence, attention_mask, packed_und_token_indexes,
                                           packed_gen_token_indexes):
            # split residual layout (fuse_train): the residual stream is kept as the und / gen expert rows
            # (exactly the rows every per-expert op of forward_train gathers), so the per-layer gathers and
            # zero-filled scatter buffers disappear.  Same values; see forward_train_split.
            hu = packed_sequence[packed_und_token_indexes]
            hg = packed_sequence[packed_gen_token_indexes]
            rope_und = (cos[packed_und_token_indexes], sin[packed_und_token_indexes])
            rope_gen = (cos[packed_gen_token_indexes], sin[packed_gen_token_indexes])
            for decoder_layer in self.layers:
                hu, hg = decoder_layer(
                    packed_sequence=None,
                    sample_lens=sample_lens,
                    attention_mask=attention_mask,
                    packed_position_embeddings=None,
                    split_sequence=(hu, hg),
                    split_rope=(rope_und, rope_gen),
                    **extra_inputs
                )
            packed_sequence_ = torch.zeros_like(packed_sequence)
            packed_sequence_[packed_und_token_indexes] = _rms_norm(self.norm, hu, plus_zero=True)
            packed_sequence_[packed_gen_token_indexes] = _rms_norm(self.norm_moe_gen, hg, plus_zero=True)
            return packed_sequence_

        for decoder_layer in self.layers:
            packed_sequence = decoder_layer(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                **extra_inputs
            )

        if self.use_moe:
            packed_sequence_ = torch.zeros_like(packed_sequence)
            packed_sequence_[packed_und_token_indexes] = _rms_norm(self.norm, packed_sequence[packed_und_token_indexes])
            if self.config.freeze_und:
                packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
            packed_sequence_[packed_gen_token_indexes] = _rms_norm(self.norm_moe_gen, packed_sequence[packed_gen_token_indexes])
            return packed_sequence_
        else:
            return self.norm(packed_sequence)

    def _split_ok(self, packed_sequence, attention_mask, und_idx, gen_idx):
        """The split residual layout is used when every per-expert op takes its fused, bitwise path."""
        from modeling.bagel import fused_train_ops as _fto
        if not (_SPLIT_RESIDUAL and _fto._ENABLED) or self.config.freeze_und or isinstance(attention_mask, list):
            return False
        inner = [getattr(l, '_checkpoint_wrapped_module', l) for l in self.layers]
        if not all(isinstance(l, Qwen2MoTDecoderLayer) and isinstance(l.self_attn, PackedAttentionMoT)
                   and isinstance(l.mlp.act_fn, torch.nn.SiLU) and isinstance(l.mlp_moe_gen.act_fn, torch.nn.SiLU)
                   for l in inner):
            return False
        n_u, n_g = und_idx.shape[0], gen_idx.shape[0]
        return (packed_sequence.dtype == torch.bfloat16 and packed_sequence.is_cuda and n_u >= 16 and n_g >= 16
                and packed_sequence.shape[0] == n_u + n_g and bool(self.config.qk_norm))

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[Union[NaiveCache, NaiveCacheMultiSeq]] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        selected_cache_indices: Optional[List[int]] = None,
        validated_mot_layout=None,
    ) -> BaseNavitOutputWithPast:

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)

        extra_inputs = {}
        mot_inference_layout = None
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == 'gen':
                assert packed_vae_token_indexes is not None
                assert packed_text_indexes is not None
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                )
                if (not torch.is_grad_enabled() and
                        self.config.layer_module == "Qwen2MoTDecoderLayer"):
                    # A call-scoped flow graph can supply the same validated
                    # partition while its index tensors remain immutable.
                    mot_inference_layout = validated_mot_layout
                    if mot_inference_layout is None:
                        mot_inference_layout = _mot_inference_layout(
                            packed_query_sequence, packed_text_indexes, packed_vae_token_indexes,
                        )
                    if mot_inference_layout is not None:
                        packed_query_sequence = packed_query_sequence[mot_inference_layout[1]]
                        extra_inputs["mot_inference_layout"] = mot_inference_layout

        fused = (mot_inference_layout is not None and self._fused_flow_eligible(
            packed_query_sequence, query_lens, past_key_values, update_past_key_values, is_causal,
            selected_cache_indices, mot_inference_layout, packed_query_position_embeddings))
        if fused:
            # final normalized hidden states in the original token order (fused path ends with the model norm)
            return BaseNavitOutputWithPast(
                packed_query_sequence=self._fused_flow_layers(
                    packed_query_sequence, query_lens, packed_query_position_embeddings, past_key_values,
                    mot_inference_layout, packed_text_indexes, packed_vae_token_indexes),
                past_key_values=past_key_values,
            )
        static_cache = past_key_values if fused_ops.ENABLED and isinstance(past_key_values, StaticKVCache) else None
        if static_cache is not None:
            # Per-forward scope for the layer-invariant cache index/length tensors (_static_kv_cache_attention).
            static_cache._forward_prep = {}
        und_fused = (fused_ops.ENABLED and self.use_moe and mode == "und" and mot_inference_layout is None
                     and self._all_mot_layers())
        und_pending = None
        try:
            for decoder_layer in (self.layers if und_fused else ()):
                # fused und path: each layer's last residual add is fused into the next RMSNorm
                actual_layer = getattr(decoder_layer, "_checkpoint_wrapped_module", decoder_layer)
                packed_query_sequence, und_pending, past_key_values = actual_layer.forward_inference_und_fused(
                    packed_query_sequence, und_pending,
                    query_lens=query_lens,
                    packed_query_position_embeddings=packed_query_position_embeddings,
                    packed_query_indexes=packed_query_indexes,
                    past_key_values=past_key_values,
                    key_values_lens=key_values_lens,
                    packed_key_value_indexes=packed_key_value_indexes,
                    update_past_key_values=update_past_key_values,
                    is_causal=is_causal,
                    selected_cache_indices=selected_cache_indices,
                )
            for decoder_layer in (() if und_fused else self.layers):
                actual_layer = getattr(decoder_layer, "_checkpoint_wrapped_module", decoder_layer)
                packed_query_sequence, past_key_values = actual_layer(
                    packed_query_sequence=packed_query_sequence,
                    query_lens=query_lens,
                    packed_query_position_embeddings=packed_query_position_embeddings,
                    packed_query_indexes=packed_query_indexes,
                    past_key_values=past_key_values,
                    key_values_lens=key_values_lens,
                    packed_key_value_indexes=packed_key_value_indexes,
                    update_past_key_values=update_past_key_values,
                    is_causal=is_causal,
                    selected_cache_indices=selected_cache_indices,
                    **extra_inputs,
                )
        finally:
            if static_cache is not None:
                static_cache._forward_prep = None

        if mot_inference_layout is not None:
            packed_query_sequence = packed_query_sequence[mot_inference_layout[2]]

        if self.use_moe:
            if mode == "und":
                normed = None
                if und_pending is not None:
                    normed = fused_ops.rms_norm(und_pending, self.norm.weight, self.norm.variance_epsilon,
                                                residual=packed_query_sequence)
                    if normed is None:
                        packed_query_sequence = packed_query_sequence + und_pending
                    else:
                        normed = normed[0]
                elif fused_ops.ENABLED:
                    normed = fused_ops.rms_norm(packed_query_sequence, self.norm.weight, self.norm.variance_epsilon)
                packed_query_sequence = normed if normed is not None else self.norm(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = self.norm(packed_query_sequence[packed_text_indexes])
                packed_query_sequence_[packed_vae_token_indexes] = self.norm_moe_gen(packed_query_sequence[packed_vae_token_indexes])
                packed_query_sequence = packed_query_sequence_
        else:
            packed_query_sequence = self.norm(packed_query_sequence)

        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )

    def _fused_flow_eligible(self, sequence, query_lens, past_key_values, update_past_key_values, is_causal,
                             selected_cache_indices, layout, position_embeddings):
        """Fused kernels only for the readonly static-KV image-flow backbone whose eager operations they
        reproduce bit for bit (expert-sorted MoT layout, bf16 weights/activations, qk-norm, SiLU MLP, and
        ATen reduction configurations covered by fused_flow)."""
        from . import fused_flow as ff
        if not ff.ENABLED:
            return False
        cfg = self.config
        reason = None
        if (torch.is_grad_enabled() or self.training or not isinstance(past_key_values, StaticKVCache)
                or update_past_key_values or is_causal or selected_cache_indices is not None):
            reason = 'mode'
        elif not (sequence.is_cuda and sequence.dtype == torch.bfloat16 and sequence.is_contiguous()
                  and sequence.dim() == 2):
            reason = 'sequence'
        elif (cfg.layer_module != "Qwen2MoTDecoderLayer" or not cfg.qk_norm or cfg.hidden_act != 'silu'
              or cfg.hidden_size // cfg.num_attention_heads != 128 or cfg.hidden_size % 512):
            reason = 'config'
        else:
            text_count = layout[0]
            gen_count = sequence.shape[0] - text_count
            nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
            cos, sin = position_embeddings
            if (text_count < 1 or gen_count < 1
                    or ff.hidden_bw(text_count, cfg.hidden_size) is None
                    or ff.hidden_bw(gen_count, cfg.hidden_size) is None
                    or not all(ff.head_ok(rows * heads) for rows in (text_count, gen_count) for heads in (nh, nkv))):
                reason = 'reduce_config'
            elif not (cos.dtype == torch.bfloat16 and sin.dtype == torch.bfloat16 and cos.is_contiguous()
                      and sin.is_contiguous() and tuple(cos.shape) == (sequence.shape[0], 128)):
                reason = 'rope'
            else:
                layer = getattr(self.layers[0], "_checkpoint_wrapped_module", self.layers[0])
                attn = layer.self_attn
                weights = (layer.input_layernorm.weight, layer.input_layernorm_moe_gen.weight,
                           layer.post_attention_layernorm.weight, attn.q_norm.weight, attn.k_norm.weight,
                           attn.q_proj.weight, attn.q_proj_moe_gen.weight, layer.mlp.gate_proj.weight,
                           self.norm.weight, self.norm_moe_gen.weight)
                if not all(w.dtype == torch.bfloat16 and w.is_contiguous() for w in weights):
                    reason = 'weights'
        if reason is not None:
            ff._note_fallback(reason)
            return False
        return True

    def _fused_flow_layers(self, h, query_lens, position_embeddings, past_key_values, layout,
                           packed_text_indexes, packed_vae_token_indexes):
        """All decoder layers of the eager MoT gen path (expert-sorted residual stream), with the norms, RoPE,
        concatenations, permutations, K/V merge and SwiGLU elementwise work fused; GEMMs and attention are the
        same calls on identically laid-out inputs. Returns the sorted residual stream after the last layer."""
        from . import fused_flow as ff
        cfg = self.config
        text_count, _, inverse = layout
        T = text_count
        H, nh, nkv, hd = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, 128
        n = h.shape[0]
        cos, sin = position_embeddings
        meta, packing = _static_kv_packing_meta(past_key_values, query_lens, None, h.device)
        total_kv = meta['sum_total']
        pending = None
        for li, decoder_layer in enumerate(self.layers):
            layer = getattr(decoder_layer, "_checkpoint_wrapped_module", decoder_layer)
            attn = layer.self_attn
            # input RMSNorm per expert (after the previous layer's MLP residual add), into one [n, H] buffer
            x = torch.empty_like(h)
            if pending is None:
                ff.rms_rows(h[:T], layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon, y_out=x[:T])
                ff.rms_rows(h[T:], layer.input_layernorm_moe_gen.weight,
                            layer.input_layernorm_moe_gen.variance_epsilon, y_out=x[T:])
            else:
                hn = torch.empty_like(h)
                ff.rms_rows(h[:T], layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon,
                            add=pending[0], h_out=hn[:T], y_out=x[:T])
                ff.rms_rows(h[T:], layer.input_layernorm_moe_gen.weight,
                            layer.input_layernorm_moe_gen.variance_epsilon, add=pending[1], h_out=hn[T:],
                            y_out=x[T:])
                h = hn
            xt, xg = x[:T], x[T:]
            q_t, q_g = attn.q_proj(xt), attn.q_proj_moe_gen(xg)
            k_t, k_g = attn.k_proj(xt), attn.k_proj_moe_gen(xg)
            v_t, v_g = attn.v_proj(xt), attn.v_proj_moe_gen(xg)
            q = torch.empty((n, nh, hd), dtype=torch.bfloat16, device=h.device)
            ff.head_norm_rope(q_t, q_g, attn.q_norm.weight, attn.q_norm_moe_gen.weight, cos, sin, inverse,
                              attn.q_norm.variance_epsilon, nh, q)
            merged_k = torch.empty((total_kv, nkv, hd), dtype=torch.bfloat16, device=h.device)
            merged_v = torch.empty((total_kv, nkv, hd), dtype=torch.bfloat16, device=h.device)
            if meta['sum_eff'] > 0:
                ff.past_copy(past_key_values.key_cache[attn.layer_idx], past_key_values.value_cache[attn.layer_idx],
                             packing['past_batch'], packing['past_pos'], packing['past_dest'], merged_k, merged_v)
            ff.head_norm_rope(k_t, k_g, attn.k_norm.weight, attn.k_norm_moe_gen.weight, cos, sin, inverse,
                              attn.k_norm.variance_epsilon, nkv, merged_k, dest=packing['new_dest'],
                              vt=v_t, vg=v_g, vout=merged_v)
            attn_out = flash_attn_varlen_func(
                q=q, k=merged_k, v=merged_v,
                cu_seqlens_q=packing['cu_seqlens_q'], cu_seqlens_k=packing['cu_seqlens_k'],
                max_seqlen_q=meta['max_q'], max_seqlen_k=meta['max_total'], causal=False,
            ).reshape(-1, H)
            o_t = attn.o_proj(attn_out[packed_text_indexes])
            o_g = attn.o_proj_moe_gen(attn_out[packed_vae_token_indexes])
            # residual add + post-attention RMSNorm per expert (separate outputs, as the eager slices' norms)
            hn = torch.empty_like(h)
            yt = torch.empty((T, H), dtype=torch.bfloat16, device=h.device)
            yg = torch.empty((n - T, H), dtype=torch.bfloat16, device=h.device)
            ff.rms_rows(h[:T], layer.post_attention_layernorm.weight, layer.post_attention_layernorm.variance_epsilon,
                        add=o_t, h_out=hn[:T], y_out=yt)
            ff.rms_rows(h[T:], layer.post_attention_layernorm_moe_gen.weight,
                        layer.post_attention_layernorm_moe_gen.variance_epsilon, add=o_g, h_out=hn[T:], y_out=yg)
            h = hn
            m_t = layer.mlp.down_proj(ff.silu_mul(layer.mlp.gate_proj(yt), layer.mlp.up_proj(yt)))
            m_g = layer.mlp_moe_gen.down_proj(ff.silu_mul(layer.mlp_moe_gen.gate_proj(yg), layer.mlp_moe_gen.up_proj(yg)))
            pending = (m_t, m_g)
        # last residual add + the model's final per-expert norm, stored at the original token positions: the
        # eager `seq[inverse]`, zeros_like, norm(seq[text_idx]) / norm_moe_gen(seq[vae_idx]) and index_puts
        # (seq[text_idx] are exactly the sorted rows [:T], seq[vae_idx] the rows [T:]).
        out = torch.empty_like(h)
        ff.rms_rows(h[:T], self.norm.weight, self.norm.variance_epsilon, add=pending[0], y_out=out,
                       write_h=False, out_index=packed_text_indexes)
        ff.rms_rows(h[T:], self.norm_moe_gen.weight, self.norm_moe_gen.variance_epsilon, add=pending[1],
                       y_out=out, write_h=False, out_index=packed_vae_token_indexes)
        ff.stats['fused_forwards'] += 1
        return out


    def _all_mot_layers(self):
        cached = getattr(self, "_all_mot_layers_cache", None)
        if cached is None or cached[0] != len(self.layers):
            ok = all(isinstance(getattr(l, "_checkpoint_wrapped_module", l), Qwen2MoTDecoderLayer) for l in self.layers)
            cached = (len(self.layers), ok)
            self._all_mot_layers_cache = cached
        return cached[1]


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def init_moe(self):
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                original_name = name.replace("_moe_gen", "")
                param.data.copy_(self.state_dict()[original_name].data)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self, *args, **kwargs):
        if self.training or kwargs.get('ref_forward', False):
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
        ref_forward: Optional[bool] = False,
        **keep_inputs,
    ) -> torch.Tensor:

        outputs = self.model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
            ref_forward=ref_forward,
            **keep_inputs,
        )
        return outputs

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[Union[NaiveCache, NaiveCacheMultiSeq]] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        selected_cache_indices: Optional[List[int]] = None,
        validated_mot_layout=None,
    ) -> BaseNavitOutputWithPast:
        
        self.model.forward
        outputs = self.model(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
            selected_cache_indices=selected_cache_indices,
            validated_mot_layout=validated_mot_layout,
        )

        return outputs
