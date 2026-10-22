"""Replay unchanged readonly-KV MoT backbones within one image generation call."""

import torch
import torch.distributed as dist

from .qwen2_navit import (
    StaticKVCache,
    _mot_inference_layout,
    _static_kv_packing_meta,
)


class StaticFlowGraph:
    """Metadata and weights must stay fixed until the owning image call returns.

    Only the packed query values change between denoising steps. Each cache and
    position/index signature gets its own entry; strong references keep those
    identities valid. Pools never survive a rollout reshard or parameter update.
    """

    def __init__(self, language_model):
        self.language_model = language_model
        self.entries = {}
        self.stream = None
        self.eager_calls = 0
        self.captures = 0
        self.replays = 0

    def __call__(self, **kwargs):
        sequence = kwargs['packed_query_sequence']
        cache = kwargs['past_key_values']
        model = self.language_model.model
        eligible = (
            sequence.is_cuda and sequence.is_contiguous()
            and not torch.is_grad_enabled() and not self.language_model.training
            and isinstance(cache, StaticKVCache)
            and kwargs.get('update_past_key_values') is False
            and kwargs.get('is_causal') is False
            and kwargs.get('selected_cache_indices') is None
            and kwargs.get('mode') == 'gen'
            and isinstance(kwargs.get('packed_text_indexes'), torch.Tensor)
            and isinstance(kwargs.get('packed_vae_token_indexes'), torch.Tensor)
            and model.config.layer_module == 'Qwen2MoTDecoderLayer'
            and 'dynamic' not in model.rotary_emb.rope_type
        )
        if not eligible:
            self.eager_calls += 1
            return self.language_model.forward_inference(**kwargs)

        key = (tuple(sequence.shape), tuple(sequence.stride()), sequence.dtype,
               sequence.device, tuple((name, id(value)) for name, value in
                                      sorted(kwargs.items())
                                      if name != 'packed_query_sequence'))
        entry = self.entries.get(key)
        if entry is None:
            # This original integer coverage check cannot run inside capture.
            # Its inputs are immutable for the owning generate_image* call.
            layout = _mot_inference_layout(
                sequence, kwargs['packed_text_indexes'], kwargs['packed_vae_token_indexes'],
            )
            if layout is None:
                self.eager_calls += 1
                return self.language_model.forward_inference(**kwargs)
            inputs = dict(kwargs)
            inputs['packed_query_sequence'] = sequence.clone()
            inputs['validated_mot_layout'] = layout
            entry = dict(inputs=inputs, graph=None)
            self.entries[key] = entry
            self.eager_calls += 1
            # Initialize kernels with the first real call, never extra warmup.
            return self.language_model.forward_inference(**inputs)

        inputs = entry['inputs']
        inputs['packed_query_sequence'].copy_(sequence)
        if entry['graph'] is None:
            # Re-establish this entry's original integer KV packing outside
            # capture. Every layer still reads its actual K/V tensor values.
            entry['kv_meta'], entry['packing'] = _static_kv_packing_meta(
                cache, inputs['query_lens'], None, sequence.device,
            )
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=sequence.device)
            graph = torch.cuda.CUDAGraph()
            current = torch.cuda.current_stream()
            self.stream.wait_stream(current)
            with torch.cuda.graph(graph, stream=self.stream):
                entry['output'] = self.language_model.forward_inference(**inputs)
            current.wait_stream(self.stream)
            entry['graph'] = graph
            self.captures += 1
        # Capture records operations; execute exactly once for this real call.
        entry['graph'].replay()
        self.replays += 1
        return entry['output']

    def report(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[static-flow-graph rank={rank}] eager={self.eager_calls} "
              f"captures={self.captures} replays={self.replays}", flush=True)
