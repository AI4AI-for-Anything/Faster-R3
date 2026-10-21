"""Replay the unchanged, RNG-free static-KV text backbone within one rollout."""

import torch

from .qwen2_navit import _static_kv_host_meta


class StaticTextGraph:
    def __init__(self, language_model, cache, extra_inputs):
        self.language_model = language_model
        self.cache = cache
        self.extra_inputs = extra_inputs
        self.entries = {}
        self.stream = torch.cuda.Stream(device=cache.seq_lens.device)
        self.eager_calls = 0
        self.captures = 0
        self.replays = 0

    def _forward(self, entry):
        return self.language_model.forward_inference(
            packed_query_sequence=entry['embedding'],
            query_lens=entry['query_lens'],
            packed_query_position_ids=entry['positions'],
            packed_query_indexes=None,
            past_key_values=self.cache,
            key_values_lens=None,
            packed_key_value_indexes=None,
            update_past_key_values=True,
            is_causal=True,
            selected_cache_indices=entry['indices'],
            **self.extra_inputs,
        )

    def __call__(self, embedding, query_lens, positions, indices):
        # A shrinking active batch needs its own original kernel shapes. Buffers
        # and graph pools live only until this generate_text_batch call returns,
        # before FSDP resharding or any parameter update.
        key = (tuple(embedding.shape), embedding.dtype)
        entry = self.entries.get(key)
        if entry is None:
            entry = dict(embedding=embedding.clone(), query_lens=query_lens.clone(),
                         positions=positions.clone(), indices=indices.clone(), graph=None)
            self.entries[key] = entry
            self.eager_calls += 1
            # The first real token initializes library kernels; no extra forward
            # or cache update is performed for warmup.
            return self._forward(entry)

        entry['embedding'].copy_(embedding)
        entry['query_lens'].copy_(query_lens)
        entry['positions'].copy_(positions)
        entry['indices'].copy_(indices)
        # Refresh the original overflow check outside capture, including on
        # replays. All layers use these same query/index objects during capture.
        meta = _static_kv_host_meta(self.cache, entry['query_lens'], entry['indices'])
        if meta['max_total'] > self.cache.max_len:
            raise RuntimeError(
                f"StaticKVCache overflow: need {meta['max_total']} tokens, "
                f"but max_len={self.cache.max_len}"
            )
        if entry['graph'] is None:
            graph = torch.cuda.CUDAGraph()
            current = torch.cuda.current_stream()
            self.stream.wait_stream(current)
            with torch.cuda.graph(graph, stream=self.stream):
                entry['output'] = self._forward(entry)
            current.wait_stream(self.stream)
            entry['graph'] = graph
            self.captures += 1
        # Capture only records the operations. Execute exactly once for this
        # token; sampling and seq_lens advancement stay in the original caller.
        entry['graph'].replay()
        self.replays += 1
        return entry['output']
