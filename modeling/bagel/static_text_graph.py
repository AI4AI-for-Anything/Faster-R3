"""Replay the unchanged, RNG-free static-KV text backbone within one rollout."""

import torch

from .qwen2_navit import _static_kv_host_meta


class StaticTextGraph:
    """Call-scoped decode graph; max_steps declares one-token-only cache writes.

    The bounded caller must be the sole cache writer and advance each selected
    sequence once per call. Omitting the budget keeps the original checks.
    """

    def __init__(self, language_model, cache, extra_inputs, max_steps=None):
        self.language_model = language_model
        self.cache = cache
        self.extra_inputs = extra_inputs
        self.entries = {}
        self.stream = torch.cuda.Stream(device=cache.seq_lens.device)
        self.eager_calls = 0
        self.captures = 0
        self.replays = 0
        self.calls = 0
        self.host_checks = 0
        self.bounded_replays = 0
        # The caller advances each active sequence by one per iteration. A
        # shrinking subset cannot exceed this bound, including the EOS step.
        # Other callers retain the original per-replay overflow check.
        self.max_steps = max_steps
        self.capacity_proven = False
        if max_steps is not None and max_steps >= 0:
            initial_max = max(cache.seq_lens.tolist(), default=0)
            self.capacity_proven = initial_max + max_steps <= cache.max_len

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
        self.calls += 1
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
        bounded_replay = (entry['graph'] is not None and self.capacity_proven
                          and self.calls <= self.max_steps
                          and entry['single_token_query'])
        if bounded_replay:
            self.bounded_replays += 1
        else:
            # Every capture still initializes its original metadata outside
            # capture. The graph reads live CUDA cache lengths on every replay.
            meta = _static_kv_host_meta(self.cache, entry['query_lens'], entry['indices'])
            self.host_checks += 1
            entry['single_token_query'] = meta['uniform_q'] and meta['q'][0] == 1
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
