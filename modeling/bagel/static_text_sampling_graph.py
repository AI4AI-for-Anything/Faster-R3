"""Replay the original text head and sampler for one unchanged active shape."""

import torch
import torch.nn.functional as F

from . import graph_capture


class StaticTextSamplingGraph:
    def __init__(self, model, do_sample, temperature, topk, top_p, generator):
        self.model = model
        self.do_sample = do_sample
        self.temperature = temperature
        self.topk = topk
        self.top_p = top_p
        self.generator = generator
        self.entries = {}
        self.stream = torch.cuda.Stream(device=model.device)
        self.eager_calls = 0
        self.captures = 0
        self.replays = 0

    def _forward(self, hidden):
        pred_logits = self.model.language_model.lm_head(hidden)
        if self.do_sample:
            next_tokens, _ = self.model._sample_next_token(
                pred_logits,
                temperature=self.temperature,
                topk=self.topk,
                top_p=self.top_p,
                generator=self.generator,
            )
        else:
            next_tokens = torch.argmax(pred_logits, dim=-1)
        raw_probs = F.softmax(pred_logits.float(), dim=-1)
        next_tokens_probs = raw_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
        return next_tokens, next_tokens_probs

    def __call__(self, hidden):
        if not hidden.is_contiguous():
            self.eager_calls += 1
            return self._forward(hidden)
        # The graph and all parameter pointers live only during this text call,
        # before FSDP resharding or an optimizer update. Preserve active shapes.
        key = (tuple(hidden.shape), tuple(hidden.stride()), hidden.dtype)
        entry = self.entries.get(key)
        if entry is None:
            entry = dict(hidden=hidden.clone(), graph=None)
            self.entries[key] = entry
            self.eager_calls += 1
            # Initialize using the first real token: never draw extra samples.
            return self._forward(entry['hidden'])

        entry['hidden'].copy_(hidden)
        if entry['graph'] is None:
            graph = torch.cuda.CUDAGraph()
            if self.generator is not None:
                graph.register_generator_state(self.generator)
            current = torch.cuda.current_stream()
            self.stream.wait_stream(current)
            entry['output'] = graph_capture.capture(graph, self.stream, lambda: self._forward(entry['hidden']))
            current.wait_stream(self.stream)
            entry['graph'] = graph
            self.captures += 1
        # Graph-safe Philox replay advances the same generator by the sum of
        # the original RNG-op offsets. Capture itself does not sample a token.
        entry['graph'].replay()
        self.replays += 1
        return entry['output']
