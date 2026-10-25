# Eight-GPU judge and evidence

The current result uses the original bitwise comparison and paired timing rules, ported to eight H100 GPUs.
Baseline records were regenerated on that host; the old four-GPU records are never used as numerical targets
or timing denominators. The historical account is [judge-4gpu.md](judge-4gpu.md).

## Two comparison scopes

| Round | Baseline configuration | Candidate configuration | Minimum ratio |
|---|---|---|---:|
| `fusion_b_total2` | Upstream + original reward service | fusion_b + faster reward service | **2.083129×** |
| `fusion_b_r` | Upstream + faster reward service | fusion_b + faster reward service | **1.880298×** |

Both sides use eight ranks, eight shards, the same ZeRO-3/loading adapter, public model and GenEval prompts,
seeds 4396 / 42, and eight-prompt step-0 evaluation fixture. Reward model/weights and scoring semantics do not
change. The faster service limits CPU threads and loads colour crops in-process. The full speedup includes this
execution change. It was measured directly, not inferred from separate service and training gains.

## Bitwise check

Two upstream baseline runs first reproduced one another on this host. The candidate then ran deterministic
steps 0–13 (seven generate and seven refine steps). The following match on **all eight ranks**:

- raw text/image losses and collected training statistics;
- rollout token hashes/counts and generated-image hashes;
- GenEval scores, total/format rewards, text/image advantages;
- gradient norm before clipping and learning rates;
- final SHA-256 digests of local parameter shards and every Adam state tensor.

Both final rounds are `identical=true`, with no differences. Rank-0 final parameter/Adam digest is
`7ee0e76bcc53ddd6b5b8d905bb99a861302d2bba9829699f101a43bd99f4d41a`.
The candidate source is fingerprinted before and after execution. Current runtime file hashes were checked
against the immutable `fusion_b` snapshot and are included in [h100.json](validation/h100.json).

**Known wandb gap:** the initialization of wandb replaces the earlier logging hook. Rank-0 logged lists were
empty for both sides. Direct per-rank training observations remain checked, but values available only through
wandb, including step-0 evaluation scores, were not compared. This gap is not a numerical pass on those values.

## Timing

Each round runs baseline A → candidate → baseline B as independent launches on the same host, using production
settings. Each launch has 16 steps; steps 0–9 warm up, and windows 10–11, 12–13 and 14–15 each contain a generate
and a refine step. The optimizer post-step observer synchronizes CUDA; the slowest rank determines time.

For each window, `ratio = min(baseline_A, baseline_B) / candidate`. The reported speedup is `min(ratios)`.
Baseline spread is `max(abs(A-B) / min(A,B))`; the benefit threshold is `max(1.02, 1 + 2*spread)`.

| Scope | Baseline A (s/step) | Candidate (s/step) | Baseline B (s/step) | Ratios |
|---|---|---|---|---|
| Total | 30.7326 / 25.8916 / 27.9101 | 11.9883 / 11.8666 / 12.2042 | 31.8582 / 24.7197 / 28.0978 | 2.5636 / **2.0831** / 2.2869 |
| Training code only | 22.9874 / 23.2564 / 22.9266 | 12.1949 / 11.9496 / 11.9846 | 23.0628 / 22.4688 / 22.5615 | 1.8850 / **1.8803** / 1.8825 |

Total spread is 4.7405%, threshold 1.0948×. Training-code-only spread is 3.5053%, threshold 1.0701×.
Both pass. A first total attempt was interrupted by a reward-service file-read error; `fusion_b_total2` is the
completed repeat, not an average with that failed attempt. All displayed values derive from completed round JSON.

## Memory

Maxima over all eight ranks and the full 16-step launches, including warm-up. Baseline values take the larger
of A and B. PyTorch allocator counters cover the trainer; device peaks also include reward workers and other
CUDA overhead. Device peaks were sampled and can miss brief transients.

| Scope / implementation | Peak allocated GiB | Peak reserved GiB | Sampled peak device MiB |
|---|---:|---:|---:|
| Total / baseline | 67.19 | 72.20 | 80,165 |
| Total / fusion_b | 68.21 | 72.94 | 80,953 |
| Training-only / baseline | 67.20 | 72.38 | 80,351 |
| Training-only / fusion_b | 68.25 | 72.94 | 80,953 |

No final run reported OOM. There is little spare device memory. The old four-GPU 131.0 → 104.8 GiB reduction
belongs to a different sharding setup and is not a claim for this release.

## Evidence and limits

[validation/h100.json](validation/h100.json) contains aggregate timing windows, numerical verdicts, memory
maxima, source identity and a runtime-file hash manifest. Machine addresses, private paths, per-example data,
model weights, reward traces and raw profiler artifacts are excluded. The original full-tree digest includes
experiment tools and is not expected to equal a digest of this documentation-enriched release.

The full frozen judge, baseline tensor records and launch orchestration remain external. Published GPU
operator diagnostics are narrower checks, not replacements for that judge. The portable launch scripts have
not been rerun end to end. Current publication checks establish source and evidence consistency, not a new
benchmark run.

Numerical identity covers only 14 deterministic steps. Timing uses production settings whose numerical values
were not compared. Full training, save/resume, other datasets, hardware, dependency versions or GPU counts are
not certified. The owner accepts the completed bit-exact checks as this release's numerical acceptance
criterion. Additional long-training consistency or quality runs, including the previously proposed 200-step /
three-seed protocol, are not required and are not pending release work. This acceptance does not expand the
observed 14-step scope or constitute a measurement of time to quality or generalization.
