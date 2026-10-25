# Faster-R3

[English](README.md) | [简体中文](README.zh-CN.md)

**2.08× faster R3 Tree-RL steps for BAGEL-7B-MoT on 8 NVIDIA H100 GPUs, optimized by AI agents.**

![Faster-R3: 2.08× total RL-step speedup](docs/performance-views/h100-benchmark.svg)

| Implementation | Time / RL step (s) ↓ | Speedup ↑ |
|---|---:|---:|
| Baseline | 24.72 | 1.00× |
| **Faster-R3** | **11.87** | **2.08×** |

The table and chart show window 2, which sets the minimum paired speedup; the baseline is the faster of runs A and B in that window.

**The 2.08× result includes GenEval reward-service execution improvements**; a separate paired measurement with the optimized reward configuration on both sides gives **1.88× for the training code alone**.

Measured in a short benchmark on **8 NVIDIA H100 GPUs (80 GB)** with FSDP2 ZeRO-3, BF16 compute, FP32 main weights and Adam state, and GenEval reward workers sharing the training GPUs. Group size is 16, with policy groups of 4; precision is unchanged and **no FP8** is used.

## Quick Start

Prepare the [training environment, public BAGEL checkpoint and GenEval reward server](docs/training.md). Fusion paths are enabled by default. Start the reward server in its own environment:

```bash
cd /path/to/reward-server
# Eight workers, four OpenMP/MKL threads each, in-process colour crops.
gunicorn -c /path/to/Faster-R3/scripts/reward/gunicorn_geneval.py "app_geneval:create_app()"
```

Then, from Faster-R3 in the training environment:

```bash
export MODEL_PATH=/path/to/BAGEL-7B-MoT
export LOG_DIR=/path/to/output
export REWARD_URL=http://127.0.0.1:18085
bash scripts/train_faster_r3_geneval.sh
```

The launcher uses eight ranks, eight shards, an eight-prompt step-0 evaluation fixture, and the shared [ZeRO-3/loading adapter](scripts/h100_adapt.py). Tested kernels depend on PyTorch 2.9.1, Triton 3.5.1 and flash-attn 2.5.8. The portable launcher is a packaging of the measured recipe; it has not itself been rerun end to end. See [baseline reproduction and memory requirements](docs/training.md#launch).

## Agent Optimization History

The optimizations combine static CUDA graphs, reward overlap, host work, and fused kernels for text, image flow and training. Each milestone below was measured against fresh baselines on the same eight-GPU host. The three fusion groups were tested individually on the same pre-fusion version, then combined in the final configuration.

![Eight-GPU optimization history with separate comparison scopes](docs/performance-views/h100-history.svg)

The earlier **1.4709× on four GPUs** is retained in the [historical report](docs/history-4gpu.md). It used different hardware, sharding and batch composition and is not a point on the eight-GPU curve.

## What changed

- **Reward service and overlap:** cap eight workers at four OpenMP/MKL threads each and load colour crops in-process. Submit rewards as before, then run packing and the reference/policy forwards while waiting; resolve rewards and advantages before computing the loss.
- **Host work and redundant computation:** read latent scalars once per latent rather than once per token; reuse shared image transforms, block masks and frozen ViT outputs. Keep the original packed layout while skipping expert GEMMs for CFG-copy rows whose outputs do not contribute to the loss.
- **Rollout scheduling:** pipeline text decoding with host bookkeeping, reuse the identical edit CFG image prefix, and capture static graphs without unconditional allocator flushing when enough memory is available.
- **Text fusion:** RMSNorm with optional residual add, Q/K norm plus RoPE, and SiLU × up; reuse layer-invariant indexes and overlap small independent projections.
- **Image-flow fusion:** normalize and restore expert order together, write K/V directly into merged caches, fuse residual/norm/MLP work, and capture each CFG branch after its first call.
- **Training fusion:** RMSNorm, SwiGLU and RoPE forward/backward kernels, plus a per-expert residual layout that avoids repeated gathers, zero-filled buffers and scatters.

![Optimization modules](docs/change-views/h100-module-map.svg)

The fused reductions reproduce the tested PyTorch reduction order, BF16 rounding boundaries and non-flushing division behavior. The production code under `modeling/` and `train/` is byte-for-byte identical to the measured snapshot; [source hashes and aggregate results](docs/validation/h100.json) accompany this release. See the [switches and source map](docs/training.md#changes-and-switches).

## Validation

**Numerical check.** Both comparison configurations matched the baseline bit for bit on every recorded item over **14 deterministic steps (0–13) on all eight ranks**: training losses/statistics, rewards, advantages, rollout token/image hashes, gradient norms, learning rates, and final parameter/Adam digests.

**Operator checks.** Additional diagnostics on one H100 passed 37 training-operator cases and 169 image-flow cases, supplementing the eight-rank training checks. See the [validation receipt](docs/validation/publication-check.json).

**Timing.** Baseline A → Faster-R3 → baseline B run as independent launches on the same host. Each launch runs 16 steps: 10 warm-up steps, then three two-step windows containing generation and refinement. An RL step includes rollout, reward and one optimizer update. The speedup is the smallest of the three ratios, using the faster baseline in each window.

| Window (steps) | Baseline (s/step) ↓ | Faster-R3 (s/step) ↓ | Paired speedup ↑ |
|---|---:|---:|---:|
| 1 (10–11) | 30.73 | 11.99 | 2.5636× |
| **2 (12–13)** | **24.72** | **11.87** | **2.0831×** |
| 3 (14–15) | 27.91 | 12.20 | 2.2869× |

The minimum is **2.0831×** for the full configuration and **1.8803×** in the separate training-code comparison. Ratios use unrounded timings; the table rounds times to two decimals.

![Paired timing for total and training-code-only results](docs/performance-views/h100-paired-timing.svg)

**Scope.** Numerical comparisons cover the deterministic steps above. Production-mode timing values and evaluation-only wandb metrics were not compared. The speedup measures short-run RL steps; full-training time and time to quality were not measured. Memory remains close to the device limit. See the [comparison records and acceptance criteria](docs/judge.md) and [memory table](docs/judge.md#memory).

## More AI4AI-for-Anything projects

[Faster-BAGEL](https://github.com/AI4AI-for-Anything/Faster-BAGEL) · [Faster-YOCO](https://github.com/AI4AI-for-Anything/Faster-YOCO) · [Faster-JiT](https://github.com/AI4AI-for-Anything/Faster-JiT) · [Faster-RAE](https://github.com/AI4AI-for-Anything/Faster-RAE) · [Faster-LLaVA](https://github.com/AI4AI-for-Anything/Faster-LLaVA) · [Faster-DMD](https://github.com/AI4AI-for-Anything/Faster-DMD) · [Faster-SelfForcing](https://github.com/AI4AI-for-Anything/Faster-SelfForcing)

## Upstream R3 documentation

Based on [sen-ye/R3](https://github.com/sen-ye/R3) at `761b0b0`, the code of *Understanding vs. Generation: Navigating Optimization Dilemma in Multimodal Models*. [Apache-2.0 license](LICENSE). The `upstream` branch holds the original code; installation, tutorial and citation remain in [README.upstream.md](README.upstream.md).
