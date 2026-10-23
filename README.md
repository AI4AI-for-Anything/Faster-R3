# Faster-R3

[English](README.md) | [简体中文](README.zh-CN.md)

**1.47× faster R3 Tree-RL training steps for BAGEL-7B-MoT on 4 H200-class GPUs, with bitwise-identical training records, optimized by AI agents.**

![Faster-R3: 1.47× RL-step speedup on four H200-class GPUs](docs/performance-views/01-benchmark.svg)

| Implementation | Seconds per RL step, windows 1 / 2 / 3 ↓ | Speedup ↑ |
|---|---:|---:|
| Baseline (faster of the two baseline runs) | 23.40 / 23.59 / 23.76 | 1.00× |
| **Faster-R3** | **15.36 / 16.04 / 16.03** | **1.47×** |

Measured on **4 H200-class GPUs** (Hopper, about 140 GiB each) with the upstream R3 recipe for the GenEval reward: BAGEL-7B-MoT (14.61B parameters) trained by Tree-RL (GRPO on the text reasoning, Mix-GRPO on the image denoising), alternating a generate step and a reflect-and-refine step, group size 16 over a policy group of 4 GPUs, FSDP2 with the upstream sharding, BF16 compute, the real GenEval reward server on the same GPUs, and **no FP8 or other precision change**. An RL step is one rollout, its rewards and one optimizer update. Baseline, Faster-R3 and baseline again run as independent launches on one host; after 10 warm-up steps, each run is timed over three windows of two steps (one generate and one refine step). The speedup is the smallest of the three window ratios (1.5237, 1.4709, 1.4827): **1.4709×**. It is a per-step benchmark of 16-step runs, not a measurement of a full training run.

Before it was timed, the same tree trained **14 steps from the released checkpoint** and every recorded value on all four ranks was **bitwise identical** to the upstream code: losses, rewards, advantages, rollout tokens and images, gradient norms, learning rates, and the digests of every parameter shard and Adam state after the last step. In the timing runs, peak allocated memory per GPU fell from 131.0 to 104.8 GiB (reserved 133.7 vs 130.7 GiB).

## Quick Start

Prepare the [environment, the BAGEL-7B-MoT checkpoint and the GenEval reward server](docs/training.md). The changes need no switches: this tree runs the upstream trainer with them always on. The launcher runs the measured 4-GPU GenEval recipe:

```bash
export MODEL_PATH=/path/to/BAGEL-7B-MoT
export LOG_DIR=/path/to/output
export REWARD_URL=http://127.0.0.1:18085   # GenEval reward server, see docs/training.md#reward-server
bash scripts/train_faster_r3_geneval.sh
```

The arguments are those of upstream `scripts/train_rl.sh` with the [recorded deviations](docs/training.md#launch) (4 GPUs, `--num_shard 4`, the GenEval data and reward of the upstream tutorial, a 4-prompt validation file). For the baseline, run the same command on the branch `upstream`. Tested only with **PyTorch 2.9.1 (CUDA 12.8 wheels), Triton 3.5.1, flash-attn 2.5.8 and transformers 4.46.2** (the upstream pins) on 4 GPUs.

## Agent Optimization History

Each point is one round of the same frozen bit-exact bench on the same host, 4 H200-class GPUs.

![R3 optimization history, from 1.06× to 1.47×](docs/performance-views/02-history.svg)

## What changed

An R3 step spends much of its time in the rollout: text reasoning decoded token by token, and images denoised with up to three classifier-free-guidance (CFG) branches per denoising step. Most rounds removed host-side overhead there: synchronizations on scalar reads from the GPU and the launch of many small kernels.

- **Static KV-cache metadata:** the per-sequence lengths of the static KV cache are copied to the host once per forward instead of with several `.item()` synchronizations in every layer; the integer packing indexes are computed once and reused by all layers and denoising steps while the lengths are unchanged; one check per decoding call proves that the cache cannot overflow, so the per-token host checks are skipped.
- **Text decoding:** the decoder backbone for each active batch shape is captured as a CUDA graph after its first eager call and replayed for the later tokens; the LM head and the top-k sampler are replayed the same way, with the generator registered to the graph so it consumes the same Philox offsets as the eager code. The token bookkeeping reads each batch from the GPU once instead of one scalar at a time.
- **Image flow (denoising):** timesteps are expanded on the GPU instead of being rebuilt from host scalars; during no-grad generation the mixture-of-transformers layers keep text and image tokens in expert order instead of gathering and scattering them in every sub-module; the read-only LM backbone of each CFG branch is captured as a CUDA graph within each image-generation call and replayed for the later denoising steps, and the independent branch replays run concurrently on separate streams, joined before the unchanged CFG arithmetic.
- **Training-step memory:** the full weights of the frozen KL-reference model are resharded right after its forward, before the policy forward and backward. This lowered peak allocation by about 27 GiB and made room for the expert-order layout.

![Optimization map: static KV metadata, text decoding, image flow, training-step memory](docs/change-views/module-map.svg)

See the [change table](docs/training.md#changes). The changes were made to leave every number unchanged (same arithmetic, reduction order and random-number consumption), and the bench confirmed it for the 14 checked steps. There are no switches; the CUDA-graph and expert-order paths fall back to the upstream code when their preconditions do not hold.

## Validation

**Bit-exact check.** Every round was judged by a [frozen bit-exact bench](docs/judge.md) built before the first optimization. The candidate trains steps 0–13 from the released BAGEL-7B-MoT checkpoint on the 4 GPUs (seed 4396, PyTorch deterministic algorithms on), and every recorded value must equal the record of the unmodified upstream code, which had first reproduced itself bitwise on the same host:

- per step and rank: the raw text and image losses, the additional statistics, the gradient norm before clipping, the learning rates, a hash of the rollout token ids and of every generated image, the GenEval rewards (scores, totals, format), and the text and image advantages;
- after the last step: the SHA-256 of every local parameter shard and every Adam state tensor.

Faster-R3 (`e71a7bd`) was identical in all of them; the final rank-0 digest equals the baseline's (`c2b02b63…`). Only bitwise-identical trees were timed. Unchanged upstream code timed as a candidate scored 0.9906× against a threshold of 1.0432, as expected for no change.

**Limits.** The check covers 14 steps in deterministic mode; the timing runs use the production settings, whose values were not compared. The bench was meant to compare rank 0's `wandb.log` values too, but `wandb.init` replaced its hook, so those lists were empty for both sides; the training values listed above were still compared directly, and the supervisor of the run waived that item ([details](docs/judge.md#known-gap-wandb-values)). The evaluation pass at step 0 is logged only through wandb, so its rewards were not compared. No long training run, other dataset (GenEval++, TIIF) or 8-GPU setup was tested.

![Paired timing of the unchanged control and of the final version](docs/performance-views/03-paired-timing.svg)

## Why it stopped at 1.47×

The goal was 2×; it was not reached. The owner ended the bit-exact work after round 17 and split the remaining work into separate rollout-only and train-only efforts, so **no ceiling is claimed**. Memory limited several directions: the baseline already allocates 131 GiB of about 140 GiB per GPU, and the GenEval workers on the same GPUs use about 3.9 GiB more. Five candidates ran out of memory: three that kept more weights or activations resident (rounds 6, 6b and 13), one that freed gradients earlier (round 7), and the expert-order layout before round 9 had freed memory (round 8). Round 11 was bitwise identical but was timed while the two baseline runs differed by 27%, which raised its threshold to 1.54×; it was reverted. Round 14 was accepted at 1.45×, below round 12, but allocated about 10.7 GiB more, so it was left out. The last complete profile is of round 16, before round 17.

See the [rounds that were not adopted](docs/training.md#not-adopted) and the [judge record](docs/judge.md).

## More AI4AI-for-Anything projects

[Faster-BAGEL](https://github.com/AI4AI-for-Anything/Faster-BAGEL) · [Faster-YOCO](https://github.com/AI4AI-for-Anything/Faster-YOCO) · [Faster-JiT](https://github.com/AI4AI-for-Anything/Faster-JiT) · [Faster-RAE](https://github.com/AI4AI-for-Anything/Faster-RAE) · [Faster-LLaVA](https://github.com/AI4AI-for-Anything/Faster-LLaVA) · [Faster-DMD](https://github.com/AI4AI-for-Anything/Faster-DMD) · [Faster-SelfForcing](https://github.com/AI4AI-for-Anything/Faster-SelfForcing)

## Upstream R3 documentation

Based on [sen-ye/R3](https://github.com/sen-ye/R3) at `761b0b0`, the code of [Understanding vs. Generation: Navigating Optimization Dilemma in Multimodal Models](https://arxiv.org/abs/2602.15772). [Apache-2.0 license](LICENSE). The branch `upstream` holds the unmodified code. The original README is kept in [README.upstream.md](README.upstream.md) and below.

<details>
<summary>Original R3 README: overview, installation, usage, citation</summary>

# Understanding VS. Generation: Navigating Optimization Dilemma in Multimodal Models

This repository contains the code for the paper [Understanding VS. Generation: Navigating Optimization Dilemma in Multimodal Models](https://arxiv.org/abs/2602.15772)

> **Understanding VS. Generation: Navigating Optimization Dilemma in Multimodal Models**   
> Sen Ye, Mengde Xu, Shuyang Gu, Di He, Liwei Wang, Han Hu   
> PKU, Tencent 


## Overview
We propose the **Reason-Reflect-Refine (R3)** framework. R3 re-frames the single-step generation process into a multi-step process of "generate-understand-regenerate". We optimize R3 with **Tree-RL** strategy, which simultaneously optimizes model's generation and understanding capabilities.

This repository contains the code for training Bagel model with R3 framework. We utilize **Tree-RL** to split the rollout process into multiple stages. We adopt GRPO to optimize the text reasoning process and Mix-GRPO to optimize the diffusion process.
![Framework](./figs/R3_infer.png)



## Installation
```bash
conda create -n bagel_r3 python=3.10 -y
conda activate bagel_r3
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

Download the model checkpoint from [Hugging Face](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) and place it in the `checkpoints` directory.


## Usage
We provide a doc in tutorial.md detailing how to train R3 on various datasets.

## Citation
If you find this work useful, please consider citing:
```bibtex
@misc{ye2026understandingvsgenerationnavigating,
      title={Understanding vs. Generation: Navigating Optimization Dilemma in Multimodal Models}, 
      author={Sen Ye and Mengde Xu and Shuyang Gu and Di He and Liwei Wang and Han Hu},
      year={2026},
      eprint={2602.15772},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.15772}, 
}
```

</details>
