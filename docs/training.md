# Training and reproduction

Current `main` packages `fusion_b` (`1367ad7`) from the eight-H100 experiment. All 37 Python files under
`modeling/` and `train/` match the measured snapshot exactly. The release adds portable scripts, documentation
and aggregate evidence; it does not include the full machine-specific judge. The old four-GPU setup is archived
in [training-4gpu.md](training-4gpu.md) and is reproducible from commit `44d1fac`.

## Environment

The measured host had eight NVIDIA H100 80 GB GPUs, 32 CPU cores and about 976 GiB host-memory capacity.
Use a Linux environment compatible with CUDA 12.8 PyTorch wheels (glibc 2.28 or newer):

| Component | Measured version |
|---|---|
| Python | 3.10 |
| PyTorch / torchvision | 2.9.1 / 0.24.1, CUDA 12.8 wheels |
| Triton | 3.5.1 |
| flash-attn | 2.5.8 |
| transformers / accelerate | 4.46.2 / 0.34.2 |
| numpy / safetensors / huggingface_hub | 1.26.4 / 0.6.2 / 0.36.0 |

```bash
python3.10 -m venv r3-env
source r3-env/bin/activate
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
# Build flash-attn against this torch installation; CUDA toolkit and a C++ compiler are required.
pip install -r requirements.txt --no-build-isolation
pip install psutil pytz
# Headless hosts used this instead of opencv_python to avoid a libGL dependency:
pip uninstall -y opencv-python
pip install opencv-python-headless==4.7.0.72
```

`requirements.txt` retains upstream pins. The observed Triton version is 3.5.1; confirm it before trusting
bitwise kernel results. Graph replay and fused reductions are version-sensitive; upgrading PyTorch or Triton
requires new operator and end-to-end checks.

## Model and data

Use the public [ByteDance-Seed/BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)
checkpoint, revision `5019f57d`, with `ema.safetensors`, `ae.safetensors`, `llm_config.json`, `vit_config.json`
and tokenizer files. The main checkpoint contains 14.61B parameters (EMA SHA-256 starts `0b41c438`).
`MODEL_PATH` points to this directory; the trainer uses `--finetune_from_ema True`.

Training uses this repository's public `data/rl_train/train_geneval.jsonl` (12,359 prompts), as in the upstream
GenEval tutorial. The measured step-0 evaluation uses the first **eight** lines of
`data/rl_test/test_geneval.jsonl`, one per rank. It is a launch fixture, not a generation-quality benchmark.
The launcher creates it under `LOG_DIR` unless `VAL_DATA_PATH` is supplied. A different validation set changes
the measured recipe. Weights, generated images and per-example result records are not included.

## Reward server

Set up [yifan123/reward-server](https://github.com/yifan123/reward-server) and its public GenEval detector and
CLIP weights as instructed upstream. The server uses a separate Python 3.10 environment with PyTorch 2.1.2+cu121,
torchvision 0.16.2, mmcv-full 1.7.2, mmdet 2.28.2, open_clip_torch 2.31.0, Flask 3.1.0 and gunicorn 23.0.0.
Validate the upstream example requests before running training.

```bash
cd /path/to/reward-server
export R3_REWARD_GPUS=8
export R3_REWARD_BIND=127.0.0.1:18085
gunicorn -c /path/to/Faster-R3/scripts/reward/gunicorn_geneval.py "app_geneval:create_app()"
```

Eight synchronous workers share the eight training GPUs. Do not use `--preload`: CUDA visibility and CPU thread
limits must be set before importing the model. The measured faster configuration sets `OMP_NUM_THREADS=4`,
`MKL_NUM_THREADS=4` and the colour-crop DataLoader's `num_workers=0`. It leaves detector, CLIP, crop ordering,
reward arithmetic, shared listener, timeout and GPU assignment unchanged. Per-GPU listener routing is disabled.
The loader override is scoped to each reward-server worker and assumes that upstream service's usage.

For the **total-speedup baseline**, use [gunicorn_geneval_baseline.py](../scripts/reward/gunicorn_geneval_baseline.py)
instead, with OMP/MKL variables unset (the experiment used the default CPU-thread configuration and four
colour-loader workers). For the **training-code-only baseline**, keep the faster configuration on both sides.
Restart the server for each independent run. Service configuration is part of the benchmark, not a model change.

## Launch

```bash
export MODEL_PATH=/path/to/BAGEL-7B-MoT
export LOG_DIR=/path/to/output
export REWARD_URL=http://127.0.0.1:18085
bash scripts/train_faster_r3_geneval.sh
```

The launcher runs eight processes and eight FSDP shards. Both baseline and candidate need the same
[h100_adapt.py](../scripts/h100_adapt.py): ZeRO-3 (`reshard_after_forward=True`), a direct GPU copy of the frozen
reference, and optional staggered initialization. Upstream ZeRO-2 does not fit this recipe on 80 GB GPUs.
The measured `R3_LOAD_CONCURRENCY=8` used roughly 450 GiB host memory while loading. Lower concurrency reduces
that transient host-memory demand but has not been separately benchmarked. Do not silently run with an
insufficient host-memory limit.

The recipe follows upstream `scripts/train_rl.sh`: group 16 / policy group 4, two rounds, learning rate 5e-6,
10 warm-up steps then constant LR, AdamW beta2 0.95 and eps 1e-8, gradient clipping 1.0, CE/MSE weights 1/2,
text/image KL weights 0.001, frozen ViT, 196/384 reasoning tokens, 16/21 denoising steps, CFG text/image 4.0/1.5,
SDE sampling and max latent size 64. Deviations are GenEval data/reward, the eight-prompt evaluation fixture,
offline wandb and explicit default seeds (`global_seed=4396`, `data_seed=42`). Two policy groups now process
two prompts' groups per step; the four-GPU experiment had one policy group.

To run upstream with **exactly the same adapter and launcher**, use a separate checkout:

```bash
git worktree add --detach ../R3-baseline upstream
export R3_SOURCE_TREE="$(cd ../R3-baseline && pwd)"
export LOG_DIR=/path/to/baseline-output
bash scripts/train_faster_r3_geneval.sh
unset R3_SOURCE_TREE
```

Select the matching reward configuration above. Setting fusion switches to zero does not restore upstream;
other optimizations remain active. The launcher defaults to the upstream 1,000-step duration and checkpointing
schedule. Only 14-step numerical runs and 16-step timing runs were measured; full training, save/resume and
full-set evaluation are not certified by these results. The judge stopped after an optimizer post-step hook;
setting `TOTAL_STEPS=16` alone does not reproduce its observation hooks or benchmark protocol.

This portable entry imports the selected trainer and applies the original shared adapter before `main()`.
It has been checked structurally, but has not itself completed an eight-GPU run. The original measurements used
the judge's entry point. This distinction also applies to the published reward configuration's environment-variable
wrappers; their measured defaults are unchanged.

## Changes and switches

| Area | Source | Change |
|---|---|---|
| Reward overlap | `train/online_rl.py` | Resolve reward futures after packing and reference/policy forwards, before loss/advantages are used |
| Packing, image reuse | `train/rollout_controller.py` | Read latent scalar once; cache shared transforms; copy identical edit CFG prefix |
| Shared forward inputs | `train/online_rl.py`, `modeling/bagel/bagel.py` | Reuse frozen ViT output and block mask between reference and policy |
| Dead CFG rows | `modeling/bagel/qwen2_navit.py` | Keep packing, skip expert GEMMs for inactive CFG-copy rows |
| Graph scheduling | `modeling/bagel/graph_capture.py`, `static_*graph.py`, `bagel.py` | Conditional allocator cleanup, pipelined text bookkeeping, earlier flow capture |
| Text fusion | `modeling/bagel/fused_ops.py` | Residual/RMSNorm, QK norm/RoPE, SiLU product |
| Flow fusion | `modeling/bagel/fused_flow.py` | Norm/reorder, merged KV writes, residual and MLP fusion |
| Training fusion | `modeling/bagel/fused_train_ops.py`, `qwen2_navit.py` | Forward/backward kernels and split residual streams |

All defaults below are the measured settings. They are diagnostic controls, not independently validated recipes.

| Environment variable | Default | Meaning |
|---|---:|---|
| `R3_FUSED_DECODE` | 1 | Fused text decode/prefill kernels |
| `R3_FUSED_DECODE_STREAMS` | 1 | Concurrent small text projections |
| `R3_FUSE_FLOW` | 1 | Fused image-flow path |
| `R3_FUSED_TRAIN_OPS` | 1 | Fused training operators |
| `R3_FUSE_TRAIN_SPLIT` | 1 | Split per-expert training residual stream |
| `R3_FUSED_SWIGLU_CONTRACT` | 1 | Measured SwiGLU backward contraction; do not change for the reported result |

The new kernels preserve the tested ATen reduction order and store boundaries, including subnormal behavior.
Unsupported shapes use eager fallbacks where implemented. Preserve graph diagnostic output: no replay means
that the graph path was not exercised. The measured memory table is in [judge.md](judge.md#memory).

## Verifying the release

```bash
python scripts/validate_release.py
python docs/performance-views/render_h100.py
```

The first command checks measured runtime source hashes, all recorded window arithmetic, eight-rank completion
and the final numerical verdicts. It needs no GPU and does not rerun training. The second regenerates the new
figures from the same JSON. Old four-GPU figures and `history.json` remain separately archived.

GPU diagnostics under `tools/fuse_decode`, `tools/fuse_flow` and `tools/fuse_train` are imported from the experiment;
only their checkpoint paths are made portable with `MODEL_PATH`. See [validation/README.md](validation/README.md)
for prerequisites and limitations. No captured activations or checkpoint tensors are shipped.
