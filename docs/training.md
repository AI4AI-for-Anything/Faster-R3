# Training and reproduction

Faster-R3 is upstream [sen-ye/R3](https://github.com/sen-ye/R3) at `761b0b0` (branch `upstream`) plus the
bit-exact rounds on branch `main`. There are no switches: the changed code paths are always on. The code under
`modeling/`, `train/` and `data/` at the head of `main` is that of commit `e71a7bd` (round 17), the tree that was
measured; later commits only add documentation and scripts.

## Environment

The checks ran on 4 H200-class GPUs (Hopper, sm_90, 143,771 MiB each) in one host, driver 570.148.08, with the
upstream pins from `requirements.txt`:

| Package | Version |
|---|---|
| Python | 3.10.19 |
| PyTorch / torchvision | 2.9.1 / 0.24.1 (PyPI wheels, CUDA 12.8) |
| Triton | 3.5.1 |
| flash-attn | 2.5.8 (built with `--no-build-isolation`, as upstream says) |
| transformers / accelerate | 4.46.2 / 0.34.2 |
| numpy / safetensors / huggingface_hub | 1.26.4 / 0.6.2 / 0.36.0 |
| wandb | 0.30.0 (runs were offline) |

```bash
python3.10 -m venv r3-env && source r3-env/bin/activate
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

The CUDA graphs need a GPU build of PyTorch with `torch.cuda.graph`, `CUDAGraph.register_generator_state` and FSDP2
(`torch.distributed.fsdp.fully_shard`); other PyTorch versions were not tested.

## Model and data

- **Model:** [ByteDance-Seed/BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) revision
  `5019f57d`: `ema.safetensors` (1,223 BF16 tensors, 14.61B elements, SHA-256 `0b41c438…`), `ae.safetensors`,
  `llm_config.json`, `vit_config.json` and the tokenizer files. `--resume_from` points at this directory with
  `--finetune_from_ema True`, as in the upstream script.
- **Training prompts:** `data/rl_train/train_geneval.jsonl` from this repository (12,359 prompts), the GenEval setup
  of the upstream [tutorial](../tutorial.md).
- **Validation prompts:** the first 4 lines of `data/rl_test/test_geneval.jsonl`. With `--eval_freq 50` the trainer
  evaluates at step 0, inside the warm-up of every timed run; the full 2,212-prompt test file would have made
  each step-0 evaluation far longer than the timed steps. This file is not a quality evaluation.

No checkpoints, generated images, rewards or logs are included in this repository.

## Reward server

The GenEval reward comes from the Flow-GRPO reward server ([yifan123/reward-server](https://github.com/yifan123/reward-server)),
as the upstream tutorial instructs. It ran in a separate Python 3.10 environment (PyTorch 2.1.2+cu121, torchvision
0.16.2, mmcv-full 1.7.2, mmdet 2.28.2, open_clip_torch 2.31.0, Flask 3.1.0, gunicorn 23.0.0) with its official
GenEval detector and CLIP weights, and returned the README's reference result on its two example images before
any run. Four synchronous workers, one on each training GPU, listen on `127.0.0.1:18085`
([`scripts/reward/gunicorn_geneval.py`](../scripts/reward/gunicorn_geneval.py)):

```bash
cd /path/to/reward-server          # set up as in its README
gunicorn -c /path/to/Faster-R3/scripts/reward/gunicorn_geneval.py "app_geneval:create_app()"
```

Each worker held about 3.9 GiB on its GPU next to the trainer. The trainer sends every rank's images to one URL
(`--reward_server_urls http://127.0.0.1:18085 --reward_server_port -1`).

## Launch

```bash
export MODEL_PATH=/path/to/BAGEL-7B-MoT
export LOG_DIR=/path/to/output
export REWARD_URL=http://127.0.0.1:18085
bash scripts/train_faster_r3_geneval.sh    # NPROC_PER_NODE=4 by default
```

[`scripts/train_faster_r3_geneval.sh`](../scripts/train_faster_r3_geneval.sh) runs `torchrun ... train/online_rl.py`
with the arguments of the measured runs. They are upstream `scripts/train_rl.sh` (learning rate 5e-6, constant after
10 warm-up steps, AdamW with β₂ 0.95 and eps 1e-8, gradient clipping at 1.0, CE weight 1.0, MSE weight 2.0, KL
weights 0.001 for text and image, ViT frozen, `--rounds 2`, `--group_size 16`, `--policy_group_size 4`, think mode,
196 / 384 reasoning tokens for generate / refine, 16 / 21 denoising steps, CFG text 4.0 and image 1.5, SDE sampling
in both stages, `--max_latent_size 64`, ...) except for these recorded deviations:

1. 4 processes on one node and `--num_shard 4` instead of 8 and 8. The per-GPU load is unchanged
   (`group_size / policy_group_size` = 4 rollouts per GPU); with 4 GPUs there is one policy group, so each step
   trains on one prompt's group of 16 instead of two;
2. the GenEval data and reward of the upstream tutorial (`--dataset_name geneval --reward_fn geneval`) instead of
   the script's GenEval++ default, which needs a Qwen2.5-VL-72B reward model;
3. the 4-prompt validation file above;
4. `--wandb_offline True` instead of `0`, `--global_seed 4396 --data_seed 42` written out (the trainer's defaults),
   and a per-run `--log_dir`.

The script prints `[static-text-graph ...]`, `[static-text-sampling-graph ...]` and `[static-flow-graph ...]` lines
with eager, capture and replay counts after each text or image call; a run that never replays did not take the
graph paths. The launcher itself was not run as written: the measured runs passed the same arguments to
`train.online_rl.main` through the bench's entry point (see [judge.md](judge.md)).

**Checkpoints.** The checked runs ran 14 or 16 steps, so no checkpoint was saved (`--save_every 50`). Saving and
resuming with this tree were not tested.

## Changes

Rounds refer to the [history](#optimization-history). Every round's tree passed the bit-exact check on its own, and
the final tree (round 17) contains every round marked "main".

| Round | Commit | What it does | Files |
|---|---|---|---|
| R01 | `9a7c69b` | The static-KV attention read the per-sequence cache lengths with several `.item()` calls in every layer. One device-to-host copy per forward now provides them, cached on the cache object for that forward's index tensors; every writer of the lengths invalidates the copy (including the two in the rollout controller). | `modeling/bagel/qwen2_navit.py`, `train/rollout_controller.py` |
| R02 | `0860070` | Text decoding with the static KV cache: the RNG-free backbone forward for each active batch shape runs eagerly once, is then captured as a CUDA graph and replayed for the later tokens of that call. Not used for dynamic RoPE, training mode or non-static caches. Graphs and their pools live only for one `generate_text_batch` call, before any resharding or parameter update. | `modeling/bagel/bagel.py`, `modeling/bagel/static_text_graph.py` (new) |
| R03 | `35cd537` | The integer packing indexes that merge cached and new K/V (gather positions, destinations, `cu_seqlens`) are built once while the lengths are unchanged and reused by every layer and denoising step. K/V values are always read fresh. | `modeling/bagel/qwen2_navit.py` |
| R04 | `8f86d88` | Per-token timesteps of each denoising step are `t.expand(n).clone()` on the GPU instead of `torch.tensor([t] * n)`, which read the CUDA scalar on the host `n` times. Same values. | `modeling/bagel/bagel.py` |
| R05 | `3d9841b` | The per-token bookkeeping of text decoding reads the active indices, tokens and probabilities with one `.tolist()` each per step instead of one `.item()` per sequence. Sampling and cache updates are unchanged. | `modeling/bagel/bagel.py` |
| R09 | `7c145be` | The frozen KL-reference model's FSDP modules are resharded right after its forward, so its full weights are not held during the policy forward and backward. Observed peak allocation dropped by about 27 GiB. | `train/online_rl.py` |
| R10 | `1fc390c` | No-grad `gen`-mode forwards of the MoT layers keep the residual stream in expert order (text tokens, then image tokens) when the index tensors form a complete, disjoint partition, so each expert's norms, projections and MLP read a contiguous slice; RoPE, attention and cache packing still see the original order, and the inverse permutation restores it after the last layer. The same BF16 store boundaries are kept. Training forwards are unchanged. | `modeling/bagel/qwen2_navit.py` |
| R12 | `fc169bd` | The LM head, top-k / top-p sampler and token probability of text decoding are captured and replayed per active shape. The CUDA generator is registered with the graph (`register_generator_state`), so each replay advances it by the same Philox offsets as the eager ops; the first call of each shape runs eagerly. | `modeling/bagel/bagel.py`, `modeling/bagel/static_text_sampling_graph.py` (new) |
| R15 | `7e91f38` | At the start of a decoding call, `max(seq_lens) + max_steps <= max_len` proves that one-token steps cannot overflow the cache, so replays within that bound skip the per-token host check. Other callers keep the original checks. | `modeling/bagel/bagel.py`, `modeling/bagel/static_text_graph.py` |
| R16 | `d18c6ac` | Image denoising: each read-only LM backbone call (one per CFG branch and per cache and index signature) runs eagerly once, is captured on its second call and replayed for the remaining steps of that image call. The expert-order partition is validated outside the capture and passed in. Measured as round "r16a" after the first launch of the same tree was refused by the bench's idle-GPU admission. | `modeling/bagel/bagel.py`, `modeling/bagel/qwen2_navit.py`, `modeling/bagel/static_flow_graph.py` (new) |
| R17 | `e71a7bd` | When all branches of a denoising step are already captured, their replays are launched on their own streams (each with a private pool and library workspace) and joined before the `llm2vae` projection and the CFG arithmetic, which run in the original order. | `modeling/bagel/bagel.py`, `modeling/bagel/static_flow_graph.py` |

## Not adopted

These commits are in the history of `main` together with their reverts; their code is not in the final tree.

| Round | Commit | Idea | Outcome |
|---|---|---|---|
| R06 | `f6be523` | Keep the gathered policy weights from the rollout through the training forward | Bitwise identical, then CUDA out of memory in step 9's backward of the timing run |
| R06b | `85a9d90` | Keep only the decoder weights across the rollout boundary | CUDA out of memory in step 6's rollout all-gather |
| R07 | `de43b14` | Release the previous step's gradients before the rollout allocations | CUDA out of memory in step 3's backward |
| R08 | `be03e3c` | The R10 expert-order layout, before R09 freed memory | Bitwise identical, then out of memory in the timing run; redone as R10 |
| R11 | `05d77a0` | Overlap the host bookkeeping of text decoding with GPU work | Bitwise identical, 1.4612×, but the two baseline runs differed by 27% and raised the threshold to 1.5415×; rejected |
| R13 | `40cffe8` | Keep (not recompute) the activations of the last eight policy MoT layers | CUDA out of memory in step 1's policy forward on all ranks |
| R14 | `b8c4a67` | Keep the policy weights gathered from the rollout through the backward | Accepted at 1.4485× (threshold 1.0543), but allocated about 10.7 GiB more than R12 without a better score; left out |

`7bb65c2` restores the R05 tree after R06 and R06b. Every "out of memory" here is a failure of the tested scope on
these GPUs, not a proof that smaller variants cannot fit.

## Optimization history

Speedups from the bench's round records ([history.json](performance-views/history.json)). Each round compares one
tree against two freshly timed baseline runs; rounds are not paired with each other.

| Round | Commit | Speedup | Threshold | Reference spread | Status |
|---|---|---:|---:|---:|---|
| R00 | — | 0.9906× | 1.0432 | 2.2% | control: unmodified upstream timed as the candidate |
| R01 | `9a7c69b` | 1.0553× | 1.0233 | 1.2% | in main |
| R02 | `0860070` | 1.3008× | 1.0536 | 2.7% | in main |
| R03 | `35cd537` | 1.3589× | 1.0200 | 0.6% | in main |
| R04 | `8f86d88` | 1.3865× | 1.0407 | 2.0% | in main |
| R05 | `3d9841b` | 1.4095× | 1.0692 | 3.5% | in main |
| R09 | `7c145be` | 1.4146× | 1.0200 | 0.7% | in main |
| R10 | `1fc390c` | 1.4203× | 1.0519 | 2.6% | in main |
| R11 | `05d77a0` | 1.4612× | 1.5415 | 27.1% | rejected, reverted |
| R12 | `fc169bd` | 1.4609× | 1.0250 | 1.2% | in main |
| R14 | `b8c4a67` | 1.4485× | 1.0543 | 2.7% | accepted, left out |
| R15 | `7e91f38` | 1.4343× | 1.0361 | 1.8% | in main |
| R16 | `d18c6ac` | 1.4503× | 1.0200 | 1.0% | in main |
| R17 | `e71a7bd` | **1.4709×** | 1.0476 | 2.4% | in main, final |

Every timed row was bitwise identical over the 14 checked steps, with the same final rank-0 digest as the baseline.
Rounds 2, 3, 8 and 16 also have earlier attempts that the bench refused to launch because a GPU was not idle, or that
were stopped before timing; none produced a result.

## Memory

| | Baseline | Faster-R3 |
|---|---:|---:|
| Peak allocated per GPU (max over ranks, steps 0–12 of a timing run) | 131.03 GiB | 104.78 GiB |
| Peak reserved per GPU (same runs) | 133.73 GiB | 130.69 GiB |
| GenEval worker on the same GPU | about 3.9 GiB | about 3.9 GiB |

The numbers are the trainer's own per-step log (`mem_allocated`, `mem_cache`: cumulative maxima), which the bench
excludes from its comparison. The baseline row is a baseline timing run of round 2, the Faster-R3 row the round-17
candidate. Most of the lower allocation comes from R09 (about 27 GiB in its own timing run); the reserved peak
changed little.

## Not tested

- Other datasets and rewards of the upstream tutorial (GenEval++, TIIF with a Qwen2.5-VL-72B reward model), the
  8-GPU default of `scripts/train_rl.sh`, multi-node runs, other group sizes, and dynamic RoPE.
- Long training. The check covers 14 steps; nothing beyond 16 steps was run with this tree.
- The evaluation pass: it uses the same generation code, so the graph paths run there too, but its rewards are only
  logged through wandb and were not compared (see [judge.md](judge.md#known-gap-wandb-values)).
- Checkpoint saving and resuming.

## Verifying the tree

The bench fingerprints each tree with the digest below (files sorted by path; `.git`, `__pycache__`, `wandb`,
`outputs`, `figures` and `*.pyc` skipped). `git archive <commit>` of every commit in the history table reproduces the
digest recorded for its round, for example `761b0b0` → `f993d78b…` (baseline) and `e71a7bd` → `315a5ebc…` (R17).

```python
import hashlib, pathlib, sys
SKIP = {'.git', '__pycache__', 'wandb', 'outputs', 'figures'}
def digest(tree):
    tree, h = pathlib.Path(tree), hashlib.sha256()
    for p in sorted(tree.rglob('*')):
        if p.is_file() and not set(p.relative_to(tree).parts) & SKIP and p.suffix != '.pyc':
            h.update(str(p.relative_to(tree)).encode() + b'\0' + hashlib.sha256(p.read_bytes()).hexdigest().encode() + b'\n')
    return h.hexdigest()
print(digest(sys.argv[1]))
```
