> Historical result: use commit `44d1fac` for its code, launcher and figures. This page does not describe current `main`.

# The judge

Every speedup in this repository comes from a frozen **bit-exact bench**, fixed before the first optimization round
and used unchanged for all of them. Only changes that leave every recorded number identical could pass it, so no
statistical tolerance is involved. The bench (scripts, launch rules and baseline records) is a separate artifact and
is not included; this page describes what it checks.

## What is compared

- **Baseline:** the unmodified upstream tree (branch `upstream`, sen-ye/R3 at `761b0b0`).
- **Candidate:** an immutable snapshot of the optimized tree. The source digest of the tree (see
  [training.md](training-4gpu.md#verifying-the-tree)) is taken when the job is formed and again after it ends; a change
  in between fails the run.
- **Recipe:** the arguments of [training.md](training-4gpu.md#launch), one environment for both trees. The recipe file,
  the validation fixture and the verification records of the model, the environment, the reward server and its
  CLIP weights are hashed at the start and the end of every run.
- **Entry:** the bench imports the tree's own `train.online_rl` and calls its unchanged `main()` with the recipe as
  `sys.argv`, under `torchrun` with 4 processes. It only observes: an optimizer post-step hook reads the trainer's
  local variables, and a wrapper around `wandb.log` was meant to record rank 0's logged values. A run stops at a
  barrier after its last step.
- **Launch:** every run is a fresh launch: a host-wide GPU lock, an idle check of the 4 GPUs (no compute process,
  0% utilization, at most 256 MiB used, sampled five times), a check of the reward server's offline caches, the four
  GenEval workers on the same GPUs, then one `torchrun`. A run whose GPUs are not idle is refused, not retried with
  looser rules.

## Bitwise check

The bench sets the flags; the tree's code path is the same in both modes.

- **Mode:** `seed_everything(4396)`, deterministic flags and `torch.use_deterministic_algorithms(True)`.
- **Length:** steps 0–13 from the released checkpoint (7 generate and 7 refine steps, including the step-0
  evaluation and the 10-step learning-rate warm-up).
- **Recorded per step and rank:** the raw text and image losses, the trainer's additional statistics, the gradient
  norm before clipping, the learning rate of every parameter group, the SHA-256 of the rollout token ids, the token
  counts, the SHA-256 of every generated image, the GenEval rewards (scores, totals, format rewards), and the text and
  image advantages.
- **Recorded after the last step:** the SHA-256 of every local parameter shard and of every Adam state tensor of the
  FSDP2 model, and one digest over all of them.

The candidate is **identical** only if every per-step record of every rank and the final digests equal those of
the baseline record, and rank 0's logged values agree (excluding `step_time`, `mem_allocated` and `mem_cache`).
Before any round, two baseline runs on the host had to be identical to each other; they were (final rank-0 digest
`c2b02b63537fc5576515f96744058e5799e6a1fb996516560558b846983d2175`). Every round is compared with the first of them.

## Timing

Only identical candidates are timed: baseline A → candidate → baseline B, each a fresh launch on the same host in
the same session, with production settings (no deterministic flags), 16 steps each.

- The end of each step is taken at the optimizer post-step hook after `torch.cuda.synchronize()`; the slowest
  rank counts.
- Steps 0–9 are warm-up (they include start-up and the step-0 evaluation). Three
  windows cover steps 10–11, 12–13 and 14–15, each one generate and one refine step; seconds per step = window
  time / 2.
- Per window, ratio = min(baseline A, baseline B) / candidate. **Speedup = the smallest of the three ratios.**
- Baseline spread = the largest relative difference between baseline A and B over the windows. The round counts
  only if speedup ≥ max(1.02, 1 + 2 × spread).
- Timing is never compared with a stored number; every round times its own baselines.

## Acceptance

| Case | Expected | Result |
|---|---|---|
| CPU tests of the bench: the launch rules are kept, flags are set only by the bench, the bitwise comparison ignores wall-clock fields and catches a changed loss, a changed final state and a changed logged value, the min-of-ratios arithmetic | all pass | all pass |
| Baseline run twice on the host | identical | identical, 14 steps |
| Unmodified upstream tree as the candidate (round 0) | identical, no benefit | identical; 0.9906× against a threshold of 1.0432 (ratios 1.0082, 1.0117, 0.9906; spread 2.2%) |

The round-0 control shows the noise of the timing: about ±1% per window with this spread. The bench files, the
launch rules and the baseline records (17 files) were hashed before the first round and were unchanged when the
work ended.

## Known gap: wandb values

The bench wrapped `wandb.log` before calling the trainer's `main()`, but wandb 0.30.0 rebinds `wandb.log` inside
`wandb.init()`, so the wrapper never saw a call. This was found during round 2: the baseline records and the
candidate records all hold an empty list, and the comparison of the logged values compared two empty lists. The
supervisor of the run ruled the item redundant for acceptance, because the per-step losses, rewards, advantages,
gradient norm, learning rates and the final parameter and Adam digests are compared directly, and every round was
judged under that ruling. Values that exist only in the wandb log, such as the evaluation rewards of step 0, were not
compared. Repairing the hook would have required new baseline records; it was
not done.

## Limits

- Bitwise identity was shown for 14 steps in deterministic mode. A change that alters numbers only later, or only
  without the deterministic flags, would not be caught; the timing runs themselves are not compared.
- The timing covers six steps per run after warm-up. It measures steady-state seconds per RL step, not the
  wall-clock time of a full training run with evaluation and checkpoints.
- Rounds are judged one at a time against fresh baselines; the speedups of two rounds are not a paired comparison.
- The CUDA graphs live for one text or image call, so their captures recur in every step and are inside the timed
  windows.
- Everything ran on one host with 4 H200-class GPUs and the GenEval recipe.

An earlier, statistical judge for this workload was built and then set aside when the work was restricted to
bit-exact changes; none of its results are used here.
