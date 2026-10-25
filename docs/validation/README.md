# Validation artifacts

- `h100.json`: aggregate eight-GPU records exported from the completed 2026-10-24 experiment. `runtime_files`
  holds SHA-256 for all 37 measured Python files under `modeling/` and `train/`; the public source was compared
  byte-for-byte with the immutable candidate. No runtime optimization was rewritten for publication.
  `source_commit` identifies the experimental repository; that complete experimental history is not imported here.
- `../performance-views/history.json`: the separate four-GPU history; do not mix its speedups with H100 results.
- `../../scripts/validate_release.py`: CPU-only source/evidence consistency check, not a training judge.

## Publication checks (2026-10-25)

The portable training and flow diagnostics were rerun on one H100, under the host GPU locks, against the
byte-identical measured runtime snapshot. **37 training operator cases** passed forward/backward RMSNorm, RoPE
and both SwiGLU contraction checks; **169 flow cases** reported `all_bitwise=true`. PyTorch was 2.9.1+cu128.
The [publication receipt](publication-check.json) records these aggregates and diagnostic-file hashes.
These are narrow GPU checks; the eight-rank training benchmark was not rerun. The capture-dependent text test
and full-layer test were not rerun for publication.

CPU checks also passed runtime-file hashes, all ten timing records, adapter AST equivalence, launcher arguments
and fixture construction (including paths with spaces), reward-worker allocation/replacement and execution settings,
Python syntax, local documentation file links and private-path/credential scans. All four new figures were visually
inspected. Two inherited trailing spaces in the measured model source were preserved to retain exact source hashes.

## GPU diagnostics

Use the tested environment and public checkpoint (`MODEL_PATH`). Respect the host's GPU reservation mechanism
before running these scripts. They were used during optimization; the portable path wrappers have not been
rerun as an end-to-end training launch.

```bash
export MODEL_PATH=/path/to/BAGEL-7B-MoT
mkdir -p output/validation
python tools/fuse_train/test_fused_ops.py --out output/validation/train-ops.json
python tools/fuse_flow/fusetest.py --tree "$PWD" --task units --out output/validation/flow
```

Training checks cover fused forward outputs and input/weight gradients against eager autograd on realistic shapes,
using real checkpoint norm weights when present. Check `all_rms`, `all_rope` and `swiglu_contract_ok["1"]` for the
measured contraction configuration. The script also probes contraction zero, which is not the reported setup.
Flow `units` needs the checkpoint's configuration. Its JSON must have `all_bitwise=true`.

`tools/fuse_decode/test_fused_ops.py --acts <capture-dir> --out <report.json>` additionally requires locally captured
`acts_decode.pt` and `acts_prefill.pt`; those tensors are not bundled. It reports missing captures as failures.
`tools/fuse_train/test_layer_bitwise.py` requires a separate pre-fusion `stack_b` reference tree and checkpoint;
it checks full-layer outputs and gradients. These are research diagnostics, not an automatic reproduction of
the eight-rank judge. The original diagnostic scripts report verdicts in JSON; inspect those verdicts rather
than assuming exit status zero implies success.

The final source passed the external eight-rank, 14-step deterministic judge twice, once in each timing scope.
See [judge.md](../judge.md) for the observed fields, empty-wandb gap, memory and missing long-training evidence.
