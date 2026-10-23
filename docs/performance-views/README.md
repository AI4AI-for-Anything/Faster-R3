# Performance figures

Regenerate all figures with `python docs/performance-views/render.py` (matplotlib and numpy, no GPU). The script
recomputes every ratio, speedup, spread and threshold from the recorded windows in [history.json](history.json) and
checks them against the recorded values.

- `01-benchmark.svg`: round 17, commit `e71a7bd`. Seconds per RL step in the three windows: baseline A 23.401 /
  24.154 / 24.225, Faster-R3 15.358 / 16.039 / 16.028, baseline B 23.468 / 23.592 / 23.764. Ratios (faster
  baseline / Faster-R3) 1.5237 / 1.4709 / 1.4827; the speedup is the smallest, 1.4709×. The bars show window 2,
  the one that sets it. Four H200-class GPUs.
- `02-history.svg`: the eleven rounds in `main`, each its own paired measurement against fresh baseline runs on the
  same host. Points are cumulative trees, not isolated ablations, and are not paired with each other.
- `03-paired-timing.svg`: the per-window seconds of the round-0 control (unmodified upstream timed as the
  candidate) and of round 17.
- `../change-views/module-map.svg`: the changed files grouped by area.

`history.json` keeps, per round, the commit, the source digest of the measured tree, the bitwise verdict and the
final rank-0 digest, the three windows of each run, the ratios, the speedup, the reference spread and the threshold,
plus the rounds that ran out of memory before a verdict. The values are copied from the bench's round records
without host names or paths.
