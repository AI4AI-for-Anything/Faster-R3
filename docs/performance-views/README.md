# Current eight-H100 figures

Run `python docs/performance-views/render_h100.py` (matplotlib and numpy). It first validates the source manifest
and recomputes timing arithmetic from [../validation/h100.json](../validation/h100.json).

- `h100-benchmark.svg`: total 2.0831×, explicitly including the reward service change; code-only 1.8803× shown separately.
- `h100-history.svg`: individual and combined eight-GPU milestones, separated by comparison scope.
- `h100-paired-timing.svg`: all three windows of both final A → candidate → B measurements.
- `../change-views/h100-module-map.svg`: the latest optimization areas.

These figures match the English and Chinese README. Optional `R3_FIGURE_PREVIEW_DIR` writes PNG previews outside
the repository for visual inspection. The four-GPU assets below remain unchanged for the historical report.

# Historical four-GPU figures

Regenerate the historical figures with `python docs/performance-views/render.py` (matplotlib and numpy, no GPU). The script
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
final rank-0 digest, the three windows of each run, the ratios, the speedup, the baseline spread and the threshold,
plus the rounds that ran out of memory before a verdict. The values are copied from the bench's round records
without host names or paths.
