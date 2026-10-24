#!/usr/bin/env python3
"""Render the Faster-R3 README figures from archived measurements; no GPU work.

Input: history.json (one row per round of the bit-exact bench, copied from its round records).
Outputs: 01-benchmark.svg, 02-history.svg, 03-paired-timing.svg and ../change-views/module-map.svg.
Usage: python docs/performance-views/render.py   (needs matplotlib and numpy)
"""
from pathlib import Path
import json
import math
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
RED = '#8C0000'
LIGHT = '#D7BDBF'
GREY = '#A99A9C'
MUTED = '#79696B'
INK = '#292122'
GRID = '#EADFE0'
CARD = '#FAF5F5'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'svg.hashsalt': 'faster-r3-crimson', 'font.size': 12})


def save(fig, path):
    target = path.with_suffix('.svg')
    fig.savefig(target, facecolor='white', metadata={'Date': None})
    target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
    plt.close(fig)


def frame(ax, axis='y'):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0, labelsize=11.5, colors=MUTED, pad=9)
    ax.grid(axis=axis, color=GRID, linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def load_history():
    """Recompute every ratio, speedup and threshold from the recorded windows and check them."""
    data = json.loads((ROOT / 'history.json').read_text())
    for row in data['rows']:
        ratios = [min(a, b) / c for a, c, b in zip(row['reference_a_s'], row['candidate_s'], row['reference_b_s'])]
        spread = max(abs(a - b) / min(a, b) for a, b in zip(row['reference_a_s'], row['reference_b_s']))
        assert all(math.isclose(x, y, abs_tol=2e-4) for x, y in zip(ratios, row['ratios'])), row['round']
        assert math.isclose(min(ratios), row['speedup'], abs_tol=2e-4), row['round']
        assert math.isclose(spread, row['reference_spread'], abs_tol=2e-4), row['round']
        assert math.isclose(max(1.02, 1 + 2 * spread), row['threshold'], abs_tol=2e-4), row['round']
        assert row['passed_threshold'] == (row['speedup'] >= row['threshold']), row['round']
        assert row['bitwise_identical'] and row['bitwise_steps'] == 14, row['round']
        assert row['final_digest_rank0'] == data['reference']['final_digest_rank0'], row['round']
    return data


def deciding_window(row):
    i = int(np.argmin(row['ratios']))
    return i, min(row['reference_a_s'][i], row['reference_b_s'][i]), row['candidate_s'][i]


def hero():
    final = load_history()['rows'][-1]
    i, ref, cand = deciding_window(final)
    fig = plt.figure(figsize=(9, 4.5), facecolor='white')
    fig.text(.06, .88, 'Faster-R3, optimized with AI agents', color=RED, weight='bold', fontsize=19)
    fig.text(.06, .78, 'R3 Tree-RL of BAGEL-7B-MoT · GenEval reward · four H200-class GPUs · bitwise-identical records',
             color=MUTED, fontsize=10.5)
    ax = fig.add_axes([.24, .24, .39, .45])
    ax.barh([1, 0], [ref, cand], color=[LIGHT, RED], height=.24, zorder=3)
    ax.set_yticks([1, 0], ['Baseline', 'Faster-R3'])
    for y, value in zip([1, 0], [ref, cand]):
        ax.text(value + .5, y, f'{value:.2f} s', va='center', color=INK, fontsize=11.5, weight='bold')
    ax.set_xlim(0, 30)
    ax.set_ylim(-.6, 1.6)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    frame(ax, 'x')
    ax.tick_params(labelsize=10)
    ax.set_xlabel('Seconds per RL step (rollout, reward, update) ↓', color=MUTED, labelpad=12, fontsize=11)
    fig.text(.85, .54, f"{final['speedup']:.2f}×", color=RED, fontsize=35, weight='bold', ha='center')
    fig.text(.85, .43, 'RL-step speedup', color=MUTED, fontsize=12, ha='center')
    fig.text(.85, .36, '14 steps bitwise identical', color=MUTED, fontsize=10, ha='center')
    fig.text(.06, .075, f'Baseline → Faster-R3 → baseline in independent launches on one host; bars show window {i + 1} of '
             '3, the one with the smallest ratio.', color=MUTED, fontsize=9.5)
    fig.text(.06, .03, 'Baseline = the faster of the two baseline runs in that window. Speedup = the smallest of the '
             f"three window ratios ({', '.join(f'{x:.4f}' for x in final['ratios'])}).", color=MUTED, fontsize=9.5)
    save(fig, ROOT / '01-benchmark')


def history():
    data = load_history()
    rows = [row for row in data['rows'] if row['in_main']]
    height = 10.0
    top = lambda inches: 1 - inches / height       # figure fraction from inches below the top edge
    fig = plt.figure(figsize=(9, height), facecolor='white')
    fig.text(.09, top(.36), 'Faster-R3', fontsize=11, weight='bold', color=RED)
    fig.text(.09, top(.79), 'Agent Optimization History', fontsize=21.5, weight='bold', color=INK)
    fig.text(.94, top(.65), f"{rows[-1]['speedup']:.2f}×", fontsize=29, ha='right', weight='bold', color=RED)
    fig.text(.94, top(.88), 'RL-step speedup', fontsize=10.5, ha='right', color=MUTED)
    fig.text(.09, top(1.31), f"Speedup of each accepted round · 4 H200-class GPUs · every round bitwise identical · "
             f"{len(rows)} of {len(data['rows']) - 1 + len(data['incomplete'])} rounds in main",
             fontsize=11, color=MUTED)
    ax = fig.add_axes([.09, top(4.25), .85, 2.67 / height])
    x = list(range(len(rows)))
    values = [row['speedup'] for row in rows]
    ax.plot(x, values, color=RED, linewidth=2.3, zorder=2)
    ax.plot(x, values, linestyle='none', marker='o', markersize=7, color=RED, zorder=3)
    for i, value in enumerate(values):
        offset, align = ((0, -23), 'center') if i % 2 == 0 else ((0, 11), 'center')
        if i == 1:                                   # keep the steep R01 -> R02 segment clear
            offset, align = (-10, 4), 'right'
        if i == len(rows) - 1:
            offset = (0, 11)
        ax.annotate(f'{value:.2f}×', (x[i], value), xytext=offset, textcoords='offset points', ha=align,
                    fontsize=10.5, color=RED if i == len(rows) - 1 else INK, weight='bold')
    ax.set_xticks(x, [row['round'].upper() for row in rows])
    ax.tick_params(axis='x', labelsize=10)
    ax.set_yticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.1f}×'))
    ax.set_ylim(.95, 1.56)
    ax.set_xlim(-.6, len(rows) - .4)
    frame(ax)
    for i, row in enumerate(rows):
        bx, by = .09 + (i % 2) * .44, top(5.2 + (i // 2) * .66)
        fig.add_artist(FancyBboxPatch((bx, by - .38 / height), .41, .54 / height,
                       boxstyle='round,pad=0.004,rounding_size=0.007',
                       transform=fig.transFigure, facecolor=CARD, edgecolor='none'))
        fig.text(bx + .014, by - .11 / height, row['round'].upper(), fontsize=11.5, weight='bold', color=RED,
                 va='center')
        label = textwrap.fill(row['label'], width=34, break_long_words=False, break_on_hyphens=False)
        fig.text(bx + .072, by - .11 / height, label, fontsize=10, color=INK, weight='bold', va='center',
                 linespacing=1.25)
    fig.text(.09, top(9.7), 'Each point is one round of the same frozen bit-exact bench on the same host: the paired '
             'speedup of that tree against\nfreshly timed upstream runs. Rounds are separate measurements, not paired '
             'with each other: R15 and R16 scored\nbelow R12, and the baseline runs of accepted rounds differed by up '
             'to 3.5%. Not in main: R11 and R14 (reverted)\nand five rounds that ran out of memory.', fontsize=9.3, color=MUTED,
             linespacing=1.4)
    save(fig, ROOT / '02-history')


def paired_timing():
    data = load_history()
    pick = {row['round']: row for row in data['rows']}
    panels = [(pick['r00'], 'Control: upstream timed as the candidate'), (pick['r17'], 'R17: final Faster-R3')]
    fig = plt.figure(figsize=(10, 5.2), facecolor='white')
    fig.text(.06, .92, 'How the speedup is measured', color=INK, weight='bold', fontsize=19)
    fig.text(.06, .86, 'Seconds per RL step in each 2-step window (after 10 warm-up steps) · baseline A → candidate → '
             'baseline B, independent launches', color=MUTED, fontsize=10.2)
    for k, (row, title) in enumerate(panels):
        ax = fig.add_axes([.07 + k * .49, .2, .41, .52])
        xs = np.arange(3)
        w = .24
        ax.bar(xs - w, row['reference_a_s'], w, color=LIGHT, zorder=3)
        ax.bar(xs, row['candidate_s'], w, color=RED if k else GREY, zorder=3)
        ax.bar(xs + w, row['reference_b_s'], w, color=LIGHT, zorder=3, hatch='///', edgecolor='white')
        for j in xs:
            peak = max(row['reference_a_s'][j], row['reference_b_s'][j])
            ax.text(j, peak + .8, f"{row['ratios'][j]:.3f}×", ha='center', fontsize=10.5, color=INK, weight='bold')
        ax.set_xticks(xs, [f'window {j + 1}' for j in xs])
        ax.set_ylim(0, 29)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        frame(ax)
        ax.tick_params(labelsize=10)
        ax.set_title(title, loc='left', color=INK, fontsize=12, weight='bold', pad=10)
        verdict = (f"speedup {row['speedup']:.4f}× < threshold {row['threshold']:.4f}: no benefit" if k == 0 else
                   f"speedup {row['speedup']:.4f}× ≥ threshold {row['threshold']:.4f}: accepted")
        ax.text(0, -.2, verdict, transform=ax.transAxes, fontsize=10, color=RED if k else MUTED)
    fig.legend(handles=[Patch(facecolor=LIGHT, label='baseline A'),
                        Patch(facecolor=LIGHT, hatch='///', edgecolor='white', label='baseline B'),
                        Patch(facecolor=GREY, label='candidate (control)'),
                        Patch(facecolor=RED, label='candidate (Faster-R3)')],
               loc='lower center', ncol=4, frameon=False, fontsize=9.5, labelcolor=INK, bbox_to_anchor=(.5, .0))
    save(fig, ROOT / '03-paired-timing')


def modulemap():
    fig = plt.figure(figsize=(9, 6.8), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(.055, .925, 'What changed in Faster-R3', fontsize=23, color=INK, weight='bold')
    cards = [
        ('Static KV-cache metadata', 'Lengths read from the GPU once per forward\nPacking indexes reused across layers\n'
         'Capacity proven once per decoding call',
         'modeling/bagel/qwen2_navit.py\nmodeling/bagel/static_text_graph.py\ntrain/rollout_controller.py',
         'R01 · R03 · R15'),
        ('Text decoding', 'Backbone replayed as a CUDA graph\nLM head and sampler replayed, same RNG\n'
         'One host read per token batch',
         'modeling/bagel/bagel.py\nmodeling/bagel/static_text_graph.py\nmodeling/bagel/static_text_sampling_graph.py',
         'R02 · R05 · R12'),
        ('Image flow (denoising)', 'Timesteps built on the GPU\nMoT tokens kept in expert order\n'
         'Backbones replayed, CFG branches concurrent',
         'modeling/bagel/bagel.py\nmodeling/bagel/qwen2_navit.py\nmodeling/bagel/static_flow_graph.py',
         'R04 · R10 · R16 · R17'),
        ('Training-step memory', 'Full KL-reference weights resharded\nafter the reference forward,\n'
         'before the policy forward and backward',
         'train/online_rl.py', 'R09'),
    ]
    for i, (title, description, paths, rounds) in enumerate(cards):
        x, y = .055 + (i % 2) * .455, .5 - (i // 2) * .42
        ax.add_patch(FancyBboxPatch((x, y), .425, .37,
                     boxstyle='round,pad=0.009,rounding_size=0.012', facecolor=CARD, edgecolor=GRID))
        ax.plot([x, x + .425], [y + .37, y + .37], color=RED, linewidth=2, solid_capstyle='round')
        ax.text(x + .014, y + .33, f'{i + 1:02d}', fontsize=11, color=RED, weight='bold')
        ax.text(x + .41, y + .33, rounds, fontsize=10, color=MUTED, ha='right')
        ax.text(x + .014, y + .284, title, fontsize=12.5, color=INK, weight='bold')
        ax.text(x + .014, y + .232, description, fontsize=10.2, color=INK, va='top', linespacing=1.6)
        ax.text(x + .014, y + .03, paths, fontsize=9.2, color=MUTED, linespacing=1.5)
    ax.text(.055, .014, '7 files: 3 new, 4 upstream files edited. There are no switches: the changes are always on. The '
            'CUDA-graph and expert-order\npaths fall back to the upstream code when their preconditions do not hold.',
            fontsize=9.3, color=MUTED, linespacing=1.4)
    save(fig, ROOT.parent / 'change-views/module-map')


if __name__ == '__main__':
    hero()
    history()
    paired_timing()
    modulemap()
