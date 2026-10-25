#!/usr/bin/env python3
"""Regenerate H100 figures from validated aggregate records; historical figures are unchanged."""
from pathlib import Path
import importlib.util
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('validate_release', ROOT / 'scripts/validate_release.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
DATA = check.validate()
ROWS = {r['name']: r for r in DATA['rows']}
RED, LIGHT, INK, MUTED, GRID = '#8C0000', '#D7BDBF', '#292122', '#79696B', '#EADFE0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none',
                     'svg.hashsalt': 'faster-r3-h100-crimson'})


def save(fig, path):
    fig.savefig(path, facecolor='white', metadata={'Date': None})
    if os.environ.get('R3_FIGURE_PREVIEW_DIR'):
        preview = Path(os.environ['R3_FIGURE_PREVIEW_DIR'])
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / (path.stem + '.png'), dpi=140, facecolor='white')
    plt.close(fig)


def frame(ax, axis='x'):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0, colors=MUTED, pad=8)
    ax.grid(axis=axis, color=GRID, linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def hero():
    t = ROWS['fusion_b_total2']['timing']
    idx = int(np.argmin(t['ratios']))
    baseline, cand = min(t['ref_a'][idx], t['ref_b'][idx]), t['cand'][idx]
    fig = plt.figure(figsize=(11, 4.5))
    fig.text(.05, .90, 'Faster-R3, optimized with AI agents', color=RED, fontsize=21, weight='bold')
    fig.text(.05, .81, 'BAGEL-7B-MoT Tree-RL · 8 × H100 80 GB · GenEval · 14 steps bitwise identical', color=MUTED)
    ax = fig.add_axes([.25, .25, .39, .42])
    ax.barh([1, 0], [baseline, cand], color=[LIGHT, RED], height=.29, zorder=3)
    ax.set_yticks([1, 0], ['Upstream + original\nreward service', 'Faster-R3 + faster\nreward service'])
    for y, x in zip([1, 0], [baseline, cand]):
        ax.text(x+.5, y, f'{x:.2f} s', va='center', weight='bold', color=INK)
    ax.set_xlim(0, 30); ax.set_ylim(-.6, 1.6); frame(ax)
    ax.set_xlabel('Seconds per RL step ↓', color=MUTED, labelpad=10)
    fig.text(.81, .53, '2.08×', fontsize=39, weight='bold', color=RED, ha='center')
    fig.text(.81, .42, 'Total step speedup', color=INK, ha='center')
    fig.text(.81, .32, 'Training code alone: 1.88×\n(separate paired measurement)', color=MUTED, fontsize=10, ha='center')
    fig.text(.05, .095, 'Baseline A → candidate → baseline B; 10 warm-up steps + 3 windows of 2 steps. Bars show the slowest-ratio window.', color=MUTED, fontsize=9.3)
    fig.text(.05, .045, 'Total includes reward-service execution changes. Short-run timing; no full-training or time-to-quality claim.', color=MUTED, fontsize=9.5)
    save(fig, OUT/'h100-benchmark.svg')


def history():
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.0), gridspec_kw={'width_ratios':[1.25, 1]})
    fig.subplots_adjust(left=.20, right=.96, top=.78, bottom=.18, wspace=.78)
    fig.text(.05, .91, 'Eight-GPU milestones, two comparison scopes', color=RED, fontsize=20, weight='bold')
    fig.text(.05, .845, 'Minimum of three paired ratios; each row has its own fresh baseline timings.', color=MUTED)
    names=['rwfast_h1_r17','stack_a_rw','stack_b_rw','fuse_decode_1r','fuse_train_1r','fuse_flow_1r','fusion_b_r']
    labels=['R17 on H100','Overlap + host work','Stack B','Stack B + text fusion','Stack B + train fusion','Stack B + flow fusion','All three: fusion_b']
    for ax, ns, ls, title in [(axes[0],names,labels,'Training code only'),(axes[1],['stack_b_total','fusion_b_total2'],['Stack B','fusion_b'],'Total, including reward service')]:
        vals=[ROWS[n]['timing']['speedup_min_of_ratios'] for n in ns]
        y=np.arange(len(ns))
        ax.barh(y,vals,height=.50,color=[LIGHT]*(len(ns)-1)+[RED],zorder=3)
        ax.set_yticks(y,ls); ax.invert_yaxis(); ax.set_xlim(0,2.5)
        for i,v in enumerate(vals): ax.text(v+.05,i,f'{v:.3f}×',va='center',fontsize=10,color=INK)
        ax.set_title(title,color=INK,fontsize=11,pad=16); frame(ax)
        ax.set_xlabel('Paired speedup ↑',color=MUTED)
    fig.text(.05,.08,'Text / train / flow rows are individual additions to Stack B, not a cumulative sequence. All shown candidates passed bitwise checks.',color=MUTED,fontsize=9)
    fig.text(.05,.035,'The older four-GPU 1.47× result is a separate workload configuration and is excluded here.',color=MUTED,fontsize=9.5)
    save(fig,OUT/'h100-history.svg')


def paired():
    fig,axes=plt.subplots(1,2,figsize=(11,5.4),sharey=True)
    fig.subplots_adjust(left=.08,right=.97,top=.75,bottom=.24,wspace=.16)
    fig.text(.05,.91,'Fresh baselines bracket each candidate run',color=RED,fontsize=20,weight='bold')
    fig.text(.05,.84,'Seconds per step; every two-step window contains generation and refinement.',color=MUTED)
    for ax,name,title in zip(axes,['fusion_b_total2','fusion_b_r'],['Total: original → faster reward service','Training code: faster reward service on both sides']):
        t=ROWS[name]['timing']; x=np.arange(3); w=.24
        for offset,key,label,color in [(-w,'ref_a','Baseline A',LIGHT),(0,'cand','Faster-R3',RED),(w,'ref_b','Baseline B','#A99A9C')]:
            ax.bar(x+offset,t[key],w,label=label,color=color,zorder=3)
        ax.set_xticks(x,['Steps 10–11','Steps 12–13','Steps 14–15']); ax.set_ylim(0,38)
        ax.set_title(title,fontsize=10,color=INK,pad=13); frame(ax,'y')
        for i,r in enumerate(t['ratios']): ax.text(i,max(t['ref_a'][i],t['ref_b'][i])+1,f'{r:.3f}×',ha='center',color=RED,weight='bold')
    axes[0].set_ylabel('Seconds / RL step ↓',color=MUTED)
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.07),ncol=3,frameon=False)
    fig.text(.05,.035,'Speedup = min over windows of min(A, B) / candidate. Baseline spread: total 4.74%; training code 3.51%.',color=MUTED,fontsize=9.5)
    save(fig,OUT/'h100-paired-timing.svg')


def modules():
    fig=plt.figure(figsize=(11,6.7)); ax=fig.add_axes([0,0,1,1]); ax.axis('off')
    ax.text(.05,.92,'Where the work was removed or overlapped',color=RED,fontsize=20,weight='bold')
    cards=[('Reward service','4 CPU threads / worker\nColour crops loaded in-process\nSame detector, CLIP and scoring'),('Overlap and host path','Rewards behind packing + forwards\nLatent scalars read once\nShared masks and frozen ViT'),('Rollout scheduling','Pipelined text host bookkeeping\nReuse identical edit CFG prefix\nConditional graph-capture cleanup'),('Text decode / prefill','Residual add + RMSNorm\nQ/K norm + RoPE; SiLU × up\nConcurrent small projections'),('Image flow','Norm + expert-order restoration\nK/V direct to merged cache\nEarly per-branch graph capture'),('Training forward / backward','Fused norm, SwiGLU and RoPE\nSplit expert residual streams\nSkip inactive CFG-row GEMMs')]
    for i,(title,body) in enumerate(cards):
        x=.05+(i%3)*.31; y=.53-(i//3)*.35
        ax.add_patch(FancyBboxPatch((x,y),.285,.285,boxstyle='round,pad=0.008,rounding_size=0.012',edgecolor=GRID,facecolor='#FAF5F5'))
        ax.text(x+.015,y+.23,title,color=RED,fontsize=11.4,weight='bold')
        ax.text(x+.015,y+.17,body,color=INK,fontsize=10.2,va='top',linespacing=1.8)
    ax.text(.05,.065,'Same measured arithmetic, reduction order and RNG consumption; FP32 main weights / Adam, BF16 compute.',color=MUTED,fontsize=10)
    ax.text(.05,.025,'Validated on eight H100 GPUs for 14 deterministic steps. Kernel exactness is tied to the tested PyTorch / Triton versions.',color=MUTED,fontsize=9.5)
    save(fig,ROOT/'docs/change-views/h100-module-map.svg')


if __name__=='__main__':
    hero(); history(); paired(); modules()
