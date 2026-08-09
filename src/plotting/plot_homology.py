#!/usr/bin/env python
"""
Plot homology detection metrics by classification level and method.

Input:
    src/plotting/{dataset_dir}/metrics/*.tsv and *.npy  (from calc_homology.py)

Output:
    src/plotting/{dataset_dir}/plots/*.svg

Usage:
    python -m src.plotting.plot_homology --dataset cath
    python -m src.plotting.plot_homology --dataset scope40
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

plt.rcParams['svg.fonttype'] = 'none'

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIRS = {
    "cath": ("cath", ['class', 'architecture', 'topology', 'superfamily']),
    "scope40": ("scope", ['class', 'fold', 'superfamily', 'family']),
}

# methods and visual properties
methods = ['tmvec1', 'tmvec2', 'tmvec2s', 'tmalign', 'foldseek']
zorder = [4, 5, 3, 2, 1]

cmap = plt.get_cmap('tab10')
palette = {
    'tmvec1':   cmap(0),  # blue
    'tmvec2':   cmap(5),  # cyan
    'tmvec2s':  cmap(1),  # orange
    'tmalign':  cmap(2),  # green
    'foldseek': cmap(4),  # purple
    'plmblast': cmap(6),  # pink
}

fw = 8
margs = dict(marker='o', alpha=0.75, ms=5)
npt = 5000
ns = np.array([1, 5, 10])
ks = np.array([1, 5, 10, 50, 100])


def _fmt_val(val):
    return '{:.3f}'.format(val).lstrip('0')


def main():
    parser = argparse.ArgumentParser(description="Plot homology detection metrics")
    parser.add_argument("--dataset", choices=DATASET_DIRS.keys(), required=True)
    parser.add_argument("--metrics-dir", default=None, help="Input metrics directory (auto-detected)")
    parser.add_argument("--output-dir", default=None, help="Output plots directory (auto-detected)")
    parser.add_argument("--suffix", default="", help="Suffix for output filenames (e.g. '.all')")
    args = parser.parse_args()

    dataset_dir, levels = DATASET_DIRS[args.dataset]
    plot_dir = REPO_ROOT / "src" / "plotting" / dataset_dir
    indir = Path(args.metrics_dir) if args.metrics_dir else plot_dir / "metrics"
    outdir = Path(args.output_dir) if args.output_dir else plot_dir / "plots"
    suffix = args.suffix
    nl = len(levels)

    outdir.mkdir(parents=True, exist_ok=True)

    # Read calculated metrics
    dfs = {x: pd.read_table(indir / f'{x}.tsv', index_col=0) for x in levels}

    # Filter to available methods (from first level with data)
    first_valid = next((l for l in levels if 'ap' in dfs[l].columns), None)
    if first_valid is None:
        raise ValueError("No metrics found. Run calc_homology first.")
    available = [m for m in methods if m in dfs[first_valid].index]
    avail_zorder = [z for m, z in zip(methods, zorder) if m in available]

    # PR curves and AP
    fig, axes = plt.subplots(
        1, nl, figsize=(fw, 2.25), sharey=True, constrained_layout=True)
    for i, level in enumerate(levels):
        ax = axes[i]
        if 'ap' not in dfs[level].columns:
            ax.set_title(f"{level.capitalize()}\n(no data)")
            continue
        aps = dfs[level]['ap']
        for (method, z) in zip(available, avail_zorder):
            npy_path = indir / f'pr_{level}_{method}.npy'
            if not npy_path.exists():
                continue
            precision, recall = np.load(npy_path).T
            step = max(1, len(precision) // npt)
            precision, recall = precision[::step], recall[::step]
            PrecisionRecallDisplay(precision, recall).plot(ax=ax, curve_kwargs=dict(
                label=_fmt_val(aps.loc[method]), color=palette[method], zorder=z))
            ax.set_aspect('auto')
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(handlelength=1, handletextpad=0.25)
        for handle in leg.get_lines():
            handle.set_marker('o')
            handle.set_ms(5)
            handle.set_linestyle('none')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision' if i == 0 else None)
        ax.set_title(level.capitalize())
    fig.savefig(outdir / f'pr_curve{suffix}.svg')

    # Mean AP
    fig, axes = plt.subplots(1, nl, figsize=(fw, 1.5), constrained_layout=True)
    for i, level in enumerate(levels):
        ax = axes[i]
        if 'mean_ap' not in dfs[level].columns:
            ax.set_title(f"{level.capitalize()}\n(no data)")
            continue
        avail = [m for m in available if m in dfs[level].index]
        vals = dfs[level].loc[avail, 'mean_ap'].to_numpy()
        max_val = vals.max() if len(vals) > 0 else 1
        ax.bar(avail, vals, color=[palette[x] for x in avail])
        ax.set_ylim(0, max_val * 1.20)
        for j, val in enumerate(vals):
            ax.text(j, val, _fmt_val(val), ha='center', va='bottom')
        ax.set_xticks(range(len(avail)), [])
        ax.set_xlabel('Method')
        if i == 0:
            ax.set_ylabel('Mean AP')
    fig.savefig(outdir / f'mean_ap{suffix}.svg')

    # ROC(n)
    rocn_cols = [f'roc_{n}' for n in ns]
    rocn_ran = np.arange(len(ns))
    fig, axes = plt.subplots(1, nl, figsize=(fw, 2), constrained_layout=True)
    for i, level in enumerate(levels):
        ax = axes[i]
        df = dfs[level]
        if not all(c in df.columns for c in rocn_cols):
            ax.set_title(f"{level.capitalize()}\n(no data)")
            continue
        for j, method in enumerate(available):
            if method not in df.index:
                continue
            offset = (j - len(available) / 2) * 0.1
            ax.plot(np.arange(len(ns)) + offset, df.loc[method, rocn_cols],
                    color=palette[method], zorder=avail_zorder[j], **margs)
        ax.set_xticks(rocn_ran, ns)
        ax.set_xmargin(0.1)
        ax.set_xlabel('n', fontstyle='italic')
        if i == 0:
            ax.set_ylabel(r'ROC ($\it{n}$)')
    fig.savefig(outdir / f'rocn{suffix}.svg')

    # Hits @ K
    hits_cols = [f'hits_at_{k}' for k in ks]
    hits_ran = np.arange(len(ks))
    fig, axes = plt.subplots(
        1, nl, figsize=(fw, 2), sharey=True, constrained_layout=True)
    for i, level in enumerate(levels):
        ax = axes[i]
        df = dfs[level]
        if not all(c in df.columns for c in hits_cols):
            ax.set_title(f"{level.capitalize()}\n(no data)")
            continue
        for j, method in enumerate(available):
            if method not in df.index:
                continue
            offset = (j - len(available) / 2) * 0.1
            ax.plot(np.arange(len(ks)) + offset, df.loc[method, hits_cols],
                    color=palette[method], zorder=avail_zorder[j], **margs)
        ax.set_xticks(hits_ran, ks)
        ax.set_xlabel('k', fontstyle='italic')
        if i == 0:
            ax.set_ylabel(r'Hits @ $\it{k}$')
    fig.savefig(outdir / f'hitsk{suffix}.svg')

    print(f"Saved homology plots to {outdir}")


if __name__ == '__main__':
    main()
