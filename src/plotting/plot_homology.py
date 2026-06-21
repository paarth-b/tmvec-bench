#!/usr/bin/env python
"""
Plot homology detection metrics by classification level and method.

Input:
    metrics/*.tsv and *.npy

Output:
    plots/*.svg

"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay


plt.rcParams['svg.fonttype'] = 'none'

indir = 'metrics'
outdir = 'plots'

# classification levels (adjust as needed)
levels = ['class', 'architecture', 'topology', 'superfamily']  # CATH
# levels = ['class', 'fold', 'superfamily', 'family']  # SCOPe
nl = len(levels)

# methods
methods = ['tmvec1', 'tmvec2s', 'tmalign', 'foldseek']

# visual order of methods
zorder = [3, 4, 2, 1]

# color scheme of methods
cmap = plt.get_cmap('tab10')
palette = {
    'tmvec1':   cmap(0),  # blue
    'tmvec2s':  cmap(1),  # orange
    'tmalign':  cmap(2),  # green
    'foldseek': cmap(4),  # purple
}

# figure width
fw = 8

# marker style
margs = dict(marker='o', alpha=0.75, ms=5)

# value formatting
def _fmt_val(val):
    return '{:.3f}'.format(val).lstrip('0')


# downsample precision-recall curve to this number of points
npt = 5000

# n-values for calculating ROC(n)
ns = np.array([1, 5, 10])

# k-values for calculating precision @ K and hits @ K
ks = np.array([1, 5, 10, 50, 100])


def main():
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Read calculated metrics
    dfs = {x: pd.read_table(f'{indir}/{x}.tsv', index_col=0) for x in levels}

    # PR curves and AP
    fig, axes = plt.subplots(
        1, nl, figsize=(fw, 2.25), sharey=True, constrained_layout=True)
    for i, level in enumerate(levels):
        ax = axes[i]
        aps = dfs[level]['ap']
        for (method, z) in zip(methods, zorder):
            precision, recall = np.load(f'{indir}/pr_{level}_{method}.npy').T
            step = max(1, len(precision) // npt)
            precision, recall = precision[::step], recall[::step]
            PrecisionRecallDisplay(precision, recall).plot(ax=ax, curve_kwargs=dict(
                label=_fmt_val(aps.loc[method]), color=palette[method], zorder=z))
            # or `ax.plot(recall, precision)`, but it will apply trapezoidal rule
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
    fig.savefig(f'{outdir}/pr_curve.svg')

    # Mean AP
    fig, axes = plt.subplots(1, nl, figsize=(fw, 1.5), constrained_layout=True)
    for i, level in enumerate(levels):
        ax = axes[i]
        vals = dfs[level].loc[methods, 'mean_ap'].to_numpy()
        max_val = vals.max()
        ax.bar(methods, vals, color=[palette[x] for x in methods])
        ax.set_ylim(0, max_val * 1.20)
        for j, val in enumerate(vals):
            ax.text(j, val, _fmt_val(val), ha='center', va='bottom')
        ax.set_xticks(range(len(methods)), [])
        ax.set_xlabel('Method')
        if i == 0:
            ax.set_ylabel('Mean AP')
    fig.savefig(f'{outdir}/mean_ap.svg')

    # ROC(n)
    rocn_cols = [f'roc_{n}' for n in ns]
    rocn_ran = np.arange(len(ns))
    fig, axes = plt.subplots(1, nl, figsize=(fw, 2), constrained_layout=True)
    for i, level in enumerate(levels):
        ax = axes[i]
        df = dfs[level]
        for j, method in enumerate(methods):
            offset = (j - 2) * 0.1
            ax.plot(np.arange(len(ns)) + offset, df.loc[method, rocn_cols],
                    color=palette[method], zorder=zorder[j], **margs)
        ax.set_xticks(rocn_ran, ns)
        ax.set_xmargin(0.1)
        ax.set_xlabel('n', fontstyle='italic')
        if i == 0:
            ax.set_ylabel(r'ROC ($\it{n}$)')
    fig.savefig(f'{outdir}/rocn.svg')

    # Hits @ K
    hits_cols = [f'hits_at_{k}' for k in ks]
    hits_ran = np.arange(len(ks))
    fig, axes = plt.subplots(
        1, nl, figsize=(fw, 2), sharey=True, constrained_layout=True)
    for i, level in enumerate(levels):
        ax = axes[i]
        df = dfs[level]
        for j, method in enumerate(methods):
            offset = (j - 2) * 0.1
            ax.plot(np.arange(len(ks)) + offset, df.loc[method, hits_cols],
                    color=palette[method], zorder=zorder[j], **margs)
        ax.set_xticks(hits_ran, ks)
        ax.set_xlabel('k', fontstyle='italic')
        if i == 0:
            ax.set_ylabel(r'Hits @ $\it{k}$')
    fig.savefig(f'{outdir}/hitsk.svg')


if __name__ == '__main__':
    main()
