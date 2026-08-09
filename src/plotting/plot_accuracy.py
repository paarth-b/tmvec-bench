#!/usr/bin/env python
"""
Plot TM-score prediction accuracy results.

Input:
    src/plotting/{dataset_dir}/results.tsv  (from merge_results.py)

Output:
    src/plotting/{dataset_dir}/plots/*.svg
    src/plotting/{dataset_dir}/metrics.tsv

Usage:
    python -m src.plotting.plot_accuracy --dataset cath
    python -m src.plotting.plot_accuracy --dataset scope40
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import average_precision_score

plt.rcParams['svg.fonttype'] = 'none'

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIRS = {
    "cath": "cath",
    "scope40": "scope",
}

methods = ['tmvec1', 'tmvec2', 'tmvec2s', 'tmalign']
names = ['TM-Vec', 'TM-Vec 2', 'TM-Vec 2s', 'TM-align']
name_map = {method: name for method, name in zip(methods, names)}

tmvecs = ['tmvec1', 'tmvec2', 'tmvec2s']

cmap = plt.get_cmap('tab10')
palette = {method: cmap(i) for i, method in enumerate(methods)}
cmaps = ['Blues', 'Oranges', 'Greens']

bins = [-np.inf, 0.17, 0.3, 0.4, 0.5, 0.6, np.inf]


def main():
    parser = argparse.ArgumentParser(description="Plot TM-score prediction accuracy")
    parser.add_argument("--dataset", choices=DATASET_DIRS.keys(), required=True)
    parser.add_argument("--input", default=None, help="Input results.tsv (auto-detected if not specified)")
    parser.add_argument("--output-dir", default=None, help="Output plots directory (auto-detected if not specified)")
    args = parser.parse_args()

    dataset_dir = DATASET_DIRS[args.dataset]
    plot_dir = REPO_ROOT / "src" / "plotting" / dataset_dir
    infile = Path(args.input) if args.input else plot_dir / "results.tsv"
    outdir = Path(args.output_dir) if args.output_dir else plot_dir / "plots"

    if not infile.exists():
        raise FileNotFoundError(
            f"Input file not found: {infile}\n"
            f"Run merge_results first: python -m src.plotting.merge_results --dataset {args.dataset}"
        )

    outdir.mkdir(parents=True, exist_ok=True)

    # Read combined results, keeping only columns that exist
    df_all = pd.read_table(infile, index_col=0)
    available = [m for m in methods if m in df_all.columns]
    tmvecs_avail = [m for m in tmvecs if m in df_all.columns]
    if 'tmalign' not in available:
        raise ValueError("tmalign column required in results.tsv")
    df = df_all[available].dropna(how='any')

    metrics = {method: {} for method in tmvecs_avail}

    # Correlation between true and predicted TM-scores
    grids = np.arange(0, 1.015, 0.015)
    text = '$\\it{{r}}$ = {:.3f}\n$\\rho$ = {:.3f}\nMAE = {:.3f}\nRMSE = {:.3f}'
    x = df['tmalign'].to_numpy()
    n_tmvecs = len(tmvecs_avail)
    fig, axes = plt.subplots(1, n_tmvecs, figsize=(4 * n_tmvecs, 3.5), sharey=True)
    if n_tmvecs == 1:
        axes = [axes]
    for i, method in enumerate(tmvecs_avail):
        ax = axes[i]
        name = name_map[method]
        color = palette[method]
        metric = metrics[method]

        y = df[method].to_numpy()
        r, r_p = pearsonr(x, y)
        rho, rho_p = spearmanr(x, y)
        diff = x - y
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))

        metric['pearson_r'] = r
        metric['pearson_pval'] = r_p
        metric['spearman_rho'] = rho
        metric['spearman_pval'] = rho_p
        metric['mae'] = mae
        metric['rmse'] = rmse

        sns.histplot(data=df, x='tmalign', y=method, ax=ax, bins=(
            grids, grids), color=color)
        sns.regplot(data=df, x='tmalign', y=method, ax=ax, scatter=False, ci=None,
                    color='r')

        ax.plot((0, 1), (0, 1), 'k--', alpha=0.5)
        ax.text(0.95, 0.05, text.format(r, rho, mae, rmse), transform=ax.transAxes,
                ha='right', va='bottom')
        ax.set_xlabel('True TM-score')
        if i == 0:
            ax.set_ylabel('Predicted TM-score')
        ax.set_title(name)
    fig.savefig(outdir / 'correlation.svg')

    # Distribution of true and predicted TM-scores
    df_melt = df.melt(var_name='Method', value_name='TM-score')
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.kdeplot(data=df_melt, x='TM-score', hue='Method', fill=True, alpha=0.25, ax=ax)
    for txt in ax.legend_.texts:
        txt.set_text(name_map.get(txt.get_text(), txt.get_text()))
    fig.savefig(outdir / 'distribution.svg')

    # Distribution of TM-score prediction errors
    df_err = df[tmvecs_avail].sub(df['tmalign'], axis=0).abs()
    df_err_melt = df_err.melt(var_name='method', value_name='error')
    fig, ax = plt.subplots(figsize=(2.5 * n_tmvecs, 3.5))
    sns.violinplot(data=df_err_melt, x='method', y='error', hue='method',
                   order=tmvecs_avail, fill=False, ax=ax)
    ax.set_ylabel('Prediction error (abs.)')
    ax.set_xlabel('Method')
    ax.set_xticks(range(n_tmvecs), [name_map[x] for x in tmvecs_avail])
    fig.savefig(outdir / 'error.svg')

    # Convert continuous TM-scores into discrete ranges
    n_bins = len(bins) - 1
    ranges = np.arange(n_bins - 1) + 0.5
    bounds = bins[1:-1]
    df_bin = df.apply(lambda col: pd.cut(col, bins=bins).cat.codes, axis=0)

    # Distribution of TM-score prediction errors by TM-score ranges
    df_err['bin'] = df_bin['tmalign']
    df_err_bin = df_err.melt(id_vars='bin', value_vars=tmvecs_avail, var_name='method',
                             value_name='error')
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.boxplot(data=df_err_bin, x='bin', y='error', hue='method', hue_order=tmvecs_avail,
                fill=False, showfliers=False, gap=0.2, legend=False)
    ax.set_xticks(ranges, bounds)
    ax.set_xlabel('TM-score range')
    ax.set_ylabel('Prediction error (abs.)')
    fig.savefig(outdir / 'error_bin.svg')

    # Confusion matrices
    y_true = df_bin['tmalign'].to_numpy()
    fig, axes = plt.subplots(1, n_tmvecs, figsize=(3.5 * n_tmvecs, 3.5))
    if n_tmvecs == 1:
        axes = [axes]
    for i, method in enumerate(tmvecs_avail):
        ax = axes[i]
        name = name_map[method]
        y_pred = df_bin[method].to_numpy()
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        ConfusionMatrixDisplay(cm).plot(
            cmap=cmaps[i], colorbar=False, ax=ax)
        ax.set_xlabel('Predicted TM-score range')
        ax.set_xticks(ranges, bounds)
        if i > 0:
            ax.set_ylabel(None)
            ax.set_yticks(ranges, [])
        else:
            ax.set_ylabel('True TM-score range')
            ax.set_yticks(ranges, bounds)
        ax.set_title(name)

        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        metrics[method]['qwk'] = qwk

    fig.tight_layout()
    fig.savefig(outdir / 'confusion.svg')

    # Average precision above threshold
    data = []
    for bound in bounds:
        df_cls = df >= bound
        y_true = df_cls['tmalign'].to_numpy()
        row = []
        for method in tmvecs_avail:
            y_pred = df_cls[method].to_numpy()
            ap = average_precision_score(y_true, y_pred)
            row.append(ap)
            metrics[method][f'ap_above_{bound}'] = ap
        data.append(row)
    dfa = pd.DataFrame(data, index=bounds, columns=tmvecs_avail).T
    dfa_melt = dfa.reset_index().melt(id_vars='index')

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bp = sns.barplot(data=dfa_melt, x='variable', y='value', hue='index', saturation=1,
                     errorbar=None, ax=ax)
    handles, _ = bp.get_legend_handles_labels()
    ax.legend(handles, [name_map[x] for x in tmvecs_avail], title='Method')
    ax.set_xlabel('Min. TM-score')
    ax.set_ylabel('Average precision')
    ax.set_title('AP above threshold')
    fig.savefig(outdir / 'ap_above.svg')

    # Metrics
    dfm = pd.DataFrame(metrics).T
    dfm.to_csv(plot_dir / 'metrics.tsv', sep='\t')
    print(f"Saved plots to {outdir} and metrics to {plot_dir / 'metrics.tsv'}")


if __name__ == '__main__':
    main()
