#!/usr/bin/env python
"""
Plot TM-score prediction accuracy results.

Input:
    results.tsv

Output:
    metrics.tsv
    plots/*.svg

"""

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

infile = 'results.tsv'
outdir = 'plots'

methods = ['tmvec1', 'tmvec2s', 'tmalign']
names = ['TM-Vec', 'TM-Vec 2s', 'TM-align']
name_map = {method: name for method, name in zip(methods, names)}

tmvecs = ['tmvec1', 'tmvec2s']

cmap = plt.get_cmap('tab10')
palette = {method: cmap(i) for i, method in enumerate(methods)}
cmaps = ['Blues', 'Oranges']

bins = [-np.inf, 0.17, 0.3, 0.4, 0.5, 0.6, np.inf]


def main():
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Read combined results
    df = pd.read_table(infile, index_col=0, usecols=['seq_pair'] + methods)
    df.dropna(how='any', inplace=True)

    metrics = {method: {} for method in tmvecs}

    # Correlation between true and predicted TM-scores
    grids = np.arange(0, 1.015, 0.015)
    text = '$\\it{{r}}$ = {:.3f}\n$\\rho$ = {:.3f}\nMAE = {:.3f}\nRMSE = {:.3f}'
    x = df['tmalign'].to_numpy()
    fig, axes = plt.subplots(1, len(tmvecs), figsize=(8, 3.5), sharey=True)
    for i, method in enumerate(tmvecs):
        ax = axes[i]
        name = name_map[method]
        color = palette[method]
        metric = metrics[method]

        # Statistics
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

        # Correlation plot
        sns.histplot(data=df, x='tmalign', y=method, ax=ax, bins=(
            grids, grids), color=color)
        sns.regplot(data=df, x='tmalign', y=method, ax=ax, scatter=False, ci=None,
                    color='r')

        ax.plot((0, 1), (0, 1), 'k--', alpha=0.5)
        ax.text(0.95, 0.05, text.format(r, rho, mae, rmse), transform=ax.transAxes,
                ha='right', va='bottom')
        ax.set_xlabel('True TM-score')
        if i == 0:
            ax.set_ylabel(f'Predicted TM-score')
        ax.set_title(name)
    fig.savefig(f'plots/correlation.svg')

    # Distribution of true and predicted TM-scores
    df_melt = df.melt(var_name='Method', value_name='TM-score')
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.kdeplot(data=df_melt, x='TM-score', hue='Method', fill=True, alpha=0.25, ax=ax)
    for text in ax.legend_.texts:
        text.set_text(name_map[text.get_text()])
    fig.savefig('plots/distribution.svg')

    # Distribution of TM-score prediction errors
    df_err = df[tmvecs].sub(df['tmalign'], axis=0).abs()
    df_err_melt = df_err.melt(var_name='method', value_name='error')
    fig, ax = plt.subplots(figsize=(2.5, 3.5))
    sns.violinplot(data=df_err_melt, x='method', y='error', hue='method',
                   order=tmvecs, fill=False, ax=ax)
    ax.set_ylabel('Prediction error (abs.)')
    ax.set_xlabel('Method')
    ax.set_xticks(range(len(tmvecs)), [name_map[x] for x in tmvecs])
    fig.savefig('plots/error.svg')

    # Convert continuous TM-scores into discrete ranges
    n_bins = len(bins) - 1
    ranges = np.arange(n_bins - 1) + 0.5
    bounds = bins[1:-1]
    df_bin = df.apply(lambda col: pd.cut(col, bins=bins).cat.codes, axis=0)

    # Distribution of TM-score prediction errors by TM-score ranges
    df_err['bin'] = df_bin['tmalign']
    df_err_bin = df_err.melt(id_vars='bin', value_vars=tmvecs, var_name='method',
                             value_name='error')
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.boxplot(data=df_err_bin, x='bin', y='error', hue='method', hue_order=tmvecs,
                fill=False, showfliers=False, gap=0.2, legend=False)
    ax.set_xticks(ranges, bounds)
    ax.set_xlabel('TM-score range')
    ax.set_ylabel('Prediction error (abs.)')
    fig.savefig('plots/error_bin.svg')

    # Confusion matrics
    y_true = df_bin['tmalign'].to_numpy()
    fig, axes = plt.subplots(1, len(tmvecs), figsize=(7, 3.5))
    for i, method in enumerate(tmvecs):
        ax=axes[i]
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

        # quadratic weighted kappa
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        metrics[method]['qwk'] = qwk

    fig.tight_layout()
    fig.savefig('plots/confusion.svg')

    # Average precision above threshold
    data = []
    for bound in bounds:
        df_cls = df >= bound
        y_true = df_cls['tmalign'].to_numpy()
        row = []
        for method in tmvecs:
            y_pred = df_cls[method].to_numpy()
            ap = average_precision_score(y_true, y_pred)
            row.append(ap)
            metrics[method][f'ap_above_{bound}'] = ap
        data.append(row)
    dfa = pd.DataFrame(data, index=bounds, columns=tmvecs).T
    dfa_melt = dfa.reset_index().melt(id_vars='index')

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bp = sns.barplot(data=dfa_melt, x='variable', y='value', hue='index', saturation=1,
                     errorbar=None, ax=ax)
    handles, _ = bp.get_legend_handles_labels()
    ax.legend(handles, [name_map[x] for x in tmvecs], title='Method')
    ax.set_xlabel('Min. TM-score')
    ax.set_ylabel('Average precision')
    ax.set_title('AP above threshold')
    fig.savefig('plots/ap_above.svg')

    # Metrics
    dfm = pd.DataFrame(metrics).T
    dfm.to_csv('metrics.tsv', sep='\t')


if __name__ == '__main__':
    main()
