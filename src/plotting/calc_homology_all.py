#!/usr/bin/env python
"""
Calculate homology detection metrics by classification level and method

Note:
    All protein domain pairs are included in the analysis.

Input:
    results.tsv

Output:
    metrics/counts.tsv: numbers of queries and pairs per level
    metrics/{level}.tsv: performance metrics per method per level
    metrics/*.npy: ROC and PR curves per method

"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
)


# Classification levels (from broad to narrow)
levels = ['class', 'architecture', 'topology', 'superfamily']  # CATH
# levels = ['class', 'fold', 'superfamily', 'family']  # SCOPe

# Methods to compare
methods = ['tmvec1', 'tmvec2s', 'tmalign', 'foldseek']

# n-values for calculating ROC(n)
ns = np.array([1, 5, 10])

# k-values for calculating precision @ K and hits @ K
ks = np.array([1, 5, 10, 50, 100])


def main():
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Read combined results
    df = pd.read_table('results.tsv', index_col=0)
    print('Total hit count', df.shape[0], sep=': ')

    # Separate seq1 (query) and seq2 (subject)
    # Note: seq1 and seq2 are always in lexicographical order.
    df.index = df.index.str.split(',', expand=True)
    df.index.names = ['seq1', 'seq2']
    df.sort_index(inplace=True)

    # Check if all hits have classifications
    assert df[levels].notna().all().all()

    # Per-method hit count
    print('Per-method hit count:')
    for method in methods:
        print(method, df[method].notna().sum(), sep=': ')

    # Confirm that all reported TM-scores are > -1 (theoretically they should be > 0,
    # but there could be numerical issues), which will justify setting missing hits to
    # -1 later to represent worse-than-all hits.
    assert not (df[methods] <= -1).any().any()

    # Duplicate and swap reciprocal hits. This assumes that all methods are symmetric
    # (i.e., TM-score of A -> B equals to that of B -> A).
    df = pd.concat([df, df.swaplevel()]).sort_index()
    assert not df.index.has_duplicates

    # Number of positives per query. This value equals to the size of the unit - 1.
    grouped = df[levels].groupby(level=0, group_keys=False)
    dfp = grouped[levels].sum()

    # Number of negatives per query. This value equals to the total number of hits
    # outside the unit.
    dfn = -dfp.sub(grouped.size(), axis=0)

    # Subtract the positive count of each level from that of its parent level, except
    # for the lowest level. This value will be used for weighing.
    dfp = dfp.assign(base=0).diff(periods=-1, axis=1).drop(columns='base')

    # Perform tests at each level
    counts = []
    for i, level in enumerate(levels):
        print('Level', level, sep=': ')

        # Identify queries that have at least one positive and one negative.
        queries = dfp.index[(dfp[level] > 0) & (dfn[level] > 0)]
        df_valid = df.loc[queries,]
        print('Queries', n_queries := queries.shape[0], sep=': ')

        # Exclude subjects within the same child unit per query (except for the lowest
        # level).
        if i < len(levels) - 1:
            df_valid.query(f'{levels[i + 1]} == 0', inplace=True)

        print('Hits', n_hits := df_valid.shape[0], sep=': ')
        counts.append({'level': level, 'queries': n_queries, 'hits': n_hits})

        # Calculate weights (1 / positive count)
        weights = 1.0 / dfp.loc[queries, level]

        results = {}
        for method in methods:
            print('Method', method, sep=': ')
            result = {}

            # Get ground truth classifications and TM-scores predicted/calculated by
            # each method. At this point, all hits (query-subject pairs) are included.
            y_true = df_valid[level].to_numpy()

            # Replace NaN with -1, such that unreported hits represent hits worse than
            # all reported hits by each method.
            y_score = df_valid[method].fillna(-1).to_numpy()

            # Global AUROC (i.e., all query-subject pairs are pooled)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            save_curve(fpr, tpr, f'roc_{level}_{method}')
            result['auroc'] = auc(fpr, tpr)

            # Global average precision (AP)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            save_curve(precision, recall, f'pr_{level}_{method}')
            result['ap'] = ap(precision, recall)

            # Size-weighted global AUROC and AP
            w = weights.loc[df_valid.index.get_level_values(0)].to_numpy()

            fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=w)
            save_curve(fpr, tpr, f'weighted_roc_{level}_{method}')
            result['weighted_auroc'] = auc(fpr, tpr)

            precision, recall, _ = precision_recall_curve(
                y_true, y_score, sample_weight=w)
            save_curve(precision, recall, f'weighted_pr_{level}_{method}')
            result['weighted_ap'] = ap(precision, recall)

            # Sort all TM-scores once to facilitate all downstream calculations.
            df_sorted = df_valid[[level, method]].sort_values(
                method, ascending=False, kind='stable', na_position='last')

            # Group by query
            grouped = df_sorted.fillna(-1).groupby(level=0, group_keys=False)

            # Mean AUROC and AP over queries
            aurocs = grouped.apply(lambda x: roc_auc_score(x[level], x[method]))
            result['mean_auroc'] = aurocs.mean()

            aps = grouped.apply(lambda x: average_precision_score(x[level], x[method]))
            result['mean_ap'] = aps.mean()

            # Drop NaN scores (rather than setting to 0) for downstream calculations
            grouped = df_sorted.dropna(how='any').groupby(level=0, group_keys=False)

            # ROC_n (sensitivity up to the n-th false positive (FP))
            # For each query, ROC_n = TPs before n-th FP / total TPs. Queries without any
            # hit are set to 0. Then average across queries.
            rocns = grouped.apply(tp_before_ns, level=level, ns=ns).reindex(
                queries, fill_value=0).div(dfp.loc[queries, level], axis=0)
            result.update(rocns.mean(axis=0).to_dict())

            # Precision @ K (TP of top k hits / k) and Hits @ K (top k hits contain at
            # least one TP). Averaged across queries.
            tps = grouped.apply(tp_at_ks, level=level, ks=ks).reindex(
                queries, fill_value=0)

            hits_at_k = (tps > 0).mean(axis=0)
            hits_at_k.index = [f'hits_at_{k}' for k in ks]
            result.update(hits_at_k.to_dict())

            # Note: Precision@1 == Hits@1
            precision_at_k = tps.div(ks, axis=1).mean(axis=0).iloc[1:]
            precision_at_k.index = [f'precision_at_{k}' for k in ks[1:]]
            result.update(precision_at_k.to_dict())

            results[method] = result

        dfr = pd.DataFrame(results).T
        dfr.to_csv(f'{level}.tsv', sep='\t', float_format='%.7g')

    dfc = pd.DataFrame(counts)
    dfc.to_csv('counts.tsv', sep='\t', index=False)


def save_curve(x, y, fname):
    """Save curve to file."""
    data = np.column_stack((x, y)).astype('float32').round(5)
    np.save(f'{outdir}/{fname}.npy', data)


def ap(precision, recall):
    """Calculate average precision.

    This function generates the same result as `average_precision_score`, based on the
    output of `precision_recall_curve`. Whereas `auc` doesn't.

    """
    return max(0.0, -np.sum(np.diff(recall) * precision[:-1]))


def tp_before_ns(group, level, ns):
    """Calculate number of TPs before hitting the n-th FP for multiple n's.

    First get all FP positions (value = 0). For n > total FP count, n-th FP
    won't be reached, so just return total TP count. For other n's, at n-th
    FP position p, there are p preceding hits, in which n - 1 are FPs, and
    p - (n - 1) are TPs.

    """
    vals = group[level].to_numpy()
    fp_pos = np.flatnonzero(vals == 0)
    has_n = ns <= fp_pos.size
    idx = ns[has_n] - 1
    tps = np.full(len(ns), vals.sum())
    tps[has_n] = fp_pos[idx] - idx
    return pd.Series(tps, index=[f'roc_{n}' for n in ns])


def tp_at_ks(group, level, ks):
    """Calculate number of TPs within the top k hits."""
    return pd.Series([group[level].iloc[:k].sum() for k in ks])


if __name__ == '__main__':
    main()
