#!/usr/bin/env python
"""
Merge test results by multiple models.

Input:
    results/
    domains.txt
    truth.tsv

Output:
    results.tsv

"""

import pandas as pd


# test dataset name (adjust as needed)
key = 'cath'
# key = 'scope40'

# directory of test results (adjust as needed)
resdir = 'results'

# methods to be compared
methods = ['tmvec1', 'tmvec2s', 'tmalign', 'foldseek']


def main():
    # read domains
    with open('domains.txt', 'r') as fh:
        domains = set(fh.read().splitlines())

    dfs = []
    for method in methods:
        # rename "tmvec2_student" as "tmvec2s" for conciseness.
        fname = 'tmvec2_student' if method == 'tmvec2s' else method

        # read test results
        df = pd.read_csv(f'{resdir}/{key}_{fname}_similarities.csv.xz')

        # drop non-existent domains
        # NOTE: Several domains were broken down into chains in the Foldseek analysis,
        #   and they cannot be compared with other results. Remove them.
        df.query('seq1_id in @domains & seq2_id in @domains', inplace=True)

        # rename TM-score column to method name
        # NOTE: Foldseek additionally reports E-values. They are the last column.
        df.rename(columns={'tm_score': method}, inplace=True)

        # sort and merge domain IDs
        df['seq_pair'] = df.apply(lambda row: ','.join(sorted(
            [row['seq1_id'], row['seq2_id']])), axis=1)
        df.drop(['seq1_id', 'seq2_id'], axis=1, inplace=True)
        df.set_index('seq_pair', inplace=True)

        dfs.append(df)

    # combine results
    merged = pd.concat(dfs, axis=1)

    # append ground truth
    truth = pd.read_table('truth.tsv', index_col=0)
    res = pd.concat([truth, merged], axis=1)

    res.to_csv('results.tsv', sep='\t')


if __name__ == '__main__':
    main()
