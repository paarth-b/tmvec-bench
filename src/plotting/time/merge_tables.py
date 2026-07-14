#!/usr/bin/env python
"""
Generate two tables with combined encoding and query times, respectively.
"""

import pandas as pd


# Path to the result directory. Adjust as needed.
resdir = 'results'

# Methods being compared.
methods = ['tmvec1', 'tmvec2', 'tmvec2_student', 'foldseek', 'foldseek_gpu', 'prostt5_gpu', 'diamond']


def main():
    # Encoding
    dfs = []
    for method in methods:
        if method == 'diamond':
            continue
        dfs.append(pd.read_csv(f'{resdir}/{method}_encoding_times.csv')[[
            'encoding_size', 'mean_seconds']].assign(method=method))
    df = pd.concat(dfs, axis=0)
    df.to_csv('encoding.tsv', sep='\t', index=False)

    # Query
    dfs = []
    for method in methods:
        df = pd.read_csv(f'{resdir}/{method}_query_times.csv')
        if method == 'diamond':
            df.rename(columns={'search_mean': 'total_mean'}, inplace=True)
        dfs.append(df[['query_size', 'database_size', 'total_mean']].assign(method=method))
    df = pd.concat(dfs, axis=0)
    df.to_csv('query.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
