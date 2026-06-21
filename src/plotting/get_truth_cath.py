#!/usr/bin/env python
"""
Generate ground truth of CATH S100 protein classifications.

Input:
    domains.txt: list of protein domains to include

Output:
    truth.tsv: Same classification unit (1) or not (0) of each pair of domains at each
    classification level.

Notes:
    CATH data files were retrieved from:
    https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/

    Specifically, domain classifications are defined in: cath-domain-list-S100.txt.
    Column definitions were adopted from: README-cath-list-file-format.txt.
    domain.lst stores the first 1000 domains extracted from: cath-domain-seqs-S100.fa.

"""

from itertools import combinations
import pandas as pd


# path to CATH S100 domain list (adjust as needed)
refile = 'cath-domain-list-S100.txt'

# levels of protein structure classificaton
levels = ['class', 'architecture', 'topology', 'superfamily']

# additional column names in the data file
columns = levels + ['S35', 'S60', 'S95', 'S100', 'count', 'length', 'resolution']


def main():
    # read reference data
    df = pd.read_csv(refile, sep=r'\s+', names=columns, index_col=0)

    # generate hierarchical classification units
    for level in levels:
        df[level] = df[level].astype(str)
    df['architecture'] = df['class'].str.cat(df['architecture'], sep='.')
    df['topology'] = df['architecture'].str.cat(df['topology'], sep='.')
    df['superfamily'] = df['topology'].str.cat(df['superfamily'], sep='.')

    # filter to tested domains
    with open('domains.txt', 'r') as fh:
        domains = fh.read().splitlines()
    df = df.loc[domains, levels]
    df.index.name = 'seq'

    # calculate unit sizes
    # sizes = df.apply(lambda x: x.map(x.value_counts()))
    # sizes.to_csv('sizes.tsv', sep='\t')

    # convert units into codes
    codes = df.astype('category').apply(lambda s: s.cat.codes).to_numpy()

    # generate groud truths (a pair of sequences from the same unit or not)
    with open('truth.tsv', 'w') as fh:
        print('seq_pair', *levels, sep='\t', file=fh)
        for i, j in combinations(range(len(domains)), 2):
            matches = (codes[i] == codes[j]).astype(int)
            print(f'{domains[i]},{domains[j]}', *matches, sep='\t', file=fh)


if __name__ == '__main__':
    main()
