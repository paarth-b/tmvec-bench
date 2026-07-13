#!/usr/bin/env python
"""
Generate ground truth of SCOPe 40 protein classifications.

Input:
    domains.txt: list of protein domains to include

Output:
    truth.tsv: Same classification unit (1) or not (0) of each pair of domains at each
    classification level.

Notes:
    SCOPe sequence data (clustered at 40% sequence identity) were retrieved from:
    https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/

    Which is the data repository supporting the Foldseek paper:
    https://www.nature.com/articles/s41587-023-01773-0

    SCOPe 2.01 classifications were retrieved from the official SCOPe website:
    https://scop.berkeley.edu/downloads/parse/dir.des.scope.2.01-stable.txt

"""

from itertools import combinations
import pandas as pd


# path to SCOPe 40 domain list (adjust as needed)
refile = 'dir.des.scope.2.01-stable.txt'

# levels of protein structure classificaton
levels = ['class', 'fold', 'superfamily', 'family']

# additional column names in the data file
columns = ['sunid', 'sid', 'family', 'domain', 'description']


def main():
    # read reference data
    df = pd.read_table(refile, names=columns, comment='#')
    df = df.query('domain != "-"').set_index('domain')

    # generate hierarchical classification units
    assert (df['family'].str.split('.').str.len() == 4).all()
    df['superfamily'] = df['family'].str.rsplit('.', n=1).str[0]
    df['fold'] = df['superfamily'].str.rsplit('.', n=1).str[0]
    df['class'] = df['fold'].str.rsplit('.', n=1).str[0]

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
