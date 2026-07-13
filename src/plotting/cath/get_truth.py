#!/usr/bin/env python
"""Generate ground truth matches between CATH S100 proteins.

Notes:
    CATH data files were retrieved from:
    ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_4_0/

    Specifically, domain classifications are defined in: cath-domain-list-S100.txt.
    Column definitions were adopted from: README-cath-list-file-format.txt.
    domain.lst stores the first 1000 domains extracted from: cath-domain-seqs-S100.fa.
"""

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Generate CATH ground truth")
    parser.add_argument("--data-dir", default=str(Path(__file__).parent),
                        help="Directory containing cath-domain-list-S100.txt and domain.lst")
    parser.add_argument("--output", default=None, help="Output path (default: {data-dir}/truth.tsv)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output = Path(args.output) if args.output else data_dir / "truth.tsv"

    levels = ['class', 'architecture', 'topology', 'superfamily']
    columns = levels + ['S35', 'S60', 'S95', 'S100', 'count', 'length', 'resolution']
    df = pd.read_csv(data_dir / 'cath-domain-list-S100.txt', sep=r'\s+', names=columns, index_col=0)

    for level in levels:
        df[level] = df[level].astype(str)
    df['architecture'] = df['class'].str.cat(df['architecture'], sep='.')
    df['topology'] = df['architecture'].str.cat(df['topology'], sep='.')
    df['superfamily'] = df['topology'].str.cat(df['superfamily'], sep='.')

    domains = (data_dir / 'domain.lst').read_text().splitlines()[::2]
    df = df.loc[domains]

    with open(output, 'w') as fh:
        print('a', 'b', *levels, sep='\t', file=fh)
        for a, b in combinations(domains, 2):
            out = [a, b]
            for level in levels:
                out.append(str(int(df.loc[a, level] == df.loc[b, level])))
            print(*out, sep='\t', file=fh)

    print(f"Saved {output}")


if __name__ == '__main__':
    main()
