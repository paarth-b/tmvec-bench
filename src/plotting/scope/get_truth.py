#!/usr/bin/env python
"""Generate ground truth matches between SCOPe 40 proteins.

Notes:
    SCOPe sequence data (clustered at 40% sequence identity) were retrieved from:
    https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/

    SCOPe 2.01 classifications were retrieved from the official SCOPe website:
    https://scop.berkeley.edu/downloads/parse/dir.des.scope.2.01-stable.txt
"""

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Generate SCOPe ground truth")
    parser.add_argument("--data-dir", default=str(Path(__file__).parent),
                        help="Directory containing dir.des.scope.2.01-stable.txt and domain.lst")
    parser.add_argument("--output", default=None, help="Output path (default: {data-dir}/truth.tsv)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output = Path(args.output) if args.output else data_dir / "truth.tsv"

    levels = ['class', 'fold', 'superfamily', 'family']
    columns = ['sunid', 'sid', 'family', 'domain', 'description']
    df = pd.read_table(data_dir / 'dir.des.scope.2.01-stable.txt', names=columns, comment='#')
    df = df.query('domain != "-"').set_index('domain')

    assert (df['family'].str.split('.').str.len() == 4).all()
    df['superfamily'] = df['family'].str.rsplit('.', n=1).str[0]
    df['fold'] = df['superfamily'].str.rsplit('.', n=1).str[0]
    df['class'] = df['fold'].str.rsplit('.', n=1).str[0]

    domains = (data_dir / 'domain.lst').read_text().splitlines()
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
