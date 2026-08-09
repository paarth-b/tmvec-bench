#!/usr/bin/env python
"""
Generate ground truth of CATH S100 protein classifications.

Input:
    data/fasta/cath-domain-list-S100.txt: CATH domain classifications
    src/plotting/cath/domains.txt: list of protein domains to include

Output:
    src/plotting/cath/truth.tsv: Same classification unit (1) or not (0) of each
    pair of domains at each classification level.

Notes:
    CATH data files were retrieved from:
    https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/

    Specifically, domain classifications are defined in: cath-domain-list-S100.txt.
    Column definitions were adopted from: README-cath-list-file-format.txt.

Usage:
    python -m src.plotting.get_truth_cath
    python src/plotting/get_truth_cath.py
"""

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = {
    "ref_file": REPO_ROOT / "data/fasta/cath-domain-list-S100.txt",
    "domains": REPO_ROOT / "src/plotting/cath/domains.txt",
    "output": REPO_ROOT / "src/plotting/cath/truth.tsv",
}

# levels of protein structure classification
levels = ['class', 'architecture', 'topology', 'superfamily']

# additional column names in the data file
columns = levels + ['S35', 'S60', 'S95', 'S100', 'count', 'length', 'resolution']


def main():
    parser = argparse.ArgumentParser(description="Generate CATH ground truth classifications")
    parser.add_argument("--ref-file", default=str(DEFAULTS["ref_file"]),
                        help="Path to cath-domain-list-S100.txt")
    parser.add_argument("--domains", default=str(DEFAULTS["domains"]),
                        help="Path to domains.txt")
    parser.add_argument("--output", default=str(DEFAULTS["output"]),
                        help="Output truth.tsv path")
    args = parser.parse_args()

    # read reference data
    df = pd.read_csv(args.ref_file, sep=r'\s+', names=columns, index_col=0)

    # generate hierarchical classification units
    for level in levels:
        df[level] = df[level].astype(str)
    df['architecture'] = df['class'].str.cat(df['architecture'], sep='.')
    df['topology'] = df['architecture'].str.cat(df['topology'], sep='.')
    df['superfamily'] = df['topology'].str.cat(df['superfamily'], sep='.')

    # filter to tested domains
    with open(args.domains, 'r') as fh:
        domains = fh.read().splitlines()
    df = df.loc[domains, levels]
    df.index.name = 'seq'

    # convert units into codes
    codes = df.astype('category').apply(lambda s: s.cat.codes).to_numpy()

    # generate ground truths (a pair of sequences from the same unit or not)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as fh:
        print('seq_pair', *levels, sep='\t', file=fh)
        for i, j in combinations(range(len(domains)), 2):
            matches = (codes[i] == codes[j]).astype(int)
            print(f'{domains[i]},{domains[j]}', *matches, sep='\t', file=fh)

    print(f"Saved {len(domains) * (len(domains) - 1) // 2:,} pairs to {output_path}")


if __name__ == '__main__':
    main()
