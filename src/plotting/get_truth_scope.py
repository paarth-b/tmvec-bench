#!/usr/bin/env python
"""
Generate ground truth of SCOPe 40 protein classifications.

Input:
    data/fasta/dir.des.scope.2.01-stable.txt: SCOPe domain classifications
    src/plotting/scope/domains.txt: list of protein domains to include

Output:
    src/plotting/scope/truth.tsv: Same classification unit (1) or not (0) of each
    pair of domains at each classification level.

Notes:
    SCOPe sequence data (clustered at 40% sequence identity) were retrieved from:
    https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/

    Which is the data repository supporting the Foldseek paper:
    https://www.nature.com/articles/s41587-023-01773-0

    SCOPe 2.01 classifications were retrieved from the official SCOPe website:
    https://scop.berkeley.edu/downloads/parse/dir.des.scope.2.01-stable.txt

Usage:
    python -m src.plotting.get_truth_scope
    python src/plotting/get_truth_scope.py
"""

import argparse
import subprocess
import urllib.request
from itertools import combinations
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

SCOPE_URL = "https://scop.berkeley.edu/downloads/parse/dir.des.scope.2.01-stable.txt"

DEFAULTS = {
    "ref_file": REPO_ROOT / "data/fasta/dir.des.scope.2.01-stable.txt",
    "domains": REPO_ROOT / "src/plotting/scope/domains.txt",
    "output": REPO_ROOT / "src/plotting/scope/truth.tsv",
}

# levels of protein structure classification
levels = ['class', 'fold', 'superfamily', 'family']

# additional column names in the data file
columns = ['sunid', 'sid', 'family', 'domain', 'description']


def main():
    parser = argparse.ArgumentParser(description="Generate SCOPe ground truth classifications")
    parser.add_argument("--ref-file", default=str(DEFAULTS["ref_file"]),
                        help="Path to dir.des.scope.2.01-stable.txt (auto-downloaded if missing)")
    parser.add_argument("--domains", default=str(DEFAULTS["domains"]),
                        help="Path to domains.txt")
    parser.add_argument("--output", default=str(DEFAULTS["output"]),
                        help="Output truth.tsv path")
    args = parser.parse_args()

    ref_path = Path(args.ref_file)
    if not ref_path.exists():
        print(f"SCOPe classification file not found at {ref_path}, downloading...")
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        # Try multiple download methods (some environments have SSL cert issues)
        downloaded = False
        for cmd in [
            ["wget", "-q", "-O", str(ref_path), SCOPE_URL],
            ["wget", "-q", "--no-check-certificate", "-O", str(ref_path), SCOPE_URL],
            ["curl", "-ksfo", str(ref_path), SCOPE_URL],
        ]:
            try:
                subprocess.run(cmd, check=True)
                downloaded = True
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        if not downloaded:
            urllib.request.urlretrieve(SCOPE_URL, ref_path)
        print(f"Downloaded to {ref_path}")

    # read reference data
    df = pd.read_table(ref_path, names=columns, comment='#')
    df = df.query('domain != "-"').set_index('domain')

    # generate hierarchical classification units
    assert (df['family'].str.split('.').str.len() == 4).all()
    df['superfamily'] = df['family'].str.rsplit('.', n=1).str[0]
    df['fold'] = df['superfamily'].str.rsplit('.', n=1).str[0]
    df['class'] = df['fold'].str.rsplit('.', n=1).str[0]

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
