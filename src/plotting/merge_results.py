#!/usr/bin/env python
"""
Merge benchmark results from multiple methods into a single table.

Input:
    results/{dataset}_{method}_similarities.csv
    src/plotting/{dataset_dir}/domains.txt
    src/plotting/{dataset_dir}/truth.tsv

Output:
    src/plotting/{dataset_dir}/results.tsv

Usage:
    python -m src.plotting.merge_results --dataset cath
    python -m src.plotting.merge_results --dataset scope40
"""

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIRS = {
    "cath": "cath",
    "scope40": "scope",
}

# methods to be compared (file name -> display name)
METHODS = {
    "tmvec1": "tmvec1",
    "tmvec2": "tmvec2",
    "tmvec2_student": "tmvec2s",
    "tmalign": "tmalign",
    "foldseek": "foldseek",
    "plmblast": "plmblast",
}


def main():
    parser = argparse.ArgumentParser(description="Merge benchmark results into one table")
    parser.add_argument("--dataset", choices=DATASET_DIRS.keys(), required=True,
                        help="Dataset name")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"),
                        help="Directory containing per-method CSV files")
    parser.add_argument("--output", default=None,
                        help="Output results.tsv path (auto-detected if not specified)")
    args = parser.parse_args()

    dataset_dir = DATASET_DIRS[args.dataset]
    plot_dir = REPO_ROOT / "src" / "plotting" / dataset_dir
    domains_file = plot_dir / "domains.txt"
    truth_file = plot_dir / "truth.tsv"
    output_file = Path(args.output) if args.output else plot_dir / "results.tsv"

    # read domains
    with open(domains_file, 'r') as fh:
        domains = set(fh.read().splitlines())

    dfs = []
    for fname, display_name in METHODS.items():
        csv_path = Path(args.results_dir) / f"{args.dataset}_{fname}_similarities.csv"
        if not csv_path.exists():
            print(f"  Skipping {fname}: {csv_path} not found")
            continue

        print(f"  Loading {fname}...")
        df = pd.read_csv(csv_path)

        # drop non-existent domains
        df.query('seq1_id in @domains & seq2_id in @domains', inplace=True)

        # rename TM-score column to method display name
        df.rename(columns={'tm_score': display_name}, inplace=True)

        # sort and merge domain IDs
        df['seq_pair'] = df.apply(lambda row: ','.join(sorted(
            [row['seq1_id'], row['seq2_id']])), axis=1)
        df.drop(['seq1_id', 'seq2_id'], axis=1, inplace=True)
        df.set_index('seq_pair', inplace=True)

        dfs.append(df)

    if not dfs:
        raise ValueError(f"No result files found in {args.results_dir} for dataset '{args.dataset}'")

    # combine results
    merged = pd.concat(dfs, axis=1)

    # append ground truth
    truth = pd.read_table(truth_file, index_col=0)
    res = pd.concat([truth, merged], axis=1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(output_file, sep='\t')
    print(f"Saved merged table ({len(res):,} pairs, {len(res.columns)} columns) to {output_file}")


if __name__ == '__main__':
    main()
