#!/usr/bin/env python
"""
Foldseek Benchmark: Generate pairwise TM-score predictions for protein structures.
"""

from pathlib import Path
import argparse
import subprocess
import pandas as pd
import tempfile
import os

DATASETS = {
    "cath": {
        "structure_dir": "data/pdb/cath-s100",
        "output": "results/cath_foldseek_similarities.csv",
    },
    "scope40": {
        "structure_dir": "data/pdb/scope40",
        "output": "results/scope40_foldseek_similarities.csv",
    },
}


def get_pdb_files(structure_dir):
    """Return sorted PDB file paths from the structure directory."""
    with os.scandir(structure_dir) as entries:
        pdb_files = [Path(e.path) for e in entries if e.is_file()]
    pdb_files.sort()
    print(f"Found {len(pdb_files)} structure files")
    return pdb_files


def run_foldseek(structure_dir, foldseek_bin, threads):
    """Run Foldseek all-vs-all search."""
    print("Running Foldseek all-vs-all search...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tsv_path = Path(tmp_dir) / "results.tsv"

        cmd = [
            foldseek_bin, "easy-search",
            structure_dir, structure_dir,
            str(tsv_path), tmp_dir,
            "--exhaustive-search", "1",
            "--format-output", "query,target,alntmscore,evalue",
            "--threads", str(threads),
            "--gpu", "1",
            "-e", "10",
            "--max-seqs", "100000",
            "--min-ungapped-score", "0"
        ]

        # Don't capture output, so Foldseek progress stays visible.
        result = subprocess.run(cmd)

        if result.returncode != 0:
            raise RuntimeError("Foldseek failed")

        # Read results before tmp_dir is deleted.
        df = pd.read_csv(tsv_path, sep='\t', header=None,
                        names=['query', 'target', 'alntmscore', 'evalue'],
                        low_memory=False)

    print(f"Loaded {len(df)} alignments")
    return df


def parse_results(df):
    """Extract unique pairs and average the two bidirectional scores.

    Vectorized so it scales to millions of alignment rows.
    """
    print("Parsing results...")

    # Extract the domain ID from each file path.
    df = df.copy()
    df['q_id'] = df['query'].str.extract(r'/([^/]+)\.[^.]+$')[0]
    df['t_id'] = df['target'].str.extract(r'/([^/]+)\.[^.]+$')[0]

    # Fall back to a plain filename when there was no path to match.
    mask_q = df['q_id'].isna()
    mask_t = df['t_id'].isna()
    if mask_q.any():
        df.loc[mask_q, 'q_id'] = df.loc[mask_q, 'query'].str.replace(r'\.[^.]+$', '', regex=True)
    if mask_t.any():
        df.loc[mask_t, 't_id'] = df.loc[mask_t, 'target'].str.replace(r'\.[^.]+$', '', regex=True)

    # Strip any _MODEL_* suffix.
    df['q_id'] = df['q_id'].str.split('_MODEL_').str[0]
    df['t_id'] = df['t_id'].str.split('_MODEL_').str[0]

    df = df[df['q_id'] != df['t_id']]
    print(f"Processing {len(df):,} non-self alignments...")

    # Canonical pair key so (A,B) and (B,A) collapse together.
    df['seq1_id'] = df[['q_id', 't_id']].min(axis=1)
    df['seq2_id'] = df[['q_id', 't_id']].max(axis=1)

    print("Aggregating bidirectional scores...")
    result_df = df.groupby(['seq1_id', 'seq2_id']).agg(
        tm_score=('alntmscore', 'mean'),
        evalue=('evalue', 'min')
    ).reset_index()

    print(f"Extracted {len(result_df):,} unique pairs")
    return result_df.to_dict('records')


def save_results(pairs, output_path):
    """Save results to CSV."""
    df = pd.DataFrame(pairs)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(pairs):,} pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Foldseek benchmark")
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="cath",
                        help="Dataset to use (cath or scope40)")
    parser.add_argument("--structure-dir", default=None,
                        help="PDB structure directory (overrides dataset default)")
    parser.add_argument("--output", default=None, help="Output CSV path (overrides dataset default)")
    parser.add_argument("--foldseek-bin", default="binaries/foldseek", help="Path to foldseek binary")
    parser.add_argument("--threads", type=int, default=32, help="Number of CPU threads")
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    structure_dir = args.structure_dir or config["structure_dir"]
    output = args.output or config["output"]

    print("=" * 80)
    print("Foldseek Benchmark")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Structure dir: {structure_dir}")
    print(f"Output: {output}")
    print(f"Threads: {args.threads}")
    print("=" * 80)

    if not Path(structure_dir).exists():
        raise ValueError(f"Structure directory not found: {structure_dir}")
    if not Path(args.foldseek_bin).exists():
        raise ValueError(f"Foldseek binary not found: {args.foldseek_bin}")

    pdb_files = get_pdb_files(structure_dir)
    if not pdb_files:
        raise ValueError(f"No structure files found in {structure_dir}")

    df = run_foldseek(structure_dir, args.foldseek_bin, args.threads)
    pairs = parse_results(df)
    save_results(pairs, output)

    print("=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
