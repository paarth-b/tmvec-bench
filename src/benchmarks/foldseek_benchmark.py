#!/usr/bin/env python
"""
Foldseek Benchmark: Generate pairwise TM-score predictions for protein structures.
"""

from pathlib import Path
import subprocess
import pandas as pd
import tempfile
import shutil
import argparse
from tqdm import tqdm


def get_pdb_files(structure_dir):
    """Get all PDB files from structure directory."""
    pdb_files = [f for f in Path(structure_dir).iterdir() if f.is_file()]
    pdb_files.sort()
    print(f"Found {len(pdb_files)} PDB files")
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

        # Run without capturing output so we can see progress
        result = subprocess.run(cmd)

        if result.returncode != 0:
            raise RuntimeError("Foldseek failed")

        # Read results before tmp_dir is deleted
        df = pd.read_csv(tsv_path, sep='\t', header=None,
                        names=['query', 'target', 'alntmscore', 'evalue'],
                        low_memory=False)

    print(f"Loaded {len(df)} alignments")
    return df


def parse_results(df):
    """Extract unique pairwise comparisons and average bidirectional scores."""
    print("Parsing results...")

    # Collect scores for both directions
    scores_dict = {}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        q_id = Path(row['query']).stem
        t_id = Path(row['target']).stem

        # Remove _MODEL_* suffix if present
        q_id = q_id.split('_MODEL_')[0] if '_MODEL_' in q_id else q_id
        t_id = t_id.split('_MODEL_')[0] if '_MODEL_' in t_id else t_id

        if q_id != t_id:
            pair_key = tuple(sorted([q_id, t_id]))
            if pair_key not in scores_dict:
                scores_dict[pair_key] = {'scores': [], 'evalues': []}
            scores_dict[pair_key]['scores'].append(row['alntmscore'])
            scores_dict[pair_key]['evalues'].append(row['evalue'])

    # Average bidirectional scores
    pairs = []
    for pair_key, data in scores_dict.items():
        pairs.append({
            'seq1_id': pair_key[0],
            'seq2_id': pair_key[1],
            'tm_score': sum(data['scores']) / len(data['scores']),
            'evalue': min(data['evalues'])  # Use minimum e-value
        })

    print(f"Extracted {len(pairs)} unique pairs")
    return pairs


def save_results(pairs, output_path):
    """Save results to CSV."""
    df = pd.DataFrame(pairs)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(pairs):,} pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Foldseek benchmark")
    parser.add_argument("--structure-dir", required=True, help="Directory with PDB files")
    parser.add_argument("--foldseek-bin", required=True, help="Path to foldseek binary")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--fasta", help="Fasta file to filter structures")
    parser.add_argument("--threads", type=int, default=32, help="Number of threads")
    args = parser.parse_args()

    print("=" * 80)
    print("Foldseek Benchmark")
    print(f"Structure dir: {args.structure_dir}")
    print(f"Output: {args.output}")
    print(f"Threads: {args.threads}")
    print("=" * 80)

    pdb_files = get_pdb_files(args.structure_dir)
    if not pdb_files:
        raise ValueError(f"No files found in {args.structure_dir}")

    df = run_foldseek(args.structure_dir, args.foldseek_bin, args.threads)
    pairs = parse_results(df)
    save_results(pairs, args.output)

    print("=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
