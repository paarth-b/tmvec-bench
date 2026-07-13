#!/usr/bin/env python
"""TMalign benchmark for CATH and SCOPe.

Runs all-vs-all pairwise TM-align comparisons using multiple CPU cores.
"""

import argparse
import os
import subprocess
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DATASETS = {
    "cath": {
        "fasta": "data/fasta/cath-s100-unique-10k.fa",
        "pdb_dir": "data/pdb/CATH",
        "output": "/work/nvme/beut/paarthbatra/data/results/cath_tmalign_similarities.csv",
    },
    "scope40": {
        "fasta": "data/fasta/scop40.fasta",
        "pdb_dir": "data/pdb/SCOPe40",
        "output": "/work/nvme/beut/paarthbatra/data/results/scope40_tmalign_similarities.csv",
    },
}

TMALIGN_TIMEOUT = 60  # seconds per pair


def parse_fasta_ids(fasta_path):
    """Extract domain IDs from a FASTA file."""
    ids = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                seq_id = line.strip()[1:].split("|")[-1].split("/")[0]
                ids.append(seq_id)
    return ids


def find_structures(domain_ids, pdb_dir):
    """Map domain IDs to PDB file paths. Returns {id: Path} for found files."""
    pdb_dir = Path(pdb_dir)
    structures = {}
    for did in domain_ids:
        for path in [pdb_dir / f"{did}.pdb", pdb_dir / did]:
            if path.exists():
                structures[did] = path
                break
    print(f"Found {len(structures)}/{len(domain_ids)} structures")
    return structures


def run_tmalign_pair(args):
    """Run TMalign on one pair. Returns (id1, id2, score) or None on failure."""
    id1, pdb1, id2, pdb2, binary = args
    try:
        result = subprocess.run(
            [binary, str(pdb1), str(pdb2), "-a", "T"],
            capture_output=True,
            text=True,
            timeout=TMALIGN_TIMEOUT,
        )
        for line in result.stdout.split("\n"):
            if line.startswith("TM-score=") and "average length" in line:
                score = float(line.split()[1])
                return (id1, id2, score)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
        pass
    return None


def calculate_all_scores(structures, binary, num_workers):
    """Calculate all pairwise TM-scores using multiple CPU cores."""
    ids = list(structures.keys())
    pair_args = [
        (id1, structures[id1], id2, structures[id2], binary)
        for id1, id2 in combinations(ids, 2)
    ]
    total_pairs = len(pair_args)
    print(f"Computing {total_pairs:,} pairs with {num_workers} workers...")

    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(run_tmalign_pair, pair_args, chunksize=256),
            total=total_pairs,
            desc="TMalign",
        ):
            if result is not None:
                results.append(result)

    return pd.DataFrame(results, columns=["seq1_id", "seq2_id", "tm_score"])


def main():
    parser = argparse.ArgumentParser(description="TMalign benchmark")
    parser.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        default="cath",
    )
    parser.add_argument("--fasta", help="Override FASTA path")
    parser.add_argument("--pdb-dir", help="Override PDB directory")
    parser.add_argument("--output", help="Override output CSV path")
    parser.add_argument("--binary", default="binaries/TMalign")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers (default: all CPUs)",
    )
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    fasta = args.fasta or config["fasta"]
    pdb_dir = args.pdb_dir or config["pdb_dir"]
    output = args.output or config["output"]

    print(f"Dataset: {args.dataset}")
    print(f"FASTA:   {fasta}")
    print(f"PDB dir: {pdb_dir}")
    print(f"Output:  {output}")
    print(f"Workers: {args.workers}")

    ids = parse_fasta_ids(fasta)
    print(f"Found {len(ids)} sequences in FASTA")

    structures = find_structures(ids, pdb_dir)
    if not structures:
        raise ValueError("No structures found!")

    df = calculate_all_scores(structures, args.binary, args.workers)
    if df.empty:
        raise ValueError("No scores computed!")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\nSaved {len(df):,} scores to {output}")
    print(f"Mean: {df['tm_score'].mean():.4f}, Std: {df['tm_score'].std():.4f}")


if __name__ == "__main__":
    main()
