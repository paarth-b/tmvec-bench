#!/usr/bin/env python
"""
Time benchmark for Diamond (sequence-based alignment).
Measures database creation and query times for different dataset sizes.
"""

import argparse
import pandas as pd
import numpy as np
import subprocess
import time
import tempfile
import shutil
import os
from datetime import datetime
from pathlib import Path

try:
    from .benchmark_env import capture_benchmark_environment, write_json
except ImportError:
    from benchmark_env import capture_benchmark_environment, write_json


# ==============================================================================
# TIMING UTILITIES
# ==============================================================================

def timed_run(func, *args, **kwargs):
    """Run a function with timing."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def benchmark_function(func, num_runs=3, warmup_runs=1, *args, **kwargs):
    """Benchmark a function with warmup and multiple runs."""
    for _ in range(warmup_runs):
        func(*args, **kwargs)

    times = []
    result = None
    for _ in range(num_runs):
        result, elapsed = timed_run(func, *args, **kwargs)
        times.append(elapsed)

    times = np.array(times)
    return result, times.mean(), times.std(), times.tolist()


# ==============================================================================
# FASTA UTILITIES
# ==============================================================================

def load_fasta_sequences(fasta_path, max_sequences=None):
    """Load sequences from FASTA file."""
    sequences = []
    current_id, current_seq = None, []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id:
                    sequences.append((current_id, ''.join(current_seq)))
                    if max_sequences and len(sequences) >= max_sequences:
                        break
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id and (not max_sequences or len(sequences) < max_sequences):
            sequences.append((current_id, ''.join(current_seq)))

    print(f"Loaded {len(sequences)} sequences")
    return sequences


def write_subset_fasta(sequences, output_path, count):
    """Write a subset of sequences to a FASTA file."""
    subset = sequences[:count]
    with open(output_path, 'w') as f:
        for seq_id, seq in subset:
            f.write(f">{seq_id}\n{seq}\n")
    return output_path


# ==============================================================================
# DIAMOND OPERATIONS
# ==============================================================================

def create_diamond_db(diamond_bin, fasta_path, db_path, threads=1):
    """Create a Diamond database."""
    cmd = [
        diamond_bin, "makedb",
        "--in", str(fasta_path),
        "--db", str(db_path),
        "--threads", str(threads)
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Diamond makedb failed: {result.stderr}")
    return db_path


def search_diamond(diamond_bin, query_fasta, db_path, output_path, threads=1):
    """Run Diamond blastp search."""
    cmd = [
        diamond_bin, "blastp",
        "--query", str(query_fasta),
        "--db", str(db_path),
        "--out", str(output_path),
        "--outfmt", "6",
        "--threads", str(threads),
        "--max-target-seqs", "100000",
        "--evalue", "10",
        "--sensitive"
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Diamond blastp failed: {result.stderr}")
    return output_path


# ==============================================================================
# BENCHMARKS
# ==============================================================================

def run_database_creation_benchmark(sequences, diamond_bin, encoding_sizes,
                                     threads=1, num_runs=3, warmup_runs=1):
    """Benchmark database creation times for different sizes."""
    results = []

    print("\n" + "=" * 60)
    print("Benchmarking database creation times...")
    print(f"(warmup_runs={warmup_runs}, num_runs={num_runs})")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for size in encoding_sizes:
            print(f"\nCreating database for {size} sequences...")

            if size > len(sequences):
                # Duplicate sequences to reach size
                repeats = (size // len(sequences)) + 1
                seqs = (sequences * repeats)[:size]
            else:
                seqs = sequences[:size]

            fasta_path = Path(tmp_dir) / f"seqs_{size}.fa"
            write_subset_fasta(seqs, fasta_path, size)

            counter = [0]

            def create_db():
                counter[0] += 1
                db_path = Path(tmp_dir) / f"db_{size}_{counter[0]}"
                create_diamond_db(diamond_bin, fasta_path, db_path, threads)
                # Clean up
                for f in Path(tmp_dir).glob(f"db_{size}_{counter[0]}*"):
                    f.unlink()
                return db_path

            _, mean_time, std_time, all_times = benchmark_function(
                create_db,
                num_runs=num_runs,
                warmup_runs=warmup_runs
            )

            seqs_per_sec = size / mean_time if mean_time > 0 else 0

            print(f"Creating db for {size:>6} sequences: {mean_time:.3f}s ± {std_time:.3f}s "
                  f"({seqs_per_sec:.1f} seq/s)")

            results.append({
                "encoding_size": size,
                "mean_seconds": mean_time,
                "std_seconds": std_time,
                "seqs_per_second": seqs_per_sec,
                "all_times": all_times,
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
            })

    return pd.DataFrame(results)


def run_search_benchmark(sequences, diamond_bin, database_sizes, query_sizes,
                         threads=1, num_runs=3, warmup_runs=1):
    """Benchmark search times for different database and query sizes."""
    results = []

    print("\n" + "=" * 60)
    print("Benchmarking search times...")
    print(f"(warmup_runs={warmup_runs}, num_runs={num_runs})")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for db_size in database_sizes:
            print(f"\nBuilding database with {db_size} sequences...")

            if db_size > len(sequences):
                repeats = (db_size // len(sequences)) + 1
                db_seqs = (sequences * repeats)[:db_size]
            else:
                db_seqs = sequences[:db_size]

            db_fasta = Path(tmp_dir) / f"db_{db_size}.fa"
            write_subset_fasta(db_seqs, db_fasta, db_size)

            db_path = Path(tmp_dir) / f"db_{db_size}"
            start_db = time.perf_counter()
            create_diamond_db(diamond_bin, db_fasta, db_path, threads)
            db_build_time = time.perf_counter() - start_db

            print(f"Database build (one-time): {db_build_time:.3f}s")

            for query_size in query_sizes:
                print(f"Running queries ({query_size} sequences)...")

                if query_size > len(sequences):
                    repeats = (query_size // len(sequences)) + 1
                    query_seqs = (sequences * repeats)[:query_size]
                else:
                    query_seqs = sequences[:query_size]

                query_fasta = Path(tmp_dir) / f"query_{query_size}.fa"
                write_subset_fasta(query_seqs, query_fasta, query_size)

                counter = [0]

                def do_search():
                    counter[0] += 1
                    output = Path(tmp_dir) / f"result_{counter[0]}.tsv"
                    search_diamond(diamond_bin, query_fasta, db_path, output, threads)
                    output.unlink()
                    return output

                _, search_mean, search_std, search_times = benchmark_function(
                    do_search,
                    num_runs=num_runs,
                    warmup_runs=warmup_runs
                )

                print(
                    f"Query {query_size:>5} vs {db_size:>6} database: "
                    f"search={search_mean:.3f}s±{search_std:.3f}s"
                )

                results.append({
                    "query_size": query_size,
                    "database_size": db_size,
                    "db_build_time_one_time": db_build_time,
                    "search_mean": search_mean,
                    "search_std": search_std,
                    "num_runs": num_runs,
                    "warmup_runs": warmup_runs,
                })

    return pd.DataFrame(results)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Diamond Time Benchmark")
    parser.add_argument("--fasta", required=True,
                        help="Path to FASTA file")
    parser.add_argument("--diamond-bin", default="diamond",
                        help="Path to diamond binary")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory")
    parser.add_argument("--max-sequences", type=int, default=100000,
                        help="Maximum sequences to use")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of threads")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of timed runs per benchmark")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Number of warmup runs before timing")
    args = parser.parse_args()

    print(f"Using diamond binary: {args.diamond_bin}")
    print(f"Using {args.threads} thread(s)")

    start_time = time.perf_counter()

    sequences = load_fasta_sequences(args.fasta, args.max_sequences)

    if len(sequences) == 0:
        raise ValueError(f"No sequences found in {args.fasta}")

    print(f"Using {len(sequences)} sequences for benchmark")

    # Encoding sizes matching other benchmarks
    encoding_sizes = [10, 100, 1000, 5000, 10000, 50000]
    database_sizes = [1000, 10000, 100000]
    query_sizes = [10, 100, 1000]

    # Filter sizes based on available sequences
    encoding_sizes = [s for s in encoding_sizes if s <= len(sequences)]
    database_sizes = [s for s in database_sizes if s <= len(sequences)]

    encoding_df = run_database_creation_benchmark(
        sequences, args.diamond_bin, encoding_sizes,
        threads=args.threads,
        num_runs=args.num_runs, warmup_runs=args.warmup_runs
    )

    search_df = run_search_benchmark(
        sequences, args.diamond_bin, database_sizes, query_sizes,
        threads=args.threads,
        num_runs=args.num_runs, warmup_runs=args.warmup_runs
    )

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    encoding_df.to_csv(output_dir / "diamond_encoding_times.csv", index=False)
    search_df.to_csv(output_dir / "diamond_query_times.csv", index=False)

    # Save benchmark config
    config = {
        "diamond_bin": args.diamond_bin,
        "fasta": args.fasta,
        "num_sequences": len(sequences),
        "threads": args.threads,
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "encoding_sizes": encoding_sizes,
        "database_sizes": database_sizes,
        "query_sizes": query_sizes,
        "system_info_file": "diamond_system_info.json",
    }
    pd.Series(config).to_json(output_dir / "diamond_benchmark_config.json")
    write_json(
        output_dir / "diamond_system_info.json",
        capture_benchmark_environment(requested_threads=args.threads, accelerator="cpu"),
    )

    total_benchmark_time = time.perf_counter() - start_time
    print(f"\n{'=' * 60}")
    print(f"Total benchmark time: {total_benchmark_time:.2f}s")
    print(f"Results saved in: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
