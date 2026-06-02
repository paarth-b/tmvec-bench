#!/usr/bin/env python
"""
Time benchmark for TM-align (structural alignment baseline).
Measures pairwise alignment times for different numbers of structures.

Since TM-align has no embedding or database-indexing step, encoding_times.csv
measures all-vs-all pairwise alignment time for N structures (O(n^2)), and
query_times.csv measures the time to align Q query structures against a
D-structure database.
"""

import argparse
import subprocess
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    from .benchmark_env import capture_benchmark_environment, write_json
except ImportError:
    from benchmark_env import capture_benchmark_environment, write_json


# ==============================================================================
# STRUCTURE LOADING
# ==============================================================================

def collect_structure_files(structure_dir, extension=".pdb", max_files=None):
    """Collect PDB structure files from a directory."""
    structure_dir = Path(structure_dir)
    files = sorted(structure_dir.glob(f"*{extension}"))
    if max_files:
        files = files[:max_files]
    return list(files)


def duplicate_structures_to_size(structures, target_size):
    """Duplicate structure list to reach target_size."""
    if len(structures) >= target_size:
        return structures[:target_size]
    repeats = (target_size // len(structures)) + 1
    return (structures * repeats)[:target_size]


# ==============================================================================
# TMALIGN RUNNER
# ==============================================================================

def _run_pair(args):
    """Run TMalign on one pair of structures (top-level for pickling)."""
    pdb1, pdb2, binary = args
    try:
        subprocess.run(
            [str(binary), str(pdb1), str(pdb2), "-a", "T"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass


def _run_pairs(pairs, binary, threads):
    """Run a list of (pdb1, pdb2) pairs through TMalign and return elapsed time."""
    tasks = [(p1, p2, binary) for p1, p2 in pairs]
    start = time.perf_counter()
    if threads > 1:
        with ProcessPoolExecutor(max_workers=threads) as executor:
            list(executor.map(_run_pair, tasks))
    else:
        for task in tasks:
            _run_pair(task)
    return time.perf_counter() - start


# ==============================================================================
# BENCHMARKS
# ==============================================================================

def run_encoding_benchmark(structure_files, binary, encoding_sizes,
                           threads=1, num_runs=1, warmup_runs=0):
    """
    Benchmark all-vs-all TM-align timing (proxy for encoding/database-creation time).
    For N structures this runs N*(N-1)/2 pairwise alignments.
    """
    results = []
    print("\n" + "=" * 60)
    print("Benchmarking all-vs-all alignment times (encoding proxy)...")
    print(f"  warmup_runs={warmup_runs}, num_runs={num_runs}, threads={threads}")
    print("=" * 60)

    for size in encoding_sizes:
        files = duplicate_structures_to_size(structure_files, size)
        pairs = [(files[i], files[j]) for i in range(len(files))
                 for j in range(i + 1, len(files))]
        n_pairs = len(pairs)

        # Warmup
        warmup_pairs = pairs[:min(10, n_pairs)]
        for _ in range(warmup_runs):
            _run_pairs(warmup_pairs, binary, threads)

        times = []
        for _ in range(num_runs):
            elapsed = _run_pairs(pairs, binary, threads)
            times.append(elapsed)

        times = np.array(times)
        pairs_per_sec = n_pairs / times.mean() if times.mean() > 0 else 0
        print(f"  {size:>4} structures ({n_pairs:>6} pairs): "
              f"{times.mean():.3f}s ± {times.std():.3f}s  "
              f"({pairs_per_sec:.1f} pairs/s)")

        results.append({
            "encoding_size": size,
            "mean_seconds": times.mean(),
            "std_seconds": times.std(),
            "n_pairs": n_pairs,
            "pairs_per_second": pairs_per_sec,
            "all_times": times.tolist(),
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
        })

    return pd.DataFrame(results)


def run_query_benchmark(structure_files, binary, database_sizes, query_sizes,
                        threads=1, num_runs=1, warmup_runs=0):
    """
    Benchmark query timing: Q query structures aligned against a D-structure database.
    """
    results = []
    print("\n" + "=" * 60)
    print("Benchmarking query alignment times...")
    print(f"  warmup_runs={warmup_runs}, num_runs={num_runs}, threads={threads}")
    print("=" * 60)

    for db_size in database_sizes:
        db_files = duplicate_structures_to_size(structure_files, db_size)

        for q_size in query_sizes:
            q_files = duplicate_structures_to_size(structure_files, q_size)
            pairs = [(qf, df) for qf in q_files for df in db_files]
            n_pairs = len(pairs)

            # Warmup
            warmup_pairs = pairs[:min(10, n_pairs)]
            for _ in range(warmup_runs):
                _run_pairs(warmup_pairs, binary, threads)

            times = []
            for _ in range(num_runs):
                elapsed = _run_pairs(pairs, binary, threads)
                times.append(elapsed)

            times = np.array(times)
            print(f"  query {q_size:>4} vs db {db_size:>4} ({n_pairs:>6} pairs): "
                  f"{times.mean():.3f}s ± {times.std():.3f}s")

            results.append({
                "query_size": q_size,
                "database_size": db_size,
                "total_mean": times.mean(),
                "total_std": times.std(),
                "n_pairs": n_pairs,
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
            })

    return pd.DataFrame(results)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="TM-align Time Benchmark")
    parser.add_argument("--structure-dir", default="data/pdb/cath-s100",
                        help="Directory containing PDB structure files")
    parser.add_argument("--tmalign-binary", default="binaries/TMalign",
                        help="Path to TMalign binary")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--max-structures", type=int, default=1000,
                        help="Maximum structures to load from disk")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of parallel processes for TM-align")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of timed runs per benchmark point")
    parser.add_argument("--warmup-runs", type=int, default=0,
                        help="Number of warmup runs before timing")
    parser.add_argument("--structure-extension", default=".pdb",
                        help="Structure file extension (default: .pdb)")
    args = parser.parse_args()

    binary = Path(args.tmalign_binary)
    if not binary.exists():
        raise FileNotFoundError(f"TMalign binary not found at {args.tmalign_binary}")

    print(f"Using TMalign binary: {args.tmalign_binary}")
    print(f"Using {args.threads} thread(s)")

    start_time = time.perf_counter()

    structure_files = collect_structure_files(
        args.structure_dir, args.structure_extension, max_files=args.max_structures
    )
    if not structure_files:
        raise ValueError(f"No structure files found in {args.structure_dir}")
    print(f"Found {len(structure_files)} structures")

    # Small sizes: TM-align is O(n^2) and ~0.07s/pair on CPU
    encoding_sizes = [10, 25, 50]
    database_sizes = [50, 100]
    query_sizes = [10]

    if len(structure_files) < max(encoding_sizes + database_sizes):
        print(f"Note: only {len(structure_files)} structures available; "
              "larger sizes will use duplication")

    encoding_df = run_encoding_benchmark(
        structure_files, binary, encoding_sizes,
        threads=args.threads, num_runs=args.num_runs, warmup_runs=args.warmup_runs,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    encoding_df.to_csv(output_dir / "tmalign_encoding_times.csv", index=False)

    query_df = run_query_benchmark(
        structure_files, binary, database_sizes, query_sizes,
        threads=args.threads, num_runs=args.num_runs, warmup_runs=args.warmup_runs,
    )
    query_df.to_csv(output_dir / "tmalign_query_times.csv", index=False)

    config = {
        "comparison_mode": "tmalign",
        "threads": args.threads,
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "encoding_sizes": encoding_sizes,
        "database_sizes": database_sizes,
        "query_sizes": query_sizes,
        "system_info_file": "tmalign_system_info.json",
    }
    pd.Series(config).to_json(output_dir / "tmalign_benchmark_config.json")
    write_json(
        output_dir / "tmalign_system_info.json",
        capture_benchmark_environment(requested_threads=args.threads, accelerator="cpu"),
    )

    total_time = time.perf_counter() - start_time
    print(f"\n{'=' * 60}")
    print(f"Total benchmark time: {total_time:.2f}s")
    print(f"Results saved in: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
