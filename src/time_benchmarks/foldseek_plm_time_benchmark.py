#!/usr/bin/env python
"""
Time benchmark for Foldseek ProstT5 mode (sequence-based structural search).
Measures database creation and query times for different database and query sizes.

ProstT5 uses a protein language model to predict 3Di structural sequences
directly from amino acid sequences, bypassing the need for actual structures.

Supports both CPU and GPU modes via --use-gpu flag.
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
    """
    Run a function with timing.
    Returns (result, elapsed_seconds).
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def benchmark_function(func, num_runs=3, warmup_runs=1, *args, **kwargs):
    """
    Benchmark a function with warmup and multiple runs.
    Returns (result, mean_time, std_time, all_times).
    """
    # Warmup runs (not timed)
    for _ in range(warmup_runs):
        func(*args, **kwargs)

    # Timed runs
    times = []
    result = None
    for _ in range(num_runs):
        result, elapsed = timed_run(func, *args, **kwargs)
        times.append(elapsed)

    times = np.array(times)
    return result, times.mean(), times.std(), times.tolist()


# ==============================================================================
# SEQUENCE UTILITIES
# ==============================================================================

def load_fasta_sequences(fasta_path, max_sequences=None):
    """
    Load sequences from a FASTA file.
    Returns list of (id, sequence) tuples.
    """
    sequences = []
    current_id = None
    current_seq = []

    print(f"Loading sequences from {fasta_path}...")

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences.append((current_id, ''.join(current_seq)))
                    if max_sequences and len(sequences) >= max_sequences:
                        break
                current_id = line[1:].split()[0]  # Take first word after >
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget last sequence
        if current_id is not None and (not max_sequences or len(sequences) < max_sequences):
            sequences.append((current_id, ''.join(current_seq)))

    print(f"Loaded {len(sequences)} sequences")
    return sequences


def duplicate_sequences_to_size(sequences, target_size):
    """
    Duplicate sequence list to reach target size.
    Repeats the sequence list as many times as needed, then truncates.
    """
    if len(sequences) >= target_size:
        return sequences[:target_size]

    # Calculate how many times to repeat
    repeats_needed = (target_size // len(sequences)) + 1
    duplicated = sequences * repeats_needed
    return duplicated[:target_size]


def write_fasta(sequences, output_path):
    """
    Write sequences to a FASTA file.
    sequences: list of (id, sequence) tuples
    """
    with open(output_path, 'w') as f:
        for i, (seq_id, seq) in enumerate(sequences):
            # Add index to make IDs unique when duplicating
            f.write(f">{seq_id}_{i}\n{seq}\n")
    return output_path


# ==============================================================================
# FOLDSEEK PROSTT5 OPERATIONS
# ==============================================================================

def run_foldseek_command(foldseek_binary, cmd_args, verbose=False):
    """
    Run a foldseek command and capture output.
    Returns (stdout, stderr, returncode).
    """
    cmd = [str(foldseek_binary)] + cmd_args

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    if result.returncode != 0 and verbose:
        print(f"Command failed with return code {result.returncode}")
        print(f"stderr: {result.stderr}")

    return result.stdout, result.stderr, result.returncode


def download_prostt5_weights(foldseek_binary, weights_dir, temp_dir=None):
    """
    Download ProstT5 model weights using foldseek databases command.
    """
    weights_path = Path(weights_dir)
    weights_path.mkdir(parents=True, exist_ok=True)

    scratch = temp_dir if temp_dir else tempfile.mkdtemp(prefix="prostt5_download_")

    args = [
        "databases",
        "ProstT5",
        str(weights_path / "prostt5"),
        str(scratch)
    ]

    print(f"Downloading ProstT5 weights to {weights_path}...")
    stdout, stderr, returncode = run_foldseek_command(foldseek_binary, args, verbose=True)

    if not temp_dir and Path(scratch).exists():
        shutil.rmtree(scratch, ignore_errors=True)

    if returncode != 0:
        raise RuntimeError(f"Failed to download ProstT5 weights: {stderr}")

    return str(weights_path / "prostt5")


def create_prostt5_database(foldseek_binary, fasta_path, output_db,
                             prostt5_weights, threads=1, use_gpu=False,
                             temp_dir=None):
    """
    Create a foldseek database from FASTA using ProstT5 model.

    Args:
        fasta_path: Path to FASTA file
        output_db: Path for the output database
        prostt5_weights: Path to ProstT5 model weights
        threads: Number of threads to use
        use_gpu: Use GPU for ProstT5 inference
        temp_dir: Dedicated temp directory for foldseek internal use
    """
    args = [
        "createdb",
        str(fasta_path),
        str(output_db),
        "--prostt5-model", str(prostt5_weights),
        "--threads", str(threads)
    ]

    if use_gpu:
        args.extend(["--gpu", "1"])

    # Set environment to use dedicated temp directory if provided
    env = os.environ.copy()
    if temp_dir:
        env['TMPDIR'] = str(temp_dir)

    cmd = [str(foldseek_binary)] + args
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to create ProstT5 database: {result.stderr}")

    return output_db


def pad_database_for_gpu(foldseek_binary, input_db, output_db):
    """
    Pad database for GPU search using makepaddedseqdb.
    Required for GPU searches.
    """
    args = [
        "makepaddedseqdb",
        str(input_db),
        str(output_db)
    ]

    cmd = [str(foldseek_binary)] + args
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to pad database: {result.stderr}")

    return output_db


def search_foldseek(foldseek_binary, query_db, target_db, result_db,
                    threads=1, sensitivity=None, temp_dir=None, use_gpu=False):
    """
    Run foldseek search.
    Note: GPU flag here is for search acceleration, not ProstT5 inference.
    """
    scratch_dir = temp_dir if temp_dir else tempfile.mkdtemp(prefix="foldseek_scratch_")

    args = [
        "search",
        str(query_db),
        str(target_db),
        str(result_db),
        str(scratch_dir),
        "--threads", str(threads)
    ]

    if sensitivity is not None:
        args.extend(["-s", str(sensitivity)])

    if use_gpu:
        args.extend(["--gpu", "1"])

    stdout, stderr, returncode = run_foldseek_command(foldseek_binary, args)

    if not temp_dir and Path(scratch_dir).exists():
        shutil.rmtree(scratch_dir, ignore_errors=True)

    if returncode != 0:
        raise RuntimeError(f"Failed to search: {stderr}")

    return result_db


def cleanup_foldseek_db(db_path):
    """
    Clean up all files associated with a foldseek database.
    """
    db_path = Path(db_path)
    parent = db_path.parent
    db_name = db_path.name

    for f in parent.iterdir():
        if f.name.startswith(db_name):
            try:
                f.unlink()
            except OSError:
                pass


# ==============================================================================
# BENCHMARKS
# ==============================================================================

def run_encoding_benchmark(sequences, foldseek_binary, prostt5_weights,
                           encoding_sizes, threads=1, use_gpu=False,
                           num_runs=3, warmup_runs=1,
                           benchmark_temp_dir=None):
    """
    Benchmark ProstT5 database creation times for different sizes.
    This measures the time to convert FASTA sequences to 3Di databases.
    """
    results = []

    mode_str = "GPU" if use_gpu else "CPU"
    print("\n" + "="*60)
    print(f"Benchmarking ProstT5 encoding times ({mode_str} mode)...")
    print(f"(warmup_runs={warmup_runs}, num_runs={num_runs})")
    print("="*60)

    master_temp_dir = benchmark_temp_dir or tempfile.mkdtemp(prefix="prostt5_bench_")
    master_temp_path = Path(master_temp_dir)

    try:
        for enc_size in encoding_sizes:
            print(f"\nEncoding {enc_size} sequences with ProstT5...")

            if enc_size > len(sequences):
                print(f"Duplicating sequences to reach {enc_size} (have {len(sequences)} sequences)")
                enc_seqs = duplicate_sequences_to_size(sequences, enc_size)
            else:
                enc_seqs = sequences[:enc_size]

            size_temp_dir = master_temp_path / f"encoding_{enc_size}"
            size_temp_dir.mkdir(exist_ok=True)

            # Write FASTA file
            fasta_path = size_temp_dir / "sequences.fasta"
            write_fasta(enc_seqs, fasta_path)

            db_counter = [0]

            def create_db():
                db_counter[0] += 1
                temp_db = size_temp_dir / f"prostt5db_{db_counter[0]}"
                create_prostt5_database(
                    foldseek_binary, fasta_path, str(temp_db),
                    prostt5_weights, threads=threads, use_gpu=use_gpu,
                    temp_dir=str(size_temp_dir)
                )
                cleanup_foldseek_db(temp_db)
                return str(temp_db)

            _, mean_time, std_time, all_times = benchmark_function(
                create_db,
                num_runs=num_runs,
                warmup_runs=warmup_runs
            )

            seqs_per_sec = enc_size / mean_time if mean_time > 0 else 0

            print(f"Encoding {enc_size:>6} sequences: {mean_time:.3f}s +/- {std_time:.3f}s "
                  f"({seqs_per_sec:.1f} seq/s) [runs: {[f'{t:.3f}' for t in all_times]}]")

            results.append({
                "encoding_size": enc_size,
                "mean_seconds": mean_time,
                "std_seconds": std_time,
                "seqs_per_second": seqs_per_sec,
                "all_times": all_times,
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
                "mode": mode_str,
            })

            shutil.rmtree(size_temp_dir, ignore_errors=True)

    finally:
        if not benchmark_temp_dir:
            shutil.rmtree(master_temp_dir, ignore_errors=True)

    return pd.DataFrame(results)


def run_search_benchmark(sequences, foldseek_binary, prostt5_weights,
                         database_sizes, query_sizes,
                         threads=1, sensitivity=None, use_gpu=False,
                         num_runs=3, warmup_runs=1,
                         benchmark_temp_dir=None):
    """
    Benchmark search times for different database and query sizes.

    Target database creation is done once and NOT included in the query timing.
    Only query database creation + search are timed.
    """
    results = []

    mode_str = "GPU" if use_gpu else "CPU"
    print("\n" + "="*60)
    print(f"Benchmarking ProstT5 search times ({mode_str} mode)...")
    print(f"(warmup_runs={warmup_runs}, num_runs={num_runs})")
    print("="*60)

    master_temp_dir = benchmark_temp_dir or tempfile.mkdtemp(prefix="prostt5_search_bench_")
    master_temp_path = Path(master_temp_dir)

    try:
        for database_size in database_sizes:
            print(f"\nBuilding database with {database_size} sequences...")

            if database_size > len(sequences):
                print(f"Duplicating sequences to reach {database_size} (have {len(sequences)} sequences)")
                db_seqs = duplicate_sequences_to_size(sequences, database_size)
            else:
                db_seqs = sequences[:database_size]

            db_size_temp_dir = master_temp_path / f"db_{database_size}"
            db_size_temp_dir.mkdir(exist_ok=True)

            # Write database FASTA
            db_fasta_path = db_size_temp_dir / "database.fasta"
            write_fasta(db_seqs, db_fasta_path)
            target_db = db_size_temp_dir / "target_db"

            try:
                # Create target database ONCE (not included in query benchmark timing)
                start_db = time.perf_counter()
                create_prostt5_database(
                    foldseek_binary, db_fasta_path, str(target_db),
                    prostt5_weights, threads=threads, use_gpu=use_gpu,
                    temp_dir=str(db_size_temp_dir)
                )
                # For GPU mode, pad the main target_db (this also creates the padded _ss db)
                if use_gpu:
                    target_db_padded = db_size_temp_dir / "target_db_pad"
                    pad_database_for_gpu(foldseek_binary, str(target_db), str(target_db_padded))
                    search_target_db = str(target_db_padded)
                else:
                    search_target_db = str(target_db)
                db_build_time = time.perf_counter() - start_db

                print(f"Database build (one-time): {db_build_time:.3f}s (NOT included in query timings)")

                for query_size in query_sizes:
                    print(f"Running queries ({query_size} sequences, {mode_str})...")

                    if query_size > len(sequences):
                        query_seqs = duplicate_sequences_to_size(sequences, query_size)
                    else:
                        query_seqs = sequences[:query_size]

                    query_temp_dir = db_size_temp_dir / f"query_{query_size}"
                    query_temp_dir.mkdir(exist_ok=True)

                    # Write query FASTA
                    query_fasta_path = query_temp_dir / "query.fasta"
                    write_fasta(query_seqs, query_fasta_path)
                    query_db = query_temp_dir / "query_db"

                    try:
                        # Create query database once for search benchmark
                        create_prostt5_database(
                            foldseek_binary, query_fasta_path, str(query_db),
                            prostt5_weights, threads=threads, use_gpu=use_gpu,
                            temp_dir=str(query_temp_dir)
                        )

                        # Benchmark query encoding separately
                        enc_counter = [0]
                        def create_query_db():
                            enc_counter[0] += 1
                            temp_query_db = query_temp_dir / f"query_db_bench_{enc_counter[0]}"
                            create_prostt5_database(
                                foldseek_binary, query_fasta_path, str(temp_query_db),
                                prostt5_weights, threads=threads, use_gpu=use_gpu,
                                temp_dir=str(query_temp_dir)
                            )
                            cleanup_foldseek_db(temp_query_db)
                            return str(temp_query_db)

                        _, enc_mean, enc_std, enc_times = benchmark_function(
                            create_query_db,
                            num_runs=num_runs,
                            warmup_runs=warmup_runs
                        )

                        # Search benchmark
                        search_counter = [0]
                        def do_search():
                            search_counter[0] += 1
                            result_db = query_temp_dir / f"result_db_{search_counter[0]}"
                            search_foldseek(
                                foldseek_binary, str(query_db), search_target_db,
                                str(result_db), threads=threads, sensitivity=sensitivity,
                                temp_dir=str(query_temp_dir), use_gpu=use_gpu
                            )
                            cleanup_foldseek_db(result_db)
                            return str(result_db)

                        _, search_mean, search_std, search_times = benchmark_function(
                            do_search,
                            num_runs=num_runs,
                            warmup_runs=warmup_runs
                        )

                        total_mean = enc_mean + search_mean
                        total_std = np.sqrt(enc_std**2 + search_std**2)

                        print(
                            f"Query {query_size:>5} vs {database_size:>6} database: "
                            f"encode={enc_mean:.3f}s+/-{enc_std:.3f}s, "
                            f"search={search_mean:.3f}s+/-{search_std:.3f}s, "
                            f"total={total_mean:.3f}s+/-{total_std:.3f}s"
                        )

                        results.append({
                            "query_size": query_size,
                            "database_size": database_size,
                            "encode_mean": enc_mean,
                            "encode_std": enc_std,
                            "db_build_time_one_time": db_build_time,
                            "search_mean": search_mean,
                            "search_std": search_std,
                            "total_mean": total_mean,
                            "total_std": total_std,
                            "num_runs": num_runs,
                            "warmup_runs": warmup_runs,
                            "mode": mode_str,
                        })

                    finally:
                        shutil.rmtree(query_temp_dir, ignore_errors=True)

            finally:
                shutil.rmtree(db_size_temp_dir, ignore_errors=True)

    finally:
        if not benchmark_temp_dir:
            shutil.rmtree(master_temp_dir, ignore_errors=True)

    return pd.DataFrame(results)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Foldseek ProstT5 Time Benchmark")
    parser.add_argument("--fasta", required=True,
                        help="Path to FASTA file with protein sequences")
    parser.add_argument("--foldseek-binary",
                        default="binaries/foldseek",
                        help="Path to foldseek binary")
    parser.add_argument("--prostt5-weights",
                        default="weights/prostt5",
                        help="Path to ProstT5 weights (will download if not exists)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--max-sequences", type=int, default=100000,
                        help="Maximum sequences to use (default: 100000)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of threads for foldseek")
    parser.add_argument("--sensitivity", type=float, default=None,
                        help="Sensitivity parameter for foldseek search")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of timed runs per benchmark")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Number of warmup runs before timing")
    parser.add_argument("--temp-dir", default=None,
                        help="Base temp directory for benchmark files")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Enable GPU acceleration for ProstT5 inference and search")
    parser.add_argument("--encoding-sizes", default=None,
                        help="Comma-separated list of encoding sizes to override the default "
                             "(e.g. '50000' to only fill in a single point)")
    parser.add_argument("--database-sizes", default=None,
                        help="Comma-separated list of database sizes for the search benchmark "
                             "to override the default (e.g. '100000')")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip the query/search benchmark (only run encoding)")
    parser.add_argument("--skip-encoding", action="store_true",
                        help="Skip the encoding benchmark (only run query/search)")
    args = parser.parse_args()

    # Verify foldseek binary exists
    foldseek_path = Path(args.foldseek_binary)
    if not foldseek_path.exists():
        raise FileNotFoundError(f"Foldseek binary not found at {args.foldseek_binary}")

    # Verify FASTA exists
    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found at {args.fasta}")

    mode_str = "GPU" if args.use_gpu else "CPU"
    print(f"Using foldseek binary: {args.foldseek_binary}")
    print(f"Mode: {mode_str}")
    print(f"Using {args.threads} thread(s)")

    # Check/download ProstT5 weights
    prostt5_weights = Path(args.prostt5_weights)
    if not prostt5_weights.exists():
        print(f"ProstT5 weights not found at {prostt5_weights}, downloading...")
        weights_dir = prostt5_weights.parent
        weights_dir.mkdir(parents=True, exist_ok=True)
        download_prostt5_weights(foldseek_path, str(weights_dir))
    else:
        print(f"Using ProstT5 weights: {prostt5_weights}")

    start_time = time.perf_counter()

    # Load sequences
    sequences = load_fasta_sequences(
        args.fasta,
        max_sequences=args.max_sequences
    )

    if len(sequences) == 0:
        raise ValueError(f"No sequences found in {args.fasta}")

    print(f"Using {len(sequences)} sequences for benchmark")

    # Encoding sizes (match the Foldseek structural benchmark so plots align)
    encoding_sizes = [10, 100, 1000, 5000, 10000, 50000]
    if args.encoding_sizes:
        encoding_sizes = [int(s) for s in args.encoding_sizes.split(",") if s.strip()]
    database_sizes = [1000, 10000]
    if args.database_sizes:
        database_sizes = [int(s) for s in args.database_sizes.split(",") if s.strip()]
    query_sizes = [10, 100, 1000]

    # Warn if duplication is needed (fine for timing purposes)
    if len(sequences) < max(encoding_sizes + database_sizes):
        print(f"Note: only {len(sequences)} sequences available; larger sizes will use duplication")

    # Create dedicated temp directory if specified
    benchmark_temp_dir = None
    if args.temp_dir:
        benchmark_temp_dir = Path(args.temp_dir)
        benchmark_temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using temp directory: {benchmark_temp_dir}")

    # Set up output dir (needed by both encoding and search outputs)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = "_gpu" if args.use_gpu else "_cpu"
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/time_benchmarks") / f"prostt5{mode_suffix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_encoding:
        encoding_df = run_encoding_benchmark(
            sequences, foldseek_path, str(prostt5_weights),
            encoding_sizes, threads=args.threads, use_gpu=args.use_gpu,
            num_runs=args.num_runs, warmup_runs=args.warmup_runs,
            benchmark_temp_dir=str(benchmark_temp_dir) if benchmark_temp_dir else None
        )
        encoding_df.to_csv(output_dir / "encoding_times.csv", index=False)

    if not args.skip_search:
        search_df = run_search_benchmark(
            sequences, foldseek_path, str(prostt5_weights),
            database_sizes, query_sizes,
            threads=args.threads, sensitivity=args.sensitivity, use_gpu=args.use_gpu,
            num_runs=args.num_runs, warmup_runs=args.warmup_runs,
            benchmark_temp_dir=str(benchmark_temp_dir) if benchmark_temp_dir else None
        )

        search_df.to_csv(output_dir / "query_times.csv", index=False)

    # Save benchmark config
    config = {
        "foldseek_binary": str(args.foldseek_binary),
        "fasta_file": str(args.fasta),
        "prostt5_weights": str(prostt5_weights),
        "num_sequences": len(sequences),
        "threads": args.threads,
        "sensitivity": args.sensitivity,
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "encoding_sizes": encoding_sizes,
        "database_sizes": database_sizes,
        "query_sizes": query_sizes,
        "use_gpu": args.use_gpu,
        "mode": mode_str,
        "system_info_file": "system_info.json",
    }
    pd.Series(config).to_json(output_dir / "benchmark_config.json")
    write_json(
        output_dir / "system_info.json",
        capture_benchmark_environment(
            requested_threads=args.threads,
            accelerator="gpu" if args.use_gpu else "cpu",
        ),
    )

    total_benchmark_time = time.perf_counter() - start_time
    print(f"\n{'='*60}")
    print(f"Total benchmark time: {total_benchmark_time:.2f}s")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
