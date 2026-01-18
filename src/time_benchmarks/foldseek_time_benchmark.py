#!/usr/bin/env python
"""
Time benchmark for Foldseek (3Di structural search).
Measures database creation and query times for different database and query sizes.

This benchmark uses subprocess timing to measure foldseek performance.
"""

import argparse
import pandas as pd
import numpy as np
import subprocess
import time
import tempfile
import shutil
from datetime import datetime
from pathlib import Path


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
# STRUCTURE FILE UTILITIES
# ==============================================================================

def collect_structure_files(structure_dir, extension=".pdb"):
    """Collect all structure files from a directory."""
    structure_path = Path(structure_dir)
    structure_files = sorted(structure_path.glob(f"*{extension}"))
    print(f"Found {len(structure_files)} structure files in {structure_dir}")
    return structure_files


def duplicate_files_to_size(files, target_size):
    """
    Duplicate file list to reach target size.
    Repeats the file list as many times as needed, then truncates.
    """
    if len(files) >= target_size:
        return files[:target_size]
    
    # Calculate how many times to repeat
    repeats_needed = (target_size // len(files)) + 1
    duplicated = files * repeats_needed
    return duplicated[:target_size]


def create_temp_structure_dir(structure_files, temp_base_dir=None):
    """
    Create a temporary directory and symlink structure files into it.
    Returns the temporary directory path.
    """
    if temp_base_dir:
        temp_dir = tempfile.mkdtemp(dir=temp_base_dir)
    else:
        temp_dir = tempfile.mkdtemp()
    
    temp_path = Path(temp_dir)
    
    for i, src_file in enumerate(structure_files):
        dst_file = temp_path / f"structure_{i:06d}{src_file.suffix}"
        dst_file.symlink_to(src_file.absolute())
    
    return temp_dir


# ==============================================================================
# FOLDSEEK OPERATIONS
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
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0 and verbose:
        print(f"Command failed with return code {result.returncode}")
        print(f"stderr: {result.stderr}")
    
    return result.stdout, result.stderr, result.returncode


def create_foldseek_database(foldseek_binary, structure_dir, output_db, threads=1):
    """
    Create a foldseek database from structure files.
    """
    args = [
        "createdb",
        str(structure_dir),
        str(output_db),
        "--threads", str(threads)
    ]
    
    stdout, stderr, returncode = run_foldseek_command(foldseek_binary, args)
    
    if returncode != 0:
        raise RuntimeError(f"Failed to create database: {stderr}")
    
    return output_db


def search_foldseek(foldseek_binary, query_db, target_db, result_db, 
                    threads=1, sensitivity=None):
    """
    Run foldseek search.
    """
    args = [
        "search",
        str(query_db),
        str(target_db),
        str(result_db),
        "/tmp",  # temporary directory for foldseek
        "--threads", str(threads)
    ]
    
    if sensitivity is not None:
        args.extend(["-s", str(sensitivity)])
    
    stdout, stderr, returncode = run_foldseek_command(foldseek_binary, args)
    
    if returncode != 0:
        raise RuntimeError(f"Failed to search: {stderr}")
    
    return result_db


def convert_search_results(foldseek_binary, result_db, query_db, target_db, 
                           output_file, format_mode=0):
    """
    Convert foldseek search results to readable format.
    """
    args = [
        "convertalis",
        str(query_db),
        str(target_db),
        str(result_db),
        str(output_file),
        "--format-mode", str(format_mode)
    ]
    
    stdout, stderr, returncode = run_foldseek_command(foldseek_binary, args)
    
    if returncode != 0:
        raise RuntimeError(f"Failed to convert results: {stderr}")
    
    return output_file


# ==============================================================================
# BENCHMARKS
# ==============================================================================

def run_database_creation_benchmark(structure_files, foldseek_binary, 
                                     database_sizes, threads=1, 
                                     num_runs=3, warmup_runs=1):
    """Benchmark database creation times for different sizes."""
    results = []
    
    print("\n" + "="*60)
    print("Benchmarking encoding times...")
    print(f"(warmup_runs={warmup_runs}, num_runs={num_runs})")
    print("="*60)
    
    for db_size in database_sizes:
        print(f"\nEncoding {db_size} structures...")
        
        if db_size > len(structure_files):
            print(f"Duplicating structures to reach {db_size} (have {len(structure_files)} structures)")
            db_files = duplicate_files_to_size(structure_files, db_size)
        else:
            db_files = structure_files[:db_size]
        
        # Create temporary directory with structures
        temp_dir = create_temp_structure_dir(db_files)
        
        try:
            # Benchmark function wrapper
            def create_db():
                temp_db = tempfile.mktemp(suffix="_foldseekdb")
                create_foldseek_database(foldseek_binary, temp_dir, temp_db, threads=threads)
                # Clean up database files
                for f in Path(temp_db).parent.glob(f"{Path(temp_db).name}*"):
                    f.unlink()
                return temp_db
            
            _, mean_time, std_time, all_times = benchmark_function(
                create_db,
                num_runs=num_runs,
                warmup_runs=warmup_runs
            )
            
            structs_per_sec = db_size / mean_time if mean_time > 0 else 0
            
            print(f"Encoding {db_size:>6} structures: {mean_time:.3f}s ± {std_time:.3f}s "
                  f"({structs_per_sec:.1f} struct/s) [runs: {[f'{t:.3f}' for t in all_times]}]")
            
            results.append({
                "encoding_size": db_size,
                "mean_seconds": mean_time,
                "std_seconds": std_time,
                "structs_per_second": structs_per_sec,
                "all_times": all_times,
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
            })
        
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
    
    return pd.DataFrame(results)


def run_search_benchmark(structure_files, foldseek_binary, database_sizes, 
                        query_sizes, threads=1, sensitivity=None,
                        num_runs=3, warmup_runs=1):
    """Benchmark search times for different database and query sizes."""
    results = []
    
    print("\n" + "="*60)
    print("Benchmarking search times...")
    print(f"(warmup_runs={warmup_runs}, num_runs={num_runs})")
    print("="*60)
    
    for database_size in database_sizes:
        print(f"\nBuilding database with {database_size} structures...")
        
        if database_size > len(structure_files):
            print(f"Duplicating structures to reach {database_size} (have {len(structure_files)} structures)")
            db_files = duplicate_files_to_size(structure_files, database_size)
        else:
            db_files = structure_files[:database_size]
        
        # Create database directory and database
        db_temp_dir = create_temp_structure_dir(db_files)
        target_db = tempfile.mktemp(suffix="_targetdb")
        
        try:
            # Create target database (timed)
            _, db_mean, db_std, db_times = benchmark_function(
                create_foldseek_database,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
                foldseek_binary=foldseek_binary,
                structure_dir=db_temp_dir,
                output_db=target_db,
                threads=threads
            )
            
            print(f"Database build: {db_mean:.3f}s ± {db_std:.3f}s")
            
            for query_size in query_sizes:
                print(f"Running queries ({query_size} structures)...")
                
                if query_size > len(structure_files):
                    query_files = duplicate_files_to_size(structure_files, query_size)
                else:
                    query_files = structure_files[:query_size]
                
                # Create query directory and database
                query_temp_dir = create_temp_structure_dir(query_files)
                query_db = tempfile.mktemp(suffix="_querydb")
                
                try:
                    # Create query database (timed)
                    _, enc_mean, enc_std, enc_times = benchmark_function(
                        create_foldseek_database,
                        num_runs=num_runs,
                        warmup_runs=warmup_runs,
                        foldseek_binary=foldseek_binary,
                        structure_dir=query_temp_dir,
                        output_db=query_db,
                        threads=threads
                    )
                    
                    # Search (timed)
                    def do_search():
                        result_db = tempfile.mktemp(suffix="_resultdb")
                        search_foldseek(foldseek_binary, query_db, target_db, 
                                       result_db, threads=threads, sensitivity=sensitivity)
                        # Clean up result files
                        for f in Path(result_db).parent.glob(f"{Path(result_db).name}*"):
                            f.unlink()
                        return result_db
                    
                    _, search_mean, search_std, search_times = benchmark_function(
                        do_search,
                        num_runs=num_runs,
                        warmup_runs=warmup_runs
                    )
                    
                    total_mean = enc_mean + db_mean + search_mean
                    total_std = np.sqrt(enc_std**2 + db_std**2 + search_std**2)
                    
                    print(
                        f"Query {query_size:>5} vs {database_size:>6} database: "
                        f"encode={enc_mean:.3f}s±{enc_std:.3f}s, "
                        f"db_build={db_mean:.3f}s±{db_std:.3f}s, "
                        f"search={search_mean:.3f}s±{search_std:.3f}s, "
                        f"total={total_mean:.3f}s±{total_std:.3f}s"
                    )
                    
                    results.append({
                        "query_size": query_size,
                        "database_size": database_size,
                        "encode_mean": enc_mean,
                        "encode_std": enc_std,
                        "db_build_mean": db_mean,
                        "db_build_std": db_std,
                        "search_mean": search_mean,
                        "search_std": search_std,
                        "total_mean": total_mean,
                        "total_std": total_std,
                        "num_runs": num_runs,
                        "warmup_runs": warmup_runs,
                    })
                
                finally:
                    # Clean up query database
                    shutil.rmtree(query_temp_dir)
                    for f in Path(query_db).parent.glob(f"{Path(query_db).name}*"):
                        f.unlink(missing_ok=True)
        
        finally:
            # Clean up target database
            shutil.rmtree(db_temp_dir)
            for f in Path(target_db).parent.glob(f"{Path(target_db).name}*"):
                f.unlink(missing_ok=True)
    
    return pd.DataFrame(results)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Foldseek Time Benchmark")
    parser.add_argument("--structure-dir", required=True,
                        help="Directory containing PDB structure files")
    parser.add_argument("--foldseek-binary", 
                        default="/scratch/akeluska/ismb_submission/tmvec2/binaries/foldseek",
                        help="Path to foldseek binary")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--max-structures", type=int, default=None,
                        help="Maximum structures to use")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of threads for foldseek")
    parser.add_argument("--sensitivity", type=float, default=None,
                        help="Sensitivity parameter for foldseek search")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of timed runs per benchmark")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Number of warmup runs before timing")
    parser.add_argument("--structure-extension", default=".pdb",
                        help="Structure file extension (default: .pdb)")
    args = parser.parse_args()
    
    # Verify foldseek binary exists
    foldseek_path = Path(args.foldseek_binary)
    if not foldseek_path.exists():
        raise FileNotFoundError(f"Foldseek binary not found at {args.foldseek_binary}")
    
    print(f"Using foldseek binary: {args.foldseek_binary}")
    print(f"Using {args.threads} thread(s)")
    
    start_time = time.perf_counter()
    
    # Collect structure files
    structure_files = collect_structure_files(args.structure_dir, args.structure_extension)
    
    if args.max_structures:
        structure_files = structure_files[:args.max_structures]
        print(f"Limited to {len(structure_files)} structures")
    
    if len(structure_files) == 0:
        raise ValueError(f"No structure files found in {args.structure_dir}")
    
    # Define benchmark sizes based on available structures
    max_structures = len(structure_files)
    
    # Database creation benchmark
    db_creation_sizes = [10, 100, 1000]
    if max_structures >= 5000:
        db_creation_sizes.extend([5000, 10000])
    
    db_creation_df = run_database_creation_benchmark(
        structure_files, foldseek_path, db_creation_sizes,
        threads=args.threads,
        num_runs=args.num_runs, warmup_runs=args.warmup_runs
    )
    
    # Search benchmark
    if max_structures >= 1000:
        database_sizes = [100, 1000]
        if max_structures >= 10000:
            database_sizes.append(10000)
        query_sizes = [10, 100]
        if max_structures >= 1000:
            query_sizes.append(1000)
    else:
        database_sizes = [min(100, max_structures)]
        query_sizes = [min(10, max_structures)]
    
    search_df = run_search_benchmark(
        structure_files, foldseek_path, database_sizes, query_sizes,
        threads=args.threads, sensitivity=args.sensitivity,
        num_runs=args.num_runs, warmup_runs=args.warmup_runs
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/time_benchmarks") / f"foldseek_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    db_creation_df.to_csv(output_dir / "encoding_times.csv", index=False)
    search_df.to_csv(output_dir / "query_times.csv", index=False)
    
    # Save benchmark config
    config = {
        "foldseek_binary": str(args.foldseek_binary),
        "structure_dir": args.structure_dir,
        "num_structures": len(structure_files),
        "threads": args.threads,
        "sensitivity": args.sensitivity,
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "db_creation_sizes": db_creation_sizes,
        "database_sizes": database_sizes,
        "query_sizes": query_sizes,
    }
    pd.Series(config).to_json(output_dir / "benchmark_config.json")
    
    total_benchmark_time = time.perf_counter() - start_time
    print(f"\n{'='*60}")
    print(f"Total benchmark time: {total_benchmark_time:.2f}s")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
