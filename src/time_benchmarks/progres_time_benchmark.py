#!/usr/bin/env python
"""
Time benchmark for Progres (Protein Graph Embedding Search).
Measures embedding and similarity calculation times for different dataset sizes.
"""

import argparse
import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .benchmark_env import capture_benchmark_environment, write_json
except ImportError:
    from benchmark_env import capture_benchmark_environment, write_json

# Add progres to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "progres"))

import progres as pg


# ==============================================================================
# TIMING UTILITIES
# ==============================================================================

def timed_run(func, *args, **kwargs):
    """
    Run a function with timing.
    Returns (result, elapsed_seconds).
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
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

def collect_structure_files(structure_dir, extension=".pdb", max_files=None):
    """
    Collect structure files from a directory efficiently.
    """
    structure_path = Path(structure_dir)
    structure_files = []

    print(f"Scanning {structure_dir} for {extension} files...")

    with os.scandir(structure_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(extension):
                structure_files.append(Path(entry.path))
                if max_files and len(structure_files) >= max_files:
                    break

    structure_files.sort()
    print(f"Found {len(structure_files)} structure files in {structure_dir}")
    return structure_files


def duplicate_files_to_size(files, target_size):
    """
    Duplicate file list to reach target size.
    """
    if len(files) >= target_size:
        return files[:target_size]

    repeats_needed = (target_size // len(files)) + 1
    duplicated = files * repeats_needed
    return duplicated[:target_size]


# ==============================================================================
# PROGRES OPERATIONS
# ==============================================================================

def embed_structures(structure_files, model, device='cpu'):
    """
    Generate Progres embeddings for a list of structure files.
    Returns tensor of embeddings.
    """
    embeddings = []
    for fp in structure_files:
        try:
            emb = pg.embed_structure(str(fp), device=device, model=model)
            embeddings.append(emb.cpu())
        except Exception:
            # Skip failed structures
            continue

    if not embeddings:
        raise RuntimeError("No structures could be embedded")

    return torch.stack(embeddings)


def calculate_similarity(embeddings):
    """
    Calculate pairwise similarity matrix from embeddings.
    Progres similarity: (1 + cosine) / 2
    """
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    cosine_sim = torch.mm(embeddings_norm, embeddings_norm.t())
    return (1 + cosine_sim) / 2


# ==============================================================================
# BENCHMARKS
# ==============================================================================

def run_embedding_benchmark(structure_files, model, device,
                            encoding_sizes, num_runs=3, warmup_runs=1):
    """
    Benchmark embedding times for different numbers of structures.
    """
    results = []

    print("\n" + "=" * 60)
    print("Benchmarking embedding times...")
    print(f"(warmup_runs={warmup_runs}, num_runs={num_runs})")
    print("=" * 60)

    for size in encoding_sizes:
        print(f"\nEmbedding {size} structures...")

        if size > len(structure_files):
            print(f"Duplicating structures to reach {size}")
            files = duplicate_files_to_size(structure_files, size)
        else:
            files = structure_files[:size]

        def embed_batch():
            return embed_structures(files, model, device)

        _, mean_time, std_time, all_times = benchmark_function(
            embed_batch,
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )

        structs_per_sec = size / mean_time if mean_time > 0 else 0

        print(f"Embedding {size:>6} structures: {mean_time:.3f}s ± {std_time:.3f}s "
              f"({structs_per_sec:.1f} struct/s)")

        results.append({
            "encoding_size": size,
            "mean_seconds": mean_time,
            "std_seconds": std_time,
            "structs_per_second": structs_per_sec,
            "all_times": all_times,
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
        })

    return pd.DataFrame(results)


def run_search_benchmark(structure_files, model, device,
                         database_sizes, query_sizes,
                         num_runs=3, warmup_runs=1):
    """
    Benchmark search times for different database and query sizes.

    For Progres, search is matrix multiplication of embeddings.
    Database embedding is done once (not included in query timing).
    """
    results = []

    print("\n" + "=" * 60)
    print("Benchmarking search times...")
    print(f"(warmup_runs={warmup_runs}, num_runs={num_runs})")
    print("=" * 60)

    for db_size in database_sizes:
        print(f"\nBuilding database with {db_size} structures...")

        if db_size > len(structure_files):
            db_files = duplicate_files_to_size(structure_files, db_size)
        else:
            db_files = structure_files[:db_size]

        # Pre-embed database (not timed for query benchmark)
        start_db = time.perf_counter()
        db_embeddings = embed_structures(db_files, model, device)
        db_embed_time = time.perf_counter() - start_db
        print(f"Database embedding (one-time): {db_embed_time:.3f}s")

        for query_size in query_sizes:
            print(f"Running queries ({query_size} structures)...")

            if query_size > len(structure_files):
                query_files = duplicate_files_to_size(structure_files, query_size)
            else:
                query_files = structure_files[:query_size]

            # Benchmark query embedding
            def embed_queries():
                return embed_structures(query_files, model, device)

            _, enc_mean, enc_std, enc_times = benchmark_function(
                embed_queries,
                num_runs=num_runs,
                warmup_runs=warmup_runs
            )

            # Get query embeddings for search benchmark
            query_embeddings = embed_structures(query_files, model, device)

            # Benchmark similarity calculation (search)
            def do_search():
                query_norm = F.normalize(query_embeddings, p=2, dim=1)
                db_norm = F.normalize(db_embeddings, p=2, dim=1)
                return torch.mm(query_norm, db_norm.t())

            _, search_mean, search_std, search_times = benchmark_function(
                do_search,
                num_runs=num_runs,
                warmup_runs=warmup_runs
            )

            total_mean = enc_mean + search_mean
            total_std = np.sqrt(enc_std ** 2 + search_std ** 2)

            print(
                f"Query {query_size:>5} vs {db_size:>6} database: "
                f"encode={enc_mean:.3f}s±{enc_std:.3f}s, "
                f"search={search_mean:.3f}s±{search_std:.3f}s, "
                f"total={total_mean:.3f}s±{total_std:.3f}s"
            )

            results.append({
                "query_size": query_size,
                "database_size": db_size,
                "encode_mean": enc_mean,
                "encode_std": enc_std,
                "db_embed_time_one_time": db_embed_time,
                "search_mean": search_mean,
                "search_std": search_std,
                "total_mean": total_mean,
                "total_std": total_std,
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
            })

    return pd.DataFrame(results)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Progres Time Benchmark")
    parser.add_argument("--structure-dir", required=True,
                        help="Directory containing PDB structure files")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--max-structures", type=int, default=100000,
                        help="Maximum structures to use")
    parser.add_argument("--device", default=None,
                        help="Device (cuda/cpu, auto-detects if not specified)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of threads (for torch)")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of timed runs per benchmark")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Number of warmup runs before timing")
    parser.add_argument("--structure-extension", default=".pdb",
                        help="Structure file extension (default: .pdb)")
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(args.threads)

    print(f"Using device: {device}")
    print(f"Using {args.threads} thread(s)")

    # Load model once
    print("Loading Progres model...")
    model = pg.load_trained_model(device)

    start_time = time.perf_counter()

    structure_files = collect_structure_files(
        args.structure_dir,
        args.structure_extension,
        max_files=args.max_structures
    )

    if len(structure_files) == 0:
        raise ValueError(f"No structure files found in {args.structure_dir}")

    print(f"Using {len(structure_files)} structures for benchmark")

    # Encoding sizes matching other benchmarks
    encoding_sizes = [10, 100, 1000]
    database_sizes = [1000]
    query_sizes = [10, 100, 1000]

    # Filter sizes based on available structures
    encoding_sizes = [s for s in encoding_sizes if s <= len(structure_files)]
    database_sizes = [s for s in database_sizes if s <= len(structure_files)]

    encoding_df = run_embedding_benchmark(
        structure_files, model, device, encoding_sizes,
        num_runs=args.num_runs, warmup_runs=args.warmup_runs
    )

    search_df = run_search_benchmark(
        structure_files, model, device, database_sizes, query_sizes,
        num_runs=args.num_runs, warmup_runs=args.warmup_runs
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) if args.output_dir else Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    encoding_df.to_csv(output_dir / "progres_encoding_times.csv", index=False)
    search_df.to_csv(output_dir / "progres_query_times.csv", index=False)

    # Save benchmark config
    config = {
        "device": device,
        "structure_dir": args.structure_dir,
        "num_structures": len(structure_files),
        "threads": args.threads,
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "encoding_sizes": encoding_sizes,
        "database_sizes": database_sizes,
        "query_sizes": query_sizes,
        "system_info_file": "progres_system_info.json",
    }
    pd.Series(config).to_json(output_dir / "progres_benchmark_config.json")
    write_json(
        output_dir / "progres_system_info.json",
        capture_benchmark_environment(requested_threads=args.threads, accelerator=device),
    )

    total_benchmark_time = time.perf_counter() - start_time
    print(f"\n{'=' * 60}")
    print(f"Total benchmark time: {total_benchmark_time:.2f}s")
    print(f"Results saved in: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
