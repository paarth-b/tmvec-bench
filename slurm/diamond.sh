#!/bin/bash
set -e

# Get the repository root directory (parent of scripts directory)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BENCHMARK_THREADS="${BENCHMARK_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${BENCHMARK_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${BENCHMARK_THREADS}}"

echo "CPUs: ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Benchmark threads: ${BENCHMARK_THREADS}"
echo "Start: $(date)"
echo ""

# # SCOPe40 dataset
# echo "=========================================="
# echo "Running Diamond blastp on SCOPe40-1000..."
# echo ""
# echo "Model: Diamond (sequence-based baseline)"
# echo "FASTA: data/fasta/scope40-1000.fa (1000 sequences)"
# echo "Output: results/scope40_diamond_similarities.csv"
# echo ""
# python -m src.accuracy_benchmarks.diamond --dataset scope40 --threads ${SLURM_CPUS_PER_TASK:-32}
# echo ""
# echo "=========================================="

# # CATH dataset
# echo "=========================================="
# echo "Running Diamond blastp on CATH S100..."
# echo ""
# echo "Model: Diamond (sequence-based baseline)"
# echo "FASTA: data/fasta/cath-domain-seqs-S100-1k.fa (1000 sequences)"
# echo "Output: results/cath_diamond_similarities.csv"
# echo ""
# python -m src.accuracy_benchmarks.diamond --dataset cath --threads ${SLURM_CPUS_PER_TASK:-32}
# echo "=========================================="

# echo ""
# echo "=========================================="
# echo "Generating density scatter plots for Diamond..."
# echo "=========================================="
# python src/util/graphs.py diamond
# echo "=========================================="

echo ""
echo "=========================================="
echo "Running Diamond Time Benchmark..."
echo "=========================================="
uv run python src/time_benchmarks/diamond_time_benchmark.py \
    --fasta data/fasta/cath-domain-seqs-S100.fa \
    --diamond-bin binaries/diamond \
    --threads ${BENCHMARK_THREADS}
echo "=========================================="

echo ""
echo "Complete: $(date)"
