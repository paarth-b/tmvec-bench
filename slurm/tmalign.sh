#!/bin/bash
set -e

# Get the repository root directory (parent of scripts directory)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BENCHMARK_THREADS="${BENCHMARK_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${BENCHMARK_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${BENCHMARK_THREADS}}"

echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Benchmark threads: ${BENCHMARK_THREADS}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""

# Set hydra's verbosity to full error
export HYDRA_FULL_ERROR=1

# CUSTOMIZE TO YOUR MACHINE: Load required software and activate environment
# module load python/miniforge3_pytorch/2.7.0

FASTA_FILE="$REPO_ROOT/data/fasta/scop40.fasta"
OUTPUT_FILE="$REPO_ROOT/results/scope40_tmalign_similarities.csv"
echo "=========================================="
echo "Running TM-align predictions on SCOPe40..."
echo ""
echo "Model: TM-align binaries/TMalign"
echo "FASTA: ${FASTA_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo ""
uv run python -m src.accuracy_benchmarks.tmalign --dataset scope40
echo ""
echo "=========================================="

FASTA_FILE="$REPO_ROOT/data/fasta/cath-s100-unique-10k.fa"
OUTPUT_FILE="$REPO_ROOT/results/cath_tmalign_similarities.csv"
echo "=========================================="
echo "Running TM-align predictions on CATH ..."
echo ""
echo "Model: TM-align binaries/TMalign"
echo "FASTA: ${FASTA_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo ""
uv run python -m src.accuracy_benchmarks.tmalign --dataset cath
echo "=========================================="

echo ""
echo "=========================================="
echo "Generating density scatter plots for TM-align..."
echo "=========================================="
uv run python -m src.util.graphs tmalign
echo "=========================================="

echo ""
echo "=========================================="
echo "Running TM-align Model Time Benchmark..."
echo "=========================================="
uv run python -m src.time_benchmarks.tmalign_time_benchmark \
    --threads ${BENCHMARK_THREADS}
