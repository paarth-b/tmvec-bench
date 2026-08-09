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

# CUSTOMIZE TO YOUR MACHINE: Load required software and activate environment
# module load python/miniforge3_pytorch/2.7.0
# module load mamba/latest && source activate tmvec_distill

# NOTE: Foldseek accuracy results are identical between CPU and GPU modes
# (GPU only accelerates the prefilter step, not the alignment scoring).
# Run scripts/foldseek.sh for accuracy benchmarks and plots.

echo ""
echo "=========================================="
echo "Running Foldseek GPU default Time Benchmark..."
echo "=========================================="
uv run python src/time_benchmarks/foldseek_time_benchmark.py \
    --structure-dir data/pdb/cath-s100 \
    --threads ${BENCHMARK_THREADS} \
    --use-gpu \
    --output-dir results/time_benchmarks/foldseek_gpu_default
