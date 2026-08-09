#!/bin/bash
#SBATCH --job-name=foldseek-bench
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --account=beut-dtai-gh
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j/%x.out
#SBATCH --error=logs/%j/%x.err
#SBATCH --exclusive

set -e

# Get the repository root directory (parent of slurm directory)
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
# module load uv run python/miniforge3_pytorch/2.7.0
# module load mamba/latest && source activate tmvec_distill

echo "=========================================="
echo "Running Foldseek predictions on full SCOPe40..."
echo ""
uv run python -m src.accuracy_benchmarks.foldseek --dataset scope40
echo ""
echo "=========================================="

echo "=========================================="
echo "Running Foldseek predictions on full CATH..."
echo ""
uv run python -m src.accuracy_benchmarks.foldseek
echo "=========================================="

echo ""
echo "=========================================="
echo "Generating density scatter plots for Foldseek..."
echo "=========================================="
uv run python -m src.util.graphs foldseek
echo "=========================================="

echo ""
echo "=========================================="
echo "Running Foldseek Model Time Benchmark..."
echo "=========================================="
uv run python -m src.time_benchmarks.foldseek_time_benchmark \
    --structure-dir data/pdb/cath-s100 \
    --threads ${BENCHMARK_THREADS}
