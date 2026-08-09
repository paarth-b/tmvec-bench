#!/bin/bash
#SBATCH --job-name=tm1-bench
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""

export HYDRA_FULL_ERROR=1

module load cuda-compat/13.0

echo "=========================================="
echo "Running TM-Vec 1 predictions on full SCOPe40..."
echo ""
uv run python -m src.accuracy_benchmarks.tmvec1 --dataset scope40
echo ""
echo "=========================================="

echo "=========================================="
echo "Running TM-Vec 1 predictions on full CATH..."
echo ""
uv run python -m src.accuracy_benchmarks.tmvec1
echo "=========================================="

echo ""
echo "=========================================="
echo "Generating density scatter plots for TM-Vec 1..."
echo "=========================================="
uv run python -m src.util.graphs tmvec1
echo "=========================================="

echo ""
echo "=========================================="
echo "Running TM-Vec 1 Model Time Benchmark..."
echo "=========================================="
uv run python -m src.time_benchmarks.tmvec1_time_benchmark
