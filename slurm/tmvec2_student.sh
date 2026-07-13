#!/bin/bash
#SBATCH --job-name=tm2-student-bench
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

# Get the repository root directory
REPO_ROOT="/u/paarthbatra/git/tmvec-bench"
cd "$REPO_ROOT"

echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""

# Set hydra's verbosity to full error
export HYDRA_FULL_ERROR=1

# CUSTOMIZE TO YOUR MACHINE: Load required software and activate environment
module load cuda-compat/13.0

echo "=========================================="
echo "Running TM-Vec 2 Student predictions on full SCOPe40..."
echo ""
uv run python -m src.accuracy_benchmarks.tmvec2_student scope40
echo ""
echo "=========================================="

echo "=========================================="
echo "Running TM-Vec 2 Student predictions on full CATH..."
echo ""
uv run python -m src.accuracy_benchmarks.tmvec2_student
echo "=========================================="

echo ""
echo "=========================================="
echo "Generating density scatter plots for TM-Vec 2 Student..."
echo "=========================================="
uv run python -m src.util.graphs tmvec2_student
echo "=========================================="

echo ""
echo "=========================================="
echo "Running TM-Vec 2 Student Model Time Benchmark..."
echo "=========================================="
uv run python -m src.time_benchmarks.student_time_benchmark
