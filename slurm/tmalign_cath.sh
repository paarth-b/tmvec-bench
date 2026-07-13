#!/bin/bash
#SBATCH --job-name=tmalign-cath-bench
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --account=beut-dtai-gh
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j/%x.out
#SBATCH --error=logs/%j/%x.err
#SBATCH --exclusive

set -e

REPO_ROOT="/u/paarthbatra/git/tmvec-bench"
cd "$REPO_ROOT"

echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo ""

module load cuda-compat/13.0

echo "=========================================="
echo "Running TM-align predictions on full CATH..."
echo ""
uv run python -m src.accuracy_benchmarks.tmalign --dataset cath
echo "=========================================="
