#!/bin/bash
#SBATCH --job-name=plmblast-bench
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --account=beut-dtai-gh
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j/%x.out
#SBATCH --error=logs/%j/%x.err
#SBATCH --exclusive

set -e

DATASET="${1:?usage: sbatch slurm/plmblast.sh <scope40|cath> [cpc]}"
CPC="${2:-90}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Dataset: $DATASET"
echo "CPC:     $CPC"
echo "CPUs:    $SLURM_CPUS_PER_TASK"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start:   $(date)"
echo ""

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# pLM-BLAST is a sibling repo with its own virtualenv; the benchmark script
# auto-detects <repo>/benchmark/bin/python when PLMBLAST_REPO is set.
export PLMBLAST_REPO="${PLMBLAST_REPO:-$(realpath ../pLM-BLAST)}"

uv run python -u -m src.accuracy_benchmarks.plmblast \
    --dataset "$DATASET" \
    --cosine-cutoff "$CPC" \
    --workers "${SLURM_CPUS_PER_TASK:-16}"
