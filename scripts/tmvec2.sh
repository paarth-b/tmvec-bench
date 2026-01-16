#!/bin/bash
#SBATCH --job-name=tm2-bench
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

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""

# Customize to your machine: Load override module from deltaAI
# module load python/miniforge3_pytorch/2.7.0

# Configure PYTHONPATH
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

DATASET=${1:-cath}

if [ "$DATASET" = "scope40" ]; then
    echo "Model: TM-Vec 2 (Lobster-24M) tmvec2_model.ckpt"
    echo "FASTA: data/fasta/scope40-2500.fa (2500 sequences)"
    echo "Output: results/scope40_tmvec2_similarities.csv"
    echo ""
    echo "Running TM-Vec 2 predictions on SCOPe40-2500..."
    echo ""
    python -m src.benchmarks.tmvec_2 scope40
    echo ""
    echo "=========================================="
    echo "TM-Vec 2 Predictions Complete!"
    echo "End: $(date)"
    echo "=========================================="
    echo ""
    echo "Results:"
    echo "  results/scope40_tmvec2_similarities.csv"
else
    echo "Model: TM-Vec 2 (Lobster-24M) tmvec2_model.ckpt"
    echo "FASTA: data/fasta/cath-domain-seqs-S100-1k.fa (1000 sequences)"
    echo "Output: results/tmvec2_similarities.csv"
    echo ""
    echo "Running TM-Vec 2 predictions on CATH S100..."
    echo ""
    python -m src.benchmarks.tmvec_2
    echo ""
    echo "=========================================="
    echo "TM-Vec 2 Predictions Complete!"
    echo "End: $(date)"
    echo "=========================================="
    echo ""
    echo "Results:"
    echo "  results/tmvec2_similarities.csv"
fi
