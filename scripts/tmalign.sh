#!/bin/bash
#SBATCH --job-name=tmalign-bench
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

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo ""

# Customize to your machine: Load required software and activate environment
module load python/miniforge3_pytorch/2.7.0

DATASET=${1:-cath}

if [ "$DATASET" = "scope40" ]; then
    echo "FASTA: data/fasta/scope40-2500.fa (2500 sequences)"
    echo "Output: results/scope40_tmalign_similarities.csv"
    echo ""
    echo "Running TMalign benchmark on SCOPe40-2500..."
    echo ""
    python -m src.benchmarks.tmalign_benchmark scope40
    echo ""
    echo "=========================================="
    echo "| TMalign Benchmark Complete!            |"
    echo "| End: $(date)                           |"
    echo "=========================================="
    echo ""
    echo "Results:"
    echo "  results/scope40_tmalign_similarities.csv"
else
    echo "FASTA: data/fasta/cath-domain-seqs-S100-1k.fa (1000 sequences)"
    echo "Output: results/tmalign_similarities.csv"
    echo ""
    echo "Running TMalign benchmark on CATH S100-1k..."
    echo ""
    python -m src.benchmarks.tmalign_benchmark
    echo ""
    echo "=========================================="
    echo "| TMalign Benchmark Complete!            |"
    echo "| End: $(date)                           |"
    echo "=========================================="
    echo ""
    echo "Results:"
    echo "  results/tmalign_similarities.csv"
fi
