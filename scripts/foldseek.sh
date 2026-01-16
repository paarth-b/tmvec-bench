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

mkdir -p logs/$SLURM_JOB_ID

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo ""

# CUSTOMIZE TO YOUR MACHINE: Load required software and activate environment
# module load mamba/latest            # Replace with your module system
# source activate tmvec_distill       # Replace with your environment name
# module load python/miniforge3_pytorch/2.7.0

DATASET=${1:-cath}
FOLDSEEK_BIN=binaries/foldseek

# Local default
THREADS=${SLURM_CPUS_PER_TASK:-8}

if [ "$DATASET" = "scope40" ]; then
    STRUCTURE_DIR=data/scope40pdb
    OUTPUT_FILE=results/scope40_foldseek_similarities.parquet
    echo "Foldseek binary: $FOLDSEEK_BIN"
    echo "Structure dir: $STRUCTURE_DIR"
    echo "Output: $OUTPUT_FILE"
    echo ""
    echo "Running Foldseek benchmark on SCOPe40-2500..."
    echo ""
    python -m src.benchmarks.foldseek_benchmark \
        --structure-dir "$STRUCTURE_DIR" \
        --foldseek-bin "$FOLDSEEK_BIN" \
        --output "$OUTPUT_FILE" \
        --threads "$THREADS"
    echo ""
    echo "=========================================="
    echo "Foldseek Benchmark Complete!"
    echo "End: $(date)"
    echo "=========================================="
    echo ""
    echo "Results:"
    echo "  results/scope40_foldseek_similarities.parquet"
else
    STRUCTURE_DIR=data/pdb/cath-s100
    OUTPUT_FILE=results/foldseek_similarities.csv
    echo "Foldseek binary: $FOLDSEEK_BIN"
    echo "Structure dir: $STRUCTURE_DIR"
    echo "Output: $OUTPUT_FILE"
    echo ""
    echo "Running Foldseek benchmark on CATH S100-1k..."
    echo ""
    python -m src.benchmarks.foldseek_benchmark \
        --structure-dir "$STRUCTURE_DIR" \
        --foldseek-bin "$FOLDSEEK_BIN" \
        --output "$OUTPUT_FILE" \
        --threads "$THREADS"
    echo ""
    echo "=========================================="
    echo "Foldseek Benchmark Complete!"
    echo "End: $(date)"
    echo "=========================================="
    echo ""
    echo "Results:"
    echo "  results/foldseek_similarities.csv"
fi
