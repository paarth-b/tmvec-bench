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

# Get the repository root directory (parent of scripts directory)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""

# Set hydra's verbosity to full error
export HYDRA_FULL_ERROR=1

# CUSTOMIZE TO YOUR MACHINE: Load required software and activate environment
# module load python/miniforge3_pytorch/2.7.0
# module load mamba/latest && source activate tmvec_distill

FASTA_FILE="$REPO_ROOT/data/fasta/scope40-1000.fa"
OUTPUT_FILE="$REPO_ROOT/results/scope40_foldseek_similarities.csv"
echo "=========================================="
echo "Running Foldseek predictions on SCOPe40-1000..."
echo ""
echo "Model: Foldseek binaries/foldseek"
echo "FASTA: ${FASTA_FILE} (1000 sequences)"
echo "Output: ${OUTPUT_FILE}"
echo ""
python -m src.benchmarks.foldseek scope40
echo ""
echo "=========================================="

FASTA_FILE="$REPO_ROOT/data/fasta/cath-domain-seqs-S100-1k.fa"
OUTPUT_FILE="$REPO_ROOT/results/foldseek_similarities.csv"
echo "=========================================="
echo "Running Foldseek predictions on CATH S100..."
echo ""
echo "Model: Foldseek binaries/foldseek"
echo "FASTA: ${FASTA_FILE} (1000 sequences)"
echo "Output: ${OUTPUT_FILE}"
echo ""
python -m src.benchmarks.foldseek
echo "=========================================="

echo ""
echo "=========================================="
echo "Generating density scatter plots for Foldseek..."
echo "=========================================="
python src/util/graphs.py foldseek
echo "=========================================="

echo ""
echo "=========================================="
echo "Running Foldseek Model Time Benchmark..."
echo "=========================================="
python /scratch/akeluska/ismb_submission/tmvec2/src/time_benchmarks/foldseek_time_benchmark.py \
    --structure-dir /scratch/akeluska/tm-bench/pdb \
    --threads $SLURM_CPUS_PER_TASK 
