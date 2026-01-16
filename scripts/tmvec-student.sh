#!/bin/bash
#SBATCH -A grp_qzhu44               # CUSTOMIZE: your account
#SBATCH -N 1                        # number of nodes
#SBATCH -c 4                        # CUSTOMIZE: number of cores
#SBATCH -t 1-00:00:00               # CUSTOMIZE: time in d-hh:mm:ss
#SBATCH -p public                   # CUSTOMIZE: partition
#SBATCH -G a100:1                   # CUSTOMIZE: GPU type and count
#SBATCH --mem=80G                   # CUSTOMIZE: memory
#SBATCH -q public                   # CUSTOMIZE: QOS
#SBATCH -o slurm_logs/slurm.%j.out  # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm_logs/lurm.%j.err   # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"    # CUSTOMIZE: your email
#SBATCH --export=NONE               # Purge the job-submitting shell environment

# Get the repository root directory (parent of scripts directory)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""

# Set hydra's verbosity to full error
export HYDRA_FULL_ERROR=1

# Set HF_HOME to cache directory (customize as needed)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# CUSTOMIZE TO YOUR MACHINE: Load required software and activate environment
module load python/miniforge3_pytorch/2.7.0

DATASET=${1:-cath}

if [ "$DATASET" = "scope40" ]; then
    FASTA_FILE="$REPO_ROOT/data/fasta/scope40-2500.fa"
    OUTPUT_FILE="$REPO_ROOT/results/scope40_tmvec_student_similarities.csv"
    echo "Model: TM-Vec Student binaries/tmvec_student.pt"
    echo "FASTA: ${FASTA_FILE} (2500 sequences)"
    echo "Output: ${OUTPUT_FILE}"
    echo ""
    echo "Running TM-Vec Student predictions on SCOPe40-2500..."
    echo ""
    python -m src.benchmarks.tmvec_student scope40
else
    FASTA_FILE="$REPO_ROOT/data/fasta/cath-domain-seqs-S100-1k.fa"
    OUTPUT_FILE="$REPO_ROOT/results/tmvec_student_similarities.csv"
    echo "Model: TM-Vec Student binaries/tmvec_student.pt"
    echo "FASTA: ${FASTA_FILE} (1000 sequences)"
    echo "Output: ${OUTPUT_FILE}"
    echo ""
    echo "Running TM-Vec Student predictions on CATH S100..."
    echo ""
    python -m src.benchmarks.tmvec_student
fi

echo ""
echo "=========================================="
echo "TM-Vec Student Predictions Complete!"
echo "End: $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  ${OUTPUT_FILE}"
