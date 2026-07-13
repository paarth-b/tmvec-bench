#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BENCHMARK_THREADS="${BENCHMARK_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${BENCHMARK_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${BENCHMARK_THREADS}}"

echo "CPUs: ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Benchmark threads: ${BENCHMARK_THREADS}"
echo "Start: $(date)"
echo ""

export HYDRA_FULL_ERROR=1

echo ""
echo "=========================================="
echo "Running Foldseek CPU default Time Benchmark..."
echo "=========================================="
python3 src/time_benchmarks/foldseek_time_benchmark.py \
    --structure-dir data/pdb/cath-s100 \
    --threads ${BENCHMARK_THREADS} \
    --output-dir results/time_benchmarks/foldseek_cpu_default
