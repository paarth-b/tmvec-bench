# TM-Vec 2: Structure Similarity Benchmarking

Benchmarking suite for evaluating protein structure similarity methods on CATH and SCOPe datasets, as described in the ISMB 2026 submission.

## Overview

This toolkit benchmarks four protein structure similarity methods against TM-Align scores:
- **Foldseek**: Fast structure comparison using 3Di sequences
- **TM-Vec**: Neural network model for TM-score prediction from ProtT5-XL embeddings
- **TM-Vec 2**: Optimized architecture using Lobster-24M foundation model
- **TM-Vec 2s**: BiLSTM student model distilled from TM-Vec 2

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/paarth-b/tmvec-bench.git
cd tmvec-bench
```

### 2. Install Python Dependencies

Using `uv` (recommended):
```bash
uv sync
source .venv/bin/activate
```

Or using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Download Required Binaries

#### TMalign Binary

The provided binary `binaries/TMalign` requires x86-64 architecture. For other architectures (e.g., Apple Silicon), download from [Zhang Group website](https://zhanggroup.org/TM-align/).


#### Foldseek Binary

D https://github.com/steineggerlab/foldseek/releases/
Place the Foldseek executable in `binaries/foldseek`:

```bash
# For Linux x86-64:
wget https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-x86_64.tar.gz
tar xvzf foldseek-linux-x86_64.tar.gz
mv foldseek/bin/foldseek binaries/foldseek
chmod +x binaries/foldseek

# For macOS (Intel):
wget https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-osx-universal.tar.gz
tar xvzf foldseek-osx-universal.tar.gz
mv foldseek/bin/foldseek binaries/foldseek
chmod +x binaries/foldseek
```

Verify installation:
```bash
binaries/foldseek version
```

### 4. Download Model Checkpoints

#### TM-Vec (original model)

Download the TM-Vec CATH checkpoint:

```bash
# Using huggingface-cli (recommended):
hf download scikit-bio/tmvec-cath tm_vec_cath_model.ckpt --local-dir binaries/
```

Or download manually from [HuggingFace Hub](https://huggingface.co/scikit-bio/tmvec-cath/tree/main) and place `tm_vec_cath_model.ckpt` in `binaries/`.

#### TM-Vec 2 Models

```bash
# TM-Vec 2 (Lobster-based teacher model)
hf download scikit-bio/tmvec-2 --local-dir models/tmvec-2

# TM-Vec 2s (student model) - already provided in binaries/
# File: binaries/tmvec2_student.pt
```

The configuration file `binaries/tm_vec_cath_model_params.json` is already included in the repository.

## Dataset Setup

### CATH S100 Dataset

The benchmarks use the first 1,000 domains from CATH S100 (non-redundant at 100% sequence identity).

The FASTA file is already provided at `data/cath-top1k.fa`. Create the directory and download structures for the 1000 domains:

```bash
mkdir -p data/pdb/cath-s100

python src/util/download_structures.py \
    --fasta data/cath-top1k.fa \
    --output-dir data/pdb/cath-s100 \
    --dataset cath
```

This will download ~1000 PDB structures from RCSB PDB.


### SCOPe40 Dataset

The benchmarks use 1,000 domains from SCOPe 2.01 clustered at 40% sequence identity.

The FASTA file is already provided at `data/fasta/scope40-1000.fa`. Download structures for the 1000 domains:

```bash
mkdir -p data/scope40pdb

python src/util/download_structures.py \
    --fasta data/fasta/scope40-1000.fa \
    --output-dir data/scope40pdb \
    --dataset scope40
```

This downloads from ASTRAL/RCSB PDB.

## Running Benchmarks

All benchmark scripts are in `scripts/`. They can be run locally or on SLURM clusters.

```bash
# CATH
sbatch scripts/tmvec2_student.sh
sbatch scripts/foldseek.sh
sbatch scripts/tmvec2.sh
sbatch scripts/tmvec1.sh
sbatch scripts/tmalign.sh
```

## Output Files

### Similarity Results

All benchmarks generate CSV files in `results/` with the following format:

| seq1_id | seq2_id | tm_score | evalue (Foldseek only) |
|---------|---------|----------|------------------------|
| 107lA00 | 108lA00 | 0.8523   | 1.2e-10               |
| 107lA00 | 109lA00 | 0.7234   | 3.4e-08               |

For **1000 sequences**, each benchmark produces:
- **~499,500 pairwise comparisons** (all-vs-all excluding self)
- **CSV file size**: ~15-30 MB per benchmark

### Visualization

Generate plots from results:

```bash
# CATH visualizations
cd src/plotting/cath
jupyter notebook plot.ipynb

# SCOPe visualizations
cd ../scope
jupyter notebook plot.ipynb

# Runtime benchmarks
cd ../time
jupyter notebook plot.ipynb
```

Plots are saved to `figures/` and include:
- ROC curves (homology detection at different classification levels)
- PR curves (precision-recall)
- Density scatter plots (predicted vs. true TM-scores)
- Runtime comparisons (encoding and query times)

## Validation of Published Results

To validate the results in the ISMB 2026 paper:

1. **Table 1 (Prediction Accuracy)**: Run all benchmarks on both CATH and SCOPe40, then compare the generated CSVs against TM-align ground truth using the plotting notebooks.

2. **Figure 4 (TM-score Prediction)**: Generate density scatter plots showing correlation between predicted and true TM-scores.

3. **Figure 5 (Homology Detection)**: Use the ground truth classification files to compute ROC/PR curves at different hierarchy levels (Class → Superfamily/Family).

4. **Supplementary Tables (Runtime)**: Time benchmarks are in `src/time_benchmarks/`. Results should match the encoding/query time tables.

Expected runtime for full benchmark suite (per dataset):
- TM-align: ~12-24 hours (CPU-bound, can parallelize)
- Foldseek: ~30-60 minutes (GPU-accelerated)
- TM-Vec: ~6-8 hours (ProtT5-XL embedding generation)
- TM-Vec 2: ~1-2 hours (Lobster-24M is much faster)
- TM-Vec 2s: ~5-10 minutes (no PLM, direct BiLSTM)

## File Structure

After setup, your directory should look like:

```
tmvec-bench/
├── binaries/
│   ├── foldseek                      # Downloaded
│   ├── TMalign                       # Downloaded
│   ├── tm_vec_cath_model.ckpt       # Downloaded from HF
│   ├── tm_vec_cath_model_params.json # Provided
│   └── tmvec2_student.pt            # Provided
├── data/
│   ├── cath-top1k.fa                # Provided
│   ├── fasta/
│   │   ├── scope40-1000.fa          # Provided
│   │   └── cath-domain-seqs-S100-1k.fa  # Alternative CATH sequences
│   ├── pdb/
│   │   └── cath-s100/               # Downloaded (~1000 PDB files)
│   └── scope40pdb/                  # Downloaded (~1000 PDB files)
├── models/
│   └── tmvec-2/                     # Downloaded from HF
├── results/                         # Generated by benchmarks
│   ├── cath_tmalign_similarities.csv
│   ├── cath_foldseek_similarities.csv
│   ├── cath_tmvec1_similarities.csv
│   ├── cath_tmvec2_similarities.csv
│   ├── cath_tmvec2_student_similarities.csv
│   └── scope40_*.csv                # SCOPe results
├── figures/                         # Generated by plotting scripts
└── src/
    ├── benchmarks/                  # Benchmark implementations
    ├── plotting/                    # Visualization notebooks
    ├── time_benchmarks/             # Runtime profiling
    └── util/                        # Helper scripts
```
