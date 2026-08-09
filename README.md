# TM-Vec 2 Benchmarking

Benchmarking library for TMVec-2 Suite, comparing to structure alignment methods like Foldseek and TMAlign.

## Description

This repo benchmarks five protein structure similarity methods against TM-Align scores:
- **Foldseek**: Fast structure comparison using 3Di sequences
- **TM-Vec**: Neural network model for TM-score prediction from ProtT5-XL embeddings
- **TM-Vec 2**: Optimized architecture using Lobster-24M foundation model
- **TM-Vec 2s**: BiLSTM student model distilled from TM-Vec 2
- **pLM-BLAST**: Local alignment of per-residue ProtT5 embeddings ([labstructbioinf/pLM-BLAST](https://github.com/labstructbioinf/pLM-BLAST))

## Installation

### 1. Clone Repository

Clone with the supplementary submodule (includes pre-bundled datasets, binaries,
and time benchmark results):

```bash
git clone --recursive https://github.com/paarth-b/tmvec-bench.git
cd tmvec-bench
```

If you already cloned without `--recursive`, initialize the submodule:

```bash
git submodule update --init --recursive
```

The submodule (`tmvec_bench_supplementary/`) provides three symlinked directories:
- `binaries/` — TMalign binary, student model checkpoint, config files
- `data/` — CATH/SCOPe FASTA files, CATH PDB zip, classification data
- `results/` — pre-computed time benchmark results

Without the submodule, these directories will be empty symlinks. The benchmark
and plotting code will still run once you download the required data manually
(see Dataset Setup below).

### 2. Install Python Dependencies

Using `uv` (recommended):

Install `uv` if not already installed:
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Install dependencies using `uv`:
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

Download from [Foldseek GitHub releases](https://github.com/steineggerlab/foldseek/releases/).
Place the Foldseek executable in `binaries/foldseek`:

```bash
# Linux AVX2 build (check using: cat /proc/cpuinfo | grep avx2)
wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
tar xvzf foldseek-linux-avx2.tar.gz
mv foldseek/bin/foldseek binaries/foldseek
chmod +x binaries/foldseek
```

```bash
# Linux ARM64 build
wget https://mmseqs.com/foldseek/foldseek-linux-arm64.tar.gz
tar xvzf foldseek-linux-arm64.tar.gz
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

Using huggingface cli (recommended):
```bash
huggingface-cli download scikit-bio/tmvec-cath tm_vec_cath_model.ckpt --local-dir binaries/
```

Or download manually from [HuggingFace Hub](https://huggingface.co/scikit-bio/tmvec-cath/tree/main) and place `tm_vec_cath_model.ckpt` in `binaries/`.

#### TM-Vec 2 Models

```bash
# TM-Vec 2 (Lobster-based teacher model) - auto-downloaded from HuggingFace
# on first run via huggingface_hub. To pre-download:
huggingface-cli download scikit-bio/tmvec-2

# TM-Vec 2s (student model) - already provided in binaries/
# File: binaries/tmvec2_student.pt
```

The configuration file `binaries/tm_vec_cath_model_params.json` is already included in the repository.

## Dataset Setup

### CATH S100 Dataset

The benchmarks use 10,000 non-redundant domains from CATH S100. The FASTA file is already provided at `data/fasta/cath-s100-unique-10k.fa`.

**PDB structures:** A zip file with 1,000 CATH S100 PDB structures is provided for convenience. For the full 10,000-domain benchmark, use the download script (below).

```bash
# Option A: Use provided 1,000-structure subset (for quick testing)
unzip data/cath-pdb.zip -d data/
# Structures will be at data/pdb/cath-s100/
```

```bash
# Option B: Download all 10,000 structures from CATH Database
python src/util/download_structures.py \
    --fasta data/fasta/cath-s100-unique-10k.fa \
    --output-dir data/pdb/cath-s100 \
    --dataset cath
```

Alternatively, use the Slurm batch script:
```bash
bash slurm/download_cath_10k.sh
```

### SCOPe40 Dataset

The benchmarks use 11,211 domains from SCOPe 2.01 clustered at 40% sequence identity. The FASTA file is already provided at `data/fasta/scop40.fasta`.

**PDB structures:** Download the SCOPe40 PDB structures (hosted on Google Drive):

```bash
wget "https://drive.usercontent.google.com/download?id=1HjtC7Dv-MZABO9wr5PYr5DPLZ6S642P6&export=download&confirm=t" -O data/scope40-pdb.zip
unzip data/scope40-pdb.zip -d data/
# Structures should be at data/pdb/scope40/
```

Alternatively, download structures from the SCOPe/ASTRAL database:
```bash
python src/util/download_structures.py \
    --fasta data/fasta/scop40.fasta \
    --output-dir data/pdb/scope40 \
    --dataset scope40
```

### CATH S100 Full Sequence File (optional)

For time benchmarks that require the full CATH S100 sequence set, unzip the provided archive:

```bash
unzip data/fasta/cath-domain-seqs.zip -d data/fasta
# Produces data/fasta/cath-domain-seqs.fa
```

The CATH domain classification file (`data/fasta/cath-domain-list-S100.txt`) and full sequence file (`data/fasta/cath-domain-seqs-S100.fa`) are already included.

## Running Benchmarks

### Accuracy Benchmarks

All accuracy benchmarks write results to `results/` as CSV files with the format:

| seq1_id | seq2_id | tm_score | evalue (Foldseek only) |
|---------|---------|----------|------------------------|
| 107lA00 | 108lA00 | 0.8523   | 1.2e-10               |

Run individual benchmarks:

```bash
# TM-Vec 2 (teacher model, Lobster-24M based)
uv run python -m src.accuracy_benchmarks.tmvec2 --dataset cath
uv run python -m src.accuracy_benchmarks.tmvec2 --dataset scope40

# TM-Vec 2s (student model)
uv run python -m src.accuracy_benchmarks.tmvec2_student --dataset cath
uv run python -m src.accuracy_benchmarks.tmvec2_student --dataset scope40

# TM-Vec (original ProtT5-based model)
uv run python -m src.accuracy_benchmarks.tmvec1 --dataset cath
uv run python -m src.accuracy_benchmarks.tmvec1 --dataset scope40

# Foldseek (structure alignment)
uv run python -m src.accuracy_benchmarks.foldseek --dataset cath
uv run python -m src.accuracy_benchmarks.foldseek --dataset scope40

# TM-align (ground truth, CPU-only, may take >10 hours for 10k domains)
uv run python -m src.accuracy_benchmarks.tmalign --dataset cath
uv run python -m src.accuracy_benchmarks.tmalign --dataset scope40
```

> **_NOTE:_**  TM-align is a CPU-based script and may take a long time (>10 hours) to generate pairwise scores for 10,000 domains (49,995,000 pairs). To run on a smaller subset, pass a shorter FASTA file with `--fasta`.

### pLM-BLAST

pLM-BLAST lives in a sibling repository. Clone it and expose the path:

```bash
git clone https://github.com/labstructbioinf/pLM-BLAST.git ../pLM-BLAST
# Install its requirements in a dedicated venv (see pLM-BLAST README)
export PLMBLAST_REPO=$(realpath ../pLM-BLAST)
```

The benchmark auto-detects `$PLMBLAST_REPO/benchmark/bin/python` if that venv exists,
otherwise it falls back to the active interpreter.

```bash
uv run python -m src.accuracy_benchmarks.plmblast --dataset cath
uv run python -m src.accuracy_benchmarks.plmblast --dataset scope40
```

### Time Benchmarks

Time benchmarks measure encoding and query throughput for each method:

```bash
uv run python -m src.time_benchmarks.tmvec2_time_benchmark
uv run python -m src.time_benchmarks.tmvec1_time_benchmark
uv run python -m src.time_benchmarks.student_time_benchmark
uv run python -m src.time_benchmarks.foldseek_time_benchmark --structure-dir data/pdb/cath-s100
uv run python -m src.time_benchmarks.tmalign_time_benchmark
uv run python -m src.time_benchmarks.diamond_time_benchmark --fasta data/fasta/cath-domain-seqs-S100.fa
```

### Slurm Batch Scripts

For cluster environments, use the scripts in `slurm/`:

```bash
# These run the benchmarks on both CATH and SCOPe40 datasets, generate plots,
# and run time benchmarks.
bash slurm/tmvec2_student.sh
bash slurm/tmvec2.sh
bash slurm/tmvec1.sh
bash slurm/foldseek.sh
bash slurm/tmalign.sh

# pLM-BLAST requires a dataset argument:
bash slurm/plmblast.sh cath
bash slurm/plmblast.sh scope40
```

> **_NOTE:_** Slurm scripts contain SBATCH directives for a specific cluster. Edit the `#SBATCH` lines (partition, account, etc.) to match your cluster configuration.

## Visualization

### Density Scatter Plots

Generate density scatter plots comparing predicted vs true TM-scores:

```bash
uv run python -m src.util.graphs tmvec2
uv run python -m src.util.graphs tmvec1
uv run python -m src.util.graphs tmvec2_student
uv run python -m src.util.graphs foldseek
```

Plots are saved to `figures/{dataset}/{method}/density_scatter.png`.

### ROC Curves (Homology Detection)

```bash
# Generate ground truth classification files first
uv run python -m src.plotting.get_truth_cath
uv run python -m src.plotting.get_truth_scope

# Generate ROC curves
uv run python -m src.plotting.plot_roc --dataset cath
uv run python -m src.plotting.plot_roc --dataset scope40
```

Plots are saved to `figures/{dataset}/roc.png`.

### Accuracy and Homology Analysis

Merge all method results into a single table, then generate accuracy and homology plots:

```bash
# Merge results into a combined table
uv run python -m src.plotting.merge_results --dataset cath
uv run python -m src.plotting.merge_results --dataset scope40

# Plot TM-score prediction accuracy (correlation, error, confusion matrices)
uv run python -m src.plotting.plot_accuracy --dataset cath
uv run python -m src.plotting.plot_accuracy --dataset scope40

# Calculate homology detection metrics
uv run python -m src.plotting.calc_homology --dataset cath
uv run python -m src.plotting.calc_homology --dataset scope40

# Plot homology detection metrics (PR curves, mean AP, ROC(n), hits@K)
uv run python -m src.plotting.plot_homology --dataset cath
uv run python -m src.plotting.plot_homology --dataset scope40
```

For the "all pairs" variant (including pairs not reported by all methods):

```bash
uv run python -m src.plotting.calc_homology_all --dataset cath
uv run python -m src.plotting.plot_homology --dataset cath --suffix .all --metrics-dir src/plotting/cath/metrics_all
```

Plots are saved to `src/plotting/{cath,scope}/plots/`.

### Runtime Plots

```bash
uv run python src/plotting/time/plot.py
```

Plots are saved to `src/plotting/time/plots/`.

## Validation of Published Results

To validate the results in the ISMB 2026 paper:

1. **Table 1 (Prediction Accuracy)**: Run all benchmarks on both CATH and SCOPe40, then run `merge_results` + `plot_accuracy` to generate correlation, error, and confusion matrix plots.

2. **Figure 4 (TM-score Prediction)**: Generate density scatter plots using `src/util/graphs.py` showing correlation between predicted and true TM-scores.

3. **Figure 5 (Homology Detection)**: Run `get_truth` + `plot_roc` to compute ROC/PR curves at different hierarchy levels (Class → Superfamily/Family).

4. **Supplementary Tables (Runtime)**: Time benchmarks are in `src/time_benchmarks/`. Run `src/plotting/time/plot.py` to generate runtime comparison plots.
