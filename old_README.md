# TM-Vec Student: Structure Similarity Benchmarking

Benchmarking suite for evaluating other TM-score prediction methods on protein structures.

## Overview

This toolkit benchmarks three protein structure similarity methods against TM-Align scores:
- **Foldseek**: Fast structure comparison using 3Di sequences
- **TM-Vec**: Neural network model for TM-score prediction from sequence embeddings
- **TM-Vec Student**: BiLSTM Architecture distilling TM-Vec embeddings to predict from sequence


## Installation

1. Install dependencies using uv or pip:
```bash
uv sync
# or
pip install -r requirements.txt
```
```bash
# activate your environment:
source .venv/bin/activate
```

```bash
# then, unzip pdb files in the data folder (or download)
unzip 'data/cath-s100-structures.zip'
python 'src/util/pdb_downloader.py'
# make sure to move the cath-s100 unzipped folder into data/
```
> **Note:** Both TMAlign and Foldseek binary can be compiled from source as well. The code to download foldseek directly is provided below. 
2. Place required binaries in `binaries/`:
   - `foldseek` - Download from [Foldseek repository](https://github.com/steineggerlab/foldseek)
   OR 
      ```bash 
      wget https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-gpu.tar.gz
      ```

   - `TMAlign` - Use provided binary or download from Bioconda:
      ```bash
      conda install bioconda::tmalign
      ```
   
3. Download model checkpoints from Hugging Face Hub:

   ```bash
   hf download scikit-bio/tmvec-cath --local-dir models/tmvec-cath
   ```


   **Option C - Manual download:**
   Download from [tmvec-cath](https://huggingface.co/scikit-bio/tmvec-cath/tree/main) and place in `models/tmvec-cath/`

## Running Benchmarks

Run benchmarks directly using Python modules:

**Foldseek Benchmark:**
```bash
bash scripts/foldseek.sh
```

**TMalign Benchmark:**
```bash
bash scripts/tmalign.sh
```
> **Note:** Given TMalign binary requires x86-64 architecture. See Installation section for details.

**TM-Vec Benchmark:**
```bash
bash scripts/tmvec1.sh
```

**TM-Vec Student Model:**
```bash
bash scripts/tmvec-student.sh
```

## Input Data

### FASTA Files
Protein sequences in FASTA format (e.g., `data/fasta/cath-domain-seqs-S100-1k.fa`):
```
>cath|4_4_0|107lA00
MDPSTPPGVPPGETVSGGDNFTVKKLRKEGWVS...
>cath|4_4_0|108lA00
MKLLPLTALLLLGTVALVAAEAAPLKDVEQSSSQ...
```

### PDB Structures
PDB files must be placed in `data/pdb/cath-s100/` directory. The structures can be loaded from the `cath-s100-structures.zip` file, or automatically downloaded using the PDB downloader utility.

## Output Files

All benchmarks generate CSV files in `results/` with pairwise similarity scores:

| query_id | target_id | tm_score |
|----------|-----------|----------|
| cath\|4_4_0\|107lA00 | cath\|4_4_0\|108lA00 | 0.8523 |
| cath\|4_4_0\|107lA00 | cath\|4_4_0\|109lA00 | 0.7234 |

Visualization plots are saved to `figures/`.
