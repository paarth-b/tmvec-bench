#!/bin/bash
#SBATCH --job-name=download-cath-10k
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --account=beut-dtai-gh
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j/%x.out
#SBATCH --error=logs/%j/%x.err

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Start: $(date)"

NEW_DIR="data/pdb/CATH_new"
OLD_DIR="data/pdb/CATH"

mkdir -p "$NEW_DIR"

echo "=========================================="
echo "Downloading CATH domain structures for 10k set..."
echo "=========================================="
uv run python src/util/download_structures.py \
    --fasta data/fasta/cath-s100-unique-10k.fa \
    --output-dir "$NEW_DIR" \
    --dataset cath

DOWNLOADED=$(ls "$NEW_DIR" | wc -l)
echo "Downloaded $DOWNLOADED structures"

if [ "$DOWNLOADED" -lt 9000 ]; then
    echo "ERROR: Too few structures downloaded ($DOWNLOADED). Aborting replacement."
    exit 1
fi

echo "=========================================="
echo "Replacing $OLD_DIR with $NEW_DIR..."
echo "=========================================="
rm -rf "$OLD_DIR"
mv "$NEW_DIR" "$OLD_DIR"

echo "Done: $(ls $OLD_DIR | wc -l) structures in $OLD_DIR"
echo "End: $(date)"
