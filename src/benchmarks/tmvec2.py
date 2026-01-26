#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from src.model.tmvec_2_model import TMScorePredictor, TMVecConfig
from lobster.model import LobsterPMLM

def generate_embeddings(sequences, batch_size=32, max_length=512, device='cuda'):
    """
    Generate TMvec-2 embeddings for protein sequences.

    Uses streaming processing to avoid OOM: each batch goes through Lobster -> TMvec-2
    immediately rather than accumulating all Lobster embeddings in RAM first.
    """
    # Load Lobster model
    print("Loading Lobster-24M model...")
    lobster_model = LobsterPMLM("asalam91/lobster_24M")
    tokenizer = lobster_model.tokenizer
    lobster_model.to(device)
    lobster_model.eval()

    # Load TMvec-2 model
    print("Loading TMvec-2 model from HuggingFace...")
    checkpoint_path = hf_hub_download(
        repo_id="scikit-bio/tmvec-2",
        filename="tmvec-2.ckpt"
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = TMVecConfig(
        d_model=408,
        num_layers=4,
        projection_hidden_dim=1024
    )
    tmvec_model = TMScorePredictor(config)
    tmvec_model.load_state_dict(checkpoint['state_dict'])
    tmvec_model.to(device)
    tmvec_model.eval()

    # Stream processing: Lobster -> TMvec-2 for each batch immediately
    # This avoids storing all Lobster embeddings in RAM
    print("Generating embeddings (streaming Lobster -> TMvec-2)...")
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i + batch_size]

            # Lobster encoding
            encoded = tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            outputs = lobster_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            lobster_emb = outputs.hidden_states[-1]

            # TMvec-2 encoding - process immediately, don't store Lobster output
            padding_mask = (attention_mask == 0)
            tmvec_emb = tmvec_model.encode_sequence(lobster_emb, padding_mask)

            # Store only the final embedding (batch_size x 512)
            all_embeddings.append(tmvec_emb.cpu().numpy())

            # Clear intermediate tensors
            del lobster_emb, outputs, input_ids, attention_mask, encoded, padding_mask

    # Free model memory
    del lobster_model, tmvec_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return np.concatenate(all_embeddings, axis=0)


def load_fasta(fasta_path, max_sequences=None):
    """Load sequences from FASTA file."""
    seq_ids, sequences = [], []
    current_id, current_seq = None, []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if current_id:
                    seq_ids.append(current_id)
                    sequences.append(''.join(current_seq))
                    if max_sequences and len(seq_ids) >= max_sequences:
                        break
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id and (not max_sequences or len(seq_ids) < max_sequences):
            seq_ids.append(current_id)
            sequences.append(''.join(current_seq))

    print(f"Loaded {len(seq_ids)} sequences")
    return seq_ids, sequences

def calculate_scores(embeddings):
    print("Calculating pairwise scores...")
    embeddings_tensor = torch.from_numpy(embeddings)
    embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=1)
    tm_matrix = torch.mm(embeddings_norm, embeddings_norm.t()).numpy()
    print(f"Mean: {tm_matrix.mean():.4f}, Std: {tm_matrix.std():.4f}")
    return tm_matrix


def save_results(seq_ids, tm_matrix, output_path):
    print(f"Saving to {output_path}...")
    seq1_ids = []
    seq2_ids = []
    tm_scores = []

    for i in range(len(seq_ids)):
        for j in range(i + 1, len(seq_ids)):
            seq1_ids.append(seq_ids[i])
            seq2_ids.append(seq_ids[j])
            tm_scores.append(float(tm_matrix[i, j]))

    df = pd.DataFrame({
        'seq1_id': seq1_ids,
        'seq2_id': seq2_ids,
        'tm_score': tm_scores
    })
    df.to_csv(output_path, index=False)
    print(f"Saved {len(tm_scores):,} scores")


def main():
    parser = argparse.ArgumentParser(description="TMvec-2 TM-score prediction")
    parser.add_argument("--dataset", choices=['cath', 'scope40'], default='cath',
                        help="Dataset to use (cath or scope40)")
    parser.add_argument("--fasta", default=None, help="FASTA file path (overrides dataset default)")
    parser.add_argument("--output", default=None, help="Output CSV path (overrides dataset default)")
    parser.add_argument("--max-sequences", type=int, default=1000, help="Maximum sequences to process")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding generation")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu, auto-detects if not specified)")

    args = parser.parse_args()

    # Set dataset-specific defaults
    if args.dataset == 'scope40':
        fasta = args.fasta or "data/fasta/scope40-1000.fa"
        output = args.output or "results/scope40_tmvec2_similarities.csv"
    else:
        fasta = args.fasta or "data/cath-top1k.fa"
        output = args.output or "results/cath_tmvec2_similarities.csv"

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("TMvec-2 TM-Score Prediction")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Device: {device}")
    print(f"FASTA: {fasta}")
    print(f"Output: {output}")
    print(f"Max sequences: {args.max_sequences}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)

    seq_ids, sequences = load_fasta(fasta, args.max_sequences)
    tmvec_embeddings = generate_embeddings(sequences, args.batch_size, device=device)
    tm_matrix = calculate_scores(tmvec_embeddings)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    save_results(seq_ids, tm_matrix, output)

    print("=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
