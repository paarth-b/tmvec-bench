#!/usr/bin/env python
"""TM-Vec Student: TM-score predictions for CATH and SCOPe."""

import sys
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

from src.model.student_model import StudentModel, encode_sequence


def load_fasta(fasta_path, max_sequences):
    """Load sequences from FASTA file."""
    seq_ids, sequences = [], []
    current_id, current_seq = None, []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id:
                    seq_ids.append(current_id)
                    sequences.append("".join(current_seq))
                    if len(seq_ids) >= max_sequences:
                        break
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id and len(seq_ids) < max_sequences:
            seq_ids.append(current_id)
            sequences.append("".join(current_seq))

    print(f"Loaded {len(seq_ids)} sequences")
    return seq_ids, sequences


def load_model(checkpoint_path, device):
    """Load student model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

    model = StudentModel()
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model


def compute_embeddings(model, sequences, max_length, batch_size, device):
    """Encode sequences to embeddings."""
    print("Encoding sequences...")
    tokens = torch.stack([encode_sequence(seq, max_length) for seq in sequences])
    embeddings = []

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc="Encoding"):
            end = min(len(sequences), start + batch_size)
            batch = tokens[start:end].to(device)
            embeddings.append(model.seq_encoder(batch).cpu())

    return torch.cat(embeddings, dim=0)


def predict_scores(model, embeddings, seq_ids, output_path, batch_size, device):
    """Compute pairwise TM-scores using batched operations."""
    n = len(seq_ids)
    total_pairs = n * (n - 1) // 2

    print(f"Scoring {total_pairs:,} pairs...")

    predictor = model.tm_predictor.to(device)
    embeddings = embeddings.to(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all results in memory
    all_seq1_ids = []
    all_seq2_ids = []
    all_scores = []

    with torch.no_grad():
        with tqdm(total=total_pairs, desc="Predicting") as pbar:
            for i in range(n - 1):
                emb_i = embeddings[i:i + 1]
                emb_j = embeddings[i + 1:]

                # Process in chunks to manage memory
                for start in range(0, len(emb_j), batch_size):
                    end = min(len(emb_j), start + batch_size)
                    batch = emb_j[start:end]
                    batch_count = len(batch)

                    # Create pairwise features in both directions
                    emb_i_expanded = emb_i.expand(batch_count, -1)

                    # Direction 1: [emb_i, emb_j, emb_i * emb_j, |emb_i - emb_j|]
                    features_ij = torch.cat([
                        emb_i_expanded,
                        batch,
                        emb_i_expanded * batch,
                        torch.abs(emb_i_expanded - batch)
                    ], dim=1)

                    # Direction 2: [emb_j, emb_i, emb_j * emb_i, |emb_j - emb_i|]
                    features_ji = torch.cat([
                        batch,
                        emb_i_expanded,
                        batch * emb_i_expanded,
                        torch.abs(batch - emb_i_expanded)
                    ], dim=1)

                    # Predict TM-scores in both directions and average
                    scores_ij = predictor(features_ij).squeeze(-1)
                    scores_ji = predictor(features_ji).squeeze(-1)
                    scores = ((scores_ij + scores_ji) / 2).cpu()

                    if batch_count == 1:
                        scores = [scores]

                    # Collect results
                    for k, score in enumerate(scores):
                        j = i + 1 + start + k
                        all_seq1_ids.append(seq_ids[i])
                        all_seq2_ids.append(seq_ids[j])
                        all_scores.append(float(score))

                    pbar.update(batch_count)

    # Write to CSV
    df = pd.DataFrame({
        'seq1_id': all_seq1_ids,
        'seq2_id': all_seq2_ids,
        'tm_score': all_scores
    })
    df.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")


def main():
    is_scope40 = len(sys.argv) > 1 and sys.argv[1] == "scope40"

    if is_scope40:
        fasta = "data/fasta/scope40-1000.fa"
        output = "results/scope40_tmvec_student_similarities.csv"
        max_seq = 1000
    else:
        fasta = "data/fasta/cath-domain-seqs-S100-1k.fa"
        output = "results/tmvec_student_similarities.csv"
        max_seq = 1000

    checkpoint = "binaries/tmvec_student.pt"
    max_length = 600
    embed_batch = 128
    pred_batch = 4096
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"FASTA: {fasta}")
    print(f"Output: {output}")

    seq_ids, sequences = load_fasta(fasta, max_seq)
    model = load_model(checkpoint, device)
    embeddings = compute_embeddings(model, sequences, max_length, embed_batch, device)
    predict_scores(model, embeddings, seq_ids, Path(output), pred_batch, device)


if __name__ == "__main__":
    main()
