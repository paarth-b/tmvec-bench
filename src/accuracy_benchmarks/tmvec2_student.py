#!/usr/bin/env python
"""TM-Vec Student: TM-score predictions for CATH and SCOPe using cosine similarity."""

import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.accuracy_benchmarks import save_pairwise_scores
from src.models.tmvec2_student_model import StudentModel, encode_sequence
from src.util.fasta import load_fasta


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


def calculate_scores(embeddings):
    """Pairwise cosine similarity (L2-normalize, then normalized dot product)."""
    print("Calculating pairwise cosine similarities...")
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    tm_matrix = torch.mm(embeddings_norm, embeddings_norm.t()).cpu().numpy()
    print(f"Cosine similarity stats - Mean: {tm_matrix.mean():.4f}, Std: {tm_matrix.std():.4f}")
    return tm_matrix


def main():
    is_scope40 = len(sys.argv) > 1 and sys.argv[1] == "scope40"

    if is_scope40:
        fasta = "data/fasta/scop40.fasta"
        output = "/work/nvme/beut/paarthbatra/data/results/scope40_tmvec2_student_similarities.csv"
    else:
        fasta = "data/fasta/cath-s100-unique-10k.fa"
        output = "/work/nvme/beut/paarthbatra/data/results/cath_tmvec2_student_similarities.csv"

    checkpoint = "binaries/tmvec2_student.pt"
    max_length = 600
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"FASTA: {fasta}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output}")

    seq_ids, sequences = load_fasta(fasta, None)
    model = load_model(checkpoint, device)
    embeddings = compute_embeddings(model, sequences, max_length, batch_size, device)
    tm_matrix = calculate_scores(embeddings)
    save_pairwise_scores(seq_ids, tm_matrix, output)


if __name__ == "__main__":
    main()
