#!/usr/bin/env python
"""TM-Vec Student: TM-score predictions for CATH and SCOPe using cosine similarity."""

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.accuracy_benchmarks import save_pairwise_scores
from src.models.tmvec2_student_model import StudentModel, encode_sequence
from src.util.fasta import load_fasta

DATASETS = {
    "cath": {
        "fasta": "data/fasta/cath-s100-unique-10k.fa",
        "output": "results/cath_tmvec2_student_similarities.csv",
    },
    "scope40": {
        "fasta": "data/fasta/scop40.fasta",
        "output": "results/scope40_tmvec2_student_similarities.csv",
    },
}


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
    parser = argparse.ArgumentParser(description="TM-Vec Student TM-score prediction")
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="cath",
                        help="Dataset to use (cath or scope40)")
    parser.add_argument("--fasta", default=None, help="FASTA file path (overrides dataset default)")
    parser.add_argument("--output", default=None, help="Output CSV path (overrides dataset default)")
    parser.add_argument("--checkpoint", default="binaries/tmvec2_student.pt",
                        help="Path to student model checkpoint")
    parser.add_argument("--max-length", type=int, default=600, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for encoding")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu, auto-detects if not specified)")
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    fasta = args.fasta or config["fasta"]
    output = args.output or config["output"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Device: {device}")
    print(f"FASTA: {fasta}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output}")

    seq_ids, sequences = load_fasta(fasta, None)
    model = load_model(args.checkpoint, device)
    embeddings = compute_embeddings(model, sequences, args.max_length, args.batch_size, device)
    tm_matrix = calculate_scores(embeddings)
    save_pairwise_scores(seq_ids, tm_matrix, output)


if __name__ == "__main__":
    main()
