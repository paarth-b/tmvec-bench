"""Shared helpers for the accuracy benchmarks."""

from pathlib import Path

import pandas as pd


def cosine_similarity_matrix(embeddings):
    """Pairwise cosine-similarity matrix for a numpy array of embeddings."""
    import torch
    import torch.nn.functional as F

    print("Calculating pairwise scores...")
    normed = F.normalize(torch.from_numpy(embeddings), p=2, dim=1)
    tm_matrix = torch.mm(normed, normed.t()).numpy()
    print(f"Mean: {tm_matrix.mean():.4f}, Std: {tm_matrix.std():.4f}")
    return tm_matrix


def save_pairwise_scores(seq_ids, tm_matrix, output_path):
    """Write upper-triangle pairwise scores as seq1_id,seq2_id,tm_score CSV."""
    print(f"Saving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (seq_ids[i], seq_ids[j], float(tm_matrix[i, j]))
        for i in range(len(seq_ids))
        for j in range(i + 1, len(seq_ids))
    ]
    df = pd.DataFrame(rows, columns=["seq1_id", "seq2_id", "tm_score"])
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} scores")
