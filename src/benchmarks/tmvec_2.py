#!/usr/bin/env python
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from ..model.tmvec_2_model import TMScorePredictor, TMVecConfig
from lobster.model import LobsterPMLM


def generate_embeddings(sequences, batch_size=32, max_length=512, device='cuda'):
    print("Generating Lobster-24M embeddings...")
    model = LobsterPMLM.from_pretrained("asalam91/lobster_24M")
    model.to(device)
    model.eval()

    all_embeddings = []
    all_lengths = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i + batch_size]

            batch_seqs = [seq[:max_length] for seq in batch_seqs]
            lengths = [len(seq) for seq in batch_seqs]
            all_lengths.append(lengths)

            embeddings = model.get_embeddings(batch_seqs, mean_embedding=False)
            all_embeddings.append(embeddings.cpu())

    return all_embeddings, all_lengths


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


def transform_embeddings(base_embeddings, lengths_per_batch, device):
    print("Loading TMvec-2 model from HuggingFace...")
    checkpoint_path = hf_hub_download(
        repo_id="scikit-bio/tmvec-2",
        filename="tmvec-2.ckpt"
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint['state_dict']

    config = TMVecConfig(d_model=408)
    model = TMScorePredictor(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Transforming embeddings...")
    all_embeddings = []

    with torch.no_grad():
        for batch, lengths in tqdm(zip(base_embeddings, lengths_per_batch), desc="TMvec-2 encoding", total=len(base_embeddings)):
            batch = batch.to(device)
            batch_size, seq_len = batch.shape[:2]

            # Create proper padding mask based on actual sequence lengths
            padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            for idx, length in enumerate(lengths):
                padding_mask[idx, :length] = False

            emb = model.encode_sequence(batch, padding_mask)
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


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
    is_scope40 = len(sys.argv) > 1 and sys.argv[1] == "scope40"

    if is_scope40:
        fasta = "data/fasta/scope40-1000.fa"
        output = "results/scope40_tmvec2_similarities.csv"
        max_seq = 1000
    else:
        fasta = "data/fasta/cath-domain-seqs-S100-1k.fa"
        output = "results/tmvec2_similarities.csv"
        max_seq = 1000

    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"FASTA: {fasta}")
    print(f"Output: {output}")

    seq_ids, sequences = load_fasta(fasta, max_seq)
    base_embeddings, lengths_per_batch = generate_embeddings(sequences, batch_size, device=device)
    tmvec_embeddings = transform_embeddings(base_embeddings, lengths_per_batch, device)
    tm_matrix = calculate_scores(tmvec_embeddings)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    save_results(seq_ids, tm_matrix, output)


if __name__ == "__main__":
    main()
