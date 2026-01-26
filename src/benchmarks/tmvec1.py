#!/usr/bin/env python
"""TMvec-1: TM-score predictions for CATH and SCOPe."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..model.tmvec_1_model import TransformerEncoderModule, TransformerEncoderModuleConfig


def generate_embeddings(sequences, batch_size=32, max_length=512, device='cuda',
                        checkpoint_path="binaries/tm_vec_cath_model.ckpt"):
    """
    Generate TMvec1 embeddings for protein sequences.

    Uses streaming processing to avoid OOM: each batch goes through ProtT5 -> TMvec
    immediately rather than accumulating all ProtT5 embeddings in RAM first.
    """
    from transformers import T5Tokenizer, T5EncoderModel

    # Load both models upfront
    print("Loading ProtT5 model...")
    prot_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    prot_model.to(device)
    prot_model.eval()

    print("Loading TMvec model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = TransformerEncoderModuleConfig(d_model=1024)
    tmvec_model = TransformerEncoderModule(config)
    tmvec_model.load_state_dict(checkpoint['state_dict'])
    tmvec_model.to(device)
    tmvec_model.eval()

    # Stream processing: ProtT5 -> TMvec for each batch immediately
    # This avoids storing all ProtT5 embeddings in RAM
    print("Generating embeddings (streaming ProtT5 -> TMvec)...")
    all_embeddings = []
    sequences_spaced = [" ".join(list(seq)) for seq in sequences]

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences_spaced), batch_size)):
            batch_seqs = sequences_spaced[i:i + batch_size]

            # ProtT5 encoding
            encoded = tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            outputs = prot_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            prot_emb = outputs.last_hidden_state

            # TMvec encoding - process immediately, don't store ProtT5 output
            batch_size_curr, seq_len = prot_emb.shape[:2]
            padding_mask = torch.zeros(batch_size_curr, seq_len, dtype=torch.bool, device=device)
            tmvec_emb = tmvec_model(prot_emb, src_mask=None, src_key_padding_mask=padding_mask)

            # Store only the final embedding (batch_size x 512)
            all_embeddings.append(tmvec_emb.cpu().numpy())

            # Clear intermediate tensors
            del prot_emb, outputs, input_ids, attention_mask, encoded

    # Free model memory
    del prot_model, tmvec_model
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
    """Calculate pairwise TM-scores via cosine similarity."""
    print("Calculating pairwise scores...")
    embeddings_tensor = torch.from_numpy(embeddings)
    embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=1)
    tm_matrix = torch.mm(embeddings_norm, embeddings_norm.t()).numpy()
    print(f"Mean: {tm_matrix.mean():.4f}, Std: {tm_matrix.std():.4f}")
    return tm_matrix


def save_results(seq_ids, tm_matrix, output_path):
    """Save pairwise scores to CSV."""
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
        output = "results/scope40_tmvec1_similarities.csv"
        max_seq = 1000
    else:
        fasta = "data/cath-top1k.fa"
        output = "results/cath_tmvec1_similarities.csv"
        max_seq = 1000

    checkpoint = "binaries/tm_vec_cath_model.ckpt"
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"FASTA: {fasta}")
    print(f"Output: {output}")

    seq_ids, sequences = load_fasta(fasta, max_seq)
    tmvec_embeddings = generate_embeddings(
        sequences, batch_size, device=device, checkpoint_path=checkpoint
    )
    tm_matrix = calculate_scores(tmvec_embeddings)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    save_results(seq_ids, tm_matrix, output)


if __name__ == "__main__":
    main()
