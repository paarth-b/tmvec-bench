#!/usr/bin/env python
"""TMvec-1: TM-score predictions for CATH and SCOPe."""

import sys
import numpy as np
import torch
from tqdm import tqdm

from src.accuracy_benchmarks import cosine_similarity_matrix, save_pairwise_scores
from src.models.tmvec_1_model import TransformerEncoderModule, TransformerEncoderModuleConfig
from src.util.fasta import load_fasta


def generate_embeddings(sequences, batch_size=32, max_length=512, device='cuda'):
    """Generate ProtT5 embeddings for protein sequences."""
    from transformers import T5Tokenizer, T5EncoderModel

    print("Generating ProtT5 embeddings...")
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model.to(device)
    model.eval()

    all_embeddings = []
    sequences_spaced = [" ".join(list(seq)) for seq in sequences]

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences_spaced), batch_size)):
            batch_seqs = sequences_spaced[i:i + batch_size]

            encoded = tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            embeddings = outputs.last_hidden_state
            all_embeddings.append(embeddings.cpu())

    return all_embeddings


def transform_embeddings(base_embeddings, checkpoint_path, device):
    """Transform embeddings with TMvec model."""
    print("Loading TMvec model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = TransformerEncoderModuleConfig(d_model=1024)
    model = TransformerEncoderModule(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    print("Transforming embeddings...")
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(base_embeddings, desc="TMvec encoding"):
            batch = batch.to(device)
            batch_size, seq_len = batch.shape[:2]
            padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            emb = model(batch, src_mask=None, src_key_padding_mask=padding_mask)
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main():
    is_scope40 = len(sys.argv) > 1 and sys.argv[1] == "scope40"

    if is_scope40:
        fasta = "data/fasta/scop40.fasta"
        output = "/work/nvme/beut/paarthbatra/data/results/scope40_tmvec1_similarities.csv"
    else:
        fasta = "data/fasta/cath-s100-unique-10k.fa"
        output = "/work/nvme/beut/paarthbatra/data/results/cath_tmvec1_similarities.csv"

    checkpoint = "binaries/tm_vec_cath_model.ckpt"
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"FASTA: {fasta}")
    print(f"Output: {output}")

    seq_ids, sequences = load_fasta(fasta, None)
    base_embeddings = generate_embeddings(sequences, batch_size, device=device)
    tmvec_embeddings = transform_embeddings(base_embeddings, checkpoint, device)
    tm_matrix = cosine_similarity_matrix(tmvec_embeddings)
    save_pairwise_scores(seq_ids, tm_matrix, output)


if __name__ == "__main__":
    main()
