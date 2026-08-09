#!/usr/bin/env python
import argparse
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from src.accuracy_benchmarks import cosine_similarity_matrix, save_pairwise_scores
from src.models.tmvec_2_model import TMScorePredictor, TMVecConfig
from src.util.fasta import load_fasta
from lobster.model._mlm import LobsterPMLM


def generate_embeddings(sequences, batch_size=32, max_length=512, device='cuda'):
    """Generate LOBSTER embeddings for protein sequences using tokenizer approach."""
    print("Generating Lobster-24M embeddings...")
    model = LobsterPMLM("asalam91/lobster_24M")
    tokenizer = model.tokenizer
    model.to(device)
    model.eval()

    all_embeddings = []
    all_attention_masks = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i + batch_size]

            encoded = tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            embeddings = outputs.hidden_states[-1]

            all_embeddings.append(embeddings.cpu())
            all_attention_masks.append(attention_mask.cpu())

    print(f"Generated LOBSTER embeddings: {all_embeddings[0].shape}")
    return all_embeddings, all_attention_masks


def transform_embeddings(base_embeddings, attention_masks, device):
    """Transform base embeddings into structure-aware embeddings using TMvec-2 model."""
    print("Loading TMvec-2 model from HuggingFace...")
    checkpoint_path = hf_hub_download(
        repo_id="scikit-bio/tmvec-2",
        filename="tmvec-2.ckpt"
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint['state_dict']

    config = TMVecConfig(
        d_model=408,
        num_layers=4,
        projection_hidden_dim=1024
    )
    model = TMScorePredictor(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Transforming embeddings...")
    all_embeddings = []

    with torch.no_grad():
        for batch_emb, attn_mask in tqdm(zip(base_embeddings, attention_masks), desc="TMvec-2 encoding", total=len(base_embeddings)):
            batch_emb = batch_emb.to(device)
            attn_mask = attn_mask.to(device)

            # Convert attention_mask to padding_mask (attention_mask: 1=real, 0=padding)
            # padding_mask: True=padding, False=real
            padding_mask = (attn_mask == 0)

            emb = model.encode_sequence(batch_emb, padding_mask)
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(description="TMvec-2 TM-score prediction")
    parser.add_argument("--dataset", choices=['cath', 'scope40'], default='cath',
                        help="Dataset to use (cath or scope40)")
    parser.add_argument("--fasta", default=None, help="FASTA file path (overrides dataset default)")
    parser.add_argument("--output", default=None, help="Output CSV path (overrides dataset default)")
    parser.add_argument("--max-sequences", type=int, default=None, help="Maximum sequences to process")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding generation")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu, auto-detects if not specified)")

    args = parser.parse_args()

    # Set dataset-specific defaults
    if args.dataset == 'scope40':
        fasta = args.fasta or "data/fasta/scop40.fasta"
        output = args.output or "results/scope40_tmvec2_similarities.csv"
    else:
        fasta = args.fasta or "data/fasta/cath-s100-unique-10k.fa"
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
    base_embeddings, attention_masks = generate_embeddings(sequences, args.batch_size, device=device)
    tmvec_embeddings = transform_embeddings(base_embeddings, attention_masks, device)
    tm_matrix = cosine_similarity_matrix(tmvec_embeddings)
    save_pairwise_scores(seq_ids, tm_matrix, output)

    print("=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
