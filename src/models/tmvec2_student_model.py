"""TM-Vec 2 Student Model: BiLSTM encoder for TM-score prediction (inference only)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Amino acid vocabulary
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYXUBZ'
VOCAB_SIZE = len(AMINO_ACIDS) + 2
PAD_TOKEN = 0
UNK_TOKEN = 1
AA_TO_IDX = {aa: i + 2 for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX['<PAD>'] = PAD_TOKEN
AA_TO_IDX['<UNK>'] = UNK_TOKEN


def encode_sequence(sequence, max_length=None):
    """Encode amino acid sequence to tensor."""
    encoded = [AA_TO_IDX.get(aa, UNK_TOKEN) for aa in sequence]

    if max_length:
        if len(encoded) > max_length:
            encoded = encoded[:max_length]
        else:
            encoded.extend([PAD_TOKEN] * (max_length - len(encoded)))

    return torch.tensor(encoded, dtype=torch.long)


class ProteinSequenceEncoder(nn.Module):
    """BiLSTM encoder with attention pooling."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512, output_dim=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim // 2, 2,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        mask = (x != PAD_TOKEN).float()
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1)

        pooled = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        output = self.output_proj(pooled)
        output = self.norm(output)
        return output


class StudentModel(nn.Module):
    """Prediction: CosineSimilarity(Enc(A), Enc(B))"""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=512,
                 seq_embed_dim=512, dropout=0.1):
        super().__init__()
        self.seq_encoder = ProteinSequenceEncoder(
            vocab_size, embed_dim, hidden_dim, seq_embed_dim, dropout
        )

    def forward(self, seq_a, seq_b=None):
        repr_a = self.seq_encoder(seq_a)
        if seq_b is not None:
            repr_b = self.seq_encoder(seq_b)
            cosine_sim = F.cosine_similarity(repr_a, repr_b, dim=1)
            return repr_a, repr_b, cosine_sim
        return repr_a
