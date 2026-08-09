"""TM-Vec 2: Transformer encoder for TM-score prediction (inference only)."""

from typing import Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn


class TMVecConfig:
    """Configuration for TM-Vec 2 model."""

    def __init__(
        self,
        d_model: int = 408,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.2,
        activation: str = 'gelu',
        out_dim: int = 512,
        projection_hidden_dim: Optional[int] = None,
        **kwargs,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.out_dim = out_dim
        self.projection_hidden_dim = projection_hidden_dim or d_model


class TMScorePredictor(pl.LightningModule):
    """
    TM-Vec 2: Predicts TM-scores from protein embedding pairs.

    Architecture:
        Embeddings -> Transformer -> Pool -> MLP -> Cosine Similarity -> TM-score
    """

    def __init__(self, config: Optional[TMVecConfig] = None, **kwargs):
        super().__init__()

        self.config = config if config else TMVecConfig(**kwargs)
        self.save_hyperparameters(vars(self.config))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation=self.config.activation,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)

        # Projection head
        self.dropout = nn.Dropout(self.config.dropout)
        self.projection = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.projection_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.projection_hidden_dim, self.config.out_dim)
        )

        # Similarity
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        self._init_weights()

    def _init_weights(self):
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_sequence(self, embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Encode sequence: Transformer -> Mean Pool -> Project.

        Args:
            embeddings: [B, L, D]
            padding_mask: [B, L] True=padding

        Returns:
            [B, out_dim]
        """
        hidden = self.encoder(embeddings, src_key_padding_mask=padding_mask)

        # Mean pooling (ignore padding)
        lengths = (~padding_mask).sum(dim=1, keepdim=True).float().clamp(min=1e-9)
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        pooled = (hidden * mask_expanded).sum(dim=1) / lengths

        pooled = self.dropout(pooled)
        return self.projection(pooled)

    def forward(
        self,
        seq1_embeddings: torch.Tensor,
        seq2_embeddings: torch.Tensor,
        seq1_padding_mask: torch.Tensor,
        seq2_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (emb1, emb2, cosine_similarity)."""
        emb1 = self.encode_sequence(seq1_embeddings, seq1_padding_mask)
        emb2 = self.encode_sequence(seq2_embeddings, seq2_padding_mask)
        cos_sim = self.cos_sim(emb1, emb2)
        return emb1, emb2, cos_sim
