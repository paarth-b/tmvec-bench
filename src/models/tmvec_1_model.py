"""TM-Vec 1: Transformer encoder for TM-score prediction (inference only)."""

import inspect
from typing import Union

import lightning as L
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from transformers import PretrainedConfig


class TransformerEncoderModuleConfig(PretrainedConfig):
    def __init__(self,
                 d_model=1024,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=2048,
                 out_dim=512,
                 dropout=0.2,
                 activation='gelu',
                 **kwargs):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.out_dim = out_dim
        self.dropout = dropout
        self.activation = activation
        super().__init__(**kwargs)


class TransformerEncoderModule(L.LightningModule, PyTorchModelHubMixin):
    """TransformerEncoder with global mean pooling and MLP projection."""

    def __init__(self,
                 config: Union[TransformerEncoderModuleConfig, dict],
                 random_seed: int = 42):
        super().__init__()

        torch.manual_seed(random_seed)

        if isinstance(config, TransformerEncoderModuleConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = TransformerEncoderModuleConfig(**config)
        else:
            raise ValueError("Invalid config type")

        encoder_args = {
            k: getattr(self.config, k)
            for k in inspect.signature(nn.TransformerEncoderLayer).parameters
            if hasattr(self.config, k)
        }

        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, **encoder_args)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)

        self.dropout = nn.Dropout(self.config.dropout)
        self.mlp = nn.Linear(self.config.d_model, self.config.out_dim)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor,
                src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        lens = torch.logical_not(src_key_padding_mask).sum(dim=1)
        x = x.sum(dim=1) / lens.unsqueeze(1)
        x = self.dropout(x)
        x = self.mlp(x)
        return x
