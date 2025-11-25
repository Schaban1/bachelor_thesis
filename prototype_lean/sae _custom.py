import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SparseAutoencoder(nn.Module):
    def __init__(self, n_input_features: int = 1024, n_learned_features: int = 8192):
        super().__init__()
        # Encoder
        self.encoder_weight = nn.Parameter(torch.empty(n_input_features, n_learned_features))
        self.encoder_bias   = nn.Parameter(torch.zeros(n_learned_features))
        # Decoder
        self.decoder_weight = nn.Parameter(torch.empty(n_input_features, n_learned_features))
        # Tied bias
        self.tied_bias = nn.Parameter(torch.zeros(1, n_input_features))

        nn.init.xavier_uniform_(self.encoder_weight)
        nn.init.xavier_uniform_(self.decoder_weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.tied_bias
        hidden = F.relu(F.linear(x,self.encoder_weight, self.encoder_bias))
        decoder_normed = self.decoder_weight / self.decoder_weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        recon = F.linear(hidden, decoder_normed.t()) + self.tied_bias
        return hidden, recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.tied_bias
        return F.relu(F.linear(x, self.encoder_weight, self.encoder_bias))