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
        self.tied_bias = nn.Parameter(torch.zeros(n_input_features))

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

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda"):
        state = torch.load(path, map_location=device)

        print("\n=== SAE CHECKPOINT DEBUG INFO ===")
        print(f"Keys: {list(state.keys())}")
        print(f"encoder._weight shape : {state['encoder._weight'].shape}")
        print(f"decoder._weight shape : {state['decoder._weight'].shape}")
        if 'pre_encoder_bias._bias_reference' in state:
            print(f"pre_encoder_bias._bias_reference shape : {state['pre_encoder_bias._bias_reference'].shape}")
        if 'post_decoder_bias._bias_reference' in state:
            print(f"post_decoder_bias._bias_reference shape : {state['post_decoder_bias._bias_reference'].shape}")
        print(f"encoder._bias shape   : {state['encoder._bias'].shape}")
        print("=================================\n")

        enc_raw = state["encoder._weight"].squeeze(0)
        dec_raw = state["decoder._weight"].squeeze(0)
        bias_in = state["pre_encoder_bias._bias_reference"].squeeze(0)
        bias_out = state["encoder._bias"].squeeze(0)

        encoder_w = enc_raw.t().clone().contiguous()
        decoder_w = dec_raw.clone().contiguous()
        tied_b = bias_in.clone().contiguous()
        enc_b = bias_out.clone().contiguous()

        model = cls().to(device)

        ''''
        model.encoder_weight.data = state['encoder._weight'].squeeze(0).T  # [1,8192,1024] → [1024,8192]
        model.decoder_weight.data = state['decoder._weight'].squeeze(0)  # [1,1024,8192] → [1024,8192]
        model.tied_bias.data = state['pre_encoder_bias._bias_reference'].squeeze(0)  # [1,1024] → [1024]
        model.encoder_bias.data = state['encoder._bias'].squeeze(0) # [1,8192] → [8192]
        '''''

        model.encoder_weight = nn.Parameter(encoder_w)
        model.decoder_weight = nn.Parameter(decoder_w)
        model.tied_bias = nn.Parameter(tied_b)
        model.encoder_bias = nn.Parameter(enc_b)

        print("\nSHAPES AFTER copy_():")
        print(f"  encoder_weight : {model.encoder_weight.shape}")
        print(f"  decoder_weight : {model.decoder_weight.shape}")
        print(f"  tied_bias      : {model.tied_bias.shape}")
        print(f"  encoder_bias   : {model.encoder_bias.shape}")
        print("")

        return model.eval()