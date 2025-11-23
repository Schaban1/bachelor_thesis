from sparse_autoencoder.sparse_autoencoder.autoencoder.model import SparseAutoencoder
import torch

print('SparseAutoencoder geladen – ohne wandb-Fehler?')
sae = SparseAutoencoder(1024, 8192)
x = torch.randn(1, 1024)
acts, recon = sae.forward(x)
print('forward pass successful')