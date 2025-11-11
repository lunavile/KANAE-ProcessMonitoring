import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_layers, latent_size, alpha=1e-3, l1_lambda=0.0, l2_lambda=0.0):
        super(OrthogonalAutoencoder, self).__init__()
        self.alpha = alpha
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # --- Encoder ---
        encoder_layers = []
        for i, h in enumerate(hidden_layers):
            in_dim = input_size if i == 0 else hidden_layers[i - 1]
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        self.z_layer = nn.Linear(hidden_layers[-1], latent_size)

        # --- Decoder ---
        decoder_layers = []
        for i in reversed(range(len(hidden_layers))):
            in_dim = latent_size if i == len(hidden_layers) - 1 else hidden_layers[i + 1]
            decoder_layers.append(nn.Linear(in_dim, hidden_layers[i]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden_layers[0], input_size))
        self.decoder = nn.Sequential(*decoder_layers)

        # Efficiently store only encoder weights (exclude biases and decoder)
        self.weight_params = [
            p for n, p in self.named_parameters()
            if 'weight' in n and ('encoder' in n or 'z_layer' in n)
        ]

    def encode(self, x):
        return self.z_layer(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def orthogonality_penalty(self, z):
        """
        Orthogonality loss using Frobenius norm: ||ZᵀZ - I||²
        """
        gram = torch.matmul(z.T, z) / z.size(0)
        identity = torch.eye(z.size(1), device=z.device)
        return torch.norm(gram - identity, p='fro') ** 2

    def elastic_net_regularization(self):
        """
        Efficient Elastic Net regularization using encoder and latent layer weights.
        Combines L1 (Lasso) and L2 (Ridge) regularization.
        """
        weights = torch.cat([p.view(-1) for p in self.weight_params])
        l1_loss = self.l1_lambda * torch.norm(weights, p=1)  # L1: ||θ||₁
        l2_loss = self.l2_lambda * torch.norm(weights, p=2)**2  # L2: ||θ||²
        return l1_loss + l2_loss

    def loss_function(self, x, x_hat, z):
        """
        Compute total loss using:
        - Reconstruction Loss (MSE)
        - Orthogonality Penalty
        - Elastic Net Regularization
        """
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        reg_loss = self.alpha * self.orthogonality_penalty(z)
        enet_loss = self.elastic_net_regularization()

        total_loss = recon_loss + reg_loss + enet_loss
        return total_loss, recon_loss, reg_loss, enet_loss
    
    def regularization_loss(self, z=None):
        """
       Returns total regularization loss:
       - Orthogonality (needs z)
       - Elastic net (encoder weights)
       """
        if z is None:
         return self.elastic_net_regularization()  # fallback if z not available
        
        reg_orth = self.alpha * self.orthogonality_penalty(z)
        reg_enet = self.elastic_net_regularization()
        return reg_orth + reg_enet    

def get_orthogonalAE_model(input_size, **kwargs):
    return OrthogonalAutoencoder(
        input_size=input_size,
        hidden_layers=kwargs["hidden_layers"],
        latent_size=kwargs["latent_size"],
        alpha=kwargs.get("alpha", 1e-3),
        l1_lambda=kwargs.get("l1_lambda", 0.0),
        l2_lambda=kwargs.get("l2_lambda", 0.0)
    )