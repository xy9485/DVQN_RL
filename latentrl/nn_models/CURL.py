import torch
from torch import Tensor, nn

from nn_models.components import ResidualLayer, ResidualLinearLayer


class CURL(nn.Module):
    """
    CURL: Contrastive Unsupervised Representation Learning
    """

    def __init__(self, z_dim, critic, critic_target):
        super(CURL, self).__init__()

        self.encoder = critic.encoder

        self.encoder_target = critic_target.encoder

        n_redisual_layers = 2
        residual_blocks = [nn.ReLU()]
        # for _ in range(n_redisual_layers):
        residual_blocks.append(ResidualLinearLayer(dim_in=z_dim, dim_out=z_dim))
        residual_blocks.append(nn.ReLU())
        residual_blocks.append(ResidualLinearLayer(dim_in=z_dim, dim_out=z_dim))

        self.residual_module = nn.Sequential(*residual_blocks)
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.ln = nn.LayerNorm(z_dim)

    def encode_anchor(self, x):

        z_out = self.encoder(x)
        # z_out = self.residual_module(z_out)
        # z_out = self.ln(z_out)
        return z_out

    def encode_positive(self, x):
        with torch.no_grad():
            z_out = self.encoder_target(x)
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits
