import torch
from torch import Tensor, nn

from nn_models.components import ResidualLayer, ResidualLinearLayer, MlpModel
from nn_models.encoder import Encoder


class CURL(nn.Module):
    """
    CURL: Contrastive Unsupervised Representation Learning
    """

    def __init__(
        self,
        encoder: Encoder,
        encoder_target: Encoder,
        anchor_mlp_hidden_dims: int | list | None = None,
    ):
        super(CURL, self).__init__()

        self.encoder = encoder
        self.encoder_target = encoder_target
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        z_dim = self.encoder.linear_out_dim
        if anchor_mlp_hidden_dims is not None:
            self.anchor_MLP = MlpModel(z_dim, anchor_mlp_hidden_dims, z_dim)
        else:
            self.anchor_MLP = None

        # n_redisual_layers = 2
        # residual_blocks = [nn.ReLU()]
        # # for _ in range(n_redisual_layers):
        # residual_blocks.append(ResidualLinearLayer(dim_in=z_dim, dim_out=z_dim))
        # residual_blocks.append(nn.ReLU())
        # residual_blocks.append(ResidualLinearLayer(dim_in=z_dim, dim_out=z_dim))
        # self.residual_module = nn.Sequential(*residual_blocks)

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        # self.ln = nn.LayerNorm(z_dim)

    def encode_anchor(self, x):

        z_out = self.encoder(x)
        # z_out = self.residual_module(z_out)
        # z_out = self.ln(z_out)
        return z_out

    def encode_positive(self, x):
        with torch.no_grad():
            z_out = self.encoder_target(x)
        return z_out

    def compute_logits_bilinear(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        if self.anchor_MLP is not None:
            z_a = self.anchor_MLP(z_a)
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def compute_logits(self, z_a, z_pos):
        if self.anchor_MLP is not None:
            z_a = self.anchor_MLP(z_a)
        return torch.matmul(z_a, z_pos.T)
