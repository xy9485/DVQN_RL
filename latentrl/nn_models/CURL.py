import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

from nn_models.components import ResidualLayer, ResidualLinearLayer, MlpModel
from nn_models.encoder import Encoder


class CURL(nn.Module):
    """
    CURL: Contrastive Unsupervised Representation Learning
    """

    def __init__(
        self,
        encoder: Encoder,
        projection_hidden_dims: int | list | None = None,
    ):
        super(CURL, self).__init__()

        self.encoder = encoder
        self.projection_hidden_dims = projection_hidden_dims
        # z_dim = self.encoder.linear_out_dim
        self.feature_dim = self.encoder.linear_out_dim
        if projection_hidden_dims != [-1]:
            # self.g = MlpModel(
            #     self.encoder.linear_out_dim,
            #     hidden_dims=projection_hidden_dims[:-1],
            #     output_dim=projection_hidden_dims[-1],
            #     # hidden_dims=projection_hidden_dims,
            #     # output_dim=self.encoder.linear_out_dim,
            #     activation=nn.ReLU,
            #     norm=nn.BatchNorm1d,
            #     # norm=nn.LayerNorm,
            # )

            self.g = nn.Sequential(
                nn.Linear(self.encoder.linear_out_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                # nn.BatchNorm1d(256),
                # nn.ReLU(),
                # nn.Sigmoid(),
            )
            self.feature_dim = 128

        self.W = nn.Parameter(torch.rand(self.feature_dim, self.feature_dim))
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward(self, x, projection=True):
        z_out = self.encoder(x)
        if projection and hasattr(self, "g"):
            z_out = self.g(z_out)
        return z_out
        # return F.normalize(z_out, dim=-1)

    # def encode_anchor(self, x):

    #     z_out = self.encoder(x)
    #     # z_out = self.residual_module(z_out)
    #     # z_out = self.ln(z_out)
    #     return z_out

    # def encode_positive(self, x):
    #     with torch.no_grad():
    #         z_out = self.encoder_target(x)
    #     return z_out

    def compute_logits_bilinear(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        # if self.g is not None:
        #     z_a = self.g(z_a)
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def compute_logits(self, z_a, z_pos):
        # if self.g is not None:
        #     z_a = self.g(z_a)
        return torch.matmul(z_a, z_pos.T)

    def curl_loss(self, anc: Tensor, pos: Tensor, temperature: float = 0.07):
        # anc = torch.cat([anc, pos], dim=0)
        # pos = torch.cat([pos, anc], dim=0)
        # anc = F.normalize(anc, dim=1)
        # pos = F.normalize(pos, dim=1)
        # logits = torch.matmul(anc, pos.T)
        # logits = self.compute_logits(anc, pos)
        logits = self.compute_logits_bilinear(anc, pos)
        logits = logits / temperature
        labels = torch.arange(logits.shape[0]).long().to(logits.device)
        # loss = F.cross_entropy(logits, labels)

        return logits, labels


class CURL_ATC(nn.Module):
    """
    CURL: Contrastive Unsupervised Representation Learning
    """

    def __init__(
        self,
        encoder: Encoder,
        encoder_target: Encoder,
        anchor_projection: bool,
    ):
        super().__init__()

        self.encoder = encoder
        self.encoder_target = encoder_target
        self.anchor_projection = anchor_projection
        # z_dim = self.encoder.linear_out_dim
        self.feature_dim = self.encoder.linear_out_dim

        if anchor_projection:
            self.anchor_mlp = nn.Sequential(
                nn.Linear(self.encoder.linear_out_dim, 256),
                # nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.BatchNorm1d(256),
                # nn.ReLU(),
                # nn.Sigmoid(),
            )
            self.feature_dim = 256
        else:
            self.anchor_mlp = None

        self.W = nn.Parameter(torch.rand(self.feature_dim, self.feature_dim))

    def forward(self, anc, pos):
        z_anc = self.encoder(anc)
        with torch.no_grad():
            z_pos = self.encoder_target(pos)
        if self.anchor_mlp is not None:
            z_anc = self.anchor_mlp(z_anc) + z_anc
        # z_anc = F.normalize(z_anc, dim=1)
        # z_pos = F.normalize(z_pos, dim=1)
        logits = self.compute_logits_bilinear(z_anc, z_pos)
        return logits
        # return F.normalize(z_out, dim=-1)

    # def encode_anchor(self, x):

    #     z_out = self.encoder(x)
    #     # z_out = self.residual_module(z_out)
    #     # z_out = self.ln(z_out)
    #     return z_out

    # def encode_positive(self, x):
    #     with torch.no_grad():
    #         z_out = self.encoder_target(x)
    #     return z_out

    def compute_logits_bilinear(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        # if self.g is not None:
        #     z_a = self.g(z_a)
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def compute_logits(self, z_a, z_pos):
        # if self.g is not None:
        #     z_a = self.g(z_a)
        return torch.matmul(z_a, z_pos.T)

def simclr_loss(anc: Tensor, pos: Tensor, temperature: float = 1) -> Tensor:
    batch_size = anc.shape[0]
    out = torch.cat([anc, pos], dim=0)
    out = F.normalize(out, dim=1)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (
        torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)
    ).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(anc * pos, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss


def simclr_debiased_loss(
    out_1: Tensor,
    out_2: Tensor,
    temperature: float,
    debiased: bool,
    tau_plus: float,
) -> Tensor:
    def get_negative_mask(batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    batch_size = out_1.shape[0]
    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    # estimator g()
    if debiased:
        N = batch_size * 2 - 2
        Ng = (-tau_plus * N * pos + neg.sum(dim=-1)) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
    else:
        Ng = neg.sum(dim=-1)

    # contrastive loss
    loss = (-torch.log(pos / (pos + Ng))).mean()
    return loss


def simclr_loss2(anc: Tensor, pos: Tensor, temperature: float = 0.07):
    device = anc.device
    batch_size = anc.shape[0]
    out = torch.cat([anc, pos], dim=0)
    # out = F.normalize(out, dim=1)  # require to compute cosine similarity
    similarity_matrix = torch.matmul(out, out.T) / temperature

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    # loss = F.cross_entropy(logits, labels)

    return logits, labels
