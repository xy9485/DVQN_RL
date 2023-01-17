import torch

# from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

from typing import Any, Dict, List, Optional, Tuple, Type, Union, TypeVar
from torch import Tensor


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        # try detach
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = (
            torch.sum(flat_latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        )  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return (
            quantized_latents.permute(0, 3, 1, 2).contiguous(),
            vq_loss,
        )  # [B x D x H x W]


class VectorQuantizerLinear(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        # try detach
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        """
        latent # [B x D]
        """

        # Compute L2 distance between latents and embedding weights
        dist = (
            torch.sum(latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(latents, self.embedding.weight.t())
        )  # [B x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [B, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [B x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [B, D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        avg_probs = torch.mean(encoding_one_hot, dim=0)

        # Compute vq entropy

        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        entrophy_vq = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        # vq_probs = dist / torch.sum(dist, dim=1, keepdim=True)
        # vq_probs = -vq_probs * torch.log(vq_probs + 1e-10)
        # entrophy_vq = -torch.sum(vq_probs, dim=1).mean()

        # entrophy_vq = torch.std(dist, dim=1).mean()
        cluster_metric = dist[encoding_one_hot.bool()].mean().item()
        output_dict = {
            "hard_encoding_inds": encoding_inds,
            "hard_quantized_latents": quantized_latents,
            "cluster_metric": cluster_metric,
        }

        return (
            quantized_latents,  # [B x D]
            vq_loss,
            entrophy_vq,
            output_dict,
        )


class VQSoftAttention(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, beta: float = 0.25, softmax_alpha: float = 10
    ):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.softmax_alpha = softmax_alpha

        self.embedding = nn.Embedding(self.K, self.D)
        # try detach
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        """
        latent # [B x D]
        """
        latents = F.normalize(latents, dim=1)  # [B x D]
        codebook = F.normalize(self.embedding.weight.data, dim=1)  # [KxD]
        attention = torch.matmul(latents, codebook.t())  # this is also cosine similarity [BxK]
        soft_assign = F.softmax(self.softmax_alpha * attention, dim=1)  # [BxK]

        # [Quantize the latents]
        quantized_latents = torch.matmul(soft_assign, self.embedding.weight)  # [B, D]

        # [Compute the VQ Losses[]
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        # quantized_latents = latents + (quantized_latents - latents).detach()
        avg_probs = torch.mean(soft_assign, dim=0)

        # [Compute vq entropy]
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        entrophy_vq = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        # vq_probs = dist / torch.sum(dist, dim=1, keepdim=True)
        # vq_probs = -vq_probs * torch.log(vq_probs + 1e-10)
        # entrophy_vq = -torch.sum(vq_probs, dim=1).mean()

        # entrophy_vq = torch.std(dist, dim=1).mean()

        # dist_std = torch.std(dist, dim=1)
        # entrophy_vq = entrophy_vq + dist_std.mean()

        # [Get Hard Quantization]
        device = latents.device
        argm_attention = torch.argmax(attention, dim=1, keepdim=True)  # [B, 1]
        encoding_one_hot = torch.zeros(argm_attention.size(0), self.K, device=device).scatter_(
            1, argm_attention, 1
        )  # [B x K]
        hard_quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [B, D]
        cluster_metric = attention[encoding_one_hot.bool()].mean().item()
        output_dict = {
            "hard_encoding_inds": argm_attention,
            "hard_quantized_latents": hard_quantized_latents,
            "cluster_assignment": soft_assign,
            "cluster_metric": cluster_metric,
        }

        return (
            quantized_latents,  # [B x D]
            vq_loss,
            entrophy_vq,
            output_dict,
        )


class VectorQuantizerLinearSoft(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, beta: float = 0.25, softmin_beta: float = 10
    ):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.softmin_beta = softmin_beta

        self.embedding = nn.Embedding(self.K, self.D)
        # self.embedding = nn.Embedding(self.K, self.D).requires_grad_(False)
        # try detach
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)
        # self.embedding.weight.data.uniform_(0, 1)
        # self.embedding.weight.data.orthogonal_()
        # nn.init.orthogonal_(self.embedding.weight.data)

    def forward(self, latents: Tensor) -> Tensor:
        """
        latent # [B x D]
        """

        # Compute L2 distance between latents and embedding weights
        dist = (
            torch.sum(latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(latents, self.embedding.weight.t())
        )  # [B x K]

        # [Get the encoding that has the min distance]
        # encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [B, 1]
        encoding_inds = F.softmin(self.softmin_beta * dist, dim=1)

        # [Quantize the latents]
        quantized_latents = torch.matmul(encoding_inds, self.embedding.weight)  # [B, D]

        # [Compute the VQ Losses[]
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        # quantized_latents = latents + (quantized_latents - latents).detach()
        avg_probs = torch.mean(encoding_inds, dim=0)

        # [Compute vq entropy]
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        entrophy_vq = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        # vq_probs = dist / torch.sum(dist, dim=1, keepdim=True)
        # vq_probs = -vq_probs * torch.log(vq_probs + 1e-10)
        # entrophy_vq = -torch.sum(vq_probs, dim=1).mean()

        # entrophy_vq = torch.std(dist, dim=1).mean()

        # dist_std = torch.std(dist, dim=1)
        # entrophy_vq = entrophy_vq + dist_std.mean()

        # [Get Hard Quantization]
        device = latents.device
        hard_encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [B, 1]
        encoding_one_hot = torch.zeros(hard_encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, hard_encoding_inds, 1)  # [B x K]
        hard_quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [B, D]
        cluster_metric = dist[encoding_one_hot.bool()].mean().item()
        output_dict = {
            "hard_encoding_inds": hard_encoding_inds,
            "hard_quantized_latents": hard_quantized_latents,
            "cluster_assignment": encoding_inds,
            "cluster_metric": cluster_metric,
        }

        return (
            quantized_latents,  # [B x D]
            vq_loss,
            entrophy_vq,
            output_dict,
        )


class VectorQuantizerLinearDiffable(nn.Module):
    """
    make argmin differantiable
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, beta: float = 0.25, softmin_beta: float = 10
    ):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.softmin_beta = softmin_beta

        self.embedding = nn.Embedding(self.K, self.D)
        # try detach
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        """
        latent # [B x D]
        """

        # Compute L2 distance between latents and embedding weights
        dist = (
            torch.sum(latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(latents, self.embedding.weight.t())
        )  # [B x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [B, 1]
        encoding_inds_soft = F.softmin(self.softmin_beta * dist, dim=1)

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [B x K]

        encoding_one_hot = (encoding_one_hot - encoding_inds_soft).detach() + encoding_inds_soft

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [B, D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss
        avg_probs = torch.mean(encoding_one_hot, dim=0)

        # Compute vq entropy
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        entrophy_vq = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        # vq_probs = dist / torch.sum(dist, dim=1, keepdim=True)
        # vq_probs = -vq_probs * torch.log(vq_probs + 1e-10)
        # entrophy_vq = -torch.sum(vq_probs, dim=1).mean()

        # entrophy_vq = torch.std(dist, dim=1).mean()
        cluster_metric = dist[encoding_one_hot.bool()].mean().item()
        output_dict = {
            "hard_encoding_inds": encoding_inds,
            "hard_quantized_latents": quantized_latents,
            "cluster_assignment": encoding_inds_soft,
            "cluster_metric": cluster_metric,
        }
        return (
            quantized_latents,  # [B x D]
            vq_loss,
            entrophy_vq,
            output_dict,
        )


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity, encodings
