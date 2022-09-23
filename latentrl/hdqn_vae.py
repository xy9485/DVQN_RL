from cmath import e
import copy
import datetime
from json import decoder
import math
import os
from pickle import FALSE
import random
import re
from sys import modules
from pprint import pp
import PIL
from PIL import Image
import io

# from this import d
import time
from collections import namedtuple
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

# from unicodedata import decimal

import gym
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch import Tensor, nn
from torchsummary import summary
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.colors as colors
from sympy.solvers import solve
from sympy import Symbol

import wandb
from latentrl.policies.utils import ReplayMemory
from latentrl.utils.learning import EarlyStopping, ReduceLROnPlateau
from latentrl.utils.misc import (
    get_linear_fn,
    linear_schedule,
    polyak_sync,
    soft_sync_params,
    update_learning_rate,
    wandb_log_image,
)


def make_encoder(encoder_type, n_channel_input, n_channel_output, observation_space, hidden_dims):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        n_channel_input, n_channel_output, observation_space, hidden_dims
    )


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)
    elif isinstance(m, nn.LayerNorm):
        m.bias.data.zero_()
        m.weight.data.fill_(1.0)


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


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_embeddings,
        commitment_factor=0.25,
        decay=0.99,
        epsilon=1e-5,
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_factor = commitment_factor

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
        loss = self._commitment_factor * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        vq_entrophy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))

        # convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, vq_entrophy, encodings


class Batch_KMeans(nn.Module):
    def __init__(
        self,
        n_clusters,
        embedding_dim,
        decay=0.99,
        epsilon=1e-5,
        device="cuda",
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.register_buffer("centroids", torch.randn(n_clusters, embedding_dim))
        # self.centroids = torch.randn(self.n_clusters, self.embedding_dim)
        # self.centroids.requires_grad_(False)
        # self.centroids.retain_grad()
        # self.count = 100 * torch.zeros(self.n_clusters)
        self.register_buffer("_ema_cluster_size", torch.zeros(n_clusters))
        self.register_buffer("_ema_w", torch.randn(n_clusters, embedding_dim))
        # self._ema_w = torch.randn(n_clusters, embedding_dim)
        self._decay = decay
        self._epsilon = epsilon

    @torch.no_grad()
    def _compute_distances(self, X):
        # X = torch.norm(X, p=2, dim=1)
        # pass
        X = F.normalize(X, p=2, dim=1)
        distances = (
            torch.sum(X**2, dim=1, keepdim=True)
            + torch.sum(self.centroids**2, dim=1)
            - 2 * torch.matmul(X, self.centroids.t())
        )

        return distances

    @torch.no_grad()
    def init_cluster(self, X: Tensor):
        """Generate initial clusters using sklearn.Kmeans"""
        print("========== Start Initial Clustering ==========")
        self.model = KMeans(n_clusters=self.n_clusters, n_init=20)
        # X = preprocessing.normalize(X)
        self.init_cluster_indices = self.model.fit_predict(preprocessing.normalize(X.cpu().numpy()))
        self.centroids = torch.from_numpy(self.model.cluster_centers_).to(X.device)  # copy clusters
        print("========== End Initial Clustering ==========")

    @torch.no_grad()
    def assign_clusters(self, X):
        distances = self._compute_distances(X)
        return torch.argmin(distances, dim=1)

    @torch.no_grad()
    def assign_centroid(self, X, update_centroid=True):
        distances = self._compute_distances(X)
        cluster_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        cluster_onehots = torch.zeros(cluster_indices.shape[0], self.n_clusters, device=X.device)
        cluster_onehots.scatter_(1, cluster_indices, 1)
        quantized = torch.matmul(cluster_onehots, self.centroids)

        if update_centroid:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(cluster_onehots, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon) / (n + self.n_clusters * self._epsilon) * n
            )

            dw = torch.matmul(cluster_onehots.t(), X)
            self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw

            self.centroids = self._ema_w / self._ema_cluster_size.unsqueeze(1)

        return quantized, cluster_indices

    # def fit(self, inputs):
    #     inputs = inputs.permute(0, 2, 3, 1).contiguous()
    #     inputs = inputs.view(-1, self.embedding_dim)
    #     for i in range(self.num_embeddings):
    #         self.centroids[i] = torch.mean(
    #             inputs[torch.argmax(torch.abs(inputs - self.centroids[i]))]
    #         )
    #     return self.centroids

    # def predict(self, inputs):
    #     inputs = inputs.permute(0, 2, 3, 1).contiguous()
    #     inputs = inputs.view(-1, self.embedding_dim)
    #     distances = (
    #         torch.sum(inputs**2, dim=1, keepdim=True)
    #         + torch.sum(self.centroids**2, dim=1)
    #         - 2 * torch.matmul(inputs, self.centroids.t())
    #     )
    #     encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
    #     encodings = torch.zeros(
    #         encoding_indices.shape[0], self.num_embeddings, device=inputs.device
    #     )
    #     encodings.scatter_(1, encoding_indices, 1)
    #     return encodings


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        batch_norm: bool = True,
        negative_slope: Optional[float] = None,
    ):
        super().__init__()
        blocks = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm)
        ]
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_channels))
        if negative_slope:
            blocks.append(nn.LeakyReLU(negative_slope=negative_slope))
        else:
            # blocks.append(nn.ReLU())
            blocks.append(nn.ELU())
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class DeConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        batch_norm: bool = True,
        negative_slope: Optional[float] = None,
    ):
        super().__init__()
        blocks = [
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm
            )
        ]
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_channels))
        if negative_slope:
            blocks.append(nn.LeakyReLU(negative_slope=negative_slope))
        else:
            # blocks.append(nn.ReLU())
            blocks.append(nn.ELU())
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Encoder2(nn.Module):
    def __init__(
        self,
        n_channel_input: int,
        n_channel_output: int,
        observation_space: gym.spaces.box.Box = None,
        hidden_dims: List = [32, 64],
        n_redisual_layers: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.forward_call = 0

        self.convs = nn.Sequential(
            ConvBlock(
                n_channel_input,
                hidden_dims[0],
                kernel_size=8,
                stride=4,
                padding=0,
                batch_norm=True,
            ),
            ConvBlock(
                hidden_dims[0],
                hidden_dims[1],
                kernel_size=4,
                stride=2,
                padding=0,
                batch_norm=True,
            ),
            ConvBlock(
                hidden_dims[1],
                n_channel_output,
                kernel_size=3,
                stride=2,
                padding=0,
                batch_norm=True,
            ),
        )
        if n_redisual_layers:
            residuals = []
            for _ in range(n_redisual_layers):
                residuals.append(ResidualLayer(n_channel_output, n_channel_output))
            residuals.append(nn.ReLU())
            self.residual_layers = nn.Sequential(*residuals)

        # Compute shape by doing one forward pass
        # with torch.no_grad():
        #     x = observation_space.sample()
        #     x = T.ToTensor()(x).unsqueeze(0)
        #     x = self.convs(x.float())
        #     # x = self.flatten_layer(x)
        #     # n_flatten = x.shape[1]
        #     C, H, W = x[1:]

        # self.fc = nn.Linear(n_flatten, self.feature_dim)
        # self.ln = nn.LayerNorm([C, H, W])

        # self.outputs = dict()

    def forward(self, x):
        x = self.convs(x)
        if hasattr(self, "residual_layers"):
            x = self.residual_layers(x)
        # x = self.ln(x)
        # result = self.ln(result)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        linear_out_dim: Optional[int] = None,  # latent_dim
        observation_space: gym.spaces.box.Box = None,
        hidden_dims: List = [32, 64],
        n_redisual_layers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.forward_call = 0

        blocks = [
            ConvBlock(
                in_channels,
                hidden_dims[0],
                kernel_size=8,
                stride=4,
                padding=0,
                batch_norm=False,
            ),
            ConvBlock(
                hidden_dims[0],
                hidden_dims[1],
                kernel_size=4,
                stride=2,
                padding=0,
                batch_norm=False,
            ),
            ConvBlock(
                hidden_dims[1],
                hidden_dims[1],
                kernel_size=3,
                stride=2,
                padding=0,
                batch_norm=False,
            ),
        ]
        if n_redisual_layers > 0:
            for _ in range(n_redisual_layers):
                blocks.append(ResidualLayer(hidden_dims[1], hidden_dims[1]))
            blocks.append(nn.ReLU())

        # Compute shape by doing one forward pass
        self.blocks = nn.Sequential(*blocks)
        with torch.no_grad():
            x = observation_space.sample()
            x = T.ToTensor()(x).unsqueeze(0)
            x = self.blocks(x.float())
            self.shape_conv_output = x.shape
            # shape of the last feature map of the encoder, [B, C, H, W]
            self.n_flatten = torch.prod(torch.tensor(x.shape[1:])).item()
            # self.shape_latent_h_w = x.shape[2:]
            # C, H, W = x[1:]
        if linear_out_dim:  # if encoder_out_dim is not None than followed by a linear layer
            blocks.extend(
                [
                    nn.Flatten(),
                    nn.Linear(self.n_flatten, linear_out_dim),
                ]
            )
        # self.ln = nn.LayerNorm([C, H, W])

        # self.outputs = dict()
        # self.fc_mu = nn.Linear(self.n_flatten, out_channels)
        # self.fc_std = nn.Linear(self.n_flatten, out_channels)
        self.blocks = nn.Sequential(*blocks)

    # def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    def forward(self, x):
        return self.blocks(x)

        # x = self.blocks(x)
        # x = x.view(-1, self.n_flatten)
        # mu = self.fc_mu(x)
        # std = self.fc_std(x)
        # return self.reparameterize(mu, std), mu, std


class Encoder_MiniGrid(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        linear_out_dim: Optional[int] = None,  # latent_dim
        observation_space: gym.spaces.box.Box = None,
        hidden_dims: List = [16, 32],
        n_redisual_layers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.forward_call = 0
        final_channels = hidden_dims[1]
        blocks = [
            ConvBlock(
                in_channels,
                hidden_dims[0],
                kernel_size=2,
                stride=2,
                padding=0,
                batch_norm=False,
            ),
            ConvBlock(
                hidden_dims[0],
                hidden_dims[1],
                kernel_size=2,
                stride=1,
                padding=0,
                batch_norm=False,
            ),
            ConvBlock(
                hidden_dims[1],
                final_channels,
                kernel_size=2,
                stride=1,
                padding=0,
                batch_norm=False,
            ),
        ]
        if n_redisual_layers > 0:
            for _ in range(n_redisual_layers):
                blocks.append(ResidualLayer(final_channels, final_channels))
            blocks.append(nn.ReLU())

        # Compute shape by doing one forward pass
        self.blocks = nn.Sequential(*blocks)
        with torch.no_grad():
            x = observation_space.sample()
            x = T.ToTensor()(x).unsqueeze(0)
            x = self.blocks(x.float())
            self.shape_conv_output = x.shape
            # shape of the last feature map of the encoder, [B, C, H, W]
            self.n_flatten = torch.prod(torch.tensor(x.shape[1:])).item()
            # self.shape_latent_h_w = x.shape[2:]
            # C, H, W = x[1:]
        if linear_out_dim:  # if linear_out_dim is not None than followed by a linear layer
            blocks.extend(
                [
                    nn.Flatten(),
                    nn.Linear(self.n_flatten, linear_out_dim),
                ]
            )
        # self.ln = nn.LayerNorm([C, H, W])

        # self.outputs = dict()
        # self.fc_mu = nn.Linear(self.n_flatten, out_channels)
        # self.fc_std = nn.Linear(self.n_flatten, out_channels)
        self.blocks = nn.Sequential(*blocks)

    # def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    def forward(self, x):
        return self.blocks(x)

        # x = self.blocks(x)
        # x = x.view(-1, self.n_flatten)
        # mu = self.fc_mu(x)
        # std = self.fc_std(x)
        # return self.reparameterize(mu, std), mu, std


class EncoderRes(nn.Module):
    def __init__(
        self,
        n_channel_input: int,
        n_channel_output: int,
        # num_embeddings: int,
        # observation_space: gym.spaces.box.Box = None,
        # device: torch.device,
        hidden_dims: List = None,
        # beta: float = 0.25,
        # img_size: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()

        # self.embedding_dim = embedding_dim
        # self.num_embeddings = num_embeddings
        # self.img_size = img_size
        # self.beta = beta
        self.forward_call = 0

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64]

        # Build Encoder
        n_channel = n_channel_input
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        n_channel,
                        out_channels=h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            n_channel = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(n_channel, n_channel, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
        )

        for _ in range(1):
            modules.append(ResidualLayer(n_channel, n_channel))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(n_channel, n_channel_output, kernel_size=1, stride=1),
                nn.LeakyReLU(),
            )
        )

        self.convs = nn.Sequential(*modules)

        # Compute shape by doing one forward pass
        # with torch.no_grad():
        #     x = observation_space.sample()
        #     x = T.ToTensor()(x).unsqueeze(0)
        #     x = self.convs(x.float())
        #     # x = self.flatten_layer(x)
        #     # n_flatten = x.shape[1]
        #     C, H, W = x[1:]

        # self.fc = nn.Linear(n_flatten, self.feature_dim)
        # self.ln = nn.LayerNorm([C, H, W])

        # self.outputs = dict()

    def forward(self, obs):
        result = self.convs(obs)
        # result = self.ln(result)
        return result


class ReparameterizeModule(nn.Module):
    def __init__(self, out_dim: int, shape_encoder_fm: torch.Size) -> None:
        super().__init__()
        self.n_flatten = np.prod(shape_encoder_fm[1:])
        self.fc_mu = nn.Linear(self.n_flatten, out_dim)
        self.fc_std = nn.Linear(self.n_flatten, out_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        x = x.view(-1, self.n_flatten)
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        return self.reparameterize(mu, std), mu, std


class Decoder_VQ(nn.Module):
    def __init__(
        self,
        n_channel_input: int,
        n_channel_output: int,
        hidden_dims: List = None,
        n_redisual_layers: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_forward_call = 0
        hidden_dims.reverse()

        if n_redisual_layers:
            residuals = []
            for _ in range(n_redisual_layers):
                residuals.append(ResidualLayer(n_channel_input, n_channel_input))
            residuals.append(nn.ReLU())
            self.residual_layers = nn.Sequential(*residuals)

        self.convs = nn.Sequential(
            DeConvBlock(
                n_channel_input,
                hidden_dims[0],
                kernel_size=3,
                stride=2,
                padding=0,
                batch_norm=True,
            ),
            DeConvBlock(
                hidden_dims[0],
                hidden_dims[1],
                kernel_size=4,
                stride=2,
                padding=0,
                batch_norm=True,
            ),
            DeConvBlock(
                hidden_dims[1],
                n_channel_output,
                kernel_size=8,
                stride=4,
                padding=0,
                batch_norm=True,
            ),
        )

        # Compute shape by doing one forward pass
        # with torch.no_grad():
        #     x = observation_space.sample()
        #     x = T.ToTensor()(x).unsqueeze(0)
        #     x = self.convs(x.float())
        #     # x = self.flatten_layer(x)
        #     # n_flatten = x.shape[1]
        #     C, H, W = x[1:]

        # self.fc = nn.Linear(n_flatten, self.feature_dim)
        # self.ln = nn.LayerNorm([C, H, W])

        # self.outputs = dict()

    def forward(self, x):
        if hasattr(self, "residual_layers"):
            x = self.residual_layers(x)
        x = self.convs(x)
        self.n_forward_call += 1
        # result = self.ln(result)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_channels: int,
        shape_conv_output: torch.Size,
        hidden_dims: List = [32, 64],
        n_redisual_layers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_forward_call = 0
        self.shape_encoder_fm = shape_conv_output
        hidden_dims.reverse()

        self.n_flatten = torch.prod(torch.tensor(shape_conv_output[1:])).item()
        self.decoder_input = nn.Linear(in_dim, self.n_flatten)
        blocks = []
        if n_redisual_layers:
            for _ in range(n_redisual_layers):
                blocks.append(ResidualLayer(hidden_dims[0], hidden_dims[0]))
            blocks.append(nn.ReLU())

        blocks += [
            DeConvBlock(
                hidden_dims[0],
                hidden_dims[0],
                kernel_size=3,
                stride=2,
                padding=0,
                batch_norm=False,
            ),
            DeConvBlock(
                hidden_dims[0],
                hidden_dims[1],
                kernel_size=4,
                stride=2,
                padding=0,
                batch_norm=False,
            ),
            # DeConvBlock(
            #     hidden_dims[1],
            #     out_channels,
            #     kernel_size=8,
            #     stride=4,
            #     padding=0,
            #     batch_norm=True,
            # ),
            nn.ConvTranspose2d(hidden_dims[1], out_channels, 8, 4, 0),
            nn.Sigmoid(),
        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        self.n_forward_call += 1
        x = self.decoder_input(x)
        x = x.view(-1, self.shape_encoder_fm[1], self.shape_encoder_fm[2], self.shape_encoder_fm[3])
        return self.blocks(x)


class Decoder_MiniGrid(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_channels: int,
        shape_conv_output: torch.Size,
        hidden_dims: List = [16, 32],
        n_redisual_layers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_forward_call = 0
        self.shape_encoder_fm = shape_conv_output
        first_channels = shape_conv_output[1]

        self.n_flatten = torch.prod(torch.tensor(shape_conv_output[1:])).item()
        self.decoder_input = nn.Linear(in_dim, self.n_flatten)
        blocks = []
        if n_redisual_layers:
            for _ in range(n_redisual_layers):
                blocks.append(ResidualLayer(first_channels, first_channels))
            blocks.append(nn.ReLU())

        hidden_dims.reverse()
        blocks += [
            # DeConvBlock(
            #     first_channels,
            #     hidden_dims[0],
            #     kernel_size=2,
            #     stride=1,
            #     padding=0,
            #     batch_norm=False,
            # ),
            DeConvBlock(
                hidden_dims[0],
                hidden_dims[1],
                kernel_size=2,
                stride=1,
                padding=0,
                batch_norm=False,
            ),
            DeConvBlock(
                hidden_dims[1],
                out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                batch_norm=False,
            ),
        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        self.n_forward_call += 1
        x = self.decoder_input(x)
        x = x.view(-1, self.shape_encoder_fm[1], self.shape_encoder_fm[2], self.shape_encoder_fm[3])
        return self.blocks(x)


class DecoderRes(nn.Module):
    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
        # embedding_dim: int,
        # num_embeddings: int,
        hidden_dims: List = None,
        # beta: float = 0.25,
        # img_size: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()

        # self.embedding_dim = embedding_dim
        # self.num_embeddings = num_embeddings
        # self.img_size = img_size
        # self.beta = beta
        # self.reconstruction_path = reconstruction_path

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64]

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    n_input_channels,
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.LeakyReLU(),
            )
        )

        for _ in range(1):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    out_channels=n_output_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.Tanh(),
            )
        )

        self.convs = nn.Sequential(*modules)
        self.n_forward_call = 0

    def forward(self, x):
        recon = self.convs(x)
        # result = self.ln(result)
        self.n_forward_call += 1
        return recon


class Q_MLP(nn.Module):
    def __init__(self, input_dim, action_space, flatten=False, hidden_dim=64):
        super().__init__()
        blocks = []
        if flatten:
            blocks.extend(
                [
                    nn.Flatten(),
                    nn.Linear(np.prod(input_dim), hidden_dim),
                    nn.ReLU(),
                ]
            )
        else:
            blocks.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    # nn.ReLU(),
                    nn.ELU(),
                    # nn.Tanh(),
                ]
            )
        blocks.extend(
            [
                # nn.Linear(256, 256),
                # nn.ReLU(),
                # nn.Linear(hidden_dim, hidden_dim),
                # nn.ReLU(),
                # nn.Tanh(),
                nn.Linear(hidden_dim, action_space.n),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class V_MLP(nn.Module):
    def __init__(self, input_dim, flatten=False):
        super().__init__()
        blocks = []
        if flatten:
            blocks.extend(
                [
                    nn.Flatten(),
                    nn.Linear(np.prod(input_dim), 512),
                    nn.ReLU(),
                ]
            )
        else:
            blocks.extend(
                [
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                ]
            )
        blocks.extend(
            [
                # nn.Linear(256, 256),
                # nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            ]
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class DQN(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.box.Box = None,
        action_space: gym.spaces.discrete.Discrete = None,
        embedding_dim: int = 32,
        hidden_dims: List = [32, 64],
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = Encoder_MiniGrid(
            observation_space.shape[-1],
            linear_out_dim=embedding_dim,
            observation_space=observation_space,
            hidden_dims=hidden_dims,
        )

        with torch.no_grad():
            x = observation_space.sample()
            x = T.ToTensor()(x).unsqueeze(0)
            x = self.encoder(x.float())
            # x = self.flatten_layer(x)
            # n_flatten = x.shape[1]
            self.encoder_out_shape = x.shape[1:]

        self.critic = Q_MLP(embedding_dim, action_space, flatten=False, hidden_dim=64)

    def forward_conv(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return self.critic(x), x


class DQN_Repara(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.box.Box = None,
        action_space: gym.spaces.discrete.Discrete = None,
        embedding_dim: int = 32,
        hidden_dims: List = [32, 64],
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        # self.conv_block = Encoder_MiniGrid(
        #     observation_space.shape[-1],
        #     linear_out_dim=None,
        #     observation_space=observation_space,
        #     hidden_dims=hidden_dims,
        # )
        self.conv_block = Encoder(
            observation_space.shape[-1],
            linear_out_dim=None,
            observation_space=observation_space,
            hidden_dims=hidden_dims,
        )

        self.repara = ReparameterizeModule(embedding_dim, self.conv_block.shape_conv_output)
        self.encoder = nn.Sequential(self.conv_block, self.repara)
        self.Q_linear = Q_MLP(embedding_dim, action_space, flatten=False)

    def forward_conv(self, x: Tensor):
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        x, mu, std = self.encoder(x)
        return self.Q_linear(x), x, mu, std


class HDQN(nn.Module):
    def __init__(self, config, env: gym.Env) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = int(time.time())
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        self.set_hparams(config)
        self.env = env
        # self.n_actions = env.action_space.n
        # self.learn_with_ae = config.learn_with_ae
        # self.init_clustering = config.init_clustering

        # self.kmeans = Batch_KMeans(
        #     n_clusters=config.n_clusters, embedding_dim=config.embedding_dim, device=self.device
        # ).to(self.device)

        # self.aug = RandomShiftsAug(pad=4)

        self.outputs = dict()
        # self.apply(weight_init)

        # Initialize experience replay buffer
        self.memory = ReplayMemory(self.size_replay_memory, self.device)
        # self.Transition = namedtuple(
        #     "Transition", ("state", "action", "next_state", "reward", "done")
        # )
        self.exploration_scheduler = get_linear_fn(
            config.exploration_initial_eps,
            config.exploration_final_eps,
            config.exploration_fraction,
        )

        self.timesteps_done = 0
        self.episodes_done = 0
        # self.n_call_train = 0
        self._current_progress_remaining = 1.0
        self.to_buffer = False  # for func maybe_buffer_recent_states

        # self._create_optimizers(config)
        # self.reset_training_info()
        # self.train()

    def train(self, training=True):
        raise NotImplementedError

    def reset_training_info(self):
        raise NotImplementedError

    def log_training_info(self, wandb_log=True):
        if wandb_log:
            metrics = {
                "loss/ground_Q_error": mean(self.training_info["ground_Q_error"]),
                "loss/abstract_V_error": mean(self.training_info["abstract_V_error"]),
                "train/exploration_rate": self.exploration_rate,
                "train/current_progress_remaining": self._current_progress_remaining,
                "lr/lr_ground_Q_optimizer": self.ground_Q_optimizer.param_groups[0]["lr"],
                "lr/lr_abstract_V_optimizer": self.abstract_V_optimizer.param_groups[0]["lr"],
            }
            wandb.log(metrics)

    def load_states_from_memory(self, unique=True):
        transitions = random.sample(self.memory.memory, len(self.memory))
        batch = self.memory.Transition(*zip(*transitions))
        state_batch = np.stack(batch.state, axis=0).transpose(0, 3, 1, 2)
        if unique:
            state_batch = np.unique(state_batch, axis=0)
        state_batch = torch.from_numpy(state_batch).contiguous().float().to(self.device)

        # Use when states are cached as Tensor
        # batch = self.memory.sample(batch_size=len(self.memory))
        # state_batch = torch.cat(batch.next_state)
        # state_batch = torch.unique(state_batch, dim=0).float().to(self.device)

        return state_batch

    def triangulation_for_triheatmap(self, M, N):
        # M: number of columns, N: number of rows
        xv, yv = np.meshgrid(
            np.arange(-0.5, M), np.arange(-0.5, N)
        )  # vertices of the little squares
        xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
        x = np.concatenate([xv.ravel(), xc.ravel()])
        y = np.concatenate([yv.ravel(), yc.ravel()])
        cstart = (M + 1) * (N + 1)  # indices of the centers
        print(cstart)

        trianglesN = [
            (i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
            for j in range(N)
            for i in range(M)
        ]
        trianglesE = [
            (i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
            for j in range(N)
            for i in range(M)
        ]
        trianglesS = [
            (i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
            for j in range(N)
            for i in range(M)
        ]
        trianglesW = [
            (i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
            for j in range(N)
            for i in range(M)
        ]
        return [
            Triangulation(x, y, triangles)
            for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]
        ]

    def visualize_clusters_minigrid(self):
        # trainsitions_in_memory = self.memory.sample(batch_size=len(self.memory))
        # arrays = np.stack(trainsitions_in_memory.state, axis=0)
        # arrays_unique = np.unique(arrays, axis=0)
        # tensors_unique = (
        #     torch.from_numpy(arrays_unique.transpose(0, 3, 1, 2))
        #     .contiguous()
        #     .float()
        #     .to(self.device)
        # )

        batch = self.memory.sample(batch_size=len(self.memory))
        state_batch = torch.cat(batch.state)
        state_batch = state_batch.cpu().numpy()
        unique_array = np.unique(state_batch, axis=0)
        tensors_unique = torch.from_numpy(unique_array).float().to(self.device)

        with torch.no_grad():
            embeddings = self.ground_Q.encoder(tensors_unique)[0]
            cluster_indices = self.kmeans.assign_clusters(embeddings)
        # states_in_memory = self.load_states_from_memory()
        # arrays = torch.unique(states_in_memory, dim=0)
        # arrays = arrays.cpu().numpy().transpose(0, 3, 1, 2)
        batch, channels, width, height = tensors_unique.shape
        list_of_agent_pos_dir = []
        # clustersN = np.empty(shape=(height, width))
        # clustersS = np.empty(shape=(height, width))
        # clustersW = np.empty(shape=(height, width))
        # clustersE = np.empty(shape=(height, width))
        clustersN = np.full(shape=(height, width), fill_value=4)
        clustersS = np.full(shape=(height, width), fill_value=4)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=4)
        clustersE = np.full(shape=(height, width), fill_value=4)
        for idx, array in enumerate(tensors_unique):
            # break
            for i in range(width):
                for j in range(height):
                    type_idx, color_idx, state = array[:, i, j]
                    if type_idx == 10:  # if type is agent
                        assert 0 <= state < 4
                        if state == 3:
                            clustersN[j, i] = 0
                        elif state == 2:
                            clustersW[j, i] = 1
                        elif state == 1:
                            clustersS[j, i] = 2
                        elif state == 0:
                            clustersE[j, i] = 3
                    # agent_pos = (i, j)
                    # agent_dir = state
                    # list_of_agent_pos_dir.append((agent_pos, agent_dir))
        values = [clustersN, clustersE, clustersS, clustersW]
        triangulations = self.triangulation_for_triheatmap(width, height)
        fig, ax = plt.subplots()
        vmax = 4
        vmin = 0
        imgs = [
            ax.tripcolor(
                t,
                np.ravel(val),
                vmin=vmin,
                vmax=vmax,
                cmap="gist_ncar",
                ec="black",
            )
            for t, val in zip(triangulations, values)
        ]
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    def visualize_clusters_minigrid2(self):
        # take out all unique states from replay buffer and visualize their clusters
        # This approach might not cover all the states in the environment

        batch = self.memory.sample(batch_size=len(self.memory))
        # ===When states are cached as channel-first tensors===
        state_batch = torch.cat(batch.next_state)
        state_batch = state_batch.cpu().numpy()
        unique_array, indices = np.unique(state_batch, return_index=True, axis=0)
        unique_info_list = [batch.info[i] for i in indices]
        unique_tensor = torch.from_numpy(unique_array).float().to(self.device)

        # ===When states are cached as numpy arrays===
        # state_batch = np.stack(batch.next_state, axis=0)
        # unique_array, indices = np.unique(state_batch, return_index=True, axis=0)
        # unique_info_list = [batch.info[i] for i in indices]
        # unique_tensor = (
        #     torch.from_numpy(unique_array.transpose(0, 3, 1, 2))
        #     .contiguous()
        #     .float()
        #     .to(self.device)
        # )

        with torch.no_grad():
            embeddings = self.ground_Q.encoder(unique_tensor)[0]
            cluster_indices = self.kmeans.assign_clusters(embeddings)
        # states_in_memory = self.load_states_from_memory()
        # arrays = torch.unique(states_in_memory, dim=0)
        # arrays = arrays.cpu().numpy().transpose(0, 3, 1, 2)
        width = self.env.width
        height = self.env.height
        num_cluster = self.kmeans.n_clusters
        # clustersN = np.empty(shape=(height, width))
        # clustersS = np.empty(shape=(height, width))
        # clustersW = np.empty(shape=(height, width))
        # clustersE = np.empty(shape=(height, width))
        clustersN = np.full(shape=(height, width), fill_value=num_cluster)
        clustersS = np.full(shape=(height, width), fill_value=num_cluster)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=num_cluster)
        clustersE = np.full(shape=(height, width), fill_value=num_cluster)

        print(cluster_indices.shape, len(unique_info_list))
        n, w, s, e = 0, 0, 0, 0
        for cluster_idx, info in zip(cluster_indices, unique_info_list):
            agent_pos = info["agent_pos"]
            agent_dir = info["agent_dir"]
            assert 0 <= agent_dir < 4
            if agent_dir == 3:
                n += 1
                clustersN[agent_pos[1], agent_pos[0]] = cluster_idx
            if agent_dir == 2:
                w += 1
                clustersW[agent_pos[1], agent_pos[0]] = cluster_idx
            if agent_dir == 1:
                s += 1
                clustersS[agent_pos[1], agent_pos[0]] = cluster_idx
            if agent_dir == 0:
                e += 1
                clustersE[agent_pos[1], agent_pos[0]] = cluster_idx
        print(n, w, s, e)
        values = [clustersN, clustersE, clustersS, clustersW]
        triangulations = self.triangulation_for_triheatmap(width, height)
        fig, ax = plt.subplots()
        vmax = num_cluster
        vmin = 0
        imgs = [
            ax.tripcolor(t, np.ravel(val), vmin=vmin, vmax=vmax, cmap="gist_ncar", ec="black")
            for t, val in zip(triangulations, values)
        ]
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    # def cluster_visualize_memory3(self):
    #         # take out all unique states from replay buffer and visualize their clusters
    #         # This approach might not cover all the states in the environment
    #         trainsitions_in_memory = self.memory.sample(batch_size=len(self.memory))
    #         arrays = np.stack(trainsitions_in_memory.state, axis=0)
    #         infos = trainsitions_in_memory.info
    #         arrays_unique, indices = np.unique(arrays, return_index=True, axis=0)
    #         infos = infos[indices]
    #         tensors_unique = (
    #             torch.from_numpy(arrays_unique.transpose(0, 3, 1, 2))
    #             .contiguous()
    #             .float()
    #             .to(self.device)
    #         )
    #         with torch.no_grad():
    #             embeddings = self.ground_Q.encoder(tensors_unique)[0]
    #             cluster_indices = self.kmeans.assign_clusters(embeddings)
    #         # states_in_memory = self.load_states_from_memory()
    #         # arrays = torch.unique(states_in_memory, dim=0)
    #         # arrays = arrays.cpu().numpy().transpose(0, 3, 1, 2)
    #         width = self.env.width
    #         height = self.env.height
    #         # clustersN = np.empty(shape=(height, width))
    #         # clustersS = np.empty(shape=(height, width))
    #         # clustersW = np.empty(shape=(height, width))
    #         # clustersE = np.empty(shape=(height, width))
    #         clustersN = np.full(shape=(height, width), fill_value=4)
    #         clustersS = np.full(shape=(height, width), fill_value=4)
    #         # clustersS = np.random.randint(0, 4, size=(height, width))
    #         clustersW = np.full(shape=(height, width), fill_value=4)
    #         clustersE = np.full(shape=(height, width), fill_value=4)

    #         for i in range(width):
    #             for j in range(height):
    #                 pass

    #         for cluster_idx, info in zip(cluster_indices, infos):
    #             agent_pos = info["agent_pos"]
    #             agent_dir = info["agent_dir"]
    #             if agent_dir == 3:
    #                 clustersN[agent_pos[1], agent_pos[0]] = cluster_idx
    #             if agent_dir == 2:
    #                 clustersW[agent_pos[1], agent_pos[0]] = cluster_idx
    #             if agent_dir == 1:
    #                 clustersS[agent_pos[1], agent_pos[0]] = cluster_idx
    #             if agent_dir == 0:
    #                 clustersE[agent_pos[1], agent_pos[0]] = cluster_idx

    #         values = [clustersN, clustersE, clustersS, clustersW]
    #         triangulations = self.triangulation_for_triheatmap(width, height)
    #         fig, ax = plt.subplots()
    #         vmax = 4
    #         vmin = 0
    #         imgs = [
    #             ax.tripcolor(t, np.ravel(val), vmin=vmin, vmax=vmax, cmap="gist_ncar", ec="black")
    #             for t, val in zip(triangulations, values)
    #         ]
    #         ax.invert_yaxis()
    #         plt.tight_layout()
    #         plt.show()

    def set_hparams(self, config):
        raise NotImplementedError

    def _create_optimizers(self, config):
        raise NotImplementedError

    def _update_current_progress_remaining(self, timesteps_done, total_timesteps):
        # self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)
        finished_time_steps_after_init = timesteps_done - self.init_steps
        if finished_time_steps_after_init < 0:
            self._current_progress_remaining = 1.0
        else:
            self._current_progress_remaining = (
                1.0 - finished_time_steps_after_init / total_timesteps
            )

    @torch.no_grad()
    def maybe_buffer_recent_states(self, state, buffer_length=30):
        if self.timesteps_done % self.buffer_recent_states_every == 0 and self.timesteps_done > 0:
            self.recent_states = []
            self.to_buffer = True
        if self.to_buffer:
            self.recent_states.append(state)
            if len(self.recent_states) == buffer_length:
                print("##Check how good abstraction is##")
                # convert self.recent_states to torch tensor
                self.recent_states = (
                    (torch.tensor(self.recent_states).permute(0, 3, 1, 2).contiguous())
                    .float()
                    .to(self.device)
                )
                grd_q, encoded = self.ground_Q(self.recent_states)
                quantized, vq_loss, vq_entrophy, encodings = self.vector_quantizer(encoded)
                abs_v = self.abstract_V(quantized)
                (clusters, inverse_indice, counts) = torch.unique(
                    quantized,
                    return_inverse=True,
                    return_counts=True,
                    dim=0,
                )
                print("number of clusters:\n", len(clusters))
                print("inverse_indice:\n", inverse_indice.tolist())
                print("counts:\n", counts.tolist())
                # log n_abstract_states by wandb
                wandb.log({"abstraction/n_clusters_in_buffer": len(clusters)})
                wandb.log({"abstraction/value_difference": torch.abs(abs_v - grd_q).mean().item()})
                self.to_buffer = False

    def cache(self, state, action, next_state, reward, terminated, info):
        """Add the experience to memory"""
        # if state_type == "rgb":
        #     state = T.ToTensor()(state).float().unsqueeze(0)
        #     next_state = T.ToTensor()(next_state).float().unsqueeze(0)
        # else:
        #     state = torch.from_numpy(state.transpose((2, 0, 1))).contiguous().float().unsqueeze(0)
        #     next_state = (
        #         torch.from_numpy(next_state.transpose((2, 0, 1))).contiguous().float().unsqueeze(0)
        #     )
        # if state_type == "img":
        #     state = state / 255.0
        #     next_state = next_state / 255.0
        # action = torch.tensor([action]).unsqueeze(0)
        # reward = torch.tensor([reward]).unsqueeze(0)
        # terminated = torch.tensor([terminated]).unsqueeze(0)

        self.memory.push(state, action, next_state, reward, terminated, info)

    def act(self, state):
        self._update_current_progress_remaining(self.timesteps_done, self.total_timesteps)
        self.exploration_rate = self.exploration_scheduler(self._current_progress_remaining)
        with torch.no_grad():
            state = T.ToTensor()(state).float().unsqueeze(0).to(self.device)

            if random.random() > self.exploration_rate:
                action = self.ground_Q(state)[0].max(1)[1].item()
            else:
                action = random.randrange(self.n_actions)

        self.timesteps_done += 1
        return action

    def update(self):
        raise NotImplementedError

    def update_grdQ_pure(self, state, action, next_state, reward, terminated, shaping=True):
        if hasattr(self, "lr_scheduler_ground_Q"):
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_scheduler_ground_Q(self._current_progress_remaining),
            )

        state = self.aug(state)
        next_state = self.aug(next_state)

        # [Update ground Q network]
        grd_q, encoded, mu, std = self.ground_Q(state)
        grd_q = grd_q.gather(1, action)

        with torch.no_grad():

            # Vanilla DQN
            grd_q_next, encoded_next = self.ground_Q_target(next_state)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)

            # Double DQN
            # action_argmax_target = self.ground_target_Q_net(next_state_batch).argmax(
            #     dim=1, keepdim=True
            # )
            # ground_next_max_Q = self.ground_Q_net(next_state_batch).gather(1, action_argmax_target)

            # Compute ground target Q value

            grd_q_target = (reward + (1 - terminated.float()) * self.gamma * grd_q_next_max).float()

        criterion = nn.SmoothL1Loss()
        ground_td_error = criterion(grd_q, grd_q_target)

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        ground_td_error.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()

        self.training_info["ground_Q_error"].append(ground_td_error.item())

    def update_abs_Table(self):
        raise NotImplementedError


class HDQN_ManualAbs(HDQN):
    def __init__(self, config, env, use_table4grd=False):
        super().__init__(config, env)
        # self.set_hparams(config)
        self.n_actions = env.action_space.n
        self.n_clusters = config.n_clusters
        self.use_table4grd = use_table4grd

        # self.abs_ticks = np.array([5, 10, 14])
        # self.abs_ticks = np.array(config.abs_ticks)
        # self.abs_txt_ticks = ((np.insert(self.abs_ticks, 0, 0) + np.append(self.abs_ticks, 0)) / 2)[
        #     :-1
        # ]

        if use_table4grd:
            self.grd_Q_table = np.zeros((env.width - 2, env.height - 2, 4, env.action_space.n))
            self.lr_grd_Q = config.lr_ground_Q
            self.grd_visits = np.zeros((env.height, env.width, 4))
        else:
            self.ground_Q = DQN(
                env.observation_space,
                env.action_space,
                config.embedding_dim,
                hidden_dims=config.hidden_dims,
                # embedding_dim=config.latent_dim,
            ).to(self.device)

            self.ground_Q_target = DQN(
                env.observation_space,
                env.action_space,
                config.embedding_dim,
                hidden_dims=config.hidden_dims,
                # embedding_dim=config.latent_dim,
            ).to(self.device)
            self.ground_Q_target.load_state_dict(self.ground_Q.state_dict())
            self.ground_Q_target.train()

        self.abstract_V_array = np.zeros((config.n_clusters))
        self.lr_abs_V = config.lr_abstract_V
        self.abstract_eligibllity_list = []

        self.aug = RandomShiftsAug(pad=4)

        self.timesteps_done = 0
        self.episodes_done = 0
        self._current_progress_remaining = 1.0
        self.to_buffer_recent_states = False  # for func maybe_buffer_recent_states
        self.count_vis = 0

        if not use_table4grd:
            self._create_optimizers(config)
            self.train()
        self.reset_training_info()

    def train(self, training=True):
        self.training = training
        self.ground_Q.train(training)

    def reset_training_info(self):
        self.training_info = {
            "ground_Q_error": [],
            "abstract_V_error": [],
            "avg_shaping": [],
        }

    def set_abs_ticks(self, config, idx_abs_layout):
        self.abs_ticks = np.array(config.abs_ticks[idx_abs_layout])
        self.abs_txt_ticks = ((np.insert(self.abs_ticks, 0, 0) + np.append(self.abs_ticks, 0)) / 2)[
            :-1
        ]

    def set_hparams(self, config):
        # Hyperparameters
        # self.total_episodes = config.total_episodes
        self.total_timesteps = config.total_timesteps
        self.init_steps = config.init_steps  # min. experiences before training
        self.batch_size = config.batch_size
        self.size_replay_memory = config.size_replay_memory
        self.gamma = config.gamma
        self.omega = config.omega
        self.ground_tau = config.ground_tau
        # self.encoder_tau = config.encoder_tau
        # self.abstract_tau = config.abstract_tau

        self.ground_learn_every = config.ground_learn_every
        self.ground_sync_every = config.ground_sync_every
        self.ground_gradient_steps = config.ground_gradient_steps
        # self.abstract_learn_every = config.abstract_learn_every
        # self.abstract_sync_every = config.abstract_sync_every
        # self.abstract_gradient_steps = config.abstract_gradient_steps

        # self.validate_every = config.validate_every
        self.save_model_every = config.save_model_every
        self.reset_training_info_every = config.reset_training_info_every
        # self.save_recon_every = config.save_recon_every
        # self.buffer_recent_states_every = config.buffer_recent_states_every

        self.clip_grad = config.clip_grad

    def log_training_info(self, wandb_log=True):
        if wandb_log:
            metrics = {
                "training_info/ground_Q_error": mean(self.training_info["ground_Q_error"]),
                "training_info/abstract_V_error": mean(self.training_info["abstract_V_error"]),
                "training_info/avg_shaping": mean(self.training_info["avg_shaping"]),
                "training_info/timesteps_done": self.timesteps_done,
                "training_info/episodes_done": self.episodes_done,
                "training_info/exploration_rate": self.exploration_rate,
                "training_info/current_progress_remaining": self._current_progress_remaining,
                # "lr/lr_ground_Q_optimizer": self.ground_Q_optimizer.param_groups[0]["lr"],
                # "lr/lr_abstract_V_optimizer": self.abstract_V_optimizer.param_groups[0]["lr"],
            }
            wandb.log(metrics)

            # print("logging training info:")
            # pp(metrics)

    def vis_abstraction_values(self, prefix: str):
        width = self.env.width
        height = self.env.height
        # num_cluster = self.kmeans.n_clusters
        # clustersN = np.empty(shape=(height, width))
        # clustersS = np.empty(shape=(height, width))
        # clustersW = np.empty(shape=(height, width))
        # clustersE = np.empty(shape=(height, width))
        clustersN = np.full(shape=(height, width), fill_value=-1)
        clustersS = np.full(shape=(height, width), fill_value=-1)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=-1)
        clustersE = np.full(shape=(height, width), fill_value=-1)

        clustersN2 = np.full(shape=(height, width), fill_value=-1.0, dtype=np.float32)
        clustersS2 = np.full(shape=(height, width), fill_value=-1.0, dtype=np.float32)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW2 = np.full(shape=(height, width), fill_value=-1.0, dtype=np.float32)
        clustersE2 = np.full(shape=(height, width), fill_value=-1.0, dtype=np.float32)

        for w in range(width - 2):
            w += 1
            for h in range(height - 2):
                h += 1
                abstract_state_idx, abstract_value = self.get_abstract_value((w, h))
                clustersN[h, w] = abstract_state_idx
                clustersE[h, w] = abstract_state_idx
                clustersS[h, w] = abstract_state_idx
                clustersW[h, w] = abstract_state_idx

                clustersN2[h, w] = abstract_value
                clustersE2[h, w] = abstract_value
                clustersS2[h, w] = abstract_value
                clustersW2[h, w] = abstract_value

        values = [clustersN, clustersE, clustersS, clustersW]
        values2 = [clustersN2, clustersE2, clustersS2, clustersW2]

        triangulations = self.triangulation_for_triheatmap(width, height)

        xx, yy = np.meshgrid(self.abs_txt_ticks, self.abs_txt_ticks)
        xx = xx.flatten()
        yy = yy.flatten()

        # [Plot Abstraction]
        fig_abs, ax_abs = plt.subplots(figsize=(5, 5))
        vmax = self.n_clusters
        vmin = 0
        my_cmap = copy.copy(plt.cm.get_cmap("gist_ncar"))
        my_cmap.set_under(color="dimgray")
        imgs = [
            ax_abs.tripcolor(t, np.ravel(val), vmin=vmin, vmax=vmax, cmap=my_cmap, ec="black")
            for t, val in zip(triangulations, values)
        ]
        for i, (x, y) in enumerate(zip(xx, yy)):
            ax_abs.text(
                x,
                y,
                str(i),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=13,
                color="black",
                fontweight="semibold",
            )
        ax_abs.invert_yaxis()
        fig_abs.tight_layout()
        plt.close(fig_abs)

        # [Plot Abstract Values]
        vmin = self.abstract_V_array.min()
        vmax = self.abstract_V_array.max() + 0.00001
        # vmin = vmin - 0.07 * (vmax - vmin)
        my_cmap = copy.copy(plt.cm.get_cmap("hot"))
        my_cmap.set_under(color="green")
        fig_abs_v, ax_abs_v = plt.subplots(figsize=(5, 5))
        imgs2 = [
            ax_abs_v.tripcolor(
                t,
                np.ravel(val),
                vmin=vmin,
                vmax=vmax,
                # norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=my_cmap,
                ec="black",
            )
            for t, val in zip(triangulations, values2)
        ]

        abstract_V_array = self.abstract_V_array.round(3)
        for i, (x, y) in enumerate(zip(xx, yy)):
            ax_abs_v.text(
                x,
                y,
                abstract_V_array[i],
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=13,
                color="green",
                fontweight="semibold",
            )
        ax_abs_v.invert_yaxis()
        # ax_abs_v.set_xlabel(f"Abs_V:{abstract_V_array}")
        fig_abs_v.tight_layout()
        plt.close(fig_abs_v)
        # self.count_vis += 1
        # f_name = f"/workspace/repos_dev/VQVAE_RL/figures/minigrid_abstraction/{prefix}_{self.count_vis}.png"
        # fig.savefig(
        #     f_name,
        #     dpi=200,
        #     facecolor="w",
        #     edgecolor="w",
        #     orientation="portrait",
        #     format=None,
        #     transparent=False,
        #     bbox_inches=None,
        #     pad_inches=0.1,
        # )
        # print(f"saving fig: {f_name}")

        # pil_object = PIL.Image.frombytes(
        #     "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        # )

        # [alternative plot of abstract values]
        # test = self.abstract_V_array.reshape(3, 3)
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.imshow(
        #     test,
        #     # norm=colors.LogNorm(),
        #     cmap="hot",
        # )
        # fig.savefig(
        #     f"/workspace/repos_dev/VQVAE_RL/figures/minigrid_abstraction/{prefix}_{self.count_vis}_test.png",
        #     dpi=200,
        #     facecolor="w",
        #     edgecolor="w",
        #     orientation="portrait",
        #     format=None,
        #     transparent=False,
        #     bbox_inches=None,
        #     pad_inches=0.1,
        # )
        # plt.close(fig)

    def vis_abstraction(self, prefix: str = None):
        width = self.env.width
        height = self.env.height

        clustersN = np.full(shape=(height, width), fill_value=-1)
        clustersS = np.full(shape=(height, width), fill_value=-1)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=-1)
        clustersE = np.full(shape=(height, width), fill_value=-1)

        for w in range(width - 2):
            w += 1
            for h in range(height - 2):
                h += 1
                abstract_state_idx, abstract_value = self.get_abstract_value((w, h))
                clustersN[h, w] = abstract_state_idx
                clustersE[h, w] = abstract_state_idx
                clustersS[h, w] = abstract_state_idx
                clustersW[h, w] = abstract_state_idx

        values = [clustersN, clustersE, clustersS, clustersW]
        triangulations = self.triangulation_for_triheatmap(width, height)

        xx, yy = np.meshgrid(self.abs_txt_ticks, self.abs_txt_ticks)
        xx = xx.flatten()
        yy = yy.flatten()

        # [Plot Abstraction]
        fig_abs, ax_abs = plt.subplots(figsize=(5, 5))
        vmax = self.n_clusters
        vmin = 0
        my_cmap = copy.copy(plt.cm.get_cmap("gist_ncar"))
        my_cmap.set_under(color="dimgray")
        imgs = [
            ax_abs.tripcolor(
                t,
                np.ravel(val),
                vmin=vmin,
                vmax=vmax,
                cmap=my_cmap,
                ec="black",
            )
            for t, val in zip(triangulations, values)
        ]
        for i, (x, y) in enumerate(zip(xx, yy)):
            ax_abs.text(
                x,
                y,
                str(i),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=13,
                color="black",
                fontweight="semibold",
            )
        ax_abs.invert_yaxis()
        fig_abs.tight_layout()

        img_buffer = io.BytesIO()
        fig_abs.savefig(
            img_buffer,
            dpi=100,
            # facecolor="w",
            # edgecolor="w",
            # orientation="portrait",
            # transparent=False,
            # bbox_inches=None,
            # pad_inches=0.1,
            format="png",
        )
        img = Image.open(img_buffer)
        wandb.log({"Images/abstraction": wandb.Image(img)})
        img_buffer.close()
        plt.close(fig_abs)

    def vis_abstract_values(self, prefix: str = None):
        width = self.env.width
        height = self.env.height
        clustersN = np.full(shape=(height, width), fill_value=-1.0, dtype=np.float32)
        clustersE = np.full(shape=(height, width), fill_value=-1.0, dtype=np.float32)
        clustersS = np.full(shape=(height, width), fill_value=-1.0, dtype=np.float32)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=-1.0, dtype=np.float32)

        for w in range(width - 2):
            w += 1
            for h in range(height - 2):
                h += 1
                abstract_state_idx, abstract_value = self.get_abstract_value((w, h))
                clustersN[h, w] = abstract_value
                clustersE[h, w] = abstract_value
                clustersS[h, w] = abstract_value
                clustersW[h, w] = abstract_value

        values = [clustersN, clustersE, clustersS, clustersW]

        triangulations = self.triangulation_for_triheatmap(width, height)
        xx, yy = np.meshgrid(self.abs_txt_ticks, self.abs_txt_ticks)
        xx = xx.flatten()
        yy = yy.flatten()

        # [Plot Abstract Values]
        fig_abs_v, ax_abs_v = plt.subplots(figsize=(5, 5))
        vmin = self.abstract_V_array.min()
        vmax = self.abstract_V_array.max() + 0.00001
        # vmin = vmin - 0.07 * (vmax - vmin)
        my_cmap = copy.copy(plt.cm.get_cmap("hot"))
        my_cmap.set_under(color="green")
        imgs2 = [
            ax_abs_v.tripcolor(
                t,
                np.ravel(val),
                vmin=vmin,
                vmax=vmax,
                # norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=my_cmap,
                ec="black",
                lw=0.02,
            )
            for t, val in zip(triangulations, values)
        ]
        # round the values
        abstract_V_array = self.abstract_V_array.round(3)
        for i, (x, y) in enumerate(zip(xx, yy)):
            ax_abs_v.text(
                x,
                y,
                abstract_V_array[i],
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=11,
                color="blue",
                fontweight="semibold",
            )
        ax_abs_v.invert_yaxis()
        # ax_abs_v.set_xlabel(f"Abs_V:{abstract_V_array}")
        fig_abs_v.tight_layout()

        img_buffer = io.BytesIO()
        fig_abs_v.savefig(
            img_buffer,
            dpi=100,
            # facecolor="w",
            # edgecolor="w",
            # orientation="portrait",
            # transparent=False,
            # bbox_inches=None,
            # pad_inches=0.1,
            format="png",
        )
        # wandb.define_metric("Images/time_steps_done")
        # wandb.define_metric("Images/abs_values", step_metric="Images/time_steps_done")
        img = Image.open(img_buffer)
        wandb.log({"Images/abs_values": wandb.Image(img)})
        img_buffer.close()
        plt.close(fig_abs_v)

    def vis_grd_visits(self):
        width = self.env.width
        height = self.env.height
        clustersN = np.full(shape=(height, width), fill_value=-1)
        clustersE = np.full(shape=(height, width), fill_value=-1)
        clustersS = np.full(shape=(height, width), fill_value=-1)
        clustersW = np.full(shape=(height, width), fill_value=-1)

        for w in range(width - 2):
            w += 1
            for h in range(height - 2):
                h += 1
                for i in range(4):
                    if i == 0:
                        clustersE[h, w] = self.grd_visits[h, w, i]
                    if i == 1:
                        clustersS[h, w] = self.grd_visits[h, w, i]
                    if i == 2:
                        clustersW[h, w] = self.grd_visits[h, w, i]
                    if i == 3:
                        clustersN[h, w] = self.grd_visits[h, w, i]

        values = [clustersN, clustersE, clustersS, clustersW]

        triangulations = self.triangulation_for_triheatmap(width, height)

        # [Plot Abstract Values]
        fig_grd_visits, ax_grd_visits = plt.subplots(figsize=(5, 5))
        vmin = self.grd_visits.min()
        vmax = self.grd_visits.max()
        my_cmap = copy.copy(plt.cm.get_cmap("hot"))
        my_cmap.set_under(color="green")
        imgs = [
            ax_grd_visits.tripcolor(
                t,
                np.ravel(val),
                vmin=vmin,
                vmax=vmax,
                # norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=my_cmap,
                ec="black",
                lw=0.02,
            )
            for t, val in zip(triangulations, values)
        ]
        ax_grd_visits.invert_yaxis()
        # cax = fig_grd_visits.add_axes([0.9, 0.23, 0.03, 0.5])
        # fig_grd_visits.colorbar(ax_grd_visits, cax=cax)
        ax_grd_visits.set_xlabel(f"vmin:{vmin}, vmax:{vmax}")
        fig_grd_visits.tight_layout()
        img_buffer = io.BytesIO()
        fig_grd_visits.savefig(
            img_buffer,
            dpi=100,
            # facecolor="w",
            # edgecolor="w",
            # orientation="portrait",
            # transparent=False,
            # bbox_inches=None,
            # pad_inches=0.1,
            format="png",
        )
        # wandb.define_metric("Images/time_steps_done")
        # wandb.define_metric("Images/abs_values", step_metric="Images/time_steps_done")
        img = Image.open(img_buffer)
        # print("save visits img")
        # img.save("/workspace/repos_dev/VQVAE_RL/figures/minigrid_abstraction/grd_visits.png")
        wandb.log({"Images/grd_visits": wandb.Image(img)})
        img_buffer.close()
        plt.close(fig_grd_visits)

        self.grd_visits = np.zeros((self.env.height, self.env.width, 4))

    def _create_optimizers(self, config):

        if isinstance(config.lr_ground_Q, str) and config.lr_ground_Q.startswith("lin"):
            self.lr_scheduler_ground_Q = linear_schedule(float(config.lr_ground_Q.split("_")[1]))
            lr_ground_Q = self.lr_scheduler_ground_Q(self._current_progress_remaining)

        elif isinstance(config.lr_ground_Q, float):
            lr_ground_Q = config.lr_ground_Q

        # self.ground_Q_optimizer = optim.RMSprop(
        #     self.ground_Q_net.parameters(), lr=lr_ground_Q, alpha=0.95, momentum=0, eps=0.01
        # )
        # self.abstract_V_optimizer = optim.RMSprop(
        #     self.ground_Q_net.parameters(), lr=lr_abstract_V, alpha=0.95, momentum=0.95, eps=0.01
        # )

        if hasattr(self, "ground_Q"):
            self.ground_Q_optimizer = optim.Adam(self.ground_Q.parameters(), lr=lr_ground_Q)

    def act(self, state):
        self._update_current_progress_remaining(self.timesteps_done, self.total_timesteps)
        self.exploration_rate = self.exploration_scheduler(self._current_progress_remaining)
        with torch.no_grad():
            state = T.ToTensor()(state).float().unsqueeze(0).to(self.device)

            if random.random() > self.exploration_rate:
                action = self.ground_Q(state)[0].max(1)[1].item()
            else:
                action = random.randrange(self.n_actions)

        # [maintain eligibility]
        # found = False
        # for x in self.abstract_eligibllity_list:
        #     if x[0] == state:
        #         x[1] = 1
        #         found = True
        # if not found:
        #     self.abstract_eligibllity_list.append((state, 1))

        self.timesteps_done += 1
        return action

    def act_table(self, info):
        self._update_current_progress_remaining(self.timesteps_done, self.total_timesteps)
        self.exploration_rate = self.exploration_scheduler(self._current_progress_remaining)
        agent_pos = info["agent_pos2"]
        agent_dir = info["agent_dir2"]

        if random.random() > self.exploration_rate:
            q_values = np.array(
                [
                    self.grd_Q_table[agent_pos[1] - 1, agent_pos[0] - 1, agent_dir, a]
                    for a in range(self.n_actions)
                ]
            )
            action = np.random.choice(np.flatnonzero(q_values == q_values.max()))

            assert action in range(self.n_actions)
        else:
            action = random.randrange(self.n_actions)

        # [maintain eligibility]
        # found = False
        # for x in self.abstract_eligibllity_list:
        #     if x[0] == state:
        #         x[1] = 1
        #         found = True
        # if not found:
        #     self.abstract_eligibllity_list.append((state, 1))

        self.timesteps_done += 1
        return action

    def update_grd_visits(self, info):
        agent_pos = info["agent_pos2"]
        agent_dir = info["agent_dir2"]
        self.grd_visits[agent_pos[1], agent_pos[0], agent_dir] += 1

    def get_abstract_state_idx(self, agent_pos):
        """
        currently onlu support Minigrid Empty-style of envs
        """
        # H, W, C = self.env.observation_space.shape
        # real_H = H - 2
        # real_W = W - 2
        # size = real_H

        # bin = math.sqrt(self.n_cluster)
        # # check if bin is an interger
        # assert bin == int(bin)
        # ceil = math.ceil(size / self.n_clusters)
        # floor = math.floor(size / self.n_clusters)
        # a = Symbol("a")
        # b = Symbol("b")
        # result = solve((a * floor + b * ceil - 13, a + b - 3), a, b)
        # abstract_ticks = [1]
        # for i in range(a):
        #     abstract_ticks.append((i + 1) * floor)
        # for i in range(b):
        #     abstract_ticks.append((i + 1) * ceil)
        n_bin = math.sqrt(self.n_clusters)
        # check if bin is an interger
        assert n_bin == int(n_bin)
        n_bin = int(n_bin)
        assert len(self.abs_ticks) == n_bin

        for i in range(len(self.abs_ticks)):
            if agent_pos[1] <= self.abs_ticks[i]:
                abstract_state_idx = i * n_bin
                break
        for i in range(len(self.abs_ticks)):
            if agent_pos[0] <= self.abs_ticks[i]:
                abstract_state_idx += i
                break
        return abstract_state_idx

    def get_abstract_value(self, agent_pos):
        abstract_state_idx = self.get_abstract_state_idx(agent_pos)
        abstract_value = self.abstract_V_array[abstract_state_idx]
        return abstract_state_idx, abstract_value

    def update_grdQ_shaping(
        self, state, action, next_state, reward, terminated, abs_value_l, abs_value_next_l
    ):
        if hasattr(self, "lr_scheduler_ground_Q"):
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_scheduler_ground_Q(self._current_progress_remaining),
            )

        # [Update ground Q network]
        grd_q, encoded = self.ground_Q(state)
        grd_q = grd_q.gather(1, action)

        with torch.no_grad():

            # Vanilla DQN
            grd_q_next, encoded_next = self.ground_Q_target(next_state)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)

            # Double DQN
            # action_argmax_target = self.ground_target_Q_net(next_state_batch).argmax(
            #     dim=1, keepdim=True
            # )
            # ground_next_max_Q = self.ground_Q_net(next_state_batch).gather(1, action_argmax_target)

            # Compute ground target Q value

            # shaping = []
            # abs_value_l = []
            # abs_value_next_l = []
            # abs_indices = []
            # abs_indices_next = []
            # # shaping = self.gamma * self.abstract_V_table[quantized_next] - self.abstract_V_table[quantized]
            # for i, info_i in enumerate(info):
            #     agent_pos = info_i["agent_pos1"]
            #     agent_pos_next = info_i["agent_pos2"]
            #     abs_idx, abs_value = self.get_abstract_value(agent_pos)
            #     abs_idx_next, abs_value_next = self.get_abstract_value(agent_pos_next)
            #     # shaping.append(abs_value_next - abs_value)
            #     abs_indices.append(abs_idx)
            #     abs_indices_next.append(abs_idx_next)
            #     abs_value_l.append(abs_value)
            #     abs_value_next_l.append(abs_value_next)
            #     delta = self.gamma * abs_value_next - abs_value
            #     shaping.append(delta)
            #     # if delta != 0:
            #     #     print("####################delta!=0")
            # # abs_value_l = torch.tensor(abs_value_l)
            # # abs_value_next_l = torch.tensor(abs_value_next_l)
            # # shaping = (abs_value_next_l - abs_value_l).unsqueeze(1).to(self.device)
            # shaping = torch.tensor(shaping).unsqueeze(1).to(self.device)

            abs_value_l = torch.tensor(abs_value_l).unsqueeze(1).to(self.device)
            abs_value_next_l = torch.tensor(abs_value_next_l).unsqueeze(1).to(self.device)
            shaping = self.gamma * abs_value_next_l - abs_value_l

            grd_q_target = (
                reward
                + self.omega * shaping * (1 - terminated.float())
                + self.gamma * grd_q_next_max * (1 - terminated.float())
            ).float()

        criterion = nn.SmoothL1Loss()
        ground_td_error = criterion(grd_q, grd_q_target)

        # [Update abstract V network]
        # mask_ = ~torch.tensor(
        #     [
        #         [torch.equal(a_state, next_a_state)]
        #         for a_state, next_a_state in zip(quantized, quantized_next)
        #     ]
        # ).to(self.device)
        # abs_v *= mask_
        # abs_v_target *= mask_

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        ground_td_error.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()

        self.training_info["ground_Q_error"].append(ground_td_error.item())
        self.training_info["avg_shaping"].append(torch.mean(shaping).item())

    def update_grdQ_table_shaping(
        self,
        action: Tensor,
        reward: Tensor,
        terminated: Tensor,
        info: tuple,
        abs_value_l: list,
        abs_value_next_l: list,
        use_shaping: bool,
    ):
        action = action.squeeze().tolist()
        reward = reward.squeeze().tolist()
        terminated = terminated.squeeze().tolist()

        # [Update ground Q network]
        delta_l = []
        shaping_l = []
        for i, info_i in enumerate(info):
            agent_pos1 = info_i["agent_pos1"]
            agent_dir1 = info_i["agent_dir1"]
            agent_pos2 = info_i["agent_pos2"]
            agent_dir2 = info_i["agent_dir2"]
            q = self.grd_Q_table[agent_pos1[1] - 1, agent_pos1[0] - 1, agent_dir1, action[i]]
            max_q_prime = self.grd_Q_table[
                agent_pos2[1] - 1, agent_pos2[0] - 1, agent_dir2, :
            ].max()
            if use_shaping:
                # shaping = self.gamma * abs_value_next_l[i] - abs_value_l[i]
                shaping = abs_value_next_l[i] - abs_value_l[i]
            else:
                shaping = 0
            q_target = (
                reward[i]
                + self.omega * shaping * (1 - terminated[i])
                + self.gamma * max_q_prime * (1 - terminated[i])
            )
            delta = q_target - q
            self.grd_Q_table[agent_pos1[1] - 1, agent_pos1[0] - 1, agent_dir1, action[i]] += (
                self.lr_grd_Q * delta
            )
            delta_l.append(delta)
            shaping_l.append(shaping)

        self.training_info["ground_Q_error"].append(mean(delta_l))
        self.training_info["avg_shaping"].append(mean(shaping_l))

    def update_absV(
        self,
        abs_indices: list,
        abs_indices_next: list,
        abs_value_l: list,
        abs_value_next_l: list,
        reward: Tensor,
        terminated: Tensor,
    ):
        # target = reward + self.gamma * abs_value_next_l
        # delta = target - abs_value_l

        # reward = reward / 10

        reward = reward.squeeze().tolist()
        terminated = terminated.float().squeeze().tolist()

        delta_l = []
        for i, (abs_idx, abs_idx_next) in enumerate(zip(abs_indices, abs_indices_next)):
            if abs_idx == abs_idx_next and reward[i] == 0:
                delta_l.append(0)
            else:
                target = reward[i] + self.gamma * abs_value_next_l[i] * (1 - terminated[i])
                delta = target - abs_value_l[i]
                # if delta <= 0:
                #     delta_l.append(0)
                #     continue
                self.abstract_V_array[abs_idx] += self.lr_abs_V * delta
                delta_l.append(delta)

        self.training_info["abstract_V_error"].append(mean(delta_l))

    def update(self):
        if self.timesteps_done == self.init_steps:
            print("Init steps done")

        if (
            self.timesteps_done == self.init_steps
            or self.timesteps_done % self.ground_learn_every == 0
        ):
            abs_update_step = 10
            grd_update_step = 1
            for _ in range(abs_update_step):
                state, action, next_state, reward, terminated, info = self.memory.sample(
                    self.batch_size
                )
                # [data augmentation]
                # state = self.aug(state)
                # next_state = self.aug(next_state)

                # [extract abstract information]
                (
                    abs_indices,
                    abs_indices_next,
                    abs_value_l,
                    abs_value_next_l,
                ) = self.get_abs_indices_values(info)

                # [update abstract_V]
                abs_value_l, abs_value_next_l = self.update_absV(
                    abs_indices, abs_indices_next, abs_value_l, abs_value_next_l, reward
                )

                # [update ground_Q with reward shaping]
                if grd_update_step > 0:
                    self.update_grdQ_shaping(
                        state, action, next_state, reward, terminated, abs_value_l, abs_value_next_l
                    )
                    grd_update_step -= 1

                # [purely update ground Q]
                # self.update_grdQ_pure(state, action, next_state, reward, terminated, info)

        if self.timesteps_done % self.ground_sync_every == 0:
            soft_sync_params(
                self.ground_Q.parameters(),
                self.ground_Q_target.parameters(),
                self.ground_tau,
            )
            # soft_sync_params(
            #     self.ground_Q.encoder.parameters(),
            #     self.ground_Q_target.encoder.parameters(),
            #     self.encoder_tau,
            # )
            # soft_sync_params(
            #     self.ground_Q.critic.parameters(),
            #     self.ground_Q_target.critic.parameters(),
            #     self.ground_tau,
            # )

        if self.timesteps_done % self.save_model_every == 0:
            pass

        if self.timesteps_done % self.reset_training_info_every == 0:
            self.log_training_info(wandb_log=True)
            self.reset_training_info()

    def update_table(self, use_shaping: bool):
        if self.timesteps_done == self.init_steps:
            print("Init steps done")

        if (
            self.timesteps_done == self.init_steps
            or self.timesteps_done % self.ground_learn_every == 0
        ):
            abs_update_step = 1
            grd_update_step = 1
            for _ in range(abs_update_step):
                state, action, next_state, reward, terminated, info = self.memory.sample(
                    self.batch_size
                )
                # [data augmentation]
                # state = self.aug(state)
                # next_state = self.aug(next_state)

                # [extract abstract information]
                (
                    abs_indices,
                    abs_indices_next,
                    abs_value_l,
                    abs_value_next_l,
                ) = self.get_abs_indices_values(info)

                # [update abstract_V]
                self.update_absV(
                    abs_indices,
                    abs_indices_next,
                    abs_value_l,
                    abs_value_next_l,
                    reward,
                    terminated,
                )

                # [update ground_Q with reward shaping, purely update ground Q with use_shaping=False]
                if grd_update_step > 0:
                    # state, action, next_state, reward, terminated, info = self.memory.sample(
                    #     self.batch_size
                    # )

                    # (
                    #     _,
                    #     _,
                    #     abs_value_l,
                    #     abs_value_next_l,
                    # ) = self.get_abs_indices_values(info)

                    self.update_grdQ_table_shaping(
                        action,
                        reward,
                        terminated,
                        info,
                        abs_value_l,
                        abs_value_next_l,
                        use_shaping=use_shaping,
                    )
                    grd_update_step -= 1

        if self.timesteps_done % self.save_model_every == 0:
            pass

        if self.timesteps_done % self.reset_training_info_every == 0:
            self.log_training_info(wandb_log=True)
            self.reset_training_info()

    def get_abs_indices_values(self, info: tuple):
        abs_value_l = []
        abs_value_next_l = []
        abs_indices = []
        abs_indices_next = []
        # shaping = self.gamma * self.abstract_V_table[quantized_next] - self.abstract_V_table[quantized]
        for i, info_i in enumerate(info):
            agent_pos = info_i["agent_pos1"]
            agent_pos_next = info_i["agent_pos2"]
            abs_idx, abs_value = self.get_abstract_value(agent_pos)
            abs_idx_next, abs_value_next = self.get_abstract_value(agent_pos_next)
            # shaping.append(abs_value_next - abs_value)
            abs_indices.append(abs_idx)
            abs_indices_next.append(abs_idx_next)
            abs_value_l.append(abs_value)
            abs_value_next_l.append(abs_value_next)

        return abs_indices, abs_indices_next, abs_value_l, abs_value_next_l

    def update_grdQ_pure(self, state, action, next_state, reward, terminated, info):
        if hasattr(self, "lr_scheduler_ground_Q"):
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_scheduler_ground_Q(self._current_progress_remaining),
            )

        # [Update ground Q network]
        grd_q, encoded = self.ground_Q(state)
        grd_q = grd_q.gather(1, action)

        with torch.no_grad():

            # Vanilla DQN
            grd_q_next, encoded_next = self.ground_Q_target(next_state)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)

            # Double DQN
            # action_argmax_target = self.ground_target_Q_net(next_state_batch).argmax(
            #     dim=1, keepdim=True
            # )
            # ground_next_max_Q = self.ground_Q_net(next_state_batch).gather(1, action_argmax_target)

            # Compute ground target Q value
            grd_q_target = (reward + self.gamma * grd_q_next_max * (1 - terminated.float())).float()

        criterion = nn.SmoothL1Loss()
        ground_td_error = criterion(grd_q, grd_q_target)

        # [Update abstract V network]
        # mask_ = ~torch.tensor(
        #     [
        #         [torch.equal(a_state, next_a_state)]
        #         for a_state, next_a_state in zip(quantized, quantized_next)
        #     ]
        # ).to(self.device)
        # abs_v *= mask_
        # abs_v_target *= mask_

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        ground_td_error.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()

        self.training_info["ground_Q_error"].append(ground_td_error.item())


class HDQN_KMeans_VAE(nn.Module):
    def __init__(
        self,
        config,
        env: gym.Env,
        # hidden_dims: List = [32, 64],
        beta: float = 0.25,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = int(time.time())
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.set_hparams(config)
        self.env = env
        self.n_actions = env.action_space.n
        self.learn_with_ae = config.learn_with_ae
        self.init_clustering = config.init_clustering

        self.kmeans = Batch_KMeans(
            n_clusters=config.n_clusters, embedding_dim=config.embedding_dim, device=self.device
        ).to(self.device)

        self.ground_Q = DQN_Repara(
            env.observation_space,
            env.action_space,
            config.embedding_dim,
            hidden_dims=config.hidden_dims,
            # embedding_dim=config.latent_dim,
        ).to(self.device)

        self.ground_Q_target = DQN_Repara(
            env.observation_space,
            env.action_space,
            config.embedding_dim,
            hidden_dims=config.hidden_dims,
            # embedding_dim=config.latent_dim,
        ).to(self.device)
        self.ground_Q_target.load_state_dict(self.ground_Q.state_dict())
        self.ground_Q_target.train()

        self.decoder = Decoder(
            in_dim=config.embedding_dim,
            out_channels=env.observation_space.shape[-1],
            shape_conv_output=self.ground_Q.conv_block.shape_conv_output,
            hidden_dims=config.hidden_dims,
        ).to(self.device)

        self.abstract_V = V_MLP(config.embedding_dim, flatten=False).to(self.device)
        self.abstract_V_target = V_MLP(config.embedding_dim, flatten=False).to(self.device)
        self.abstract_V_target.load_state_dict(self.abstract_V.state_dict())
        self.abstract_V_target.train()

        self.aug = RandomShiftsAug(pad=4)

        # for coding test
        # summary(self.ground_Q.encoder, (4, 84, 84))
        # summary(self.decoder, (32,))

        self.outputs = dict()
        # self.apply(weight_init)

        # Initialize experience replay buffer
        self.memory = ReplayMemory(self.size_replay_memory, self.device)
        # self.Transition = namedtuple(
        #     "Transition", ("state", "action", "next_state", "reward", "done")
        # )
        self.exploration_scheduler = get_linear_fn(
            config.exploration_initial_eps,
            config.exploration_final_eps,
            config.exploration_fraction,
        )

        self.timesteps_done = 0
        self.episodes_done = 0
        self.n_call_train = 0
        self._current_progress_remaining = 1.0
        self.to_buffer = False  # for func maybe_buffer_recent_states

        self._create_optimizers(config)
        self.reset_training_info()
        self.train()

    def train(self, training=True):
        self.training = training
        self.ground_Q.train(training)
        self.abstract_V.train(training)
        self.decoder.train(training)

    def reset_training_info(self):
        self.training_info = {
            "ground_Q_error": [],
            "abstract_V_error": [],
            "kld": [],
            "recon_loss": [],
            "commitment_loss": [],
        }

    def log_training_info(self, wandb_log=True):
        if wandb_log:
            metrics = {
                "loss/ground_Q_error": mean(self.training_info["ground_Q_error"]),
                "loss/abstract_V_error": mean(self.training_info["abstract_V_error"]),
                "loss/kld_loss": mean(self.training_info["kld"]),
                "loss/recon_loss": mean(self.training_info["recon_loss"]),
                "loss/vae_loss": mean(self.training_info["kld"])
                + mean(self.training_info["recon_loss"]),
                "loss/commitment_loss": mean(self.training_info["commitment_loss"]),
                "train/exploration_rate": self.exploration_rate,
                "train/current_progress_remaining": self._current_progress_remaining,
                "lr/lr_ground_Q_optimizer": self.ground_Q_optimizer.param_groups[0]["lr"],
                "lr/lr_abstract_V_optimizer": self.abstract_V_optimizer.param_groups[0]["lr"],
            }
            wandb.log(metrics)

    def load_states_from_memory(self, unique=True):
        transitions = random.sample(self.memory.memory, len(self.memory))
        batch = self.memory.Transition(*zip(*transitions))
        state_batch = np.stack(batch.state, axis=0).transpose(0, 3, 1, 2)
        if unique:
            state_batch = np.unique(state_batch, axis=0)
        state_batch = torch.from_numpy(state_batch).contiguous().float().to(self.device)

        # Use when states are cached as Tensor
        # batch = self.memory.sample(batch_size=len(self.memory))
        # state_batch = torch.cat(batch.next_state)
        # state_batch = torch.unique(state_batch, dim=0).float().to(self.device)

        return state_batch

    def triangulation_for_triheatmap(self, M, N):
        # M: number of columns, N: number of rows
        xv, yv = np.meshgrid(
            np.arange(-0.5, M), np.arange(-0.5, N)
        )  # vertices of the little squares
        xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
        x = np.concatenate([xv.ravel(), xc.ravel()])
        y = np.concatenate([yv.ravel(), yc.ravel()])
        cstart = (M + 1) * (N + 1)  # indices of the centers
        print(cstart)

        trianglesN = [
            (i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
            for j in range(N)
            for i in range(M)
        ]
        trianglesE = [
            (i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
            for j in range(N)
            for i in range(M)
        ]
        trianglesS = [
            (i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
            for j in range(N)
            for i in range(M)
        ]
        trianglesW = [
            (i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
            for j in range(N)
            for i in range(M)
        ]
        return [
            Triangulation(x, y, triangles)
            for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]
        ]

    def cluster_visualize_memory(self):
        # trainsitions_in_memory = self.memory.sample(batch_size=len(self.memory))
        # arrays = np.stack(trainsitions_in_memory.state, axis=0)
        # arrays_unique = np.unique(arrays, axis=0)
        # tensors_unique = (
        #     torch.from_numpy(arrays_unique.transpose(0, 3, 1, 2))
        #     .contiguous()
        #     .float()
        #     .to(self.device)
        # )

        batch = self.memory.sample(batch_size=len(self.memory))
        state_batch = torch.cat(batch.state)
        state_batch = state_batch.cpu().numpy()
        unique_array = np.unique(state_batch, axis=0)
        tensors_unique = torch.from_numpy(unique_array).float().to(self.device)

        with torch.no_grad():
            embeddings = self.ground_Q.encoder(tensors_unique)[0]
            cluster_indices = self.kmeans.assign_clusters(embeddings)
        # states_in_memory = self.load_states_from_memory()
        # arrays = torch.unique(states_in_memory, dim=0)
        # arrays = arrays.cpu().numpy().transpose(0, 3, 1, 2)
        batch, channels, width, height = tensors_unique.shape
        list_of_agent_pos_dir = []
        # clustersN = np.empty(shape=(height, width))
        # clustersS = np.empty(shape=(height, width))
        # clustersW = np.empty(shape=(height, width))
        # clustersE = np.empty(shape=(height, width))
        clustersN = np.full(shape=(height, width), fill_value=4)
        clustersS = np.full(shape=(height, width), fill_value=4)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=4)
        clustersE = np.full(shape=(height, width), fill_value=4)
        for idx, array in enumerate(tensors_unique):
            # break
            for i in range(width):
                for j in range(height):
                    type_idx, color_idx, state = array[:, i, j]
                    if type_idx == 10:  # if type is agent
                        assert 0 <= state < 4
                        if state == 3:
                            clustersN[j, i] = 0
                        elif state == 2:
                            clustersW[j, i] = 1
                        elif state == 1:
                            clustersS[j, i] = 2
                        elif state == 0:
                            clustersE[j, i] = 3
                    # agent_pos = (i, j)
                    # agent_dir = state
                    # list_of_agent_pos_dir.append((agent_pos, agent_dir))
        values = [clustersN, clustersE, clustersS, clustersW]
        triangulations = self.triangulation_for_triheatmap(width, height)
        fig, ax = plt.subplots()
        vmax = 4
        vmin = 0
        imgs = [
            ax.tripcolor(
                t,
                np.ravel(val),
                vmin=vmin,
                vmax=vmax,
                cmap="gist_ncar",
                ec="black",
            )
            for t, val in zip(triangulations, values)
        ]
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    def cluster_visualize_memory2(self):
        # take out all unique states from replay buffer and visualize their clusters
        # This approach might not cover all the states in the environment

        batch = self.memory.sample(batch_size=len(self.memory))
        # ===When states are cached as channel-first tensors===
        state_batch = torch.cat(batch.next_state)
        state_batch = state_batch.cpu().numpy()
        unique_array, indices = np.unique(state_batch, return_index=True, axis=0)
        unique_info_list = [batch.info[i] for i in indices]
        unique_tensor = torch.from_numpy(unique_array).float().to(self.device)

        # ===When states are cached as numpy arrays===
        # state_batch = np.stack(batch.next_state, axis=0)
        # unique_array, indices = np.unique(state_batch, return_index=True, axis=0)
        # unique_info_list = [batch.info[i] for i in indices]
        # unique_tensor = (
        #     torch.from_numpy(unique_array.transpose(0, 3, 1, 2))
        #     .contiguous()
        #     .float()
        #     .to(self.device)
        # )

        with torch.no_grad():
            embeddings = self.ground_Q.encoder(unique_tensor)[0]
            cluster_indices = self.kmeans.assign_clusters(embeddings)
        # states_in_memory = self.load_states_from_memory()
        # arrays = torch.unique(states_in_memory, dim=0)
        # arrays = arrays.cpu().numpy().transpose(0, 3, 1, 2)
        width = self.env.width
        height = self.env.height
        num_cluster = self.kmeans.n_clusters
        # clustersN = np.empty(shape=(height, width))
        # clustersS = np.empty(shape=(height, width))
        # clustersW = np.empty(shape=(height, width))
        # clustersE = np.empty(shape=(height, width))
        clustersN = np.full(shape=(height, width), fill_value=num_cluster)
        clustersS = np.full(shape=(height, width), fill_value=num_cluster)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=num_cluster)
        clustersE = np.full(shape=(height, width), fill_value=num_cluster)

        print(cluster_indices.shape, len(unique_info_list))
        n, w, s, e = 0, 0, 0, 0
        for cluster_idx, info in zip(cluster_indices, unique_info_list):
            agent_pos = info["agent_pos"]
            agent_dir = info["agent_dir"]
            assert 0 <= agent_dir < 4
            if agent_dir == 3:
                n += 1
                clustersN[agent_pos[1], agent_pos[0]] = cluster_idx
            if agent_dir == 2:
                w += 1
                clustersW[agent_pos[1], agent_pos[0]] = cluster_idx
            if agent_dir == 1:
                s += 1
                clustersS[agent_pos[1], agent_pos[0]] = cluster_idx
            if agent_dir == 0:
                e += 1
                clustersE[agent_pos[1], agent_pos[0]] = cluster_idx
        print(n, w, s, e)
        values = [clustersN, clustersE, clustersS, clustersW]
        triangulations = self.triangulation_for_triheatmap(width, height)
        fig, ax = plt.subplots()
        vmax = num_cluster
        vmin = 0
        imgs = [
            ax.tripcolor(t, np.ravel(val), vmin=vmin, vmax=vmax, cmap="gist_ncar", ec="black")
            for t, val in zip(triangulations, values)
        ]
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    # def cluster_visualize_memory3(self):
    #         # take out all unique states from replay buffer and visualize their clusters
    #         # This approach might not cover all the states in the environment
    #         trainsitions_in_memory = self.memory.sample(batch_size=len(self.memory))
    #         arrays = np.stack(trainsitions_in_memory.state, axis=0)
    #         infos = trainsitions_in_memory.info
    #         arrays_unique, indices = np.unique(arrays, return_index=True, axis=0)
    #         infos = infos[indices]
    #         tensors_unique = (
    #             torch.from_numpy(arrays_unique.transpose(0, 3, 1, 2))
    #             .contiguous()
    #             .float()
    #             .to(self.device)
    #         )
    #         with torch.no_grad():
    #             embeddings = self.ground_Q.encoder(tensors_unique)[0]
    #             cluster_indices = self.kmeans.assign_clusters(embeddings)
    #         # states_in_memory = self.load_states_from_memory()
    #         # arrays = torch.unique(states_in_memory, dim=0)
    #         # arrays = arrays.cpu().numpy().transpose(0, 3, 1, 2)
    #         width = self.env.width
    #         height = self.env.height
    #         # clustersN = np.empty(shape=(height, width))
    #         # clustersS = np.empty(shape=(height, width))
    #         # clustersW = np.empty(shape=(height, width))
    #         # clustersE = np.empty(shape=(height, width))
    #         clustersN = np.full(shape=(height, width), fill_value=4)
    #         clustersS = np.full(shape=(height, width), fill_value=4)
    #         # clustersS = np.random.randint(0, 4, size=(height, width))
    #         clustersW = np.full(shape=(height, width), fill_value=4)
    #         clustersE = np.full(shape=(height, width), fill_value=4)

    #         for i in range(width):
    #             for j in range(height):
    #                 pass

    #         for cluster_idx, info in zip(cluster_indices, infos):
    #             agent_pos = info["agent_pos"]
    #             agent_dir = info["agent_dir"]
    #             if agent_dir == 3:
    #                 clustersN[agent_pos[1], agent_pos[0]] = cluster_idx
    #             if agent_dir == 2:
    #                 clustersW[agent_pos[1], agent_pos[0]] = cluster_idx
    #             if agent_dir == 1:
    #                 clustersS[agent_pos[1], agent_pos[0]] = cluster_idx
    #             if agent_dir == 0:
    #                 clustersE[agent_pos[1], agent_pos[0]] = cluster_idx

    #         values = [clustersN, clustersE, clustersS, clustersW]
    #         triangulations = self.triangulation_for_triheatmap(width, height)
    #         fig, ax = plt.subplots()
    #         vmax = 4
    #         vmin = 0
    #         imgs = [
    #             ax.tripcolor(t, np.ravel(val), vmin=vmin, vmax=vmax, cmap="gist_ncar", ec="black")
    #             for t, val in zip(triangulations, values)
    #         ]
    #         ax.invert_yaxis()
    #         plt.tight_layout()
    #         plt.show()

    def set_hparams(self, config):
        # Hyperparameters
        # self.total_episodes = config.total_episodes
        self.total_timesteps = config.total_timesteps
        self.init_steps = config.init_steps  # min. experiences before training
        self.batch_size = config.batch_size
        self.size_replay_memory = config.size_replay_memory
        self.gamma = config.gamma
        self.omega = config.omega
        self.ground_tau = config.ground_tau
        self.encoder_tau = config.encoder_tau
        self.abstract_tau = config.abstract_tau

        self.ground_learn_every = config.ground_learn_every
        self.ground_sync_every = config.ground_sync_every
        self.ground_gradient_steps = config.ground_gradient_steps
        self.abstract_learn_every = config.abstract_learn_every
        self.abstract_sync_every = config.abstract_sync_every
        self.abstract_gradient_steps = config.abstract_gradient_steps

        self.validate_every = config.validate_every
        self.save_model_every = config.save_model_every
        self.reset_training_info_every = config.reset_training_info_every
        self.save_recon_every = config.save_recon_every
        self.buffer_recent_states_every = config.buffer_recent_states_every

        self.clip_grad = config.clip_grad

    def _create_optimizers(self, config):

        if isinstance(config.lr_ground_Q, str) and config.lr_ground_Q.startswith("lin"):
            self.lr_scheduler_ground_Q = linear_schedule(float(config.lr_ground_Q.split("_")[1]))
            lr_ground_Q = self.lr_scheduler_ground_Q(self._current_progress_remaining)

        elif isinstance(config.lr_ground_Q, float):
            lr_ground_Q = config.lr_ground_Q

        if isinstance(config.lr_abstract_V, str) and config.lr_abstract_V.startswith("lin"):
            self.lr_scheduler_abstract_V = linear_schedule(
                float(config.lr_abstract_V.split("_")[1])
            )
            lr_abstract_V = self.lr_scheduler_abstract_V(self._current_progress_remaining)

        elif isinstance(config.lr_abstract_V, float):
            lr_abstract_V = config.lr_abstract_V

        # self.ground_Q_optimizer = optim.RMSprop(
        #     self.ground_Q_net.parameters(), lr=lr_ground_Q, alpha=0.95, momentum=0, eps=0.01
        # )
        # self.abstract_V_optimizer = optim.RMSprop(
        #     self.ground_Q_net.parameters(), lr=lr_abstract_V, alpha=0.95, momentum=0.95, eps=0.01
        # )

        if hasattr(self, "ground_Q"):
            self.ground_Q_optimizer = optim.Adam(self.ground_Q.parameters(), lr=lr_ground_Q)
        if hasattr(self, "abstract_V"):
            self.abstract_V_optimizer = optim.Adam(self.abstract_V.parameters(), lr=lr_abstract_V)

        if hasattr(self, "decoder"):
            self.encoder_optimizer = optim.Adam(
                self.ground_Q.encoder.parameters(), lr=config.lr_encoder
            )
            self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=config.lr_decoder)

    def _update_current_progress_remaining(self, timesteps_done, total_timesteps):
        # self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)
        finished_time_steps_after_init = timesteps_done - self.init_steps
        if finished_time_steps_after_init < 0:
            self._current_progress_remaining = 1.0
        else:
            self._current_progress_remaining = (
                1.0 - finished_time_steps_after_init / total_timesteps
            )

    @torch.no_grad()
    def maybe_buffer_recent_states(self, state, buffer_length=30):
        if self.timesteps_done % self.buffer_recent_states_every == 0 and self.timesteps_done > 0:
            self.recent_states = []
            self.to_buffer = True
        if self.to_buffer:
            self.recent_states.append(state)
            if len(self.recent_states) == buffer_length:
                print("##Check how good abstraction is##")
                # convert self.recent_states to torch tensor
                self.recent_states = (
                    (torch.tensor(self.recent_states).permute(0, 3, 1, 2).contiguous())
                    .float()
                    .to(self.device)
                )
                grd_q, encoded = self.ground_Q(self.recent_states)
                quantized, vq_loss, vq_entrophy, encodings = self.vector_quantizer(encoded)
                abs_v = self.abstract_V(quantized)
                (clusters, inverse_indice, counts) = torch.unique(
                    quantized,
                    return_inverse=True,
                    return_counts=True,
                    dim=0,
                )
                print("number of clusters:\n", len(clusters))
                print("inverse_indice:\n", inverse_indice.tolist())
                print("counts:\n", counts.tolist())
                # log n_abstract_states by wandb
                wandb.log({"abstraction/n_clusters_in_buffer": len(clusters)})
                wandb.log({"abstraction/value_difference": torch.abs(abs_v - grd_q).mean().item()})
                self.to_buffer = False

    def cache(self, state, action, next_state, reward, terminated, info):
        """Add the experience to memory"""
        # if state_type == "rgb":
        #     state = T.ToTensor()(state).float().unsqueeze(0)
        #     next_state = T.ToTensor()(next_state).float().unsqueeze(0)
        # else:
        #     state = torch.from_numpy(state.transpose((2, 0, 1))).contiguous().float().unsqueeze(0)
        #     next_state = (
        #         torch.from_numpy(next_state.transpose((2, 0, 1))).contiguous().float().unsqueeze(0)
        #     )
        # if state_type == "img":
        #     state = state / 255.0
        #     next_state = next_state / 255.0
        # action = torch.tensor([action]).unsqueeze(0)
        # reward = torch.tensor([reward]).unsqueeze(0)
        # terminated = torch.tensor([terminated]).unsqueeze(0)

        self.memory.push(state, action, next_state, reward, terminated, copy.deepcopy(info))

    def act(self, state):
        self._update_current_progress_remaining(self.timesteps_done, self.total_timesteps)
        self.exploration_rate = self.exploration_scheduler(self._current_progress_remaining)
        with torch.no_grad():
            state = T.ToTensor()(state).float().unsqueeze(0).to(self.device)

            if random.random() > self.exploration_rate:
                action = self.ground_Q(state)[0].max(1)[1].item()
            else:
                action = random.randrange(self.n_actions)

        self.timesteps_done += 1
        return action

    def update(self):
        n_steps_update = 1

        if self.timesteps_done == self.init_steps:
            states_in_memory = self.load_states_from_memory(unique=False)
            # states_in_memory_copy = states_in_memory.clone().detach()
            if self.learn_with_ae:
                self.pretrain_ae(states_in_memory, batch_size=1, epochs=30)
            if self.init_clustering:
                embedding_in_memory = self.ground_Q.encoder(states_in_memory)[0]
                self.kmeans.init_cluster(embedding_in_memory)
                # self.cluster_visualize_memory2()
            n_steps_update = 10

            # for _ in range(int(self.init_steps / 100)):
        if (
            self.timesteps_done == self.init_steps
            or self.timesteps_done % self.ground_learn_every == 0
        ):
            for _ in range(n_steps_update):
                state, action, next_state, reward, terminated = self.memory.sample(self.batch_size)

                # [data augmentation]
                state = self.aug(state)
                next_state = self.aug(next_state)

                # [update ground_Q with reward shaping]
                # quantized, quantized_next = self.update_grdQ_shaping(
                #     state, action, next_state, reward, terminated, use_shaping=True
                # )
                # [update abstract_V]
                # self.update_absV(quantized, quantized_next, reward, terminated)

                # [update vae, centroids of kmeans, maybe commitment loss]
                # self.train_vae_kmeans(
                #     state=state,
                #     next_state=None,
                #     update_centriods=True,
                #     kld_beta_vae=3,
                #     optimize_commitment=False,  # Ture or False needs further experiments
                #     save_recon_every=None,
                # )
                # [Or, use this: all in one]
                self.update_grdQ_absV(
                    state, action, next_state, reward, terminated, update_absV=True
                )

        if self.timesteps_done % self.ground_sync_every == 0:
            soft_sync_params(
                self.ground_Q.encoder.parameters(),
                self.ground_Q_target.encoder.parameters(),
                self.encoder_tau,
            )
            soft_sync_params(
                self.ground_Q.Q_linear.parameters(),
                self.ground_Q_target.Q_linear.parameters(),
                self.ground_tau,
            )

        if self.timesteps_done % self.abstract_sync_every == 0:
            soft_sync_params(
                self.abstract_V.parameters(),
                self.abstract_V_target.parameters(),
                self.abstract_tau,
            )

        if self.timesteps_done % self.save_model_every == 0:
            pass

        if self.timesteps_done % self.reset_training_info_every == 0:
            self.log_training_info(wandb_log=True)
            self.reset_training_info()

    def vae_loss_function(self, recon_x, x, mu, logvar, beta=3):
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld, recon_loss, kld

    def train_ae_kmeans(
        self, state_batch, next_state_batch, train_ae=True, train_kmeans=True, save_recon_every=1000
    ):
        if next_state_batch is not None:
            state_batch = torch.cat((state_batch, next_state_batch), dim=0)
            # shuffle the batch by the first dimension
            state_batch = state_batch[torch.randperm(state_batch.size(0)), :]

        recon_loss = torch.tensor(0.0).to(self.device)
        commitment_loss = torch.tensor(0.0).to(self.device)

        if train_ae:
            # Optimize Classic AE
            encoded = self.ground_Q.encoder(state_batch)
            recon = self.decoder(encoded)
            recon_loss = F.mse_loss(recon, state_batch)
            # if self.decoder.n_forward_call % save_recon_every == 0:
            #     stacked = torch.cat((recon[:7, :1], state_batch[:1, :1]), dim=0)
            #     wandb_log_image(stacked)

        if train_kmeans:
            quantized, cluster_indices = self.kmeans.assign_centroid(encoded, update_centroid=True)
            commitment_loss = F.mse_loss(quantized, encoded)

        if train_ae or train_kmeans:
            self.encoder_optimizer.zero_grad(set_to_none=True)
        if train_ae:
            self.decoder_optimizer.zero_grad(set_to_none=True)

        if train_ae or train_kmeans:
            combined_loss = recon_loss + commitment_loss
            combined_loss.backward()

        if train_ae or train_kmeans:
            self.encoder_optimizer.step()
        if train_ae:
            self.decoder_optimizer.step()

        return recon_loss, commitment_loss

    def train_vae_kmeans(
        self,
        state,
        next_state=None,
        update_centriods=True,
        kld_beta_vae=3,
        optimize_commitment=False,
        save_recon_every=None,
    ):
        if not update_centriods:
            optimize_commitment = False

        if next_state is not None:
            state = torch.cat((state, next_state), dim=0)
            # shuffle the batch by the first dimension
            state = state[torch.randperm(state.size(0)), :]

        # Optimize VAE
        encoded, mu, std = self.ground_Q.encoder(state)
        recon = self.decoder(encoded)
        vae_loss, recon_loss, kld = self.vae_loss_function(recon, state, mu, std, kld_beta_vae)
        # Wether or not assign centroids
        quantized, cluster_indices = self.kmeans.assign_centroid(
            encoded, update_centroid=update_centriods
        )

        if optimize_commitment:
            commitment_loss = F.mse_loss(quantized, encoded)
        else:
            commitment_loss = torch.tensor(0.0).to(self.device)

        self.encoder_optimizer.zero_grad(set_to_none=True)
        self.decoder_optimizer.zero_grad(set_to_none=True)

        combined_loss = vae_loss + commitment_loss
        combined_loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if save_recon_every and self.decoder.n_forward_call % save_recon_every == 0:
            stacked = torch.cat((recon[:7, :1], state[:1, :1]), dim=0)
            wandb_log_image(stacked)

        self.training_info["recon_loss"].append(recon_loss.item())
        self.training_info["kld"].append(kld.item())
        self.training_info["commitment_loss"].append(commitment_loss.item())

        return recon_loss, kld, commitment_loss

    def update_ae(
        self,
        state_batch,
        next_state_batch,
        save_recon_every=None,
        update_kmeans_centriods=True,
    ):
        if next_state_batch is not None:
            state_batch = torch.cat((state_batch, next_state_batch), dim=0)
            # shuffle the batch by the first dimension
            state_batch = state_batch[torch.randperm(state_batch.size(0)), :]

        # Optimize AE
        encoded = self.ground_Q.encoder(state_batch)
        recon = self.decoder(encoded)
        recon_loss = F.mse_loss(recon, state_batch)

        self.encoder_optimizer.zero_grad(set_to_none=True)
        self.decoder_optimizer.zero_grad(set_to_none=True)
        recon_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if update_kmeans_centriods:
            quantized, cluster_indices = self.kmeans.assign_centroid(encoded, update_centroid=True)

        if save_recon_every and (self.decoder.n_forward_call % save_recon_every) == 0:
            stacked = torch.cat((recon[:7, :1], state_batch[:1, :1]), dim=0)
            wandb_log_image(stacked)

        self.training_info["recon_loss"].append(recon_loss.item())

        return recon_loss, encoded, quantized

    def update_vae(
        self,
        state_batch,
        next_state_batch=None,
        kld_beta_vae=3,
        save_recon_every=None,
        update_kmeans_centriods=True,
        optimize_commitment=False,
    ):
        if next_state_batch is not None:
            state_batch = torch.cat((state_batch, next_state_batch), dim=0)
            # shuffle the batch by the first dimension
            state_batch = state_batch[torch.randperm(state_batch.size(0)), :]

        # Optimize VAE
        encoded, mu, std = self.ground_Q.encoder(state_batch)
        recon = self.decoder(encoded)

        # [wether or not update centroids]
        quantized, cluster_indices = self.kmeans.assign_centroid(
            encoded, update_centroid=update_kmeans_centriods
        )

        vae_loss, recon_loss, kld = self.vae_loss_function(
            recon, state_batch, mu, std, kld_beta_vae
        )
        if optimize_commitment:
            commitment_loss = F.mse_loss(encoded, quantized)
        else:
            commitment_loss = torch.tensor(0.0).to(self.device)

        self.encoder_optimizer.zero_grad(set_to_none=True)
        self.decoder_optimizer.zero_grad(set_to_none=True)
        (vae_loss + commitment_loss).backward(retain_graph=True)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if save_recon_every and (self.decoder.n_forward_call % save_recon_every) == 0:
            stacked = torch.cat((recon[:7, :1], state_batch[:1, :1]), dim=0)
            wandb_log_image(stacked)

        self.training_info["recon_loss"].append(recon_loss.item())
        self.training_info["kld"].append(kld.item())
        self.training_info["commitment_loss"].append(commitment_loss.item())

        return recon_loss, kld, encoded, quantized

    def optimize_commitment(self, encoded, quantized):
        """
        Currently this function gets error, when "one of the variables needed for gradient computation has been modified by an inplace operation" occurs
        """
        commitment_loss = F.mse_loss(encoded, quantized)
        self.encoder_optimizer.zero_grad(set_to_none=True)
        commitment_loss.backward()
        self.encoder_optimizer.step()
        self.training_info["commitment_loss"].append(commitment_loss.item())

    def pretrain_ae(self, data, batch_size=128, epochs=30):
        dataset = Dataset_pretrain(data)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("========== Start AE pretraining ==========")
        list_recon_loss = []
        list_kld = []
        list_commitment_loss = []
        for e in range(epochs):
            for i, (state_batch) in enumerate(train_loader):
                # only update vae without kmeans involved
                recon_loss, kld, commitment_loss = self.train_vae_kmeans(
                    state=state_batch,
                    next_state=None,
                    update_centriods=False,
                    kld_beta_vae=3,
                    optimize_commitment=False,
                    save_recon_every=None,
                )
                list_recon_loss.append(recon_loss.item())
                list_kld.append(kld.item())

            if i % 5 == 0:
                print(
                    f"Pretrain_Epoch {e}/{epochs}, recon_loss: {mean(list_recon_loss)}, kld: {mean(list_kld)}"
                )
        print("========== End AE pretraining ==========")

    def update_grdQ_absV(self, state, action, next_state, reward, terminated, update_absV=True):
        """
        This function is the combination of update_grdQ and update_absV and update_vae
        """
        if hasattr(self, "lr_scheduler_ground_Q"):
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_scheduler_ground_Q(self._current_progress_remaining),
            )
        if hasattr(self, "lr_scheduler_abstract_V"):
            update_learning_rate(
                self.abstract_V_optimizer,
                self.lr_scheduler_abstract_V(self._current_progress_remaining),
            )

        # batch = self.memory.sample(batch_size=self.batch_size)
        # state_batch = np.stack(batch.state, axis=0).transpose(0, 3, 1, 2)
        # next_state_batch = np.stack(batch.next_state, axis=0).transpose(0, 3, 1, 2)
        # state_batch = torch.from_numpy(state_batch).contiguous().float().to(self.device)
        # next_state_batch = torch.from_numpy(next_state_batch).contiguous().float().to(self.device)

        # # batch = self.memory.lazy_sample(batch_size=self.batch_size)
        # # state_batch = torch.cat(batch.state).to(self.device)
        # # next_state_batch = torch.cat(batch.next_state).to(self.device)
        # # action_batch = torch.cat(batch.action).to(self.device)
        # # reward_batch = torch.cat(batch.reward).to(self.device)
        # # terminated_batch = torch.cat(batch.terminated).to(self.device)

        # action_batch = torch.tensor(batch.action).unsqueeze(0).to(self.device)
        # reward_batch = torch.tensor(batch.reward).unsqueeze(0).to(self.device)
        # terminated_batch = torch.tensor(batch.terminated).unsqueeze(0).to(self.device)

        # mask = torch.eq(state_batch, next_state_batch)
        # num_same_aggregation = 0
        # for sample_mask in mask:
        #     if torch.all(sample_mask):
        #         num_same_aggregation += 1
        # print("num_same_aggregation:", num_same_aggregation)

        # [Data augmentation]
        state = self.aug(state)
        next_state = self.aug(next_state)

        # [Update ground Q network]
        grd_q, encoded, mu, std = self.ground_Q(state)
        grd_q = grd_q.gather(1, action)

        with torch.no_grad():

            # Vanilla DQN
            grd_q_next, encoded_next, mu, std = self.ground_Q_target(next_state)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)

            # Double DQN
            # action_argmax_target = self.ground_target_Q_net(next_state_batch).argmax(
            #     dim=1, keepdim=True
            # )
            # ground_next_max_Q = self.ground_Q_net(next_state_batch).gather(1, action_argmax_target)

            # Compute ground target Q value
            # abs_v = self.abstract_V_target(quantized)
            # abs_v_next = self.abstract_V_target(quantized_next)
            if update_absV:
                quantized, cluster_indices = self.kmeans.assign_centroid(
                    encoded, update_centroid=False
                )

                quantized_next, cluster_indices = self.kmeans.assign_centroid(
                    encoded_next, update_centroid=False
                )

                abs_v = self.abstract_V(quantized)
                abs_v_next = self.abstract_V(quantized_next)
                # shaping = self.gamma * abs_v_next - abs_v
                shaping = abs_v_next - abs_v
            else:
                shaping = 0

            grd_q_target = (
                reward
                + self.omega * shaping * (1 - terminated.float())
                + (1 - terminated.float()) * self.gamma * grd_q_next_max
            ).float()

        criterion = nn.SmoothL1Loss()
        ground_td_error = criterion(grd_q, grd_q_target)

        if update_absV:
            # [Update abstract V network]
            # mask_ = ~torch.tensor(
            #     [
            #         [torch.equal(a_state, next_a_state)]
            #         for a_state, next_a_state in zip(quantized, quantized_next)
            #     ]
            # ).to(self.device)
            # abs_v *= mask_
            # abs_v_target *= mask_

            abs_v = self.abstract_V(quantized)
            abs_v_target = torch.zeros(abs_v.shape[0]).to(self.device)
            with torch.no_grad():
                abs_v_next = self.abstract_V_target(quantized_next)
                for i, (a_state, next_a_state) in enumerate(zip(quantized, quantized_next)):
                    if torch.equal(a_state, next_a_state) and reward[i] == 0:
                        abs_v_target[i] = abs_v_next[i]
                    else:
                        abs_v_target[i] = reward[i] / 10 + self.gamma * abs_v_next[i]
            abs_v_target = abs_v_target.unsqueeze(1)
            criterion = nn.SmoothL1Loss()
            abstract_td_error = criterion(abs_v, abs_v_target)

            # [Optimize RL network]
            rl_loss = abstract_td_error + ground_td_error
        else:
            rl_loss = ground_td_error

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        if update_absV:
            self.abstract_V_optimizer.zero_grad(set_to_none=True)
        rl_loss.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                param.grad.data.clamp_(-1, 1)
            if update_absV:
                for param in self.abstract_V.parameters():
                    param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()
        if update_absV:
            self.abstract_V_optimizer.step()

        # [Update autoencoder and ema-kmeans]
        recon_loss, kld, commitment_loss = self.train_vae_kmeans(
            state,
            next_state=None,
            update_centriods=True,
            kld_beta_vae=3,
            optimize_commitment=False,
            save_recon_every=None,
        )

        self.training_info["ground_Q_error"].append(ground_td_error.item())
        if update_absV:
            self.training_info["abstract_V_error"].append(abstract_td_error.item())
        self.training_info["recon_loss"].append(recon_loss.item())
        self.training_info["kld"].append(kld.item())
        self.training_info["commitment_loss"].append(commitment_loss.item())

    def update_grdQ_shaping(self, state, action, next_state, reward, terminated, use_shaping: bool):
        if hasattr(self, "lr_scheduler_ground_Q"):
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_scheduler_ground_Q(self._current_progress_remaining),
            )

        # [Update ground Q network]
        grd_q, encoded, mu, std = self.ground_Q(state)
        grd_q = grd_q.gather(1, action)

        with torch.no_grad():

            # Vanilla DQN
            grd_q_next, encoded_next, mu, std = self.ground_Q_target(next_state)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)

            # Double DQN
            # action_argmax_target = self.ground_target_Q_net(next_state_batch).argmax(
            #     dim=1, keepdim=True
            # )
            # ground_next_max_Q = self.ground_Q_net(next_state_batch).gather(1, action_argmax_target)

            # Compute ground target Q value
            # abs_v = self.abstract_V_target(quantized)
            # abs_v_next = self.abstract_V_target(quantized_next)
            if use_shaping:
                quantized, cluster_indices = self.kmeans.assign_centroid(
                    encoded, update_centroid=False
                )

                quantized_next, cluster_indices = self.kmeans.assign_centroid(
                    encoded_next, update_centroid=False
                )

                abs_v = self.abstract_V(quantized)
                abs_v_next = self.abstract_V(quantized_next)
                # shaping = self.gamma * abs_v_next - abs_v
                shaping = abs_v_next - abs_v
            else:
                shaping = 0

            grd_q_target = (
                reward
                + self.omega * shaping * (1 - terminated.float())
                + (1 - terminated.float()) * self.gamma * grd_q_next_max
            ).float()

        criterion = nn.SmoothL1Loss()
        ground_td_error = criterion(grd_q, grd_q_target)

        # [Update abstract V network]
        # mask_ = ~torch.tensor(
        #     [
        #         [torch.equal(a_state, next_a_state)]
        #         for a_state, next_a_state in zip(quantized, quantized_next)
        #     ]
        # ).to(self.device)
        # abs_v *= mask_
        # abs_v_target *= mask_

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        ground_td_error.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()

        self.training_info["ground_Q_error"].append(ground_td_error.item())

        return quantized, quantized_next

    def update_absV(self, quantized, quantized_next, reward, terminated):
        if hasattr(self, "lr_scheduler_abstract_V"):
            update_learning_rate(
                self.abstract_V_optimizer,
                self.lr_scheduler_abstract_V(self._current_progress_remaining),
            )
        # [Update abstract V network]
        # mask_ = ~torch.tensor(
        #     [
        #         [torch.equal(a_state, next_a_state)]
        #         for a_state, next_a_state in zip(quantized, quantized_next)
        #     ]
        # ).to(self.device)
        # abs_v *= mask_
        # abs_v_target *= mask_

        abs_v = self.abstract_V(quantized)
        abs_v_target = torch.zeros(abs_v.shape[0]).to(self.device)
        with torch.no_grad():
            abs_v_next = self.abstract_V_target(quantized_next)
            for i, (a_state, next_a_state) in enumerate(zip(quantized, quantized_next)):
                if torch.equal(a_state, next_a_state) and reward[i] == 0:
                    abs_v_target[i] = abs_v_next[i]
                else:
                    abs_v_target[i] = reward[i] / 10 + self.gamma * abs_v_next[i]
        abs_v_target = abs_v_target.unsqueeze(1)
        criterion = nn.SmoothL1Loss()
        abstract_td_error = criterion(abs_v, abs_v_target)

        self.abstract_V_optimizer.zero_grad(set_to_none=True)
        abstract_td_error.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.abstract_V.parameters():
                param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.abstract_V_optimizer.step()

        # [Update autoencoder and ema-kmeans]
        # recon_loss, kld, commitment_loss = self.train_vae_kmeans(
        #     state_batch=state_batch,
        #     next_state_batch=next_state_batch,
        #     train_ae=True,
        #     update_kmeans=True,
        # )

        self.training_info["abstract_V_error"].append(abstract_td_error.item())

    def update_grdQ_pure(self, state, action, next_state, reward, terminated, shaping=True):
        if hasattr(self, "lr_scheduler_ground_Q"):
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_scheduler_ground_Q(self._current_progress_remaining),
            )

        state = self.aug(state)
        next_state = self.aug(next_state)

        # [Update ground Q network]
        grd_q, encoded, mu, std = self.ground_Q(state)
        grd_q = grd_q.gather(1, action)

        with torch.no_grad():

            # Vanilla DQN
            grd_q_next, encoded_next = self.ground_Q_target(next_state)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)

            # Double DQN
            # action_argmax_target = self.ground_target_Q_net(next_state_batch).argmax(
            #     dim=1, keepdim=True
            # )
            # ground_next_max_Q = self.ground_Q_net(next_state_batch).gather(1, action_argmax_target)

            # Compute ground target Q value

            grd_q_target = (reward + (1 - terminated.float()) * self.gamma * grd_q_next_max).float()

        criterion = nn.SmoothL1Loss()
        ground_td_error = criterion(grd_q, grd_q_target)

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        ground_td_error.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()

        self.training_info["ground_Q_error"].append(ground_td_error.item())


class Dataset_pretrain(torch.utils.data.Dataset):
    def __init__(self, data_list: Tensor):
        data_list = data_list[torch.randperm(data_list.size(0)), :]
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        state = self.data_list[idx]
        return state


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


_AVAILABLE_ENCODERS = {"pixel": EncoderRes}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder = Encoder(
    #     inchannels=4,
    #     linear_out_dim=64,
    #     observation_space=gym.spaces.box.Box(low=0, high=1, shape=(84, 84, 4)),
    #     n_redisual_layers=0,
    # ).to(device)
    # summary(encoder, (4, 84, 84))

    # # repara = ReparameterizeModule(out_dim=32, shape_encoder_fm=encoder.shape_encoder_fm).to(device)
    # # summary(repara, (64, 4, 4))

    # decoder = Decoder(
    #     32,
    #     4,
    #     shape_encoder_fm=encoder.shape_encoder_fm,
    # ).to(device)
    # summary(decoder, (32,))

    encoder = Encoder_MiniGrid(
        inchannels=3,
        linear_out_dim=64,
        observation_space=gym.spaces.box.Box(low=0, high=1, shape=(8, 8, 3)),
        n_redisual_layers=0,
    ).to(device)
    summary(encoder, (3, 8, 8))

    decoder = Decoder_MiniGrid(
        in_dim=64,
        out_channels=3,
        shape_conv_output=encoder.shape_encoder_fm,
    ).to(device)
    summary(decoder, (64,))
