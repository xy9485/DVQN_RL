import datetime
from json import decoder
import os
import random
from sys import modules
from tarfile import BLOCKSIZE

# from this import d
import time
from collections import namedtuple
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

# from unicodedata import decimal

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch import Tensor, nn
from torchsummary import summary
from torchvision.utils import save_image

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
        aa = torch.sum(flat_input**2, dim=1, keepdim=True)
        bb = torch.sum(self._embedding.weight**2, dim=1)
        dd = 2 * torch.matmul(flat_input, self._embedding.weight.t())
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
        negative_slope: float = 0.2,
    ):
        super().__init__()
        blocks = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm)
        ]
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_channels))
        if negative_slope != 0:
            blocks.append(nn.LeakyReLU(negative_slope))
        else:
            blocks.append(nn.ReLU())
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
        negative_slope: float = 0.2,
    ):
        super().__init__()
        blocks = [
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm
            )
        ]
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_channels))
        if negative_slope != 0:
            blocks.append(nn.LeakyReLU(negative_slope))
        else:
            blocks.append(nn.ReLU())
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Encoder(nn.Module):
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
                batch_norm=False,
                negative_slope=0.2,
            ),
            ConvBlock(
                hidden_dims[0],
                hidden_dims[1],
                kernel_size=4,
                stride=2,
                padding=0,
                batch_norm=True,
                negative_slope=0.2,
            ),
            ConvBlock(
                hidden_dims[1],
                n_channel_output,
                kernel_size=3,
                stride=2,
                padding=0,
                batch_norm=True,
                negative_slope=0.2,
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


class EncoderVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,  # latent_dim
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
                batch_norm=True,
            ),
            ConvBlock(
                hidden_dims[1],
                hidden_dims[1],
                kernel_size=3,
                stride=2,
                padding=0,
                batch_norm=True,
            ),
        ]
        if n_redisual_layers > 0:
            for _ in range(n_redisual_layers):
                blocks.append(ResidualLayer(hidden_dims[1], hidden_dims[1]))
            blocks.append(nn.ReLU())

        self.blocks = nn.Sequential(*blocks)
        # Compute shape by doing one forward pass
        with torch.no_grad():
            x = observation_space.sample()
            x = T.ToTensor()(x).unsqueeze(0)
            x = self.blocks(x.float())
            self.shape_encoder_fm = (
                x.shape
            )  # shape of the last feature map of the encoder, [B, C, H, W]
            self.n_flatten = torch.prod(torch.tensor(x.shape[1:])).item()
            # self.shape_latent_h_w = x.shape[2:]
            # C, H, W = x[1:]

        # self.fc = nn.Linear(n_flatten, self.feature_dim)
        # self.ln = nn.LayerNorm([C, H, W])

        # self.outputs = dict()
        self.fc_mu = nn.Linear(self.n_flatten, out_channels)
        self.fc_std = nn.Linear(self.n_flatten, out_channels)

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
        x = self.blocks(x)
        x = x.view(-1, self.n_flatten)
        return self.fc_mu(x), self.fc_std(x)


class ReparameterizeModule(nn.Module):
    def __init__(self, shape_encoder_fm: torch.Size, out_channels: int) -> None:
        super().__init__()
        self.n_flatten = torch.prod(torch.tensor(shape_encoder_fm[1:])).item()
        self.fc_mu = nn.Linear(self.n_flatten, out_channels)
        self.fc_std = nn.Linear(self.n_flatten, out_channels)

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


class Decoder(nn.Module):
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
                negative_slope=0.2,
            ),
            DeConvBlock(
                hidden_dims[0],
                hidden_dims[1],
                kernel_size=4,
                stride=2,
                padding=0,
                batch_norm=True,
                negative_slope=0.2,
            ),
            # DeConvBlock(
            #     hidden_dims[1],
            #     n_channel_output,
            #     kernel_size=8,
            #     stride=4,
            #     padding=0,
            #     batch_norm=True,
            #     negative_slope=0.2,
            # ),
            nn.ConvTranspose2d(hidden_dims[1], n_channel_output, 8, 4, 0),
            nn.Sigmoid(),
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


class DecoderVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shape_encoder_fm: torch.Size,
        hidden_dims: List = [32, 64],
        n_redisual_layers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_forward_call = 0
        self.shape_encoder_fm = shape_encoder_fm
        hidden_dims.reverse()

        self.n_flatten = torch.prod(torch.tensor(shape_encoder_fm[1:])).item()
        self.decoder_input = nn.Linear(in_channels, self.n_flatten)
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
        x = self.decoder_input(x)
        x = x.view(-1, self.shape_encoder_fm[1], self.shape_encoder_fm[2], self.shape_encoder_fm[3])
        print(x.shape)
        x = self.blocks(x)

        self.n_forward_call += 1
        # result = self.ln(result)
        return x


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
    def __init__(self, input_dim, action_space, conv_first=True):
        super().__init__()
        self.conv_first = conv_first
        if conv_first:
            self.conv = nn.Sequential(
                nn.Conv2d(input_dim[0], input_dim[0], kernel_size=1, stride=1),
                nn.ReLU(),
            )

        self.flatten_layer = nn.Flatten()
        self.linears = nn.Sequential(
            nn.Linear(np.prod(input_dim), 512),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(512, action_space.n),
            # nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.conv_first:
            x = self.conv(x)
        x = self.flatten_layer(x)
        return self.linears(x)


class V_MLP(nn.Module):
    def __init__(self, input_dim, conv_first=True):
        super().__init__()
        self.conv_first = conv_first
        if conv_first:
            self.conv = nn.Conv2d(input_dim[0], input_dim[0], kernel_size=1, stride=1)

        self.flatten_layer = nn.Flatten()
        self.fc1 = nn.Linear(np.prod(input_dim), 512)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 256)
        self.head = nn.Linear(512, 1)

    def forward(self, x: Tensor) -> Tensor:
        if self.conv_first:
            x = F.relu(self.conv(x))
        x = self.flatten_layer(x)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        return self.head(x)


class DQN(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.box.Box = None,
        action_space: gym.spaces.discrete.Discrete = None,
        dim_encoder_out: int = 64,
        hidden_dims: List = None,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            n_channel_input=observation_space.shape[-1],
            n_channel_output=dim_encoder_out,
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

        self.Q_linear = Q_MLP(self.encoder_out_shape, action_space)

    def forward_conv(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_conv(x)
        return self.Q_linear(x), x


class HDQN(nn.Module):
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

        self.ground_Q = DQN(
            env.observation_space,
            env.action_space,
            config.dim_encoder_out,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        self.ground_Q_target = DQN(
            env.observation_space,
            env.action_space,
            config.dim_encoder_out,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        self.ground_Q_target.load_state_dict(self.ground_Q.state_dict())
        # self.dqn_target.eval()

        # self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, beta)
        self.vector_quantizer = VectorQuantizerEMA(
            config.dim_encoder_out, config.vq_n_codebook, commitment_factor=beta
        ).to(self.device)
        self.vector_quantizer.train()

        self.decoder = Decoder(
            n_channel_input=config.dim_encoder_out,
            n_channel_output=env.observation_space.shape[-1],
            hidden_dims=config.hidden_dims,
        ).to(self.device)

        self.abstract_V = V_MLP(self.ground_Q.encoder_out_shape).to(self.device)
        self.abstract_V_target = V_MLP(self.ground_Q.encoder_out_shape).to(self.device)
        self.abstract_V_target.load_state_dict(self.abstract_V.state_dict())
        # self.abstract_V_target.eval()

        # summary(self.ground_Q.encoder, (4, 84, 84))
        # summary(self.decoder, (32, 4, 4))

        self.outputs = dict()
        # self.apply(weight_init)

        # Initialize experience replay buffer
        self.memory = ReplayMemory(self.size_replay_memory)
        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "done")
        )
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

    def reset_training_info(self, wandb_log=False):
        if wandb_log:
            metrics = {
                "loss/ground_Q_error": mean(self.training_info["ground_Q_error"]),
                "loss/abstract_V_error": mean(self.training_info["abstract_V_error"]),
                "loss/vq_loss": mean(self.training_info["vq_loss"]),
                "loss/perplexity": mean(self.training_info["perplexity"]),
                "loss/recon_loss": mean(self.training_info["recon_loss"]),
                "loss/vqvae_loss": mean(self.training_info["vq_loss"])
                + mean(self.training_info["recon_loss"]),
            }
            wandb.log(metrics)
        self.training_info = {
            "ground_Q_error": [],
            "abstract_V_error": [],
            "vq_loss": [],
            "perplexity": [],
            "recon_loss": [],
            "vq_loss_next": [],
            "perplexity_next": [],
        }

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
        if hasattr(self, "vector_quantizer"):
            self.vector_quantizer_optimizer = optim.Adam(
                self.vector_quantizer.parameters(), lr=config.lr_vq
            )

        if hasattr(self, "decoder"):
            self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=config.lr_decoder)
            self.encoder_optimizer = optim.Adam(
                self.ground_Q.encoder.parameters(), lr=config.lr_encoder
            )

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

    def cache(self, state, action, next_state, reward, done):
        """Add the experience to memory"""
        state = T.ToTensor()(state).float().unsqueeze(0)
        next_state = T.ToTensor()(next_state).float().unsqueeze(0)
        action = torch.tensor([action]).unsqueeze(0)
        reward = torch.tensor([reward]).unsqueeze(0)
        done = torch.tensor([done]).unsqueeze(0)

        self.memory.push(state, action, next_state, reward, done)

    @torch.no_grad()
    def act(self, state):
        self._update_current_progress_remaining(self.timesteps_done, self.total_timesteps)
        self.exploration_rate = self.exploration_scheduler(self._current_progress_remaining)

        state = T.ToTensor()(state).float().unsqueeze(0).to(self.device)

        if random.random() > self.exploration_rate:
            action = self.ground_Q(state)[0].max(1)[1].view(1, 1)
        else:
            action = random.randrange(self.n_actions)

        self.timesteps_done += 1
        return action

    def update(self):
        if self.timesteps_done < self.init_steps:
            return None, None, None, None

        if self.timesteps_done == self.init_steps:
            for _ in range(int(self.init_steps / 100)):
                self.train()

        if self.timesteps_done % self.ground_learn_every == 0:
            for _ in range(self.ground_gradient_steps):
                self.train()

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
            self.reset_training_info(wandb_log=True)

    def train_vqvae(self, state_batch, beta=1.0, save_recon_every=1000):
        # Optimize VQVAE
        encoded = self.ground_Q.forward_conv(state_batch)
        quantized, vq_loss, perplexity, encodings = self.vector_quantizer(encoded)
        recon = self.decoder(quantized)

        recon_loss = F.mse_loss(recon, state_batch)
        decode_loss = vq_loss * beta + recon_loss

        self.encoder_optimizer.zero_grad(set_to_none=True)
        self.vector_quantizer_optimizer.zero_grad(set_to_none=True)
        self.decoder_optimizer.zero_grad(set_to_none=True)
        decode_loss.backward()
        self.encoder_optimizer.step()
        self.vector_quantizer_optimizer.step()
        self.decoder_optimizer.step()

        if self.decoder.n_forward_call % save_recon_every == 0:
            stacked = torch.cat((recon[:7, :1], state_batch[:1, :1]), dim=0)
            wandb_log_image(stacked)

        return vq_loss, recon_loss, perplexity

    def train(self):
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

        batch = self.memory.sample(batch_size=self.batch_size)    
        # batch = self.memory.lazy_sample(batch_size=self.batch_size)
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        # mask = torch.eq(state_batch, next_state_batch)
        # num_same_aggregation = 0
        # for sample_mask in mask:
        #     if torch.all(sample_mask):
        #         num_same_aggregation += 1
        # print("num_same_aggregation:", num_same_aggregation)

        # Compute ground TD error
        grd_q, encoded = self.ground_Q(state_batch)
        grd_q = grd_q.gather(1, action_batch)

        with torch.no_grad():
            self.ground_Q_target.eval()
            self.abstract_V.eval()

            # Vanilla DQN
            grd_q_next, encoded_next = self.ground_Q_target(next_state_batch)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)

            # Double DQN
            # action_argmax_target = self.ground_target_Q_net(next_state_batch).argmax(
            #     dim=1, keepdim=True
            # )
            # ground_next_max_Q = self.ground_Q_net(next_state_batch).gather(1, action_argmax_target)

            # Compute ground target Q value
            quantized, _, _, _ = self.vector_quantizer(encoded)
            quantized_next, vq_loss_next, perplexity_next, _ = self.vector_quantizer(encoded_next)
            abs_v = self.abstract_V(quantized)
            abs_v_next = self.abstract_V(quantized_next)
            # abs_v = self.abstract_V_target(quantized)
            # abs_v_next = self.abstract_V_target(quantized_next)
            # shaping = self.gamma * abs_v_next - abs_v
            shaping = abs_v_next - abs_v
            grd_q_target = (
                reward_batch
                + self.omega * shaping * (1 - done_batch.float())
                + (1 - done_batch.float()) * self.gamma * grd_q_next_max
            ).float()

        criterion = nn.SmoothL1Loss()
        ground_td_error = criterion(grd_q, grd_q_target)

        # Compute abstract TD error
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
            self.abstract_V_target.eval()
            abs_v_next = self.abstract_V_target(quantized_next)
            for i, (a_state, next_a_state, reward) in enumerate(
                zip(quantized, quantized_next, reward_batch)
            ):
                if torch.equal(a_state, next_a_state) and reward == 0:
                    abs_v_target[i] = abs_v_next[i]
                else:
                    abs_v_target[i] = reward_batch[i] / 10 + self.gamma * abs_v_next[i]
        abs_v_target = abs_v_target.unsqueeze(1)
        criterion = nn.SmoothL1Loss()
        abstract_td_error = criterion(abs_v, abs_v_target)

        # Optimize RL network
        rl_loss = abstract_td_error + ground_td_error

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        self.abstract_V_optimizer.zero_grad(set_to_none=True)
        rl_loss.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                param.grad.data.clamp_(-1, 1)

            for param in self.abstract_V.parameters():
                param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()
        self.abstract_V_optimizer.step()

        vq_loss, recon_loss, perplexity = self.train_vqvae(
            state_batch=state_batch, beta=1.0, save_recon_every=self.save_recon_every
        )

        self.n_call_train += 1
        self.training_info["ground_Q_error"].append(ground_td_error.item())
        self.training_info["abstract_V_error"].append(abstract_td_error.item())
        self.training_info["vq_loss"].append(vq_loss.item())
        self.training_info["perplexity"].append(perplexity.item())
        self.training_info["recon_loss"].append(recon_loss.item())
        self.training_info["vq_loss_next"].append(vq_loss_next.item())
        self.training_info["perplexity_next"].append(perplexity_next.item())


_AVAILABLE_ENCODERS = {"pixel": EncoderRes}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = EncoderVAE(
        4,
        32,
        gym.spaces.box.Box(low=0, high=1, shape=(84, 84, 4)),
        n_redisual_layers=0,
    ).to(device)
    summary(encoder, (4, 84, 84))

    decoder = DecoderVAE(
        32,
        4,
        shape_encoder_fm=encoder.shape_encoder_fm,
    ).to(device)
    summary(decoder, (32,))
