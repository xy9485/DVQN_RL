from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor, nn

from nn_models.components import ConvBlock, ResidualLayer


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
        self.linear_out_dim = linear_out_dim
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
        with torch.no_grad():
            x = observation_space.sample()
            x = T.ToTensor()(x).unsqueeze(0)
            x = nn.Sequential(*blocks)(x.float())
            self.shape_conv_output = x.shape
            # shape of the last feature map of the encoder, [B, C, H, W]
            self.n_flatten = torch.prod(torch.tensor(x.shape[1:])).item()
            # self.shape_latent_h_w = x.shape[2:]
            # C, H, W = x[1:]
        if self.linear_out_dim:  # if encoder_out_dim is not None than followed by a linear layer
            blocks.extend(
                [
                    nn.Flatten(),
                    nn.Linear(self.n_flatten, self.linear_out_dim),
                    nn.ReLU(),
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
        x = x / 255.0
        x = x.permute(0, 3, 2, 1).float()
        return self.blocks(x)

        # x = self.blocks(x)
        # x = x.view(-1, self.n_flatten)
        # mu = self.fc_mu(x)
        # std = self.fc_std(x)
        # return self.reparameterize(mu, std), mu, std


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


class Encoder_MiniGrid(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        linear_out_dim: Optional[int] = None,  # latent_dim
        observation_space: gym.spaces.box.Box = None,
        hidden_dims: List = [16, 32, 64],
        n_redisual_layers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.forward_call = 0
        self.linear_out_dim = linear_out_dim
        final_channels = hidden_dims[-1]
        # encoder architecture #1
        # blocks = [
        #     ConvBlock(
        #         in_channels,
        #         hidden_dims[0],
        #         kernel_size=2,
        #         stride=1,
        #         padding=0,
        #         batch_norm=False,
        #     ),
        #     nn.MaxPool2d((2, 2)),
        #     ConvBlock(
        #         hidden_dims[0],
        #         hidden_dims[1],
        #         kernel_size=2,
        #         stride=1,
        #         padding=0,
        #         batch_norm=False,
        #     ),
        #     nn.MaxPool2d((2, 2)),
        #     # nn.MaxPool2d((2, 2)),
        #     # ConvBlock(
        #     #     hidden_dims[1],
        #     #     hidden_dims[2],
        #     #     kernel_size=2,
        #     #     stride=1,
        #     #     padding=0,
        #     #     batch_norm=False,
        #     # ),
        # ]
        # encoder architeture #2
        blocks = [
            ConvBlock(
                in_channels,
                hidden_dims[0],
                kernel_size=5,
                stride=2,
                padding=0,
                batch_norm=False,
            ),
            ConvBlock(
                hidden_dims[0],
                hidden_dims[1],
                kernel_size=3,
                stride=2,
                padding=0,
                batch_norm=False,
            ),
            ConvBlock(
                hidden_dims[1],
                final_channels,
                kernel_size=3,
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
        with torch.no_grad():
            x = observation_space.sample()
            x = torch.from_numpy(x).unsqueeze(0).transpose(1, 3).transpose(2, 3).float()
            x = nn.Sequential(*blocks)(x)
            self.shape_conv_output = x.shape
            # shape of the last feature map of the encoder, [B, C, H, W]
            # self.n_flatten = torch.prod(torch.tensor(x.shape[1:])).item()
            self.n_flatten = nn.Flatten()(x).shape[1]
            # self.shape_latent_h_w = x.shape[2:]
            # C, H, W = x[1:]
        if self.linear_out_dim:  # if linear_out_dim is not None than followed by a linear layer
            blocks.extend(
                [
                    nn.Flatten(),
                    nn.Linear(self.n_flatten, self.linear_out_dim),
                    nn.LayerNorm(self.linear_out_dim),
                    nn.Tanh(),
                    # nn.ReLU(),
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

    def forward(self, x: Tensor) -> Tensor:
        # x = x.transpose(1, 3).transpose(2, 3).float()
        x = x / 10
        x = x.permute(0, 3, 2, 1).float()
        return self.blocks(x)

        # x = self.blocks(x)
        # x = x.view(-1, self.n_flatten)
        # mu = self.fc_mu(x)
        # std = self.fc_std(x)
        # return self.reparameterize(mu, std), mu, std


class Encoder_MiniGrid_PartialObs(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        linear_out_dim: Optional[int] = None,  # latent_dim
        observation_space: gym.spaces.box.Box = None,
        hidden_dims: List = [16, 32],
        **kwargs,
    ) -> None:
        super().__init__()
        self.forward_call = 0
        # final_channels = hidden_dims[-1]*2
        self.linear_out_dim = linear_out_dim
        self.random_encoder = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )

        # Compute shape by doing one forward pass

        with torch.no_grad():
            x = observation_space.sample()
            x = T.ToTensor()(x).unsqueeze(0)
            x = self.random_encoder(x.float())
            self.shape_conv_output = x.shape
            # shape of the last feature map of the encoder, [B, C, H, W]
            self.n_flatten = nn.Flatten()(x).shape[1]

        blocks = [self.random_encoder, nn.Flatten()]
        if self.linear_out_dim:  # if linear_out_dim is not None than followed by a linear layer
            blocks.extend(
                [
                    nn.Linear(self.n_flatten, self.linear_out_dim),
                    # nn.LayerNorm(linear_out_dim),
                    nn.Tanh(),
                    # nn.ReLU(),
                ]
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        # x /= 10
        x = x.transpose(1, 3).transpose(2, 3).float()
        return self.blocks(x)


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
