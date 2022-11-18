from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from nn_models.components import ConvBlock, ResidualLayer


class RandomEncoder(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, feature_dim) -> None:
        super().__init__()
        n_input_channels = observation_space.shape[0]

        # test_layer = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
        # print("test_layer.weight.size():", test_layer.weight.size())

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.flatten_layer = nn.Flatten()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            x = observation_space.sample()
            x = torch.from_numpy(x).unsqueeze(0).float()
            # x = torch.from_numpy(x).unsqueeze(0).permute(0, 3, 1, 2).float()
            x = self.cnn(x)
            x = self.flatten_layer(x)
            n_flatten = x.shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        x = x.float()
        # x = x.permute(0, 3, 2, 1).float()
        x = self.cnn(x)
        x = self.flatten_layer(x)
        return self.head(x)


class RandomEncoderMiniGrid(nn.Module):
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
        final_channels = hidden_dims[-1]
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
        self.blocks = nn.Sequential(*blocks)
        with torch.no_grad():
            x = observation_space.sample()
            x = T.ToTensor()(x).unsqueeze(0)
            x = self.blocks(x.float())
            # self.shape_conv_output = x.shape
            # shape of the last feature map of the encoder, [B, C, H, W]
            # self.n_flatten = torch.prod(torch.tensor(x.shape[1:])).item()
            self.n_flatten = nn.Flatten()(x).shape[1]
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
        x /= 10
        return self.blocks(x)

        # x = self.blocks(x)
        # x = x.view(-1, self.n_flatten)
        # mu = self.fc_mu(x)
        # std = self.fc_std(x)
        # return self.reparameterize(mu, std), mu, std
