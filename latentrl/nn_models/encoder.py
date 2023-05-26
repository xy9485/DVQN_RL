import copy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor, nn
from torchsummary import summary
from nn_models.components import ConvBlock, ResidualLayer


class Encoder(nn.Module):
    def __init__(self, linear_dims: int | list | None) -> None:
        super().__init__()
        assert isinstance(linear_dims, list) and isinstance(linear_dims[0], int)

        self.linear_dims = linear_dims
        self.linear_out_dim = None

    def maybe_add_linear_module(self, output_logits=True) -> None:
        self.blocks.append(nn.Flatten())
        if self.linear_dims == [-1]:
            self.linear_out_dim = self.cnn_flatten_dim
            return
        if len(self.linear_dims) == 1:
            # self.linear_dims = [self.linear_dims]
            self.blocks.append(nn.Linear(self.cnn_flatten_dim, self.linear_dims[0]))
            # self.blocks.append(nn.LayerNorm(self.linear_dims))
            # self.blocks.append(nn.ReLU())
            # self.blocks.append(nn.Sigmoid())
            # self.blocks.append(nn.Tanh())
            # self.blocks.append(nn.Linear(self.linear_dims[-1], self.linear_dims[-1]))
            if not output_logits:
                self.blocks.append(nn.Tanh())
            self.linear_out_dim = self.linear_dims[0]
            return

        for n_in, n_out in zip([self.cnn_flatten_dim] + self.linear_dims[:-1], self.linear_dims):
            self.blocks.extend([nn.Linear(n_in, n_out), nn.ReLU()])

        self.blocks.append(nn.Linear(self.linear_dims[-1], self.linear_dims[-1]))
        # self.blocks.append(nn.LayerNorm(self.linear_dims[-1]))
        if not output_logits:
            self.blocks.append(nn.Tanh())
        self.linear_out_dim = self.linear_dims[-1]
        return

    @property
    @torch.no_grad()
    def cnn_flatten_dim(self) -> int:
        x = nn.Sequential(*self.cnn_module)(self.example_x)
        x = nn.Flatten()(x)
        return x.shape[1]

    @property
    @torch.no_grad()
    def cnn_last_feature_map_dim(self) -> Tuple[int, int, int]:
        """Returns the shape[B, C, H, W] of the last feature map of the CNN."""
        x = nn.Sequential(*self.cnn_module)(self.example_x)
        return x.shape


class EncoderImg(Encoder):
    def __init__(
        self,
        observation_space: gym.spaces.box.Box = None,
        hidden_channels: List = [32, 64, 64],
        linear_dims: int | list | None = None,  # latent_dim
        n_redisual_layers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(linear_dims)
        self.blocks = [
            ConvBlock(
                min(observation_space.shape),
                hidden_channels[0],
                kernel_size=8,
                stride=4,
                padding=0,
                batch_norm=False,
            ),
            ConvBlock(
                hidden_channels[0],
                hidden_channels[1],
                kernel_size=4,
                stride=2,
                padding=0,
                batch_norm=False,
            ),
            ConvBlock(
                hidden_channels[1],
                hidden_channels[2],
                kernel_size=3,
                stride=2,
                padding=0,
                batch_norm=False,
            ),
        ]
        # self.blocks = [
        #     nn.Conv2d(input_channels, 32, 5, stride=5, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 5, stride=5, padding=0),
        #     nn.ReLU(),
        # ]
        if n_redisual_layers > 0:
            for _ in range(n_redisual_layers):
                self.blocks.append(ResidualLayer(hidden_channels[-1], hidden_channels[-1]))
            self.blocks.append(nn.ReLU())

        # self.cnn_module = nn.Sequential(*self.blocks)
        self.cnn_module = self.blocks[:]
        # Compute shape by doing one forward pass
        self.example_x = observation_space.sample()
        self.example_x = torch.from_numpy(self.example_x).unsqueeze(0).float()

        # with torch.no_grad():
        #     x = observation_space.sample()
        #     x = torch.from_numpy(x).unsqueeze(0).float()
        #     # when using atari wrapper, first dim is already channels, so no need to permute
        #     x = nn.Sequential(*self.blocks)(x)
        #     self.shape_conv_output = x.shape

        #     self.n_flatten = nn.Flatten()(x).shape[1]
        self.maybe_add_linear_module()
        # if self.linear_dims:
        #     self.blocks.extend(
        #         [
        #             nn.Flatten(),
        #             nn.Linear(self.n_flatten, self.linear_dims),
        #             nn.ReLU(),
        #         ]
        #     )
        # else:
        #     self.blocks.append(nn.Flatten())
        # # self.ln = nn.LayerNorm([C, H, W])

        # self.outputs = dict()
        # self.fc_mu = nn.Linear(self.n_flatten, out_channels)
        # self.fc_std = nn.Linear(self.n_flatten, out_channels)
        self.blocks = nn.Sequential(*self.blocks)

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
        # x = x.float()
        # x = x.permute(0, 3, 1, 2).float()
        return self.blocks(x)

        # x = self.blocks(x)
        # x = x.view(-1, self.n_flatten)
        # mu = self.fc_mu(x)
        # std = self.fc_std(x)
        # return self.reparameterize(mu, std), mu, std


_AVAILABLE_ENCODERS = {
    "full_img": EncoderImg,
}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )


def make_encoder(
    input_format,
    observation_space=None,
    hidden_channels: int = None,
    linear_dims: int | list = None,
):
    if input_format == "full_img":
        encoder = EncoderImg(
            observation_space=observation_space,
            # hidden_channels=hidden_channels,
            linear_dims=linear_dims,
        )
    else:
        raise NotImplementedError

    return encoder


# class EncoderMaker:
#     def __init__(self, input_format, agent):
#         self.input_format = input_format
#         self.agent = agent

#     def make(self):
#         if self.input_format == "partial_obs":
#             encoder = Encoder_MiniGrid_PartialObs(
#                 linear_out_dim=None,
#                 observation_space=self.agent.env.observation_space,
#             )
#         elif self.input_format == "full_obs":
#             encoder = Encoder_MiniGrid(
#                 input_channels=self.agent.env.observation_space.shape[-1],
#                 linear_dims=self.agent.grd_encoder_linear_dims,
#                 observation_space=self.agent.env.observation_space,
#                 hidden_channels=self.agent.grd_hidden_channels,
#             )
#         elif self.input_format == "full_img":
#             encoder = EncoderImg(
#                 input_channels=self.agent.env.observation_space.shape[0],
#                 linear_dims=self.agent.grd_encoder_linear_dims,
#                 observation_space=self.agent.env.observation_space,
#                 hidden_channels=self.agent.grd_hidden_channels,
#             )
#         elif self.input_format == "full_img_small":
#             encoder = Encoder_MinAtar(
#                 input_channels=self.agent.env.observation_space.shape[-1],
#                 linear_dims=self.agent.grd_encoder_linear_dims,
#                 observation_space=self.agent.env.observation_space,
#                 hidden_channels=self.agent.grd_hidden_channels,
#             )
#         else:
#             raise NotImplementedError

#         return encoder
