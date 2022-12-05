from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch
from torch import Tensor, nn

from nn_models.components import ReparameterizeModule, MlpModel
from nn_models.encoder import (
    Encoder,
    EncoderImg,
    Encoder_MiniGrid,
    Encoder_MiniGrid_PartialObs,
    EncoderRes,
)


class Q_MLP(nn.Module):
    def __init__(self, input_dim, action_space, flatten=False, hidden_dim=64):
        super().__init__()
        blocks = []
        if flatten:
            blocks.extend(
                [
                    nn.Flatten(),
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
        else:
            blocks.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    # nn.ELU(),
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
    def __init__(self, input_dim, mlp_hidden_dim_abs, flatten=False):
        super().__init__()
        blocks = []
        if flatten:
            blocks.extend(
                [
                    nn.Flatten(),
                    nn.Linear(np.prod(input_dim), mlp_hidden_dim_abs),
                    nn.ReLU(),
                ]
            )
        else:
            blocks.extend(
                [
                    nn.Linear(input_dim, mlp_hidden_dim_abs),
                    nn.ReLU(),
                    nn.Linear(mlp_hidden_dim_abs, mlp_hidden_dim_abs),
                    nn.ReLU(),
                ]
            )
        blocks.extend(
            [
                # nn.Linear(256, 256),
                # nn.ReLU(),
                # nn.Linear(512, 512),
                # nn.ReLU(),
                nn.Linear(mlp_hidden_dim_abs, 1),
            ]
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class DQN(nn.Module):
    def __init__(
        self,
        # observation_space: gym.spaces.box.Box,
        action_space: gym.spaces.discrete.Discrete,
        encoder: Encoder,
        mlp_hidden_dim_grd: int | list,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        # with torch.no_grad():
        #     x = observation_space.sample()  # 0-255 in defualt
        #     # x = x.transpose(2, 0, 1)
        #     x = torch.tensor(x).unsqueeze(0)
        #     # x = T.ToTensor()(x).unsqueeze(0)
        #     x = self.encoder(x)
        #     # x = self.flatten_layer(x)
        #     # n_flatten = x.shape[1]
        #     self.encoder_out_shape = x.shape[1:]

        self.critic_input_dim = self.encoder.linear_out_dim
        # self.critic = Q_MLP(
        #     self.critic_input_dim, action_space, flatten=False, hidden_dim=mlp_hidden_dim_grd
        # )
        self.critic = MlpModel(
            input_dim=self.critic_input_dim,
            hidden_dims=mlp_hidden_dim_grd,
            output_dim=action_space.n,
            activation=nn.ReLU,
        )

    def forward_conv(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor):
        x = self.encoder(x)
        return self.critic(x), x

    def forward_enc_detach(self, x: Tensor):
        x = self.encoder(x).detach()
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
        self.conv_block = EncoderImg(
            observation_space.shape[-1],
            linear_dims=None,
            observation_space=observation_space,
            hidden_channels=hidden_dims,
        )

        self.repara = ReparameterizeModule(embedding_dim, self.conv_block.shape_conv_output)
        self.encoder = nn.Sequential(self.conv_block, self.repara)
        self.Q_linear = Q_MLP(embedding_dim, action_space, flatten=False)

    def forward_conv(self, x: Tensor):
        return self.encoder(x)

    def forward(self, x: Tensor):
        x, mu, std = self.encoder(x)
        return self.Q_linear(x), x, mu, std
