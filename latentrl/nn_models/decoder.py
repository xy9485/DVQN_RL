from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
from torch import Tensor, nn

from nn_models.components import DeConvBlock, ResidualLayer


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
