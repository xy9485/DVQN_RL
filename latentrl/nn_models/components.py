from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MlpModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU) -> None:
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        elif hidden_dims is None:
            hidden_dims = []
        blocks = []
        for n_in, n_out in zip([input_dim] + hidden_dims[:-1], hidden_dims):
            if activation:
                blocks.extend([nn.Linear(n_in, n_out), activation()])
            else:
                blocks.extend([nn.Linear(n_in, n_out)])

        last_hidden_dim = hidden_dims[-1] if hidden_dims else input_dim
        blocks.extend([nn.Linear(last_hidden_dim, output_dim)])

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


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


class ResidualLinearLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out),
            nn.ReLU(True),
            nn.Linear(dim_out, dim_out),
            nn.LayerNorm(dim_out),
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
    ):
        super().__init__()
        blocks = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm)
        ]
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_channels))

        # blocks.append(nn.LeakyReLU())
        blocks.append(nn.ReLU())
        # blocks.append(nn.ELU())
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


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
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
