import datetime
import os
import time

import torch

# from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

# from .types_ import *
from typing import Any, Dict, List, Optional, Tuple, Type, Union, TypeVar

from common.utils import wandb_log_image

Tensor = TypeVar("Tensor")


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


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        num_embeddings: int,
        # device: torch.device,
        hidden_dims: List = None,
        beta: float = 0.25,
        img_size: int = 64,
        reconstruction_path=None,
        **kwargs,
    ) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta
        self.reconstruction_path = reconstruction_path
        # self.device = device
        self.forward_call = 0
        self.initial_in_channels = in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
        )

        for _ in range(1):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU(),
            )
        )

        self.encoder = nn.Sequential(*modules)

        # self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)
        self.vq_layer = VectorQuantizerEMA(num_embeddings, embedding_dim)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1),
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
                    out_channels=self.initial_in_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.Tanh(),
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        # input = input.to(self.device)
        encoded = self.encode(input)[0]
        # quantized_inputs, vq_loss = self.vq_layer(encoded)
        quantized_inputs, vq_loss, _, _ = self.vq_layer(encoded)  # EMA
        recon = self.decode(quantized_inputs)
        # print(type(recon), recon.size())
        if (
            self.forward_call % 10000 == 0
            and self.reconstruction_path
            # and self.initial_in_channels == 3
        ):
            # current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
            # save_to = os.path.join(self.reconstruction_path, f"recon_{current_time}.png")
            # save_image(input[:8],save_to)
            # save_image(recon[:8], save_to)
            # print("save input and recon to path: ", save_to)
            stacked = torch.cat((recon[:7, :1], input[:1, :1]), dim=0)
            wandb_log_image(stacked)
            print("log recons as image to wandb")

        self.forward_call += 1
        return [recon, quantized_inputs, encoded, vq_loss]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "VQ_Loss": vq_loss}

    def sample(self, num_samples: int, current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning("VQVAE sampler is not implemented.")

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


if __name__ == "__main__":
    from torchsummary import summary

    # model = VQVAE(3, 64, 128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VQVAE(
        in_channels=3,
        embedding_dim=16,
        num_embeddings=64,
        reconstruction_path=None,
    ).to(device)

    # summary(model, (3, 64, 64))
    # print(model)
    summary(model.encoder, (3, 84, 84))
    summary(model.decoder, (16, 21, 21))
    # summary(model.vq_layer, (16, 21, 21))
    # print(model.encoder.parameters)
    # print(model.encoder._parameters)
    # print(model.encoder._modules)
