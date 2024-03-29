import datetime
import os
import time
import gym

import torch

# from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

# from .types_ import *
from typing import Any, Dict, List, Optional, Tuple, Type, Union, TypeVar
import torchvision.transforms as T

from common.utils import wandb_log_image

Tensor = TypeVar("Tensor")


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


class VQVAE2(nn.Module):
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
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta
        self.reconstruction_path = reconstruction_path
        # self.device = device
        self.forward_call = 0
        self.n_input_channels = in_channels

        # modules = []
        # if hidden_dims is None:
        #     hidden_dims = [64, 128]

        # # Build Encoder
        # in_channels = self.n_input_channels
        # for h_dim in hidden_dims:
        #     modules.append(
        #         nn.Sequential(
        #             nn.Conv2d(
        #                 in_channels,
        #                 out_channels=h_dim,
        #                 kernel_size=4,
        #                 stride=2,
        #                 padding=1,
        #             ),
        #             nn.LeakyReLU(),
        #         )
        #     )
        #     in_channels = h_dim

        # modules.append(
        #     nn.Sequential(
        #         nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        #         nn.LeakyReLU(),
        #     )
        # )

        # for _ in range(2):
        #     modules.append(ResidualLayer(in_channels, in_channels))
        # modules.append(nn.LeakyReLU())

        # modules.append(
        #     nn.Sequential(
        #         nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
        #         nn.LeakyReLU(),
        #     )
        # )

        # self.encoder = nn.Sequential(*modules)

        # self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)

        # # Build Decoder
        # modules = []
        # modules.append(
        #     nn.Sequential(
        #         nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1),
        #         nn.LeakyReLU(),
        #     )
        # )

        # for _ in range(2):
        #     modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        # modules.append(nn.LeakyReLU())

        # hidden_dims.reverse()

        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(
        #                 hidden_dims[i],
        #                 hidden_dims[i + 1],
        #                 kernel_size=4,
        #                 stride=2,
        #                 padding=1,
        #             ),
        #             nn.LeakyReLU(),
        #         )
        #     )

        # modules.append(
        #     nn.Sequential(
        #         nn.ConvTranspose2d(
        #             hidden_dims[-1],
        #             out_channels=self.n_input_channels,
        #             kernel_size=4,
        #             stride=2,
        #             padding=1,
        #         ),
        #         nn.Tanh(),
        #     )
        # )

        # self.decoder = nn.Sequential(*modules)

        # ========================================================================================
        # use dqn of the original paper as encoder of vqvae
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # nn.Conv2d(64, embedding_dim, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(),
        )

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(embedding_dim, 64, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(),
            nn.ConvTranspose2d(embedding_dim, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.n_input_channels, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
        )

        # ========================================================================================

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return result

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
        encoded = self.encode(input)
        quantized_inputs, vq_loss = self.vq_layer(encoded)
        recon = self.decode(quantized_inputs)
        # print(type(recon), recon.size())
        if (
            self.forward_call % 10000 == 0
            and self.reconstruction_path
            # and self.n_input_channels == 3
        ):
            current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
            save_to = os.path.join(self.reconstruction_path, f"recon_{current_time}.png")
            # save_image(input[:8],save_to)
            # save_image(recon[:8], save_to)
            # print("save input and recon to path: ", save_to)
            # log recon as image to wandb
            stacked = torch.cat((recon[:7, :1], input[:1, :1]), dim=0)
            wandb_log_image(stacked)
            # take the first channel to log no matter how many frame stacked
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

    env_id = "Boxing-v0"
    env = gym.make(env_id)
    print(env.observation_space.shape)
    model = VQVAE2(
        observation_space=env.observation_space,
        in_channels=3,
        embedding_dim=32,
        num_embeddings=64,
        reconstruction_path=None,
    ).to(device)

    # summary(model, (3, 84, 84))
    # print(model)
    summary(model.encoder, (3, 64, 64))
    # summary(model.decoder, (32, 7, 7))
    # print(model.encoder.parameters)
    # print(model.encoder._parameters)
    # print(model.encoder._modules)
    print("model summary done")
