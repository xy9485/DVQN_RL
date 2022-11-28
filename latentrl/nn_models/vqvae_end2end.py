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


class VectorQuantizerLinear(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        # try detach
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        """
        latent # [B x D]
        """

        # Compute L2 distance between latents and embedding weights
        dist = (
            torch.sum(latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(latents, self.embedding.weight.t())
        )  # [B x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [B, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [B x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [B, D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        avg_probs = torch.mean(encoding_one_hot, dim=0)

        # Compute vq entropy

        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        entrophy_vq = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        # vq_probs = dist / torch.sum(dist, dim=1, keepdim=True)
        # vq_probs = -vq_probs * torch.log(vq_probs + 1e-10)
        # entrophy_vq = -torch.sum(vq_probs, dim=1).mean()

        # entrophy_vq = torch.std(dist, dim=1).mean()
        cluster_metric = dist[encoding_one_hot.bool()].mean().item()
        output_dict = {
            "hard_encoding_inds": encoding_inds,
            "hard_quantized_latents": quantized_latents,
            "cluster_metric": cluster_metric,
        }

        return (
            quantized_latents,  # [B x D]
            vq_loss,
            entrophy_vq,
            output_dict,
        )


class VectorQuantizerLinearSoft(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, beta: float = 0.25, softmin_beta: float = 10
    ):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.softmin_beta = softmin_beta

        self.embedding = nn.Embedding(self.K, self.D)
        # try detach
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        """
        latent # [B x D]
        """

        # Compute L2 distance between latents and embedding weights
        dist = (
            torch.sum(latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(latents, self.embedding.weight.t())
        )  # [B x K]

        # [Get the encoding that has the min distance]
        # encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [B, 1]
        encoding_inds = F.softmin(self.softmin_beta * dist, dim=1)

        # [Quantize the latents]
        quantized_latents = torch.matmul(encoding_inds, self.embedding.weight)  # [B, D]

        # [Compute the VQ Losses[]
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        # quantized_latents = latents + (quantized_latents - latents).detach()
        avg_probs = torch.mean(encoding_inds, dim=0)

        # [Compute vq entropy]
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        entrophy_vq = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        # vq_probs = dist / torch.sum(dist, dim=1, keepdim=True)
        # vq_probs = -vq_probs * torch.log(vq_probs + 1e-10)
        # entrophy_vq = -torch.sum(vq_probs, dim=1).mean()

        # entrophy_vq = torch.std(dist, dim=1).mean()

        # dist_std = torch.std(dist, dim=1)
        # entrophy_vq = entrophy_vq + dist_std.mean()

        # [Get Hard Quantization]
        device = latents.device
        hard_encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [B, 1]
        encoding_one_hot = torch.zeros(hard_encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, hard_encoding_inds, 1)  # [B x K]
        hard_quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [B, D]
        cluster_metric = dist[encoding_one_hot.bool()].mean().item()
        output_dict = {
            "hard_encoding_inds": hard_encoding_inds,
            "hard_quantized_latents": hard_quantized_latents,
            "cluster_assignment": encoding_inds,
            "cluster_metric": cluster_metric,
        }

        return (
            quantized_latents,  # [B x D]
            vq_loss,
            entrophy_vq,
            output_dict,
        )


class VectorQuantizerLinearDiffable(nn.Module):
    """
    make argmin differantiable
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, beta: float = 0.25, softmin_beta: float = 10
    ):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.softmin_beta = softmin_beta

        self.embedding = nn.Embedding(self.K, self.D)
        # try detach
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        """
        latent # [B x D]
        """

        # Compute L2 distance between latents and embedding weights
        dist = (
            torch.sum(latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(latents, self.embedding.weight.t())
        )  # [B x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [B, 1]
        encoding_inds_soft = F.softmin(self.softmin_beta * dist, dim=1)

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [B x K]

        encoding_one_hot = (encoding_one_hot - encoding_inds_soft).detach() + encoding_inds_soft

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [B, D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss
        avg_probs = torch.mean(encoding_one_hot, dim=0)

        # Compute vq entropy
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        entrophy_vq = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        # vq_probs = dist / torch.sum(dist, dim=1, keepdim=True)
        # vq_probs = -vq_probs * torch.log(vq_probs + 1e-10)
        # entrophy_vq = -torch.sum(vq_probs, dim=1).mean()

        # entrophy_vq = torch.std(dist, dim=1).mean()
        cluster_metric = dist[encoding_one_hot.bool()].mean().item()
        output_dict = {
            "hard_encoding_inds": encoding_inds,
            "hard_quantized_latents": quantized_latents,
            "cluster_assignment": encoding_inds_soft,
            "cluster_metric": cluster_metric,
        }
        return (
            quantized_latents,  # [B x D]
            vq_loss,
            entrophy_vq,
            output_dict,
        )


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

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
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity, encodings


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
