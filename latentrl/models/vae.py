import torch
import torch.nn as nn
from torchvision.utils import save_image
import time
import os


class VAE(nn.Module):
    def __init__(self, img_channels, latent_dims, hidden_dims=None, vae_version=None):
        super().__init__()
        self.img_channels = img_channels
        self.latent_dims = latent_dims
        self.vae_version = vae_version
        os.makedirs(f"../reconstruction/{self.vae_version}/", exist_ok=True)
        self.forward_call = 0
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]   # last layer of encoder hidden_dims[-1] * 4
            # hidden_dims = [32, 64, 128, 256]    # last layer of encoder hidden_dims[-1] * 16
        self.encoder = Encoder(img_channels, latent_dims, hidden_dims)
        self.decoder = Decoder(img_channels, latent_dims, hidden_dims)

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
        # print("input:")
        # print(x)
        mu, logvar = self.encoder(x)
        # sigma = logsigma.exp()
        z = self.reparameterize(mu, logvar)
        y = self.decoder(z)
        if self.forward_call % 5000 == 0:
            print("save input and recon")
            # save_image(x[:8], f"../reconstruction/{self.vae_version}/input_{time.time()}.png")
            save_image(y[:8], f"../reconstruction/{self.vae_version}/recon_{time.time()}.png")
        self.forward_call += 1
        return y, mu, logvar

    def loss_function(self,recon_x, x, mu, logvar):
        recon_loss = nn.MSELoss(size_average=False)
        BCE = recon_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD

    def sample_and_decode(self, current_device, num_samples=1):
        z = torch.randn(num_samples,
                        self.latent_dims)

        z = z.to(current_device)

        samples = self.decoder(z)
        return samples

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dims, hidden_dims):
        super().__init__()

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dims)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dims)

    def forward(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, in_channels, latent_dims, hidden_dims):
        super().__init__()

        self.decoder_input = nn.Linear(latent_dims, hidden_dims[-1] * 4)

        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

