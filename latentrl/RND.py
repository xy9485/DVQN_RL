import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from nn_models.components import MlpModel
from nn_models.encoder import make_encoder


class RNDNet(nn.Module):
    def __init__(self, env, config, feature_dim):
        super().__init__()
        self.encoder = make_encoder(
            input_format=config.input_format,
            input_channels=min(env.observation_space.shape),
            linear_dims=config.grd_encoder_linear_dims,
            observation_space=env.observation_space,
            hidden_channels=config.grd_hidden_channels,
        )
        self.linear = MlpModel(
            input_dim=self.encoder.linear_out_dim,
            hidden_dims=None,
            output_dim=feature_dim,
            activation=nn.ReLU,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        return x


class RND:
    def __init__(self, env, config, feature_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target = RNDNet(env, config, feature_dim).to(self.device)
        self.predictor = RNDNet(env, config, feature_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=0.00005)
        # freeze the network parameters
        # for p in self.target.parameters():
        #     p.requires_grad = False
        self.beta = 1.0
        self.beta_t = self.beta
        self.kappa = 1e-5

    @torch.no_grad()
    def compute_int_reward(self, x, time_steps):
        x = torch.as_tensor(np.array(x)).unsqueeze(0).to(self.device)
        y_true = self.target(x)
        y_pred = self.predictor(x)
        reward = torch.pow(y_pred - y_true, 2).sum() / 2
        # reward = F.mse_loss(y_pred, y_true)
        self.beta_t = self.beta * np.power(1.0 - self.kappa, time_steps)
        return reward * self.beta_t

    def update(self, x):
        y_true = self.target(x).detach()
        y_pred = self.predictor(x)
        loss = F.mse_loss(y_pred, y_true)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
