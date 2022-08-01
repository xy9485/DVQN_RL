import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T


class DQN_paper(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, action_space, n_latent_channel) -> None:
        super().__init__()
        n_input_channels = observation_space.shape[-1]

        # test_layer = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
        # print("test_layer.weight.size():", test_layer.weight.size())

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, n_latent_channel, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.flatten_layer = nn.Flatten()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            x = observation_space.sample()
            x = T.ToTensor()(x).unsqueeze(0)
            x = self.cnn(x.float())
            x = self.flatten_layer(x)
            n_flatten = x.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(512, action_space.n),
            # nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.flatten_layer(x)
        return self.linear(x)


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

    def forward(self, input):
        return input + self.resblock(input)


class DQN_Big(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, action_space, n_latent_channel) -> None:
        super().__init__()

        modules = []
        hidden_dims = [32, 64]
        in_channels = observation_space.shape[-1]
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
                nn.Conv2d(in_channels, n_latent_channel, kernel_size=1, stride=1),
                nn.LeakyReLU(),
            )
        )
        self.cnn = nn.Sequential(*modules)

        self.flatten_layer = nn.Flatten()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            x = observation_space.sample()
            x = T.ToTensor()(x).unsqueeze(0)
            x = self.cnn(x.float())
            x = self.flatten_layer(x)
            n_flatten = x.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(512, action_space.n),
            # nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.flatten_layer(x)
        return self.linear(x)


class DVN_paper(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box) -> None:
        super().__init__()
        n_input_channels = observation_space.shape[-1]

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
            x = self.cnn(T.ToTensor()(observation_space.sample()).unsqueeze(0).float())
            x = self.flatten_layer(x)
            n_flatten = x.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.flatten_layer(self.cnn(observations)))


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = x.to(device)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = F.relu(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc2(x))
        return self.head(x)


class DQN_MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN_MLP, self).__init__()
        # print("input_dim", input_dim)
        self.flatten_layer = nn.Flatten()
        self.fc1 = nn.Linear(np.prod(input_dim), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 256)
        self.head = nn.Linear(256, output_dim)

    def forward(self, x):
        # x = x.to(device)
        x = self.flatten_layer(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        return self.head(x)


class DVN(nn.Module):
    def __init__(self, input_dim):
        super(DVN, self).__init__()
        self.flatten_layer = nn.Flatten()
        self.fc1 = nn.Linear(np.prod(input_dim), 256)
        self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 256)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        # x = x.to(device)
        x = self.flatten_layer(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        return self.head(x)


class DQN_MLP(nn.Module):
    def __init__(self, input_dim, action_space):
        super().__init__()
        self.flatten_layer = nn.Flatten()
        self.linears = nn.Sequential(
            nn.Linear(np.prod(input_dim), 512),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(512, action_space.n),
            # nn.ReLU(),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.flatten_layer(x)
        return self.linears(x)
