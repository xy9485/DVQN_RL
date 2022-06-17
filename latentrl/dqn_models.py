import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T


class DQN_paper(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, action_space) -> None:
        super().__init__()
        n_input_channels = observation_space.shape[-1]

        # test_layer = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
        # print("test_layer.weight.size():", test_layer.weight.size())

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            temp = observation_space.sample()
            temp = T.ToTensor()(temp)
            temp = temp.unsqueeze(0)
            n_flatten = self.cnn(temp.float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_flatten),
            nn.ReLU(),
            nn.Linear(n_flatten, action_space.n),
            # nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


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
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                T.ToTensor()(observation_space.sample()).unsqueeze(0).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, 1), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


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
        self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 256)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        # x = x.to(device)
        x = self.flatten_layer(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        return self.head(x)
