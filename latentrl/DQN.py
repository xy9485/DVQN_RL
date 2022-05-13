import sys
import os
from os import makedirs
import time
from tkinter import N
from types import SimpleNamespace
import colored_traceback.auto
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from tensorboard import summary

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from data2 import RolloutDatasetNaive

from vqvae_end2end import VQVAE
from utils.learning import EarlyStopping, ReduceLROnPlateau
from utils.misc import get_linear_fn, make_vec_env_customized
from wrappers import (
    ActionRepetitionWrapper,
    EncodeStackWrapper,
    FrameStackWrapper,
    PreprocessObservationWrapper,
    pack_env_wrappers,
)

from transforms import transform_dict
from torchsummary import summary
import wandb
import GPUtil
from pprint import pprint
import datetime


def make_env(
    env_id,
    config,
):
    env = gym.make(env_id).unwrapped

    wrapper_class_list = [
        # ActionDiscreteWrapper,
        ActionRepetitionWrapper,
        # EpisodeEarlyStopWrapper,
        # Monitor,
        # CarRandomStartWrapper,
        PreprocessObservationWrapper,
        # EncodeStackWrapper,
        # PunishRewardWrapper,
        FrameStackWrapper,
    ]
    wrapper_kwargs_list = [
        {"action_repetition": config.action_repetition},
        # {"max_neg_rewards": max_neg_rewards, "punishment": punishment},
        # {'filename': monitor_dir},
        # {"filename": os.path.join(monitor_dir, "train")},  # just single env in this case
        # {
        #     "warm_up_steps": hparams["learning_starts"],
        #     "n_envs": n_envs,
        #     "always_random_start": always_random_start,
        #     "no_random_start": no_random_start,
        # },
        {
            "vertical_cut_d": 84,
            "shape": 84,
            "num_output_channels": 3,
        },
        # {
        #     "n_stack": n_stack,
        #     "vae_f": vae_path,
        #     "vae_sample": vae_sample,
        #     "vae_inchannel": vae_inchannel,
        #     "latent_dim": vae_latent_dim,
        # },
        # {'max_neg_rewards': max_neg_rewards, "punishment": punishment}
        {"n_frame_stack": config.n_frame_stack},
    ]

    wrapper = pack_env_wrappers(wrapper_class_list, wrapper_kwargs_list)
    env = wrapper(env)
    env.seed(config.seed)

    return env


# Q-Network


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

        self.linear = nn.Sequential(nn.Linear(n_flatten, action_space.n), nn.ReLU())

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
        self.fc4 = nn.Linear(256, 256)
        self.head = nn.Linear(256, output_dim)

    def forward(self, x):
        # x = x.to(device)
        x = self.flatten_layer(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.head(x)


class DVN(nn.Module):
    def __init__(self, input_dim):
        super(DVN, self).__init__()
        self.flatten_layer = nn.Flatten()
        self.fc1 = nn.Linear(np.prod(input_dim), 128)
        self.fc2 = nn.Linear(128, 256)
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


class Agent:
    def __init__(
        self,
        env,
        config,
    ):
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        # self.save_dir = save_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = int(time.time())
        np.random.seed(seed)
        torch.manual_seed(seed)

        # >>>>>>>>>>>VQVAE>>>>>>>>>>>>>
        # game_name = "Boxing-v0"
        # vae_version = "vqvae_c3_embedding16x64_3_end2end"
        # reconstruction_path = os.path.join(
        #     "/workspace/repos_dev/VQVAE_RL/reconstruction", game_name, vae_version
        # )
        if config.reconstruction_path:
            os.makedirs(
                config.reconstruction_path,
                exist_ok=True,
            )
        self.vqvae_model = VQVAE(
            in_channels=config.vqvae_inchannel,
            embedding_dim=config.vqvae_latent_channel,
            num_embeddings=config.vqvae_num_embeddings,
            reconstruction_path=config.reconstruction_path,
        ).to(self.device)

        self.vqvae_optimizer = optim.Adam(self.vqvae_model.parameters(), lr=5e-4)
        # or Adam

        # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

        self.scheduler = ReduceLROnPlateau(self.vqvae_optimizer, "min")

        self.earlystopping = EarlyStopping("min", patience=30)
        # <<<<<<<<<<<VQVAE<<<<<<<<<<<<

        # DQN_MLP
        # Get number of actions from gym action space
        self.n_actions = env.action_space.n
        self.obs_height = env.observation_space.shape[0]
        self.obs_width = env.observation_space.shape[1]

        self.policy_mlp_net = DQN_MLP(16 * 21 * 21, self.n_actions).to(self.device)
        self.target_mlp_net = DQN_MLP(16 * 21 * 21, self.n_actions).to(self.device)
        # or
        # self.policy_mlp_net = DQN_paper(env.observation_space, env.action_space).to(self.device)
        # self.target_mlp_net = DQN_paper(env.observation_space, env.action_space).to(self.device)

        self.target_mlp_net.load_state_dict(self.policy_mlp_net.state_dict())
        self.target_mlp_net.eval()

        self.dqn_optimizer = optim.Adam(self.policy_mlp_net.parameters(), lr=5e-4)
        # RMSProp

        print(self.policy_mlp_net)
        # summary(policy_mlp_net, (3, obs_height, obs_width))

        # self.use_cuda = torch.cuda.is_available()
        # if self.use_cuda:
        #     self.net = self.net.to(device="cuda")

        # self.curr_step = 0
        self.total_steps_done = 0
        self.episodes_done = 0

        # Hyperparameters
        self.num_episodes_train = config.num_episodes_train
        self.batch_size = config.batch_size
        self.validation_size = config.validation_size
        self.size_replay_memory = config.size_replay_memory
        self.gamma = config.gamma
        self.eps_start = config.eps_start
        self.eps_end = config.eps_end
        self.eps_decay = config.eps_decay
        self.target_update = config.target_update

        self.exploration_rate = config.exploration_rate
        self.exploration_rate_decay = config.exploration_rate_decay
        self.exploration_rate_min = config.exploration_rate_min

        self.save_model_every = config.save_model_every

        # Initialize experience replay buffer
        self.memory = ReplayMemory(self.size_replay_memory)
        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "done")
        )

        self.init_steps = config.init_steps  # min. experiences before training
        self.learn_every = config.learn_every  # no. of experiences between updates to Q_online
        self.sync_every = config.sync_every  # no. of experiences between updates to Q_target
        self.validate_every = config.validate_every
        # self.exploration_schedule = get_linear_fn(
        #     self.exploration_initial_eps,
        #     self.exploration_final_eps,
        #     self.exploration_fraction,
        # )

        self._current_progress_remaining = 1.0
        self.n_call_learn = 0
        self.eps_threshold = None

    @torch.no_grad()
    def act(self, state):
        if self.episodes_done < math.ceil(0.99 * self.num_episodes_train):
            self.eps_threshold = (
                self.eps_start - self.eps_start / self.num_episodes_train * self.episodes_done
            )
        # linear schedule on epsilon
        # eps_threshold = (self.eps_end + (self.eps_start - self.eps_end) / math.ceil(self.train_episode * 0.9)
        #                  * self.episodes_done)

        # exploration rate exponential decay
        # self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
        #     -1.0 * self.total_steps_done / self.eps_decay
        # )

        # # decrease exploration_rate
        # self.exploration_rate *= self.exploration_rate_decay
        # self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        state = T.ToTensor()(state).float().unsqueeze(0).to(self.device)
        latent_state = self.vqvae_model.encoder(state)
        quantized_latent_state = self.vqvae_model.vq_layer(latent_state)[0]

        self.total_steps_done += 1

        sample = random.random()
        if sample > self.eps_threshold:
            # with torch.no_grad():
            action = self.policy_mlp_net(quantized_latent_state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )

        return action

    def cache(self, state, action, next_state, reward, done):
        """Add the experience to memory"""
        state = T.ToTensor()(state).float().unsqueeze(0)
        next_state = T.ToTensor()(next_state).float().unsqueeze(0)
        reward = torch.tensor([reward])

        self.memory.push(state, action, next_state, reward, done)

    def recall(self):
        """Sample experiences from memory"""
        pass

    def td_estimate(self, state, action):
        pass

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        pass

    def update_Q_online(self, td_estimate, td_target):
        pass

    def sync_Q_target(self):
        self.target_mlp_net.load_state_dict(self.policy_mlp_net.state_dict())

    def validate_vqvae(self):
        batch = self.memory.sample(validation_size=self.validation_size)

        state_batch = torch.cat(batch.state).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        recon_batch, quantized, input, vqloss = self.vqvae_model(state_batch)
        recon_batch2, quantized2, input2, vqloss2 = self.vqvae_model(next_state_batch)

        recon_loss = F.mse_loss(recon_batch, state_batch)
        recon_loss2 = F.mse_loss(recon_batch2, next_state_batch)

        validate_loss = (vqloss + vqloss2 + recon_loss + recon_loss2) / 2

        return validate_loss

    def learn(self, tb_writer):
        if self.total_steps_done % self.sync_every == 0:
            self.sync_Q_target()

        if self.total_steps_done % self.save_model_every == 0:
            pass
            # self.save()

        if self.total_steps_done < self.init_steps:
            return None, None, None

        if self.total_steps_done % self.learn_every != 0:
            return None, None, None

        if self.n_call_learn % self.validate_every == 0:
            # validate_loss = self.validate_vqvae()
            # self.scheduler.step(validate_loss)
            # self.earlystopping.step(validate_loss)
            pass

        """Update online action value (Q) function with a batch of experiences"""
        batch = self.memory.sample(batch_size=self.batch_size)

        # state = []
        # next_state = []
        # for state, next_state in zip(batch.state, batch.next_state):
        #     state.append(torch.tensor(state).float().unsqueeze(0).to(self.device))
        #     next_state.append(torch.tensor(next_state).float().unsqueeze(0).to(self.device))

        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        done_batch = torch.tensor(batch.done).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run sample()")
        recon_batch, quantized, input, vqloss = self.vqvae_model(state_batch)
        recon_batch2, quantized2, input2, vqloss2 = self.vqvae_model(next_state_batch)
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run vqvae_model")

        recon_loss = F.mse_loss(recon_batch, state_batch)
        recon_loss2 = F.mse_loss(recon_batch2, next_state_batch)

        current_Q = self.policy_mlp_net(quantized).gather(1, action_batch)
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run policy_mlp_net")

        # Compute next_state_max_Q
        # next_state_max_Q = torch.zeros(self.batch_size, device=self.device)
        # non_final_next_states = quantized2[~done_batch]
        # next_state_max_Q[~done_batch] = self.target_mlp_net(
        #     non_final_next_states).max(1)[0].detach()
        # target_Q = (
        #     next_state_max_Q * self.gamma) + reward_batch

        # Or Compute next_state_max_Q
        with torch.no_grad():
            next_state_max_Q = self.target_mlp_net(quantized2).max(1)[0].detach()

            target_Q = (
                reward_batch + (1 - done_batch.float()) * self.gamma * next_state_max_Q
            ).float()

        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run target_Q")

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        q_loss = criterion(current_Q, target_Q.unsqueeze(1))
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run q_loss")

        if not self.earlystopping.stop:
            total_loss = 0.5 * (vqloss + vqloss2 + recon_loss + recon_loss2) + q_loss

        else:
            total_loss = q_loss

            # make sure q_loss no longer effect weights of the encoder
            for param in self.vqvae_model.parameters():
                param.requires_grad = False

            self.dqn_optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            self.dqn_optimizer.step()

            for name, param in self.policy_mlp_net.named_parameters():
                tb_writer.add_histogram(f"gradients/dqn/{name}", param.grad, self.n_call_learn)
                tb_writer.add_histogram(f"weight_bias/dqn/{name}", param, self.n_call_learn)

            self.n_call_learn += 1
            q_loss = q_loss.item()
            return None, None, q_loss

        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("get total_loss")

        # Optimize the model
        self.dqn_optimizer.zero_grad(set_to_none=True)
        self.vqvae_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")

        # 1 clamp gradients to avoid exploding gradient
        # for param in self.policy_mlp_net.parameters():
        #     param.grad.data.clamp_(-1, 1)

        # for param in self.vqvae_model.parameters():
        #     param.grad.data.clamp_(-1, 1)

        # 2 Clip gradient norm
        # max_grad_norm = 10
        # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
        # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)

        self.dqn_optimizer.step()
        self.vqvae_optimizer.step()

        self.n_call_learn += 1

        if self.n_call_learn % 500 == 0:
            for name, param in self.policy_mlp_net.named_parameters():
                tb_writer.add_histogram(f"gradients/dqn/{name}", param.grad, self.n_call_learn)
                tb_writer.add_histogram(f"weight_bias/dqn/{name}", param, self.n_call_learn)
            for name, param in self.vqvae_model.encoder.named_parameters():
                tb_writer.add_histogram(f"gradients/encoder/{name}", param.grad, self.n_call_learn)
                tb_writer.add_histogram(f"weight_bias/encoder/{name}", param, self.n_call_learn)
            for name, param in self.vqvae_model.decoder.named_parameters():
                tb_writer.add_histogram(f"gradients/decoder/{name}", param.grad, self.n_call_learn)
                tb_writer.add_histogram(f"weight_bias/decoder/{name}", param, self.n_call_learn)

        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run step()")

        # print("learn() finished")
        mean_recon_loss = ((recon_loss + recon_loss2) / 2).item()
        mean_vq_loss = ((vqloss + vqloss2) / 2).item()
        q_loss = q_loss.item()
        return mean_recon_loss, mean_vq_loss, q_loss

    def _update_current_progress_remaining(self, num_timesteps, total_timesteps):
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)


class DuoLayerAgent:
    def __init__(
        self,
        env,
        config,
    ):
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        # self.save_dir = save_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = int(time.time())
        np.random.seed(seed)
        torch.manual_seed(seed)

        # >>>>>>>>>>>VQVAE>>>>>>>>>>>>>
        # game_name = "Boxing-v0"
        # vae_version = "vqvae_c3_embedding16x64_3_end2end"
        # reconstruction_path = os.path.join(
        #     "/workspace/repos_dev/VQVAE_RL/reconstruction", game_name, vae_version
        # )
        if config.reconstruction_path:
            os.makedirs(
                config.reconstruction_path,
                exist_ok=True,
            )
        self.vqvae_model = VQVAE(
            in_channels=config.vqvae_inchannel,
            embedding_dim=config.vqvae_latent_channel,
            num_embeddings=config.vqvae_num_embeddings,
            reconstruction_path=config.reconstruction_path,
        ).to(self.device)

        self.vqvae_optimizer = optim.Adam(self.vqvae_model.parameters(), lr=5e-4)
        # or Adam

        # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

        self.scheduler = ReduceLROnPlateau(self.vqvae_optimizer, "min")

        self.earlystopping = EarlyStopping("min", patience=5)
        # <<<<<<<<<<<VQVAE<<<<<<<<<<<<

        # DQN_MLP
        # Get number of actions from gym action space
        self.n_actions = env.action_space.n
        self.obs_height = env.observation_space.shape[0]
        self.obs_width = env.observation_space.shape[1]

        # define ground leve policy
        self.ground_Q_net = DQN(84, 84, self.n_actions).to(self.device)
        self.ground_target_Q_net = DQN(84, 84, self.n_actions).to(self.device)
        self.ground_target_Q_net.load_state_dict(self.ground_Q_net.state_dict())
        self.ground_target_Q_net.eval()

        self.ground_Q_optimizer = optim.Adam(self.ground_Q_net.parameters(), lr=5e-4)

        # define latent level value network
        self.abstract_V_net = DVN(16 * 21 * 21).to(self.device)
        self.abstract_target_V_net = DVN(16 * 21 * 21).to(self.device)
        self.abstract_target_V_net.load_state_dict(self.abstract_V_net.state_dict())
        self.abstract_target_V_net.eval()

        self.abstract_V_optimizer = optim.Adam(self.abstract_V_net.parameters(), lr=5e-4)

        # define latent level policy network
        # self.abstract_Q_net = DQN_MLP(16 * 21 * 21, self.n_actions).to(self.device)
        # self.abstract_target_Q_net = DQN_MLP(16 * 21 * 21, self.n_actions).to(self.device)
        # self.abstract_target_Q_net.load_state_dict(self.abstract_Q_net.state_dict())
        # self.abstract_target_Q_net.eval()

        # self.abstract_Q_optimizer = optim.Adam(self.abstract_Q_net.parameters(), lr=5e-4)
        # RMSProp

        # print(self.policy_mlp_net)
        # summary(policy_mlp_net, (3, obs_height, obs_width))

        # self.use_cuda = torch.cuda.is_available()
        # if self.use_cuda:
        #     self.net = self.net.to(device="cuda")

        # self.curr_step = 0
        self.total_steps_done = 0
        self.episodes_done = 0

        # Hyperparameters
        self.num_episodes_train = config.num_episodes_train
        self.batch_size = config.batch_size
        self.validation_size = config.validation_size
        self.size_replay_memory = config.size_replay_memory
        self.gamma = config.gamma
        self.omega = config.omega
        self.eps_start = config.eps_start
        self.eps_end = config.eps_end
        self.eps_decay = config.eps_decay
        self.target_update = config.target_update

        self.exploration_rate = config.exploration_rate
        self.exploration_rate_decay = config.exploration_rate_decay
        self.exploration_rate_min = config.exploration_rate_min

        self.save_model_every = config.save_model_every

        # Initialize experience replay buffer
        self.memory = ReplayMemory(self.size_replay_memory)
        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "done")
        )

        self.init_steps = config.init_steps  # min. experiences before training
        self.learn_every = config.learn_every  # no. of experiences between updates to Q_online
        self.sync_every = config.sync_every
        # no. of experiences between updates to abstract Q_target
        self.sync_ground_every = self.sync_every
        # no. of experiences between updates to ground Q_target
        self.validate_every = config.validate_every
        # self.exploration_schedule = get_linear_fn(
        #     self.exploration_initial_eps,
        #     self.exploration_final_eps,
        #     self.exploration_fraction,
        # )

        self._current_progress_remaining = 1.0
        self.n_call_learn = 0
        self.eps_threshold = None

    @torch.no_grad()
    def act(self, state):
        if self.episodes_done < math.ceil(0.99 * self.num_episodes_train):
            self.eps_threshold = (
                self.eps_start - self.eps_start / self.num_episodes_train * self.episodes_done
            )
        # linear schedule on epsilon
        # eps_threshold = (self.eps_end + (self.eps_start - self.eps_end) / math.ceil(self.train_episode * 0.9)
        #                  * self.episodes_done)

        # exploration rate exponential decay
        # self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
        #     -1.0 * self.total_steps_done / self.eps_decay
        # )

        # # decrease exploration_rate
        # self.exploration_rate *= self.exploration_rate_decay
        # self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        state = T.ToTensor()(state).float().unsqueeze(0).to(self.device)

        sample = random.random()
        if sample > self.eps_threshold:
            action = self.ground_Q_net(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )

        self.total_steps_done += 1

        return action

    def cache(self, state, action, next_state, reward, done):
        """Add the experience to memory"""
        state = T.ToTensor()(state).float().unsqueeze(0)
        next_state = T.ToTensor()(next_state).float().unsqueeze(0)
        reward = torch.tensor([reward])

        self.memory.push(state, action, next_state, reward, done)

    def recall(self):
        """Sample experiences from memory"""
        pass

    def td_estimate(self, state, action):
        pass

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        pass

    def update_Q_online(self, td_estimate, td_target):
        pass

    def sync_ground_Q_target(self):
        # self.abstract_target_Q_net.load_state_dict(self.abstract_Q_net.state_dict())
        self.ground_target_Q_net.load_state_dict(self.ground_Q_net.state_dict())

    def sync_abstract_V_target(self):
        self.abstract_target_V_net.load_state_dict(self.abstract_V_net.state_dict())

    def validate_vqvae(self):
        batch = self.memory.sample(validation_size=self.validation_size)

        state_batch = torch.cat(batch.state).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        recon_batch, quantized, input, vqloss = self.vqvae_model(state_batch)
        recon_batch2, quantized2, input2, vqloss2 = self.vqvae_model(next_state_batch)

        recon_loss = F.mse_loss(recon_batch, state_batch)
        recon_loss2 = F.mse_loss(recon_batch2, next_state_batch)

        validate_loss = (vqloss + vqloss2 + recon_loss + recon_loss2) / 2

        return validate_loss

    def learn(self, tb_writer):
        if self.total_steps_done % self.sync_every == 0:
            self.sync_ground_Q_target()
            self.sync_abstract_V_target()

        if self.total_steps_done % self.save_model_every == 0:
            pass
            # self.save()

        if self.total_steps_done < self.init_steps:
            return None, None, None

        if self.total_steps_done % self.learn_every != 0:
            return None, None, None

        if self.n_call_learn % self.validate_every == 0:
            validate_loss = self.validate_vqvae()
            self.scheduler.step(validate_loss)
            self.earlystopping.step(validate_loss)

        """Update online action value (Q) function with a batch of experiences"""
        batch = self.memory.sample(batch_size=self.batch_size)

        # state = []
        # next_state = []
        # for state, next_state in zip(batch.state, batch.next_state):
        #     state.append(torch.tensor(state).float().unsqueeze(0).to(self.device))
        #     next_state.append(torch.tensor(next_state).float().unsqueeze(0).to(self.device))

        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        done_batch = torch.tensor(batch.done).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        ground_current_Q = self.ground_Q_net(state_batch).gather(1, action_batch)

        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run sample()")
        recon_batch, quantized, input, vqloss = self.vqvae_model(state_batch)
        recon_batch2, quantized2, input2, vqloss2 = self.vqvae_model(next_state_batch)
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run vqvae_model")

        recon_loss = F.mse_loss(recon_batch, state_batch)
        recon_loss2 = F.mse_loss(recon_batch2, next_state_batch)

        # current_Q = self.abstract_Q_net(quantized).gather(1, action_batch)
        abstract_current_V = self.abstract_V_net(quantized)

        curent_composed = ground_current_Q + self.omega * abstract_current_V

        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run policy_mlp_net")

        # Compute next_state_max_Q
        # next_state_max_Q = torch.zeros(self.batch_size, device=self.device)
        # non_final_next_states = quantized2[~done_batch]
        # next_state_max_Q[~done_batch] = self.target_mlp_net(
        #     non_final_next_states).max(1)[0].detach()
        # target_Q = (
        #     next_state_max_Q * self.gamma) + reward_batch

        # Or Compute next_state_max_Q
        with torch.no_grad():
            next_ground_state_max_Q = self.ground_target_Q_net(next_state_batch).max(1)[0].detach()

            # next_abstract_state_max_Q = self.abstract_target_Q_net(quantized2).max(1)[0].detach()

            next_abstract_state_V = self.abstract_target_V_net(quantized2)

            # target_Q = (
            #     reward_batch + (1 - done_batch.float()) * self.gamma * next_state_max_Q
            # ).float()
            target_composed = reward_batch + (1 - done_batch.float()) * (
                self.omega * next_abstract_state_V.squeeze() + self.gamma * next_ground_state_max_Q
            )

        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run target_Q")

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        composed_loss = criterion(curent_composed, target_composed.unsqueeze(1))
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run q_loss")

        if not self.earlystopping.stop:
            total_loss = vqloss + vqloss2 + recon_loss + recon_loss2 + composed_loss

        else:
            total_loss = composed_loss

            # make sure q_loss no longer effect weights of the encoder
            for param in self.vqvae_model.parameters():
                param.requires_grad = False

            self.ground_Q_optimizer.zero_grad(set_to_none=True)
            self.abstract_V_optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.abstract_V_optimizer.parameters(), max_grad_norm)
            self.ground_Q_optimizer.step()
            self.abstract_V_optimizer.step()

            for name, param in self.ground_Q_net.named_parameters():
                tb_writer.add_histogram(f"gradients/ground_Q_net/{name}", param, self.n_call_learn)
                tb_writer.add_histogram(
                    f"weight_bias/ground_Q_net/{name}", param, self.n_call_learn
                )
            for name, param in self.abstract_V_net.named_parameters():
                tb_writer.add_histogram(
                    f"gradients/abstract_V_net/{name}", param.grad, self.n_call_learn
                )
                tb_writer.add_histogram(
                    f"weight_bias/abstract_V_net/{name}", param, self.n_call_learn
                )

            self.n_call_learn += 1

            composed_loss = composed_loss.item()
            return None, None, composed_loss

        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("get total_loss")

        # Optimize the model
        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        self.abstract_V_optimizer.zero_grad(set_to_none=True)
        self.vqvae_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")

        # 1 clamp gradients to avoid exploding gradient
        # for param in self.policy_mlp_net.parameters():
        #     param.grad.data.clamp_(-1, 1)

        # for param in self.vqvae_model.parameters():
        #     param.grad.data.clamp_(-1, 1)

        # 2 Clip gradient norm
        # max_grad_norm = 10
        # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
        # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()
        self.abstract_V_optimizer.step()
        self.vqvae_optimizer.step()

        self.n_call_learn += 1

        for name, param in self.abstract_V_net.named_parameters():
            tb_writer.add_histogram(
                f"gradients/abstract_V_net/{name}", param.grad, self.n_call_learn
            )
            tb_writer.add_histogram(f"weight_bias/abstract_V_net/{name}", param, self.n_call_learn)
        for name, param in self.vqvae_model.encoder.named_parameters():
            tb_writer.add_histogram(f"gradients/encoder/{name}", param.grad, self.n_call_learn)
            tb_writer.add_histogram(f"weight_bias/encoder/{name}", param, self.n_call_learn)
        for name, param in self.vqvae_model.decoder.named_parameters():
            tb_writer.add_histogram(f"gradients/decoder/{name}", param.grad, self.n_call_learn)
            tb_writer.add_histogram(f"weight_bias/decoder/{name}", param, self.n_call_learn)

        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run step()")

        # print("learn() finished")

        # return total_loss, q_loss

        mean_recon_loss = ((recon_loss + recon_loss2) / 2).item()
        mean_vq_loss = ((vqloss + vqloss2) / 2).item()
        composed_loss = composed_loss.item()
        return mean_recon_loss, mean_vq_loss, composed_loss

    def _update_current_progress_remaining(self, num_timesteps, total_timesteps):
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)


class ReplayMemory(object):
    def __init__(
        self,
        capacity,
    ):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "done")
        )

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(
        self,
        batch_size=None,
        validation_size=None,
    ):
        if validation_size:
            transitions = random.sample(self.memory, validation_size)
        else:
            transitions = random.sample(self.memory, batch_size)
        # This converts batch-array of Transitions
        # to Transition of batch-arrays.
        return self.Transition(*zip(*transitions))
        # return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


def train_single_layer():
    env_id = "Boxing-v0"  # CarRacing-v0, ALE/Skiing-v5, Boxing-v0
    vae_version = "vqvae_c3_embedding16x64_3_end2end_2"
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")

    for _ in range(1):
        # 🐝 initialise a wandb run
        wandb.init(
            project="vqvae+latent_rl",
            config={
                "env_id": "Boxing-v0",
                "total_time_steps": 1e6,
                "action_repetition": 2,
                "n_frame_stack": 1,
                # "lr": 1e-3,
                # "dropout": random.uniform(0.01, 0.80),
                "vqvae_inchannel": int(3 * 1),
                "vqvae_latent_channel": 16,
                "vqvae_num_embeddings": 64,
                "reconstruction_path": os.path.join(
                    "/workspace/repos_dev/VQVAE_RL/reconstruction/singlelayer", env_id, current_time
                ),
                # "reconstruction_path": None,
                "num_episodes_train": 4000,
                "batch_size": 128,
                "validation_size": 128,
                "validate_every": 10,
                "size_replay_memory": int(1e6),
                "gamma": 0.97,
                "eps_start": 0.9,
                "eps_end": 0.05,
                "eps_decay": 200,
                "target_update": 10,
                "exploration_rate": 0.1,
                "exploration_rate_decay": 0.99999975,
                "exploration_rate_min": 0.1,
                "save_model_every": 5e5,
                "init_steps": 1e4,
                "learn_every": 4,
                "sync_every": 8,
                "seed": int(time.time()),
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            },
        )
        config = wandb.config
        print(type(config))

        # config = {
        #     "env_id": "Boxing-v0",
        #     "total_time_steps": 1e6,
        #     "action_repetition": 2,
        #     "n_frame_stack": 4,
        #     # "lr": 1e-3,
        #     # "dropout": random.uniform(0.01, 0.80),
        #     "vqvae_inchannel": int(3 * 4),
        #     "vqvae_latent_channel": 16,
        #     "vqvae_num_embeddings": 64,
        #     # "reconstruction_path": os.path.join(
        #     #     "/workspace/repos_dev/VQVAE_RL/reconstruction", env_id, vae_version
        #     # ),
        #     "reconstruction_path": None,
        #     "num_episodes_train": 100,
        #     "batch_size": 128,
        #     "gamma": 0.99,
        #     "eps_start": 0.9,
        #     "eps_end": 0.05,
        #     "eps_decay": 200,
        #     "target_update": 10,
        #     "exploration_rate": 0.1,
        #     "exploration_rate_decay": 0.99999975,
        #     "exploration_rate_min": 0.1,
        #     "save_model_every": 5e5,
        #     "init_steps": 1e4,
        #     "learn_every": 4,
        #     "sync_every": 8,
        #     "seed": int(time.time()),
        #     "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # }
        # config = SimpleNamespace(**config)

        # The main training loop
        env = make_env(env_id, config)
        agent = Agent(env, config)
        print("agent.policy_mlp_net:", agent.policy_mlp_net)
        print("agent.vqvae_model:", agent.vqvae_model)

        wandb.watch(agent.policy_mlp_net, log_freq=100)
        # wandb.watch(agent.target_mlp_net, log_freq=100)
        wandb.watch(agent.vqvae_model, log_freq=100)

        comment = ""
        log_dir_tensorboard = f"/workspace/repos_dev/VQVAE_RL/log_tensorboard/singlelayer/{env_id}/{current_time}_{comment}"
        tb_writer = SummaryWriter(log_dir_tensorboard)
        print("log_dir_tensorboard:", log_dir_tensorboard)

        time_start_training = time.time()

        # transformer = transform_dict["Boxing-v0"]
        for i_episode in range(config.num_episodes_train):
            time_start_episode = time.time()
            # Initialize the environment and state
            state = env.reset()
            # last_screen = get_screen()
            # current_screen = get_screen()
            # state = current_screen - last_screen
            episodic_reward = 0
            episodic_negative_reward = 0
            episodic_non_zero_reward = 0
            loss_list = []
            for t in count():
                # Select and perform an action
                action = agent.act(state)
                # print(
                #     "memory_allocated: {:.5f} MB".format(
                #         torch.cuda.memory_allocated() / (1024 * 1024)
                #     )
                # )
                # print("agent.act")

                next_state, reward, done, _ = env.step(action.item())
                episodic_reward += reward
                if reward < 0:
                    episodic_negative_reward += reward
                else:
                    episodic_non_zero_reward += reward

                # # Observe new state
                # last_screen = current_screen
                # current_screen = get_screen()
                # if not done:
                #     next_state = current_screen - last_screen
                # else:
                #     next_state = None

                # Store the transition in memory
                # agent.memory.push(state, action, next_state, reward, done)
                agent.cache(state, action, next_state, reward, done)
                # print(
                #     "memory_allocated: {:.5f} MB".format(
                #         torch.cuda.memory_allocated() / (1024 * 1024)
                #     )
                # )
                # print("agent.cache")

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                # optimize_model()
                recon_loss, vq_loss, q_loss = agent.learn(tb_writer)
                loss_list.append([recon_loss, vq_loss, q_loss])
                # print(
                #     "memory_allocated: {:.5f} MB".format(
                #         torch.cuda.memory_allocated() / (1024 * 1024)
                #     )
                # )
                # print("agent.learn")
                if done:
                    # print("sys.getsizeof(agent.memory)", sys.getsizeof(agent.memory))
                    # print(torch.cuda.memory_reserved()/(1024*1024), "MB")
                    # print(torch.cuda.memory_allocated()/(1024*1024), "MB")

                    agent.episodes_done += 1
                    # episode_durations.append(t + 1)
                    # plot_durations()
                    metrics = {
                        "train/episodic_reward": episodic_reward,
                        "train/episodic_negative_reward": episodic_negative_reward,
                        "train/episodic_non_zero_reward": episodic_non_zero_reward,
                        "train/total_steps_done": agent.total_steps_done,
                        "train/time_elapsed": time.time() - time_start_training,
                        "train/episode_length": t + 1,
                        "train/episodes": i_episode,
                        "train/epsilon": agent.eps_threshold,
                        "train/episodic_fps": int((t + 1) / (time.time() - time_start_episode)),
                    }
                    wandb.log({**metrics})

                    print(">>>>>>>>>>>>>>>>Episode Done>>>>>>>>>>>>>>>>>")
                    print("time cost so far: {:.1f} s".format(time.time() - time_start_training))
                    print("episodic time cost: {:.1f} s".format(time.time() - time_start_episode))
                    print("agent.total_steps_done:", agent.total_steps_done)
                    print("episodic_fps:", int((t + 1) / (time.time() - time_start_episode)))
                    print("Episode finished after {} timesteps".format(t + 1))
                    print("Episode reward: {}".format(episodic_reward))

                    print(
                        "memory_allocated: {:.1f} MB".format(
                            torch.cuda.memory_allocated() / (1024 * 1024)
                        )
                    )
                    print(
                        "memory_reserved: {:.1f} MB".format(
                            torch.cuda.memory_reserved() / (1024 * 1024)
                        )
                    )
                    print("agent.earlystopping.stop", agent.earlystopping.stop)

                    loss_list = np.array(loss_list, dtype=float)
                    mean_losses = np.nanmean(loss_list, axis=0)
                    print("mean losses(recon, vq, q):", np.around(mean_losses, decimals=4))

                    break

            # # Update the target network, copying all weights and biases in DQN
            # if i_episode % agent.target_update == 0:
            #     agent.target_mlp_net.load_state_dict(
            #         agent.policy_mlp_net.state_dict())

        wandb.finish()

    print("Complete")
    env.close()


def train_duolayer():
    env_id = "Boxing-v0"  # CarRacing-v0, ALE/Skiing-v5, Boxing-v0
    vae_version = "vqvae_c3_embedding16x64_3_duolayer"

    for _ in range(1):
        # 🐝 initialise a wandb run
        wandb.init(
            project="vqvae+latent_rl",
            config={
                "env_id": "Boxing-v0",
                "total_time_steps": 1e6,
                "action_repetition": 2,
                "n_frame_stack": 1,  # make sure matching with vqvae_inchannel
                # "lr": 1e-3,
                # "dropout": random.uniform(0.01, 0.80),
                "vqvae_inchannel": int(3 * 1),
                "vqvae_latent_channel": 16,
                "vqvae_num_embeddings": 64,
                "reconstruction_path": os.path.join(
                    "/workspace/repos_dev/VQVAE_RL/reconstruction", env_id, vae_version
                ),  # using when n_frame_stack=1
                # "reconstruction_path": None,
                "num_episodes_train": 1000,
                "batch_size": 128,
                "validation_size": 128,
                "validate_every": 10,
                "size_replay_memory": int(1e6),
                "gamma": 0.97,
                "omega": 2.5e-3,
                "eps_start": 0.9,
                "eps_end": 0.05,
                "eps_decay": 200,
                "target_update": 10,
                "exploration_rate": 0.1,
                "exploration_rate_decay": 0.99999975,
                "exploration_rate_min": 0.1,
                "save_model_every": 5e5,
                "init_steps": 1e4,
                "learn_every": 4,
                "sync_every": 8,
                "seed": int(time.time()),
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            },
        )
        config = wandb.config
        print(type(config))

        # config = {
        #     "env_id": "Boxing-v0",
        #     "total_time_steps": 1e6,
        #     "action_repetition": 2,
        #     "n_frame_stack": 4,
        #     # "lr": 1e-3,
        #     # "dropout": random.uniform(0.01, 0.80),
        #     "vqvae_inchannel": int(3 * 4),
        #     "vqvae_latent_channel": 16,
        #     "vqvae_num_embeddings": 64,
        #     # "reconstruction_path": os.path.join(
        #     #     "/workspace/repos_dev/VQVAE_RL/reconstruction", env_id, vae_version
        #     # ),
        #     "reconstruction_path": None,
        #     "num_episodes_train": 100,
        #     "batch_size": 128,
        #     "gamma": 0.99,
        #     "eps_start": 0.9,
        #     "eps_end": 0.05,
        #     "eps_decay": 200,
        #     "target_update": 10,
        #     "exploration_rate": 0.1,
        #     "exploration_rate_decay": 0.99999975,
        #     "exploration_rate_min": 0.1,
        #     "save_model_every": 5e5,
        #     "init_steps": 1e4,
        #     "learn_every": 4,
        #     "sync_every": 8,
        #     "seed": int(time.time()),
        #     "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # }
        # config = SimpleNamespace(**config)

        # The main training loop
        env = make_env(env_id, config)
        agent = DuoLayerAgent(env, config)
        # print("agent.policy_mlp_net:", agent.ground_Q_net)
        # print("agent.vqvae_model:", agent.vqvae_model)

        wandb.watch(agent.ground_Q_net, log_freq=1000)
        wandb.watch(agent.abstract_V_net, log_freq=1000)
        wandb.watch(agent.vqvae_model, log_freq=1000)

        current_time = datetime.datetime.now() + datetime.timedelta(hours=2)
        current_time = current_time.strftime("%b%d_%H-%M-%S")

        comment = ""
        log_dir_tensorboard = f"/workspace/repos_dev/VQVAE_RL/log_tensorboard/end2end_duolayer/{env_id}/{current_time}_{comment}"
        tb_writer = SummaryWriter(log_dir_tensorboard)
        print("log_dir_tensorboard:", log_dir_tensorboard)
        print("reconstruction_path:", config.reconstruction_path)

        time_start_training = time.time()

        # transformer = transform_dict["Boxing-v0"]
        for i_episode in range(config.num_episodes_train):
            time_start_episode = time.time()
            # Initialize the environment and state
            state = env.reset()
            # last_screen = get_screen()
            # current_screen = get_screen()
            # state = current_screen - last_screen
            episodic_reward = 0
            episodic_negative_reward = 0
            episodic_non_zero_reward = 0
            loss_list = []
            for t in count():
                # Select and perform an action
                action = agent.act(state)
                # print(
                #     "memory_allocated: {:.5f} MB".format(
                #         torch.cuda.memory_allocated() / (1024 * 1024)
                #     )
                # )
                # print("agent.act")

                next_state, reward, done, _ = env.step(action.item())
                episodic_reward += reward
                if reward < 0:
                    episodic_negative_reward += reward
                else:
                    episodic_non_zero_reward += reward

                # # Observe new state
                # last_screen = current_screen
                # current_screen = get_screen()
                # if not done:
                #     next_state = current_screen - last_screen
                # else:
                #     next_state = None

                # Store the transition in memory
                # agent.memory.push(state, action, next_state, reward, done)
                agent.cache(state, action, next_state, reward, done)
                # print(
                #     "memory_allocated: {:.5f} MB".format(
                #         torch.cuda.memory_allocated() / (1024 * 1024)
                #     )
                # )
                # print("agent.cache")

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                # optimize_model()
                recon_loss, vq_loss, q_loss = agent.learn(tb_writer)
                loss_list.append([recon_loss, vq_loss, q_loss])
                # print(
                #     "memory_allocated: {:.5f} MB".format(
                #         torch.cuda.memory_allocated() / (1024 * 1024)
                #     )
                # )
                # print("agent.learn")
                if done:
                    # print("sys.getsizeof(agent.memory)", sys.getsizeof(agent.memory))
                    # print(torch.cuda.memory_reserved()/(1024*1024), "MB")
                    # print(torch.cuda.memory_allocated()/(1024*1024), "MB")

                    agent.episodes_done += 1
                    # episode_durations.append(t + 1)
                    # plot_durations()
                    metrics = {
                        "train/episodic_reward": episodic_reward,
                        "train/episodic_negative_reward": episodic_negative_reward,
                        "train/episodic_non_zero_reward": episodic_non_zero_reward,
                        "train/total_steps_done": agent.total_steps_done,
                        "train/time_elapsed": time.time() - time_start_training,
                        "train/episode_length": t + 1,
                        "train/episodes": i_episode,
                        "train/epsilon": agent.eps_threshold,
                        "train/episodic_fps": int((t + 1) / (time.time() - time_start_episode)),
                    }
                    wandb.log({**metrics})

                    print(">>>>>>>>>>>>>>>>Episode Done>>>>>>>>>>>>>>>>>")
                    print("time cost so far: {:.1f} s".format(time.time() - time_start_training))
                    print("episodic time cost: {:.1f} s".format(time.time() - time_start_episode))
                    print("agent.total_steps_done:", agent.total_steps_done)
                    print("episodic_fps:", int((t + 1) / (time.time() - time_start_episode)))
                    print("Episode finished after {} timesteps".format(t + 1))
                    print("Episode reward: {}".format(episodic_reward))

                    print(
                        "memory_allocated: {:.1f} MB".format(
                            torch.cuda.memory_allocated() / (1024 * 1024)
                        )
                    )
                    print(
                        "memory_reserved: {:.1f} MB".format(
                            torch.cuda.memory_reserved() / (1024 * 1024)
                        )
                    )
                    print("agent.earlystopping.stop", agent.earlystopping.stop)
                    loss_list = np.array(loss_list, dtype=float)
                    mean_losses = np.nanmean(loss_list, axis=0)
                    print("mean losses(recon, vq, q):", np.around(mean_losses, decimals=4))

                    break

            # # Update the target network, copying all weights and biases in DQN
            # if i_episode % agent.target_update == 0:
            #     agent.target_mlp_net.load_state_dict(
            #         agent.policy_mlp_net.state_dict())

        wandb.finish()

    print("Complete")
    env.close()


if __name__ == "__main__":

    # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Get the first available GPU
    DEVICE_ID_LIST = GPUtil.getAvailable(
        order="random",
        limit=4,
        maxLoad=0.5,
        maxMemory=0.5,
        includeNan=False,
        excludeID=[],
        excludeUUID=[],
    )
    assert len(DEVICE_ID_LIST) > 0, "no availible cuda currently"
    print("availible CUDAs:", DEVICE_ID_LIST)
    DEVICE_ID = DEVICE_ID_LIST[0]  # grab first element from list
    # os.environ["DISPLAY"] = ":199"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    os.environ["GPU_DEBUG"] = str(DEVICE_ID)
    # from utils.gpu_profile import gpu_profile

    # sys.settrace(gpu_profile)

    train_single_layer()
    # train_duolayer()
