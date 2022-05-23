import math
import os
import random
import time
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from latentrl.dqn_models import DQN, DQN_MLP, DVN
from latentrl.policies.utils import ReplayMemory
from latentrl.utils.learning import EarlyStopping, ReduceLROnPlateau
from latentrl.utils.misc import get_linear_fn, linear_schedule, update_learning_rate
from latentrl.vqvae_end2end import VQVAE
from latentrl.vqvae_prototype import VQVAE2


class SingelLayerAgent:
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
        self.vqvae_model = VQVAE2(
            in_channels=env.observation_space.shape[-1],
            embedding_dim=config.vqvae_latent_channel,
            num_embeddings=config.vqvae_num_embeddings,
            reconstruction_path=config.reconstruction_path,
        ).to(self.device)

        self.vqvae_optimizer = optim.Adam(self.vqvae_model.parameters(), lr=config.lr_vqvae)

        # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

        self.scheduler = ReduceLROnPlateau(self.vqvae_optimizer, "min")

        self.earlystopping = EarlyStopping("min", patience=30)
        # <<<<<<<<<<<VQVAE<<<<<<<<<<<<

        # DQN_MLP
        # Get number of actions from gym action space
        self.n_actions = env.action_space.n
        self.obs_height = env.observation_space.shape[0]
        self.obs_width = env.observation_space.shape[1]

        # get the output size of the self.vqvae_model.encoder
        with torch.no_grad():
            sample = T.ToTensor()(env.observation_space.sample()).unsqueeze(0).to(self.device)
            encoder_output_size = self.vqvae_model.encoder(sample).shape
        input_dim = (encoder_output_size[1], encoder_output_size[2], encoder_output_size[3])

        self.policy_mlp_net = DQN_MLP(input_dim, self.n_actions).to(self.device)
        self.target_mlp_net = DQN_MLP(input_dim, self.n_actions).to(self.device)
        # or
        # self.policy_mlp_net = DQN_paper(env.observation_space, env.action_space).to(self.device)
        # self.target_mlp_net = DQN_paper(env.observation_space, env.action_space).to(self.device)

        self.target_mlp_net.load_state_dict(self.policy_mlp_net.state_dict())
        # self.target_mlp_net.eval()

        self._current_progress_remaining = 1.0
        if isinstance(config.lr_dqn, str) and config.lr_dqn.startswith("lin"):
            self.lr_scheduler = linear_schedule(float(config.lr_dqn.split("_")[1]))
            # self.vqvae_optimizer = optim.Adam(
            #     self.vqvae_model.parameters(),
            #     lr=self.lr_scheduler(self._current_progress_remaining),
            # )
            self.dqn_optimizer = optim.Adam(
                self.policy_mlp_net.parameters(),
                lr=self.lr_scheduler(self._current_progress_remaining),
            )
        elif isinstance(config.lr_dqn, float):
            # self.vqvae_optimizer = optim.Adam(
            #     self.vqvae_model.parameters(), lr=config.learning_rate
            # )
            self.dqn_optimizer = optim.Adam(
                self.policy_mlp_net.parameters(),
                lr=config.lr_dqn,
            )

        # print(self.policy_mlp_net)
        # summary(policy_mlp_net, (3, obs_height, obs_width))

        # self.use_cuda = torch.cuda.is_available()
        # if self.use_cuda:
        #     self.net = self.net.to(device="cuda")

        # self.curr_step = 0
        self.timesteps_done = 0
        self.episodes_done = 0

        # Hyperparameters
        self.total_timesteps = config.total_timesteps
        # self.total_episodes = config.total_episodes
        self.batch_size = config.batch_size
        self.validation_size = config.validation_size
        self.size_replay_memory = config.size_replay_memory
        self.gamma = config.gamma

        self.exploration_initial_eps = config.exploration_initial_eps
        self.exploration_final_eps = config.exploration_final_eps
        self.exploration_fraction = config.exploration_fraction

        # self.eps_start = config.eps_start
        # self.eps_end = config.eps_end
        # self.eps_decay = config.eps_decay

        # self.exploration_rate = config.exploration_rate
        # self.exploration_rate_decay = config.exploration_rate_decay
        # self.exploration_rate_min = config.exploration_rate_min

        self.save_model_every = config.save_model_every

        # Initialize experience replay buffer
        self.memory = ReplayMemory(self.size_replay_memory)
        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "done")
        )

        self.init_steps = config.init_steps  # min. experiences before training
        self.learn_every = config.learn_every  # no. of experiences between updates to Q_online
        self.sync_every = config.sync_every  # no. of experiences between updates to Q_target
        # self.validate_every = config.validate_every
        self.gradient_steps = config.gradient_steps

        self.exploration_scheduler = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        self.n_call_train = 0
        self.exploration_rate = None

    @torch.no_grad()
    def act(self, state):
        # if self.episodes_done < math.ceil(0.99 * self.num_episodes_train):
        #     self.eps_threshold = (
        #         self.eps_start - self.eps_start / self.num_episodes_train * self.episodes_done
        #     )
        self._update_current_progress_remaining(self.timesteps_done, self.total_timesteps)
        self.exploration_rate = self.exploration_scheduler(self._current_progress_remaining)
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

        self.timesteps_done += 1

        sample = random.random()
        if sample > self.exploration_rate:
            # with torch.no_grad():
            # action = self.policy_mlp_net(quantized_latent_state)  # for debug
            action = self.policy_mlp_net(quantized_latent_state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor(
                [[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long
            )

        return action

    @torch.no_grad()
    def act2(self, state):
        # if self.episodes_done < math.ceil(0.99 * self.num_episodes_train):
        #     self.eps_threshold = (
        #         self.eps_start - self.eps_start / self.num_episodes_train * self.episodes_done
        #     )
        self._update_current_progress_remaining(self.timesteps_done, self.total_timesteps)
        self.exploration_rate = self.exploration_scheduler(self._current_progress_remaining)
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

        self.timesteps_done += 1

        sample = random.random()
        if sample > self.exploration_rate:
            # with torch.no_grad():
            # action = self.policy_mlp_net(quantized_latent_state)  # for debug
            action = self.policy_mlp_net(latent_state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor(
                [[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long
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
        if self.timesteps_done % self.sync_every == 0:
            self.sync_Q_target()

        if self.timesteps_done % self.save_model_every == 0:
            pass
            # self.save()

        if self.timesteps_done < self.init_steps:
            return None, None, None

        if self.timesteps_done == self.init_steps:
            loss_list = []
            for _ in range(int(self.init_steps / 10)):
                mean_recon_loss, mean_vq_loss, q_loss = self.train(tb_writer)
                loss_list.append([mean_recon_loss, mean_vq_loss, q_loss])
            loss_list = np.array(loss_list, dtype=float)
            loss_list = np.nanmean(loss_list, axis=0)
            return [item for item in loss_list]

        if self.timesteps_done % self.learn_every != 0:
            return None, None, None

        # if self.n_call_train % self.validate_every == 0:
        #     # validate_loss = self.validate_vqvae()
        #     # self.scheduler.step(validate_loss)
        #     # self.earlystopping.step(validate_loss)
        #     pass

        loss_list = []
        for _ in range(self.gradient_steps):
            mean_recon_loss, mean_vq_loss, q_loss = self.train(tb_writer)
            loss_list.append([mean_recon_loss, mean_vq_loss, q_loss])
        loss_list = np.array(loss_list, dtype=float)
        loss_list = np.nanmean(loss_list, axis=0)
        return [item for item in loss_list]

    def train2(self, tb_writer):
        # update_learning_rate(
        #     self.vqvae_optimizer, self.lr_scheduler(self._current_progress_remaining)
        # )
        update_learning_rate(
            self.dqn_optimizer, self.lr_scheduler(self._current_progress_remaining)
        )

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

        current_Q = self.policy_mlp_net(quantized.detach()).gather(1, action_batch)
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
            # next_state_max_Q = next_state_max_Q   # for debug
            next_state_max_Q = self.target_mlp_net(quantized2).max(1)[0]

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

        # if not self.earlystopping.stop:
        #     total_loss = 1 * (vqloss + vqloss2 + recon_loss + recon_loss2) + 1 * q_loss

        # else:
        #     total_loss = q_loss

        #     # make sure q_loss no longer effect weights of the encoder
        #     for param in self.vqvae_model.parameters():
        #         param.requires_grad = False

        #     self.dqn_optimizer.zero_grad(set_to_none=True)
        #     total_loss.backward()
        #     # max_grad_norm = 10
        #     # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
        #     self.dqn_optimizer.step()
        #     if self.n_call_learn % 500 == 0:
        #         for name, param in self.policy_mlp_net.named_parameters():
        #             tb_writer.add_histogram(f"gradients/dqn/{name}", param.grad, self.n_call_learn)
        #             tb_writer.add_histogram(f"weight_bias/dqn/{name}", param, self.n_call_learn)

        #     self.n_call_learn += 1
        #     q_loss = q_loss.item()
        #     return None, None, q_loss
        vqvae_loss = vqloss + vqloss2 + recon_loss + recon_loss2
        if vqvae_loss > 0.0001:
            # Optimize the model
            self.vqvae_optimizer.zero_grad(set_to_none=True)
            # self.vqvae_optimizer.zero_grad()
            vqvae_loss.backward()
            # 1 clamp gradients to avoid exploding gradient
            # for param in self.policy_mlp_net.parameters():
            #     param.grad.data.clamp_(-1, 1)

            # for param in self.vqvae_model.parameters():
            #     param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
            self.vqvae_optimizer.step()
        else:
            # Optimize the model
            self.dqn_optimizer.zero_grad(set_to_none=True)
            # self.dqn_optimizer.zero_grad()
            q_loss.backward()
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

        self.n_call_train += 1

        # if self.n_call_train % 500 == 0:
        #     for name, param in self.policy_mlp_net.named_parameters():
        #         tb_writer.add_histogram(f"gradients/dqn/{name}", param.grad, self.n_call_train)
        #         tb_writer.add_histogram(f"weight_bias/dqn/{name}", param, self.n_call_train)
        #     for name, param in self.vqvae_model.encoder.named_parameters():
        #         tb_writer.add_histogram(f"gradients/encoder/{name}", param.grad, self.n_call_train)
        #         tb_writer.add_histogram(f"weight_bias/encoder/{name}", param, self.n_call_train)
        #     for name, param in self.vqvae_model.decoder.named_parameters():
        #         tb_writer.add_histogram(f"gradients/decoder/{name}", param.grad, self.n_call_train)
        #         tb_writer.add_histogram(f"weight_bias/decoder/{name}", param, self.n_call_train)

        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run step()")

        # print("learn() finished")
        mean_recon_loss = ((recon_loss + recon_loss2) / 2).item()
        mean_vq_loss = ((vqloss + vqloss2) / 2).item()
        q_loss = q_loss.item()
        return mean_recon_loss, mean_vq_loss, q_loss

    def train(self, tb_writer):
        # update_learning_rate(
        #     self.vqvae_optimizer, self.lr_scheduler(self._current_progress_remaining)
        # )
        update_learning_rate(
            self.dqn_optimizer, self.lr_scheduler(self._current_progress_remaining)
        )

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
            # next_state_max_Q = next_state_max_Q   # for debug
            next_state_max_Q = self.target_mlp_net(quantized2).max(1)[0]

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
            total_loss = 1 * (vqloss + vqloss2 + recon_loss + recon_loss2) + 1 * q_loss

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
            if self.n_call_train % 500 == 0:
                for name, param in self.policy_mlp_net.named_parameters():
                    tb_writer.add_histogram(f"gradients/dqn/{name}", param.grad, self.n_call_train)
                    tb_writer.add_histogram(f"weight_bias/dqn/{name}", param, self.n_call_train)

            self.n_call_train += 1
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

        self.n_call_train += 1

        if self.n_call_train % 500 == 0:
            for name, param in self.policy_mlp_net.named_parameters():
                tb_writer.add_histogram(f"gradients/dqn/{name}", param.grad, self.n_call_train)
                tb_writer.add_histogram(f"weight_bias/dqn/{name}", param, self.n_call_train)
            for name, param in self.vqvae_model.encoder.named_parameters():
                tb_writer.add_histogram(f"gradients/encoder/{name}", param.grad, self.n_call_train)
                tb_writer.add_histogram(f"weight_bias/encoder/{name}", param, self.n_call_train)
            for name, param in self.vqvae_model.decoder.named_parameters():
                tb_writer.add_histogram(f"gradients/decoder/{name}", param.grad, self.n_call_train)
                tb_writer.add_histogram(f"weight_bias/decoder/{name}", param, self.n_call_train)

        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run step()")

        # print("learn() finished")
        mean_recon_loss = ((recon_loss + recon_loss2) / 2).item()
        mean_vq_loss = ((vqloss + vqloss2) / 2).item()
        q_loss = q_loss.item()
        return mean_recon_loss, mean_vq_loss, q_loss

    def train3(self, tb_writer):
        # update_learning_rate(
        #     self.vqvae_optimizer, self.lr_scheduler(self._current_progress_remaining)
        # )
        update_learning_rate(
            self.dqn_optimizer, self.lr_scheduler(self._current_progress_remaining)
        )

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

        latent_state = self.vqvae_model.encoder(state_batch)
        next_latent_state = self.vqvae_model.encoder(next_state_batch)

        current_Q = self.policy_mlp_net(latent_state).gather(1, action_batch)
        with torch.no_grad():
            # next_state_max_Q = next_state_max_Q   # for debug
            next_state_max_Q = self.target_mlp_net(next_latent_state).max(1)[0]

            target_Q = (
                reward_batch + (1 - done_batch.float()) * self.gamma * next_state_max_Q
            ).float()

        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run target_Q")

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        q_loss = criterion(current_Q, target_Q.unsqueeze(1))

        self.dqn_optimizer.zero_grad(set_to_none=True)
        self.vqvae_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
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

        self.n_call_train += 1

        # if self.n_call_train % 500 == 0:
        #     for name, param in self.policy_mlp_net.named_parameters():
        #         tb_writer.add_histogram(f"gradients/dqn/{name}", param.grad, self.n_call_train)
        #         tb_writer.add_histogram(f"weight_bias/dqn/{name}", param, self.n_call_train)
        #     for name, param in self.vqvae_model.encoder.named_parameters():
        #         tb_writer.add_histogram(f"gradients/encoder/{name}", param.grad, self.n_call_train)
        #         tb_writer.add_histogram(f"weight_bias/encoder/{name}", param, self.n_call_train)
        #     for name, param in self.vqvae_model.decoder.named_parameters():
        #         tb_writer.add_histogram(f"gradients/decoder/{name}", param.grad, self.n_call_train)
        #         tb_writer.add_histogram(f"weight_bias/decoder/{name}", param, self.n_call_train)

        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run step()")

        q_loss = q_loss.item()
        return None, None, q_loss

    def _update_current_progress_remaining(self, timesteps_done, total_timesteps):
        finished_time_steps_after_init = timesteps_done - self.init_steps
        if finished_time_steps_after_init < 0:
            self._current_progress_remaining = 1.0
        else:
            self._current_progress_remaining = (
                1.0 - finished_time_steps_after_init / total_timesteps
            )

        # self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    # def _update_current_progress_remaining(self, finished: int, total: int) -> None:
    #     """
    #     Compute current progress remaining (starts from 1 and ends to 0)

    #     :param num_timesteps: current number of timesteps
    #     :param total_timesteps:
    #     """
    #     self._current_progress_remaining = 1.0 - finished / total

    @property
    def warm_up_done(self):
        return self.timesteps_done >= self.init_steps


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
