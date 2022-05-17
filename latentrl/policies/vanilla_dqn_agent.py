import random
import time
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from latentrl.dqn_models import DQN_paper
from latentrl.policies.utils import ReplayMemory
from latentrl.utils.learning import EarlyStopping, ReduceLROnPlateau
from latentrl.utils.misc import get_linear_fn, linear_schedule, update_learning_rate
from torchsummary import summary


class VanillaDQNAgent:
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

        # DQN_MLP
        # Get number of actions from gym action space
        self.n_actions = env.action_space.n
        if len(env.observation_space.shape) > 1:
            self.obs_height = env.observation_space.shape[0]
            self.obs_width = env.observation_space.shape[1]

        # self.policy_mlp_net = DQN_MLP(env.observation_space.shape, self.n_actions).to(self.device)
        # self.target_mlp_net = DQN_MLP(env.observation_space.shape, self.n_actions).to(self.device)

        self.policy_mlp_net = DQN_paper(env.observation_space, env.action_space).to(self.device)
        self.target_mlp_net = DQN_paper(env.observation_space, env.action_space).to(self.device)

        print("policy_mlp_net", self.policy_mlp_net)
        if len(env.observation_space.shape) == 3:
            temp = env.observation_space.sample()
            sample = T.ToTensor()(temp)
            summary(self.policy_mlp_net, (12, 84, 84))
        else:
            summary(self.policy_mlp_net, env.observation_space.shape)
        # or
        # self.policy_mlp_net = DQN_paper(env.observation_space, env.action_space).to(self.device)
        # self.target_mlp_net = DQN_paper(env.observation_space, env.action_space).to(self.device)

        self.target_mlp_net.load_state_dict(self.policy_mlp_net.state_dict())
        # self.target_mlp_net.eval()

        self._current_progress_remaining = 1.0
        if isinstance(config.learning_rate, str) and config.learning_rate.startswith("lin"):
            self.lr_scheduler = linear_schedule(float(config.learning_rate.split("_")[1]))
            self.dqn_optimizer = optim.Adam(
                self.policy_mlp_net.parameters(),
                lr=self.lr_scheduler(self._current_progress_remaining),
            )
        elif isinstance(config.learning_rate, float):
            self.dqn_optimizer = optim.Adam(
                self.policy_mlp_net.parameters(), lr=config.learning_rate
            )

        self.scheduler = ReduceLROnPlateau(self.dqn_optimizer, "min")

        self.earlystopping = EarlyStopping("min", patience=30)

        # print(self.policy_mlp_net)
        # summary(policy_mlp_net, (3, obs_height, obs_width))

        # self.curr_step = 0
        self.total_steps_done = 0
        self.num_episodes_finished = 0

        # Hyperparameters
        self.num_episodes_train = config.num_episodes_train
        self.batch_size = config.batch_size
        # self.validation_size = config.validation_size
        self.size_replay_memory = config.size_replay_memory
        self.gamma = config.gamma
        self.exploration_initial_eps = config.exploration_initial_eps
        self.exploration_final_eps = config.exploration_final_eps
        self.exploration_fraction = config.exploration_fraction
        # self.eps_decay = config.eps_decay
        # self.target_update = config.target_update

        # self.exploration_rate = config.exploration_rate
        # self.exploration_rate_decay = config.exploration_rate_decay
        # self.exploration_rate_min = config.exploration_rate_min

        # self.save_model_every = config.save_model_every

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
        # self.exploration_schedule = get_linear_fn(
        #     self.exploration_initial_eps,
        #     self.exploration_final_eps,
        #     self.exploration_fraction,
        # )

        self.exploration_scheduler = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        self.n_call_learn = 0
        self.exploration_rate = None

    def _update_current_progress_remaining(self, finished: int, total: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(finished) / float(total)

    @torch.no_grad()
    def act(self, state):
        # if self.num_episodes_finished < math.ceil(0.99 * self.num_episodes_train):
        #     self.exploration_rate = (
        #         self.exploration_initial_eps
        #         - self.exploration_initial_eps / self.num_episodes_train * self.num_episodes_finished
        #     )
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

        if len(state.shape) > 2:
            state = T.ToTensor()(state).float().unsqueeze(0).to(self.device)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        sample = random.random()
        if sample > self.exploration_rate:
            # with torch.no_grad():
            # action = self.policy_mlp_net(quantized_latent_state)  # for debug
            action = self.policy_mlp_net(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor(
                [[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long
            )

        self.total_steps_done += 1
        self._update_current_progress_remaining(self.num_episodes_finished, self.num_episodes_train)
        return action

    def cache(self, state, action, next_state, reward, done):
        """Add the experience to memory"""
        # if state is an image, convert it to a tensor
        if len(state.shape) > 2:
            state = T.ToTensor()(state).float().unsqueeze(0)
            next_state = T.ToTensor()(next_state).float().unsqueeze(0)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
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

    def learn(self, tb_writer):
        if self.total_steps_done % self.sync_every == 0:
            self.sync_Q_target()

        # if self.total_steps_done % self.save_model_every == 0:
        #     pass
        # self.save()

        if self.total_steps_done < self.init_steps:
            return None

        if self.total_steps_done % self.learn_every != 0:
            return None

        # if self.n_call_learn % self.validate_every == 0:
        # validate_loss = self.validate_vqvae()
        # self.scheduler.step(validate_loss)
        # self.earlystopping.step(validate_loss)
        # pass

        loss_list = []
        for _ in range(self.gradient_steps):
            q_loss = self.train(tb_writer)
            loss_list.append(q_loss)
        mean_q_loss = np.mean(loss_list)

        return mean_q_loss

    def train(self, tb_writer):
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

        current_Q = self.policy_mlp_net(state_batch).gather(1, action_batch)

        # Or Compute next_state_max_Q
        with torch.no_grad():
            # next_state_max_Q = next_state_max_Q   # for debug
            next_state_max_Q = self.target_mlp_net(next_state_batch).max(1)[0]

            target_Q = (
                reward_batch + (1 - done_batch.float()) * self.gamma * next_state_max_Q
            ).float()

        criterion = nn.SmoothL1Loss()
        q_loss = criterion(current_Q, target_Q.unsqueeze(1))

        self.dqn_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        # max_grad_norm = 10
        # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
        self.dqn_optimizer.step()

        self.n_call_learn += 1

        if self.n_call_learn % 500 == 0:
            for name, param in self.policy_mlp_net.named_parameters():
                tb_writer.add_histogram(f"gradients/dqn/{name}", param.grad, self.n_call_learn)
                tb_writer.add_histogram(f"weight_bias/dqn/{name}", param, self.n_call_learn)

        return q_loss.item()
