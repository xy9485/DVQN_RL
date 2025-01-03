import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from common.utils import (
    linear_schedule,
    soft_sync_params,
    update_learning_rate,
)
from nn_models import (
    DQN,
    DVN,
    DuelDQN,
    RandomShiftsAug,
    CURL,
    CURL_ATC,
)
from nn_models.encoder import make_encoder
from nn_models.CURL import simclr_loss, simclr_loss2, simclr_debiased_loss
from PIL import Image
from policies.HDQN import HDQN
from policies.utils import (
    ReplayBufferNStep,
    transition_np2torch,
)
from PER import Memory
from common.Logger import LoggerWandb, Logger
from torch import nn


class DVQN(HDQN):
    def __init__(self, args, env, logger: LoggerWandb):
        super().__init__(args, env, logger)
        self.act_boltzmann = False
        self.init_act_boltzmann_temperature = 0.1
        self.act_boltzmann_temperature = self.init_act_boltzmann_temperature
        self.boltzmann_decay_rate = self.explore_final_fraction ** (1/self.total_timesteps)
        self.n_update = 1
        if args.criterion == "l1":
            self.criterion = nn.SmoothL1Loss()
        elif args.criterion == "l2":
            self.criterion = nn.MSELoss()
        if args.algo == "dueldqn":
            QNet = DuelDQN
        else:
            QNet = DQN
        self.Q = QNet(
            action_space=env.action_space,
            encoder=make_encoder(
                input_format=args.input_format,
                observation_space=env.observation_space,
                linear_dims=args.Q_encoder_linear_dims,
            ),
            mlp_hidden_dims=args.Q_critic_dims,
        ).to(self.device)

        self.Q_target = QNet(
            action_space=env.action_space,
            encoder=make_encoder(
                input_format=args.input_format,
                observation_space=env.observation_space,
                linear_dims=args.Q_encoder_linear_dims,
            ),
            mlp_hidden_dims=args.Q_critic_dims,
        ).to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q.train()
        self.Q_target.train()

        if self.algo == "cddqn":
            self.Q2 = QNet(
                action_space=env.action_space,
                encoder=make_encoder(
                    input_format=args.input_format,
                    observation_space=env.observation_space,
                    linear_dims=args.Q_encoder_linear_dims,
                ),
                mlp_hidden_dims=args.Q_critic_dims,
            ).to(self.device)

            self.Q2_target = QNet(
                action_space=env.action_space,
                encoder=make_encoder(
                    input_format=args.input_format,
                    observation_space=env.observation_space,
                    linear_dims=args.Q_encoder_linear_dims,
                ),
                mlp_hidden_dims=args.Q_critic_dims,
            ).to(self.device)
            self.Q2_target.load_state_dict(self.Q2.state_dict())
            self.Q2.train()
            self.Q_target.train()

        if self.algo == "avgdqn":
            self.target_q_net_list = []
            self.num_active_target = 1
            self.avgdqn_k = args.avgdqn_k
            for k in range(args.avgdqn_k):
                Q_target = QNet(
                    action_space=env.action_space,
                    encoder=make_encoder(
                        input_format=args.input_format,
                        observation_space=env.observation_space,
                        linear_dims=args.Q_encoder_linear_dims,
                    ),
                    mlp_hidden_dims=args.Q_critic_dims,
                ).to(self.device)
                Q_target.load_state_dict(self.Q.state_dict())
                self.target_q_net_list.append(Q_target)

        if args.algo == "dvqn":
            self.Q2 = QNet(
                action_space=env.action_space,
                encoder=make_encoder(
                    input_format=args.input_format,
                    observation_space=env.observation_space,
                    linear_dims=args.Q_encoder_linear_dims,
                ),
                mlp_hidden_dims=args.Q_critic_dims,
            ).to(self.device)
            if args.share_encoder:
                assert args.algo == "dvqn"
                self.V = DVN(
                    encoder=self.Q.encoder,
                    mlp_hidden_dims=args.V_critic_dims,
                ).to(self.device)
            else:
                self.V = DVN(
                    encoder=make_encoder(
                        input_format=args.input_format,
                        observation_space=env.observation_space,
                        linear_dims=args.V_encoder_linear_dims,
                    ),
                    mlp_hidden_dims=args.V_critic_dims,
                ).to(self.device)

            self.V_target = DVN(
                encoder=make_encoder(
                    input_format=args.input_format,
                    observation_space=env.observation_space,
                    linear_dims=self.V.encoder.linear_dims,
                ),
                mlp_hidden_dims=self.V.mlp_hidden_dims,
            ).to(self.device)

            self.V_target.load_state_dict(self.V.state_dict())
            self.V.train()
            self.V_target.train()

        if args.use_curl:
            if args.use_curl == "on_V":
                curl_encoder = self.V.encoder
            elif args.use_curl == "on_Q":
                curl_encoder = self.Q.encoder
            if self.curl_pair == "atc":
                self.curl = CURL_ATC(
                    encoder=curl_encoder,
                    encoder_target=make_encoder(
                        input_format=args.input_format,
                        observation_space=env.observation_space,
                        linear_dims=curl_encoder.linear_dims,
                    ),
                    anchor_projection=True,
                ).to(self.device)
                self.curl.train()
            else:
                self.curl = CURL(
                    encoder=curl_encoder,
                    projection_hidden_dims=args.curl_projection_dims,
                ).to(self.device)

                self.curl_ema = CURL(
                    encoder=make_encoder(
                        input_format=args.input_format,
                        observation_space=env.observation_space,
                        linear_dims=self.curl.encoder.linear_dims,
                    ),
                    projection_hidden_dims=self.curl.projection_hidden_dims,
                ).to(self.device)

                self.curl_ema.load_state_dict(self.curl.state_dict())
                for param in self.curl_ema.parameters():
                    param.requires_grad = False
                self.curl.train()
                self.curl_ema.train()

        self.aug = RandomShiftsAug(pad=4)

        if self.per:
            self.memory = Memory(self.size_replay_memory, self.total_timesteps)
        else:
            self.memory = ReplayBufferNStep(
                self.size_replay_memory,
                self.device,
                gamma=args.gamma,
                batch_size=args.batch_size,
            )
            self.memory2 = ReplayBufferNStep(
                self.Q_learn_every,
                self.device,
                gamma=args.gamma,
                batch_size=args.batch_size,
            )

        self.count_vis = 0
        self.goal_found = False

        self._create_optimizers(args)
        # self.train()

    def train(self, training=True):
        self.training = training
        self.Q.train(training)
        self.V.train(training)

    def set_hparams(self, args):
        # Hyperparameters
        self.input_format = args.input_format
        self.algo = args.algo
        self.use_curl = args.use_curl
        self.curl_pair = args.curl_pair
        self.use_vq = args.use_vq
        # self.use_noisynet = args.use_noisynet
        self.curl_vq_cfg = args.curl_vq_cfg
        self.curl_enc_detach = args.curl_enc_detach

        self.Q_enc_detach = args.Q_enc_detach
        self.critic_upon_vq = args.critic_upon_vq
        # self.total_episodes = config.total_episodes
        self.total_timesteps = int(args.total_timesteps)
        self.init_steps = int(args.init_steps)  # min. experiences before training
        self.batch_size = int(args.batch_size)
        self.batch_size_repre = self.batch_size
        self.size_replay_memory = int(args.size_replay_memory)
        self.gamma = args.gamma

        self.Q_encoder_linear_dims = args.Q_encoder_linear_dims
        self.Q_learn_every = args.freq_Q_learn
        self.Q_gradient_steps = args.Q_gradient_steps
        self.Q_sync_every = args.freq_Q_sync
        self.Q_encoder_tau = args.tau_Q_encoder
        self.Q_critic_tau = args.tau_Q_critic

        if args.algo == "dvqn":
            self.V_learn_every = args.freq_V_learn
            self.V_gradient_steps = args.V_gradient_steps
            self.V_sync_every = args.freq_V_sync
            self.V_encoder_tau = args.tau_V_encoder
            self.V_critic_tau = args.tau_V_critic
            self.V_enc_detach = args.V_enc_detach
            self.share_encoder = args.share_encoder
            self.use2Q = args.use2Q
            self.use_n_newdata = args.use_n_newdata

        if args.use_curl:
            self.curl_learn_every = args.freq_curl_learn
            self.curl_sync_every = args.freq_curl_sync
            self.curl_gradient_steps = 1
            self.curl_tau = args.tau_curl

        if args.use_vq:
            self.num_vq_embeddings = args.num_vq_embeddings
            self.use_vq = args.use_vq
            self.vq_tau = args.tau_vq

        self.clip_grad_mode = args.clip_grad_mode
        self.clip_reward = args.clip_reward
        self.epsilon_decay = args.epsilon_decay
        self.lr_decay = args.lr_decay
        self.lr_min = args.lr_min
        self.explore_final_fraction = args.explore_final_fraction
        self.per = args.per


    def cache(self, state, action, next_state, reward, terminated, info):
        """Add the experience to memory"""
        if terminated:
            gamma = 0.0
        else:
            gamma = self.gamma
        self.memory.push((state, action, next_state, reward, gamma, info))

    def _create_optimizers(self, args):

        if args.lr_Q.startswith("lin"):
            self.lr_scheduler_Q = linear_schedule(float(args.lr_Q.split("_")[1]))
            self.lr_Q = self.lr_scheduler_Q(self._current_progress_remaining)
        else:
            self.lr_Q = float(args.lr_Q)

        if args.algo == "dvqn":
            if args.lr_V.startswith("lin"):
                self.lr_scheduler_V = linear_schedule(float(args.lr_V.split("_")[1]))
                self.lr_V = self.lr_scheduler_V(self._current_progress_remaining)
            else:
                self.lr_V = float(args.lr_V)

        if args.use_vq:
            if args.lr_vq.startswith("lin"):
                self.lr_scheduler_vq = linear_schedule(float(args.lr_vq.split("_")[1]))
                self.lr_vq = self.lr_scheduler_vq(self._current_progress_remaining)
            else:
                self.lr_vq = float(args.lr_vq)
        if args.use_curl:
            if args.lr_curl.startswith("lin"):
                self.lr_scheduler_curl = linear_schedule(float(args.lr_curl.split("_")[1]))
                self.lr_curl = self.lr_scheduler_curl(self._current_progress_remaining)
            else:
                self.lr_curl = float(args.lr_curl)

        if args.optimizer == "adam":
            OPT = optim.Adam
        elif args.optimizer == "adamw":
            OPT = optim.AdamW
        elif args.optimizer == "rmsprop":
            OPT = optim.RMSprop
        if args.use_curl:
            self.curl_optimizer = OPT(self.curl.parameters(), lr=self.lr_curl)
        if args.use_vq:
            self.vq_optimizer = OPT(self.vq.parameters(), lr=self.lr_vq)
        if args.algo == "dvqn":
            self.V_optimizer = OPT(self.V.parameters(), lr=self.lr_V)
            # if self.share_encoder:
            #     self.whole_optimizer = OPT(
            #         chain(self.V.critic.parameters(), self.Q.parameters()),
            #         lr=self.lr_Q,
            #     )
            # else:
            #     self.whole_optimizer = OPT(
            #         chain(self.V.parameters(), self.Q.parameters()),
            #         lr=self.lr_Q,
            #     )

        self.Q_optimizer = OPT(self.Q.parameters(), lr=self.lr_Q)
        # self.whole_optimizer = OPT(
        #     chain(self.V.critic.parameters(), self.Q.parameters()), lr=self.lr_Q
        # )
        if self.algo == "cddqn" or self.algo == "dvqn":
            self.Q_optimizer2 = OPT(self.Q2.parameters(), lr=self.lr_Q)
 

    def act(self, state):
        # warm up phase
        if self.timesteps_done < self.init_steps:
            # action = self.env.action_space.sample()
            action = random.randrange(self.n_actions)
            # self.exploration_rate = 0.0
            action_prob = 1/self.n_actions
            return action, action_prob

        self._update_current_progress_remaining(self.timesteps_done, self.total_timesteps)
        # [if linear decay]
        # self.exploration_rate = self.exploration_scheduler(self._current_progress_remaining)
        # [if exponential decay]
        self.exploration_rate *= self.boltzmann_decay_rate
        # self.exploration_rate = max(self.exploration_rate * self.epsilon_decay, self.epsilon_min)
        with torch.no_grad():
            state = state[:]  # this operation required when using LazyFrames
            state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            
            if self.act_boltzmann:
                if self.algo == "dvqn":
                    q = self.Q(state)[0]
                    q2 = self.Q2(state)[0]
                    # elementwise min between q and q2
                    # q = torch.minimum(q, q2)
                    q = (q+q2)/2
                else:
                    q = self.Q(state)[0]
                action_probs = F.softmax(q/self.act_boltzmann_temperature, dim=1).squeeze()
                action = torch.multinomial(action_probs, num_samples=1).item()                
                action_prob = action_probs[action].item()
                # self.act_boltzmann_temperature = self.init_act_boltzmann_temperature * self._current_progress_remaining + 1e-10
                self.act_boltzmann_temperature *= self.boltzmann_decay_rate
                self.L.log(
                    {
                        "Info/act/explore_rate": int(action == q.argmax(dim=1).item()),
                        "Info/act/self.act_boltzmann_temperature": self.act_boltzmann_temperature,
                     })
                return action, action_prob

            if random.random() > self.exploration_rate:
                if self.algo == "dvqn" and self.use2Q:
                    q = (self.Q(state)[0] + self.Q2(state)[0]) / 2
                    # q = torch.minimum(self.Q(state)[0], self.Q2(state)[0])
                else:
                    q = self.Q(state)[0]
                action = q.argmax(dim=1).item()
                action_prob = (1-self.exploration_rate) + self.exploration_rate/self.n_actions
            else:
                action = random.randrange(self.n_actions)
                action_prob = self.exploration_rate/self.n_actions

        return action, action_prob

    def act_e_greedy(self, state, epsilon=0.001):
        with torch.no_grad():
            state = torch.from_numpy(state[:]).unsqueeze(0).to(self.device)
            if random.random() > epsilon:
                action = self.Q(state)[0].argmax(dim=1).item()
            else:
                action = random.randrange(self.n_actions)
            return action


    def update_V(
        self,
        obs,
        n_obs,
        rew,
        gamma,
        gradient_step=True,
        clip_grad_mode="clamp",
        criterion = nn.SmoothL1Loss(),
    ):
        if self.clip_reward:
            rew.clamp_(-1, 1)

        v = self.V(obs)
        with torch.no_grad():
            n_v = self.V_target(n_obs)
            # n_V = self.V(n_obs)
            v_target = rew + gamma * n_v

        td_error = criterion(v, v_target)

        if gradient_step:
            self.V_optimizer.zero_grad(set_to_none=True)
            td_error.backward()

            if clip_grad_mode == "clamp":
                for param in self.V.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-1, 1)
            elif clip_grad_mode == "norm":
                nn.utils.clip_grad_norm_(self.V.parameters(), max_norm=1.0)

            self.V_optimizer.step()
        metric = {
            "Info/V/td_error": td_error.item(),
            "Info/V/v": v.mean().item(),

        }
        self.L.log(metric)

        return

    def update_Q(
        self,
        obs,
        act,
        n_obs,
        rew,
        gamma,
        gradient_step=True,
        clip_grad_mode="clamp",
        criterion = nn.SmoothL1Loss(),
    ):
        if self.clip_reward:
            rew.clamp_(-1, 1)

        # [Update Q network]
        q, _ = self.Q(obs)
        q_std = torch.std(q, dim=1, keepdim=True)

        q = q.gather(1, act)

        q2, _ = self.Q2(obs)
        q2 = q2.gather(1, act)

        with torch.no_grad():
            v = self.V(obs)
            # V = self.V_target(obs)
            n_v = self.V(n_obs)
            q_target = rew + gamma * n_v

        td_error = criterion(q, q_target)
        td_error2 = criterion(q2, q_target)

        # [gradient descent]
        if gradient_step:
            self.Q_optimizer.zero_grad(set_to_none=True)
            self.Q_optimizer2.zero_grad(set_to_none=True)
            (td_error + td_error2).backward()

            if clip_grad_mode == "clamp":
                for param in self.Q.parameters():
                    if param.grad is not None:  # make sure grad is not None
                        param.grad.data.clamp_(-1, 1)
                for param in self.Q2.parameters():
                    if param.grad is not None:  # make sure grad is not None
                        param.grad.data.clamp_(-1, 1)
            elif clip_grad_mode == "norm":
                nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=1.0)
                nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=1.0)
            self.Q_optimizer.step()
            self.Q_optimizer2.step()

        with torch.no_grad():
            vq_diff_l1 = (v - q).mean()
            vq_diff_l2 = F.mse_loss(v, q)

        metric = {
            "Info/Q/q": q.mean().item(),
            "Info/Q/q_std": q_std.mean().item(),
            "Info/Q/q_max": q.max(1)[0].unsqueeze(1).mean().item(),
            "Info/Q/vq_l1": vq_diff_l1.item(),
            "Info/Q/vq_l2": vq_diff_l2.item(),
            "Info/Q/td_error": ((td_error + td_error2)/2).item(),
        }
        self.L.log(metric)

        return

    def update_ddqn(self, obs, act, n_obs, rew, gamma, criterion = nn.SmoothL1Loss(), clip_grad_mode="clamp"):

        if self.clip_reward:
            rew.clamp_(-1, 1)


        # [Update Q network]
        q, encoded = self.Q(obs)
        q = q.gather(1, act)

        with torch.no_grad():            # [Double DQN]
            q_next_max = self.Q_target(n_obs)[0].gather(
                1, self.Q(n_obs)[0].argmax(dim=1, keepdim=True)
            )
            # Compute target Q value
            q_target = rew + gamma * q_next_max

        td_error = criterion(q, q_target)

        self.Q_optimizer.zero_grad(set_to_none=True)
        td_error.backward()

        if clip_grad_mode == "clamp":
            for param in self.Q.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
        elif clip_grad_mode == "norm":
            nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=1.0)

        self.Q_optimizer.step()

        metric = {
            "Info/Q/td_error": td_error.item(),
            "Info/Q/q": q.mean().item(),
            "Info/Q/q_max": q.max(1)[0].unsqueeze(1).mean().item(),
        }

        self.L.log(metric)

        return

    def update_cddqn(self, obs, act, n_obs, rew, gamma, criterion = nn.SmoothL1Loss(), clip_grad_mode="clamp"):

        if self.clip_reward:
            rew.clamp_(-1, 1)

        q, encoded = self.Q(obs)
        q = q.gather(1, act)

        q2, encoded2 = self.Q2(obs)
        q2 = q2.gather(1, act)
        with torch.no_grad():
            q_next, encoded_next = self.Q_target(n_obs)
            q_next_max1 = q_next.max(1)[0].unsqueeze(1)

            q_next_max2 = self.Q2_target(n_obs)[0].gather(
                1, q_next.argmax(dim=1, keepdim=True)
            )

            q_next_max = torch.minimum(q_next_max1, q_next_max2)
            q_target = rew + gamma * q_next_max

        td_error = criterion(q, q_target)
        td_error2 = criterion(q2, q_target)
        self.Q_optimizer.zero_grad(set_to_none=True)
        self.Q_optimizer2.zero_grad(set_to_none=True)
        (td_error + td_error2).backward()
        if clip_grad_mode == "clamp":
            # 1 clamp gradients to avoid exploding gradient
            for param in self.Q.parameters():
                if param.grad is not None:  # make sure grad is not None
                    param.grad.data.clamp_(-1, 1)
            for param in self.Q2.parameters():
                if param.grad is not None:  # make sure grad is not None
                    param.grad.data.clamp_(-1, 1)
        elif clip_grad_mode == "norm":
            nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=1.0)
        self.Q_optimizer.step()
        self.Q_optimizer2.step()
        metric = {
            "Info/Q/td_error": (0.5 * td_error + 0.5 * td_error2).item(),
            "Info/Q/q": q.mean().item(),
            "Info/Q/q2": q2.mean().item(),
            "Info/Q/q_max": q.max(1)[0].unsqueeze(1).mean().item(),
            "Info/Q/q_max2": q2.max(1)[0].unsqueeze(1).mean().item(),
        }

        self.L.log(metric)

        return

    def update_avgdqn(self, obs, act, n_obs, rew, gamma, criterion = nn.SmoothL1Loss(), clip_grad_mode="clamp"):

        if self.clip_reward:
            rew.clamp_(-1, 1)

        # [Update Q network]
        q, encoded = self.Q(obs)
        q = q.gather(1, act)

        with torch.no_grad():
            q_value_list = [
                q_func(n_obs)[0] for q_func in self.target_q_net_list[-self.num_active_target :]
            ]
            avg_q_value = sum(q_value_list) / len(q_value_list)
            q_next_max = avg_q_value.max(1)[0].unsqueeze(1)

            # Compute target Q value
            q_target = rew + gamma * q_next_max

        td_error = criterion(q, q_target)

        self.Q_optimizer.zero_grad(set_to_none=True)
        td_error.backward()
        if clip_grad_mode == "clamp":
            for param in self.Q.parameters():
                if param.grad is not None:  # make sure grad is not None
                    param.grad.data.clamp_(-1, 1)
        elif clip_grad_mode == "norm":
            nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=1.0)
        self.Q_optimizer.step()

        metric = {
            "Info/Q/td_error": td_error.item(),
            "Info/Q/q": q.mean().item(),
            "Info/Q/q_max": q.max(1)[0].unsqueeze(1).mean().item(),
        }

        self.L.log(metric)

        return

    def update_dqn(self, obs, act, n_obs, rew, gamma, criterion = nn.SmoothL1Loss(), clip_grad_mode="clamp"):

        if self.clip_reward:
            rew.clamp_(-1, 1)


        # [Update Q network]
        q, encoded = self.Q(obs)
        q = q.gather(1, act)

        with torch.no_grad():
            # [Vanilla DQN]
            q_next, encoded_next = self.Q_target(n_obs)
            q_next_max = q_next.max(1)[0].unsqueeze(1)

            # Compute target Q value
            q_target = rew + gamma * q_next_max

        td_error = criterion(q, q_target)

        self.Q_optimizer.zero_grad(set_to_none=True)
        td_error.backward()
        if clip_grad_mode == "clamp":
            for param in self.Q.parameters():
                if param.grad is not None:  # make sure grad is not None
                    param.grad.data.clamp_(-1, 1)
        elif clip_grad_mode == "norm":
            nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=1.0)
        self.Q_optimizer.step()

        metric = {
            "Info/Q/td_error": td_error.item(),
            "Info/Q/q": q.mean().item(),
            "Info/Q/q_max": q.max(1)[0].unsqueeze(1).mean().item(),
        }

        self.L.log(metric)

        return

    def update_dueldqn(self, obs, act, n_obs, rew, gamma, criterion = nn.SmoothL1Loss(), clip_grad_mode="clamp"):
        assert isinstance(self.Q, DuelDQN)

        self.update_dqn(obs, act, n_obs, rew, gamma, criterion, clip_grad_mode)

    def update_sarsa(
        self,
        obs,
        act,
        n_obs,
        rew,
        gamma,
        criterion = nn.SmoothL1Loss(),
        clip_grad_mode="clamp",
    ):
        if self.clip_reward:
            rew.clamp_(-1, 1)

        # [Update Q network]
        q, _ = self.Q(obs)

        q = q.gather(1, act)

        with torch.no_grad():
            q_next, encoded_next = self.Q_target(n_obs)
            
            # boltzmann policy
            # next_action_probs = F.softmax(q_next/self.act_boltzmann_temperature, dim=1)
            # next_action = torch.multinomial(next_action_probs, num_samples=1)  

            # epsilon greedy policy
            next_action = q_next.argmax(dim=1, keepdim=True)
            mask = torch.randn_like(next_action.float()) > self.exploration_rate 
            next_action = next_action * mask + torch.randint_like(next_action, self.n_actions) * ~mask
            
            # action_prob = next_action_probs[action].item()
            q_target = rew + gamma * (q_next.gather(1, next_action))

        td_error = criterion(q, q_target)

        # [gradient descent]
        self.Q_optimizer.zero_grad(set_to_none=True)
        td_error.backward()

        if clip_grad_mode == "clamp":
            for param in self.Q.parameters():
                if param.grad is not None:  # make sure grad is not None
                    param.grad.data.clamp_(-1, 1)
        elif clip_grad_mode == "norm":
            nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=1.0)
        self.Q_optimizer.step()

        metric = {
            "Info/Q/td_error": td_error.item(),
            "Info/Q/q": q.mean().item(),
            "Info/Q/q_max": q.max(1)[0].unsqueeze(1).mean().item(),
        }
        self.L.log(metric)

    def update_contrastive(self, anc_obs, pos_obs, ema=False, clip_grad_mode="clamp"):
        anc_encoded = self.curl(anc_obs, projection=True)
        anc_encoded_l2norm = torch.linalg.norm(anc_encoded, dim=1).mean()
        # anc_encoded = F.normalize(anc_encoded, dim=1)
        anc, anc_vq_loss, anc_entrophy_vq, anc_output_dict = self.vq(anc_encoded)
        anc = F.normalize(anc, dim=1)

        # positive sample
        if ema:
            with torch.no_grad():
                pos_encoded = self.curl_ema(pos_obs, projection=True)
                pos_encoded_l2norm = torch.linalg.norm(pos_encoded, dim=1).mean()
                # pos_encoded = F.normalize(pos_encoded, dim=1)
                pos, pos_vq_loss, pos_entrophy_vq, pos_output_dict = self.vq_ema(pos_encoded)
                pos = F.normalize(pos, dim=1)
        else:
            pos_encoded = self.curl(pos_obs, projection=True)
            pos_encoded_l2norm = torch.linalg.norm(pos_encoded, dim=1).mean()
            # pos_encoded = F.normalize(pos_encoded, dim=1)
            pos, pos_vq_loss, pos_entrophy_vq, pos_output_dict = self.vq(pos_encoded)
            # pos = F.normalize(pos, dim=1)

        vq_entropy = anc_entrophy_vq + pos_entrophy_vq
        vq_loss = anc_vq_loss + pos_vq_loss
        # Normalize the codebook first
        # codebook = F.normalize(self.vq.embedding.weight, dim=1)
        # or not
        codebook = self.vq.embedding.weight

        codebook_l2norm = torch.linalg.norm(codebook, dim=1).mean()

        cb_diversity = torch.matmul(codebook, codebook.T)
        # Using mean() only makes sense when the codebook is of non-negative vectors
        # cb_diversity = cb_diversity.mean()
        # cb_diversity = torch.einsum("ij,mj->im", [codebook, codebook]).mean()

        # Or like below, cb_diversity=(W*W_T - I), like below:
        I = torch.eye(cb_diversity.shape[0])
        I = I.float().to(self.device)
        # cb_diversity = torch.sum(torch.abs((cb_diversity - I)))
        cb_diversity = torch.linalg.norm((cb_diversity - I))

        # Or a less constrained loss, cb_diversity=(W*W_T*(1-I)),like below:
        # mask1 = torch.ones(cb_diversity.shape) - torch.eye(cb_diversity.shape[0])
        # mask1 = mask1.float().to(self.device)
        # cb_diversity = torch.sum(torch.abs((cb_diversity * mask1)))
        # cb_diversity = torch.linalg.norm(cb_diversity * mask1)

        # Another way to do the loss above:
        # mask1 = ~torch.eye(codebook.shape[0], dtype=torch.bool, device=self.device)
        # cb_diversity = cb_diversity[mask1].mean()
        # cb_diversity = torch.sum(torch.abs(cb_diversity[mask1]))
        # cb_diversity = torch.linalg.norm(cb_diversity[mask1])

        # Compute the negative diversity
        mask2 = ~torch.eye(anc.shape[0], dtype=torch.bool, device=self.device)
        neg_diversity = (
            torch.matmul(anc, anc.T)[mask2].mean() + torch.matmul(pos, pos.T)[mask2].mean()
        )

        # anc = anc_output_dict["cluster_assignment"]
        # pos = pos_output_dict["cluster_assignment"]
        # anc = anc / (anc.norm(dim=1, keepdim=True)+1e-8)
        # pos = pos / (pos.norm(dim=1, keepdim=True)+1e-8)
        # anc = anc / anc.norm(dim=1, keepdim=True)
        # pos = pos / pos.norm(dim=1, keepdim=True)
        # anc = anc_output_dict["dist"]
        # pos = pos_output_dict["dist"]
        if ema:
            # [curl_loss, use MOCO style]
            logits, labels = self.curl.curl_loss(anc, pos, temperature=1)
        else:
            # [simclr_loss2 use one style of computing simclr loss, simclr_loss1 use another style, they are equivalent]
            logits, labels = simclr_loss2(anc, pos, temperature=0.5)
        loss1 = F.cross_entropy(logits, labels)
        with torch.no_grad():
            correct = torch.argmax(logits, dim=1) == labels
            contrast_acc = torch.mean(correct.float())
        # loss2 = self.push_away(pos)

        # [compute simclr loss with debiased, same style as simclr_loss1]
        # loss1 = simclr_debiased_loss(
        #     anc, pos, temperature=0.5, debiased=True, tau_plus=1 / self.num_vq_embeddings
        # )
        # contrast_acc = torch.tensor(0.0)

        total_loss = (
            loss1 * 1.0
            + cb_diversity * self.curl_vq_cfg[0]
            - vq_entropy * self.curl_vq_cfg[1]
            + neg_diversity * 0.0
            + vq_loss * self.curl_vq_cfg[2]
        )
        # total_loss = loss1 - vq_entropy * 0.5
        # total_loss = loss1 + loss2 - anc_entrophy_vq * 0.5 - pos_entrophy_vq * 0.5

        self.curl_optimizer.zero_grad()
        self.vq_optimizer.zero_grad()
        total_loss.backward()

        self.curl_optimizer.step()
        self.vq_optimizer.step()

        cluster_metrics = anc_output_dict["cluster_metric"]

        metric = {
            "Info/contrastive_vq/contrastive_loss": loss1.item(),
            "Info/contrastive_vq/vq_loss": vq_loss.item(),
            "Info/contrastive_vq/entrophy_vq": anc_entrophy_vq.item(),
            "Info/contrastive_vq/update_contrastive total_loss": total_loss.item(),
            "Info/contrastive_vq/contrast_acc": contrast_acc.item(),
            "Info/contrastive_vq/cluster_metrics": cluster_metrics,
            "Info/contrastive_vq/vq_codebook_diversity": cb_diversity.item(),
            "Info/contrastive_vq/neg_diversity": neg_diversity.item(),
            "Info/contrastive_vq/anc_encoded_l2norm": anc_encoded_l2norm.item(),
            "Info/contrastive_vq/codebook_l2norm": codebook_l2norm.item(),
        }
        self.L.log(metric)

    def update_contrastive_novq(self, anc_obs, pos_obs, ema=False):
        # # [data augmentation]
        # if self.input_format == "full_img":
        #     anc_obs = self.aug(anc_obs)
        #     pos_obs = self.aug(pos_obs)
        anc = self.curl(anc_obs)
        anc = F.normalize(anc, dim=1)
        if ema:
            pos = self.curl_ema(pos_obs)
        else:
            pos = self.curl(pos_obs)
        pos = F.normalize(pos, dim=1)

        mask2 = ~torch.eye(anc.shape[0], dtype=torch.bool, device=self.device)
        neg_diversity = (
            torch.matmul(anc, anc.T)[mask2].mean() + torch.matmul(pos, pos.T)[mask2].mean()
        )

        if ema:
            logits, labels = self.curl.curl_loss(anc, pos, temperature=1)
        else:
            logits, labels = simclr_loss2(anc, pos, temperature=0.5)
        loss1 = F.cross_entropy(logits, labels)
        with torch.no_grad():
            correct = torch.argmax(logits, dim=1) == labels
            contrast_acc = torch.mean(correct.float())
        # loss2 = self.push_away(pos)

        total_loss = loss1 * 1.0 + neg_diversity * 0.0
        # total_loss = loss1 - vq_entropy * 0.5
        # total_loss = loss1 + loss2 - anc_entrophy_vq * 0.5 - pos_entrophy_vq * 0.5

        self.curl_optimizer.zero_grad()
        total_loss.backward()
        self.curl_optimizer.step()
        metric = {
            "Info/contrastive/contrastive_loss": loss1.item(),
            "Info/contrastive/update_contrastive total_loss": total_loss.item(),
            "Info/contrastive/contrast_acc": contrast_acc.item(),
            "Info/contrastive/neg_diversity": neg_diversity.item(),
        }
        self.L.log(metric)

        return total_loss

    def update_contrastive_atc(self, anc_obs, pos_obs):
        # # [data augmentation]
        # if self.input_format == "full_img":
        #     anc_obs = self.aug(anc_obs)
        #     pos_obs = self.aug(pos_obs)
        logits = self.curl(anc_obs, pos_obs)
        labels = torch.arange(logits.shape[0]).long().to(logits.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            correct = torch.argmax(logits, dim=1) == labels
            contrast_acc = torch.mean(correct.float())

        self.curl_optimizer.zero_grad()
        loss.backward()
        self.curl_optimizer.step()
        metric = {
            "Info/contrastive/contrastive_loss": loss.item(),
            "Info/contrastive/contrast_acc": contrast_acc.item(),
        }
        self.L.log(metric)



    def update_hp(self):
        if hasattr(self, "lr_scheduler_curl"):
            self.lr_curl = self.lr_scheduler_curl(self._current_progress_remaining)
            update_learning_rate(self.curl_optimizer, self.lr_curl)

        if hasattr(self, "lr_scheduler_vq"):
            self.lr_vq = self.lr_scheduler_vq(self._current_progress_remaining)
            update_learning_rate(self.vq_optimizer, self.lr_vq)

        if hasattr(self, "lr_scheduler_Q"):
            self.lr_Q = self.lr_scheduler_Q(self._current_progress_remaining)
            update_learning_rate(
                self.Q_optimizer,
                self.lr_Q,
            )

        if hasattr(self, "lr_scheduler_V"):
            self.lr_V = self.lr_scheduler_V(self._current_progress_remaining)
            update_learning_rate(self.V_optimizer, self.lr_V)

        self.L.log(
            {
                "HP/lr_Q": self.lr_Q,
                "HP/exploration_rate": self.exploration_rate,
                "HP/act_boltzmann_temperature": self.act_boltzmann_temperature,
                "HP/current_progress_remaining": self._current_progress_remaining,
            }
        )
        if self.algo == "dvqn":
            self.L.log(
                {
                    "HP/lr_V": self.lr_V,
                }
            )
        if self.use_curl:
            self.L.log({"HP/lr_curl": self.lr_curl})
        if self.use_vq:
            self.L.log({"HP/lr_vq": self.lr_vq})

    def update(self):
        self.update_hp()
        self.n_update += 1

        if self.timesteps_done < self.init_steps:
            return
        if self.timesteps_done == self.init_steps:
            print("Warm up done")

        steps = self.timesteps_done - self.init_steps
        if steps % self.Q_learn_every == 0:
            if self.algo == "dvqn" and self.use_n_newdata > 0:
                obs, act, n_obs, rew, gamma, info = self.memory.sample(self.batch_size)
                obs2, act2, n_obs2, rew2, gamma2, info2 = self.memory.sample_latest_n(self.use_n_newdata)
                obs = torch.cat([obs, obs2], dim=0)
                act = torch.cat([act, act2], dim=0)
                n_obs = torch.cat([n_obs, n_obs2], dim=0)
                rew = torch.cat([rew, rew2], dim=0)
                gamma= torch.cat([gamma, gamma2], dim=0)
                # permute the data
                idx = torch.randperm(obs.shape[0])
                obs = obs[idx]
                act = act[idx]
                n_obs = n_obs[idx]
                rew = rew[idx]
                gamma = gamma[idx]
            else:
                obs, act, n_obs, rew, gamma, info = self.memory.sample(self.batch_size)

            # [data augmentation]
            if self.input_format == "full_img":
                with torch.no_grad():
                    obs = self.aug(obs)
                    n_obs = self.aug(n_obs)
                    if self.curl_pair == "raw":
                        pos = self.aug(obs)
                    else:
                        pos = n_obs
            if self.algo == "dvqn":
                self.update_V(
                    obs,
                    n_obs,
                    rew,
                    gamma,
                    gradient_step=True,
                    clip_grad_mode=self.clip_grad_mode,
                    criterion=self.criterion
                )
                self.update_Q(
                    obs,
                    act,
                    n_obs,
                    rew,
                    gamma,
                    gradient_step=True,
                    clip_grad_mode=self.clip_grad_mode,
                    criterion=self.criterion
                )

            elif self.algo == "ddqn":
                self.update_ddqn(
                    obs,
                    act,
                    n_obs,
                    rew,
                    gamma,
                    criterion=self.criterion,
                    clip_grad_mode=self.clip_grad_mode,
                )
            elif self.algo == "cddqn":
                self.update_cddqn(
                    obs,
                    act,
                    n_obs,
                    rew,
                    gamma,
                    criterion=self.criterion,
                    clip_grad_mode=self.clip_grad_mode,
                )
            elif self.algo == "sarsa":
                self.update_sarsa(
                    obs,
                    act,
                    n_obs,
                    rew,
                    gamma,
                    criterion=self.criterion,
                    clip_grad_mode=self.clip_grad_mode,
                )
            elif self.algo == "dueldqn":
                self.update_dueldqn(
                    obs,
                    act,
                    n_obs,
                    rew,
                    gamma,
                    criterion=self.criterion,
                    clip_grad_mode=self.clip_grad_mode,
                )
            elif self.algo == "dqn":
                self.update_dqn(
                    obs, 
                    act, 
                    n_obs, 
                    rew, 
                    gamma, 
                    criterion=self.criterion, 
                    clip_grad_mode=self.clip_grad_mode
                )
            elif self.algo == "avgdqn":
                self.update_avgdqn(
                    obs,
                    act,
                    n_obs,
                    rew,
                    gamma,
                    criterion=self.criterion,
                    clip_grad_mode=self.clip_grad_mode,
                )
            
        if self.use_curl and steps % self.curl_learn_every == 0:
            for _ in range(self.curl_gradient_steps):
                if self.use_vq:
                    self.update_contrastive(obs, pos, ema=True)
                elif self.curl_pair == "atc":
                    self.update_contrastive_atc(obs, pos)
                else:
                    self.update_contrastive_novq(obs, pos, ema=True)

        if steps % self.Q_sync_every == 0:
            if self.algo == "avgdqn":
                if self.num_active_target < self.avgdqn_k:
                    self.num_active_target += 1

                for idx, target_q_func in enumerate(self.target_q_net_list):
                    if idx != len(self.target_q_net_list) - 1:  # if not last target q function
                        target_q_func.load_state_dict(self.target_q_net_list[idx + 1].state_dict())
                    else:
                        target_q_func.load_state_dict(self.Q.state_dict())
            else:
                soft_sync_params(
                    self.Q.parameters(),
                    self.Q_target.parameters(),
                    self.Q_encoder_tau,
                )
            if hasattr(self, "Q2") and hasattr(self, "Q2_target"):
                soft_sync_params(
                    self.Q2.parameters(),
                    self.Q2_target.parameters(),
                    self.Q_encoder_tau,
                )
            # soft_sync_params(
            #     self.Q.encoder.parameters(),
            #     self.Q_target.encoder.parameters(),
            #     self.Q_encoder_tau,
            # )

            # soft_sync_params(
            #     self.Q.critic.parameters(),
            #     self.Q_target.critic.parameters(),
            #     self.Q_critic_tau,
            # )

        if self.algo == 'dvqn' and steps % self.V_sync_every == 0:
            soft_sync_params(
                self.V.parameters(),
                self.V_target.parameters(),
                self.V_encoder_tau,
            )
            # soft_sync_params(
            #     self.V.encoder.parameters(),
            #     self.V_target.encoder.parameters(),
            #     self.V_encoder_tau,
            # )
            # soft_sync_params(
            #     self.V.critic.parameters(),
            #     self.V_target.critic.parameters(),
            #     self.V_critic_tau,
            # )

        if self.use_curl and steps % self.curl_sync_every == 0:
            if self.curl_pair == "atc":
                soft_sync_params(
                    self.curl.encoder.parameters(),
                    self.curl.encoder_target.parameters(),
                    self.curl_tau,
                )
            else:
                soft_sync_params(
                    self.curl.parameters(),
                    self.curl_ema.parameters(),
                    self.curl_tau,
                )
        self.L.dump2wandb(agent=self)