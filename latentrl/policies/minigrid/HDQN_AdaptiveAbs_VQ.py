import copy
import io
import math
import random
from collections import Counter, deque, namedtuple
from pprint import pprint
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import wandb
from common.batch_kmeans import Batch_KMeans
from common.learning_scheduler import EarlyStopping, ReduceLROnPlateau
from common.utils import (
    Dataset_pretrain,
    get_linear_fn,
    linear_schedule,
    polyak_sync,
    soft_sync_params,
    update_learning_rate,
    wandb_log_image,
)
from envs import FourRoomsEnv
from matplotlib import pyplot as plt
from nn_models import (
    DQN,
    V_MLP,
    Decoder,
    Decoder_MiniGrid,
    DQN_Repara,
    Encoder,
    Encoder_MiniGrid_PartialObs,
    Encoder_MiniGrid,
    RandomShiftsAug,
    RandomEncoder,
    RandomEncoderMiniGrid,
    CURL,
    VectorQuantizerLinear,
    VectorQuantizerLinearSoft,
)
from PIL import Image
from policies.HDQN import HDQN
from policies.utils import EncoderMaker, ReplayMemory, ReplayMemoryWithCluster
from sklearn.cluster import KMeans
from sympy.solvers import solve
from torch import Tensor, nn
from torchsummary import summary
from torchvision.utils import save_image

from minigrid import Wall


class HDQN_AdaptiveAbs_VQ(HDQN):
    def __init__(self, config, env):
        super().__init__(config, env)
        # self.set_hparams(config)
        self.n_actions = env.action_space.n

        self.ground_Q = DQN(
            observation_space=env.observation_space,
            action_space=env.action_space,
            encoder_maker=EncoderMaker(input_format=config.input_format, agent=self),
            mlp_hidden_dim_grd=config.mlp_hidden_dim_grd,
        ).to(self.device)

        self.ground_Q_target = DQN(
            observation_space=env.observation_space,
            action_space=env.action_space,
            encoder_maker=EncoderMaker(input_format=config.input_format, agent=self),
            mlp_hidden_dim_grd=config.mlp_hidden_dim_grd,
        ).to(self.device)

        self.ground_Q_target.load_state_dict(self.ground_Q.state_dict())
        # self.ground_Q_target.train()

        # self.vq = VectorQuantizerLinear(
        #     num_embeddings=config.num_vq_embeddings,
        #     embedding_dim=config.dim_vq_embeddings,
        #     beta=0.25,
        # ).to(self.device)

        self.vq = VectorQuantizerLinearSoft(
            num_embeddings=config.num_vq_embeddings,
            embedding_dim=config.dim_vq_embeddings,
            beta=0.25,
            softmin_beta=config.vq_softmin_beta,
        ).to(self.device)

        self.abs_V_MLP = V_MLP(
            input_dim=self.ground_Q.critic_input_dim, mlp_hidden_dim_abs=config.mlp_hidden_dim_abs
        ).to(self.device)
        self.abs_V_MLP_target = V_MLP(
            input_dim=self.ground_Q.critic_input_dim, mlp_hidden_dim_abs=config.mlp_hidden_dim_abs
        ).to(self.device)
        self.abs_V_MLP_target.load_state_dict(self.abs_V_MLP.state_dict())

        self.aug = RandomShiftsAug(pad=4)
        self.memory = ReplayMemory(self.size_replay_memory, self.device)

        self.count_vis = 0
        self.goal_found = False

        self._create_optimizers(config)
        self.train()
        self.reset_training_info()

    def train(self, training=True):
        self.training = training
        self.ground_Q.train(training)
        self.abs_V_MLP.train(training)

    def reset_training_info(self):
        self.training_info = {
            "ground_Q_error": [],
            "abstract_V_error": [],
            "avg_shaping": [],
            "entrophy_vq": [],
            "vq_loss": [],
            "absV_grdQ(d)_diff": [],
            "absV_grdQ(d)_mse": [],
            "total_loss": [],
        }

    def set_hparams(self, config):
        # Hyperparameters
        # self.total_episodes = config.total_episodes
        self.total_timesteps = config.total_timesteps
        self.init_steps = config.init_steps  # min. experiences before training
        self.batch_size = config.batch_size
        self.size_replay_memory = config.size_replay_memory
        self.gamma = config.gamma
        self.abs_gamma = config.abstract_gamma
        self.omega = config.omega
        self.ground_Q_critic_tau = config.ground_Q_critic_tau
        self.ground_Q_encoder_tau = config.ground_Q_encoder_tau
        self.abstract_tau = config.abstract_tau
        self.grd_hidden_channels = config.grd_hidden_channels
        self.grd_embedding_dim = config.grd_embedding_dim
        self.mlp_hidden_dim_abs = config.mlp_hidden_dim_abs
        self.mlp_hidden_dim_grd = config.mlp_hidden_dim_grd

        self.num_vq_embeddings = config.num_vq_embeddings
        self.dim_vq_embeddings = config.dim_vq_embeddings

        self.ground_learn_every = config.ground_learn_every
        self.ground_gradient_steps = config.ground_gradient_steps
        self.ground_sync_every = config.ground_sync_every

        self.abstract_learn_every = config.abstract_learn_every
        self.abstract_gradient_steps = config.abstract_gradient_steps
        self.abstract_sync_every = config.abstract_sync_every

        self.reset_training_info_every = config.reset_training_info_every

        self.clip_grad = config.clip_grad
        self.clip_reward = config.clip_reward
        self.encoded_detach4abs = config.encoded_detach4abs
        self.input_format = config.input_format
        self.use_shaping = config.use_shaping

    def log_training_info(self, wandb_log=True):
        if wandb_log:
            metrics = {
                "Info/ground_Q_error": mean(self.training_info["ground_Q_error"])
                if len(self.training_info["ground_Q_error"]) > 0
                else 0,
                "Info/abstract_V_error": mean(self.training_info["abstract_V_error"])
                if len(self.training_info["abstract_V_error"])
                else 0,
                "Info/avg_shaping": mean(self.training_info["avg_shaping"])
                if len(self.training_info["avg_shaping"])
                else 0,
                "Info/entrophy_vq": mean(self.training_info["entrophy_vq"])
                if len(self.training_info["entrophy_vq"])
                else 0,
                "Info/vq_loss": mean(self.training_info["vq_loss"])
                if len(self.training_info["vq_loss"])
                else 0,
                "Info/absV_grdQ(d)_diff": mean(self.training_info["absV_grdQ(d)_diff"])
                if len(self.training_info["absV_grdQ(d)_diff"])
                else 0,
                "Info/absV_grdQ(d)_mse": mean(self.training_info["absV_grdQ(d)_mse"])
                if len(self.training_info["absV_grdQ(d)_mse"])
                else 0,
                "Info/total_loss": mean(self.training_info["total_loss"])
                if len(self.training_info["total_loss"])
                else 0,
                # "info/timesteps_done": self.timesteps_done,
                # "info/episodes_done": self.episodes_done,
                "Info/exploration_rate": self.exploration_rate,
                "Info/current_progress_remaining": self._current_progress_remaining,
                # "lr/lr_ground_Q_optimizer": self.ground_Q_optimizer.param_groups[0]["lr"],
                # "lr/lr_abstract_V_optimizer": self.abstract_V_optimizer.param_groups[0]["lr"],
                "Info/lr_ground_Q": self.lr_grd_Q,
                "Info/lr_abstract_V": self.lr_abs_V,
                "Info/timesteps_done": self.timesteps_done,
                "Info/episodes_done": self.episodes_done,
            }
            wandb.log(metrics)

            # print("logging training info:")
            # pp(metrics)

    @torch.no_grad()
    def encode_state(self, state):
        # assert isinstance(state, np.ndarray)
        if len(state.shape) == 4:
            return self.ground_Q.forward_conv(state)
            # return self.ground_Q.encoder(state).cpu().numpy()
        state = state[:]
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        return self.ground_Q.forward_conv(state)
        # return self.ground_Q.encoder(state).squeeze().cpu().numpy()

    def cache(self, state, action, next_state, reward, terminated, info):
        """Add the experience to memory"""
        self.memory.push(state, action, next_state, reward, terminated, info)
        return 0

    def cache_goal_transition(self):
        print("cache goal transitions")
        # temp = np.zeros((self.env.width, self.env.height, 3))

        state1 = self.env.unwrapped.grid.copy().encode()
        state1[self.env.width - 3][self.env.height - 2] = np.array([10, 0, 0])
        next_state1 = self.env.unwrapped.grid.copy().encode()
        next_state1[self.env.width - 2][self.env.height - 2] = np.array([10, 0, 0])

        state2 = self.env.unwrapped.grid.copy().encode()
        state2[self.env.width - 2][self.env.height - 3] = np.array([10, 0, 1])
        next_state2 = self.env.unwrapped.grid.copy().encode()
        next_state2[self.env.width - 2][self.env.height - 2] = np.array([10, 0, 1])

        goal_pos = (self.env.width - 2, self.env.height - 2)
        info1 = {
            "agent_pos1": (goal_pos[0] - 1, goal_pos[1]),
            "agent_dir1": 0,
            "agent_pos2": goal_pos,
            "agent_dir2": 0,
            "interval4SemiMDP": 1,
        }
        info2 = {
            "agent_pos1": (goal_pos[0], goal_pos[1] - 1),
            "agent_dir1": 1,
            "agent_pos2": goal_pos,
            "agent_dir2": 1,
            "interval4SemiMDP": 1,
        }
        # sample a reward from uniform distribution in range [0, 0.5)
        reward1 = np.random.uniform(0, 0.5)
        reward2 = np.random.uniform(0, 0.5)
        self.cache(state1, 2, next_state1, reward1, True, info1)
        self.cache(state2, 2, next_state2, reward2, True, info2)
        print("Inject goal transitions with reward1: {}, reward2: {}".format(reward1, reward2))

    @torch.no_grad()
    def get_abs_state_idx(self, state):
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        encoded_state = self.ground_Q.forward_conv(state)
        _, _, _, output_dict = self.vq(encoded_state)
        abstract_state_inds = output_dict["hard_encoding_inds"]
        return abstract_state_inds.squeeze().item()

    @torch.no_grad()
    def get_abs_value(self, state, mode="soft"):
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        encoded = self.ground_Q.forward_conv(state)
        quantized, _, _, output_dict = self.vq(encoded)
        if mode == "soft":
            abs_value = self.abs_V_MLP(quantized)
        elif mode == "hard":
            abs_value = self.abs_V_MLP(output_dict["hard_quantized_latents"])
        return abs_value.squeeze().item()

    @torch.no_grad()
    def get_grd_reduction_v(self, state, reduction_mode="max"):
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        grd_q, encoded = self.ground_Q(state)
        # grd_q, _ = self.ground_Q_target(state)
        if reduction_mode == "max":
            return grd_q.squeeze().max().item()
        elif reduction_mode == "mean":
            return grd_q.squeeze().mean().item()

    def vis_abstraction(self, prefix: str = None):
        width = self.env.width
        height = self.env.height

        clustersN = np.full(shape=(height, width), fill_value=-1)
        clustersS = np.full(shape=(height, width), fill_value=-1)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=-1)
        clustersE = np.full(shape=(height, width), fill_value=-1)
        for w in range(width):
            # w += 1
            for h in range(height):
                # h += 1
                if not isinstance(self.env.grid.get(w, h), Wall):
                    for dir in range(4):
                        if self.input_format == "partial_obs":
                            env_ = copy.deepcopy(self.env)
                            env_.agent_pos = (w, h)
                            env_.agent_dir = dir
                            grid, vis_mask = env_.gen_obs_grid(agent_view_size=7)
                            state = grid.encode(vis_mask)
                        elif self.input_format == "full_img":
                            state = self.env.unwrapped.grid.copy().render(
                                tile_size=self.env.tile_size, agent_pos=(w, h), agent_dir=dir
                            )
                        elif self.input_format == "full_obs":
                            state = self.env.unwrapped.grid.copy().encode()
                            state[w][h] = np.array([10, 0, dir])
                        abstract_state_idx = self.get_abs_state_idx(state)
                        if dir == 3:
                            clustersN[h, w] = abstract_state_idx
                        if dir == 0:
                            clustersE[h, w] = abstract_state_idx
                        if dir == 1:
                            clustersS[h, w] = abstract_state_idx
                        if dir == 2:
                            clustersW[h, w] = abstract_state_idx

        values = [clustersN, clustersE, clustersS, clustersW]
        triangulations = self.triangulation_for_triheatmap(width, height)

        # [Plot Abstraction]
        fig_abs, ax_abs = plt.subplots(figsize=(5, 5))
        vmax = self.num_vq_embeddings
        vmin = 0
        my_cmap = copy.copy(plt.cm.get_cmap("gist_ncar"))
        my_cmap.set_under(color="dimgray")
        imgs = [
            ax_abs.tripcolor(
                t,
                np.ravel(val),
                vmin=vmin,
                vmax=vmax,
                cmap=my_cmap,
                ec="black",
            )
            for t, val in zip(triangulations, values)
        ]

        ax_abs.invert_yaxis()
        fig_abs.tight_layout()

        img_buffer = io.BytesIO()
        fig_abs.savefig(
            img_buffer,
            dpi=100,
            # facecolor="w",
            # edgecolor="w",
            # orientation="portrait",
            # transparent=False,
            # bbox_inches=None,
            # pad_inches=0.1,
            format="png",
        )
        img = Image.open(img_buffer)
        wandb.log({"Images/abstraction": wandb.Image(img)})
        img_buffer.close()
        plt.close(fig_abs)

    def vis_abstract_values(self, prefix: str = None, mode="soft"):
        width = self.env.width
        height = self.env.height
        clustersN = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)
        clustersE = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)
        clustersS = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)

        abs_values = []
        for w in range(width):
            # w += 1
            for h in range(height):
                # h += 1
                # abstract_state_idx, abstract_value = self.get_abstract_value((w, h))
                if not isinstance(self.env.grid.get(w, h), Wall):
                    for dir in range(4):
                        if self.input_format == "partial_obs":
                            env_ = copy.deepcopy(self.env)
                            env_.agent_pos = (w, h)
                            env_.agent_dir = dir
                            grid, vis_mask = env_.gen_obs_grid(agent_view_size=7)
                            state = grid.encode(vis_mask)
                        elif self.input_format == "full_img":
                            state = self.env.unwrapped.grid.copy().render(
                                tile_size=self.env.tile_size, agent_pos=(w, h), agent_dir=dir
                            )
                        elif self.input_format == "full_obs":
                            state = self.env.unwrapped.grid.copy().encode()
                            state[w][h] = np.array([10, 0, dir])
                        abstract_value = self.get_abs_value(state, mode=mode)
                        abs_values.append(abstract_value)
                        if dir == 3:
                            clustersN[h, w] = abstract_value
                        if dir == 0:
                            clustersE[h, w] = abstract_value
                        if dir == 1:
                            clustersS[h, w] = abstract_value
                        if dir == 2:
                            clustersW[h, w] = abstract_value

        values = [clustersN, clustersE, clustersS, clustersW]

        triangulations = self.triangulation_for_triheatmap(width, height)

        # [Plot Abstract Values]
        fig_abs_v, ax_abs_v = plt.subplots(figsize=(5, 5))
        vmin = min(abs_values)
        vmax = max(abs_values) + 1e-10
        # vmin = vmin - 0.07 * (vmax - vmin)
        my_cmap = copy.copy(plt.cm.get_cmap("hot"))
        my_cmap.set_bad(color="green")
        imgs2 = [
            ax_abs_v.tripcolor(
                t,
                np.ravel(val),
                vmin=vmin,
                vmax=vmax,
                # norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=my_cmap,
                ec="black",
                lw=0.02,
            )
            for t, val in zip(triangulations, values)
        ]
        # round the values

        # xx, yy = np.meshgrid(self.abs_txt_ticks, self.abs_txt_ticks)
        # xx = xx.flatten()
        # yy = yy.flatten()

        ax_abs_v.invert_yaxis()
        ax_abs_v.set_xlabel(f"vmin: {vmin:.6f}, vmax: {vmax:.6f}")
        fig_abs_v.tight_layout()

        img_buffer = io.BytesIO()
        fig_abs_v.savefig(
            img_buffer,
            dpi=100,
            # facecolor="w",
            # edgecolor="w",
            # orientation="portrait",
            # transparent=False,
            # bbox_inches=None,
            # pad_inches=0.1,
            format="png",
        )
        # wandb.define_metric("Images/time_steps_done")
        # wandb.define_metric("Images/abs_values", step_metric="Images/time_steps_done")
        img = Image.open(img_buffer)
        wandb.log({f"Images/abs_values_{mode}": wandb.Image(img)})
        img_buffer.close()
        plt.close(fig_abs_v)

    def vis_grd_q_values(self, prefix: str = None, norm_log=50, reduction_mode="max"):
        width = self.env.width
        height = self.env.height
        clustersN = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)
        clustersE = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)
        clustersS = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)

        grd_v_avg_values = []
        for w in range(width):
            # w += 1
            for h in range(height):
                # h += 1
                # abstract_state_idx, abstract_value = self.get_abstract_value((w, h))
                if not isinstance(self.env.grid.get(w, h), Wall):
                    for dir in range(4):
                        if self.input_format == "partial_obs":
                            env_ = copy.deepcopy(self.env)
                            env_.agent_pos = (w, h)
                            env_.agent_dir = dir
                            grid, vis_mask = env_.gen_obs_grid(agent_view_size=7)
                            state = grid.encode(vis_mask)
                        elif self.input_format == "full_img":
                            state = self.env.unwrapped.grid.copy().render(
                                tile_size=self.env.tile_size, agent_pos=(w, h), agent_dir=dir
                            )
                        elif self.input_format == "full_obs":
                            state = self.env.unwrapped.grid.copy().encode()
                            state[w][h] = np.array([10, 0, dir])
                        grd_v_avg = self.get_grd_reduction_v(state, reduction_mode=reduction_mode)
                        grd_v_avg = np.log(grd_v_avg + 1e-5) / np.log(norm_log)
                        grd_v_avg_values.append(grd_v_avg)
                        if dir == 3:
                            clustersN[h, w] = grd_v_avg
                        if dir == 0:
                            clustersE[h, w] = grd_v_avg
                        if dir == 1:
                            clustersS[h, w] = grd_v_avg
                        if dir == 2:
                            clustersW[h, w] = grd_v_avg

        values = [clustersN, clustersE, clustersS, clustersW]

        triangulations = self.triangulation_for_triheatmap(width, height)

        # [Plot Abstract Values]
        fig_abs_v, ax_grd_v = plt.subplots(figsize=(5, 5))
        vmin = min(grd_v_avg_values)
        vmax = max(grd_v_avg_values) + 1e-10
        # vmin = vmin - 0.07 * (vmax - vmin)
        my_cmap = copy.copy(plt.cm.get_cmap("hot"))
        my_cmap.set_bad(color="green")
        imgs2 = [
            ax_grd_v.tripcolor(
                t,
                np.ravel(val),
                vmin=vmin,
                vmax=vmax,
                # norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=my_cmap,
                ec="black",
                lw=0.02,
            )
            for t, val in zip(triangulations, values)
        ]
        # round the values

        # xx, yy = np.meshgrid(self.abs_txt_ticks, self.abs_txt_ticks)
        # xx = xx.flatten()
        # yy = yy.flatten()

        ax_grd_v.invert_yaxis()
        ax_grd_v.set_xlabel(f"vmin: {vmin:.6f}, vmax: {vmax:.6f}")
        fig_abs_v.tight_layout()

        img_buffer = io.BytesIO()
        fig_abs_v.savefig(
            img_buffer,
            dpi=100,
            # facecolor="w",
            # edgecolor="w",
            # orientation="portrait",
            # transparent=False,
            # bbox_inches=None,
            # pad_inches=0.1,
            format="png",
        )
        # wandb.define_metric("Images/time_steps_done")
        # wandb.define_metric("Images/abs_values", step_metric="Images/time_steps_done")
        img = Image.open(img_buffer)
        wandb.log({f"Images/grd_values_{reduction_mode}": wandb.Image(img)})
        img_buffer.close()
        plt.close(fig_abs_v)

    def vis_grd_visits(self, norm_log: 0, suffix: str = None):
        width = self.env.width
        height = self.env.height
        clustersN = np.full(shape=(height, width), fill_value=np.nan)
        clustersE = np.full(shape=(height, width), fill_value=np.nan)
        clustersS = np.full(shape=(height, width), fill_value=np.nan)
        clustersW = np.full(shape=(height, width), fill_value=np.nan)

        grd_visits_to_vis = copy.deepcopy(self.grd_visits)
        if norm_log:
            grd_visits_to_vis = np.log(grd_visits_to_vis + 1) / np.log(norm_log)

        for w in range(width - 2):
            w += 1
            for h in range(height - 2):
                h += 1
                for i in range(4):
                    if i == 0:
                        clustersE[h, w] = grd_visits_to_vis[h, w, i]
                    if i == 1:
                        clustersS[h, w] = grd_visits_to_vis[h, w, i]
                    if i == 2:
                        clustersW[h, w] = grd_visits_to_vis[h, w, i]
                    if i == 3:
                        clustersN[h, w] = grd_visits_to_vis[h, w, i]

        values = [clustersN, clustersE, clustersS, clustersW]

        triangulations = self.triangulation_for_triheatmap(width, height)

        # [Plotting]
        fig_grd_visits, ax_grd_visits = plt.subplots(figsize=(5, 5))
        vmin = grd_visits_to_vis.min()
        vmax = grd_visits_to_vis.max()
        my_cmap = copy.copy(plt.cm.get_cmap("hot"))
        my_cmap.set_bad(color="green")
        imgs = [
            ax_grd_visits.tripcolor(
                t,
                np.ravel(val),
                vmin=vmin,
                vmax=vmax,
                # norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=my_cmap,
                ec="black",
                lw=0.02,
            )
            for t, val in zip(triangulations, values)
        ]
        ax_grd_visits.invert_yaxis()
        # cax = fig_grd_visits.add_axes([0.9, 0.23, 0.03, 0.5])
        # fig_grd_visits.colorbar(ax_grd_visits, cax=cax)
        xlabel = f"vmin:{self.grd_visits.min()}, vmax:{self.grd_visits.max()}, sum:{self.grd_visits.sum()}, eps:{self.exploration_rate}"
        ax_grd_visits.set_xlabel(xlabel)
        fig_grd_visits.tight_layout()
        img_buffer = io.BytesIO()
        fig_grd_visits.savefig(
            img_buffer,
            dpi=100,
            # facecolor="w",
            # edgecolor="w",
            # orientation="portrait",
            # transparent=False,
            # bbox_inches=None,
            # pad_inches=0.1,
            format="png",
        )
        # wandb.define_metric("Images/time_steps_done")
        # wandb.define_metric("Images/abs_values", step_metric="Images/time_steps_done")
        img = Image.open(img_buffer)
        # print("save visits img")
        # img.save("/workspace/repos_dev/VQVAE_RL/figures/minigrid_abstraction/grd_visits.png")
        wandb.log({f"Images/grd_visits_log{norm_log}_{suffix}": wandb.Image(img)})
        img_buffer.close()
        plt.close(fig_grd_visits)

    def vis_reward_distribution(self, suffix: str = None):
        width = self.env.width
        height = self.env.height
        clustersN = np.full(shape=(height, width), fill_value=np.nan)
        clustersE = np.full(shape=(height, width), fill_value=np.nan)
        clustersS = np.full(shape=(height, width), fill_value=np.nan)
        clustersW = np.full(shape=(height, width), fill_value=np.nan)

        for w in range(width - 2):
            w += 1
            for h in range(height - 2):
                h += 1
                for i in range(4):
                    if i == 0:
                        clustersE[h, w] = self.grd_reward_distribution[h, w, i]
                    if i == 1:
                        clustersS[h, w] = self.grd_reward_distribution[h, w, i]
                    if i == 2:
                        clustersW[h, w] = self.grd_reward_distribution[h, w, i]
                    if i == 3:
                        clustersN[h, w] = self.grd_reward_distribution[h, w, i]

        values = [clustersN, clustersE, clustersS, clustersW]

        triangulations = self.triangulation_for_triheatmap(width, height)

        # [Plotting]
        fig_grd_visits, ax_grd_visits = plt.subplots(figsize=(5, 5))
        vmin = self.grd_reward_distribution.min()
        vmax = self.grd_reward_distribution.max()
        my_cmap = copy.copy(plt.cm.get_cmap("hot"))
        my_cmap.set_bad(color="green")
        imgs = [
            ax_grd_visits.tripcolor(
                t,
                np.ravel(val),
                vmin=vmin,
                vmax=vmax,
                # norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=my_cmap,
                ec="black",
                lw=0.02,
            )
            for t, val in zip(triangulations, values)
        ]
        ax_grd_visits.invert_yaxis()
        # cax = fig_grd_visits.add_axes([0.9, 0.23, 0.03, 0.5])
        # fig_grd_visits.colorbar(ax_grd_visits, cax=cax)
        xlabel = f"vmin:{vmin}, vmax:{vmax}"
        ax_grd_visits.set_xlabel(xlabel)
        fig_grd_visits.tight_layout()
        img_buffer = io.BytesIO()
        fig_grd_visits.savefig(
            img_buffer,
            dpi=100,
            # facecolor="w",
            # edgecolor="w",
            # orientation="portrait",
            # transparent=False,
            # bbox_inches=None,
            # pad_inches=0.1,
            format="png",
        )
        # wandb.define_metric("Images/time_steps_done")
        # wandb.define_metric("Images/abs_values", step_metric="Images/time_steps_done")
        img = Image.open(img_buffer)
        # print("save visits img")
        # img.save("/workspace/repos_dev/VQVAE_RL/figures/minigrid_abstraction/grd_visits.png")
        wandb.log({f"Images/reward_distribution": wandb.Image(img)})
        img_buffer.close()
        plt.close(fig_grd_visits)

    def vis_shaping_distribution(self, norm_log: 0, suffix: str = None):
        width = self.env.width
        height = self.env.height
        clustersN = np.full(shape=(height, width), fill_value=np.nan)
        clustersE = np.full(shape=(height, width), fill_value=np.nan)
        clustersS = np.full(shape=(height, width), fill_value=np.nan)
        clustersW = np.full(shape=(height, width), fill_value=np.nan)

        shaping_distribution = copy.deepcopy(self.shaping_distribution)
        if norm_log:
            shaping_distribution = np.log(
                shaping_distribution - shaping_distribution.min() + 1
            ) / np.log(norm_log)

        for w in range(width - 2):
            w += 1
            for h in range(height - 2):
                h += 1
                for i in range(4):
                    if i == 0:
                        clustersE[h, w] = shaping_distribution[h, w, i]
                    if i == 1:
                        clustersS[h, w] = shaping_distribution[h, w, i]
                    if i == 2:
                        clustersW[h, w] = shaping_distribution[h, w, i]
                    if i == 3:
                        clustersN[h, w] = shaping_distribution[h, w, i]

        values = [clustersN, clustersE, clustersS, clustersW]

        triangulations = self.triangulation_for_triheatmap(width, height)

        # [Plotting]
        fig_grd_visits, ax_grd_visits = plt.subplots(figsize=(5, 5))
        vmin = shaping_distribution.min()
        vmax = shaping_distribution.max()
        my_cmap = copy.copy(plt.cm.get_cmap("hot"))
        my_cmap.set_bad(color="green")
        imgs = [
            ax_grd_visits.tripcolor(
                t,
                np.ravel(val),
                vmin=vmin,
                vmax=vmax,
                # norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=my_cmap,
                ec="black",
                lw=0.02,
            )
            for t, val in zip(triangulations, values)
        ]
        ax_grd_visits.invert_yaxis()
        # cax = fig_grd_visits.add_axes([0.9, 0.23, 0.03, 0.5])
        # fig_grd_visits.colorbar(ax_grd_visits, cax=cax)
        xlabel_vmin = round(self.shaping_distribution.min(), 3)
        xlabel_vmax = round(self.shaping_distribution.max(), 3)
        xlabel_sum = round(self.shaping_distribution.sum(), 3)
        xlabel_eps = round(self.exploration_rate, 3)
        xlabel = f"vmin:{xlabel_vmin}, vmax:{xlabel_vmax}, sum:{xlabel_sum}, eps:{xlabel_eps}"
        ax_grd_visits.set_xlabel(xlabel)
        fig_grd_visits.tight_layout()
        img_buffer = io.BytesIO()
        fig_grd_visits.savefig(
            img_buffer,
            dpi=100,
            # facecolor="w",
            # edgecolor="w",
            # orientation="portrait",
            # transparent=False,
            # bbox_inches=None,
            # pad_inches=0.1,
            format="png",
        )
        # wandb.define_metric("Images/time_steps_done")
        # wandb.define_metric("Images/abs_values", step_metric="Images/time_steps_done")
        img = Image.open(img_buffer)
        # print("save visits img")
        # img.save("/workspace/repos_dev/VQVAE_RL/figures/minigrid_abstraction/grd_visits.png")
        wandb.log({f"Images/shaping_distribution_log{norm_log}_{suffix}": wandb.Image(img)})
        img_buffer.close()
        plt.close(fig_grd_visits)

        # self.grd_visits = np.zeros((height, width, 4))

    def _create_optimizers(self, config):

        if isinstance(config.lr_ground_Q, str) and config.lr_ground_Q.startswith("lin"):
            self.lr_scheduler_ground_Q = linear_schedule(float(config.lr_ground_Q.split("_")[1]))
            self.lr_grd_Q = self.lr_scheduler_ground_Q(self._current_progress_remaining)
        else:
            self.lr_grd_Q = config.lr_ground_Q

        if isinstance(config.lr_abstract_V, str) and config.lr_abstract_V.startswith("lin"):
            self.lr_scheduler_abstract_V = linear_schedule(
                float(config.lr_abstract_V.split("_")[1])
            )
            self.lr_abs_V = self.lr_scheduler_abstract_V(self._current_progress_remaining)
        else:
            self.lr_abs_V = config.lr_abstract_V

        if isinstance(config.lr_vq, str) and config.lr_vq.startswith("lin"):
            self.lr_scheduler_vq = linear_schedule(float(config.lr_vq.split("_")[1]))
            self.lr_vq = self.lr_scheduler_vq(self._current_progress_remaining)
        else:
            self.lr_vq = config.lr_vq
        # self.ground_Q_optimizer = optim.RMSprop(
        #     self.ground_Q_net.parameters(), lr=lr_ground_Q, alpha=0.95, momentum=0, eps=0.01
        # )
        # self.abstract_V_optimizer = optim.RMSprop(
        #     self.ground_Q_net.parameters(), lr=lr_abstract_V, alpha=0.95, momentum=0.95, eps=0.01
        # )

        self.ground_Q_optimizer = optim.Adam(self.ground_Q.parameters(), lr=self.lr_grd_Q)
        self.abs_V_MLP_optimizer = optim.Adam(self.abs_V_MLP.parameters(), lr=self.lr_abs_V)
        self.vq_optimizer = optim.Adam(self.vq.parameters(), lr=self.lr_vq)

    def act(self, state):
        # warm up phase
        if self.timesteps_done < self.init_steps:
            # action = self.env.action_space.sample()
            action = random.randrange(self.n_actions)
            self.exploration_rate = 1.0
            return action

        self._update_current_progress_remaining(self.timesteps_done, self.total_timesteps)
        self.exploration_rate = self.exploration_scheduler(self._current_progress_remaining)
        with torch.no_grad():
            state = state[:]  # this operation required when using LazyFrames
            # state = state.transpose(2, 0, 1)[np.newaxis, ...]
            state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            # state = T.ToTensor()(state).float().unsqueeze(0).to(self.device)

            if random.random() > self.exploration_rate:
                # action = self.ground_Q(state.unsqueeze(0))[0].max(1)[1].item()
                action = self.ground_Q(state)[0].argmax(dim=1).item()
            else:
                action = random.randrange(self.n_actions)

        # [maintain eligibility]
        # found = False
        # for x in self.abstract_eligibllity_list:
        #     if x[0] == state:
        #         x[1] = 1
        #         found = True
        # if not found:
        #     self.abstract_eligibllity_list.append((state, 1))

        # self.timesteps_done += 1
        return action

    def update_grd_visits(self, info):
        agent_pos = info["agent_pos2"]
        agent_dir = info["agent_dir2"]
        self.grd_visits[agent_pos[1], agent_pos[0], agent_dir] += 1

    def update_grd_rewards(self, info):
        agent_pos = info["agent_pos2"]
        agent_dir = info["agent_dir2"]
        self.grd_reward_distribution[agent_pos[1], agent_pos[0], agent_dir] += info[
            "original_reward"
        ]

    def update_grdQ_absV(
        self,
        state,
        action,
        next_state,
        reward,
        terminated,
        info,
    ):
        if hasattr(self, "lr_scheduler_ground_Q"):
            self.lr_grd_Q = self.lr_scheduler_ground_Q(self._current_progress_remaining)
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_grd_Q,
            )
        if hasattr(self, "lr_scheduler_abstract_V"):
            self.lr_abs_V = self.lr_scheduler_abstract_V(self._current_progress_remaining)
            update_learning_rate(self.abs_V_MLP_optimizer, self.lr_abs_V)
        if self.clip_reward:
            reward.clamp_(-1, 1)

        # [data augmentation]
        if self.input_format == "full_img":
            state = self.aug(state)
            next_state = self.aug(next_state)

        # [Update ground Q network]
        grd_q, encoded = self.ground_Q(state)
        grd_q_reduction = torch.mean(grd_q, dim=1, keepdim=True)
        # grd_q_reduction = torch.amax(grd_q.detach(), dim=1, keepdim=True)
        # grd_q_mean_bytarg = torch.mean(self.ground_Q_target(state)[0].detach(), dim=1, keepdim=True)
        grd_q = grd_q.gather(1, action)
        if self.encoded_detach4abs:
            encoded = encoded.detach()
        quantized, vq_loss, entrophy_vq, _ = self.vq(encoded)
        abs_v = self.abs_V_MLP(quantized)

        with torch.no_grad():

            # Vanilla DQN
            grd_q_next, encoded_next = self.ground_Q_target(next_state)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)
            quantized_next, _, _, _ = self.vq(encoded_next.detach())

            # Double DQN
            # action_argmax_target = self.ground_target_Q_net(next_state_batch).argmax(
            #     dim=1, keepdim=True
            # )
            # ground_next_max_Q = self.ground_Q_net(next_state_batch).gather(1, action_argmax_target)

            grd_q_target = (
                reward
                # + self.omega * shaping * (1 - terminated.float())
                + self.gamma * grd_q_next_max * (1 - terminated.float())
            )
            # mask = (quantized == quantized_next) & (reward == 0)
            # mask = (quantized != quantized_next) | (reward != 0)
            mask = (quantized != quantized_next).any(dim=1).unsqueeze(1)
            mask = mask | (reward != 0)
            mask = mask.float()
            abs_v_next = self.abs_V_MLP_target(quantized_next)
            abs_v_target = reward + self.abs_gamma * abs_v_next * (1 - terminated.float())
            abs_v_target = abs_v_target * mask + abs_v_next * (1 - mask)

        # criterion = nn.SmoothL1Loss()
        criterion = F.mse_loss
        ground_td_error = criterion(grd_q, grd_q_target)
        abs_td_error = criterion(abs_v, abs_v_target)
        # abs_grd_diff = criterion(abs_v, grd_q)
        # commit_error_grd2abs = criterion(abs_v.detach(), grd_q_reduction)
        # commit_error_abs2grd = criterion(abs_v, grd_q_reduction.detach())
        commit_error_abs2grd = criterion(abs_v, grd_q_target)

        # commit_error_grd2abs = torch.linalg.norm(F.relu(abs_v.detach() - grd_q_reduction))
        # commit_error_abs2grd = torch.linalg.norm(F.relu(grd_q_target - abs_v))

        # commit_error_grd2abs = F.relu(abs_v.detach() - grd_q_reduction).mean()
        # commit_error_abs2grd = F.relu(grd_q_target - abs_v).mean()

        diff_l2_abs_grd = F.mse_loss(abs_v, grd_q_reduction)
        diff_l1_abs_grd = F.l1_loss(abs_v, grd_q_reduction)

        # [Compute total loss]
        total_loss = ground_td_error + vq_loss - entrophy_vq + commit_error_abs2grd

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        self.abs_V_MLP_optimizer.zero_grad(set_to_none=True)
        self.vq_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                if param.grad is not None:  # make sure grad is not None
                    param.grad.data.clamp_(-1, 1)
            for param in self.abs_V_MLP.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            for param in self.vq.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()
        self.abs_V_MLP_optimizer.step()
        self.vq_optimizer.step()

        # for i, info_i in enumerate(info):
        #     self.shaping_distribution[
        #         info_i["agent_pos2"][1], info_i["agent_pos2"][0], info_i["agent_dir2"]
        #     ] += shaping[i]

        self.training_info["ground_Q_error"].append(ground_td_error.item())
        self.training_info["abstract_V_error"].append(abs_td_error.item())
        self.training_info["entrophy_vq"].append(entrophy_vq.item())
        self.training_info["vq_loss"].append(vq_loss.item())
        self.training_info["absV_grdQ(d)_diff"].append(diff_l1_abs_grd.item())
        self.training_info["absV_grdQ(d)_mse"].append(diff_l2_abs_grd.item())
        self.training_info["total_loss"].append(total_loss.item())
        # self.training_info["avg_shaping"].append(torch.mean(shaping).item())

    def update_absV(
        self,
        state,
        action,
        next_state,
        reward,
        terminated,
    ):
        if hasattr(self, "lr_scheduler_abstract_V"):
            self.lr_abs_V = self.lr_scheduler_abstract_V(self._current_progress_remaining)
            update_learning_rate(self.abs_V_MLP_optimizer, self.lr_abs_V)

        # [data augmentation]
        # state = self.aug(state)
        # next_state = self.aug(next_state)

        # [Update ground Q network]
        grd_q, encoded = self.ground_Q(state)
        grd_q_reduction = torch.mean(grd_q, dim=1, keepdim=True)
        # grd_q_reduction = torch.amax(grd_q.detach(), dim=1, keepdim=True)
        # grd_q_mean_bytarg = torch.mean(self.ground_Q_target(state)[0].detach(), dim=1, keepdim=True)
        # grd_q = grd_q.gather(1, action)
        quantized, vq_loss, entrophy_vq, _ = self.vq(encoded.detach())
        abs_v = self.abs_V_MLP(quantized)

        # nn.SmoothL1Loss()
        abs2grd_commit_mse = F.mse_loss(abs_v, grd_q_reduction.detach())
        abs2grd_commit_l1 = F.l1_loss(abs_v, grd_q_reduction.detach())

        # [Compute total loss]
        total_loss = vq_loss - entrophy_vq + abs2grd_commit_mse

        self.abs_V_MLP_optimizer.zero_grad(set_to_none=True)
        self.vq_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.abs_V_MLP.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            for param in self.vq.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)

        self.abs_V_MLP_optimizer.step()
        self.vq_optimizer.step()

        self.training_info["entrophy_vq"].append(entrophy_vq.item())
        self.training_info["vq_loss"].append(vq_loss.item())
        self.training_info["absV_grdQ(d)_diff"].append(abs2grd_commit_l1.item())
        self.training_info["absV_grdQ(d)_mse"].append(abs2grd_commit_mse.item())

    def update_grdQ_pure(self, state, action, next_state, reward, terminated):
        if hasattr(self, "lr_scheduler_ground_Q"):
            self.lr_grd_Q = self.lr_scheduler_ground_Q(self._current_progress_remaining)
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_grd_Q,
            )

        if self.clip_reward:
            reward.clamp_(-1, 1)

        # [data augmentation]
        if self.input_format == "full_img":
            state = self.aug(state)
            next_state = self.aug(next_state)

        # [Update ground Q network]
        grd_q, encoded = self.ground_Q(state)
        grd_q = grd_q.gather(1, action)

        with torch.no_grad():

            # Vanilla DQN
            grd_q_next, encoded_next = self.ground_Q_target(next_state)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)

            # Double DQN
            # action_argmax_target = self.ground_target_Q_net(next_state_batch).argmax(
            #     dim=1, keepdim=True
            # )
            # ground_next_max_Q = self.ground_Q_net(next_state_batch).gather(1, action_argmax_target)

            # Compute ground target Q value
            grd_q_target = reward + self.gamma * grd_q_next_max * (1 - terminated.float())

        # criterion = nn.SmoothL1Loss()
        criterion = F.mse_loss
        ground_td_error = criterion(grd_q, grd_q_target)

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        ground_td_error.backward()
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                if param.grad is not None:  # make sure grad is not None
                    param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()

        self.training_info["ground_Q_error"].append(ground_td_error.item())

    def init_vq_codebook(
        self,
    ):
        encoded_l = []
        for _ in range(10):
            state, action, next_state, reward, terminated, info = self.memory.sample(
                self.batch_size, mode=None
            )
            encoded = self.ground_Q.forward_conv(state).detach().cpu().numpy()
            encoded_l.append(np.unique(encoded, axis=0))

        self.kmeans = KMeans(n_clusters=self.num_vq_embeddings, n_init=20, random_state=0).fit(
            np.concatenate(encoded_l, axis=0)
        )
        # self.vq_codebook = self.kmeans.cluster_centers_
        self.vq.embedding.weight.data = torch.as_tensor(self.kmeans.cluster_centers_).to(
            self.device
        )

    def update(self, use_shaping: bool):
        """
        update with adaptive abstraction
        """
        if self.timesteps_done == self.init_steps:
            # self.init_vq_codebook()
            # self.vis_abstraction()
            print("Warm up done")
            pass
        if self.timesteps_done == self.init_steps + 1:
            for _ in range(3):
                # self.cache_goal_transition()
                pass
        steps = self.timesteps_done - self.init_steps
        if use_shaping:
            if steps % self.ground_learn_every == 0:
                state, action, next_state, reward, terminated, info = self.memory.sample(
                    self.batch_size, mode=None
                )

                # [update ground_Q with reward shaping]
                self.update_grdQ_absV(state, action, next_state, reward, terminated, info)
        else:
            # [purely update ground Q]
            if steps % self.ground_learn_every == 0:
                state, action, next_state, reward, terminated, info = self.memory.sample(
                    self.batch_size, mode=None
                )
                self.update_grdQ_pure(state, action, next_state, reward, terminated)

        if steps % self.ground_sync_every == 0:
            # soft_sync_params(
            #     self.ground_Q.parameters(),
            #     self.ground_Q_target.parameters(),
            #     self.ground_tau,
            # )
            soft_sync_params(
                self.ground_Q.encoder.parameters(),
                self.ground_Q_target.encoder.parameters(),
                self.ground_Q_encoder_tau,
            )

            soft_sync_params(
                self.ground_Q.critic.parameters(),
                self.ground_Q_target.critic.parameters(),
                self.ground_Q_critic_tau,
            )

        if steps % self.abstract_sync_every == 0:
            soft_sync_params(
                self.abs_V_MLP.parameters(),
                self.abs_V_MLP_target.parameters(),
                self.abstract_tau,
            )

        if steps % self.reset_training_info_every == 0:
            self.log_training_info(wandb_log=True)
            self.reset_training_info()
