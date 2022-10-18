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


class HDQN_AdaptiveAbs(HDQN):
    def __init__(self, config, env, use_table4grd=False):
        super().__init__(config, env)
        # self.set_hparams(config)
        self.n_actions = env.action_space.n
        self.use_table4grd = use_table4grd

        # self.abs_ticks = np.array([5, 10, 14])
        # self.abs_ticks = np.array(config.abs_ticks)
        # self.abs_txt_ticks = ((np.insert(self.abs_ticks, 0, 0) + np.append(self.abs_ticks, 0)) / 2)[
        #     :-1
        # ]

        if use_table4grd:
            # self.grd_Q_table = np.random.rand(env.width, env.height, 4, env.action_space.n) / 1e15
            self.grd_Q_table = np.zeros((env.width, env.height, 4, env.action_space.n))
            # self.lr_grd_Q = config.lr_ground_Q
        else:
            self.ground_Q = DQN(
                observation_space=env.observation_space,
                action_space=env.action_space,
                encoder_maker=EncoderMaker(input_format=config.input_format, agent=self),
            ).to(self.device)

            self.ground_Q_target = DQN(
                observation_space=env.observation_space,
                action_space=env.action_space,
                encoder_maker=EncoderMaker(input_format=config.input_format, agent=self),
            ).to(self.device)

            self.ground_Q_target.load_state_dict(self.ground_Q.state_dict())
            self.ground_Q_target.train()

        self.grd_visits = np.zeros((env.height, env.width, 4))
        self.shaping_distribution = np.zeros((env.height, env.width, 4))

        self.abstract_V_array = np.zeros((config.n_clusters))
        # self.lr_abs_V = config.lr_abstract_V
        self.abstract_eligibllity_list = []

        # self.random_encoder = RandomEncoderMiniGrid(
        #     observation_space=env.observation_space,
        #     linear_out_dim=config.cluster_embedding_dim,
        #     hidden_dims=config.hidden_dims,
        # ).to(self.device)
        self.random_encoder = Encoder_MiniGrid_PartialObs(
            in_channels=3,
            observation_space=env.observation_space,
            linear_out_dim=None,
            hidden_dims=config.hidden_dims,
        ).to(self.device)

        # self.random_encoder = self.ground_Q_target.encoder

        self.aug = RandomShiftsAug(pad=4)
        self.memory = ReplayMemoryWithCluster(self.size_replay_memory, self.device)
        self.buffer_before_kmeans = set()
        self.abs_centroids = np.zeros((config.n_clusters, config.cluster_embedding_dim))

        self.list_recent_state_encoded = []
        self._ema_cluster_size = np.zeros((config.n_clusters))
        # self._ema_w = np.random.randn(config.n_clusters, 2)
        self._ema_w = np.zeros((config.n_clusters, 2))
        self.ema_window = 100
        self._decay = 0.99
        self._epsilon = 1e-5

        # self.timesteps_done = 0
        # self.episodes_done = 0
        # self._current_progress_remaining = 1.0
        # self.to_buffer_recent_states = False  # for func maybe_buffer_recent_states
        self.count_vis = 0
        self.goal_found = False
        self.n_abs_updates = 60

        if not use_table4grd:
            self._create_optimizers(config)
            self.train()
        self.reset_training_info()

    def train(self, training=True):
        self.training = training
        self.ground_Q.train(training)

    def reset_training_info(self):
        self.training_info = {
            "ground_Q_error": [],
            "abstract_V_error": [],
            "avg_shaping": [],
            "n_cross_upper_bound": [],
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
        self.ground_tau = config.ground_tau
        # self.encoder_tau = config.encoder_tau
        # self.abstract_tau = config.abstract_tau
        self.grd_hidden_dims = config.grd_hidden_dims
        self.grd_embedding_dim = config.grd_embedding_dim
        self.cluster_embedding_dim = config.cluster_embedding_dim

        self.ground_learn_every = config.ground_learn_every
        self.ground_sync_every = config.ground_sync_every
        self.ground_gradient_steps = config.ground_gradient_steps
        self.abstract_learn_every = config.abstract_learn_every
        self.abstract_sync_every = config.abstract_sync_every
        self.abstract_gradient_steps = config.abstract_gradient_steps

        # self.validate_every = config.validate_every
        self.save_model_every = config.save_model_every
        self.reset_training_info_every = config.reset_training_info_every
        # self.save_recon_every = config.save_recon_every
        # self.buffer_recent_states_every = config.buffer_recent_states_every

        self.clip_grad = config.clip_grad
        self.input_format = config.input_format
        self.n_clusters = config.n_clusters

        if isinstance(config.lr_ground_Q, str) and config.lr_ground_Q.startswith("lin"):
            self.lr_scheduler_ground_Q = linear_schedule(float(config.lr_ground_Q.split("_")[1]))
            self.lr_grd_Q = self.lr_scheduler_ground_Q(1.0)
        elif isinstance(config.lr_ground_Q, float):
            self.lr_grd_Q = config.lr_ground_Q

        if isinstance(config.lr_abstract_V, str) and config.lr_abstract_V.startswith("lin"):
            self.lr_scheduler_abstract_V = linear_schedule(
                float(config.lr_abstract_V.split("_")[1])
            )
            self.lr_abs_V = self.lr_scheduler_abstract_V(1.0)
        elif isinstance(config.lr_abstract_V, float):
            self.lr_abs_V = config.lr_abstract_V

    def log_training_info(self, wandb_log=True):
        if wandb_log:
            metrics = {
                "Info/ground_Q_error": mean(self.training_info["ground_Q_error"]),
                "Info/abstract_V_error": mean(self.training_info["abstract_V_error"])
                if len(self.training_info["abstract_V_error"])
                else 0,
                # "training_info/n_cross_upper_bound": mean(
                #     self.training_info.get("n_cross_upper_bound", [0])
                # ),
                "Info/avg_shaping": mean(self.training_info["avg_shaping"])
                if len(self.training_info["avg_shaping"])
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
        assert isinstance(state, np.ndarray)
        # state_ = state.transpose(2, 0, 1)[np.newaxis, ...]
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        state_encoded = self.random_encoder(state).squeeze().cpu()
        return tuple(state_encoded.tolist())

    # def assign_abs_state(self, state_encoded):
    #     dist = []
    #     for i in range(self.n_clusters):
    #         dist.append(np.sum(np.absolute(state_encoded - self.abs_centroids[i])))
    #     return np.argmin(dist)

    # def assign_abs_state(self, state_encoded):
    #     diff = self.abs_centroids - state_encoded
    #     dist = np.sum(np.absolute(diff), axis=-1)
    #     return np.argmin(dist)

    def assign_abs_state(self, state_encoded):
        if len(state_encoded) > 2:
            diff = self.abs_centroids - state_encoded
            dist = np.linalg.norm(np.absolute(diff), axis=-1)
            return np.argmin(dist)
        elif len(state_encoded) == 2:
            # use Hanming distance
            diff = self.abs_centroids - state_encoded
            dist = np.sum(np.absolute(diff), axis=-1)
            return np.argmin(dist)

    def assign_abs_state_matrixwise(self, X: np.ndarray):
        distances = X[:, np.newaxis, :] - self.abs_centroids
        distances = np.sum(np.absolute(distances, out=distances), axis=-1)
        return np.argmin(distances, axis=-1)

    def replace_abstraction_in_memory(self):
        for _ in range(len(self.memory)):
            trs = self.memory.pop()
            if self.cluster_embedding_dim > 2:
                state_encoded = self.encode_state(trs.state)
                next_state_encoded = self.encode_state(trs.next_state)
            else:
                state_encoded = trs.info["agent_pos1"]
                next_state_encoded = trs.info["agent_pos2"]

            abs_state = self.assign_abs_state(state_encoded)
            next_abs_state = self.assign_abs_state(next_state_encoded)
            self.memory.push(
                trs.state,
                abs_state,
                trs.action,
                trs.next_state,
                next_abs_state,
                trs.reward,
                trs.terminated,
                trs.info,
            )

    def cache(self, state, action, next_state, reward, terminated, info):
        """Add the experience to memory"""
        if self.cluster_embedding_dim > 2:
            state_encoded = self.encode_state(state)
            next_state_encoded = self.encode_state(next_state)
        else:
            state_encoded = info["agent_pos1"]
            next_state_encoded = info["agent_pos2"]

        abs_state = self.assign_abs_state(state_encoded)
        next_abs_state = self.assign_abs_state(next_state_encoded)

        if self.timesteps_done < self.init_steps:
            self.buffer_before_kmeans.add(state_encoded)
        elif self.timesteps_done == self.init_steps:
            print(f"Init Kmeans clustering...buffered {len(self.buffer_before_kmeans)} states...")
            self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=0).fit(
                np.array(list(self.buffer_before_kmeans))
            )
            abs_state_counter_dict: Dict = Counter(self.kmeans.labels_)
            self.abs_state_count = np.array(
                [abs_state_counter_dict[i] for i in range(self.n_clusters)]
            )
            self.abs_centroids = self.kmeans.cluster_centers_
            self.replace_abstraction_in_memory()

            self.vis_grd_visits(norm_log=50)
            self.vis_grd_visits(norm_log=0)
            self.vis_abstraction()
            self.vis_abstract_values()
        else:
            # abs_label = self.kmeans.predict([state_encoded])[0]
            self.abs_state_count[abs_state] += 1
            self.abs_centroids[abs_state, :] += (
                1
                / self.abs_state_count[abs_state]
                * (state_encoded - self.abs_centroids[abs_state, :])
            )

        self.memory.push(
            state, abs_state, action, next_state, next_abs_state, reward, terminated, info
        )

        if abs_state != next_abs_state:
            abs_value = self.abstract_V_array[abs_state]
            abs_value_next = self.abstract_V_array[next_abs_state]
            shaping = abs_value_next - abs_value
        else:
            shaping = 0

        return shaping

    def cache_ema(self, state, action, next_state, reward, terminated, info):
        """Add the experience to memory"""
        state_encoded = info["agent_pos1"]
        next_state_encoded = info["agent_pos2"]
        abs_state = self.assign_abs_state(state_encoded)
        next_abs_state = self.assign_abs_state(next_state_encoded)
        if abs_state != next_abs_state:
            abs_value = self.abstract_V_array[abs_state]
            abs_value_next = self.abstract_V_array[next_abs_state]
            shaping = abs_value_next - abs_value
            episodic_shaped_reward += shaping

        if self.timesteps_done < self.init_steps:
            self.buffer_before_kmeans.append(state_encoded)
        elif self.timesteps_done == self.init_steps:
            self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=0).fit(
                np.array(self.buffer_before_kmeans)
            )
            abs_states_onehot = np.eye(self.n_clusters)[self.kmeans.labels_]
            self._ema_cluster_size = np.sum(abs_states_onehot, axis=0)
            self._ema_w = np.matmul(abs_states_onehot.T, np.array(self.buffer_before_kmeans))
            # abs_state_counter_dict: Dict = Counter(self.kmeans.labels_)
            # self.abs_state_count = np.array(
            #     [abs_state_counter_dict[i] for i in range(self.n_clusters)]
            # )
            self.abs_centroids = self.kmeans.cluster_centers_
            # reassign abs_state for states in memory
            for _ in range(len(self.memory)):
                trs = self.memory.pop()
                abs_state = self.assign_abs_state(trs.info["agent_pos1"])
                next_abs_state = self.assign_abs_state(trs.info["agent_pos2"])
                self.memory.push(
                    trs.state,
                    abs_state,
                    trs.action,
                    trs.next_state,
                    next_abs_state,
                    trs.reward,
                    trs.terminated,
                    trs.info,
                )
            self.vis_abstraction()
        else:
            self.list_recent_state_encoded.append(state_encoded)
            if len(self.list_recent_state_encoded) > self.ema_window:
                X = np.array(self.list_recent_state_encoded)
                abs_states = self.assign_abs_state_matrixwise(X)
                abs_states_onehot = np.eye(self.n_clusters)[abs_states]
                self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                    1 - self._decay
                ) * np.sum(abs_states_onehot, axis=0)

                # Laplace smoothing of the cluster size
                n = np.sum(self._ema_cluster_size)
                self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self.n_clusters * self._epsilon)
                    * n
                )

                dw = np.matmul(abs_states_onehot.T, X)
                self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw
                self.abs_centroids = self._ema_w / self._ema_cluster_size[:, np.newaxis]

                self.list_recent_state_encoded = []

        self.memory.push(
            state, abs_state, action, next_state, next_abs_state, reward, terminated, info
        )

    def cache_goal_transition(self):
        print("cache goal transitions")
        temp = np.zeros((self.env.width, self.env.height, 3))
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
        self.cache(temp, 2, temp, reward1, True, info1)
        self.cache(temp, 2, temp, reward2, True, info2)

    def vis_abstraction_and_values(self, prefix: str):
        width = self.env.width
        height = self.env.height
        # num_cluster = self.kmeans.n_clusters
        # clustersN = np.empty(shape=(height, width))
        # clustersS = np.empty(shape=(height, width))
        # clustersW = np.empty(shape=(height, width))
        # clustersE = np.empty(shape=(height, width))
        clustersN = np.full(shape=(height, width), fill_value=-1)
        clustersS = np.full(shape=(height, width), fill_value=-1)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=-1)
        clustersE = np.full(shape=(height, width), fill_value=-1)

        clustersN2 = np.full(shape=(height, width), fill_value=-1.0, dtype=np.float32)
        clustersS2 = np.full(shape=(height, width), fill_value=-1.0, dtype=np.float32)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW2 = np.full(shape=(height, width), fill_value=-1.0, dtype=np.float32)
        clustersE2 = np.full(shape=(height, width), fill_value=-1.0, dtype=np.float32)

        for w in range(width - 2):
            w += 1
            for h in range(height - 2):
                h += 1
                abstract_state_idx, abstract_value = self.get_abstract_value((w, h))
                clustersN[h, w] = abstract_state_idx
                clustersE[h, w] = abstract_state_idx
                clustersS[h, w] = abstract_state_idx
                clustersW[h, w] = abstract_state_idx

                clustersN2[h, w] = abstract_value
                clustersE2[h, w] = abstract_value
                clustersS2[h, w] = abstract_value
                clustersW2[h, w] = abstract_value

        values = [clustersN, clustersE, clustersS, clustersW]
        values2 = [clustersN2, clustersE2, clustersS2, clustersW2]

        triangulations = self.triangulation_for_triheatmap(width, height)

        xx, yy = np.meshgrid(self.abs_txt_ticks, self.abs_txt_ticks)
        xx = xx.flatten()
        yy = yy.flatten()

        # [Plot Abstraction]
        fig_abs, ax_abs = plt.subplots(figsize=(5, 5))
        vmax = self.n_clusters
        vmin = 0
        my_cmap = copy.copy(plt.cm.get_cmap("gist_ncar"))
        my_cmap.set_under(color="dimgray")
        imgs = [
            ax_abs.tripcolor(t, np.ravel(val), vmin=vmin, vmax=vmax, cmap=my_cmap, ec="black")
            for t, val in zip(triangulations, values)
        ]
        for i, (x, y) in enumerate(zip(xx, yy)):
            ax_abs.text(
                x,
                y,
                str(i),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=13,
                color="black",
                fontweight="semibold",
            )
        ax_abs.invert_yaxis()
        fig_abs.tight_layout()
        plt.close(fig_abs)

        # [Plot Abstract Values]
        vmin = self.abstract_V_array.min()
        vmax = self.abstract_V_array.max() + 0.00001
        # vmin = vmin - 0.07 * (vmax - vmin)
        my_cmap = copy.copy(plt.cm.get_cmap("hot"))
        my_cmap.set_under(color="green")
        fig_abs_v, ax_abs_v = plt.subplots(figsize=(5, 5))
        imgs2 = [
            ax_abs_v.tripcolor(
                t,
                np.ravel(val),
                vmin=vmin,
                vmax=vmax,
                # norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=my_cmap,
                ec="black",
            )
            for t, val in zip(triangulations, values2)
        ]

        abstract_V_array = self.abstract_V_array.round(3)
        for i, (x, y) in enumerate(zip(xx, yy)):
            ax_abs_v.text(
                x,
                y,
                abstract_V_array[i],
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=13,
                color="green",
                fontweight="semibold",
            )
        ax_abs_v.invert_yaxis()
        # ax_abs_v.set_xlabel(f"Abs_V:{abstract_V_array}")
        fig_abs_v.tight_layout()
        plt.close(fig_abs_v)
        # self.count_vis += 1
        # f_name = f"/workspace/repos_dev/VQVAE_RL/figures/minigrid_abstraction/{prefix}_{self.count_vis}.png"
        # fig.savefig(
        #     f_name,
        #     dpi=200,
        #     facecolor="w",
        #     edgecolor="w",
        #     orientation="portrait",
        #     format=None,
        #     transparent=False,
        #     bbox_inches=None,
        #     pad_inches=0.1,
        # )
        # print(f"saving fig: {f_name}")

        # pil_object = PIL.Image.frombytes(
        #     "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        # )

        # [alternative plot of abstract values]
        # test = self.abstract_V_array.reshape(3, 3)
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.imshow(
        #     test,
        #     # norm=colors.LogNorm(),
        #     cmap="hot",
        # )
        # fig.savefig(
        #     f"/workspace/repos_dev/VQVAE_RL/figures/minigrid_abstraction/{prefix}_{self.count_vis}_test.png",
        #     dpi=200,
        #     facecolor="w",
        #     edgecolor="w",
        #     orientation="portrait",
        #     format=None,
        #     transparent=False,
        #     bbox_inches=None,
        #     pad_inches=0.1,
        # )
        # plt.close(fig)

    def vis_abstraction(self, prefix: str = None):
        width = self.env.width
        height = self.env.height

        clustersN = np.full(shape=(height, width), fill_value=-1)
        clustersS = np.full(shape=(height, width), fill_value=-1)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=-1)
        clustersE = np.full(shape=(height, width), fill_value=-1)

        if self.cluster_embedding_dim > 2:
            center_coords = np.zeros((self.n_clusters, 2))

        for w in range(width):
            # w += 1
            for h in range(height):
                # h += 1
                if not isinstance(self.env.grid.get(w, h), Wall):
                    if self.cluster_embedding_dim > 2:

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
                            encoded = self.encode_state(state)
                            abstract_state_idx = self.assign_abs_state(encoded)
                            if dir == 3:
                                clustersN[h, w] = abstract_state_idx
                            if dir == 0:
                                clustersE[h, w] = abstract_state_idx
                            if dir == 1:
                                clustersS[h, w] = abstract_state_idx
                            if dir == 2:
                                clustersW[h, w] = abstract_state_idx
                    elif self.cluster_embedding_dim == 2:
                        abstract_state_idx = self.assign_abs_state((w, h))
                        clustersN[h, w] = abstract_state_idx
                        clustersE[h, w] = abstract_state_idx
                        clustersS[h, w] = abstract_state_idx
                        clustersW[h, w] = abstract_state_idx

        values = [clustersN, clustersE, clustersS, clustersW]
        triangulations = self.triangulation_for_triheatmap(width, height)

        # [Plot Abstraction]
        fig_abs, ax_abs = plt.subplots(figsize=(5, 5))
        vmax = self.n_clusters
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

        # xx, yy = np.meshgrid(self.abs_txt_ticks, self.abs_txt_ticks)
        # xx = xx.flatten()
        # yy = yy.flatten()
        if self.cluster_embedding_dim > 2:
            pass
        elif self.cluster_embedding_dim == 2:
            for i, (x, y) in enumerate(self.abs_centroids):
                ax_abs.text(
                    x,
                    y,
                    str(i),
                    # horizontalalignment="center",
                    # verticalalignment="center",
                    fontsize=9,
                    color="k",
                    fontweight="semibold",
                    # fontweight="normal",
                    bbox=dict(
                        boxstyle="round,pad=0.08, rounding_size=0.2",
                        fc=(1.0, 0.8, 0.8),
                        ec="k",
                        lw=1.5,
                    ),
                )

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

    def vis_abstract_values(self, prefix: str = None):
        width = self.env.width
        height = self.env.height
        clustersN = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)
        clustersE = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)
        clustersS = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)
        abstract_V_array_to_vis = copy.deepcopy(self.abstract_V_array)
        # abstract_V_array_to_vis -= np.min(abstract_V_array_to_vis)
        # if len(np.argwhere(abstract_V_array_to_vis)) > 0:
        #     print("abstract_V_array_to_vis:")
        #     pprint(abstract_V_array_to_vis)
        #     print(f"np.argwhere(abstract_V_array_to_vis): {np.argwhere(abstract_V_array_to_vis)}")
        #     goal_pos = (self.env.width - 2, self.env.height - 2)
        #     print(f"abs_state where goal located: {self.assign_abs_state(goal_pos)}")

        for w in range(width):
            # w += 1
            for h in range(height):
                # h += 1
                # abstract_state_idx, abstract_value = self.get_abstract_value((w, h))
                if not isinstance(self.env.grid.get(w, h), Wall):
                    if self.cluster_embedding_dim > 2:
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
                            encoded = self.encode_state(state)
                            abstract_state_idx = self.assign_abs_state(encoded)
                            abstract_value = abstract_V_array_to_vis[abstract_state_idx]
                            if dir == 3:
                                clustersN[h, w] = abstract_value
                            if dir == 0:
                                clustersE[h, w] = abstract_value
                            if dir == 1:
                                clustersS[h, w] = abstract_value
                            if dir == 2:
                                clustersW[h, w] = abstract_value

                    elif self.cluster_embedding_dim == 2:
                        abstract_state_idx = self.assign_abs_state((w, h))
                        abstract_value = abstract_V_array_to_vis[abstract_state_idx]
                        clustersN[h, w] = abstract_value
                        clustersE[h, w] = abstract_value
                        clustersS[h, w] = abstract_value
                        clustersW[h, w] = abstract_value

        values = [clustersN, clustersE, clustersS, clustersW]

        triangulations = self.triangulation_for_triheatmap(width, height)

        # [Plot Abstract Values]
        fig_abs_v, ax_abs_v = plt.subplots(figsize=(5, 5))
        vmin = abstract_V_array_to_vis.min()
        vmax = abstract_V_array_to_vis.max() + 1e-10
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
        abstract_V_array = self.abstract_V_array.round(3)
        # xx, yy = np.meshgrid(self.abs_txt_ticks, self.abs_txt_ticks)
        # xx = xx.flatten()
        # yy = yy.flatten()
        if self.cluster_embedding_dim > 2:
            pass
        elif self.cluster_embedding_dim == 2:
            for i, (x, y) in enumerate(self.abs_centroids):
                ax_abs_v.text(
                    x,
                    y,
                    abstract_V_array[i],
                    # horizontalalignment="center",
                    # verticalalignment="center",
                    fontsize=11,
                    color="k",
                    fontweight="semibold",
                    bbox=dict(
                        boxstyle="round,pad=0.08, rounding_size=0.2",
                        fc=(1.0, 0.8, 0.8),
                        ec="k",
                        lw=1.5,
                    ),
                )

        ax_abs_v.invert_yaxis()
        # ax_abs_v.set_xlabel(f"Abs_V:{abstract_V_array}")
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
        wandb.log({"Images/abs_values": wandb.Image(img)})
        img_buffer.close()
        plt.close(fig_abs_v)

    def vis_grd_q_values(self, norm_log: 0, suffix: str = None):
        width = self.env.width
        height = self.env.height
        clustersN = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)
        clustersE = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)
        clustersS = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)
        clustersW = np.full(shape=(height, width), fill_value=np.nan, dtype=np.float32)

        grd_Q_table_to_vis = copy.deepcopy(self.grd_Q_table)
        if norm_log:
            grd_Q_table_to_vis = np.log(grd_Q_table_to_vis - grd_Q_table_to_vis.min() + 1) / np.log(
                norm_log
            )

        for w in range(width - 2):
            w += 1
            for h in range(height - 2):
                h += 1
                for i in range(4):
                    if i == 0:
                        clustersE[h, w] = grd_Q_table_to_vis[h, w, i, :].max()
                    if i == 1:
                        clustersS[h, w] = grd_Q_table_to_vis[h, w, i, :].max()
                    if i == 2:
                        clustersW[h, w] = grd_Q_table_to_vis[h, w, i, :].max()
                    if i == 3:
                        clustersN[h, w] = grd_Q_table_to_vis[h, w, i, :].max()

        values = [clustersN, clustersE, clustersS, clustersW]

        triangulations = self.triangulation_for_triheatmap(width, height)

        # [Plotting]
        fig_grd_q, ax_grd_q = plt.subplots(figsize=(5, 5))

        vmin = grd_Q_table_to_vis.min()
        vmax = grd_Q_table_to_vis.max()

        my_cmap = copy.copy(plt.cm.get_cmap("hot"))
        my_cmap.set_bad(color="green")

        imgs = [
            ax_grd_q.tripcolor(
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
        ax_grd_q.invert_yaxis()
        # cax = fig_grd_visits.add_axes([0.9, 0.23, 0.03, 0.5])
        # fig_grd_visits.colorbar(ax_grd_visits, cax=cax)
        xlabel_median = round(np.median(self.grd_Q_table[np.nonzero(self.grd_Q_table)]), 3)
        xlabel_vmin = round(self.grd_Q_table.min(), 3)
        xlabel_vmax = round(self.grd_Q_table.max(), 3)
        xlabel_sum = round(np.sum(self.grd_Q_table), 3)

        ax_grd_q.set_xlabel(
            f"vmin:{xlabel_vmin}, vmax:{xlabel_vmax}, median:{xlabel_median}, sum:{xlabel_sum}"
        )
        fig_grd_q.tight_layout()
        img_buffer = io.BytesIO()
        fig_grd_q.savefig(
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
        wandb.log({f"Images/grd_q_max_log{norm_log}": wandb.Image(img)})
        img_buffer.close()
        plt.close(fig_grd_q)

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
            lr_ground_Q = self.lr_scheduler_ground_Q(self._current_progress_remaining)

        elif isinstance(config.lr_ground_Q, float):
            lr_ground_Q = config.lr_ground_Q

        # self.ground_Q_optimizer = optim.RMSprop(
        #     self.ground_Q_net.parameters(), lr=lr_ground_Q, alpha=0.95, momentum=0, eps=0.01
        # )
        # self.abstract_V_optimizer = optim.RMSprop(
        #     self.ground_Q_net.parameters(), lr=lr_abstract_V, alpha=0.95, momentum=0.95, eps=0.01
        # )

        if hasattr(self, "ground_Q"):
            self.ground_Q_optimizer = optim.Adam(self.ground_Q.parameters(), lr=lr_ground_Q)

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

    def act_table(self, info, exploit_only=False):
        self._update_current_progress_remaining(self.timesteps_done, self.total_timesteps)
        self.exploration_rate = self.exploration_scheduler(self._current_progress_remaining)
        agent_pos = info["agent_pos2"]
        agent_dir = info["agent_dir2"]

        if exploit_only:
            rand = 1.0
        else:
            rand = random.random()

        if rand > self.exploration_rate:

            q_values = np.array(
                [
                    self.grd_Q_table[agent_pos[1], agent_pos[0], agent_dir, a]
                    for a in range(self.n_actions)
                ]
            )
            action = np.random.choice(np.flatnonzero(q_values == q_values.max()))

            # naive greedy
            # action = np.argmax(self.grd_Q_table[agent_pos[1], agent_pos[0], agent_dir, :])

            assert action in range(self.n_actions)
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
        return action

    def update_grd_visits(self, info):
        agent_pos = info["agent_pos2"]
        agent_dir = info["agent_dir2"]
        self.grd_visits[agent_pos[1], agent_pos[0], agent_dir] += 1

    def get_abstract_state_idx(self, agent_pos):
        """
        currently onlu support Minigrid Empty-style of envs
        """
        # H, W, C = self.env.observation_space.shape
        # real_H = H - 2
        # real_W = W - 2
        # size = real_H

        # bin = math.sqrt(self.n_cluster)
        # # check if bin is an interger
        # assert bin == int(bin)
        # ceil = math.ceil(size / self.n_clusters)
        # floor = math.floor(size / self.n_clusters)
        # a = Symbol("a")
        # b = Symbol("b")
        # result = solve((a * floor + b * ceil - 13, a + b - 3), a, b)
        # abstract_ticks = [1]
        # for i in range(a):
        #     abstract_ticks.append((i + 1) * floor)
        # for i in range(b):
        #     abstract_ticks.append((i + 1) * ceil)
        n_bin = math.sqrt(self.n_clusters)
        # check if bin is an interger
        assert n_bin == int(n_bin)
        n_bin = int(n_bin)
        assert len(self.abs_ticks) == n_bin

        for i in range(len(self.abs_ticks)):
            if agent_pos[1] <= self.abs_ticks[i]:
                abstract_state_idx = i * n_bin
                break
        for i in range(len(self.abs_ticks)):
            if agent_pos[0] <= self.abs_ticks[i]:
                abstract_state_idx += i
                break
        return abstract_state_idx

    def get_abstract_value(self, agent_pos):
        abstract_state_idx = self.get_abstract_state_idx(agent_pos)
        abstract_value = self.abstract_V_array[abstract_state_idx]
        return abstract_state_idx, abstract_value

    def get_abs_indices_values(self, info: tuple):
        abs_value_l = []
        abs_value_next_l = []
        abs_indices = []
        abs_indices_next = []
        # shaping = self.gamma * self.abstract_V_table[quantized_next] - self.abstract_V_table[quantized]
        for i, info_i in enumerate(info):
            agent_pos = info_i["agent_pos1"]
            agent_pos_next = info_i["agent_pos2"]
            abs_idx, abs_value = self.get_abstract_value(agent_pos)
            abs_idx_next, abs_value_next = self.get_abstract_value(agent_pos_next)
            # shaping.append(abs_value_next - abs_value)
            abs_indices.append(abs_idx)
            abs_indices_next.append(abs_idx_next)
            abs_value_l.append(abs_value)
            abs_value_next_l.append(abs_value_next)

        return abs_indices, abs_indices_next, abs_value_l, abs_value_next_l

    def update_grdQ_shaping(
        self,
        state,
        action,
        next_state,
        reward,
        terminated,
        info,
        abs_indices,
        abs_indices_next,
    ):
        if hasattr(self, "lr_scheduler_ground_Q"):
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_scheduler_ground_Q(self._current_progress_remaining),
            )
        abs_indices = list(abs_indices)
        abs_indices_next = list(abs_indices_next)
        abs_value_l = self.abstract_V_array[abs_indices]
        abs_value_next_l = self.abstract_V_array[abs_indices_next]

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

            # shaping = []
            # abs_value_l = []
            # abs_value_next_l = []
            # abs_indices = []
            # abs_indices_next = []
            # # shaping = self.gamma * self.abstract_V_table[quantized_next] - self.abstract_V_table[quantized]
            # for i, info_i in enumerate(info):
            #     agent_pos = info_i["agent_pos1"]
            #     agent_pos_next = info_i["agent_pos2"]
            #     abs_idx, abs_value = self.get_abstract_value(agent_pos)
            #     abs_idx_next, abs_value_next = self.get_abstract_value(agent_pos_next)
            #     # shaping.append(abs_value_next - abs_value)
            #     abs_indices.append(abs_idx)
            #     abs_indices_next.append(abs_idx_next)
            #     abs_value_l.append(abs_value)
            #     abs_value_next_l.append(abs_value_next)
            #     delta = self.gamma * abs_value_next - abs_value
            #     shaping.append(delta)
            #     # if delta != 0:
            #     #     print("####################delta!=0")
            # # abs_value_l = torch.tensor(abs_value_l)
            # # abs_value_next_l = torch.tensor(abs_value_next_l)
            # # shaping = (abs_value_next_l - abs_value_l).unsqueeze(1).to(self.device)
            # shaping = torch.tensor(shaping).unsqueeze(1).to(self.device)
            abs_indices = torch.tensor(abs_indices).to(self.device)
            abs_indices_next = torch.tensor(abs_indices_next).to(self.device)
            abs_value_l = torch.tensor(abs_value_l).to(self.device)
            abs_value_next_l = torch.tensor(abs_value_next_l).to(self.device)
            # abs_value_l -= self.abstract_V_array.min()
            # abs_value_next_l -= self.abstract_V_array.min()
            mask = abs_indices != abs_indices_next
            # shaping = self.gamma * abs_value_next_l - abs_value_l
            shaping = abs_value_next_l - abs_value_l
            shaping = (shaping * mask.float()).unsqueeze(1)
            grd_q_target = (
                reward
                + self.omega * shaping * (1 - terminated.float())
                + self.gamma * grd_q_next_max * (1 - terminated.float())
            ).float()

        criterion = nn.SmoothL1Loss()
        ground_td_error = criterion(grd_q, grd_q_target)

        # [Update abstract V network]
        # mask_ = ~torch.tensor(
        #     [
        #         [torch.equal(a_state, next_a_state)]
        #         for a_state, next_a_state in zip(quantized, quantized_next)
        #     ]
        # ).to(self.device)
        # abs_v *= mask_
        # abs_v_target *= mask_

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        ground_td_error.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()

        for i, info_i in enumerate(info):
            self.shaping_distribution[
                info_i["agent_pos2"][1], info_i["agent_pos2"][0], info_i["agent_dir2"]
            ] += shaping[i]

        self.training_info["ground_Q_error"].append(ground_td_error.item())
        self.training_info["avg_shaping"].append(torch.mean(shaping).item())

    def update_grdQ_table_shaping(
        self,
        action: tuple,
        reward: tuple,
        terminated: tuple,
        info: tuple,
        abs_indices: list,
        abs_indices_next: list,
        abs_value_l: list,
        abs_value_next_l: list,
        use_shaping: bool,
        action_prime: int = None,
    ):

        if hasattr(self, "lr_scheduler_ground_Q"):
            self.lr_grd_Q = self.lr_scheduler_ground_Q(self._current_progress_remaining)

        # [Update ground Q network]
        delta_l = []
        shaping_l = []
        for i, info_i in enumerate(info):
            agent_pos1 = info_i["agent_pos1"]
            agent_dir1 = info_i["agent_dir1"]
            agent_pos2 = info_i["agent_pos2"]
            agent_dir2 = info_i["agent_dir2"]
            if use_shaping:
                if abs_indices_next[i] != abs_indices[i]:
                    shaping = self.gamma * abs_value_next_l[i] - abs_value_l[i]
                    # shaping = abs_value_next_l[i] - abs_value_l[i]
                    self.shaping_distribution[agent_pos2[1], agent_pos2[0], agent_dir2] += shaping
                else:
                    shaping = 0
            else:
                shaping = 0

            q = self.grd_Q_table[agent_pos1[1], agent_pos1[0], agent_dir1, action[i]]
            # Use Sarsa Backup if action prime is given, otherwise use Q-learning backup
            if action_prime:
                q_prime = self.grd_Q_table[agent_pos2[1], agent_pos2[0], agent_dir2, action_prime]
                q_target = (
                    reward[i]
                    + self.omega * shaping * (1 - terminated[i])
                    + self.gamma * q_prime * (1 - terminated[i])
                )
            else:
                max_q_prime = self.grd_Q_table[agent_pos2[1], agent_pos2[0], agent_dir2, :].max()
                q_target = (
                    reward[i]
                    + self.omega * shaping * (1 - terminated[i])
                    + self.gamma * max_q_prime * (1 - terminated[i])
                )
            delta = q_target - q
            self.grd_Q_table[agent_pos1[1], agent_pos1[0], agent_dir1, action[i]] += (
                self.lr_grd_Q * delta
            )
            delta_l.append(delta)
            shaping_l.append(shaping)

        self.training_info["ground_Q_error"].append(mean(delta_l))
        self.training_info["avg_shaping"].append(mean(shaping_l))

    def update_absV(
        self,
        abs_indices: tuple,
        abs_indices_next: tuple,
        reward: tuple,
        terminated: tuple,
        info: tuple,
    ):
        if hasattr(self, "lr_scheduler_abstract_V"):
            self.lr_abs_V = self.lr_scheduler_abstract_V(self._current_progress_remaining)
        # target = reward + self.gamma * abs_value_next_l
        # delta = target - abs_value_l
        abs_indices = list(abs_indices)
        abs_indices_next = list(abs_indices_next)
        abs_value_l = self.abstract_V_array[abs_indices]
        abs_value_next_l = self.abstract_V_array[abs_indices_next]

        # reward = reward / 10
        update_flag = np.full(len(self.abstract_V_array), 20, dtype=np.int8)
        delta_l = []
        for i, (abs_idx, abs_idx_next) in enumerate(zip(abs_indices, abs_indices_next)):
            if update_flag[abs_idx] > 0:
                if abs_idx == abs_idx_next and reward[i] == 0:
                    delta_l.append(0)
                else:
                    target = reward[i] + self.abs_gamma * abs_value_next_l[i] * (1 - terminated[i])
                    # target = reward[i] + self.gamma ** info[i]["interval4SemiMDP"] * abs_value_next_l[
                    #     i
                    # ] * (1 - terminated[i])
                    # target = reward[i] + self.gamma ** info[i]["interval4SemiMDP"] * abs_value_next_l[i]
                    delta = target - abs_value_l[i]
                    # if delta <= 0:
                    #     delta_l.append(0)
                    #     continue
                    self.abstract_V_array[abs_idx] += self.lr_abs_V * delta
                    update_flag[abs_idx] -= 1
                    delta_l.append(delta)
            else:
                delta_l.append(0)

        self.training_info["abstract_V_error"].append(mean(delta_l))

        return mean(delta_l)

    def update_absV_immediate(
        self,
        abs_indices: tuple,
        abs_indices_next: tuple,
        reward: tuple,
        terminated: tuple,
        info: tuple,
    ):
        if hasattr(self, "lr_scheduler_abstract_V"):
            self.lr_abs_V = self.lr_scheduler_abstract_V(self._current_progress_remaining)
        reward = list(reward)
        # target = reward + self.gamma * abs_value_next_l
        # delta = target - abs_value_l
        # abs_indices = list(abs_indices)
        # abs_indices_next = list(abs_indices_next)
        # abs_value_l = self.abstract_V_array[abs_indices]
        # abs_value_next_l = self.abstract_V_array[abs_indices_next]

        # reward = reward / 10
        delta_l = []
        for i, (abs_idx, abs_idx_next) in enumerate(zip(abs_indices, abs_indices_next)):
            if reward[i] < 0:
                reward[i] = 0
            if abs_idx == abs_idx_next and reward[i] == 0:
                delta_l.append(0)
            else:
                target = reward[i] + self.abs_gamma * self.abstract_V_array[abs_idx_next] * (
                    1 - terminated[i]
                )
                # target = reward[i] + self.gamma ** info[i]["interval4SemiMDP"] * abs_value_next_l[
                #     i
                # ] * (1 - terminated[i])
                # target = reward[i] + self.gamma ** info[i]["interval4SemiMDP"] * abs_value_next_l[i]
                delta = target - self.abstract_V_array[abs_idx]
                # if delta <= 0:
                #     delta_l.append(0)
                # else:
                self.abstract_V_array[abs_idx] += self.lr_abs_V * delta
                delta_l.append(delta)

        self.training_info["abstract_V_error"].append(mean(delta_l))

        return mean(delta_l)

    def maybe_update_absV(
        self,
        abs_indices: list,
        abs_indices_next: list,
        abs_value_l: list,
        abs_value_next_l: list,
        reward: Tensor,
        terminated: Tensor,
        info: tuple,
    ):
        """Update abstract value function when abstract error acrosses the threshold"""
        # target = reward + self.gamma * abs_value_next_l
        # delta = target - abs_value_l

        # reward = reward / 10
        abs_indices = np.array(abs_indices)
        abs_indices_next = np.array(abs_indices_next)
        abs_value_l = np.array(abs_value_l)
        abs_value_next_l = np.array(abs_value_next_l)
        reward = reward.squeeze().cpu().numpy()
        terminated = terminated.float().squeeze().cpu().numpy()
        interval4SemiMDP = np.array([info_i["interval4SemiMDP"] for info_i in info])

        mask = (abs_indices != abs_indices_next) | (reward != 0)

        target = reward + np.power(self.gamma, interval4SemiMDP) * abs_value_next_l * (
            1 - terminated
        )
        delta = target - abs_value_l
        delta *= mask

        if mean(delta) > 1e-5:
            self.abstract_V_array[abs_indices] += self.lr_abs_V * delta
            self.training_info["abstract_V_error"].append(mean(delta))
            to_update_grd_q = False
            return to_update_grd_q
        else:
            to_update_grd_q = True
            return to_update_grd_q

    def update_absV_potential_like(
        self,
        abs_indices: list,
        abs_indices_next: list,
        abs_value_l: list,
        abs_value_next_l: list,
        reward: Tensor,
        terminated: Tensor,
        info: tuple,
    ):
        # target = reward + self.gamma * abs_value_next_l
        # delta = target - abs_value_l

        # reward = reward / 10

        reward = reward.squeeze().tolist()
        terminated = terminated.float().squeeze().tolist()
        # abs_value_updated = np.zeros_like(self.abstract_V_array)
        sorted_abs_V_indices = np.argsort(self.abstract_V_array)
        upper_bounds = np.zeros_like(self.abstract_V_array, dtype=np.float32)
        # length = len(sorted_abs_V_indices)
        # for i, v in enumerate(sorted_abs_V_indices):
        #     t = 1
        #     while i + t < length:
        #         if self.abstract_V_array[sorted_abs_V_indices[i + t]] <= self.abstract_V_array[v]:
        #             t += 1
        #         else:
        #             break
        #     if i + t < length:
        #         upper_bounds[v] = self.abstract_V_array[sorted_abs_V_indices[i + t]]
        #     else:
        #         upper_bounds[v] = np.inf

        for i, j in enumerate(upper_bounds):
            if i < 8:
                if i < 6:
                    if (i + 1) % 3 == 0:
                        upper_bounds[i] = self.abstract_V_array[i + 3]
                    else:
                        upper_bounds[i] = max(
                            self.abstract_V_array[i + 1], self.abstract_V_array[i + 3]
                        )
                else:
                    upper_bounds[i] = self.abstract_V_array[i + 1]

        delta_l = []
        n_cross_upper_bound = 0
        for i, (abs_idx, abs_idx_next) in enumerate(zip(abs_indices, abs_indices_next)):
            if abs_idx == abs_idx_next and reward[i] == 0:
                delta_l.append(0)
            else:
                # target = reward[i] + self.gamma * abs_value_next_l[i] * (1 - terminated[i])
                target = reward[i] + self.gamma ** info[i]["interval4SemiMDP"] * abs_value_next_l[
                    i
                ] * (1 - terminated[i])
                delta = target - abs_value_l[i]
                # if delta <= 0:
                #     delta_l.append(0)
                #     continue
                # if self.abstract_V_array[abs_idx] + self.lr_abs_V * delta < upper_bounds[abs_idx]:
                #     self.abstract_V_array[abs_idx] += self.lr_abs_V * delta
                #     delta_l.append(delta)
                # else:
                #     delta_l.append(0)
                #     n_cross_upper_bound += 1

                self.abstract_V_array[abs_idx] += self.lr_abs_V * delta
                if self.abstract_V_array[abs_idx] + self.lr_abs_V * delta > upper_bounds[abs_idx]:
                    n_cross_upper_bound += 1

        self.training_info["abstract_V_error"].append(mean(delta_l))
        self.training_info["n_cross_upper_bound"].append(n_cross_upper_bound)

    def update(self, use_shaping: bool):
        """
        update with adaptive abstraction
        """
        if self.timesteps_done == self.init_steps:
            print("Init steps done")
        if self.timesteps_done == self.init_steps + 1:
            # self.cache_goal_transition()
            pass

        if use_shaping:
            if self.timesteps_done % self.abstract_learn_every == 0:
                for _ in range(self.abstract_gradient_steps):
                    (
                        state,
                        abs_state,
                        action,
                        next_state,
                        next_abs_state,
                        reward,
                        terminated,
                        info,
                    ) = self.memory.sample(self.batch_size, mode="pure")
                    # [data augmentation]
                    # state = self.aug(state)
                    # next_state = self.aug(next_state)

                    # [update abstract_V]
                    self.update_absV_immediate(
                        abs_state,
                        next_abs_state,
                        reward,
                        terminated,
                        info,
                    )

            if self.timesteps_done % self.ground_learn_every == 0:
                for _ in range(self.ground_gradient_steps):
                    (
                        state,
                        abs_state,
                        action,
                        next_state,
                        next_abs_state,
                        reward,
                        terminated,
                        info,
                    ) = self.memory.sample(self.batch_size, mode=None)

                    # [data augmentation]
                    # state = self.aug(state)
                    # next_state = self.aug(next_state)

                    # [update ground_Q with reward shaping]
                    self.update_grdQ_shaping(
                        state,
                        action,
                        next_state,
                        reward,
                        terminated,
                        info,
                        abs_state,
                        next_abs_state,
                    )
        else:
            # [purely update ground Q]
            if self.timesteps_done % self.ground_learn_every == 0:
                state, _, action, next_state, _, reward, terminated, info = self.memory.sample(
                    self.batch_size, mode=None
                )
                self.update_grdQ_pure(state, action, next_state, reward, terminated, info)

        if self.timesteps_done % self.ground_sync_every == 0:
            soft_sync_params(
                self.ground_Q.parameters(),
                self.ground_Q_target.parameters(),
                self.ground_tau,
            )
            # soft_sync_params(
            #     self.ground_Q.encoder.parameters(),
            #     self.ground_Q_target.encoder.parameters(),
            #     self.encoder_tau,
            # )
            # soft_sync_params(
            #     self.ground_Q.critic.parameters(),
            #     self.ground_Q_target.critic.parameters(),
            #     self.ground_tau,
            # )

        if self.timesteps_done % self.save_model_every == 0:
            pass

        if self.timesteps_done % self.reset_training_info_every == 0:
            self.log_training_info(wandb_log=True)
            self.reset_training_info()

    def update_table(self, use_shaping: bool):
        if self.timesteps_done == self.init_steps:
            print("Init steps done")

        if hasattr(self, "lr_scheduler_ground_Q"):
            self.lr_grd_Q = self.lr_scheduler_ground_Q(self._current_progress_remaining)
        if hasattr(self, "lr_scheduler_abstract_V"):
            self.lr_abs_V = self.lr_scheduler_abstract_V(self._current_progress_remaining)

        if self.timesteps_done % self.abstract_learn_every == 0:
            for _ in range(self.abstract_gradient_steps):
                state, action, next_state, reward, terminated, info = self.memory.sample(
                    self.batch_size, mode="pure"
                )
                # [data augmentation]
                # state = self.aug(state)
                # next_state = self.aug(next_state)

                # [extract abstract information]
                (
                    abs_indices,
                    abs_indices_next,
                    abs_value_l,
                    abs_value_next_l,
                ) = self.get_abs_indices_values(info)

                # [update abstract_V]
                self.update_absV(
                    abs_indices,
                    abs_indices_next,
                    abs_value_l,
                    abs_value_next_l,
                    reward,
                    terminated,
                    info,
                )

        if self.timesteps_done % self.ground_learn_every == 0:
            for _ in range(self.ground_gradient_steps):
                # [update ground_Q with reward shaping, purely update ground Q with use_shaping=False]
                # if grd_update_step > 0:
                state, action, next_state, reward, terminated, info = self.memory.sample(
                    self.batch_size, mode="pure"
                )

                (
                    abs_indices,
                    abs_indices_next,
                    abs_value_l,
                    abs_value_next_l,
                ) = self.get_abs_indices_values(info)

                self.update_grdQ_table_shaping(
                    action,
                    reward,
                    terminated,
                    info,
                    abs_indices,
                    abs_indices_next,
                    abs_value_l,
                    abs_value_next_l,
                    use_shaping=use_shaping,
                    action_prime=None,
                )
                # grd_update_step -= 1

        if self.timesteps_done % self.save_model_every == 0:
            pass

        if self.timesteps_done % self.reset_training_info_every == 0:
            self.log_training_info(wandb_log=True)
            self.reset_training_info()

    def update_table2(self, use_shaping: bool):
        if self.timesteps_done == self.init_steps:
            print("Init steps done")

        if self.timesteps_done % self.abstract_learn_every == 0:
            for _ in range(self.abstract_gradient_steps):
                state, action, next_state, reward, terminated, info = self.memory.sample(
                    self.batch_size, mode="pure"
                )
                # [data augmentation]
                # state = self.aug(state)
                # next_state = self.aug(next_state)

                # [extract abstract information]
                (
                    abs_indices,
                    abs_indices_next,
                    abs_value_l,
                    abs_value_next_l,
                ) = self.get_abs_indices_values(info)

                # [update abstract_V]
                self.update_absV(
                    abs_indices,
                    abs_indices_next,
                    abs_value_l,
                    abs_value_next_l,
                    reward,
                    terminated,
                    info,
                )

                # if self.timesteps_done % self.ground_learn_every == 0:
                #     for _ in range(self.ground_gradient_steps):
                # [update ground_Q with reward shaping, purely update ground Q with use_shaping=False]
                # if grd_update_step > 0:
                # state, action, next_state, reward, terminated, info = self.memory.sample(
                #     self.batch_size, mode="pure"
                # )

                (
                    abs_indices,
                    abs_indices_next,
                    abs_value_l,
                    abs_value_next_l,
                ) = self.get_abs_indices_values(info)

                self.update_grdQ_table_shaping(
                    action,
                    reward,
                    terminated,
                    info,
                    abs_indices,
                    abs_indices_next,
                    abs_value_l,
                    abs_value_next_l,
                    use_shaping=use_shaping,
                    action_prime=None,
                )
                # grd_update_step -= 1

        if self.timesteps_done % self.save_model_every == 0:
            pass

        if self.timesteps_done % self.reset_training_info_every == 0:
            self.log_training_info(wandb_log=True)
            self.reset_training_info()

    def update_table_no_memory(self, use_shaping: bool, action_prime: int):
        if self.timesteps_done == self.init_steps:
            print("Init steps done")
        if hasattr(self, "lr_scheduler_ground_Q"):
            self.lr_grd_Q = self.lr_scheduler_ground_Q(self._current_progress_remaining)
        if hasattr(self, "lr_scheduler_abstract_V"):
            self.lr_abs_V = self.lr_scheduler_abstract_V(self._current_progress_remaining)

        latest_transition = self.memory.latest_transition
        state = (latest_transition.state,)
        action = (latest_transition.action,)
        next_state = (latest_transition.next_state,)
        reward = (latest_transition.reward,)
        terminated = (latest_transition.terminated,)
        info = (latest_transition.info,)

        # [data augmentation]
        # state = self.aug(state)
        # next_state = self.aug(next_state)

        # [extract abstract information]
        (
            abs_indices,
            abs_indices_next,
            abs_value_l,
            abs_value_next_l,
        ) = self.get_abs_indices_values(info)

        # [update abstract_V]
        self.update_absV(
            abs_indices,
            abs_indices_next,
            abs_value_l,
            abs_value_next_l,
            reward,
            terminated,
            info,
        )

        # [update ground_Q with reward shaping, purely update ground Q with use_shaping=False]
        # state, action, next_state, reward, terminated, info = self.memory.sample(
        #     self.batch_size
        # )

        (
            abs_indices,
            abs_indices_next,
            abs_value_l,
            abs_value_next_l,
        ) = self.get_abs_indices_values(info)

        self.update_grdQ_table_shaping(
            action,
            reward,
            terminated,
            info,
            abs_indices,
            abs_indices_next,
            abs_value_l,
            abs_value_next_l,
            use_shaping=use_shaping,
            action_prime=action_prime,
        )

        if self.timesteps_done % self.save_model_every == 0:
            pass

        if self.timesteps_done % self.reset_training_info_every == 0:
            self.log_training_info(wandb_log=True)
            self.reset_training_info()

    def update_table_abs_update_non_parallel(self, use_shaping: bool):
        if self.timesteps_done == self.init_steps:
            print("Init steps done")

        if (
            self.timesteps_done == self.init_steps
            or self.timesteps_done % self.ground_learn_every == 0
        ):
            # abs_update_step = 20
            grd_update_step = 1
            if self.goal_found:
                while self.n_abs_updates > 0:
                    state, action, next_state, reward, terminated, info = self.memory.sample(
                        self.batch_size
                    )
                    # [data augmentation]
                    # state = self.aug(state)
                    # next_state = self.aug(next_state)

                    # [extract abstract information]
                    (
                        abs_indices,
                        abs_indices_next,
                        abs_value_l,
                        abs_value_next_l,
                    ) = self.get_abs_indices_values(info)

                    # [update abstract_V]
                    abs_error = self.update_absV(
                        abs_indices,
                        abs_indices_next,
                        abs_value_l,
                        abs_value_next_l,
                        reward,
                        terminated,
                        info,
                    )
                    if abs_error != 0:
                        self.n_abs_updates -= 1

                # [update ground_Q with reward shaping, purely update ground Q with use_shaping=False]
            for _ in range(grd_update_step):
                state, action, next_state, reward, terminated, info = self.memory.sample(
                    self.batch_size
                )

                (
                    _,
                    _,
                    abs_value_l,
                    abs_value_next_l,
                ) = self.get_abs_indices_values(info)

                self.update_grdQ_table_shaping(
                    action,
                    reward,
                    terminated,
                    info,
                    abs_value_l,
                    abs_value_next_l,
                    use_shaping=use_shaping,
                )

        if self.timesteps_done % self.save_model_every == 0:
            pass

        if self.timesteps_done % self.reset_training_info_every == 0:
            self.log_training_info(wandb_log=True)
            self.reset_training_info()

    def update_table_abs_update_non_parallel2(self, use_shaping: bool):
        if self.timesteps_done == self.init_steps:
            print("Init steps done")

        if (
            self.timesteps_done == self.init_steps
            or self.timesteps_done % self.ground_learn_every == 0
        ):
            abs_update_step = 1
            grd_update_step = 1
            for _ in range(abs_update_step):
                state, action, next_state, reward, terminated, info = self.memory.sample(
                    self.batch_size
                )
                # [data augmentation]
                # state = self.aug(state)
                # next_state = self.aug(next_state)

                # [extract abstract information]
                (
                    abs_indices,
                    abs_indices_next,
                    abs_value_l,
                    abs_value_next_l,
                ) = self.get_abs_indices_values(info)

                # [update abstract_V]
                to_update_grd_q = self.maybe_update_absV(
                    abs_indices,
                    abs_indices_next,
                    abs_value_l,
                    abs_value_next_l,
                    reward,
                    terminated,
                    info,
                )

            # [update ground_Q with reward shaping, purely update ground Q with use_shaping=False]
            if to_update_grd_q:
                for _ in range(grd_update_step):
                    state, action, next_state, reward, terminated, info = self.memory.sample(
                        self.batch_size
                    )

                    (
                        _,
                        _,
                        abs_value_l,
                        abs_value_next_l,
                    ) = self.get_abs_indices_values(info)

                    self.update_grdQ_table_shaping(
                        action,
                        reward,
                        terminated,
                        info,
                        abs_value_l,
                        abs_value_next_l,
                        use_shaping=use_shaping,
                    )

        if self.timesteps_done % self.save_model_every == 0:
            pass

        if self.timesteps_done % self.reset_training_info_every == 0:
            self.log_training_info(wandb_log=True)
            self.reset_training_info()

    def update_grdQ_pure(self, state, action, next_state, reward, terminated, info):
        if hasattr(self, "lr_scheduler_ground_Q"):
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_scheduler_ground_Q(self._current_progress_remaining),
            )

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
            grd_q_target = (reward + self.gamma * grd_q_next_max * (1 - terminated.float())).float()

        criterion = nn.SmoothL1Loss()
        ground_td_error = criterion(grd_q, grd_q_target)

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        ground_td_error.backward()
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()

        self.training_info["ground_Q_error"].append(ground_td_error.item())
