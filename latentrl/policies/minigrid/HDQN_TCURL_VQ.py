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
from torchvision import transforms
import wandb
from tqdm import tqdm
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
    DVN,
    MlpModel,
    V_MLP,
    Decoder,
    Decoder_MiniGrid,
    DQN_Repara,
    EncoderImg,
    Encoder_MiniGrid_PartialObs,
    Encoder_MiniGrid,
    RandomShiftsAug,
    RandomEncoder,
    RandomEncoderMiniGrid,
    CURL,
    MOCO,
    VectorQuantizerLinear,
    VectorQuantizerLinearSoft,
    VectorQuantizerLinearDiffable,
    VQSoftAttention,
)
from PIL import Image
from policies.HDQN import HDQN
from policies.utils import (
    EncoderMaker,
    ReplayMemory,
    ReplayMemoryWithCluster,
    ReplayBufferNStep,
    transition_np2torch,
    make_encoder,
)
from common.Logger import LoggerWandb, Logger

from sklearn.cluster import KMeans
from sympy.solvers import solve
from torch import Tensor, nn
from torchsummary import summary
from torchvision.utils import save_image

from minigrid import Wall


class HDQN_TCURL_VQ(HDQN):
    def __init__(self, config, env, logger: LoggerWandb):
        super().__init__(config, env, logger)
        # self.set_hparams(config)
        self.ground_Q = DQN(
            # observation_space=env.observation_space,
            action_space=env.action_space,
            # encoder=EncoderMaker(input_format=config.input_format, agent=self).make(),
            encoder=make_encoder(
                input_format=config.input_format,
                input_channels=min(env.observation_space.shape),
                linear_dims=config.grd_encoder_linear_dims,
                observation_space=env.observation_space,
                hidden_channels=config.grd_hidden_channels,
            ),
            mlp_hidden_dim_grd=config.mlp_hidden_dim_grd,
        ).to(self.device)

        self.ground_Q_target = DQN(
            # observation_space=env.observation_space,
            action_space=env.action_space,
            # encoder=EncoderMaker(input_format=config.input_format, agent=self).make(),
            encoder=make_encoder(
                input_format=config.input_format,
                input_channels=min(env.observation_space.shape),
                linear_dims=config.grd_encoder_linear_dims,
                observation_space=env.observation_space,
                hidden_channels=config.grd_hidden_channels,
            ),
            mlp_hidden_dim_grd=config.mlp_hidden_dim_grd,
        ).to(self.device)

        self.ground_Q_target.load_state_dict(self.ground_Q.state_dict())
        # self.ground_Q_target.train()

        # self.abs_V = MlpModel(
        #     input_dim=self.curl.encoder.linear_out_dim,
        #     hidden_dims=config.mlp_hidden_dim_abs,
        #     output_dim=1,
        #     activation=nn.ReLU,
        # ).to(self.device)
        # self.abs_V_target = MlpModel(
        #     input_dim=self.curl.encoder_target.linear_out_dim,
        #     hidden_dims=config.mlp_hidden_dim_abs,
        #     output_dim=1,
        #     activation=nn.ReLU,
        # ).to(self.device)

        self.abs_V = DVN(
            # encoder=EncoderMaker(input_format=config.input_format, agent=self).make(),
            encoder=make_encoder(
                input_format=config.input_format,
                input_channels=min(env.observation_space.shape),
                linear_dims=config.abs_encoder_linear_dims,
                observation_space=env.observation_space,
                hidden_channels=config.grd_hidden_channels,
            ),
            mlp_hidden_dim_abs=config.mlp_hidden_dim_abs,
        ).to(self.device)

        self.abs_V_target = DVN(
            # encoder=EncoderMaker(input_format=config.input_format, agent=self).make(),
            encoder=make_encoder(
                input_format=config.input_format,
                input_channels=min(env.observation_space.shape),
                linear_dims=config.abs_encoder_linear_dims,
                observation_space=env.observation_space,
                hidden_channels=config.grd_hidden_channels,
            ),
            mlp_hidden_dim_abs=config.mlp_hidden_dim_abs,
        ).to(self.device)

        self.abs_V_target.load_state_dict(self.abs_V.state_dict())

        self.abs_V_array = np.zeros((config.num_vq_embeddings))

        self.curl = CURL(
            encoder=self.abs_V.encoder,
            encoder_target=self.abs_V_target.encoder,
        ).to(self.device)

        self.abs_encoder = self.abs_V.encoder

        # self.vq = VQSoftAttention(
        #     num_embeddings=config.num_vq_embeddings,
        #     embedding_dim=self.curl.encoder.linear_out_dim,
        #     beta=0.25,
        #     softmax_alpha=config.vq_softmin_beta,
        # ).to(self.device)

        self.vq = VectorQuantizerLinearSoft(
            num_embeddings=config.num_vq_embeddings,
            embedding_dim=self.curl.encoder.linear_out_dim,
            beta=0.25,
            softmin_beta=config.vq_softmin_beta,
        ).to(self.device)

        # self.moco = MOCO(
        #     encoder_maker=EncoderMaker(input_format=config.input_format, agent=self),
        #     vq=self.vq,
        #     K=32 * 2048,  # 32 can be batch size
        #     m=0.999,
        #     T=0.07,
        # )

        self.aug = RandomShiftsAug(pad=4)
        self.memory = ReplayBufferNStep(
            self.size_replay_memory,
            self.device,
            gamma=config.gamma,
            batch_size=config.batch_size,
        )

        self.count_vis = 0
        self.goal_found = False

        self._create_optimizers(config)
        self.train()

    def train(self, training=True):
        self.training = training
        self.ground_Q.train(training)
        self.abs_V.train(training)

    def set_hparams(self, config):
        # Hyperparameters
        # self.total_episodes = config.total_episodes
        self.total_timesteps = config.total_timesteps
        self.init_steps = config.init_steps  # min. experiences before training
        self.batch_size = config.batch_size
        self.batch_size_repre = config.batch_size_repre
        self.size_replay_memory = config.size_replay_memory
        self.gamma = config.gamma
        self.abs_gamma = config.abstract_gamma
        self.omega = config.omega

        self.grd_hidden_channels = config.grd_hidden_channels
        self.grd_encoder_linear_dims = config.grd_encoder_linear_dims
        self.mlp_hidden_dim_grd = config.mlp_hidden_dim_grd
        self.ground_learn_every = config.ground_learn_every
        # self.ground_gradient_steps = config.ground_gradient_steps
        self.ground_sync_every = config.ground_sync_every
        self.ground_Q_critic_tau = config.ground_Q_critic_tau
        self.ground_Q_encoder_tau = config.ground_Q_encoder_tau

        self.mlp_hidden_dim_abs = config.mlp_hidden_dim_abs
        self.abstract_learn_every = config.abstract_learn_every
        # self.abstract_gradient_steps = config.abstract_gradient_steps
        self.abstract_sync_every = config.abstract_sync_every
        self.abs_V_encoder_tau = config.abstract_V_encoder_tau
        self.abs_V_critic_tau = config.abstract_V_critic_tau

        self.curl_vq_learn_every = config.curl_vq_learn_every
        self.curl_vq_sync_every = config.curl_vq_sync_every
        self.curl_vq_gradient_steps = config.curl_vq_gradient_steps
        self.curl_tau = config.curl_tau
        self.num_vq_embeddings = config.num_vq_embeddings
        # self.dim_vq_embeddings = config.dim_vq_embeddings

        self.clip_grad = config.clip_grad
        self.clip_reward = config.clip_reward
        self.encoded_detach4abs = config.encoded_detach4abs
        self.input_format = config.input_format
        self.use_shaping = config.use_shaping

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
        # encoded = self.ground_Q.forward_conv(state)
        # encoded = self.curl.encoder(state)
        encoded = self.abs_encoder(state)
        _, _, _, output_dict = self.vq(encoded)
        abstract_state_inds = output_dict["hard_encoding_inds"]
        return abstract_state_inds.squeeze().item()

    @torch.no_grad()
    def get_abs_value(self, state, mode="hard"):
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        # encoded = self.ground_Q.forward_conv(state)
        # encoded = self.curl.encoder(state)
        encoded = self.abs_encoder(state)
        quantized, _, _, output_dict = self.vq(encoded)
        if mode == "table":
            abstract_state_inds = output_dict["hard_encoding_inds"]
            abs_value = self.abs_V_array[abstract_state_inds.squeeze().item()]
        elif mode == "hard":
            abs_value = self.abs_V.critic(output_dict["hard_quantized_latents"]).squeeze().item()
        elif mode == "target_hard":
            abs_value = (
                self.abs_V_target.critic(output_dict["hard_quantized_latents"]).squeeze().item()
            )
        elif mode == "soft":
            abs_value = self.abs_V.critic(quantized).squeeze().item()
        elif mode == "target_soft":
            abs_value = self.abs_V_target.critic(quantized).squeeze().item()
        return abs_value

    @torch.no_grad()
    def get_grd_reduction_v(self, state, reduction_mode="max"):
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        grd_q, encoded = self.ground_Q(state)
        # grd_q, _ = self.ground_Q_target(state)
        if reduction_mode == "max":
            return grd_q.cpu().numpy().max()
        elif reduction_mode == "mean":
            return grd_q.cpu().numpy().mean()

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
                cell = self.env.grid.get(w, h)
                if cell is None or cell.can_overlap():
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
        self.L.log_img("Images/abstraction", img)
        img_buffer.close()
        plt.close(fig_abs)

    def vis_abstract_values(self, prefix: str = None, mode=None):
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
                cell = self.env.grid.get(w, h)
                if cell is None or cell.can_overlap():
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
        # wandb.log({f"Images/abs_values_{mode}": wandb.Image(img)})
        self.L.log_img(f"Images/abs_values_{mode}", img)
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
                # if not isinstance(self.env.grid.get(w, h), Wall):
                cell = self.env.grid.get(w, h)
                if cell is None or cell.can_overlap():
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
                        # grd_v_avg = np.log(grd_v_avg + 1e-5) / np.log(norm_log)
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
        vmax = max(grd_v_avg_values)
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
        # wandb.log({f"Images/grd_values_{reduction_mode}": wandb.Image(img)})
        self.L.log_img(f"Images/grd_values_{reduction_mode}", img)
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

        for w in range(width):
            # w += 1
            for h in range(height):
                # h += 1
                cell = self.env.grid.get(w, h)
                if cell is None or cell.can_overlap():
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
        # wandb.log({f"Images/grd_visits_log{norm_log}_{suffix}": wandb.Image(img)})
        self.L.log_img(f"Images/grd_visits_log{norm_log}_{suffix}", img)
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

        if isinstance(config.lr_curl, str) and config.lr_curl.startswith("lin"):
            self.lr_scheduler_curl = linear_schedule(float(config.lr_curl.split("_")[1]))
            self.lr_curl = self.lr_scheduler_curl(self._current_progress_remaining)
        else:
            self.lr_curl = config.lr_curl

        if isinstance(config.safe_ratio, str) and config.safe_ratio.startswith("lin"):
            self.safe_ratio_scheduler = linear_schedule(
                float(config.safe_ratio.split("_")[1]), reduce=True
            )
            self.safe_ratio = self.safe_ratio_scheduler(self._current_progress_remaining)
        else:
            self.safe_ratio = config.safe_ratio
        # self.ground_Q_optimizer = optim.RMSprop(
        #     self.ground_Q_net.parameters(), lr=lr_ground_Q, alpha=0.95, momentum=0, eps=0.01
        # )
        # self.abstract_V_optimizer = optim.RMSprop(
        #     self.ground_Q_net.parameters(), lr=lr_abstract_V, alpha=0.95, momentum=0.95, eps=0.01
        # )
        self.curl_optimizer = optim.Adam(self.curl.parameters(), lr=self.lr_curl)
        self.vq_optimizer = optim.Adam(self.vq.parameters(), lr=self.lr_vq)
        self.abs_V_optimizer = optim.Adam(self.abs_V.parameters(), lr=self.lr_abs_V)
        self.ground_Q_optimizer = optim.Adam(self.ground_Q.parameters(), lr=self.lr_grd_Q)
        # self.curl_optimizer = optim.RMSprop(self.curl.parameters(), lr=self.lr_curl)
        # self.vq_optimizer = optim.RMSprop(self.vq.parameters(), lr=self.lr_vq)
        # self.abs_V_optimizer = optim.RMSprop(self.abs_V.parameters(), lr=self.lr_abs_V)
        # self.ground_Q_optimizer = optim.RMSprop(self.ground_Q.parameters(), lr=self.lr_grd_Q)
        # self.curl_optimizer = optim.RMSprop(
        #     self.curl.parameters(), lr=self.lr_curl, alpha=0.95, centered=True, eps=0.01
        # )
        # self.vq_optimizer = optim.RMSprop(
        #     self.vq.parameters(), lr=self.lr_vq, alpha=0.95, centered=True, eps=0.01
        # )
        # self.abs_V_optimizer = optim.RMSprop(
        #     self.abs_V.parameters(), lr=self.lr_abs_V, alpha=0.95, centered=True, eps=0.01
        # )
        # self.ground_Q_optimizer = optim.RMSprop(
        #     self.ground_Q.parameters(), lr=self.lr_grd_Q, alpha=0.95, centered=True, eps=0.01
        # )  # created for Minatar, used by original authors

        # self.moco_optimizer = optim.Adam(self.moco.parameters(), lr=self.lr_curl)

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
        gamma,
        info,
    ):
        # abs and grd sharing the same CNN
        if hasattr(self, "lr_scheduler_ground_Q"):
            self.lr_grd_Q = self.lr_scheduler_ground_Q(self._current_progress_remaining)
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_grd_Q,
            )
        if hasattr(self, "lr_scheduler_abstract_V"):
            self.lr_abs_V = self.lr_scheduler_abstract_V(self._current_progress_remaining)
            update_learning_rate(self.abs_V_optimizer, self.lr_abs_V)
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
        abs_v = self.abs_V(quantized)

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
                # + self.omega * shaping
                + gamma * grd_q_next_max
            )
            # mask = (quantized == quantized_next) & (reward == 0)
            # mask = (quantized != quantized_next) | (reward != 0)
            mask = (quantized != quantized_next).any(dim=1).unsqueeze(1)
            mask = mask | (reward != 0)
            mask = mask.float()
            abs_v_next = self.abs_V_target(quantized_next)
            abs_v_target = reward + gamma * abs_v_next
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
        self.abs_V_optimizer.zero_grad(set_to_none=True)
        self.vq_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                if param.grad is not None:  # make sure grad is not None
                    param.grad.data.clamp_(-1, 1)
            for param in self.abs_V.parameters():
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
        self.abs_V_optimizer.step()
        self.vq_optimizer.step()

        # for i, info_i in enumerate(info):
        #     self.shaping_distribution[
        #         info_i["agent_pos2"][1], info_i["agent_pos2"][0], info_i["agent_dir2"]
        #     ] += shaping[i]

        # self.training_info["ground_Q_error"].append(ground_td_error.item())
        # self.training_info["abstract_V_error"].append(abs_td_error.item())
        # self.training_info["entrophy_vq"].append(entrophy_vq.item())
        # self.training_info["vq_loss"].append(vq_loss.item())
        # self.training_info["absV_grdQ(rd)_l1"].append(diff_l1_abs_grd.item())
        # self.training_info["absV_grdQ(rd)_l2"].append(diff_l2_abs_grd.item())
        # self.training_info["total_loss"].append(total_loss.item())
        # self.training_info["avg_shaping"].append(torch.mean(shaping).item())

    def update_grdQ(
        self,
        obs,
        act,
        n_obs,
        rew,
        gamma,
        use_vq=True,
        use_shaping=False,
        approach_abs=False,
    ):
        # abs and grd sharing the same CNN

        if self.clip_reward:
            rew.clamp_(-1, 1)

        # [data augmentation]
        # if self.input_format == "full_img":
        #     obs = self.aug(obs)
        #     n_obs = self.aug(n_obs)

        # [Update ground Q network]
        grd_q, _ = self.ground_Q(obs)
        grd_q_reduction = torch.mean(grd_q, dim=1, keepdim=True)
        # grd_q_reduction = torch.amax(grd_q.detach(), dim=1, keepdim=True)
        # grd_q_mean_bytarget = torch.mean(self.ground_Q_target(state)[0].detach(), dim=1, keepdim=True)
        grd_q = grd_q.gather(1, act)

        with torch.no_grad():
            if use_vq:
                encoded = self.abs_V.encoder(obs)
                encoded, _, _, _ = self.vq(encoded)
                abs_v = self.abs_V_target.critic(encoded)
                n_encoded = self.abs_V.encoder(n_obs)
                n_encoded, _, _, _ = self.vq(n_encoded)
                n_abs_v = self.abs_V_target.critic(n_encoded)
            else:
                abs_v = self.abs_V_target(obs)
                n_abs_v = self.abs_V_target(n_obs)
            # [Vanilla DQN]
            grd_q_next, encoded_next = self.ground_Q_target(n_obs)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)

            # [Double DQN]
            # action_argmax_target = self.ground_Q_target(n_obs)[0].argmax(dim=1, keepdim=True)
            # grd_q_next_max = self.ground_Q(n_obs)[0].gather(1, action_argmax_target)

            grd_q_target = rew + gamma * grd_q_next_max
            if use_shaping:
                # shaping = self.abs_V_target(quantized_next) - self.abs_V_target(quantized)
                # shaping = abs_v_next_hard - abs_v_hard
                shaping = n_abs_v - abs_v
                grd_q_target += self.omega * shaping
                self.L.log({"avg_shaping": shaping.mean().item()})
            # mask = (quantized != quantized_next).any(dim=1).unsqueeze(1)
            # mask = mask | (reward != 0)
            # mask = mask.float()
            # abs_v_target = reward + gamma * abs_v_next
            # abs_v_target = abs_v_target * mask + abs_v_next * (1 - mask)

        criterion = nn.SmoothL1Loss()
        # criterion = F.mse_loss
        # ground_td_error = F.mse_loss(grd_q, grd_q_target)
        # ground_td_error = criterion(grd_q, grd_q_target)
        # target_value = (
        #     self.safe_ratio * (rew + gamma * abs_v_next) + (1 - self.safe_ratio) * grd_q_target
        # )
        if random.random() < self.safe_ratio:
            target_value = rew + gamma * n_abs_v
        else:
            target_value = grd_q_target

        ground_td_error = criterion(grd_q, target_value)
        # abs_td_error = F.mse_loss(abs_v, abs_v_target)
        # abs_grd_diff = criterion(abs_v, grd_q)
        # commit_error_grd2abs = criterion(abs_v.detach(), grd_q_reduction)
        # commit_error_abs2grd = criterion(abs_v, grd_q_reduction.detach())
        # commit_error_abs2grd = F.mse_loss(abs_v, grd_q_target)

        # commit_error_grd2abs = torch.linalg.norm(F.relu(abs_v.detach() - grd_q_reduction))
        # commit_error_abs2grd = torch.linalg.norm(F.relu(grd_q_target - abs_v))

        # commit_error_grd2abs = F.relu(abs_v.detach() - grd_q_reduction).mean()
        # commit_error_abs2grd = F.relu(grd_q_target - abs_v).mean()
        with torch.no_grad():
            diff_l2_abs_grd = F.mse_loss(abs_v, grd_q)
            # diff_l1_abs_grd = F.l1_loss(abs_v, grd_q_reduction)
            diff_l1_abs_grd = (abs_v - grd_q).mean()

        # [Compute total loss]
        total_loss = ground_td_error
        if approach_abs:
            # grd_match_abs_err = F.mse_loss(grd_q, abs_v)
            grd_match_abs_err = F.mse_loss(grd_q, rew + gamma * n_abs_v)
            total_loss += 0.1 * grd_match_abs_err

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        # self.abs_V_optimizer.zero_grad(set_to_none=True)
        # self.vq_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                if param.grad is not None:  # make sure grad is not None
                    param.grad.data.clamp_(-1, 1)
            # for param in self.abs_V.parameters():
            #     if param.grad is not None:
            #         param.grad.data.clamp_(-1, 1)
            # for param in self.vq.parameters():
            #     if param.grad is not None:
            #         param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()
        # self.abs_V_optimizer.step()
        # self.vq_optimizer.step()

        # for i, info_i in enumerate(info):
        #     self.shaping_distribution[
        #         info_i["agent_pos2"][1], info_i["agent_pos2"][0], info_i["agent_dir2"]
        #     ] += shaping[i]

        metric = {
            "Training_Info/update_grdQ/ground_Q_error": ground_td_error.item(),
            "Training_Info/update_grdQ/update_Q total_loss": total_loss.item(),
            "Training_Info/update_grdQ/grd_q": grd_q.mean().item(),
            "Training_Info/update_grdQ/grd_q_mean": grd_q_reduction.mean().item(),
            "Training_Info/update_grdQ/absV_grdQ_l1": diff_l1_abs_grd.item(),
            "Training_Info/update_grdQ/absV_grdQ_l2": diff_l2_abs_grd.item(),
        }
        self.L.log(metric)

        return total_loss

    def update_grdQ_critic(
        self,
        obs,
        act,
        n_obs,
        rew,
        gamma,
        use_vq,
        approach_abs,
    ):

        if self.clip_reward:
            rew.clamp_(-1, 1)

        # [data augmentation]
        if self.input_format == "full_img":
            obs = self.aug(obs)
            n_obs = self.aug(n_obs)

        # [Update ground Q network]
        # grd_q, encoded = self.ground_Q(obs)
        with torch.no_grad():
            encoded = self.curl.encoder(obs)
            if use_vq:
                encoded, _, _, _ = self.vq(encoded)
        grd_q = self.ground_Q.critic(encoded)
        grd_q_reduction = torch.mean(grd_q, dim=1, keepdim=True)
        grd_q = grd_q.gather(1, act)

        with torch.no_grad():
            n_encoded = self.curl.encoder(n_obs)
            if use_vq:
                n_encoded, _, _, _ = self.vq(n_encoded)

            # [Vanilla DQN]
            # grd_q_next, encoded_next = self.ground_Q_target(n_obs)
            grd_q_next = self.ground_Q_target.critic(n_encoded)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)

            # [Double DQN]
            # selected_action = self.ground_Q.critic(n_quantized).argmax(dim=1, keepdim=True)
            # grd_q_next_max = self.ground_Q_target.critic(n_quantized).gather(1, selected_action)

            grd_q_target = rew + gamma * grd_q_next_max

        ground_td_error = F.mse_loss(grd_q, grd_q_target)

        # [Compute total loss]
        total_loss = ground_td_error
        if approach_abs:
            with torch.no_grad():
                abs_v = self.abs_V_target(encoded)
            grd_match_abs_err = F.mse_loss(grd_q_reduction, abs_v)
            total_loss += 0.1 * grd_match_abs_err

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        # self.abs_V_optimizer.zero_grad(set_to_none=True)
        # self.vq_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                if param.grad is not None:  # make sure grad is not None
                    param.grad.data.clamp_(-1, 1)
            # for param in self.abs_V.parameters():
            #     if param.grad is not None:
            #         param.grad.data.clamp_(-1, 1)
            # for param in self.vq.parameters():
            #     if param.grad is not None:
            #         param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()

        # self.training_info["ground_Q_error"].append(ground_td_error.item())
        # self.training_info["abstract_V_error"].append(abs_td_error.item())
        # self.training_info["entrophy_vq"].append(entrophy_vq.item())
        # self.training_info["vq_loss"].append(vq_loss.item())

    def update_absV(
        self,
        obs,
        n_obs,
        rew,
        gamma,
        use_vq=False,
        detach_encoder=False,
    ):
        if self.clip_reward:
            rew.clamp_(-1, 1)
        # # [data augmentation]
        # if self.input_format == "full_img":
        #     obs = self.aug(obs)
        #     n_obs = self.aug(n_obs)
        if use_vq:
            encoded = self.abs_V.encoder(obs)
            encoded, _, _, _ = self.vq(encoded)
            if detach_encoder:
                encoded = encoded.detach()
            abs_v = self.abs_V.critic(encoded)
            with torch.no_grad():
                n_encoded = self.abs_V.encoder(n_obs)
                n_encoded, _, _, _ = self.vq(n_encoded)
                n_abs_v = self.abs_V_target.critic(n_encoded)
        else:
            abs_v = self.abs_V(obs, detach_encoder=detach_encoder)
            with torch.no_grad():
                n_abs_v = self.abs_V_target(n_obs)
        abs_v_target = rew + gamma * n_abs_v
        # mask = (quantized != n_quantized).any(dim=1).unsqueeze(1)
        # mask = mask | (rew != 0)
        # mask = mask.float()
        # abs_v_target = abs_v_target * mask + abs_v * (1 - mask)
        # criterion = F.smooth_l1_loss
        criterion = F.mse_loss
        # criterion = nn.SmoothL1Loss()
        abs_td_error = criterion(abs_v, abs_v_target)

        self.abs_V_optimizer.zero_grad(set_to_none=True)
        abs_td_error.backward()
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.abs_V.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)

        self.abs_V_optimizer.step()
        metric = {
            "Training_Info/update_absV/abstract_V_error": abs_td_error.item(),
            "Training_Info/update_absV/abs_v": abs_v.mean().item(),
        }
        self.L.log(metric)

        return abs_td_error

    def update_absV_immediate(
        self,
        abs_indices: np.ndarray,
        abs_indices_next: np.ndarray,
        reward: Tensor,
        gamma: Tensor,
    ):
        if hasattr(self, "lr_scheduler_abstract_V"):
            self.lr_abs_V = self.lr_scheduler_abstract_V(self._current_progress_remaining)
        reward = reward.squeeze().float().tolist()
        gamma = gamma.squeeze().float().tolist()
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
                target = reward[i] + gamma * self.abs_V_array[abs_idx_next]
                # target = reward[i] + self.gamma ** info[i]["interval4SemiMDP"] * abs_value_next_l[
                #     i
                # ] * (1 - terminated[i])
                # target = reward[i] + self.gamma ** info[i]["interval4SemiMDP"] * abs_value_next_l[i]
                delta = target - self.abs_V_array[abs_idx]
                # if delta <= 0:
                #     delta_l.append(0)
                # else:
                self.abs_V_array[abs_idx] += self.lr_abs_V * delta
                delta_l.append(delta)

        return mean(delta_l)

    def update_tcurl_vq_nstep(self):
        anc_obs, pos_obs, neg_obs = self.memory.sample_windowed_transitions()
        # anchor
        anc_grd_q, anc_encoded = self.ground_Q(anc_obs)
        anc_quantized, anc_vq_loss, anc_entrophy_vq, _ = self.vq(anc_encoded)
        # positive sample
        pos_grd_q, pos_encoded = self.ground_Q(pos_obs)
        pos_quantized, pos_vq_loss, pos_entrophy_vq, _ = self.vq(pos_encoded)
        # negative sample
        neg_grd_q, neg_encoded = self.ground_Q(neg_obs)
        neg_quantized, neg_vq_loss, neg_entrophy_vq, _ = self.vq(neg_encoded)

        pos_loss = F.mse_loss(anc_quantized, pos_quantized)
        neg_loss = F.mse_loss(anc_quantized, neg_quantized)

        total_loss = pos_loss + neg_loss

        total_loss.backward()

        self.ground_Q_optimizer.zero_grad()
        self.vq_optimizer.step()

        total_loss.backward()

        self.ground_Q_optimizer.step()
        self.vq_optimizer.step()

    def simclr_loss(self, anc: Tensor, pos: Tensor, temperature: float = 1) -> Tensor:
        batch_size = anc.shape[0]
        out = torch.cat([anc, pos], dim=0)
        out = F.normalize(out, dim=1)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (
            torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)
        ).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(anc * pos, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        return loss

    def simclr_loss2(self, anc: Tensor, pos: Tensor, temperature: float = 0.07):
        batch_size = anc.shape[0]
        out = torch.cat([anc, pos], dim=0)
        # out = F.normalize(out, dim=1)
        similarity_matrix = torch.matmul(out, out.T) / temperature

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # loss = F.cross_entropy(logits, labels)

        return logits, labels

    def curl_loss(self, anc: Tensor, pos: Tensor, temperature: float = 0.07):
        # anc = torch.cat([anc, pos], dim=0)
        # pos = torch.cat([pos, anc], dim=0)
        # anc = F.normalize(anc, dim=1)
        # pos = F.normalize(pos, dim=1)
        # logits = torch.matmul(anc, pos.T)
        # logits = self.curl.compute_logits(anc, pos)
        logits = self.curl.compute_logits_bilinear(anc, pos)
        logits = logits / temperature
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        # loss = F.cross_entropy(logits, labels)

        return logits, labels

    def push_away(self, x: Tensor):
        labels = torch.arange(x.shape[0]).long().to(self.device)
        loss = F.cross_entropy(torch.matmul(x, x.T), labels)
        return loss

    def update_contrastive(self, anc_obs, pos_obs):
        anc_encoded = self.curl.encoder(anc_obs)
        # anc_encoded = F.normalize(anc_encoded, dim=1)
        anc, anc_vq_loss, anc_entrophy_vq, anc_output_dict = self.vq(anc_encoded)
        # anc = F.normalize(anc, dim=1)

        # positive sample
        # with torch.no_grad():
        pos_encoded = self.curl.encoder(pos_obs)
        # pos_encoded = F.normalize(pos_encoded, dim=1)
        pos, pos_vq_loss, pos_entrophy_vq, pos_output_dict = self.vq(pos_encoded)
        # pos = F.normalize(pos, dim=1)

        vq_entropy = anc_entrophy_vq + pos_entrophy_vq

        # Normalize the codebook first
        # codebook = F.normalize(self.vq.embedding.weight, dim=1)
        # or not
        codebook = self.vq.embedding.weight

        cb_diversity = torch.matmul(codebook, codebook.T)
        # Using mean() only makes sense when the codebook is of non-negative vectors
        # cb_diversity = cb_diversity.mean()
        # cb_diversity = torch.einsum("ij,mj->im", [codebook, codebook]).mean()

        # Or like below, cb_diversity=(W*W_T - I), like below:
        # I = torch.eye(cb_diversity.shape[0])
        # I = I.float().to(self.device)
        # cb_diversity = torch.sum(torch.abs((cb_diversity - I)))
        # cb_diversity = torch.linalg.norm((cb_diversity - I))

        # Or a less constrained loss, cb_diversity=(W*W_T*(1-I)),like below:
        # mask1 = torch.ones(cb_diversity.shape) - torch.eye(cb_diversity.shape[0])
        # mask1 = mask1.float().to(self.device)
        # cb_diversity = torch.sum(torch.abs((cb_diversity * mask1)))
        # cb_diversity = torch.linalg.norm(cb_diversity * mask1)

        # Another way to do the loss above:
        mask1 = ~torch.eye(codebook.shape[0], dtype=torch.bool, device=self.device)
        cb_diversity = cb_diversity[mask1].mean()
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
        # logits, labels = self.curl_loss(anc, pos, temperature=1)
        logits, labels = self.simclr_loss2(anc, pos, temperature=1)
        loss1 = F.cross_entropy(logits, labels)
        with torch.no_grad():
            correct = torch.argmax(logits, dim=1) == labels
            contrast_acc = torch.mean(correct.float())
        # loss2 = self.push_away(pos)

        total_loss = (
            loss1 * 1.0
            + cb_diversity * 0.5
            - vq_entropy * 0.25
            + neg_diversity * 0.0
            + anc_vq_loss * 0.0
            + pos_vq_loss * 0.0
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
            "Training_Info/update_contrastive_vq/contrastive_loss": loss1.item(),
            "Training_Info/update_contrastive_vq/vq_loss": 0.5 * (anc_vq_loss + pos_vq_loss).item(),
            "Training_Info/update_contrastive_vq/entrophy_vq": 0.5
            * (anc_entrophy_vq + pos_entrophy_vq).item(),
            "Training_Info/update_contrastive_vq/update_contrastive total_loss": total_loss.item(),
            "Training_Info/update_contrastive_vq/contrast_acc": contrast_acc.item(),
            "Training_Info/update_contrastive_vq/cluster_metrics": cluster_metrics,
            "Training_Info/update_contrastive_vq/vq_codebook_diversity": cb_diversity.item(),
            "Training_Info/update_contrastive_vq/neg_diversity": neg_diversity.item(),
        }
        self.L.log(metric)

    def update_contrastive_novq(self, anc_obs, pos_obs):
        # # [data augmentation]
        # if self.input_format == "full_img":
        #     anc_obs = self.aug(anc_obs)
        #     pos_obs = self.aug(pos_obs)

        anc = self.curl.encoder(anc_obs)
        # anc = F.normalize(anc, dim=1)
        with torch.no_grad():
            pos = self.curl.encoder_target(pos_obs)
            # pos = F.normalize(pos, dim=1)

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
        logits, labels = self.curl_loss(anc, pos, temperature=1)
        # logits, labels = self.simclr_loss2(anc, pos, temperature=1)
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
            "Training_Info/update_contrastive/contrastive_loss": loss1.item(),
            "Training_Info/update_contrastive/update_contrastive total_loss": total_loss.item(),
            "Training_Info/update_contrastive/contrast_acc": contrast_acc.item(),
            "Training_Info/update_contrastive/neg_diversity": neg_diversity.item(),
        }
        self.L.log(metric)

        return total_loss

    def update_contrastive_grdEncoder(self, anc_obs, pos_obs):
        anc_encoded = self.ground_Q.encoder(anc_obs)
        # anc_encoded = F.normalize(anc_encoded, dim=1)
        anc, anc_vq_loss, anc_entrophy_vq, anc_output_dict = self.vq(anc_encoded)

        # positive sample
        # with torch.no_grad():
        pos_encoded = self.ground_Q.encoder(pos_obs)
        # pos_encoded = F.normalize(pos_encoded, dim=1)
        pos, pos_vq_loss, pos_entrophy_vq, pos_output_dict = self.vq(pos_encoded)

        vq_entropy = anc_entrophy_vq + pos_entrophy_vq
        codebook = F.normalize(self.vq.embedding.weight, dim=1)
        codebook_diversity = torch.einsum("ij,mj->im", [codebook, codebook]).mean()

        # anc = anc_output_dict["cluster_assignment"]
        # pos = pos_output_dict["cluster_assignment"]
        # anc = anc / (anc.norm(dim=1, keepdim=True)+1e-8)
        # pos = pos / (pos.norm(dim=1, keepdim=True)+1e-8)
        # anc = anc / anc.norm(dim=1, keepdim=True)
        # pos = pos / pos.norm(dim=1, keepdim=True)
        # logits, labels = self.curl_loss(anc, pos, temperature=1)
        logits, labels = self.simclr_loss2(anc, pos, temperature=1)
        loss1 = F.cross_entropy(logits, labels)
        with torch.no_grad():
            correct = torch.argmax(logits, dim=1) == labels
            contrast_acc = torch.mean(correct.float())
        # loss2 = self.push_away(pos)

        total_loss = loss1 + codebook_diversity * 1.0 - vq_entropy * 0.2
        # total_loss = loss1 - vq_entropy * 0.5
        # total_loss = loss1 + loss2 - anc_entrophy_vq * 0.5 - pos_entrophy_vq * 0.5

        self.ground_Q_optimizer.zero_grad()
        self.vq_optimizer.zero_grad()
        total_loss.backward()

        self.ground_Q_optimizer.step()
        self.vq_optimizer.step()

        cluster_metrics = anc_output_dict["cluster_metric"]

        # self.training_info["contrastive_loss"].append(loss1.item())
        # self.training_info["vq_loss"].append(anc_vq_loss.item())
        # self.training_info["entrophy_vq"].append(anc_entrophy_vq.item())
        # self.training_info["total_loss"].append(total_loss.item())
        # self.training_info["contrast_acc"].append(contrast_acc.item())
        # self.training_info["cluster_metrics"].append(cluster_metrics)

    def update_curl_vq2(self, anc_obs, pos_obs):
        anc_grd_q, anc_encoded = self.ground_Q(anc_obs)
        anc_quantized, anc_vq_loss, anc_entrophy_vq, _ = self.vq(anc_encoded)

        # positive sample
        with torch.no_grad():
            pos_grd_q, pos_encoded = self.ground_Q(pos_obs)
            pos_quantized, pos_vq_loss, pos_entrophy_vq, _ = self.vq(pos_encoded)

        dist = torch.matmul(anc_quantized, pos_quantized.T)
        labels = torch.arange(dist.shape[0]).long().to(self.device)
        loss = F.cross_entropy(dist, labels)
        total_loss = loss - anc_entrophy_vq

        self.ground_Q_optimizer.zero_grad()
        self.vq_optimizer.zero_grad()
        total_loss.backward()

        self.ground_Q_optimizer.step()
        self.vq_optimizer.step()

        # self.training_info["vq_loss"].append(anc_vq_loss.item())
        # self.training_info["entrophy_vq"].append(anc_entrophy_vq.item())
        # self.training_info["total_loss"].append(total_loss.item())

    def update_grdQ_pure(self, state, action, next_state, reward, gamma):

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

            # [Vanilla DQN]
            grd_q_next, encoded_next = self.ground_Q_target(next_state)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)

            # [Double DQN]
            # action_argmax_target = self.ground_Q_target(next_state)[0].argmax(dim=1, keepdim=True)
            # grd_q_next_max = self.ground_Q(next_state)[0].gather(1, action_argmax_target)

            # Compute ground target Q value
            grd_q_target = reward + gamma * grd_q_next_max

        criterion = nn.SmoothL1Loss()
        # criterion = F.mse_loss
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

        self.L.log({"ground_Q_error", ground_td_error.item()})

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
        self.vq.embedding.weight = torch.as_tensor(self.kmeans.cluster_centers_).to(self.device)

    def update_hp(self):
        if hasattr(self, "lr_scheduler_curl"):
            self.lr_curl = self.lr_scheduler_curl(self._current_progress_remaining)
            update_learning_rate(self.curl_optimizer, self.lr_curl)

        if hasattr(self, "lr_scheduler_vq"):
            self.lr_vq = self.lr_scheduler_vq(self._current_progress_remaining)
            update_learning_rate(self.vq_optimizer, self.lr_vq)

        if hasattr(self, "lr_scheduler_ground_Q"):
            self.lr_grd_Q = self.lr_scheduler_ground_Q(self._current_progress_remaining)
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_grd_Q,
            )

        if hasattr(self, "lr_scheduler_abstract_V"):
            self.lr_abs_V = self.lr_scheduler_abstract_V(self._current_progress_remaining)
            update_learning_rate(self.abs_V_optimizer, self.lr_abs_V)

        metric = {
            "HP/lr_ground_Q": self.lr_grd_Q,
            "HP/lr_abstract_V": self.lr_abs_V,
            "HP/lr_vq": self.lr_vq,
            "HP/lr_curl": self.lr_curl,
            "HP/safe_ratio": self.safe_ratio,
            "HP/exploration_rate": self.exploration_rate,
            "HP/current_progress_remaining": self._current_progress_remaining,
        }
        self.L.log(metric)

    def update(self, use_shaping: bool):
        """
        update with adaptive abstraction
        """
        self.update_hp()
        if hasattr(self, "safe_ratio_scheduler"):
            self.safe_ratio = self.safe_ratio_scheduler(self._current_progress_remaining)

        if self.timesteps_done < self.init_steps:
            return
        if self.timesteps_done == self.init_steps:
            # self.init_vq_codebook()
            # self.vis_abstraction()
            print("Warm up done")
            # if use_shaping:
            # for _ in tqdm(range(self.curl_vq_gradient_steps * 30)):
            # [1-step]
            # obs, act, n_obs, rew, gamma, info = self.memory.sample()
            # self.update_contrastive(obs, n_obs)
            # self.update_contrastive_novq(n_obs, obs)
            # [n-step]
            # N_Step_T: List[dict] = self.memory.sample_n_step_transits(
            #     n_step=3, batch_size=self.batch_size
            # )
            # obs = []
            # n_obs = []
            # for transit in N_Step_T:
            #     if transit:
            #         obs.extend(list(transit["obs"].values()))
            #         n_obs.extend(list(transit["n_obs"].values()))
            # if len(obs) > 0:
            #     obs = torch.as_tensor(np.array(obs)).to(self.device)
            #     n_obs = torch.as_tensor(np.array(n_obs)).to(self.device)
            #     # self.update_contrastive(obs, n_obs)
            #     self.update_contrastive_novq(n_obs, obs)
        # if self.timesteps_done == self.init_steps + 1:
        #     for _ in range(3):
        #         # self.cache_goal_transition()
        #         pass

        steps = self.timesteps_done - self.init_steps
        if use_shaping:
            if steps % self.abstract_learn_every == 0 or steps % self.ground_learn_every == 0:
                obs, act, n_obs, rew, gamma, info = self.memory.sample(self.batch_size)
                # sample n-step transitions
                # while True:
                #     N_Step_T: List[dict] = self.memory.sample_n_step_transits(
                #         n_step=5, batch_size=self.batch_size
                #     )
                #     if N_Step_T[-1] != None:
                #         break
                # obs, act, n_obs, rew, gamma, info = [], [], [], [], [], []
                # for transit in N_Step_T:
                #     if transit:
                #         obs.extend(list(transit["obs"].values()))
                #         act.extend(list(transit["act"].values()))
                #         n_obs.extend(list(transit["n_obs"].values()))
                #         rew.extend(list(transit["rew"].values()))
                #         gamma.extend(list(transit["gamma"].values()))
                # if len(obs) > 0:
                #     obs = torch.as_tensor(np.array(obs)).to(self.device)
                #     n_obs = torch.as_tensor(np.array(n_obs)).to(self.device)
                #     act = torch.as_tensor(act).unsqueeze(1).to(self.device)
                #     rew = torch.as_tensor(rew).unsqueeze(1).to(self.device)
                #     gamma = torch.as_tensor(gamma).unsqueeze(1).to(self.device)

                # [data augmentation]
                if self.input_format == "full_img":
                    obs = self.aug(obs)
                    n_obs = self.aug(n_obs)
            if steps % self.abstract_learn_every == 0:
                abs_loss = self.update_absV(
                    obs, n_obs, rew, gamma, use_vq=True, detach_encoder=True
                )
                pass
            if steps % self.ground_learn_every == 0:
                # self.update_grdQ(obs, act, n_obs, rew, gamma, use_shaping=True, approach_abs=False)
                grd_loss = self.update_grdQ(
                    obs, act, n_obs, rew, gamma, use_vq=True, use_shaping=False, approach_abs=False
                )
                # self.update_grdQ_critic(
                #     obs, act, n_obs, rew, gamma, use_vq=False, approach_abs=False
                # )
                pass
            if steps % self.curl_vq_learn_every == 0:
                for _ in range(self.curl_vq_gradient_steps):
                    pass
                    # [1-step]
                    # obs, act, n_obs, rew, gamma, info = self.memory.sample(self.batch_size_repre)
                    # self.update_contrastive(obs, n_obs)
                    # ct_loss = self.update_contrastive_novq(n_obs, obs)
                    # [n-step]
                    N_Step_T: List[dict] = self.memory.sample_n_step_transits(
                        n_step=3, batch_size=self.batch_size
                    )
                    obs = []
                    n_obs = []
                    for transit in N_Step_T:
                        if transit:
                            obs.extend(list(transit["obs"].values()))
                            n_obs.extend(list(transit["n_obs"].values()))
                    if len(obs) > 0:
                        obs = torch.as_tensor(np.array(obs)).to(self.device)
                        n_obs = torch.as_tensor(np.array(n_obs)).to(self.device)
                        self.update_contrastive(obs, n_obs)
                        # self.update_contrastive_novq(n_obs, obs)
            # if steps > 30000:
            # self.update_grdQ_critic(obs, act, n_obs, rew, gamma)
            # self.update_grdQ_absV(state, action, next_state, reward, gamma, info)
            # [update ground_Q with reward shaping]
            # total_loss = abs_loss + grd_loss + ct_loss
            # self.abs_V_optimizer.zero_grad(set_to_none=True)
            # self.ground_Q_optimizer.zero_grad(set_to_none=True)
            # self.curl_optimizer.zero_grad(set_to_none=True)
            # total_loss.backward()
            # if self.clip_grad:
            #     # 1 clamp gradients to avoid exploding gradient
            #     for param in self.abs_V.parameters():
            #         if param.grad is not None:
            #             param.grad.data.clamp_(-1, 1)
            #     for param in self.ground_Q.parameters():
            #         if param.grad is not None:  # make sure grad is not None
            #             param.grad.data.clamp_(-1, 1)
            #     for param in self.curl.parameters():
            #         if param.grad is not None:
            #             param.grad.data.clamp_(-1, 1)
            #     # 2 Clip gradient norm
            #     # max_grad_norm = 10
            #     # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            #     # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)

            # self.abs_V_optimizer.step()
            # self.ground_Q_optimizer.step()
            # self.curl_optimizer.step()
        else:
            # [purely update ground Q]
            if steps % self.ground_learn_every == 0:
                obs, act, n_obs, rew, gamma, info = self.memory.sample(self.batch_size)
                # anc = torch.concat([obs, n_obs], dim=0)
                # pos = torch.concat([n_obs, obs], dim=0)
                # anc = obs
                # pos = n_obs
                self.update_grdQ_pure(obs, act, n_obs, rew, gamma)
                # self.update_contrastive(anc, pos)

                # N_Step_T: List[dict] = self.memory.sample_n_step_transits(n_step=3)
                # obs = []
                # n_obs = []
                # for transit in N_Step_T:
                #     if transit:
                #         obs.extend(list(transit["obs"].values()))
                #         n_obs.extend(list(transit["n_obs"].values()))
                # if len(obs) > 0:
                #     anc = obs + n_obs
                #     pos = n_obs + obs
                #     # anc = obs
                #     # pos = n_obs
                #     anc = torch.as_tensor(np.array(anc)).to(self.device)
                #     pos = torch.as_tensor(np.array(pos)).to(self.device)
                #     # self.update_curl_vq(obs, n_obs)
                #     # self.update_curl_vq(n_obs, obs)
                #     self.update_contrastive(anc, pos)

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
                self.abs_V.encoder.parameters(),
                self.abs_V_target.encoder.parameters(),
                self.abs_V_encoder_tau,
            )
            soft_sync_params(
                self.abs_V.critic.parameters(),
                self.abs_V_target.critic.parameters(),
                self.abs_V_critic_tau,
            )

        # if steps % self.curl_vq_sync_every == 0:
        #     soft_sync_params(
        #         self.curl.encoder.parameters(),
        #         self.curl.encoder_target.parameters(),
        #         self.curl_tau,
        #     )
        self.L.dump2wandb(agent=self)
