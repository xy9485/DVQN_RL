import copy
import io
from itertools import chain
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
    DotDict,
)
from envs import FourRoomsEnv
from matplotlib import pyplot as plt
from nn_models import (
    DQN,
    DVN,
    DuelDQN,
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
    CURL_ATC,
    MOCO,
    VectorQuantizerLinear,
    VectorQuantizerLinearSoft,
    VectorQuantizerLinearDiffable,
    VQSoftAttention,
)
from nn_models.encoder import make_encoder
from nn_models.CURL import simclr_loss, simclr_loss2, simclr_debiased_loss
from PIL import Image
from policies.HDQN import HDQN
from policies.utils import (
    ReplayMemory,
    ReplayMemoryWithCluster,
    ReplayBufferNStep,
    transition_np2torch,
)
from PER import Memory
from common.Logger import LoggerWandb, Logger
from RND import RND

from sklearn.cluster import KMeans
from sympy.solvers import solve
from torch import Tensor, nn
from torchsummary import summary
from torchvision.utils import save_image

from minigrid import Wall


class HDQN_TCURL_VQ(HDQN):
    def __init__(self, args, env, logger: LoggerWandb):
        super().__init__(args, env, logger)
        # self.set_hparams(config)
        if args.use_dueling:
            QNet = DuelDQN
        else:
            QNet = DQN
        self.ground_Q = QNet(
            # observation_space=env.observation_space,
            action_space=env.action_space,
            # encoder=EncoderMaker(input_format=config.input_format, agent=self).make(),
            encoder=make_encoder(
                input_format=args.input_format,
                observation_space=env.observation_space,
                # hidden_channels=args.grd_hidden_channels,
                linear_dims=args.grd_encoder_linear_dims,
            ),
            mlp_hidden_dims=args.grd_critic_dims,
            # noisy=args.use_noisynet,
        ).to(self.device)

        self.ground_Q_target = QNet(
            # observation_space=env.observation_space,
            action_space=env.action_space,
            # encoder=EncoderMaker(input_format=config.input_format, agent=self).make(),
            encoder=make_encoder(
                input_format=args.input_format,
                observation_space=env.observation_space,
                # hidden_channels=args.grd_hidden_channels,
                linear_dims=args.grd_encoder_linear_dims,
            ),
            mlp_hidden_dims=args.grd_critic_dims,
            # noisy=args.use_noisynet,
        ).to(self.device)
        self.ground_Q_target.load_state_dict(self.ground_Q.state_dict())
        self.ground_Q.train()
        self.ground_Q_target.train()

        if self.grd_mode == "cddqn":
            self.ground_Q2 = QNet(
                # observation_space=env.observation_space,
                action_space=env.action_space,
                # encoder=EncoderMaker(input_format=config.input_format, agent=self).make(),
                encoder=make_encoder(
                    input_format=args.input_format,
                    observation_space=env.observation_space,
                    # hidden_channels=args.grd_hidden_channels,
                    linear_dims=args.grd_encoder_linear_dims,
                ),
                mlp_hidden_dims=args.grd_critic_dims,
                noisy=args.use_noisynet,
            ).to(self.device)

            self.ground_Q2_target = QNet(
                # observation_space=env.observation_space,
                action_space=env.action_space,
                # encoder=EncoderMaker(input_format=config.input_format, agent=self).make(),
                encoder=make_encoder(
                    input_format=args.input_format,
                    observation_space=env.observation_space,
                    # hidden_channels=args.grd_hidden_channels,
                    linear_dims=args.grd_encoder_linear_dims,
                ),
                mlp_hidden_dims=args.grd_critic_dims,
                noisy=args.use_noisynet,
            ).to(self.device)
            self.ground_Q2_target.load_state_dict(self.ground_Q2.state_dict())
            self.ground_Q2.train()
            self.ground_Q_target.train()

        if args.use_abs_V:
            if args.share_encoder:
                self.abs_V = DVN(
                    encoder=self.ground_Q.encoder,
                    mlp_hidden_dims=args.abs_critic_dims,
                    noisy=args.use_noisynet,
                ).to(self.device)
            else:
                self.abs_V = DVN(
                    # encoder=EncoderMaker(input_format=config.input_format, agent=self).make(),
                    encoder=make_encoder(
                        input_format=args.input_format,
                        observation_space=env.observation_space,
                        # hidden_channels=args.grd_hidden_channels,
                        linear_dims=args.abs_encoder_linear_dims,
                    ),
                    mlp_hidden_dims=args.abs_critic_dims,
                    noisy=args.use_noisynet,
                ).to(self.device)

            self.abs_V_target = DVN(
                # encoder=EncoderMaker(input_format=config.input_format, agent=self).make(),
                encoder=make_encoder(
                    input_format=args.input_format,
                    observation_space=env.observation_space,
                    # hidden_channels=args.grd_hidden_channels,
                    linear_dims=self.abs_V.encoder.linear_dims,
                ),
                mlp_hidden_dims=self.abs_V.mlp_hidden_dims,
                noisy=args.use_noisynet,
            ).to(self.device)

            self.abs_V_target.load_state_dict(self.abs_V.state_dict())
            self.abs_V.train()
            self.abs_V_target.train()
            self.abs_encoder = self.abs_V.encoder

        if args.use_curl:
            if args.use_curl == "on_abs":
                curl_encoder = self.abs_V.encoder
            elif args.use_curl == "on_grd":
                curl_encoder = self.ground_Q.encoder
            if self.curl_pair == "atc":
                # assert self.abs_enc_detach == True
                # assert self.grd_enc_detach == True
                self.curl = CURL_ATC(
                    encoder=curl_encoder,
                    encoder_target=make_encoder(
                        input_format=args.input_format,
                        observation_space=env.observation_space,
                        # hidden_channels=args.grd_hidden_channels,
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
                        # hidden_channels=args.grd_hidden_channels,
                        linear_dims=self.curl.encoder.linear_dims,
                    ),
                    projection_hidden_dims=self.curl.projection_hidden_dims,
                ).to(self.device)

                self.curl_ema.load_state_dict(self.curl.state_dict())
                for param in self.curl_ema.parameters():
                    param.requires_grad = False
                self.curl.train()
                self.curl_ema.train()

        if args.use_vq:
            self.vq = VectorQuantizerLinearSoft(
                num_embeddings=args.num_vq_embeddings,
                embedding_dim=self.curl.feature_dim,
                commitment_beta=0.25,
                softmin_beta=args.vq_softmin_beta,
            ).to(self.device)
            self.vq_ema = VectorQuantizerLinearSoft(
                num_embeddings=args.num_vq_embeddings,
                embedding_dim=self.curl.feature_dim,
                commitment_beta=0.25,
                softmin_beta=args.vq_softmin_beta,
            ).to(self.device)

            self.vq_ema.load_state_dict(self.vq.state_dict())
            for param in self.vq_ema.parameters():
                param.requires_grad = False
            self.vq.train()
            self.vq_ema.train()
            self.abs_V_array = np.zeros((args.num_vq_embeddings))

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
        if args.use_curiosity:
            self.rnd = RND(env, args, feature_dim=32)

        self.count_vis = 0
        self.goal_found = False

        self._create_optimizers(args)
        # self.train()

    def train(self, training=True):
        self.training = training
        self.ground_Q.train(training)
        self.abs_V.train(training)

    def set_hparams(self, args):
        # Hyperparameters
        self.input_format = args.input_format
        self.grd_mode = args.grd_mode
        self.grd_lower_bound = args.grd_lower_bound
        self.dan = args.dan
        self.use_curiosity = args.use_curiosity
        self.use_abs_V = args.use_abs_V
        self.use_curl = args.use_curl
        self.curl_pair = args.curl_pair
        self.use_vq = args.use_vq
        self.use_noisynet = args.use_noisynet
        self.curl_vq_cfg = args.curl_vq_cfg
        self.curl_enc_detach = args.curl_enc_detach
        self.abs_enc_detach = args.abs_enc_detach
        self.grd_enc_detach = args.grd_enc_detach
        self.critic_upon_vq = args.critic_upon_vq
        # self.total_episodes = config.total_episodes
        self.total_timesteps = int(args.total_timesteps)
        self.init_steps = int(args.init_steps)  # min. experiences before training
        self.batch_size = int(args.batch_size)
        self.batch_size_repre = self.batch_size
        self.size_replay_memory = int(args.size_replay_memory)
        self.gamma = args.gamma
        # self.abs_gamma = self.gamma
        self.omega = args.omega  # for reward shaping

        # self.grd_hidden_channels = args.grd_hidden_channels
        self.grd_encoder_linear_dims = args.grd_encoder_linear_dims
        self.mlp_hidden_dim_grd = args.grd_critic_dims
        self.ground_learn_every = args.freq_grd_learn
        # self.ground_gradient_steps = config.ground_gradient_steps
        self.ground_sync_every = args.freq_grd_sync
        self.ground_Q_encoder_tau = args.tau_grd_encoder
        self.ground_Q_critic_tau = args.tau_grd_critic

        if args.use_abs_V:
            self.abstract_learn_every = args.freq_abs_learn
            # self.abstract_gradient_steps = config.abstract_gradient_steps
            self.abstract_sync_every = args.freq_abs_sync
            self.abs_V_encoder_tau = args.tau_abs_encoder
            self.abs_V_critic_tau = args.tau_abs_critic

        if args.use_curl:
            self.curl_learn_every = args.freq_curl_learn
            self.curl_sync_every = args.freq_curl_sync
            self.curl_gradient_steps = 1
            self.curl_tau = args.tau_curl

        if args.use_vq:
            self.num_vq_embeddings = args.num_vq_embeddings
            self.use_vq = args.use_vq
            self.vq_tau = args.tau_vq

        self.clip_grad = args.clip_grad
        self.clip_reward = args.clip_reward
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.lr_decay = args.lr_decay
        self.lr_min = args.lr_min
        self.share_encoder = args.share_encoder
        self.per = args.per

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
        if terminated:
            gamma = 0.0
        else:
            gamma = self.gamma
        self.memory.push((state, action, next_state, reward, gamma, info))
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
        if mode == "soft_novq":
            return self.abs_V.critic(encoded).squeeze().item()
        if mode == "target_soft_novq":
            return self.abs_V_target.critic(encoded).squeeze().item()

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
        else:
            # here raise an error, saying invalid mode
            raise Exception("Invalid mode for non-vq abs value")
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

    def _create_optimizers(self, args):

        if args.lr_grd_Q.startswith("lin"):
            self.lr_scheduler_ground_Q = linear_schedule(float(args.lr_grd_Q.split("_")[1]))
            self.lr_grd_Q = self.lr_scheduler_ground_Q(self._current_progress_remaining)
        else:
            self.lr_grd_Q = float(args.lr_grd_Q)

        if args.use_abs_V:
            if args.lr_abs_V.startswith("lin"):
                self.lr_scheduler_abstract_V = linear_schedule(float(args.lr_abs_V.split("_")[1]))
                self.lr_abs_V = self.lr_scheduler_abstract_V(self._current_progress_remaining)
            else:
                self.lr_abs_V = float(args.lr_abs_V)

            if args.conservative_ratio.startswith("lin"):
                self.safe_ratio_scheduler = linear_schedule(
                    float(args.conservative_ratio.split("_")[1]), reduce=True
                )
                self.safe_ratio = self.safe_ratio_scheduler(self._current_progress_remaining)
            else:
                self.safe_ratio = float(args.conservative_ratio)

            if args.approach_abs_factor.startswith("lin"):
                self.close_factor_scheduler = linear_schedule(
                    float(args.approach_abs_factor.split("_")[1]), reduce=True
                )
                self.close_factor = self.close_factor_scheduler(self._current_progress_remaining)
            else:
                self.close_factor = float(args.approach_abs_factor)
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

        # self.ground_Q_optimizer = optim.RMSprop(
        #     self.ground_Q_net.parameters(), lr=lr_ground_Q, alpha=0.95, momentum=0, eps=0.01
        # )
        # self.abstract_V_optimizer = optim.RMSprop(
        #     self.ground_Q_net.parameters(), lr=lr_abstract_V, alpha=0.95, momentum=0.95, eps=0.01
        # )
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
        if args.use_abs_V:
            self.abs_V_optimizer = OPT(self.abs_V.parameters(), lr=self.lr_abs_V)
            if self.share_encoder:
                self.whole_optimizer = OPT(
                    chain(self.abs_V.critic.parameters(), self.ground_Q.parameters()),
                    lr=self.lr_grd_Q,
                )
            else:
                self.whole_optimizer = OPT(
                    chain(self.abs_V.parameters(), self.ground_Q.parameters()),
                    lr=self.lr_grd_Q,
                )

        self.ground_Q_optimizer = OPT(self.ground_Q.parameters(), lr=self.lr_grd_Q)
        # self.whole_optimizer = OPT(
        #     chain(self.abs_V.critic.parameters(), self.ground_Q.parameters()), lr=self.lr_grd_Q
        # )
        if self.grd_mode == "cddqn":
            self.ground_Q_optimizer2 = OPT(self.ground_Q2.parameters(), lr=self.lr_grd_Q)
        # elif args.optimizer == "sgd":
        #     OPT = optim.SGD
        #     if args.use_curl:
        #         self.curl_optimizer = OPT(
        #             self.curl.parameters(), lr=self.lr_curl, weight_decay=0.0, momentum=0.9
        #         )
        #     if args.use_vq:
        #         self.vq_optimizer = OPT(
        #             self.vq.parameters(), lr=self.lr_vq, weight_decay=0.0, momentum=0.9
        #         )
        #     if args.use_abs_V:
        #         self.abs_V_optimizer = OPT(self.abs_V.parameters(), lr=self.lr_abs_V)
        #     self.ground_Q_optimizer = OPT(self.ground_Q.parameters(), lr=self.lr_grd_Q)

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
        if self.use_noisynet:
            with torch.no_grad():
                state = torch.from_numpy(state[:]).unsqueeze(0).to(self.device)
                action = self.ground_Q(state)[0].argmax(dim=1).item()
                self.exploration_rate = 0.0
                return action
        # warm up phase
        if self.timesteps_done < self.init_steps:
            # action = self.env.action_space.sample()
            action = random.randrange(self.n_actions)
            # self.exploration_rate = 0.0
            return action

        self._update_current_progress_remaining(self.timesteps_done, self.total_timesteps)
        # [if linear decay]
        self.exploration_rate = self.exploration_scheduler(self._current_progress_remaining)
        # [if exponential decay]
        # self.exploration_rate = max(self.exploration_rate * self.epsilon_decay, self.epsilon_min)
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

    def act_e_greedy(self, state, epsilon=0.001):
        with torch.no_grad():
            state = torch.from_numpy(state[:]).unsqueeze(0).to(self.device)
            if random.random() > epsilon:
                action = self.ground_Q(state)[0].argmax(dim=1).item()
            else:
                action = random.randrange(self.n_actions)
            return action

    def update_grd_visits(self, info):
        agent_pos = info["agent_pos2"]
        agent_dir = info["agent_dir2"]
        self.grd_visits[agent_pos[1], agent_pos[0], agent_dir] += 1

    def update_grd_rewards(self, info, reward):
        agent_pos = info["agent_pos2"]
        agent_dir = info["agent_dir2"]
        # self.grd_reward_distribution[agent_pos[1], agent_pos[0], agent_dir] += info[
        #     "original_reward"
        # ]
        self.grd_reward_distribution[agent_pos[1], agent_pos[0], agent_dir] += reward

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
        weight=None,
        via_vq=False,
        detach_encoder=False,
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
        grd_q, _ = self.ground_Q(obs, detach_encoder)
        grd_q_reduction = torch.mean(grd_q, dim=1, keepdim=True)
        grd_q_std = torch.std(grd_q, dim=1, keepdim=True)
        # grd_q_reduction = torch.amax(grd_q.detach(), dim=1, keepdim=True)
        # grd_q_mean_bytarget = torch.mean(self.ground_Q_target(state)[0].detach(), dim=1, keepdim=True)
        grd_q = grd_q.gather(1, act)

        with torch.no_grad():
            if via_vq:
                encoded = self.abs_V.encoder(obs)
                encoded, _, _, _ = self.vq(encoded)
                abs_v = self.abs_V_target.critic(encoded)
                n_encoded = self.abs_V.encoder(n_obs)
                n_encoded, _, _, _ = self.vq(n_encoded)
                n_abs_v = self.abs_V_target.critic(n_encoded)
            else:
                abs_v = self.abs_V(obs)
                # abs_v = self.abs_V_target(obs)
                n_abs_v = self.abs_V(n_obs)
                # n_abs_v = self.abs_V_target(n_obs)

            # [Vanilla DQN]
            grd_q_next, encoded_next = self.ground_Q_target(n_obs)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)

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
        if self.per:
            criterion = nn.SmoothL1Loss(reduction="none")
        else:
            criterion = nn.SmoothL1Loss()
        # criterion = F.mse_loss
        # ground_td_error = F.mse_loss(grd_q, grd_q_target)
        # ground_td_error = criterion(grd_q, grd_q_target)
        # target_value = (
        #     self.safe_ratio * (rew + gamma * n_abs_v) + (1 - self.safe_ratio) * grd_q_target
        # )
        if random.random() < self.safe_ratio:
            target_value = rew + gamma * n_abs_v
        else:
            target_value = grd_q_target

        if self.dan:
            ground_td_error = criterion(grd_q + abs_v, rew + gamma * (grd_q_next_max + n_abs_v))
            td_error_per = grd_q + abs_v - rew - gamma * (grd_q_next_max + n_abs_v)
        else:
            ground_td_error = criterion(grd_q, target_value)
            td_error_per = grd_q - target_value
        if self.per:
            ground_td_error = ground_td_error * weight
            ground_td_error = ground_td_error.mean()
        # [learn advantage function]
        # ground_td_error = criterion(grd_q + abs_v, target_value)
        # abs_td_error = F.mse_loss(abs_v, abs_v_target)
        # abs_grd_diff = criterion(abs_v, grd_q)
        # commit_error_grd2abs = criterion(abs_v.detach(), grd_q_reduction)
        # commit_error_abs2grd = criterion(abs_v, grd_q_reduction.detach())
        # commit_error_abs2grd = F.mse_loss(abs_v, grd_q_target)

        # commit_error_grd2abs = torch.linalg.norm(F.relu(abs_v.detach() - grd_q_reduction))
        # commit_error_abs2grd = torch.linalg.norm(F.relu(grd_q_target - abs_v))

        # commit_error_grd2abs = F.relu(abs_v.detach() - grd_q_reduction).mean()
        # commit_error_abs2grd = F.relu(grd_q_target - abs_v).mean()

        # [Compute total loss]
        total_loss = ground_td_error
        if approach_abs:
            criterion = nn.SmoothL1Loss()
            # criterion = F.mse_loss
            if self.dan:
                grd_match_abs_err = criterion(grd_q + abs_v, rew + gamma * n_abs_v)
                # grd_match_abs_err = criterion(grd_q + abs_v, abs_v)
                # grd_match_abs_err = criterion(grd_q, torch.zeros_like(grd_q))
                # grd_match_abs_err = criterion(grd_q_reduction, torch.zeros_like(grd_q_reduction))
                # grd_match_abs_err = torch.linalg.norm(grd_q_reduction)
                # grd_match_abs_err += torch.linalg.norm(grd_q_std)
            else:
                # grd_match_abs_err = criterion(grd_q, abs_v)
                # grd_match_abs_err = criterion(grd_q_reduction, abs_v)
                grd_match_abs_err = criterion(grd_q, rew + gamma * n_abs_v)
            total_loss = (
                1 - self.close_factor
            ) * total_loss + self.close_factor * grd_match_abs_err

        # [gradient descent]
        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                if param.grad is not None:  # make sure grad is not None
                    param.grad.data.clamp_(-1, 1)
        #     # 2 Clip gradient norm
        #     # max_grad_norm = 10
        #     # torch.nn.utils.clip_grad_norm_(self.ground_Q.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()

        with torch.no_grad():
            diff_l2_abs_grd = F.mse_loss(abs_v, grd_q)
            # diff_l1_abs_grd = F.l1_loss(abs_v, grd_q_reduction)
            diff_l1_abs_grd = (abs_v - grd_q).mean()

        metric = {
            "Info/grdQ/ground_Q_error": ground_td_error.item(),
            "Info/grdQ/update_Q total_loss": total_loss.item(),
            "Info/grdQ/grd_q": grd_q.mean().item(),
            "Info/grdQ/grd_q_max": grd_q.max(1)[0].mean().item(),
            "Info/grdQ/grd_q_next_max": grd_q_next_max.mean().item(),
            "Info/grdQ/grd_q_std": grd_q_std.mean().item(),
            "Info/grdQ/grd_q_reduction": grd_q_reduction.mean().item(),
            "Info/grdQ/absV_grdQ_l1": diff_l1_abs_grd.item(),
            "Info/grdQ/absV_grdQ_l2": diff_l2_abs_grd.item(),
        }
        if self.dan:
            metric.update(
                {
                    "Info/grdQ/grd_q": (grd_q + abs_v).mean().item(),
                    "Info/grdQ/grd_q_max": (grd_q_next_max + n_abs_v).mean().item(),
                    "Info/grdQ/advantage": grd_q.mean().item(),
                }
            )
        if approach_abs:
            metric.update(
                {
                    "Info/grdQ/grd_match_abs_err": grd_match_abs_err.item(),
                }
            )
        self.L.log(metric)

        return total_loss, td_error_per

    def update_grdQ_ddqn(
        self,
        obs,
        act,
        n_obs,
        rew,
        gamma,
        via_vq,
        detach_encoder=False,
        use_shaping=False,
        approach_abs=False,
        lower_bound=False,
    ):
        if self.clip_reward:
            rew.clamp_(-1, 1)

        # [Update ground Q network]
        grd_q, _ = self.ground_Q(obs, detach_encoder)
        grd_q_reduction = torch.mean(grd_q, dim=1, keepdim=True)
        # grd_q_reduction = torch.amax(grd_q.detach(), dim=1, keepdim=True)
        # grd_q_mean_bytarget = torch.mean(self.ground_Q_target(state)[0].detach(), dim=1, keepdim=True)
        grd_q = grd_q.gather(1, act)

        with torch.no_grad():
            if via_vq:
                encoded = self.abs_V.encoder(obs)
                encoded, _, _, _ = self.vq(encoded)
                abs_v = self.abs_V_target.critic(encoded)
                n_encoded = self.abs_V.encoder(n_obs)
                n_encoded, _, _, _ = self.vq(n_encoded)
                n_abs_v = self.abs_V_target.critic(n_encoded)
            else:
                # abs_v = self.abs_V_target(obs)
                # n_abs_v = self.abs_V_target(n_obs)
                abs_v = self.abs_V(obs)
                n_abs_v = self.abs_V(n_obs)
            # [Double DQN]
            grd_q_next_max = self.ground_Q_target(n_obs)[0].gather(
                1, self.ground_Q(n_obs)[0].argmax(dim=1, keepdim=True)
            )
            if lower_bound:
                assert approach_abs == False
                assert self.safe_ratio == 0.0
                grd_q_next_max = torch.maximum(grd_q_next_max, n_abs_v)

            grd_q_target = rew + gamma * grd_q_next_max
            if use_shaping:
                # shaping = self.abs_V_target(quantized_next) - self.abs_V_target(quantized)
                # shaping = abs_v_next_hard - abs_v_hard
                shaping = n_abs_v - abs_v
                grd_q_target += self.omega * shaping
                self.L.log({"avg_shaping": shaping.mean().item()})
            # if self.safe_ratio > 0.0:
            #     if random.random() < self.safe_ratio:
            #         target_value = rew + gamma * n_abs_v
            #     else:
            #         target_value = grd_q_target
            target_value = grd_q_target

        criterion = nn.SmoothL1Loss()
        ground_td_error = criterion(grd_q, target_value)
        if self.dan:
            ground_td_error = criterion(grd_q + abs_v, rew + gamma * (grd_q_next_max + n_abs_v))

        # [Compute total loss]
        total_loss = ground_td_error
        if approach_abs:
            assert lower_bound == False
            criterion = nn.SmoothL1Loss()
            # criterion = F.mse_loss
            # grd_match_abs_err = criterion(grd_q_reduction, abs_v)
            grd_match_abs_err = criterion(grd_q, abs_v)
            # grd_match_abs_err = criterion(grd_q, rew + gamma * n_abs_v)
            if self.dan:
                grd_match_abs_err = criterion(grd_q + abs_v, rew + gamma * n_abs_v)
            # grd_match_abs_err = criterion(grd_q, rew + gamma * n_abs_v)
            # total_loss += self.close_factor * grd_match_abs_err
            total_loss = (
                1 - self.close_factor
            ) * total_loss + self.close_factor * grd_match_abs_err

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
        self.ground_Q_optimizer.step()

        with torch.no_grad():
            diff_l2_abs_grd = F.mse_loss(abs_v, grd_q)
            # diff_l1_abs_grd = F.l1_loss(abs_v, grd_q_reduction)
            diff_l1_abs_grd = (abs_v - grd_q).mean()

        metric = {
            "Info/grdQ/ground_Q_error": ground_td_error.item(),
            "Info/grdQ/update_Q total_loss": total_loss.item(),
            "Info/grdQ/grd_q": grd_q.mean().item(),
            "Info/grdQ/grd_q_max": grd_q.max(1)[0].mean().item(),
            "Info/grdQ/grd_q_next_max": grd_q_next_max.mean().item(),
            "Info/grdQ/grd_q_reduction": grd_q_reduction.mean().item(),
            "Info/grdQ/absV_grdQ_l1": diff_l1_abs_grd.item(),
            "Info/grdQ/absV_grdQ_l2": diff_l2_abs_grd.item(),
        }
        self.L.log(metric)

        return total_loss

    def update_grdQ_cddqn(
        self,
        obs,
        act,
        n_obs,
        rew,
        gamma,
        via_vq,
        detach_encoder=False,
        use_shaping=False,
        approach_abs=False,
        lower_bound=False,
    ):
        if self.clip_reward:
            rew.clamp_(-1, 1)

        # [Update ground Q network]
        grd_q, _ = self.ground_Q(obs, detach_encoder)
        grd_q = grd_q.gather(1, act)

        grd_q2, _ = self.ground_Q2(obs, detach_encoder)
        grd_q2 = grd_q2.gather(1, act)

        grd_q_reduction = torch.mean(grd_q, dim=1, keepdim=True)
        # grd_q_reduction = torch.amax(grd_q.detach(), dim=1, keepdim=True)

        with torch.no_grad():
            if via_vq:
                encoded = self.abs_V.encoder(obs)
                encoded, _, _, _ = self.vq(encoded)
                abs_v = self.abs_V_target.critic(encoded)
                n_encoded = self.abs_V.encoder(n_obs)
                n_encoded, _, _, _ = self.vq(n_encoded)
                n_abs_v = self.abs_V_target.critic(n_encoded)
            else:
                # abs_v = self.abs_V_target(obs)
                # n_abs_v = self.abs_V_target(n_obs)
                abs_v = self.abs_V(obs)
                n_abs_v = self.abs_V(n_obs)
            # [Clipped Double DQN]
            grd_q_next, encoded_next = self.ground_Q_target(n_obs)
            grd_q_next_max1 = grd_q_next.max(1)[0].unsqueeze(1)

            grd_q_next_max2 = self.ground_Q2_target(n_obs)[0].gather(
                1, self.ground_Q_target(n_obs)[0].argmax(dim=1, keepdim=True)
            )

            grd_q_next_max = torch.minimum(grd_q_next_max1, grd_q_next_max2)

            if lower_bound:
                assert approach_abs == False
                assert use_shaping == False
                assert self.safe_ratio == 0.0
                grd_q_next_max = torch.maximum(grd_q_next_max, n_abs_v)

            grd_q_target = rew + gamma * grd_q_next_max
            if use_shaping:
                # shaping = self.abs_V_target(quantized_next) - self.abs_V_target(quantized)
                # shaping = abs_v_next_hard - abs_v_hard
                shaping = n_abs_v - abs_v
                grd_q_target += self.omega * shaping
                self.L.log({"avg_shaping": shaping.mean().item()})

            # if self.safe_ratio > 0:
            #     if random.random() < self.safe_ratio:
            #         target_value = rew + gamma * n_abs_v
            #     else:
            #         target_value = grd_q_target
            target_value = grd_q_target

        criterion = nn.SmoothL1Loss()
        ground_td_error = criterion(grd_q, target_value)
        ground_td_error2 = criterion(grd_q2, target_value)
        if self.dan:
            ground_td_error = criterion(grd_q + abs_v, rew + gamma * (grd_q_next_max + n_abs_v))

        # [Compute total loss]
        total_loss = ground_td_error + ground_td_error2
        if approach_abs:
            assert lower_bound == False
            criterion = nn.SmoothL1Loss()
            # criterion = F.mse_loss
            # grd_match_abs_err = criterion(grd_q_reduction, abs_v)
            grd_match_abs_err = criterion(grd_q, abs_v)
            if self.dan:
                grd_match_abs_err = criterion(grd_q + abs_v, rew + gamma * n_abs_v)
            # grd_match_abs_err = criterion(grd_q, rew + gamma * n_abs_v)
            # total_loss += self.close_factor * grd_match_abs_err
            total_loss = (
                1 - self.close_factor
            ) * total_loss + self.close_factor * grd_match_abs_err

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        self.ground_Q_optimizer2.zero_grad(set_to_none=True)
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
                for param in self.ground_Q2.parameters():
                    if param.grad is not None:  # make sure grad is not None
                        param.grad.data.clamp_(-1, 1)

        self.ground_Q_optimizer.step()
        self.ground_Q_optimizer2.step()

        with torch.no_grad():
            diff_l2_abs_grd = F.mse_loss(abs_v, grd_q)
            # diff_l1_abs_grd = F.l1_loss(abs_v, grd_q_reduction)
            diff_l1_abs_grd = (abs_v - grd_q).mean()

        metric = {
            "Info/grdQ/ground_Q_error": ground_td_error.item(),
            "Info/grdQ/ground_Q_error2": ground_td_error2.item(),
            "Info/grdQ/update_Q total_loss": total_loss.item(),
            "Info/grdQ/grd_q": grd_q.mean().item(),
            # "Info/grdQ/grd_q2": grd_q2.mean().item(),
            "Info/grdQ/grd_q_max": grd_q.max(1)[0].mean().item(),
            # "Info/grdQ/grd_q2_max": grd_q2.max(1)[0].mean().item(),
            "Info/grdQ/grd_q_next_max": grd_q_next_max.mean().item(),
            # "Info/grdQ/grd_q_next_max2": grd_q_next_max2.mean().item(),
            "Info/grdQ/grd_q_reduction": grd_q_reduction.mean().item(),
            "Info/grdQ/absV_grdQ_l1": diff_l1_abs_grd.item(),
            "Info/grdQ/absV_grdQ_l2": diff_l2_abs_grd.item(),
        }
        self.L.log(metric)

        return total_loss

    def update_absV(
        self,
        obs,
        n_obs,
        rew,
        gamma,
        via_vq=False,
        detach_encoder=False,
    ):
        if self.clip_reward:
            rew.clamp_(-1, 1)
        # # [data augmentation]
        # if self.input_format == "full_img":
        #     obs = self.aug(obs)
        #     n_obs = self.aug(n_obs)
        if via_vq:
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
        # criterion = F.mse_loss
        criterion = nn.SmoothL1Loss()
        abs_td_error = criterion(abs_v, abs_v_target)
        # codebook_v = self.abs_V.critic(self.vq.embedding.weight.data)
        # with torch.no_grad():
        #     codebook_v_old = self.abs_V_target.critic(self.vq.embedding.weight.data)
        # stability_loss = criterion(codebook_v, codebook_v_old)

        # if via_vq:
        #     # let abs_v of hard quanitized approach its members' abs_v
        #     with torch.no_grad():
        #         encoded = self.abs_V.encoder(obs)
        #         encoded, _, _, output = self.vq(encoded)
        #     loss2 = criterion(self.abs_V.critic(output["hard_quantized_latents"]), abs_v_target)
        #     # abs_td_error += 0.01 * loss2
        # else:
        #     loss2 = torch.tensor(0.0)
        stable_factor = (
            0.1 * (self.timesteps_done - self.init_steps) / (self.total_timesteps - self.init_steps)
        )
        loss = abs_td_error
        self.abs_V_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.abs_V.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)

        #     # 2 Clip gradient norm
        #     # max_grad_norm = 10
        #     # torch.nn.utils.clip_grad_norm_(self.abs_V.parameters(), max_grad_norm)

        self.abs_V_optimizer.step()
        metric = {
            "Info/update_absV/abstract_V_error": abs_td_error.item(),
            "Info/update_absV/signed_absV_error": (abs_v - abs_v_target).mean().item(),
            "Info/update_absV/abs_v": abs_v.mean().item(),
            # "Info/update_absV/centroid_abs_V_error": loss2.item(),
            # "Info/update_absV/stability_loss": stability_loss.item(),
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

    def push_away(self, x: Tensor):
        labels = torch.arange(x.shape[0]).long().to(self.device)
        loss = F.cross_entropy(torch.matmul(x, x.T), labels)
        return loss

    @torch.no_grad()
    def vq_confidence(self, x):
        x = torch.as_tensor(np.array(x)).unsqueeze(0).to(self.device)
        x = self.curl(x)
        x, vq_loss, entrophy_vq, output_dict = self.vq(x)

        # encoding_inds = F.softmin(output_dict["dist"] / 0.1, dim=1)
        # avg_probs = torch.mean(encoding_inds, dim=0)
        # entrophy_vq = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))

        # return entrophy_vq as intrinsic reward in sense of confidence
        # return entrophy_vq * 0.1 * np.power(1 - 1e-5, self.timesteps_done - self.init_steps)
        return entrophy_vq * 0.1

    def update_contrastive(self, anc_obs, pos_obs, ema=False):
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

    def update_grdQ_pure(self, obs, act, n_obs, rew, gamma):

        if self.clip_reward:
            rew.clamp_(-1, 1)

        if self.grd_mode == "cddqn":
            grd_q, encoded = self.ground_Q(obs)
            grd_q = grd_q.gather(1, act)

            grd_q2, encoded2 = self.ground_Q2(obs)
            grd_q2 = grd_q2.gather(1, act)
            with torch.no_grad():
                grd_q_next, encoded_next = self.ground_Q_target(n_obs)
                grd_q_next_max1 = grd_q_next.max(1)[0].unsqueeze(1)

                grd_q_next_max2 = self.ground_Q2_target(n_obs)[0].gather(
                    1, self.ground_Q_target(n_obs)[0].argmax(dim=1, keepdim=True)
                )

                grd_q_next_max = torch.minimum(grd_q_next_max1, grd_q_next_max2)
                grd_q_target = rew + gamma * grd_q_next_max

            criterion = nn.SmoothL1Loss()
            ground_td_error = criterion(grd_q, grd_q_target)
            ground_td_error2 = criterion(grd_q2, grd_q_target)
            self.ground_Q_optimizer.zero_grad(set_to_none=True)
            self.ground_Q_optimizer2.zero_grad(set_to_none=True)
            (ground_td_error + ground_td_error2).backward()
            if self.clip_grad:
                # 1 clamp gradients to avoid exploding gradient
                for param in self.ground_Q.parameters():
                    if param.grad is not None:  # make sure grad is not None
                        param.grad.data.clamp_(-1, 1)
                for param in self.ground_Q2.parameters():
                    if param.grad is not None:  # make sure grad is not None
                        param.grad.data.clamp_(-1, 1)

                # 2 Clip gradient norm
                # max_grad_norm = 10
                # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
                # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
            self.ground_Q_optimizer.step()
            self.ground_Q_optimizer2.step()
            metric = {
                "Info/grdQ/ground_Q_error1": ground_td_error.item(),
                "Info/grdQ/ground_Q_error2": ground_td_error2.item(),
                "Info/grdQ/grd_q": grd_q.mean().item(),
                "Info/grdQ/grd_q2": grd_q2.mean().item(),
                "Info/grdQ/grd_q_max": grd_q_next_max.mean().item(),
                "Info/grdQ/grd_q_max2": grd_q_next_max2.mean().item(),
            }

            self.L.log(metric)
            return

        # [Update ground Q network]
        grd_q, encoded = self.ground_Q(obs)
        grd_q = grd_q.gather(1, act)

        with torch.no_grad():
            # [Vanilla DQN]
            if self.grd_mode == "dqn":
                grd_q_next, encoded_next = self.ground_Q_target(n_obs)
                grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)
            elif self.grd_mode == "ddqn":
                # [Double DQN]
                grd_q_next_max = self.ground_Q_target(n_obs)[0].gather(
                    1, self.ground_Q(n_obs)[0].argmax(dim=1, keepdim=True)
                )

            # Compute ground target Q value
            grd_q_target = rew + gamma * grd_q_next_max

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

        metric = {
            "Info/grdQ/ground_Q_error": ground_td_error.item(),
            "Info/grdQ/grd_q": grd_q.mean().item(),
            "Info/grdQ/grd_q_max": grd_q_next_max.mean().item(),
        }

        self.L.log(metric)

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

        if hasattr(self, "safe_ratio_scheduler"):
            self.safe_ratio = self.safe_ratio_scheduler(self._current_progress_remaining)

        # conservative learning and optimistic learning alternating
        # if ((self.timesteps_done - self.init_steps) // 10000) % 2 == 0:
        #     self.safe_ratio = 1.0
        # else:
        #     self.safe_ratio = 0.0

        if hasattr(self, "close_factor_scheduler"):
            self.close_factor = self.close_factor_scheduler(self._current_progress_remaining)

        self.L.log(
            {
                "HP/lr_ground_Q": self.lr_grd_Q,
                "HP/exploration_rate": self.exploration_rate,
                "HP/current_progress_remaining": self._current_progress_remaining,
            }
        )
        if self.use_abs_V:
            self.L.log(
                {
                    "HP/lr_abstract_V": self.lr_abs_V,
                    "HP/safe_ratio": self.safe_ratio,
                    "HP/close_factor": self.close_factor,
                }
            )
        if self.use_curl:
            self.L.log({"HP/lr_curl": self.lr_curl})
        if self.use_vq:
            self.L.log({"HP/lr_vq": self.lr_vq})

    def update(self):
        """
        update with adaptive abstraction
        """
        if self.use_noisynet:
            if self.timesteps_done * self.ground_learn_every == 0:
                self.ground_Q.reset_noise()
                self.ground_Q_target.reset_noise()
            if self.use_abs_V and self.timesteps_done * self.abstract_learn_every == 0:
                self.abs_V.reset_noise()
                self.abs_V_target.reset_noise()
        self.update_hp()

        if self.timesteps_done < self.init_steps:
            return
        if self.timesteps_done == self.init_steps:
            # self.init_vq_codebook()
            # self.vis_abstraction()
            print("Warm up done")
            # if use_shaping:
            for _ in tqdm(range(1000)):
                pass
                # [1-step]
                # obs, act, n_obs, rew, gamma, info = self.memory.sample(self.batch_size_repre)
                # # [data augmentation]
                # if self.input_format == "full_img":
                #     with torch.no_grad():
                #         obs = self.aug(obs)
                #         pos = self.aug(obs)
                # self.update_contrastive(obs, n_obs)
                # self.update_contrastive_novq(obs, n_obs, ema=True)
                # self.update_contrastive_novq(obs, pos, ema=True)
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
                #     # self.update_contrastive(obs, n_obs, ema=False)
                #     self.update_contrastive_novq(obs, n_obs, ema=False)
        # if self.timesteps_done == self.init_steps + 1:
        #     for _ in range(3):
        #         # self.cache_goal_transition()
        #         pass

        steps = self.timesteps_done - self.init_steps
        if self.use_abs_V:
            if steps % self.abstract_learn_every == 0 or steps % self.ground_learn_every == 0:
                if self.per:
                    batch, idxs, weight = self.memory.sample(self.batch_size, steps)
                    obs, act, n_obs, rew, gamma, info = zip(*batch)
                    obs = torch.as_tensor(np.array(obs)).to(self.device)
                    n_obs = torch.as_tensor(np.array(n_obs)).to(self.device)
                    act = torch.as_tensor(act).unsqueeze(1).to(self.device)
                    rew = torch.as_tensor(rew).unsqueeze(1).to(self.device)
                    gamma = torch.as_tensor(gamma).unsqueeze(1).to(self.device)
                    weight = torch.as_tensor(weight).unsqueeze(1).to(self.device)
                else:
                    obs, act, n_obs, rew, gamma, info = self.memory.sample(self.batch_size)
                    weight = None
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
                    with torch.no_grad():
                        obs = self.aug(obs)
                        n_obs = self.aug(n_obs)
                        if self.curl_pair == "raw":
                            pos = self.aug(obs)
                        else:
                            pos = n_obs
                if self.use_curiosity:
                    loss = self.rnd.update(obs)
                    # loss = 0
                    self.L.log(
                        {
                            "RND/rnd_loss": loss,
                            "RND/beta_t": self.rnd.beta_t,
                        }
                    )
            if steps % self.abstract_learn_every == 0:
                absV_loss = self.update_absV(
                    obs,
                    n_obs,
                    rew,
                    gamma,
                    via_vq=self.critic_upon_vq,
                    detach_encoder=self.abs_enc_detach,
                )
                pass
            if steps % self.ground_learn_every == 0:
                if self.grd_mode == "dqn":
                    grdQ_loss, td_error_per = self.update_grdQ(
                        obs,
                        act,
                        n_obs,
                        rew,
                        gamma,
                        weight=weight,
                        via_vq=self.critic_upon_vq,
                        detach_encoder=self.grd_enc_detach,
                        use_shaping=False,
                        approach_abs=self.close_factor > 0.0,
                    )
                    if self.per:
                        td_error_per = td_error_per.squeeze().detach().cpu().numpy()
                        for i in range(self.batch_size):
                            self.memory.update(idx=idxs[i], error=td_error_per[i])
                        self.L.log({"Info/grdQ/td_error_per": td_error_per.mean()})
                elif self.grd_mode == "ddqn":
                    grdQ_loss = self.update_grdQ_ddqn(
                        obs,
                        act,
                        n_obs,
                        rew,
                        gamma,
                        via_vq=self.critic_upon_vq,
                        detach_encoder=self.grd_enc_detach,
                        use_shaping=False,
                        approach_abs=self.close_factor > 0.0,
                        lower_bound=self.grd_lower_bound,
                    )
                elif self.grd_mode == "cddqn":
                    grdQ_loss = self.update_grdQ_cddqn(
                        obs,
                        act,
                        n_obs,
                        rew,
                        gamma,
                        via_vq=self.critic_upon_vq,
                        detach_encoder=self.grd_enc_detach,
                        use_shaping=False,
                        approach_abs=self.close_factor > 0.0,
                        lower_bound=self.grd_lower_bound,
                    )
            if self.use_curl and steps % self.curl_learn_every == 0:
                for _ in range(self.curl_gradient_steps):
                    pass
                    # [1-step]
                    # obs, act, n_obs, rew, gamma, info = self.memory.sample(self.batch_size_repre)
                    # [data augmentation]
                    # if self.input_format == "full_img":
                    # with torch.no_grad():
                    # obs = self.aug(obs)
                    # pos = self.aug(obs)
                    if self.use_vq:
                        self.update_contrastive(obs, pos, ema=True)
                    elif self.curl_pair == "atc":
                        self.update_contrastive_atc(obs, pos)
                    else:
                        self.update_contrastive_novq(obs, pos, ema=True)
                    # ct_loss = self.update_contrastive_novq(obs, pos, ema=True)
                    # [n-step]
                    # N_Step_T: List[dict] = self.memory.sample_n_step_transits(
                    #     n_step=3, batch_size=self.batch_size_repre
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
                    #     # self.update_contrastive(obs, n_obs, ema=False)
                    #     self.update_contrastive_novq(obs, n_obs, ema=False)
            # [update ground_Q with reward shaping]
            # total_loss = absV_loss + grdQ_loss
            # self.whole_optimizer.zero_grad()
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
            # for param in self.curl.parameters():
            # if param.grad is not None:
            # param.grad.data.clamp_(-1, 1)
            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
            # self.whole_optimizer.step()
            # self.abs_V_optimizer.step()
            # self.abs_V.encoder.zero_grad()
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
                if hasattr(self, "rnd"):
                    loss = self.rnd.update(obs)
                    self.L.log(
                        {
                            "RND/rnd_loss": loss,
                            "RND/beta_t": self.rnd.beta_t,
                        }
                    )
                with torch.no_grad():
                    obs = self.aug(obs)
                    n_obs = self.aug(n_obs)
                    if self.curl_pair == "raw":
                        pos = self.aug(obs)
                    else:
                        pos = n_obs
                self.update_grdQ_pure(obs, act, n_obs, rew, gamma)
                if self.use_curl == "on_grd":
                    if self.use_vq:
                        self.update_contrastive(obs, pos, ema=True)
                    else:
                        self.update_contrastive_novq(obs, pos, ema=True)
                # self.update_contrastive(anc, pos)

                # self.rnd.update(n_obs)

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
            soft_sync_params(
                self.ground_Q.parameters(),
                self.ground_Q_target.parameters(),
                self.ground_Q_encoder_tau,
            )
            if self.grd_mode == "cddqn":
                soft_sync_params(
                    self.ground_Q2.parameters(),
                    self.ground_Q2_target.parameters(),
                    self.ground_Q_encoder_tau,
                )
            # soft_sync_params(
            #     self.ground_Q.encoder.parameters(),
            #     self.ground_Q_target.encoder.parameters(),
            #     self.ground_Q_encoder_tau,
            # )

            # soft_sync_params(
            #     self.ground_Q.critic.parameters(),
            #     self.ground_Q_target.critic.parameters(),
            #     self.ground_Q_critic_tau,
            # )

        if self.use_abs_V and steps % self.abstract_sync_every == 0:
            soft_sync_params(
                self.abs_V.parameters(),
                self.abs_V_target.parameters(),
                self.abs_V_encoder_tau,
            )
            # soft_sync_params(
            #     self.abs_V.encoder.parameters(),
            #     self.abs_V_target.encoder.parameters(),
            #     self.abs_V_encoder_tau,
            # )
            # soft_sync_params(
            #     self.abs_V.critic.parameters(),
            #     self.abs_V_target.critic.parameters(),
            #     self.abs_V_critic_tau,
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

        if self.use_vq and steps % self.curl_sync_every == 0:
            soft_sync_params(
                self.vq.parameters(),
                self.vq_ema.parameters(),
                self.vq_tau,
            )
        self.L.dump2wandb(agent=self)
