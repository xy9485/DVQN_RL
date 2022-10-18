import copy
from pickle import FALSE
import random
from pprint import pp
import time
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

# from unicodedata import decimal

import gym
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch import Tensor, nn
from torchsummary import summary
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation


import wandb
from nn_models import (
    DQN,
    DQN_Repara,
    V_MLP,
    Decoder,
    Decoder_MiniGrid,
    Encoder_MiniGrid,
    RandomShiftsAug,
)
from policies.utils import ReplayMemory, ReplayMemoryWithCluster
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


class HDQN_KMeans_VAE(nn.Module):
    def __init__(
        self,
        config,
        env: gym.Env,
        # hidden_dims: List = [32, 64],
        beta: float = 0.25,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = int(time.time())
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.set_hparams(config)
        self.env = env
        self.n_actions = env.action_space.n
        self.learn_with_ae = config.learn_with_ae
        self.init_clustering = config.init_clustering

        self.kmeans = Batch_KMeans(
            n_clusters=config.n_clusters, embedding_dim=config.embedding_dim, device=self.device
        ).to(self.device)

        self.ground_Q = DQN_Repara(
            env.observation_space,
            env.action_space,
            config.embedding_dim,
            hidden_dims=config.hidden_dims,
            # embedding_dim=config.latent_dim,
        ).to(self.device)

        self.ground_Q_target = DQN_Repara(
            env.observation_space,
            env.action_space,
            config.embedding_dim,
            hidden_dims=config.hidden_dims,
            # embedding_dim=config.latent_dim,
        ).to(self.device)
        self.ground_Q_target.load_state_dict(self.ground_Q.state_dict())
        self.ground_Q_target.train()

        self.decoder = Decoder(
            in_dim=config.embedding_dim,
            out_channels=env.observation_space.shape[-1],
            shape_conv_output=self.ground_Q.conv_block.shape_conv_output,
            hidden_dims=config.hidden_dims,
        ).to(self.device)

        self.abstract_V = V_MLP(config.embedding_dim, flatten=False).to(self.device)
        self.abstract_V_target = V_MLP(config.embedding_dim, flatten=False).to(self.device)
        self.abstract_V_target.load_state_dict(self.abstract_V.state_dict())
        self.abstract_V_target.train()

        self.aug = RandomShiftsAug(pad=4)

        # for coding test
        # summary(self.ground_Q.encoder, (4, 84, 84))
        # summary(self.decoder, (32,))

        self.outputs = dict()
        # self.apply(weight_init)

        # Initialize experience replay buffer
        self.memory = ReplayMemory(self.size_replay_memory, self.device)
        # self.Transition = namedtuple(
        #     "Transition", ("state", "action", "next_state", "reward", "done")
        # )
        self.exploration_scheduler = get_linear_fn(
            config.exploration_initial_eps,
            config.exploration_final_eps,
            config.exploration_fraction,
        )

        self.timesteps_done = 0
        self.episodes_done = 0
        self.n_call_train = 0
        self._current_progress_remaining = 1.0
        self.to_buffer = False  # for func maybe_buffer_recent_states

        self._create_optimizers(config)
        self.reset_training_info()
        self.train()

    def train(self, training=True):
        self.training = training
        self.ground_Q.train(training)
        self.abstract_V.train(training)
        self.decoder.train(training)

    def reset_training_info(self):
        self.training_info = {
            "ground_Q_error": [],
            "abstract_V_error": [],
            "kld": [],
            "recon_loss": [],
            "commitment_loss": [],
        }

    def log_training_info(self, wandb_log=True):
        if wandb_log:
            metrics = {
                "loss/ground_Q_error": mean(self.training_info["ground_Q_error"]),
                "loss/abstract_V_error": mean(self.training_info["abstract_V_error"]),
                "loss/kld_loss": mean(self.training_info["kld"]),
                "loss/recon_loss": mean(self.training_info["recon_loss"]),
                "loss/vae_loss": mean(self.training_info["kld"])
                + mean(self.training_info["recon_loss"]),
                "loss/commitment_loss": mean(self.training_info["commitment_loss"]),
                "train/exploration_rate": self.exploration_rate,
                "train/current_progress_remaining": self._current_progress_remaining,
                "lr/lr_ground_Q_optimizer": self.ground_Q_optimizer.param_groups[0]["lr"],
                "lr/lr_abstract_V_optimizer": self.abstract_V_optimizer.param_groups[0]["lr"],
            }
            wandb.log(metrics)

    def load_states_from_memory(self, unique=True):
        transitions = random.sample(self.memory.memory, len(self.memory))
        batch = self.memory.Transition(*zip(*transitions))
        state_batch = np.stack(batch.state, axis=0).transpose(0, 3, 1, 2)
        if unique:
            state_batch = np.unique(state_batch, axis=0)
        state_batch = torch.from_numpy(state_batch).contiguous().float().to(self.device)

        # Use when states are cached as Tensor
        # batch = self.memory.sample(batch_size=len(self.memory))
        # state_batch = torch.cat(batch.next_state)
        # state_batch = torch.unique(state_batch, dim=0).float().to(self.device)

        return state_batch

    def triangulation_for_triheatmap(self, M, N):
        # M: number of columns, N: number of rows
        xv, yv = np.meshgrid(
            np.arange(-0.5, M), np.arange(-0.5, N)
        )  # vertices of the little squares
        xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
        x = np.concatenate([xv.ravel(), xc.ravel()])
        y = np.concatenate([yv.ravel(), yc.ravel()])
        cstart = (M + 1) * (N + 1)  # indices of the centers
        print(cstart)

        trianglesN = [
            (i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
            for j in range(N)
            for i in range(M)
        ]
        trianglesE = [
            (i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
            for j in range(N)
            for i in range(M)
        ]
        trianglesS = [
            (i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
            for j in range(N)
            for i in range(M)
        ]
        trianglesW = [
            (i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
            for j in range(N)
            for i in range(M)
        ]
        return [
            Triangulation(x, y, triangles)
            for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]
        ]

    def cluster_visualize_memory(self):
        # trainsitions_in_memory = self.memory.sample(batch_size=len(self.memory))
        # arrays = np.stack(trainsitions_in_memory.state, axis=0)
        # arrays_unique = np.unique(arrays, axis=0)
        # tensors_unique = (
        #     torch.from_numpy(arrays_unique.transpose(0, 3, 1, 2))
        #     .contiguous()
        #     .float()
        #     .to(self.device)
        # )

        batch = self.memory.sample(batch_size=len(self.memory))
        state_batch = torch.cat(batch.state)
        state_batch = state_batch.cpu().numpy()
        unique_array = np.unique(state_batch, axis=0)
        tensors_unique = torch.from_numpy(unique_array).float().to(self.device)

        with torch.no_grad():
            embeddings = self.ground_Q.encoder(tensors_unique)[0]
            cluster_indices = self.kmeans.assign_clusters(embeddings)
        # states_in_memory = self.load_states_from_memory()
        # arrays = torch.unique(states_in_memory, dim=0)
        # arrays = arrays.cpu().numpy().transpose(0, 3, 1, 2)
        batch, channels, width, height = tensors_unique.shape
        list_of_agent_pos_dir = []
        # clustersN = np.empty(shape=(height, width))
        # clustersS = np.empty(shape=(height, width))
        # clustersW = np.empty(shape=(height, width))
        # clustersE = np.empty(shape=(height, width))
        clustersN = np.full(shape=(height, width), fill_value=4)
        clustersS = np.full(shape=(height, width), fill_value=4)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=4)
        clustersE = np.full(shape=(height, width), fill_value=4)
        for idx, array in enumerate(tensors_unique):
            # break
            for i in range(width):
                for j in range(height):
                    type_idx, color_idx, state = array[:, i, j]
                    if type_idx == 10:  # if type is agent
                        assert 0 <= state < 4
                        if state == 3:
                            clustersN[j, i] = 0
                        elif state == 2:
                            clustersW[j, i] = 1
                        elif state == 1:
                            clustersS[j, i] = 2
                        elif state == 0:
                            clustersE[j, i] = 3
                    # agent_pos = (i, j)
                    # agent_dir = state
                    # list_of_agent_pos_dir.append((agent_pos, agent_dir))
        values = [clustersN, clustersE, clustersS, clustersW]
        triangulations = self.triangulation_for_triheatmap(width, height)
        fig, ax = plt.subplots()
        vmax = 4
        vmin = 0
        imgs = [
            ax.tripcolor(
                t,
                np.ravel(val),
                vmin=vmin,
                vmax=vmax,
                cmap="gist_ncar",
                ec="black",
            )
            for t, val in zip(triangulations, values)
        ]
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    def cluster_visualize_memory2(self):
        # take out all unique states from replay buffer and visualize their clusters
        # This approach might not cover all the states in the environment

        batch = self.memory.sample(batch_size=len(self.memory))
        # ===When states are cached as channel-first tensors===
        state_batch = torch.cat(batch.next_state)
        state_batch = state_batch.cpu().numpy()
        unique_array, indices = np.unique(state_batch, return_index=True, axis=0)
        unique_info_list = [batch.info[i] for i in indices]
        unique_tensor = torch.from_numpy(unique_array).float().to(self.device)

        # ===When states are cached as numpy arrays===
        # state_batch = np.stack(batch.next_state, axis=0)
        # unique_array, indices = np.unique(state_batch, return_index=True, axis=0)
        # unique_info_list = [batch.info[i] for i in indices]
        # unique_tensor = (
        #     torch.from_numpy(unique_array.transpose(0, 3, 1, 2))
        #     .contiguous()
        #     .float()
        #     .to(self.device)
        # )

        with torch.no_grad():
            embeddings = self.ground_Q.encoder(unique_tensor)[0]
            cluster_indices = self.kmeans.assign_clusters(embeddings)
        # states_in_memory = self.load_states_from_memory()
        # arrays = torch.unique(states_in_memory, dim=0)
        # arrays = arrays.cpu().numpy().transpose(0, 3, 1, 2)
        width = self.env.width
        height = self.env.height
        num_cluster = self.kmeans.n_clusters
        # clustersN = np.empty(shape=(height, width))
        # clustersS = np.empty(shape=(height, width))
        # clustersW = np.empty(shape=(height, width))
        # clustersE = np.empty(shape=(height, width))
        clustersN = np.full(shape=(height, width), fill_value=num_cluster)
        clustersS = np.full(shape=(height, width), fill_value=num_cluster)
        # clustersS = np.random.randint(0, 4, size=(height, width))
        clustersW = np.full(shape=(height, width), fill_value=num_cluster)
        clustersE = np.full(shape=(height, width), fill_value=num_cluster)

        print(cluster_indices.shape, len(unique_info_list))
        n, w, s, e = 0, 0, 0, 0
        for cluster_idx, info in zip(cluster_indices, unique_info_list):
            agent_pos = info["agent_pos"]
            agent_dir = info["agent_dir"]
            assert 0 <= agent_dir < 4
            if agent_dir == 3:
                n += 1
                clustersN[agent_pos[1], agent_pos[0]] = cluster_idx
            if agent_dir == 2:
                w += 1
                clustersW[agent_pos[1], agent_pos[0]] = cluster_idx
            if agent_dir == 1:
                s += 1
                clustersS[agent_pos[1], agent_pos[0]] = cluster_idx
            if agent_dir == 0:
                e += 1
                clustersE[agent_pos[1], agent_pos[0]] = cluster_idx
        print(n, w, s, e)
        values = [clustersN, clustersE, clustersS, clustersW]
        triangulations = self.triangulation_for_triheatmap(width, height)
        fig, ax = plt.subplots()
        vmax = num_cluster
        vmin = 0
        imgs = [
            ax.tripcolor(t, np.ravel(val), vmin=vmin, vmax=vmax, cmap="gist_ncar", ec="black")
            for t, val in zip(triangulations, values)
        ]
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    # def cluster_visualize_memory3(self):
    #         # take out all unique states from replay buffer and visualize their clusters
    #         # This approach might not cover all the states in the environment
    #         trainsitions_in_memory = self.memory.sample(batch_size=len(self.memory))
    #         arrays = np.stack(trainsitions_in_memory.state, axis=0)
    #         infos = trainsitions_in_memory.info
    #         arrays_unique, indices = np.unique(arrays, return_index=True, axis=0)
    #         infos = infos[indices]
    #         tensors_unique = (
    #             torch.from_numpy(arrays_unique.transpose(0, 3, 1, 2))
    #             .contiguous()
    #             .float()
    #             .to(self.device)
    #         )
    #         with torch.no_grad():
    #             embeddings = self.ground_Q.encoder(tensors_unique)[0]
    #             cluster_indices = self.kmeans.assign_clusters(embeddings)
    #         # states_in_memory = self.load_states_from_memory()
    #         # arrays = torch.unique(states_in_memory, dim=0)
    #         # arrays = arrays.cpu().numpy().transpose(0, 3, 1, 2)
    #         width = self.env.width
    #         height = self.env.height
    #         # clustersN = np.empty(shape=(height, width))
    #         # clustersS = np.empty(shape=(height, width))
    #         # clustersW = np.empty(shape=(height, width))
    #         # clustersE = np.empty(shape=(height, width))
    #         clustersN = np.full(shape=(height, width), fill_value=4)
    #         clustersS = np.full(shape=(height, width), fill_value=4)
    #         # clustersS = np.random.randint(0, 4, size=(height, width))
    #         clustersW = np.full(shape=(height, width), fill_value=4)
    #         clustersE = np.full(shape=(height, width), fill_value=4)

    #         for i in range(width):
    #             for j in range(height):
    #                 pass

    #         for cluster_idx, info in zip(cluster_indices, infos):
    #             agent_pos = info["agent_pos"]
    #             agent_dir = info["agent_dir"]
    #             if agent_dir == 3:
    #                 clustersN[agent_pos[1], agent_pos[0]] = cluster_idx
    #             if agent_dir == 2:
    #                 clustersW[agent_pos[1], agent_pos[0]] = cluster_idx
    #             if agent_dir == 1:
    #                 clustersS[agent_pos[1], agent_pos[0]] = cluster_idx
    #             if agent_dir == 0:
    #                 clustersE[agent_pos[1], agent_pos[0]] = cluster_idx

    #         values = [clustersN, clustersE, clustersS, clustersW]
    #         triangulations = self.triangulation_for_triheatmap(width, height)
    #         fig, ax = plt.subplots()
    #         vmax = 4
    #         vmin = 0
    #         imgs = [
    #             ax.tripcolor(t, np.ravel(val), vmin=vmin, vmax=vmax, cmap="gist_ncar", ec="black")
    #             for t, val in zip(triangulations, values)
    #         ]
    #         ax.invert_yaxis()
    #         plt.tight_layout()
    #         plt.show()

    def set_hparams(self, config):
        # Hyperparameters
        # self.total_episodes = config.total_episodes
        self.total_timesteps = config.total_timesteps
        self.init_steps = config.init_steps  # min. experiences before training
        self.batch_size = config.batch_size
        self.size_replay_memory = config.size_replay_memory
        self.gamma = config.gamma
        self.omega = config.omega
        self.ground_tau = config.ground_tau
        self.encoder_tau = config.encoder_tau
        self.abstract_tau = config.abstract_tau

        self.ground_learn_every = config.ground_learn_every
        self.ground_sync_every = config.ground_sync_every
        self.ground_gradient_steps = config.ground_gradient_steps
        self.abstract_learn_every = config.abstract_learn_every
        self.abstract_sync_every = config.abstract_sync_every
        self.abstract_gradient_steps = config.abstract_gradient_steps

        self.validate_every = config.validate_every
        self.save_model_every = config.save_model_every
        self.reset_training_info_every = config.reset_training_info_every
        self.save_recon_every = config.save_recon_every
        self.buffer_recent_states_every = config.buffer_recent_states_every

        self.clip_grad = config.clip_grad

    def _create_optimizers(self, config):

        if isinstance(config.lr_ground_Q, str) and config.lr_ground_Q.startswith("lin"):
            self.lr_scheduler_ground_Q = linear_schedule(float(config.lr_ground_Q.split("_")[1]))
            lr_ground_Q = self.lr_scheduler_ground_Q(self._current_progress_remaining)

        elif isinstance(config.lr_ground_Q, float):
            lr_ground_Q = config.lr_ground_Q

        if isinstance(config.lr_abstract_V, str) and config.lr_abstract_V.startswith("lin"):
            self.lr_scheduler_abstract_V = linear_schedule(
                float(config.lr_abstract_V.split("_")[1])
            )
            lr_abstract_V = self.lr_scheduler_abstract_V(self._current_progress_remaining)

        elif isinstance(config.lr_abstract_V, float):
            lr_abstract_V = config.lr_abstract_V

        # self.ground_Q_optimizer = optim.RMSprop(
        #     self.ground_Q_net.parameters(), lr=lr_ground_Q, alpha=0.95, momentum=0, eps=0.01
        # )
        # self.abstract_V_optimizer = optim.RMSprop(
        #     self.ground_Q_net.parameters(), lr=lr_abstract_V, alpha=0.95, momentum=0.95, eps=0.01
        # )

        if hasattr(self, "ground_Q"):
            self.ground_Q_optimizer = optim.Adam(self.ground_Q.parameters(), lr=lr_ground_Q)
        if hasattr(self, "abstract_V"):
            self.abstract_V_optimizer = optim.Adam(self.abstract_V.parameters(), lr=lr_abstract_V)

        if hasattr(self, "decoder"):
            self.encoder_optimizer = optim.Adam(
                self.ground_Q.encoder.parameters(), lr=config.lr_encoder
            )
            self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=config.lr_decoder)

    def _update_current_progress_remaining(self, timesteps_done, total_timesteps):
        # self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)
        finished_time_steps_after_init = timesteps_done - self.init_steps
        if finished_time_steps_after_init < 0:
            self._current_progress_remaining = 1.0
        else:
            self._current_progress_remaining = (
                1.0 - finished_time_steps_after_init / total_timesteps
            )

    @torch.no_grad()
    def maybe_buffer_recent_states(self, state, buffer_length=30):
        if self.timesteps_done % self.buffer_recent_states_every == 0 and self.timesteps_done > 0:
            self.recent_states = []
            self.to_buffer = True
        if self.to_buffer:
            self.recent_states.append(state)
            if len(self.recent_states) == buffer_length:
                print("##Check how good abstraction is##")
                # convert self.recent_states to torch tensor
                self.recent_states = (
                    (torch.tensor(self.recent_states).permute(0, 3, 1, 2).contiguous())
                    .float()
                    .to(self.device)
                )
                grd_q, encoded = self.ground_Q(self.recent_states)
                quantized, vq_loss, vq_entrophy, encodings = self.vector_quantizer(encoded)
                abs_v = self.abstract_V(quantized)
                (clusters, inverse_indice, counts) = torch.unique(
                    quantized,
                    return_inverse=True,
                    return_counts=True,
                    dim=0,
                )
                print("number of clusters:\n", len(clusters))
                print("inverse_indice:\n", inverse_indice.tolist())
                print("counts:\n", counts.tolist())
                # log n_abstract_states by wandb
                wandb.log({"abstraction/n_clusters_in_buffer": len(clusters)})
                wandb.log({"abstraction/value_difference": torch.abs(abs_v - grd_q).mean().item()})
                self.to_buffer = False

    def cache(self, state, action, next_state, reward, terminated, info):
        """Add the experience to memory"""
        # if state_type == "rgb":
        #     state = T.ToTensor()(state).float().unsqueeze(0)
        #     next_state = T.ToTensor()(next_state).float().unsqueeze(0)
        # else:
        #     state = torch.from_numpy(state.transpose((2, 0, 1))).contiguous().float().unsqueeze(0)
        #     next_state = (
        #         torch.from_numpy(next_state.transpose((2, 0, 1))).contiguous().float().unsqueeze(0)
        #     )
        # if state_type == "img":
        #     state = state / 255.0
        #     next_state = next_state / 255.0
        # action = torch.tensor([action]).unsqueeze(0)
        # reward = torch.tensor([reward]).unsqueeze(0)
        # terminated = torch.tensor([terminated]).unsqueeze(0)

        self.memory.push(state, action, next_state, reward, terminated, copy.deepcopy(info))

    def act(self, state):
        self._update_current_progress_remaining(self.timesteps_done, self.total_timesteps)
        self.exploration_rate = self.exploration_scheduler(self._current_progress_remaining)
        with torch.no_grad():
            state = T.ToTensor()(state).float().unsqueeze(0).to(self.device)

            if random.random() > self.exploration_rate:
                action = self.ground_Q(state)[0].max(1)[1].item()
            else:
                action = random.randrange(self.n_actions)

        self.timesteps_done += 1
        return action

    def update(self):
        n_steps_update = 1

        if self.timesteps_done == self.init_steps:
            states_in_memory = self.load_states_from_memory(unique=False)
            # states_in_memory_copy = states_in_memory.clone().detach()
            if self.learn_with_ae:
                self.pretrain_ae(states_in_memory, batch_size=1, epochs=30)
            if self.init_clustering:
                embedding_in_memory = self.ground_Q.encoder(states_in_memory)[0]
                self.kmeans.init_cluster(embedding_in_memory)
                # self.cluster_visualize_memory2()
            n_steps_update = 10

            # for _ in range(int(self.init_steps / 100)):
        if (
            self.timesteps_done == self.init_steps
            or self.timesteps_done % self.ground_learn_every == 0
        ):
            for _ in range(n_steps_update):
                state, action, next_state, reward, terminated = self.memory.sample(self.batch_size)

                # [data augmentation]
                state = self.aug(state)
                next_state = self.aug(next_state)

                # [update ground_Q with reward shaping]
                # quantized, quantized_next = self.update_grdQ_shaping(
                #     state, action, next_state, reward, terminated, use_shaping=True
                # )
                # [update abstract_V]
                # self.update_absV(quantized, quantized_next, reward, terminated)

                # [update vae, centroids of kmeans, maybe commitment loss]
                # self.train_vae_kmeans(
                #     state=state,
                #     next_state=None,
                #     update_centriods=True,
                #     kld_beta_vae=3,
                #     optimize_commitment=False,  # Ture or False needs further experiments
                #     save_recon_every=None,
                # )
                # [Or, use this: all in one]
                self.update_grdQ_absV(
                    state, action, next_state, reward, terminated, update_absV=True
                )

        if self.timesteps_done % self.ground_sync_every == 0:
            soft_sync_params(
                self.ground_Q.encoder.parameters(),
                self.ground_Q_target.encoder.parameters(),
                self.encoder_tau,
            )
            soft_sync_params(
                self.ground_Q.Q_linear.parameters(),
                self.ground_Q_target.Q_linear.parameters(),
                self.ground_tau,
            )

        if self.timesteps_done % self.abstract_sync_every == 0:
            soft_sync_params(
                self.abstract_V.parameters(),
                self.abstract_V_target.parameters(),
                self.abstract_tau,
            )

        if self.timesteps_done % self.save_model_every == 0:
            pass

        if self.timesteps_done % self.reset_training_info_every == 0:
            self.log_training_info(wandb_log=True)
            self.reset_training_info()

    def vae_loss_function(self, recon_x, x, mu, logvar, beta=3):
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld, recon_loss, kld

    def train_ae_kmeans(
        self, state_batch, next_state_batch, train_ae=True, train_kmeans=True, save_recon_every=1000
    ):
        if next_state_batch is not None:
            state_batch = torch.cat((state_batch, next_state_batch), dim=0)
            # shuffle the batch by the first dimension
            state_batch = state_batch[torch.randperm(state_batch.size(0)), :]

        recon_loss = torch.tensor(0.0).to(self.device)
        commitment_loss = torch.tensor(0.0).to(self.device)

        if train_ae:
            # Optimize Classic AE
            encoded = self.ground_Q.encoder(state_batch)
            recon = self.decoder(encoded)
            recon_loss = F.mse_loss(recon, state_batch)
            # if self.decoder.n_forward_call % save_recon_every == 0:
            #     stacked = torch.cat((recon[:7, :1], state_batch[:1, :1]), dim=0)
            #     wandb_log_image(stacked)

        if train_kmeans:
            quantized, cluster_indices = self.kmeans.assign_centroid(encoded, update_centroid=True)
            commitment_loss = F.mse_loss(quantized, encoded)

        if train_ae or train_kmeans:
            self.encoder_optimizer.zero_grad(set_to_none=True)
        if train_ae:
            self.decoder_optimizer.zero_grad(set_to_none=True)

        if train_ae or train_kmeans:
            combined_loss = recon_loss + commitment_loss
            combined_loss.backward()

        if train_ae or train_kmeans:
            self.encoder_optimizer.step()
        if train_ae:
            self.decoder_optimizer.step()

        return recon_loss, commitment_loss

    def train_vae_kmeans(
        self,
        state,
        next_state=None,
        update_centriods=True,
        kld_beta_vae=3,
        optimize_commitment=False,
        save_recon_every=None,
    ):
        if not update_centriods:
            optimize_commitment = False

        if next_state is not None:
            state = torch.cat((state, next_state), dim=0)
            # shuffle the batch by the first dimension
            state = state[torch.randperm(state.size(0)), :]

        # Optimize VAE
        encoded, mu, std = self.ground_Q.encoder(state)
        recon = self.decoder(encoded)
        vae_loss, recon_loss, kld = self.vae_loss_function(recon, state, mu, std, kld_beta_vae)
        # Wether or not assign centroids
        quantized, cluster_indices = self.kmeans.assign_centroid(
            encoded, update_centroid=update_centriods
        )

        if optimize_commitment:
            commitment_loss = F.mse_loss(quantized, encoded)
        else:
            commitment_loss = torch.tensor(0.0).to(self.device)

        self.encoder_optimizer.zero_grad(set_to_none=True)
        self.decoder_optimizer.zero_grad(set_to_none=True)

        combined_loss = vae_loss + commitment_loss
        combined_loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if save_recon_every and self.decoder.n_forward_call % save_recon_every == 0:
            stacked = torch.cat((recon[:7, :1], state[:1, :1]), dim=0)
            wandb_log_image(stacked)

        self.training_info["recon_loss"].append(recon_loss.item())
        self.training_info["kld"].append(kld.item())
        self.training_info["commitment_loss"].append(commitment_loss.item())

        return recon_loss, kld, commitment_loss

    def update_ae(
        self,
        state_batch,
        next_state_batch,
        save_recon_every=None,
        update_kmeans_centriods=True,
    ):
        if next_state_batch is not None:
            state_batch = torch.cat((state_batch, next_state_batch), dim=0)
            # shuffle the batch by the first dimension
            state_batch = state_batch[torch.randperm(state_batch.size(0)), :]

        # Optimize AE
        encoded = self.ground_Q.encoder(state_batch)
        recon = self.decoder(encoded)
        recon_loss = F.mse_loss(recon, state_batch)

        self.encoder_optimizer.zero_grad(set_to_none=True)
        self.decoder_optimizer.zero_grad(set_to_none=True)
        recon_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if update_kmeans_centriods:
            quantized, cluster_indices = self.kmeans.assign_centroid(encoded, update_centroid=True)

        if save_recon_every and (self.decoder.n_forward_call % save_recon_every) == 0:
            stacked = torch.cat((recon[:7, :1], state_batch[:1, :1]), dim=0)
            wandb_log_image(stacked)

        self.training_info["recon_loss"].append(recon_loss.item())

        return recon_loss, encoded, quantized

    def update_vae(
        self,
        state_batch,
        next_state_batch=None,
        kld_beta_vae=3,
        save_recon_every=None,
        update_kmeans_centriods=True,
        optimize_commitment=False,
    ):
        if next_state_batch is not None:
            state_batch = torch.cat((state_batch, next_state_batch), dim=0)
            # shuffle the batch by the first dimension
            state_batch = state_batch[torch.randperm(state_batch.size(0)), :]

        # Optimize VAE
        encoded, mu, std = self.ground_Q.encoder(state_batch)
        recon = self.decoder(encoded)

        # [wether or not update centroids]
        quantized, cluster_indices = self.kmeans.assign_centroid(
            encoded, update_centroid=update_kmeans_centriods
        )

        vae_loss, recon_loss, kld = self.vae_loss_function(
            recon, state_batch, mu, std, kld_beta_vae
        )
        if optimize_commitment:
            commitment_loss = F.mse_loss(encoded, quantized)
        else:
            commitment_loss = torch.tensor(0.0).to(self.device)

        self.encoder_optimizer.zero_grad(set_to_none=True)
        self.decoder_optimizer.zero_grad(set_to_none=True)
        (vae_loss + commitment_loss).backward(retain_graph=True)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if save_recon_every and (self.decoder.n_forward_call % save_recon_every) == 0:
            stacked = torch.cat((recon[:7, :1], state_batch[:1, :1]), dim=0)
            wandb_log_image(stacked)

        self.training_info["recon_loss"].append(recon_loss.item())
        self.training_info["kld"].append(kld.item())
        self.training_info["commitment_loss"].append(commitment_loss.item())

        return recon_loss, kld, encoded, quantized

    def optimize_commitment(self, encoded, quantized):
        """
        Currently this function gets error, when "one of the variables needed for gradient computation has been modified by an inplace operation" occurs
        """
        commitment_loss = F.mse_loss(encoded, quantized)
        self.encoder_optimizer.zero_grad(set_to_none=True)
        commitment_loss.backward()
        self.encoder_optimizer.step()
        self.training_info["commitment_loss"].append(commitment_loss.item())

    def pretrain_ae(self, data, batch_size=128, epochs=30):
        dataset = Dataset_pretrain(data)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("========== Start AE pretraining ==========")
        list_recon_loss = []
        list_kld = []
        list_commitment_loss = []
        for e in range(epochs):
            for i, (state_batch) in enumerate(train_loader):
                # only update vae without kmeans involved
                recon_loss, kld, commitment_loss = self.train_vae_kmeans(
                    state=state_batch,
                    next_state=None,
                    update_centriods=False,
                    kld_beta_vae=3,
                    optimize_commitment=False,
                    save_recon_every=None,
                )
                list_recon_loss.append(recon_loss.item())
                list_kld.append(kld.item())

            if i % 5 == 0:
                print(
                    f"Pretrain_Epoch {e}/{epochs}, recon_loss: {mean(list_recon_loss)}, kld: {mean(list_kld)}"
                )
        print("========== End AE pretraining ==========")

    def update_grdQ_absV(self, state, action, next_state, reward, terminated, update_absV=True):
        """
        This function is the combination of update_grdQ and update_absV and update_vae
        """
        if hasattr(self, "lr_scheduler_ground_Q"):
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_scheduler_ground_Q(self._current_progress_remaining),
            )
        if hasattr(self, "lr_scheduler_abstract_V"):
            update_learning_rate(
                self.abstract_V_optimizer,
                self.lr_scheduler_abstract_V(self._current_progress_remaining),
            )

        # batch = self.memory.sample(batch_size=self.batch_size)
        # state_batch = np.stack(batch.state, axis=0).transpose(0, 3, 1, 2)
        # next_state_batch = np.stack(batch.next_state, axis=0).transpose(0, 3, 1, 2)
        # state_batch = torch.from_numpy(state_batch).contiguous().float().to(self.device)
        # next_state_batch = torch.from_numpy(next_state_batch).contiguous().float().to(self.device)

        # # batch = self.memory.lazy_sample(batch_size=self.batch_size)
        # # state_batch = torch.cat(batch.state).to(self.device)
        # # next_state_batch = torch.cat(batch.next_state).to(self.device)
        # # action_batch = torch.cat(batch.action).to(self.device)
        # # reward_batch = torch.cat(batch.reward).to(self.device)
        # # terminated_batch = torch.cat(batch.terminated).to(self.device)

        # action_batch = torch.tensor(batch.action).unsqueeze(0).to(self.device)
        # reward_batch = torch.tensor(batch.reward).unsqueeze(0).to(self.device)
        # terminated_batch = torch.tensor(batch.terminated).unsqueeze(0).to(self.device)

        # mask = torch.eq(state_batch, next_state_batch)
        # num_same_aggregation = 0
        # for sample_mask in mask:
        #     if torch.all(sample_mask):
        #         num_same_aggregation += 1
        # print("num_same_aggregation:", num_same_aggregation)

        # [Data augmentation]
        state = self.aug(state)
        next_state = self.aug(next_state)

        # [Update ground Q network]
        grd_q, encoded, mu, std = self.ground_Q(state)
        grd_q = grd_q.gather(1, action)

        with torch.no_grad():

            # Vanilla DQN
            grd_q_next, encoded_next, mu, std = self.ground_Q_target(next_state)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)

            # Double DQN
            # action_argmax_target = self.ground_target_Q_net(next_state_batch).argmax(
            #     dim=1, keepdim=True
            # )
            # ground_next_max_Q = self.ground_Q_net(next_state_batch).gather(1, action_argmax_target)

            # Compute ground target Q value
            # abs_v = self.abstract_V_target(quantized)
            # abs_v_next = self.abstract_V_target(quantized_next)
            if update_absV:
                quantized, cluster_indices = self.kmeans.assign_centroid(
                    encoded, update_centroid=False
                )

                quantized_next, cluster_indices = self.kmeans.assign_centroid(
                    encoded_next, update_centroid=False
                )

                abs_v = self.abstract_V(quantized)
                abs_v_next = self.abstract_V(quantized_next)
                # shaping = self.gamma * abs_v_next - abs_v
                shaping = abs_v_next - abs_v
            else:
                shaping = 0

            grd_q_target = (
                reward
                + self.omega * shaping * (1 - terminated.float())
                + (1 - terminated.float()) * self.gamma * grd_q_next_max
            ).float()

        criterion = nn.SmoothL1Loss()
        ground_td_error = criterion(grd_q, grd_q_target)

        if update_absV:
            # [Update abstract V network]
            # mask_ = ~torch.tensor(
            #     [
            #         [torch.equal(a_state, next_a_state)]
            #         for a_state, next_a_state in zip(quantized, quantized_next)
            #     ]
            # ).to(self.device)
            # abs_v *= mask_
            # abs_v_target *= mask_

            abs_v = self.abstract_V(quantized)
            abs_v_target = torch.zeros(abs_v.shape[0]).to(self.device)
            with torch.no_grad():
                abs_v_next = self.abstract_V_target(quantized_next)
                for i, (a_state, next_a_state) in enumerate(zip(quantized, quantized_next)):
                    if torch.equal(a_state, next_a_state) and reward[i] == 0:
                        abs_v_target[i] = abs_v_next[i]
                    else:
                        abs_v_target[i] = reward[i] / 10 + self.gamma * abs_v_next[i]
            abs_v_target = abs_v_target.unsqueeze(1)
            criterion = nn.SmoothL1Loss()
            abstract_td_error = criterion(abs_v, abs_v_target)

            # [Optimize RL network]
            rl_loss = abstract_td_error + ground_td_error
        else:
            rl_loss = ground_td_error

        self.ground_Q_optimizer.zero_grad(set_to_none=True)
        if update_absV:
            self.abstract_V_optimizer.zero_grad(set_to_none=True)
        rl_loss.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.ground_Q.parameters():
                param.grad.data.clamp_(-1, 1)
            if update_absV:
                for param in self.abstract_V.parameters():
                    param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.ground_Q_optimizer.step()
        if update_absV:
            self.abstract_V_optimizer.step()

        # [Update autoencoder and ema-kmeans]
        recon_loss, kld, commitment_loss = self.train_vae_kmeans(
            state,
            next_state=None,
            update_centriods=True,
            kld_beta_vae=3,
            optimize_commitment=False,
            save_recon_every=None,
        )

        self.training_info["ground_Q_error"].append(ground_td_error.item())
        if update_absV:
            self.training_info["abstract_V_error"].append(abstract_td_error.item())
        self.training_info["recon_loss"].append(recon_loss.item())
        self.training_info["kld"].append(kld.item())
        self.training_info["commitment_loss"].append(commitment_loss.item())

    def update_grdQ_shaping(self, state, action, next_state, reward, terminated, use_shaping: bool):
        if hasattr(self, "lr_scheduler_ground_Q"):
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_scheduler_ground_Q(self._current_progress_remaining),
            )

        # [Update ground Q network]
        grd_q, encoded, mu, std = self.ground_Q(state)
        grd_q = grd_q.gather(1, action)

        with torch.no_grad():

            # Vanilla DQN
            grd_q_next, encoded_next, mu, std = self.ground_Q_target(next_state)
            grd_q_next_max = grd_q_next.max(1)[0].unsqueeze(1)

            # Double DQN
            # action_argmax_target = self.ground_target_Q_net(next_state_batch).argmax(
            #     dim=1, keepdim=True
            # )
            # ground_next_max_Q = self.ground_Q_net(next_state_batch).gather(1, action_argmax_target)

            # Compute ground target Q value
            # abs_v = self.abstract_V_target(quantized)
            # abs_v_next = self.abstract_V_target(quantized_next)
            if use_shaping:
                quantized, cluster_indices = self.kmeans.assign_centroid(
                    encoded, update_centroid=False
                )

                quantized_next, cluster_indices = self.kmeans.assign_centroid(
                    encoded_next, update_centroid=False
                )

                abs_v = self.abstract_V(quantized)
                abs_v_next = self.abstract_V(quantized_next)
                # shaping = self.gamma * abs_v_next - abs_v
                shaping = abs_v_next - abs_v
            else:
                shaping = 0

            grd_q_target = (
                reward
                + self.omega * shaping * (1 - terminated.float())
                + (1 - terminated.float()) * self.gamma * grd_q_next_max
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

        self.training_info["ground_Q_error"].append(ground_td_error.item())

        return quantized, quantized_next

    def update_absV(self, quantized, quantized_next, reward, terminated):
        if hasattr(self, "lr_scheduler_abstract_V"):
            update_learning_rate(
                self.abstract_V_optimizer,
                self.lr_scheduler_abstract_V(self._current_progress_remaining),
            )
        # [Update abstract V network]
        # mask_ = ~torch.tensor(
        #     [
        #         [torch.equal(a_state, next_a_state)]
        #         for a_state, next_a_state in zip(quantized, quantized_next)
        #     ]
        # ).to(self.device)
        # abs_v *= mask_
        # abs_v_target *= mask_

        abs_v = self.abstract_V(quantized)
        abs_v_target = torch.zeros(abs_v.shape[0]).to(self.device)
        with torch.no_grad():
            abs_v_next = self.abstract_V_target(quantized_next)
            for i, (a_state, next_a_state) in enumerate(zip(quantized, quantized_next)):
                if torch.equal(a_state, next_a_state) and reward[i] == 0:
                    abs_v_target[i] = abs_v_next[i]
                else:
                    abs_v_target[i] = reward[i] / 10 + self.gamma * abs_v_next[i]
        abs_v_target = abs_v_target.unsqueeze(1)
        criterion = nn.SmoothL1Loss()
        abstract_td_error = criterion(abs_v, abs_v_target)

        self.abstract_V_optimizer.zero_grad(set_to_none=True)
        abstract_td_error.backward()
        # print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        # print("run backward")
        if self.clip_grad:
            # 1 clamp gradients to avoid exploding gradient
            for param in self.abstract_V.parameters():
                param.grad.data.clamp_(-1, 1)

            # 2 Clip gradient norm
            # max_grad_norm = 10
            # torch.nn.utils.clip_grad_norm_(self.policy_mlp_net.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.vqvae_model.parameters(), max_grad_norm)
        self.abstract_V_optimizer.step()

        # [Update autoencoder and ema-kmeans]
        # recon_loss, kld, commitment_loss = self.train_vae_kmeans(
        #     state_batch=state_batch,
        #     next_state_batch=next_state_batch,
        #     train_ae=True,
        #     update_kmeans=True,
        # )

        self.training_info["abstract_V_error"].append(abstract_td_error.item())

    def update_grdQ_pure(self, state, action, next_state, reward, terminated, shaping=True):
        if hasattr(self, "lr_scheduler_ground_Q"):
            update_learning_rate(
                self.ground_Q_optimizer,
                self.lr_scheduler_ground_Q(self._current_progress_remaining),
            )

        state = self.aug(state)
        next_state = self.aug(next_state)

        # [Update ground Q network]
        grd_q, encoded, mu, std = self.ground_Q(state)
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

            grd_q_target = (reward + (1 - terminated.float()) * self.gamma * grd_q_next_max).float()

        criterion = nn.SmoothL1Loss()
        ground_td_error = criterion(grd_q, grd_q_target)

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

        self.training_info["ground_Q_error"].append(ground_td_error.item())
