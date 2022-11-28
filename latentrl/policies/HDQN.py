import copy
import random
import io

# from this import d
import time
from collections import Counter, deque, namedtuple
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from PIL import Image
import gym
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from torch import Tensor, nn

from policies.utils import ReplayMemory, ReplayMemoryWithCluster
from common.utils import (
    Dataset_pretrain,
    get_linear_fn,
    linear_schedule,
    polyak_sync,
    soft_sync_params,
    update_learning_rate,
    wandb_log_image,
)
from minigrid import Wall


class HDQN(nn.Module):
    def __init__(self, config, env: gym.Env) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = int(time.time())
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        self.set_hparams(config)
        self.env = env
        self.n_actions = env.action_space.n
        # self.n_actions = env.action_space.n
        # self.learn_with_ae = config.learn_with_ae
        # self.init_clustering = config.init_clustering

        # self.kmeans = Batch_KMeans(
        #     n_clusters=config.n_clusters, embedding_dim=config.embedding_dim, device=self.device
        # ).to(self.device)

        # self.aug = RandomShiftsAug(pad=4)

        self.outputs = dict()
        # self.apply(weight_init)

        # Initialize experience replay buffer
        # self.memory = ReplayMemory(self.size_replay_memory, self.device)
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
        self._current_progress_remaining = 1.0
        self.to_buffer = False  # for func maybe_buffer_recent_states

        # self._create_optimizers(config)
        # self.reset_training_info()
        # self.train()

    def train(self, training=True):
        raise NotImplementedError

    def reset_training_info(self):
        raise NotImplementedError

    def log_training_info(self, wandb_log=True):
        if wandb_log:
            metrics = {
                "loss/ground_Q_error": mean(self.training_info["ground_Q_error"]),
                "loss/abstract_V_error": mean(self.training_info["abstract_V_error"]),
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
        # print(cstart)

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

    def visualize_clusters_minigrid(self):
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

    def visualize_clusters_minigrid2(self):
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
        raise NotImplementedError

    def _create_optimizers(self, config):
        raise NotImplementedError

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

        # self.memory.push(state, action, next_state, reward, terminated, info)
        pass

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
        raise NotImplementedError

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

    def update_abs_Table(self):
        raise NotImplementedError

    # @torch.no_grad()
    # def encode_state(self, state):
    #     assert isinstance(state, np.ndarray)
    #     state_ = state.transpose(2, 0, 1)[np.newaxis, ...]
    #     state_ = torch.from_numpy(state_).contiguous().float().to(self.device)
    #     state_encoded = self.random_encoder(state_).squeeze().cpu()
    #     return tuple(state_encoded.tolist())

    # def assign_abs_state(self, state_encoded):
    #     if len(state_encoded) > 2:
    #         diff = self.abs_centroids - state_encoded
    #         dist = np.linalg.norm(np.absolute(diff), axis=-1)
    #         return np.argmin(dist)
    #     elif len(state_encoded) == 2:
    #         # use Hanming distance
    #         diff = self.abs_centroids - state_encoded
    #         dist = np.sum(np.absolute(diff), axis=-1)
    #         return np.argmin(dist)

    # def assign_abs_state_matrixwise(self, X: np.ndarray):
    #     distances = X[:, np.newaxis, :] - self.abs_centroids
    #     distances = np.sum(np.absolute(distances, out=distances), axis=-1)
    #     return np.argmin(distances, axis=-1)

    # def vis_abstraction(self, manual_abs: bool = False, prefix: str = None):
    #     width = self.env.width
    #     height = self.env.height

    #     clustersN = np.full(shape=(height, width), fill_value=-1)
    #     clustersS = np.full(shape=(height, width), fill_value=-1)
    #     # clustersS = np.random.randint(0, 4, size=(height, width))
    #     clustersW = np.full(shape=(height, width), fill_value=-1)
    #     clustersE = np.full(shape=(height, width), fill_value=-1)

    #     if self.cluster_embedding_dim > 2:
    #         center_coords = np.zeros((self.n_clusters, 2))

    #     for w in range(width):
    #         # w += 1
    #         for h in range(height):
    #             # h += 1
    #             if not isinstance(self.env.grid.get(w, h), Wall):
    #                 if self.cluster_embedding_dim > 2:

    #                     for dir in range(4):
    #                         if manual_abs:
    #                             encoded = (w, h)
    #                         elif self.input_format == "partial_obs":
    #                             env_ = copy.deepcopy(self.env)
    #                             env_.agent_pos = (w, h)
    #                             env_.agent_dir = dir
    #                             grid, vis_mask = env_.gen_obs_grid(agent_view_size=7)
    #                             state = grid.encode(vis_mask)
    #                             state = state.transpose(0, 2, 1)
    #                             encoded = self.encode_state(state)
    #                         elif self.input_format == "full_img":
    #                             state = self.env.unwrapped.grid.copy().render(
    #                                 tile_size=self.env.tile_size, agent_pos=(w, h), agent_dir=dir
    #                             )
    #                             encoded = self.encode_state(state)
    #                         elif self.input_format == "full_obs":
    #                             state = self.env.unwrapped.grid.copy().encode()
    #                             state[w][h] = np.array([10, 0, dir])
    #                             state = state.transpose(0, 2, 1)
    #                             encoded = self.encode_state(state)
    #                         abstract_state_idx = self.assign_abs_state(encoded)
    #                         if dir == 3:
    #                             clustersN[h, w] = abstract_state_idx
    #                         if dir == 0:
    #                             clustersE[h, w] = abstract_state_idx
    #                         if dir == 1:
    #                             clustersS[h, w] = abstract_state_idx
    #                         if dir == 2:
    #                             clustersW[h, w] = abstract_state_idx
    #                 elif self.cluster_embedding_dim == 2:
    #                     abstract_state_idx = self.assign_abs_state((w, h))
    #                     clustersN[h, w] = abstract_state_idx
    #                     clustersE[h, w] = abstract_state_idx
    #                     clustersS[h, w] = abstract_state_idx
    #                     clustersW[h, w] = abstract_state_idx

    #     values = [clustersN, clustersE, clustersS, clustersW]
    #     triangulations = self.triangulation_for_triheatmap(width, height)

    #     # [Plot Abstraction]
    #     fig_abs, ax_abs = plt.subplots(figsize=(5, 5))
    #     vmax = self.n_clusters
    #     vmin = 0
    #     my_cmap = copy.copy(plt.cm.get_cmap("gist_ncar"))
    #     my_cmap.set_under(color="dimgray")
    #     imgs = [
    #         ax_abs.tripcolor(
    #             t,
    #             np.ravel(val),
    #             vmin=vmin,
    #             vmax=vmax,
    #             cmap=my_cmap,
    #             ec="black",
    #         )
    #         for t, val in zip(triangulations, values)
    #     ]

    #     # xx, yy = np.meshgrid(self.abs_txt_ticks, self.abs_txt_ticks)
    #     # xx = xx.flatten()
    #     # yy = yy.flatten()
    #     if self.cluster_embedding_dim > 2:
    #         pass
    #     elif self.cluster_embedding_dim == 2:
    #         for i, (x, y) in enumerate(self.abs_centroids):
    #             ax_abs.text(
    #                 x,
    #                 y,
    #                 str(i),
    #                 # horizontalalignment="center",
    #                 # verticalalignment="center",
    #                 fontsize=9,
    #                 color="k",
    #                 fontweight="semibold",
    #                 # fontweight="normal",
    #                 bbox=dict(
    #                     boxstyle="round,pad=0.08, rounding_size=0.2",
    #                     fc=(1.0, 0.8, 0.8),
    #                     ec="k",
    #                     lw=1.5,
    #                 ),
    #             )

    #     ax_abs.invert_yaxis()
    #     fig_abs.tight_layout()

    #     img_buffer = io.BytesIO()
    #     fig_abs.savefig(
    #         img_buffer,
    #         dpi=100,
    #         # facecolor="w",
    #         # edgecolor="w",
    #         # orientation="portrait",
    #         # transparent=False,
    #         # bbox_inches=None,
    #         # pad_inches=0.1,
    #         format="png",
    #     )
    #     img = Image.open(img_buffer)
    #     wandb.log({"Images/abstraction": wandb.Image(img)})
    #     img_buffer.close()
    #     plt.close(fig_abs)
