import datetime
import math
import os
import random
import sys
import time
from collections import deque, namedtuple
from itertools import count
from os import makedirs
from pprint import pprint
from tkinter import N
from types import SimpleNamespace

# import tracemalloc

import colored_traceback.auto
import GPUtil
import gym
from gym.wrappers import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from cv2 import mean
from PIL import Image

# from tensorboard import summary
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

print("sys.path:", sys.path)
import wandb
from data2 import RolloutDatasetNaive
from policies.hrl_dqn_agent import DuoLayerAgent, SingelLayerAgent
from policies.vanilla_dqn_agent import VanillaDQNAgent
from transforms import transform_dict
from utils.learning import EarlyStopping, ReduceLROnPlateau
from utils.misc import get_linear_fn, linear_schedule, make_vec_env_customized, update_learning_rate
from wrappers import (
    ActionDiscreteWrapper,
    ActionRepetitionWrapper,
    CarRandomStartWrapper,
    ClipRewardEnvCustom,
    EncodeStackWrapper,
    EpisodeEarlyStopWrapper,
    EpisodicLifeEnvCustom,
    FrameStackCustom,
    FrameStackWrapper,
    PreprocessObservationWrapper,
    pack_env_wrappers,
)

from atari_wrapper import *


def make_env(
    env_id,
    config,
):
    # env = gym.make(env_id).unwrapped
    env = gym.make(env_id)
    # env = gym.make(
    #     env_id,
    #     # frameskip=(3, 7),
    #     # repeat_action_probability=0.25,
    #     full_action_space=False,
    #     # render_mode="human",
    # )

    # For atari, using gym wappers or third-party wappers
    # wrapper_class_list = [
    #     ClipRewardEnvCustom,
    #     EpisodicLifeEnvCustom,
    #     GrayScaleObservation,
    #     ResizeObservation,
    #     FrameStackCustom,
    # ]
    # wrapper_kwargs_list = [
    #     None,
    #     None,
    #     {"keep_dim": True},  # gym default wrapper
    #     {"shape": 84},  # gym default wrapper
    #     # {"num_stack": config.n_frame_stack},  # gym default wrapper
    #     {"k": config.n_frame_stack},  # custom wrapper
    # ]

    # For atari, but using custom wrapper
    # wrapper_class_list = [
    #     # ActionDiscreteWrapper,
    #     # ActionRepetitionWrapper,
    #     # EpisodeEarlyStopWrapper,
    #     # Monitor,
    #     # CarRandomStartWrapper,
    #     PreprocessObservationWrapper,
    #     # EncodeStackWrapper,
    #     # PunishRewardWrapper,
    #     FrameStackWrapper,
    # ]
    # wrapper_kwargs_list = [
    #     # {"action_repetition": config.action_repetition},
    #     # {"max_neg_rewards": max_neg_rewards, "punishment": punishment},
    #     # {'filename': monitor_dir},
    #     # {"filename": os.path.join(monitor_dir, "train")},  # just single env in this case
    #     # {
    #     #     "warm_up_steps": hparams["learning_starts"],
    #     #     "n_envs": n_envs,
    #     #     "always_random_start": always_random_start,
    #     #     "no_random_start": no_random_start,
    #     # },
    #     {
    #         "vertical_cut_d": 84,
    #         "shape": 84,
    #         "num_output_channels": 1,
    #     },
    #     # {
    #     #     "n_stack": n_stack,
    #     #     "vae_f": vae_path,
    #     #     "vae_sample": vae_sample,
    #     #     "vae_inchannel": vae_inchannel,
    #     #     "latent_dim": vae_latent_dim,
    #     # },
    #     # {'max_neg_rewards': max_neg_rewards, "punishment": punishment}
    #     {"n_frame_stack": config.n_frame_stack},
    # ]

    # For carracing-v0
    wrapper_class_list = [
        ActionDiscreteWrapper,
        # ActionRepetitionWrapper,
        EpisodeEarlyStopWrapper,
        # Monitor,
        # CarRandomStartWrapper,
        PreprocessObservationWrapper,
        # EncodeStackWrapper,
        # PunishRewardWrapper,
        FrameStackWrapper,
    ]
    wrapper_kwargs_list = [
        {"action_repetition": config.action_repetition},
        {"max_neg_rewards": 100, "punishment": -20.0},
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
            "num_output_channels": 1,
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


def train_single_layer():
    env_id = "CarRacing-v0"  # CarRacing-v0, ALE/Skiing-v5, Boxing-v0
    for _ in range(3):
        current_time = datetime.datetime.now() + datetime.timedelta(hours=2)
        current_time = current_time.strftime("%b%d_%H-%M-%S")
        # 🐝 initialise a wandb run
        wandb.init(
            project="vqvae+latent_rl",
            tags=[
                "vqvae+latentdqn",
                # "as_vanilla_dqn",
                "vqvae2(32-64-64)",
                "encoder&vq_attach",
            ],  # vqvae2(32-64-128), vqvae(128-256)
            config={
                "env_id": env_id,
                "total_timesteps": 1000_000,
                "init_steps": 10000,
                "action_repetition": 3,
                "n_frame_stack": 4,
                # "dropout": random.uniform(0.01, 0.80),
                # "vqvae_inchannel": int(3 * 1),
                "vqvae_latent_channel": 16,
                "vqvae_num_embeddings": 64,
                "reconstruction_path": os.path.join(
                    "/workspace/repos_dev/VQVAE_RL/reconstruction/singlelayer", env_id, current_time
                ),
                # "reconstruction_path": None,
                # "total_episodes": 1000,
                "lr_vqvae": 5e-4,
                "lr_dqn": "lin_2.5e-4",
                "batch_size": 256,
                "validation_size": 128,
                "validate_every": 10,
                "size_replay_memory": int(5e5),
                "gamma": 0.97,
                "exploration_fraction": 0.9,
                "exploration_initial_eps": 0.1,
                "exploration_final_eps": 0.01,
                # "eps_start": 0.1,
                # "eps_end": 0.05,
                # "eps_decay": 200,
                # "exploration_rate": 0.1,
                # "exploration_rate_decay": 0.99999975,
                # "exploration_rate_min": 0.1,
                "save_model_every": 5e5,
                "learn_every": 4,
                "sync_every": 8,
                "gradient_steps": 1,
                "seed": int(time.time()),
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            },
        )
        config = wandb.config
        print(type(config))

        # config = {
        #     "env_id": "Boxing-v0",
        #     "total_timesteps": 1000_000,
        #     "init_steps": 10000,
        #     "action_repetition": 2,
        #     "n_frame_stack": 2,
        #     # "dropout": random.uniform(0.01, 0.80),
        #     "vqvae_inchannel": int(3 * 1),
        #     "vqvae_latent_channel": 16,
        #     "vqvae_num_embeddings": 64,
        #     "reconstruction_path": os.path.join(
        #         "/workspace/repos_dev/VQVAE_RL/reconstruction/singlelayer", env_id, current_time
        #     ),
        #     # "reconstruction_path": None,
        #     # "total_episodes": 1000,
        #     "lr_vqvae": 3e-4,
        #     "lr_dqn": "lin_5.3e-4",
        #     "batch_size": 256,
        #     "validation_size": 128,
        #     "validate_every": 10,
        #     "size_replay_memory": int(1e6),
        #     "gamma": 0.97,
        #     "exploration_fraction": 0.5,
        #     "exploration_initial_eps": 1,
        #     "exploration_final_eps": 0.1,
        #     # "eps_start": 0.1,
        #     # "eps_end": 0.05,
        #     # "eps_decay": 200,
        #     # "exploration_rate": 0.1,
        #     # "exploration_rate_decay": 0.99999975,
        #     # "exploration_rate_min": 0.1,
        #     "save_model_every": 5e5,
        #     "learn_every": 8,
        #     "sync_every": 32,
        #     "gradient_steps": 2,
        #     "seed": int(time.time()),
        #     "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # }
        # config = SimpleNamespace(**config)

        # The main training loop
        env = make_env(env_id, config)
        agent = SingelLayerAgent(env, config)
        print("agent.policy_mlp_net:", agent.policy_mlp_net)
        print("agent.vqvae_model:", agent.vqvae_model)

        # wandb.watch(agent.policy_mlp_net, log_freq=100)
        # wandb.watch(agent.target_mlp_net, log_freq=100)
        # wandb.watch(agent.vqvae_model, log_freq=100)

        comment = ""
        log_dir_tensorboard = f"/workspace/repos_dev/VQVAE_RL/log_tensorboard/singlelayer/{env_id}/{current_time}_{comment}"
        tb_writer = SummaryWriter(log_dir_tensorboard)
        print("log_dir_tensorboard:", log_dir_tensorboard)

        time_start_training = time.time()

        # transformer = transform_dict["Boxing-v0"]
        # for i_episode in range(config.num_episodes_train):
        # old_snapshot = tracemalloc.take_snapshot()
        while agent.timesteps_done < int(config.total_timesteps + config.init_steps):
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

                if agent.timesteps_done >= int(config.total_timesteps + config.init_steps):
                    print("training finished! Cost timesteps:", agent.timesteps_done)
                    print("agent.lr_dqn:", agent.dqn_optimizer.param_groups[0]["lr"])
                    print("train/exploration_rate:", agent.exploration_rate)
                    print("_current_progress_remaining:", agent._current_progress_remaining)
                    loss_list = np.array(loss_list, dtype=float)
                    mean_losses = np.nanmean(loss_list, axis=0)
                    metrics = {
                        "train/episodic_reward": episodic_reward,
                        "train/episodic_negative_reward": episodic_negative_reward,
                        "train/episodic_non_zero_reward": episodic_non_zero_reward,
                        "train/timesteps_done": agent.timesteps_done,
                        "train/time_elapsed": (time.time() - time_start_training) / 3600,
                        "train/episode_length": t + 1,
                        "train/episodes_done": agent.episodes_done,
                        "train/exploration_rate": agent.exploration_rate,
                        "train/lr_dqn": agent.dqn_optimizer.param_groups[0]["lr"],
                        "train/lr_vqvae": agent.vqvae_optimizer.param_groups[0]["lr"],
                        "train/episodic_fps": int((t + 1) / (time.time() - time_start_episode)),
                        "train/recon_loss": mean_losses[0],
                        "train/vq_loss": mean_losses[1],
                        "train/vqvae_loss": mean_losses[0] + mean_losses[1],
                        "train/q_loss": mean_losses[2],
                        "train/current_progress_remaining": agent._current_progress_remaining,
                    }
                    wandb.log({**metrics})
                    break

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
                # if agent.timesteps_done == agent.init_steps:
                #     for i in range(int(agent.init_steps / 10)):
                #         recon_loss, vq_loss, q_loss = agent.learn(tb_writer)
                # else:
                #     recon_loss, vq_loss, q_loss = agent.learn(tb_writer)

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
                    loss_list = np.array(loss_list, dtype=float)
                    mean_losses = np.nanmean(loss_list, axis=0)

                    agent.episodes_done += 1
                    # episode_durations.append(t + 1)
                    # plot_durations()
                    metrics = {
                        "train/episodic_reward": episodic_reward,
                        "train/episodic_negative_reward": episodic_negative_reward,
                        "train/episodic_non_zero_reward": episodic_non_zero_reward,
                        "train/timesteps_done": agent.timesteps_done,
                        "train/time_elapsed": (time.time() - time_start_training) / 3600,
                        "train/episode_length": t + 1,
                        "train/episodes_done": agent.episodes_done,
                        "train/exploration_rate": agent.exploration_rate,
                        "train/lr_dqn": agent.dqn_optimizer.param_groups[0]["lr"],
                        "train/lr_vqvae": agent.vqvae_optimizer.param_groups[0]["lr"],
                        "train/episodic_fps": int((t + 1) / (time.time() - time_start_episode)),
                        "train/recon_loss": mean_losses[0],
                        "train/vq_loss": mean_losses[1],
                        "train/vqvae_loss": mean_losses[0] + mean_losses[1],
                        "train/q_loss": mean_losses[2],
                        "train/current_progress_remaining": agent._current_progress_remaining,
                    }
                    wandb.log({**metrics})

                    print(">>>>>>>>>>>>>>>>Episode Done>>>>>>>>>>>>>>>>>")
                    print(
                        "time cost so far: {:.3f} h".format(
                            (time.time() - time_start_training) / 3600
                        )
                    )
                    print("episodic time cost: {:.1f} s".format(time.time() - time_start_episode))
                    print("Total_steps_done:", agent.timesteps_done)
                    print("Episodic_fps:", int((t + 1) / (time.time() - time_start_episode)))
                    print("Episode finished after {} timesteps".format(t + 1))
                    print("Episode {} reward: {}".format(agent.episodes_done, episodic_reward))
                    # print(
                    #     "memory_allocated: {:.1f} MB".format(
                    #         torch.cuda.memory_allocated() / (1024 * 1024)
                    #     )
                    # )
                    # print(
                    #     "memory_reserved: {:.1f} MB".format(
                    #         torch.cuda.memory_reserved() / (1024 * 1024)
                    #     )
                    # )
                    print("agent.earlystopping.stop", agent.earlystopping.stop)

                    print("mean losses(recon, vq, q):", np.around(mean_losses, decimals=6))
                    print("_current_progress_remaining:", agent._current_progress_remaining)
                    print("train/exploration_rate:", agent.exploration_rate)

                    print("number of vqvae model forward passes:", agent.vqvae_model.forward_call)

                    print(
                        "size of agent.memory: {} entries and {} mb".format(
                            len(agent.memory), sys.getsizeof(agent.memory) / (1024 * 1024)
                        )
                    )

                    # current_snapshot = tracemalloc.take_snapshot()
                    # top_stats = current_snapshot.compare_to(old_snapshot, "lineno")
                    # # top_stats = current_snapshot.statistics("lineno")
                    # # top_stats = snapshot.statistics("filename")
                    # print("[ Top 10 ]")
                    # for stat in top_stats[:10]:
                    #     print(stat)

                    # old_snapshot = current_snapshot

                    break

            # # Update the target network, copying all weights and biases in DQN
            # if i_episode % agent.target_update == 0:
            #     agent.target_mlp_net.load_state_dict(
            #         agent.policy_mlp_net.state_dict())

        wandb.finish()

    print("Complete")
    env.close()


def train_duolayer():
    env_id = "CarRacing-v0"
    # CarRacing-v0, ALE/Skiing-v5, Boxing-v0, ALE/Freeway-v5, ALE/Pong-v5, ALE/Breakout-v5, BreakoutNoFrameskip-v4, RiverraidNoFrameskip-v4
    # vae_version = "vqvae_c3_embedding16x64_3_duolayer"

    for rep in range(5):
        current_time = datetime.datetime.now() + datetime.timedelta(hours=2)
        current_time = current_time.strftime("%b%d_%H-%M-%S")
        # 🐝 initialise a wandb run
        wandb.init(
            project="vqvae+latent_rl",
            # mode="disabled",
            group="VQVAE_Duo_EMA_sync8",
            # group="Vanilla DQN",
            tags=[
                # "duolayer",
                # "as_vanilla_dqn",
                # "vqvae2(32-64-64)",
                # "encoder_detach",
                # "DDQN"
                # "polyak_abstract",
                # "vanilla dqn",
                # "learn2",
            ],  # vqvae2(32-64-128), vqvae(128-256)
            config={
                "env_id": env_id,
                "total_timesteps": 3e5,
                "init_steps": 1000,
                "action_repetition": 3,  # 3 for carracing, 2 for boxing
                "n_frame_stack": 4,
                # "dropout": random.uniform(0.01, 0.80),
                "vqvae_inchannel": int(3 * 1),
                "vqvae_latent_channel": 16,  # 16
                "vqvae_num_embeddings": 64,  # 64
                "reconstruction_path": os.path.join(
                    "/workspace/repos_dev/VQVAE_RL/reconstruction/duolayer", env_id, current_time
                ),
                # "reconstruction_path": None,
                # "total_episodes": 1000,
                "lr_vqvae": 5e-4,
                "lr_ground_Q": 2.5e-4,  # "lin_5.3e-4", 5e-4
                "lr_abstract_V": 2.5e-4,  # "lin_5.3e-4", 5e-4
                "batch_size": 256,
                "validation_size": 128,
                "validate_every": 10,
                "size_replay_memory": int(1e5),
                "gamma": 0.97,
                "omega": 0,  # 2.5e-3, 1
                "tau": "None",
                "exploration_fraction": 0.9,
                "exploration_initial_eps": 0.1,
                "exploration_final_eps": 0.01,
                "save_model_every": 5e5,
                "ground_learn_every": 4,
                "ground_sync_every": 8,
                "ground_gradient_steps": 1,
                "abstract_learn_every": 4,
                "abstract_sync_every": 8,
                "abstract_gradient_steps": 1,
                "seed": int(time.time()),
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                "run_func_name": "train_duo_layer",
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
        # env = make_atari(env_id)
        # env = wrap_deepmind(env)
        agent = DuoLayerAgent(env, config)
        # print("agent.policy_mlp_net:", agent.ground_Q_net)
        # print("agent.vqvae_model:", agent.vqvae_model)

        # wandb.watch(agent.ground_Q_net, log_freq=1000)
        # wandb.watch(agent.abstract_V_net, log_freq=1000)
        # wandb.watch(agent.vqvae_model, log_freq=1000)

        current_time = datetime.datetime.now() + datetime.timedelta(hours=2)
        current_time = current_time.strftime("%b%d_%H-%M-%S")

        comment = ""
        log_dir_tensorboard = f"/workspace/repos_dev/VQVAE_RL/log_tensorboard/end2end_duolayer/{env_id}/{current_time}_{comment}"
        # tb_writer = SummaryWriter(log_dir_tensorboard)
        tb_writer = None
        print("log_dir_tensorboard:", log_dir_tensorboard)
        print("reconstruction_path:", config.reconstruction_path)

        time_start_training = time.time()

        # transformer = transform_dict["Boxing-v0"]
        # for i_episode in range(config.total_episodes):
        while agent.timesteps_done < int(config.total_timesteps + config.init_steps):
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

                next_state, reward, done, _ = env.step(action)
                # env.render()
                episodic_reward += reward
                if reward < 0:
                    episodic_negative_reward += reward
                else:
                    episodic_non_zero_reward += reward

                if agent.timesteps_done >= int(config.total_timesteps + config.init_steps):
                    break

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
                # agent.cache_lazy(state, action, next_state, reward, done)
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
                loss_list.append(agent.learn2_improved(tb_writer))
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
                    loss_list = np.array(loss_list, dtype=float)
                    mean_losses = np.nanmean(loss_list, axis=0)

                    agent.episodes_done += 1
                    # episode_durations.append(t + 1)
                    # plot_durations()
                    metrics = {
                        "train/episodic_reward": episodic_reward,
                        "train/episodic_negative_reward": episodic_negative_reward,
                        "train/episodic_non_zero_reward": episodic_non_zero_reward,
                        "train/timesteps_done": agent.timesteps_done,
                        "train/time_elapsed": (time.time() - time_start_training) / 3600,
                        "train/episode_length": t + 1,
                        "train/episodes_done": agent.episodes_done,
                        "train/exploration_rate": agent.exploration_rate,
                        "train/episodic_fps": int((t + 1) / (time.time() - time_start_episode)),
                        "train/lr_vqvae_optimizer": agent.vqvae_optimizer.param_groups[0]["lr"],
                        "train/lr_ground_Q_optimizer": agent.ground_Q_optimizer.param_groups[0][
                            "lr"
                        ],
                        "train/lr_abstract_V_optimizer": agent.abstract_V_optimizer.param_groups[0][
                            "lr"
                        ],
                        "train/recon_loss": mean_losses[0],
                        "train/vq_loss": mean_losses[1],
                        "train/vqvae_loss": mean_losses[0] + mean_losses[1],
                        "train/abstract_td_error": mean_losses[2],
                        "train/ground_td_error": mean_losses[3],
                        "train/current_progress_remaining": agent._current_progress_remaining,
                    }
                    wandb.log({**metrics})

                    print(f">>>>>>>>>>>>>>>>Episode Done| Repetition {rep}>>>>>>>>>>>>>>>>>")
                    print(
                        "time cost so far: {:.3f} h".format(
                            (time.time() - time_start_training) / 3600
                        )
                    )
                    print("episodic time cost: {:.1f} s".format(time.time() - time_start_episode))
                    print("Total_steps_done:", agent.timesteps_done)
                    print("Episodic_fps:", int((t + 1) / (time.time() - time_start_episode)))
                    print("Episode finished after {} timesteps".format(t + 1))
                    print("Episode {} reward: {}".format(agent.episodes_done, episodic_reward))

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
                    print("agent.vqvae_earlystopping.stop", agent.vqvae_earlystopping.stop)

                    print(
                        "mean losses(recon, vq, abstract_td_error, ground_td_error):",
                        np.around(mean_losses, decimals=6),
                    )
                    print("_current_progress_remaining:", agent._current_progress_remaining)
                    print("train/exploration_rate:", agent.exploration_rate)

                    print("number of vqvae model forward passes:", agent.vqvae_model.forward_call)

                    print(
                        "size of agent.memory: {} entries and {} mb".format(
                            len(agent.memory), sys.getsizeof(agent.memory) / (1024 * 1024)
                        )
                    )

                    break

            # # Update the target network, copying all weights and biases in DQN
            # if i_episode % agent.target_update == 0:
            #     agent.target_mlp_net.load_state_dict(
            #         agent.policy_mlp_net.state_dict())

        wandb.finish()

    print("Complete")
    env.close()


def train_vanilla_dqn():
    env_id = "Boxing-v0"  # CarRacing-v0, ALE/Skiing-v5, Boxing-v0
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")

    for _ in range(1):
        # 🐝 initialise a wandb run
        wandb.init(
            project="vqvae+latent_rl",
            config={
                "env_id": env_id,
                # "total_time_steps": 1e6,
                "action_repetition": 2,
                "n_frame_stack": 1,
                # "lr": 1e-3,
                # "dropout": random.uniform(0.01, 0.80),
                # "vqvae_inchannel": int(3 * 1),
                # "vqvae_latent_channel": 16,
                # "vqvae_num_embeddings": 64,
                # "reconstruction_path": os.path.join(
                #     "/workspace/repos_dev/VQVAE_RL/reconstruction/singlelayer", env_id, current_time
                # ),
                # "reconstruction_path": None,
                "num_episodes_train": 1000,
                "batch_size": 128,
                # "validation_size": 128,
                # "validate_every": 10,
                "size_replay_memory": 100_000,
                "gamma": 0.97,
                "exploration_fraction": 0.99,
                "exploration_initial_eps": 0.1,
                "exploration_final_eps": 0.01,
                # "eps_decay": 200,
                # "target_update": 250,
                # "exploration_rate": 0.5,
                # "exploration_rate_decay": 0.99999975,
                # "exploration_rate_min": 0.1,
                # "save_model_every": 5e5,
                "init_steps": 1e4,
                "learn_every": 4,
                "sync_every": 8,
                "gradient_steps": 1,
                "learning_rate": "lin_5.3e-4",
                "seed": int(time.time()),
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                "run_func_name": "train_vanilla_dqn",
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
        env = make_env(env_id, config)  # when running atari game
        # env = gym.make(env_id)
        agent = VanillaDQNAgent(env, config)
        print("agent.policy_mlp_net:", agent.policy_mlp_net)

        wandb.watch(agent.policy_mlp_net, log_freq=100)
        # wandb.watch(agent.target_mlp_net, log_freq=100)

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
                # if agent.total_steps_done == agent.init_steps:
                #     for i in range(int(agent.init_steps / 10)):
                #         recon_loss, vq_loss, q_loss = agent.learn(tb_writer)
                q_loss = agent.learn(tb_writer)
                loss_list.append(q_loss)
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

                    agent.num_episodes_finished += 1
                    # episode_durations.append(t + 1)
                    # plot_durations()
                    loss_list = np.array(loss_list, dtype=float)
                    mean_q_loss = np.nanmean(loss_list)
                    metrics = {
                        "train/episodic_reward": episodic_reward,
                        "train/episodic_negative_reward": episodic_negative_reward,
                        "train/episodic_non_zero_reward": episodic_non_zero_reward,
                        "train/total_steps_done": agent.total_steps_done,
                        "train/time_elapsed": (time.time() - time_start_training) / 3600,
                        "train/episode_length": t + 1,
                        "train/episodes": i_episode,
                        "train/exploration_rate": agent.exploration_rate,
                        "train/lr_dqn": agent.dqn_optimizer.param_groups[0]["lr"],
                        "train/episodic_fps": int((t + 1) / (time.time() - time_start_episode)),
                        "train/q_loss": mean_q_loss,
                    }
                    wandb.log({**metrics})

                    print(">>>>>>>>>>>>>>>>Episode Done>>>>>>>>>>>>>>>>>")
                    print(
                        "time cost so far: {:.3f} h".format(
                            (time.time() - time_start_training) / 3600
                        )
                    )
                    print("episodic time cost: {:.1f} s".format(time.time() - time_start_episode))
                    print("Total_steps_done:", agent.total_steps_done)
                    print("Episodic_fps:", int((t + 1) / (time.time() - time_start_episode)))
                    print("Episode finished after {} timesteps".format(t + 1))
                    print("Episode({}) reward: {}".format(i_episode, episodic_reward))
                    print("Agent.exploration_rate:", agent.exploration_rate)
                    # print(
                    #     "memory_allocated: {:.1f} MB".format(
                    #         torch.cuda.memory_allocated() / (1024 * 1024)
                    #     )
                    # )
                    # print(
                    #     "memory_reserved: {:.1f} MB".format(
                    #         torch.cuda.memory_reserved() / (1024 * 1024)
                    #     )
                    # )
                    print("agent.earlystopping.stop", agent.earlystopping.stop)

                    print("mean q loss:", np.around(mean_q_loss, decimals=4))

                    break

            # # Update the target network, copying all weights and biases in DQN
            # if i_episode % agent.target_update == 0:
            #     agent.target_mlp_net.load_state_dict(
            #         agent.policy_mlp_net.state_dict())

        wandb.finish()

    print("Complete")
    env.close()


def find_gpu():
    # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Get the first available GPU
    DEVICE_ID_LIST = GPUtil.getAvailable(
        order="random",
        limit=4,
        maxLoad=0.95,
        maxMemory=0.75,
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


if __name__ == "__main__":

    # from utils.gpu_profile import gpu_profile
    find_gpu()
    # sys.settrace(gpu_profile)
    torch.set_num_threads(1)
    # tracemalloc.start()
    # set number of threads to 1, when using T.ToTensor() it will cause very high cpu usage and using milti-threads

    # train_single_layer()
    train_duolayer()
    # train_vanilla_dqn()
