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
import yaml

# import tracemalloc

import colored_traceback.auto
import GPUtil
import gym
from gym.wrappers import TimeLimit
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

from latentrl.hdqn import HDQN
from latentrl.hdqn_vae import HDQN_KMeans_VAE, HDQN_ManualAbs
from latentrl.policies import utils

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
    FrameStack,
    FrameStackCustom,
    FrameStackWrapper,
    LimitNumberActionsWrapper,
    MinigridEmptyRewardWrapper,
    MinigridInfoWrapper,
    StateBonusCustom,
    WarpFrameRGB,
    PreprocessObservationWrapper,
    pack_env_wrappers,
)

# from atari_wrapper import (
#     # NoopResetEnv,
#     # MaxAndSkipEnv,
#     # FireResetEnv,
#     # EpisodicLifeEnv,
#     # WarpFrame,
#     # ClipRewardEnv,
#     # FrameStack,
#     # ScaledFloatFrame,
#     make_env_atari,
# )
from gym_minigrid.wrappers import RGBImgObsWrapper, FullyObsWrapper, ImgObsWrapper, StateBonus
from gym_minigrid.envs.empty import EmptyEnv
from gym_minigrid.minigrid import MiniGridEnv


def make_env_carracing(env_id):
    # env = gym.make(env_id).unwrapped
    env = gym.make(
        env_id,
        continuous=True,
        # render_mode="human",
    )
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
        ActionRepetitionWrapper,
        EpisodeEarlyStopWrapper,
        # Monitor,
        # CarRandomStartWrapper,
        PreprocessObservationWrapper,
        # EncodeStackWrapper,
        # PunishRewardWrapper,
        # FrameStackWrapper,
        FrameStack,
    ]
    wrapper_kwargs_list = [
        None,
        {"action_repetition": 2},
        {"max_neg_rewards": 100, "punishment": -20.0},
        {"vertical_cut_d": 84, "shape": 84, "num_output_channels": 1, "preprocess_mode": "cv2"},
        {"n_frames": 4},
    ]

    wrapper = pack_env_wrappers(wrapper_class_list, wrapper_kwargs_list)
    env = wrapper(env)
    # env.seed(seed)

    return env


def make_env_minigrid(config):
    # env = gym.make(
    #     env_id,
    #     # new_step_api=True,
    #     # render_mode="human",
    # )
    env = EmptyEnv(
        size=config.env_size,
        # agent_start_pos=tuple(config.agent_start_pos),
    )
    # env = LimitNumberActionsWrapper(env, limit=3)
    # env = TimeLimit(env, max_episode_steps=3000, new_step_api=True)
    # env = StateBonus(env)
    env = FullyObsWrapper(env)
    # env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    # env = PreprocessObservationWrapper(env, shape=84, num_output_channels=1)
    # env = WarpFrameRGB(env)
    # env = StateBonusCustom(env)
    # env = MinigridEmptyRewardWrapper(env)
    env = MinigridInfoWrapper(env)
    # env = FrameStack(env, n_frames=1)
    # env.new_step_api = True
    return env


MAKE_ENV_FUNCS = {
    # "Atari": make_env_atari,
    "CarRacing": make_env_carracing,
    "GymMiniGrid": make_env_minigrid,
}


def train_hdqn():
    env_id = "CarRacing-v2"
    # CarRacing-v0, ALE/Skiing-v5, Boxing-v0, ALE/Freeway-v5, ALE/Pong-v5, ALE/Breakout-v5, BreakoutNoFrameskip-v4, RiverraidNoFrameskip-v4
    # vae_version = "vqvae_c3_embedding16x64_3_duolayer"

    for rep in range(5):
        current_time = datetime.datetime.now() + datetime.timedelta(hours=2)
        current_time = current_time.strftime("%b%d_%H-%M-%S")
        # 🐝 initialise a wandb run
        wandb.init(
            project="HDQN",
            mode="disabled",
            group="vqvae+dqn",
            # group="Vanilla DQN",
            tags=[
                # "duolayer",
                # "as_vanilla_dqn",
                # "vqvae2(32-64-64)",
                # "encoder_detach",
                # "DDQN"
                # "polyak_abstract",
                # "vanilla dqn",
                "random seed",
                "omega0",
            ],  # vqvae2(32-64-128), vqvae(128-256)
            config={
                "env_id": env_id,
                "total_timesteps": 3e5,
                "init_steps": 1000,
                # "action_repetition": 3,  # 3 for carracing, 2 for boxing
                # "n_frame_stack": 4,
                "hidden_dims": [32, 64],
                "dim_encoder_out": 50,  # 16
                "vq_n_codebook": 32,  # 64
                "lr_ground_Q": 1.5e-4,  # "lin_5.3e-4", 5e-4
                "lr_abstract_V": 1.5e-4,  # "lin_5.3e-4", 5e-4
                "lr_vq": 2.5e-4,
                "lr_encoder": 2.5e-4,
                "lr_decoder": 2.5e-4,
                "batch_size": 128,
                "size_replay_memory": int(1e5),
                "gamma": 0.97,
                "omega": 0,  # 2.5e-3, 1
                "ground_tau": 0.02,  # 0.01
                "abstract_tau": 0.02,
                "encoder_tau": 0.1,  # 0.05
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
                "validate_every": 1000,
                "reset_training_info_every": 4000,
                "save_recon_every": 1000,
                "buffer_recent_states_every": 1000,
                "clip_grad": False,  # False for carcing, True for atari
            },
        )
        config = wandb.config

        env = make_env_carracing(env_id, config, seed=int(time.time()))
        # env = make_atari(env_id)
        # env = wrap_deepmind(env, seed=int(time.time()))
        agent = HDQN(config, env)

        # comment = ""
        # log_dir_tensorboard = f"/workspace/repos_dev/VQVAE_RL/log_tensorboard/end2end_duolayer/{env_id}/{current_time}_{comment}"
        # tb_writer = SummaryWriter(log_dir_tensorboard)
        # tb_writer = None
        # print("log_dir_tensorboard:", log_dir_tensorboard)
        # print("reconstruction_path:", config.reconstruction_path)

        time_start_training = time.time()

        while agent.timesteps_done < int(config.total_timesteps + config.init_steps):
            time_start_episode = time.time()
            # Initialize the environment and state
            state = env.reset()
            episodic_reward = 0
            episodic_negative_reward = 0
            episodic_non_negative_reward = 0
            for t in count():
                # Select and perform an action
                action = agent.act(state)

                next_state, reward, done, _ = env.step(action)
                # env.render()
                episodic_reward += reward
                if reward < 0:
                    episodic_negative_reward += reward
                else:
                    episodic_non_negative_reward += reward

                if agent.timesteps_done >= int(config.total_timesteps + config.init_steps):
                    break

                # Store the transition in memory
                agent.cache(state, action, next_state, reward, done)
                # agent.cache_lazy(state, action, next_state, reward, done)

                state = next_state

                agent.update()
                agent.maybe_buffer_recent_states(state)

                if done:
                    # print("sys.getsizeof(agent.memory)", sys.getsizeof(agent.memory))
                    # print(torch.cuda.memory_reserved()/(1024*1024), "MB")
                    # print(torch.cuda.memory_allocated()/(1024*1024), "MB")
                    agent.episodes_done += 1

                    metrics = {
                        "reward/episodic_reward": episodic_reward,
                        "reward/episodic_negative_reward": episodic_negative_reward,
                        "reward/episodic_non_negative_reward": episodic_non_negative_reward,
                        "train/timesteps_done": agent.timesteps_done,
                        "train/time_elapsed": (time.time() - time_start_training) / 3600,
                        "train/episode_length": t + 1,
                        "train/episodes_done": agent.episodes_done,
                        "train/exploration_rate": agent.exploration_rate,
                        "train/episodic_fps": int((t + 1) / (time.time() - time_start_episode)),
                        "train/current_progress_remaining": agent._current_progress_remaining,
                        "lr/lr_vq_optimizer": agent.vector_quantizer_optimizer.param_groups[0][
                            "lr"
                        ],
                        "lr/lr_ground_Q_optimizer": agent.ground_Q_optimizer.param_groups[0]["lr"],
                        "lr/lr_abstract_V_optimizer": agent.abstract_V_optimizer.param_groups[0][
                            "lr"
                        ],
                    }
                    wandb.log(metrics)

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

                    # print(
                    #     "mean losses(recon, vq, abstract_td_error, ground_td_error):",
                    #     np.around(mean_losses, decimals=6),
                    # )
                    print("_current_progress_remaining:", agent._current_progress_remaining)
                    print("train/exploration_rate:", agent.exploration_rate)
                    print("number of vqvae model forward passes:", agent.decoder.n_forward_call)
                    print(
                        "size of agent.memory: {} entries and {} mb".format(
                            len(agent.memory), sys.getsizeof(agent.memory) / (1024 * 1024)
                        )
                    )

                    break

        wandb.finish()

    print("Complete")
    env.close()


def train_dqn_kmeans():
    # env_id = "MiniGrid-Empty-Random-6x6-v0"
    env_id = "MiniGrid-Empty-16x16-v0"
    # env_id = "CarRacing-v2"
    # CarRacing-v05, ALE/Skiing-v5, Boxing-v0, ALE/Freeway-v5, ALE/Pong-v5, ALE/Breakout-v5, BreakoutNoFrameskip-v4, RiverraidNoFrameskip-v4
    # vae_version = "vqvae_c3_embedding16x64_3_duolayer"

    # cfg_key = "MiniGrid-Empty-RGB"
    cfg_key = "MiniGrid-Empty-16x16-v0"
    for rep in range(8):
        # load hyperparameters from yaml config file
        with open("/workspace/repos_dev/VQVAE_RL/hyperparams/dqn_ae.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)[cfg_key]
            pprint(config)
        current_time = datetime.datetime.now() + datetime.timedelta(hours=2)
        current_time = current_time.strftime("%b%d_%H-%M-%S")
        # 🐝 initialise a wandb run
        run = wandb.init(
            project="HDQN_Neo",
            mode="disabled",
            group=config["wandb_group"],
            tags=[
                f"omega{config['omega']}",
            ],
            config=config,
        )
        config = wandb.config
        make_env = MAKE_ENV_FUNCS[config.env_type]
        env = make_env(env_id)

        # agent = HDQN_KMeans_VAE(config, env)
        agent = HDQN_ManualAbs(config, env)
        # agent.vis_abstraction(prefix=f"rep{rep}")

        # comment = ""
        # log_dir_tensorboard = f"/workspace/repos_dev/VQVAE_RL/log_tensorboard/end2end_duolayer/{env_id}/{current_time}_{comment}"
        # tb_writer = SummaryWriter(log_dir_tensorboard)
        # tb_writer = None
        # print("log_dir_tensorboard:", log_dir_tensorboard)
        # print("reconstruction_path:", config.reconstruction_path)

        goal_found = False
        time_start_training = time.time()
        # env.reset()
        total_steps = int(config.total_timesteps + config.init_steps)
        while agent.timesteps_done < total_steps:
            time_start_episode = time.time()
            # Initialize the environment and state
            # env.seed(seed=int(time.time()))
            state, info = env.reset()
            episodic_reward = 0
            episodic_negative_reward = 0
            episodic_non_negative_reward = 0
            for t in count():
                # Select and perform an action
                with utils.eval_mode(agent):
                    action = agent.act(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                # if isinstance(env.unwrapped, MiniGridEnv):
                #     info["agent_pos"] = env.agent_pos
                #     info["agent_dir"] = env.agent_dir
                # print(env.agent_pos, env.agent_dir)
                # time.sleep(10)

                # Store the transition in memory
                agent.cache(state, action, next_state, reward, terminated, info)
                # agent.cache_lazy(state, action, next_state, reward, terminated)

                # reward = info["original_reward"]
                episodic_reward += reward
                if reward < 0:
                    episodic_negative_reward += reward
                else:
                    episodic_non_negative_reward += reward

                if agent.timesteps_done >= total_steps:
                    truncated = True

                state = next_state
                if agent.timesteps_done >= config.init_steps:
                    agent.update()

                for i in range(5):
                    if agent.timesteps_done == (i + 1) * total_steps / 5:
                        # agent.vis_abstraction(prefix=f"rep{rep}")
                        break
                # agent.maybe_buffer_recent_states(state)

                if terminated or truncated:
                    if terminated:
                        goal_found = True
                    # print("sys.getsizeof(agent.memory)", sys.getsizeof(agent.memory))
                    # print(torch.cuda.memory_reserved()/(1024*1024), "MB")
                    # print(torch.cuda.memory_allocated()/(1024*1024), "MB")
                    agent.episodes_done += 1

                    metrics = {
                        "reward/episodic_reward": episodic_reward,
                        "reward/episodic_negative_reward": episodic_negative_reward,
                        "reward/episodic_non_negative_reward": episodic_non_negative_reward,
                        "train/timesteps_done": agent.timesteps_done,
                        "train/time_elapsed": (time.time() - time_start_training) / 3600,
                        "train/episode_length": t + 1,
                        "train/episodes_done": agent.episodes_done,
                        "train/fps_per_episode": int((t + 1) / (time.time() - time_start_episode)),
                    }
                    wandb.log(metrics)

                    print(
                        f">>>>>>>>>>>>>>>>Episode{agent.episodes_done} Done| Repetition {rep}>>>>>>>>>>>>>>>>>"
                    )
                    print("terminal, trauncated:", terminated, truncated)
                    print(
                        "time cost so far: {:.3f} h".format(
                            (time.time() - time_start_training) / 3600
                        )
                    )
                    print("episodic time cost: {:.1f} s".format(time.time() - time_start_episode))
                    print("Total_steps_done:", agent.timesteps_done)
                    print("Episodic_fps:", int((t + 1) / (time.time() - time_start_episode)))
                    print("Episode finished after {} timesteps".format(t + 1))
                    print(
                        "### Episode {} reward: {} ###".format(agent.episodes_done, episodic_reward)
                    )

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

                    # print(
                    #     "mean losses(recon, vq, abstract_td_error, ground_td_error):",
                    #     np.around(mean_losses, decimals=6),
                    # )
                    print("_current_progress_remaining:", agent._current_progress_remaining)
                    print("train/exploration_rate:", agent.exploration_rate)
                    # print("number of vqvae model forward passes:", agent.decoder.n_forward_call)
                    print(
                        "size of agent.memory: {} entries and {} mb".format(
                            len(agent.memory), sys.getsizeof(agent.memory) / (1024 * 1024)
                        )
                    )

                    break

        wandb.finish()

        if goal_found:
            print("====Goal Found====")
        else:
            print("====Goal Not Found in this repetition, deleting this run from wandb====")
            if run.mode != "disabled":
                api = wandb.Api()
                run = api.run(f"team-yuan/HDQN_Neo/{run.id}")
                run.delete()

    print("Complete")
    env.close()


def train_hq_table():
    # env_id = "MiniGrid-Empty-Random-6x6-v0"
    # env_id = "MiniGrid-Empty-16x16-v0"
    # env_id = "CarRacing-v2"
    # CarRacing-v05, ALE/Skiing-v5, Boxing-v0, ALE/Freeway-v5, ALE/Pong-v5, ALE/Breakout-v5, BreakoutNoFrameskip-v4, RiverraidNoFrameskip-v4
    # vae_version = "vqvae_c3_embedding16x64_3_duolayer"

    # cfg_key = "MiniGrid-Empty-RGB"
    cfg_key = "MiniGrid-Empty-v0-table"
    # load hyperparameters from yaml config file
    with open("/workspace/repos_dev/VQVAE_RL/hyperparams/dqn_ae.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)[cfg_key]
        pprint(cfg)
    for rep in range(30):
        print(f"====Starting Repetition {rep}====")
        current_time = datetime.datetime.now() + datetime.timedelta(hours=2)
        current_time = current_time.strftime("%b%d_%H-%M-%S")
        # 🐝 initialise a wandb run
        run = wandb.init(
            project="HDQN_TABLE",
            # mode="disabled",
            group=cfg["wandb_group"],
            tags=[
                "tbl",
                "shp" if cfg["use_shaping"] else "no_shp",
                f"omg{cfg['omega']}",
                f"env{cfg['env_size']}x{cfg['env_size']}",
                f"start({cfg['agent_start_pos']}, {cfg['agent_start_pos']})",
            ],
            notes=cfg["wandb_notes"],
            config=cfg,
        )
        config = wandb.config
        make_env = MAKE_ENV_FUNCS[config.env_type]
        env = make_env(config)

        # agent = HDQN_KMeans_VAE(config, env)
        agent = HDQN_ManualAbs(config, env, use_table4grd=True)
        agent.set_abs_ticks(config, 0)
        # agent.vis_abstraction(prefix=f"rep{rep}")

        # comment = ""
        # log_dir_tensorboard = f"/workspace/repos_dev/VQVAE_RL/log_tensorboard/end2end_duolayer/{env_id}/{current_time}_{comment}"
        # tb_writer = SummaryWriter(log_dir_tensorboard)
        # tb_writer = None
        # print("log_dir_tensorboard:", log_dir_tensorboard)
        # print("reconstruction_path:", config.reconstruction_path)
        agent.vis_abstraction()
        goal_found = False
        steps_after_goal_found = 0
        episodes_since_goal_found = 0
        interval4SemiMDP = 0
        time_start_training = time.time()
        # gym.reset(seed=int(time.time()))
        total_steps = int(config.total_timesteps + config.init_steps)
        agent.cache_goal_transition()
        while agent.timesteps_done < total_steps:
            time_start_episode = time.time()
            # Initialize the environment and state
            state, info = env.reset()
            episodic_reward = 0
            episodic_negative_reward = 0
            episodic_non_negative_reward = 0
            episodic_shaped_reward = 0
            # action = agent.act_table(info)
            for t in count():
                # Select and perform an action
                action = agent.act_table(info)
                if goal_found:
                    steps_after_goal_found += 1
                # [Step]
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.timesteps_done += 1

                abs_state1, abs_value1 = agent.get_abstract_value(info["agent_pos1"])
                abs_state2, abs_value2 = agent.get_abstract_value(info["agent_pos2"])
                interval4SemiMDP += 1
                info["interval4SemiMDP"] = interval4SemiMDP
                if not (abs_state1 == abs_state2 and reward == 0):
                    # this conditino should match the one in update_absV
                    interval4SemiMDP = 0
                if abs_state1 != abs_state2:
                    episodic_shaped_reward += config.gamma * abs_value2 - abs_value1

                agent.update_grd_visits(info)

                for i in range(len(config.abs_ticks) - 1):
                    if agent.timesteps_done == (i + 1) * total_steps / len(config.abs_ticks):
                        agent.set_abs_ticks(config, i + 1)
                # if isinstance(env.unwrapped, MiniGridEnv):
                #     info["agent_pos"] = env.agent_pos
                #     info["agent_dir"] = env.agent_dir
                # print(env.agent_pos, env.agent_dir)
                # time.sleep(10)

                # [Store the transition in memory]
                agent.cache(state, action, next_state, reward, terminated, info)
                # agent.cache_lazy(state, action, next_state, reward, terminated)

                # reward = info["original_reward"]
                episodic_reward += reward
                if reward < 0:
                    episodic_negative_reward += reward
                else:
                    episodic_non_negative_reward += reward
                # [update]
                action_prime = agent.act_table(info)
                if agent.timesteps_done >= config.init_steps:
                    # here we use table to do update
                    agent.update_table(use_shaping=config.use_shaping)
                    # agent.update_table_no_memory(
                    #     use_shaping=config.use_shaping, action_prime=action_prime
                    # )
                    # agent.update_table_abs_update_non_parallel2(use_shaping=config.use_shaping)
                # [visualization]
                for i in range(10):
                    if agent.timesteps_done == (i + 1) * total_steps / 10:
                        # agent.vis_grd_visits(norm_log=50)
                        # agent.vis_grd_visits(norm_log=0)
                        # agent.grd_visits = np.zeros_like(agent.grd_visits)
                        agent.vis_grd_q_values(norm_log=100)
                        agent.vis_grd_q_values(norm_log=1e8)
                        agent.vis_abstract_values()
                        break
                # agent.maybe_buffer_recent_states(state)
                if agent.timesteps_done >= total_steps:
                    truncated = True

                state = next_state
                # action = action_prime

                if terminated or truncated:
                    if terminated:
                        goal_found = True
                        print("!!Goal Found!!")
                        agent.goal_found = True
                    # print("sys.getsizeof(agent.memory)", sys.getsizeof(agent.memory))
                    # print(torch.cuda.memory_reserved()/(1024*1024), "MB")
                    # print(torch.cuda.memory_allocated()/(1024*1024), "MB")
                    agent.episodes_done += 1

                    metrics = {
                        "reward/episodic_reward": episodic_reward,
                        "reward/episodic_negative_reward": episodic_negative_reward,
                        "reward/episodic_non_negative_reward": episodic_non_negative_reward,
                        "reward/episodic_shaped_reward": episodic_shaped_reward,
                        "time/timesteps_done": agent.timesteps_done,
                        "time/time_elapsed": (time.time() - time_start_training) / 3600,
                        "time/episode_length": t + 1,
                        "time/episodes_done": agent.episodes_done,
                        "time/fps_per_episode": int((t + 1) / (time.time() - time_start_episode)),
                    }
                    if goal_found:
                        episodes_since_goal_found += 1
                        metrics.update(
                            {
                                "After_goal_found/episode_length_after_first_found": t + 1,
                                "After_goal_found/episodic_reward": episodic_reward,
                                "After_goal_found/episodic_shaped_reward": episodic_shaped_reward,
                                "After_goal_found/episodes_done_after_first_found": episodes_since_goal_found,
                                "After_goal_found/steps_done_after_first_found": steps_after_goal_found,
                            }
                        )
                    wandb.log(metrics)

                    print(
                        f">>>>>>>>>>>>>>>>Episode{agent.episodes_done} Done| Repetition {rep}>>>>>>>>>>>>>>>>>"
                    )
                    print("terminal, trauncated:", terminated, truncated)
                    print(
                        "time cost so far: {:.3f} h".format(
                            (time.time() - time_start_training) / 3600
                        )
                    )
                    print("episodic time cost: {:.1f} s".format(time.time() - time_start_episode))
                    print("Total_steps_done:", agent.timesteps_done)
                    print("Episodic_fps:", int((t + 1) / (time.time() - time_start_episode)))
                    print("Episode finished after {} timesteps".format(t + 1))
                    print(f"++++++Episode: {agent.episodes_done} reward: {episodic_reward}++++++")
                    print(f"agent.goal_found: {agent.goal_found}")

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

                    # print(
                    #     "mean losses(recon, vq, abstract_td_error, ground_td_error):",
                    #     np.around(mean_losses, decimals=6),
                    # )
                    print("_current_progress_remaining:", agent._current_progress_remaining)
                    print("train/exploration_rate:", agent.exploration_rate)
                    # print("number of vqvae model forward passes:", agent.decoder.n_forward_call)
                    print(
                        "size of agent.memory: {} entries and {} mb".format(
                            len(agent.memory), sys.getsizeof(agent.memory) / (1024 * 1024)
                        )
                    )
                    # End this episode
                    break
            if agent.episodes_done > 0 and agent.episodes_done % 2 == 0:
                print("Evaluate the agent by visualizing grd visits")
                evaluate_agent(env, agent, exploit_only=True)
                evaluate_agent(env, agent, exploit_only=False)
        wandb.finish()

        # if goal_found:
        #     print("====Goal Found====")
        # else:
        #     print("====Goal Not Found in this repetition, deleting this run from wandb====")
        # if not isinstance(run.mode, wandb.sdk.lib.disabled.RunDisabled):
        #     api = wandb.Api()
        #     run = api.run(f"team-yuan/HDQN_Neo/{run.id}")
        #     run.delete()

    print("Complete")
    env.close()


def evaluate_agent(env, agent: HDQN_ManualAbs, exploit_only=True):
    agent.grd_visits = np.zeros_like(agent.grd_visits)
    episodic_reward = 0
    timesteps_done = 0
    state, info = env.reset()
    for t in count():
        action = agent.act_table(info, exploit_only=exploit_only)
        next_state, reward, terminated, truncated, info = env.step(action)
        timesteps_done += 1
        agent.update_grd_visits(info)
        episodic_reward += reward
        if terminated or truncated:
            break
    agent.vis_grd_visits(norm_log=50, suffix=f"eval_exploitOnly{exploit_only}")
    agent.vis_grd_visits(norm_log=0, suffix=f"eval_exploitOnly{exploit_only}")
    agent.grd_visits = np.zeros_like(agent.grd_visits)


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

    # train_hdqn()
    # train_dqn_kmeans()
    train_hq_table()

    # env = make_env_minigrid(env_id="MiniGrid-Empty-6x6-v0")
    # print(env.observation_space.shape)
