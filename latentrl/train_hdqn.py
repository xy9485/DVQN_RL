import argparse
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
import pathlib

print(pathlib.Path(__file__).parent.parent.absolute())

import colored_traceback.auto
import GPUtil
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import yaml
from statistics import mean

# from cv2 import mean
import minigrid

import minigrid.envs

# from minigrid import FullyObsWrapper
# from minigrid.envs import EmptyEnv

# from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper, StateBonus
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import wandb


# from data2 import RolloutDatasetNaive
# from common.learning_scheduler import EarlyStopping, ReduceLROnPlateau
import common
import common.wrappers
from common.Logger import LoggerWandb

# from common.transforms import transform_dict
# from common.utils import (
#     get_linear_fn,
#     linear_schedule,
#     make_vec_env_customized,
#     update_learning_rate,
# )
import envs

# from envs.fourrooms import FourRoomsEnv
from policies import (
    HDQN,
    HDQN_TCURL_VQ,
    HDQN_KMeans_CURL,
    HDQN_ManualAbs,
    utils,
)

from common.make_env import make_env_atari, make_env_carracing, make_env_minatar, make_env_minigrid

# from policies.hrl_dqn_agent import DuoLayerAgent, SingelLayerAgent
# from policies.vanilla_dqn_agent import VanillaDQNAgent


MAKE_ENV_FUNCS = {
    "Atari": make_env_atari,
    "MinAtar": make_env_minatar,
    "CarRacing": make_env_carracing,
    "GymMiniGrid": make_env_minigrid,
}


def parse_args():
    cli = argparse.ArgumentParser()

    # cli.add_argument("--mode", default="grd", type=str)
    cli.add_argument("--args_from_cli", default=False, action="store_true")
    # cli.add_argument("--use_grd_Q", default=True, action="store_true")
    cli.add_argument("--grd_mode", default="dqn", choices=["dqn", "ddqn", "cddqn"], type=str)
    cli.add_argument("--grd_lower_bound", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument("--use_abs_V", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument("--use_curl", default=None, choices=["on_abs", "on_grd", "off"], type=str)
    cli.add_argument("--share_encoder", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument("--curl_pair", default="raw", choices=["raw", "temp", "atc"], type=str)
    cli.add_argument("--use_vq", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument(
        "--curl_vq_cfg",
        nargs=3,
        default=[0.0, 0.0, 0.0],
        type=float,
        help="factors for cb_diversity, vq_entropy, vq_loss",
    )
    cli.add_argument("--use_dueling", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument("--use_noisynet", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument("--use_curiosity", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument("--clip_reward", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument("--clip_grad", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument("--dan", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument(
        "--per",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="prioritized experience replay",
    )

    cli.add_argument("--domain_name", default="Breakout-v5", type=str)
    cli.add_argument("--domain_type", default="Atari", type=str)
    cli.add_argument("--input_format", default="full_img", type=str)
    cli.add_argument("--env_seed", default=940805, type=int)

    cli.add_argument("--total_timesteps", default=1e5, type=int)
    cli.add_argument("--init_steps", default=1e4, type=int)
    cli.add_argument("--batch_size", default=128, type=int)
    cli.add_argument("--size_replay_memory", default=1e5, type=int)
    cli.add_argument("--gamma", default=0.99, type=float)
    cli.add_argument("--exploration", nargs=3, default=[0.99, 0.1, 0.1], type=float)
    cli.add_argument("--epsilon_decay", default=0.99998, type=float)
    cli.add_argument("--epsilon_min", default=0.01, type=float)

    cli.add_argument("--conservative_ratio", default="0.0", type=str)
    cli.add_argument("--approach_abs_factor", default="0.0", type=str)
    cli.add_argument("--omega", default=0.0, type=float, help="factor of reward shaping")
    # params for grd_Q
    cli.add_argument("--grd_encoder_linear_dims", nargs="*", default=[-1], type=int)
    cli.add_argument("--grd_critic_dims", nargs="*", default=[256, 256], type=int)
    cli.add_argument("--abs_encoder_linear_dims", default=[-1], nargs="*", type=int)
    cli.add_argument("--abs_critic_dims", default=[256, 256], nargs="*", type=int)
    cli.add_argument("--curl_projection_dims", default=[-1], nargs="*", type=int)
    cli.add_argument("--curl_enc_detach", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument("--critic_upon_vq", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument("--abs_enc_detach", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument("--grd_enc_detach", default=False, action=argparse.BooleanOptionalAction)

    cli.add_argument("--num_vq_embeddings", default=16, type=int)
    # cli.add_argument("--dim_vq_embeddings", default=128, type=int)
    cli.add_argument("--vq_softmin_beta", default=0.5, type=float)

    cli.add_argument("--lr_grd_Q", default="0.0001", type=str, help="sometimes start with lin")
    cli.add_argument("--lr_abs_V", default="0.0001", type=str, help="sometimes start with lin")
    cli.add_argument("--lr_curl", default="0.0001", type=str)
    cli.add_argument("--lr_vq", default="0.0001", type=str)
    cli.add_argument("--lr_decay", default=0.9999, type=float)
    cli.add_argument("--lr_min", default=0.01, type=float)

    cli.add_argument("--freq_grd_learn", default=1, type=int)
    cli.add_argument("--freq_grd_sync", default=1000, type=int)
    cli.add_argument("--freq_abs_learn", default=1, type=int)
    cli.add_argument("--freq_abs_sync", default=1000, type=int)
    cli.add_argument("--freq_curl_learn", default=1, type=int)
    cli.add_argument("--freq_curl_sync", default=1, type=int)

    cli.add_argument("--tau_grd_encoder", default=1, type=float)
    cli.add_argument("--tau_grd_critic", default=1, type=float)
    cli.add_argument("--tau_abs_encoder", default=1, type=float)
    cli.add_argument("--tau_abs_critic", default=1, type=float)
    cli.add_argument("--tau_curl", default=0.001, type=float)
    cli.add_argument("--tau_vq", default=0.001, type=float)

    cli.add_argument("--optimizer", default="rmsprop", type=str)
    cli.add_argument("--freq_eval", default=1e4, type=int)
    cli.add_argument("--evaluation_episodes", default=10, type=int)
    # cli.add_argument("--wandb_group_name", default=None, type=str)
    cli.add_argument("--wandb_tags", default=None, nargs="*", type=str)
    cli.add_argument("--wandb_mode", default="online", type=str)
    cli.add_argument("--extra_note", default="", type=str)
    cli.add_argument("--repetitions", default=20, type=int)
    cli.add_argument("--htcondor_procid", default=None, type=str)
    args = cli.parse_args()

    if not args.args_from_cli:
        # find_gpu()
        print("loading args from txt file...")
        with open(
            # "/workspace/repos_dev/VQVAE_RL/latentrl/htcondor_args/atari/abs_grd.txt",
            f"/workspace/repos_dev/VQVAE_RL/latentrl/htcondor_args/atari/temp.txt",
            "r",
        ) as f:
            for args_str in f:
                args = cli.parse_args(args_str.split())
                break
        # args.wandb_mode = "disabled"
    if args.use_curl == "off":
        args.use_curl = None
    pprint(args)
    return args


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
            # mode="disabled",
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
    # env_id = "MiniGrid-Empty-16x16-v0"
    # env_id = "CarRacing-v2"
    # CarRacing-v05, ALE/Skiing-v5, Boxing-v0, ALE/Freeway-v5, ALE/Pong-v5, ALE/Breakout-v5, BreakoutNoFrameskip-v4, RiverraidNoFrameskip-v4
    # vae_version = "vqvae_c3_embedding16x64_3_duolayer"

    # cfg_key = "MiniGrid-Empty-RGB"
    cfg_key = "CarRacing-v2"
    # load hyperparameters from yaml config file
    with open("/workspace/repos_dev/VQVAE_RL/hyperparams/dqn_ae.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)[cfg_key]
        pprint(cfg)
    for rep in range(8):
        current_time = datetime.datetime.now() + datetime.timedelta(hours=2)
        current_time = current_time.strftime("%b%d_%H-%M-%S")
        # 🐝 initialise a wandb run
        run = wandb.init(
            project="HDQN_Carracing",
            mode="disabled",
            group=cfg["wandb_group"],
            tags=[
                "tbl",
                "shp" if cfg["use_shaping"] else "no_shp",
                f"omg{cfg['omega']}",
            ],
            config=cfg,
        )
        config = wandb.config
        make_env = MAKE_ENV_FUNCS[config.env_type]
        env = make_env(config)

        # agent = HDQN_KMeans_VAE(config, env)
        # agent = HDQN_ManualAbs(config, env)
        agent = HDQN_KMeans_CURL(config, env)
        # agent.vis_abstraction(prefix=f"rep{rep}")
        goal_found = False
        steps_after_goal_found = 0
        episodes_since_goal_found = 0
        interval4SemiMDP = 0
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
            episodic_shaped_reward = 0
            for t in count():
                # Select and perform an action
                with utils.eval_mode(agent):
                    action = agent.act(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                agent.timesteps_done += 1
                if goal_found:
                    steps_after_goal_found += 1

                # abs_state1, abs_value1 = agent.get_abstract_value(info["agent_pos1"])
                # abs_state2, abs_value2 = agent.get_abstract_value(info["agent_pos2"])
                # interval4SemiMDP += 1
                # info["interval4SemiMDP"] = interval4SemiMDP
                # if not (abs_state1 == abs_state2 and reward == 0):
                #     # this conditino should match the one in update_absV
                #     interval4SemiMDP = 0
                # if abs_state1 != abs_state2:
                #     episodic_shaped_reward += config.gamma * abs_value2 - abs_value1

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
                if agent.timesteps_done >= config.init_steps:
                    agent.update(use_shaping=config.use_shaping)

                for i in range(5):
                    if agent.timesteps_done == (i + 1) * total_steps / 5:
                        # agent.vis_abstraction(prefix=f"rep{rep}")
                        break
                # agent.maybe_buffer_recent_states(state)

                state = next_state

                if terminated or truncated:
                    if terminated:
                        goal_found = True
                    print("sys.getsizeof(agent.memory)", sys.getsizeof(agent.memory))
                    print(torch.cuda.memory_reserved() / (1024 * 1024), "MB")
                    print(torch.cuda.memory_allocated() / (1024 * 1024), "MB")
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
                    print(f"***Episodic Reward: {episodic_reward}")
                    print(f"Goal_found: {goal_found}")

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

    print("Complete")
    env.close()


def train_manual_absT_grdTN():
    """
    abstract level using table
    ground level using table or network, by setting use_table4grd

    """
    # env_id = "MiniGrid-Empty-Random-6x6-v0"
    # env_id = "MiniGrid-Empty-16x16-v0"
    # env_id = "CarRacing-v2"
    # CarRacing-v05, ALE/Skiing-v5, Boxing-v0, ALE/Freeway-v5, ALE/Pong-v5, ALE/Breakout-v5, BreakoutNoFrameskip-v4, RiverraidNoFrameskip-v4
    # vae_version = "vqvae_c3_embedding16x64_3_duolayer"

    # cfg_key = "MiniGrid-Empty-RGB"
    cfg_key = "MiniGrid-Empty-v0"
    # load hyperparameters from yaml config file
    with open("/workspace/repos_dev/VQVAE_RL/hyperparams/minigrid/manual_abstraction.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)[cfg_key]
        pprint(cfg)
    for rep in range(30):
        print(f"====Starting Repetition {rep}====")
        current_time = datetime.datetime.now() + datetime.timedelta(hours=2)
        current_time = current_time.strftime("%b%d_%H-%M-%S")
        # 🐝 initialise a wandb run
        run = wandb.init(
            project="HDQN_AbsTable_GrdNN_Empty",
            # mode="disabled",
            group=cfg["wandb_group"],
            tags=[
                "tbl",
                "shp" if cfg["use_shaping"] else "no_shp",
                f"omg{cfg['omega']}",
                f"env{cfg['env_size']}x{cfg['env_size']}",
                f"start({cfg['agent_start_pos'][0]}, {cfg['agent_start_pos'][1]})",
            ],
            notes=cfg["wandb_notes"],
            config=cfg,
        )
        config = wandb.config
        make_env = MAKE_ENV_FUNCS[config.env_type]
        env = make_env(config, env_id=cfg_key)

        # agent = HDQN_KMeans_VAE(config, env)
        agent = HDQN_ManualAbs(config, env, use_table4grd=False)
        agent.set_abs_ticks(config, 0)
        # agent.vis_abstraction(prefix=f"rep{rep}")

        # comment = ""
        # log_dir_tensorboard = f"/workspace/repos_dev/VQVAE_RL/log_tensorboard/end2end_duolayer/{env_id}/{current_time}_{comment}"
        # tb_writer = SummaryWriter(log_dir_tensorboard)
        # tb_writer = None
        # print("log_dir_tensorboard:", log_dir_tensorboard)
        # print("reconstruction_path:", config.reconstruction_path)

        goal_found = False
        steps_after_goal_found = 0
        episodes_since_goal_found = 0
        interval4SemiMDP = 0
        time_start_training = time.time()
        # gym.reset(seed=int(time.time()))
        total_steps = int(config.total_timesteps + config.init_steps)
        while agent.timesteps_done < total_steps:
            time_start_episode = time.time()
            # Initialize the environment and state
            state, info = env.reset()
            episodic_reward = 0
            episodic_negative_reward = 0
            episodic_non_negative_reward = 0
            episodic_shaped_reward = 0
            # action = agent.act_table(info)
            agent.grd_visits = np.zeros((env.height, env.width, 4))
            for t in count():
                # [Select and perform an action]
                # action = agent.act_table(info)
                with utils.eval_mode(agent):
                    action = agent.act(state)
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
                    # shaping = config.gamma * abs_value2 - abs_value1
                    shaping = abs_value2 - abs_value1
                    episodic_shaped_reward += shaping
                    # agent.shaping_distribution[
                    #     info["agent_pos2"][1], info["agent_pos2"][0], info["agent_dir2"]
                    # ] += shaping
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
                # action_prime = agent.act_table(info)
                if agent.timesteps_done >= config.init_steps:
                    agent.update(use_shaping=config.use_shaping)
                    # here we use table to do update
                    # agent.update_table(use_shaping=config.use_shaping)
                    # agent.update_table_no_memory(
                    #     use_shaping=config.use_shaping, action_prime=action_prime
                    # )
                    # agent.update_table_abs_update_non_parallel2(use_shaping=config.use_shaping)
                # [visualization]
                for i in range(20):
                    if agent.timesteps_done == (i + 1) * total_steps / 20:
                        # agent.vis_grd_visits(norm_log=50)
                        # agent.vis_grd_visits(norm_log=0)
                        # agent.grd_visits = np.zeros_like(agent.grd_visits)
                        # agent.vis_shaping_distribution(norm_log=100)
                        # agent.vis_shaping_distribution(norm_log=0)
                        # agent.vis_grd_q_values(norm_log=100)
                        # agent.vis_grd_q_values(norm_log=1e8)
                        # agent.vis_abstract_values()
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
                    if agent.episodes_done > 0 and agent.episodes_done % 1 == 0:
                        # agent.vis_abstraction()
                        agent.vis_abstract_values()
                        agent.vis_grd_visits(norm_log=50)
                        agent.vis_grd_visits(norm_log=0)
                        agent.vis_shaping_distribution(norm_log=100)
                        agent.vis_shaping_distribution(norm_log=0)

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
            # if agent.episodes_done > 0 and agent.episodes_done % 3 == 0:
            #     agent.vis_abstract_values()
            #     print("Evaluate the agent by visualizing grd visits")
            #     # evaluate_agent(env, agent, exploit_only=True)
            #     # evaluate_agent(env, agent, exploit_only=False)
            #     evaluate_dqn_agent(env, agent)
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


def train_adaptive_absT_grdTN(args):
    """
    abstract level using table
    ground level using table or network, by setting use_table4grd

    """
    # env_id = "MiniGrid-Empty-Random-6x6-v0"
    # env_id = "MiniGrid-Empty-16x16-v0"
    # env_id = "CarRacing-v2"
    # CarRacing-v05, ALE/Skiing-v5, Boxing-v0, ALE/Freeway-v5, ALE/Pong-v5, ALE/Breakout-v5, BreakoutNoFrameskip-v4, RiverraidNoFrameskip-v4
    # vae_version = "vqvae_c3_embedding16x64_3_duolayer"

    # cfg_key = "MiniGrid-Empty-RGB"
    # cfg_key = "MiniGrid-Empty-v0"
    cfg_key = "MiniGrid-FourRooms-v0"
    # cfg_key = "MiniGrid-MultiRooms-v0"
    # cfg_key = "MiniGrid-Crossing-v0"
    # cfg_key = "MiniGrid-DistShift-v0"
    # cfg_key = "MiniGrid-Custom"
    # load hyperparameters from yaml config file
    # with open(
    #     "/workspace/repos_dev/VQVAE_RL/hyperparams/minigrid/adaptive_abstraction_vq.yaml"
    # ) as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)[cfg_key]
    #     pprint(cfg)
    external_rng = np.random.default_rng(seed=args.env_seed)
    fix_rn = external_rng.integers(1e3, 1e10, 30)
    for rep in range(args.repetitions):
        print(f"====Starting Repetition {rep}====")
        current_time = datetime.datetime.now() + datetime.timedelta(hours=2)
        current_time = current_time.strftime("%b%d_%H-%M-%S")
        # 🐝 initialise a wandb run
        project_name = args.domain_name.replace("/", "_")
        wandb.init(
            # project="HDQN_AbsTable_GrdNN_Empty",
            # project="HDQN_AbsTable_GrdNN_4Rooms",
            project=f"HDQN_{project_name}",
            mode=args.wandb_mode,
            group=args.wandb_group_name,
            tags=args.wandb_tags,
            # notes=cfg["wandb_notes"],
            config=vars(args),
        )
        wandb.run.log_code(".")
        L = LoggerWandb()
        env = MAKE_ENV_FUNCS[args.domain_type](args.domain_name, seed=args.env_seed)

        # agent = HDQN_AdaptiveAbs_VQ(config, env)
        agent = HDQN_TCURL_VQ(args, env, logger=L)
        if agent.use_vq:
            wandb.watch(agent.vq, log="all", log_freq=100, idx=0)
        # wandb.watch(agent.abs_V, log="all", log_freq=100, idx=1)
        # wandb.watch(agent.ground_Q, log="all", log_freq=100, idx=2)
        if agent.use_curl:
            wandb.watch(agent.curl, log="all", log_freq=100, idx=3)

        first_goal_found = False
        steps_after_goal_found = 0
        episodes_since_goal_found = 0
        interval4SemiMDP = 0
        time_start_training = time.time()
        # gym.reset(seed=int(time.time()))
        total_steps = int(args.total_timesteps + args.init_steps)
        # agent.cache_goal_transition()
        while agent.timesteps_done < total_steps:
            time_start_episode = time.time()
            # Initialize the environment and state
            state, info = env.reset()
            episodic_reward = 0
            episodic_combined_reward = 0
            episodic_intrinsic_reward = 0
            episodic_negative_reward = 0
            episodic_non_negative_reward = 0
            episodic_shaped_reward = 0
            # action = agent.act_table(info)
            agent.grd_visits = np.zeros((env.height, env.width, 4))
            agent.grd_reward_distribution = np.zeros((env.height, env.width, 4))
            agent.shaping_distribution = np.zeros((env.height, env.width, 4))
            for t in count():
                # [Select and perform an action]
                # action = agent.act_table(info)
                with utils.eval_mode(agent):
                    action = agent.act(state)
                if first_goal_found:
                    steps_after_goal_found += 1
                # [Step]
                next_state, reward, terminated, truncated, info = env.step(action)
                reward = info["original_reward"]
                episodic_reward += reward
                if reward < 0:
                    episodic_negative_reward += reward
                else:
                    episodic_non_negative_reward += reward
                if agent.use_curiosity:
                    int_reward = (
                        agent.rnd.compute_int_reward(state, agent.timesteps_done - args.init_steps)
                        .clamp_(-0.1, 0.1)
                        .item()
                    )
                    # int_reward = agent.vq_confidence(state)
                    episodic_intrinsic_reward += int_reward
                    reward += int_reward
                    episodic_combined_reward += reward

                agent.timesteps_done += 1

                # abs_state1, abs_value1 = agent.get_abstract_value(info["agent_pos1"])
                # abs_state2, abs_value2 = agent.get_abstract_value(info["agent_pos2"])
                # interval4SemiMDP += 1
                # info["interval4SemiMDP"] = interval4SemiMDP
                # if not (abs_state1 == abs_state2 and reward == 0):
                #     # this conditino should match the one in update_absV
                #     interval4SemiMDP = 0
                # if abs_state1 != abs_state2:
                #     shaping = config.gamma * abs_value2 - abs_value1
                #     episodic_shaped_reward += shaping
                #     agent.shaping_distribution[
                #         info["agent_pos2"][1], info["agent_pos2"][0], info["agent_dir2"]
                #     ] += shaping
                agent.update_grd_visits(info)
                agent.update_grd_rewards(info, reward)

                # for i in range(len(config.abs_ticks) - 1):
                #     if agent.timesteps_done == (i + 1) * total_steps / len(config.abs_ticks):
                #         agent.set_abs_ticks(config, i + 1)
                # if isinstance(env.unwrapped, MiniGridEnv):
                #     info["agent_pos"] = env.agent_pos
                #     info["agent_dir"] = env.agent_dir
                # print(env.agent_pos, env.agent_dir)
                # time.sleep(10)

                # [Store the transition in memory]
                shaping = agent.cache(state, action, next_state, reward, terminated, info)
                # shaping = agent.cache_ema(state, action, next_state, reward, terminated, info)
                # agent.cache_lazy(state, action, next_state, reward, terminated)
                episodic_shaped_reward += shaping

                # [update]
                # action_prime = agent.act_table(info)
                if agent.timesteps_done >= args.init_steps:
                    agent.update()
                    # here we use table to do update
                    # agent.update_table(use_shaping=config.use_shaping)
                    # agent.update_table_no_memory(
                    #     use_shaping=config.use_shaping, action_prime=action_prime
                    # )
                    # agent.update_table_abs_update_non_parallel2(use_shaping=config.use_shaping)
                # [visualization]
                for i in range(20):
                    if agent.timesteps_done == (i + 1) * total_steps / 20:
                        # agent.vis_grd_visits(norm_log=50)
                        # agent.vis_grd_visits(norm_log=0)
                        # agent.grd_visits = np.zeros_like(agent.grd_visits)
                        # agent.vis_shaping_distribution(norm_log=100)
                        # agent.vis_shaping_distribution(norm_log=0)
                        # agent.vis_grd_q_values(norm_log=100)
                        # agent.vis_grd_q_values(norm_log=1e8)
                        # agent.vis_abstract_values()
                        break
                # agent.maybe_buffer_recent_states(state)
                if agent.timesteps_done >= total_steps:
                    truncated = True

                state = next_state
                # action = action_prime

                if terminated or truncated:
                    agent.episodes_done += 1
                    if agent._current_progress_remaining > 0.8:
                        plot_every = 1
                    elif agent._current_progress_remaining > 0.6:
                        plot_every = 5
                    elif agent._current_progress_remaining > 0.4:
                        plot_every = 15
                    elif agent._current_progress_remaining > 0.2:
                        plot_every = 30

                    if agent.episodes_done > 0 and agent.episodes_done % 1 == 0:
                        agent.vis_grd_visits(norm_log=50)
                        # agent.vis_grd_visits(norm_log=0)
                        pass
                        if agent.timesteps_done >= args.init_steps:
                            agent.vis_grd_q_values(reduction_mode="max", norm_log=50)
                            # agent.vis_grd_q_values(reduction_mode="mean", norm_log=50)
                            # agent.vis_reward_distribution()
                            agent.vis_abstraction()
                            agent.vis_abstract_values(mode="soft_novq")
                            # agent.vis_abstract_values(mode="soft")
                            # agent.vis_abstract_values(mode="hard")
                            # agent.vis_abstract_values(mode="target_soft_novq")
                            # agent.vis_abstract_values(mode="target_soft")
                            # agent.vis_abstract_values(mode="target_hard")
                            # agent.vis_shaping_distribution(norm_log=100)
                            # agent.vis_shaping_distribution(norm_log=0)
                            pass

                    metrics = {
                        "Episodic/reward": episodic_reward,
                        "Episodic/negative_reward": episodic_negative_reward,
                        "Episodic/non_negative_reward": episodic_non_negative_reward,
                        "Episodic/shaped_reward": episodic_shaped_reward,
                        "Episodic/combined_reward": episodic_combined_reward,
                        "Episodic/intrinsic_reward": episodic_intrinsic_reward,
                        "reward/episodic_reward": episodic_reward,
                        "reward/episodic_negative_reward": episodic_negative_reward,
                        "reward/episodic_non_negative_reward": episodic_non_negative_reward,
                        "time/timesteps_done": agent.timesteps_done,
                        "Episodic/length": t + 1,
                        "Time/total_time_elapsed": (time.time() - time_start_training) / 3600,
                        "Time/fps_per_episode": int((t + 1) / (time.time() - time_start_episode)),
                    }
                    # if terminated:
                    #     first_goal_found = True
                    #     print("!!Goal Found!!")
                    #     agent.goal_found = True
                    # if goal_found:
                    #     episodes_since_goal_found += 1
                    #     metrics.update(
                    #         {
                    #             "After_1stTerminated/episode_length": t + 1,
                    #             "After_1stTerminated/episodic_reward": episodic_reward,
                    #             "After_1stTerminated/episodic_shaped_reward": episodic_shaped_reward,
                    #             "After_1stTerminated/episodes_done": episodes_since_goal_found,
                    #             "After_1stTerminated/timesteps_done": steps_after_goal_found,
                    #         }
                    #     )
                    L.log(metrics)
                    L.dump2wandb(agent=agent, force=True)

                    print2console(
                        agent=agent,
                        episodic_reward=episodic_reward,
                        terminated=terminated,
                        truncated=truncated,
                        t=t,
                        time_start_episode=time_start_episode,
                        rep=rep,
                    )

                    break
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


def evaluate_dqn_agent(env, agent: HDQN_ManualAbs):
    agent.grd_visits = np.zeros_like(agent.grd_visits)
    episodic_reward = 0
    timesteps_done = 0
    state, info = env.reset()
    for t in count():
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        timesteps_done += 1
        agent.update_grd_visits(info)
        episodic_reward += reward
        if terminated or truncated:
            break
    agent.vis_grd_visits(norm_log=50, suffix=f"eval")
    agent.vis_grd_visits(norm_log=0, suffix=f"eval")
    agent.grd_visits = np.zeros_like(agent.grd_visits)


def train_atari_absT_grdN(args):
    """
    abstract level using table
    ground level using table or network, by setting use_table4grd

    """
    cfg_key = "Atari"
    # cfg_key = "MinAtar/Breakout-v0"
    # cfg_key = "MinAtar/Asterix-v0"
    # cfg_key = "MinAtar/SpaceInvaders-v0"
    # cfg_key = "MinAtar/Freeway-v0"
    # cfg_key = "MinAtar/Seaquest-v0"

    # load hyperparameters from yaml config file
    # with open("/workspace/repos_dev/VQVAE_RL/hyperparams/atari/atari_soft_vq.yaml") as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)[cfg_key]
    #     pprint(cfg)
    # with open("/workspace/repos_dev/VQVAE_RL/hyperparams/minatar/minatar.yaml") as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)["MinAtar"]
    #     pprint(cfg)
    # with open("/workspace/repos_dev/VQVAE_RL/hyperparams/carracing.yaml") as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)["CarRacing-v2"]
    #     pprint(cfg)
    best_eval_reward = -np.inf
    best_train_reward = -np.inf
    for rep in range(args.repetitions):
        print(f"====Starting Repetition {rep}====")
        current_time = datetime.datetime.now() + datetime.timedelta(hours=2)
        current_time = current_time.strftime("%b%d_%H-%M-%S")
        # 🐝 initialise a wandb run
        project_name = args.domain_name
        curl_projection = 0 if args.curl_projection_dims == [-1] else 1
        if args.use_curl == None:
            curl_mode = "off"
        else:
            curl_mode = args.use_curl.replace("on_", "")

        group_name = f"A{int(args.use_abs_V)}_AEncD{int(args.abs_enc_detach)}_GEncD{int(args.grd_enc_detach)}_ShrEnc{int(args.share_encoder)}_Curl|{curl_mode},{args.curl_pair},P{curl_projection}|_VQ{int(args.use_vq)}|{args.num_vq_embeddings},{args.vq_softmin_beta},{int(args.critic_upon_vq)},{args.curl_vq_cfg}|_bs{args.batch_size}_ms{int(args.size_replay_memory/1000)}k_close{args.approach_abs_factor}|{args.extra_note}"

        run = wandb.init(
            # project="HDQN_AbsTable_GrdNN_Atari",
            project=f"HDQN_Atari_{project_name}",
            # project="HDQN_MinAtar",
            # project="HDQN_Neo_Carracing",
            mode=args.wandb_mode,
            group=group_name,
            tags=args.wandb_tags,
            # notes=cfg["wandb_notes"],
            config=vars(args),
        )
        # wandb.run.log_code(".")
        if not args.args_from_cli:
            log_dir_root = os.path.join(
                "/workspace/repos_dev/VQVAE_RL/results",
                args.domain_type,
                args.domain_name,
                group_name,
            )
        else:
            log_dir_root = os.path.join(
                "/storage/raid/xue/rlyuan/repos_dev/VQVAE_RL/results",
                args.domain_type,
                args.domain_name,
                group_name,
            )
        os.makedirs(os.path.join(log_dir_root, "best_models"), exist_ok=True)
        current_time = datetime.datetime.now()
        current_time = current_time.strftime("%b%d_%H-%M-%S")
        log_dir = os.path.join(log_dir_root, current_time)
        os.makedirs(log_dir, exist_ok=True)
        L = LoggerWandb(log_dir)
        env = MAKE_ENV_FUNCS[args.domain_type]("ALE/" + args.domain_name, seed=args.env_seed)

        # agent = HDQN_Pixel(config, env)
        agent = HDQN_TCURL_VQ(args, env, logger=L)
        if agent.use_vq:
            wandb.watch(agent.vq, log="all", log_freq=100, idx=0)
        # wandb.watch(agent.abs_V, log="all", log_freq=100, idx=1)
        # wandb.watch(agent.ground_Q, log="all", log_freq=100, idx=2)
        if agent.use_curl:
            wandb.watch(agent.curl, log="all", log_freq=100, idx=3)

        time_start_training = time.time()
        # gym.reset(seed=int(time.time()))
        total_steps = int(args.total_timesteps + args.init_steps)
        # agent.cache_goal_transition()
        episodic_reward_window = deque(maxlen=15)
        eval_rwd_window = deque(maxlen=6)
        ema_reward_list = []
        time_steps_list = []
        recent_dropping_episodes = 0
        while agent.timesteps_done < total_steps:
            time_start_episode = time.time()
            # Initialize the environment and state
            state, info = env.reset()
            episodic_reward = 0
            episodic_negative_reward = 0
            episodic_non_negative_reward = 0
            episodic_shaped_reward = 0
            for t in count():
                # [Select and perform an action]
                # with utils.eval_mode(agent):
                action = agent.act(state)
                # [Step]
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.timesteps_done += 1

                # abs_state1, abs_value1 = agent.get_abstract_value(info["agent_pos1"])
                # abs_state2, abs_value2 = agent.get_abstract_value(info["agent_pos2"])
                # interval4SemiMDP += 1
                # info["interval4SemiMDP"] = interval4SemiMDP
                # if not (abs_state1 == abs_state2 and reward == 0):
                #     # this conditino should match the one in update_absV
                #     interval4SemiMDP = 0
                # if abs_state1 != abs_state2:
                #     shaping = config.gamma * abs_value2 - abs_value1
                #     episodic_shaped_reward += shaping
                #     agent.shaping_distribution[
                #         info["agent_pos2"][1], info["agent_pos2"][0], info["agent_dir2"]
                #     ] += shaping

                # for i in range(len(config.abs_ticks) - 1):
                #     if agent.timesteps_done == (i + 1) * total_steps / len(config.abs_ticks):
                #         agent.set_abs_ticks(config, i + 1)
                # if isinstance(env.unwrapped, MiniGridEnv):
                #     info["agent_pos"] = env.agent_pos
                #     info["agent_dir"] = env.agent_dir
                # print(env.agent_pos, env.agent_dir)
                # time.sleep(10)

                # [Store the transition in memory]
                shaping = agent.cache(state, action, next_state, reward, terminated, info)
                # agent.cache_ema(state, action, next_state, reward, terminated, info)
                # agent.cache_lazy(state, action, next_state, reward, terminated)
                episodic_shaped_reward += shaping

                # reward = info["original_reward"]
                episodic_reward += reward
                if reward < 0:
                    episodic_negative_reward += reward
                else:
                    episodic_non_negative_reward += reward
                # [update]
                # action_prime = agent.act_table(info)
                if agent.timesteps_done >= args.init_steps:
                    agent.update()

                    if agent.timesteps_done % args.freq_eval == 0:
                        print("start eval ...")
                        avg_eval_rwd = test(agent, args, L)
                        eval_rwd_window.append(avg_eval_rwd)
                        print("eval end")
                    # here we use table to do update
                    # agent.update_table(use_shaping=config.use_shaping)
                    # agent.update_table_no_memory(
                    #     use_shaping=config.use_shaping, action_prime=action_prime
                    # )
                    # agent.update_table_abs_update_non_parallel2(use_shaping=config.use_shaping)

                # agent.maybe_buffer_recent_states(state)
                if agent.timesteps_done >= total_steps:
                    truncated = True

                state = next_state
                # action = action_prime

                if terminated or truncated:
                    agent.episodes_done += 1
                    episodic_reward_window.append(episodic_reward)
                    if agent.episodes_done > 0 and agent.episodes_done % 1 == 0:
                        if agent.timesteps_done > args.init_steps:
                            # agent.vis_abstraction()
                            # agent.vis_abstract_values()
                            # agent.vis_grd_visits(norm_log=50)
                            # agent.vis_grd_visits(norm_log=0)
                            # agent.vis_shaping_distribution(norm_log=100)
                            # agent.vis_shaping_distribution(norm_log=0)
                            pass

                    metrics = {
                        "Episodic/reward": episodic_reward,
                        "Episodic/negative_reward": episodic_negative_reward,
                        "Episodic/non_negative_reward": episodic_non_negative_reward,
                        "Episodic/shaped_reward": episodic_shaped_reward,
                        # "Episodic/ema_reward": mean(episodic_reward_window),
                        "reward/episodic_reward": episodic_reward,
                        "reward/episodic_negative_reward": episodic_negative_reward,
                        "reward/episodic_non_negative_reward": episodic_non_negative_reward,
                        "time/timesteps_done": agent.timesteps_done,
                        "Episodic/length": t + 1,
                        "Time/total_time_elapsed": (time.time() - time_start_training) / 3600,
                        "Time/fps_per_episode": int((t + 1) / (time.time() - time_start_episode)),
                        # "RND/beta_t": agent.rnd.beta_t,
                        # "Episodic/ema_reward_derivative": ema_reward_derivative,
                    }

                    L.log_and_dump(metrics, agent)
                    # L.dump2wandb(agent=agent, force=True)

                    print2console(
                        agent=agent,
                        episodic_reward=episodic_reward,
                        terminated=terminated,
                        truncated=truncated,
                        t=t,
                        time_start_episode=time_start_episode,
                        rep=rep,
                    )

                    break
        if mean(eval_rwd_window) > best_eval_reward:
            best_eval_reward = mean(eval_rwd_window)
            torch.save(agent.ground_Q.state_dict(), f"{log_dir_root}/best_models/eval_grd_q.pt")
            torch.save(agent.abs_V.state_dict(), f"{log_dir_root}/best_models/eval_abs_v.pt")
        if mean(episodic_reward_window) > best_train_reward:
            best_train_reward = mean(episodic_reward_window)
            torch.save(agent.ground_Q.state_dict(), f"{log_dir_root}/best_models/train_grd_q.pt")
            torch.save(agent.abs_V.state_dict(), f"{log_dir_root}/best_models/train_abs_v.pt")
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


def test(agent, args, L: LoggerWandb):
    env = MAKE_ENV_FUNCS[args.domain_type]("ALE/" + args.domain_name)
    # env.eval()

    episodic_rews = []
    episodic_non_negative_rews = []
    episodic_negative_rews = []

    # Test performance over several episodes
    terminated = True
    for _ in range(args.evaluation_episodes):
        while True:
            if terminated:
                state, info = env.reset()
                reward_sum = 0
                non_negative_reward_sum = 0
                negative_reward_sum = 0
                terminated = False

            action = agent.act_e_greedy(state, epsilon=0.01)  # Choose an action ε-greedily
            state, reward, terminated, truncated, info = env.step(action)  # Step
            reward_sum += reward
            if reward >= 0:
                non_negative_reward_sum += reward
            else:
                negative_reward_sum += reward

            if terminated:
                episodic_rews.append(reward_sum)
                episodic_non_negative_rews.append(non_negative_reward_sum)
                episodic_negative_rews.append(negative_reward_sum)
                break
    env.close()
    if len(episodic_non_negative_rews) == 0:
        episodic_non_negative_rews.append(0)
    if len(episodic_negative_rews) == 0:
        episodic_negative_rews.append(0)
    avg_reward = sum(episodic_rews) / len(episodic_rews)
    avg_non_negative_reward = sum(episodic_non_negative_rews) / len(episodic_non_negative_rews)
    avg_negative_reward = sum(episodic_negative_rews) / len(episodic_negative_rews)
    # Return average reward and Q-value

    metrics = {
        "Evaluation/avg_episodic_reward": avg_reward,
        "Evaluation/avg_episodic_non_negative_reward": avg_non_negative_reward,
        "Evaluation/avg_episodic_negative_reward": avg_negative_reward,
        "Evaluation/timesteps_done": agent.timesteps_done,
        "Evaluation/episodes_done": agent.episodes_done,
    }
    L.log_and_dump(metrics, agent, mode="eval")

    return avg_reward


def print2console(agent, episodic_reward, terminated, truncated, t, time_start_episode, rep):
    print(f"===========Episode {agent.episodes_done} Done| Repetition {rep}=====")
    print("[Total_steps_done]:", agent.timesteps_done)
    print(f"[Episode {agent.episodes_done} Reward]: {episodic_reward}")
    print("[Exploration_rate]:", agent.exploration_rate)
    print("[Episodic_fps]:", int((t + 1) / (time.time() - time_start_episode)))
    print("[Episodic time cost]: {:.1f} s".format(time.time() - time_start_episode))
    print("[Episodic timesteps]: {} ".format(t + 1))
    print(f"[Terminal:{terminated} | Truncated: {truncated}]")
    print("[Current_progress_remaining]:", agent._current_progress_remaining)
    print(f"[wandb run name]: {wandb.run.project}/{wandb.run.group}/{wandb.run.name}")


def find_gpu():
    # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Get the first available GPU
    DEVICE_ID_LIST = GPUtil.getAvailable(
        order="random",
        limit=4,
        maxLoad=0.8,
        maxMemory=0.85,
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
    # print("sys.path:", sys.path)

    # from utils.gpu_profile import gpu_profile
    # find_gpu()
    # sys.settrace(gpu_profile)
    # torch.set_num_threads(1)
    # torch.autograd.set_detect_anomaly(True)
    # tracemalloc.start()
    # set number of threads to 1, when using T.ToTensor() it will cause very high cpu usage and using milti-threads
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No GPU available, aborting")
        raise SystemExit
    # train_hdqn()
    # train_dqn_kmeans()
    # train_manual_absT_grdTN()
    # train_adaptive_absT_grdTN(args)
    train_atari_absT_grdN(args)

    # env = make_env_minigrid(env_id="MiniGrid-Empty-6x6-v0")
    # print(env.observation_space.shape)

    # import gymnasium

    # env = gymnasium.make("MiniGrid-LavaCrossingS11N5-v0", render_mode="human")
    # env.reset()
    # for _ in range(100000):
    #     action = env.action_space.sample()
    #     # print("action: ", action)
    #     # print("[Before Step] env.agent_pos, env.agent_dir: ", env.agent_pos, env.agent_dir)
    #     next_state, reward, terminated, truncated, info = env.step(action)
    #     state = next_state
    #     if terminated or truncated:
    #         state, info = env.reset()
