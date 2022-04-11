import argparse
import copy
import sys
import os
import GPUtil
# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Get the first available GPU
DEVICE_ID_LIST = GPUtil.getAvailable(order = 'random', limit = 4, maxLoad = 0.5, maxMemory = 0.5,
                                     includeNan=False, excludeID=[], excludeUUID=[])
assert len(DEVICE_ID_LIST) > 0, "no availible cuda currently"
print("availible CUDAs:", DEVICE_ID_LIST)
DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list
# os.environ["DISPLAY"] = ":199"
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

from os.path import join, exists
from os import mkdir, makedirs, getpid
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
from stable_baselines3.common.logger import TensorBoardOutputFormat

from hparams import HyperParams as hp

import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary

from latentrl.custom_callbacks import HparamsWriterCallback, EarlyStopCallback, TrainingRewardWriterCallback, \
    TrainingRewardWriterCallback_both, EpisodeCounterCallback, MyRewardWriterCallback, SaveOnBestTrainingRewardCallback
from models.vae import VAE
from wrappers import LatentWrapper, NaiveWrapper, ShapingWrapper, PreprocessObservationWrapper, EncodeWrapper, \
    ShapeRewardWrapper, VecShapeReward, VecShapeReward2, EncodeStackWrapper, ShapeRewardStackWrapper, \
    CarRandomStartWrapper, ActionDiscreteWrapper, FrameStackWrapper, EpisodeEarlyStopWrapper, PunishRewardWrapper
from wrappers import pack_env_wrappers
# from my_callbacks import ImageRecorderCallback
from transforms import transform_dict
from utils.misc import ProcessFrame, linear_schedule, make_vec_env_customized

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize
from stable_baselines3 import A2C, SAC, PPO, TD3, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed, safe_mean
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init

def process_hparams_for_model(hparams:Dict, algo_class, env_id):

    temp_hparams = {}
    if isinstance(hparams['learning_rate'], str) and hparams['learning_rate'].startswith('lin'):
        temp_hparams.update(learning_rate=linear_schedule(eval(hparams['learning_rate'].split('_')[1])))

    pi_config = eval(hparams['policy_kwargs'].split('-')[1])
    qf_config = eval(hparams['policy_kwargs'].split('-')[-1])
    temp_hparams.update(policy_kwargs=dict(net_arch=dict(pi=pi_config, qf=qf_config)))

    temp_hparams.update(tensorboard_log=f"../tensorboard_log/{algo_class.__name__}_{env_id}/")
    temp_hparams.update(verbose=1)

    hparams.update(temp_hparams)

    return hparams
# Example:
# class HPARAMS:
#     learning_rate = 'lin_7.3e-4'
#     buffer_size = 1_000_000
#     learning_starts = 1000
#     batch_size = 256
#     tau = 0.02
#     gamma = 0.98
#     train_freq = 8
#     gradient_steps = 8
#     use_sde = False
#     use_sde_at_warmup = False
#     # set the structure of policy and critic network.
#     policy_kwargs = f"pi-{[256, 256, 256]}-qf-{[256, 256, 256]}"
#
# hparams = {key: value for key, value in HPARAMS.__dict__.items() if not key.startswith('__') and not callable(key)}
# _hparams = process_hparams_for_model(copy.deepcopy(hparams), SAC, env_id)

def train_vanilla_discrete():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    algo_class = DQN # DQN
    policy = 'CnnPolicy'
    env_id = 'CarRacing-v0'
    total_time_steps = 1_000_000  # int(3e6)
    n_envs = 1
    n_stack = 4
    stack_mode = 'gym_stack'  # or 'venv_stack or 'gym_stack' or None
    eval_freq = 50
    action_repetition = 3
    max_neg_rewards = 100
    punishment = -20



    time_tag = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time()))
    run_name = f"vanilla_discrete_{algo_class.__name__}_{time_tag}_normobs"
    monitor_dir = f"../log_monitor/{run_name}"

    always_random_start = True
    no_random_start = True

    save_final_model = True
    seed = int(time.time())



    hparams = {
        "learning_rate": linear_schedule(7.3e-4),  # linear_schedule(7.3e-4), 0.0003, 4e-4(John)
        "buffer_size": 100_000,
        "learning_starts": 10_000,  # 100_000   #10_000(John)
        "batch_size": 128,
        "tau": 1,  # 0.02/0.005/1
        "gamma": 0.97,  #0.95(John)
        "train_freq": 4,  # or int(8/n_envs) ?, 4(John)
        "gradient_steps": 1,  # -1, 1(John)
        "target_update_interval": 8, #update target per 10000 steps in John, 1
        "exploration_fraction": 0.99,
        "exploration_initial_eps":  0.1,
        "exploration_final_eps": 0.01,
        # "tensorboard_log": f"../tensorboard_log/{algo_class.__name__}_{env_id}_5/",
        "tensorboard_log": f"../tensorboard_log/{env_id}/",
        "policy_kwargs": dict(net_arch=[256, 256, 256]),
        # "policy_kwargs": dict(share_features_extractor=False, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
        # "policy_kwargs": dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
        # "policy_kwargs": dict(net_arch=dict(pi=[512, 512], qf=[512, 512])),
        "verbose": 1,
        # "accelerate_warmup": False  # only for warmup stage
    }

    # @@prepare stuff for wappers
    if not stack_mode or stack_mode == 'venv_stack':
        wrapper_class_list = [
            ActionDiscreteWrapper,
            EpisodeEarlyStopWrapper,
            CarRandomStartWrapper,
            PreprocessObservationWrapper,
        ]
        wrapper_kwargs_list = [
            {},
            {},
            {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
             'always_random_start': always_random_start, 'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': 3},

        ]
        wrapper_class_list_eval = [
            ActionDiscreteWrapper,
            EpisodeEarlyStopWrapper,
            # CarRandomStartWrapper,
            PreprocessObservationWrapper,
        ]
        wrapper_kwargs_list_eval = [
            {},
            {},
            # {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
            #  'always_random_start': always_random_start,  'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': 3}
        ]

    elif stack_mode == 'gym_stack':
        wrapper_class_list = [
            ActionDiscreteWrapper,
            EpisodeEarlyStopWrapper,
            Monitor,
            CarRandomStartWrapper,
            # Monitor,
            PreprocessObservationWrapper,
            FrameStackWrapper,
            # PunishRewardWrapper,
        ]
        wrapper_kwargs_list = [
            {'action_repetition': action_repetition},   #ActionDiscreteWrapper
            {'max_neg_rewards': max_neg_rewards, 'punishment': punishment},   #EpisodeEarlyStopWrapper
            {'filename': monitor_dir},  #Monitor
            {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
             'always_random_start': always_random_start, 'no_random_start': no_random_start},   #CarRandomStartWrapper
            # {'filename': monitor_dir},    #Monitor
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': 1}, #PreprocessObservationWrapper
            {'n_stack': n_stack},   #FrameStackWrapper
            # {'max_neg_rewards': max_neg_rewards, "punishment": punishment}  #PunishRewardWrapper
        ]

        wrapper_class_list_eval = [
            ActionDiscreteWrapper,
            # EpisodeEarlyStopWrapper,
            Monitor,
            # CarRandomStartWrapper,
            # Monitor,
            PreprocessObservationWrapper,
            FrameStackWrapper
        ]
        wrapper_kwargs_list_eval = [
            {'action_repetition': action_repetition},
            # {'max_neg_rewards': max_neg_rewards, 'punishment': punishment},
            {'filename': monitor_dir},
            # {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
            #  'always_random_start': always_random_start,  'no_random_start': no_random_start},
            # {'filename': monitor_dir},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': 1},
            {'n_stack': n_stack}
        ]
    else:
        wrapper_class_list = []
        wrapper_kwargs_list = []
        wrapper_class_list_eval = []
        wrapper_kwargs_list_eval = []

    # env = make_vec_env(env_id, n_envs, seed=seed, monitor_dir=monitor_dir,
    #                    wrapper_class=pack_env_wrappers(wrapper_class_list, wrapper_kwargs_list),
    #                    # vec_env_cls=SubprocVecEnv
    #                    )
    # eval_env = make_vec_env(env_id, 1, seed=seed + 1, monitor_dir=monitor_dir,
    #                         wrapper_class=pack_env_wrappers(wrapper_class_list_eval, wrapper_kwargs_list_eval),
    #                         # vec_env_cls=SubprocVecEnv
    #                         )
    env = make_vec_env_customized(env_id, n_envs, seed=seed,
                                  wrapper_class=pack_env_wrappers(wrapper_class_list, wrapper_kwargs_list),
                                  # vec_env_cls=SubprocVecEnv
                            )
    eval_env = make_vec_env_customized(env_id, 1, seed=seed + 1,
                                       wrapper_class=pack_env_wrappers(wrapper_class_list_eval, wrapper_kwargs_list_eval),
                                       # vec_env_cls=SubprocVecEnv
                            )
    # env = VecNormalize(env)
    # eval_env = VecNormalize(eval_env)

    if stack_mode == 'venv_stack' and n_stack > 1:
        print("Using VecFrameStack VenvWrapper")
        env = VecFrameStack(env, n_stack)
        eval_env = VecFrameStack(eval_env, n_stack)

    set_random_seed(seed + 2)

    extra_hparams = dict(stack_config=f"{stack_mode}({n_stack})",
                         omega='no_need',
                         total_time_steps=total_time_steps,
                         vae_inchannel='no_need',
                         vae_latent_dim='no_need',
                         vae_sample='no_need',
                         always_random_start=always_random_start,
                         no_random_start=no_random_start,
                         max_neg_rewards=max_neg_rewards,
                         punishment=punishment,
                         action_repetition=action_repetition
                         )
    # os.makedirs('../tensorboard_log/', exist_ok=True)

    # @@custom config
    model = DQN(policy, env, **hparams)  # CnnPolicy or MlpPolicy
    # @@default config
    # model = SAC(policy, env, policy_kwargs=_hparams['policy_kwargs'], verbose=1,
    #             tensorboard_log=f"../tensorboard_log/{algo_class.__name__}_{env_id}/",
    #             device=device)  # CnnPolicy or MlpPolicy

    # @@load model in order to continue training
    # model = SAC.load("/workspace/VQVAE_RL/log_evaluation/CarRacing-v0_latent_SAC_02-03-21-47/best_model",
    #                  tensorboard_log="CarRacing-v0_latent_SAC_02-03-21-47_1", verbose=1)
    # env = make_vec_env(env_id, 4, seed=seed, monitor_dir=monitor_dir,
    #                    wrapper_class=pack_wrappers(wrapper_class_list, wrapper_kwargs_list),
    #                    vec_env_cls=SubprocVecEnv
    #                    )
    # model.set_env(env)

    # @@prepare callbacks
    hp_callback = HparamsWriterCallback(run_name, hparams, extra_hparams)
    # org_rwd_callback = TrainingRewardWriterCallback_both(stack_mode=stack_mode)
    myreward_callback = MyRewardWriterCallback(average_window_size=5)
    callback = CallbackList([hp_callback, myreward_callback])

    # or
    # callback = HparamsWriterCallback(run_name, hparams, extra_hparams)

    model.learn(total_time_steps, tb_log_name=run_name, eval_env=eval_env,
                eval_freq=int(total_time_steps / (n_envs * eval_freq)),
                eval_log_path=f"../log_evaluation/{run_name}", callback=callback)
    # model.learn(total_time_steps, tb_log_name=run_name, callback=callback)

    # bit dirty to write hyperparameters into tensorboard when learning finished
    # for outputformat in model.logger.output_formats:
    #     if isinstance(outputformat, TensorBoardOutputFormat):
    #         torch_summary_writer = outputformat.writer
    #         torch_summary_writer.add_hparams(hparams, {'hparam/accuracy': 10}, run_name='.')

    if save_final_model:
        model.save(f"../saved_final_model/{run_name}")

def train_vanilla_continuous():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    algo_class = DQN # DQN
    policy = 'CnnPolicy'
    env_id = 'CarRacing-v0'
    total_time_steps = 2_000_000  # int(3e6)
    n_envs = 1
    n_stack = 4
    stack_mode = 'gym_stack'  # or 'venv_stack or None
    eval_freq = 50


    time_tag = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time()))
    run_name = f"vanilla_conti_{algo_class.__name__}_{time_tag}SAC"

    always_random_start = False
    no_random_start = False

    save_final_model = True
    seed = int(time.time())

    hparams = {
        "learning_rate": linear_schedule(7.3e-4),  # linear_schedule(7.3e-4), 0.0003
        "buffer_size": 1_000_000,
        "learning_starts": 100_000,  # 100_000
        "batch_size": 256,
        "tau": 0.02,  # 0.02/0.005
        "gamma": 0.98,
        "train_freq": 1,  # or int(8/n_envs) ?
        "gradient_steps": -1,  # -1
        "ent_coef": "auto",  # "auto",0.2
        "target_update_interval": 1,
        # "target_entropy": "auto",
        "use_sde": False,
        "use_sde_at_warmup": False,
        "tensorboard_log": f"../tensorboard_log/{algo_class.__name__}_{env_id}_5/",
        "policy_kwargs": dict(share_features_extractor=True, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
        # "policy_kwargs": dict(share_features_extractor=False, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
        # "policy_kwargs": dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
        # "policy_kwargs": dict(net_arch=dict(pi=[512, 512], qf=[512, 512])),
        "verbose": 1,
        "accelerate_warmup": False  # only for warmup stage
    }

    # @@prepare stuff for wappers
    if not stack_mode or stack_mode == 'venv_stack':
        wrapper_class_list = [
            CarRandomStartWrapper,
            PreprocessObservationWrapper,
        ]
        wrapper_kwargs_list = [
            {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
             'always_random_start': always_random_start, 'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': 3},

        ]
    elif stack_mode == 'gym_stack':
        wrapper_class_list = [
            CarRandomStartWrapper,
            PreprocessObservationWrapper,
        ]
        wrapper_kwargs_list = [
            {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
             'always_random_start': always_random_start, 'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': 3},
        ]

        wrapper_class_list_eval = [
            # CarRandomStartWrapper,
            PreprocessObservationWrapper,
        ]
        wrapper_kwargs_list_eval = [
            # {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
            #  'always_random_start': always_random_start,  'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': 3},
        ]
    else:
        wrapper_class_list = []
        wrapper_kwargs_list = []
        wrapper_class_list_eval = []
        wrapper_kwargs_list_eval = []

    monitor_dir = f"../log_monitor/{run_name}"
    env = make_vec_env(env_id, n_envs, seed=seed, monitor_dir=monitor_dir,
                       wrapper_class=pack_env_wrappers(wrapper_class_list, wrapper_kwargs_list),
                       # vec_env_cls=SubprocVecEnv
                       )
    eval_env = make_vec_env(env_id, 1, seed=seed + 1, monitor_dir=monitor_dir,
                            wrapper_class=pack_env_wrappers(wrapper_class_list_eval, wrapper_kwargs_list_eval),
                            # vec_env_cls=SubprocVecEnv
                            )
    if stack_mode == 'venv_stack' and n_stack > 1:
        print("Using VecFrameStack VenvWrapper")
        env = VecFrameStack(env, n_stack)
        eval_env = VecFrameStack(eval_env, n_stack)

    set_random_seed(seed + 2)

    extra_hparams = dict(stack_config=f"{stack_mode}({n_stack})",
                         omega='no_need',
                         total_time_steps=total_time_steps,
                         vae_inchannel='no_need',
                         vae_latent_dim='no_need',
                         vae_sample='no_need',
                         always_random_start=always_random_start,
                         no_random_start=no_random_start)
    # os.makedirs('../tensorboard_log/', exist_ok=True)

    # @@custom config
    model = SAC(policy, env, **hparams)  # CnnPolicy or MlpPolicy
    # @@default config
    # model = SAC(policy, env, policy_kwargs=_hparams['policy_kwargs'], verbose=1,
    #             tensorboard_log=f"../tensorboard_log/{algo_class.__name__}_{env_id}/",
    #             device=device)  # CnnPolicy or MlpPolicy

    # @@load model in order to continue training
    # model = SAC.load("/workspace/VQVAE_RL/log_evaluation/CarRacing-v0_latent_SAC_02-03-21-47/best_model",
    #                  tensorboard_log="CarRacing-v0_latent_SAC_02-03-21-47_1", verbose=1)
    # env = make_vec_env(env_id, 4, seed=seed, monitor_dir=monitor_dir,
    #                    wrapper_class=pack_wrappers(wrapper_class_list, wrapper_kwargs_list),
    #                    vec_env_cls=SubprocVecEnv
    #                    )
    # model.set_env(env)

    # @@prepare callbacks
    # hp_callpack = HparamsWriterCallback()
    # early_stop_callback = EarlyStopCallback(reward_threshold=0, progress_remaining_threshold=0.4)
    # callback = CallbackList([hp_callpack, early_stop_callback])
    # or
    callback = HparamsWriterCallback(run_name, extra_hparams)

    model.learn(total_time_steps, tb_log_name=run_name, eval_env=eval_env,
                eval_freq=int(total_time_steps / (n_envs * eval_freq)), n_eval_episodes=10,
                eval_log_path=f"../log_evaluation/{run_name}", callback=callback)
    # model.learn(total_time_steps, tb_log_name=run_name, callback=callback)

    # bit dirty to write hyperparameters into tensorboard when learning finished
    # for outputformat in model.logger.output_formats:
    #     if isinstance(outputformat, TensorBoardOutputFormat):
    #         torch_summary_writer = outputformat.writer
    #         torch_summary_writer.add_hparams(hparams, {'hparam/accuracy': 10}, run_name='.')

    if save_final_model:
        model.save(f"../saved_final_model/{run_name}")

def train_latent():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    # make vec_envs # first way to achieve wrapping
    # load vae
    # best_filename = join(vae_path, 'best.tar')
    # vae_model = VAE(3, 128)  #  latent_size: 128
    # device_vae = torch.device('cpu')
    # vae_model.load_state_dict(torch.load(best_filename, map_location=device_vae)['state_dict'])
    # # vae_model.to(device)
    # vae_model.eval()
    #
    # env = make_vec_env(env_id, 1, seed=seed, monitor_dir=monitor_dir, wrapper_class=LatentWrapper,
    #                    wrapper_kwargs={'encoder': vae_model.encoder,
    #                                    'process_frame': ProcessFrame(vertical_cut=84, resize=(64,64))})
    #
    #
    # eval_env = make_vec_env(env_id, 1, seed=seed+1, monitor_dir=monitor_dir, wrapper_class=LatentWrapper,
    #                         wrapper_kwargs={'encoder': vae_model.encoder,
    #                                         'process_frame': ProcessFrame(vertical_cut=84, resize=(64,64))})
    algo_class = SAC
    policy = 'MlpPolicy'
    env_id = 'CarRacing-v0'
    total_time_steps = 2_000_000  #int(3e6)
    n_envs = 1
    n_stack = 4
    stack_mode = 'gym_stack'  # or 'venv_stack or None
    eval_freq = 50
    vae_inchannel = 1
    vae_latent_dim = 32

    time_tag = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time()))
    run_name = f"latent_{algo_class.__name__}_{time_tag}_alwaysRS_AC"
    # vae_path = "/workspace/VQVAE_RL/logdir/beta3_vae32_channel3" #beta3_vae32_channel3
    vae_path = "/workspace/VQVAE_RL/logdir/beta3_vae32_channel1"
    # vae_path = "/workspace/VQVAE_RL/logdir/vae"

    vae_sample = True
    always_random_start = True
    no_random_start = True

    save_final_model = True
    seed = int(time.time())


    hparams = {
        "learning_rate": linear_schedule(7.3e-4), #linear_schedule(7.3e-4), 0.0003
        "buffer_size": 1_000_000,
        "learning_starts": 100_000,# 100_000
        "batch_size": 256,
        "tau": 0.02, #0.02/0.005
        "gamma": 0.98,
        "train_freq": 1, #or int(8/n_envs) ?
        "gradient_steps": -1, # -1
        "ent_coef": "auto",   # "auto",0.2
        "target_update_interval": 1,
        # "target_entropy": "auto",
        "use_sde": False,
        "use_sde_at_warmup": False,
        "tensorboard_log": f"../tensorboard_log/{algo_class.__name__}_{env_id}_5/",
        "policy_kwargs": dict(share_features_extractor=True, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
        # "policy_kwargs": dict(share_features_extractor=False, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
        # "policy_kwargs": dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
        # "policy_kwargs": dict(net_arch=dict(pi=[512, 512], qf=[512, 512])),
        "verbose": 1,
        # "accelerate_warmup": False # only for warmup stage
    }
    # make vec_envs # second way to achieve wrapping, this should be more beautiful
    # because following the style of stacking wrappers one by one

    # @@prepare stuff for wappers
    if not stack_mode or stack_mode == 'venv_stack':
        wrapper_class_list = [
            CarRandomStartWrapper,
            PreprocessObservationWrapper,
            EncodeWrapper
        ]
        wrapper_kwargs_list = [
            {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
             'always_random_start': always_random_start,  'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': vae_inchannel},
            {'vae_f': vae_path, 'vae_sample': vae_sample,
             "vae_inchannel": vae_inchannel, "latent_dim": vae_latent_dim}
        ]
    elif stack_mode == 'gym_stack':
        wrapper_class_list = [
            CarRandomStartWrapper,
            PreprocessObservationWrapper,
            EncodeStackWrapper
        ]
        wrapper_kwargs_list = [
            {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
             'always_random_start': always_random_start, 'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': vae_inchannel},
            {'n_stack': n_stack, 'vae_f': vae_path, 'vae_sample': vae_sample,
              "vae_inchannel": vae_inchannel, "latent_dim": vae_latent_dim}
        ]

        wrapper_class_list_eval = [
            # CarRandomStartWrapper,
            PreprocessObservationWrapper,
            EncodeStackWrapper
        ]
        wrapper_kwargs_list_eval = [
            # {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
            #  'always_random_start': always_random_start,  'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': vae_inchannel},
            {'n_stack': n_stack, 'vae_f': vae_path, 'vae_sample': vae_sample,
             "vae_inchannel": vae_inchannel, "latent_dim": vae_latent_dim}
        ]
    else:
        wrapper_class_list = []
        wrapper_kwargs_list = []
        wrapper_class_list_eval = []
        wrapper_kwargs_list_eval = []

    monitor_dir = f"../log_monitor/{run_name}"
    env = make_vec_env(env_id, n_envs, seed=seed, monitor_dir=monitor_dir,
                       wrapper_class=pack_env_wrappers(wrapper_class_list, wrapper_kwargs_list),
                       # vec_env_cls=SubprocVecEnv
                       )
    eval_env = make_vec_env(env_id, 1, seed=seed+1, monitor_dir=monitor_dir,
                            wrapper_class=pack_env_wrappers(wrapper_class_list_eval, wrapper_kwargs_list_eval),
                            # vec_env_cls=SubprocVecEnv
                            )
    if stack_mode == 'venv_stack' and n_stack > 1:
        print("Using VecFrameStack VenvWrapper")
        env = VecFrameStack(env, n_stack)
        eval_env = VecFrameStack(eval_env, n_stack)

    set_random_seed(seed+2)

    extra_hparams = dict(stack_config=f"{stack_mode}({n_stack})",
                         omega='no_need',
                         total_time_steps=total_time_steps,
                         vae_inchannel = vae_inchannel,
                         vae_latent_dim = vae_latent_dim,
                         vae_sample = vae_sample,
                         always_random_start = always_random_start,
                         no_random_start=no_random_start)
    # os.makedirs('../tensorboard_log/', exist_ok=True)

    # @@custom config
    model = SAC(policy, env, **hparams)  # CnnPolicy or MlpPolicy
    # @@default config
    # model = SAC(policy, env, policy_kwargs=_hparams['policy_kwargs'], verbose=1,
    #             tensorboard_log=f"../tensorboard_log/{algo_class.__name__}_{env_id}/",
    #             device=device)  # CnnPolicy or MlpPolicy

    # @@load model in order to continue training
    # model = SAC.load("/workspace/VQVAE_RL/log_evaluation/CarRacing-v0_latent_SAC_02-03-21-47/best_model",
    #                  tensorboard_log="CarRacing-v0_latent_SAC_02-03-21-47_1", verbose=1)
    # env = make_vec_env(env_id, 4, seed=seed, monitor_dir=monitor_dir,
    #                    wrapper_class=pack_wrappers(wrapper_class_list, wrapper_kwargs_list),
    #                    vec_env_cls=SubprocVecEnv
    #                    )
    # model.set_env(env)

    # @@prepare callbacks
    # hp_callpack = HparamsWriterCallback()
    # early_stop_callback = EarlyStopCallback(reward_threshold=0, progress_remaining_threshold=0.4)
    # callback = CallbackList([hp_callpack, early_stop_callback])
    # or
    callback = HparamsWriterCallback(run_name, extra_hparams)

    model.learn(total_time_steps, tb_log_name=run_name, eval_env=eval_env,
                eval_freq=int(total_time_steps / (n_envs * eval_freq)),
                eval_log_path=f"../log_evaluation/{run_name}", callback=callback)
    # model.learn(total_time_steps, tb_log_name=run_name, callback=callback)

    # bit dirty to write hyperparameters into tensorboard when learning finished
    # for outputformat in model.logger.output_formats:
    #     if isinstance(outputformat, TensorBoardOutputFormat):
    #         torch_summary_writer = outputformat.writer
    #         torch_summary_writer.add_hparams(hparams, {'hparam/accuracy': 10}, run_name='.')

    if save_final_model:
        model.save(f"../saved_final_model/{run_name}")


def train_latent2():
    # cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if cuda else "cpu")
    algo_class = DQN
    policy = 'MlpPolicy'    #MlpPolicy/CnnPolicy
    env_id = 'CarRacing-v0'
    total_time_steps = 1_000_000  # int(3e6)
    n_envs = 1
    n_stack = 4
    stack_mode = 'gym_stack'  # or 'venv_stack or None
    eval_freq = 50
    vae_inchannel = 1
    vae_latent_dim = 32
    action_repetition = 3
    max_neg_rewards = 100
    punishment = -20.0


    time_tag = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time()))
    run_name = f"latent_{algo_class.__name__}_{time_tag}_discrete_vq_c1"
    # vae_path = "/workspace/VQVAE_RL/logdir/beta3_vae32_channel3" #beta3_vae32_channel3
    # vae_path = "/workspace/VQVAE_RL/logdir/beta3_vae32_channel1"
    # vae_path = "/workspace/VQVAE_RL/logdir/vae"
    # vae_path = "/workspace/VQVAE_RL/logdir/vqvae"
    # vae_path = "/workspace/VQVAE_RL/logdir/vqvae"
    # vae_path = "/workspace/VQVAE_RL/logdir/vqvae_c3_embedding16x64"
    vae_path = "/workspace/VQVAE_RL/logdir/vqvae_c1_embedding16x64"
    monitor_dir = f"../log_monitor/{run_name}"

    vae_sample = True

    always_random_start = True
    no_random_start = True

    save_final_model = True
    seed = int(time.time())

    hparams = {
        "learning_rate": linear_schedule(4e-4),  # linear_schedule(7.3e-4)/0.0003/4e-4(John)
        "buffer_size": 100_000,
        "learning_starts": 10_000,  # 100_000
        "batch_size": 128,
        "tau": 1,  # 0.02/0.005/1/1e-2(John)
        "gamma": 0.97,
        "train_freq": 4,  # or int(8/n_envs) ?
        "gradient_steps": 1,  # -1
        "target_update_interval": 8,
        "exploration_fraction": 0.99,
        "exploration_initial_eps": 0.1,
        "exploration_final_eps": 0.01,
        # "tensorboard_log": f"../tensorboard_log/{algo_class.__name__}_{env_id}_5/",
        "tensorboard_log": f"../tensorboard_log/{env_id}/",
        "policy_kwargs": dict(net_arch=[256, 256, 256]),
        "verbose": 1,
        # "accelerate_warmup": False  # only for warmup stage
    }
    # make vec_envs # second way to achieve wrapping, this should be more beautiful
    # because following the style of stacking wrappers one by one

    # @@prepare stuff for wappers
    if not stack_mode or stack_mode == 'venv_stack':
        wrapper_class_list = [
            CarRandomStartWrapper,
            PreprocessObservationWrapper,
            EncodeWrapper
        ]
        wrapper_kwargs_list = [
            {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
             'always_random_start': always_random_start, 'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': vae_inchannel},
            {'vae_f': vae_path, 'vae_sample': vae_sample,
             "vae_inchannel": vae_inchannel, "latent_dim": vae_latent_dim}
        ]
    elif stack_mode == 'gym_stack':

        wrapper_class_list = [
            ActionDiscreteWrapper,
            EpisodeEarlyStopWrapper,
            Monitor,
            CarRandomStartWrapper,
            PreprocessObservationWrapper,
            EncodeStackWrapper,
            # PunishRewardWrapper,
        ]
        wrapper_kwargs_list = [
            {'action_repetition': action_repetition},
            {'max_neg_rewards': max_neg_rewards, 'punishment': punishment},
            # {'filename': monitor_dir},
            {'filename': os.path.join(monitor_dir, 'train')}, #just single env in this case
            {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
             'always_random_start': always_random_start, 'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': vae_inchannel},
            {'n_stack': n_stack, 'vae_f': vae_path, 'vae_sample': vae_sample,
             "vae_inchannel": vae_inchannel, "latent_dim": vae_latent_dim},
            # {'max_neg_rewards': max_neg_rewards, "punishment": punishment}
        ]

        wrapper_class_list_eval = [
            ActionDiscreteWrapper,
            # EpisodeEarlyStopWrapper,
            Monitor,
            # CarRandomStartWrapper,
            PreprocessObservationWrapper,
            EncodeStackWrapper
        ]
        wrapper_kwargs_list_eval = [
            {'action_repetition': action_repetition},
            # {'max_neg_rewards': max_neg_rewards, 'punishment': punishment},
            # {'filename': monitor_dir},
            {'filename': os.path.join(monitor_dir, 'eval')}, #just single env in this case
            # {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
            #  'always_random_start': always_random_start, 'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': vae_inchannel},
            {'n_stack': n_stack, 'vae_f': vae_path, 'vae_sample': vae_sample,
             "vae_inchannel": vae_inchannel, "latent_dim": vae_latent_dim}
        ]
    else:
        wrapper_class_list = []
        wrapper_kwargs_list = []
        wrapper_class_list_eval = []
        wrapper_kwargs_list_eval = []

    # env = make_vec_env(env_id, n_envs, seed=seed, monitor_dir=monitor_dir,
    #                    wrapper_class=pack_env_wrappers(wrapper_class_list, wrapper_kwargs_list),
    #                    # vec_env_cls=SubprocVecEnv
    #                    )
    # eval_env = make_vec_env(env_id, 1, seed=seed + 1, monitor_dir=monitor_dir,
    #                         wrapper_class=pack_env_wrappers(wrapper_class_list_eval, wrapper_kwargs_list_eval),
    #                         # vec_env_cls=SubprocVecEnv
    #                         )

    env = make_vec_env_customized(env_id, n_envs, seed=seed, monitor_dir=monitor_dir,
                                  wrapper_class=pack_env_wrappers(wrapper_class_list, wrapper_kwargs_list),
                                  # vec_env_cls=SubprocVecEnv
                                  )
    eval_env = make_vec_env_customized(env_id, 1, seed=seed + 1, monitor_dir=monitor_dir,
                                       wrapper_class=pack_env_wrappers(wrapper_class_list_eval,
                                                                       wrapper_kwargs_list_eval),
                                       # vec_env_cls=SubprocVecEnv
                                       )

    if stack_mode == 'venv_stack' and n_stack > 1:
        print("Using VecFrameStack VenvWrapper")
        env = VecFrameStack(env, n_stack)
        eval_env = VecFrameStack(eval_env, n_stack)

    set_random_seed(seed + 2)

    extra_hparams = dict(stack_config=f"{stack_mode}({n_stack})",
                         omega='no_need',
                         total_time_steps=total_time_steps,
                         vae_inchannel=vae_inchannel,
                         vae_latent_dim=vae_latent_dim,
                         vae_sample=vae_sample,
                         always_random_start=always_random_start,
                         no_random_start=no_random_start,
                         max_neg_rewards=max_neg_rewards,
                         punishment=punishment,
                         action_repetition=action_repetition
                         )
    # os.makedirs('../tensorboard_log/', exist_ok=True)

    # @@custom config
    model = DQN(policy, env, **hparams)  # CnnPolicy or MlpPolicy
    # @@default config
    # model = SAC(policy, env, policy_kwargs=_hparams['policy_kwargs'], verbose=1,
    #             tensorboard_log=f"../tensorboard_log/{algo_class.__name__}_{env_id}/",
    #             device=device)  # CnnPolicy or MlpPolicy

    # @@load model in order to continue training
    # model = SAC.load("/workspace/VQVAE_RL/log_evaluation/CarRacing-v0_latent_SAC_02-03-21-47/best_model",
    #                  tensorboard_log="CarRacing-v0_latent_SAC_02-03-21-47_1", verbose=1)
    # env = make_vec_env(env_id, 4, seed=seed, monitor_dir=monitor_dir,
    #                    wrapper_class=pack_wrappers(wrapper_class_list, wrapper_kwargs_list),
    #                    vec_env_cls=SubprocVecEnv
    #                    )
    # model.set_env(env)

    # @@prepare callbacks
    hp_callpack = HparamsWriterCallback(run_name, hparams, extra_hparams)
    myreward_callback = MyRewardWriterCallback(average_window_size=5)
    # early_stop_callback = EarlyStopCallback(reward_threshold=0, progress_remaining_threshold=0.4)
    save_best_on_training_back = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=monitor_dir)
    callback = CallbackList([hp_callpack, myreward_callback, save_best_on_training_back])
    # or
    # callback = HparamsWriterCallback(run_name, hparams, extra_hparams)

    model.learn(total_time_steps, tb_log_name=run_name, eval_env=eval_env,
                eval_freq=int(total_time_steps / (n_envs * eval_freq)), n_eval_episodes=10,
                eval_log_path=f"../log_evaluation/{run_name}", callback=callback)
    # model.learn(total_time_steps, tb_log_name=run_name, callback=callback)

    # bit dirty to write hyperparameters into tensorboard when learning finished
    # for outputformat in model.logger.output_formats:
    #     if isinstance(outputformat, TensorBoardOutputFormat):
    #         torch_summary_writer = outputformat.writer
    #         torch_summary_writer.add_hparams(hparams, {'hparam/accuracy': 10}, run_name='.')

    if save_final_model:
        model.save(f"../saved_final_model/{run_name}")


def train_shaping():
    # cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if cuda else "cpu")

    # policy_kwargs = dict(share_features_extractor=True, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))
    # policy_kwargs = dict(features_extractor_class=NatureCNN, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))
    # features_extractor_class: NatureCNN(CNN from DQN paper, without Maxpooling)
    algo_class = SAC
    policy = 'CnnPolicy'
    env_id = 'CarRacing-v0'
    total_time_steps = 1_000_000  # int(3e6)
    omega = 50
    n_envs = 1
    n_stack = 4
    stack_mode = 'gym_stack'  # 'gym_stack' or 'venv_stack' or None
    eval_freq = 50
    vae_inchannel = 1
    vae_latent_dim = 32

    time_tag = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time()))
    run_name = f"shaping_{algo_class.__name__}_{time_tag}_t"

    # vae_path = "/workspace/VQVAE_RL/logdir/vae"
    vae_path = "/workspace/VQVAE_RL/logdir/beta3_vae32_channel1"

    # latent_model_path = "/workspace/VQVAE_RL/log_evaluation/CarRacing-v0_latent_SAC_02-07-10-59-48_e1/best_model"
    # latent_model_path = "/workspace/VQVAE_RL/log_evaluation/CarRacing-v0_latent_SAC_02-08-19-15-20/best_model"
    # latent_model_path = "/workspace/VQVAE_RL/log_evaluation/CarRacing-v0_latent_SAC_02-08-19-15-20/best_model"

    #latent model trained on gym_stack mode
    # latent_model_path = "/workspace/VQVAE_RL/log_evaluation/CarRacing-v0_latent_SAC_02-10-20-57-29/best_model"
    latent_model_path = "/workspace/VQVAE_RL/log_evaluation/latent_SAC_02-21-17-07-13_alwaysRS_AC/best_model"


    vae_sample = True
    latent_deterministic = True
    always_random_start = True
    no_random_start = False

    save_final_model = True
    seed = int(time.time())

    hparams = {
        "learning_rate": linear_schedule(7.3e-4),
        "buffer_size": 1_000_000,
        "learning_starts": 100_000,
        "batch_size": 256,
        "tau": 0.02,
        "gamma": 0.98,
        "train_freq": 1, # time steps for each rollout(def collect_rollouts): n_envs * train_freq
        "gradient_steps": -1,
        "ent_coef": "auto",  # "auto",0.2
        "target_update_interval": 1,
        # "target_entropy": "auto",
        "accelerate_warmup": True, # only for warmup stage
        "use_sde": False,
        "use_sde_at_warmup": False,
        "tensorboard_log": f"../tensorboard_log/{algo_class.__name__}_{env_id}_5/",
        "policy_kwargs": dict(share_features_extractor=True, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
        # "policy_kwargs": dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
        # "policy_kwargs": dict(net_arch=dict(pi=[256,256], qf=[256,256])),
        "verbose": 1,
    }

    if not stack_mode:
        wrapper_class_list = [PreprocessObservationWrapper, ShapeRewardWrapper]
        wrapper_kwargs_list = [{'vertical_cut_d': 84, 'shape': 64},
                               {'vae_f': vae_path, 'latent_model_f': latent_model_path}
                               ]
    elif stack_mode == 'gym_stack':
        wrapper_class_list = [CarRandomStartWrapper, PreprocessObservationWrapper, ShapeRewardStackWrapper]
        wrapper_kwargs_list = [
            {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
             'always_random_start': always_random_start, 'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': vae_inchannel},
            {'n_stack': n_stack, 'gamma': hparams['gamma'], 'omega': omega,
             'vae_f': vae_path, "vae_inchannel": vae_inchannel, "latent_dim": vae_latent_dim, 'vae_sample': vae_sample,
             'latent_model_f': latent_model_path, 'latent_deterministic': latent_deterministic,
             'train': True}
        ]
        wrapper_class_list_eval = [
            # CarRandomStartWrapper,
            PreprocessObservationWrapper,
            ShapeRewardStackWrapper]
        wrapper_kwargs_list_eval = [
            # {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
            #  'always_random_start': always_random_start, 'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': vae_inchannel},
            {'n_stack': n_stack, 'gamma': hparams['gamma'], 'omega': omega,
             'vae_f': vae_path, "vae_inchannel": vae_inchannel, "latent_dim": vae_latent_dim, 'vae_sample': vae_sample,
             'latent_model_f': latent_model_path, 'latent_deterministic': latent_deterministic,
             'train': False}
        ]
    elif stack_mode == 'venv_stack':
        wrapper_class_list = [PreprocessObservationWrapper]
        wrapper_kwargs_list = [{'vertical_cut_d': 84, 'shape': 64}
                               ]
        wrapper_kwargs_list_eval = [{'vertical_cut_d': 84, 'shape': 64}
                               ]
    print("+++++")
    monitor_dir = f"../log_monitor/{run_name}"
    env = make_vec_env(env_id, n_envs, seed=seed, monitor_dir=monitor_dir,
                       wrapper_class=pack_env_wrappers(wrapper_class_list, wrapper_kwargs_list),
                       # vec_env_cls=SubprocVecEnv
                       )
    eval_env = make_vec_env(env_id, 1, seed=seed+1, monitor_dir=monitor_dir,
                            wrapper_class=pack_env_wrappers(wrapper_class_list_eval, wrapper_kwargs_list_eval),
                            # vec_env_cls=SubprocVecEnv
                            )
    print("+++++")
    if stack_mode == 'venv_stack':
        print("Using VecFrameStack and VecShapeReward2 as VenvWrapper")
        env = VecFrameStack(env, n_stack)
        env = VecShapeReward2(env, gamma=hparams['gamma'], omega=omega,
                              vae_f=vae_path, latent_model_f=latent_model_path,
                              vae_sample=vae_sample, latent_deterministic=latent_deterministic,
                              get_potential_mode='max',train=True)
        eval_env = VecFrameStack(eval_env, n_stack)
        eval_env = VecShapeReward2(eval_env, vae_f=vae_path, latent_model_f=latent_model_path,
                                   vae_sample=vae_sample, latent_deterministic=latent_deterministic,
                                   get_potential_mode='max', train=False)

    set_random_seed(seed+2)
    # os.makedirs('../tensorboard_log/', exist_ok=True)

    model = SAC(policy, env, **hparams)  # CnnPolicy or MlpPolicy
    # model = SAC(policy, env, policy_kwargs=_hparams['policy_kwargs'], verbose=1,
    #             tensorboard_log=f"../tensorboard_log/{algo_class.__name__}_{env_id}/",
    #             device=device)  # CnnPolicy or MlpPolicy

    # load model in order to continue training
    # model = SAC.load("/workspace/VQVAE_RL/log_evaluation/CarRacing-v0_latent_SAC_02-03-21-47/best_model",
    #                  tensorboard_log="CarRacing-v0_latent_SAC_02-03-21-47_1", verbose=1)
    # env = make_vec_env(env_id, 4, seed=seed, monitor_dir=monitor_dir,
    #                    wrapper_class=pack_wrappers(wrapper_class_list, wrapper_kwargs_list),
    #                    vec_env_cls=SubprocVecEnv
    #                    )
    # model.set_env(env)


    extra_hparams = dict(stack_config=f"{stack_mode}({n_stack})",
                         omega=omega,
                         total_time_steps=total_time_steps,
                         vae_inchannel=vae_inchannel,
                         vae_latent_dim=vae_latent_dim,
                         vae_sample=vae_sample,
                         always_random_start=always_random_start,
                         no_random_start=no_random_start
                         )
    # prepare callbacks
    hp_callback = HparamsWriterCallback(run_name, extra_hparams)
    org_rwd_callback = TrainingRewardWriterCallback_both(stack_mode=stack_mode)
    # early_stop_callback = EarlyStopCallback(reward_threshold=0, progress_remaining_threshold=0.4)
    callback = CallbackList([hp_callback, org_rwd_callback])
    # or
    # callback = HparamsWriterCallback(run_name, extra_hparams)

    model.learn(total_time_steps, tb_log_name=run_name, eval_env=eval_env,
                eval_freq=int(total_time_steps / (n_envs * eval_freq)), eval_log_path=f"../log_evaluation/{run_name}",
                callback=callback)
    # model.learn(total_time_steps, tb_log_name=run_name, callback=callback)

    if save_final_model:
        model.save(f"../saved_final_model/{run_name}")

def train_shaping2():
    algo_class = DQN
    policy = 'CnnPolicy'
    env_id = 'CarRacing-v0'
    total_time_steps = 1_000_000  # int(3e6)
    omega = 2.5e-3
    n_envs = 1
    n_stack = 4
    stack_mode = 'gym_stack'  # or 'venv_stack or None
    eval_freq = 50
    vae_inchannel = 1
    vae_latent_dim = 32
    action_repetition = 3
    max_neg_rewards = 100
    punishment = -20.0
    vae_type = 'vae'

    time_tag = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time()))
    run_name = f"shaping_{algo_class.__name__}_{time_tag}_vae_discrete"
    # vae_path = "/workspace/VQVAE_RL/logdir/vae"
    vae_path = "/workspace/VQVAE_RL/logdir/beta3_vae32_channel1"
    # vae_path = "/workspace/VQVAE_RL/logdir/vqvae_c1_embedding16x64"
    monitor_dir = f"../log_monitor/{run_name}"

    # latent_model_path = "/workspace/VQVAE_RL/log_evaluation/CarRacing-v0_latent_SAC_02-07-10-59-48_e1/best_model"
    # latent_model_path = "/workspace/VQVAE_RL/log_evaluation/CarRacing-v0_latent_SAC_02-08-19-15-20/best_model"
    # latent_model_path = "/workspace/VQVAE_RL/log_evaluation/CarRacing-v0_latent_SAC_02-08-19-15-20/best_model"

    #latent model trained on gym_stack mode
    # latent_model_path = "/workspace/VQVAE_RL/log_evaluation/CarRacing-v0_latent_SAC_02-10-20-57-29/best_model"
    # latent_model_path = "/workspace/VQVAE_RL/log_evaluation/latent_SAC_02-21-17-07-13_alwaysRS_AC/best_model"
    latent_model_path = "/workspace/VQVAE_RL/saved_final_model/latent_DQN_03-08-19-44-57_discrete.zip"
    # latent_model_path = "/workspace/VQVAE_RL/saved_final_model/latent_DQN_04-06-19-11-05_discrete_vq_c1.zip"


    vae_sample = True
    latent_deterministic = True
    always_random_start = True
    no_random_start = True

    save_final_model = True
    seed = int(time.time())

    hparams = {
        "learning_rate": linear_schedule(4e-4),  # linear_schedule(7.3e-4), 0.0003, 0.0004(John)
        "buffer_size": 100_000,
        "learning_starts": 10_000,  # 100_000   #10_000(John)
        "batch_size": 128,
        "tau": 1,  # 0.02/0.005/1
        "gamma": 0.97,  # 0.95(John)
        "train_freq": 4,  # or int(8/n_envs) ?, 4(John)
        "gradient_steps": 1,  # -1, 1(John)
        "target_update_interval": 8,  # update target per 10000 steps in John, 1
        "exploration_fraction": 0.99,
        "exploration_initial_eps": 0.1,
        "exploration_final_eps": 0.01,
        # "tensorboard_log": f"../tensorboard_log/{algo_class.__name__}_{env_id}_5/",
        "tensorboard_log": f"../tensorboard_log/{env_id}/",
        "policy_kwargs": dict(net_arch=[256, 256, 256]),
        # "policy_kwargs": dict(share_features_extractor=False, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
        # "policy_kwargs": dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
        # "policy_kwargs": dict(net_arch=dict(pi=[512, 512], qf=[512, 512])),
        "verbose": 1,
        # "accelerate_warmup": False  # only for warmup stage
    }

    if not stack_mode:
        wrapper_class_list = [PreprocessObservationWrapper, ShapeRewardWrapper]
        wrapper_kwargs_list = [{'vertical_cut_d': 84, 'shape': 64},
                               {'vae_f': vae_path, 'latent_model_f': latent_model_path}
                               ]
    elif stack_mode == 'gym_stack':
        wrapper_class_list = [
            ActionDiscreteWrapper,
            EpisodeEarlyStopWrapper,
            Monitor,
            CarRandomStartWrapper,
            PreprocessObservationWrapper,
            ShapeRewardStackWrapper,
            # PunishRewardWrapper,
        ]
        wrapper_kwargs_list = [
            {'action_repetition': action_repetition},
            {'max_neg_rewards': max_neg_rewards, 'punishment': punishment},
            {'filename': os.path.join(monitor_dir, 'train')},
            {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
             'always_random_start': always_random_start, 'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': vae_inchannel},
            {'n_stack': n_stack, 'vae_type':vae_type, 'gamma': hparams['gamma'], 'omega': omega,
             'vae_f': vae_path, "vae_inchannel": vae_inchannel, "latent_dim": vae_latent_dim, 'vae_sample': vae_sample,
             'latent_model_f': latent_model_path, 'latent_deterministic': latent_deterministic,
             'train': True, 'latent_model_class':DQN}
        ]
        wrapper_class_list_eval = [
            ActionDiscreteWrapper,
            # EpisodeEarlyStopWrapper,
            Monitor,
            # CarRandomStartWrapper,
            PreprocessObservationWrapper,
            ShapeRewardStackWrapper
        ]
        wrapper_kwargs_list_eval = [
            {'action_repetition': action_repetition},
            # {'max_neg_rewards': max_neg_rewards, 'punishment': punishment},
            {'filename': os.path.join(monitor_dir, 'eval')},
            # {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
            #  'always_random_start': always_random_start, 'no_random_start': no_random_start},
            {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': vae_inchannel},
            {'n_stack': n_stack, 'vae_type':vae_type, 'gamma': hparams['gamma'], 'omega': omega,
             'vae_f': vae_path, "vae_inchannel": vae_inchannel, "latent_dim": vae_latent_dim, 'vae_sample': vae_sample,
             'latent_model_f': latent_model_path, 'latent_deterministic': latent_deterministic,
             'train': False, 'latent_model_class':DQN}
        ]
    elif stack_mode == 'venv_stack':
        wrapper_class_list = [PreprocessObservationWrapper]
        wrapper_kwargs_list = [{'vertical_cut_d': 84, 'shape': 64}
                               ]
        wrapper_kwargs_list_eval = [{'vertical_cut_d': 84, 'shape': 64}
                               ]
    # env = make_vec_env(env_id, n_envs, seed=seed, monitor_dir=monitor_dir,
    #                    wrapper_class=pack_env_wrappers(wrapper_class_list, wrapper_kwargs_list),
    #                    # vec_env_cls=SubprocVecEnv
    #                    )
    # eval_env = make_vec_env(env_id, 1, seed=seed+1, monitor_dir=monitor_dir,
    #                         wrapper_class=pack_env_wrappers(wrapper_class_list_eval, wrapper_kwargs_list_eval),
    #                         # vec_env_cls=SubprocVecEnv
    #                         )
    env = make_vec_env_customized(env_id, n_envs, seed=seed, monitor_dir=monitor_dir,
                                  wrapper_class=pack_env_wrappers(wrapper_class_list, wrapper_kwargs_list),
                                  # vec_env_cls=SubprocVecEnv
                                  )
    eval_env = make_vec_env_customized(env_id, 1, seed=seed + 1, monitor_dir=monitor_dir,
                                       wrapper_class=pack_env_wrappers(wrapper_class_list_eval,
                                                                       wrapper_kwargs_list_eval),
                                       # vec_env_cls=SubprocVecEnv
                                       )

    if stack_mode == 'venv_stack':
        print("Using VecFrameStack and VecShapeReward2 as VenvWrapper")
        env = VecFrameStack(env, n_stack)
        env = VecShapeReward2(env, gamma=hparams['gamma'], omega=omega,
                              vae_f=vae_path, latent_model_f=latent_model_path,
                              vae_sample=vae_sample, latent_deterministic=latent_deterministic,
                              get_potential_mode='max',train=True)
        eval_env = VecFrameStack(eval_env, n_stack)
        eval_env = VecShapeReward2(eval_env, vae_f=vae_path, latent_model_f=latent_model_path,
                                   vae_sample=vae_sample, latent_deterministic=latent_deterministic,
                                   get_potential_mode='max', train=False)

    set_random_seed(seed+2)
    # os.makedirs('../tensorboard_log/', exist_ok=True)

    model = DQN(policy, env, **hparams)  # CnnPolicy or MlpPolicy
    # model = SAC(policy, env, policy_kwargs=_hparams['policy_kwargs'], verbose=1,
    #             tensorboard_log=f"../tensorboard_log/{algo_class.__name__}_{env_id}/",
    #             device=device)  # CnnPolicy or MlpPolicy

    # load model in order to continue training
    # model = SAC.load("/workspace/VQVAE_RL/log_evaluation/CarRacing-v0_latent_SAC_02-03-21-47/best_model",
    #                  tensorboard_log="CarRacing-v0_latent_SAC_02-03-21-47_1", verbose=1)
    # env = make_vec_env(env_id, 4, seed=seed, monitor_dir=monitor_dir,
    #                    wrapper_class=pack_wrappers(wrapper_class_list, wrapper_kwargs_list),
    #                    vec_env_cls=SubprocVecEnv
    #                    )
    # model.set_env(env)
    extra_hparams = dict(stack_config=f"{stack_mode}({n_stack})",
                         omega=omega,
                         total_time_steps=total_time_steps,
                         vae_inchannel=vae_inchannel,
                         vae_latent_dim=vae_latent_dim,
                         vae_sample=vae_sample,
                         always_random_start=always_random_start,
                         no_random_start=no_random_start,
                         max_neg_rewards=max_neg_rewards,
                         punishment=punishment,
                         action_repetition=action_repetition
                         )

    # prepare callbacks
    hp_callback = HparamsWriterCallback(run_name, hparams, extra_hparams)
    # org_rwd_callback = TrainingRewardWriterCallback_both(stack_mode=stack_mode)
    myreward_callback = MyRewardWriterCallback(average_window_size=5)
    # early_stop_callback = EarlyStopCallback(reward_threshold=0, progress_remaining_threshold=0.4)
    callback = CallbackList([hp_callback, myreward_callback])
    # or
    # callback = HparamsWriterCallback(run_name, extra_hparams)

    model.learn(total_time_steps, tb_log_name=run_name, eval_env=eval_env,
                eval_freq=int(total_time_steps / (n_envs * eval_freq)), n_eval_episodes=10,
                eval_log_path=f"../log_evaluation/{run_name}",
                callback=callback)
    # model.learn(total_time_steps, tb_log_name=run_name, callback=callback)

    if save_final_model:
        model.save(f"../saved_final_model/{run_name}")

def train_vanilla(algo_class, policy: str, env_id: str, total_time_steps: int, device=None, seed: int =123):
    env = make_vec_env(env_id, 3, seed=seed)
    # env = VecFrameStack(env, 4)
    # print(env.action_space)
    eval_env = make_vec_env(env_id, 1, seed=seed+1)
    # eval_env = VecFrameStack(eval_env, 4)

    set_random_seed(seed)
    os.makedirs('./tensorboard_log/', exist_ok=True)

    model = algo_class(policy, env, verbose=1,
                       # tensorboard_log=f"../tensorboard_log/{algo_class.__name__}_{env_id}/",
                       # device=device
                       )  # CnnPolicy
    model.learn(total_time_steps,
                # tb_log_name=f'vanilla_ts{total_time_steps:.0e}',
                eval_env=eval_env,
                eval_freq=int(total_time_steps / 50))

def run_model():
    algo_class = DQN  # DQN
    policy = 'CnnPolicy'
    env_id = 'CarRacing-v0'
    total_time_steps = 1_000_000  # int(3e6)
    n_envs = 1
    n_stack = 4
    stack_mode = 'gym_stack'  # or 'venv_stack or 'gym_stack' or None
    eval_freq = 50
    max_neg_rewards = 100
    action_repetition = 3

    time_tag = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time()))
    run_name = f"vanilla_discrete_{algo_class.__name__}_{time_tag}"
    monitor_dir = f"../log_monitor/{run_name}"

    always_random_start = True
    no_random_start = True

    save_final_model = True
    seed = int(time.time())

    hparams = {
        "learning_rate": 0.0004,  # linear_schedule(7.3e-4), 0.0003, 0.0004(John)
        "buffer_size": 100_000,
        "learning_starts": 10_000,  # 100_000   #10_000(John)
        "batch_size": 128,
        "tau": 1,  # 0.02/0.005/1
        "gamma": 0.95,  # 0.95(John)
        "train_freq": 4,  # or int(8/n_envs) ?, 4(John)
        "gradient_steps": 1,  # -1, 1(John)
        "target_update_interval": 1,  # update target per 10000 steps in John
        "exploration_fraction": 0.9,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        # "tensorboard_log": f"../tensorboard_log/{algo_class.__name__}_{env_id}_5/",
        "tensorboard_log": f"../tensorboard_log/{env_id}/",
        "policy_kwargs": dict(net_arch=[256, 256, 256]),
        # "policy_kwargs": dict(share_features_extractor=False, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
        # "policy_kwargs": dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
        # "policy_kwargs": dict(net_arch=dict(pi=[512, 512], qf=[512, 512])),
        "verbose": 1,
        # "accelerate_warmup": False  # only for warmup stage
    }

    wrapper_class_list = [
        ActionDiscreteWrapper,
        EpisodeEarlyStopWrapper,
        # Monitor,
        # CarRandomStartWrapper,
        # Monitor,
        PreprocessObservationWrapper,
        FrameStackWrapper
    ]
    wrapper_kwargs_list = [
        {'action_repetition': action_repetition},
        {'max_neg_rewards': max_neg_rewards},
        # {'filename': monitor_dir},
        # {'warm_up_steps': hparams['learning_starts'], 'n_envs': n_envs,
        #  'always_random_start': always_random_start, 'no_random_start': no_random_start},
        # {'filename': monitor_dir},
        {'vertical_cut_d': 84, 'shape': 64, 'num_output_channels': 1},
        {'n_stack': n_stack}
    ]

    env = make_vec_env_customized(env_id, n_envs, seed=seed,
                                  wrapper_class=pack_env_wrappers(wrapper_class_list, wrapper_kwargs_list),
                                  # vec_env_cls=SubprocVecEnv
                                  )

    # model = DQN.load("/workspace/VQVAE_RL/saved_final_model/vanilla_discrete_DQN_03-05-16-31-37.zip", env=env)
    model = DQN.load('/workspace/VQVAE_RL/log_evaluation/vanilla_discrete_DQN_03-05-16-31-37/best_model.zip', env=env)
    # env = model.env
    # Enjoy trained agent
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if dones[0] == True:
            obs = env.reset()
        env.render()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Trainer')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--logdir', type=str, default='/workspace/VQVAE_RL/logdir/', help='Directory where results are logged')
    parser.add_argument('--noreload', action='store_true',
                        help='Best model is not reloaded if specified')
    parser.add_argument('-o2f', '--output2file', action='store_true',
                        help='print to a file when using nohup')
    args = parser.parse_args()

    print("torch.cuda.device_count()", torch.cuda.device_count())
    print("torch.cuda.current_device()", torch.cuda.current_device())

    # train_vanilla(algo_class=SAC, policy='CnnPolicy', env_id='CarRacing-v0', total_time_steps=int(1e6),
    #               device=device, seed=int(time.time()))

    # train_latent()
    # train_shaping()
    # train_vanilla_continuous()
    # train_vanilla_discrete()
    # train_latent2()
    train_shaping2()
    # train_vanilla(DQN, 'MlpPolicy', 'LunarLander-v2', 10000)
    # run_model()





    # env = gym.make('CarRacing-v0')
    # env = Monitor(LatentWrapper(env, 1, vae_model.encoder, transform_easy))
    # env = DummyVecEnv([lambda: env])
    #
    # eval_env = gym.make('CarRacing-v0')
    # eval_env = Monitor(LatentWrapper(eval_env, 0, vae_model.encoder, transform_easy))
    # eval_env = DummyVecEnv([lambda: eval_env])

    # Create action noise because TD3 and DDPG use a deterministic policy
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # model = SAC("MlpPolicy", 'CarRacing-v0', action_noise=None, verbose=1, tensorboard_log="./tensorboard_log/td3_carracing_tensorboard/")
    # model.learn(2000)
    # model.learn(10000, tb_log_name="first_run")
    # model.learn(2000, eval_env=eval_env, eval_freq=50, n_eval_episodes=5)



