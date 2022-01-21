
import numpy as np
import multiprocessing as mp
import gym
import os
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d.car_racing import CarRacing

# env = CarRacing()
# env = gym.make('CarRacing-v0')
# observation = env.reset()
import torch
import gym
import Box2D
import math
import numpy as np
# env = gym.make("CartPole-v1")
# env = gym.make('CarRacing-v0')
# print(type(env))
# print("env._max_episode_steps:", env._max_episode_steps)
# action = env.action_space.sample()
# print(action)
# observation1 = env.reset()
# observation2 = env.reset()
# env.reset()
# print(np.argwhere(observation1 != observation2).shape)

# for _ in range(10):
#     t = 0
#     while True:
#         t+=1
#         if t > 60:
#             env.render()
#         if t % 100 == 0:
#             print("per 100 steps:",t)
#         action = env.action_space.sample() # your agent here (this takes random actions)
#         observation, reward, done, info = env.step(action)
#         # env.env.viewer.window.dispatch_events()
#         # print(observation.shape)
#
#         if done:
#             print("done:",t)
#             observation = env.reset()
#             break
#     env.close()

# t = []
# n = np.array([1,2,3])
# p = np.array([4,5,6])
# t += [n]
# t += [p]
# # t.append(n)
# print(t)
# print(np.array(t))
# print(np.stack(t,axis=0))

# arrays = [np.random.randn(3, 4) for _ in range(2)]
#
# print(np.array(arrays))
# print(np.stack(arrays,axis=0))

# import torch
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
#
#
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )
#
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )
#
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# print(type(training_data))
# print(isinstance(training_data, torch.Tensor))
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#
#     img, label = training_data[sample_idx]
#     # print(training_data[sample_idx])
#
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

import glob
# a = glob.glob('datasets/rollout_ep_[0-9][0-9][0-9].npz')
# print(a)
# data = np.load('datasets/rollout_ep_000.npz')
# print(type(data))
# for k, v in data.items():
#     print(k,v)
#     break

# def give(x):
#     a = x*10
#     b = x*100
#     return a, b
# u = [2,3,4,5]
# l1, l2 = [give(x) for x in u]
# # print(

# from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
# import gym
# env = gym.make('CarRacing-v0')
# print(env.reset().shape)

import gym

# from stable_baselines3 import A2C, PPO

# env = gym.make('CartPole-v1')
#
# model = A2C('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=1000)

# model = PPO('MlpPolicy', "CartPole-v1", verbose=1).learn(1000)

# print(env.observation_space)

# from wrappers import CarRacingWrapper


# print(f"<1s>")
# for _ in range(20):
#     env.reset()
#     for _ in range(1000):
#         # env.render()
#         env.step(env.action_space.sample()) # take a random action
# env.close()
#
# class A:
#     def __init__(self, env):
#         self.env = env
# class B(A):
#     def reward(self):
#         pass
# class My(B):
#     def __init__(self, env):
#         super().__init__(env)
#
# print(My(5).env)
from typing import TypeVar, Generic
# from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
# from stable_baselines3.common.vec_env import DummyVecEnv
# env = DummyVecEnv([lambda: gym.make("CartPole-v0")])
# print(env.observation_space)
# print(obs_space_info(env.observation_space))


# def decorator(f):
#     def new_function():
#         print("Extra Functionality")
#         f()
#     return new_function
#
# @decorator
# def initial_function():
#     print("Initial Functionality")
#
# initial_function()
# def rolling_window(array: np.ndarray, window: int) -> np.ndarray:
#     """
#     Apply a rolling window to a np.ndarray
#     :param array: the input Array
#     :param window: length of the rolling window
#     :return: rolling window on the input array
#     """
#     shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
#     s1 = array.strides
#     s2 = array.strides[-1]
#     strides = s1 + (s2,)
#     return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
#
# nd = np.reshape(np.arange(3*4), (3,4))
# re = rolling_window(nd, window=3)
# print(re)

import gym
# env = gym.make("CarRacing-v0")
# observation = env.reset()
# for _ in range(1000):
#   env.render()
#   action = env.action_space.sample() # your agent here (this takes random actions)
#   observation, reward, done, info = env.step(action)
#   print(info)
#
#   if done:
#     observation = env.reset()
# env.close()

from wrappers import LatentWrapper, NaiveWrapper, NaiveWrapper
from my_callbacks import ImageRecorderCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, SAC, PPO, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# env = gym.make('CarRacing-v0')
# env = NaiveWrapper(env, train=1)
# env = make_vec_env(NaiveWrapper, 2, env_kwargs={'env': env, 'train': True})
# print(env)
# env = Monitor(NaiveWrapper(env, train=1))
# env = DummyVecEnv([lambda: env])


import numpy as np
import multiprocessing as mp
import gym
import os
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d.car_racing import CarRacing

# env = CarRacing()
# env = gym.make('CarRacing-v0')
# observation = env.reset()

import gym
import Box2D
import math
import numpy as np
# env = gym.make("CartPole-v1")
# env = gym.make('CarRacing-v0')
# print(type(env))
# print("env._max_episode_steps:", env._max_episode_steps)
# action = env.action_space.sample()
# print(action)
# observation1 = env.reset()
# observation2 = env.reset()
# env.reset()
# print(np.argwhere(observation1 != observation2).shape)

# for _ in range(10):
#     t = 0
#     while True:
#         t+=1
#         if t > 60:
#             env.render()
#         if t % 100 == 0:
#             print("per 100 steps:",t)
#         action = env.action_space.sample() # your agent here (this takes random actions)
#         observation, reward, done, info = env.step(action)
#         # env.env.viewer.window.dispatch_events()
#         # print(observation.shape)
#
#         if done:
#             print("done:",t)
#             observation = env.reset()
#             break
#     env.close()

# t = []
# n = np.array([1,2,3])
# p = np.array([4,5,6])
# t += [n]
# t += [p]
# # t.append(n)
# print(t)
# print(np.array(t))
# print(np.stack(t,axis=0))

# arrays = [np.random.randn(3, 4) for _ in range(2)]
#
# print(np.array(arrays))
# print(np.stack(arrays,axis=0))

# import torch
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
#
#
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )
#
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )
#
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# print(type(training_data))
# print(isinstance(training_data, torch.Tensor))
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#
#     img, label = training_data[sample_idx]
#     # print(training_data[sample_idx])
#
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

import glob
# a = glob.glob('datasets/rollout_ep_[0-9][0-9][0-9].npz')
# print(a)
# data = np.load('datasets/rollout_ep_000.npz')
# print(type(data))
# for k, v in data.items():
#     print(k,v)
#     break

# def give(x):
#     a = x*10
#     b = x*100
#     return a, b
# u = [2,3,4,5]
# l1, l2 = [give(x) for x in u]
# # print(

# from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
# import gym
# env = gym.make('CarRacing-v0')
# print(env.reset().shape)

import gym

# from stable_baselines3 import A2C, PPO

# env = gym.make('CartPole-v1')
#
# model = A2C('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=1000)

# model = PPO('MlpPolicy', "CartPole-v1", verbose=1).learn(1000)

# print(env.observation_space)

# from wrappers import CarRacingWrapper


# print(f"<1s>")
# for _ in range(20):
#     env.reset()
#     for _ in range(1000):
#         # env.render()
#         env.step(env.action_space.sample()) # take a random action
# env.close()
#
# class A:
#     def __init__(self, env):
#         self.env = env
# class B(A):
#     def reward(self):
#         pass
# class My(B):
#     def __init__(self, env):
#         super().__init__(env)
#
# print(My(5).env)
from typing import TypeVar, Generic
# from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
# from stable_baselines3.common.vec_env import DummyVecEnv
# env = DummyVecEnv([lambda: gym.make("CartPole-v0")])
# print(env.observation_space)
# print(obs_space_info(env.observation_space))


# def decorator(f):
#     def new_function():
#         print("Extra Functionality")
#         f()
#     return new_function
#
# @decorator
# def initial_function():
#     print("Initial Functionality")
#
# initial_function()
# def rolling_window(array: np.ndarray, window: int) -> np.ndarray:
#     """
#     Apply a rolling window to a np.ndarray
#     :param array: the input Array
#     :param window: length of the rolling window
#     :return: rolling window on the input array
#     """
#     shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
#     s1 = array.strides
#     s2 = array.strides[-1]
#     strides = s1 + (s2,)
#     return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
#
# nd = np.reshape(np.arange(3*4), (3,4))
# re = rolling_window(nd, window=3)
# print(re)

import gym
# env = gym.make("CarRacing-v0")
# observation = env.reset()
# for _ in range(1000):
#   env.render()
#   action = env.action_space.sample() # your agent here (this takes random actions)
#   observation, reward, done, info = env.step(action)
#   print(info)
#
#   if done:
#     observation = env.reset()
# env.close()

from wrappers import LatentWrapper, NaiveWrapper, NaiveWrapper
from my_callbacks import ImageRecorderCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, SAC, PPO, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

env = gym.make('CarRacing-v0')
print(env.action_space)
# print(env.spec)
# print(env.spec.max_episode_steps)
# env = NaiveWrapper(env, train=1)
# env = make_vec_env(NaiveWrapper, 2, env_kwargs={'env': env, 'train': True})
# print(env)
# env = Monitor(NaiveWrapper(env, train=1))
# env = DummyVecEnv([lambda: env])

# cuda = torch.cuda.is_available()
# device = torch.device("cuda" if cuda else "cpu")
# print("torch.cuda.device_count()", torch.cuda.device_count())
# print("torch.cuda.current_device()", torch.cuda.current_device())
#
# os.makedirs('./tensorboard_log/', exist_ok=True)
# # eval_env = gym.make('CarRacing-v0')
# # eval_env = Monitor(eval_env)
# # eval_env = DummyVecEnv([lambda: eval_env])
# eval_env = make_vec_env('CarRacing-v0', 1, seed=0)
#
# total_time_steps = int(5e6)
# model = PPO("CnnPolicy", 'CarRacing-v0', verbose=1, tensorboard_log="./tensorboard_log/sac_cartpole_tensorboard/", device=device)  # CnnPolicy
# model.learn(total_time_steps, tb_log_name=f'vanilla_ts{total_time_steps:.0e}', eval_env=eval_env,
#             eval_freq=int(total_time_steps / 20))


# import gym
#
# from stable_baselines3 import SAC
# from stable_baselines3.common.env_util import make_vec_env
#
# env = make_vec_env("Pendulum-v0", n_envs=4, seed=0)
#
# # We collect 4 transitions per call to `ènv.step()`
# # and performs 2 gradient steps per call to `ènv.step()`
# # if gradient_steps=-1, then we would do 4 gradients steps per call to `ènv.step()`
# model = TD3('MlpPolicy', env, train_freq=1, gradient_steps=2, verbose=1)
# model.learn(total_timesteps=10_000)