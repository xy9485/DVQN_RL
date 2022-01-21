import argparse
import sys
import os
from os.path import join, exists
from os import mkdir, makedirs,getpid
import time

import gym
import numpy as np
from hparams import HyperParams as hp

import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary
from models.vae import VAE
from wrappers import LatentWrapper, NaiveWrapper, ShapingWrapper
from my_callbacks import ImageRecorderCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import A2C, SAC, PPO, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed



class PrintTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        pass

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
        return True


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
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Trainer')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--logdir', type=str, default='./logdir', help='Directory where results are logged')
    parser.add_argument('--noreload', action='store_true',
                        help='Best model is not reloaded if specified')
    parser.add_argument('-o2f', '--output2file', action='store_true',
                        help='print to a file when using nohup')
    args = parser.parse_args()

    vae_dir = join(args.logdir, 'vae_buffersize200')
    best_filename = join(vae_dir, 'best.tar')

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print("torch.cuda.device_count()", torch.cuda.device_count())
    print("torch.cuda.current_device()", torch.cuda.current_device())


    vae_model = VAE(3, hp.vsize)  # we do not specify pretrained=True, i.e. do not load default weights
    device = torch.device('cpu')
    vae_model.load_state_dict(torch.load(best_filename, map_location=device)['state_dict'])
    # vae_model.to(device)
    vae_model.eval()

    transform_easy = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # env = Monitor(CarRacingWrapper(env, vae_model.encoder, transform_easy))
    # env = DummyVecEnv([lambda: env])

    # Create the callback: check every 1000 steps
    # eval_env = gym.make('CarRacing-v0')
    # eval_env = Monitor(CarRacingWrapper(gym.make('CarRacing-v0'), vae_model.encoder, transform_easy))
    # eval_callback = EvalCallback(eval_env, eval_freq=50,
    #                              deterministic=True, render=False)

    # print_training_reward_callback = PrintTrainingRewardCallback(50)

    # env = gym.make('CarRacing-v0')
    # env = Monitor(LatentWrapper(env, 1, vae_model.encoder, transform_easy))
    # env = DummyVecEnv([lambda: env])
    #
    # eval_env = gym.make('CarRacing-v0')
    # eval_env = Monitor(LatentWrapper(eval_env, 0, vae_model.encoder, transform_easy))
    # eval_env = DummyVecEnv([lambda: eval_env])

    env = gym.make('CarRacing-v0')
    env = make_vec_env('CarRacing-v0', 1, seed=1234, wrapper_class=LatentWrapper,
                       wrapper_kwargs={'train': 1, 'encoder':vae_model.encoder, 'transform':transform_easy})

    eval_env = gym.make('CarRacing-v0')
    eval_env = make_vec_env('CarRacing-v0', 1, seed=1233, wrapper_class=LatentWrapper,
                       wrapper_kwargs={'train': 0, 'encoder':vae_model.encoder, 'transform':transform_easy})

    # print(isinstance(eval_env, gym.Env))

    # Create action noise because TD3 and DDPG use a deterministic policy
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # model = TD3("MlpPolicy", 'CarRacing-v0', action_noise=None, verbose=1, tensorboard_log="./tensorboard_log/td3_carracing_tensorboard/")
    # model.learn(2000)
    # model.learn(10000, tb_log_name="first_run")
    # model.learn(2000, eval_env=eval_env, eval_freq=50, n_eval_episodes=5)

    os.makedirs('./tensorboard_log/', exist_ok=True)
    total_time_steps = int(5e6)
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_log/sac_cartpole_tensorboard/", device=device)  # CnnPolicy
    model.learn(total_time_steps, tb_log_name=f'latent_ts{total_time_steps:.0e}', eval_env=eval_env, eval_freq=int(total_time_steps/10))

    # model.learn(20000,  eval_env=eval_env, eval_freq=500, n_eval_episodes=5)
    # model = SAC("CnnPolicy", env, verbose=1).learn(2000)

    # eval_env = Monitor(CarRacingWrapper(gym.make('CarRacing-v0'), vae_model.encoder, transform_easy))
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

    # env_id = 'CarRacing-v0'
    # NUM_EXPERIMENTS = 3 # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
    # TRAIN_STEPS = 5000
    # # Number of episodes for evaluation
    # EVAL_EPS = 20
    # ALGO = A2C
    #
    # # We will create one environment to evaluate the agent on
    # eval_env = gym.make(env_id)
    #
    # times = []
    # rewards = []
    # for experiment in range(NUM_EXPERIMENTS):
    #         # it is recommended to run several experiments due to variability in results
    #         env.reset()
    #         model = ALGO('MlpPolicy', env, verbose=0)
    #         start = time.time()
    #         model.learn(total_timesteps=TRAIN_STEPS)
    #         times.append(time.time() - start)
    #         mean_reward, _  = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
    #         rewards.append(mean_reward)