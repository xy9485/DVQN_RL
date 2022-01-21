import numpy as np
import os, sys, glob
import gym
from hparams import HyperParams as hp
from car_racing import CarRacingWrapper
import math
from PIL import Image

def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions

def rollout(noise_type='brown'):
    # env = gym.make("CarRacing-v0")
    env = CarRacingWrapper()

    # if we dont use gym.make("CarRacing-v0"), we dont need below, since CarRacingWrapper has no attribute _max_episode_steps
    # env._max_episode_steps = 1000
    # OR
    # env.spec.max_episode_steps = 1000

    seq_len = 1000+60 # the first 60 frames are zooming in.
    max_ep = hp.n_rollout
    feat_dir = hp.data_dir

    os.makedirs(feat_dir, exist_ok=True)

    obs_global = []

    for ep in range(max_ep):
        print("episode:", ep)
        # seed = np.random.randint(int(1e7))
        # env.seed(seed)

        obs_lst = []
        # action_lst = []
        reward_lst = []
        next_obs_lst = []
        done_lst = []

        if noise_type == 'white':
            action_lst = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == 'brown':
            action_lst = sample_continuous_policy(env.action_space, seq_len, 1. / 50)

        obs = env.reset()
        t = 0
        # action = action_lst[t]
        # obs, reward, done, _ = env.step(action)
        done = False


        while not done and t < seq_len:
            # if t%60 == 0:
            #     Image.fromarray(obs).save(f'{t}.jpeg')
            action = action_lst[t]
            t += 1
            if t <= 60:
                continue
            # env.render()
            # action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)

            # np.savez(
            #     os.path.join(feat_dir, 'rollout_{:03d}_{:04d}'.format(ep, t)),
            #     obs=obs,
            #     action=action,
            #     reward=reward,
            #     next_obs=next_obs,
            #     done=done,
            # )

            obs_lst.append(obs)
            action_lst.append(action)
            reward_lst.append(reward)
            next_obs_lst.append(next_obs)
            done_lst.append(done)

            obs = next_obs
            # print("obs.shape:", obs.shape)
        print("obs_lst.shape", np.stack(obs_lst, axis=0).shape)
        # np.savez(
        #     os.path.join(feat_dir, 'rollout_ep_{:03d}'.format(ep)),
        #     obs=np.stack(obs_lst, axis=0),  # (T, H, W, C)
        #     action=np.stack(action_lst, axis=0),  # (T, a)
        #     reward=np.stack(reward_lst, axis=0),  # (T, 1)
        #     next_obs=np.stack(next_obs_lst, axis=0),  # (T, H, W, C)
        #     done=np.stack(done_lst, axis=0),  # (T, 1)
        # )


if __name__ == '__main__':
    # np.random.seed(hp.seed)
    rollout()