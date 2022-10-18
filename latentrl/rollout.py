import time
import numpy as np
import os, sys, glob
import gym
from hparams import HyperParams as hp
from car_racing import CarRacingWrapper
import math
from PIL import Image
from latentrl.common.utils import sample_continuous_policy
from gym.envs.box2d.car_dynamics import Car


def generate_action(prev_action):
    if np.random.randint(3) % 3:  # probability to repeat the previous action
        return prev_action

    index = np.random.randn(3)
    # Favor acceleration over the others:
    index[1] = np.abs(index[1])
    index = np.argmax(index)
    mask = np.zeros(3)
    mask[index] = 1

    action = np.random.randn(3)
    action = np.tanh(action)
    action[1] = (action[1] + 1) / 2
    action[2] = (action[2] + 1) / 2

    # if action[0] < 0 and np.random.randint(3)==0: # try to prefer turning left
    #     mask[0] = 1

    return action * mask


def rollout(name_env, noise_type="white"):
    env = gym.make("CarRacing-v0")
    # env = gym.make(name_env)
    # if we dont use gym.make("CarRacing-v0"), we dont need below, since CarRacingWrapper has no attribute _max_episode_steps
    # env._max_episode_steps = 1000
    # OR
    env.spec.max_episode_steps = 1000
    max_ep = 1000
    seq_len = 1000  # the first 60 frames are zooming in.

    feat_dir = f"datasets/{name_env}"

    os.makedirs(feat_dir, exist_ok=True)

    obs_global = []

    for ep in range(max_ep):
        print("episode:", ep)
        seed = np.random.randint(int(1e7))
        env.seed(seed)

        obs_lst = []
        # action_lst = []
        reward_lst = []
        next_obs_lst = []
        done_lst = []

        if noise_type == "white":
            action_lst = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == "brown":
            action_lst = sample_continuous_policy(env.action_space, seq_len, 1.0 / 50)

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
            env.render()

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

            reward_lst.append(reward)
            next_obs_lst.append(next_obs)
            done_lst.append(done)

            obs = next_obs
            # print("obs.shape:", obs.shape)
        # print("obs_lst.shape", np.stack(obs_lst, axis=0).shape)
        # np.savez(
        #     os.path.join(feat_dir, 'rollout_ep_{:03d}'.format(ep)),
        #     obs=np.stack(obs_lst, axis=0),  # (T, H, W, C)
        #     action=np.stack(action_lst, axis=0),  # (T, a)
        #     reward=np.stack(reward_lst, axis=0),  # (T, 1)
        #     next_obs=np.stack(next_obs_lst, axis=0),  # (T, H, W, C)
        #     done=np.stack(done_lst, axis=0),  # (T, 1)
        # )
        np.savez(
            os.path.join(feat_dir, "rollout_ep_{:03d}".format(ep)),
            obs=np.array(obs_lst),  # (T, H, W, C)
            action=np.array(action_lst),  # (T, a)
            reward=np.array(reward_lst),  # (T, 1)
            next_obs=np.array(next_obs_lst),  # (T, H, W, C)
            done=np.array(done_lst),  # (T, 1)
        )


def rollout_naive(name_env):
    env = gym.make(name_env)
    # env = gym.make(name_env)
    print(env.action_space)
    max_ep = 1000
    # start2save = 60
    # seq_len = 200 + start2save  # the first 60 frames are zooming in for Carracing-v0.
    seq_len = 1000
    # env.spec.max_episode_steps = seq_len  #not working
    env._max_episode_steps = seq_len

    feat_dir = f"/workspace/repos_dev/VQVAE_RL/datasets/{name_env}_1000x1000"

    # os.makedirs(feat_dir, exist_ok=True)

    obs_global = []

    for ep in range(max_ep):
        print("episode:", ep)
        seed = np.random.randint(int(1e7))
        env.seed(seed)

        obs_lst = []
        action_lst = []
        reward_lst = []
        done_lst = []

        # env.full_action_space = True # for skiing

        env.reset()
        action = env.action_space.sample()
        # Little hack to make the Car start at random positions in the race-track
        # position = np.random.randint(len(env.track))
        # env.car = Car(env.world, *env.track[position][1:4])

        # action = action_lst[t]
        # obs, reward, done, _ = env.step(action)
        done = False

        t = 0
        while not done and t < seq_len:

            action = env.action_space.sample()
            # action = generate_action(action)
            obs, reward, done, _ = env.step(action)
            print("reward:", reward)
            if done:
                print("done")
            t += 1
            # if t <= 20:
            #     continue

            env.render()

            # np.savez(
            #     os.path.join(feat_dir, 'rollout_{:03d}_{:04d}'.format(ep, t)),
            #     obs=obs,
            #     action=action,
            #     reward=reward,
            #     done=done,
            # )

            obs_lst.append(obs)
            action_lst.append(action)
            reward_lst.append(reward)
            done_lst.append(done)

            # print("obs.shape:", obs.shape)
        print("obs_lst.shape", np.array(obs_lst).shape)
        print("action_lst.shape", np.array(action_lst).shape)

        # np.savez(
        #     os.path.join(feat_dir, "rollout_ep_{:03d}".format(ep)),
        #     obs=np.array(obs_lst),  # (T, H, W, C)
        #     action=np.array(action_lst),  # (T, a)
        #     reward=np.array(reward_lst),  # (T, 1)
        #     done=np.array(done_lst),  # (T, 1)
        # )


if __name__ == "__main__":
    import colored_traceback.auto

    np.random.seed(int(time.time()))
    # name_env = "ALE/Freeway-v5"
    # name_env = "ALE/Boxing-v5"
    name_env = "Boxing-v0"
    # name_env = "ALE/Skiing-v5"
    # name_env = "CarRacing-v0"
    rollout_naive(name_env)
