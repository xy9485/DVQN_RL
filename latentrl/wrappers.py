import math
from os import truncate
import time
from collections import deque
from statistics import mean
import itertools as it
from turtle import position
import gym

# from gym.spaces.box import Box
# from gym.spaces.discrete import Discrete
from gym.spaces import Box
from gym.spaces import Discrete
from gym.envs.box2d.car_racing import CarRacing
from gym.envs.box2d.car_dynamics import Car
import numpy as np
from PIL import Image, ImageOps
from gym import spaces
from stable_baselines3 import SAC, DQN
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from torchvision import transforms as T
from os.path import join
import torch
from models.vae import VAE
from models.vqvae import VQVAE
import importlib
from typing import List, Set, Dict, Tuple, Optional, Union, Any, Callable
import cv2

cv2.ocl.setUseOpenCL(False)


class GeneralWrapper(gym.Wrapper):
    def __init__(self, env, train):
        super().__init__(env)
        if train:
            self.env_purpose = "train"
        else:
            self.env_purpose = "eval"
        self.step_episode = 0

    def step(self):
        if self.env.spec.id == "CarRacing-v0":
            #  https://github.com/openai/gym/issues/976
            self.viewer.window.dispatch_events()
        self.step_episode += 1

    def reset(self):
        if self.env.spec.id == "CarRacing-v0":
            #  https://github.com/openai/gym/issues/976
            self.viewer.window.dispatch_events()
        print(
            f"====Env Reset: env_purpose: {self.env_purpose} | steps of this episode: {self.step_episode}===="
        )
        self.step_episode = 0


class LatentWrapper(gym.Wrapper):
    def __init__(self, env, encoder=None, process_frame=None, seed=None):
        super().__init__(env)  # self.env = env happens in init of gym.Wrapper

        if seed:
            self.env.seed(int(seed))

        #  new observation space to deal with resize
        self.observation_space = Box(low=0, high=1, shape=(128,), dtype=np.float32)

        self._encoder = encoder
        self._process_frame = process_frame

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # super().step()
        obs = self._process_frame(obs)
        latent_obs = self.encode(obs)
        return latent_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        # super().reset()
        obs = self._process_frame(obs)
        latent_obs = self.encode(obs)
        return latent_obs

    # def process_frame(self, obs):
    #     obs = self._transform(obs)
    #     obs = obs.unsqueeze(0)
    #     return self._encoder(obs)[0].detach().numpy()   # using mu

    def encode(self, obs):
        obs = T.ToTensor()(obs)
        obs = obs.unsqueeze(0)
        return self._encoder(obs)[0].squeeze().detach().numpy()  # using mu


class ShapingWrapper(gym.Wrapper):
    def __init__(self, env, encoder=None, process_frame=None, policy=None, seed=None):
        super().__init__(env)

        if seed:
            self.env.seed(int(seed))
        #  new observation space to deal with resize
        self.observation_space = Box(low=0, high=255, shape=(64, 64) + (3,), dtype=np.uint8)

        self._encoder = encoder
        self._process_frame = process_frame
        self.policy = policy
        self.value_last_latent_state = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # super().step()

        obs = self._process_frame(obs)
        latent_obs = self.encode(obs)
        action = self.policy.actor(latent_obs)
        new_value = self.policy.critic(latent_obs, action)
        shaping = new_value - self.value_last_latent_state
        self.value_last_latent_state = new_value

        obs = self._process_frame(obs)
        return obs, reward + shaping, done, info

    def reset(self):
        obs = self.env.reset()
        # super().reset()

        obs = self._process_frame(obs)
        latent_obs = self.encode(obs)
        action = self.policy.actor(latent_obs)
        new_value = self.policy.critic(latent_obs, action)
        self.value_last_latent_state = new_value

        obs = self._process_frame(obs)
        return obs

    # def process_frame(self, obs, to_encode=False):
    #     obs = self._transform(obs)
    #     if to_encode:
    #         obs = obs.unsqueeze(0)
    #         return self._encoder(obs)[0].detach().numpy()
    #     return obs.detach().permute(1,2,0).numpy()  # without scaling, dtype:uint8
    #     # return np.array(obs) / max_val    #with scaling

    def encode(self, obs):
        obs = T.ToTensor()(obs)
        obs = obs.unsqueeze(0)
        return self._encoder(obs)[0].squeeze().detach().numpy()  # using mu


class NaiveWrapper(gym.Wrapper):
    def __init__(self, env, train):
        super().__init__(env, train)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # super().step()
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        # super.reset()
        return obs


class CropObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, vertical_cut=None, horizontal_cut=None):
        super().__init__(env)
        self.horizontal_cut = horizontal_cut
        self.vertical_cut = vertical_cut

        self.observation_space = Box(low=0, high=255, shape=(64, 64) + (3,), dtype=np.uint8)

    def observation(self, obs):
        obs = obs[: self.vertical_cut, : self.horizontal_cut, :]
        return obs


class ResizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape

        self.observation_space = Box(
            low=0, high=255, shape=(shape[0], shape[1]) + (3,), dtype=np.uint8
        )

    def observation(self, obs):
        obs_image = Image.fromarray(obs, mode="RGB")
        obs = obs_image.resize(self.resize, Image.BILINEAR)
        return np.array(obs)


class CarRandomStartWrapper(gym.Wrapper):
    def __init__(self, env, warm_up_steps, n_envs, always_random_start, no_random_start=False):
        super().__init__(env)
        self.warm_up_steps = int(warm_up_steps / n_envs)
        self.finished_steps = 0
        self.always_random_start = always_random_start
        self.no_random_start = no_random_start
        # self.prev_action = self.env.action_space.sample()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.finished_steps += 1
        return obs, reward, done, info

    def reset(self):
        self.env.unwrapped.random_start = False
        # obs = self.env.reset()
        if self.no_random_start:
            return self.env.reset()
        if self.always_random_start or (self.finished_steps < self.warm_up_steps):
            self.env.unwrapped.random_start = True
            return self.env.reset()
        return self.env.reset()

        # position = np.random.randint(len(self.env.track))
        # self.env.car = Car(self.env.world, *self.env.track[position][1:4])
        # if self.accelerated_exploration:
        # self._action = self.env.action_space.sample()
        # obs, reward, _, _  = self.env.step(self._action)


class PreprocessObservationWrapper(
    gym.ObservationWrapper
):  # combination of CropObservationWrapper and ResizeObservationWrapper
    def __init__(
        self,
        env,
        vertical_cut_u=None,
        vertical_cut_d=None,
        horizontal_cut_l=None,
        horizontal_cut_r=None,
        shape=64,
        num_output_channels=3,
        preprocess_mode="torchvision",  # "torchvision" or "PIL" or "cv2"
    ):
        super().__init__(env)
        self.shape = (shape, shape)
        self.num_output_channels = num_output_channels
        # PIL or transforms(for Grayscale)
        self.preprocess_mode = preprocess_mode

        self.vertical_cut_u = vertical_cut_u
        self.vertical_cut_d = vertical_cut_d
        self.horizontal_cut_l = horizontal_cut_l
        self.horizontal_cut_r = horizontal_cut_r

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(shape, shape) + (num_output_channels,),
            dtype=np.uint8,  # this is important to have
        )

    def observation(self, obs):
        obs = obs[
            self.vertical_cut_u : self.vertical_cut_d,
            self.horizontal_cut_l : self.horizontal_cut_r,
            :,
        ]
        if self.preprocess_mode == "PIL":
            obs = Image.fromarray(obs, mode="RGB")
            obs = obs.resize(self.shape, Image.BILINEAR)
            if self.num_output_channels == 1:
                obs = ImageOps.grayscale(obs)
            return np.array(obs)
        elif self.preprocess_mode == "cv2":
            obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
            if self.num_output_channels == 1:
                obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
                obs = np.array(obs)[..., np.newaxis]
            return np.array(obs)
        elif self.preprocess_mode == "torchvision":
            obs = T.ToPILImage()(obs)
            # interpolation=InterpolationMode.BILINEAR
            obs = T.Resize(self.shape)(obs)
            if self.num_output_channels == 1:
                obs = T.Grayscale(num_output_channels=self.num_output_channels)(obs)
                return np.array(obs)[..., np.newaxis]
                # or return np.expand_dims(np.array(obs), -1)
            return np.array(obs)


class WarpFrameRGB(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.size = 84
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=(self.size, self.size) + (3,),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame):
        """returns the current observation from a frame"""
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        # cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_NEAREST
        return frame


class FrameStack(gym.Wrapper):
    """Stack n_frames last frames.

    :param gym.Env env: the environment to wrap.
    :param int n_frames: the number of frames to stack.
    """

    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * n_frames,)),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        obs, info = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        # the original wrapper use `LazyFrames` but since we use np buffer,
        # it has no effect
        return np.concatenate(self.frames, axis=2)


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.repeat_axis = -1
        self.stack_dimension = -1
        low = np.repeat(self.env.observation_space.low, self.n_frames, axis=self.repeat_axis)
        high = np.repeat(self.env.observation_space.high, self.n_frames, axis=self.repeat_axis)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=self.env.observation_space.dtype
        )

        self.stackedobs = np.zeros(self.observation_space.low.shape, dtype=np.uint8)
        self.shift_size_obs = self.env.observation_space.low.shape[self.stack_dimension]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # start_time = time.time()

        self.stackedobs = np.roll(
            self.stackedobs, shift=-self.shift_size_obs, axis=self.stack_dimension
        )
        self.stackedobs[..., -self.shift_size_obs :] = obs

        # self.time_step_wait = time.time() - start_time
        return self.stackedobs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.stackedobs = np.zeros(self.observation_space.low.shape, dtype=np.uint8)
        obs, info = self.env.reset(**kwargs)
        self.stackedobs[..., -self.shift_size_obs :] = obs

        return self.stackedobs, info


class EncodeWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        vae_f=None,
        vae_sample=False,
        vae_inchannel=3,
        latent_dim=128,
        seed=None,
    ):
        super().__init__(env)  # self.env = env happens in init of gym.Wrapper

        if seed:
            self.env.seed(int(seed))

        #  new observation space to deal with resize
        self.observation_space = Box(low=-5, high=5, shape=(128,), dtype=np.float32)

        # load vae
        best_filename = join(vae_f, "best.tar")
        vae_model = VAE(vae_inchannel, latent_dim)  # latent_size: 128
        device_vae = torch.device("cpu")
        vae_model.load_state_dict(torch.load(best_filename, map_location=device_vae)["state_dict"])
        # vae_model.to(device)
        vae_model.eval()
        self._vae_model = vae_model
        self._encoder = vae_model.encoder
        self.vae_sample = vae_sample

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        latent_obs = self.encode(obs)
        return latent_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        latent_obs = self.encode(obs)
        return latent_obs

    def encode(self, obs):
        obs = T.ToTensor()(obs)
        obs = obs.unsqueeze(0)
        if self.vae_sample:
            latent_obs = self._vae_model.reparameterize(*self._encoder(obs))
        else:
            latent_obs = self._encoder(obs)[0]
        return latent_obs.squeeze().detach().numpy()  # using mu


class ShapeRewardWrapper(gym.Wrapper):
    def __init__(self, env, vae_f=None, latent_model_f=None, latent_model_class=SAC, seed=None):
        super().__init__(env)

        if seed:
            self.env.seed(int(seed))
        #  new observation space to deal with resize
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(64, 64) + (3,),
            dtype=np.uint8,  # this is important to have
        )

        # load vae
        best_filename = join(vae_f, "best.tar")
        vae_model = VAE(3, 128)  # latent_size: 128
        device_vae = torch.device("cpu")
        vae_model.load_state_dict(torch.load(best_filename, map_location=device_vae)["state_dict"])
        # vae_model.to(device)
        vae_model.eval()
        self._encoder = vae_model.encoder

        # load latent policy
        self._latent_model_f = latent_model_f
        self._latent_model_class = latent_model_class
        self._latent_policy = self._latent_model_class.load(self._latent_model_f).policy
        self._latent_policy.eval()
        self._latent_policy.to("cpu")
        self._last_potential_value = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        latent_obs = self.encode(obs)
        action = self._latent_policy.actor(latent_obs, deterministic=True)
        q_values = [
            q_value.squeeze().item() for q_value in self._latent_policy.critic(latent_obs, action)
        ]
        new_potential_value = max(q_values)  # in SAC there are 2 q-networks

        assert (
            self._last_potential_value is not None
        ), "potential value of the old state can't be none"
        shaping = new_potential_value - self._last_potential_value
        self._last_potential_value = new_potential_value

        return obs, reward + shaping, done, info

    def reset(self):
        obs = self.env.reset()
        latent_obs = self.encode(obs)
        action = self._latent_policy.actor(latent_obs, deterministic=True)
        q_values = [
            q_value.squeeze().item() for q_value in self._latent_policy.critic(latent_obs, action)
        ]
        new_potential_value = max(q_values)  # in SAC there are 2 q-networks
        self._last_potential_value = new_potential_value

        return obs

    def encode(self, obs):
        obs = T.ToTensor()(obs)
        obs = obs.unsqueeze(0)
        return self._encoder(obs)[0].detach()
        # no more need of squeeze and conversion to numpy, otherwise errors rise


class EncodeStackWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        n_stack,
        vae_type="vqvae",
        vae_f=None,
        vae_inchannel=3,
        latent_dim=128,
        vae_sample=False,
    ):
        super().__init__(env)
        self.n_stack = n_stack
        self.repeat_axis = -1
        self.stack_dimension = -1
        self.vae_type = vae_type
        if self.vae_type == "vae":
            observation_space = Box(low=-5, high=5, shape=(latent_dim,), dtype=np.float32)
            low = np.repeat(observation_space.low, self.n_stack, axis=self.repeat_axis)
            high = np.repeat(observation_space.high, self.n_stack, axis=self.repeat_axis)
            self.observation_space = spaces.Box(low=low, high=high, dtype=observation_space.dtype)
            # load vae
            best_filename = join(vae_f, "best.tar")
            vae_model = VAE(vae_inchannel, latent_dim)  # latent_size: 128
            device_vae = torch.device("cpu")
            vae_model.load_state_dict(
                torch.load(best_filename, map_location=device_vae)["state_dict"]
            )
            # vae_model.to(device)
            vae_model.eval()
            self._vae_model = vae_model
            self._encoder = vae_model.encoder
            self.vae_inchannel = vae_inchannel
            self.latent_dim = latent_dim
            self.vae_sample = vae_sample
            self.shift_size = latent_dim

        elif self.vae_type == "vqvae":
            observation_space = Box(low=-5, high=5, shape=(21, 21, 16), dtype=np.float32)
            # Carracing shape(16, 16, 16)
            # Atari shape(21, 21, 16)， original image is resized to 84x84

            low = np.repeat(observation_space.low, self.n_stack, axis=self.repeat_axis)
            high = np.repeat(observation_space.high, self.n_stack, axis=self.repeat_axis)
            self.observation_space = spaces.Box(low=low, high=high, dtype=observation_space.dtype)
            # self.stacked_latent_obs = np.zeros(self.observation_space.low.shape, dtype=np.uint8)
            self.shift_size = 16
            # load vqvae
            vqvae_model = VQVAE(in_channels=vae_inchannel, embedding_dim=16, num_embeddings=64)
            best_filename = join(vae_f, "best.tar")

            device_vae = torch.device("cpu")
            vqvae_model.load_state_dict(
                torch.load(best_filename, map_location=device_vae)["state_dict"]
            )

            # self.device_vae = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # vqvae_model.load_state_dict(torch.load(best_filename)["state_dict"])
            # vqvae_model.to(self.device_vae)

            vqvae_model.eval()
            # vqvae_model = vqvae_model.float()
            self._vae_model = vqvae_model
            self._encoder = vqvae_model.encoder
            self.embedding_dim = 16
            self.num_embeddings = 64

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        latent_obs = self.encode(obs)
        self.stacked_latent_obs = np.roll(
            self.stacked_latent_obs, shift=-self.shift_size, axis=self.stack_dimension
        )
        self.stacked_latent_obs[..., -self.shift_size :] = latent_obs
        # print(np.nonzero(self.stacked_latent_obs))
        return self.stacked_latent_obs, reward, done, info

    def reset(self):
        # print("self._vae_model.encoder.weight.dtype:", next(self._vae_model.parameters()).dtype)
        self.reset_stacked_latent_obs()
        # self.stacked_latent_obs = np.zeros(self.n_stack * self.latent_dim, np.float32)
        obs = self.env.reset()
        latent_obs = self.encode(obs)
        self.stacked_latent_obs[..., -self.shift_size :] = latent_obs
        # print(np.nonzero(self.stacked_latent_obs))
        return self.stacked_latent_obs

    def encode(self, obs):
        with torch.inference_mode():
            obs = T.ToTensor()(obs)
            obs = obs.unsqueeze(0)
            if self.vae_type == "vqvae":
                encoding = self._vae_model.encoder(obs.float())
                quantized_inputs, vq_loss = self._vae_model.vq_layer(encoding)
                return quantized_inputs.permute(0, 2, 3, 1).squeeze().numpy()
                # [B x D x H x W] -> [B x H x W x D]
            elif self.vae_type == "vae":
                if self.vae_sample:
                    latent_obs = self._vae_model.reparameterize(*self._encoder(obs))
                else:
                    latent_obs = self._encoder(obs)[0]
                return latent_obs.squeeze().detach().numpy()  # using mu

    def reset_stacked_latent_obs(self):
        if self.vae_type == "vae":
            self.stacked_latent_obs = np.zeros(self.n_stack * self.latent_dim, np.float32)
        elif self.vae_type == "vqvae":
            self.stacked_latent_obs = np.zeros(self.observation_space.low.shape, np.float32)


class ShapeRewardStackWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        n_stack,
        vae_type="vqvae",
        gamma=0.96,
        omega=100,
        vae_f=None,
        vae_inchannel=3,
        latent_dim=128,
        vae_sample=False,
        latent_model_f=None,
        latent_deterministic=True,
        train=True,
        latent_model_class=SAC,
    ):
        super().__init__(env)
        self.n_stack = n_stack
        self.vae_type = vae_type
        self.gamma = gamma
        self.omega = omega
        self.train = train

        self.repeat_axis = -1
        self.stack_dimension = -1
        low = np.repeat(self.env.observation_space.low, self.n_stack, axis=self.repeat_axis)
        high = np.repeat(self.env.observation_space.high, self.n_stack, axis=self.repeat_axis)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=self.env.observation_space.dtype
        )

        self.stackedobs = np.zeros(self.observation_space.low.shape, dtype=np.uint8)

        self.shift_size_obs = self.env.observation_space.low.shape[self.stack_dimension]

        # load vae
        best_filename = join(vae_f, "best.tar")
        if self.vae_type == "vae":
            vae_model = VAE(vae_inchannel, latent_dim)  # latent_size: 128
            device_vae = torch.device("cpu")
            vae_model.load_state_dict(
                torch.load(best_filename, map_location=device_vae)["state_dict"]
            )
            # vae_model.to(device)
            vae_model.eval()
            self._vae_model = vae_model
            self._encoder = vae_model.encoder
            self.vae_sample = vae_sample
            self.latent_dim = latent_dim
            self.shift_size_latent_obs = latent_dim
        elif self.vae_type == "vqvae":
            vqvae_model = VQVAE(in_channels=vae_inchannel, embedding_dim=16, num_embeddings=64)
            # device_vae = torch.device("cpu")
            # vqvae_model.load_state_dict(
            #     torch.load(best_filename, map_location=torch.device("cpu"))["state_dict"]
            # )

            self.device_vae = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vqvae_model.load_state_dict(torch.load(best_filename)["state_dict"])
            vqvae_model.to(self.device_vae)

            vqvae_model.eval()
            self._vae_model = vqvae_model
            self._encoder = vqvae_model.encoder
            self.latent_obs_shape = (21, 21, 16)  # (16, 16, 16)
            self.shift_size_latent_obs = 16

        # load latent policy
        self._latent_model_f = latent_model_f
        self._latent_model_class = latent_model_class
        self._latent_model = self._latent_model_class.load(self._latent_model_f)
        self._latent_policy = self._latent_model.policy
        self._latent_policy.eval()
        self._latent_policy.to(self.device_vae)
        self._last_potential_value = None
        self.latent_deterministic = latent_deterministic

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        start_time = time.time()
        self.stackedobs = np.roll(
            self.stackedobs, shift=-self.shift_size_obs, axis=self.stack_dimension
        )
        self.stackedobs[..., -self.shift_size_obs :] = obs
        if not self.train:  # for evaluation, no need for computing shaping
            return self.stackedobs, reward, done, info
        latent_obs = self.encode(obs)
        with torch.inference_mode():
            self.stacked_latent_obs = torch.roll(
                self.stacked_latent_obs,
                shifts=-self.shift_size_latent_obs,
                dims=self.stack_dimension,
            )
            self.stacked_latent_obs[..., -self.shift_size_latent_obs :] = latent_obs
            if self._latent_model_class == SAC:
                action = self._latent_policy.actor(
                    self.stacked_latent_obs, deterministic=self.latent_deterministic
                )
                # action = self._latent_model.predict(self.stacked_latent_obs, deterministic=self.latent_deterministic)
                q_values = [
                    q_value.squeeze().item()
                    for q_value in self._latent_policy.critic_target(
                        self.stacked_latent_obs, action
                    )
                ]
                new_potential_value = sum(q_values) / len(q_values)  # in SAC there are 2 q-networks
            elif self._latent_model_class == DQN:
                q_value = self._latent_policy.q_net(self.stacked_latent_obs).max()
                new_potential_value = q_value.item()
        shaping = self.omega * (self.gamma * new_potential_value - self._last_potential_value)
        shaping = max(-50.0, min(shaping, 50.0))  # clip the value of shaping
        self._last_potential_value = new_potential_value
        self.shaping = shaping
        self.reward = reward
        self.time_step_wait = time.time() - start_time
        return self.stackedobs, reward + shaping, done, info

    def reset(self):
        print("memory_allocated: {:.5f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
        print("memory_reserved: {:.1f} MB".format(torch.cuda.memory_reserved() / (1024 * 1024)))
        self.stackedobs = np.zeros(self.observation_space.low.shape, dtype=np.uint8)
        self.reset_stacked_latent_obs()
        # self.stacked_latent_obs = torch.zeros((1, self.n_stack * self.latent_dim), dtype=torch.float32)

        obs = self.env.reset()
        self.stackedobs[..., -self.shift_size_obs :] = obs

        latent_obs = self.encode(obs)
        self.stacked_latent_obs[..., -self.shift_size_latent_obs :] = latent_obs
        with torch.inference_mode():
            if self._latent_model_class == SAC:
                # action = self._latent_model.predict(self.stacked_latent_obs, deterministic=self.latent_deterministic)
                action = self._latent_policy.actor(
                    self.stacked_latent_obs, deterministic=self.latent_deterministic
                )
                q_values = [
                    q_value.squeeze().item()
                    for q_value in self._latent_policy.critic(self.stacked_latent_obs, action)
                ]
                self._last_potential_value = sum(q_values) / len(
                    q_values
                )  # in SAC there are 2 q-networks
            elif self._latent_model_class == DQN:
                self._last_potential_value = (
                    self._latent_policy.q_net(self.stacked_latent_obs).max().item()
                )

        return self.stackedobs

    def reset_stacked_latent_obs(self):
        if self.vae_type == "vae":
            self.stacked_latent_obs = torch.zeros(
                (1, self.n_stack * self.latent_dim), dtype=torch.float32
            )
        elif self.vae_type == "vqvae":
            self.stacked_latent_obs = torch.zeros(
                (
                    1,
                    self.latent_obs_shape[0],
                    self.latent_obs_shape[1],
                    self.latent_obs_shape[2] * self.n_stack,
                ),
                dtype=torch.float32,
                device=self.device_vae,
            )
            # temp_tensor = torch.zeros((1, *self.latent_obs_shape), dtype=torch.float32)
            # self.stacked_latent_obs = torch.repeat_interleave(temp_tensor, self.n_stack, dim=self.stack_dimension)

    def encode(self, obs):
        with torch.inference_mode():
            obs = T.ToTensor()(obs)
            obs = obs.unsqueeze(0)
            if self.vae_type == "vqvae":
                encoding = self._vae_model.encoder(obs.float().to(self.device_vae))
                quantized_inputs, vq_loss = self._vae_model.vq_layer(encoding)
                return quantized_inputs.permute(0, 2, 3, 1).squeeze()
                # [B x D x H x W] -> [B x H x W x D]
            elif self.vae_type == "vae":
                if self.vae_sample:
                    latent_obs = self._vae_model.reparameterize(*self._encoder(obs))
                else:
                    latent_obs = self._encoder(obs)[0]
                return latent_obs.squeeze().detach()  # using mu

    def encode_(self, obs):  # return flatten latent representation of stacked obs
        with torch.inference_mode():
            obs = T.ToTensor()(obs)
            obs = obs.unsqueeze(0)
            if self.vae_sample:
                latent_obs = self._vae_model.reparameterize(*self._encoder(obs))
            else:
                latent_obs = self._encoder(obs)[0]
            return latent_obs.squeeze().detach()  # using mu


class VecShapeReward2(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        gamma=0.96,
        omega=100,
        vae_f=None,
        vae_sample=False,
        vae_inchannel=3,
        latent_dim=128,
        latent_model_f=None,
        latent_deterministic=True,
        get_potential_mode="mean",
        train=True,
        latent_model_class=SAC,
        seed=None,
    ):
        super().__init__(venv=venv)
        self.n_stack = venv.n_stack
        self.gamma = gamma
        self.omega = omega
        self.get_potential_mode = get_potential_mode
        self.train = train
        # load vae
        best_filename = join(vae_f, "best.tar")
        vae_model = VAE(vae_inchannel, latent_dim)  # latent_size: 128
        device_vae = torch.device("cpu")
        vae_model.load_state_dict(torch.load(best_filename, map_location=device_vae)["state_dict"])
        # vae_model.to(device)
        vae_model.eval()
        self.vae_model = vae_model
        self._encoder = vae_model.encoder
        self.vae_sample = vae_sample
        self.vae_inchannel = vae_inchannel
        self.latent_dim = latent_dim

        # load latent policy
        self._latent_model_f = latent_model_f
        self._latent_model_class = latent_model_class
        self._latent_policy = self._latent_model_class.load(self._latent_model_f).policy
        self._latent_policy.eval()
        self._latent_policy.to("cpu")
        self._last_potential_value = None
        self.latent_deterministic = latent_deterministic

        self.shift_size_latent_obs = latent_dim

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        self.stacked_latent_obs = torch.zeros(
            (self.num_envs, 1, self.n_stack * self.latent_dim), dtype=torch.float32
        )
        self._last_potential_value = np.zeros(len(obs))
        for i_env in range(len(obs)):
            latent_obs = self.encode2(obs[i_env])
            self.stacked_latent_obs[i_env, ..., -self.shift_size_latent_obs :] = latent_obs
            action = self._latent_policy.actor(
                self.stacked_latent_obs[i_env], deterministic=self.latent_deterministic
            )
            q_values = [
                q.squeeze().item()
                for q in self._latent_policy.critic(self.stacked_latent_obs[i_env], action)
            ]
            if self.get_potential_mode == "mean":
                self._last_potential_value[i_env] = sum(q_values) / len(q_values)
            elif self.get_potential_mode == "max":
                self._last_potential_value[i_env] = max(q_values)
            elif self.get_potential_mode == "min":
                self._last_potential_value[i_env] = min(q_values)
            else:
                raise Exception("invalid get_potential_mode")

        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        start_time = time.time()
        # self._last_potential_value = np.empty(len(obs))
        if not self.train:
            return obs, reward, done, info
        shaping = np.zeros(len(obs))
        for i_env in range(len(obs)):
            latent_obs = self.encode2(obs[i_env])
            if done[i_env]:
                self.stacked_latent_obs[i_env] = 0
            self.stacked_latent_obs[i_env, ..., -self.shift_size_latent_obs :] = latent_obs
            action = self._latent_policy.actor(self.stacked_latent_obs[i_env], deterministic=True)
            q_values = [
                q.squeeze().item()
                for q in self._latent_policy.critic(self.stacked_latent_obs[i_env], action)
            ]
            if self.get_potential_mode == "mean":
                new_potential_value = sum(q_values) / len(q_values)
            elif self.get_potential_mode == "max":
                new_potential_value = max(q_values)
            elif self.get_potential_mode == "min":
                new_potential_value = min(q_values)
            else:
                raise Exception("invalid get_potential_mode")
            shaping[i_env] = self.omega * (
                self.gamma * new_potential_value - self._last_potential_value[i_env]
            )
            self._last_potential_value[i_env] = new_potential_value
        self.shaping = np.clip(shaping, -35, 35)
        self.reward = reward
        self.time_step_wait = time.time() - start_time
        return obs, reward + shaping, done, info

    def encode(self, obs):  # return flatten latent representation of stacked obs
        with torch.no_grad():
            splitted_obs = np.split(obs, self.n_stack, axis=-1)
            splitted_obs = np.array(splitted_obs)
            splitted_obs = torch.from_numpy(splitted_obs)
            splitted_obs = splitted_obs.permute(0, 3, 1, 2)
            splitted_obs = torch.div(splitted_obs, 255.0)
            splitted_obs = self._encoder(splitted_obs)[0]
            return splitted_obs.view(-1, torch.numel(splitted_obs))

    def encode2(self, stack_obs):  # return flatten latent representation of stacked obs
        with torch.no_grad():
            obs = stack_obs[..., -self.vae_inchannel :]
            obs = T.ToTensor()(obs)
            obs = obs.unsqueeze(0)
            if self.vae_sample:
                obs = self._vae_model.reparameterize(self._encoder(obs))
            else:
                obs = self._encoder(obs)[0]
            latent_obs = obs.squeeze()
            return latent_obs

    def encode3(self, obs):
        obs = T.ToTensor()(obs)
        obs = obs.unsqueeze(0)
        return self._encoder(obs)[0].detach()


class VecShapeReward(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        vae_f=None,
        latent_model_f=None,
        latent_model_class=SAC,
        seed=None,
    ):
        super().__init__(venv=venv)
        self.n_stack = venv.n_stack
        # load vae
        best_filename = join(vae_f, "best.tar")
        vae_model = VAE(3, 128)  # latent_size: 128
        device_vae = torch.device("cpu")
        vae_model.load_state_dict(torch.load(best_filename, map_location=device_vae)["state_dict"])
        # vae_model.to(device)
        vae_model.eval()
        self._encoder = vae_model.encoder

        # load latent policy
        self._latent_model_f = latent_model_f
        self._latent_model_class = latent_model_class
        self._latent_policy = self._latent_model_class.load(self._latent_model_f).policy
        self._latent_policy.eval()
        self._latent_policy.to("cpu")
        self._last_potential_value = None

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        self._last_potential_value = np.zeros(len(obs))
        for i_env in range(len(obs)):
            new_potential_value = []
            latent_obs = []
            for j in range(self.n_stack):
                # print("%%%", obs[i_env].shape)
                # print("%%%",self.encode(obs[i_env][..., j*3:(j+1)*3]).shape)
                latent_code = self.encode(obs[i_env][..., j * 3 : (j + 1) * 3])
                latent_obs.append(latent_code)
                # if np.all(latent_obs==0): # at the beginning, the stack might not be full
                #     continue
            latent_stacked_obs = torch.cat(latent_obs, dim=1)
            # print("%%%", latent_stacked_obs.shape)
            action = self._latent_policy.actor(latent_stacked_obs, deterministic=True)
            q_values = [
                q_value.squeeze().item()
                for q_value in self._latent_policy.critic(latent_stacked_obs, action)
            ]
            # in SAC there are 2 q-networks
            new_potential_value = max(q_values)

            self._last_potential_value[i_env] = new_potential_value

        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        # self._last_potential_value = np.empty(len(obs))
        shaping = np.zeros(len(obs))
        for i_env in range(len(obs)):
            new_potential_value = []
            latent_obs = []
            for j in range(self.venv.n_stack):
                latent_code = self.encode(obs[i_env][..., j * 3 : (j + 1) * 3])
                latent_obs.append(latent_code)
                # if np.all(latent_obs == 0):  # at the beginning, the stack might not be full
                #     continue
            latent_stacked_obs = torch.cat(latent_obs, dim=1)
            action = self._latent_policy.actor(latent_stacked_obs, deterministic=True)
            q_values = [
                q_value.squeeze().item()
                for q_value in self._latent_policy.critic(latent_stacked_obs, action)
            ]
            # in SAC there are 2 q-networks
            new_potential_value = max(q_values)

            # assert self._last_potential_value[i] is not None, "potential value of the old state can't be none"
            shaping[i_env] = new_potential_value - self._last_potential_value[i_env]
            self._last_potential_value[i_env] = new_potential_value

        return obs, reward + shaping, done, info

    def encode(self, obs):
        obs = T.ToTensor()(obs)
        obs = obs.unsqueeze(0)
        return self._encoder(obs)[0].detach()
        # no more need of squeeze and conversion to numpy, otherwise errors rise


def pack_env_wrappers(
    wrapper_class_list: List, wrapper_kwargs_list: List[Dict], **kwargs_dict
) -> Callable:
    # version 1: return a callable, which doesn't require kwargs

    # wrapper_classes = []
    # wrapper_kwargs = []
    #
    # def get_module_name(wrapper_name):
    #     return ".".join(wrapper_name.split(".")[:-1])
    #
    # def get_class_name(wrapper_name):
    #     return wrapper_name.split(".")[-1]
    #
    # wrapper_module = importlib.import_module(get_module_name(wrapper_name))
    # wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
    #
    # assert len(wrapper_class_list) == len(kwargs_list), "wrapper_class_list and kwargs_list must share the same length"
    # for wrapper_class, kwargs in zip(wrapper_class_list, kwargs_list)
    #     wrapper_classes.append(wrapper_class)
    #     wrapper_kwargs.append(kwargs)

    # def wrap_env(env):
    #     for WrapperClassStr, _kwargs in kwargs_dict.items():
    #         WrapperClass = globals()[WrapperClassStr]
    #         env = WrapperClass(env, _kwargs)
    #     return env
    # return wrap_env
    if len(wrapper_class_list) == 0:
        return None
    assert len(wrapper_class_list) == len(
        wrapper_kwargs_list
    ), "two arguments(List) should share same length"

    def wrap_env(env: gym.Env):
        for wrapper_class, kwargs in zip(wrapper_class_list, wrapper_kwargs_list):
            if kwargs is None:
                env = wrapper_class(env)
                continue
            env = wrapper_class(env, **kwargs)
        return env

    return wrap_env


def pack_wrappers2(
    env, **kwargs_dict
):  # version 1: define a callable(function), which requires kwargs
    for WrapperClassStr, _kwargs in kwargs_dict.items():
        WrapperClass = globals()[WrapperClassStr]
        env = WrapperClass(env, _kwargs)
    return env


class ActionDiscreteWrapper(gym.ActionWrapper):
    # action space is discrete, designed for CarRacing-v0
    def __init__(self, env):
        super().__init__(env)
        self.action_list = [k for k in it.product([-1, -0.5, 0, 0.5, 1], [1, 0], [0.2, 0])]
        # self.action_list = [k for k in it.product([-1, -0.5, 0, 0.5, 1], [1, 0.5], [0.8, 0])]
        self.action_space = Discrete(20)

    def action(self, action):
        return self.action_list[action]


class ActionRepetitionWrapper(gym.Wrapper):
    def __init__(self, env, action_repetition=3):
        super().__init__(env)
        self.action_repetition = action_repetition

    def step(self, action: int):
        accumulated_reward = 0
        for _ in range(self.action_repetition):
            obs, reward, terminated, truncated, info = self.env.step(action)
            accumulated_reward += reward
            if terminated or truncated:
                break
        return obs, accumulated_reward, terminated, truncated, info


class EpisodeEarlyStopWrapper(gym.Wrapper):
    def __init__(self, env, max_neg_rewards=100, punishment=-20):
        super().__init__(env)
        self.neg_reward_counter = 0
        self.max_neg_rewards = max_neg_rewards  # 12(John)
        self.episode_reward = 0
        self.punishment = punishment

    def step(self, action: int):
        # modify rew
        obs, reward, terminated, truncated, info = self.env.step(action)
        early_done, punishment = self.check_early_stop2(reward)
        if early_done:
            reward += punishment
            truncated = True
        self.episode_reward += reward

        # if done or early_done:
        #     self.episode_reward = 0
        #     done = True

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.episode_reward = 0
        return obs

    def check_early_stop(self, reward):
        # return False, 0.0
        if reward < 0:
            self.neg_reward_counter += 1
            early_done = self.neg_reward_counter > self.max_neg_rewards
            # punishment = -30.0
            if early_done and self.episode_reward <= 500:
                punishment = -20.0
            else:
                punishment = 0.0

            if early_done:
                self.neg_reward_counter = 0

            return early_done, punishment
        else:
            self.neg_reward_counter = 0
            return False, 0.0

    def check_early_stop2(self, reward):
        # return False, 0.0
        if reward < 0:
            self.neg_reward_counter += 1
            early_done = self.neg_reward_counter > self.max_neg_rewards

            if early_done:
                punishment = self.punishment
                self.neg_reward_counter = 0
            else:
                punishment = 0

            return early_done, punishment
        else:
            self.neg_reward_counter = 0
            return False, 0.0


class PunishRewardWrapper(gym.Wrapper):
    def __init__(self, env, max_neg_rewards=12, punishment=-1):
        super().__init__(env)
        self.neg_reward_counter = 0
        self.max_neg_rewards = max_neg_rewards  # 12(John)
        self.punishment = punishment

    def step(self, action: int):
        # modify rew
        obs, reward, done, info = self.env.step(action)
        if reward < 0:
            self.neg_reward_counter += 1
            if self.neg_reward_counter > self.max_neg_rewards:
                reward = self.punishment
        else:
            self.neg_reward_counter = 0

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.neg_reward_counter = 0
        return obs


class PositiveRewardTracker(gym.Wrapper):
    def __init__(
        self,
        env,
    ):
        super().__init__(env)

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        if reward >= 0:
            self.non_zero_accumulated_reward += reward
        else:
            self.negative_accumulated_reward += reward
        if done:
            info["non_zero_accumulated_reward"] = self.non_zero_accumulated_reward
            info["negative_accumulated_reward"] = self.negative_accumulated_reward

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.non_zero_accumulated_reward = 0
        self.negative_accumulated_reward = 0
        return obs


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        if self._grayscale:
            frame = np.expand_dims(frame, 0)
        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class ChannelsFirstImageShape(gym.ObservationWrapper):
    """
    Change image shape to CWH
    """

    def __init__(self, env):
        super(ChannelsFirstImageShape, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1])
        )

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)


class ClipRewardEnvCustom(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class EpisodicLifeEnvCustom(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


# class Merge2Dimensions(gym.ObservationWrapper):
#     """
#     Change image shape to CWH
#     """

#     def __init__(self, env):
#         super().__init__(env)

#     def observation(self, observation):
#         return np.concatenate(observation.tolist(), axis=-1)


class LazyFramesCustom(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class FrameStackCustom(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * n_frames),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return LazyFramesCustom(list(self.frames))


class MinigridInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        agent_pos1 = self.env.agent_pos
        agent_dir1 = self.env.agent_dir
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["agent_pos2"] = self.env.agent_pos
        info["agent_dir2"] = self.env.agent_dir
        info["agent_pos1"] = agent_pos1
        info["agent_dir1"] = agent_dir1
        return obs, reward, terminated, truncated, info

    def step_(self, action):
        info["agent_pos1"] = info["agent_pos2"]
        info["agent_dir1"] = info["agent_dir2"]
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["agent_pos2"] = self.env.agent_pos
        info["agent_dir2"] = self.env.agent_dir

        return obs, reward, terminated, truncated, info

    def reset(self):
        env, info = self.env.reset()
        info["agent_pos2"] = self.env.agent_pos
        info["agent_dir2"] = self.env.agent_dir
        # info["agent_pos1"] = None
        # info["agent_dir1"] = None
        return env, info


class MinigridEmptyRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            reward = 1
        else:
            reward = -0.01
        return obs, reward, terminated, truncated, info


class StateBonusCustom(gym.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = reward

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = tuple(env.agent_pos)

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        info["bonus"] = bonus

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


if __name__ == "__main__":
    env = gym.make("CarRacing-v0")
    print(env.action_space)
    env = PreprocessObservationWrapper(env, vertical_cut_d=84)
    print(env.action_space)
    env.reset()
