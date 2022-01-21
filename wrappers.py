import gym
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing
import numpy as np
from PIL import Image

class GeneralWrapper(gym.Wrapper):
    def __init__(self, env, train):
        super().__init__(env)
        if train:
            self.env_purpose = "train"
        else:
            self.env_purpose = "eval"
        self.step_episode = 0

    def step_episode_add(self):
        self.step_episode += 1

    def print_at_reset(self):
        print(f"====Env Reset: env_purpose: {self.env_purpose} | steps of this episode: {self.step_episode}====")
        self.step_episode = 0

class LatentWrapper(GeneralWrapper):
    def __init__(self, env, train, encoder=None, transform=None, seed=None):
        super().__init__(env, train) # self.env = env happens in init of gym.Wrapper

        if seed:
            self.env.seed(int(seed))

        #  new observation space to deal with resize
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(128,),
            # dtype=np.float32
        )

        self._encoder = encoder
        self._transform = transform

    def step(self, action):
        """ one step through the environment """
        frame, reward, done, info = self.env.step(action)

        #  needed to get image rendering
        #  https://github.com/openai/gym/issues/976
        self.viewer.window.dispatch_events()

        obs = self.process_frame(frame)
        self.step_episode_add()
        return obs, reward, done, info

    def reset(self):
        """ resets and returns initial observation """
        raw = self.env.reset()
        # print("######raw.shape:", raw.shape)
        #  needed to get image rendering
        #  https://github.com/openai/gym/issues/976
        self.viewer.window.dispatch_events()

        obs = self.process_frame(raw)
        self.print_at_reset()
        return obs

    def process_frame(self, frame, vertical_cut=84, resized_screen=(64, 64)):
        """ crops, scales & convert to float """
        frame = frame[:vertical_cut, :, :]
        frame = Image.fromarray(frame, mode='RGB')
        obs = frame.resize(resized_screen, Image.BILINEAR)
        obs = np.array(obs)
        # obs = np.array(obs) / max_val     # with normalization

        obs = self._transform(obs)
        obs = obs.unsqueeze(0)
        return self._encoder(obs)[0].detach().numpy()

class ShapingWrapper(GeneralWrapper):
    def __init__(self, env, train, encoder=None, transform=None, policy=None, seed=None):
        super().__init__(env, train)

        if seed:
            self.env.seed(int(seed))
        #  new observation space to deal with resize
        self.observation_space = Box(
            low=0,
            high=255,
            shape=self.screen_size + (3,)
        )

        self._encoder = encoder
        self._transform = transform
        self.policy = policy
        self.value_last_latent_state = 0

    def step(self, action):
        """ one step through the environment """
        frame, reward, done, info = self.env.step(action)

        #  needed to get image rendering
        #  https://github.com/openai/gym/issues/976
        self.viewer.window.dispatch_events()

        latent_obs = self.process_frame(frame, to_encode=True)
        action = self.policy.actor(latent_obs)
        new_value = self.policy.critic(latent_obs, action)
        shaping = new_value - self.value_last_latent_state
        self.value_last_latent_state = new_value

        obs = self.process_frame(frame)
        self.step_episode_add()
        return obs, reward+shaping, done, info

    def reset(self):
        """ resets and returns initial observation """
        raw = self.env.reset()
        # print("######raw.shape:", raw.shape)
        #  needed to get image rendering
        #  https://github.com/openai/gym/issues/976
        self.viewer.window.dispatch_events()

        latent_obs = self.process_frame(raw, to_encode=True)    # or abstract obs
        action = self.policy.actor(latent_obs)
        new_value = self.policy.critic(latent_obs, action)
        self.value_last_latent_state = new_value

        obs = self.process_frame(raw)  # or ground obs

        self.print_at_reset()
        return obs

    def process_frame(self, frame, vertical_cut=84, resized_screen=(64, 64), to_encode=False):
        """ crops, scales & convert to float """
        frame = frame[:vertical_cut, :, :]
        frame = Image.fromarray(frame, mode='RGB')
        obs = frame.resize(resized_screen, Image.BILINEAR)
        obs = np.array(obs)
        if to_encode:
            obs = self._transform(obs)
            obs = obs.unsqueeze(0)
            return self._encoder(obs)[0].detach().numpy()
        return obs  # without scaling, dtype:uint8
        # return np.array(obs) / max_val    #with scaling

class NaiveWrapper(GeneralWrapper):
    def __init__(self, env, train):
        super().__init__(env, train)

    def step(self, action):
        """ one step through the environment """
        frame, reward, done, info = self.env.step(action)

        #  needed to get image rendering
        #  https://github.com/openai/gym/issues/976
        self.viewer.window.dispatch_events()

        self.step_episode_add()
        return frame, reward, done, info

    def reset(self):
        """ resets and returns initial observation """
        # print(f"reset the env:", self.env_purpose)
        raw = self.env.reset()
        # print("######raw.shape:", raw.shape)
        #  needed to get image rendering
        #  https://github.com/openai/gym/issues/976
        self.viewer.window.dispatch_events()

        self.print_at_reset()
        return raw




