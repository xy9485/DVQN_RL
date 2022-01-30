import gym
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing
import numpy as np
from PIL import Image
from torchvision import transforms as T

class GeneralWrapper(gym.Wrapper):
    def __init__(self, env, train):
        super().__init__(env)
        if train:
            self.env_purpose = "train"
        else:
            self.env_purpose = "eval"
        self.step_episode = 0

    def step(self):
        if self.env.spec.id == 'CarRacing-v0':
            #  https://github.com/openai/gym/issues/976
            self.viewer.window.dispatch_events()
        self.step_episode += 1

    def reset(self):
        if self.env.spec.id == 'CarRacing-v0':
            #  https://github.com/openai/gym/issues/976
            self.viewer.window.dispatch_events()
        print(f"====Env Reset: env_purpose: {self.env_purpose} | steps of this episode: {self.step_episode}====")
        self.step_episode = 0

class LatentWrapper(gym.Wrapper):
    def __init__(self, env, train, encoder=None, process_frame=None, seed=None):
        super().__init__(env) # self.env = env happens in init of gym.Wrapper

        if seed:
            self.env.seed(int(seed))

        #  new observation space to deal with resize
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(128,),
            dtype=np.float32
        )

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
    def __init__(self, env, train, encoder=None, process_frame=None, policy=None, seed=None):
        super().__init__(env)

        if seed:
            self.env.seed(int(seed))
        #  new observation space to deal with resize
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(64,64) + (3,),
            dtype=np.uint8
        )

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
        return obs, reward+shaping, done, info

    def reset(self):
        obs = self.env.reset()
        # super().reset()

        obs = self._process_frame(obs)
        latent_obs = self.encode(obs)
        action = self.policy.actor(latent_obs)
        new_value = self.policy.critic(latent_obs, action)
        self.value_last_latent_state = new_value

        obs = self.process_frame(obs)
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




