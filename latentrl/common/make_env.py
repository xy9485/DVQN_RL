import gymnasium as gym
from gymnasium.spaces import Box, Discrete

class RedundantActionWrapper(gym.ActionWrapper):
    def __init__(self, env, action_redundancy=2):
        super().__init__(env)
        self.action_redundancy = action_redundancy
        self.origin_action_space_n = self.action_space.n
        # [variant 1]
        # self.action_space = Discrete(self.action_space.n * action_redundancy)
        # [variant 2]
        # assert action_redundancy > self.origin_action_space_n
        # self.action_space = Discrete(action_redundancy)
        # [variant 3]
        self.action_space = Discrete(self.origin_action_space_n + action_redundancy)

    def action(self, action):
        # return action // self.action_redundancy  # floor division
        if action < self.origin_action_space_n:
            return action
        else:
            return 0

def make_env_atari(env_id, **kwargs):
    env = gym.make(
        env_id,
        frameskip=1,
    )
    action_redundancy = kwargs.get("redundant_actions", 0)
    if action_redundancy > 0:
        env = RedundantActionWrapper(env, action_redundancy=action_redundancy)
    env = gym.wrappers.AtariPreprocessing(env)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=3000)
    return env