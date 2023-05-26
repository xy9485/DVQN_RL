import gym


def make_env_atari(env_id, **kwargs):
    env = gym.make(
        env_id,
        frameskip=1,
    )
    # env = common.wrappers.RedundantActionWrapper(env, action_redundancy=10)
    env = gym.wrappers.AtariPreprocessing(env)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=3000)
    return env
