import gym
import minigrid

import common
import common.wrappers
import envs


def make_env_carracing(env_id, **kwargs):
    # env = gym.make(env_id).unwrapped
    env = gym.make(
        env_id,
        continuous=True,
        # render_mode="human",
    )
    # env = gym.make(
    #     env_id,
    #     # frameskip=(3, 7),
    #     # repeat_action_probability=0.25,
    #     full_action_space=False,
    #     # render_mode="human",
    # )

    # For atari, using gym wappers or third-party wappers
    # wrapper_class_list = [
    #     ClipRewardEnvCustom,
    #     EpisodicLifeEnvCustom,
    #     GrayScaleObservation,
    #     ResizeObservation,
    #     FrameStackCustom,
    # ]
    # wrapper_kwargs_list = [
    #     None,
    #     None,
    #     {"keep_dim": True},  # gym default wrapper
    #     {"shape": 84},  # gym default wrapper
    #     # {"num_stack": config.n_frame_stack},  # gym default wrapper
    #     {"k": config.n_frame_stack},  # custom wrapper
    # ]

    # For atari, but using custom wrapper
    # wrapper_class_list = [
    #     # ActionDiscreteWrapper,
    #     # ActionRepetitionWrapper,
    #     # EpisodeEarlyStopWrapper,
    #     # Monitor,
    #     # CarRandomStartWrapper,
    #     PreprocessObservationWrapper,
    #     # EncodeStackWrapper,
    #     # PunishRewardWrapper,
    #     FrameStackWrapper,
    # ]
    # wrapper_kwargs_list = [
    #     # {"action_repetition": config.action_repetition},
    #     # {"max_neg_rewards": max_neg_rewards, "punishment": punishment},
    #     # {'filename': monitor_dir},
    #     # {"filename": os.path.join(monitor_dir, "train")},  # just single env in this case
    #     # {
    #     #     "warm_up_steps": hparams["learning_starts"],
    #     #     "n_envs": n_envs,
    #     #     "always_random_start": always_random_start,
    #     #     "no_random_start": no_random_start,
    #     # },
    #     {
    #         "vertical_cut_d": 84,
    #         "shape": 84,
    #         "num_output_channels": 1,
    #     },
    #     # {
    #     #     "n_stack": n_stack,
    #     #     "vae_f": vae_path,
    #     #     "vae_sample": vae_sample,
    #     #     "vae_inchannel": vae_inchannel,
    #     #     "latent_dim": vae_latent_dim,
    #     # },
    #     # {'max_neg_rewards': max_neg_rewards, "punishment": punishment}
    #     {"n_frame_stack": config.n_frame_stack},
    # ]

    # For carracing-v0
    wrapper_class_list = [
        common.wrappers.ActionDiscreteWrapper,
        common.wrappers.ActionRepetitionWrapper,
        common.wrappers.EpisodeEarlyStopWrapper,
        # Monitor,
        # CarRandomStartWrapper,
        common.wrappers.PreprocessObservationWrapper,
        # EncodeStackWrapper,
        # PunishRewardWrapper,
        # FrameStackWrapper,
        # common.wrappers.FrameStack,
        gym.wrappers.FrameStack,
    ]
    wrapper_kwargs_list = [
        None,
        {"action_repetition": 3},
        {"max_neg_rewards": 100, "punishment": -20.0},
        {"vertical_cut_d": 84, "shape": 84, "num_output_channels": 1, "preprocess_mode": "cv2"},
        {"num_stack": 4},
    ]

    wrapper = common.wrappers.pack_env_wrappers(wrapper_class_list, wrapper_kwargs_list)
    env = wrapper(env)
    # env.seed(seed)

    return env


def make_env_minigrid(env_id, **kwargs):
    # env = gym.make(
    #     env_id,
    #     # new_step_api=True,
    #     # render_mode="human",
    # )

    if env_id == "MiniGrid-Empty-v0":
        env = minigrid.envs.EmptyEnv(
            # size=env_cfg.env_size,
            # agent_start_pos=tuple(config.agent_start_pos),
            # render_mode="human",
        )
    if env_id == "MiniGrid-MultiRooms-v0":
        env = envs.MultiRoomEnv(
            minNumRooms=8,
            maxNumRooms=8,
            max_steps=4000,
            # render_mode="human",
            seed=kwargs["seed"],
        )
    if env_id == "MiniGrid-FourRooms-v0":
        # use custom FourRoomsEnv because we fix positions of four doors
        env = envs.FourRoomsEnv(
            # agent_pos=tuple(env_cfg.agent_start_pos),
            agent_pos=(1, 1),
            # goal_pos=tuple(env_cfg.goal_pos),
            goal_pos=(17, 17),
            # render_mode="human",
        )
        # env = minigrid.envs.FourRoomsEnv(
        #     agent_pos=tuple(config.agent_start_pos),
        #     goal_pos=tuple(config.goal_pos),
        #     # render_mode="human",
        # )
    if env_id == "MiniGrid-Crossing-v0":
        env = envs.CrossingEnv(
            # size=env_cfg.env_size,
            num_crossings=1,
            obstacle_type=minigrid.core.world_object.Wall,
            # max_steps=2000,
            # render_mode="human",
        )
    if env_id == "MiniGrid-DistShift-v0":
        env = envs.DistShiftEnv(
            # width=env_cfg.env_size,
            # height=env_cfg.env_size,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            strip_rows=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            # max_steps=2000,
            # render_mode="human",
        )
    if env_id == "CollectFlags-basic":
        env = envs.CollectFlags(
            maze_name="basic",
            max_steps=8000,
            # render_mode="human",
        )

    if not env_id.start("CollectFlags"):
        env = common.wrappers.LimitNumberActionsWrapper(env, limit=3)
    # env = TimeLimit(env, max_episode_steps=3000, new_step_api=True)
    # env = minigrid.wrappers.StateBonus(env)
    # if env_cfg.input_format == "partial_obs":
    #     pass
    # if env_cfg.input_format == "full_obs":
    env = minigrid.wrappers.FullyObsWrapper(env)
    # elif env_cfg.input_format == "full_img":
    #     env = minigrid.wrappers.RGBImgObsWrapper(env, tile_size=env_cfg.tile_size)
    # else:
    # raise NotImplementedError

    env = minigrid.wrappers.ImgObsWrapper(env)
    # env = PreprocessObservationWrapper(env, shape=84, num_output_channels=1)
    # env = WarpFrameRGB(env)
    # env = minigrid.wrappers.StateBonus(env)
    # env = common.wrappers.MinigridRewardWrapper(env)
    env = common.wrappers.MinigridInfoWrapper(env)
    # env = common.wrappers.StateBonusCustom(env, 5000)
    # env = FrameStack(env, n_frames=1)
    # env.new_step_api = True
    return env


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


def make_env_minatar(env_id, **kwargs):
    env = gym.make(
        env_id,
        # frameskip=1,
    )
    # env = common.wrappers.FrameStack(env, 4)
    return env
