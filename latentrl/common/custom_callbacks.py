from pprint import pprint
from statistics import mean
import time
import os
import json
import numpy as np
from stable_baselines3 import DQN, SAC

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.logger import Image, TensorBoardOutputFormat
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecTransposeImage

from common.utils import pretty_json


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
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
        return True


class MyCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _init_callback(self) -> None:
        # Create folder if needed
        pass

    def _on_step(self) -> bool:
        if self.locals["done"]:
            print()


class ImageRecorderCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(ImageRecorderCallback, self).__init__(verbose)

    def _on_step(self):
        image = self.training_env.render(mode="rgb_array")
        # "HWC" specify the dataformat of the image, here channel last
        # (H for height, W for width, C for channel)
        # See https://pytorch.org/docs/stable/tensorboard.html
        # for supported formats
        self.logger.record(
            "trajectory/image",
            Image(image, "HWC"),
            exclude=("stdout", "log", "json", "csv"),
        )
        # self.logger.dump(self.num_timesteps)
        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        return True


class HparamsWriterCallback(BaseCallback):
    def __init__(self, run_name, hparams: dict, extra_hparams: dict, log_freq=4):
        super().__init__()
        self._log_freq = log_freq
        self.run_name = run_name
        self.extra_hparams = extra_hparams
        # self.hparams = {
        #     'buffer_size': None,
        #     'learning_starts': None,
        #     'batch_size': None,
        #     'tau': None,
        #     'gamma': None,
        #     'train_freq': None,
        #     'gradient_steps': None,
        #     "ent_coef": None,
        #     "target_update_interval": None,
        #     'n_stack': None,
        #     'n_envs': None,
        #     'use_sde': None,
        #     'use_sde_at_warmup': None,
        #     'policy_kwargs': None,
        #     "accelerate_warmup": "None_"
        # }

        self.hparams = {
            # "learning_rate": linear_schedule(7.3e-4),  # linear_schedule(7.3e-4), 0.0003, 0.0004(John)
            "buffer_size": 100_000,
            "learning_starts": 10_000,  # 100_000   #10_000(John)
            "batch_size": 128,
            "tau": 1,  # 0.02/0.005/1
            "gamma": 0.95,  # 0.95(John)
            "train_freq": 4,  # or int(8/n_envs) ?, 4(John)
            "gradient_steps": 1,  # -1, 1(John)
            "target_update_interval": 8,  # update target per 10000 steps in John
            "exploration_fraction": 0.9,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.01,
            # "tensorboard_log": f"../tensorboard_log/{algo_class.__name__}_{env_id}_5/",
            # "tensorboard_log": f"../tensorboard_log/{env_id}/",
            "policy_kwargs": dict(net_arch=[256, 256, 256]),
            # "policy_kwargs": dict(share_features_extractor=False, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
            # "policy_kwargs": dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
            # "policy_kwargs": dict(net_arch=dict(pi=[512, 512], qf=[512, 512])),
            "verbose": 1,
            # "accelerate_warmup": False  # only for warmup stage
        }

        # self.hparams = hparams

    def _on_training_start(self):
        self.episode_finished = 0
        for key in self.model.__dict__:
            if key in self.hparams.keys():
                if key == "train_freq":
                    self.hparams[key] = self.model.__dict__[key].frequency
                    continue
                elif key == "policy_kwargs" and isinstance(self.model, SAC):
                    pi = self.model.__dict__[key]["net_arch"]["pi"]
                    qf = self.model.__dict__[key]["net_arch"]["qf"]
                    sharing = self.model.__dict__[key].get("share_features_extractor", None)
                    if sharing:
                        self.hparams[key] = f"pi{pi}-qf{qf}-sharing"
                    else:
                        self.hparams[key] = f"pi{pi}-qf{qf}-nosharing"
                    continue
                elif key == "policy_kwargs" and isinstance(self.model, DQN):
                    self.hparams[key] = f"{self.model.__dict__[key]['net_arch']}"
                    continue
                self.hparams[key] = self.model.__dict__[key]
        if isinstance(self.model.env, VecTransposeImage):
            self.hparams["n_stack"] = self.model.env.venv.__dict__.get("n_stack", 1)
        else:
            self.hparams["n_stack"] = self.training_env.__dict__.get("n_stack", 1)
        # pprint(self.model.env.__dict__, width=1)
        self.extra_hparams.update(self.hparams)
        print("updated extra_hparams from callback:")
        pprint(self.extra_hparams, width=1)

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )
        self.tb_formatter.writer.add_text(
            f"hparams/{self.run_name}", pretty_json(self.extra_hparams), global_step=0
        )

    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_finished += 1
                if self.episode_finished % self._log_freq == 0:
                    # if self.n_calls % self._log_freq == 0:
                    print("@@@writing hparams:")
                    recent_mean_training_reward = float(
                        safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                    )
                    self.tb_formatter.writer.add_hparams(
                        self.extra_hparams,
                        {"train_rwd": recent_mean_training_reward},
                        run_name=".",
                    )
                    self.tb_formatter.writer.flush()
                    pprint(self.extra_hparams, width=1)
                    print("@@@run_name:", self.run_name)

        return True


class EarlyStopCallback(BaseCallback):  # need to optimize
    def __init__(
        self,
        reward_threshold: float = 0,
        progress_remaining_threshold: float = 0.5,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.reward_threshold = reward_threshold
        self.progress_remaining_threshold = progress_remaining_threshold

    def _on_step(self) -> bool:
        continue_training = True
        if self.model._episode_num > 1:
            if bool(self.model._current_progress_remaining < self.progress_remaining_threshold):
                mean_train_reward = float(
                    safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                )
                if bool(mean_train_reward < self.reward_threshold):
                    continue_training = False

        if self.verbose > 0 and not continue_training:
            print(
                f"Stopping training because the mean reward: {mean_train_reward:.2f} "
                f"doesn't reach the threshold: {self.reward_threshold} "
                f"before {int(self.progress_remaining_threshold * 100)}% of total time steps"
            )
        return continue_training


class TrainingRewardWriterCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._log_freq = 10

    def _on_training_start(self) -> None:
        self.episode_finished = 0
        self.ep_rewards = []
        self.ep_returns = []
        for _ in range(self.model.n_envs):
            self.ep_rewards.append([])

        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

        self.training_time_start = time.time()
        self.on_step_time_used = 0
        self.percentage = 0

    def _on_step(self) -> bool:
        on_step_time_start = time.time()
        if self.model.num_timesteps <= self.model.learning_starts:
            return
        for idx, done in enumerate(self.locals["dones"]):
            shaping = self.model.env.venv.envs[idx].shaping
            original_reward = self.model.env.venv.envs[idx].reward
            self.ep_rewards[idx].append(original_reward)
            if done:
                self.episode_finished += 1
                # print("self.episode_finished += 1")
                self.ep_returns.append(sum(self.ep_rewards[idx]))
                self.ep_rewards[idx] = []
                if self.episode_finished % self._log_freq == 0:
                    # recent_returns = self.ep_returns[-self._log_freq:]
                    recent_returns = self.ep_returns
                    recent_mean_actual_training_reward = sum(recent_returns) / len(recent_returns)
                    self.tb_formatter.writer.add_scalar(
                        "rollout/recent_rew_mean",
                        recent_mean_actual_training_reward,
                        self.model.num_timesteps,
                    )
                    # print("writing success", self.model.num_timesteps)
                    print(f"time cost of this callback: {self.percentage}")
        self.on_step_time_used += time.time() - on_step_time_start
        self.percentage = 100 * self.on_step_time_used / (time.time() - self.training_time_start)

        return True


class TrainingRewardWriterCallback_both(BaseCallback):
    def __init__(self, stack_mode="gym_stack"):
        super().__init__()
        self.stack_mode = stack_mode
        self._log_freq = 3

    def _on_training_start(self) -> None:
        self.episode_finished = 0
        self.ep_rewards = []
        self.ep_returns = []
        for _ in range(self.model.n_envs):
            self.ep_rewards.append([])

        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

        self.training_time_start = time.time()
        self.on_step_time_used = 0
        self.percentage_this_callback = 0
        self.env_time_used = 0
        self.percentage_env = 0

    def _on_step(self) -> bool:
        on_step_time_start = time.time()

        # if self.model.num_timesteps <= self.model.learning_starts:
        # return

        if self.stack_mode == "venv_stack":
            original_reward = self.model.env.venv.reward
            shaping = self.model.env.venv.shaping
            self.env_time_used += self.model.env.venv.time_step_wait
        elif self.stack_mode == "gym_stack":
            for idx, done in enumerate(self.locals["dones"]):
                self.env_time_used += self.model.env.venv.envs[idx].time_step_wait

        for idx, done in enumerate(self.locals["dones"]):
            if self.stack_mode == "gym_stack":
                shaping = self.model.env.venv.envs[idx].shaping
                original_reward = self.model.env.venv.envs[idx].reward
                self.ep_rewards[idx].append(original_reward)
            elif self.stack_mode == "venv_stack":
                self.ep_rewards[idx].append(original_reward[idx])
            else:
                raise Exception("wrong stack_mode in callback")
            if done:
                self.episode_finished += 1
                # print("self.episode_finished += 1")
                self.ep_returns.append(sum(self.ep_rewards[idx]))
                self.ep_rewards[idx] = []
                if self.episode_finished % self._log_freq == 0:
                    # recent_returns = self.ep_returns[-self._log_freq:]
                    recent_returns = self.ep_returns
                    recent_mean_actual_training_reward = sum(recent_returns) / len(recent_returns)
                    self.tb_formatter.writer.add_scalar(
                        "rollout/recent_rew_mean",
                        recent_mean_actual_training_reward,
                        self.model.num_timesteps,
                    )
                    # print("writing success", self.model.num_timesteps)
                    print(f"TIME COST of this callback: {self.percentage_this_callback}")
                    self.percentage_env = (
                        100 * self.env_time_used / (time.time() - self.training_time_start)
                    )
                    print(f"TIME COST of the env stepping: {self.percentage_env}")
        self.on_step_time_used += time.time() - on_step_time_start
        self.percentage_this_callback = (
            100 * self.on_step_time_used / (time.time() - self.training_time_start)
        )

        return True


class MyRewardWriterCallback(BaseCallback):
    def __init__(self, average_window_size=1):
        super().__init__()
        self.average_window_size = average_window_size
        self.episode_rewards = []
        self.episode_lengths = []
        # self.episode_times = []

    def _on_training_start(self) -> None:
        self.episode_finished = 0
        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_finished += 1
                info = self.locals["infos"][idx]
                if "episode" in info.keys():
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
                    # self.episode_times.append(info["episode"]["t"])
                    if self.episode_finished % self.average_window_size == 0:
                        recent_returns = self.episode_rewards[-self.average_window_size :]
                        self.tb_formatter.writer.add_scalar(
                            f"rollout/recent_rew({self.average_window_size}eps)_timesteps",
                            sum(recent_returns) / len(recent_returns),
                            self.model.num_timesteps,
                        )
                        self.tb_formatter.writer.add_scalar(
                            f"rollout/recent_rew({self.average_window_size}eps)_eps",
                            sum(recent_returns) / len(recent_returns),
                            self.episode_finished,
                        )
                return True


class NonZeroRewardWriterCallback(BaseCallback):
    def __init__(self, average_window_size=5):
        super().__init__()
        self.average_window_size = average_window_size
        self.episode_nonzero_rewards = []
        self.episode_negative_rewards = []

    def _on_training_start(self) -> None:
        self.episode_finished = 0
        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_finished += 1
                info = self.locals["infos"][idx]
                if "non_zero_accumulated_reward" in info.keys():
                    self.episode_nonzero_rewards.append(info["non_zero_accumulated_reward"])
                    if self.episode_finished % self.average_window_size == 0:
                        recent_returns = self.episode_nonzero_rewards[-self.average_window_size :]
                        self.tb_formatter.writer.add_scalar(
                            f"rollout/recent_non_zero_rew({self.average_window_size}eps)_timesteps",
                            sum(recent_returns) / len(recent_returns),
                            self.model.num_timesteps,
                        )
                        self.tb_formatter.writer.add_scalar(
                            f"rollout/recent_non_zero_rew({self.average_window_size}eps)_eps",
                            sum(recent_returns) / len(recent_returns),
                            self.episode_finished,
                        )
                if "negative_accumulated_reward" in info.keys():
                    self.episode_negative_rewards.append(info["negative_accumulated_reward"])
                    if self.episode_finished % self.average_window_size == 0:
                        recent_returns = self.episode_negative_rewards[-self.average_window_size :]
                        self.tb_formatter.writer.add_scalar(
                            f"rollout/recent_negative_rew({self.average_window_size}eps)_timesteps",
                            sum(recent_returns) / len(recent_returns),
                            self.model.num_timesteps,
                        )
                        self.tb_formatter.writer.add_scalar(
                            f"rollout/recent_negative_rew({self.average_window_size}eps)_eps",
                            sum(recent_returns) / len(recent_returns),
                            self.episode_finished,
                        )

                # else:
                #     print("no key named non_zero_accumulated_reward")


class EpisodeCounterCallback(BaseCallback):
    def __init__(self, num_episodes=8000):
        super().__init__()
        self.num_episodes = num_episodes

    def _on_training_start(self) -> None:
        self.finished_episodes = 0

    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                self.finished_episodes += 1
                print("finished episode:", self.finished_episodes)
        if self.finished_episodes == self.num_episodes:
            return False
        else:
            return True


class OmegaScheduler(BaseCallback):
    def __init__(self):
        super().__init__()


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model_train.zip")
        self.best_mean_reward = -np.inf
        self.episode_finished = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        # if self.save_path is not None:
        #     os.makedirs(self.save_path, exist_ok=True)
        pass

    def _on_step2(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-30:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    # if self.verbose > 0:
                    print(
                        f"Saving new best model with reward:{self.best_mean_reward:.2f} to {self.save_path}"
                    )
                    self.model.save(self.save_path)

    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_finished += 1
                # Retrieve training reward
                x, y = ts2xy(load_results(self.log_dir), "episodes")
                if len(x) > 0:
                    mean_reward = np.mean(y[-20:])
                    # New best model, you could save the agent here
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        # Example for saving best model
                        # if self.verbose > 0:
                        print(
                            f"Saving new best model with reward:{self.best_mean_reward:.2f} to {self.save_path}"
                        )
                        self.model.save(self.save_path)

        return True
