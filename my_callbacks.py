import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.logger import Image

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

class MyCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _init_callback(self) -> None:
        # Create folder if needed
        pass

    def _on_step(self) -> bool:
        if self.locals['done']:
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
        self.logger.record("trajectory/image", Image(image, "HWC"), exclude=("stdout", "log", "json", "csv"))
        # self.logger.dump(self.num_timesteps)
        return True
