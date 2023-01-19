from abc import ABC, abstractmethod
from collections import defaultdict
import torch
import wandb
from statistics import mean

# create a abstract class for logger,
class Logger(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def log(self, key_value: dict):
        raise NotImplementedError

    def dump2wandb(self):
        raise NotImplementedError

    def dump2console(self):
        raise NotImplementedError


class LoggerWandb(Logger):
    def __init__(self):
        self.metrics = defaultdict(list)
        self.dump_interval = 100
        self.log_counter = 0
        wandb.define_metric("General/timesteps_done")
        wandb.define_metric("Training_Info/*", step_metric="General/timesteps_done")

    def log(self, key_value: dict):
        for key, value in key_value.items():
            self.metrics[key].append(value)
        self.log_counter += 1

    def log_img(self, key, img):
        wandb.log({key: wandb.Image(img)})

    def dump2wandb(self, agent, force=False):
        if force or self.log_counter % self.dump_interval == 0:
            for key, value in self.metrics.items():
                self.metrics[key] = sum(value) / len(value)
            self.metrics.update(
                {
                    "General/timesteps_done": agent.timesteps_done,
                    "General/episodes_done": agent.episodes_done,
                }
            )
            wandb.log(self.metrics)
            self.metrics = defaultdict(list)
