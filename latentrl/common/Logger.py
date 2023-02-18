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
        self.M = defaultdict(list)
        self.dump_interval = 100
        self.dump_counter = 0
        wandb.define_metric("General/timesteps_done")
        wandb.define_metric("Info/*", step_metric="General/timesteps_done")
        wandb.define_metric("Episodic/*", step_metric="General/timesteps_done")

    def log(self, metrics: dict):
        for key, value in metrics.items():
            self.M[key].append(value)
    
    def log_and_dump(self, metrics: dict, agent):
        metrics.update(
            {
                "General/timesteps_done": agent.timesteps_done,
                "General/episodes_done": agent.episodes_done,
            }
        )
        wandb.log(metrics)

    def log_img(self, key, img):
        wandb.log({key: wandb.Image(img)})

    def dump2wandb(self, agent, force=False, clear=True):
        if not force:
            self.dump_counter += 1
        if force or self.dump_counter % self.dump_interval == 0:
            for key, value in self.M.items():
                self.M[key] = sum(value) / len(value)
            self.M.update(
                {
                    "General/timesteps_done": agent.timesteps_done,
                    "General/episodes_done": agent.episodes_done,
                }
            )
            wandb.log(self.M)
            if clear:
                self.M = defaultdict(list)
