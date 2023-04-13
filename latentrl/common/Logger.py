from abc import ABC, abstractmethod
from collections import defaultdict
import torch
import wandb
from statistics import mean
import json
import os


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
    def __init__(self, log_dir, dump_interval=100):
        self.M = defaultdict(list)

        self.log_train_path = os.path.join(log_dir, "train.log")
        self.log_train_episodic_path = os.path.join(log_dir, "train_episodic.log")
        self.log_eval_path = os.path.join(log_dir, "eval.log")
        if os.path.exists(self.log_train_path):
            os.remove(self.log_train_path)
        if os.path.exists(self.log_train_episodic_path):
            os.remove(self.log_train_episodic_path)
        if os.path.exists(self.log_eval_path):
            os.remove(self.log_eval_path)

        self.dump_interval = dump_interval
        self.dump_counter = 0
        wandb.define_metric("General/timesteps_done")
        wandb.define_metric("Info/*", step_metric="General/timesteps_done")
        wandb.define_metric("Episodic/*", step_metric="General/timesteps_done")

    def log(self, metrics: dict):
        for key, value in metrics.items():
            self.M[key].append(value)

    def log_and_dump(self, metrics: dict, agent, mode="train"):
        metrics.update(
            {
                "General/timesteps_done": agent.timesteps_done,
                "General/episodes_done": agent.episodes_done,
            }
        )
        wandb.log(metrics)
        if mode == "train":
            with open(self.log_train_episodic_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        elif mode == "eval":
            data = dict()
            for key, value in metrics.items():
                data[key.replace("Evaluation/", "")] = value
            with open(self.log_eval_path, "a") as f:
                f.write(json.dumps(data) + "\n")
        else:
            raise ValueError("Unknown mode")

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
            with open(self.log_train_path, "a") as f:
                f.write(json.dumps(self.M) + "\n")
            if clear:
                self.M = defaultdict(list)
