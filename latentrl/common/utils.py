""" Various auxiliary utilities """
import json
import math
import os
from itertools import zip_longest
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
import torchvision
import wandb
from PIL import Image, ImageColor, ImageDraw, ImageFont

# from models import MDRNNCell, VAE, Controller
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from torch import Tensor
from torchvision.utils import make_grid


def sample_continuous_policy(action_space, seq_len, dt):
    """Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    actions = [action_space.vae_sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(
                actions[-1] + math.sqrt(dt) * daction_dt,
                action_space.low,
                action_space.high,
            )
        )
    return actions


def save_checkpoint(state, is_best, filename, best_filename):
    """Save state in filename. Also save in best_filename if is_best."""
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


# def flatten_parameters(params):
#     """ Flattening parameters.
#
#     :args params: generator of parameters (as returned by module.parameters())
#
#     :returns: flattened parameters (i.e. one tensor of dimension 1 with all
#         parameters concatenated)
#     """
#     return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()


# def unflatten_parameters(params, example, device):
#     """ Unflatten parameters.
#
#     :args params: parameters as a single 1D np array
#     :args example: generator of parameters (as returned by module.parameters()),
#         used to reshape params
#     :args device: where to store unflattened parameters
#
#     :returns: unflattened parameters
#     """
#     params = torch.Tensor(params).to(device)
#     idx = 0
#     unflattened = []
#     for e_p in example:
#         unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
#         idx += e_p.numel()
#     return unflattened


def process_frame(
    frame,
    resize=(64, 64),  # default value for carracing
    vertical_cut=84,  # default value for carracing
    horizontal_cut=None,
):
    frame = frame[:vertical_cut, :horizontal_cut, :]
    frame = Image.fromarray(frame, mode="RGB")
    obs = frame.resize(resize, Image.BILINEAR)
    return np.array(obs)


class ProcessFrame:
    def __init__(
        self, vertical_cut=84, horizontal_cut=None, resize=(64, 64)
    ):  # default value for CarRacing-v0
        self.resize = resize
        self.horizontal_cut = horizontal_cut
        self.vertical_cut = vertical_cut

    def __call__(self, frame):
        frame = frame[: self.vertical_cut, : self.horizontal_cut, :]
        frame = Image.fromarray(frame, mode="RGB")
        obs = frame.resize(self.resize, Image.BILINEAR)
        return np.array(obs)


def linear_schedule(initial_value: float, reduce: bool = True) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if reduce:
            return progress_remaining * initial_value
        else:
            return (1 - progress_remaining) * initial_value

    return func


def get_linear_fn(start: float, end: float, end_fraction: float):
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.
    :return:
    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def update_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.

    :param optimizer:
    :param learning_rate:
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))


def make_vec_env_customized(  # to customize the order of Monitor wrapper
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    # monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: the environment ID or the environment class
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    # monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            if monitor_dir:
                os.makedirs(
                    monitor_dir, exist_ok=True
                )  # when monitor_dir is given, there has to be only one env

            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


@torch.no_grad()
def wandb_log_image(tensor, nrow, section_name, **kwargs) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, nrow=nrow, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    ndarr = grid.permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    wandb.log({section_name: wandb.Image(im)})


def polyak_sync(
    params: Iterable[torch.nn.Parameter],
    target_params: Iterable[torch.nn.Parameter],
    tau: float,
) -> None:
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(tau)
            target_param.data.add_((1 - tau) * param.data)
            # torch.add(target_param.data, param.data, alpha=(1 - tau), out=target_param.data)


def soft_sync_params(
    params: Iterable[torch.nn.Parameter],
    target_params: Iterable[torch.nn.Parameter],
    tau: float,
) -> None:
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * param.data)
            # torch.add(target_param.data, param.data, alpha=(1 - tau), out=target_param.data)


def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


class Dataset_pretrain(torch.utils.data.Dataset):
    def __init__(self, data_list: Tensor):
        data_list = data_list[torch.randperm(data_list.size(0)), :]
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        state = self.data_list[idx]
        return state
