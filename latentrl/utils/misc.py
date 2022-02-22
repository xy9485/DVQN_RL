""" Various auxiliary utilities """
import json
import math
from typing import Callable

import torch
import numpy as np
from PIL import Image
# from models import MDRNNCell, VAE, Controller


def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

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
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions


def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
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
    frame = Image.fromarray(frame, mode='RGB')
    obs = frame.resize(resize, Image.BILINEAR)
    return np.array(obs)


class ProcessFrame:
    def __init__(self, vertical_cut=84, horizontal_cut=None, resize=(64, 64)):  #default value for CarRacing-v0
        self.resize = resize
        self.horizontal_cut = horizontal_cut
        self.vertical_cut = vertical_cut

    def __call__(self, frame):
        frame = frame[:self.vertical_cut, :self.horizontal_cut, :]
        frame = Image.fromarray(frame, mode='RGB')
        obs = frame.resize(self.resize, Image.BILINEAR)
        return np.array(obs)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
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
        return progress_remaining * initial_value

    return func

def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))